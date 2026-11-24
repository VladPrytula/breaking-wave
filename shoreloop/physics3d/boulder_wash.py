"""Breaking-wave integration gates for resolved boulders on the true beach."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from shoreloop.config import BC, BC3DConfig, Grid3DConfig, Solver3DConfig
from shoreloop.physics3d.cut_cells import CutCellGeometry, initialize_scene_cut_cells
from shoreloop.physics3d.diagnostic_render import render_field_slice
from shoreloop.physics3d.pcg import PCGConfig
from shoreloop.physics3d.projection import ProjectionConfig
from shoreloop.physics3d.relaxation import WaveRelaxationZone
from shoreloop.physics3d.solver import Solver3D
from shoreloop.physics3d.solids import measure_solid_flux
from shoreloop.physics3d.timestep import TimeStepConfig
from shoreloop.physics3d.vof import initialize_box, measure_vof
from shoreloop.physics3d.waves import LinearWaveInlet, PeriodicWaveComponent
from shoreloop.physics3d.wet_dry import WispRemover
from shoreloop.scene.geometry3d import (
    AnalyticBeach,
    SceneGeometry,
    deterministic_boulders,
)


@dataclass(frozen=True)
class OneBoulderWashMetrics:
    """Flow, contact, wake, and conservation measurements for AW-508."""

    grid: tuple[int, int, int]
    cells: int
    cut_cells: int
    stabilized_cells: int
    minimum_open_fraction: float
    maximum_upstream_surface: float
    maximum_downstream_surface: float
    maximum_surface_head_difference: float
    maximum_side_velocity: float
    maximum_boulder_pressure: float
    maximum_boulder_contact_fraction: float
    maximum_boulder_flux_residual: float
    maximum_vorticity: float
    maximum_reverse_flow_cells: int
    maximum_reverse_speed: float
    maximum_recirculation_length: float
    maximum_split_depth_contrast: float
    maximum_downstream_rejoin_depth: float
    backwash_start_time: float | None
    maximum_backwash_speed: float
    closed_face_absolute_flux: float
    volume_source: float
    advection_volume_drift: float
    clipped_volume: float
    wisp_volume_removed: float
    relative_volume_residual: float
    alpha_min: float
    alpha_max: float
    solid_water_volume: float
    max_divergence: float
    max_pressure_iterations: int
    accepted_steps: int

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(asdict(self), indent=2) + "\n")


def _cell_velocity(solver: Solver3D) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = solver.state.get_interior("u")
    v = solver.state.get_interior("v")
    w = solver.state.get_interior("w")
    return (
        0.5 * (u[:-1] + u[1:]),
        0.5 * (v[:, :-1] + v[:, 1:]),
        0.5 * (w[:, :, :-1] + w[:, :, 1:]),
    )


def _vorticity(
    velocity: tuple[np.ndarray, np.ndarray, np.ndarray],
    spacing: tuple[float, float, float],
) -> np.ndarray:
    u, v, w = velocity
    dx, dy, dz = spacing
    du_dy = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2.0 * dy)
    dw_dy = (np.roll(w, -1, axis=1) - np.roll(w, 1, axis=1)) / (2.0 * dy)
    du_dz = np.gradient(u, dz, axis=2)
    dv_dx = np.gradient(v, dx, axis=0)
    dv_dz = np.gradient(v, dz, axis=2)
    dw_dx = np.gradient(w, dx, axis=0)
    return np.sqrt(
        (dw_dy - dv_dz) ** 2
        + (du_dz - dw_dx) ** 2
        + (dv_dx - du_dy) ** 2
    )


def _aperture_flux_divergence(solver: Solver3D) -> np.ndarray:
    state = solver.state
    g = solver.cfg.grid
    u = state.get_interior("u")
    v = state.get_interior("v")
    w = state.get_interior("w")
    au = state.get_geometry_interior("u_aperture")
    av = state.get_geometry_interior("v_aperture")
    aw = state.get_geometry_interior("w_aperture")
    return (
        (au[1:] * u[1:] - au[:-1] * u[:-1]) / g.dx
        + (av[:, 1:] * v[:, 1:] - av[:, :-1] * v[:, :-1]) / g.dy
        + (aw[:, :, 1:] * w[:, :, 1:]
           - aw[:, :, :-1] * w[:, :, :-1]) / g.dz
    )


def _render_boulder_surface_fields(
    path: str | Path,
    geometry: CutCellGeometry,
    boulder_cut: np.ndarray,
    alpha: np.ndarray,
    pressure: np.ndarray,
    vorticity: np.ndarray,
) -> None:
    """Plot contact, pressure, and vorticity on the embedded surface."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    points = geometry.boundary_centroids[boulder_cut]
    fields = (
        (alpha[boulder_cut], "water contact", "Blues", 0.0, 1.0),
        (pressure[boulder_cut], "pressure [Pa]", "coolwarm", None, None),
        (vorticity[boulder_cut], "vorticity [s⁻¹]", "magma", 0.0, None),
    )
    figure = plt.figure(figsize=(13.0, 4.2), constrained_layout=True)
    for index, (values, title, cmap, vmin, vmax) in enumerate(fields, start=1):
        axis = figure.add_subplot(1, 3, index, projection="3d")
        surface = axis.scatter(
            points[:, 0], points[:, 1], points[:, 2], c=values,
            cmap=cmap, vmin=vmin, vmax=vmax, s=10.0,
        )
        axis.set_title(title)
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.set_zlabel("z [m]")
        axis.view_init(elev=24.0, azim=-64.0)
        figure.colorbar(surface, ax=axis, shrink=0.68)
    figure.savefig(output, dpi=140)
    plt.close(figure)


def run_one_boulder_wash_gate(
    *,
    device: str = "cuda:0",
    output_dir: str = "out/wave_tank3d/one-boulder-wash",
    write_frames: bool = False,
) -> OneBoulderWashMetrics:
    """Run one resolved breaker through impact, wrap, wake, and retreat."""
    grid = Grid3DConfig(
        nx=104, ny=32, nz=24,
        Lx=4.16, Ly=1.6, Lz=0.48,
    )
    cfg = Solver3DConfig(
        grid=grid,
        bc=BC3DConfig(
            xlo=BC.WALL, xhi=BC.WALL,
            ylo=BC.PERIODIC, yhi=BC.PERIODIC,
            zlo=BC.WALL, zhi=BC.WALL,
        ),
        device=device,
        output_dir=output_dir,
    )
    depth = 0.28
    period = 0.8
    component = PeriodicWaveComponent(
        period=period,
        depth=depth,
        alongshore_length=grid.Ly,
        amplitude=0.09,
        gravity=cfg.physics.g,
    )
    beach = AnalyticBeach(1.35, 0.18, grid.z0)
    boulder = deterministic_boulders()[0]
    scene = SceneGeometry(beach, (boulder,))
    solver = Solver3D(
        cfg,
        ProjectionConfig(
            pcg=PCGConfig(
                relative_tolerance=2.0e-5,
                max_iterations=300,
                preconditioner="multigrid",
            ),
            rms_divergence_tolerance=8.0e-4,
        ),
        TimeStepConfig(dt_max=0.005),
    )
    initialize_box(
        solver.state,
        (grid.x0, grid.y0, grid.z0),
        (grid.x0 + grid.Lx, grid.y0 + grid.Ly, grid.z0 + depth),
    )
    geometry = initialize_scene_cut_cells(solver.state, scene)
    inlet = LinearWaveInlet((component,))
    solver.set_wave_relaxation(WaveRelaxationZone(
        cfg, inlet, component.wavelength, timescale=0.02 * period,
    ))
    solver.set_wisp_remover(WispRemover(cfg))

    flat = PeriodicWaveComponent(
        period=period,
        depth=depth,
        alongshore_length=grid.Ly,
        amplitude=0.0,
        gravity=cfg.physics.g,
    )
    flat_zone = WaveRelaxationZone(
        cfg, LinearWaveInlet((flat,)), component.wavelength,
        timescale=0.05 * period,
    )
    x, y, _ = solver.state.grid.coordinates("cell")
    centre = np.asarray(boulder.centre)
    radii = np.asarray(boulder.radii)
    centre_i = int(np.argmin(np.abs(x - centre[0])))
    centre_j = int(np.argmin(np.abs(y - centre[1])))
    upstream_i = int(np.argmin(np.abs(x - (centre[0] - radii[0] - 0.08))))
    downstream_i = int(np.argmin(np.abs(x - (centre[0] + radii[0] + 0.12))))
    side_indices = (
        int(np.argmin(np.abs(y - (centre[1] - radii[1] - grid.dy)))),
        int(np.argmin(np.abs(y - (centre[1] + radii[1] + grid.dy)))),
    )
    boulder_cells = (geometry.material_ids == boulder.material_id) & (
        geometry.theta > 0.0
    )
    boulder_cut = boulder_cells & geometry.cut_mask
    near_x = np.abs(x[:, None, None] - centre[0]) < radii[0] + 0.75
    near_y_distance = np.abs(y[None, :, None] - centre[1])
    near_y_distance = np.minimum(near_y_distance, grid.Ly - near_y_distance)
    near_y = near_y_distance < radii[1] + 0.30
    downstream = (
        (x[:, None, None] > centre[0] + radii[0])
        & (x[:, None, None] < centre[0] + radii[0] + 0.75)
        & near_y
    )
    upstream = (
        (x[:, None, None] > centre[0] - radii[0] - 0.45)
        & (x[:, None, None] < centre[0] - radii[0])
        & near_y
    )

    generation_end = 1.5 * period
    end_time = 4.0 * period
    sample_interval = 0.04
    next_sample = sample_interval
    absorbing = False
    initial_volume = measure_vof(solver.state).volume
    source_volume = 0.0
    advection_volume_drift = 0.0
    clipped_volume = 0.0
    wisp_volume_removed = 0.0
    max_divergence = 0.0
    max_pressure_iterations = 0
    maximum_upstream_surface = grid.z0 + depth
    maximum_downstream_surface = grid.z0 + depth
    maximum_head_difference = 0.0
    maximum_side_velocity = 0.0
    maximum_pressure = 0.0
    maximum_contact = 0.0
    maximum_boulder_flux_residual = 0.0
    maximum_vorticity = 0.0
    maximum_reverse_cells = 0
    maximum_reverse_speed = 0.0
    maximum_recirculation_length = 0.0
    maximum_split_contrast = 0.0
    maximum_rejoin_depth = 0.0
    wash_times = []
    wash_velocities = []
    closed_face_flux = 0.0
    best_frame_score = 0.0
    best_frame_fields = None

    while solver.time < end_time - 1.0e-12:
        target = min(next_sample, end_time)
        if not absorbing:
            target = min(target, generation_end)
        timestep = solver.compute_dt()
        dt = min(timestep.dt, target - solver.time)
        projection = solver.step(dt, timestep=timestep)
        source_volume += solver.last_relaxation.volume_source
        advection_volume_drift += solver.last_advection.vof.volume_drift
        clipped_volume += solver.last_advection.vof.clipped_volume
        wisp_volume_removed += solver.last_wisp_removal.removed_volume
        max_divergence = max(max_divergence, projection.max_divergence_after)
        max_pressure_iterations = max(
            max_pressure_iterations, projection.pressure_solve.iterations,
        )
        if not absorbing and abs(solver.time - generation_end) < 1.0e-10:
            solver.set_wave_relaxation(flat_zone)
            absorbing = True

        if abs(solver.time - next_sample) < 1.0e-10:
            alpha = solver.state.get_interior("alpha")
            theta = solver.state.get_geometry_interior("theta")
            liquid = alpha * theta
            water_depth = liquid.sum(axis=2) * grid.dz
            beach_height = np.asarray(beach.elevation(x))[:, None]
            surface = beach_height + water_depth
            upstream_surface = float(surface[upstream_i, centre_j])
            downstream_surface = float(surface[downstream_i, centre_j])
            maximum_upstream_surface = max(
                maximum_upstream_surface, upstream_surface,
            )
            maximum_downstream_surface = max(
                maximum_downstream_surface, downstream_surface,
            )
            maximum_head_difference = max(
                maximum_head_difference,
                upstream_surface - downstream_surface,
            )

            velocity = _cell_velocity(solver)
            speed = np.sqrt(sum(component_velocity**2 for component_velocity in velocity))
            maximum_side_velocity = max(
                maximum_side_velocity,
                *(float(np.max(speed[centre_i, side_j]))
                  for side_j in side_indices),
            )
            pressure = solver.state.get_interior("p")
            boulder_active = boulder_cut & (alpha >= 0.5)
            boulder_resolved = boulder_active & (
                theta >= geometry.stabilization_threshold
            )
            if np.any(boulder_active):
                maximum_pressure = max(
                    maximum_pressure,
                    float(np.max(np.abs(pressure[boulder_resolved]))),
                )
                maximum_contact = max(
                    maximum_contact,
                    float(np.count_nonzero(alpha[boulder_cut] > 0.05))
                    / int(np.count_nonzero(boulder_cut)),
                )
                flux_residual = _aperture_flux_divergence(solver)
                maximum_boulder_flux_residual = max(
                    maximum_boulder_flux_residual,
                    float(np.max(np.abs(flux_residual[boulder_active]))),
                )
            omega = _vorticity(velocity, (grid.dx, grid.dy, grid.dz))
            wet_near = (liquid > 0.05) & near_x & near_y
            if np.any(wet_near):
                maximum_vorticity = max(
                    maximum_vorticity, float(np.max(omega[wet_near])),
                )

            upstream_wet = upstream & (liquid > 0.05)
            incident_velocity = (
                float(np.mean(velocity[0][upstream_wet]))
                if np.any(upstream_wet) else 0.0
            )
            reversed_flow = downstream & (liquid > 0.05) & (velocity[0] < -0.02)
            if incident_velocity > 0.05 and np.any(reversed_flow):
                reverse_cells = int(np.count_nonzero(reversed_flow))
                maximum_reverse_cells = max(maximum_reverse_cells, reverse_cells)
                maximum_reverse_speed = max(
                    maximum_reverse_speed,
                    float(-np.min(velocity[0][reversed_flow])),
                )
                reverse_x = np.flatnonzero(np.any(reversed_flow, axis=(1, 2)))
                maximum_recirculation_length = max(
                    maximum_recirculation_length,
                    float(x[reverse_x[-1]] - x[reverse_x[0]] + grid.dx),
                )

            side_depth = 0.5 * (
                water_depth[centre_i, side_indices[0]]
                + water_depth[centre_i, side_indices[1]]
            )
            maximum_split_contrast = max(
                maximum_split_contrast,
                float(side_depth - water_depth[centre_i, centre_j]),
            )
            maximum_rejoin_depth = max(
                maximum_rejoin_depth,
                float(water_depth[downstream_i, centre_j]),
            )
            wash_weight = np.where(downstream, liquid, 0.0)
            wash_velocities.append(float(
                np.sum(wash_weight * velocity[0], dtype=np.float64)
                / np.sum(wash_weight, dtype=np.float64)
            ))
            wash_times.append(solver.time)
            closed_face_flux = max(
                closed_face_flux, measure_solid_flux(solver.state).absolute_flux,
            )
            score = maximum_split_contrast + 0.1 * float(np.max(omega[wet_near]))
            if write_frames and score > best_frame_score:
                best_frame_score = score
                best_frame_fields = (
                    alpha.copy(), pressure.copy(), omega.copy(),
                )
            next_sample += sample_interval

    if write_frames:
        peak_alpha, peak_pressure, peak_vorticity = best_frame_fields
        _render_boulder_surface_fields(
            Path(output_dir) / "one_boulder_peak_surface.png",
            geometry, boulder_cut,
            peak_alpha, peak_pressure, peak_vorticity,
        )
        render_field_slice(
            solver.state, Path(output_dir) / "one_boulder_final_xz.png",
            plane="xz", index=centre_j,
        )
    final = measure_vof(solver.state)
    wash_velocity = np.asarray(wash_velocities)
    peak_wash = int(np.argmax(wash_velocity))
    backwash_samples = np.flatnonzero(wash_velocity[peak_wash:] < 0.0)
    backwash_start_time = (
        float(wash_times[peak_wash + int(backwash_samples[0])])
        if backwash_samples.size else None
    )
    maximum_backwash_speed = max(
        0.0, float(-np.min(wash_velocity[peak_wash:])),
    )
    residual = (
        final.volume - initial_volume - source_volume + wisp_volume_removed
    )
    alpha = solver.state.get_interior("alpha")
    solid = solver.state.get_solid_interior().astype(bool)
    solid_water = float(alpha[solid].sum(dtype=np.float64)) * (
        grid.dx * grid.dy * grid.dz
    )
    return OneBoulderWashMetrics(
        grid=grid.shape,
        cells=grid.cell_count,
        cut_cells=int(np.count_nonzero(geometry.cut_mask)),
        stabilized_cells=geometry.stabilized_cells,
        minimum_open_fraction=geometry.minimum_open_fraction,
        maximum_upstream_surface=maximum_upstream_surface,
        maximum_downstream_surface=maximum_downstream_surface,
        maximum_surface_head_difference=maximum_head_difference,
        maximum_side_velocity=maximum_side_velocity,
        maximum_boulder_pressure=maximum_pressure,
        maximum_boulder_contact_fraction=maximum_contact,
        maximum_boulder_flux_residual=maximum_boulder_flux_residual,
        maximum_vorticity=maximum_vorticity,
        maximum_reverse_flow_cells=maximum_reverse_cells,
        maximum_reverse_speed=maximum_reverse_speed,
        maximum_recirculation_length=maximum_recirculation_length,
        maximum_split_depth_contrast=maximum_split_contrast,
        maximum_downstream_rejoin_depth=maximum_rejoin_depth,
        backwash_start_time=backwash_start_time,
        maximum_backwash_speed=maximum_backwash_speed,
        closed_face_absolute_flux=closed_face_flux,
        volume_source=source_volume,
        advection_volume_drift=advection_volume_drift,
        clipped_volume=clipped_volume,
        wisp_volume_removed=wisp_volume_removed,
        relative_volume_residual=residual / initial_volume,
        alpha_min=final.alpha_min,
        alpha_max=final.alpha_max,
        solid_water_volume=solid_water,
        max_divergence=max_divergence,
        max_pressure_iterations=max_pressure_iterations,
        accepted_steps=solver.steps,
    )
