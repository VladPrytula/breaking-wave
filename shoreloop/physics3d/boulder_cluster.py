"""Breaking wash and backwash through the deterministic boulder cluster."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from shoreloop.config import BC, BC3DConfig, Grid3DConfig, Solver3DConfig
from shoreloop.physics3d.boulder_wash import (
    _aperture_flux_divergence,
    _cell_velocity,
    _render_boulder_surface_fields,
    _vorticity,
)
from shoreloop.physics3d.cut_cells import initialize_scene_cut_cells
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
    RoundedBoulder,
    SceneGeometry,
    deterministic_boulders,
)


@dataclass(frozen=True)
class BoulderClusterMetrics:
    """Gap, shelter, wake, and conservation measurements for AW-509."""

    grid: tuple[int, int, int]
    cells: int
    cut_cells: int
    stabilized_cells: int
    minimum_open_fraction: float
    gap_open_areas: tuple[float, float]
    maximum_forward_gap_discharges: tuple[float, float]
    maximum_reverse_gap_discharges: tuple[float, float]
    maximum_gap_speeds: tuple[float, float]
    maximum_gap_jet_amplification: float
    reversing_gaps: int
    first_gap_reversal_time: float | None
    maximum_reverse_flow_cells: int
    maximum_reverse_speed: float
    maximum_recirculation_length: float
    maximum_vorticity: float
    maximum_interacting_wake_cells: int
    maximum_interacting_wake_span: float
    maximum_sheltered_water_volume: float
    minimum_shelter_velocity_ratio: float
    maximum_boulder_pressure: float
    maximum_boulder_contact_fraction: float
    maximum_boulder_flux_residual: float
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
    minimum_dt: float
    accepted_steps: int

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(asdict(self), indent=2) + "\n")


@dataclass(frozen=True)
class _GapSection:
    face_index: int
    mask: np.ndarray
    open_area: float


def _gap_section(
    solver: Solver3D,
    lower: RoundedBoulder,
    upper: RoundedBoulder,
    x_position: float,
) -> _GapSection:
    grid = solver.cfg.grid
    x, y, z = solver.state.grid.coordinates("u")
    face_index = int(np.argmin(np.abs(x - x_position)))
    yy, zz = np.meshgrid(y, z, indexing="ij")
    top = min(
        lower.centre[2] + lower.radii[2],
        upper.centre[2] + upper.radii[2],
    )
    mask = (
        (yy > lower.centre[1])
        & (yy < upper.centre[1])
        & (zz < top)
    )
    aperture = solver.state.get_geometry_interior("u_aperture")[face_index]
    open_area = float(np.sum(aperture[mask], dtype=np.float64)) * (
        grid.dy * grid.dz
    )
    return _GapSection(face_index, mask, open_area)


def _section_flow(
    solver: Solver3D,
    section: _GapSection,
    alpha: np.ndarray,
) -> tuple[float, float, float]:
    index = section.face_index
    u = solver.state.get_interior("u")[index]
    aperture = solver.state.get_geometry_interior("u_aperture")[index]
    face_alpha = 0.5 * (alpha[index - 1] + alpha[index])
    wet_fraction = aperture * face_alpha * section.mask
    face_area = solver.cfg.grid.dy * solver.cfg.grid.dz
    wet_area = float(np.sum(wet_fraction, dtype=np.float64)) * face_area
    discharge = float(np.sum(
        wet_fraction * u, dtype=np.float64,
    )) * face_area
    active = (wet_fraction > 0.05) & section.mask
    speed = float(np.max(np.abs(u[active]))) if np.any(active) else 0.0
    mean_velocity = discharge / wet_area if wet_area > 0.0 else 0.0
    return discharge, mean_velocity, speed


def _approach_velocity(solver: Solver3D, alpha: np.ndarray) -> float:
    x, _, z = solver.state.grid.coordinates("u")
    index = int(np.argmin(np.abs(x - 2.05)))
    u = solver.state.get_interior("u")[index]
    aperture = solver.state.get_geometry_interior("u_aperture")[index]
    face_alpha = 0.5 * (alpha[index - 1] + alpha[index])
    mask = z[None, :] < 0.40
    weight = aperture * face_alpha * mask
    return float(
        np.sum(weight * u, dtype=np.float64)
        / np.sum(weight, dtype=np.float64)
    )


def _periodic_y_distance(
    y: np.ndarray,
    centre: float,
    length: float,
) -> np.ndarray:
    distance = np.abs(y - centre)
    return np.minimum(distance, length - distance)


def run_boulder_cluster_gate(
    *,
    device: str = "cuda:0",
    output_dir: str = "out/wave_tank3d/boulder-cluster",
    write_frames: bool = False,
) -> BoulderClusterMetrics:
    """Run breaking wash, interacting wakes, and retreat through two gaps."""
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
    boulders = deterministic_boulders()
    scene = SceneGeometry(beach, boulders)
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

    first, second, third = boulders
    gaps = (
        _gap_section(solver, first, second, 2.60),
        _gap_section(solver, third, second, 2.90),
    )
    x, y, _ = solver.state.grid.coordinates("cell")
    xx = x[:, None, None]
    yy = y[None, :, None]
    wake_masks = []
    for boulder in boulders:
        centre = np.asarray(boulder.centre)
        radii = np.asarray(boulder.radii)
        wake_masks.append(
            (xx > centre[0] + radii[0])
            & (xx < centre[0] + radii[0] + 0.70)
            & (_periodic_y_distance(yy, centre[1], grid.Ly)
               < radii[1] + 0.25)
        )
    wake_count = sum(wake_masks)
    wake_union = wake_count > 0
    wake_overlap = wake_count > 1
    shadow = (
        (xx > 3.35) & (xx < 3.75)
        & (yy > 0.35) & (yy < 1.25)
    )
    exposed = (
        (xx > 3.35) & (xx < 3.75)
        & (yy > 0.02) & (yy < 0.27)
    )
    boulder_cut = (
        (geometry.material_ids == first.material_id) & geometry.cut_mask
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
    minimum_dt = np.inf
    forward_gap_discharge = np.zeros(2)
    maximum_gap_speed = np.zeros(2)
    maximum_jet_amplification = 0.0
    maximum_reverse_cells = 0
    maximum_reverse_speed = 0.0
    maximum_recirculation_length = 0.0
    maximum_vorticity = 0.0
    maximum_interacting_cells = 0
    maximum_interacting_span = 0.0
    maximum_sheltered_volume = 0.0
    minimum_shelter_ratio = np.inf
    maximum_pressure = 0.0
    maximum_contact = 0.0
    maximum_flux_residual = 0.0
    closed_face_flux = 0.0
    times = []
    gap_discharge_history = []
    history = []
    best_frame_score = 0.0
    best_frame_fields = None

    while solver.time < end_time - 1.0e-12:
        target = min(next_sample, end_time)
        if not absorbing:
            target = min(target, generation_end)
        timestep = solver.compute_dt()
        step_limit = timestep.dt
        dt = min(step_limit, target - solver.time)
        minimum_dt = min(minimum_dt, step_limit)
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
            velocity = _cell_velocity(solver)
            omega = _vorticity(velocity, (grid.dx, grid.dy, grid.dz))
            approach = _approach_velocity(solver, alpha)
            gap_flow = tuple(
                _section_flow(solver, section, alpha) for section in gaps
            )
            discharges = np.asarray([flow[0] for flow in gap_flow])
            gap_means = np.asarray([flow[1] for flow in gap_flow])
            gap_speeds = np.asarray([flow[2] for flow in gap_flow])
            forward_gap_discharge = np.maximum(
                forward_gap_discharge, discharges,
            )
            maximum_gap_speed = np.maximum(maximum_gap_speed, gap_speeds)
            if approach > 0.05:
                maximum_jet_amplification = max(
                    maximum_jet_amplification,
                    float(np.max(gap_means)) / approach,
                )

            wet_wake = wake_union & (liquid > 0.05)
            reversed_flow = wet_wake & (velocity[0] < -0.02)
            reverse_cells = int(np.count_nonzero(reversed_flow))
            if approach > 0.05 and reverse_cells:
                maximum_reverse_cells = max(
                    maximum_reverse_cells, reverse_cells,
                )
                maximum_reverse_speed = max(
                    maximum_reverse_speed,
                    float(-np.min(velocity[0][reversed_flow])),
                )
                reverse_x = np.flatnonzero(np.any(reversed_flow, axis=(1, 2)))
                maximum_recirculation_length = max(
                    maximum_recirculation_length,
                    float(x[reverse_x[-1]] - x[reverse_x[0]] + grid.dx),
                )

            wet_cluster = (liquid > 0.05) & (
                (xx > 2.1) & (xx < 3.9)
            )
            maximum_vorticity = max(
                maximum_vorticity, float(np.max(omega[wet_cluster])),
            )
            interacting = wake_overlap & (liquid > 0.05) & (omega > 5.0)
            interacting_cells = int(np.count_nonzero(interacting))
            maximum_interacting_cells = max(
                maximum_interacting_cells, interacting_cells,
            )
            if interacting_cells:
                active_y = np.flatnonzero(np.any(interacting, axis=(0, 2)))
                maximum_interacting_span = max(
                    maximum_interacting_span,
                    float(y[active_y[-1]] - y[active_y[0]] + grid.dy),
                )

            cell_volume = grid.dx * grid.dy * grid.dz
            shadow_weight = np.where(shadow, liquid, 0.0)
            exposed_weight = np.where(exposed, liquid, 0.0)
            shadow_volume = float(np.sum(
                shadow_weight, dtype=np.float64,
            )) * cell_volume
            maximum_sheltered_volume = max(
                maximum_sheltered_volume, shadow_volume,
            )
            shadow_mass = float(np.sum(shadow_weight, dtype=np.float64))
            exposed_mass = float(np.sum(exposed_weight, dtype=np.float64))
            shadow_velocity = (
                float(np.sum(
                    shadow_weight * velocity[0], dtype=np.float64,
                )) / shadow_mass
                if shadow_mass > 0.0 else 0.0
            )
            exposed_velocity = (
                float(np.sum(
                    exposed_weight * velocity[0], dtype=np.float64,
                )) / exposed_mass
                if exposed_mass > 0.0 else 0.0
            )
            shelter_ratio = None
            if exposed_velocity > 0.05 and shadow_volume > cell_volume:
                shelter_ratio = shadow_velocity / exposed_velocity
                minimum_shelter_ratio = min(
                    minimum_shelter_ratio, shelter_ratio,
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
                maximum_flux_residual = max(
                    maximum_flux_residual,
                    float(np.max(np.abs(flux_residual[boulder_active]))),
                )
            closed_face_flux = max(
                closed_face_flux, measure_solid_flux(solver.state).absolute_flux,
            )

            score = float(np.sum(np.maximum(discharges, 0.0))) + (
                1.0e-4 * interacting_cells
            )
            if write_frames and score > best_frame_score:
                best_frame_score = score
                best_frame_fields = (
                    alpha.copy(), pressure.copy(), omega.copy(),
                )
            times.append(solver.time)
            gap_discharge_history.append(discharges)
            history.append({
                "time": solver.time,
                "approach_velocity": approach,
                "gap_discharges": discharges.tolist(),
                "gap_mean_velocities": gap_means.tolist(),
                "gap_speeds": gap_speeds.tolist(),
                "reverse_flow_cells": reverse_cells,
                "peak_vorticity": float(np.max(omega[wet_cluster])),
                "interacting_wake_cells": interacting_cells,
                "sheltered_water_volume": shadow_volume,
                "shelter_velocity_ratio": shelter_ratio,
            })
            next_sample += sample_interval

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "boulder_cluster_history.json").write_text(
        json.dumps(history, indent=2) + "\n",
    )
    if write_frames:
        peak_alpha, peak_pressure, peak_vorticity = best_frame_fields
        _render_boulder_surface_fields(
            output / "boulder_cluster_peak_surface.png",
            geometry, boulder_cut,
            peak_alpha, peak_pressure, peak_vorticity,
        )
        render_field_slice(
            solver.state, output / "boulder_cluster_final_xy.png",
            plane="xy", index=round(depth / grid.dz) - 1,
        )

    time_array = np.asarray(times)
    discharge_array = np.asarray(gap_discharge_history)
    reverse_gap_discharge = np.zeros(2)
    reversal_times = []
    for index in range(2):
        peak = int(np.argmax(discharge_array[:, index]))
        post_peak = discharge_array[peak:, index]
        reverse_gap_discharge[index] = max(0.0, float(-np.min(post_peak)))
        reversed_samples = np.flatnonzero(post_peak < 0.0)
        if reversed_samples.size:
            reversal_times.append(float(
                time_array[peak + int(reversed_samples[0])],
            ))

    final = measure_vof(solver.state)
    residual = (
        final.volume - initial_volume - source_volume + wisp_volume_removed
    )
    alpha = solver.state.get_interior("alpha")
    solid = solver.state.get_solid_interior().astype(bool)
    solid_water = float(alpha[solid].sum(dtype=np.float64)) * (
        grid.dx * grid.dy * grid.dz
    )
    return BoulderClusterMetrics(
        grid=grid.shape,
        cells=grid.cell_count,
        cut_cells=int(np.count_nonzero(geometry.cut_mask)),
        stabilized_cells=geometry.stabilized_cells,
        minimum_open_fraction=geometry.minimum_open_fraction,
        gap_open_areas=tuple(section.open_area for section in gaps),
        maximum_forward_gap_discharges=tuple(
            float(value) for value in forward_gap_discharge
        ),
        maximum_reverse_gap_discharges=tuple(
            float(value) for value in reverse_gap_discharge
        ),
        maximum_gap_speeds=tuple(
            float(value) for value in maximum_gap_speed
        ),
        maximum_gap_jet_amplification=maximum_jet_amplification,
        reversing_gaps=len(reversal_times),
        first_gap_reversal_time=min(reversal_times) if reversal_times else None,
        maximum_reverse_flow_cells=maximum_reverse_cells,
        maximum_reverse_speed=maximum_reverse_speed,
        maximum_recirculation_length=maximum_recirculation_length,
        maximum_vorticity=maximum_vorticity,
        maximum_interacting_wake_cells=maximum_interacting_cells,
        maximum_interacting_wake_span=maximum_interacting_span,
        maximum_sheltered_water_volume=maximum_sheltered_volume,
        minimum_shelter_velocity_ratio=minimum_shelter_ratio,
        maximum_boulder_pressure=maximum_pressure,
        maximum_boulder_contact_fraction=maximum_contact,
        maximum_boulder_flux_residual=maximum_flux_residual,
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
        minimum_dt=float(minimum_dt),
        accepted_steps=solver.steps,
    )
