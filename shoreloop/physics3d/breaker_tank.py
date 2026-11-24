"""Quasi-two-dimensional shoaling and breaker integration scenarios."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from shoreloop.config import (BC, BC3DConfig, Grid3DConfig, Solver3DConfig)
from shoreloop.physics3d.diagnostic_render import render_field_slice
from shoreloop.physics3d.cut_cells import initialize_scene_cut_cells
from shoreloop.physics3d.pcg import PCGConfig
from shoreloop.physics3d.projection import ProjectionConfig
from shoreloop.physics3d.relaxation import WaveRelaxationZone
from shoreloop.physics3d.solver import Solver3D
from shoreloop.physics3d.solids import PlaneBeach
from shoreloop.physics3d.state import State3D
from shoreloop.physics3d.timestep import TimeStepConfig
from shoreloop.physics3d.topology import (BreakingTopologyDiagnostics,
                                          TopologyTracker)
from shoreloop.physics3d.vof import initialize_box, measure_vof
from shoreloop.physics3d.waves import (LinearWaveInlet, PeriodicWaveComponent)
from shoreloop.physics3d.wet_dry import WispRemover
from shoreloop.scene.geometry3d import SceneGeometry


@dataclass(frozen=True)
class ShoalingWaveMetrics:
    refinement: int
    cells: int
    dx: float
    dz: float
    period: float
    target_amplitude: float
    incident_crest: float
    maximum_shoaling_crest: float
    crest_amplification: float
    maximum_offshore_slope: float
    maximum_shoaling_slope: float
    slope_amplification: float
    steepest_position: float
    steepest_time: float
    celerity_probe_a: float
    celerity_probe_b: float
    crest_time_a: float
    crest_time_b: float
    measured_celerity: float
    volume_source: float
    advection_volume_drift: float
    clipped_volume: float
    wisp_volume_removed: float
    relative_volume_residual: float
    alpha_min: float
    alpha_max: float
    max_divergence: float
    max_pressure_iterations: int
    accepted_steps: int
    minimum_dt: float
    max_courant: float

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(asdict(self), indent=2) + "\n")


@dataclass(frozen=True)
class NaturalBreakerMetrics:
    refinement: int
    cells: int
    dx: float
    dz: float
    period: float
    wavelength: float
    target_amplitude: float
    first_topology_time: float | None
    last_topology_time: float | None
    breaker_position: float | None
    breaker_time: float | None
    maximum_downward_interface_cells: int
    maximum_multi_transition_columns: int
    maximum_vertical_transitions: int
    maximum_consecutive_topology_frames: int
    component_mergers: int
    component_splits: int
    maximum_kinetic_energy: float
    maximum_surface_slope: float
    initial_shoreline: float
    maximum_shoreline: float
    maximum_shoreline_time: float
    post_peak_minimum_shoreline: float
    post_peak_minimum_time: float
    shoreline_advance: float
    shoreline_recession: float
    maximum_swash_velocity: float
    minimum_post_wash_swash_velocity: float
    backwash_start_time: float | None
    maximum_runup: float
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
    minimum_dt: float
    max_courant: float
    topology_samples: int
    diagnostic_frames: int

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(asdict(self), indent=2) + "\n")


def _write_breaker_surface(
    state: State3D,
    output_dir: str,
    topology: BreakingTopologyDiagnostics,
) -> dict[str, float | int | str | None]:
    from shoreloop.render.surface_mesh import extract_state_free_surface_mesh

    raw_mesh = extract_state_free_surface_mesh(state)
    smooth_mesh = extract_state_free_surface_mesh(
        state, smoothing_iterations=2,
    )
    raw_downward = raw_mesh.normals[:, 2] < -0.05
    smooth_downward = smooth_mesh.normals[:, 2] < -0.05
    raw_weight = np.maximum(-raw_mesh.normals[:, 2], 0.0)
    smooth_weight = np.maximum(-smooth_mesh.normals[:, 2], 0.0)
    raw_weight_sum = float(np.sum(raw_weight, dtype=np.float64))
    smooth_weight_sum = float(np.sum(smooth_weight, dtype=np.float64))
    raw_underside_x = (
        float(np.sum(
            raw_weight * raw_mesh.vertices[:, 0], dtype=np.float64,
        ) / raw_weight_sum)
        if raw_weight_sum > 0.0 else None
    )
    smooth_underside_x = (
        float(np.sum(
            smooth_weight * smooth_mesh.vertices[:, 0], dtype=np.float64,
        ) / smooth_weight_sum)
        if smooth_weight_sum > 0.0 else None
    )
    underside_shift = (
        smooth_underside_x - raw_underside_x
        if (raw_underside_x is not None and smooth_underside_x is not None)
        else None
    )
    surface_path = Path(output_dir) / (
        f"breaker_surface_t{topology.time:04.2f}.npz"
    )
    smooth_mesh.write(surface_path)
    return {
        "time": topology.time,
        "path": str(surface_path),
        "downward_interface_cells": topology.downward_interface_cells,
        "multi_transition_columns": topology.multi_transition_columns,
        "raw_vertices": len(raw_mesh.vertices),
        "smoothed_vertices": len(smooth_mesh.vertices),
        "raw_triangles": len(raw_mesh.triangles),
        "smoothed_triangles": len(smooth_mesh.triangles),
        "raw_downward_vertices": int(np.count_nonzero(raw_downward)),
        "smoothed_downward_vertices": int(np.count_nonzero(smooth_downward)),
        "raw_minimum_normal_z": float(np.min(raw_mesh.normals[:, 2])),
        "smoothed_minimum_normal_z": float(np.min(smooth_mesh.normals[:, 2])),
        "raw_underside_x": raw_underside_x,
        "smoothed_underside_x": smooth_underside_x,
        "underside_position_shift": underside_shift,
    }


def _build_beach_solver(
    *,
    amplitude: float,
    refinement: int,
    device: str,
    output_dir: str,
    tank_length: float = 3.6,
) -> tuple[Solver3D, PeriodicWaveComponent, PlaneBeach, float]:
    grid = Grid3DConfig(
        nx=round(tank_length / 0.04) * refinement,
        ny=2,
        nz=24 * refinement,
        Lx=tank_length,
        Ly=0.08,
        Lz=0.48,
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
        amplitude=amplitude,
        gravity=cfg.physics.g,
    )
    beach = PlaneBeach(toe_x=1.35, slope=0.18,
                       base_elevation=grid.z0)
    solver = Solver3D(
        cfg,
        ProjectionConfig(pcg=PCGConfig(relative_tolerance=5.0e-6,
                                       max_iterations=500)),
        TimeStepConfig(dt_max=0.005),
    )
    initialize_box(
        solver.state,
        (grid.x0, grid.y0, grid.z0),
        (grid.x0 + grid.Lx, grid.y0 + grid.Ly, grid.z0 + depth),
    )
    initialize_scene_cut_cells(solver.state, SceneGeometry(beach))
    solver.set_wave_relaxation(
        WaveRelaxationZone(
            cfg, LinearWaveInlet((component,)), component.wavelength,
            timescale=0.02 * period,
        ),
    )
    solver.set_wisp_remover(WispRemover(cfg))
    return solver, component, beach, depth


def _surface(solver: Solver3D, bed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    alpha = solver.state.get_interior("alpha")
    theta = solver.state.get_geometry_interior("theta")
    water_depth = (alpha * theta).mean(axis=1).sum(axis=1) * (
        solver.cfg.grid.dz
    )
    return bed + water_depth, water_depth


def run_shoaling_wave_gate(
    *,
    refinement: int = 1,
    device: str = "cuda:0",
    output_dir: str = "out/wave_tank3d/shoaling-wave",
) -> ShoalingWaveMetrics:
    """Launch one nonlinear packet and measure its shoaling before run-up."""
    if refinement not in (1, 2):
        raise ValueError("shoaling refinement must be 1 or 2")
    amplitude = 0.060
    solver, component, beach, depth = _build_beach_solver(
        amplitude=amplitude,
        refinement=refinement,
        device=device,
        output_dir=output_dir,
    )
    grid = solver.cfg.grid
    x = grid.x0 + (np.arange(grid.nx) + 0.5) * grid.dx
    bed = np.asarray(beach.elevation(x))
    offshore = ((x > component.wavelength + 0.08)
                & (x < beach.toe_x - 0.04))
    still_water_shoreline = beach.toe_x + depth / beach.slope
    shoaling = ((x > beach.toe_x + 0.08)
                & (x < still_water_shoreline - 0.28))
    probe_a = 1.58
    probe_b = 1.98
    probe_a_index = int(np.argmin(np.abs(x - probe_a)))
    probe_b_index = int(np.argmin(np.abs(x - probe_b)))

    flat_component = PeriodicWaveComponent(
        period=component.period,
        depth=depth,
        alongshore_length=grid.Ly,
        amplitude=0.0,
        gravity=solver.cfg.physics.g,
    )
    flat_zone = WaveRelaxationZone(
        solver.cfg, LinearWaveInlet((flat_component,)), component.wavelength,
        timescale=0.05 * component.period,
    )
    generation_end = 2.5 * component.period
    end_time = 4.0 * component.period
    absorbing = False
    times: list[float] = []
    probe_a_series: list[float] = []
    probe_b_series: list[float] = []
    incident_crest = 0.0
    maximum_shoaling_crest = 0.0
    maximum_offshore_slope = 0.0
    maximum_shoaling_slope = 0.0
    steepest_position = beach.toe_x
    steepest_time = 0.0
    source_volume = 0.0
    advection_volume_drift = 0.0
    clipped_volume = 0.0
    wisp_volume_removed = 0.0
    max_divergence = 0.0
    max_pressure_iterations = 0
    minimum_dt = np.inf
    max_courant = 0.0
    initial_volume = measure_vof(solver.state).volume

    while solver.time < end_time - 1.0e-12:
        target_time = end_time if absorbing else generation_end
        timestep = solver.compute_dt()
        dt = min(timestep.dt, target_time - solver.time)
        projection = solver.step(dt, timestep=timestep)
        source_volume += solver.last_relaxation.volume_source
        advection_volume_drift += solver.last_advection.vof.volume_drift
        clipped_volume += solver.last_advection.vof.clipped_volume
        wisp_volume_removed += solver.last_wisp_removal.removed_volume
        max_divergence = max(max_divergence, projection.max_divergence_after)
        max_pressure_iterations = max(
            max_pressure_iterations, projection.pressure_solve.iterations,
        )
        minimum_dt = min(minimum_dt, dt)
        max_courant = max(max_courant, solver.last_advection.vof.max_courant)

        surface, water_depth = _surface(solver, bed)
        elevation = surface - (grid.z0 + depth)
        slope = np.abs(np.gradient(surface, grid.dx))
        resolved_water = water_depth > 2.0 * grid.dz
        offshore_samples = offshore & resolved_water
        shoaling_samples = shoaling & resolved_water
        if np.any(offshore_samples):
            incident_crest = max(
                incident_crest, float(np.max(elevation[offshore_samples])),
            )
            maximum_offshore_slope = max(
                maximum_offshore_slope,
                float(np.max(slope[offshore_samples])),
            )
        if np.any(shoaling_samples):
            maximum_shoaling_crest = max(
                maximum_shoaling_crest,
                float(np.max(elevation[shoaling_samples])),
            )
            masked_slope = np.where(shoaling_samples, slope, -np.inf)
            index = int(np.argmax(masked_slope))
            if masked_slope[index] > maximum_shoaling_slope:
                maximum_shoaling_slope = float(masked_slope[index])
                steepest_position = float(x[index])
                steepest_time = solver.time
        times.append(solver.time)
        probe_a_series.append(float(elevation[probe_a_index]))
        probe_b_series.append(float(elevation[probe_b_index]))

        if not absorbing and abs(solver.time - generation_end) < 1.0e-10:
            solver.set_wave_relaxation(flat_zone)
            absorbing = True

    times_array = np.asarray(times)
    probe_a_values = np.asarray(probe_a_series)
    probe_b_values = np.asarray(probe_b_series)
    first_crest_window = ((times_array > 0.7 * component.period)
                          & (times_array < 2.5 * component.period))
    crest_a_index = int(np.flatnonzero(first_crest_window)[
        np.argmax(probe_a_values[first_crest_window])
    ])
    crest_time_a = float(times_array[crest_a_index])
    paired_window = ((times_array > crest_time_a + 0.12)
                     & (times_array < crest_time_a + 0.8))
    crest_b_index = int(np.flatnonzero(paired_window)[
        np.argmax(probe_b_values[paired_window])
    ])
    crest_time_b = float(times_array[crest_b_index])
    measured_celerity = (x[probe_b_index] - x[probe_a_index]) / (
        crest_time_b - crest_time_a
    )
    final_vof = measure_vof(solver.state)
    final_volume = final_vof.volume
    conservation_residual = (final_volume - initial_volume
                             - source_volume + wisp_volume_removed)
    return ShoalingWaveMetrics(
        refinement=refinement,
        cells=grid.cell_count,
        dx=grid.dx,
        dz=grid.dz,
        period=component.period,
        target_amplitude=amplitude,
        incident_crest=incident_crest,
        maximum_shoaling_crest=maximum_shoaling_crest,
        crest_amplification=maximum_shoaling_crest / incident_crest,
        maximum_offshore_slope=maximum_offshore_slope,
        maximum_shoaling_slope=maximum_shoaling_slope,
        slope_amplification=maximum_shoaling_slope / maximum_offshore_slope,
        steepest_position=steepest_position,
        steepest_time=steepest_time,
        celerity_probe_a=float(x[probe_a_index]),
        celerity_probe_b=float(x[probe_b_index]),
        crest_time_a=crest_time_a,
        crest_time_b=crest_time_b,
        measured_celerity=float(measured_celerity),
        volume_source=source_volume,
        advection_volume_drift=advection_volume_drift,
        clipped_volume=clipped_volume,
        wisp_volume_removed=wisp_volume_removed,
        relative_volume_residual=conservation_residual / initial_volume,
        alpha_min=final_vof.alpha_min,
        alpha_max=final_vof.alpha_max,
        max_divergence=max_divergence,
        max_pressure_iterations=max_pressure_iterations,
        accepted_steps=solver.steps,
        minimum_dt=float(minimum_dt),
        max_courant=max_courant,
    )


def run_natural_breaker_gate(
    *,
    refinement: int = 1,
    device: str = "cuda:0",
    output_dir: str = "out/wave_tank3d/natural-breaker",
    write_frames: bool = False,
    write_surface_meshes: bool = False,
) -> NaturalBreakerMetrics:
    """Run through shoaling, resolved overturning, wash, and backwash onset."""
    if refinement not in (1, 2):
        raise ValueError("breaker refinement must be 1 or 2")
    amplitude = 0.090
    solver, component, beach, depth = _build_beach_solver(
        amplitude=amplitude,
        refinement=refinement,
        device=device,
        output_dir=output_dir,
        tank_length=4.2,
    )
    grid = solver.cfg.grid
    x = grid.x0 + (np.arange(grid.nx) + 0.5) * grid.dx
    bed = np.asarray(beach.elevation(x))
    still_water_shoreline = beach.toe_x + depth / beach.slope
    shoreline_tolerance = 0.25 * grid.dz
    breaker_zone_start = beach.toe_x - 0.12
    breaker_zone_end = still_water_shoreline - 0.45
    shoaling = ((x > breaker_zone_start)
                & (x < breaker_zone_end))

    flat_component = PeriodicWaveComponent(
        period=component.period,
        depth=depth,
        alongshore_length=grid.Ly,
        amplitude=0.0,
        gravity=solver.cfg.physics.g,
    )
    flat_zone = WaveRelaxationZone(
        solver.cfg, LinearWaveInlet((flat_component,)), component.wavelength,
        timescale=0.05 * component.period,
    )
    generation_end = 1.5 * component.period
    end_time = 4.0 * component.period
    sample_interval = 0.02
    next_sample = sample_interval
    absorbing = False
    tracker = TopologyTracker(
        solver.cfg,
        x_bounds=(breaker_zone_start, breaker_zone_end),
    )
    topology_series: list[BreakingTopologyDiagnostics] = [
        tracker.sample(solver.state, time=0.0),
    ]
    frame_paths: list[str] = []
    initial_surface, _ = _surface(solver, bed)
    initial_wet = initial_surface > bed + shoreline_tolerance
    initial_shoreline = float(x[np.flatnonzero(initial_wet)[-1]] + 0.5 * grid.dx)
    shorelines = [initial_shoreline]
    shoreline_times = [0.0]
    swash_velocities = [0.0]
    history: list[dict[str, float | int]] = []
    surface_history: list[dict[str, float | int | str | None]] = []

    first_topology_time: float | None = None
    last_topology_time: float | None = None
    breaker_position: float | None = None
    breaker_time: float | None = None
    maximum_downward = 0
    maximum_multi_columns = 0
    maximum_transitions = 0
    consecutive_topology = 0
    maximum_consecutive_topology = 0
    component_mergers = 0
    component_splits = 0
    maximum_kinetic_energy = topology_series[0].kinetic_energy
    maximum_surface_slope = 0.0
    topology_score = 0
    source_volume = 0.0
    advection_volume_drift = 0.0
    clipped_volume = 0.0
    wisp_volume_removed = 0.0
    max_divergence = 0.0
    max_pressure_iterations = 0
    minimum_dt = np.inf
    max_courant = 0.0
    initial_volume = topology_series[0].water_volume

    while solver.time < end_time - 1.0e-12:
        target_time = min(next_sample, end_time)
        if not absorbing:
            target_time = min(target_time, generation_end)
        timestep = solver.compute_dt()
        dt = min(timestep.dt, target_time - solver.time)
        projection = solver.step(dt, timestep=timestep)
        source_volume += solver.last_relaxation.volume_source
        advection_volume_drift += solver.last_advection.vof.volume_drift
        clipped_volume += solver.last_advection.vof.clipped_volume
        wisp_volume_removed += solver.last_wisp_removal.removed_volume
        max_divergence = max(max_divergence, projection.max_divergence_after)
        max_pressure_iterations = max(
            max_pressure_iterations, projection.pressure_solve.iterations,
        )
        minimum_dt = min(minimum_dt, dt)
        max_courant = max(max_courant, solver.last_advection.vof.max_courant)

        if not absorbing and abs(solver.time - generation_end) < 1.0e-10:
            solver.set_wave_relaxation(flat_zone)
            absorbing = True

        if abs(solver.time - next_sample) < 1.0e-10:
            topology = tracker.sample(
                solver.state,
                time=solver.time,
                rms_divergence=projection.rms_divergence_after,
                max_divergence=projection.max_divergence_after,
                pressure_iterations=projection.pressure_solve.iterations,
            )
            topology_series.append(topology)
            surface, water_depth = _surface(solver, bed)
            resolved = shoaling & (water_depth > grid.dz)
            slope = np.abs(np.gradient(surface, grid.dx))
            elevation = surface - (grid.z0 + depth)
            crest_index = int(np.argmax(np.where(
                resolved, elevation, -np.inf,
            )))
            if np.any(resolved):
                maximum_surface_slope = max(
                    maximum_surface_slope, float(np.max(slope[resolved])),
                )
            wet = water_depth > shoreline_tolerance
            shoreline = float(x[np.flatnonzero(wet)[-1]] + 0.5 * grid.dx)
            shorelines.append(shoreline)
            shoreline_times.append(solver.time)
            alpha = solver.state.get_interior("alpha")
            u = solver.state.get_interior("u")
            cell_u = 0.5 * (u[:-1] + u[1:])
            swash = x > still_water_shoreline - 0.25
            theta = solver.state.get_geometry_interior("theta")
            swash_weight = (alpha * theta)[swash]
            swash_velocity = float(
                np.sum(swash_weight * cell_u[swash], dtype=np.float64)
                / np.sum(swash_weight, dtype=np.float64)
            )
            swash_velocities.append(swash_velocity)
            history.append({
                "time": solver.time,
                "shoreline": shoreline,
                "swash_velocity": swash_velocity,
                "surface_slope": (
                    float(np.max(slope[resolved])) if np.any(resolved) else 0.0
                ),
                "crest_position": float(x[crest_index]),
                "crest_elevation": float(elevation[crest_index]),
                "downward_interface_cells":
                    topology.downward_interface_cells,
                "multi_transition_columns":
                    topology.multi_transition_columns,
                "maximum_vertical_transitions":
                    topology.maximum_vertical_transitions,
            })

            active = (topology.downward_interface_cells > 0
                      and topology.multi_transition_columns > 0)
            if active:
                if first_topology_time is None:
                    first_topology_time = solver.time
                last_topology_time = solver.time
                consecutive_topology += 1
                score = (topology.downward_interface_cells
                         + 4 * topology.multi_transition_columns)
                if score > topology_score and np.any(resolved):
                    topology_score = score
                    masked_slope = np.where(resolved, slope, -np.inf)
                    breaker_index = int(np.argmax(masked_slope))
                    breaker_position = float(x[breaker_index])
                    breaker_time = solver.time
                if write_frames:
                    frame = render_field_slice(
                        solver.state,
                        Path(output_dir) / f"breaker_t{solver.time:04.2f}.png",
                        plane="xz",
                    )
                    frame_paths.append(str(frame.path))
            else:
                consecutive_topology = 0
            if (write_surface_meshes
                    and topology.downward_interface_cells > 0):
                surface_history.append(_write_breaker_surface(
                    solver.state, output_dir, topology,
                ))
            maximum_consecutive_topology = max(
                maximum_consecutive_topology, consecutive_topology,
            )
            maximum_downward = max(
                maximum_downward, topology.downward_interface_cells,
            )
            maximum_multi_columns = max(
                maximum_multi_columns, topology.multi_transition_columns,
            )
            maximum_transitions = max(
                maximum_transitions, topology.maximum_vertical_transitions,
            )
            component_mergers += topology.component_mergers
            component_splits += topology.component_splits
            maximum_kinetic_energy = max(
                maximum_kinetic_energy, topology.kinetic_energy,
            )
            next_sample += sample_interval

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "breaker_topology.json").write_text(
        json.dumps([asdict(sample) for sample in topology_series], indent=2)
        + "\n",
    )
    (output / "breaker_history.json").write_text(
        json.dumps(history, indent=2) + "\n",
    )
    if write_surface_meshes:
        (output / "breaker_surface_history.json").write_text(
            json.dumps(surface_history, indent=2) + "\n",
        )
        strongest_surface = max(
            surface_history,
            key=lambda sample: sample["downward_interface_cells"],
        )
        if (strongest_surface["raw_downward_vertices"] == 0
                or strongest_surface["smoothed_downward_vertices"] == 0):
            raise AssertionError(
                "strongest breaker surface lost its resolved underside"
            )
        if abs(strongest_surface["underside_position_shift"]) >= 0.5 * grid.dx:
            raise AssertionError(
                "constrained smoothing moved the breaker underside by half a cell"
            )
    if write_frames:
        frame = render_field_slice(
            solver.state, output / "breaker_final.png", plane="xz",
        )
        frame_paths.append(str(frame.path))

    shoreline_array = np.asarray(shorelines)
    shoreline_time_array = np.asarray(shoreline_times)
    swash_velocity_array = np.asarray(swash_velocities)
    maximum_shoreline_index = int(np.argmax(shoreline_array))
    maximum_shoreline = float(shoreline_array[maximum_shoreline_index])
    post_peak_index = maximum_shoreline_index + int(np.argmin(
        shoreline_array[maximum_shoreline_index:],
    ))
    post_peak_minimum = float(shoreline_array[post_peak_index])
    maximum_swash_index = int(np.argmax(swash_velocity_array))
    post_wash_index = maximum_swash_index + int(np.argmin(
        swash_velocity_array[maximum_swash_index:],
    ))
    backwash_samples = np.flatnonzero(
        swash_velocity_array[maximum_swash_index:] < 0.0,
    )
    backwash_start_time = None
    if backwash_samples.size:
        backwash_start_time = float(shoreline_time_array[
            maximum_swash_index + int(backwash_samples[0])
        ])
    final_vof = measure_vof(solver.state)
    final_volume = final_vof.volume
    conservation_residual = (final_volume - initial_volume
                             - source_volume + wisp_volume_removed)
    alpha = solver.state.get_interior("alpha")
    solid = solver.state.get_solid_interior().astype(bool)
    solid_water_volume = float(alpha[solid].sum(dtype=np.float64)) * (
        grid.dx * grid.dy * grid.dz
    )
    return NaturalBreakerMetrics(
        refinement=refinement,
        cells=grid.cell_count,
        dx=grid.dx,
        dz=grid.dz,
        period=component.period,
        wavelength=component.wavelength,
        target_amplitude=amplitude,
        first_topology_time=first_topology_time,
        last_topology_time=last_topology_time,
        breaker_position=breaker_position,
        breaker_time=breaker_time,
        maximum_downward_interface_cells=maximum_downward,
        maximum_multi_transition_columns=maximum_multi_columns,
        maximum_vertical_transitions=maximum_transitions,
        maximum_consecutive_topology_frames=maximum_consecutive_topology,
        component_mergers=component_mergers,
        component_splits=component_splits,
        maximum_kinetic_energy=maximum_kinetic_energy,
        maximum_surface_slope=maximum_surface_slope,
        initial_shoreline=initial_shoreline,
        maximum_shoreline=maximum_shoreline,
        maximum_shoreline_time=float(
            shoreline_time_array[maximum_shoreline_index],
        ),
        post_peak_minimum_shoreline=post_peak_minimum,
        post_peak_minimum_time=float(shoreline_time_array[post_peak_index]),
        shoreline_advance=maximum_shoreline - initial_shoreline,
        shoreline_recession=maximum_shoreline - post_peak_minimum,
        maximum_swash_velocity=float(swash_velocity_array[maximum_swash_index]),
        minimum_post_wash_swash_velocity=float(
            swash_velocity_array[post_wash_index],
        ),
        backwash_start_time=backwash_start_time,
        maximum_runup=float(beach.elevation(maximum_shoreline))
        - (grid.z0 + depth),
        volume_source=source_volume,
        advection_volume_drift=advection_volume_drift,
        clipped_volume=clipped_volume,
        wisp_volume_removed=wisp_volume_removed,
        relative_volume_residual=conservation_residual / initial_volume,
        alpha_min=final_vof.alpha_min,
        alpha_max=final_vof.alpha_max,
        solid_water_volume=solid_water_volume,
        max_divergence=max_divergence,
        max_pressure_iterations=max_pressure_iterations,
        accepted_steps=solver.steps,
        minimum_dt=float(minimum_dt),
        max_courant=max_courant,
        topology_samples=len(topology_series),
        diagnostic_frames=len(frame_paths),
    )
