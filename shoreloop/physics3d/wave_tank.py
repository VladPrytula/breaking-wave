"""Small flat-tank integration gate for one linear finite-depth wave."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from shoreloop.config import (BC, BC3DConfig, Grid3DConfig, Solver3DConfig)
from shoreloop.physics3d.relaxation import WaveRelaxationZone
from shoreloop.physics3d.cut_cells import initialize_scene_cut_cells
from shoreloop.physics3d.pcg import PCGConfig
from shoreloop.physics3d.projection import ProjectionConfig
from shoreloop.physics3d.solver import Solver3D
from shoreloop.physics3d.solids import PlaneBeach
from shoreloop.scene.geometry3d import SceneGeometry
from shoreloop.physics3d.timestep import TimeStepConfig
from shoreloop.physics3d.topology import TopologyTracker
from shoreloop.physics3d.vof import initialize_box, measure_vof
from shoreloop.physics3d.wet_dry import WispRemover
from shoreloop.physics3d.waves import (LinearWaveInlet, PeriodicWaveComponent,
                                       fit_probe_wave_modes)


@dataclass(frozen=True)
class LinearWaveMetrics:
    refinement: int
    cells: int
    dx: float
    wavelength: float
    cells_per_wavelength: float
    target_amplitude: float
    incident_amplitude: float
    amplitude_error: float
    theoretical_phase_speed: float
    measured_phase_speed: float
    phase_speed_error: float
    reflected_amplitude: float
    reflection_coefficient: float
    wave_fit_residual: float
    volume_source: float
    advection_volume_drift: float
    clipped_volume: float
    volume_conservation_residual: float
    alpha_min: float
    alpha_max: float
    max_divergence: float
    max_pressure_iterations: int
    accepted_steps: int
    minimum_dt: float
    max_courant: float
    max_downward_interface_cells: int
    max_multi_transition_columns: int
    max_vertical_transitions: int
    max_water_components: int
    topology_events: int

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(asdict(self), indent=2) + "\n")


@dataclass(frozen=True)
class PlaneBeachMetrics:
    cells: int
    dx: float
    period: float
    amplitude: float
    shoreline_depth_tolerance: float
    still_water_shoreline: float
    initial_shoreline: float
    maximum_shoreline: float
    maximum_shoreline_time: float
    post_peak_minimum_shoreline: float
    post_peak_minimum_time: float
    shoreline_advance: float
    shoreline_recession: float
    maximum_runup: float
    final_dry_beach_fraction: float
    final_thin_film_fraction: float
    initial_volume: float
    volume_source: float
    advection_volume_drift: float
    clipped_volume: float
    wisp_volume_removed: float
    volume_conservation_residual: float
    relative_volume_residual: float
    alpha_min: float
    alpha_max: float
    solid_water_volume: float
    max_divergence: float
    max_pressure_iterations: int
    accepted_steps: int
    minimum_dt: float
    max_courant: float

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(asdict(self), indent=2) + "\n")


def _surface_elevation(solver: Solver3D, depth: float) -> np.ndarray:
    alpha = solver.state.get_interior("alpha")
    theta = solver.state.get_geometry_interior("theta")
    return (alpha * theta).sum(axis=2).mean(axis=1) * (
        solver.cfg.grid.dz
    ) - depth


def _column_water_depth(solver: Solver3D) -> np.ndarray:
    alpha = solver.state.get_interior("alpha")
    theta = solver.state.get_geometry_interior("theta")
    return (alpha * theta).mean(axis=1).sum(axis=1) * solver.cfg.grid.dz


def _shoreline_position(solver: Solver3D, depth_tolerance: float) -> float:
    wet = _column_water_depth(solver) > depth_tolerance
    if not np.any(wet):
        return solver.cfg.grid.x0
    last = int(np.flatnonzero(wet)[-1])
    return solver.cfg.grid.x0 + (last + 1) * solver.cfg.grid.dx


def run_linear_wave_gate(
    *,
    refinement: int = 1,
    device: str = "cuda:0",
    output_dir: str = "out/wave_tank3d/linear-wave",
) -> LinearWaveMetrics:
    """Spin up four periods, then fit one period at five probes."""
    if refinement not in (1, 2):
        raise ValueError("linear wave refinement must be 1 or 2")
    dx = 0.04 / refinement
    grid = Grid3DConfig(
        nx=150 * refinement,
        ny=4,
        nz=12 * refinement,
        Lx=6.0,
        Ly=0.16,
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
    amplitude = 0.005
    component = PeriodicWaveComponent(
        period=period,
        depth=depth,
        alongshore_length=grid.Ly,
        amplitude=amplitude,
        gravity=cfg.physics.g,
    )
    solver = Solver3D(
        cfg,
        ProjectionConfig(pcg=PCGConfig(relative_tolerance=5.0e-6,
                                       max_iterations=400)),
        TimeStepConfig(dt_max=0.005),
    )
    initialize_box(
        solver.state,
        (grid.x0, grid.y0, grid.z0),
        (grid.x0 + grid.Lx, grid.y0 + grid.Ly, grid.z0 + depth),
    )
    zone_width = component.wavelength
    solver.set_wave_relaxation(
        WaveRelaxationZone(
            cfg, LinearWaveInlet((component,)), zone_width,
            timescale=0.01 * period,
        ),
    )

    first_probe = round((zone_width + 0.15 * component.wavelength) / dx - 0.5)
    probe_spacing = max(2, round(0.15 * component.wavelength / dx))
    probe_indices = np.arange(first_probe, first_probe + 5 * probe_spacing,
                              probe_spacing)
    probe_positions = grid.x0 + (probe_indices + 0.5) * dx

    sample_interval = 0.02
    spinup_time = 4.0 * period
    end_time = 5.0 * period
    next_sample = spinup_time + sample_interval
    times: list[float] = []
    probe_elevations: list[np.ndarray] = []
    source_volume = 0.0
    advection_volume_drift = 0.0
    clipped_volume = 0.0
    max_divergence = 0.0
    max_pressure_iterations = 0
    minimum_dt = np.inf
    max_courant = 0.0
    max_downward_interface_cells = 0
    max_multi_transition_columns = 0
    max_vertical_transitions = 0
    max_water_components = 0
    topology_events = 0
    topology_tracker = TopologyTracker(cfg)
    initial_volume = measure_vof(solver.state).volume

    while solver.time < end_time - 1.0e-12:
        target_time = min(next_sample, end_time)
        timestep = solver.compute_dt()
        dt = min(timestep.dt, target_time - solver.time)
        projection = solver.step(dt, timestep=timestep)
        minimum_dt = min(minimum_dt, dt)
        max_courant = max(max_courant, solver.last_advection.vof.max_courant)
        advection_volume_drift += solver.last_advection.vof.volume_drift
        clipped_volume += solver.last_advection.vof.clipped_volume
        source_volume += solver.last_relaxation.volume_source
        max_divergence = max(max_divergence, projection.max_divergence_after)
        max_pressure_iterations = max(
            max_pressure_iterations, projection.pressure_solve.iterations,
        )
        if abs(solver.time - next_sample) < 1.0e-10:
            surface = _surface_elevation(solver, depth)
            times.append(solver.time)
            probe_elevations.append(surface[probe_indices].copy())
            topology = topology_tracker.sample(
                solver.state,
                time=solver.time,
                rms_divergence=projection.rms_divergence_after,
                max_divergence=projection.max_divergence_after,
                pressure_iterations=projection.pressure_solve.iterations,
            )
            max_downward_interface_cells = max(
                max_downward_interface_cells,
                topology.downward_interface_cells,
            )
            max_multi_transition_columns = max(
                max_multi_transition_columns,
                topology.multi_transition_columns,
            )
            max_vertical_transitions = max(
                max_vertical_transitions,
                topology.maximum_vertical_transitions,
            )
            max_water_components = max(
                max_water_components, topology.water_components,
            )
            topology_events += (
                topology.component_mergers + topology.component_splits
            )
            next_sample += sample_interval

    times_array = np.asarray(times)
    wave_fit = fit_probe_wave_modes(
        times_array, np.asarray(probe_elevations), probe_positions, component,
    )
    measured_phase_speed = component.omega / wave_fit.wavenumber

    final_vof = measure_vof(solver.state)
    final_volume = final_vof.volume
    return LinearWaveMetrics(
        refinement=refinement,
        cells=grid.cell_count,
        dx=dx,
        wavelength=component.wavelength,
        cells_per_wavelength=component.wavelength / dx,
        target_amplitude=amplitude,
        incident_amplitude=wave_fit.incident_amplitude,
        amplitude_error=(wave_fit.incident_amplitude - amplitude) / amplitude,
        theoretical_phase_speed=component.phase_speed,
        measured_phase_speed=measured_phase_speed,
        phase_speed_error=(measured_phase_speed - component.phase_speed)
        / component.phase_speed,
        reflected_amplitude=wave_fit.reflected_amplitude,
        reflection_coefficient=wave_fit.reflection_coefficient,
        wave_fit_residual=wave_fit.relative_residual,
        volume_source=source_volume,
        advection_volume_drift=advection_volume_drift,
        clipped_volume=clipped_volume,
        volume_conservation_residual=final_volume - initial_volume - source_volume,
        alpha_min=final_vof.alpha_min,
        alpha_max=final_vof.alpha_max,
        max_divergence=max_divergence,
        max_pressure_iterations=max_pressure_iterations,
        accepted_steps=solver.steps,
        minimum_dt=float(minimum_dt),
        max_courant=max_courant,
        max_downward_interface_cells=max_downward_interface_cells,
        max_multi_transition_columns=max_multi_transition_columns,
        max_vertical_transitions=max_vertical_transitions,
        max_water_components=max_water_components,
        topology_events=topology_events,
    )


def run_plane_beach_gate(
    *,
    device: str = "cuda:0",
    output_dir: str = "out/wave_tank3d/plane-beach",
) -> PlaneBeachMetrics:
    """Carry a gentle regular wave through one run-up/recession cycle."""
    grid = Grid3DConfig(nx=90, ny=2, nz=24, Lx=3.6, Ly=0.08, Lz=0.48)
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
    amplitude = 0.040
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
                                       max_iterations=400)),
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
            timescale=0.01 * period,
        ),
    )
    flat_component = PeriodicWaveComponent(
        period=period,
        depth=depth,
        alongshore_length=grid.Ly,
        amplitude=0.0,
        gravity=cfg.physics.g,
    )
    flat_zone = WaveRelaxationZone(
        cfg, LinearWaveInlet((flat_component,)), component.wavelength,
        timescale=0.05 * period,
    )
    wisp_tolerance = 1.0e-6
    solver.set_wisp_remover(WispRemover(cfg, tolerance=wisp_tolerance))

    shoreline_tolerance = 0.75 * grid.dz
    still_water_shoreline = beach.toe_x + (
        grid.z0 + depth - beach.base_elevation
    ) / beach.slope
    initial_shoreline = _shoreline_position(solver, shoreline_tolerance)
    shorelines = [initial_shoreline]
    shoreline_times = [0.0]
    source_volume = 0.0
    advection_volume_drift = 0.0
    clipped_volume = 0.0
    wisp_volume_removed = 0.0
    max_divergence = 0.0
    max_pressure_iterations = 0
    minimum_dt = np.inf
    max_courant = 0.0
    initial_volume = measure_vof(solver.state).volume
    generation_end = 1.5 * period
    end_time = 4.0 * period
    absorbing = False

    while solver.time < end_time - 1.0e-12:
        target_time = end_time if absorbing else generation_end
        timestep = solver.compute_dt()
        dt = min(timestep.dt, target_time - solver.time)
        projection = solver.step(dt, timestep=timestep)
        minimum_dt = min(minimum_dt, dt)
        source_volume += solver.last_relaxation.volume_source
        advection_volume_drift += solver.last_advection.vof.volume_drift
        clipped_volume += solver.last_advection.vof.clipped_volume
        max_courant = max(max_courant, solver.last_advection.vof.max_courant)
        wisp_volume_removed += solver.last_wisp_removal.removed_volume
        max_divergence = max(max_divergence, projection.max_divergence_after)
        max_pressure_iterations = max(
            max_pressure_iterations, projection.pressure_solve.iterations,
        )
        shorelines.append(_shoreline_position(solver, shoreline_tolerance))
        shoreline_times.append(solver.time)
        if not absorbing and abs(solver.time - generation_end) < 1.0e-10:
            solver.set_wave_relaxation(flat_zone)
            absorbing = True

    shoreline_array = np.asarray(shorelines)
    shoreline_time_array = np.asarray(shoreline_times)
    peak_index = int(np.argmax(shoreline_array))
    maximum_shoreline = float(shoreline_array[peak_index])
    post_peak_index = peak_index + int(np.argmin(shoreline_array[peak_index:]))
    post_peak_minimum = float(shoreline_array[post_peak_index])
    final_vof = measure_vof(solver.state)
    final_volume = final_vof.volume
    alpha = solver.state.get_interior("alpha")
    solid = solver.state.get_solid_interior().astype(bool)
    solid_water_volume = float(alpha[solid].sum(dtype=np.float64)) * (
        grid.dx * grid.dy * grid.dz
    )
    theta = solver.state.get_geometry_interior("theta")
    final_depth = (alpha * theta).mean(axis=1).sum(axis=1) * grid.dz
    beach_columns = np.asarray(
        grid.x0 + (np.arange(grid.nx) + 0.5) * grid.dx
        > still_water_shoreline + grid.dx,
    )
    dry = final_depth <= shoreline_tolerance
    thin = ((final_depth > wisp_tolerance * grid.dz)
            & (final_depth < 0.5 * grid.dz))
    beach_column_count = int(np.count_nonzero(beach_columns))
    final_dry_fraction = float(np.count_nonzero(dry & beach_columns)) / (
        beach_column_count
    )
    final_thin_fraction = float(np.count_nonzero(thin & beach_columns)) / (
        beach_column_count
    )
    conservation_residual = (final_volume - initial_volume
                             - source_volume + wisp_volume_removed)
    return PlaneBeachMetrics(
        cells=grid.cell_count,
        dx=grid.dx,
        period=period,
        amplitude=amplitude,
        shoreline_depth_tolerance=shoreline_tolerance,
        still_water_shoreline=still_water_shoreline,
        initial_shoreline=initial_shoreline,
        maximum_shoreline=maximum_shoreline,
        maximum_shoreline_time=float(shoreline_time_array[peak_index]),
        post_peak_minimum_shoreline=post_peak_minimum,
        post_peak_minimum_time=float(shoreline_time_array[post_peak_index]),
        shoreline_advance=maximum_shoreline - initial_shoreline,
        shoreline_recession=maximum_shoreline - post_peak_minimum,
        maximum_runup=float(beach.elevation(maximum_shoreline))
        - (grid.z0 + depth),
        final_dry_beach_fraction=final_dry_fraction,
        final_thin_film_fraction=final_thin_fraction,
        initial_volume=initial_volume,
        volume_source=source_volume,
        advection_volume_drift=advection_volume_drift,
        clipped_volume=clipped_volume,
        wisp_volume_removed=wisp_volume_removed,
        volume_conservation_residual=conservation_residual,
        relative_volume_residual=conservation_residual / initial_volume,
        alpha_min=final_vof.alpha_min,
        alpha_max=final_vof.alpha_max,
        solid_water_volume=solid_water_volume,
        max_divergence=max_divergence,
        max_pressure_iterations=max_pressure_iterations,
        accepted_steps=solver.steps,
        minimum_dt=float(minimum_dt),
        max_courant=max_courant,
    )
