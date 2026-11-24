"""Named AW-1101c planar, drop, and gravity--capillary validation cases."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np

from shoreloop.config import (
    BC,
    BC3DConfig,
    Grid3DConfig,
    Physics3DConfig,
    Solver3DConfig,
)
from shoreloop.physics3d.energy import EnergyLedger
from shoreloop.physics3d.free_surface import (
    FreeSurfaceProjection,
    initialize_hydrostatic_pressure,
)
from shoreloop.physics3d.pcg import PCGConfig
from shoreloop.physics3d.projection import ProjectionConfig
from shoreloop.physics3d.solver import Solver3D
from shoreloop.physics3d.state import State3D
from shoreloop.physics3d.timestep import TimeStepConfig, TimeStepController
from shoreloop.physics3d.vof import (
    initialize_plane,
    initialize_sphere,
    measure_vof,
)


WATER_SURFACE_TENSION = 0.072


@dataclass(frozen=True)
class PlanarCapillaryMetrics:
    grid: tuple[int, int, int]
    surface_area: float
    relative_surface_area_error: float
    max_abs_curvature: float
    max_abs_capillary_pressure: float
    max_abs_pressure: float
    max_abs_velocity: float
    max_divergence: float
    pressure_iterations: int


@dataclass(frozen=True)
class StaticDropMetrics:
    grid: tuple[int, int, int]
    spacing: float
    radius: float
    capillary_limit: float
    mixed_cells: int
    height_function_cells: int
    unresolved_curvature_cells: int
    mean_curvature: float
    expected_curvature: float
    relative_curvature_error: float
    mean_pressure_jump: float
    expected_pressure_jump: float
    relative_pressure_jump_error: float
    peak_parasitic_speed: float
    relative_volume_residual: float
    alpha_min: float
    alpha_max: float
    max_divergence: float
    max_pressure_iterations: int
    initial_surface_energy: float
    final_surface_energy: float
    capillary_exchange: float
    numerical_energy_residual: float
    relative_energy_residual: float


@dataclass(frozen=True)
class StandingWaveMetrics:
    grid: tuple[int, int, int]
    surface_tension: float
    wavenumber: float
    depth: float
    quarter_period_time: float
    measured_angular_frequency: float
    accepted_steps: int
    minimum_dt: float
    capillary_limit: float | None
    relative_volume_residual: float
    alpha_min: float
    alpha_max: float
    max_divergence: float
    max_pressure_iterations: int


@dataclass(frozen=True)
class DispersionPairMetrics:
    grid: tuple[int, int, int]
    gravity_only: StandingWaveMetrics
    gravity_capillary: StandingWaveMetrics
    expected_capillary_frequency_squared: float
    measured_capillary_frequency_squared: float
    relative_capillary_dispersion_error: float


@dataclass(frozen=True)
class SurfaceTensionValidationMetrics:
    story: str
    planar: PlanarCapillaryMetrics
    coarse_drop: StaticDropMetrics
    fine_drop: StaticDropMetrics
    coarse_wave: DispersionPairMetrics
    fine_wave: DispersionPairMetrics

    def write(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(asdict(self), indent=2) + "\n")
        return destination


def _periodic_box() -> BC3DConfig:
    return BC3DConfig(
        xlo=BC.PERIODIC,
        xhi=BC.PERIODIC,
        ylo=BC.PERIODIC,
        yhi=BC.PERIODIC,
        zlo=BC.PERIODIC,
        zhi=BC.PERIODIC,
    )


def run_planar_capillary_case(device: str) -> PlanarCapillaryMetrics:
    grid = Grid3DConfig(
        nx=18, ny=14, nz=20, Lx=0.18, Ly=0.14, Lz=0.20,
    )
    cfg = Solver3DConfig(
        grid=grid,
        physics=Physics3DConfig(g=0.0, surface_tension=WATER_SURFACE_TENSION),
        bc=BC3DConfig(
            xlo=BC.PERIODIC,
            xhi=BC.PERIODIC,
            ylo=BC.PERIODIC,
            yhi=BC.PERIODIC,
            zlo=BC.WALL,
            zhi=BC.WALL,
        ),
        device=device,
    )
    state = State3D(cfg)
    initialize_plane(state, (0.0, 0.0, 1.0), 0.103, samples_per_axis=8)
    projection = FreeSurfaceProjection(
        cfg,
        ProjectionConfig(
            pcg=PCGConfig(relative_tolerance=1.0e-7, max_iterations=300),
            rms_divergence_tolerance=1.0e-5,
        ),
    )
    result = projection.project(state, 0.001)
    maximum_velocity = max(
        float(np.max(np.abs(state.get_interior(name))))
        for name in ("u", "v", "w")
    )
    expected_area = grid.Lx * grid.Ly
    return PlanarCapillaryMetrics(
        grid=grid.shape,
        surface_area=result.surface_area,
        relative_surface_area_error=result.surface_area / expected_area - 1.0,
        max_abs_curvature=result.max_abs_curvature,
        max_abs_capillary_pressure=result.max_abs_capillary_pressure,
        max_abs_pressure=float(np.max(np.abs(state.get_interior("p")))),
        max_abs_velocity=maximum_velocity,
        max_divergence=result.max_divergence_after,
        pressure_iterations=result.pressure_solve.iterations,
    )


def run_static_drop_case(n: int, device: str) -> StaticDropMetrics:
    length = 0.10
    radius = 0.025
    centre = (0.05, 0.05, 0.05)
    grid = Grid3DConfig(
        nx=n, ny=n, nz=n, Lx=length, Ly=length, Lz=length,
    )
    cfg = Solver3DConfig(
        grid=grid,
        physics=Physics3DConfig(g=0.0, surface_tension=WATER_SURFACE_TENSION),
        bc=_periodic_box(),
        device=device,
    )
    solver = Solver3D(
        cfg,
        ProjectionConfig(
            pcg=PCGConfig(relative_tolerance=1.0e-7, max_iterations=1000),
            rms_divergence_tolerance=2.0e-4,
        ),
        TimeStepConfig(dt_max=1.0),
    )
    initialize_sphere(
        solver.state, centre, radius, samples_per_axis=16,
    )
    initial_vof = measure_vof(solver.state)
    liquid_mass = cfg.physics.rho * initial_vof.volume
    ledger = EnergyLedger(cfg)
    solver.set_energy_ledger(ledger)
    initial_energy = ledger.measure(solver.state)
    first_limit = solver.compute_dt()
    peak_speed = 0.0
    maximum_divergence = 0.0
    maximum_iterations = 0
    mean_curvature = 0.0
    mean_pressure = 0.0
    mixed_cells = 0
    height_function_cells = 0
    unresolved_cells = 0

    for step_index in range(8):
        timestep = first_limit if step_index == 0 else solver.compute_dt()
        projection = solver.step(0.25 * timestep.dt, timestep=timestep)
        step_energy = solver.last_energy.final
        peak_speed = max(
            peak_speed,
            math.sqrt(2.0 * step_energy.kinetic / liquid_mass),
        )
        maximum_divergence = max(
            maximum_divergence, projection.max_divergence_after,
        )
        maximum_iterations = max(
            maximum_iterations, projection.pressure_solve.iterations,
        )
        if step_index == 0:
            geometry = solver.projection.pressure_solver.geometry.surface_tension
            slices = solver.state.grid.interior_slices("cell")
            alpha = solver.state.get_interior("alpha")
            mixed = (alpha > 1.0e-6) & (alpha < 1.0 - 1.0e-6)
            curvature = geometry.curvature.numpy()[slices]
            mean_curvature = float(np.mean(curvature[mixed]))
            mixed_cells = geometry.mixed_cells
            height_function_cells = geometry.height_function_cells
            unresolved_cells = geometry.unresolved_cells
            x, y, z = solver.state.grid.coordinates("cell")
            distance = np.sqrt(
                (x[:, None, None] - centre[0]) ** 2
                + (y[None, :, None] - centre[1]) ** 2
                + (z[None, None, :] - centre[2]) ** 2
            )
            pressure = solver.state.get_interior("p")
            mean_pressure = float(np.mean(
                pressure[distance < radius - 2.0 * grid.dx],
            ))

    final_vof = measure_vof(solver.state)
    final_energy = ledger.measure(solver.state)
    expected_curvature = 2.0 / radius
    expected_pressure = WATER_SURFACE_TENSION * expected_curvature
    return StaticDropMetrics(
        grid=grid.shape,
        spacing=grid.dx,
        radius=radius,
        capillary_limit=first_limit.capillary_limit,
        mixed_cells=mixed_cells,
        height_function_cells=height_function_cells,
        unresolved_curvature_cells=unresolved_cells,
        mean_curvature=mean_curvature,
        expected_curvature=expected_curvature,
        relative_curvature_error=mean_curvature / expected_curvature - 1.0,
        mean_pressure_jump=mean_pressure,
        expected_pressure_jump=expected_pressure,
        relative_pressure_jump_error=mean_pressure / expected_pressure - 1.0,
        peak_parasitic_speed=peak_speed,
        relative_volume_residual=(
            final_vof.volume / initial_vof.volume - 1.0
        ),
        alpha_min=final_vof.alpha_min,
        alpha_max=final_vof.alpha_max,
        max_divergence=maximum_divergence,
        max_pressure_iterations=maximum_iterations,
        initial_surface_energy=initial_energy.surface,
        final_surface_energy=final_energy.surface,
        capillary_exchange=ledger.capillary_exchange,
        numerical_energy_residual=ledger.numerical_residual,
        relative_energy_residual=(
            ledger.numerical_residual / initial_energy.surface
        ),
    )


def _standing_wave_alpha(
    grid: Grid3DConfig,
    depth: float,
    amplitude: float,
    wavenumber: float,
) -> np.ndarray:
    x = grid.x0 + (np.arange(grid.nx) + 0.5) * grid.dx
    z = grid.z0 + (np.arange(grid.nz) + 0.5) * grid.dz
    cell_average = amplitude * (
        np.sin(wavenumber * (x + 0.5 * grid.dx))
        - np.sin(wavenumber * (x - 0.5 * grid.dx))
    ) / (wavenumber * grid.dx)
    surface = depth + cell_average
    alpha = np.clip(
        (
            surface[:, None, None]
            - (z[None, None, :] - 0.5 * grid.dz)
        ) / grid.dz,
        0.0,
        1.0,
    )
    return np.broadcast_to(alpha, grid.shape).copy().astype(np.float32)


def _surface_mode(solver: Solver3D, depth: float, wavenumber: float) -> float:
    alpha = solver.state.get_interior("alpha")
    elevation = alpha.mean(axis=1).sum(axis=1) * solver.cfg.grid.dz - depth
    x = solver.state.grid.coordinates("cell")[0]
    return float(2.0 * np.mean(elevation * np.cos(wavenumber * x)))


def run_standing_wave_case(
    n: int,
    surface_tension: float,
    device: str,
) -> StandingWaveMetrics:
    length = 0.03
    spacing = length / n
    grid = Grid3DConfig(
        nx=n, ny=4, nz=n,
        Lx=length, Ly=4.0 * spacing, Lz=length,
    )
    depth = 0.5 * length
    wavenumber = 2.0 * math.pi / length
    cfg = Solver3DConfig(
        grid=grid,
        physics=Physics3DConfig(surface_tension=surface_tension),
        bc=BC3DConfig(
            xlo=BC.PERIODIC,
            xhi=BC.PERIODIC,
            ylo=BC.PERIODIC,
            yhi=BC.PERIODIC,
            zlo=BC.WALL,
            zhi=BC.WALL,
        ),
        device=device,
    )
    solver = Solver3D(
        cfg,
        ProjectionConfig(
            pcg=PCGConfig(relative_tolerance=2.0e-6, max_iterations=500),
            rms_divergence_tolerance=3.0e-4,
        ),
        TimeStepConfig(dt_max=3.0e-4),
    )
    solver.state.set_interior(
        "alpha",
        _standing_wave_alpha(
            grid, depth, 0.2 * spacing, wavenumber,
        ),
    )
    initialize_hydrostatic_pressure(solver.state, depth)
    initial_vof = measure_vof(solver.state)
    previous_time = 0.0
    previous_mode = _surface_mode(solver, depth, wavenumber)
    maximum_divergence = 0.0
    maximum_iterations = 0
    minimum_dt = math.inf
    first_capillary_limit = math.inf
    crossing_time = 0.0

    while crossing_time == 0.0:
        timestep = solver.compute_dt()
        if solver.steps == 0:
            first_capillary_limit = timestep.capillary_limit
        projection = solver.step(timestep.dt, timestep=timestep)
        mode = _surface_mode(solver, depth, wavenumber)
        minimum_dt = min(minimum_dt, timestep.dt)
        maximum_divergence = max(
            maximum_divergence, projection.max_divergence_after,
        )
        maximum_iterations = max(
            maximum_iterations, projection.pressure_solve.iterations,
        )
        if mode <= 0.0:
            crossing_time = (
                previous_time
                - previous_mode * (solver.time - previous_time)
                / (mode - previous_mode)
            )
        previous_time = solver.time
        previous_mode = mode

    final_vof = measure_vof(solver.state)
    return StandingWaveMetrics(
        grid=grid.shape,
        surface_tension=surface_tension,
        wavenumber=wavenumber,
        depth=depth,
        quarter_period_time=crossing_time,
        measured_angular_frequency=math.pi / (2.0 * crossing_time),
        accepted_steps=solver.steps,
        minimum_dt=minimum_dt,
        capillary_limit=(
            None if math.isinf(first_capillary_limit)
            else first_capillary_limit
        ),
        relative_volume_residual=(
            final_vof.volume / initial_vof.volume - 1.0
        ),
        alpha_min=final_vof.alpha_min,
        alpha_max=final_vof.alpha_max,
        max_divergence=maximum_divergence,
        max_pressure_iterations=maximum_iterations,
    )


def run_dispersion_pair(n: int, device: str) -> DispersionPairMetrics:
    gravity = run_standing_wave_case(n, 0.0, device)
    capillary = run_standing_wave_case(n, WATER_SURFACE_TENSION, device)
    expected = (
        WATER_SURFACE_TENSION / 1000.0
        * gravity.wavenumber**3
        * math.tanh(gravity.wavenumber * gravity.depth)
    )
    measured = (
        capillary.measured_angular_frequency**2
        - gravity.measured_angular_frequency**2
    )
    return DispersionPairMetrics(
        grid=gravity.grid,
        gravity_only=gravity,
        gravity_capillary=capillary,
        expected_capillary_frequency_squared=expected,
        measured_capillary_frequency_squared=measured,
        relative_capillary_dispersion_error=measured / expected - 1.0,
    )


def run_surface_tension_validation(
    *,
    device: str = "cuda:0",
) -> SurfaceTensionValidationMetrics:
    """Run the complete focused AW-1101c CUDA gate."""
    return SurfaceTensionValidationMetrics(
        story="AW-1101c balanced single-phase surface tension",
        planar=run_planar_capillary_case(device),
        coarse_drop=run_static_drop_case(32, device),
        fine_drop=run_static_drop_case(48, device),
        coarse_wave=run_dispersion_pair(24, device),
        fine_wave=run_dispersion_pair(48, device),
    )
