"""Macroscopic rough-wall stress gates."""

import numpy as np
import pytest

from shoreloop.config import (BC, BC3DConfig, Grid3DConfig, Physics3DConfig,
                              Solver3DConfig)
from shoreloop.device import DEVICE
from shoreloop.physics3d.cut_cells import initialize_scene_cut_cells
from shoreloop.physics3d.flow_diagnostics import (cell_velocity,
                                                   turbulence_fields)
from shoreloop.physics3d.pcg import PCGConfig
from shoreloop.physics3d.projection import ProjectionConfig
from shoreloop.physics3d.solver import Solver3D
from shoreloop.physics3d.solids import measure_solid_flux
from shoreloop.physics3d.state import State3D
from shoreloop.physics3d.timestep import TimeStepConfig
from shoreloop.physics3d.vof import initialize_box
from shoreloop.physics3d.wall_stress import WallStressConfig, WallStressModel
from shoreloop.scene.geometry3d import (AnalyticBeach, RoundedBoulder,
                                        SceneGeometry)


def _flow_over_wall(slip_length: float, discretization="ghost_fluid"):
    grid = Grid3DConfig(
        nx=12, ny=6, nz=8, Lx=1.2, Ly=0.6, Lz=0.8,
    )
    cfg = Solver3DConfig(grid=grid, device=DEVICE, pressure_discretization=discretization)
    state = State3D(cfg)
    initialize_box(
        state, (grid.x0, grid.y0, grid.z0),
        (grid.x0 + grid.Lx, grid.y0 + grid.Ly, grid.z0 + grid.Lz),
    )
    initialize_scene_cut_cells(
        state,
        SceneGeometry(AnalyticBeach(toe_x=2.0, slope=0.0,
                                    base_elevation=0.2)),
    )
    state.fill_("u", 1.0)
    state.fill_("v", 0.0)
    state.fill_("w", 0.0)
    model = WallStressModel(
        cfg,
        WallStressConfig(
            wall_viscosity=5.0e-4,
            beach_slip_length=slip_length,
            boulder_slip_length=slip_length,
        ),
    )
    result = model.apply(state, 0.02)
    return state, result


@pytest.mark.parametrize("discretization", ["ghost_fluid", "liquid_mimetic"])
def test_rougher_wall_reduces_slip_without_normal_flux(discretization):
    smooth, smooth_result = _flow_over_wall(0.05, discretization)
    rough, rough_result = _flow_over_wall(0.005, discretization)
    z = rough.grid.coordinates("u")[2]
    first_fluid = int(np.flatnonzero(z > 0.2)[0])
    smooth_slip = float(np.mean(smooth.get_interior("u")[:, :, first_fluid]))
    rough_slip = float(np.mean(rough.get_interior("u")[:, :, first_fluid]))
    assert rough_slip < smooth_slip < 1.0
    assert smooth_result.energy_loss > 0.0
    assert rough_result.energy_loss > smooth_result.energy_loss
    assert rough_result.active_cells > 0
    assert measure_solid_flux(rough).absolute_flux == 0.0


def test_wall_stress_never_injects_energy_on_a_sloped_boulder_scene():
    grid = Grid3DConfig(nx=16, ny=12, nz=10, Lx=1.6, Ly=1.2, Lz=1.0)
    cfg = Solver3DConfig(grid=grid, device=DEVICE)
    state = State3D(cfg)
    state.fill_("alpha", 1.0)
    initialize_scene_cut_cells(
        state,
        SceneGeometry(
            AnalyticBeach(toe_x=0.4, slope=0.12, base_elevation=0.08),
            (RoundedBoulder(
                centre=(1.0, 0.6, 0.42), radii=(0.24, 0.20, 0.28),
                rotation_xyz=(0.13, -0.19, 0.31),
            ),),
        ),
    )
    model = WallStressModel(cfg, WallStressConfig(wall_viscosity=8.0e-4))
    rng = np.random.default_rng(603)
    for _ in range(8):
        for name in ("u", "v", "w"):
            state.set_interior(
                name,
                rng.normal(0.0, 0.6, state.grid.interior_shape(name)).astype(
                    np.float32
                ),
            )
        assert model.apply(state, 0.01).energy_loss >= 0.0


def _short_boulder_wake(slip_length: float) -> tuple[float, float, float]:
    periodic = BC3DConfig(
        xlo=BC.PERIODIC, xhi=BC.PERIODIC,
        ylo=BC.PERIODIC, yhi=BC.PERIODIC,
        zlo=BC.PERIODIC, zhi=BC.PERIODIC,
    )
    grid = Grid3DConfig(nx=24, ny=16, nz=16, Lx=1.2, Ly=0.8, Lz=0.8)
    cfg = Solver3DConfig(
        grid=grid, bc=periodic, physics=Physics3DConfig(g=0.0), device=DEVICE,
    )
    solver = Solver3D(
        cfg,
        ProjectionConfig(
            pcg=PCGConfig(
                relative_tolerance=2.0e-5,
                max_iterations=200,
                preconditioner="multigrid",
            ),
            rms_divergence_tolerance=8.0e-4,
        ),
        TimeStepConfig(dt_max=0.002),
        wall_stress_settings=WallStressConfig(
            wall_viscosity=8.0e-4,
            beach_slip_length=slip_length,
            boulder_slip_length=slip_length,
        ),
    )
    initialize_box(solver.state, (0.0, 0.0, 0.0), (1.2, 0.8, 0.8))
    initialize_scene_cut_cells(
        solver.state,
        SceneGeometry(
            AnalyticBeach(toe_x=2.0, slope=0.0, base_elevation=-1.0),
            (RoundedBoulder(
                centre=(0.45, 0.4, 0.4), radii=(0.16, 0.14, 0.15),
                rotation_xyz=(0.1, -0.15, 0.2),
            ),),
        ),
    )
    solver.state.fill_("u", 0.45)
    for _ in range(40):
        solver.step(0.002)

    velocity = cell_velocity(solver.state)
    x, y, z = solver.state.grid.coordinates("cell")
    wake = (
        (x[:, None, None] > 0.6)
        & (x[:, None, None] < 0.95)
        & (np.abs(y[None, :, None] - 0.4) < 0.22)
        & (np.abs(z[None, None, :] - 0.4) < 0.22)
        & (solver.state.get_geometry_interior("theta") > 0.0)
    )
    fields = turbulence_fields(
        velocity, (grid.dx, grid.dy, grid.dz), (True, True, True),
    )
    return (
        float(np.mean(velocity[0][wake])),
        float(np.min(velocity[0][wake])),
        float(np.mean(fields.vorticity_magnitude[wake])),
    )


def test_roughness_changes_the_short_boulder_wake_predictably():
    smooth_mean, smooth_minimum, smooth_vorticity = _short_boulder_wake(0.05)
    rough_mean, rough_minimum, rough_vorticity = _short_boulder_wake(0.005)
    assert rough_mean < smooth_mean
    assert rough_minimum < smooth_minimum
    assert rough_vorticity > smooth_vorticity
