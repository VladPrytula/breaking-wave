"""Atmospheric one-phase pressure projection regressions."""

import numpy as np

from shoreloop.config import BC, BC3DConfig, Grid3DConfig, Solver3DConfig
from shoreloop.device import DEVICE
from shoreloop.physics3d import (FreeSurfaceProjection, PCGConfig,
                                 ProjectionConfig, State3D, initialize_box,
                                 initialize_hydrostatic_pressure)


def test_hydrostatic_initial_pressure_balances_gravity_exactly():
    grid = Grid3DConfig(nx=5, ny=4, nz=10, Lx=0.5, Ly=0.4, Lz=1.0)
    cfg = Solver3DConfig(grid=grid, device=DEVICE)
    state = State3D(cfg)
    surface_elevation = 0.6
    initialize_box(
        state, (0.0, 0.0, 0.0), (0.5, 0.4, surface_elevation),
    )
    initialize_hydrostatic_pressure(state, surface_elevation)
    state.set_interior(
        "w", np.full(state.grid.interior_shape("w"),
                     -cfg.physics.g * 0.01, dtype=np.float32),
    )
    projection = FreeSurfaceProjection(
        cfg,
        ProjectionConfig(
            pcg=PCGConfig(relative_tolerance=2.0e-6, max_iterations=250),
            rms_divergence_tolerance=8.0e-5,
        ),
    )
    result = projection.project(state, 0.01)

    assert result.converged
    assert result.pressure_solve.iterations == 0
    assert result.max_divergence_after < 2.0e-6
    pressure = state.get_interior("p")
    alpha = state.get_interior("alpha")
    np.testing.assert_array_equal(pressure[alpha < 0.5], np.float32(0.0))
    z = grid.z0 + (np.arange(grid.nz) + 0.5) * grid.dz
    expected = cfg.physics.rho * cfg.physics.g * np.maximum(
        surface_elevation - z, 0.0,
    )
    np.testing.assert_allclose(
        pressure[2, 2], expected, rtol=2.0e-7, atol=2.0e-4,
    )


def test_pressure_correction_recovers_hydrostatic_balance_from_perturbation():
    grid = Grid3DConfig(nx=8, ny=6, nz=10, Lx=0.8, Ly=0.6, Lz=1.0)
    cfg = Solver3DConfig(grid=grid, device=DEVICE)
    state = State3D(cfg)
    surface_elevation = 0.6
    initialize_box(
        state, (0.0, 0.0, 0.0), (0.8, 0.6, surface_elevation),
    )
    initialize_hydrostatic_pressure(state, surface_elevation)
    pressure = state.get_interior("p")
    x_phase = np.sin(
        2.0 * np.pi * (np.arange(grid.nx) + 0.5) / grid.nx,
    )
    pressure[:, :, :6] += 25.0 * x_phase[:, None, None]
    state.set_interior("p", pressure)
    state.set_interior(
        "w", np.full(state.grid.interior_shape("w"),
                     -cfg.physics.g * 0.01, dtype=np.float32),
    )
    projection = FreeSurfaceProjection(
        cfg,
        ProjectionConfig(
            pcg=PCGConfig(relative_tolerance=2.0e-6, max_iterations=250),
            rms_divergence_tolerance=8.0e-5,
        ),
    )
    result = projection.project(state, 0.01)

    assert result.converged
    assert result.pressure_solve.iterations > 0
    assert result.max_divergence_after < 2.0e-5
    column = state.get_interior("p")[3, 2, :6]
    np.testing.assert_allclose(
        np.diff(column), -cfg.physics.rho * cfg.physics.g * grid.dz,
        rtol=8.0e-5, atol=3.0e-2,
    )


def test_water_column_projects_hydrostatically_without_air_pressure():
    grid = Grid3DConfig(nx=5, ny=4, nz=10, Lx=0.5, Ly=0.4, Lz=1.0)
    cfg = Solver3DConfig(grid=grid, device=DEVICE)
    state = State3D(cfg)
    initialize_box(state, (0.0, 0.0, 0.0), (0.5, 0.4, 0.6))
    state.set_interior(
        "w", np.full(state.grid.interior_shape("w"),
                     -cfg.physics.g * 0.01, dtype=np.float32),
    )
    projection = FreeSurfaceProjection(
        cfg,
        ProjectionConfig(
            pcg=PCGConfig(relative_tolerance=2.0e-6, max_iterations=250),
            rms_divergence_tolerance=8.0e-5,
        ),
    )
    result = projection.project(state, 0.01)

    assert result.converged
    assert result.active_cells == grid.nx * grid.ny * 6
    assert result.dirichlet_faces > 0
    assert result.rms_divergence_after < result.rms_divergence_before
    pressure = state.get_interior("p")
    alpha = state.get_interior("alpha")
    np.testing.assert_array_equal(pressure[alpha < 0.5], np.float32(0.0))
    column = pressure[2, 2, :6]
    np.testing.assert_allclose(
        np.diff(column), -cfg.physics.rho * cfg.physics.g * grid.dz,
        rtol=8.0e-5, atol=3.0e-2,
    )


def test_disconnected_water_components_do_not_make_pcg_singular():
    periodic = BC3DConfig(
        xlo=BC.PERIODIC, xhi=BC.PERIODIC,
        ylo=BC.PERIODIC, yhi=BC.PERIODIC,
        zlo=BC.WALL, zhi=BC.WALL,
    )
    grid = Grid3DConfig(nx=12, ny=8, nz=8, Lx=1.2, Ly=0.8, Lz=0.8)
    cfg = Solver3DConfig(grid=grid, bc=periodic, device=DEVICE)
    alpha = np.zeros(grid.shape, dtype=np.float32)
    alpha[1:4, 1:4, 1:4] = 1.0
    alpha[7:10, 4:7, 3:6] = 1.0
    rng = np.random.default_rng(208)
    template = State3D(cfg)
    velocities = {
        name: (0.02 * rng.standard_normal(
            template.grid.interior_shape(name),
        )).astype(np.float32)
        for name in ("u", "v", "w")
    }
    pressures = {}
    for preconditioner in ("jacobi", "multigrid"):
        state = State3D(cfg)
        state.set_interior("alpha", alpha)
        for name, velocity in velocities.items():
            state.set_interior(name, velocity)
        projection = FreeSurfaceProjection(
            cfg,
            ProjectionConfig(
                pcg=PCGConfig(
                    relative_tolerance=3.0e-6,
                    max_iterations=300,
                    preconditioner=preconditioner,
                ),
                rms_divergence_tolerance=2.0e-4,
            ),
        )
        result = projection.project(state, 0.01)

        assert result.pressure_solve.converged
        assert result.converged
        assert result.active_cells == 54
        assert result.dirichlet_faces > 0
        assert result.rms_divergence_after <= 2.0e-4
        pressures[preconditioner] = state.get_interior("p")

    np.testing.assert_allclose(
        pressures["multigrid"], pressures["jacobi"], rtol=3.0e-4,
        atol=2.0e-2,
    )
