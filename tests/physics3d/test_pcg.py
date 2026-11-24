"""Manufactured regressions for the matrix-free pressure PCG solver."""

import numpy as np

from shoreloop.config import BC, BC3DConfig, Grid3DConfig, Solver3DConfig
from shoreloop.device import DEVICE
from shoreloop.physics3d import (MACOperators, PCGConfig, PressurePCG,
                                 State3D)


def _centred(values: np.ndarray) -> np.ndarray:
    values64 = values.astype(np.float64)
    return values64 - np.mean(values64)


def test_pcg_recovers_manufactured_periodic_pressure():
    grid_cfg = Grid3DConfig(
        nx=9, ny=8, nz=7,
        Lx=1.8, Ly=1.2, Lz=0.7,
    )
    periodic = BC3DConfig(
        xlo=BC.PERIODIC, xhi=BC.PERIODIC,
        ylo=BC.PERIODIC, yhi=BC.PERIODIC,
        zlo=BC.PERIODIC, zhi=BC.PERIODIC,
    )
    cfg = Solver3DConfig(grid=grid_cfg, bc=periodic, device=DEVICE)
    state = State3D(cfg)
    operators = MACOperators(cfg)
    assert state.device.startswith("cuda")

    x, y, z = state.grid.coordinates("p")
    exact = (
        np.sin(2.0 * np.pi * (x - grid_cfg.x0) / grid_cfg.Lx)[:, None, None]
        + 0.35 * np.cos(4.0 * np.pi * (y - grid_cfg.y0) / grid_cfg.Ly)[None, :, None]
        - 0.2 * np.sin(2.0 * np.pi * (z - grid_cfg.z0) / grid_cfg.Lz)[None, None, :]
        + 3.75
    ).astype(np.float32)
    state.set_interior("p", exact)
    rhs = operators.pressure_operator(state.p)
    rhs_before = rhs.numpy().copy()

    settings = PCGConfig(relative_tolerance=3.0e-6, max_iterations=120)
    result = PressurePCG(cfg, settings).solve(rhs)
    recovered = result.pressure.numpy()[state.grid.interior_slices("p")]

    assert result.converged
    assert 0 < result.iterations <= settings.max_iterations
    assert result.relative_residual <= settings.relative_tolerance
    assert result.absolute_residual > 0.0
    np.testing.assert_array_equal(rhs.numpy(), rhs_before)
    np.testing.assert_allclose(
        _centred(recovered), _centred(exact), rtol=2.0e-5, atol=2.0e-5,
    )
    assert abs(float(np.mean(recovered, dtype=np.float64))) < 2.0e-7


def test_pcg_removes_closed_neumann_null_mode_and_uses_boundary_diagonal():
    grid_cfg = Grid3DConfig(
        nx=8, ny=6, nz=5,
        Lx=1.6, Ly=0.9, Lz=0.5,
    )
    walls = BC3DConfig(
        xlo=BC.WALL, xhi=BC.WALL,
        ylo=BC.WALL, yhi=BC.WALL,
        zlo=BC.WALL, zhi=BC.WALL,
    )
    cfg = Solver3DConfig(grid=grid_cfg, bc=walls, device=DEVICE)
    state = State3D(cfg)
    operators = MACOperators(cfg)
    solver = PressurePCG(
        cfg, PCGConfig(relative_tolerance=3.0e-6, max_iterations=120),
    )
    assert solver.device.startswith("cuda")

    diagonal = solver.diagonal.numpy()[state.grid.interior_slices("cell")]
    inverse_squares = np.array(
        [1.0 / grid_cfg.dx**2, 1.0 / grid_cfg.dy**2,
         1.0 / grid_cfg.dz**2],
    )
    np.testing.assert_allclose(diagonal[0, 0, 0], np.sum(inverse_squares),
                               rtol=2.0e-7)
    np.testing.assert_allclose(diagonal[1, 1, 1], 2.0 * np.sum(inverse_squares),
                               rtol=2.0e-7)

    x, y, z = state.grid.coordinates("p")
    exact = (
        np.cos(np.pi * (x - grid_cfg.x0) / grid_cfg.Lx)[:, None, None]
        - 0.3 * np.cos(2.0 * np.pi * (y - grid_cfg.y0) / grid_cfg.Ly)[None, :, None]
        + 0.15 * np.cos(np.pi * (z - grid_cfg.z0) / grid_cfg.Lz)[None, None, :]
        - 8.0
    ).astype(np.float32)
    state.set_interior("p", exact)
    rhs = operators.pressure_operator(state.p)
    assert operators.cell_inner_product(state.p, rhs) > 0.0

    result = solver.solve(rhs)
    recovered = result.pressure.numpy()[state.grid.interior_slices("p")]

    assert result.converged
    assert result.relative_residual <= solver.settings.relative_tolerance
    np.testing.assert_allclose(
        _centred(recovered), _centred(exact), rtol=2.0e-5, atol=2.0e-5,
    )
    assert abs(float(np.mean(recovered, dtype=np.float64))) < 2.0e-7


def test_multigrid_preconditioning_limits_iteration_growth_under_refinement():
    periodic = BC3DConfig(
        xlo=BC.PERIODIC, xhi=BC.PERIODIC,
        ylo=BC.PERIODIC, yhi=BC.PERIODIC,
        zlo=BC.PERIODIC, zhi=BC.PERIODIC,
    )
    iterations = {"jacobi": [], "multigrid": []}
    solutions = {}
    for shape in ((24, 16, 8), (48, 32, 16)):
        nx, ny, nz = shape
        cfg = Solver3DConfig(
            grid=Grid3DConfig(
                nx=nx, ny=ny, nz=nz, Lx=1.5, Ly=1.0, Lz=0.5,
            ),
            bc=periodic,
            device=DEVICE,
        )
        state = State3D(cfg)
        pressure = np.random.default_rng(403 + nx).standard_normal(
            shape,
        ).astype(np.float32)
        state.set_interior("p", pressure)
        rhs = MACOperators(cfg).pressure_operator(state.p)

        for kind in iterations:
            solver = PressurePCG(
                cfg,
                PCGConfig(
                    relative_tolerance=5.0e-6,
                    max_iterations=240,
                    preconditioner=kind,
                ),
            )
            result = solver.solve(rhs)
            assert result.converged
            assert result.relative_residual < 5.0e-6
            iterations[kind].append(result.iterations)
            solutions[(shape, kind)] = _centred(
                result.pressure.numpy()[state.grid.interior_slices()],
            )

        pressure_difference = np.linalg.norm(
            solutions[(shape, "multigrid")]
            - solutions[(shape, "jacobi")],
        ) / np.linalg.norm(solutions[(shape, "jacobi")])
        assert pressure_difference < 1.0e-3

    jacobi_growth = iterations["jacobi"][1] / iterations["jacobi"][0]
    multigrid_growth = (
        iterations["multigrid"][1] / iterations["multigrid"][0]
    )
    assert iterations["multigrid"][1] < iterations["jacobi"][1]
    assert multigrid_growth < jacobi_growth
