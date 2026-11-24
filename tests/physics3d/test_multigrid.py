"""Transfer-operator gate for the regular geometric hierarchy."""

import numpy as np
import pytest
import warp as wp

from shoreloop.config import BC, BC3DConfig, Grid3DConfig, Solver3DConfig
from shoreloop.device import DEVICE
from shoreloop.physics3d import MACOperators
from shoreloop.physics3d.multigrid import (MultigridHierarchy,
                                           MultigridVCycle)


def _field(level, interior: np.ndarray) -> wp.array:
    host = np.zeros(level.grid.cell_shape, dtype=np.float32)
    host[level.grid.interior_slices()] = interior
    return wp.array(host, dtype=wp.float32, device=DEVICE)


def test_multigrid_hierarchy_preserves_constants_and_linear_corrections():
    open_box = BC3DConfig(
        xlo=BC.OPEN, xhi=BC.OPEN,
        ylo=BC.OPEN, yhi=BC.OPEN,
        zlo=BC.OPEN, zhi=BC.OPEN,
    )
    cfg = Solver3DConfig(
        grid=Grid3DConfig(
            nx=16, ny=12, nz=8, Lx=1.6, Ly=0.9, Lz=0.8,
            x0=-0.2, y0=0.1, z0=-0.3,
        ),
        bc=open_box,
        device=DEVICE,
    )
    hierarchy = MultigridHierarchy(cfg)
    assert hierarchy.shapes == ((16, 12, 8), (8, 6, 4), (4, 3, 2))

    fine_level, coarse_level = hierarchy.levels[:2]
    constant = _field(
        fine_level,
        np.full(fine_level.cfg.grid.shape, 2.75, dtype=np.float32),
    )
    restricted = wp.zeros(
        coarse_level.grid.cell_shape, dtype=wp.float32, device=DEVICE,
    )
    hierarchy.restrict(0, constant, restricted)
    np.testing.assert_allclose(
        restricted.numpy()[coarse_level.grid.interior_slices()], 2.75,
        atol=2.0e-6,
    )

    x, y, z = coarse_level.grid.coordinates()
    coarse_linear = (
        0.4 + 1.3 * x[:, None, None]
        - 0.7 * y[None, :, None]
        + 0.2 * z[None, None, :]
    ).astype(np.float32)
    coarse_field = _field(coarse_level, coarse_linear)
    prolonged = wp.zeros(
        fine_level.grid.cell_shape, dtype=wp.float32, device=DEVICE,
    )
    hierarchy.prolong(1, coarse_field, prolonged)

    xf, yf, zf = fine_level.grid.coordinates()
    expected = (
        0.4 + 1.3 * xf[:, None, None]
        - 0.7 * yf[None, :, None]
        + 0.2 * zf[None, None, :]
    )
    actual = prolonged.numpy()[fine_level.grid.interior_slices()]
    np.testing.assert_allclose(
        actual[1:-1, 1:-1, 1:-1],
        expected[1:-1, 1:-1, 1:-1],
        atol=3.0e-6,
    )

    constant_coarse = _field(
        coarse_level,
        np.full(coarse_level.cfg.grid.shape, -1.5, dtype=np.float32),
    )
    hierarchy.prolong(1, constant_coarse, prolonged)
    np.testing.assert_allclose(
        prolonged.numpy()[fine_level.grid.interior_slices()], -1.5,
        atol=2.0e-6,
    )

    with pytest.raises(ValueError, match="even nx, ny, and nz"):
        MultigridHierarchy(Solver3DConfig(
            grid=Grid3DConfig(nx=15, ny=12, nz=8), device=DEVICE,
        ))


def test_v_cycle_reduces_smooth_and_oscillatory_error_repeatably():
    periodic = BC3DConfig(
        xlo=BC.PERIODIC, xhi=BC.PERIODIC,
        ylo=BC.PERIODIC, yhi=BC.PERIODIC,
        zlo=BC.PERIODIC, zhi=BC.PERIODIC,
    )
    for nx, ny, nz in ((16, 8, 8), (32, 16, 16), (64, 32, 32)):
        cfg = Solver3DConfig(
            grid=Grid3DConfig(
                nx=nx, ny=ny, nz=nz, Lx=2.0, Ly=1.0, Lz=1.0,
            ),
            bc=periodic,
            device=DEVICE,
        )
        hierarchy = MultigridHierarchy(cfg)
        level = hierarchy.levels[0]
        x, y, z = level.grid.coordinates()
        ii, jj, kk = np.indices((nx, ny, nz))
        exact = (
            np.sin(2.0 * np.pi * x[:, None, None] / cfg.grid.Lx)
            * np.cos(2.0 * np.pi * y[None, :, None] / cfg.grid.Ly)
            + 0.02 * (-1.0) ** (ii + jj + kk)
        ).astype(np.float32)
        exact_field = _field(level, exact)
        rhs = MACOperators(cfg).pressure_operator(exact_field)

        cycle = MultigridVCycle(cfg)
        first_pressure = wp.zeros(
            level.grid.cell_shape, dtype=wp.float32, device=DEVICE,
        )
        first = cycle.cycle(rhs, first_pressure)
        second_pressure = wp.zeros(
            level.grid.cell_shape, dtype=wp.float32, device=DEVICE,
        )
        second = cycle.cycle(rhs, second_pressure)

        assert first.reduction_factor < 0.45
        assert len(first.level_residuals) == len(cycle.hierarchy.levels)
        np.testing.assert_allclose(
            first_pressure.numpy(), second_pressure.numpy(), atol=2.0e-7,
        )
        assert second.reduction_factor == pytest.approx(
            first.reduction_factor, abs=2.0e-7,
        )
