import numpy as np

from shoreloop.config import Grid3DConfig, Solver3DConfig
from shoreloop.device import DEVICE
from shoreloop.physics3d import MACGrid, State3D


def test_asymmetric_mac_shapes_coordinates_and_index_round_trip():
    cfg = Grid3DConfig(nx=3, ny=4, nz=2,
                       Lx=0.6, Ly=1.2, Lz=0.8,
                       x0=-1.0, y0=2.0, z0=-0.4, ng=1)
    grid = MACGrid(cfg)

    assert grid.interior_shape("cell") == (3, 4, 2)
    assert grid.interior_shape("u") == (4, 4, 2)
    assert grid.interior_shape("v") == (3, 5, 2)
    assert grid.interior_shape("w") == (3, 4, 3)
    assert grid.cell_shape == (5, 6, 4)
    assert grid.u_shape == (6, 6, 4)
    assert grid.v_shape == (5, 7, 4)
    assert grid.w_shape == (5, 6, 5)
    assert grid.interior_slices("u") == (slice(1, 5), slice(1, 5), slice(1, 3))
    assert grid.storage_index("alpha", 2, 3, 1) == (3, 4, 2)
    assert grid.storage_index("u", 3, 3, 1) == (4, 4, 2)

    xc, yc, zc = grid.coordinates("cell")
    np.testing.assert_allclose(xc, (-0.9, -0.7, -0.5))
    np.testing.assert_allclose(yc, (2.15, 2.45, 2.75, 3.05))
    np.testing.assert_allclose(zc, (-0.2, 0.2))
    xu, yu, zu = grid.coordinates("u")
    np.testing.assert_allclose(xu, (-1.0, -0.8, -0.6, -0.4))
    np.testing.assert_allclose(yu, yc)
    np.testing.assert_allclose(zu, zc)
    xv, yv, zv = grid.coordinates("v")
    np.testing.assert_allclose(xv, xc)
    np.testing.assert_allclose(yv, (2.0, 2.3, 2.6, 2.9, 3.2))
    np.testing.assert_allclose(zv, zc)
    xw, yw, zw = grid.coordinates("w")
    np.testing.assert_allclose(xw, xc)
    np.testing.assert_allclose(yw, yc)
    np.testing.assert_allclose(zw, (-0.4, 0.0, 0.4))
    xug, _, _ = grid.coordinates("u", include_ghost=True)
    np.testing.assert_allclose(xug, (-1.2, -1.0, -0.8, -0.6, -0.4, -0.2))

    state = State3D(Solver3DConfig(grid=cfg, device=DEVICE))
    assert state.p.shape == grid.cell_shape
    assert state.alpha.shape == grid.cell_shape
    assert state.u.shape == grid.u_shape
    assert state.v.shape == grid.v_shape
    assert state.w.shape == grid.w_shape
    assert state.porous_storage.shape == grid.cell_shape
    assert state.foam.shape == grid.cell_shape
    assert state.solid_wetness.shape == grid.cell_shape
    assert state.solid.shape == grid.cell_shape
    assert state.theta.shape == grid.cell_shape
    assert state.u_aperture.shape == grid.u_shape
    assert state.v_aperture.shape == grid.v_shape
    assert state.w_aperture.shape == grid.w_shape
    assert state.material.shape == grid.cell_shape
    assert state.solid.dtype != state.p.dtype
    assert all(state.field(name).dtype == state.p.dtype for name in state.FIELDS)

    state.fill_("u", 3.25)
    np.testing.assert_array_equal(state.u.numpy(), np.float32(3.25))
    state.zero_("u")
    np.testing.assert_array_equal(state.u.numpy(), np.float32(0.0))

    diagnostic = np.arange(cfg.nx * cfg.ny * cfg.nz, dtype=np.float32).reshape(
        cfg.nx, cfg.ny, cfg.nz)
    state.set_interior("alpha", diagnostic)
    np.testing.assert_array_equal(state.get_interior("alpha"), diagnostic)
    assert float(state.alpha.numpy().sum()) == float(diagnostic.sum())

    solid = diagnostic.astype(np.int32) % 2
    state.set_solid_interior(solid)
    np.testing.assert_array_equal(state.get_solid_interior(), solid)
    assert state.solid_cells == int(np.count_nonzero(solid))
