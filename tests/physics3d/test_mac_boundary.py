"""Boundary ownership regression for the three-dimensional MAC grid."""

import numpy as np
import pytest

from shoreloop.config import BC, BC3DConfig, Grid3DConfig, Solver3DConfig
from shoreloop.device import DEVICE
from shoreloop.physics3d import State3D, fill_boundaries


def test_wall_periodic_and_open_ownership_on_all_axes():
    with pytest.raises(ValueError, match="paired"):
        BC3DConfig(ylo=BC.PERIODIC, yhi=BC.WALL)

    grid_cfg = Grid3DConfig(nx=3, ny=4, nz=5, ng=2)
    bc = BC3DConfig(xlo=BC.WALL, xhi=BC.WALL,
                    ylo=BC.PERIODIC, yhi=BC.PERIODIC,
                    zlo=BC.OPEN, zhi=BC.OPEN)
    state = State3D(Solver3DConfig(grid=grid_cfg, bc=bc, device=DEVICE))
    assert state.device.startswith("cuda")
    grid = state.grid
    ng = grid_cfg.ng

    # Constant cell/tangential fields survive every halo fill. The sole
    # exception is u at its x-normal wall, tested immediately below.
    for value, name in enumerate(state.FIELDS, start=1):
        state.fill_(name, -999.0)
        state.set_interior(
            name,
            np.full(grid.interior_shape(name), float(value), dtype=np.float32),
        )
    fill_boundaries(state)
    np.testing.assert_array_equal(state.p.numpy(), np.float32(1.0))
    np.testing.assert_array_equal(state.alpha.numpy(), np.float32(2.0))
    np.testing.assert_array_equal(state.v.numpy(), np.float32(4.0))
    np.testing.assert_array_equal(state.w.numpy(), np.float32(5.0))

    u = state.u.numpy()
    xlo = grid.boundary_face_index("x", "lo")
    xhi = grid.boundary_face_index("x", "hi")
    np.testing.assert_array_equal(u[xlo], np.float32(0.0))
    np.testing.assert_array_equal(u[xhi], np.float32(0.0))
    for layer in range(ng):
        np.testing.assert_array_equal(u[xlo - 1 - layer], np.float32(-3.0))
        np.testing.assert_array_equal(u[xhi + 1 + layer], np.float32(-3.0))
    np.testing.assert_array_equal(u[xlo + 1:xhi], np.float32(3.0))

    # A periodic normal velocity has one owned face: y-low. y-high duplicates
    # it, and the two ghost layers wrap from the nearest unique opposite faces.
    v_values = np.fromfunction(
        lambda i, j, k: 100.0 * i + 10.0 * j + k,
        grid.interior_shape("v"), dtype=np.float32).astype(np.float32)
    state.fill_("v", -999.0)
    state.set_interior("v", v_values)
    fill_boundaries(state)
    v = state.v.numpy()
    sx = slice(ng, ng + grid_cfg.nx)
    sz = slice(ng, ng + grid_cfg.nz)
    ylo = grid.boundary_face_index("y", "lo")
    yhi = grid.boundary_face_index("y", "hi")
    np.testing.assert_array_equal(v[sx, yhi, sz], v[sx, ylo, sz])
    for layer in range(ng):
        np.testing.assert_array_equal(
            v[sx, ylo - 1 - layer, sz], v[sx, yhi - 1 - layer, sz])
        np.testing.assert_array_equal(
            v[sx, yhi + 1 + layer, sz], v[sx, ylo + 1 + layer, sz])

    # Zero-gradient open ownership includes the physical normal faces and then
    # continues the same value through the ghosts.
    w_values = np.fromfunction(
        lambda i, j, k: 100.0 * i + 10.0 * j + k,
        grid.interior_shape("w"), dtype=np.float32).astype(np.float32)
    state.fill_("w", -999.0)
    state.set_interior("w", w_values)
    fill_boundaries(state)
    w = state.w.numpy()
    sy = slice(ng, ng + grid_cfg.ny)
    zlo = grid.boundary_face_index("z", "lo")
    zhi = grid.boundary_face_index("z", "hi")
    np.testing.assert_array_equal(w[sx, sy, zlo], w[sx, sy, zlo + 1])
    np.testing.assert_array_equal(w[sx, sy, zhi], w[sx, sy, zhi - 1])
    for layer in range(ng):
        np.testing.assert_array_equal(w[sx, sy, zlo - 1 - layer],
                                      w[sx, sy, zlo])
        np.testing.assert_array_equal(w[sx, sy, zhi + 1 + layer],
                                      w[sx, sy, zhi])
