"""Compatibility and analytic regressions for the regular-grid MAC operators."""

import numpy as np

from shoreloop.config import BC, BC3DConfig, Grid3DConfig, Solver3DConfig
from shoreloop.device import DEVICE
from shoreloop.physics3d import MACOperators, State3D, fill_boundaries


def test_divergence_and_gradient_satisfy_discrete_integration_by_parts():
    grid_cfg = Grid3DConfig(nx=7, ny=5, nz=4, Lx=1.4, Ly=0.75, Lz=0.8,
                           ng=2)
    cfg = Solver3DConfig(grid=grid_cfg, device=DEVICE)
    state = State3D(cfg)
    operators = MACOperators(cfg)
    assert state.device.startswith("cuda")

    rng = np.random.default_rng(103)
    for name in ("p", "u", "v", "w"):
        state.set_interior(
            name,
            rng.standard_normal(state.grid.interior_shape(name)).astype(np.float32),
        )
    fill_boundaries(state)

    div = operators.divergence(state.u, state.v, state.w)
    grad = operators.pressure_gradient(state.p)
    cell_term = operators.cell_inner_product(state.p, div)
    face_term = operators.face_inner_product((state.u, state.v, state.w), grad)
    scale = max(abs(cell_term), abs(face_term), 1.0)

    assert abs(cell_term + face_term) / scale < 2.0e-7


def test_linear_fields_have_constant_interior_gradient_and_divergence():
    grid_cfg = Grid3DConfig(
        nx=6, ny=5, nz=4,
        Lx=0.9, Ly=1.25, Lz=0.8,
        x0=-0.4, y0=0.2, z0=-0.3,
    )
    open_bc = BC3DConfig(
        xlo=BC.OPEN, xhi=BC.OPEN,
        ylo=BC.OPEN, yhi=BC.OPEN,
        zlo=BC.OPEN, zhi=BC.OPEN,
    )
    cfg = Solver3DConfig(grid=grid_cfg, bc=open_bc, device=DEVICE)
    state = State3D(cfg)
    operators = MACOperators(cfg)
    assert operators.device.startswith("cuda")

    px, py, pz = state.grid.coordinates("p")
    pressure_slopes = np.array((0.75, -1.25, 2.0), dtype=np.float32)
    p = (pressure_slopes[0] * px[:, None, None]
         + pressure_slopes[1] * py[None, :, None]
         + pressure_slopes[2] * pz[None, None, :] + 0.4)
    state.set_interior("p", p)

    ux, uy, uz = state.grid.coordinates("u")
    vx, vy, vz = state.grid.coordinates("v")
    wx, wy, wz = state.grid.coordinates("w")
    state.set_interior(
        "u", 0.5 * ux[:, None, None] + 0.1 * uy[None, :, None]
        - 0.2 * uz[None, None, :],
    )
    state.set_interior(
        "v", -0.3 * vx[:, None, None] - 0.75 * vy[None, :, None]
        + 0.15 * vz[None, None, :],
    )
    state.set_interior(
        "w", 0.2 * wx[:, None, None] + 0.05 * wy[None, :, None]
        + 1.25 * wz[None, None, :],
    )
    fill_boundaries(state)

    div = operators.divergence(state.u, state.v, state.w)
    div_interior = div.numpy()[state.grid.interior_slices("cell")]
    np.testing.assert_allclose(
        div_interior[1:-1, 1:-1, 1:-1],
        0.5 - 0.75 + 1.25,
        rtol=2.0e-6,
        atol=2.0e-6,
    )

    gx, gy, gz = (field.numpy()[state.grid.interior_slices(name)]
                  for field, name in zip(operators.pressure_gradient(state.p),
                                         ("u", "v", "w")))
    np.testing.assert_allclose(gx[1:-1], pressure_slopes[0],
                               rtol=2.0e-6, atol=2.0e-6)
    np.testing.assert_allclose(gy[:, 1:-1], pressure_slopes[1],
                               rtol=2.0e-6, atol=2.0e-6)
    np.testing.assert_allclose(gz[:, :, 1:-1], pressure_slopes[2],
                               rtol=2.0e-6, atol=2.0e-6)
    np.testing.assert_array_equal(gx[[0, -1]], np.float32(0.0))
    np.testing.assert_array_equal(gy[:, [0, -1]], np.float32(0.0))
    np.testing.assert_array_equal(gz[:, :, [0, -1]], np.float32(0.0))
