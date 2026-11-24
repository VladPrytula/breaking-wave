import numpy as np

from shoreloop.config import Grid3DConfig, Solver3DConfig
from shoreloop.device import DEVICE
from shoreloop.physics3d.solids import (PlaneBeach,
                                        initialize_plane_beach)
from shoreloop.physics3d.state import State3D


def test_plane_beach_clears_solid_cells_and_closes_touching_faces():
    cfg = Solver3DConfig(
        grid=Grid3DConfig(nx=8, ny=3, nz=5,
                          Lx=0.8, Ly=0.3, Lz=0.5),
        device=DEVICE,
    )
    state = State3D(cfg)
    state.fill_("p", 4.0)
    state.fill_("alpha", 1.0)
    for name in ("u", "v", "w"):
        state.fill_(name, 2.0)

    count = initialize_plane_beach(
        state, PlaneBeach(toe_x=0.2, slope=0.5),
    )
    solid = state.get_solid_interior().astype(bool)
    assert count == int(solid.sum()) > 0
    assert np.all(state.get_interior("p")[solid] == 0.0)
    assert np.all(state.get_interior("alpha")[solid] == 0.0)

    blocked_u = np.zeros(state.grid.interior_shape("u"), dtype=bool)
    blocked_u[0] = solid[0]
    blocked_u[-1] = solid[-1]
    blocked_u[1:-1] = solid[:-1] | solid[1:]
    blocked_w = np.zeros(state.grid.interior_shape("w"), dtype=bool)
    blocked_w[:, :, 0] = solid[:, :, 0]
    blocked_w[:, :, -1] = solid[:, :, -1]
    blocked_w[:, :, 1:-1] = solid[:, :, :-1] | solid[:, :, 1:]
    assert np.all(state.get_interior("u")[blocked_u] == 0.0)
    assert np.all(state.get_interior("w")[blocked_w] == 0.0)
