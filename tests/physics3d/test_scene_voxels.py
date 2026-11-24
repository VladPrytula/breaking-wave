import math

import numpy as np

from shoreloop.config import Grid3DConfig, Solver3DConfig
from shoreloop.device import DEVICE
from shoreloop.physics3d.solids import initialize_scene_voxels
from shoreloop.physics3d.state import State3D
from shoreloop.scene.geometry3d import AnalyticBeach, RoundedBoulder, SceneGeometry


def _boulder_state(n: int) -> tuple[State3D, RoundedBoulder]:
    cfg = Solver3DConfig(
        grid=Grid3DConfig(
            nx=n, ny=n, nz=n, Lx=2.0, Ly=2.0, Lz=2.0,
            x0=-1.0, y0=-1.0, z0=-1.0,
        ),
        device=DEVICE,
    )
    boulder = RoundedBoulder(
        centre=(0.07, -0.04, 0.02), radii=(0.55, 0.42, 0.36),
        rotation_xyz=(0.12, -0.18, 0.31), exponent=2.0,
    )
    state = State3D(cfg)
    initialize_scene_voxels(
        state,
        SceneGeometry(AnalyticBeach(toe_x=4.0, slope=0.0,
                                    base_elevation=-2.0), (boulder,)),
    )
    return state, boulder


def test_voxel_boulder_volume_refines_and_face_masks_align():
    errors = []
    for n in (16, 32):
        state, boulder = _boulder_state(n)
        solid = state.get_solid_interior().astype(bool)
        g = state.cfg.grid
        volume = float(solid.sum()) * g.dx * g.dy * g.dz
        exact = 4.0 * math.pi * np.prod(boulder.radii) / 3.0
        errors.append(abs(volume - exact))

        u_open = state.get_geometry_interior("u_aperture").astype(bool)
        assert not np.any(u_open[1:-1] & (solid[:-1] | solid[1:]))
        centres = np.stack(
            np.meshgrid(*state.grid.coordinates("cell"), indexing="ij"),
            axis=-1,
        )
        assert np.array_equal(solid, boulder.implicit(centres) < 0.0)
        assert np.all(state.get_geometry_interior("theta") == ~solid)
    assert errors[1] < errors[0]
