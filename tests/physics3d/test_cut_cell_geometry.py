import math

import numpy as np

from shoreloop.config import Grid3DConfig, Solver3DConfig
from shoreloop.device import DEVICE
from shoreloop.physics3d.cut_cells import (
    generate_cut_cell_geometry,
    initialize_scene_cut_cells,
)
from shoreloop.physics3d.grid import MACGrid
from shoreloop.physics3d.state import State3D
from shoreloop.scene.geometry3d import AnalyticBeach, RoundedBoulder, SceneGeometry


def _grid(n: int) -> MACGrid:
    return MACGrid(Grid3DConfig(nx=n, ny=n, nz=n, Lx=1.0, Ly=1.0, Lz=1.0))


def test_plane_cut_volume_aperture_and_normals_refine():
    beach = AnalyticBeach(toe_x=0.0, slope=0.337, base_elevation=0.173)
    scene = SceneGeometry(beach)
    exact_volume = 0.173 + 0.5 * 0.337
    exact_mid_u_open = 1.0 - (0.173 + 0.5 * 0.337)
    volume_errors = []
    aperture_errors = []
    for n in (12, 24):
        geometry = generate_cut_cell_geometry(scene, _grid(n))
        volume_errors.append(abs(float(np.mean(1.0 - geometry.theta))
                                 - exact_volume))
        aperture_errors.append(abs(
            float(np.mean(geometry.apertures[0][n // 2]))
            - exact_mid_u_open
        ))
        assert np.min(geometry.theta) >= 0.0
        assert np.max(geometry.theta) <= 1.0
        for aperture in geometry.apertures:
            assert np.min(aperture) >= 0.0
            assert np.max(aperture) <= 1.0
        expected_normal = np.array((-beach.slope, 0.0, 1.0))
        expected_normal /= np.linalg.norm(expected_normal)
        np.testing.assert_allclose(
            geometry.boundary_normals[geometry.cut_mask],
            np.broadcast_to(expected_normal,
                            geometry.boundary_normals[geometry.cut_mask].shape),
            atol=2.0e-8,
        )
        assert np.max(np.abs(scene.signed_distance(
            geometry.boundary_centroids[geometry.cut_mask],
        ))) < 2.0e-12
    assert volume_errors[1] < volume_errors[0]
    assert aperture_errors[1] < aperture_errors[0]


def test_sphere_cut_volume_and_central_aperture_refine():
    radius = 0.263
    boulder = RoundedBoulder((0.5, 0.5, 0.5), (radius, radius, radius),
                             exponent=2.0)
    scene = SceneGeometry(
        AnalyticBeach(toe_x=2.0, slope=0.0, base_elevation=-1.0),
        (boulder,),
    )
    exact_volume = 4.0 * math.pi * radius**3 / 3.0
    exact_central_open = 1.0 - math.pi * radius**2
    volume_errors = []
    aperture_errors = []
    for n in (8, 16):
        geometry = generate_cut_cell_geometry(scene, _grid(n))
        volume_errors.append(abs(float(np.mean(1.0 - geometry.theta))
                                 - exact_volume))
        assert np.count_nonzero(geometry.cut_mask) > 0
    assert volume_errors[1] < volume_errors[0]
    # The n=8 centred sphere has fortuitous area-error cancellation. Measure
    # the area over translations within one grid cell, then refine: neither
    # a special lattice phase nor a different physical sphere owns this gate.
    for n in (8, 16, 32):
        errors = []
        for phase in (0, .2, .4, .6, .8):
            translated = RoundedBoulder(
                (.5, .5 + phase / n, .5 + .37 * phase / n),
                (radius, radius, radius), exponent=2.0,
            )
            geometry = generate_cut_cell_geometry(
                SceneGeometry(scene.beach, (translated,)), _grid(n),
            )
            errors.append(float(np.mean(geometry.apertures[0][n // 2]))
                          - exact_central_open)
        aperture_errors.append(float(np.sqrt(np.mean(np.square(errors)))))
    assert aperture_errors[1] < .5 * aperture_errors[0]
    assert aperture_errors[2] < .5 * aperture_errors[1]


def test_cut_geometry_uploads_cell_volumes_and_face_apertures():
    grid = Grid3DConfig(nx=8, ny=6, nz=6, Lx=1.0, Ly=0.75, Lz=0.75)
    state = State3D(Solver3DConfig(grid=grid, device=DEVICE))
    scene = SceneGeometry(AnalyticBeach(0.2, 0.35, 0.08))
    geometry = initialize_scene_cut_cells(state, scene)
    np.testing.assert_array_equal(
        state.get_geometry_interior("theta"), geometry.theta,
    )
    for name, expected in zip(
        ("u_aperture", "v_aperture", "w_aperture"),
        geometry.apertures,
        strict=True,
    ):
        np.testing.assert_array_equal(state.get_geometry_interior(name), expected)
    assert state.solid_cells == int(np.count_nonzero(geometry.solid))
