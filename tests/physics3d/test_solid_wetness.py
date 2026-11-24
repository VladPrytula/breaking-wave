"""Persistent three-dimensional solid-surface wetness gates."""

import numpy as np

from shoreloop.config import Grid3DConfig, Solver3DConfig
from shoreloop.device import DEVICE
from shoreloop.physics3d.cut_cells import initialize_scene_cut_cells
from shoreloop.physics3d.solid_wetness import (
    SolidWetnessConfig, SolidWetnessModel, _mesh_wetness,
    sample_scene_surface_wetness,
)
from shoreloop.physics3d.state import State3D
from shoreloop.scene.geometry3d import (AnalyticBeach, RoundedBoulder,
                                        SceneGeometry, TriangleMesh)


def test_render_wetness_repeats_across_periodic_y_images():
    grid = Grid3DConfig(nx=2, ny=4, nz=2, Lx=1.0, Ly=2.0, Lz=1.0)
    state = State3D(Solver3DConfig(grid=grid, device=DEVICE))
    wetness = np.broadcast_to(
        np.arange(grid.ny, dtype=np.float32)[None, :, None],
        grid.shape,
    ).copy()
    material = np.ones(grid.shape, dtype=np.int32)
    vertices = np.asarray((
        (0.25, 0.25, 0.25),
        (0.25, 2.25, 0.25),
        (0.25, -1.75, 0.25),
        (0.25, 0.75, 0.25),
        (0.25, 2.75, 0.25),
        (0.25, -1.25, 0.25),
    ))
    mesh = TriangleMesh(
        vertices=vertices,
        triangles=np.asarray(((0, 3, 1), (1, 3, 4)), dtype=np.int32),
        material_id=1,
    )

    sampled = _mesh_wetness(state, mesh, wetness, material)

    np.testing.assert_array_equal(sampled[:3], np.zeros(3, dtype=np.float32))
    np.testing.assert_array_equal(sampled[3:], np.ones(3, dtype=np.float32))


def test_wetness_follows_boulder_contact_and_declared_time_scales():
    grid = Grid3DConfig(
        nx=20, ny=20, nz=20, Lx=1.2, Ly=1.2, Lz=1.2,
    )
    cfg = Solver3DConfig(grid=grid, device=DEVICE)
    state = State3D(cfg)
    boulder = RoundedBoulder(
        centre=(0.6, 0.6, 0.6), radii=(0.28, 0.24, 0.26),
        rotation_xyz=(0.15, -0.11, 0.23), material_id=2,
    )
    scene = SceneGeometry(
        AnalyticBeach(toe_x=2.0, slope=0.0, base_elevation=-1.0),
        (boulder,),
    )
    geometry = initialize_scene_cut_cells(state, scene)
    x, _, _ = state.grid.coordinates("cell")
    contact = (
        (x[:, None, None] < boulder.centre[0])
        & (geometry.theta > 0.0)
    )
    alpha = np.zeros(grid.shape, dtype=np.float32)
    alpha[contact] = 1.0
    state.set_interior("alpha", alpha)
    hydro_before = alpha.copy()

    model = SolidWetnessModel(
        cfg,
        SolidWetnessConfig(
            wetting_time=0.5, drying_time=2.0, contact_threshold=0.05,
        ),
    )
    wet = model.apply(state, 0.5)
    assert wet.contacted_cells > 0
    np.testing.assert_allclose(wet.maximum_wetness, 0.5, rtol=2.0e-6)
    np.testing.assert_array_equal(state.get_interior("alpha"), hydro_before)

    state.zero_("alpha")
    dry = model.apply(state, 2.0)
    np.testing.assert_allclose(dry.maximum_wetness, 0.25, rtol=2.0e-6)
    assert 0.0 <= float(state.get_interior("solid_wetness").min())
    assert float(state.get_interior("solid_wetness").max()) <= 1.0

    samples = sample_scene_surface_wetness(
        state,
        scene,
        (0.0, 1.2, 0.0, 1.2),
        beach_segments=(20, 12),
        boulder_segments=(24, 12),
    )
    boulder_sample = samples[1]
    assert boulder_sample.material_id == 2
    assert boulder_sample.primitive_id == 1
    assert boulder_sample.coordinates.shape == (len(boulder_sample.wetness), 2)
    assert np.all((boulder_sample.coordinates >= 0.0)
                  & (boulder_sample.coordinates <= 1.0))
    assert float(np.max(boulder_sample.wetness)) > 0.0
    meshes = scene.render_meshes(
        (0.0, 1.2, 0.0, 1.2),
        beach_segments=(20, 12),
        boulder_segments=(24, 12),
    )
    assert all(len(sample.wetness) == len(mesh.vertices)
               for sample, mesh in zip(samples, meshes, strict=True))
