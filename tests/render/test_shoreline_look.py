"""Deterministic surface-coordinate gates for the AW-907 shoreline look."""

import numpy as np

from shoreloop.render.shoreline_look import (
    BEACH_MICROGEOMETRY_RELIEF_BOUND,
    BEACH_NORMAL_SLOPE_LIMIT,
    BEACH_PEBBLE_RELIEF_RANGE,
    BEACH_TEXTURE_PERIOD,
    BOULDER_NORMAL_SLOPE_LIMIT,
    STONE_LOOKS,
    _encode_normal_map,
    beach_pebble_microgeometry,
    refine_boulder_render_mesh,
    shoreline_material_look,
)
from shoreloop.scene.geometry3d import (
    AnalyticBeach,
    RoundedBoulder,
    SceneGeometry,
    production_boulders,
)


def test_beach_material_is_stable_bounded_and_fixed_to_surface_coordinates():
    scene = SceneGeometry(
        AnalyticBeach(toe_x=1.75, slope=0.15),
        production_boulders(),
    )
    mesh = scene.beach.render_mesh((0.0, 4.8, 0.0, 6.0), nx=24, ny=16)
    vertices = mesh.vertices.copy()
    first = shoreline_material_look(
        scene, mesh, 0, beach_texture_resolution=512,
    )
    second = shoreline_material_look(
        scene, mesh, 0, beach_texture_resolution=512,
    )

    assert first.mapping == "bitmap"
    np.testing.assert_array_equal(first.dry_albedo, second.dry_albedo)
    np.testing.assert_array_equal(first.wet_albedo, second.wet_albedo)
    np.testing.assert_array_equal(
        first.roughness_weight, second.roughness_weight,
    )
    np.testing.assert_array_equal(first.normal_map, second.normal_map)
    np.testing.assert_array_equal(
        first.relief_height_m, second.relief_height_m,
    )
    np.testing.assert_allclose(
        first.texcoords,
        mesh.vertices[:, :2] / BEACH_TEXTURE_PERIOD,
    )
    assert 0.035 <= float(first.dry_albedo.min())
    assert float(first.dry_albedo.max()) <= 0.32
    assert float(np.std(first.dry_albedo)) > 0.01
    assert 0.0 <= float(first.roughness_weight.min())
    assert float(first.roughness_weight.max()) <= 1.0
    assert float(first.relief_height_m.min()) == 0.0
    assert BEACH_PEBBLE_RELIEF_RANGE[0] < float(
        np.quantile(first.relief_height_m, 0.95)
    )
    assert float(first.relief_height_m.max()) < (
        BEACH_PEBBLE_RELIEF_RANGE[1] + 2.0e-4
    )
    decoded = 2.0 * first.normal_map - 1.0
    np.testing.assert_allclose(
        first.normal_map,
        _encode_normal_map(
            first.relief_height_m,
            BEACH_TEXTURE_PERIOD / first.relief_height_m.shape[0],
        ),
        rtol=0.0,
        atol=1.0e-6,
    )
    slope = np.linalg.norm(decoded[..., :2], axis=-1) / decoded[..., 2]
    assert 0.03 < float(np.sqrt(np.mean(slope * slope)))
    assert float(slope.max()) <= BEACH_NORMAL_SLOPE_LIMIT + 2.0e-6
    np.testing.assert_array_equal(mesh.vertices, vertices)


def test_pebble_relief_deterministically_triangulates_about_the_mean_beach():
    scene = SceneGeometry(AnalyticBeach(toe_x=0.0, slope=0.15))
    bounds = (-0.2, 0.3, -0.25, 0.25)
    base = scene.beach.render_mesh(bounds, nx=5, ny=4)
    look = shoreline_material_look(
        scene, base, 0, beach_texture_resolution=256,
    )
    first = beach_pebble_microgeometry(
        scene.beach, bounds, look.relief_height_m, spacing=0.01,
    )
    second = beach_pebble_microgeometry(
        scene.beach, bounds, look.relief_height_m, spacing=0.01,
    )

    np.testing.assert_array_equal(first.mesh.vertices, second.mesh.vertices)
    np.testing.assert_array_equal(first.mesh.triangles, second.mesh.triangles)
    np.testing.assert_array_equal(first.vertex_normals, second.vertex_normals)
    np.testing.assert_array_equal(first.texcoords, second.texcoords)
    np.testing.assert_array_equal(first.relief_height_m, second.relief_height_m)
    np.testing.assert_allclose(
        first.texcoords,
        first.mesh.vertices[:, :2] / BEACH_TEXTURE_PERIOD,
        rtol=0.0,
        atol=1.0e-7,
    )
    base_elevation = scene.beach.elevation(first.mesh.vertices[:, 0])
    np.testing.assert_allclose(
        first.mesh.vertices[:, 2] - first.relief_height_m,
        base_elevation,
        rtol=0.0,
        atol=5.0e-10,
    )
    assert first.source_relief_mean_m > 0.0
    assert float(first.relief_height_m.min()) < 0.0
    assert float(first.relief_height_m.max()) > 0.0
    assert float(np.max(np.abs(first.relief_height_m))) <= (
        BEACH_MICROGEOMETRY_RELIEF_BOUND
    )
    assert len(first.mesh.triangles) == 2 * 50 * 50
    assert np.all(np.isfinite(first.mesh.vertices))
    assert np.all(np.isfinite(first.vertex_normals))
    np.testing.assert_allclose(
        np.linalg.norm(first.vertex_normals, axis=1), 1.0, atol=1.0e-6,
    )
    assert float(first.vertex_normals[:, 2].min()) > 0.0


def test_five_stone_classes_have_stable_variation_and_bounded_normal_detail():
    boulders = production_boulders()
    scene = SceneGeometry(AnalyticBeach(1.75, 0.15), boulders)
    meshes = scene.render_meshes(
        (0.0, 4.8, 0.0, 6.0),
        beach_segments=(8, 8),
        boulder_segments=(32, 16),
    )[1:]
    assert len(STONE_LOOKS) == len(meshes) == len(boulders) == 5

    albedos = []
    roughness = []
    for primitive_id, (boulder, mesh) in enumerate(
        zip(boulders, meshes, strict=True), start=1,
    ):
        vertices = mesh.vertices.copy()
        look = shoreline_material_look(scene, mesh, primitive_id)
        repeated = shoreline_material_look(scene, mesh, primitive_id)
        assert look.mapping == "constant"
        np.testing.assert_array_equal(look.dry_albedo, repeated.dry_albedo)
        np.testing.assert_array_equal(
            look.vertex_normals, repeated.vertex_normals,
        )
        np.testing.assert_allclose(
            np.linalg.norm(look.vertex_normals, axis=1), 1.0, atol=1.0e-6,
        )

        local = np.asarray(boulder.local_coordinates(mesh.vertices))
        scaled = local / np.asarray(boulder.radii)
        gradient = (
            np.sign(scaled)
            * np.abs(scaled) ** (boulder.exponent - 1.0)
            / np.asarray(boulder.radii)
        ) @ boulder.rotation.T
        gradient /= np.linalg.norm(gradient, axis=1, keepdims=True)
        np.testing.assert_allclose(
            look.base_vertex_normals, gradient, rtol=0.0, atol=6.0e-8,
        )
        np.testing.assert_array_equal(
            look.base_vertex_normals, repeated.base_vertex_normals,
        )
        cosine = np.sum(gradient * look.vertex_normals, axis=1)
        slope = np.sqrt(np.maximum(0.0, 1.0 / cosine**2 - 1.0))
        assert float(slope.max()) <= BOULDER_NORMAL_SLOPE_LIMIT + 2.0e-6
        assert float(slope.max()) > 0.08
        np.testing.assert_array_equal(mesh.vertices, vertices)
        albedos.append(tuple(look.dry_albedo))
        roughness.append(look.dry_roughness[0])

    assert len(set(albedos)) == len(boulders)
    assert len(set(roughness)) == len(boulders)


def test_boulder_refinement_preserves_captured_field_and_analytic_surface():
    boulder = RoundedBoulder(
        centre=(0.3, -0.5, 0.2), radii=(0.32, 0.25, 0.30),
        rotation_xyz=(0.38, -0.42, 0.65),
    )
    # This coarse analytic mesh is an octahedron. Its radial intersection is
    # explicit, providing an independent check of the transferred triangle field.
    source = boulder.render_mesh(longitude_segments=4, latitude_segments=2)
    vertices_before = source.vertices.copy()
    triangles_before = source.triangles.copy()
    local = boulder.local_coordinates(source.vertices) / boulder.radii
    gradient = np.asarray((0.17, 0.11, -0.09))
    source_values = (0.5 + local @ gradient).astype(np.float32)
    values_before = source_values.copy()
    refined, values = refine_boulder_render_mesh(
        boulder, source, source_values, levels=3,
    )
    direction = boulder.local_coordinates(refined.vertices) / boulder.radii
    projected = direction / np.sum(np.abs(direction), axis=1, keepdims=True)
    expected = 0.5 + projected @ gradient
    np.testing.assert_allclose(values, expected, rtol=0.0, atol=1.2e-7)
    np.testing.assert_array_equal(values[:len(source_values)], source_values)
    np.testing.assert_array_equal(refined.vertices[:len(source.vertices)], source.vertices)
    np.testing.assert_array_equal(source.vertices, vertices_before)
    np.testing.assert_array_equal(source.triangles, triangles_before)
    np.testing.assert_array_equal(source_values, values_before)
    assert values.min() >= source_values.min()
    assert values.max() <= source_values.max()
    assert len(refined.triangles) == len(source.triangles) * 4**3
    np.testing.assert_allclose(boulder.implicit(refined.vertices), 0.0, atol=2e-15)

    edges = np.sort(refined.triangles[:, ((0, 1), (1, 2), (2, 0))], axis=-1)
    _, counts = np.unique(edges.reshape(-1, 2), axis=0, return_counts=True)
    assert np.all(counts == 2)
    faces = refined.vertices[refined.triangles]
    outward = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
    assert np.all(np.sum(outward * (faces.mean(axis=1) - boulder.centre), axis=1) > 0)
    for constant in (0.0, 1.0):
        _, mapped = refine_boulder_render_mesh(
            boulder, source, np.full(len(source.vertices), constant, np.float32),
            levels=3,
        )
        np.testing.assert_array_equal(mapped, np.full_like(mapped, constant))
