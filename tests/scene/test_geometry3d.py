import numpy as np

from shoreloop.scene.geometry3d import (
    AnalyticBeach,
    RoundedBoulder,
    SceneGeometry,
    deterministic_boulders,
)


def test_shared_boulder_implicit_and_mesh_classification_agree():
    boulder = RoundedBoulder(
        centre=(1.2, -0.3, 0.7),
        radii=(0.8, 0.55, 0.4),
        rotation_xyz=(0.17, -0.29, 0.41),
        exponent=2.6,
    )
    mesh = boulder.render_mesh(longitude_segments=48, latitude_segments=24)
    rng = np.random.default_rng(501)
    directions = rng.normal(size=(80, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    local = directions / (
        np.sum(np.abs(directions) ** boulder.exponent, axis=1, keepdims=True)
        ** (1.0 / boulder.exponent)
    )
    scales = np.r_[np.full(40, 0.72), np.full(40, 1.28)]
    world = (
        local * np.asarray(boulder.radii) * scales[:, None]
    ) @ boulder.rotation.T + np.asarray(boulder.centre)

    implicit_inside = np.asarray(boulder.signed_distance(world)) < 0.0
    mesh_inside = np.asarray(mesh.contains(world))
    assert np.array_equal(mesh_inside, implicit_inside)
    assert np.max(np.abs(boulder.implicit(mesh.vertices))) < 2.0e-14


def test_scene_materials_bounds_and_beach_mesh_share_primitives():
    beach = AnalyticBeach(toe_x=1.0, slope=0.2, base_elevation=0.05)
    scene = SceneGeometry(beach, deterministic_boulders())
    points = np.array(((0.2, 0.5, 0.0), (2.45, 0.55, 0.22),
                       (0.2, 0.5, 0.3)))
    assert np.array_equal(scene.material_ids(points), (1, 2, 0))

    beach_mesh, *boulder_meshes = scene.render_meshes((0.0, 4.0, 0.0, 1.6))
    assert scene.materials[0].dry_albedo != scene.materials[1].dry_albedo
    for material in scene.materials:
        np.testing.assert_allclose(
            material.wet_albedo,
            0.79 * np.asarray(material.dry_albedo),
        )
        assert material.wet_roughness < material.roughness
    assert np.max(np.abs(
        beach_mesh.vertices[:, 2]
        - beach.elevation(beach_mesh.vertices[:, 0])
    )) == 0.0
    for primitive, mesh in zip(scene.boulders, boulder_meshes, strict=True):
        lower, upper = mesh.bounds
        centre = np.asarray(primitive.centre)
        assert np.all(lower < centre)
        assert np.all(upper > centre)
        assert mesh.material_id == primitive.material_id

    refined = scene.render_meshes(
        (0.0, 4.0, 0.0, 1.6),
        beach_segments=(80, 40),
        boulder_segments=(72, 36),
    )
    assert len(refined[0].triangles) == 2 * 80 * 40
    assert all(len(mesh.triangles) == 2 * 72 * (36 - 1)
               for mesh in refined[1:])


def test_deterministic_cluster_has_two_resolved_internal_gaps():
    first, second, third = deterministic_boulders()

    def gap_width(
        lower: RoundedBoulder,
        upper: RoundedBoulder,
        x: float,
        z: float,
    ) -> float:
        y = np.linspace(lower.centre[1], upper.centre[1], 10_001)
        points = np.column_stack((np.full_like(y, x), y, np.full_like(y, z)))
        lower_solid = np.asarray(lower.signed_distance(points)) < 0.0
        upper_solid = np.asarray(upper.signed_distance(points)) < 0.0
        return float(y[np.flatnonzero(upper_solid)[0]]
                     - y[np.flatnonzero(lower_solid)[-1]])

    assert gap_width(first, second, 2.60, 0.30) > 0.20
    assert gap_width(third, second, 2.90, 0.30) > 0.08
