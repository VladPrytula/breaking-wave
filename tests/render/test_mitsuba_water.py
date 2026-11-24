"""Analytic optics and multi-interface gates for Mitsuba mesh water."""

from __future__ import annotations

from dataclasses import replace

import mitsuba as mi
import numpy as np

from shoreloop.closures.ripples import RippleField, RippleSpectrum
from shoreloop.config import Grid3DConfig, GridConfig
from shoreloop.physics3d.cut_cells import generate_cut_cell_geometry
from shoreloop.physics3d.grid import MACGrid
from shoreloop.render.camera import Camera
from shoreloop.render.geometry_alignment import measure_geometry_alignment
from shoreloop.render.mitsuba_water import (
    MitsubaWaterRenderer,
    ReceiverPlane,
    SharedSolidScene,
    bounded_material_coverage,
    _ripple_texcoords,
    _solid_shape,
    periodic_ripple_normal_map,
    ripple_calmness,
    join_collar_weight,
)
from shoreloop.render.raytrace import WaterOptics
from shoreloop.render.shoreline_look import shoreline_material_look
from shoreloop.render.sky import OvercastSky, Sun
from shoreloop.render.surface_mesh import (
    WaterSurfaceMesh,
    extract_free_surface_mesh,
)
from shoreloop.scene.geometry3d import (
    AnalyticBeach,
    RoundedBoulder,
    SceneGeometry,
    deterministic_boulders,
)
from shoreloop.physics3d.solid_wetness import SurfaceWetnessSample


def _empty_attributes(
    vertices: np.ndarray,
    triangles: np.ndarray,
    normals: np.ndarray,
) -> WaterSurfaceMesh:
    return WaterSurfaceMesh(
        vertices=vertices.astype(np.float32),
        triangles=triangles.astype(np.int32),
        normals=normals.astype(np.float32),
        velocity=np.zeros((len(vertices), 3), dtype=np.float32),
        foam=np.zeros(len(vertices), dtype=np.float32),
        vertex_open_boundaries=np.zeros(len(vertices), dtype=np.uint8),
        open_boundary_edges=np.empty((0, 2), dtype=np.int32),
        open_boundary_labels=np.empty(0, dtype=np.uint8),
    )


def _flat_mesh() -> WaterSurfaceMesh:
    vertices = np.asarray((
        (-4.0, -4.0, 0.0),
        (4.0, -4.0, 0.0),
        (4.0, 4.0, 0.0),
        (-4.0, 4.0, 0.0),
    ))
    triangles = np.asarray(((0, 1, 2), (0, 2, 3)))
    normals = np.tile((0.0, 0.0, 1.0), (4, 1))
    return _empty_attributes(vertices, triangles, normals)


def test_periodic_water_uses_local_mesh_instances_without_baked_translation():
    base = _flat_mesh()
    base.vertices[:, 1] *= np.float32(0.75)
    y0 = np.float32(0.17919016)
    y1 = np.nextafter(y0, np.float32(np.inf))
    z0 = np.float32(0.24)
    z1 = np.nextafter(z0, np.float32(np.inf))
    micro = np.asarray((
        (0.4, y0, z0),
        (0.4, y0, z1),
        (0.4, y1, z0),
    ), dtype=np.float32)
    first = len(base.vertices)
    mesh = _empty_attributes(
        np.concatenate((base.vertices, micro)),
        np.concatenate((
            base.triangles,
            np.asarray(((first, first + 1, first + 2),), dtype=np.int32),
        )),
        np.concatenate((
            base.normals,
            np.tile(np.asarray(((-1.0, 0.0, 0.0),), dtype=np.float32), (3, 1)),
        )),
    )
    baked = (
        micro + np.asarray((0.0, 6.0, 0.0))
    ).astype(np.float32)
    assert np.all(np.cross(
        baked[1] - baked[0], baked[2] - baked[0],
    ) == 0.0)

    renderer = MitsubaWaterRenderer(
        mesh,
        ReceiverPlane(z=-0.5, bounds=(-4.0, 4.0, -9.0, 9.0)),
        periodic_y_period=6.0,
        variant="llvm_ad_rgb",
    )

    stored = np.asarray(
        mi.traverse(renderer.water_shape)["vertex_positions"],
    ).reshape((-1, 3))
    np.testing.assert_array_equal(stored, mesh.vertices)
    assert len(renderer.water_instances) == 3
    assert [
        (float(instance.bbox().min[1]), float(instance.bbox().max[1]))
        for instance in renderer.water_instances
    ] == [(-9.0, -3.0), (-3.0, 3.0), (3.0, 9.0)]


def _fresnel(cosine: np.ndarray, ior: float) -> np.ndarray:
    sine_transmitted = np.sqrt(1.0 - cosine * cosine) / ior
    cosine_transmitted = np.sqrt(1.0 - sine_transmitted * sine_transmitted)
    rs = (
        (cosine - ior * cosine_transmitted)
        / (cosine + ior * cosine_transmitted)
    )
    rp = (
        (ior * cosine - cosine_transmitted)
        / (ior * cosine + cosine_transmitted)
    )
    return 0.5 * (rs * rs + rp * rp)


def test_flat_mesh_matches_fresnel_snell_and_actual_beer_path():
    camera = Camera(
        pos=(0.0, -1.4, 1.0),
        yaw_deg=90.0,
        pitch_deg=40.0,
        vfov_deg=24.0,
        width=32,
        height=24,
    )
    sky = OvercastSky()
    optics = WaterOptics(scatter=(0.0, 0.0, 0.0))
    bed_radiance = np.asarray((0.16, 0.21, 0.25), dtype=np.float64)
    receiver = ReceiverPlane(
        z=-0.5,
        bounds=(-4.0, 4.0, -4.0, 4.0),
        albedo=(0.0, 0.0, 0.0),
        # Mitsuba transports radiance across the interface with the physical
        # eta^-2 factor.  This source is specified in water-side radiance so
        # the analytic expression below can use the same external convention
        # as render_fast.
        radiance=tuple((optics.n_water**2 * bed_radiance).tolist()),
    )
    renderer = MitsubaWaterRenderer(
        _flat_mesh(), receiver, camera=camera, sky=sky, optics=optics,
        variant="llvm_ad_rgb",
    )
    image = renderer.render(spp=2048, seed=19)

    direction = camera.ray_dirs().astype(np.float64)
    cosine = -direction[..., 2]
    reflectance = _fresnel(cosine, optics.n_water)
    reflected = direction.copy()
    reflected[..., 2] *= -1.0
    reflected_radiance = sky.radiance(reflected)
    sine_transmitted = np.sqrt(1.0 - cosine * cosine) / optics.n_water
    cosine_transmitted = np.sqrt(1.0 - sine_transmitted * sine_transmitted)
    path_length = -receiver.z / cosine_transmitted
    attenuation = np.exp(
        -path_length[..., None] * np.asarray(optics.kappa),
    )
    expected = (
        reflectance[..., None] * reflected_radiance
        + (1.0 - reflectance[..., None]) * bed_radiance * attenuation
    )

    error = np.abs(image - expected)
    assert float(np.median(error)) < 0.01
    assert float(np.quantile(error, 0.95)) < 0.03


def test_directional_sun_uses_the_shared_scene_to_sun_direction():
    camera = Camera(
        pos=(0.0, 0.0, 2.0),
        yaw_deg=0.0,
        pitch_deg=90.0,
        vfov_deg=24.0,
        width=12,
        height=8,
    )
    black_sky = OvercastSky(
        horizon=(0.0, 0.0, 0.0),
        zenith=(0.0, 0.0, 0.0),
    )
    sun = Sun(
        azimuth_deg=37.0,
        elevation_deg=90.0,
        irradiance=(1.0, 0.5, 0.25),
    )
    water = _flat_mesh()
    water.vertices[:, 0] += 100.0
    renderer = MitsubaWaterRenderer(
        water,
        ReceiverPlane(
            z=0.0,
            bounds=(-4.0, 4.0, -4.0, 4.0),
            albedo=(1.0, 1.0, 1.0),
        ),
        camera=camera,
        sky=black_sky,
        sun=sun,
        variant="llvm_ad_rgb",
    )

    image = renderer.render(spp=32, seed=1001)
    np.testing.assert_allclose(
        image.mean(axis=(0, 1)) / np.asarray(sun.irradiance),
        np.full(3, 1.0 / np.pi),
        rtol=0.02,
        atol=0.0,
    )


def _curled_sheet() -> WaterSurfaceMesh:
    grid = Grid3DConfig(
        nx=48, ny=24, nz=48,
        Lx=2.4, Ly=1.0, Lz=2.4,
        x0=-1.2, y0=-0.5, z0=-1.2,
    )
    axes = tuple(
        origin + (np.arange(count) + 0.5) * spacing
        for origin, count, spacing in zip(
            (grid.x0, grid.y0, grid.z0),
            grid.shape,
            (grid.dx, grid.dy, grid.dz),
            strict=True,
        )
    )
    x, y, z = np.meshgrid(*axes, indexing="ij")
    radius = np.sqrt(x * x + z * z)
    angle = np.arctan2(z, x)
    margin = np.minimum.reduce((
        0.13 - np.abs(radius - 0.72),
        0.31 - np.abs(y),
        (2.55 - np.abs(angle)) * 0.72,
    ))
    alpha = np.clip(0.5 + margin / grid.dx, 0.0, 1.0).astype(np.float32)
    return extract_free_surface_mesh(alpha, grid, smoothing_iterations=2)


def test_overturned_sheet_renders_and_alternates_medium_transitions():
    mesh = _curled_sheet()
    camera = Camera(
        pos=(-2.8, -2.6, 1.5),
        yaw_deg=42.9,
        pitch_deg=21.4,
        vfov_deg=34.0,
        width=48,
        height=36,
    )
    renderer = MitsubaWaterRenderer(
        mesh,
        ReceiverPlane(
            z=-1.15,
            bounds=(-4.0, 4.0, -4.0, 4.0),
            albedo=(0.24, 0.18, 0.12),
        ),
        camera=camera,
        variant="llvm_ad_rgb",
    )

    assert renderer.water_shape.is_medium_transition()
    direction = mi.Vector3f(0.0, 0.0, -1.0)
    origin = mi.Point3f(0.0, 0.0, 1.1)
    signs = []
    heights = []
    for _ in range(4):
        interaction = renderer.scene.ray_intersect(mi.Ray3f(origin, direction))
        assert interaction.is_valid()
        point = np.asarray(interaction.p, dtype=np.float32).reshape(3)
        normal = np.asarray(interaction.n, dtype=np.float32).reshape(3)
        heights.append(float(point[2]))
        signs.append(float(-normal[2]))
        origin = mi.Point3f(
            float(point[0]),
            float(point[1]),
            float(point[2]) - 1.0e-4,
        )

    assert np.all(np.diff(heights) < 0.0)
    assert signs[0] < 0.0
    assert signs[1] > 0.0
    assert signs[2] < 0.0
    assert signs[3] > 0.0

    image = renderer.render(spp=128, seed=23)
    assert np.all(np.isfinite(image))
    assert float(np.quantile(image, 0.95) - np.quantile(image, 0.05)) > 0.1


def test_shared_beach_and_boulders_render_on_aligned_cut_cell_geometry():
    grid = Grid3DConfig(
        nx=104, ny=32, nz=24,
        Lx=4.16, Ly=1.6, Lz=0.48,
    )
    scene = SceneGeometry(
        AnalyticBeach(1.35, 0.18, grid.z0),
        deterministic_boulders(),
    )
    geometry = generate_cut_cell_geometry(scene, MACGrid(grid))
    solids = SharedSolidScene(
        scene,
        (grid.x0, grid.x0 + grid.Lx, grid.y0, grid.y0 + grid.Ly),
        boulder_segments=(24, 12),
        boulder_subdivision_levels=2,
    )
    alignment = measure_geometry_alignment(
        geometry, grid, solids, plane_z=0.37,
    )
    assert alignment.maximum_distance < 0.5 * alignment.cell_width

    _, _, z = np.meshgrid(*(
        origin + (np.arange(count) + 0.5) * spacing
        for origin, count, spacing in zip(
            (grid.x0, grid.y0, grid.z0), grid.shape,
            (grid.dx, grid.dy, grid.dz), strict=True,
        )
    ), indexing="ij")
    alpha = np.clip(
        0.5 + (alignment.plane_z - z) / grid.dz,
        0.0,
        1.0,
    ).astype(np.float32)
    water = extract_free_surface_mesh(
        alpha, grid, open_fraction=geometry.theta,
        smoothing_iterations=1,
    )
    camera = Camera(
        pos=(4.75, -0.65, 1.4),
        yaw_deg=144.0,
        pitch_deg=24.0,
        vfov_deg=32.0,
        width=80,
        height=48,
    )
    renderer = MitsubaWaterRenderer(
        water, solids, camera=camera, variant="llvm_ad_rgb",
    )
    image = renderer.render(spp=32, seed=704)

    assert len(renderer.solid_shapes) == 4
    assert [mesh.material_id for mesh in renderer.solid_meshes] == [1, 2, 2, 2]
    assert np.all(np.isfinite(image))
    assert float(np.quantile(image, 0.95) - np.quantile(image, 0.05)) > 0.05


def test_passive_material_coverage_is_bounded_and_switchable():
    np.testing.assert_array_equal(
        bounded_material_coverage(np.asarray((-0.2, 0.0, 0.35, 1.0, 1.4))),
        np.asarray((0.0, 0.0, 0.35, 1.0, 1.0), dtype=np.float32),
    )

    camera = Camera(
        pos=(0.0, -1.4, 1.0),
        yaw_deg=90.0,
        pitch_deg=40.0,
        vfov_deg=24.0,
        width=32,
        height=24,
    )
    water = replace(
        _flat_mesh(),
        foam=np.asarray((0.0, 0.9, 0.9, 0.0), dtype=np.float32),
    )
    scene = SceneGeometry(
        AnalyticBeach(toe_x=10.0, slope=0.0, base_elevation=-0.5),
    )
    dry_solids = SharedSolidScene(
        scene,
        (-4.0, 4.0, -4.0, 4.0),
        beach_segments=(8, 8),
    )
    beach_mesh = dry_solids.meshes()[0]
    wetness = SurfaceWetnessSample(
        material_id=beach_mesh.material_id,
        primitive_id=0,
        coordinates=scene.beach.surface_coordinates(beach_mesh.vertices),
        wetness=np.ones(len(beach_mesh.vertices), dtype=np.float32),
    )
    solids = replace(dry_solids, surface_wetness=(wetness,))

    images = {}
    renderers = {}
    for name, foam_enabled, wetness_enabled in (
        ("all", True, True),
        ("no_foam", False, True),
        ("no_wetness", True, False),
        ("neither", False, False),
    ):
        renderer = MitsubaWaterRenderer(
            water,
            solids,
            camera=camera,
            foam_enabled=foam_enabled,
            wetness_enabled=wetness_enabled,
            variant="llvm_ad_rgb",
        )
        renderers[name] = renderer
        images[name] = renderer.render(spp=512, seed=705)

    foam_effect = np.mean(np.abs(images["all"] - images["no_foam"]))
    wetness_effect = np.mean(np.abs(images["all"] - images["no_wetness"]))
    assert float(foam_effect) > 0.01
    assert float(wetness_effect) > 0.001
    np.testing.assert_array_equal(
        np.asarray(renderers["all"].water_shape.attribute_buffer("vertex_foam")),
        bounded_material_coverage(water.foam),
    )
    np.testing.assert_array_equal(
        np.asarray(
            renderers["all"].solid_shapes[0].attribute_buffer(
                "vertex_wetness",
            ),
        ),
        np.ones(len(beach_mesh.vertices), dtype=np.float32),
    )
    for renderer in renderers.values():
        np.testing.assert_array_equal(
            renderer.mesh.vertices, water.vertices,
        )


def test_periodic_subgrid_ripples_change_only_calm_water_shading():
    grid = GridConfig(
        nx=256,
        ny=256,
        Lx=8.0,
        Ly=8.0,
        x0=-4.0,
        y0=-4.0,
    )
    spectrum = RippleSpectrum.sample(
        n_modes=24,
        period=2.0,
        band=(4, 18),
        rms_slope=0.12,
        seed=706,
    )
    field = RippleField(
        grid,
        np.full(grid.nx, 0.5),
        spectrum,
        "cpu",
    )
    first = periodic_ripple_normal_map(field, 0.0)
    looped = periodic_ripple_normal_map(field, spectrum.period)
    middle = periodic_ripple_normal_map(field, 0.37 * spectrum.period)

    np.testing.assert_allclose(looped, first, rtol=0.0, atol=2.0e-6)
    assert float(np.max(np.abs(middle - first))) > 0.01
    assert first.shape == (grid.ny, grid.nx + 2, 3)
    decoded = 2.0 * first[:, 1:-1] - 1.0
    slope = np.linalg.norm(decoded[..., :2], axis=-1) / decoded[..., 2]
    assert 0.03 < float(np.sqrt(np.mean(slope * slope))) < 0.25

    texture_mesh = replace(
        _flat_mesh(),
        vertices=np.asarray((
            (grid.x0, grid.y0 - grid.Ly, 0.0),
            (grid.x0, grid.y0, 0.0),
            (grid.x0, grid.y0 + grid.Ly, 0.0),
            (grid.x0, grid.y0 + 2.0 * grid.Ly, 0.0),
        ), dtype=np.float32),
    )
    texcoords = _ripple_texcoords(texture_mesh, grid)
    np.testing.assert_allclose(
        np.diff(texcoords[:, 1]),
        1.0,
        rtol=0.0,
        atol=1.0e-7,
    )

    calmness = ripple_calmness(np.asarray((
        (0.0, 0.0, 1.0),
        (0.0, np.sqrt(1.0 - 0.92**2), 0.92),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, -1.0),
    )))
    assert calmness[0] == 1.0
    assert 0.0 < calmness[1] < 1.0
    assert calmness[2] == 0.0
    assert calmness[3] == 0.0
    np.testing.assert_allclose(
        join_collar_weight(np.asarray((-1.0, 0.0, 0.5, 1.0, 2.0)), (0.0, 1.0)),
        (0.0, 0.0, 0.5, 1.0, 1.0),
    )

    camera = Camera(
        pos=(0.0, -1.4, 1.0),
        yaw_deg=90.0,
        pitch_deg=40.0,
        vfov_deg=24.0,
        width=32,
        height=24,
    )
    water = replace(
        _flat_mesh(),
        foam=np.asarray((0.4, 0.9, 0.9, 0.4), dtype=np.float32),
    )
    receiver = ReceiverPlane(
        z=-0.5,
        bounds=(-4.0, 4.0, -4.0, 4.0),
    )
    smooth = MitsubaWaterRenderer(
        water,
        receiver,
        camera=camera,
        ripple_field=field,
        ripples_enabled=False,
        variant="llvm_ad_rgb",
    )
    rippled = MitsubaWaterRenderer(
        water,
        receiver,
        camera=camera,
        ripple_field=field,
        variant="llvm_ad_rgb",
    )
    joined = MitsubaWaterRenderer(
        water,
        receiver,
        camera=camera,
        ripple_field=field,
        join_fade=(0.0, 4.0),
        variant="llvm_ad_rgb",
    )
    smooth_image = smooth.render(spp=512, seed=706)
    rippled_image = rippled.render(spp=512, seed=706)

    assert float(np.mean(np.abs(rippled_image - smooth_image))) > 0.001
    np.testing.assert_array_equal(rippled.mesh.vertices, water.vertices)
    np.testing.assert_array_equal(
        np.asarray(rippled.water_shape.attribute_buffer("vertex_ripple")),
        np.ones(len(water.vertices), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(joined.water_shape.attribute_buffer("vertex_ripple")),
        np.asarray((0.0, 1.0, 1.0, 0.0), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(joined.water_shape.attribute_buffer("vertex_foam")),
        np.asarray((0.0, 0.9, 0.9, 0.0), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(rippled.water_shape.attribute_buffer("vertex_foam")),
        bounded_material_coverage(water.foam),
    )
    np.testing.assert_array_equal(
        np.asarray(mi.traverse(rippled.water_shape)["vertex_positions"]),
        water.vertices.ravel(),
    )


def test_boulder_normal_detail_preserves_base_frame_at_ray_intersection():
    mi.set_variant("llvm_ad_rgb")
    boulder = RoundedBoulder(
        centre=(0.2, -0.3, 0.1),
        radii=(0.32, 0.25, 0.30),
        rotation_xyz=(0.38, -0.42, 0.65),
    )
    scene = SceneGeometry(AnalyticBeach(10.0, 0.0), (boulder,))
    mesh = scene.render_meshes(
        (-1.0, 1.0, -1.0, 1.0),
        beach_segments=(2, 2), boulder_segments=(24, 12),
    )[1]
    look = shoreline_material_look(scene, mesh, 1)
    material = next(
        value for value in scene.materials
        if value.material_id == mesh.material_id
    )
    triangle = mesh.triangles[137]
    weights = np.asarray((0.2, 0.3, 0.5))
    point = weights @ mesh.vertices[triangle]
    base_normal = weights @ look.base_vertex_normals[triangle]
    base_normal /= np.linalg.norm(base_normal)
    detail_normal = weights @ look.vertex_normals[triangle]
    detail_normal /= np.linalg.norm(detail_normal)
    assert np.linalg.norm(detail_normal - base_normal) > 0.01

    # The world normal must survive interpolation and conversion through a
    # rotated shading frame, with either dry or spatially varying wet material.
    for wetness in (None, np.linspace(0.1, 0.9, len(mesh.vertices))):
        shape = _solid_shape("boulder", mesh, material, wetness, True, look)
        render_scene = mi.load_dict({"type": "scene", "boulder": shape})
        interaction = render_scene.ray_intersect(mi.Ray3f(
            mi.Point3f(point + 0.1 * base_normal),
            mi.Vector3f(-base_normal),
        ))
        assert bool(interaction.is_valid()[0])
        np.testing.assert_allclose(
            np.asarray(interaction.sh_frame.n).ravel(),
            base_normal, rtol=0.0, atol=2.0e-6,
        )
        reference_shape = _solid_shape(
            "reference", mesh, material, wetness, True,
            replace(look, base_vertex_normals=None),
        )
        reference = mi.SurfaceInteraction3f(interaction)
        reference.sh_frame = mi.Frame3f(mi.Vector3f(detail_normal))
        reference.wi = reference.to_local(interaction.to_world(interaction.wi))
        context = mi.BSDFContext()
        # Along the base normal the shadow correction is unity: the response
        # must match the original material in the independently computed frame.
        outgoing = mi.Vector3f(base_normal)
        expected = np.asarray(reference_shape.bsdf().eval(
            context, reference, reference.to_local(outgoing),
        ))
        assert np.all(expected > 0.0)
        np.testing.assert_allclose(
            np.asarray(shape.bsdf().eval(
                context, interaction, interaction.to_local(outgoing),
            )),
            expected, rtol=2.0e-5, atol=1.0e-7,
        )
        # A grazing direction facing the detail has positive response in both
        # frames; native shadowing must attenuate it before the base horizon.
        tangent = detail_normal - np.dot(detail_normal, base_normal) * base_normal
        tangent /= np.linalg.norm(tangent)
        outgoing = mi.Vector3f(tangent + 0.02 * base_normal)
        outgoing /= np.linalg.norm(np.asarray(outgoing))
        corrected = np.asarray(shape.bsdf().eval(
            context, interaction, interaction.to_local(outgoing),
        ))
        uncorrected = np.asarray(reference_shape.bsdf().eval(
            context, reference, reference.to_local(outgoing),
        ))
        assert np.all(corrected > 0.0)
        assert np.all(corrected < uncorrected)
        np.testing.assert_array_equal(
            np.asarray(mi.traverse(shape)["vertex_positions"]),
            mesh.vertices.astype(np.float32).ravel(),
        )


def test_shoreline_look_switches_owned_materials_without_moving_solids():
    camera = Camera(
        pos=(0.0, -1.4, 1.0),
        yaw_deg=90.0,
        pitch_deg=40.0,
        vfov_deg=24.0,
        width=32,
        height=24,
    )
    water = _flat_mesh()
    scene = SceneGeometry(
        AnalyticBeach(toe_x=10.0, slope=0.0, base_elevation=-0.5),
        (RoundedBoulder(
            centre=(0.0, 0.0, -0.22),
            radii=(0.32, 0.25, 0.30),
            rotation_xyz=(0.08, -0.12, 0.25),
        ),),
    )
    common = {
        "geometry": scene,
        "beach_bounds": (-4.0, 4.0, -4.0, 4.0),
        "beach_segments": (8, 8),
        "boulder_segments": (24, 12),
        "beach_texture_resolution": 96,
    }
    renderers = {
        name: MitsubaWaterRenderer(
            water,
            SharedSolidScene(
                **common,
                beach_detail_enabled=beach_enabled,
                boulder_detail_enabled=boulder_enabled,
            ),
            camera=camera,
            variant="llvm_ad_rgb",
        )
        for name, beach_enabled, boulder_enabled in (
            ("all", True, True),
            ("no_beach", False, True),
            ("no_boulder", True, False),
            ("neutral", False, False),
        )
    }

    assert [look is not None for look in renderers["all"].solid_looks] == [
        True, True,
    ]
    assert [
        look is not None for look in renderers["no_beach"].solid_looks
    ] == [False, True]
    assert [
        look is not None for look in renderers["no_boulder"].solid_looks
    ] == [True, False]
    assert [
        look is not None for look in renderers["neutral"].solid_looks
    ] == [False, False]
    np.testing.assert_array_equal(
        renderers["all"].solid_looks[1].vertex_normals,
        renderers["no_beach"].solid_looks[1].vertex_normals,
    )
    np.testing.assert_array_equal(
        renderers["all"].solid_looks[0].normal_map,
        renderers["no_boulder"].solid_looks[0].normal_map,
    )
    for renderer in renderers.values():
        np.testing.assert_array_equal(renderer.mesh.vertices, water.vertices)
        for mesh, shape in zip(
            renderer.solid_meshes, renderer.solid_shapes, strict=True,
        ):
            np.testing.assert_array_equal(
                np.asarray(mi.traverse(shape)["vertex_positions"]),
                mesh.vertices.astype(np.float32).ravel(),
            )

    detailed = renderers["all"].render(spp=128, seed=907)
    neutral = renderers["neutral"].render(spp=128, seed=907)
    assert float(np.mean(np.abs(detailed - neutral))) > 0.001


def test_pebble_microgeometry_keeps_source_solids_and_remaps_beach_wetness():
    water = _flat_mesh()
    scene = SceneGeometry(
        AnalyticBeach(toe_x=10.0, slope=0.0, base_elevation=-0.5),
    )
    common = SharedSolidScene(
        scene,
        (-0.5, 0.5, -0.4, 0.4),
        beach_segments=(4, 4),
        beach_detail_enabled=True,
        beach_texture_resolution=96,
        beach_microgeometry_enabled=True,
    )
    source_mesh = common.meshes()[0]
    coordinates = scene.beach.surface_coordinates(source_mesh.vertices)
    wetness_values = (
        0.2
        + 0.3 * (coordinates[:, 0] + 0.5)
        + 0.4 * (coordinates[:, 1] + 0.4) / 0.8
    ).astype(np.float32)
    wetness = SurfaceWetnessSample(
        material_id=source_mesh.material_id,
        primitive_id=0,
        coordinates=coordinates,
        wetness=wetness_values,
    )
    renderer = MitsubaWaterRenderer(
        water,
        replace(common, surface_wetness=(wetness,)),
        variant="llvm_ad_rgb",
    )

    np.testing.assert_array_equal(
        renderer.solid_meshes[0].vertices, source_mesh.vertices,
    )
    np.testing.assert_array_equal(
        renderer.solid_meshes[0].triangles, source_mesh.triangles,
    )
    assert renderer.beach_microgeometry is not None
    np.testing.assert_array_equal(
        renderer.render_solid_meshes[0].vertices,
        renderer.beach_microgeometry.mesh.vertices,
    )
    assert renderer.solid_looks[0].normal_map is not None
    assert renderer.render_solid_looks[0].normal_map is None
    np.testing.assert_array_equal(
        renderer.solid_looks[0].dry_albedo,
        renderer.render_solid_looks[0].dry_albedo,
    )
    np.testing.assert_array_equal(
        renderer.render_solid_looks[0].texcoords,
        renderer.beach_microgeometry.texcoords,
    )
    target = renderer.beach_microgeometry.mesh.vertices[:, :2]
    expected_wetness = (
        0.2
        + 0.3 * (target[:, 0] + 0.5)
        + 0.4 * (target[:, 1] + 0.4) / 0.8
    ).astype(np.float32)
    np.testing.assert_allclose(
        renderer.render_solid_wetness[0], expected_wetness, atol=1.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray(renderer.solid_shapes[0].attribute_buffer("vertex_wetness")),
        expected_wetness,
        atol=1.0e-6,
    )
    np.testing.assert_array_equal(renderer.mesh.vertices, water.vertices)


def test_boulder_refinement_assigns_captured_wetness_without_changing_source():
    boulder = RoundedBoulder(
        centre=(0.3, -0.5, 0.2), radii=(0.32, 0.25, 0.30),
        rotation_xyz=(0.38, -0.42, 0.65),
    )
    geometry = SceneGeometry(AnalyticBeach(10.0, 0.0), (boulder,))
    solids = SharedSolidScene(
        geometry, (-1.0, 1.0, -1.0, 1.0),
        beach_segments=(2, 2), boulder_segments=(4, 2),
        boulder_detail_enabled=True,
    )
    beach, stone = solids.meshes()
    local = boulder.local_coordinates(stone.vertices) / boulder.radii
    gradient = np.asarray((0.17, 0.11, -0.09))
    values = (0.5 + local @ gradient).astype(np.float32)
    wetness = (
        SurfaceWetnessSample(
            beach.material_id, 0, geometry.beach.surface_coordinates(beach.vertices),
            np.zeros(len(beach.vertices), np.float32),
        ),
        SurfaceWetnessSample(
            stone.material_id, 1, boulder.surface_coordinates(stone.vertices), values,
        ),
    )
    water = _flat_mesh()
    for levels in (0, 2):
        renderer = MitsubaWaterRenderer(
            water, replace(solids, surface_wetness=wetness,
                           boulder_subdivision_levels=levels),
            variant="llvm_ad_rgb",
        )
        np.testing.assert_array_equal(renderer.mesh.vertices, water.vertices)
        np.testing.assert_array_equal(renderer.solid_meshes[1].vertices, stone.vertices)
        np.testing.assert_array_equal(renderer.render_solid_meshes[0].vertices, beach.vertices)
        target = renderer.render_solid_meshes[1]
        assert len(target.triangles) == len(stone.triangles) * 4**levels
        direction = boulder.local_coordinates(target.vertices) / boulder.radii
        projected = direction / np.sum(np.abs(direction), axis=1, keepdims=True)
        expected = 0.5 + projected @ gradient
        np.testing.assert_allclose(
            renderer.render_solid_wetness[1], expected, rtol=0.0, atol=1.2e-7,
        )
        np.testing.assert_allclose(
            np.asarray(renderer.solid_shapes[1].attribute_buffer("vertex_wetness")),
            expected, rtol=0.0, atol=1.2e-7,
        )
        np.testing.assert_array_equal(
            np.asarray(mi.traverse(renderer.solid_shapes[1])["vertex_positions"]),
            target.vertices.astype(np.float32).ravel(),
        )
        np.testing.assert_array_equal(
            renderer.solid_looks[1].dry_albedo,
            renderer.render_solid_looks[1].dry_albedo,
        )
    np.testing.assert_array_equal(wetness[1].wetness, values)
