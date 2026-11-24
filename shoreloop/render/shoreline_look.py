"""Stable render-only beach and boulder detail for the hero scene.

The fields are evaluated in coordinates owned by the shared analytic
primitives.  Albedo, roughness, shading normals, and the explicitly selected
pebble render mesh all reuse that atlas; analytic solid geometry and persistent
wetness remain authoritative and unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shoreloop.scene.geometry3d import (
    AnalyticBeach,
    BEACH_MATERIAL_ID,
    RoundedBoulder,
    SceneGeometry,
    SceneMaterial,
    TriangleMesh,
)


BEACH_TEXTURE_PERIOD = 8.0
BEACH_TEXTURE_RESOLUTION = 2048
BEACH_PEBBLE_SPACING = 0.045
BEACH_PEBBLE_RELIEF_RANGE = (0.0035, 0.0070)
BEACH_NORMAL_SLOPE_LIMIT = 0.55
BEACH_MICROGEOMETRY_SPACING = (
    2.0 * BEACH_TEXTURE_PERIOD / BEACH_TEXTURE_RESOLUTION
)
BEACH_MICROGEOMETRY_RELIEF_BOUND = 0.0045
BOULDER_NORMAL_SLOPE_LIMIT = 0.18


@dataclass(frozen=True)
class StoneLook:
    """One declared weathered-stone class used by a production boulder."""

    name: str
    albedo_multiplier: tuple[float, float, float]
    dry_roughness: float
    wet_roughness: float


STONE_LOOKS = (
    StoneLook("warm grey", (1.06, 1.00, 0.91), 0.62, 0.27),
    StoneLook("olive grey", (0.88, 0.95, 0.84), 0.71, 0.33),
    StoneLook("pale granite", (1.15, 1.10, 1.00), 0.58, 0.25),
    StoneLook("dark slate", (0.76, 0.84, 0.89), 0.75, 0.35),
    StoneLook("weathered ochre", (1.12, 1.01, 0.83), 0.66, 0.29),
)


@dataclass(frozen=True)
class SolidMaterialLook:
    """Material samples consumed directly by one Mitsuba solid shape.

    Normals are world-space vectors. When base_vertex_normals is present,
    vertex_normals supplies the detailed world normal to the normal-map BSDF.
    """

    mapping: str
    dry_albedo: np.ndarray
    wet_albedo: np.ndarray
    roughness_weight: np.ndarray
    dry_roughness: tuple[float, float]
    wet_roughness: tuple[float, float]
    texcoords: np.ndarray | None = None
    normal_map: np.ndarray | None = None
    relief_height_m: np.ndarray | None = None
    vertex_normals: np.ndarray | None = None
    base_vertex_normals: np.ndarray | None = None


@dataclass(frozen=True)
class BeachMicrogeometry:
    """Render-only beach surface derived from the AW-1001a relief field."""

    mesh: TriangleMesh
    texcoords: np.ndarray
    relief_height_m: np.ndarray
    vertex_normals: np.ndarray
    source_relief_mean_m: float


def refine_boulder_render_mesh(
    boulder: RoundedBoulder,
    mesh: TriangleMesh,
    wetness: np.ndarray | None,
    *,
    levels: int,
) -> tuple[TriangleMesh, np.ndarray | None]:
    """Subdivide the captured triangle field before analytic radial projection.

    Each shared edge creates one midpoint. Repeated linear subdivision carries
    the original barycentric wetness field; projection happens only afterwards,
    so no new wetness is sampled or invented. Original vertices stay exact.
    """
    if levels < 0:
        raise ValueError("boulder subdivision levels must be nonnegative")
    vertices = mesh.vertices
    triangles = mesh.triangles
    values = wetness
    for _ in range(levels):
        edges = np.sort(triangles[:, ((0, 1), (1, 2), (2, 0))], axis=-1)
        unique, inverse = np.unique(
            edges.reshape(-1, 2), axis=0, return_inverse=True,
        )
        midpoints = len(vertices) + inverse.reshape(-1, 3)
        vertices = np.concatenate((vertices, np.mean(vertices[unique], axis=1)))
        if values is not None:
            values = np.concatenate((values, np.mean(values[unique], axis=1)))
        a, b, c = triangles.T
        ab, bc, ca = midpoints.T
        triangles = np.stack((
            np.column_stack((a, ab, ca)),
            np.column_stack((ab, b, bc)),
            np.column_stack((ca, bc, c)),
            np.column_stack((ab, bc, ca)),
        ), axis=1).reshape(-1, 3).astype(np.int32)
    if levels:
        vertices[len(mesh.vertices):] = boulder.project_to_surface(
            vertices[len(mesh.vertices):],
        )
    return TriangleMesh(vertices, triangles, mesh.material_id), values


def _wet_albedo(
    dry_albedo: np.ndarray,
    material: SceneMaterial,
) -> np.ndarray:
    ratio = (
        np.asarray(material.wet_albedo, dtype=np.float64)
        / np.asarray(material.dry_albedo, dtype=np.float64)
    )
    return (dry_albedo * ratio).astype(np.float32)


def _encode_normal_map(height: np.ndarray, spacing: float) -> np.ndarray:
    dx = (np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)) / (
        2.0 * spacing
    )
    dy = (np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)) / (
        2.0 * spacing
    )
    slope = np.stack((-dx, -dy), axis=-1)
    magnitude = np.linalg.norm(slope, axis=-1)
    scale = np.minimum(
        1.0,
        BEACH_NORMAL_SLOPE_LIMIT / np.maximum(magnitude, 1.0e-12),
    )
    slope *= scale[..., None]
    normal = np.concatenate((slope, np.ones((*height.shape, 1))), axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    return (0.5 * (normal + 1.0)).astype(np.float32)


def _sample_periodic_relief(
    height: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Sample the texel-centred AW-1001a field on a rectilinear world grid."""
    rows, columns = height.shape
    px = np.mod(x * columns / BEACH_TEXTURE_PERIOD - 0.5, columns)
    py = np.mod(y * rows / BEACH_TEXTURE_PERIOD - 0.5, rows)
    ix0 = np.floor(px).astype(np.int32)
    iy0 = np.floor(py).astype(np.int32)
    ix1 = (ix0 + 1) % columns
    iy1 = (iy0 + 1) % rows
    tx = (px - ix0)[:, None]
    ty = (py - iy0)[None, :]
    lower = (
        (1.0 - tx) * height[iy0[None, :], ix0[:, None]]
        + tx * height[iy0[None, :], ix1[:, None]]
    )
    upper = (
        (1.0 - tx) * height[iy1[None, :], ix0[:, None]]
        + tx * height[iy1[None, :], ix1[:, None]]
    )
    return (1.0 - ty) * lower + ty * upper


def beach_pebble_microgeometry(
    beach: AnalyticBeach,
    bounds: tuple[float, float, float, float],
    relief_height_m: np.ndarray,
    *,
    spacing: float = BEACH_MICROGEOMETRY_SPACING,
) -> BeachMicrogeometry:
    """Triangulate the canonical pebble relief above the analytic beach.

    World ``x/y`` remain the stable beach atlas.  Only the render surface's
    vertical coordinate changes; the analytic beach continues to own cut
    cells, boulder support, porous closures, and solid wetness sampling.
    """
    xlo, xhi, ylo, yhi = bounds
    nx = int(np.ceil((xhi - xlo) / spacing))
    ny = int(np.ceil((yhi - ylo) / spacing))
    x = np.linspace(xlo, xhi, nx + 1)
    y = np.linspace(ylo, yhi, ny + 1)
    source_relief_mean = float(np.mean(relief_height_m, dtype=np.float64))
    relief = _sample_periodic_relief(
        np.asarray(relief_height_m, dtype=np.float64) - source_relief_mean,
        x,
        y,
    )
    z = np.asarray(beach.elevation(x))[:, None] + relief

    dz_dx, dz_dy = np.gradient(z, x, y, edge_order=2)
    normals = np.stack((-dz_dx, -dz_dy, np.ones_like(z)), axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)

    vertices = np.column_stack((
        np.repeat(x, ny + 1),
        np.tile(y, nx + 1),
        z.ravel(),
    ))
    i = np.arange(nx, dtype=np.int32)[:, None]
    j = np.arange(ny, dtype=np.int32)[None, :]
    a = i * (ny + 1) + j
    b = a + ny + 1
    triangles = np.stack(
        (a, b, a + 1, a + 1, b, b + 1), axis=-1,
    ).reshape(-1, 3)
    texcoords = (
        vertices[:, :2] / BEACH_TEXTURE_PERIOD
    ).astype(np.float32)
    return BeachMicrogeometry(
        mesh=TriangleMesh(vertices, triangles, beach.material_id),
        texcoords=texcoords,
        relief_height_m=relief.astype(np.float32).ravel(),
        vertex_normals=normals.astype(np.float32).reshape(-1, 3),
        source_relief_mean_m=source_relief_mean,
    )


def _beach_look(
    mesh: TriangleMesh,
    material: SceneMaterial,
    surface_coordinates: np.ndarray,
    resolution: int,
) -> SolidMaterialLook:
    cells = round(BEACH_TEXTURE_PERIOD / BEACH_PEBBLE_SPACING)
    rng = np.random.default_rng(1001)
    jitter = (rng.random((cells, cells, 2)) - 0.5) * 0.94
    angle = np.pi * rng.random((cells, cells))
    major_radius = 0.58 + 0.22 * rng.random((cells, cells))
    minor_radius = major_radius * (
        0.68 + 0.25 * rng.random((cells, cells))
    )
    relief = (
        BEACH_PEBBLE_RELIEF_RANGE[0]
        + np.ptp(BEACH_PEBBLE_RELIEF_RANGE) * rng.random((cells, cells))
    )
    palette_index = rng.integers(0, 8, size=(cells, cells))
    tone = 0.86 + 0.28 * rng.random((cells, cells))
    roughness = rng.random((cells, cells))
    mineral_phase = 2.0 * np.pi * rng.random((cells, cells))

    coordinate = (np.arange(resolution, dtype=np.float64) + 0.5) * (
        cells / resolution
    )
    gx, gy = np.meshgrid(coordinate, coordinate)
    ix0 = np.floor(gx).astype(np.int32)
    iy0 = np.floor(gy).astype(np.int32)
    height = np.zeros(gx.shape, dtype=np.float64)
    nearest_x = np.zeros(gx.shape, dtype=np.int32)
    nearest_y = np.zeros(gy.shape, dtype=np.int32)
    local_x = np.zeros(gx.shape, dtype=np.float64)
    local_y = np.zeros(gy.shape, dtype=np.float64)
    dome = np.zeros(gx.shape, dtype=np.float64)
    for offset_y in (-1, 0, 1):
        for offset_x in (-1, 0, 1):
            ix = ix0 + offset_x
            iy = iy0 + offset_y
            wrapped_x = np.mod(ix, cells)
            wrapped_y = np.mod(iy, cells)
            centre_x = ix + 0.5 + jitter[wrapped_y, wrapped_x, 0]
            centre_y = iy + 0.5 + jitter[wrapped_y, wrapped_x, 1]
            dx = gx - centre_x
            dy = gy - centre_y
            cosine = np.cos(angle[wrapped_y, wrapped_x])
            sine = np.sin(angle[wrapped_y, wrapped_x])
            rotated_x = cosine * dx + sine * dy
            rotated_y = -sine * dx + cosine * dy
            radius_squared = (
                (rotated_x / major_radius[wrapped_y, wrapped_x]) ** 2
                + (rotated_y / minor_radius[wrapped_y, wrapped_x]) ** 2
            )
            profile = np.maximum(1.0 - radius_squared, 0.0) ** 0.72
            candidate = relief[wrapped_y, wrapped_x] * profile
            selected = candidate > height
            height[selected] = candidate[selected]
            nearest_x[selected] = wrapped_x[selected]
            nearest_y[selected] = wrapped_y[selected]
            local_x[selected] = rotated_x[selected]
            local_y[selected] = rotated_y[selected]
            dome[selected] = profile[selected]

    u = gx / cells
    v = gy / cells
    height += 0.00008 * (
        np.sin(2.0 * np.pi * (31.0 * u + 23.0 * v))
        + 0.45 * np.sin(2.0 * np.pi * (47.0 * u - 37.0 * v + 0.17))
    )
    height = np.maximum(height, 0.0)

    palette = np.asarray((
        (0.74, 0.80, 0.86),
        (1.12, 1.02, 0.86),
        (0.86, 0.93, 0.82),
        (1.12, 0.91, 0.80),
        (0.69, 0.76, 0.80),
        (1.16, 1.12, 1.03),
        (0.96, 0.86, 0.73),
        (0.80, 0.88, 0.96),
    ))
    selected_palette = palette[palette_index[nearest_y, nearest_x]]
    selected_tone = tone[nearest_y, nearest_x, None]
    phase = mineral_phase[nearest_y, nearest_x]
    mineral = 1.0 + 0.018 * np.sin(
        8.0 * local_x - 5.0 * local_y + phase,
    ) + 0.010 * np.sin(
        13.0 * local_x + 11.0 * local_y - 0.7 * phase,
    )
    pebble_dry = (
        np.asarray(material.dry_albedo)[None, None, :]
        * selected_palette
        * selected_tone
        * mineral[..., None]
    )
    matrix_dry = (
        np.asarray(material.dry_albedo)[None, None, :]
        * np.asarray((0.70, 0.73, 0.76))[None, None, :]
    )
    dry = np.where((dome > 0.0)[..., None], pebble_dry, matrix_dry)
    dry = np.clip(dry, 0.035, 0.32).astype(np.float32)
    roughness_weight = np.clip(
        0.12 + 0.72 * roughness[nearest_y, nearest_x]
        + 0.16 * (dome == 0.0),
        0.0,
        1.0,
    ).astype(np.float32)
    return SolidMaterialLook(
        mapping="bitmap",
        dry_albedo=dry,
        wet_albedo=_wet_albedo(dry, material),
        roughness_weight=roughness_weight,
        dry_roughness=(0.66, 0.92),
        wet_roughness=(0.24, 0.42),
        texcoords=(surface_coordinates / BEACH_TEXTURE_PERIOD).astype(
            np.float32,
        ),
        normal_map=_encode_normal_map(
            height,
            BEACH_TEXTURE_PERIOD / resolution,
        ),
        relief_height_m=height.astype(np.float32),
    )


def _boulder_normal(
    boulder: RoundedBoulder,
    local: np.ndarray,
) -> np.ndarray:
    scaled = local / np.asarray(boulder.radii)
    gradient = (
        np.sign(scaled)
        * np.abs(scaled) ** (boulder.exponent - 1.0)
        / np.asarray(boulder.radii)
    )
    normal = gradient @ boulder.rotation.T
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    return normal


def _boulder_look(
    mesh: TriangleMesh,
    boulder: RoundedBoulder,
    material: SceneMaterial,
    primitive_id: int,
) -> SolidMaterialLook:
    stone = STONE_LOOKS[(primitive_id - 1) % len(STONE_LOOKS)]
    local = np.asarray(boulder.local_coordinates(mesh.vertices))
    scaled = local / np.asarray(boulder.radii)
    phase = 0.73 * primitive_id
    multiplier = np.asarray(stone.albedo_multiplier)
    dry = np.asarray(material.dry_albedo) * multiplier
    dry = np.clip(dry, 0.045, 0.30).astype(np.float32)

    base_normal = _boulder_normal(boulder, local)
    detail_modes = np.asarray((
        (7.0, -5.0, 9.0),
        (-11.0, 13.0, 6.0),
        (17.0, 9.0, -15.0),
    ))
    detail_amplitudes = np.asarray((0.014, 0.009, 0.005))
    detail_arguments = scaled @ detail_modes.T + phase * np.arange(2, 5)
    gradient_scaled = (
        np.cos(detail_arguments) * detail_amplitudes
    ) @ detail_modes
    gradient_world = gradient_scaled @ boulder.rotation.T
    tangent = gradient_world - (
        np.sum(gradient_world * base_normal, axis=1, keepdims=True)
        * base_normal
    )
    magnitude = np.linalg.norm(tangent, axis=1)
    tangent *= np.minimum(
        1.0,
        BOULDER_NORMAL_SLOPE_LIMIT / np.maximum(magnitude, 1.0e-12),
    )[:, None]
    normal = base_normal - tangent
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)

    return SolidMaterialLook(
        mapping="constant",
        dry_albedo=dry,
        wet_albedo=_wet_albedo(dry, material),
        roughness_weight=np.asarray(0.0, dtype=np.float32),
        dry_roughness=(stone.dry_roughness, stone.dry_roughness),
        wet_roughness=(stone.wet_roughness, stone.wet_roughness),
        vertex_normals=normal.astype(np.float32),
        base_vertex_normals=base_normal.astype(np.float32),
    )


def shoreline_material_look(
    scene: SceneGeometry,
    mesh: TriangleMesh,
    primitive_id: int,
    *,
    beach_texture_resolution: int = BEACH_TEXTURE_RESOLUTION,
) -> SolidMaterialLook:
    """Evaluate the AW-907 look on one shared analytic solid primitive."""
    materials = {
        material.material_id: material for material in scene.materials
    }
    material = materials[mesh.material_id]
    if mesh.material_id == BEACH_MATERIAL_ID:
        return _beach_look(
            mesh,
            material,
            scene.beach.surface_coordinates(mesh.vertices),
            beach_texture_resolution,
        )
    return _boulder_look(
        mesh,
        scene.boulders[primitive_id - 1],
        material,
        primitive_id,
    )
