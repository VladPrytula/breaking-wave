"""Topology, orientation, clipping, and attribute gates for VOF meshes."""

from __future__ import annotations

import numpy as np
import pytest

from shoreloop.config import Grid3DConfig
from shoreloop.physics3d.cut_cells import generate_cut_cell_geometry
from shoreloop.physics3d.grid import MACGrid
from shoreloop.render.surface_stability import evaluate_travelling_wave_surface
from shoreloop.render.surface_mesh import (
    OpenBoundary,
    WaterSurfaceMesh,
    _clip_to_open_volume,
    extract_free_surface_mesh,
)
from shoreloop.scene.geometry3d import (
    AnalyticBeach,
    SceneGeometry,
    deterministic_boulders,
)


def _cell_points(grid: Grid3DConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axes = tuple(
        origin + (np.arange(count) + 0.5) * spacing
        for origin, count, spacing in zip(
            (grid.x0, grid.y0, grid.z0), grid.shape,
            (grid.dx, grid.dy, grid.dz), strict=True,
        )
    )
    return np.meshgrid(*axes, indexing="ij")


def _fraction_from_margin(margin: np.ndarray, spacing: float) -> np.ndarray:
    return np.clip(0.5 + margin / spacing, 0.0, 1.0).astype(np.float32)


def _mesh_edges(triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.concatenate((
        triangles[:, (0, 1)], triangles[:, (1, 2)], triangles[:, (2, 0)],
    ))
    edges.sort(axis=1)
    return np.unique(edges, axis=0, return_counts=True)


def _euler_characteristic(vertex_count: int, triangles: np.ndarray) -> int:
    edges, _ = _mesh_edges(triangles)
    return vertex_count - len(edges) + len(triangles)


def _connected_vertex_count(vertex_count: int, triangles: np.ndarray) -> int:
    adjacency = [set() for _ in range(vertex_count)]
    for a, b, c in triangles:
        adjacency[a].update((int(b), int(c)))
        adjacency[b].update((int(a), int(c)))
        adjacency[c].update((int(a), int(b)))
    seen = {0}
    pending = [0]
    while pending:
        current = pending.pop()
        unseen = adjacency[current] - seen
        seen.update(unseen)
        pending.extend(unseen)
    return len(seen)


def _face_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    points = vertices[triangles]
    normals = np.cross(points[:, 1] - points[:, 0],
                       points[:, 2] - points[:, 0])
    return normals / np.linalg.norm(normals, axis=1, keepdims=True)


def _ordered_y_boundary(mesh: WaterSurfaceMesh, y: float) -> np.ndarray:
    indices = np.flatnonzero(mesh.vertices[:, 1] == np.float32(y))
    return indices[np.lexsort((
        mesh.vertices[indices, 2],
        mesh.vertices[indices, 0],
    ))]


def test_periodic_solid_cut_is_invariant_to_boundary_edge_orientation():
    vertices = np.asarray((
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 1.0),
        (0.5, 0.1, 0.6),
        (0.0, 1.0, 0.0),
        (1.0, 1.0, 1.0),
        (0.5, 0.9, 0.6),
    ), dtype=np.float32)
    # Both faces have upward winding.  Their corresponding periodic boundary
    # edge is necessarily traversed in the opposite direction.
    triangles = np.asarray(((0, 1, 2), (4, 3, 5)), dtype=np.int32)
    open_fraction = np.asarray((
        0.49, 0.9, 0.8,
        0.49, 0.9, 0.8,
    ), dtype=np.float32)

    clipped_vertices, clipped_triangles = _clip_to_open_volume(
        vertices, triangles, open_fraction,
    )
    low = np.flatnonzero(clipped_vertices[:, 1] == np.float32(0.0))
    high = np.flatnonzero(clipped_vertices[:, 1] == np.float32(1.0))
    low = low[np.lexsort((
        clipped_vertices[low, 2], clipped_vertices[low, 0],
    ))]
    high = high[np.lexsort((
        clipped_vertices[high, 2], clipped_vertices[high, 0],
    ))]

    assert len(clipped_triangles) > 0
    np.testing.assert_array_equal(
        clipped_vertices[low][:, (0, 2)],
        clipped_vertices[high][:, (0, 2)],
    )


@pytest.mark.parametrize("smoothing_iterations", (0, 2))
def test_periodic_nodal_reconstruction_matches_both_y_boundaries(
    smoothing_iterations: int,
):
    grid = Grid3DConfig(
        nx=24, ny=20, nz=18,
        Lx=1.2, Ly=1.0, Lz=0.9,
        x0=-0.1, y0=0.2, z0=-0.3,
    )
    x, y, z = _cell_points(grid)
    phase = 2.0 * np.pi * (y - grid.y0) / grid.Ly
    height = (
        0.27
        + 0.06 * np.cos(phase)
        + 0.02 * np.sin(phase)
        + 0.025 * np.sin(2.0 * np.pi * (x - grid.x0) / grid.Lx)
    )
    alpha = _fraction_from_margin(height - z, grid.dz)
    velocity = np.stack((
        0.4 + 0.1 * np.cos(phase),
        0.2 * np.sin(phase),
        0.05 * np.cos(2.0 * phase),
    ), axis=-1).astype(np.float32)
    foam = (0.3 + 0.1 * np.cos(phase)).astype(np.float32)

    unpaired = extract_free_surface_mesh(
        alpha,
        grid,
        velocity=velocity,
        foam=foam,
        smoothing_iterations=smoothing_iterations,
    )
    periodic = extract_free_surface_mesh(
        alpha,
        grid,
        velocity=velocity,
        foam=foam,
        periodic_axes=("y",),
        smoothing_iterations=smoothing_iterations,
    )
    low = _ordered_y_boundary(periodic, grid.y0)
    high = _ordered_y_boundary(periodic, grid.y0 + grid.Ly)
    unpaired_low = _ordered_y_boundary(unpaired, grid.y0)
    unpaired_high = _ordered_y_boundary(unpaired, grid.y0 + grid.Ly)

    assert len(low) == len(high) > 0
    np.testing.assert_array_equal(
        periodic.vertices[low][:, (0, 2)],
        periodic.vertices[high][:, (0, 2)],
    )
    np.testing.assert_array_equal(
        periodic.normals[low], periodic.normals[high],
    )
    np.testing.assert_array_equal(
        periodic.velocity[low], periodic.velocity[high],
    )
    np.testing.assert_array_equal(periodic.foam[low], periodic.foam[high])
    assert len(unpaired_low) == len(unpaired_high)
    assert float(np.max(np.abs(
        unpaired.vertices[unpaired_low, 2]
        - unpaired.vertices[unpaired_high, 2]
    ))) > 1.0e-3


def test_sphere_plane_and_overturned_sheet_topology_and_orientation(tmp_path):
    sphere_grid = Grid3DConfig(
        nx=32, ny=32, nz=32, Lx=2.0, Ly=2.0, Lz=2.0,
        x0=-1.0, y0=-1.0, z0=-1.0,
    )
    x, y, z = _cell_points(sphere_grid)
    radius = 0.56
    sphere_margin = radius - np.sqrt(x * x + y * y + z * z)
    sphere_alpha = _fraction_from_margin(sphere_margin, sphere_grid.dx)
    velocity = np.stack((x, 2.0 * y, -z), axis=-1).astype(np.float32)
    foam = (0.45 + 0.1 * x - 0.08 * y + 0.05 * z).astype(np.float32)
    sphere = extract_free_surface_mesh(
        sphere_alpha, sphere_grid, velocity=velocity, foam=foam,
    )

    sphere_edges, sphere_edge_counts = _mesh_edges(sphere.triangles)
    assert len(sphere.vertices) > 0
    assert np.all(sphere_edge_counts == 2)
    assert _euler_characteristic(len(sphere.vertices), sphere.triangles) == 2
    assert _connected_vertex_count(
        len(sphere.vertices), sphere.triangles,
    ) == len(sphere.vertices)
    radial = sphere.vertices / np.linalg.norm(
        sphere.vertices, axis=1, keepdims=True,
    )
    assert np.quantile(np.sum(sphere.normals * radial, axis=1), 0.01) > 0.94
    sphere_face_normals = _face_normals(sphere.vertices, sphere.triangles)
    sphere_face_centres = sphere.vertices[sphere.triangles].mean(axis=1)
    assert np.min(np.sum(sphere_face_normals * sphere_face_centres, axis=1)) > 0.0
    np.testing.assert_allclose(
        sphere.velocity,
        np.stack((sphere.vertices[:, 0], 2.0 * sphere.vertices[:, 1],
                  -sphere.vertices[:, 2]), axis=-1),
        atol=2.0e-6,
    )
    np.testing.assert_allclose(
        sphere.foam,
        0.45 + 0.1 * sphere.vertices[:, 0]
        - 0.08 * sphere.vertices[:, 1] + 0.05 * sphere.vertices[:, 2],
        atol=2.0e-6,
    )
    assert len(sphere_edges) > 0
    assert len(sphere.open_boundary_edges) == 0
    path = sphere.write(tmp_path / "sphere_surface.npz")
    saved = WaterSurfaceMesh.read(path)
    np.testing.assert_array_equal(saved.triangles, sphere.triangles)
    np.testing.assert_array_equal(saved.velocity, sphere.velocity)
    np.testing.assert_array_equal(saved.foam, sphere.foam)

    plane_grid = Grid3DConfig(
        nx=24, ny=20, nz=18, Lx=1.2, Ly=1.0, Lz=0.9,
        x0=-0.1, y0=0.2, z0=-0.3,
    )
    x, y, z = _cell_points(plane_grid)
    expected_normal = np.asarray((0.31, -0.23, 0.92), dtype=np.float64)
    expected_normal /= np.linalg.norm(expected_normal)
    centre = np.asarray((
        plane_grid.x0 + 0.5 * plane_grid.Lx,
        plane_grid.y0 + 0.5 * plane_grid.Ly,
        plane_grid.z0 + 0.5 * plane_grid.Lz,
    ))
    signed = (
        expected_normal[0] * (x - centre[0])
        + expected_normal[1] * (y - centre[1])
        + expected_normal[2] * (z - centre[2])
    )
    plane = extract_free_surface_mesh(
        _fraction_from_margin(-signed, min(
            plane_grid.dx, plane_grid.dy, plane_grid.dz,
        )),
        plane_grid,
        open_boundaries=("xlo", "xhi", "ylo", "yhi", "zlo", "zhi"),
    )

    _, plane_edge_counts = _mesh_edges(plane.triangles)
    assert _euler_characteristic(len(plane.vertices), plane.triangles) == 1
    assert _connected_vertex_count(
        len(plane.vertices), plane.triangles,
    ) == len(plane.vertices)
    assert np.min(plane.normals @ expected_normal) > 0.96
    assert np.min(_face_normals(
        plane.vertices, plane.triangles,
    ) @ expected_normal) > 0.96
    assert len(plane.open_boundary_edges) == int(np.count_nonzero(
        plane_edge_counts == 1,
    ))
    assert np.all(plane.open_boundary_labels != OpenBoundary.NONE)

    sheet_grid = Grid3DConfig(
        nx=48, ny=24, nz=48, Lx=2.4, Ly=1.0, Lz=2.4,
        x0=-1.2, y0=-0.5, z0=-1.2,
    )
    x, y, z = _cell_points(sheet_grid)
    radius = np.sqrt(x * x + z * z)
    angle = np.arctan2(z, x)
    margin = np.minimum.reduce((
        0.13 - np.abs(radius - 0.72),
        0.31 - np.abs(y),
        (2.55 - np.abs(angle)) * 0.72,
    ))
    sheet_alpha = _fraction_from_margin(margin, min(
        sheet_grid.dx, sheet_grid.dy, sheet_grid.dz,
    ))
    sheet = extract_free_surface_mesh(sheet_alpha, sheet_grid)

    _, sheet_edge_counts = _mesh_edges(sheet.triangles)
    assert np.all(sheet_edge_counts == 2)
    assert _euler_characteristic(len(sheet.vertices), sheet.triangles) == 2
    assert _connected_vertex_count(
        len(sheet.vertices), sheet.triangles,
    ) == len(sheet.vertices)
    transitions = np.count_nonzero(
        (sheet_alpha[:, :, 1:] >= 0.5)
        != (sheet_alpha[:, :, :-1] >= 0.5),
        axis=2,
    )
    assert int(np.max(transitions)) >= 4
    assert np.count_nonzero(sheet.normals[:, 2] > 0.7) > 0
    assert np.count_nonzero(sheet.normals[:, 2] < -0.7) > 0

    thin_margin = np.minimum.reduce((
        0.06 - np.abs(radius - 0.72),
        0.31 - np.abs(y),
        (2.55 - np.abs(angle)) * 0.72,
    ))
    thin_alpha = _fraction_from_margin(thin_margin, sheet_grid.dx)
    thin_transitions = np.count_nonzero(
        (thin_alpha[:, :, 1:] >= 0.5)
        != (thin_alpha[:, :, :-1] >= 0.5),
        axis=2,
    )
    assert int(np.max(thin_transitions)) >= 4
    thin_raw = extract_free_surface_mesh(thin_alpha, sheet_grid)
    thin_smooth = extract_free_surface_mesh(
        thin_alpha, sheet_grid, smoothing_iterations=2,
    )
    _, thin_edge_counts = _mesh_edges(thin_smooth.triangles)
    assert np.all(thin_edge_counts == 2)
    assert _euler_characteristic(
        len(thin_smooth.vertices), thin_smooth.triangles,
    ) == 2
    np.testing.assert_array_equal(thin_smooth.triangles, thin_raw.triangles)
    displacement = np.linalg.norm(
        thin_smooth.vertices - thin_raw.vertices, axis=1,
    )
    assert float(np.max(displacement)) < 0.4 * sheet_grid.dx
    thin_radius = np.hypot(
        thin_smooth.vertices[:, 0], thin_smooth.vertices[:, 2],
    )
    assert np.count_nonzero(thin_radius < 0.72) > 500
    assert np.count_nonzero(thin_radius > 0.72) > 500
    assert np.count_nonzero(thin_smooth.normals[:, 2] > 0.7) > 500
    assert np.count_nonzero(thin_smooth.normals[:, 2] < -0.7) > 500


def test_solid_mask_clips_the_surface_without_open_domain_labels():
    grid = Grid3DConfig(nx=32, ny=32, nz=24, Lx=1.0, Ly=1.0, Lz=0.75)
    x, y, z = _cell_points(grid)
    alpha = _fraction_from_margin(0.46 - z, grid.dz)
    open_fraction = np.ones(grid.shape, dtype=np.float32)
    pillar = (
        (x >= 0.375) & (x <= 0.625)
        & (y >= 0.375) & (y <= 0.625)
    )
    open_fraction[pillar] = 0.0
    mesh = extract_free_surface_mesh(
        alpha, grid, open_fraction=open_fraction,
        open_boundaries=("xlo", "xhi", "ylo", "yhi"),
    )

    face_centres = mesh.vertices[mesh.triangles].mean(axis=1)
    half_cell = 0.5 * max(grid.dx, grid.dy)
    inside_pillar = (
        (face_centres[:, 0] > 0.375 + half_cell)
        & (face_centres[:, 0] < 0.625 - half_cell)
        & (face_centres[:, 1] > 0.375 + half_cell)
        & (face_centres[:, 1] < 0.625 - half_cell)
    )
    assert not np.any(inside_pillar)
    np.testing.assert_allclose(face_centres[:, 2], 0.46, atol=2.0e-6)
    assert np.min(_face_normals(mesh.vertices, mesh.triangles)[:, 2]) > 0.999
    _, edge_counts = _mesh_edges(mesh.triangles)
    all_boundary_edge_count = int(np.count_nonzero(edge_counts == 1))
    assert 0 < len(mesh.open_boundary_edges) < all_boundary_edge_count
    assert np.all(mesh.open_boundary_labels != OpenBoundary.NONE)


def test_smoothed_initial_boulder_surface_stays_inside_the_domain():
    grid = Grid3DConfig(
        nx=104, ny=32, nz=24,
        Lx=4.16, Ly=1.6, Lz=0.48,
    )
    scene = SceneGeometry(
        AnalyticBeach(1.35, 0.18, grid.z0),
        deterministic_boulders(),
    )
    geometry = generate_cut_cell_geometry(scene, MACGrid(grid))
    _, _, z = _cell_points(grid)
    alpha = _fraction_from_margin(0.28 - z, grid.dz)

    raw = extract_free_surface_mesh(
        alpha, grid, open_fraction=geometry.theta,
    )
    smoothed = extract_free_surface_mesh(
        alpha, grid, open_fraction=geometry.theta,
        smoothing_iterations=2,
    )

    assert len(raw.triangles) > 0
    assert len(smoothed.triangles) > 0
    assert np.all(np.isfinite(raw.normals))
    assert np.all(np.isfinite(smoothed.normals))
    for mesh in (raw, smoothed):
        points = mesh.vertices[mesh.triangles]
        area_vectors = np.cross(
            points[:, 1] - points[:, 0], points[:, 2] - points[:, 0],
        )
        assert np.all(np.any(area_vectors != 0.0, axis=1))
        assert len(np.unique(mesh.triangles)) == len(mesh.vertices)
    lower = np.asarray((grid.x0, grid.y0, grid.z0))
    upper = lower + np.asarray((grid.Lx, grid.Ly, grid.Lz))
    tolerance = 2.0e-5 * min(grid.dx, grid.dy, grid.dz)
    assert np.all(smoothed.vertices >= lower - tolerance)
    assert np.all(smoothed.vertices <= upper + tolerance)
    assert np.any(smoothed.vertices[:, 1] == grid.y0)
    assert np.any(smoothed.vertices[:, 1] == grid.y0 + grid.Ly)


def test_smooth_travelling_wave_normals_and_silhouette_are_temporally_stable():
    metrics = evaluate_travelling_wave_surface().metrics

    assert metrics.stable_normal_error_p95_degrees < 1.2
    assert metrics.stable_normal_change_p95_degrees < 1.0
    assert metrics.temporal_normal_reduction > 0.45
    assert metrics.silhouette_range_p95_cells < 0.02
    assert metrics.silhouette_range_max_cells < 0.02
