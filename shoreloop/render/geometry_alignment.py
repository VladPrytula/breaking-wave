"""Measured alignment between cut-cell solids and their render tessellation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation
from skimage.measure import find_contours

from shoreloop.config import Grid3DConfig
from shoreloop.physics3d.cut_cells import CutCellGeometry
from shoreloop.render.mitsuba_water import SharedSolidScene
from shoreloop.render.shoreline_look import refine_boulder_render_mesh
from shoreloop.scene.geometry3d import BOULDER_MATERIAL_ID, TriangleMesh


@dataclass(frozen=True)
class GeometryAlignment:
    """Horizontal contact-plane comparison in physical coordinates."""

    plane_z: float
    cell_width: float
    cut_segments: np.ndarray
    render_segments: np.ndarray
    cut_to_render_distance: float
    render_to_cut_distance: float

    @property
    def maximum_distance(self) -> float:
        return max(
            self.cut_to_render_distance,
            self.render_to_cut_distance,
        )

    @property
    def maximum_distance_cells(self) -> float:
        return self.maximum_distance / self.cell_width


def _triangle_plane_segments(
    mesh: TriangleMesh,
    plane_z: float,
) -> list[np.ndarray]:
    segments = []
    tolerance = 1.0e-12
    for triangle in mesh.vertices[mesh.triangles]:
        signed = triangle[:, 2] - plane_z
        intersections = []
        for first, second in ((0, 1), (1, 2), (2, 0)):
            a = signed[first]
            b = signed[second]
            if abs(a) <= tolerance:
                intersections.append(triangle[first, :2])
            if a * b < 0.0:
                weight = a / (a - b)
                intersections.append(
                    triangle[first, :2]
                    + weight * (triangle[second, :2] - triangle[first, :2])
                )
        unique = []
        for point in intersections:
            if not any(np.linalg.norm(point - other) <= tolerance
                       for other in unique):
                unique.append(point)
        if len(unique) >= 2:
            points = np.asarray(unique)
            separation = np.linalg.norm(
                points[:, None, :] - points[None, :, :], axis=2,
            )
            first, second = np.unravel_index(
                np.argmax(separation), separation.shape,
            )
            segments.append(points[[first, second]])
    return segments


def _render_boulder_segments(
    solids: SharedSolidScene,
    plane_z: float,
) -> np.ndarray:
    segments = []
    for primitive_id, mesh in enumerate(solids.meshes()):
        if mesh.material_id != BOULDER_MATERIAL_ID:
            continue
        if solids.boulder_subdivision_levels:
            mesh, _ = refine_boulder_render_mesh(
                solids.geometry.boulders[primitive_id - 1], mesh, None,
                levels=solids.boulder_subdivision_levels,
            )
        for segment in _triangle_plane_segments(mesh, plane_z):
            midpoint = np.mean(segment, axis=0)
            point = (midpoint[0], midpoint[1], plane_z)
            if solids.geometry.beach.signed_distance(point) >= 0.0:
                segments.append(segment)
    return np.asarray(segments, dtype=np.float64).reshape((-1, 2, 2))


def _cut_boulder_segments(
    geometry: CutCellGeometry,
    grid: Grid3DConfig,
    plane_index: int,
) -> np.ndarray:
    boulder_neighbourhood = binary_dilation(
        geometry.material_ids[:, :, plane_index] == BOULDER_MATERIAL_ID,
        iterations=1,
    )
    segments = []
    for contour in find_contours(
        geometry.theta[:, :, plane_index].T,
        level=0.5,
    ):
        row = contour[:, 0]
        column = contour[:, 1]
        points = np.column_stack((
            grid.x0 + (column + 0.5) * grid.dx,
            grid.y0 + (row + 0.5) * grid.dy,
        ))
        contour_segments = np.stack((points[:-1], points[1:]), axis=1)
        midpoint_index = 0.5 * (contour[:-1] + contour[1:])
        j = np.clip(
            np.rint(midpoint_index[:, 0]).astype(np.int64),
            0,
            grid.ny - 1,
        )
        i = np.clip(
            np.rint(midpoint_index[:, 1]).astype(np.int64),
            0,
            grid.nx - 1,
        )
        segments.extend(contour_segments[boulder_neighbourhood[i, j]])
    return np.asarray(segments, dtype=np.float64).reshape((-1, 2, 2))


def _segment_samples(segments: np.ndarray) -> np.ndarray:
    fraction = np.linspace(0.0, 1.0, 5)
    return (
        segments[:, None, 0]
        + fraction[None, :, None]
        * (segments[:, None, 1] - segments[:, None, 0])
    ).reshape((-1, 2))


def _maximum_distance(
    source: np.ndarray,
    target: np.ndarray,
) -> float:
    points = _segment_samples(source)
    start = target[:, 0]
    direction = target[:, 1] - start
    relative = points[:, None, :] - start[None, :, :]
    fraction = np.sum(
        relative * direction[None, :, :], axis=2,
    ) / np.sum(direction * direction, axis=1)[None, :]
    fraction = np.clip(fraction, 0.0, 1.0)
    closest = start[None, :, :] + fraction[:, :, None] * direction[None, :, :]
    distance = np.linalg.norm(points[:, None, :] - closest, axis=2)
    return float(np.max(np.min(distance, axis=1)))


def measure_geometry_alignment(
    geometry: CutCellGeometry,
    grid: Grid3DConfig,
    solids: SharedSolidScene,
    *,
    plane_z: float,
) -> GeometryAlignment:
    """Compare cut-cell and render-mesh boulder contours near ``plane_z``."""
    z = grid.z0 + (np.arange(grid.nz) + 0.5) * grid.dz
    plane_index = int(np.argmin(np.abs(z - plane_z)))
    sampled_z = float(z[plane_index])
    cut_segments = _cut_boulder_segments(geometry, grid, plane_index)
    render_segments = _render_boulder_segments(solids, sampled_z)
    return GeometryAlignment(
        plane_z=sampled_z,
        cell_width=min(grid.dx, grid.dy),
        cut_segments=cut_segments,
        render_segments=render_segments,
        cut_to_render_distance=_maximum_distance(
            cut_segments, render_segments,
        ),
        render_to_cut_distance=_maximum_distance(
            render_segments, cut_segments,
        ),
    )


def write_geometry_alignment_overlay(
    alignment: GeometryAlignment,
    path: str | Path,
) -> Path:
    """Write the measured cut-cell/render-mesh contact-plane overlay."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9.0, 4.5), constrained_layout=True)
    for index, segment in enumerate(alignment.cut_segments):
        axis.plot(
            segment[:, 0], segment[:, 1], color="#f08a24", linewidth=2.4,
            label="cut-cell theta=0.5" if index == 0 else None,
        )
    for index, segment in enumerate(alignment.render_segments):
        axis.plot(
            segment[:, 0], segment[:, 1], color="#168aad", linewidth=1.2,
            label="render-mesh intersection" if index == 0 else None,
        )
    axis.set_aspect("equal")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_title(
        f"z={alignment.plane_z:.3f} m, max separation "
        f"{alignment.maximum_distance:.4f} m "
        f"({alignment.maximum_distance_cells:.3f} cell)"
    )
    axis.legend()
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    return destination
