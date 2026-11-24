"""Measured temporal-stability gate for extracted water surfaces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from shoreloop.config import Grid3DConfig
from shoreloop.render.surface_mesh import (
    WaterSurfaceMesh,
    _vertex_normals,
    extract_free_surface_mesh,
)


@dataclass(frozen=True)
class SurfaceStabilityMetrics:
    """Normal and silhouette motion over one sub-cell wave translation."""

    frames: int
    grid_spacing: float
    phase_step: float
    geometry_normal_error_p95_degrees: float
    stable_normal_error_p95_degrees: float
    geometry_normal_change_p95_degrees: float
    stable_normal_change_p95_degrees: float
    temporal_normal_reduction: float
    silhouette_range_p95_cells: float
    silhouette_range_max_cells: float

    def write(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(asdict(self), indent=2) + "\n")
        return destination


@dataclass(frozen=True)
class TravellingWaveEvaluation:
    """Synthetic frames retained for the short normal-stability render."""

    grid: Grid3DConfig
    phases: np.ndarray
    meshes: tuple[WaterSurfaceMesh, ...]
    geometry_normals: tuple[np.ndarray, ...]
    metrics: SurfaceStabilityMetrics


def _cell_points(
    grid: Grid3DConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axes = tuple(
        origin + (np.arange(count) + 0.5) * spacing
        for origin, count, spacing in zip(
            (grid.x0, grid.y0, grid.z0), grid.shape,
            (grid.dx, grid.dy, grid.dz), strict=True,
        )
    )
    return np.meshgrid(*axes, indexing="ij")


def _exact_normals(
    vertices: np.ndarray,
    phase: float,
    amplitude: float,
    wavenumber: float,
) -> np.ndarray:
    normals = np.stack((
        amplitude * wavenumber * np.sin(
            wavenumber * (vertices[:, 0] - phase),
        ),
        np.zeros(len(vertices)),
        np.ones(len(vertices)),
    ), axis=1)
    return normals / np.linalg.norm(normals, axis=1, keepdims=True)


def _angular_error_degrees(
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    cosine = np.sum(first * second, axis=1)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _temporal_normal_changes(
    meshes: tuple[WaterSurfaceMesh, ...],
    normals: tuple[np.ndarray, ...],
    grid: Grid3DConfig,
    phase_step: float,
) -> np.ndarray:
    changes: list[np.ndarray] = []
    for first, second, first_normals, second_normals in zip(
        meshes[:-1], meshes[1:], normals[:-1], normals[1:], strict=True,
    ):
        advected = first.vertices.astype(np.float64, copy=True)
        advected[:, 0] += phase_step
        interior = (
            (advected[:, 0] > grid.x0 + 2.0 * grid.dx)
            & (advected[:, 0] < grid.x0 + grid.Lx - 2.0 * grid.dx)
            & (advected[:, 1] > grid.y0 + grid.dy)
            & (advected[:, 1] < grid.y0 + grid.Ly - grid.dy)
        )
        distance, match = cKDTree(second.vertices).query(advected[interior])
        paired = distance < 0.8 * min(grid.dx, grid.dy, grid.dz)
        changes.append(_angular_error_degrees(
            first_normals[interior][paired], second_normals[match[paired]],
        ))
    return np.concatenate(changes)


def _side_section(mesh: WaterSurfaceMesh) -> tuple[np.ndarray, np.ndarray]:
    """Intersect a smooth-wave mesh with the centre ``y=0`` plane."""
    points: list[np.ndarray] = []
    for triangle in mesh.vertices[mesh.triangles]:
        distance = triangle[:, 1]
        for first, second in ((0, 1), (1, 2), (2, 0)):
            if distance[first] == 0.0:
                points.append(triangle[first, (0, 2)])
            if distance[first] * distance[second] < 0.0:
                fraction = -distance[first] / (
                    distance[second] - distance[first]
                )
                point = triangle[first] + fraction * (
                    triangle[second] - triangle[first]
                )
                points.append(point[(0, 2),])
    section = np.asarray(points)
    order = np.argsort(section[:, 0])
    section = section[order]
    x, first = np.unique(section[:, 0], return_index=True)
    z = np.add.reduceat(section[:, 1], first) / np.diff(
        np.r_[first, len(section)],
    )
    return x, z


def evaluate_travelling_wave_surface() -> TravellingWaveEvaluation:
    """Extract nine sub-cell phases and measure co-moving mesh changes."""
    grid = Grid3DConfig(
        nx=80, ny=8, nz=32,
        Lx=4.0, Ly=0.4, Lz=1.6,
        x0=0.0, y0=-0.2, z0=-0.8,
    )
    x, _, z = _cell_points(grid)
    amplitude = 0.16
    wavelength = 1.6
    wavenumber = 2.0 * np.pi / wavelength
    phase_step = grid.dx / 8.0
    phases = np.arange(9, dtype=np.float64) * phase_step
    meshes: list[WaterSurfaceMesh] = []
    geometry_normals: list[np.ndarray] = []
    geometry_errors: list[np.ndarray] = []
    stable_errors: list[np.ndarray] = []

    for phase in phases:
        elevation = amplitude * np.cos(wavenumber * (x - phase))
        alpha = np.clip(
            0.5 + (elevation - z) / grid.dz, 0.0, 1.0,
        ).astype(np.float32)
        mesh = extract_free_surface_mesh(alpha, grid)
        geometric = _vertex_normals(mesh.vertices, mesh.triangles)
        exact = _exact_normals(
            mesh.vertices, phase, amplitude, wavenumber,
        )
        interior = (
            (mesh.vertices[:, 0] > grid.x0 + 2.0 * grid.dx)
            & (mesh.vertices[:, 0] < grid.x0 + grid.Lx - 2.0 * grid.dx)
            & (mesh.vertices[:, 1] > grid.y0 + grid.dy)
            & (mesh.vertices[:, 1] < grid.y0 + grid.Ly - grid.dy)
        )
        meshes.append(mesh)
        geometry_normals.append(geometric)
        geometry_errors.append(_angular_error_degrees(
            geometric[interior], exact[interior],
        ))
        stable_errors.append(_angular_error_degrees(
            mesh.normals[interior], exact[interior],
        ))

    mesh_tuple = tuple(meshes)
    geometry_tuple = tuple(geometry_normals)
    stable_tuple = tuple(mesh.normals for mesh in meshes)
    geometry_changes = _temporal_normal_changes(
        mesh_tuple, geometry_tuple, grid, phase_step,
    )
    stable_changes = _temporal_normal_changes(
        mesh_tuple, stable_tuple, grid, phase_step,
    )

    sample_coordinate = np.linspace(0.3, 3.5, 800)
    silhouettes = []
    for phase, mesh in zip(phases, meshes, strict=True):
        section_x, section_z = _side_section(mesh)
        silhouettes.append(np.interp(
            sample_coordinate + phase, section_x, section_z,
        ))
    silhouette_range = np.ptp(np.asarray(silhouettes), axis=0) / grid.dz

    geometry_change_p95 = float(np.quantile(geometry_changes, 0.95))
    stable_change_p95 = float(np.quantile(stable_changes, 0.95))
    metrics = SurfaceStabilityMetrics(
        frames=len(phases),
        grid_spacing=grid.dx,
        phase_step=phase_step,
        geometry_normal_error_p95_degrees=float(np.quantile(
            np.concatenate(geometry_errors), 0.95,
        )),
        stable_normal_error_p95_degrees=float(np.quantile(
            np.concatenate(stable_errors), 0.95,
        )),
        geometry_normal_change_p95_degrees=geometry_change_p95,
        stable_normal_change_p95_degrees=stable_change_p95,
        temporal_normal_reduction=(
            1.0 - stable_change_p95 / geometry_change_p95
        ),
        silhouette_range_p95_cells=float(np.quantile(
            silhouette_range, 0.95,
        )),
        silhouette_range_max_cells=float(np.max(silhouette_range)),
    )
    return TravellingWaveEvaluation(
        grid=grid,
        phases=phases,
        meshes=mesh_tuple,
        geometry_normals=geometry_tuple,
        metrics=metrics,
    )
