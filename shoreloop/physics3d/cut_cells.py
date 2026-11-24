"""Sampled cut-cell geometry from the shared analytic scene SDF."""

from __future__ import annotations

from dataclasses import dataclass
import itertools

import numpy as np

from shoreloop.config import BC
from shoreloop.physics3d.grid import MACGrid
from shoreloop.physics3d.plic import solve_plic_offset
from shoreloop.physics3d.polyhedron import open_face_area_host
from shoreloop.physics3d.solids import enforce_voxel_solids
from shoreloop.physics3d.state import State3D
from shoreloop.scene.geometry3d import SceneGeometry


@dataclass(frozen=True)
class CutCellGeometry:
    """Open cell volumes, Cartesian face apertures, and embedded boundaries."""

    cell_sdf: np.ndarray
    theta: np.ndarray
    apertures: tuple[np.ndarray, np.ndarray, np.ndarray]
    boundary_centroids: np.ndarray
    boundary_normals: np.ndarray
    material_ids: np.ndarray
    stabilization_threshold: float
    open_normals: np.ndarray
    open_offsets: np.ndarray
    max_plane_displacement: float

    @property
    def cut_mask(self) -> np.ndarray:
        return (self.theta > 0.0) & (self.theta < 1.0)

    @property
    def solid(self) -> np.ndarray:
        return self.theta == 0.0

    @property
    def stabilized_cells(self) -> int:
        return int(np.count_nonzero(
            (self.theta > 0.0) & (self.theta < self.stabilization_threshold),
        ))

    @property
    def minimum_open_fraction(self) -> float:
        return float(np.min(self.theta[self.theta > 0.0], initial=1.0))


def _lattice_points(
    axes: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)


def _sample_cell_volumes(
    scene: SceneGeometry,
    grid: MACGrid,
    samples: int,
) -> np.ndarray:
    g = grid.config
    axes = grid.coordinates("cell")
    spacing = (g.dx, g.dy, g.dz)
    offsets = (np.arange(samples) + 0.5) / samples - 0.5
    fluid = np.zeros(g.shape, dtype=np.int32)
    for oi, oj, ok in itertools.product(offsets, repeat=3):
        points = _lattice_points(tuple(
            axis + offset * delta
            for axis, offset, delta in zip(axes, (oi, oj, ok), spacing)
        ))
        fluid += np.asarray(scene.signed_distance(points)) >= 0.0
    return fluid.astype(np.float64) / samples**3


def _plane_face_apertures(theta, normals, offsets, periodic):
    """One common open footprint on each shared Cartesian face."""
    apertures = []
    for axis in range(3):
        shape = list(theta.shape)
        shape[axis] += 1
        aperture = np.ones(shape, dtype=np.float32)
        affected = np.zeros(shape, dtype=bool)
        low_faces = [slice(None)] * 3
        high_faces = [slice(None)] * 3
        low_faces[axis] = slice(None, -1)
        high_faces[axis] = slice(1, None)
        affected[tuple(low_faces)] |= theta < 1
        affected[tuple(high_faces)] |= theta < 1
        if periodic[axis]:
            first = [slice(None)] * 3; first[axis] = 0
            last = [slice(None)] * 3; last[axis] = -1
            seam = affected[tuple(first)] | affected[tuple(last)]
            affected[tuple(first)] = seam
            affected[tuple(last)] = seam
        for index in np.argwhere(affected):
            low = index.copy(); low[axis] -= 1
            high = index.copy()
            cells = []
            for cell in (low, high):
                valid = 0 <= cell[axis] < theta.shape[axis]
                if periodic[axis]:
                    cell[axis] %= theta.shape[axis]
                    valid = True
                cells.append(tuple(cell) if valid else None)
            lo, hi = cells
            if any(cell is not None and theta[cell] == 0 for cell in cells):
                aperture[tuple(index)] = 0
                continue
            ln = np.zeros(3) if lo is None else normals[lo].astype(float)
            hn = np.zeros(3) if hi is None else normals[hi].astype(float)
            ld = 0.0 if lo is None else float(offsets[lo]) - ln[axis]
            hd = 0.0 if hi is None else float(offsets[hi])
            aperture[tuple(index)] = open_face_area_host(axis, ln, ld, hn, hd)
        apertures.append(aperture)
    return tuple(apertures)


_CORNERS = np.asarray(tuple(itertools.product((-0.5, 0.5), repeat=3)))
_EDGES = tuple(
    (a, b) for a in range(8) for b in range(a + 1, 8)
    if np.count_nonzero(_CORNERS[a] != _CORNERS[b]) == 1
)


def _surface_centroids(
    scene: SceneGeometry,
    centres: np.ndarray,
    spacing: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    corners = centres[:, None, :] + _CORNERS[None, :, :] * spacing
    values = np.asarray(scene.signed_distance(corners))
    centroids = np.empty_like(centres)
    for index in range(len(centres)):
        intersections = []
        for first, second in _EDGES:
            a = values[index, first]
            b = values[index, second]
            if (a <= 0.0) != (b <= 0.0):
                weight = a / (a - b)
                intersections.append(
                    corners[index, first]
                    + weight * (corners[index, second] - corners[index, first])
                )
        centroids[index] = (
            np.mean(intersections, axis=0) if intersections else centres[index]
        )

    epsilon = 1.0e-4 * float(np.min(spacing))
    gradient = np.empty_like(centroids)
    for _ in range(3):
        for axis in range(3):
            delta = np.zeros(3)
            delta[axis] = epsilon
            gradient[:, axis] = (
                np.asarray(scene.signed_distance(centroids + delta))
                - np.asarray(scene.signed_distance(centroids - delta))
            ) / (2.0 * epsilon)
        distance = np.asarray(scene.signed_distance(centroids))
        centroids -= distance[:, None] * gradient / np.sum(
            gradient * gradient, axis=1, keepdims=True,
        )
    lengths = np.linalg.norm(gradient, axis=1, keepdims=True)
    return centroids, gradient / lengths


def generate_cut_cell_geometry(
    scene: SceneGeometry,
    grid: MACGrid,
    *,
    samples_per_axis: int = 4,
    stabilization_threshold: float = 0.125,
    periodic: tuple[bool, bool, bool] = (False, False, False),
) -> CutCellGeometry:
    """Generate sampled open fractions and embedded-boundary geometry."""
    cell_points = _lattice_points(grid.coordinates("cell"))
    cell_sdf = np.asarray(scene.signed_distance(cell_points), dtype=np.float64)
    theta = _sample_cell_volumes(scene, grid, samples_per_axis)
    centroids = np.zeros((*grid.config.shape, 3), dtype=np.float64)
    normals = np.zeros_like(centroids)
    material = np.asarray(scene.material_ids(cell_points), dtype=np.int32)
    cut = (theta > 0.0) & (theta < 1.0)
    spacing = np.asarray((grid.config.dx, grid.config.dy, grid.config.dz))
    cut_centroids, cut_normals = _surface_centroids(
        scene, cell_points[cut], spacing,
    )
    centroids[cut] = cut_centroids
    normals[cut] = cut_normals
    material[cut] = np.asarray(scene.material_ids(
        cut_centroids - 0.01 * float(np.min(spacing)) * cut_normals,
    ))
    # Reconstruct the sampled open amount with the shared analytic normal.
    # The normal is fixed; only the plane offset changes to reproduce theta.
    # This names and measures the solid quadrature's geometric approximation
    # without changing any retained alpha*theta. Face areas below are common
    # intersections of the neighbouring open planes, used by both D and fluxes.
    open_normals = np.zeros_like(normals, dtype=np.float32)
    open_offsets = np.zeros(grid.config.shape, dtype=np.float32)
    local_normals = -cut_normals * spacing
    local_normals /= np.abs(local_normals).sum(axis=1, keepdims=True)
    offsets = np.asarray([
        solve_plic_offset(normal, float(fraction))
        for normal, fraction in zip(local_normals, theta[cut], strict=True)
    ])
    open_normals[cut] = local_normals
    open_offsets[cut] = offsets
    origins = cell_points[cut] - .5 * spacing
    centroid_local = (cut_centroids - origins) / spacing
    displacement = np.abs(
        np.sum(local_normals * centroid_local, axis=1) - offsets
    ) / np.linalg.norm(local_normals / spacing, axis=1)
    apertures = _plane_face_apertures(theta, open_normals, open_offsets, periodic)
    return CutCellGeometry(
        cell_sdf=cell_sdf,
        theta=theta.astype(np.float32),
        apertures=tuple(value.astype(np.float32) for value in apertures),
        boundary_centroids=centroids,
        boundary_normals=normals,
        material_ids=material,
        stabilization_threshold=stabilization_threshold,
        open_normals=open_normals,
        open_offsets=open_offsets,
        max_plane_displacement=float(np.max(displacement, initial=0.0)),
    )


def initialize_scene_cut_cells(
    state: State3D,
    scene: SceneGeometry,
    *,
    samples_per_axis: int = 4,
    stabilization_threshold: float = 0.125,
) -> CutCellGeometry:
    """Generate and upload the scene's cut-cell representation."""
    geometry = generate_cut_cell_geometry(
        scene, state.grid, samples_per_axis=samples_per_axis,
        stabilization_threshold=stabilization_threshold,
        periodic=tuple(getattr(state.cfg.bc, axis + "lo") == BC.PERIODIC
                       for axis in "xyz"),
    )
    state.set_solid_interior(geometry.solid.astype(np.int32))
    state.set_material_interior(geometry.material_ids)
    state.set_geometry_interior("theta", geometry.theta)
    state.set_open_planes(geometry.open_normals, geometry.open_offsets)
    for name, aperture in zip(
        ("u_aperture", "v_aperture", "w_aperture"),
        geometry.apertures,
        strict=True,
    ):
        state.set_geometry_interior(name, aperture)
    state.small_cell_threshold = geometry.stabilization_threshold
    state.small_cell_count = geometry.stabilized_cells
    state.minimum_open_fraction = geometry.minimum_open_fraction
    enforce_voxel_solids(state)
    return geometry
