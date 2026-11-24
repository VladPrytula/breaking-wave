"""Offline triangle-surface extraction from authoritative three-dimensional VOF.

The VOF field remains the hydrodynamic state; extraction reads it and writes
nothing back. A ``liquid_mimetic`` state offers two named render surfaces of
the same cell-averaged ``alpha``/``theta`` field (:data:`RENDER_SURFACES`):

``plic_boundary`` (default)
    The exact boundary of the union of the solver's cell-local cut/PLIC
    polyhedra (:mod:`shoreloop.render.liquid_surface`). Volume-exact and
    unsmoothed; a mismatch between two neighbouring PLIC planes is closed by
    a cell-face patch.
``nodal_isosurface``
    The ``alpha = 0.5`` marching-cubes isosurface of the theta-weighted nodal
    average of the same cells, periodic along periodic axes and clipped to
    the sampled open volume. It is a second reconstruction of the cell
    averages, not a smoothing of the PLIC mesh: it has no grid-plane facets
    by construction, but it is not volume-exact and cannot represent liquid
    thinner than about half a cell (sub-cell films and one-cell sheets).

The legacy raw-array path is the nodal reconstruction. Both paths sample
passive attributes without a height-field assumption.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from enum import IntFlag
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.measure import marching_cubes

from shoreloop.config import BC, Grid3DConfig

if TYPE_CHECKING:
    from shoreloop.physics3d.state import State3D


class OpenBoundary(IntFlag):
    """Bit labels carried by vertices and boundary edges on open box sides."""

    NONE = 0
    X_LO = 1 << 0
    X_HI = 1 << 1
    Y_LO = 1 << 2
    Y_HI = 1 << 3
    Z_LO = 1 << 4
    Z_HI = 1 << 5


_OPEN_BOUNDARY_BITS = {
    "xlo": OpenBoundary.X_LO,
    "xhi": OpenBoundary.X_HI,
    "ylo": OpenBoundary.Y_LO,
    "yhi": OpenBoundary.Y_HI,
    "zlo": OpenBoundary.Z_LO,
    "zhi": OpenBoundary.Z_HI,
}


@dataclass(frozen=True)
class WaterSurfaceMesh:
    """General water-air surface and its render-time vertex attributes."""

    vertices: np.ndarray
    triangles: np.ndarray
    normals: np.ndarray
    velocity: np.ndarray
    foam: np.ndarray
    vertex_open_boundaries: np.ndarray
    open_boundary_edges: np.ndarray
    open_boundary_labels: np.ndarray

    @classmethod
    def read(cls, path: str | Path) -> WaterSurfaceMesh:
        """Read the direct NumPy interchange written by :meth:`write`."""
        with np.load(path) as saved:
            return cls(
                vertices=saved["vertices"].copy(),
                triangles=saved["triangles"].copy(),
                normals=saved["normals"].copy(),
                velocity=saved["velocity"].copy(),
                foam=saved["foam"].copy(),
                vertex_open_boundaries=(
                    saved["vertex_open_boundaries"].copy()
                ),
                open_boundary_edges=saved["open_boundary_edges"].copy(),
                open_boundary_labels=saved["open_boundary_labels"].copy(),
            )

    def write(self, path: str | Path) -> Path:
        """Write the direct NumPy interchange consumed by later render work."""
        destination = Path(path)
        if destination.suffix.lower() != ".npz":
            raise ValueError("water surface output must use a .npz suffix")
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            destination,
            vertices=self.vertices,
            triangles=self.triangles,
            normals=self.normals,
            velocity=self.velocity,
            foam=self.foam,
            vertex_open_boundaries=self.vertex_open_boundaries,
            open_boundary_edges=self.open_boundary_edges,
            open_boundary_labels=self.open_boundary_labels,
        )
        return destination


def _field(
    name: str,
    values: np.ndarray,
    shape: tuple[int, ...],
) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise FloatingPointError(f"{name} contains non-finite values")
    return result


def _cell_to_nodes(
    values: np.ndarray,
    weights: np.ndarray,
    periodic: tuple[bool, bool, bool] = (False, False, False),
) -> np.ndarray:
    """Average cell-centred scalars or vectors onto the corner lattice."""
    cell_shape = values.shape[:3]
    component_shape = values.shape[3:]
    node_shape = tuple(count + 1 for count in cell_shape)
    total = np.zeros((*node_shape, *component_shape), dtype=np.float64)
    weight_sum = np.zeros(node_shape, dtype=np.float64)
    weighted = values.astype(np.float64)
    if component_shape:
        weighted = weighted * weights[(...,) + (None,) * len(component_shape)]
    else:
        weighted = weighted * weights

    for di in (0, 1):
        for dj in (0, 1):
            for dk in (0, 1):
                target = (
                    slice(di, di + cell_shape[0]),
                    slice(dj, dj + cell_shape[1]),
                    slice(dk, dk + cell_shape[2]),
                )
                total[target] += weighted
                weight_sum[target] += weights

    # A periodic box has one geometric node represented at both array ends.
    # Combine the two adjacent-cell contributions, then write the same nodal
    # value at both coordinates before extraction.
    for axis, enabled in enumerate(periodic):
        if not enabled:
            continue
        low = [slice(None)] * total.ndim
        high = [slice(None)] * total.ndim
        low[axis] = 0
        high[axis] = -1
        combined_total = total[tuple(low)] + total[tuple(high)]
        combined_weight = weight_sum[tuple(low[:3])] + weight_sum[
            tuple(high[:3])
        ]
        total[tuple(low)] = combined_total
        total[tuple(high)] = combined_total
        weight_sum[tuple(low[:3])] = combined_weight
        weight_sum[tuple(high[:3])] = combined_weight

    denominator = weight_sum[(...,) + (None,) * len(component_shape)]
    np.divide(total, denominator, out=total, where=denominator > 0.0)
    return total.astype(np.float32)


def _sample_nodes(
    values: np.ndarray,
    vertices: np.ndarray,
    grid: Grid3DConfig,
) -> np.ndarray:
    """Trilinearly sample one nodal scalar or vector field."""
    origin = np.asarray((grid.x0, grid.y0, grid.z0), dtype=np.float64)
    spacing = np.asarray((grid.dx, grid.dy, grid.dz), dtype=np.float64)
    coordinates = (vertices.astype(np.float64) - origin) / spacing
    extent = np.asarray((grid.Lx, grid.Ly, grid.Lz), dtype=np.float64)
    for axis in range(3):
        source_type = vertices.dtype
        at_low = vertices[:, axis] == np.asarray(
            origin[axis], dtype=source_type,
        )
        at_high = vertices[:, axis] == np.asarray(
            origin[axis] + extent[axis], dtype=source_type,
        )
        coordinates[at_low, axis] = 0.0
        coordinates[at_high, axis] = values.shape[axis] - 1
    lower = np.floor(coordinates).astype(np.int64)
    upper = np.minimum(lower + 1, np.asarray(values.shape[:3]) - 1)
    fraction = coordinates - lower
    result = np.zeros((len(vertices), *values.shape[3:]), dtype=np.float64)
    for di in (0, 1):
        for dj in (0, 1):
            for dk in (0, 1):
                index = tuple(
                    np.where(bit == 0, lower[:, axis], upper[:, axis])
                    for axis, bit in enumerate((di, dj, dk))
                )
                weight = np.prod(
                    np.where(
                        np.asarray((di, dj, dk), dtype=bool),
                        fraction,
                        1.0 - fraction,
                    ),
                    axis=1,
                )
                if values.ndim == 3:
                    result += weight * values[index]
                else:
                    result += weight[:, None] * values[index]
    return result.astype(np.float32)


def _orient_triangles(
    vertices: np.ndarray,
    triangles: np.ndarray,
    water_to_air_normals: np.ndarray,
) -> np.ndarray:
    """Make every face winding agree with the water-to-air field."""
    oriented = triangles.astype(np.int32, copy=True)
    points = vertices[oriented]
    face_normals = np.cross(points[:, 1] - points[:, 0],
                            points[:, 2] - points[:, 0])
    reference = np.sum(water_to_air_normals[oriented], axis=1)
    flip = np.einsum("ij,ij->i", face_normals, reference) < 0.0
    oriented[flip] = oriented[flip][:, (0, 2, 1)]
    return oriented


def _extend_through_solids(
    values: np.ndarray,
    fluid_nodes: np.ndarray,
    spacing: tuple[float, float, float],
) -> np.ndarray:
    """Extend a render scalar from the nearest fluid node into solid nodes."""
    nearest = distance_transform_edt(
        ~fluid_nodes, sampling=spacing,
        return_distances=False, return_indices=True,
    )
    extended = values.copy()
    extended[~fluid_nodes] = values[tuple(nearest[:, ~fluid_nodes])]
    return extended


def _clip_to_open_volume(
    vertices: np.ndarray,
    triangles: np.ndarray,
    open_fraction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Clip oriented triangles against the sampled ``theta=0.5`` boundary."""
    output_vertices: list[np.ndarray] = []
    output_triangles: list[tuple[int, int, int]] = []
    original_vertices: dict[int, int] = {}
    edge_vertices: dict[tuple[int, int], int] = {}

    def original(index: int) -> int:
        if index not in original_vertices:
            original_vertices[index] = len(output_vertices)
            output_vertices.append(vertices[index])
        return original_vertices[index]

    def intersection(first: int, second: int) -> int:
        first_value = float(open_fraction[first])
        second_value = float(open_fraction[second])
        if first_value == 0.5:
            return original(first)
        if second_value == 0.5:
            return original(second)
        key = (min(first, second), max(first, second))
        if key not in edge_vertices:
            # Orient every cut edge from lower to higher open fraction.  The
            # same periodic edge is traversed in opposite directions on the
            # two box sides; a scalar-defined order makes its FP32
            # interpolation bit-identical without moving either input field.
            if first_value > second_value:
                first, second = second, first
                first_value, second_value = second_value, first_value
            fraction = (0.5 - first_value) / (second_value - first_value)
            edge_vertices[key] = len(output_vertices)
            output_vertices.append(
                vertices[first] + fraction * (vertices[second] - vertices[first]),
            )
        return edge_vertices[key]

    for triangle in triangles:
        polygon: list[int] = []
        for first, second in zip(triangle, np.roll(triangle, -1), strict=True):
            first = int(first)
            second = int(second)
            first_inside = open_fraction[first] >= 0.5
            second_inside = open_fraction[second] >= 0.5
            if first_inside:
                index = original(first)
                if not polygon or polygon[-1] != index:
                    polygon.append(index)
            if first_inside != second_inside:
                index = intersection(first, second)
                if not polygon or polygon[-1] != index:
                    polygon.append(index)
        if len(polygon) > 1 and polygon[0] == polygon[-1]:
            polygon.pop()
        polygon = list(dict.fromkeys(polygon))
        for index in range(1, len(polygon) - 1):
            output_triangles.append(
                (polygon[0], polygon[index], polygon[index + 1]),
            )

    clipped_vertices = np.asarray(
        output_vertices, dtype=np.float32,
    ).reshape((-1, 3))
    clipped_triangles = np.asarray(
        output_triangles, dtype=np.int32,
    ).reshape((-1, 3))
    if len(clipped_triangles) == 0:
        return clipped_vertices[:0], clipped_triangles

    points = clipped_vertices[clipped_triangles]
    area_vectors = np.cross(
        points[:, 1] - points[:, 0], points[:, 2] - points[:, 0],
    )
    clipped_triangles = clipped_triangles[
        np.any(area_vectors != 0.0, axis=1)
    ]
    if len(clipped_triangles) == 0:
        return clipped_vertices[:0], clipped_triangles

    used = np.unique(clipped_triangles)
    compact = np.empty(len(clipped_vertices), dtype=np.int32)
    compact[used] = np.arange(len(used), dtype=np.int32)
    return clipped_vertices[used], compact[clipped_triangles]


def _vertex_normals(
    vertices: np.ndarray,
    triangles: np.ndarray,
    *,
    grid: Grid3DConfig | None = None,
    periodic: tuple[bool, bool, bool] = (False, False, False),
) -> np.ndarray:
    """Area-weighted normals, including faces across periodic box seams."""
    points = vertices[triangles]
    face_normals = np.cross(points[:, 1] - points[:, 0],
                            points[:, 2] - points[:, 0])
    normals = np.zeros_like(vertices, dtype=np.float64)
    for corner in range(3):
        np.add.at(normals, triangles[:, corner], face_normals)
    if any(periodic):
        if grid is None:
            raise ValueError("periodic vertex normals require the source grid")
        bounds = (
            (grid.x0, grid.x0 + grid.Lx),
            (grid.y0, grid.y0 + grid.Ly),
            (grid.z0, grid.z0 + grid.Lz),
        )
        for axis, enabled in enumerate(periodic):
            if not enabled:
                continue
            low = np.flatnonzero(
                vertices[:, axis] == np.asarray(
                    bounds[axis][0], dtype=vertices.dtype,
                )
            )
            high = np.flatnonzero(
                vertices[:, axis] == np.asarray(
                    bounds[axis][1], dtype=vertices.dtype,
                )
            )
            if len(low) == 0 and len(high) == 0:
                continue
            transverse = tuple(index for index in range(3) if index != axis)
            low = low[np.lexsort((
                vertices[low, transverse[1]],
                vertices[low, transverse[0]],
            ))]
            high = high[np.lexsort((
                vertices[high, transverse[1]],
                vertices[high, transverse[0]],
            ))]
            if (
                len(low) != len(high)
                or not np.array_equal(
                    vertices[low][:, transverse],
                    vertices[high][:, transverse],
                )
            ):
                raise ValueError(
                    "extracted surface does not match across periodic boundary",
                )
            combined = normals[low] + normals[high]
            normals[low] = combined
            normals[high] = combined
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    np.divide(normals, lengths, out=normals, where=lengths > 0.0)
    return normals.astype(np.float32)


def _field_gradient(
    values: np.ndarray,
    spacing: tuple[float, float, float],
    periodic: tuple[bool, bool, bool] = (False, False, False),
) -> np.ndarray:
    """Return the physical-coordinate gradient on the nodal lattice."""
    values64 = values.astype(np.float64)
    derivatives = []
    for axis, (step, enabled) in enumerate(zip(
        spacing, periodic, strict=True,
    )):
        if enabled:
            indices = np.arange(values64.shape[axis] - 1)
            core = np.take(values64, indices, axis=axis)
            derivative = (
                np.roll(core, -1, axis=axis)
                - np.roll(core, 1, axis=axis)
            ) / (2.0 * step)
            derivative = np.concatenate((
                derivative,
                np.take(derivative, (0,), axis=axis),
            ), axis=axis)
        else:
            derivative = np.gradient(
                values64, step, axis=axis, edge_order=1,
            )
        derivatives.append(derivative)
    return np.stack(derivatives, axis=-1)


def _stable_normal_gradient(
    extraction_field: np.ndarray,
    spacing: tuple[float, float, float],
    periodic: tuple[bool, bool, bool] = (False, False, False),
) -> np.ndarray:
    """Build the low-pass water-to-air gradient used only for shading."""
    core_slices = tuple(
        slice(None, -1) if enabled else slice(None)
        for enabled in periodic
    )
    stable_field = gaussian_filter(
        extraction_field[core_slices].astype(np.float64),
        sigma=1.0,
        mode=tuple("wrap" if enabled else "nearest" for enabled in periodic),
    )
    for axis, enabled in enumerate(periodic):
        if enabled:
            stable_field = np.concatenate((
                stable_field,
                np.take(stable_field, (0,), axis=axis),
            ), axis=axis)
    return -_field_gradient(stable_field, spacing, periodic)


def _sample_unit_normals(
    normal_gradient: np.ndarray,
    vertices: np.ndarray,
    grid: Grid3DConfig,
    orientation_normals: np.ndarray | None = None,
) -> np.ndarray:
    normals = _sample_nodes(normal_gradient, vertices, grid).astype(np.float64)
    lengths = np.linalg.norm(normals, axis=1)
    if orientation_normals is not None:
        local = orientation_normals.astype(np.float64)
        local_lengths = np.linalg.norm(local, axis=1)
        stable = lengths > 0.0
        oriented = local_lengths > 0.0
        if np.any(~stable & ~oriented):
            raise FloatingPointError(
                "surface normal has neither scalar-gradient nor geometric "
                "orientation",
            )
        normals[stable] /= lengths[stable, None]
        local[oriented] /= local_lengths[oriented, None]
        normals[~stable] = local[~stable]
        alignment = np.sum(normals * local, axis=1)
        # A disagreement beyond 41 degrees means the one-cell filter is
        # blending opposing faces of a resolved thin sheet.  Transition back
        # to its local oriented geometry, reaching it fully at 60 degrees.
        stable_weight = np.clip((alignment - 0.5) / 0.25, 0.0, 1.0)
        stable_weight[~oriented] = 1.0
        normals = stable_weight[:, None] * normals + (
            1.0 - stable_weight
        )[:, None] * local
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    else:
        if np.any(lengths == 0.0):
            raise FloatingPointError("surface scalar gradient is zero")
        normals /= lengths[:, None]
    return normals.astype(np.float32)


def _domain_boundary_vertices(
    vertices: np.ndarray,
    grid: Grid3DConfig,
) -> np.ndarray:
    bounds = (
        (grid.x0, grid.x0 + grid.Lx),
        (grid.y0, grid.y0 + grid.Ly),
        (grid.z0, grid.z0 + grid.Lz),
    )
    spacing = (grid.dx, grid.dy, grid.dz)
    fixed = np.zeros(len(vertices), dtype=bool)
    for axis in range(3):
        fixed |= np.isclose(
            vertices[:, axis], bounds[axis][0], rtol=0.0,
            atol=2.0e-5 * spacing[axis],
        )
        fixed |= np.isclose(
            vertices[:, axis], bounds[axis][1], rtol=0.0,
            atol=2.0e-5 * spacing[axis],
        )
    return fixed


def _constrained_smooth(
    vertices: np.ndarray,
    triangles: np.ndarray,
    extraction_field: np.ndarray,
    normal_gradient: np.ndarray,
    raw_gradient: np.ndarray,
    grid: Grid3DConfig,
    periodic: tuple[bool, bool, bool],
    iterations: int,
) -> np.ndarray:
    """Relax mesh sampling tangentially and project back to ``alpha=0.5``."""
    result = vertices.astype(np.float64, copy=True)
    directed_edges = np.concatenate((
        triangles[:, (0, 1)], triangles[:, (1, 0)],
        triangles[:, (1, 2)], triangles[:, (2, 1)],
        triangles[:, (2, 0)], triangles[:, (0, 2)],
    ))
    neighbour_count = np.bincount(
        directed_edges[:, 0], minlength=len(result),
    ).astype(np.float64)
    fixed = _domain_boundary_vertices(result, grid)
    max_step = 0.2 * min(grid.dx, grid.dy, grid.dz)
    for _ in range(iterations):
        neighbour_sum = np.zeros_like(result)
        np.add.at(
            neighbour_sum, directed_edges[:, 0], result[directed_edges[:, 1]],
        )
        displacement = neighbour_sum / neighbour_count[:, None] - result
        normals = _sample_unit_normals(
            normal_gradient, result, grid,
            _vertex_normals(
                result, triangles, grid=grid, periodic=periodic,
            ),
        )
        displacement -= (
            np.sum(displacement * normals, axis=1, keepdims=True) * normals
        )
        displacement[fixed] = 0.0
        length = np.linalg.norm(displacement, axis=1)
        scale = np.minimum(1.0, max_step / np.maximum(length, max_step))
        proposed = result + scale[:, None] * displacement

        sampled_field = _sample_nodes(
            extraction_field, proposed, grid,
        ).astype(np.float64)
        sampled_gradient = _sample_nodes(
            raw_gradient, proposed, grid,
        ).astype(np.float64)
        gradient_norm_squared = np.sum(
            sampled_gradient * sampled_gradient, axis=1,
        )
        projection_scale = np.zeros(len(proposed), dtype=np.float64)
        np.divide(
            0.5 - sampled_field,
            gradient_norm_squared,
            out=projection_scale,
            where=gradient_norm_squared > 0.0,
        )
        proposed += projection_scale[:, None] * sampled_gradient
        total_displacement = proposed - result
        total_length = np.linalg.norm(total_displacement, axis=1)
        total_scale = np.minimum(
            1.0,
            max_step / np.maximum(total_length, max_step),
        )
        proposed = result + total_scale[:, None] * total_displacement
        proposed[fixed] = vertices[fixed]
        result = proposed
    return result.astype(np.float32)


def _boundary_labels(
    vertices: np.ndarray,
    grid: Grid3DConfig,
    open_boundaries: Collection[str],
) -> np.ndarray:
    labels = np.zeros(len(vertices), dtype=np.uint8)
    limits = (
        (grid.x0, grid.x0 + grid.Lx),
        (grid.y0, grid.y0 + grid.Ly),
        (grid.z0, grid.z0 + grid.Lz),
    )
    spacing = (grid.dx, grid.dy, grid.dz)
    for side in open_boundaries:
        bit = _OPEN_BOUNDARY_BITS[side]
        axis = "xyz".index(side[0])
        value = limits[axis][side.endswith("hi")]
        selected = np.isclose(
            vertices[:, axis], value, rtol=0.0,
            atol=2.0e-5 * spacing[axis],
        )
        labels[selected] |= np.uint8(bit)
    return labels


def _labelled_boundary_edges(
    triangles: np.ndarray,
    vertex_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.concatenate((
        triangles[:, (0, 1)], triangles[:, (1, 2)], triangles[:, (2, 0)],
    ))
    edges.sort(axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = unique[counts == 1]
    labels = vertex_labels[boundary[:, 0]] & vertex_labels[boundary[:, 1]]
    selected = labels != 0
    return boundary[selected].astype(np.int32), labels[selected]


def _empty_mesh() -> WaterSurfaceMesh:
    return WaterSurfaceMesh(
        vertices=np.empty((0, 3), dtype=np.float32),
        triangles=np.empty((0, 3), dtype=np.int32),
        normals=np.empty((0, 3), dtype=np.float32),
        velocity=np.empty((0, 3), dtype=np.float32),
        foam=np.empty(0, dtype=np.float32),
        vertex_open_boundaries=np.empty(0, dtype=np.uint8),
        open_boundary_edges=np.empty((0, 2), dtype=np.int32),
        open_boundary_labels=np.empty(0, dtype=np.uint8),
    )


def extract_free_surface_mesh(
    alpha: np.ndarray,
    grid: Grid3DConfig,
    *,
    open_fraction: np.ndarray | None = None,
    velocity: np.ndarray | None = None,
    foam: np.ndarray | None = None,
    open_boundaries: Collection[str] = (),
    periodic_axes: Collection[str] = (),
    smoothing_iterations: int = 0,
) -> WaterSurfaceMesh:
    """Extract an oriented ``alpha=0.5`` mesh from one saved VOF frame.

    ``open_fraction`` is the solver's cell-volume fraction ``theta``.  It
    weights the VOF reconstruction and clips the extracted sheet at the sampled
    solid boundary, so solid cells cannot manufacture water-air faces.
    ``smoothing_iterations`` optionally relaxes vertices tangentially by at
    most 0.2 cell per pass and projects them back to the unsmoothed isosurface.
    """
    shape = grid.shape
    alpha_cell = _field("alpha", alpha, shape)
    if float(np.min(alpha_cell)) < 0.0 or float(np.max(alpha_cell)) > 1.0:
        raise ValueError("alpha must stay in [0, 1]")
    theta = (
        np.ones(shape, dtype=np.float32)
        if open_fraction is None
        else _field("open_fraction", open_fraction, shape)
    )
    if float(np.min(theta)) < 0.0 or float(np.max(theta)) > 1.0:
        raise ValueError("open_fraction must stay in [0, 1]")
    requested_boundaries = tuple(open_boundaries)
    unknown = set(requested_boundaries) - _OPEN_BOUNDARY_BITS.keys()
    if unknown:
        raise ValueError(f"unknown open-domain sides: {sorted(unknown)}")
    requested_periodic = tuple(periodic_axes)
    unknown_periodic = set(requested_periodic) - set("xyz")
    if unknown_periodic:
        raise ValueError(f"unknown periodic axes: {sorted(unknown_periodic)}")
    periodic = tuple(axis in requested_periodic for axis in "xyz")
    if (isinstance(smoothing_iterations, bool)
            or not isinstance(smoothing_iterations, int)
            or smoothing_iterations < 0):
        raise ValueError("smoothing_iterations must be a non-negative integer")

    nodal_alpha = _cell_to_nodes(alpha_cell, theta, periodic)
    nodal_open_fraction = _cell_to_nodes(
        theta, np.ones(shape, dtype=np.float32), periodic,
    )
    fluid_nodes = nodal_open_fraction >= 0.5
    active = nodal_alpha[fluid_nodes]
    if (active.size == 0 or float(np.min(active)) >= 0.5
            or float(np.max(active)) <= 0.5):
        return _empty_mesh()
    spacing = (grid.dx, grid.dy, grid.dz)
    extraction_field = _extend_through_solids(
        nodal_alpha, fluid_nodes, spacing,
    )
    normal_gradient = _stable_normal_gradient(
        extraction_field, spacing, periodic,
    )
    raw_gradient = _field_gradient(extraction_field, spacing, periodic)

    vertices, triangles, normals, _ = marching_cubes(
        extraction_field,
        level=0.5,
        spacing=spacing,
        gradient_direction="descent",
        step_size=1,
        allow_degenerate=False,
        method="lewiner",
    )
    vertices += np.asarray((grid.x0, grid.y0, grid.z0), dtype=np.float32)
    triangles = _orient_triangles(vertices, triangles, normals)
    if smoothing_iterations:
        vertices = _constrained_smooth(
            vertices, triangles, extraction_field, normal_gradient,
            raw_gradient, grid, periodic, smoothing_iterations,
        )
    vertex_open_fraction = _sample_nodes(
        nodal_open_fraction, vertices, grid,
    )
    vertices, triangles = _clip_to_open_volume(
        vertices, triangles, vertex_open_fraction,
    )
    if len(triangles) == 0:
        return _empty_mesh()
    normals = _sample_unit_normals(
        normal_gradient, vertices, grid,
        _vertex_normals(
            vertices, triangles, grid=grid, periodic=periodic,
        ),
    )

    if velocity is None:
        vertex_velocity = np.zeros((len(vertices), 3), dtype=np.float32)
    else:
        velocity_cell = _field("velocity", velocity, (*shape, 3))
        vertex_velocity = _sample_nodes(
            _cell_to_nodes(velocity_cell, theta, periodic), vertices, grid,
        )
    if foam is None:
        vertex_foam = np.zeros(len(vertices), dtype=np.float32)
    else:
        foam_cell = _field("foam", foam, shape)
        vertex_foam = _sample_nodes(
            _cell_to_nodes(foam_cell, theta, periodic), vertices, grid,
        )

    vertex_labels = _boundary_labels(
        vertices, grid, requested_boundaries,
    )
    boundary_edges, boundary_edge_labels = _labelled_boundary_edges(
        triangles, vertex_labels,
    )
    return WaterSurfaceMesh(
        vertices=vertices.astype(np.float32),
        triangles=triangles,
        normals=normals,
        velocity=vertex_velocity,
        foam=vertex_foam,
        vertex_open_boundaries=vertex_labels,
        open_boundary_edges=boundary_edges,
        open_boundary_labels=boundary_edge_labels,
    )


RENDER_SURFACES = ("plic_boundary", "nodal_isosurface")


def extract_state_free_surface_mesh(
    state: State3D,
    *,
    smoothing_iterations: int = 0,
    render_surface: str | None = None,
    reconstruction=None,
) -> WaterSurfaceMesh:
    """Download one solver frame and extract its offline render surface.

    ``render_surface`` names one of :data:`RENDER_SURFACES`. ``None`` selects
    the state's retained representation: the cut/PLIC boundary of a
    ``liquid_mimetic`` state, the nodal isosurface otherwise. The choice
    changes only the extracted mesh, never the state. A ``reconstruction``
    (the state's own PLIC planes, shared with other consumers of the same
    frame) applies to the ``plic_boundary`` surface only.
    """
    if render_surface is not None and render_surface not in RENDER_SURFACES:
        raise ValueError(
            f"unknown render surface {render_surface!r}; choose from {RENDER_SURFACES}",
        )
    if reconstruction is not None and render_surface != "plic_boundary":
        raise ValueError("a supplied reconstruction belongs to the plic_boundary surface")
    retained_boundary = state.cfg.pressure_discretization == "liquid_mimetic"
    if render_surface is None:
        render_surface = "plic_boundary" if retained_boundary else "nodal_isosurface"
    if render_surface == "plic_boundary" and not retained_boundary:
        raise ValueError("plic_boundary needs the retained cut/PLIC planes of a liquid_mimetic state")
    if retained_boundary:
        if smoothing_iterations:
            raise ValueError("liquid_mimetic render surfaces are unsmoothed reconstructions of the retained field")
        if render_surface == "plic_boundary":
            from shoreloop.render.liquid_surface import extract_liquid_surface_mesh
            return extract_liquid_surface_mesh(state, reconstruction=reconstruction)
    u = state.get_interior("u")
    v = state.get_interior("v")
    w = state.get_interior("w")
    velocity = np.stack((
        0.5 * (u[:-1] + u[1:]),
        0.5 * (v[:, :-1] + v[:, 1:]),
        0.5 * (w[:, :, :-1] + w[:, :, 1:]),
    ), axis=-1)
    open_boundaries = tuple(
        side for side in _OPEN_BOUNDARY_BITS
        if getattr(state.cfg.bc, side) == BC.OPEN
    )
    periodic_axes = tuple(
        axis for axis in "xyz"
        if (
            getattr(state.cfg.bc, f"{axis}lo") == BC.PERIODIC
            and getattr(state.cfg.bc, f"{axis}hi") == BC.PERIODIC
        )
    )
    return extract_free_surface_mesh(
        state.get_interior("alpha"),
        state.cfg.grid,
        open_fraction=state.get_geometry_interior("theta"),
        velocity=velocity,
        foam=state.get_interior("foam"),
        open_boundaries=open_boundaries,
        periodic_axes=periodic_axes,
        smoothing_iterations=smoothing_iterations,
    )
