"""Deterministic PLIC plane geometry for unit Cartesian cells.

A reconstructed mixed cell stores a local plane

``dot(normal, xi) <= offset``, ``xi in [0, 1]^3``.

The normal points from water to air and is L1-normalised.  Scaling a physical
normal by the grid spacing before normalisation makes the same representation
valid on rectangular cells.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Sequence

import warp as wp

from shoreloop.physics3d.state import State3D
from shoreloop.physics3d.polyhedron import clipped_box_moments


_PLANE_ITERATIONS = 32
_NORMAL_EPSILON = 1.0e-12
# Retained by the separate legacy surface-area formula. Volume integration
# below keeps every nonzero component, even for nearly planar normals.
_FLOAT32_GEOMETRY_EPSILON = 1.0e-4


@dataclass(frozen=True)
class PLICReconstruction:
    """Cell-local mixed-cell planes and their reconstruction diagnostics."""

    normals: wp.array
    offsets: wp.array
    centroids: wp.array
    mixed_cells: int
    max_volume_error: float


def _normal_tuple(normal: Sequence[float]) -> tuple[float, float, float]:
    try:
        values = tuple(float(value) for value in normal)
    except (TypeError, ValueError) as exc:
        raise TypeError("normal must contain three real values") from exc
    if len(values) != 3:
        raise ValueError("normal must contain exactly three values")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("normal values must be finite")
    magnitude = sum(abs(value) for value in values)
    if magnitude <= _NORMAL_EPSILON:
        raise ValueError("normal must be non-zero")
    return tuple(value / magnitude for value in values)  # type: ignore[return-value]


def _positive_cube_volume_host(coefficients: tuple[float, float, float],
                               offset: float) -> float:
    active = tuple(value for value in coefficients if value > _NORMAL_EPSILON)
    total = sum(active)
    if offset <= 0.0:
        return 0.0
    if offset >= total:
        return 1.0
    if offset > 0.5 * total:
        return 1.0 - _positive_cube_volume_host(active + (0.0,) * (3 - len(active)),
                                                total - offset)

    if len(active) == 1:
        return offset / active[0]
    if len(active) == 2:
        a, b = active
        numerator = (offset ** 2 - max(offset - a, 0.0) ** 2
                     - max(offset - b, 0.0) ** 2)
        return numerator / (2.0 * a * b)

    a, b, c = active
    numerator = (
        offset ** 3
        - max(offset - a, 0.0) ** 3
        - max(offset - b, 0.0) ** 3
        - max(offset - c, 0.0) ** 3
        + max(offset - a - b, 0.0) ** 3
        + max(offset - a - c, 0.0) ** 3
        + max(offset - b - c, 0.0) ** 3
    )
    return numerator / (6.0 * a * b * c)


def unit_cube_volume(normal: Sequence[float], offset: float) -> float:
    """Return a unit-cube fraction in the L1-normalised plane convention."""
    if not math.isfinite(offset):
        raise ValueError("offset must be finite")
    n = _normal_tuple(normal)
    # Reflect negative-normal coordinates into a positive-coefficient cube.
    reflected_offset = offset - sum(value for value in n if value < 0.0)
    return min(1.0, max(0.0, _positive_cube_volume_host(
        tuple(abs(value) for value in n), reflected_offset,
    )))


def solve_plic_offset(
    normal: Sequence[float],
    fraction: float,
    *,
    iterations: int = 48,
) -> float:
    """Solve the monotone unit-cube plane offset for one water fraction.

    The returned offset corresponds to the L1-normalised form of ``normal``.
    Empty and full fractions return the geometric lower and upper plane bounds
    without entering the mixed-cell iteration.
    """
    n = _normal_tuple(normal)
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be finite in [0, 1]")
    if (isinstance(iterations, bool) or not isinstance(iterations, int)
            or iterations <= 0):
        raise ValueError("iterations must be a positive integer")
    lower = sum(value for value in n if value < 0.0)
    upper = sum(value for value in n if value > 0.0)
    if fraction == 0.0:
        return lower
    if fraction == 1.0:
        return upper
    for _ in range(iterations):
        midpoint = 0.5 * (lower + upper)
        if unit_cube_volume(n, midpoint) < fraction:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


@wp.func
def _positive_cube_volume(
    coefficients: wp.vec3,
    offset: wp.float32,
) -> wp.float32:
    # Sort a <= b <= c and use symmetry to integrate at most half the cube.
    # Factoring the cubic differences avoids cancellation for nearly planar
    # normals without dropping components that can exceed a thin film's depth.
    a = wp.min(coefficients[0], wp.min(coefficients[1], coefficients[2]))
    c = wp.max(coefficients[0], wp.max(coefficients[1], coefficients[2]))
    b = wp.max(wp.min(coefficients[0], coefficients[1]),
               wp.min(wp.max(coefficients[0], coefficients[1]), coefficients[2]))
    total = a+b+c
    if offset <= wp.float32(0.0):
        return wp.float32(0.0)
    if offset >= total:
        return wp.float32(1.0)
    complement = offset > wp.float32(0.5)*total
    if complement:
        offset = total-offset
    volume = wp.float32(0.0)
    if b == wp.float32(0.0):
        volume = offset/c
    elif a == wp.float32(0.0):
        if offset < b:
            volume = wp.float32(0.5)*(offset/b)*(offset/c)
        else:
            volume = (offset-wp.float32(0.5)*b)/c
    elif offset < a:
        volume = (offset/a)*(offset/b)*(offset/c)/wp.float32(6.0)
    elif offset >= a+b:
        volume = (offset-wp.float32(0.5)*(a+b))/c
    else:
        # [x^3-(x-a)^3-(x-b)_+^3-(x-c)_+^3] / (6abc).
        # In this branch (x-b)_+ and (x-c)_+ never exceed a.
        pb = wp.max(offset-b, wp.float32(0.0))
        pc = wp.max(offset-c, wp.float32(0.0))
        numerator = (wp.float32(3.0)*offset*(offset-a)+a*a
                     - (pb/a)*pb*pb - (pc/a)*pc*pc)
        volume = numerator/(wp.float32(6.0)*b*c)
    if complement:
        volume = wp.float32(1.0)-volume
    return wp.clamp(volume, wp.float32(0.0), wp.float32(1.0))


@wp.func
def _unit_cube_volume_l1(normal: wp.vec3, offset: wp.float32) -> wp.float32:
    reflected_offset = offset
    if normal[0] < wp.float32(0.0):
        reflected_offset -= normal[0]
    if normal[1] < wp.float32(0.0):
        reflected_offset -= normal[1]
    if normal[2] < wp.float32(0.0):
        reflected_offset -= normal[2]
    coefficients = wp.vec3(
        wp.abs(normal[0]), wp.abs(normal[1]), wp.abs(normal[2]),
    )
    return _positive_cube_volume(coefficients, reflected_offset)


@wp.func
def _solve_plane_offset_l1(normal: wp.vec3, fraction: wp.float32) -> wp.float32:
    lower = (wp.min(normal[0], wp.float32(0.0))
             + wp.min(normal[1], wp.float32(0.0))
             + wp.min(normal[2], wp.float32(0.0)))
    upper = (wp.max(normal[0], wp.float32(0.0))
             + wp.max(normal[1], wp.float32(0.0))
             + wp.max(normal[2], wp.float32(0.0)))
    for _ in range(_PLANE_ITERATIONS):
        midpoint = wp.float32(0.5) * (lower + upper)
        if _unit_cube_volume_l1(normal, midpoint) < fraction:
            lower = midpoint
        else:
            upper = midpoint
    return wp.float32(0.5) * (lower + upper)


@wp.func
def _solve_cut_plane_offset(
    normal: wp.vec3,
    retained_volume: wp.float32,
    open_normal: wp.vec3,
    open_offset: wp.float32,
) -> wp.float32:
    lower = (wp.min(normal[0], wp.float32(0.0))
             + wp.min(normal[1], wp.float32(0.0))
             + wp.min(normal[2], wp.float32(0.0)))
    upper = (wp.max(normal[0], wp.float32(0.0))
             + wp.max(normal[1], wp.float32(0.0))
             + wp.max(normal[2], wp.float32(0.0)))
    iterations = int(_PLANE_ITERATIONS)
    for _ in range(iterations):
        midpoint = wp.float32(0.5) * (lower + upper)
        moments = clipped_box_moments(
            wp.vec3(wp.float32(0.0)), wp.vec3(wp.float32(1.0)),
            open_normal, open_offset, normal, midpoint,
        )
        if moments[0] < retained_volume:
            lower = midpoint
        else:
            upper = midpoint
    return wp.float32(0.5) * (lower + upper)


@wp.kernel
def _reconstruct_planes(
    alpha: wp.array3d(dtype=wp.float32),
    theta: wp.array3d(dtype=wp.float32),
    open_normals: wp.array3d(dtype=wp.vec3),
    open_offsets: wp.array3d(dtype=wp.float32),
    physical_normals: wp.array3d(dtype=wp.vec3),
    local_normals: wp.array3d(dtype=wp.vec3),
    offsets: wp.array3d(dtype=wp.float32),
    centroids: wp.array3d(dtype=wp.vec3),
    counts: wp.array(dtype=wp.int32),
    maximum_error: wp.array(dtype=wp.float32),
    spacing: wp.vec3,
    mixed_tolerance: wp.float32,
    ng: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    fraction = alpha[I, J, K]
    capacity = theta[I, J, K]
    open_normal = open_normals[I, J, K]
    open_offset = open_offsets[I, J, K]
    local_normals[I, J, K] = wp.vec3(
        wp.float32(0.0), wp.float32(0.0), wp.float32(0.0),
    )
    offsets[I, J, K] = wp.float32(0.0)
    centroids[I, J, K] = wp.vec3(wp.float32(0.5))
    if fraction <= mixed_tolerance or capacity <= wp.float32(0.0):
        return
    if capacity < wp.float32(1.0) and wp.dot(open_normal, open_normal) == wp.float32(0.0):
        wp.atomic_add(counts, 2, 1)
        return
    if fraction >= wp.float32(1.0) - mixed_tolerance:
        if capacity < wp.float32(1.0):
            moments = clipped_box_moments(
                wp.vec3(wp.float32(0.0)), wp.vec3(wp.float32(1.0)),
                open_normal, open_offset, wp.vec3(wp.float32(0.0)), wp.float32(0.0),
            )
            centroids[I, J, K] = wp.vec3(moments[1], moments[2], moments[3]) / moments[0]
        return

    transformed = wp.cw_mul(physical_normals[I, J, K], spacing)
    magnitude = (wp.abs(transformed[0]) + wp.abs(transformed[1])
                 + wp.abs(transformed[2]))
    if not wp.isfinite(magnitude) or magnitude <= wp.float32(_NORMAL_EPSILON):
        wp.atomic_add(counts, 1, 1)
        return

    normal = transformed / magnitude
    offset = _solve_plane_offset_l1(normal, fraction)
    if capacity < wp.float32(1.0):
        offset = _solve_cut_plane_offset(
            normal, fraction * capacity, open_normal, open_offset,
        )
    local_normals[I, J, K] = normal
    offsets[I, J, K] = offset
    wp.atomic_add(counts, 0, 1)
    moments = clipped_box_moments(
        wp.vec3(wp.float32(0.0)), wp.vec3(wp.float32(1.0)),
        open_normal, open_offset, normal, offset,
    )
    if moments[0] <= wp.float32(0.0):
        wp.atomic_add(counts, 1, 1)
        return
    centroids[I, J, K] = wp.vec3(moments[1], moments[2], moments[3]) / moments[0]
    error = wp.abs(moments[0] / capacity - fraction)
    wp.atomic_max(maximum_error, 0, error)


def reconstruct_plic(
    state: State3D,
    physical_normals: wp.array,
    *,
    mixed_tolerance: float = 1.0e-6,
) -> PLICReconstruction:
    """Reconstruct retained water in the same open halfspace used by fluxes.

    A mixed cut cell satisfies V(cube intersect open intersect water) =
    alpha*theta. Centroids are cell-local water-volume averages, never an
    assumed water-filled Cartesian centre. This does not relocate pressure
    unknowns until the paired pressure discretisation consumes these samples.
    """
    if tuple(physical_normals.shape) != state.grid.cell_shape:
        raise ValueError(
            f"normal field must have padded shape {state.grid.cell_shape}, "
            f"got {tuple(physical_normals.shape)}",
        )
    if physical_normals.dtype != wp.vec3:
        raise TypeError(f"normal field must use vec3, got {physical_normals.dtype}")
    if (not math.isfinite(mixed_tolerance) or mixed_tolerance < 0.0
            or mixed_tolerance >= 0.5):
        raise ValueError("mixed_tolerance must be finite in [0, 0.5)")

    local_normals = wp.zeros(
        state.grid.cell_shape, dtype=wp.vec3, device=state.device,
    )
    offsets = wp.zeros(
        state.grid.cell_shape, dtype=wp.float32, device=state.device,
    )
    centroids = wp.zeros(state.grid.cell_shape, dtype=wp.vec3, device=state.device)
    counts = wp.zeros(3, dtype=wp.int32, device=state.device)
    maximum_error = wp.zeros(1, dtype=wp.float32, device=state.device)
    g = state.cfg.grid
    wp.launch(
        _reconstruct_planes,
        dim=g.shape,
        inputs=[state.alpha, state.theta, state.open_normals, state.open_offsets,
                physical_normals, local_normals, offsets, centroids, counts,
                maximum_error, wp.vec3(g.dx, g.dy, g.dz), mixed_tolerance,
                g.ng],
        device=state.device,
    )
    counts_host = counts.numpy()
    if int(counts_host[2]):
        raise ValueError("retained water in cut cells requires open-cell planes")
    invalid = int(counts_host[1])
    if invalid:
        raise RuntimeError(f"{invalid} mixed cells have invalid interface normals")
    return PLICReconstruction(
        normals=local_normals,
        offsets=offsets,
        centroids=centroids,
        mixed_cells=int(counts_host[0]),
        max_volume_error=float(maximum_error.numpy()[0]),
    )
