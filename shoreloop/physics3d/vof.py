"""Water-fraction initialisation and conservative volume diagnostics.

The authoritative phase field is ``alpha``: zero is air and one is water.
Analytic shapes are sampled at deterministic sub-cell midpoints.  This is a
bring-up initialiser, not a second interface representation; later PLIC work
always reconstructs from the resulting volume fractions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Sequence

import warp as wp

from shoreloop.physics3d.boundary import fill_field_boundaries
from shoreloop.physics3d.polyhedron import clipped_box_moments
from shoreloop.physics3d.state import State3D


@dataclass(frozen=True)
class VOFDiagnostics:
    """Global water-volume and boundedness measurements."""

    volume: float
    alpha_min: float
    alpha_max: float
    clipped_volume: float
    air_cells: int
    mixed_cells: int
    water_cells: int


@wp.kernel
def _initialize_joint_plane(
    alpha: wp.array3d(dtype=wp.float32),
    theta: wp.array3d(dtype=wp.float32),
    open_normals: wp.array3d(dtype=wp.vec3),
    open_offsets: wp.array3d(dtype=wp.float32),
    normal: wp.vec3, offset: wp.float32,
    origin: wp.vec3, spacing: wp.vec3, ng: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    capacity = theta[I, J, K]
    alpha[I, J, K] = wp.float32(0.0)
    if capacity > wp.float32(0.0):
        cell_origin = origin + wp.cw_mul(spacing, wp.vec3(wp.float32(i), wp.float32(j), wp.float32(k)))
        local_normal = wp.cw_mul(normal, spacing)
        scale = wp.abs(local_normal[0]) + wp.abs(local_normal[1]) + wp.abs(local_normal[2])
        local_offset = (offset - wp.dot(normal, cell_origin)) / scale
        volume = clipped_box_moments(
            wp.vec3(wp.float32(0.0)), wp.vec3(wp.float32(1.0)),
            open_normals[I, J, K], open_offsets[I, J, K],
            local_normal / scale, local_offset,
        )[0]
        alpha[I, J, K] = wp.clamp(volume / capacity, wp.float32(0.0), wp.float32(1.0))


@wp.kernel
def _initialize_plane(
    alpha: wp.array3d(dtype=wp.float32),
    normal: wp.vec3,
    offset: wp.float32,
    x0: wp.float32,
    y0: wp.float32,
    z0: wp.float32,
    dx: wp.float32,
    dy: wp.float32,
    dz: wp.float32,
    samples: int,
    ng: int,
):
    i, j, k = wp.tid()
    inside = int(0)
    inv_samples = wp.float32(1.0) / wp.float32(samples)
    for si in range(samples):
        x = x0 + (wp.float32(i) + (wp.float32(si) + wp.float32(0.5)) * inv_samples) * dx
        for sj in range(samples):
            y = y0 + (wp.float32(j) + (wp.float32(sj) + wp.float32(0.5)) * inv_samples) * dy
            for sk in range(samples):
                z = z0 + (wp.float32(k) + (wp.float32(sk) + wp.float32(0.5)) * inv_samples) * dz
                if wp.dot(normal, wp.vec3(x, y, z)) <= offset:
                    inside += 1
    count = samples * samples * samples
    alpha[i + ng, j + ng, k + ng] = wp.float32(inside) / wp.float32(count)


@wp.kernel
def _initialize_box(
    alpha: wp.array3d(dtype=wp.float32),
    lower: wp.vec3,
    upper: wp.vec3,
    x0: wp.float32,
    y0: wp.float32,
    z0: wp.float32,
    dx: wp.float32,
    dy: wp.float32,
    dz: wp.float32,
    samples: int,
    ng: int,
):
    i, j, k = wp.tid()
    inside = int(0)
    inv_samples = wp.float32(1.0) / wp.float32(samples)
    for si in range(samples):
        x = x0 + (wp.float32(i) + (wp.float32(si) + wp.float32(0.5)) * inv_samples) * dx
        for sj in range(samples):
            y = y0 + (wp.float32(j) + (wp.float32(sj) + wp.float32(0.5)) * inv_samples) * dy
            for sk in range(samples):
                z = z0 + (wp.float32(k) + (wp.float32(sk) + wp.float32(0.5)) * inv_samples) * dz
                if (x >= lower[0] and x <= upper[0]
                        and y >= lower[1] and y <= upper[1]
                        and z >= lower[2] and z <= upper[2]):
                    inside += 1
    count = samples * samples * samples
    alpha[i + ng, j + ng, k + ng] = wp.float32(inside) / wp.float32(count)


@wp.kernel
def _initialize_sphere(
    alpha: wp.array3d(dtype=wp.float32),
    centre: wp.vec3,
    radius_squared: wp.float32,
    x0: wp.float32,
    y0: wp.float32,
    z0: wp.float32,
    dx: wp.float32,
    dy: wp.float32,
    dz: wp.float32,
    samples: int,
    ng: int,
):
    i, j, k = wp.tid()
    inside = int(0)
    inv_samples = wp.float32(1.0) / wp.float32(samples)
    for si in range(samples):
        x = x0 + (wp.float32(i) + (wp.float32(si) + wp.float32(0.5)) * inv_samples) * dx
        for sj in range(samples):
            y = y0 + (wp.float32(j) + (wp.float32(sj) + wp.float32(0.5)) * inv_samples) * dy
            for sk in range(samples):
                z = z0 + (wp.float32(k) + (wp.float32(sk) + wp.float32(0.5)) * inv_samples) * dz
                delta = wp.vec3(x, y, z) - centre
                if wp.dot(delta, delta) <= radius_squared:
                    inside += 1
    count = samples * samples * samples
    alpha[i + ng, j + ng, k + ng] = wp.float32(inside) / wp.float32(count)


@wp.kernel
def _reduce_vof(
    alpha: wp.array3d(dtype=wp.float32),
    theta: wp.array3d(dtype=wp.float32),
    alpha_sum: wp.array(dtype=wp.float64),
    bounds: wp.array(dtype=wp.float32),
    counts: wp.array(dtype=wp.int32),
    invalid: wp.array(dtype=wp.int32),
    tolerance: wp.float32,
    ng: int,
):
    i, j, k = wp.tid()
    value = alpha[i + ng, j + ng, k + ng]
    if not wp.isfinite(value):
        wp.atomic_max(invalid, 0, 1)
        return

    wp.atomic_add(
        alpha_sum, 0,
        wp.float64(value) * wp.float64(theta[i + ng, j + ng, k + ng]),
    )
    wp.atomic_min(bounds, 0, value)
    wp.atomic_max(bounds, 1, value)
    if value <= tolerance:
        wp.atomic_add(counts, 0, 1)
    elif value >= wp.float32(1.0) - tolerance:
        wp.atomic_add(counts, 2, 1)
    else:
        wp.atomic_add(counts, 1, 1)


def _vec3(name: str, values: Sequence[float]) -> tuple[float, float, float]:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain three real values") from exc
    if len(result) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} values must be finite")
    return result  # type: ignore[return-value]


def _samples(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("samples_per_axis must be a positive integer")
    if value > 16:
        raise ValueError("samples_per_axis above 16 is not supported")
    return value


def _shape_launch(state: State3D, kernel: object, inputs: list[object]) -> None:
    wp.launch(
        kernel,
        dim=state.cfg.grid.shape,
        inputs=[state.alpha, *inputs, state.cfg.grid.ng],
        device=state.device,
    )
    fill_field_boundaries(state, "alpha")


def initialize_plane(
    state: State3D,
    normal: Sequence[float],
    offset: float,
    *,
    samples_per_axis: int = 4,
) -> None:
    """Fill water on ``dot(normal, x) <= offset``.

    The normal is normalised here, so it is also the water-to-air orientation
    recovered by interface reconstruction. ``offset`` is scaled consistently.
    """
    n = _vec3("normal", normal)
    magnitude = math.sqrt(sum(value * value for value in n))
    if magnitude == 0.0:
        raise ValueError("normal must be non-zero")
    if not math.isfinite(offset):
        raise ValueError("offset must be finite")
    n_unit = tuple(value / magnitude for value in n)
    g = state.cfg.grid
    _shape_launch(
        state,
        _initialize_plane,
        [wp.vec3(*n_unit), float(offset) / magnitude,
         g.x0, g.y0, g.z0, g.dx, g.dy, g.dz, _samples(samples_per_axis)],
    )


def initialize_joint_plane(state: State3D, normal: Sequence[float], offset: float) -> None:
    """Initialise a plane intersected with the state's reconstructed open cells.

    Unlike the sampled bring-up initialiser, this integrates thin volumes
    geometrically. Initialise solid geometry first. Existing checkpoints are
    never passed through this initialiser: their alpha*theta remains owned by
    their transport history.
    """
    n = _vec3("normal", normal)
    if sum(value * value for value in n) == 0.0:
        raise ValueError("normal must be non-zero")
    if not math.isfinite(offset):
        raise ValueError("offset must be finite")
    g = state.cfg.grid
    _shape_launch(state, _initialize_joint_plane, [
        state.theta, state.open_normals, state.open_offsets, wp.vec3(*n), offset,
        wp.vec3(g.x0, g.y0, g.z0), wp.vec3(g.dx, g.dy, g.dz),
    ])


def initialize_box(
    state: State3D,
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    samples_per_axis: int = 4,
) -> None:
    """Initialise the intersection of an axis-aligned water box and the grid."""
    lo = _vec3("lower", lower)
    hi = _vec3("upper", upper)
    if any(a >= b for a, b in zip(lo, hi)):
        raise ValueError("each lower box coordinate must be below upper")
    g = state.cfg.grid
    _shape_launch(
        state,
        _initialize_box,
        [wp.vec3(*lo), wp.vec3(*hi), g.x0, g.y0, g.z0,
         g.dx, g.dy, g.dz, _samples(samples_per_axis)],
    )


def initialize_sphere(
    state: State3D,
    centre: Sequence[float],
    radius: float,
    *,
    samples_per_axis: int = 4,
) -> None:
    """Initialise a sampled spherical water region."""
    centre3 = _vec3("centre", centre)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be positive and finite")
    g = state.cfg.grid
    _shape_launch(
        state,
        _initialize_sphere,
        [wp.vec3(*centre3), radius * radius, g.x0, g.y0, g.z0,
         g.dx, g.dy, g.dz, _samples(samples_per_axis)],
    )


def measure_vof(
    state: State3D,
    *,
    clipped_volume: float = 0.0,
    classification_tolerance: float = 1.0e-6,
) -> VOFDiagnostics:
    """Measure the physical ``alpha`` cells without downloading the field."""
    if not math.isfinite(clipped_volume) or clipped_volume < 0.0:
        raise ValueError("clipped_volume must be non-negative and finite")
    if (not math.isfinite(classification_tolerance)
            or classification_tolerance < 0.0
            or classification_tolerance >= 0.5):
        raise ValueError("classification_tolerance must be finite in [0, 0.5)")

    alpha_sum = wp.zeros(1, dtype=wp.float64, device=state.device)
    bounds = wp.array([math.inf, -math.inf], dtype=wp.float32,
                      device=state.device)
    counts = wp.zeros(3, dtype=wp.int32, device=state.device)
    invalid = wp.zeros(1, dtype=wp.int32, device=state.device)
    g = state.cfg.grid
    wp.launch(
        _reduce_vof,
        dim=g.shape,
        inputs=[state.alpha, state.theta, alpha_sum, bounds, counts, invalid,
                classification_tolerance, g.ng],
        device=state.device,
    )
    if int(invalid.numpy()[0]) != 0:
        raise FloatingPointError("non-finite water fraction in physical cells")
    bounds_host = bounds.numpy()
    counts_host = counts.numpy()
    return VOFDiagnostics(
        volume=float(alpha_sum.numpy()[0]) * g.dx * g.dy * g.dz,
        alpha_min=float(bounds_host[0]),
        alpha_max=float(bounds_host[1]),
        clipped_volume=float(clipped_volume),
        air_cells=int(counts_host[0]),
        mixed_cells=int(counts_host[1]),
        water_cells=int(counts_host[2]),
    )
