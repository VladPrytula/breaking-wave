"""PLIC curvature, Young--Laplace data, and capillary surface energy.

The transported VOF fraction remains the only interface state.  Curvature is
reconstructed afresh with signed height functions on graph patches and local
PLIC normal fits elsewhere, then extended solely so the ghost-fluid face
condition samples the same interface from either side.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import warp as wp

from shoreloop.config import BC
from shoreloop.physics3d.extrapolation import _wrapped_cell_index
from shoreloop.physics3d.interface import estimate_interface_normals
from shoreloop.physics3d.plic import (
    _FLOAT32_GEOMETRY_EPSILON,
    PLICReconstruction,
    reconstruct_plic,
)
from shoreloop.physics3d.state import State3D


CAPILLARY_MIXED_TOLERANCE = 1.0e-6
CAPILLARY_CUT_CELL_MIXED_TOLERANCE = _FLOAT32_GEOMETRY_EPSILON
# Curvature is the sum of two principal curvatures.  If |kappa| h exceeds four,
# at least one principal radius is below half a cell, outside this
# reconstruction's representable domain.
MAX_RESOLVED_GRID_SCALED_CURVATURE = 4.0


@dataclass(frozen=True)
class SurfaceTensionGeometry:
    """Resolved signed curvature, ownership, and interface-area diagnostics."""

    curvature: wp.array
    height_function_valid: wp.array
    plic_fit_valid: wp.array
    curvature_owner: wp.array
    surface_area: float
    max_abs_curvature: float
    mixed_cells: int
    height_function_cells: int
    unresolved_cells: int


@wp.func
def _mixed_fraction(value: wp.float32, tolerance: wp.float32) -> int:
    if value > tolerance and value < wp.float32(1.0) - tolerance:
        return 1
    return 0


@wp.func
def _capillary_mixed_fraction(
    value: wp.float32,
    open_fraction: wp.float32,
    mixed_tolerance: wp.float32,
) -> int:
    tolerance = mixed_tolerance
    if open_fraction < wp.float32(1.0) - mixed_tolerance:
        tolerance = wp.float32(CAPILLARY_CUT_CELL_MIXED_TOLERANCE)
    return _mixed_fraction(value, tolerance)


@wp.func
def _pure_phase(value: wp.float32, tolerance: wp.float32) -> int:
    if value <= tolerance:
        return 0
    if value >= wp.float32(1.0) - tolerance:
        return 1
    return -1


@wp.func
def _capillary_pure_phase(
    value: wp.float32,
    open_fraction: wp.float32,
    mixed_tolerance: wp.float32,
) -> int:
    tolerance = mixed_tolerance
    if open_fraction < wp.float32(1.0) - mixed_tolerance:
        tolerance = wp.float32(CAPILLARY_CUT_CELL_MIXED_TOLERANCE)
    return _pure_phase(value, tolerance)


@wp.func
def _plane_point(
    normal: wp.vec3,
    offset: wp.float32,
    origin: wp.vec3,
    spacing: wp.vec3,
) -> wp.vec3:
    metric = (
        wp.abs(normal[0]) * spacing[0]
        + wp.abs(normal[1]) * spacing[1]
        + wp.abs(normal[2]) * spacing[2]
    )
    centre_offset = wp.float32(0.5) * spacing
    centre = origin + centre_offset
    distance = (
        metric * offset - wp.dot(normal, centre_offset)
    )
    return centre + distance * normal


@wp.func
def _positive_part(value: wp.float32) -> wp.float32:
    return wp.max(value, wp.float32(0.0))


@wp.func
def _positive_cube_volume_derivative(
    coefficients: wp.vec3,
    offset: wp.float32,
) -> wp.float32:
    epsilon = wp.float32(_FLOAT32_GEOMETRY_EPSILON)
    a = coefficients[0]
    b = coefficients[1]
    c = coefficients[2]
    active = int(0)
    if a > epsilon:
        active += 1
    if b > epsilon:
        active += 1
    if c > epsilon:
        active += 1
    total = a + b + c
    if offset <= wp.float32(0.0) or offset >= total:
        return wp.float32(0.0)
    if offset > wp.float32(0.5) * total:
        offset = total - offset

    if active == 1:
        return wp.float32(1.0) / wp.max(a, wp.max(b, c))
    if active == 2:
        first = wp.float32(0.0)
        second = wp.float32(0.0)
        if a > epsilon:
            first = a
        if b > epsilon:
            if first == wp.float32(0.0):
                first = b
            else:
                second = b
        if c > epsilon:
            second = c
        numerator = (
            _positive_part(offset)
            - _positive_part(offset - first)
            - _positive_part(offset - second)
        )
        return numerator / (first * second)

    numerator = (
        _positive_part(offset) * _positive_part(offset)
        - _positive_part(offset - a) * _positive_part(offset - a)
        - _positive_part(offset - b) * _positive_part(offset - b)
        - _positive_part(offset - c) * _positive_part(offset - c)
        + _positive_part(offset - a - b) * _positive_part(offset - a - b)
        + _positive_part(offset - a - c) * _positive_part(offset - a - c)
        + _positive_part(offset - b - c) * _positive_part(offset - b - c)
    )
    return numerator / (wp.float32(2.0) * a * b * c)


@wp.func
def _plic_patch_area(
    physical_normal: wp.vec3,
    local_normal: wp.vec3,
    offset: wp.float32,
    spacing: wp.vec3,
) -> wp.float32:
    reflected_offset = offset
    if local_normal[0] < wp.float32(0.0):
        reflected_offset -= local_normal[0]
    if local_normal[1] < wp.float32(0.0):
        reflected_offset -= local_normal[1]
    if local_normal[2] < wp.float32(0.0):
        reflected_offset -= local_normal[2]
    coefficients = wp.vec3(
        wp.abs(local_normal[0]),
        wp.abs(local_normal[1]),
        wp.abs(local_normal[2]),
    )
    derivative = _positive_cube_volume_derivative(
        coefficients, reflected_offset,
    )
    metric = (
        wp.abs(physical_normal[0]) * spacing[0]
        + wp.abs(physical_normal[1]) * spacing[1]
        + wp.abs(physical_normal[2]) * spacing[2]
    )
    cell_volume = spacing[0] * spacing[1] * spacing[2]
    return cell_volume * derivative / metric


@wp.func
def _height_column(
    alpha: wp.array3d(dtype=wp.float32),
    i: int,
    j: int,
    k: int,
    axis: int,
    transverse_a: int,
    transverse_b: int,
    normal_sign: wp.float32,
    nx: int,
    ny: int,
    nz: int,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    ng: int,
) -> wp.float32:
    total = wp.float32(0.0)
    low = wp.float32(0.0)
    high = wp.float32(0.0)
    previous = wp.float32(0.0)
    graph = int(1)
    endpoint_tolerance = wp.float32(1.0e-4)
    for sample in range(13):
        offset = sample - 6
        qi = i
        qj = j
        qk = k
        if axis == 0:
            qi += offset
            qj += transverse_a
            qk += transverse_b
        elif axis == 1:
            qi += transverse_a
            qj += offset
            qk += transverse_b
        else:
            qi += transverse_a
            qj += transverse_b
            qk += offset
        qi = _wrapped_cell_index(qi, nx, periodic_x)
        qj = _wrapped_cell_index(qj, ny, periodic_y)
        qk = _wrapped_cell_index(qk, nz, periodic_z)
        if qi < 0 or qj < 0 or qk < 0:
            return wp.float32(-1.0)
        value = alpha[qi + ng, qj + ng, qk + ng]
        total += value
        if sample == 0:
            low = value
        else:
            if normal_sign > wp.float32(0.0):
                if value > previous + endpoint_tolerance:
                    graph = 0
            elif value < previous - endpoint_tolerance:
                graph = 0
        if sample == 12:
            high = value
        previous = value
    if graph == 0:
        return wp.float32(-1.0)
    if normal_sign > wp.float32(0.0):
        if (low < wp.float32(1.0) - endpoint_tolerance
                or high > endpoint_tolerance):
            return wp.float32(-1.0)
    else:
        if (low > endpoint_tolerance
                or high < wp.float32(1.0) - endpoint_tolerance):
            return wp.float32(-1.0)
    if normal_sign > wp.float32(0.0):
        return total
    return wp.float32(13.0) - total


@wp.kernel
def _apply_height_function_curvature(
    alpha: wp.array3d(dtype=wp.float32),
    theta: wp.array3d(dtype=wp.float32),
    normals: wp.array3d(dtype=wp.vec3),
    curvature: wp.array3d(dtype=wp.float32),
    valid: wp.array3d(dtype=wp.int32),
    accepted: wp.array(dtype=wp.int32),
    dx: wp.float32,
    dy: wp.float32,
    dz: wp.float32,
    nx: int,
    ny: int,
    nz: int,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    mixed_tolerance: wp.float32,
    ng: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    if _capillary_mixed_fraction(
        alpha[I, J, K], theta[I, J, K], mixed_tolerance,
    ) == 0:
        return
    normal = normals[I, J, K]
    axis = int(0)
    if wp.abs(normal[1]) > wp.abs(normal[0]):
        axis = 1
    if wp.abs(normal[2]) > wp.abs(normal[axis]):
        axis = 2
    normal_sign = wp.float32(1.0)
    if normal[axis] < wp.float32(0.0):
        normal_sign = -wp.float32(1.0)

    hmm = _height_column(
        alpha, i, j, k, axis, -1, -1, normal_sign,
        nx, ny, nz, periodic_x, periodic_y, periodic_z, ng,
    )
    h0m = _height_column(
        alpha, i, j, k, axis, 0, -1, normal_sign,
        nx, ny, nz, periodic_x, periodic_y, periodic_z, ng,
    )
    hpm = _height_column(
        alpha, i, j, k, axis, 1, -1, normal_sign,
        nx, ny, nz, periodic_x, periodic_y, periodic_z, ng,
    )
    hm0 = _height_column(
        alpha, i, j, k, axis, -1, 0, normal_sign,
        nx, ny, nz, periodic_x, periodic_y, periodic_z, ng,
    )
    h00 = _height_column(
        alpha, i, j, k, axis, 0, 0, normal_sign,
        nx, ny, nz, periodic_x, periodic_y, periodic_z, ng,
    )
    hp0 = _height_column(
        alpha, i, j, k, axis, 1, 0, normal_sign,
        nx, ny, nz, periodic_x, periodic_y, periodic_z, ng,
    )
    hmp = _height_column(
        alpha, i, j, k, axis, -1, 1, normal_sign,
        nx, ny, nz, periodic_x, periodic_y, periodic_z, ng,
    )
    h0p = _height_column(
        alpha, i, j, k, axis, 0, 1, normal_sign,
        nx, ny, nz, periodic_x, periodic_y, periodic_z, ng,
    )
    hpp = _height_column(
        alpha, i, j, k, axis, 1, 1, normal_sign,
        nx, ny, nz, periodic_x, periodic_y, periodic_z, ng,
    )
    if (hmm < wp.float32(0.0) or h0m < wp.float32(0.0)
            or hpm < wp.float32(0.0) or hm0 < wp.float32(0.0)
            or h00 < wp.float32(0.0) or hp0 < wp.float32(0.0)
            or hmp < wp.float32(0.0) or h0p < wp.float32(0.0)
            or hpp < wp.float32(0.0)):
        return

    spacing_axis = dx
    spacing_a = dy
    spacing_b = dz
    if axis == 1:
        spacing_axis = dy
        spacing_a = dx
        spacing_b = dz
    elif axis == 2:
        spacing_axis = dz
        spacing_a = dx
        spacing_b = dy
    hmm *= spacing_axis
    h0m *= spacing_axis
    hpm *= spacing_axis
    hm0 *= spacing_axis
    h00 *= spacing_axis
    hp0 *= spacing_axis
    hmp *= spacing_axis
    h0p *= spacing_axis
    hpp *= spacing_axis

    slope_a = (hp0 - hm0) / (wp.float32(2.0) * spacing_a)
    slope_b = (h0p - h0m) / (wp.float32(2.0) * spacing_b)
    second_a = (
        hp0 - wp.float32(2.0) * h00 + hm0
    ) / (spacing_a * spacing_a)
    second_b = (
        h0p - wp.float32(2.0) * h00 + h0m
    ) / (spacing_b * spacing_b)
    cross = (
        hpp - hpm - hmp + hmm
    ) / (wp.float32(4.0) * spacing_a * spacing_b)
    numerator = (
        (wp.float32(1.0) + slope_b * slope_b) * second_a
        - wp.float32(2.0) * slope_a * slope_b * cross
        + (wp.float32(1.0) + slope_a * slope_a) * second_b
    )
    denominator = wp.pow(
        wp.float32(1.0) + slope_a * slope_a + slope_b * slope_b,
        wp.float32(1.5),
    )
    value = -normal_sign * numerator / denominator
    h = wp.min(dx, wp.min(dy, dz))
    if (wp.isfinite(value)
            and wp.abs(value) * h
            <= wp.float32(MAX_RESOLVED_GRID_SCALED_CURVATURE)):
        curvature[I, J, K] = value
        valid[I, J, K] = 1
        wp.atomic_add(accepted, 0, 1)


@wp.kernel
def _fit_plic_curvature(
    alpha: wp.array3d(dtype=wp.float32),
    theta: wp.array3d(dtype=wp.float32),
    physical_normals: wp.array3d(dtype=wp.vec3),
    local_normals: wp.array3d(dtype=wp.vec3),
    offsets: wp.array3d(dtype=wp.float32),
    curvature: wp.array3d(dtype=wp.float32),
    valid: wp.array3d(dtype=wp.int32),
    mixed_cells: wp.array(dtype=wp.int32),
    area_sum: wp.array(dtype=wp.float64),
    x0: wp.float32,
    y0: wp.float32,
    z0: wp.float32,
    dx: wp.float32,
    dy: wp.float32,
    dz: wp.float32,
    nx: int,
    ny: int,
    nz: int,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    mixed_tolerance: wp.float32,
    ng: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    if _capillary_mixed_fraction(
        alpha[I, J, K], theta[I, J, K], mixed_tolerance,
    ) == 0:
        return
    wp.atomic_add(mixed_cells, 0, 1)

    spacing = wp.vec3(dx, dy, dz)
    normal = physical_normals[I, J, K]
    origin = wp.vec3(
        x0 + wp.float32(i) * dx,
        y0 + wp.float32(j) * dy,
        z0 + wp.float32(k) * dz,
    )
    point = _plane_point(normal, offsets[I, J, K], origin, spacing)
    patch_area = _plic_patch_area(
        normal, local_normals[I, J, K], offsets[I, J, K], spacing,
    )
    wp.atomic_add(area_sum, 0, wp.float64(patch_area))

    axx = wp.float32(0.0)
    axy = wp.float32(0.0)
    axz = wp.float32(0.0)
    ayy = wp.float32(0.0)
    ayz = wp.float32(0.0)
    azz = wp.float32(0.0)
    bxx = wp.float32(0.0)
    bxy = wp.float32(0.0)
    bxz = wp.float32(0.0)
    byx = wp.float32(0.0)
    byy = wp.float32(0.0)
    byz = wp.float32(0.0)
    bzx = wp.float32(0.0)
    bzy = wp.float32(0.0)
    bzz = wp.float32(0.0)
    neighbours = int(0)
    h = wp.min(dx, wp.min(dy, dz))
    inverse_h2 = wp.float32(1.0) / (h * h)
    maximum_normal_separation = wp.float32(1.5) * wp.max(
        dx, wp.max(dy, dz),
    )

    for ai in range(5):
        di = ai - 2
        qi = _wrapped_cell_index(i + di, nx, periodic_x)
        for aj in range(5):
            dj = aj - 2
            qj = _wrapped_cell_index(j + dj, ny, periodic_y)
            for ak in range(5):
                dk = ak - 2
                qk = _wrapped_cell_index(k + dk, nz, periodic_z)
                if qi < 0 or qj < 0 or qk < 0:
                    continue
                if di == 0 and dj == 0 and dk == 0:
                    continue
                QI = qi + ng
                QJ = qj + ng
                QK = qk + ng
                if _capillary_mixed_fraction(
                    alpha[QI, QJ, QK], theta[QI, QJ, QK],
                    mixed_tolerance,
                ) == 0:
                    continue
                neighbour_normal = physical_normals[QI, QJ, QK]
                alignment = wp.dot(normal, neighbour_normal)
                if alignment <= wp.float32(0.5):
                    continue
                neighbour_origin = wp.vec3(
                    x0 + wp.float32(i + di) * dx,
                    y0 + wp.float32(j + dj) * dy,
                    z0 + wp.float32(k + dk) * dz,
                )
                neighbour_point = _plane_point(
                    neighbour_normal, offsets[QI, QJ, QK],
                    neighbour_origin, spacing,
                )
                delta = neighbour_point - point
                if wp.abs(wp.dot(normal, delta)) > maximum_normal_separation:
                    continue
                normal_delta = neighbour_normal - normal
                radius2 = wp.dot(delta, delta) * inverse_h2
                weight = alignment * alignment / (
                    wp.float32(1.0) + radius2
                )
                x = delta[0]
                y = delta[1]
                z = delta[2]
                axx += weight * x * x
                axy += weight * x * y
                axz += weight * x * z
                ayy += weight * y * y
                ayz += weight * y * z
                azz += weight * z * z
                bxx += weight * normal_delta[0] * x
                bxy += weight * normal_delta[0] * y
                bxz += weight * normal_delta[0] * z
                byx += weight * normal_delta[1] * x
                byy += weight * normal_delta[1] * y
                byz += weight * normal_delta[1] * z
                bzx += weight * normal_delta[2] * x
                bzy += weight * normal_delta[2] * y
                bzz += weight * normal_delta[2] * z
                neighbours += 1

    if neighbours < 3:
        return

    # Interface points span only the two tangent directions.  Completing the
    # normal direction with an explicitly zero normal derivative makes the
    # least-squares system nonsingular without inventing an off-surface field.
    normal_scale = wp.max(
        wp.float32(0.5) * (axx + ayy + azz), h * h,
    )
    axx += normal_scale * normal[0] * normal[0]
    axy += normal_scale * normal[0] * normal[1]
    axz += normal_scale * normal[0] * normal[2]
    ayy += normal_scale * normal[1] * normal[1]
    ayz += normal_scale * normal[1] * normal[2]
    azz += normal_scale * normal[2] * normal[2]

    c00 = ayy * azz - ayz * ayz
    c01 = axz * ayz - axy * azz
    c02 = axy * ayz - axz * ayy
    c11 = axx * azz - axz * axz
    c12 = axy * axz - axx * ayz
    c22 = axx * ayy - axy * axy
    determinant = axx * c00 + axy * c01 + axz * c02
    scale = axx + ayy + azz
    if (not wp.isfinite(determinant)
            or wp.abs(determinant) <= wp.float32(1.0e-8) * scale * scale * scale):
        return
    inverse_determinant = wp.float32(1.0) / determinant
    c00 *= inverse_determinant
    c01 *= inverse_determinant
    c02 *= inverse_determinant
    c11 *= inverse_determinant
    c12 *= inverse_determinant
    c22 *= inverse_determinant
    fitted = (
        bxx * c00 + bxy * c01 + bxz * c02
        + byx * c01 + byy * c11 + byz * c12
        + bzx * c02 + bzy * c12 + bzz * c22
    )
    if (wp.isfinite(fitted)
            and wp.abs(fitted) * h
            <= wp.float32(MAX_RESOLVED_GRID_SCALED_CURVATURE)):
        curvature[I, J, K] = fitted
        valid[I, J, K] = 1


@wp.kernel
def _extend_plic_curvature(
    alpha: wp.array3d(dtype=wp.float32),
    theta: wp.array3d(dtype=wp.float32),
    normals: wp.array3d(dtype=wp.vec3),
    raw: wp.array3d(dtype=wp.float32),
    height_valid: wp.array3d(dtype=wp.int32),
    fit_valid: wp.array3d(dtype=wp.int32),
    curvature: wp.array3d(dtype=wp.float32),
    owner: wp.array3d(dtype=wp.int32),
    maximum: wp.array(dtype=wp.float32),
    unresolved: wp.array(dtype=wp.int32),
    nx: int,
    ny: int,
    nz: int,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    mixed_tolerance: wp.float32,
    ng: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    target_mixed = _capillary_mixed_fraction(
        alpha[I, J, K], theta[I, J, K], mixed_tolerance,
    )
    reference = wp.vec3(
        wp.float32(0.0), wp.float32(0.0), wp.float32(0.0),
    )
    found = int(0)
    closest = wp.float32(100.0)
    for ai in range(3):
        di = ai - 1
        qi = _wrapped_cell_index(i + di, nx, periodic_x)
        for aj in range(3):
            dj = aj - 1
            qj = _wrapped_cell_index(j + dj, ny, periodic_y)
            for ak in range(3):
                dk = ak - 1
                qk = _wrapped_cell_index(k + dk, nz, periodic_z)
                if qi < 0 or qj < 0 or qk < 0:
                    continue
                QI = qi + ng
                QJ = qj + ng
                QK = qk + ng
                if _capillary_mixed_fraction(
                    alpha[QI, QJ, QK], theta[QI, QJ, QK],
                    mixed_tolerance,
                ) == 0:
                    continue
                distance2 = wp.float32(di * di + dj * dj + dk * dk)
                if distance2 < closest:
                    closest = distance2
                    reference = normals[QI, QJ, QK]
                    found = 1
    if found == 0:
        return

    weighted = wp.float32(0.0)
    weight_sum = wp.float32(0.0)
    for ai in range(7):
        di = ai - 3
        qi = _wrapped_cell_index(i + di, nx, periodic_x)
        for aj in range(7):
            dj = aj - 3
            qj = _wrapped_cell_index(j + dj, ny, periodic_y)
            for ak in range(7):
                dk = ak - 3
                qk = _wrapped_cell_index(k + dk, nz, periodic_z)
                if qi < 0 or qj < 0 or qk < 0:
                    continue
                QI = qi + ng
                QJ = qj + ng
                QK = qk + ng
                if height_valid[QI, QJ, QK] == 0:
                    continue
                alignment = wp.dot(reference, normals[QI, QJ, QK])
                if alignment <= wp.float32(0.5):
                    continue
                distance2 = wp.float32(di * di + dj * dj + dk * dk)
                weight = alignment * alignment / (
                    wp.float32(1.0) + distance2
                )
                weighted += weight * raw[QI, QJ, QK]
                weight_sum += weight
    if weight_sum > wp.float32(0.0):
        value = weighted / weight_sum
        curvature[I, J, K] = value
        owner[I, J, K] = 1
        if target_mixed != 0:
            wp.atomic_max(maximum, 0, wp.abs(value))
        return

    for ai in range(3):
        di = ai - 1
        qi = _wrapped_cell_index(i + di, nx, periodic_x)
        for aj in range(3):
            dj = aj - 1
            qj = _wrapped_cell_index(j + dj, ny, periodic_y)
            for ak in range(3):
                dk = ak - 1
                qk = _wrapped_cell_index(k + dk, nz, periodic_z)
                if qi < 0 or qj < 0 or qk < 0:
                    continue
                QI = qi + ng
                QJ = qj + ng
                QK = qk + ng
                if (_capillary_mixed_fraction(
                        alpha[QI, QJ, QK], theta[QI, QJ, QK],
                        mixed_tolerance,
                    ) == 0 or fit_valid[QI, QJ, QK] == 0):
                    continue
                alignment = wp.dot(reference, normals[QI, QJ, QK])
                if alignment <= wp.float32(0.5):
                    continue
                distance2 = wp.float32(di * di + dj * dj + dk * dk)
                weight = alignment * alignment / (
                    wp.float32(1.0) + distance2
                )
                weighted += weight * raw[QI, QJ, QK]
                weight_sum += weight
    if weight_sum > wp.float32(0.0):
        value = weighted / weight_sum
        curvature[I, J, K] = value
        owner[I, J, K] = 2
        if target_mixed != 0:
            wp.atomic_max(maximum, 0, wp.abs(value))
    elif target_mixed != 0:
        wp.atomic_add(unresolved, 0, 1)


@wp.kernel
def _sum_plic_area(
    alpha: wp.array3d(dtype=wp.float32),
    theta: wp.array3d(dtype=wp.float32),
    physical_normals: wp.array3d(dtype=wp.vec3),
    local_normals: wp.array3d(dtype=wp.vec3),
    offsets: wp.array3d(dtype=wp.float32),
    area_sum: wp.array(dtype=wp.float64),
    spacing: wp.vec3,
    mixed_tolerance: wp.float32,
    ng: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    if _capillary_mixed_fraction(
        alpha[I, J, K], theta[I, J, K], mixed_tolerance,
    ) == 0:
        return
    area = _plic_patch_area(
        physical_normals[I, J, K], local_normals[I, J, K],
        offsets[I, J, K], spacing,
    )
    wp.atomic_add(area_sum, 0, wp.float64(area))


@wp.kernel
def _sum_pure_face_area(
    alpha: wp.array3d(dtype=wp.float32),
    theta: wp.array3d(dtype=wp.float32),
    aperture_u: wp.array3d(dtype=wp.float32),
    aperture_v: wp.array3d(dtype=wp.float32),
    aperture_w: wp.array3d(dtype=wp.float32),
    area_sum: wp.array(dtype=wp.float64),
    x_face_area: wp.float64,
    y_face_area: wp.float64,
    z_face_area: wp.float64,
    nx: int,
    ny: int,
    nz: int,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    mixed_tolerance: wp.float32,
    ng: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    phase = _capillary_pure_phase(
        alpha[I, J, K], theta[I, J, K], mixed_tolerance,
    )
    if phase < 0:
        return

    qi = _wrapped_cell_index(i + 1, nx, periodic_x)
    if qi >= 0:
        neighbour = _capillary_pure_phase(
            alpha[qi + ng, J, K], theta[qi + ng, J, K],
            mixed_tolerance,
        )
        if neighbour >= 0 and neighbour != phase:
            wp.atomic_add(
                area_sum,
                0,
                wp.float64(aperture_u[I + 1, J, K]) * x_face_area,
            )

    qj = _wrapped_cell_index(j + 1, ny, periodic_y)
    if qj >= 0:
        neighbour = _capillary_pure_phase(
            alpha[I, qj + ng, K], theta[I, qj + ng, K],
            mixed_tolerance,
        )
        if neighbour >= 0 and neighbour != phase:
            wp.atomic_add(
                area_sum,
                0,
                wp.float64(aperture_v[I, J + 1, K]) * y_face_area,
            )

    qk = _wrapped_cell_index(k + 1, nz, periodic_z)
    if qk >= 0:
        neighbour = _capillary_pure_phase(
            alpha[I, J, qk + ng], theta[I, J, qk + ng],
            mixed_tolerance,
        )
        if neighbour >= 0 and neighbour != phase:
            wp.atomic_add(
                area_sum,
                0,
                wp.float64(aperture_w[I, J, K + 1]) * z_face_area,
            )


def _add_pure_face_area(
    state: State3D,
    area_sum: wp.array,
    mixed_tolerance: float,
) -> None:
    g = state.cfg.grid
    bc = state.cfg.bc
    wp.launch(
        _sum_pure_face_area,
        dim=g.shape,
        inputs=[
            state.alpha,
            state.theta,
            state.u_aperture,
            state.v_aperture,
            state.w_aperture,
            area_sum,
            g.dy * g.dz,
            g.dx * g.dz,
            g.dx * g.dy,
            g.nx,
            g.ny,
            g.nz,
            int(bc.xlo == BC.PERIODIC),
            int(bc.ylo == BC.PERIODIC),
            int(bc.zlo == BC.PERIODIC),
            mixed_tolerance,
            g.ng,
        ],
        device=state.device,
    )


def reconstruct_surface_tension_geometry(
    state: State3D,
    physical_normals: wp.array,
    plic: PLICReconstruction,
    *,
    mixed_tolerance: float = CAPILLARY_MIXED_TOLERANCE,
) -> SurfaceTensionGeometry:
    """Fit signed curvature to the authoritative PLIC interface."""
    g = state.cfg.grid
    raw = wp.zeros(state.grid.cell_shape, dtype=wp.float32, device=state.device)
    curvature = wp.zeros(
        state.grid.cell_shape, dtype=wp.float32, device=state.device,
    )
    fit_valid = wp.zeros(
        state.grid.cell_shape, dtype=wp.int32, device=state.device,
    )
    mixed_cells = wp.zeros(1, dtype=wp.int32, device=state.device)
    area_sum = wp.zeros(1, dtype=wp.float64, device=state.device)
    bc = state.cfg.bc
    periodic = (
        int(bc.xlo == BC.PERIODIC),
        int(bc.ylo == BC.PERIODIC),
        int(bc.zlo == BC.PERIODIC),
    )
    wp.launch(
        _fit_plic_curvature,
        dim=g.shape,
        inputs=[
            state.alpha, state.theta, physical_normals,
            plic.normals, plic.offsets,
            raw, fit_valid, mixed_cells, area_sum,
            g.x0, g.y0, g.z0, g.dx, g.dy, g.dz,
            g.nx, g.ny, g.nz, *periodic, mixed_tolerance, g.ng,
        ],
        device=state.device,
    )
    _add_pure_face_area(state, area_sum, mixed_tolerance)
    height_function_cells = wp.zeros(
        1, dtype=wp.int32, device=state.device,
    )
    height_valid = wp.zeros(
        state.grid.cell_shape, dtype=wp.int32, device=state.device,
    )
    wp.launch(
        _apply_height_function_curvature,
        dim=g.shape,
        inputs=[
            state.alpha, state.theta, physical_normals, raw, height_valid,
            height_function_cells,
            g.dx, g.dy, g.dz, g.nx, g.ny, g.nz, *periodic,
            mixed_tolerance, g.ng,
        ],
        device=state.device,
    )
    maximum = wp.zeros(1, dtype=wp.float32, device=state.device)
    unresolved = wp.zeros(1, dtype=wp.int32, device=state.device)
    owner = wp.zeros(
        state.grid.cell_shape, dtype=wp.int32, device=state.device,
    )
    wp.launch(
        _extend_plic_curvature,
        dim=g.shape,
        inputs=[
            state.alpha, state.theta, physical_normals, raw, height_valid,
            fit_valid, curvature, owner, maximum, unresolved,
            g.nx, g.ny, g.nz, *periodic, mixed_tolerance, g.ng,
        ],
        device=state.device,
    )
    return SurfaceTensionGeometry(
        curvature=curvature,
        height_function_valid=height_valid,
        plic_fit_valid=fit_valid,
        curvature_owner=owner,
        surface_area=float(area_sum.numpy()[0]),
        max_abs_curvature=float(maximum.numpy()[0]),
        mixed_cells=int(mixed_cells.numpy()[0]),
        height_function_cells=int(height_function_cells.numpy()[0]),
        unresolved_cells=int(unresolved.numpy()[0]),
    )


def measure_plic_surface_area(
    state: State3D,
    *,
    mixed_tolerance: float = CAPILLARY_MIXED_TOLERANCE,
) -> float:
    """Return area of mixed-cell PLIC patches and exact pure-phase faces."""
    interface = estimate_interface_normals(
        state, mixed_tolerance=mixed_tolerance,
    )
    plic = reconstruct_plic(
        state, interface.normals, mixed_tolerance=mixed_tolerance,
    )
    area_sum = wp.zeros(1, dtype=wp.float64, device=state.device)
    g = state.cfg.grid
    wp.launch(
        _sum_plic_area,
        dim=g.shape,
        inputs=[
            state.alpha, state.theta, interface.normals,
            plic.normals, plic.offsets,
            area_sum, wp.vec3(g.dx, g.dy, g.dz), mixed_tolerance, g.ng,
        ],
        device=state.device,
    )
    _add_pure_face_area(state, area_sum, mixed_tolerance)
    return float(area_sum.numpy()[0])


def gravity_capillary_angular_frequency(
    wavenumber: float,
    depth: float,
    *,
    gravity: float = 9.81,
    density: float = 1000.0,
    surface_tension: float = 0.072,
) -> float:
    """Return ``sqrt((g k + sigma k^3 / rho) tanh(k h))``."""
    values = (wavenumber, depth, density)
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("wavenumber, depth, and density must be positive and finite")
    if not math.isfinite(gravity) or gravity < 0.0:
        raise ValueError("gravity must be non-negative and finite")
    if not math.isfinite(surface_tension) or surface_tension < 0.0:
        raise ValueError("surface_tension must be non-negative and finite")
    return math.sqrt(
        (gravity * wavenumber
         + surface_tension * wavenumber**3 / density)
        * math.tanh(wavenumber * depth),
    )
