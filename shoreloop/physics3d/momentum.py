"""Conservative staggered momentum transport coupled to geometric VOF fluxes."""

from __future__ import annotations

from dataclasses import dataclass
import math

import warp as wp

from shoreloop.config import BC, Solver3DConfig
from shoreloop.physics3d.advection import (VOFAdvectionResult, VOFAdvector,
                                           _copy_cell_field)
from shoreloop.physics3d.boundary import fill_field_boundaries
from shoreloop.physics3d.extrapolation import (VelocityExtension,
                                               _face_neighbor_index,
                                               _wrapped_cell_index,
                                               extrapolate_velocity)
from shoreloop.physics3d.solids import _solid_cell
from shoreloop.physics3d.state import State3D


@dataclass(frozen=True)
class MomentumAdvectionResult:
    """Coupled volume/momentum conservation and interface-speed diagnostics."""

    vof: VOFAdvectionResult
    momentum_before: tuple[float, float, float]
    momentum_after: tuple[float, float, float]
    momentum_drift: tuple[float, float, float]
    max_water_speed: float
    discarded_momentum: float


@wp.func
def _cell_fraction(
    alpha: wp.array3d(dtype=wp.float32),
    i: int,
    j: int,
    k: int,
    nx: int,
    ny: int,
    nz: int,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    ng: int,
) -> wp.float32:
    ci = _wrapped_cell_index(i, nx, periodic_x)
    cj = _wrapped_cell_index(j, ny, periodic_y)
    ck = _wrapped_cell_index(k, nz, periodic_z)
    if ci < 0 or cj < 0 or ck < 0:
        return wp.float32(0.0)
    return alpha[ci + ng, cj + ng, ck + ng]


@wp.kernel
def _compute_face_mass(
    alpha: wp.array3d(dtype=wp.float32),
    theta: wp.array3d(dtype=wp.float32),
    solid: wp.array3d(dtype=wp.int32),
    aperture: wp.array3d(dtype=wp.float32),
    mass: wp.array3d(dtype=wp.float32),
    flux_consistent: int,
    component_axis: int,
    nx: int,
    ny: int,
    nz: int,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    ng: int,
):
    i, j, k = wp.tid()
    lower_i = i
    lower_j = j
    lower_k = k
    if component_axis == 0:
        lower_i -= 1
    elif component_axis == 1:
        lower_j -= 1
    else:
        lower_k -= 1
    if (_solid_cell(solid, lower_i, lower_j, lower_k, nx, ny, nz,
                    periodic_x, periodic_y, periodic_z, ng) != 0
            or _solid_cell(solid, i, j, k, nx, ny, nz,
                           periodic_x, periodic_y, periodic_z, ng) != 0):
        mass[i + ng, j + ng, k + ng] = wp.float32(0.0)
        return
    lower = _cell_fraction(alpha, lower_i, lower_j, lower_k,
                           nx, ny, nz, periodic_x, periodic_y, periodic_z, ng)
    upper = _cell_fraction(alpha, i, j, k, nx, ny, nz,
                           periodic_x, periodic_y, periodic_z, ng)
    open_area = aperture[i + ng, j + ng, k + ng]
    if flux_consistent != 0:
        # The dual fluxes average the primal fluxes, which move theta*alpha,
        # so the dual volume they change is half of each cell's water. A face
        # closed by the solid carries no velocity.
        if open_area > wp.float32(0.0):
            lower *= _cell_fraction(theta, lower_i, lower_j, lower_k,
                                    nx, ny, nz, periodic_x, periodic_y, periodic_z, ng)
            upper *= _cell_fraction(theta, i, j, k, nx, ny, nz,
                                    periodic_x, periodic_y, periodic_z, ng)
            mass[i + ng, j + ng, k + ng] = wp.float32(0.5) * (lower + upper)
        else:
            mass[i + ng, j + ng, k + ng] = wp.float32(0.0)
        return
    mass[i + ng, j + ng, k + ng] = open_area * wp.float32(0.5) * (lower + upper)


@wp.kernel
def _form_momentum(
    mass: wp.array3d(dtype=wp.float32),
    velocity: wp.array3d(dtype=wp.float32),
    momentum: wp.array3d(dtype=wp.float32),
    ng: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    momentum[I, J, K] = mass[I, J, K] * velocity[I, J, K]


@wp.func
def _sample_vof_flux(
    fluxes: wp.array3d(dtype=wp.float32),
    i: int,
    j: int,
    k: int,
    transport_axis: int,
    nx: int,
    ny: int,
    nz: int,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    ng: int,
) -> wp.float32:
    sx = nx
    sy = ny
    sz = nz
    if transport_axis == 0:
        sx += 1
    elif transport_axis == 1:
        sy += 1
    else:
        sz += 1
    fi = _face_neighbor_index(i, sx, periodic_x, int(transport_axis == 0))
    fj = _face_neighbor_index(j, sy, periodic_y, int(transport_axis == 1))
    fk = _face_neighbor_index(k, sz, periodic_z, int(transport_axis == 2))
    if fi < 0 or fj < 0 or fk < 0:
        return wp.float32(0.0)
    return fluxes[fi + ng, fj + ng, fk + ng]


@wp.func
def _dual_liquid_flux(
    fluxes: wp.array3d(dtype=wp.float32),
    i: int,
    j: int,
    k: int,
    high: int,
    component_axis: int,
    transport_axis: int,
    nx: int,
    ny: int,
    nz: int,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    ng: int,
) -> wp.float32:
    first_i = i
    first_j = j
    first_k = k
    second_i = i
    second_j = j
    second_k = k
    if component_axis == transport_axis:
        if transport_axis == 0:
            first_i = i + high - 1
            second_i = i + high
        elif transport_axis == 1:
            first_j = j + high - 1
            second_j = j + high
        else:
            first_k = k + high - 1
            second_k = k + high
    else:
        if transport_axis == 0:
            first_i = i + high
            second_i = first_i
        elif transport_axis == 1:
            first_j = j + high
            second_j = first_j
        else:
            first_k = k + high
            second_k = first_k
        if component_axis == 0:
            first_i -= 1
        elif component_axis == 1:
            first_j -= 1
        else:
            first_k -= 1
    first = _sample_vof_flux(
        fluxes, first_i, first_j, first_k, transport_axis,
        nx, ny, nz, periodic_x, periodic_y, periodic_z, ng,
    )
    second = _sample_vof_flux(
        fluxes, second_i, second_j, second_k, transport_axis,
        nx, ny, nz, periodic_x, periodic_y, periodic_z, ng,
    )
    return wp.float32(0.5) * (first + second)


@wp.func
def _component_sample_index(
    index: int,
    axis: int,
    component_axis: int,
    nx: int,
    ny: int,
    nz: int,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
) -> int:
    count = nx
    periodic = periodic_x
    if axis == 1:
        count = ny
        periodic = periodic_y
    elif axis == 2:
        count = nz
        periodic = periodic_z
    duplicated = int(axis == component_axis)
    if duplicated != 0:
        count += 1
    return _face_neighbor_index(index, count, periodic, duplicated)


@wp.func
def _sample_component_value(
    values: wp.array3d(dtype=wp.float32),
    i: int,
    j: int,
    k: int,
    component_axis: int,
    nx: int,
    ny: int,
    nz: int,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    ng: int,
) -> wp.float32:
    fi = _component_sample_index(i, 0, component_axis, nx, ny, nz,
                                 periodic_x, periodic_y, periodic_z)
    fj = _component_sample_index(j, 1, component_axis, nx, ny, nz,
                                 periodic_x, periodic_y, periodic_z)
    fk = _component_sample_index(k, 2, component_axis, nx, ny, nz,
                                 periodic_x, periodic_y, periodic_z)
    if fi < 0 or fj < 0 or fk < 0:
        return wp.float32(0.0)
    return values[fi + ng, fj + ng, fk + ng]


@wp.func
def _sample_component_valid(
    valid: wp.array3d(dtype=wp.int32),
    i: int,
    j: int,
    k: int,
    component_axis: int,
    nx: int,
    ny: int,
    nz: int,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    ng: int,
) -> int:
    fi = _component_sample_index(i, 0, component_axis, nx, ny, nz,
                                 periodic_x, periodic_y, periodic_z)
    fj = _component_sample_index(j, 1, component_axis, nx, ny, nz,
                                 periodic_x, periodic_y, periodic_z)
    fk = _component_sample_index(k, 2, component_axis, nx, ny, nz,
                                 periodic_x, periodic_y, periodic_z)
    if fi < 0 or fj < 0 or fk < 0:
        return 0
    return valid[fi + ng, fj + ng, fk + ng]


@wp.func
def _minmod(left: wp.float32, right: wp.float32) -> wp.float32:
    result = wp.float32(0.0)
    if left * right > wp.float32(0.0):
        magnitude = wp.min(wp.abs(left), wp.abs(right))
        result = wp.sign(left) * magnitude
    return result


@wp.func
def _donor_face(
    i: int,
    j: int,
    k: int,
    high: int,
    liquid_flux: wp.float32,
    transport_axis: int,
) -> wp.vec3i:
    """Face whose dual volume donates ``liquid_flux`` through boundary ``high``."""
    boundary = high
    if liquid_flux > wp.float32(0.0):
        boundary -= 1
    if transport_axis == 0:
        return wp.vec3i(i + boundary, j, k)
    elif transport_axis == 1:
        return wp.vec3i(i, j + boundary, k)
    return wp.vec3i(i, j, k + boundary)


@wp.kernel
def _face_transport_budget(
    transport_values: wp.array3d(dtype=wp.float32),
    flux_x: wp.array3d(dtype=wp.float32),
    flux_y: wp.array3d(dtype=wp.float32),
    flux_z: wp.array3d(dtype=wp.float32),
    outgoing: wp.array3d(dtype=wp.float32),
    inflow_velocity: wp.array3d(dtype=wp.float32),
    component_axis: int,
    nx: int,
    ny: int,
    nz: int,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    ng: int,
    flux_threshold: wp.float32,
):
    """Liquid leaving each dual volume and the mean velocity of liquid entering it.

    Both fluxes are dimensionless liquid volumes per cell volume, like the
    face mass; the entering velocity is first-order upwind.
    """
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    total_out = wp.float32(0.0)
    total_in = wp.float32(0.0)
    momentum_in = wp.float32(0.0)
    for transport_axis in range(3):
        for high in range(2):
            flux = wp.float32(0.0)
            if transport_axis == 0:
                flux = _dual_liquid_flux(
                    flux_x, i, j, k, high, component_axis, transport_axis,
                    nx, ny, nz, periodic_x, periodic_y, periodic_z, ng,
                )
            elif transport_axis == 1:
                flux = _dual_liquid_flux(
                    flux_y, i, j, k, high, component_axis, transport_axis,
                    nx, ny, nz, periodic_x, periodic_y, periodic_z, ng,
                )
            else:
                flux = _dual_liquid_flux(
                    flux_z, i, j, k, high, component_axis, transport_axis,
                    nx, ny, nz, periodic_x, periodic_y, periodic_z, ng,
                )
            if wp.abs(flux) > flux_threshold:
                # Positive flux points along +axis: outward through the high boundary.
                outward = flux
                if high == 0:
                    outward = -flux
                if outward > wp.float32(0.0):
                    total_out += outward
                else:
                    donor = _donor_face(i, j, k, high, flux, transport_axis)
                    total_in -= outward
                    momentum_in -= outward * _sample_component_value(
                        transport_values, donor[0], donor[1], donor[2], component_axis,
                        nx, ny, nz, periodic_x, periodic_y, periodic_z, ng,
                    )
    outgoing[I, J, K] = total_out
    entering = transport_values[I, J, K]
    if total_in > wp.float32(0.0):
        entering = momentum_in / total_in
    inflow_velocity[I, J, K] = entering


@wp.func
def _transported_velocity(
    values: wp.array3d(dtype=wp.float32),
    valid: wp.array3d(dtype=wp.int32),
    mass: wp.array3d(dtype=wp.float32),
    outgoing: wp.array3d(dtype=wp.float32),
    inflow_velocity: wp.array3d(dtype=wp.float32),
    i: int,
    j: int,
    k: int,
    high: int,
    liquid_flux: wp.float32,
    component_axis: int,
    transport_axis: int,
    nx: int,
    ny: int,
    nz: int,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    ng: int,
    flux_threshold: wp.float32,
    invalid: wp.array(dtype=wp.int32),
) -> wp.float32:
    # The liquid path transports every retained nonzero volume flux. Dropping
    # a small flux here while VOF moves its water breaks uniform translation
    # and increases recovered speeds as donor mass decreases.
    if wp.abs(liquid_flux) <= flux_threshold:
        return wp.float32(0.0)
    donor = _donor_face(i, j, k, high, liquid_flux, transport_axis)
    donor_i = donor[0]
    donor_j = donor[1]
    donor_k = donor[2]
    if _sample_component_valid(
        valid, donor_i, donor_j, donor_k, component_axis,
        nx, ny, nz, periodic_x, periodic_y, periodic_z, ng,
    ) == 0:
        wp.atomic_max(invalid, 0, 1)
        return wp.float32(0.0)

    previous_i = donor_i
    previous_j = donor_j
    previous_k = donor_k
    next_i = donor_i
    next_j = donor_j
    next_k = donor_k
    if transport_axis == 0:
        previous_i -= 1
        next_i += 1
    elif transport_axis == 1:
        previous_j -= 1
        next_j += 1
    else:
        previous_k -= 1
        next_k += 1
    donor_value = _sample_component_value(
        values, donor_i, donor_j, donor_k, component_axis,
        nx, ny, nz, periodic_x, periodic_y, periodic_z, ng,
    )
    # Slopes use wet neighbours only. Extended air values are extrapolated and
    # can exceed every water speed, so they cannot bound a water velocity.
    previous = donor_value
    if (_sample_component_valid(
            valid, previous_i, previous_j, previous_k, component_axis,
            nx, ny, nz, periodic_x, periodic_y, periodic_z, ng) != 0
            and _sample_component_value(
                mass, previous_i, previous_j, previous_k, component_axis,
                nx, ny, nz, periodic_x, periodic_y, periodic_z, ng) > wp.float32(0.0)):
        previous = _sample_component_value(
            values, previous_i, previous_j, previous_k, component_axis,
            nx, ny, nz, periodic_x, periodic_y, periodic_z, ng,
        )
    following = donor_value
    if (_sample_component_valid(
            valid, next_i, next_j, next_k, component_axis,
            nx, ny, nz, periodic_x, periodic_y, periodic_z, ng) != 0
            and _sample_component_value(
                mass, next_i, next_j, next_k, component_axis,
                nx, ny, nz, periodic_x, periodic_y, periodic_z, ng) > wp.float32(0.0)):
        following = _sample_component_value(
            values, next_i, next_j, next_k, component_axis,
            nx, ny, nz, periodic_x, periodic_y, periodic_z, ng,
        )
    slope = _minmod(donor_value - previous, following - donor_value)
    transported = donor_value + wp.float32(0.5) * slope
    downwind = following
    if liquid_flux < wp.float32(0.0):
        transported = donor_value - wp.float32(0.5) * slope
        downwind = previous
    transported = wp.min(wp.max(transported, wp.min(donor_value, downwind)),
                         wp.max(donor_value, downwind))
    # The donor retains mass - outgoing of its liquid with velocity
    # donor_value + outgoing * (donor_value - transported) / (mass - outgoing).
    # Blending toward donor_value once more than half the liquid leaves keeps
    # that retained velocity inside the reconstruction range. Outflow beyond
    # the donor's own liquid is liquid that entered during this unsplit step
    # through another boundary and carries the mean entering velocity.
    donor_mass = _sample_component_value(
        mass, donor_i, donor_j, donor_k, component_axis,
        nx, ny, nz, periodic_x, periodic_y, periodic_z, ng,
    )
    donor_outgoing = _sample_component_value(
        outgoing, donor_i, donor_j, donor_k, component_axis,
        nx, ny, nz, periodic_x, periodic_y, periodic_z, ng,
    )
    if donor_outgoing > donor_mass:
        entering = _sample_component_value(
            inflow_velocity, donor_i, donor_j, donor_k, component_axis,
            nx, ny, nz, periodic_x, periodic_y, periodic_z, ng,
        )
        return (donor_mass * donor_value
                + (donor_outgoing - donor_mass) * entering) / donor_outgoing
    blend = wp.float32(1.0)
    if donor_outgoing > wp.float32(0.0):
        blend = wp.min(wp.float32(1.0),
                       (donor_mass - donor_outgoing) / donor_outgoing)
    return donor_value + blend * (transported - donor_value)


@wp.kernel
def _transport_momentum(
    momentum: wp.array3d(dtype=wp.float32),
    transport_values: wp.array3d(dtype=wp.float32),
    transport_valid: wp.array3d(dtype=wp.int32),
    transport_mass: wp.array3d(dtype=wp.float32),
    transport_outgoing: wp.array3d(dtype=wp.float32),
    transport_inflow_velocity: wp.array3d(dtype=wp.float32),
    liquid_fluxes: wp.array3d(dtype=wp.float32),
    updated: wp.array3d(dtype=wp.float32),
    bound_lower: wp.array3d(dtype=wp.float32),
    bound_upper: wp.array3d(dtype=wp.float32),
    invalid: wp.array(dtype=wp.int32),
    component_axis: int,
    transport_axis: int,
    nx: int,
    ny: int,
    nz: int,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    ng: int,
    flux_threshold: wp.float32,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    low_flux = _dual_liquid_flux(
        liquid_fluxes, i, j, k, 0, component_axis, transport_axis,
        nx, ny, nz, periodic_x, periodic_y, periodic_z, ng,
    )
    high_flux = _dual_liquid_flux(
        liquid_fluxes, i, j, k, 1, component_axis, transport_axis,
        nx, ny, nz, periodic_x, periodic_y, periodic_z, ng,
    )
    low_velocity = _transported_velocity(
        transport_values, transport_valid, transport_mass, transport_outgoing,
        transport_inflow_velocity, i, j, k, 0, low_flux,
        component_axis, transport_axis, nx, ny, nz,
        periodic_x, periodic_y, periodic_z, ng, flux_threshold, invalid,
    )
    high_velocity = _transported_velocity(
        transport_values, transport_valid, transport_mass, transport_outgoing,
        transport_inflow_velocity, i, j, k, 1, high_flux,
        component_axis, transport_axis, nx, ny, nz,
        periodic_x, periodic_y, periodic_z, ng, flux_threshold, invalid,
    )
    value = momentum[I, J, K] - (
        high_flux * high_velocity - low_flux * low_velocity
    )
    if not wp.isfinite(value):
        wp.atomic_max(invalid, 0, 1)
        return
    updated[I, J, K] = value
    # With consistent masses the recovered velocity is a convex combination of
    # the retained part (inside the face's wet axial neighbours by the minmod
    # reflection) and the transported values it received. Record that range.
    low_moves = wp.abs(low_flux) > flux_threshold
    high_moves = wp.abs(high_flux) > flux_threshold
    if low_moves or high_moves:
        lower = bound_lower[I, J, K]
        upper = bound_upper[I, J, K]
        if low_moves:
            lower = wp.min(lower, low_velocity)
            upper = wp.max(upper, low_velocity)
        if high_moves:
            lower = wp.min(lower, high_velocity)
            upper = wp.max(upper, high_velocity)
        for side in range(2):
            ni = i
            nj = j
            nk = k
            step = 2 * side - 1
            if transport_axis == 0:
                ni += step
            elif transport_axis == 1:
                nj += step
            else:
                nk += step
            if (_sample_component_valid(
                    transport_valid, ni, nj, nk, component_axis,
                    nx, ny, nz, periodic_x, periodic_y, periodic_z, ng) != 0
                    and _sample_component_value(
                        transport_mass, ni, nj, nk, component_axis,
                        nx, ny, nz, periodic_x, periodic_y, periodic_z, ng) > wp.float32(0.0)):
                neighbour = _sample_component_value(
                    transport_values, ni, nj, nk, component_axis,
                    nx, ny, nz, periodic_x, periodic_y, periodic_z, ng,
                )
                lower = wp.min(lower, neighbour)
                upper = wp.max(upper, neighbour)
        bound_lower[I, J, K] = lower
        bound_upper[I, J, K] = upper


@wp.kernel
def _finalise_velocity(
    mass: wp.array3d(dtype=wp.float32),
    momentum: wp.array3d(dtype=wp.float32),
    original_velocity: wp.array3d(dtype=wp.float32),
    velocity: wp.array3d(dtype=wp.float32),
    discarded: wp.array(dtype=wp.float64),
    maximum: wp.array(dtype=wp.float32),
    mass_threshold: wp.float32,
    velocity_limit: wp.float32,
    bound_lower: wp.array3d(dtype=wp.float32),
    bound_upper: wp.array3d(dtype=wp.float32),
    local_bound: int,
    ng: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    face_mass = mass[I, J, K]
    if face_mass > mass_threshold:
        value = momentum[I, J, K] / face_mass
        if not wp.isfinite(value):
            wp.atomic_add(discarded, 0, wp.float64(wp.abs(momentum[I, J, K])))
            value = original_velocity[I, J, K]
        bounded = value
        if local_bound != 0:
            # Zero is inside the range, so the clamp can only reduce |u|.
            bounded = wp.clamp(
                value,
                wp.min(bound_lower[I, J, K], wp.float32(0.0)),
                wp.max(bound_upper[I, J, K], wp.float32(0.0)),
            )
            if bounded != value:
                wp.atomic_add(
                    discarded, 0,
                    wp.float64(wp.abs(momentum[I, J, K] - bounded * face_mass)),
                )
        else:
            roundoff_limit = velocity_limit * wp.float32(1.0001) + wp.float32(1.0e-7)
            if wp.abs(value) > roundoff_limit:
                bounded = wp.clamp(value, -velocity_limit, velocity_limit)
                wp.atomic_add(
                    discarded, 0,
                    wp.float64(wp.abs(momentum[I, J, K] - bounded * face_mass)),
                )
        value = bounded
        velocity[I, J, K] = value
        wp.atomic_max(maximum, 0, wp.abs(value))
    elif face_mass > wp.float32(0.0):
        velocity[I, J, K] = wp.float32(0.0)
        wp.atomic_add(discarded, 0, wp.float64(wp.abs(momentum[I, J, K])))
    else:
        velocity[I, J, K] = wp.float32(0.0)
        wp.atomic_add(discarded, 0, wp.float64(wp.abs(momentum[I, J, K])))


@wp.kernel
def _assign_non_owner_velocity(
    mass: wp.array3d(dtype=wp.float32),
    owner: wp.array3d(dtype=wp.int32),
    band: wp.array3d(dtype=wp.int32),
    extension: wp.array3d(dtype=wp.float32),
    velocity: wp.array3d(dtype=wp.float32),
    discarded: wp.array(dtype=wp.float64),
    maximum: wp.array(dtype=wp.float32),
    ng: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    face_mass = mass[I, J, K]
    if face_mass <= wp.float32(0.0):
        return
    if owner[I, J, K] == 0 and band[I, J, K] != 0:
        value = extension[I, J, K]
        wp.atomic_add(discarded, 0,
                      wp.float64(wp.abs(face_mass * (velocity[I, J, K] - value))))
        velocity[I, J, K] = value
    wp.atomic_max(maximum, 0, wp.abs(velocity[I, J, K]))


@wp.kernel
def _zero_periodic_duplicate(
    momentum: wp.array3d(dtype=wp.float32),
    component_axis: int,
    count: int,
    ng: int,
):
    first, second = wp.tid()
    if component_axis == 0:
        momentum[count + ng, first + ng, second + ng] = wp.float32(0.0)
    elif component_axis == 1:
        momentum[first + ng, count + ng, second + ng] = wp.float32(0.0)
    else:
        momentum[first + ng, second + ng, count + ng] = wp.float32(0.0)


@wp.kernel
def _reduce_water_speed(
    velocity: wp.array3d(dtype=wp.float32),
    mass: wp.array3d(dtype=wp.float32),
    maximum: wp.array(dtype=wp.float32),
    mass_threshold: wp.float32,
    ng: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    if mass[I, J, K] > mass_threshold:
        wp.atomic_max(maximum, 0, wp.abs(velocity[I, J, K]))


def compute_face_liquid_mass(
    state: State3D,
    name: str,
    *,
    out: wp.array | None = None,
    flux_consistent: bool = False,
) -> wp.array:
    """Return the liquid fraction of each staggered dual control volume.

    The default weights the mean cell fraction by the face aperture. The
    flux-consistent mass is half of each adjacent cell's water, theta*alpha,
    which is the volume the averaged primal fluxes actually move.
    """
    if name not in ("u", "v", "w"):
        raise ValueError("name must be 'u', 'v', or 'w'")
    axis = ("u", "v", "w").index(name)
    expected = state.grid.padded_shape(name)
    if out is None:
        mass = wp.zeros(expected, dtype=wp.float32, device=state.device)
    else:
        if tuple(out.shape) != expected or out.dtype != wp.float32:
            raise ValueError(
                f"{name} mass output must be a float32 field of shape {expected}",
            )
        mass = out
        mass.zero_()
    g = state.cfg.grid
    bc = state.cfg.bc
    periodic = (
        int(bc.xlo == BC.PERIODIC), int(bc.ylo == BC.PERIODIC),
        int(bc.zlo == BC.PERIODIC),
    )
    wp.launch(
        _compute_face_mass,
        dim=state.grid.interior_shape(name),
        inputs=[state.alpha, state.theta, state.solid,
                getattr(state, f"{name}_aperture"), mass,
                int(flux_consistent), axis, g.nx, g.ny, g.nz,
                *periodic, g.ng],
        device=state.device,
    )
    return mass


def measure_liquid_momentum(state: State3D) -> tuple[float, float, float]:
    """Measure staggered liquid momentum with the advection control volumes."""
    g = state.cfg.grid
    bc = state.cfg.bc
    periodic = (bc.xlo == BC.PERIODIC, bc.ylo == BC.PERIODIC,
                bc.zlo == BC.PERIODIC)
    scale = state.cfg.physics.rho * g.dx * g.dy * g.dz
    reduced = wp.zeros(3, dtype=wp.float32, device=state.device)
    for axis, name in enumerate(("u", "v", "w")):
        mass = compute_face_liquid_mass(state, name)
        momentum = wp.zeros(state.grid.padded_shape(name), dtype=wp.float32,
                            device=state.device)
        wp.launch(
            _form_momentum, dim=state.grid.interior_shape(name),
            inputs=[mass, state.field(name), momentum, g.ng],
            device=state.device,
        )
        if periodic[axis]:
            plane = tuple(
                count for plane_axis, count in enumerate(g.shape)
                if plane_axis != axis
            )
            wp.launch(
                _zero_periodic_duplicate,
                dim=plane,
                inputs=[momentum, axis, g.shape[axis], g.ng],
                device=state.device,
            )
        wp.utils.array_sum(
            momentum,
            out=reduced[axis:axis + 1],
        )
    totals = reduced.numpy().astype(float) * scale
    return (float(totals[0]), float(totals[1]), float(totals[2]))


class MomentumAdvector:
    """Advance VOF and all three conservative staggered momenta together."""

    __slots__ = (
        "cfg", "device", "vof", "extension_layers", "mass_threshold",
        "speed_reference_threshold", "_momentum_totals",
    )

    def __init__(
        self,
        cfg: Solver3DConfig,
        *,
        extension_layers: int = 4,
        mass_threshold: float = 1.0e-6,
        speed_reference_threshold: float = 0.1,
    ):
        if not isinstance(cfg, Solver3DConfig):
            raise TypeError("cfg must be a Solver3DConfig")
        if (isinstance(extension_layers, bool)
                or not isinstance(extension_layers, int)
                or not 1 <= extension_layers <= 8):
            raise ValueError("extension_layers must be an integer in [1, 8]")
        if (not math.isfinite(mass_threshold) or mass_threshold <= 0.0
                or mass_threshold >= 0.5):
            raise ValueError("mass_threshold must be finite in (0, 0.5)")
        if (not math.isfinite(speed_reference_threshold)
                or speed_reference_threshold <= mass_threshold
                or speed_reference_threshold >= 0.5):
            raise ValueError(
                "speed_reference_threshold must lie between mass_threshold and 0.5",
            )
        self.cfg = cfg
        self.device = cfg.device
        self.vof = VOFAdvector(cfg)
        self.extension_layers = extension_layers
        self.mass_threshold = mass_threshold
        self.speed_reference_threshold = speed_reference_threshold
        self._momentum_totals = wp.zeros(
            3, dtype=wp.float32, device=self.device,
        )

    def _copy_field(self, source: wp.array, shape: tuple[int, int, int]) -> wp.array:
        target = wp.zeros(
            tuple(n + 2 * self.cfg.grid.ng for n in shape),
            dtype=wp.float32, device=self.device,
        )
        wp.launch(
            _copy_cell_field,
            dim=shape,
            inputs=[source, target, self.cfg.grid.ng],
            device=self.device,
        )
        return target

    def _momentum_total(self, fields: tuple[wp.array, wp.array, wp.array]
                        ) -> tuple[float, float, float]:
        g = self.cfg.grid
        bc = self.cfg.bc
        periodic = (bc.xlo == BC.PERIODIC, bc.ylo == BC.PERIODIC,
                    bc.zlo == BC.PERIODIC)
        scale = self.cfg.physics.rho * g.dx * g.dy * g.dz
        for axis, field in enumerate(fields):
            if periodic[axis]:
                plane = tuple(
                    count for plane_axis, count in enumerate(g.shape)
                    if plane_axis != axis
                )
                wp.launch(
                    _zero_periodic_duplicate,
                    dim=plane,
                    inputs=[field, axis, g.shape[axis], g.ng],
                    device=self.device,
                )
            wp.utils.array_sum(
                field,
                out=self._momentum_totals[axis:axis + 1],
            )
        reduced = self._momentum_totals.numpy().astype(float) * scale
        return (float(reduced[0]), float(reduced[1]), float(reduced[2]))

    def transport_time_step(self, state: State3D, cfl: float) -> float:
        """Multi-axis CFL limit of the VOF transport velocities on faces that hold liquid.

        Axis slabs need only positive-mass face speeds. Full-vector prisms can
        receive water from transverse donors, so their bound includes those
        faces and the transverse interpolated velocities as well. Air-band
        extrapolations can exceed the stored liquid speeds.
        """
        # Seeds are faces with a resolved neighbour; residue and air faces near
        # them carry the bounded extension, residue beyond the band its own
        # bounded value, so the bound reads governed speeds only.
        liquid = self.cfg.pressure_discretization == "liquid_mimetic"
        water_threshold = self.cfg.liquid_resolved_fraction if liquid else 1e-6
        extension = extrapolate_velocity(state, order=self.cfg.velocity_extension_order, layers=self.extension_layers,
                                         water_threshold=water_threshold, residue_fallback=liquid)
        if self.cfg.vof_flux_reconstruction == "swept_prism":
            from shoreloop.physics3d.swept_prism import swept_prism_transport_rate
            rate = swept_prism_transport_rate(state, extension.fields)
            return math.inf if rate == 0.0 else cfl/rate
        spacings = (self.cfg.grid.dx, self.cfg.grid.dy, self.cfg.grid.dz)
        rate = 0.0
        for name, field, spacing in zip(
            ("u", "v", "w"), extension.fields, spacings, strict=True,
        ):
            mass = compute_face_liquid_mass(state, name)
            maximum = wp.zeros(1, dtype=wp.float32, device=self.device)
            wp.launch(
                _reduce_water_speed,
                dim=state.grid.interior_shape(name),
                inputs=[field, mass, maximum, 0.0, self.cfg.grid.ng],
                device=self.device,
            )
            rate += float(maximum.numpy()[0]) / spacing
        return math.inf if rate == 0.0 else cfl / rate

    def advect(self, state: State3D, dt: float, *, reconstruction=None) -> MomentumAdvectionResult:
        """Advance one coupled VOF/momentum step, restoring state on failure."""
        if state.device != self.device or state.cfg.grid != self.cfg.grid:
            raise ValueError("state grid and device must match the momentum advector")
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive and finite")
        g = self.cfg.grid
        names = ("u", "v", "w")
        for name in names:
            fill_field_boundaries(state, name)
        alpha_before = self._copy_field(state.alpha, g.shape)
        velocity_before = tuple(
            self._copy_field(state.field(name), state.grid.interior_shape(name))
            for name in names
        )
        flux_consistent = self.cfg.momentum_face_mass == "flux_consistent"
        mass_before = tuple(compute_face_liquid_mass(state, name, flux_consistent=flux_consistent)
                            for name in names)
        # The liquid projection accelerates retained thin water. A speed
        # reference drawn only from mass > 0.1 would be zero for an isolated
        # thin sheet and erase its accumulated momentum every transport step.
        # On this path every positive face mass participates in both recovery
        # and the measured global speed bound. The legacy path retains its
        # original thresholds; numerical momentum limiting remains reported.
        liquid_support = self.cfg.pressure_discretization == "liquid_mimetic"
        reference_threshold = 0.0 if liquid_support else self.speed_reference_threshold
        recovery_threshold = 0.0 if liquid_support else self.mass_threshold
        # Match transport support to the positive mass retained in recovery.
        # Pressure's geometric activation criterion is separate and unchanged.
        transport_threshold = 0.0 if liquid_support else 1e-6
        # Only faces with a resolved neighbour seed the transport velocity;
        # residue faces move with the extension of the resolved water.
        seed_threshold = self.cfg.liquid_resolved_fraction if liquid_support else 1e-6
        local_bound = self.cfg.momentum_speed_bound == "local"
        velocity_limits = [0.0, 0.0, 0.0]
        if not local_bound:
            for axis, (name, mass, velocity) in enumerate(zip(
                names, mass_before, velocity_before, strict=True,
            )):
                maximum = wp.zeros(1, dtype=wp.float32, device=self.device)
                wp.launch(
                    _reduce_water_speed,
                    dim=state.grid.interior_shape(name),
                    inputs=[velocity, mass, maximum,
                            reference_threshold, g.ng],
                    device=self.device,
                )
                velocity_limits[axis] = float(maximum.numpy()[0])
        # Each face's admissible range starts at its own pre-step value and
        # widens with the values transport exchanges with it.
        bound_lower = tuple(wp.clone(velocity) for velocity in velocity_before)
        bound_upper = tuple(wp.clone(velocity) for velocity in velocity_before)
        momentum = []
        for name, mass in zip(names, mass_before):
            field = wp.zeros(
                state.grid.padded_shape(name), dtype=wp.float32,
                device=self.device,
            )
            wp.launch(
                _form_momentum,
                dim=state.grid.interior_shape(name),
                inputs=[mass, state.field(name), field, g.ng],
                device=self.device,
            )
            momentum.append(field)
        current = (momentum[0], momentum[1], momentum[2])
        initial_momentum = current
        extension: VelocityExtension = extrapolate_velocity(state, order=self.cfg.velocity_extension_order, layers=self.extension_layers, water_threshold=seed_threshold,
            residue_fallback=liquid_support,
        )
        previous_vof_steps = self.vof.steps
        previous_clipped = self.vof.total_clipped_volume
        invalid = wp.zeros(1, dtype=wp.int32, device=self.device)

        try:
            if liquid_support:
                vof_result = self.vof.advect(
                    state, dt, transport_velocity=extension.fields, reconstruction=reconstruction,
                )
            else:
                vof_result = self.vof.advect(state, dt, transport_velocity=extension.fields)
            periodic = (
                int(state.cfg.bc.xlo == BC.PERIODIC),
                int(state.cfg.bc.ylo == BC.PERIODIC),
                int(state.cfg.bc.zlo == BC.PERIODIC),
            )
            axis_fluxes = [None, None, None]
            for directional in vof_result.directional_fluxes:
                axis_fluxes[("x", "y", "z").index(directional.axis)] = directional.fluxes
            outgoing = []
            inflow_velocity = []
            for component_axis, name in enumerate(names):
                total = wp.zeros(
                    state.grid.padded_shape(name), dtype=wp.float32,
                    device=self.device,
                )
                entering = wp.zeros_like(total)
                wp.launch(
                    _face_transport_budget,
                    dim=state.grid.interior_shape(name),
                    inputs=[extension.fields[component_axis], *axis_fluxes,
                            total, entering, component_axis,
                            g.nx, g.ny, g.nz, *periodic, g.ng, transport_threshold],
                    device=self.device,
                )
                outgoing.append(total)
                inflow_velocity.append(entering)
            for directional in vof_result.directional_fluxes:
                transport_axis = ("x", "y", "z").index(directional.axis)
                next_fields = []
                for component_axis, name in enumerate(names):
                    updated = wp.zeros(
                        state.grid.padded_shape(name), dtype=wp.float32,
                        device=self.device,
                    )
                    wp.launch(
                        _transport_momentum,
                        dim=state.grid.interior_shape(name),
                        inputs=[current[component_axis],
                                extension.fields[component_axis],
                                extension.valid_masks[component_axis],
                                mass_before[component_axis],
                                outgoing[component_axis],
                                inflow_velocity[component_axis],
                                directional.fluxes, updated,
                                bound_lower[component_axis],
                                bound_upper[component_axis], invalid,
                                component_axis, transport_axis,
                                g.nx, g.ny, g.nz, *periodic, g.ng, transport_threshold],
                        device=self.device,
                    )
                    next_fields.append(updated)
                current = (next_fields[0], next_fields[1], next_fields[2])
            if int(invalid.numpy()[0]) != 0:
                raise RuntimeError(
                    "liquid momentum flux reached outside the velocity "
                    "extension band",
                )
            before = self._momentum_total(initial_momentum)

            mass_after = tuple(
                compute_face_liquid_mass(state, name, flux_consistent=flux_consistent)
                for name in names
            )
            discarded = wp.zeros(1, dtype=wp.float64, device=self.device)
            maximum = wp.zeros(1, dtype=wp.float32, device=self.device)
            for axis, name in enumerate(names):
                wp.launch(
                    _finalise_velocity,
                    dim=state.grid.interior_shape(name),
                    inputs=[mass_after[axis], current[axis],
                            velocity_before[axis], state.field(name), discarded,
                            maximum, recovery_threshold,
                            velocity_limits[axis], bound_lower[axis],
                            bound_upper[axis], int(local_bound), g.ng],
                    device=self.device,
                )
                fill_field_boundaries(state, name)
            if liquid_support:
                # Residue owns no momentum: a face without a resolved neighbour
                # inside the resolved faces' band takes their bounded extension
                # and the difference is discarded momentum. Residue beyond the
                # band has nothing to follow and keeps its bounded value.
                owners = extrapolate_velocity(
                    state, order=self.cfg.velocity_extension_order,
                    layers=self.extension_layers, water_threshold=seed_threshold)
                maximum.zero_()
                for axis, name in enumerate(names):
                    wp.launch(
                        _assign_non_owner_velocity,
                        dim=state.grid.interior_shape(name),
                        inputs=[mass_after[axis], owners.seed_masks[axis],
                                owners.band_masks[axis], owners.fields[axis],
                                state.field(name), discarded, maximum, g.ng],
                        device=self.device,
                    )
                    fill_field_boundaries(state, name)
            discarded_value = (
                float(discarded.numpy()[0]) * self.cfg.physics.rho
                * g.dx * g.dy * g.dz
            )
            after_fields = []
            for axis, name in enumerate(names):
                field = wp.zeros(
                    state.grid.padded_shape(name), dtype=wp.float32,
                    device=self.device,
                )
                wp.launch(
                    _form_momentum,
                    dim=state.grid.interior_shape(name),
                    inputs=[mass_after[axis], state.field(name), field, g.ng],
                    device=self.device,
                )
                after_fields.append(field)
            after = self._momentum_total(
                (after_fields[0], after_fields[1], after_fields[2]),
            )
        except Exception:
            wp.launch(
                _copy_cell_field, dim=g.shape,
                inputs=[alpha_before, state.alpha, g.ng], device=self.device,
            )
            fill_field_boundaries(state, "alpha")
            for name, saved in zip(names, velocity_before):
                wp.launch(
                    _copy_cell_field, dim=state.grid.interior_shape(name),
                    inputs=[saved, state.field(name), g.ng], device=self.device,
                )
                fill_field_boundaries(state, name)
            self.vof.steps = previous_vof_steps
            self.vof.total_clipped_volume = previous_clipped
            raise

        return MomentumAdvectionResult(
            vof=vof_result,
            momentum_before=before,
            momentum_after=after,
            momentum_drift=tuple(a - b for a, b in zip(after, before)),
            max_water_speed=float(maximum.numpy()[0]),
            discarded_momentum=discarded_value,
        )
