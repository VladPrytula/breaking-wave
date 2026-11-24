"""One-direction geometric liquid-volume fluxes from PLIC cells."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Sequence

import warp as wp

from shoreloop.config import BC
from shoreloop.physics3d.plic import (_NORMAL_EPSILON, PLICReconstruction,
                                     _normal_tuple, _unit_cube_volume_l1,
                                     unit_cube_volume)
from shoreloop.physics3d.solids import _solid_cell
from shoreloop.physics3d.polyhedron import _Planes, convex_box_moments
from shoreloop.physics3d.state import State3D


@dataclass(frozen=True)
class DirectionalFluxResult:
    """Dimensionless donor-cell liquid fluxes for one Cartesian sweep."""

    axis: str
    fluxes: wp.array
    max_courant: float


def _axis_index(axis: str | int) -> int:
    if isinstance(axis, bool):
        raise ValueError("axis must be 'x', 'y', 'z', 0, 1, or 2")
    if isinstance(axis, int):
        if 0 <= axis <= 2:
            return axis
    elif isinstance(axis, str) and axis in ("x", "y", "z"):
        return ("x", "y", "z").index(axis)
    raise ValueError("axis must be 'x', 'y', 'z', 0, 1, or 2")


def directional_liquid_flux(
    normal: Sequence[float],
    offset: float,
    fraction: float,
    courant: float,
    axis: str | int,
) -> float:
    """Return signed liquid volume crossing one donor-cell face.

    The result is measured as a fraction of the donor-cell volume. Positive
    Courant number sweeps the high-face slab; negative Courant number sweeps
    the low-face slab.
    """
    axis_index = _axis_index(axis)
    if not math.isfinite(offset):
        raise ValueError("offset must be finite")
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be finite in [0, 1]")
    if not math.isfinite(courant) or abs(courant) > 1.0:
        raise ValueError("courant must be finite in [-1, 1]")
    if fraction == 0.0 or courant == 0.0:
        return 0.0
    if fraction == 1.0:
        return courant

    n = _normal_tuple(normal)
    sweep = abs(courant)
    lower = [0.0, 0.0, 0.0]
    lengths = [1.0, 1.0, 1.0]
    lengths[axis_index] = sweep
    if courant > 0.0:
        lower[axis_index] = 1.0 - sweep
    scaled = tuple(n[q] * lengths[q] for q in range(3))
    shifted_offset = offset - sum(n[q] * lower[q] for q in range(3))
    magnitude = sum(abs(value) for value in scaled)
    if magnitude <= _NORMAL_EPSILON:
        return 0.0
    sub_fraction = unit_cube_volume(
        tuple(value / magnitude for value in scaled),
        shifted_offset / magnitude,
    )
    liquid = min(fraction, max(0.0, sweep * sub_fraction))
    return liquid if courant > 0.0 else -liquid


@wp.func
def _directional_liquid_flux_l1(
    normal: wp.vec3,
    offset: wp.float32,
    fraction: wp.float32,
    courant: wp.float32,
    axis: int,
    mixed_tolerance: wp.float32,
) -> wp.float32:
    if courant == wp.float32(0.0) or fraction <= wp.float32(0.0):
        return wp.float32(0.0)
    if (fraction <= mixed_tolerance
            or fraction >= wp.float32(1.0) - mixed_tolerance):
        return fraction * courant

    sweep = wp.abs(courant)
    lower = wp.vec3(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    lengths = wp.vec3(wp.float32(1.0), wp.float32(1.0), wp.float32(1.0))
    if axis == 0:
        lengths = wp.vec3(sweep, wp.float32(1.0), wp.float32(1.0))
        if courant > wp.float32(0.0):
            lower = wp.vec3(wp.float32(1.0) - sweep,
                            wp.float32(0.0), wp.float32(0.0))
    elif axis == 1:
        lengths = wp.vec3(wp.float32(1.0), sweep, wp.float32(1.0))
        if courant > wp.float32(0.0):
            lower = wp.vec3(wp.float32(0.0),
                            wp.float32(1.0) - sweep, wp.float32(0.0))
    else:
        lengths = wp.vec3(wp.float32(1.0), wp.float32(1.0), sweep)
        if courant > wp.float32(0.0):
            lower = wp.vec3(wp.float32(0.0), wp.float32(0.0),
                            wp.float32(1.0) - sweep)

    scaled = wp.cw_mul(normal, lengths)
    shifted_offset = offset - wp.dot(normal, lower)
    magnitude = (wp.abs(scaled[0]) + wp.abs(scaled[1])
                 + wp.abs(scaled[2]))
    if magnitude <= wp.float32(_NORMAL_EPSILON):
        return wp.float32(0.0)
    sub_fraction = _unit_cube_volume_l1(
        scaled / magnitude, shifted_offset / magnitude,
    )
    liquid = wp.min(fraction, wp.max(wp.float32(0.0), sweep * sub_fraction))
    if courant < wp.float32(0.0):
        liquid = -liquid
    return liquid


@wp.func
def _cut_liquid_flux(
    normal: wp.vec3,
    offset: wp.float32,
    courant: wp.float32,
    aperture: wp.float32,
    low_normal: wp.vec3,
    low_offset: wp.float32,
    high_normal: wp.vec3,
    high_offset: wp.float32,
    axis: int,
) -> wp.vec2:
    """Water flux and open volume in the common face's swept open prism.

    The unit prism maps to a slab of width abs(C). Restrict it to the donor
    open cell AND the extruded common face footprint. The aperture-C flux
    owns the swept open amount; its water saturation comes from this joint
    PLIC geometry. This retains the split operator's full-water flux a*C.
    Aperture variation along the constant-velocity prism is a first-order
    geometric approximation, not an additional mass source or correction.
    """
    sweep = wp.abs(courant)
    donor_normal = high_normal
    donor_offset = high_offset
    water_offset = offset
    if courant > wp.float32(0.0):
        donor_normal = low_normal
        # Preserve a small sweep without first rounding 1-C to 1.
        donor_offset = low_offset - low_normal[axis] + low_normal[axis] * sweep
        water_offset = offset - normal[axis] + normal[axis] * sweep
    donor_normal[axis] *= sweep
    water_normal = normal
    water_normal[axis] *= sweep
    face_low = low_normal
    face_high = high_normal
    face_low[axis] = wp.float32(0.0)
    face_high[axis] = wp.float32(0.0)
    planes = _Planes()
    planes[0] = donor_normal
    planes[1] = face_low
    planes[2] = face_high
    offsets = wp.vec4(donor_offset, low_offset - low_normal[axis],
                       high_offset, water_offset)
    lower = wp.vec3(wp.float32(0.0))
    upper = wp.vec3(wp.float32(1.0))
    open_volume = convex_box_moments(lower, upper, planes, offsets, 3)[0]
    planes[3] = water_normal
    water_volume = convex_box_moments(lower, upper, planes, offsets, 4)[0]
    flux = wp.float32(0.0)
    if open_volume > wp.float32(0.0):
        saturation = wp.clamp(water_volume / open_volume,
                              wp.float32(0.0), wp.float32(1.0))
        flux = aperture * courant * saturation
    return wp.vec2(flux, open_volume)


@wp.kernel
def _compute_directional_fluxes(
    alpha: wp.array3d(dtype=wp.float32),
    solid: wp.array3d(dtype=wp.int32),
    aperture: wp.array3d(dtype=wp.float32),
    open_normals: wp.array3d(dtype=wp.vec3),
    open_offsets: wp.array3d(dtype=wp.float32),
    normals: wp.array3d(dtype=wp.vec3),
    offsets: wp.array3d(dtype=wp.float32),
    velocity: wp.array3d(dtype=wp.float32),
    fluxes: wp.array3d(dtype=wp.float32),
    maximum_courant: wp.array(dtype=wp.float32),
    invalid: wp.array(dtype=wp.int32),
    axis: int,
    nx: int,
    ny: int,
    nz: int,
    inverse_spacing_dt: wp.float32,
    periodic: int,
    mixed_tolerance: wp.float32,
    ng: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    face_aperture = aperture[I, J, K]
    if face_aperture <= wp.float32(0.0):
        fluxes[I, J, K] = wp.float32(0.0)
        return
    periodic_x = int(0)
    periodic_y = int(0)
    periodic_z = int(0)
    if axis == 0:
        periodic_x = periodic
    elif axis == 1:
        periodic_y = periodic
    else:
        periodic_z = periodic
    low_i = i
    low_j = j
    low_k = k
    if axis == 0:
        low_i -= 1
    elif axis == 1:
        low_j -= 1
    else:
        low_k -= 1
    if (_solid_cell(solid, low_i, low_j, low_k, nx, ny, nz,
                    periodic_x, periodic_y, periodic_z, ng) != 0
            or _solid_cell(solid, i, j, k, nx, ny, nz,
                           periodic_x, periodic_y, periodic_z, ng) != 0):
        fluxes[I, J, K] = wp.float32(0.0)
        return

    speed = velocity[I, J, K]
    courant = speed * inverse_spacing_dt
    if not wp.isfinite(courant):
        wp.atomic_max(invalid, 0, 1)
        return

    q = i
    if axis == 1:
        q = j
    elif axis == 2:
        q = k
    n = nx
    if axis == 1:
        n = ny
    elif axis == 2:
        n = nz
    donor_q = q
    if courant > wp.float32(0.0):
        donor_q = q - 1
    if donor_q < 0 or donor_q >= n:
        if periodic == 0:
            fluxes[I, J, K] = wp.float32(0.0)
            return
        if donor_q < 0:
            donor_q += n
        else:
            donor_q -= n

    donor_i = i
    donor_j = j
    donor_k = k
    if axis == 0:
        donor_i = donor_q
    elif axis == 1:
        donor_j = donor_q
    else:
        donor_k = donor_q
    DI = donor_i + ng
    DJ = donor_j + ng
    DK = donor_k + ng
    fraction = alpha[DI, DJ, DK]
    if fraction <= wp.float32(0.0):
        # A dry donor never moves liquid. Its face velocity may be an extended
        # air value beyond every water speed, so it is not held to the
        # Courant bound that the swept-volume construction needs.
        fluxes[I, J, K] = wp.float32(0.0)
        return
    magnitude = wp.abs(courant)
    wp.atomic_max(maximum_courant, 0, magnitude)
    if magnitude > wp.float32(1.000001):
        wp.atomic_max(invalid, 0, 1)
        return
    result = face_aperture * _directional_liquid_flux_l1(
        normals[DI, DJ, DK], offsets[DI, DJ, DK], alpha[DI, DJ, DK],
        courant, axis, mixed_tolerance,
    )
    low_q = q - 1
    high_q = q
    if periodic != 0:
        if low_q < 0:
            low_q += n
        if high_q >= n:
            high_q -= n
    if (low_q >= 0 and high_q < n and wp.abs(courant) > wp.float32(0.0)
            and fraction > mixed_tolerance
            and fraction < wp.float32(1.0) - mixed_tolerance):
        LI = I
        LJ = J
        LK = K
        HI = I
        HJ = J
        HK = K
        if axis == 0:
            LI = low_q + ng
            HI = high_q + ng
        elif axis == 1:
            LJ = low_q + ng
            HJ = high_q + ng
        else:
            LK = low_q + ng
            HK = high_q + ng
        ln = open_normals[LI, LJ, LK]
        hn = open_normals[HI, HJ, HK]
        if wp.dot(ln, ln) + wp.dot(hn, hn) > wp.float32(0.0):
            cut_flux = _cut_liquid_flux(
                normals[DI, DJ, DK], offsets[DI, DJ, DK], courant,
                face_aperture, ln, open_offsets[LI, LJ, LK],
                hn, open_offsets[HI, HJ, HK], axis,
            )
            if cut_flux[1] <= wp.float32(0.0):
                wp.atomic_max(invalid, 0, 2)
            result = cut_flux[0]
    fluxes[I, J, K] = result


def compute_directional_liquid_fluxes(
    state: State3D,
    reconstruction: PLICReconstruction,
    axis: str | int,
    dt: float,
    *,
    velocity: wp.array | None = None,
    mixed_tolerance: float = 1.0e-6,
) -> DirectionalFluxResult:
    """Evaluate first-order face-velocity PLIC fluxes for one sweep."""
    axis_index = _axis_index(axis)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite")
    if (not math.isfinite(mixed_tolerance) or mixed_tolerance < 0.0
            or mixed_tolerance >= 0.5):
        raise ValueError("mixed_tolerance must be finite in [0, 0.5)")
    grid = state.grid
    if tuple(reconstruction.normals.shape) != grid.cell_shape:
        raise ValueError("PLIC normal field shape does not match state")
    if tuple(reconstruction.offsets.shape) != grid.cell_shape:
        raise ValueError("PLIC offset field shape does not match state")

    names = ("u", "v", "w")
    apertures = (state.u_aperture, state.v_aperture, state.w_aperture)
    spacings = (state.cfg.grid.dx, state.cfg.grid.dy, state.cfg.grid.dz)
    counts = state.cfg.grid.shape
    name = names[axis_index]
    transport_velocity = state.field(name) if velocity is None else velocity
    expected_velocity_shape = grid.padded_shape(name)
    if tuple(transport_velocity.shape) != expected_velocity_shape:
        raise ValueError(
            f"{name} transport velocity must have padded shape "
            f"{expected_velocity_shape}, got {tuple(transport_velocity.shape)}",
        )
    if transport_velocity.dtype != wp.float32:
        raise TypeError(f"{name} transport velocity must use float32")
    fluxes = wp.zeros(
        grid.padded_shape(name), dtype=wp.float32, device=state.device,
    )
    maximum = wp.zeros(1, dtype=wp.float32, device=state.device)
    invalid = wp.zeros(1, dtype=wp.int32, device=state.device)
    bc = state.cfg.bc
    periodic = (
        bc.xlo == BC.PERIODIC,
        bc.ylo == BC.PERIODIC,
        bc.zlo == BC.PERIODIC,
    )[axis_index]
    wp.launch(
        _compute_directional_fluxes,
        dim=grid.interior_shape(name),
        inputs=[state.alpha, state.solid, apertures[axis_index],
                state.open_normals, state.open_offsets,
                reconstruction.normals,
                reconstruction.offsets,
                transport_velocity, fluxes, maximum, invalid, axis_index,
                *counts, dt / spacings[axis_index], int(periodic),
                mixed_tolerance, state.cfg.grid.ng],
        device=state.device,
    )
    max_courant = float(maximum.numpy()[0])
    if int(invalid.numpy()[0]) == 2:
        raise RuntimeError("open face has no donor open volume in its swept prism")
    if int(invalid.numpy()[0]) != 0:
        raise RuntimeError(
            f"non-finite or super-unit Courant number in {names[axis_index]} "
            f"sweep (max={max_courant:.6g})",
        )
    return DirectionalFluxResult(
        axis=("x", "y", "z")[axis_index],
        fluxes=fluxes,
        max_courant=max_courant,
    )
