"""Narrow-band velocity extrapolation for geometric interface transport."""

from __future__ import annotations

from dataclasses import dataclass
import math

import warp as wp

from shoreloop.config import BC
from shoreloop.physics3d.resolved import resolved_liquid
from shoreloop.physics3d.state import State3D


@dataclass(frozen=True)
class VelocityExtension:
    """Air-band transport velocities kept separate from water momentum."""

    u: wp.array
    v: wp.array
    w: wp.array
    valid_masks: tuple[wp.array, wp.array, wp.array]
    layers: int
    extended_faces: tuple[int, int, int]
    # Faces that keep their stored value: one adjacent cell holds more than
    # the threshold fraction of a cell of liquid. Every other valid face is
    # extended from them.
    seed_masks: tuple[wp.array, wp.array, wp.array] = ()
    # The band grown from those seeds alone. With ``residue_fallback`` faces
    # holding liquid beyond it also keep their stored value and grow a band,
    # so every face a liquid flux can reach is valid.
    band_masks: tuple[wp.array, wp.array, wp.array] = ()

    @property
    def fields(self) -> tuple[wp.array, wp.array, wp.array]:
        return (self.u, self.v, self.w)


@wp.func
def _wrapped_cell_index(index: int, count: int, periodic: int) -> int:
    result = index
    if result < 0:
        if periodic != 0:
            result += count
        else:
            result = -1
    elif result >= count:
        if periodic != 0:
            result -= count
        else:
            result = -1
    return result


@wp.func
def _face_neighbor_index(
    index: int,
    count: int,
    periodic: int,
    duplicated_seam: int,
) -> int:
    if index >= 0 and index < count:
        return index
    if periodic == 0:
        return -1
    # Faces 0 and count-1 coincide on a duplicated periodic seam. Wrap by the
    # distinct span so stencils two faces past the seam land correctly.
    span = count
    if duplicated_seam != 0:
        span = count - 1
    result = index % span
    if result < 0:
        result += span
    return result


@wp.func
def _water_cell(
    alpha: wp.array3d(dtype=wp.float32),
    theta: wp.array3d(dtype=wp.float32),
    i: int,
    j: int,
    k: int,
    nx: int,
    ny: int,
    nz: int,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    threshold: wp.float32,
    ng: int,
) -> int:
    ci = _wrapped_cell_index(i, nx, periodic_x)
    cj = _wrapped_cell_index(j, ny, periodic_y)
    ck = _wrapped_cell_index(k, nz, periodic_z)
    if ci < 0 or cj < 0 or ck < 0:
        return 0
    return resolved_liquid(alpha[ci + ng, cj + ng, ck + ng], theta[ci + ng, cj + ng, ck + ng], threshold)


@wp.kernel
def _initialise_extension(
    alpha: wp.array3d(dtype=wp.float32),
    theta: wp.array3d(dtype=wp.float32),
    authoritative: wp.array3d(dtype=wp.float32),
    values: wp.array3d(dtype=wp.float32),
    valid: wp.array3d(dtype=wp.int32),
    invalid: wp.array(dtype=wp.int32),
    axis: int,
    nx: int,
    ny: int,
    nz: int,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    threshold: wp.float32,
    ng: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    value = authoritative[I, J, K]
    if not wp.isfinite(value):
        wp.atomic_max(invalid, 0, 1)
        return

    known = int(0)
    if axis == 0:
        known = wp.max(
            _water_cell(alpha, theta, i - 1, j, k, nx, ny, nz,
                        periodic_x, periodic_y, periodic_z, threshold, ng),
            _water_cell(alpha, theta, i, j, k, nx, ny, nz,
                        periodic_x, periodic_y, periodic_z, threshold, ng),
        )
    elif axis == 1:
        known = wp.max(
            _water_cell(alpha, theta, i, j - 1, k, nx, ny, nz,
                        periodic_x, periodic_y, periodic_z, threshold, ng),
            _water_cell(alpha, theta, i, j, k, nx, ny, nz,
                        periodic_x, periodic_y, periodic_z, threshold, ng),
        )
    else:
        known = wp.max(
            _water_cell(alpha, theta, i, j, k - 1, nx, ny, nz,
                        periodic_x, periodic_y, periodic_z, threshold, ng),
            _water_cell(alpha, theta, i, j, k, nx, ny, nz,
                        periodic_x, periodic_y, periodic_z, threshold, ng),
        )
    valid[I, J, K] = known
    if known != 0:
        values[I, J, K] = value


@wp.kernel
def _seed_residue(
    alpha: wp.array3d(dtype=wp.float32),
    theta: wp.array3d(dtype=wp.float32),
    authoritative: wp.array3d(dtype=wp.float32),
    values: wp.array3d(dtype=wp.float32),
    valid: wp.array3d(dtype=wp.int32),
    axis: int,
    nx: int,
    ny: int,
    nz: int,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    ng: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    if valid[I, J, K] != 0:
        return
    if axis == 0:
        wet = wp.max(_water_cell(alpha, theta, i - 1, j, k, nx, ny, nz,
                                 periodic_x, periodic_y, periodic_z, wp.float32(0.0), ng),
                     _water_cell(alpha, theta, i, j, k, nx, ny, nz,
                                 periodic_x, periodic_y, periodic_z, wp.float32(0.0), ng))
    elif axis == 1:
        wet = wp.max(_water_cell(alpha, theta, i, j - 1, k, nx, ny, nz,
                                 periodic_x, periodic_y, periodic_z, wp.float32(0.0), ng),
                     _water_cell(alpha, theta, i, j, k, nx, ny, nz,
                                 periodic_x, periodic_y, periodic_z, wp.float32(0.0), ng))
    else:
        wet = wp.max(_water_cell(alpha, theta, i, j, k - 1, nx, ny, nz,
                                 periodic_x, periodic_y, periodic_z, wp.float32(0.0), ng),
                     _water_cell(alpha, theta, i, j, k, nx, ny, nz,
                                 periodic_x, periodic_y, periodic_z, wp.float32(0.0), ng))
    if wet != 0:
        values[I, J, K] = authoritative[I, J, K]
        valid[I, J, K] = 1


@wp.kernel
def _extend_one_layer(
    current_values: wp.array3d(dtype=wp.float32),
    current_valid: wp.array3d(dtype=wp.int32),
    next_values: wp.array3d(dtype=wp.float32),
    next_valid: wp.array3d(dtype=wp.int32),
    extended: wp.array(dtype=wp.int32),
    invalid: wp.array(dtype=wp.int32),
    sx: int,
    sy: int,
    sz: int,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    velocity_axis: int,
    order: int,
    ng: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    if current_valid[I, J, K] != 0:
        next_values[I, J, K] = current_values[I, J, K]
        next_valid[I, J, K] = 1
        return

    total = wp.float32(0.0)
    candidates = int(0)
    for direction in range(6):
        di = int(0)
        dj = int(0)
        dk = int(0)
        if direction == 0:
            di = -1
        elif direction == 1:
            di = 1
        elif direction == 2:
            dj = -1
        elif direction == 3:
            dj = 1
        elif direction == 4:
            dk = -1
        else:
            dk = 1

        ni = _face_neighbor_index(i + di, sx, periodic_x,
                                  int(velocity_axis == 0))
        nj = _face_neighbor_index(j + dj, sy, periodic_y,
                                  int(velocity_axis == 1))
        nk = _face_neighbor_index(k + dk, sz, periodic_z,
                                  int(velocity_axis == 2))
        if ni < 0 or nj < 0 or nk < 0:
            continue
        NI = ni + ng
        NJ = nj + ng
        NK = nk + ng
        if current_valid[NI, NJ, NK] == 0:
            continue

        candidate = current_values[NI, NJ, NK]
        si = _face_neighbor_index(ni + di, sx, periodic_x,
                                  int(velocity_axis == 0))
        sj = _face_neighbor_index(nj + dj, sy, periodic_y,
                                  int(velocity_axis == 1))
        sk = _face_neighbor_index(nk + dk, sz, periodic_z,
                                  int(velocity_axis == 2))
        if order == 1 and si >= 0 and sj >= 0 and sk >= 0:
            SI = si + ng
            SJ = sj + ng
            SK = sk + ng
            if current_valid[SI, SJ, SK] != 0:
                candidate = (wp.float32(2.0) * candidate
                             - current_values[SI, SJ, SK])
        if not wp.isfinite(candidate):
            wp.atomic_max(invalid, 0, 1)
            continue
        total += candidate
        candidates += 1

    if candidates > 0:
        next_values[I, J, K] = total / wp.float32(candidates)
        next_valid[I, J, K] = 1
        wp.atomic_add(extended, 0, 1)


def extrapolate_velocity(
    state: State3D,
    *,
    layers: int = 3,
    water_threshold: float = 1.0e-6,
    order: int = 0,
    residue_fallback: bool = False,
) -> VelocityExtension:
    """Extend face velocity a fixed number of layers beyond resolved cells.

    A face keeps its stored value when an adjacent cell holds more than
    ``water_threshold`` of a cell of liquid (``alpha*theta``); the remaining
    faces within ``layers`` of those are extended. ``order`` 0 averages the
    valid neighbours of each new face, so no extended value leaves the range
    of the values it is built from; ``order`` 1 is the linear continuation
    2a - b of each neighbour's own gradient. With ``residue_fallback`` faces
    holding any liquid beyond that band keep their stored value and grow a
    band of their own, so a liquid flux never meets an invalid face.
    """
    if isinstance(layers, bool) or not isinstance(layers, int) or layers <= 0:
        raise ValueError("layers must be a positive integer")
    if order not in (0, 1):
        raise ValueError("order must be 0 (constant) or 1 (linear)")
    if layers > 8:
        raise ValueError("at most eight velocity-extension layers are supported")
    if (not math.isfinite(water_threshold) or water_threshold < 0.0
            or water_threshold >= 1.0):
        raise ValueError("water_threshold must be finite in [0, 1)")

    g = state.cfg.grid
    bc = state.cfg.bc
    periodic = (
        int(bc.xlo == BC.PERIODIC),
        int(bc.ylo == BC.PERIODIC),
        int(bc.zlo == BC.PERIODIC),
    )
    invalid = wp.zeros(1, dtype=wp.int32, device=state.device)
    output_fields: list[wp.array] = []
    output_masks: list[wp.array] = []
    seed_masks: list[wp.array] = []
    band_masks: list[wp.array] = []
    extended_counts: list[int] = []

    for axis, name in enumerate(("u", "v", "w")):
        shape = state.grid.interior_shape(name)
        padded = state.grid.padded_shape(name)
        current_values = wp.zeros(padded, dtype=wp.float32, device=state.device)
        current_valid = wp.zeros(padded, dtype=wp.int32, device=state.device)
        wp.launch(
            _initialise_extension,
            dim=shape,
            inputs=[state.alpha, state.theta, state.field(name), current_values,
                    current_valid, invalid, axis, g.nx, g.ny, g.nz,
                    *periodic, water_threshold, g.ng],
            device=state.device,
        )
        seed_masks.append(current_valid)
        extended = wp.zeros(1, dtype=wp.int32, device=state.device)
        for _ in range(layers):
            next_values = wp.zeros(
                padded, dtype=wp.float32, device=state.device,
            )
            next_valid = wp.zeros(
                padded, dtype=wp.int32, device=state.device,
            )
            wp.launch(
                _extend_one_layer,
                dim=shape,
                inputs=[current_values, current_valid, next_values, next_valid,
                        extended, invalid, *shape, *periodic, axis, order, g.ng],
                device=state.device,
            )
            current_values = next_values
            current_valid = next_valid
        band_masks.append(current_valid)
        if residue_fallback:
            current_values = wp.clone(current_values)
            current_valid = wp.clone(current_valid)
            wp.launch(
                _seed_residue, dim=shape,
                inputs=[state.alpha, state.theta, state.field(name), current_values,
                        current_valid, axis, g.nx, g.ny, g.nz, *periodic, g.ng],
                device=state.device,
            )
            for _ in range(layers):
                next_values = wp.zeros(padded, dtype=wp.float32, device=state.device)
                next_valid = wp.zeros(padded, dtype=wp.int32, device=state.device)
                wp.launch(
                    _extend_one_layer,
                    dim=shape,
                    inputs=[current_values, current_valid, next_values, next_valid,
                            extended, invalid, *shape, *periodic, axis, order, g.ng],
                    device=state.device,
                )
                current_values = next_values
                current_valid = next_valid
        output_fields.append(current_values)
        output_masks.append(current_valid)
        extended_counts.append(int(extended.numpy()[0]))

    if int(invalid.numpy()[0]) != 0:
        raise FloatingPointError("non-finite velocity during interface extension")
    return VelocityExtension(
        u=output_fields[0],
        v=output_fields[1],
        w=output_fields[2],
        valid_masks=(output_masks[0], output_masks[1], output_masks[2]),
        layers=layers,
        extended_faces=(extended_counts[0], extended_counts[1],
                        extended_counts[2]),
        seed_masks=(seed_masks[0], seed_masks[1], seed_masks[2]),
        band_masks=(band_masks[0], band_masks[1], band_masks[2]),
    )
