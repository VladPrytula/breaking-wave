"""Boundary ownership and halo fills for the regular MAC grid.

Ownership is deliberately fixed before the discrete operators are written:

- Cell-centred fields and velocity components tangential to a side own all
  physical samples; the boundary operation fills only their ghost samples.
- A wall owns its two physical normal-velocity faces and sets them to zero.
  Normal ghosts are odd-reflected; tangential ghosts are free-slip copies.
- An open side owns its physical normal face and copies the nearest strictly
  interior face. Every ghost is the same zero-gradient continuation. This is
  bring-up plumbing, not the later absorbing wave boundary.
- On a periodic axis the low normal face is canonical. The high physical face
  copies it, while ghosts wrap from the unique faces at the opposite side.

Axes are filled x, then y, then z. Each later pass includes existing halos in
its transverse launch, so edges and corners inherit the same rules without a
separate corner convention.
"""

from __future__ import annotations

import warp as wp

from shoreloop.physics3d.state import State3D


@wp.kernel
def _fill_axis(
    field: wp.array3d(dtype=wp.float32),
    axis: int,
    n: int,
    ng: int,
    kind_lo: int,
    kind_hi: int,
    normal: int,
):
    slot, a, b = wp.tid()
    width = ng + normal
    side = slot // width
    r = slot - side * width
    lo = ng
    hi = ng + n - 1
    kind = kind_lo
    if side != 0:
        kind = kind_hi

    q = lo
    source_q = lo
    sign = wp.float32(1.0)
    zero = int(0)

    if normal != 0 and r == 0:
        # The physical normal face itself is boundary-owned.
        if side == 0:
            q = lo
            if kind == 0:  # BC.WALL
                zero = 1
            elif kind == 2:  # BC.OPEN
                source_q = lo + 1
            else:  # low periodic face is canonical
                source_q = lo
        else:
            q = hi
            if kind == 0:
                zero = 1
            elif kind == 2:
                source_q = hi - 1
            else:  # high periodic face duplicates the low face
                source_q = lo
    else:
        layer = r
        if normal != 0:
            layer = r - 1
        if side == 0:
            q = lo - 1 - layer
            if kind == 1:  # BC.PERIODIC
                source_q = hi - layer
                if normal != 0:
                    source_q = hi - 1 - layer
            elif kind == 0 and normal != 0:
                source_q = lo + 1 + layer
                sign = -wp.float32(1.0)
            else:
                source_q = lo + normal
        else:
            q = hi + 1 + layer
            if kind == 1:
                source_q = lo + layer
                if normal != 0:
                    source_q = lo + 1 + layer
            elif kind == 0 and normal != 0:
                source_q = hi - 1 - layer
                sign = -wp.float32(1.0)
            else:
                source_q = hi - normal

    value = wp.float32(0.0)
    if zero == 0:
        if axis == 0:
            value = sign * field[source_q, a, b]
        elif axis == 1:
            value = sign * field[a, source_q, b]
        else:
            value = sign * field[a, b, source_q]

    if axis == 0:
        field[q, a, b] = value
    elif axis == 1:
        field[a, q, b] = value
    else:
        field[a, b, q] = value


_NORMAL_AXIS = {"u": 0, "v": 1, "w": 2}


def _fill_field(state: State3D, name: str) -> None:
    field = state.field(name)
    grid = state.grid
    bc = state.cfg.bc
    ng = state.cfg.grid.ng
    physical = grid.interior_shape(name)
    kinds = ((bc.xlo, bc.xhi), (bc.ylo, bc.yhi), (bc.zlo, bc.zhi))

    # Sequential launches on Warp's current stream make the x/y/z corner rule
    # deterministic without host synchronization between axes.
    for axis in range(3):
        normal = int(_NORMAL_AXIS.get(name, -1) == axis)
        width = ng + normal
        if axis == 0:
            transverse = (field.shape[1], field.shape[2])
        elif axis == 1:
            transverse = (field.shape[0], field.shape[2])
        else:
            transverse = (field.shape[0], field.shape[1])
        wp.launch(
            _fill_axis,
            dim=(2 * width, *transverse),
            inputs=[field, axis, physical[axis], ng,
                    int(kinds[axis][0]), int(kinds[axis][1]), normal],
            device=state.device,
        )


def fill_boundaries(state: State3D) -> None:
    """Apply the configured boundary ownership to every primary state field."""
    for name in state.FIELDS:
        _fill_field(state, name)


def fill_field_boundaries(state: State3D, name: str) -> None:
    """Apply boundary ownership to one named primary field."""
    state.field(name)  # validate before launching any kernels
    _fill_field(state, name)
