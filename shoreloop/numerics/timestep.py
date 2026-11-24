"""CFL reduction."""

from __future__ import annotations

import warp as wp

from shoreloop.precision import FLOAT, ONE, ZERO

from shoreloop.numerics.flux import desing_u


@wp.kernel
def max_wave_rate(
    h: wp.array2d(dtype=FLOAT), qx: wp.array2d(dtype=FLOAT), qy: wp.array2d(dtype=FLOAT),
    out: wp.array(dtype=FLOAT),
    ng: int, dx: FLOAT, dy: FLOAT, g: FLOAT, eps: FLOAT,
):
    """max over cells of (|u|+c)/dx + (|v|+c)/dy.

    This is the unsplit 2D rate; dt = cfl / rate. Using the sum (not the max)
    of the two directional rates is what the dimensionally-unsplit update
    actually requires.
    """
    i, j = wp.tid()
    I = i + ng
    J = j + ng
    hv = wp.max(h[I, J], ZERO)
    if hv > ZERO:
        c = wp.sqrt(g * hv)
        u = wp.abs(desing_u(hv, qx[I, J], eps))
        v = wp.abs(desing_u(hv, qy[I, J], eps))
        rate = (u + c) / dx + (v + c) / dy
        wp.atomic_max(out, 0, rate)
