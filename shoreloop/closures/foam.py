"""Foam as a transported closure variable.

    d(c_f)/dt + div(c_f u) = alpha_f P_break - c_f / tau_f + kappa_f lap(c_f)

Three parameters with explicit physical meaning, which is the whole point:

  alpha_f  how much foam a watt of dissipated wave energy makes  [1/(m^2 W) ish]
  tau_f    how long foam survives once production stops          [s]
  kappa_f  how much it spreads by unresolved turbulence          [m^2/s]

Foam is produced *only* where the solver measured breaking dissipation, then
follows the actual flow and decays on a physical timescale. No particles, no
curvature heuristic, no "spawn where it looks white". That causal chain --
resolved energy -> breaking -> measured dissipation -> foam -- is the part of
`plan.md` worth protecting, because every link is separately testable.

Advection is first-order upwind, which is positivity-preserving under the CFL
the solver already satisfies. Foam is a closure field with no conservation law of
its own to violate, so the accuracy loss versus the second-order hydrodynamics is
not a concern; positivity is, since a negative foam concentration is meaningless.
"""

from __future__ import annotations

import warp as wp

from shoreloop.numerics.flux import desing_u
from shoreloop.precision import FLOAT, HALF, ONE, ZERO


@wp.kernel
def advance_foam(
    cf: wp.array2d(dtype=FLOAT), cf_new: wp.array2d(dtype=FLOAT),
    h: wp.array2d(dtype=FLOAT), qx: wp.array2d(dtype=FLOAT),
    qy: wp.array2d(dtype=FLOAT), P_break: wp.array2d(dtype=FLOAT),
    dt: FLOAT, dx: FLOAT, dy: FLOAT,
    alpha: FLOAT, tau: FLOAT, kappa: FLOAT, cf_max: FLOAT,
    h_wet: FLOAT, eps: FLOAT, ng: int,
):
    i, j = wp.tid()
    I = i + ng
    J = j + ng

    hv = wp.max(h[I, J], ZERO)
    if hv <= h_wet:
        # Foam cannot sit on dry land. Whatever was here has drained or been
        # stranded; the wetness field handles the visual residue, not this.
        cf_new[I, J] = ZERO
        return

    c0 = cf[I, J]

    # --- upwind advection: -div(c u) ---
    adv = ZERO
    for axis in range(2):
        di = 0
        dj = 0
        ds = dx
        if axis == 1:
            di = 0
            dj = 1
            ds = dy
        else:
            di = 1
            dj = 0

        hm = wp.max(h[I - di, J - dj], ZERO)
        hp = wp.max(h[I + di, J + dj], ZERO)
        if axis == 0:
            um = desing_u(hm, qx[I - di, J - dj], eps)
            u0 = desing_u(hv, qx[I, J], eps)
            up = desing_u(hp, qx[I + di, J + dj], eps)
        else:
            um = desing_u(hm, qy[I - di, J - dj], eps)
            u0 = desing_u(hv, qy[I, J], eps)
            up = desing_u(hp, qy[I + di, J + dj], eps)

        # face velocities, and no transport across a wet/dry face
        u_lo = HALF * (um + u0)
        u_hi = HALF * (u0 + up)
        if hm <= h_wet:
            u_lo = ZERO
        if hp <= h_wet:
            u_hi = ZERO

        c_lo = cf[I - di, J - dj]
        if u_lo < ZERO:
            c_lo = c0
        c_hi = c0
        if u_hi < ZERO:
            c_hi = cf[I + di, J + dj]

        adv = adv - (u_hi * c_hi - u_lo * c_lo) / ds

    # --- diffusion ---
    dif = ZERO
    if kappa > ZERO:
        lap = ZERO
        if wp.max(h[I - 1, J], ZERO) > h_wet:
            lap = lap + (cf[I - 1, J] - c0) / (dx * dx)
        if wp.max(h[I + 1, J], ZERO) > h_wet:
            lap = lap + (cf[I + 1, J] - c0) / (dx * dx)
        if wp.max(h[I, J - 1], ZERO) > h_wet:
            lap = lap + (cf[I, J - 1] - c0) / (dy * dy)
        if wp.max(h[I, J + 1], ZERO) > h_wet:
            lap = lap + (cf[I, J + 1] - c0) / (dy * dy)
        dif = kappa * lap

    # --- production, then decay treated implicitly so it cannot overshoot ---
    c = c0 + dt * (adv + dif + alpha * P_break[I, J])
    if tau > ZERO:
        c = c / (ONE + dt / tau)
    if c < ZERO:
        c = ZERO
    if cf_max > ZERO and c > cf_max:
        c = cf_max
    cf_new[I, J] = c
