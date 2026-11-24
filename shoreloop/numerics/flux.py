"""Riemann solver and reconstruction primitives (Warp device functions).

Scheme: MUSCL reconstruction of (eta, u, v, b) with a minmod limiter, Audusse
hydrostatic reconstruction at faces, HLLC-style approximate Riemann solver with
Einfeldt wave speeds and explicit dry-state branches.

The combination is exactly well-balanced for the lake-at-rest state
    u = 0,  eta = h + b = const
including partially dry cells -- see numerics.update.reconstruct for the
shoreline treatment and tests/hydro/test_lake_at_rest.py for the gate.

Literals are wrapped in FLOAT(...) or use typed constants: see
shoreloop.precision for why that is mandatory rather than stylistic.
"""

from __future__ import annotations

import warp as wp

from shoreloop.precision import FLOAT, HALF, ONE, SQRT2, TWO, ZERO, vec2, vec3, vec4


@wp.func
def minmod(a: FLOAT, b: FLOAT) -> FLOAT:
    r = ZERO
    if a * b > ZERO:
        if wp.abs(a) < wp.abs(b):
            r = a
        else:
            r = b
    return r


@wp.func
def recon(am: FLOAT, a0: FLOAT, ap: FLOAT) -> vec2:
    """Limited piecewise-linear reconstruction of a cell average.

    Returns (value at the cell's low face, value at the cell's high face).
    """
    s = HALF * minmod(a0 - am, ap - a0)
    return vec2(a0 - s, a0 + s)


@wp.func
def desing_u(h: FLOAT, q: FLOAT, eps: FLOAT) -> FLOAT:
    """Kurganov-Petrova desingularised velocity.

    Exactly q/h for h >= eps, smoothly -> 0 as h -> 0, always bounded. Without
    this the shoreline produces unbounded velocities in thin films.
    """
    h2 = h * h
    h4 = h2 * h2
    e2 = eps * eps
    e4 = e2 * e2
    return SQRT2 * h * q / wp.sqrt(h4 + wp.max(h4, e4))


@wp.func
def hllc(
    hL: FLOAT, unL: FLOAT, utL: FLOAT,
    hR: FLOAT, unR: FLOAT, utR: FLOAT,
    g: FLOAT, h_dry: FLOAT,
) -> vec3:
    """Approximate Riemann flux for the 2D SWE in a face-normal frame.

    Returns (F_h, F_qn, F_qt). The tangential component is upwinded by the sign
    of the mass flux (the HLLC contact treatment), which avoids smearing
    along-shore momentum across a shock.
    """
    F = vec3(ZERO, ZERO, ZERO)

    if hL > h_dry or hR > h_dry:
        hLc = wp.max(hL, ZERO)
        hRc = wp.max(hR, ZERO)
        cL = wp.sqrt(g * hLc)
        cR = wp.sqrt(g * hRc)

        sL = ZERO
        sR = ZERO
        if hL <= h_dry:
            # Dry left: exact dry-front speeds for the right-going rarefaction.
            sL = unR - TWO * cR
            sR = unR + cR
        elif hR <= h_dry:
            sL = unL - cL
            sR = unL + TWO * cL
        else:
            # Einfeldt (HLLE) speeds: sharp in smooth flow, positivity-safe at
            # CFL <= 0.5 when combined with hydrostatic reconstruction.
            sqL = wp.sqrt(hLc)
            sqR = wp.sqrt(hRc)
            u_roe = (sqL * unL + sqR * unR) / (sqL + sqR)
            c_roe = wp.sqrt(HALF * g * (hLc + hRc))
            sL = wp.min(unL - cL, u_roe - c_roe)
            sR = wp.max(unR + cR, u_roe + c_roe)

        FL0 = hLc * unL
        FL1 = hLc * unL * unL + HALF * g * hLc * hLc
        FR0 = hRc * unR
        FR1 = hRc * unR * unR + HALF * g * hRc * hRc

        f0 = ZERO
        f1 = ZERO
        if sL >= ZERO:
            f0 = FL0
            f1 = FL1
        elif sR <= ZERO:
            f0 = FR0
            f1 = FR1
        else:
            inv = ONE / (sR - sL)
            f0 = (sR * FL0 - sL * FR0 + sL * sR * (hRc - hLc)) * inv
            f1 = (sR * FL1 - sL * FR1 + sL * sR * (hRc * unR - hLc * unL)) * inv

        ut = ZERO
        if f0 >= ZERO:
            ut = utL
        else:
            ut = utR
        F = vec3(f0, f1, f0 * ut)

    return F


@wp.func
def face_flux(
    etaL: FLOAT, unL: FLOAT, utL: FLOAT, bL: FLOAT,
    etaR: FLOAT, unR: FLOAT, utR: FLOAT, bR: FLOAT,
    g: FLOAT, h_dry: FLOAT,
) -> vec4:
    """Audusse hydrostatic reconstruction + Riemann solve at one face.

    Returns (F_h, F_qn, F_qt, b_star); b_star is needed by the divergence kernel
    to build each cell's own pressure correction.
    """
    bs = wp.max(bL, bR)
    hsL = wp.max(ZERO, etaL - bs)
    hsR = wp.max(ZERO, etaR - bs)
    F = hllc(hsL, unL, utL, hsR, unR, utR, g, h_dry)
    return vec4(F[0], F[1], F[2], bs)
