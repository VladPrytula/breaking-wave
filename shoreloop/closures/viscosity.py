"""Horizontal eddy viscosity -- a closure, not resolved physics.

Molecular viscosity of water is ~1e-6 m^2/s, which at metre scales and O(1 m/s)
velocities is utterly negligible (Re ~ 1e6). Any viscosity worth having here is
therefore an *eddy* viscosity: a stand-in for horizontal momentum mixing by
turbulence the depth-integrated equations cannot resolve. Labelling it as such
matters, because it means the coefficient is a closure parameter to be
identified, not a material constant to be looked up.

Added to the momentum equation as

    d(hu)/dt += div( nu h grad u )

with `nu` either a constant or a Smagorinsky-type function of the local strain
rate. Discretised with face-centred coefficients and central differences, so it
is a symmetric negative-semidefinite operator: for nu >= 0 it can only remove
kinetic energy, never inject it. That is required by the project contract
(docs/decisions.md) and is checked by an energy gate.

Two reasons this knob earns its place beyond looking nicer:

1. It is the cleanest control over how chaotic the surf zone is. Raising it
   damps the small-scale shoreline instabilities that make a run-up front look
   noisy rather than smooth.
2. It adds dissipation, which contracts the monodromy operator of the return
   map and therefore makes the periodic orbit *easier* to reach. When
   `eps_loop` plateaus, this is one of the first things to reach for.
"""

from __future__ import annotations

import warp as wp

from shoreloop.precision import FLOAT, FOUR, HALF, ONE, TWO, ZERO

from shoreloop.numerics.flux import desing_u


@wp.func
def smagorinsky_nu(
    dudx: FLOAT, dudy: FLOAT, dvdx: FLOAT, dvdy: FLOAT,
    delta: FLOAT, nu_const: FLOAT, cs: FLOAT,
) -> FLOAT:
    """nu = nu_const + (cs*delta)^2 * |S|,  |S| = sqrt(2 S_ij S_ij)."""
    nu = nu_const
    if cs > ZERO:
        sxy = HALF * (dudy + dvdx)
        mag = wp.sqrt(TWO * dudx * dudx + TWO * dvdy * dvdy + FOUR * sxy * sxy)
        nu = nu + (cs * delta) * (cs * delta) * mag
    return nu


@wp.func
def _vel(h: wp.array2d(dtype=FLOAT), q: wp.array2d(dtype=FLOAT),
         i: int, j: int, eps: FLOAT) -> FLOAT:
    return desing_u(wp.max(h[i, j], ZERO), q[i, j], eps)


@wp.kernel
def compute_nu(
    h: wp.array2d(dtype=FLOAT),
    qx: wp.array2d(dtype=FLOAT),
    qy: wp.array2d(dtype=FLOAT),
    nu: wp.array2d(dtype=FLOAT),
    dx: FLOAT, dy: FLOAT,
    nu_const: FLOAT, cs: FLOAT, eps: FLOAT,
):
    """Cell-centred eddy viscosity over the padded grid.

    Precomputed rather than evaluated inline so that the momentum kernel can
    build a face coefficient that is a *symmetric* function of the two adjacent
    cells. With a one-sided nu (each cell applying its own value to the shared
    face) the Smagorinsky operator is asymmetric: the two cells exchange
    unequal momentum, and kinetic energy can locally be injected -- which a
    closure is contractually forbidden to do.
    """
    i, j = wp.tid()
    n0 = h.shape[0]
    n1 = h.shape[1]
    if i < 1 or i >= n0 - 1 or j < 1 or j >= n1 - 1:
        nu[i, j] = nu_const
        return
    uE = _vel(h, qx, i + 1, j, eps)
    uW = _vel(h, qx, i - 1, j, eps)
    uN = _vel(h, qx, i, j + 1, eps)
    uS = _vel(h, qx, i, j - 1, eps)
    vE = _vel(h, qy, i + 1, j, eps)
    vW = _vel(h, qy, i - 1, j, eps)
    vN = _vel(h, qy, i, j + 1, eps)
    vS = _vel(h, qy, i, j - 1, eps)
    nu[i, j] = smagorinsky_nu(
        (uE - uW) / (TWO * dx), (uN - uS) / (TWO * dy),
        (vE - vW) / (TWO * dx), (vN - vS) / (TWO * dy),
        wp.sqrt(dx * dy), nu_const, cs,
    )


@wp.kernel
def add_viscous_momentum(
    h: wp.array2d(dtype=FLOAT),
    qx: wp.array2d(dtype=FLOAT),
    qy: wp.array2d(dtype=FLOAT),
    nu: wp.array2d(dtype=FLOAT),
    rqx: wp.array2d(dtype=FLOAT),
    rqy: wp.array2d(dtype=FLOAT),
    ng: int, dx: FLOAT, dy: FLOAT,
    h_wet: FLOAT, eps: FLOAT,
):
    i, j = wp.tid()
    I = i + ng
    J = j + ng

    if h[I, J] <= h_wet:
        return

    u0 = _vel(h, qx, I, J, eps)
    v0 = _vel(h, qy, I, J, eps)
    uE = _vel(h, qx, I + 1, J, eps)
    uW = _vel(h, qx, I - 1, J, eps)
    uN = _vel(h, qx, I, J + 1, eps)
    uS = _vel(h, qx, I, J - 1, eps)
    vE = _vel(h, qy, I + 1, J, eps)
    vW = _vel(h, qy, I - 1, J, eps)
    vN = _vel(h, qy, I, J + 1, eps)
    vS = _vel(h, qy, I, J - 1, eps)

    # Face coefficients, zeroed at a wet/dry face: momentum must not diffuse
    # into dry land, which would drive flow up a dry beach.
    #
    # Both factors are min() of the two adjacent cells, and both for the same
    # reason: the face must see one coefficient, whichever side looks at it.
    #
    # The depth factor is min(h_i, h_j), NOT (h_i+h_j)/2. This matters for
    # stability, not accuracy. We add to *momentum*, so the implied diffusivity
    # for cell i's velocity is c_face/h_i. With the arithmetic mean, a 0.4 mm
    # film beside a 0.3 m cell gets c/h_i ~ nu*0.15/0.0004 = 375*nu, violating
    # the explicit diffusion limit by that factor and blowing up the shoreline.
    # With min(), c_face/h_i <= nu always, so the stated CFL bound is valid.
    #
    # The nu factor is min(nu_i, nu_j) so the Smagorinsky operator is symmetric
    # too. Symmetric non-negative face coefficients make the whole operator
    # negative semidefinite: it can only remove kinetic energy, never inject
    # it, which is the contract (and gate) for this closure.
    h0 = h[I, J]
    nu0 = nu[I, J]
    cE = ZERO
    cW = ZERO
    cN = ZERO
    cS = ZERO
    if h[I + 1, J] > h_wet:
        cE = wp.min(nu0, nu[I + 1, J]) * wp.min(h0, h[I + 1, J])
    if h[I - 1, J] > h_wet:
        cW = wp.min(nu0, nu[I - 1, J]) * wp.min(h0, h[I - 1, J])
    if h[I, J + 1] > h_wet:
        cN = wp.min(nu0, nu[I, J + 1]) * wp.min(h0, h[I, J + 1])
    if h[I, J - 1] > h_wet:
        cS = wp.min(nu0, nu[I, J - 1]) * wp.min(h0, h[I, J - 1])

    inv_dx2 = ONE / (dx * dx)
    inv_dy2 = ONE / (dy * dy)

    lap_u = (cE * (uE - u0) + cW * (uW - u0)) * inv_dx2 \
          + (cN * (uN - u0) + cS * (uS - u0)) * inv_dy2
    lap_v = (cE * (vE - v0) + cW * (vW - v0)) * inv_dx2 \
          + (cN * (vN - v0) + cS * (vS - v0)) * inv_dy2

    rqx[I, J] += lap_u
    rqy[I, J] += lap_v


@wp.kernel
def max_viscous_rate(
    h: wp.array2d(dtype=FLOAT), nu: wp.array2d(dtype=FLOAT),
    out: wp.array(dtype=FLOAT),
    ng: int, dx: FLOAT, dy: FLOAT, h_wet: FLOAT,
):
    """max over cells of 2*nu*(1/dx^2 + 1/dy^2), the explicit diffusion limit.

    The cell's own nu bounds all four of its face coefficients (each face uses
    min(nu_i, nu_j) <= nu_i), so this is a valid upper bound on the rate.
    """
    i, j = wp.tid()
    I = i + ng
    J = j + ng
    if h[I, J] <= h_wet:
        return
    wp.atomic_max(out, 0, TWO * nu[I, J] * (ONE / (dx * dx) + ONE / (dy * dy)))
