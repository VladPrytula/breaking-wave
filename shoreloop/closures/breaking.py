"""Breaking closure: detect breaking, and *measure* the energy it dissipates.

The design departs from `plan.md` section 4 in one important way, for a reason
worth stating.

`plan.md` proposes adding dissipative momentum transport where a breaking
indicator fires. That is the right move for a Boussinesq or non-hydrostatic model
whose numerics are not shock-capturing. Ours is: a well-balanced finite-volume
scheme with an approximate Riemann solver dissipates energy at a bore
*automatically*, through the entropy production of the discrete shock. Adding a
second dissipation mechanism on top would double-count. (This is why Celeris
reports needing no empirical breaking calibration.)

So the closure here does not inject dissipation. It measures what the solver
already dissipates, which is both cleaner and exactly what the plan's causal
chain asks for:

    resolved wave energy -> breaking -> MEASURED dissipated energy -> foam source

Mechanical energy density and its flux, per unit area and divided by rho:

    E   = h|u|^2/2 + g h (h/2 + b)
    F_E = u ( h|u|^2/2 + g h (h + b) )

For smooth solutions `dE/dt + div F_E = 0`. Where the scheme captures a bore it
becomes negative, and the deficit is the dissipation:

    D = -( dE/dt + div F_E )  >= 0

`D` also contains bed friction, which we know analytically and subtract:

    P_fric = g n^2 |u|^3 / h^(1/3)

What remains, gated by the breaking indicator, is the breaking power:

    P_break = B * max(0, D - P_fric)  >= 0  by construction

Caveats, stated rather than hidden. If eddy viscosity is enabled its dissipation
is lumped into `P_break`, so the breaking gates run at nu = 0. And `D` is a
discrete residual, so it carries truncation error and is not pointwise
non-negative; the negative part is clamped and reported, not silently absorbed.
"""

from __future__ import annotations

import warp as wp

from shoreloop.numerics.flux import desing_u
from shoreloop.precision import FLOAT, HALF, ONE, ZERO


@wp.kernel
def energy_density(
    h: wp.array2d(dtype=FLOAT), qx: wp.array2d(dtype=FLOAT),
    qy: wp.array2d(dtype=FLOAT), b: wp.array2d(dtype=FLOAT),
    E: wp.array2d(dtype=FLOAT),
    g: FLOAT, eps: FLOAT,
):
    i, j = wp.tid()
    hv = wp.max(h[i, j], ZERO)
    if hv <= ZERO:
        E[i, j] = ZERO
        return
    u = desing_u(hv, qx[i, j], eps)
    v = desing_u(hv, qy[i, j], eps)
    E[i, j] = HALF * hv * (u * u + v * v) + g * hv * (HALF * hv + b[i, j])


@wp.kernel
def energy_flux(
    h: wp.array2d(dtype=FLOAT), qx: wp.array2d(dtype=FLOAT),
    qy: wp.array2d(dtype=FLOAT), b: wp.array2d(dtype=FLOAT),
    FEx: wp.array2d(dtype=FLOAT), FEy: wp.array2d(dtype=FLOAT),
    g: FLOAT, eps: FLOAT, ng: int,
):
    """Energy flux at the *low* face of each cell, upwinded by the face velocity."""
    i, j = wp.tid()
    I = i + ng
    J = j + ng

    for axis in range(2):
        di = 0
        dj = 0
        if axis == 0:
            di = 1
        else:
            dj = 1
        im = I - di
        jm = J - dj

        hA = wp.max(h[im, jm], ZERO)
        hB = wp.max(h[I, J], ZERO)
        if axis == 0:
            unA = desing_u(hA, qx[im, jm], eps)
            unB = desing_u(hB, qx[I, J], eps)
        else:
            unA = desing_u(hA, qy[im, jm], eps)
            unB = desing_u(hB, qy[I, J], eps)
        un = HALF * (unA + unB)

        # upwind the transported energy quantity
        hu = hA
        bu = b[im, jm]
        qxu = qx[im, jm]
        qyu = qy[im, jm]
        if un < ZERO:
            hu = hB
            bu = b[I, J]
            qxu = qx[I, J]
            qyu = qy[I, J]

        val = ZERO
        if hu > ZERO:
            uu = desing_u(hu, qxu, eps)
            vv = desing_u(hu, qyu, eps)
            val = un * (HALF * hu * (uu * uu + vv * vv) + g * hu * (hu + bu))
        if axis == 0:
            FEx[I, J] = val
        else:
            FEy[I, J] = val


@wp.kernel
def measure_dissipation(
    E_old: wp.array2d(dtype=FLOAT), E_new: wp.array2d(dtype=FLOAT),
    FEx: wp.array2d(dtype=FLOAT), FEy: wp.array2d(dtype=FLOAT),
    FEx2: wp.array2d(dtype=FLOAT), FEy2: wp.array2d(dtype=FLOAT),
    h: wp.array2d(dtype=FLOAT), qx: wp.array2d(dtype=FLOAT),
    qy: wp.array2d(dtype=FLOAT),
    D: wp.array2d(dtype=FLOAT), P_fric: wp.array2d(dtype=FLOAT),
    acc: wp.array(dtype=wp.float64),
    dt: FLOAT, dx: FLOAT, dy: FLOAT, g: FLOAT, n_manning: FLOAT,
    h_wet: FLOAT, eps: FLOAT, ng: int, cell_area: FLOAT,
):
    """D = -(dE/dt + div F_E) - P_fric, clamped at zero with the deficit logged."""
    i, j = wp.tid()
    I = i + ng
    J = j + ng

    hv = wp.max(h[I, J], ZERO)
    if hv <= h_wet:
        D[I, J] = ZERO
        P_fric[I, J] = ZERO
        return

    dEdt = (E_new[I, J] - E_old[I, J]) / dt
    # Trapezoidal flux: average of the pre- and post-step energy fluxes. Using
    # only the pre-step flux is forward-Euler consistent, i.e. O(dt), while the
    # state update is SSP-RK2, i.e. O(dt^2). That mismatch injects a spurious
    # oscillating residual at every bore, which then has to be clamped away and
    # inflates the apparent dissipation.
    div = HALF * ((FEx[I + 1, J] - FEx[I, J]) + (FEx2[I + 1, J] - FEx2[I, J])) / dx \
        + HALF * ((FEy[I, J + 1] - FEy[I, J]) + (FEy2[I, J + 1] - FEy2[I, J])) / dy
    total = -(dEdt + div)

    u = desing_u(hv, qx[I, J], eps)
    v = desing_u(hv, qy[I, J], eps)
    speed = wp.sqrt(u * u + v * v)
    pf = g * n_manning * n_manning * speed * speed * speed / wp.pow(hv, ONE / FLOAT(3.0))
    P_fric[I, J] = pf

    # Signed residual. Clamping happens in smooth_and_clamp, after spatial
    # averaging: see there for why the order matters.
    wp.atomic_add(acc, 2, wp.float64(pf) * wp.float64(cell_area) * wp.float64(dt))
    D[I, J] = total - pf


@wp.kernel
def breaking_indicator(
    h: wp.array2d(dtype=FLOAT), b: wp.array2d(dtype=FLOAT),
    eta_old: wp.array2d(dtype=FLOAT), Bfield: wp.array2d(dtype=FLOAT),
    dt: FLOAT, g: FLOAT, gamma: FLOAT, smooth: FLOAT,
    h_wet: FLOAT, ng: int,
):
    """B = 1 where the surface rises faster than gamma * sqrt(g h).

    The standard front-steepness criterion (Kennedy / Zelt / Schaeffer):
    `d(eta)/dt >= gamma sqrt(g h)` with gamma ~ 0.35-0.65. It fires on the
    steep, rapidly rising face of a bore and not on a passing swell of the same
    height, which a slope-only or Froude-only test cannot distinguish.

    `smooth > 0` replaces the step with a logistic of that width, giving the
    differentiable mode of the same closure. The hard and smooth forms are not
    claimed to be the same operator -- their difference is a measurable function
    of `smooth`, which is the point (plan.md section 11).
    """
    i, j = wp.tid()
    I = i + ng
    J = j + ng
    hv = wp.max(h[I, J], ZERO)
    if hv <= h_wet:
        Bfield[I, J] = ZERO
        return
    detadt = (hv + b[I, J] - eta_old[I, J]) / dt
    thresh = gamma * wp.sqrt(g * hv)
    if smooth > ZERO:
        Bfield[I, J] = ONE / (ONE + wp.exp(-(detadt - thresh) / (smooth * thresh + FLOAT(1.0e-9))))
    else:
        if detadt >= thresh:
            Bfield[I, J] = ONE
        else:
            Bfield[I, J] = ZERO


@wp.kernel
def gate_breaking_power(
    D: wp.array2d(dtype=FLOAT), Bfield: wp.array2d(dtype=FLOAT),
    P_break: wp.array2d(dtype=FLOAT), ng: int,
):
    i, j = wp.tid()
    I = i + ng
    J = j + ng
    P_break[I, J] = Bfield[I, J] * D[I, J]


@wp.kernel
def store_eta(
    h: wp.array2d(dtype=FLOAT), b: wp.array2d(dtype=FLOAT),
    eta: wp.array2d(dtype=FLOAT),
):
    i, j = wp.tid()
    eta[i, j] = wp.max(h[i, j], ZERO) + b[i, j]


@wp.kernel
def wet_stencil_count(
    h: wp.array2d(dtype=FLOAT), n: wp.array2d(dtype=FLOAT),
    h_wet: FLOAT, ng: int, nx: int, ny: int,
):
    """Wet, *interior* cells in each cell's 5-point stencil, including itself.

    This is the scatter weight that makes the smoothing filter conservative:
    each cell hands out 1/n of its value, so the domain total is unchanged no
    matter how the stencil is clipped.

    Two clippings matter and both must be counted the same way. A shoreline,
    obviously. But also the *domain boundary*: `D_raw` is only computed on
    interior cells, so a cell adjacent to the halo must not donate into it --
    otherwise that share is silently lost, which leaked ~10% of the energy
    budget at the walls before this was restricted.
    """
    i, j = wp.tid()
    lo_i = ng
    hi_i = ng + nx - 1
    lo_j = ng
    hi_j = ng + ny - 1
    if i < lo_i or i > hi_i or j < lo_j or j > hi_j:
        n[i, j] = ZERO
        return
    if wp.max(h[i, j], ZERO) <= h_wet:
        n[i, j] = ZERO
        return
    c = ONE
    if i - 1 >= lo_i and wp.max(h[i - 1, j], ZERO) > h_wet:
        c = c + ONE
    if i + 1 <= hi_i and wp.max(h[i + 1, j], ZERO) > h_wet:
        c = c + ONE
    if j - 1 >= lo_j and wp.max(h[i, j - 1], ZERO) > h_wet:
        c = c + ONE
    if j + 1 <= hi_j and wp.max(h[i, j + 1], ZERO) > h_wet:
        c = c + ONE
    n[i, j] = c


@wp.kernel
def smooth_and_clamp(
    D_raw: wp.array2d(dtype=FLOAT), h: wp.array2d(dtype=FLOAT),
    wcount: wp.array2d(dtype=FLOAT),
    D: wp.array2d(dtype=FLOAT), acc: wp.array(dtype=wp.float64),
    dt: FLOAT, h_wet: FLOAT, ng: int, cell_area: FLOAT,
):
    """Spatially average the signed residual, then clamp at zero.

    Order matters. The signed residual closes the mechanical-energy budget
    *exactly* -- verified to roundoff on a closed-domain wet-bed dam break -- but
    locally it oscillates in sign across the captured bore, because a discrete
    shock is spread over a few cells and the scheme's internal energy exchange is
    not monotone within it. Clamping pointwise therefore throws away the negative
    lobes and over-counts dissipation.

    Averaging first lets adjacent +/- pairs cancel, which is also the physically
    right thing: dissipation at a bore is distributed over the bore thickness,
    not concentrated at the one cell where the residual happens to peak. The
    remaining over-count is reported via the deficit accumulator and shrinks with
    resolution.
    """
    i, j = wp.tid()
    I = i + ng
    J = j + ng
    if wp.max(h[I, J], ZERO) <= h_wet:
        D[I, J] = ZERO
        return

    # Gather form of a conservative scatter: each contributing cell donates
    # 1/wcount of its value, so sum(D_smooth) == sum(D_raw) exactly, even where
    # the stencil is clipped by a shoreline. A plain neighbour average is NOT
    # conservative and leaks ~10% of the energy budget at the boundaries.
    d = D_raw[I, J] / wcount[I, J]
    for k in range(4):
        di = 0
        dj = 0
        if k == 0:
            di = -1
        elif k == 1:
            di = 1
        elif k == 2:
            dj = -1
        else:
            dj = 1
        wn = wcount[I + di, J + dj]
        if wn > ZERO:
            d = d + D_raw[I + di, J + dj] / wn

    w = wp.float64(cell_area) * wp.float64(dt)
    if d < ZERO:
        wp.atomic_add(acc, 1, wp.float64(-d) * w)
        d = ZERO
    else:
        wp.atomic_add(acc, 0, wp.float64(d) * w)
    D[I, J] = d
