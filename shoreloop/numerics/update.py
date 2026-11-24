"""Warp kernels for the shallow-water update.

Staged as reconstruct -> face flux -> divergence, once per direction. The stages
are separate kernels rather than one fused kernel for three reasons: each stage
is independently testable, the intermediates are exactly the tensors reverse-mode
AD wants to see, and on GB10 (~273 GB/s, bandwidth-bound) the extra traffic is
cheap next to the correctness risk of a hand-fused 5x5 stencil.
"""

from __future__ import annotations

import warp as wp

from shoreloop.precision import FLOAT, FOUR_THIRDS, HALF, ONE, TWO, ZERO

from shoreloop.numerics.flux import desing_u, face_flux, minmod


@wp.kernel
def reconstruct(
    h: wp.array2d(dtype=FLOAT),
    qx: wp.array2d(dtype=FLOAT),
    qy: wp.array2d(dtype=FLOAT),
    b: wp.array2d(dtype=FLOAT),
    etaL: wp.array2d(dtype=FLOAT),
    etaR: wp.array2d(dtype=FLOAT),
    unL: wp.array2d(dtype=FLOAT),
    unR: wp.array2d(dtype=FLOAT),
    utL: wp.array2d(dtype=FLOAT),
    utR: wp.array2d(dtype=FLOAT),
    bL: wp.array2d(dtype=FLOAT),
    bR: wp.array2d(dtype=FLOAT),
    axis: int,
    eps: FLOAT,
    h_wet: FLOAT,
):
    """MUSCL reconstruction along `axis` (0 = x, 1 = y).

    Reconstructs the *free surface* eta = h + b rather than h, which is what
    makes the lake-at-rest state reconstruct exactly over wet cells.

    Two shoreline treatments, both load-bearing.

    The problem: lake-at-rest is `eta = const` only where the cell is *wet*. In
    a dry cell h = 0, so eta = b, which sits *above* the water level. Naively
    limiting eta across a wet/dry face reconstructs a spurious surface slope,
    and hydrostatic reconstruction turns that into positive depth on dry land
    -- inventing water, breaking positivity, and driving flow in a lake that
    should be still.

    1. **Dry cells reconstruct zero depth.** Where `h <= h_wet` we force the
       eta half-slope to equal the bed half-slope, so `eta_face - b_face = 0`
       exactly at both faces, and zero the velocity slopes.

    2. **Wet cells clamp the bed to the surface**: `b_face <- min(b_face,
       eta_face)`. This is Liang & Marche's local bed modification. It makes
       the reconstructed depth non-negative by construction *without* touching
       the eta, u or v slopes, so wet cells keep full second-order accuracy
       right up to the shoreline.

    Both preserve lake-at-rest exactly. For the clamped case the algebra is:
    with `eta = eta0` constant, `h_hi = eta0 - b_R` and `h_lo = eta0 - b_L`
    (clamped), the two hydrostatically reconstructed depths at a face are equal,
    so the Riemann flux is the pressure term alone; the cell's own pressure
    correction then leaves `-(g/2)(h_hi^2 - h_lo^2)/ds`, and since
    `h_hi^2 - h_lo^2 = (b_L - b_R)(h_hi + h_lo)`, that cancels the centred
    bed-slope source `-g (h_hi + h_lo)/2 (b_R - b_L)/ds` identically.

    The approximation being made is explicit: in a *partially* dry cell the
    solver sees a bed clipped to the water surface rather than the true bed.
    That is the price of exact balance there, and it is far cheaper than the
    three-cell first-order band it replaces.
    """
    i, j = wp.tid()

    di = 0
    dj = 0
    if axis == 0:
        di = 1
    else:
        dj = 1

    n0 = h.shape[0]
    n1 = h.shape[1]
    # One-cell guard: cells on the outermost halo ring get no slope.
    if i - di < 0 or i + di >= n0 or j - dj < 0 or j + dj >= n1:
        e0 = h[i, j] + b[i, j]
        etaL[i, j] = e0
        etaR[i, j] = e0
        unL[i, j] = ZERO
        unR[i, j] = ZERO
        utL[i, j] = ZERO
        utR[i, j] = ZERO
        bL[i, j] = b[i, j]
        bR[i, j] = b[i, j]
        return

    im = i - di
    jm = j - dj
    ip = i + di
    jp = j + dj

    e_m = h[im, jm] + b[im, jm]
    e_0 = h[i, j] + b[i, j]
    e_p = h[ip, jp] + b[ip, jp]

    if axis == 0:
        qn_m = qx[im, jm]
        qn_0 = qx[i, j]
        qn_p = qx[ip, jp]
        qt_m = qy[im, jm]
        qt_0 = qy[i, j]
        qt_p = qy[ip, jp]
    else:
        qn_m = qy[im, jm]
        qn_0 = qy[i, j]
        qn_p = qy[ip, jp]
        qt_m = qx[im, jm]
        qt_0 = qx[i, j]
        qt_p = qx[ip, jp]

    un_m = desing_u(h[im, jm], qn_m, eps)
    un_0 = desing_u(h[i, j], qn_0, eps)
    un_p = desing_u(h[ip, jp], qn_p, eps)
    ut_m = desing_u(h[im, jm], qt_m, eps)
    ut_0 = desing_u(h[i, j], qt_0, eps)
    ut_p = desing_u(h[ip, jp], qt_p, eps)

    se = HALF * minmod(e_0 - e_m, e_p - e_0)
    sn = HALF * minmod(un_0 - un_m, un_p - un_0)
    st = HALF * minmod(ut_0 - ut_m, ut_p - ut_0)
    sb = HALF * minmod(b[i, j] - b[im, jm], b[ip, jp] - b[i, j])

    if h[i, j] <= h_wet:
        # Dry cell: zero *every* slope, the bed's included.
        #
        # Zeroing only the eta slope (se = sb) is not enough, and the failure is
        # instructive. A minmod-limited bed slope can place a dry cell's
        # reconstructed face bed *below* the still-water level even though its
        # cell-centre bed is above it -- e.g. b = 0.463 with b_face = 0.359
        # against eta0 = 0.400. Then b_star = max(b_face_wet, b_face_dry) sits
        # below eta0, the face sees positive depth on the wet side against dry
        # on the other, and the solver correctly computes an outflow flux
        # through what is actually solid rock. Water climbs the rock and the
        # lake starts to slosh.
        #
        # With sb = 0 the face bed is the cell average, which is above the
        # surface by definition of the cell being dry, so b_star >= b_cell,
        # both reconstructed depths clamp to zero, and the flux vanishes.
        se = ZERO
        sn = ZERO
        st = ZERO
        sb = ZERO

    eL = e_0 - se
    eR = e_0 + se
    bLv = b[i, j] - sb
    bRv = b[i, j] + sb

    # Local bed modification: never let the reconstructed bed rise above the
    # reconstructed surface. Keeps h_recon >= 0 without limiting eta/u/v.
    bLv = wp.min(bLv, eL)
    bRv = wp.min(bRv, eR)

    etaL[i, j] = eL
    etaR[i, j] = eR
    unL[i, j] = un_0 - sn
    unR[i, j] = un_0 + sn
    utL[i, j] = ut_0 - st
    utR[i, j] = ut_0 + st
    bL[i, j] = bLv
    bR[i, j] = bRv


@wp.kernel
def compute_faces(
    etaL: wp.array2d(dtype=FLOAT),
    etaR: wp.array2d(dtype=FLOAT),
    unL: wp.array2d(dtype=FLOAT),
    unR: wp.array2d(dtype=FLOAT),
    utL: wp.array2d(dtype=FLOAT),
    utR: wp.array2d(dtype=FLOAT),
    bL: wp.array2d(dtype=FLOAT),
    bR: wp.array2d(dtype=FLOAT),
    Fh: wp.array2d(dtype=FLOAT),
    Fn: wp.array2d(dtype=FLOAT),
    Ft: wp.array2d(dtype=FLOAT),
    Bs: wp.array2d(dtype=FLOAT),
    axis: int,
    ng: int,
    g: FLOAT,
    h_dry: FLOAT,
):
    """Flux at the *low* face of cell (I, J), i.e. face I-1/2 along `axis`."""
    i, j = wp.tid()
    I = i + ng
    J = j + ng

    di = 0
    dj = 0
    if axis == 0:
        di = 1
    else:
        dj = 1

    im = I - di
    jm = J - dj

    F = face_flux(
        etaR[im, jm], unR[im, jm], utR[im, jm], bR[im, jm],
        etaL[I, J], unL[I, J], utL[I, J], bL[I, J],
        g, h_dry,
    )
    Fh[I, J] = F[0]
    Fn[I, J] = F[1]
    Ft[I, J] = F[2]
    Bs[I, J] = F[3]


@wp.kernel
def accumulate_divergence(
    etaL: wp.array2d(dtype=FLOAT),
    etaR: wp.array2d(dtype=FLOAT),
    bL: wp.array2d(dtype=FLOAT),
    bR: wp.array2d(dtype=FLOAT),
    Fh: wp.array2d(dtype=FLOAT),
    Fn: wp.array2d(dtype=FLOAT),
    Ft: wp.array2d(dtype=FLOAT),
    Bs: wp.array2d(dtype=FLOAT),
    rh: wp.array2d(dtype=FLOAT),
    rqx: wp.array2d(dtype=FLOAT),
    rqy: wp.array2d(dtype=FLOAT),
    axis: int,
    ng: int,
    ds: FLOAT,
    g: FLOAT,
):
    """Add the `axis` contribution of -div(F) + S_topo to the RHS.

    Each cell rebuilds its own Audusse pressure correction from its own
    reconstructed depth and the shared face b_star, plus the centred within-cell
    bed-slope source. Together these cancel exactly at rest.
    """
    i, j = wp.tid()
    I = i + ng
    J = j + ng

    di = 0
    dj = 0
    if axis == 0:
        di = 1
    else:
        dj = 1

    Ip = I + di
    Jp = J + dj

    # The cell's own reconstructed depths at its two faces.
    hlo = wp.max(ZERO, etaL[I, J] - bL[I, J])
    hhi = wp.max(ZERO, etaR[I, J] - bR[I, J])

    # Hydrostatically reconstructed depths this cell saw at each face.
    hs_hi = wp.max(ZERO, etaR[I, J] - Bs[Ip, Jp])
    hs_lo = wp.max(ZERO, etaL[I, J] - Bs[I, J])

    Fn_hi = Fn[Ip, Jp] + HALF * g * (hhi * hhi - hs_hi * hs_hi)
    Fn_lo = Fn[I, J] + HALF * g * (hlo * hlo - hs_lo * hs_lo)

    inv = ONE / ds
    dmass = -(Fh[Ip, Jp] - Fh[I, J]) * inv
    dnorm = -(Fn_hi - Fn_lo) * inv - g * HALF * (hhi + hlo) * (bR[I, J] - bL[I, J]) * inv
    dtang = -(Ft[Ip, Jp] - Ft[I, J]) * inv

    rh[I, J] += dmass
    if axis == 0:
        rqx[I, J] += dnorm
        rqy[I, J] += dtang
    else:
        rqy[I, J] += dnorm
        rqx[I, J] += dtang


@wp.kernel
def zero3(a: wp.array2d(dtype=FLOAT), b: wp.array2d(dtype=FLOAT), c: wp.array2d(dtype=FLOAT)):
    i, j = wp.tid()
    a[i, j] = ZERO
    b[i, j] = ZERO
    c[i, j] = ZERO


@wp.kernel
def euler_stage(
    h0: wp.array2d(dtype=FLOAT), qx0: wp.array2d(dtype=FLOAT), qy0: wp.array2d(dtype=FLOAT),
    rh: wp.array2d(dtype=FLOAT), rqx: wp.array2d(dtype=FLOAT), rqy: wp.array2d(dtype=FLOAT),
    h1: wp.array2d(dtype=FLOAT), qx1: wp.array2d(dtype=FLOAT), qy1: wp.array2d(dtype=FLOAT),
    dt: FLOAT, ng: int,
):
    i, j = wp.tid()
    I = i + ng
    J = j + ng
    h1[I, J] = h0[I, J] + dt * rh[I, J]
    qx1[I, J] = qx0[I, J] + dt * rqx[I, J]
    qy1[I, J] = qy0[I, J] + dt * rqy[I, J]


@wp.kernel
def ssp_rk2_combine(
    hn: wp.array2d(dtype=FLOAT), qxn: wp.array2d(dtype=FLOAT), qyn: wp.array2d(dtype=FLOAT),
    h1: wp.array2d(dtype=FLOAT), qx1: wp.array2d(dtype=FLOAT), qy1: wp.array2d(dtype=FLOAT),
    rh: wp.array2d(dtype=FLOAT), rqx: wp.array2d(dtype=FLOAT), rqy: wp.array2d(dtype=FLOAT),
    hout: wp.array2d(dtype=FLOAT), qxout: wp.array2d(dtype=FLOAT), qyout: wp.array2d(dtype=FLOAT),
    dt: FLOAT, ng: int,
):
    """U^{n+1} = 1/2 U^n + 1/2 (U^(1) + dt L(U^(1)))."""
    i, j = wp.tid()
    I = i + ng
    J = j + ng
    hout[I, J] = HALF * (hn[I, J] + h1[I, J] + dt * rh[I, J])
    qxout[I, J] = HALF * (qxn[I, J] + qx1[I, J] + dt * rqx[I, J])
    qyout[I, J] = HALF * (qyn[I, J] + qy1[I, J] + dt * rqy[I, J])


@wp.kernel
def clamp_depth(
    h: wp.array2d(dtype=FLOAT), qx: wp.array2d(dtype=FLOAT), qy: wp.array2d(dtype=FLOAT),
    clipped: wp.array(dtype=wp.float64),
    ng: int, cell_area: FLOAT,
):
    """Enforce h >= 0 and account for the mass that had to be invented.

    The scheme is positivity-preserving in theory; this catches roundoff and
    keeps the mass-conservation gate honest instead of silently absorbing error.
    """
    i, j = wp.tid()
    I = i + ng
    J = j + ng
    hv = h[I, J]
    if hv <= ZERO:
        wp.atomic_add(clipped, 0, wp.float64(-hv) * wp.float64(cell_area))
        h[I, J] = ZERO
        qx[I, J] = ZERO
        qy[I, J] = ZERO


@wp.kernel
def apply_manning_friction(
    h: wp.array2d(dtype=FLOAT), qx: wp.array2d(dtype=FLOAT), qy: wp.array2d(dtype=FLOAT),
    dt: FLOAT, g: FLOAT, n_manning: FLOAT, eps: FLOAT, h_wet: FLOAT, ng: int,
):
    """Semi-implicit Manning friction.

    S_f = -g n^2 |u| u / h^(4/3), integrated as q <- q / (1 + dt*k) with
    k = g n^2 |u| / h^(4/3). Unconditionally stable and cannot reverse the
    velocity, which an explicit update does in thin films.

    Applied down to h_wet (the reconstruction dryness threshold), not to the
    desingularisation scale: the semi-implicit form has the correct h -> 0
    limit (q -> 0), and skipping films thinner than desing_eps would leave a
    sub-millimetre swash tongue moving frictionlessly -- exactly where Manning
    friction should dominate -- while the breaking closure subtracts friction
    power there regardless.
    """
    i, j = wp.tid()
    I = i + ng
    J = j + ng
    hv = h[I, J]
    if hv > h_wet:
        u = desing_u(hv, qx[I, J], eps)
        v = desing_u(hv, qy[I, J], eps)
        speed = wp.sqrt(u * u + v * v)
        k = g * n_manning * n_manning * speed / wp.pow(hv, FOUR_THIRDS)
        f = ONE / (ONE + dt * k)
        qx[I, J] = qx[I, J] * f
        qy[I, J] = qy[I, J] * f
