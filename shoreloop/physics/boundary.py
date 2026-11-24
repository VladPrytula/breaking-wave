"""Ghost-cell boundary conditions.

Bathymetry ghosts are filled once at setup (b is static), so the per-step kernel
only touches (h, qx, qy).
"""

from __future__ import annotations

import warp as wp

from shoreloop.precision import FLOAT, HALF, ONE, QUARTER, TWO, TWO_PI, ZERO

from shoreloop.numerics.flux import desing_u


@wp.func
def _src_index(kind: int, ghost: int, lo: bool, ng: int, n: int) -> int:
    """Interior index that a ghost cell copies from.

    kind: 0 = WALL (mirror), 1 = PERIODIC (wrap), 2 = OPEN (nearest interior).
    ghost: 0 for the ring adjacent to the interior, 1 for the next one out.
    """
    idx = int(0)
    if lo:
        if kind == 0:
            idx = ng + ghost
        elif kind == 1:
            idx = ng + n - 1 - ghost
        else:
            idx = ng
    else:
        if kind == 0:
            idx = ng + n - 1 - ghost
        elif kind == 1:
            idx = ng + ghost
        else:
            idx = ng + n - 1
    return idx


@wp.kernel
def bc_x(
    h: wp.array2d(dtype=FLOAT), qx: wp.array2d(dtype=FLOAT), qy: wp.array2d(dtype=FLOAT),
    kind_lo: int, kind_hi: int, ng: int, nx: int,
):
    ghost, j = wp.tid()

    Ilo = ng - 1 - ghost
    Slo = _src_index(kind_lo, ghost, True, ng, nx)
    sgn_lo = ONE
    if kind_lo == 0:
        sgn_lo = -ONE
    h[Ilo, j] = h[Slo, j]
    qx[Ilo, j] = sgn_lo * qx[Slo, j]
    qy[Ilo, j] = qy[Slo, j]

    Ihi = ng + nx + ghost
    Shi = _src_index(kind_hi, ghost, False, ng, nx)
    sgn_hi = ONE
    if kind_hi == 0:
        sgn_hi = -ONE
    h[Ihi, j] = h[Shi, j]
    qx[Ihi, j] = sgn_hi * qx[Shi, j]
    qy[Ihi, j] = qy[Shi, j]


@wp.kernel
def bc_y(
    h: wp.array2d(dtype=FLOAT), qx: wp.array2d(dtype=FLOAT), qy: wp.array2d(dtype=FLOAT),
    kind_lo: int, kind_hi: int, ng: int, ny: int,
):
    i, ghost = wp.tid()

    Jlo = ng - 1 - ghost
    Slo = _src_index(kind_lo, ghost, True, ng, ny)
    sgn_lo = ONE
    if kind_lo == 0:
        sgn_lo = -ONE
    h[i, Jlo] = h[i, Slo]
    qy[i, Jlo] = sgn_lo * qy[i, Slo]
    qx[i, Jlo] = qx[i, Slo]

    Jhi = ng + ny + ghost
    Shi = _src_index(kind_hi, ghost, False, ng, ny)
    sgn_hi = ONE
    if kind_hi == 0:
        sgn_hi = -ONE
    h[i, Jhi] = h[i, Shi]
    qy[i, Jhi] = sgn_hi * qy[i, Shi]
    qx[i, Jhi] = qx[i, Shi]


@wp.kernel
def bc_scalar_x(f: wp.array2d(dtype=FLOAT), kind_lo: int, kind_hi: int, ng: int, nx: int):
    ghost, j = wp.tid()
    f[ng - 1 - ghost, j] = f[_src_index(kind_lo, ghost, True, ng, nx), j]
    f[ng + nx + ghost, j] = f[_src_index(kind_hi, ghost, False, ng, nx), j]


@wp.kernel
def bc_scalar_y(f: wp.array2d(dtype=FLOAT), kind_lo: int, kind_hi: int, ng: int, ny: int):
    i, ghost = wp.tid()
    f[i, ng - 1 - ghost] = f[i, _src_index(kind_lo, ghost, True, ng, ny)]
    f[i, ng + ny + ghost] = f[i, _src_index(kind_hi, ghost, False, ng, ny)]





@wp.kernel
def bc_wavemaker_x_lo(
    h: wp.array2d(dtype=FLOAT), qx: wp.array2d(dtype=FLOAT),
    qy: wp.array2d(dtype=FLOAT), b: wp.array2d(dtype=FLOAT),
    amp: wp.array(dtype=FLOAT), harm: wp.array(dtype=FLOAT),
    pha: wp.array(dtype=FLOAT), ky: wp.array(dtype=FLOAT), nmodes: int,
    t: FLOAT, period: FLOAT, eta0: FLOAT,
    g: FLOAT, ng: int, eps: FLOAT, h_dry: FLOAT,
    y0: FLOAT, dy: FLOAT,
):
    """Generating-absorbing offshore boundary, on Riemann invariants.

    At the low-x boundary the characteristic with speed `u + c` runs *into* the
    domain, so its invariant is prescribed from the target wave; the one with
    speed `u - c` runs *out*, so its invariant is taken from the interior:

        R+ = u_t + 2 c_t          (incoming, from the forcing)
        R- = u_i - 2 c_i          (outgoing, radiated from inside)
        =>  c_b = (R+ - R-) / 4,   u_b = (R+ + R-) / 2

    A prescribed-eta boundary would instead pin both, reflecting everything that
    reaches it. Reflected energy interferes with the forcing and builds standing
    modes at basin periods unrelated to T, which does not merely add error -- it
    can destroy the existence of a T-periodic orbit (docs/decisions.md
    Correction 1). This form lets outgoing waves leave: substituting a linear
    left-running wave into the algebra returns `h_b = h_ref + eta'` and
    `u_b = u'`, i.e. the outgoing wave is reproduced exactly to linear order and
    passes through.

    Oblique components: each mode carries an alongshore wavenumber k_y, so its
    phase is `omega t - k_y y + phi` and its linear velocity splits into
    (cos theta, sin theta) with `sin theta = k_y c / omega` from the shallow-
    water dispersion relation at the boundary depth. Only the NORMAL velocity
    enters R+; the tangential velocity rides the contact characteristic (speed
    u), so it is prescribed from the target where the boundary flow enters the
    domain and taken from the interior where it leaves. Obliquely *outgoing*
    waves see a boundary tuned for normal incidence, so absorption degrades
    with angle -- measured, not hidden, by the reflection gate.

    Setting all amplitudes to zero gives a pure absorber, which is how the
    reflection-coefficient gate is measured.
    """
    ghost, j = wp.tid()
    I = ng - 1 - ghost
    Ii = ng

    bb = b[Ii, j]
    h_ref = eta0 - bb

    if h_ref <= h_dry:
        # Boundary sits on dry land: nothing to force. Fall back to a wall.
        h[I, j] = h[Ii, j]
        qx[I, j] = -qx[Ii, j]
        qy[I, j] = qy[Ii, j]
        return

    c_ref = wp.sqrt(g * h_ref)
    uv_fac = wp.sqrt(g / h_ref)
    y = y0 + (FLOAT(j - ng) + HALF) * dy

    eta_t = eta0
    u_t = ZERO
    v_t = ZERO
    for m in range(nmodes):
        om = TWO_PI * harm[m] / period
        e = amp[m] * wp.cos(om * t - ky[m] * y + pha[m])
        sth = ky[m] * c_ref / om
        # host validation guarantees |sth| < 1; the clamp is a belt
        sth = wp.clamp(sth, -FLOAT(0.999), FLOAT(0.999))
        cth = wp.sqrt(ONE - sth * sth)
        eta_t += e
        # linear progressive-wave velocity eta' sqrt(g/h), split by direction
        u_t += e * cth * uv_fac
        v_t += e * sth * uv_fac

    h_t = wp.max(eta_t - bb, ZERO)
    Rp = u_t + TWO * wp.sqrt(g * h_t)

    hi = wp.max(h[Ii, j], ZERO)
    ui = desing_u(hi, qx[Ii, j], eps)
    vi = desing_u(hi, qy[Ii, j], eps)
    Rm = ui - TWO * wp.sqrt(g * hi)

    cb = QUARTER * (Rp - Rm)
    if cb < ZERO:
        cb = ZERO
    ub = HALF * (Rp + Rm)
    hb = cb * cb / g

    # tangential velocity: from the forcing on inflow, from the interior on
    # outflow (the contact characteristic moves with the normal velocity)
    vb = vi
    if ub > ZERO:
        vb = v_t

    h[I, j] = hb
    qx[I, j] = hb * ub
    qy[I, j] = hb * vb
