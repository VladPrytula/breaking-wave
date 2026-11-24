"""render_fast: the oblique dielectric ray tracer (stage E).

Per camera ray (plan.md section 8, priorities re-ranked by measurement in
docs/reference-video.md):

1. March the ray against the height field z_s = eta + zeta (eta resolved,
   zeta the subgrid ripple closure) and the bed b, whichever comes first.
2. On a water hit: Fresnel split. The reflected ray samples the overcast sky
   dome -- this Fresnel-weighted sky reflection is the dominant term, the
   measured 4.2x far-to-near luminance ramp. The refracted ray (Snell,
   n = 1.333) marches to the bed with Beer-Lambert attenuation both on the
   view path and on the downwelling light.
3. On a bed hit (dry or through water): pebble albedo texture, darkened by
   the wetness closure state w.
4. Foam from the transported c_f field as a high-albedo Lambertian layer.
   With the closure off the input array is zero and no whiteness can appear.
5. No caustics and no sun glitter: the reference scene is overcast, with no
   directional source (docs/reference-video.md). Total internal reflection is
   handled by the Fresnel term (R -> 1), which matters only for future
   underwater cameras.

Geometry: bed and water are both single-valued height fields on a render grid
`scale`x finer than the hydro grid, marched with fixed cell-sized steps plus
bisection refinement -- no BVH, per docs/decisions.md ("Stack"). Stones are
rendered as exactly the mounds the solver flows around (b_hydro), not as
separate render-only geometry: until the stage F single-source stone field
exists, drawing sharper stones than the water feels would fake a coupling.

Named approximations, all visible in the code:
- Reflected rays sample only the sky: an emergent stone does not occlude the
  sky's reflection in the water beside it (second-bounce geometry).
- A REFLECTED ray that leaves the domain going *down* uses the horizon
  radiance -- the grazing-incidence limit of far water.
- A PRIMARY ray that exits through the offshore edge shades an analytic far
  field: the wavemaker's incoming components plus the ripple closure, both
  exact plane waves at the constant offshore depth, normal-mapped onto the
  still-water plane. Named sub-approximations: no parallax or self-occlusion
  of those waves (centimetres of height seen from tens of metres at grazing);
  incoming components only (the outgoing wave the boundary absorbs is not
  re-radiated offshore); modes finer than the pixel footprint are Gaussian-
  prefiltered into a slope-quadrature reflection lobe (view-plane slopes
  only); flat bed at mean albedo; no foam or breaking offshore -- 5 cm waves
  in 0.8 m of water do not break.
- The crossing test samples the surface at step endpoints, so a sub-cell
  crest thinner than one render cell can be stepped over.
- Downwelling underwater irradiance is attenuated by the local vertical
  depth, not integrated along refracted light paths (that is the caustics
  machinery, deliberately absent).

This renderer is deliberately NOT differentiable (docs/decisions.md): image
gradients come from Mitsuba at stage J. It is also deliberately float32-only,
whatever SHORELOOP_FP64 says: it feeds a display path, no verification gate
runs through it, and the solver fields are cast on entry (in `prep_fields`,
the one kernel that touches FLOAT arrays -- bare literals below are correct
float32 everywhere else BY this design, not by accident).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from shoreloop.closures.ripples import RippleField, RippleSpectrum
from shoreloop.config import GridConfig
from shoreloop.precision import FLOAT, HALF, ONE, ZERO
from shoreloop.render.camera import Camera, tonemap
from shoreloop.render.sky import OvercastSky, Sun
from shoreloop.render.texture import PebbleTexture, pebble_maps
from shoreloop.scene.forcing import WaveForcing

PI = 3.14159265358979


def _fresnel_host(cosi: float, n: float) -> float:
    """Host-side unpolarized dielectric reflectance, for the sun's constant
    interface transmission. Must agree with the kernel's _fresnel."""
    sini = np.sqrt(max(1.0 - cosi * cosi, 0.0))
    sint = sini / n
    cost = np.sqrt(max(1.0 - sint * sint, 0.0))
    rs = (cosi - n * cost) / (cosi + n * cost)
    rp = (n * cosi - cost) / (n * cosi + cost)
    return 0.5 * (rs * rs + rp * rp)


@dataclass
class WaterOptics:
    """Optical closure parameters -- all part of theta_render, all fittable."""

    n_water: float = 1.333
    # Beer-Lambert extinction per metre; red absorbed first, which is what
    # makes clear shallow water read as blue-green with depth.
    kappa: tuple[float, float, float] = (0.85, 0.22, 0.14)
    # Upwelling scatter of the water column itself: the radiance a ray relaxes
    # to as its path length grows. Small for a clear lake.
    scatter: tuple[float, float, float] = (0.010, 0.026, 0.036)
    foam_albedo: float = 0.85
    # Wet-pebble darkening, per channel. Measured: wet/dry luma ratio 0.79,
    # slightly warmer wet (R/B 0.465 -> 0.500), docs/reference-video.md.
    wet_dark: tuple[float, float, float] = (0.82, 0.80, 0.76)


@dataclass(frozen=True)
class OffshoreWaveField:
    """Exact plane-wave modes continued beyond the render-grid boundary.

    Each mode is ``a cos(kx*x + ky*y - 2*pi*n*t/T + phase)``.  Unlike
    ``WaveForcing``, this interchange carries spatial wavenumbers explicitly,
    so a finite-depth 3-D inlet does not get silently replaced by the shallow
    dispersion used by the SWE solver.
    """

    period: float
    harmonics: tuple[int, ...]
    kx: tuple[float, ...]
    ky: tuple[float, ...]
    amplitudes: tuple[float, ...]
    phases: tuple[float, ...]


# ---------------------------------------------------------------------------
# device functions (float32)
# ---------------------------------------------------------------------------
@wp.func
def _bilin(f: wp.array2d(dtype=wp.float32), x: wp.float32, y: wp.float32,
           x0: wp.float32, y0: wp.float32,
           inv_dx: wp.float32, inv_dy: wp.float32,
           nx: int, ny: int) -> wp.float32:
    """Bilinear sample of a cell-centred field: clamped in x, periodic in y
    (the solver's alongshore boundary is periodic, so the scene wraps)."""
    fx = (x - x0) * inv_dx - 0.5
    fy = (y - y0) * inv_dy - 0.5
    fx = wp.clamp(fx, 0.0, wp.float32(nx - 1) - 1.0e-3)
    fyn = wp.float32(ny)
    fy = fy - fyn * wp.floor(fy / fyn)
    i0 = int(fx)
    ax = fx - wp.float32(i0)
    i1 = wp.min(i0 + 1, nx - 1)
    j0 = int(fy)
    if j0 >= ny:
        j0 = ny - 1
    ay = fy - wp.float32(j0)
    j1 = j0 + 1
    if j1 >= ny:
        j1 = 0
    return (f[i0, j0] * (1.0 - ax) * (1.0 - ay) + f[i1, j0] * ax * (1.0 - ay)
            + f[i0, j1] * (1.0 - ax) * ay + f[i1, j1] * ax * ay)


@wp.func
def _bilin3(f: wp.array2d(dtype=wp.vec3f), x: wp.float32, y: wp.float32,
            x0: wp.float32, y0: wp.float32,
            inv_dx: wp.float32, inv_dy: wp.float32,
            nx: int, ny: int) -> wp.vec3f:
    fx = (x - x0) * inv_dx - 0.5
    fy = (y - y0) * inv_dy - 0.5
    fx = wp.clamp(fx, 0.0, wp.float32(nx - 1) - 1.0e-3)
    fyn = wp.float32(ny)
    fy = fy - fyn * wp.floor(fy / fyn)
    i0 = int(fx)
    ax = fx - wp.float32(i0)
    i1 = wp.min(i0 + 1, nx - 1)
    j0 = int(fy)
    if j0 >= ny:
        j0 = ny - 1
    ay = fy - wp.float32(j0)
    j1 = j0 + 1
    if j1 >= ny:
        j1 = 0
    return (f[i0, j0] * (1.0 - ax) * (1.0 - ay) + f[i1, j0] * ax * (1.0 - ay)
            + f[i0, j1] * (1.0 - ax) * ay + f[i1, j1] * ax * ay)


@wp.func
def _normal(z: wp.array2d(dtype=wp.float32), x: wp.float32, y: wp.float32,
            x0: wp.float32, y0: wp.float32,
            inv_dx: wp.float32, inv_dy: wp.float32,
            nx: int, ny: int, ex: wp.float32, ey: wp.float32) -> wp.vec3f:
    zx = (_bilin(z, x + ex, y, x0, y0, inv_dx, inv_dy, nx, ny)
          - _bilin(z, x - ex, y, x0, y0, inv_dx, inv_dy, nx, ny)) / (2.0 * ex)
    zy = (_bilin(z, x, y + ey, x0, y0, inv_dx, inv_dy, nx, ny)
          - _bilin(z, x, y - ey, x0, y0, inv_dx, inv_dy, nx, ny)) / (2.0 * ey)
    return wp.normalize(wp.vec3f(-zx, -zy, 1.0))


@wp.func
def _sky(d: wp.vec3f, zenith: wp.vec3f, horizon: wp.vec3f,
         p: wp.float32, glow: wp.vec3f, glow_dir: wp.vec3f,
         glow_sharp: wp.float32) -> wp.vec3f:
    """Sky dome; mirrors OvercastSky.radiance exactly. Downward directions
    clamp elevation to the horizon value: the grazing far-water limit. The
    glow term is the localised sunward bright patch of the dramatic presets;
    zero by default."""
    mu = wp.pow(wp.clamp(d[2], 0.0, 1.0), p)
    out = horizon * (1.0 - mu) + zenith * mu
    if glow[0] + glow[1] + glow[2] > 0.0:
        c = wp.clamp(wp.dot(d, glow_dir), 0.0, 1.0)
        out += glow * wp.pow(c, glow_sharp)
    return out


@wp.func
def _fresnel(cosi: wp.float32, eta: wp.float32) -> wp.float32:
    """Unpolarized dielectric reflectance, eta = n2/n1. R -> 1 past the
    critical angle (total internal reflection, for eta < 1)."""
    s2t = (1.0 - cosi * cosi) / (eta * eta)
    if s2t >= 1.0:
        return 1.0
    cost = wp.sqrt(1.0 - s2t)
    rs = (cosi - eta * cost) / (cosi + eta * cost)
    rp = (eta * cosi - cost) / (eta * cosi + cost)
    return 0.5 * (rs * rs + rp * rp)


@wp.func
def _beer(kappa: wp.vec3f, path: wp.float32) -> wp.vec3f:
    return wp.vec3f(wp.exp(-kappa[0] * path), wp.exp(-kappa[1] * path),
                    wp.exp(-kappa[2] * path))


@wp.func
def _march(z: wp.array2d(dtype=wp.float32),
           o: wp.vec3f, d: wp.vec3f,
           x0: wp.float32, y0: wp.float32,
           inv_dx: wp.float32, inv_dy: wp.float32,
           nx: int, ny: int,
           cell: wp.float32, t_max: wp.float32,
           z_ceiling: wp.float32) -> wp.float32:
    """First crossing of the ray below the height field; -1 for a miss.

    Fixed steps advancing ~one render cell in the horizontal, sign test at
    the endpoints, bisection to refine. When the horizontal motion is tiny
    the capped step is safe because the ray-minus-surface function is then
    monotone (|d_z| dominates any surface slope contribution).
    """
    hxy = wp.sqrt(d[0] * d[0] + d[1] * d[1])
    step = cell / wp.max(hxy, 0.05)
    t = wp.float32(0.0)
    f_prev = o[2] - _bilin(z, o[0], o[1], x0, y0, inv_dx, inv_dy, nx, ny)
    if f_prev <= 0.0:
        return 0.0
    while t < t_max:
        t_next = wp.min(t + step, t_max)
        px = o[0] + t_next * d[0]
        py = o[1] + t_next * d[1]
        pz = o[2] + t_next * d[2]
        if d[2] >= 0.0 and pz > z_ceiling:
            return -1.0
        f = pz - _bilin(z, px, py, x0, y0, inv_dx, inv_dy, nx, ny)
        if f <= 0.0:
            a = t
            bt = t_next
            for _ in range(16):
                m = 0.5 * (a + bt)
                fm = (o[2] + m * d[2]) - _bilin(z, o[0] + m * d[0],
                                                o[1] + m * d[1],
                                                x0, y0, inv_dx, inv_dy, nx, ny)
                if fm > 0.0:
                    a = m
                else:
                    bt = m
            return 0.5 * (a + bt)
        t = t_next
    return -1.0


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------
@wp.func
def _lit(z_bed: wp.array2d(dtype=wp.float32),
         o: wp.vec3f, sun_dir: wp.vec3f,
         x0: wp.float32, y0: wp.float32,
         inv_dx: wp.float32, inv_dy: wp.float32,
         nx: int, ny: int,
         step: wp.float32, t_max: wp.float32,
         z_bed_max: wp.float32) -> wp.float32:
    """1.0 if the sun is visible from o, 0.0 if a bed occluder blocks it.

    A shadow ray marched against the bed height field: emergent stones and
    (in the "pebbles" texture style) individual pebble domes cast real
    shadows. The march uses a coarser step than primary rays -- centimetre
    shadow edges, a third of the cost -- and exits as soon as the ray climbs
    above the tallest bed point.
    """
    t = _march(z_bed, o, sun_dir, x0, y0, inv_dx, inv_dy, nx, ny,
               step, t_max, z_bed_max)
    if t >= 0.0:
        return 0.0
    return 1.0


@wp.struct
class TraceParams:
    cam_pos: wp.vec3f
    fwd: wp.vec3f
    right: wp.vec3f
    up: wp.vec3f
    tanx: wp.float32
    tany: wp.float32
    x0: wp.float32
    y0: wp.float32
    inv_dx: wp.float32
    inv_dy: wp.float32
    dx: wp.float32
    dy: wp.float32
    nx: wp.int32
    ny: wp.int32
    cell: wp.float32
    t_max: wp.float32
    z_ceiling: wp.float32
    n_water: wp.float32
    kappa: wp.vec3f
    scatter: wp.vec3f
    zenith: wp.vec3f
    horizon: wp.vec3f
    sky_p: wp.float32
    glow: wp.vec3f
    glow_dir: wp.vec3f
    glow_sharp: wp.float32
    e_down: wp.vec3f
    foam_albedo: wp.float32
    wet_dark: wp.vec3f
    alb_mean: wp.vec3f  # mean bed albedo, for the offshore-extension fade
    # --- directional sun (has_sun == 0 disables every term below) ---
    has_sun: wp.int32
    sun_dir: wp.vec3f        # unit, toward the sun
    sun_irr: wp.vec3f        # irradiance on a plane perpendicular to the sun
    glit_rad: wp.vec3f       # filtered solar-disc radiance for specular glint
    glit_inv_sig2: wp.float32  # 1 / sigma^2 (sigma in radians)
    tsun_up: wp.vec3f        # from a submerged point toward the refracted sun
    sun_trans: wp.float32    # Fresnel transmission of the sun at the surface
    inv_cost_sun: wp.float32  # slant-path factor 1 / cos(theta_t)
    shadow_step: wp.float32
    z_bed_max: wp.float32    # shadow rays early-out above the tallest bed
    # --- analytic offshore far field (ff_on == 0 disables) ---
    ff_on: wp.int32
    ff_nf: wp.int32          # leading modes are forcing (no gust envelope)
    ff_nm: wp.int32          # total modes: forcing + ripple continuations
    ff_ne: wp.int32          # gustiness envelope modes (ripple modes only)
    ff_eta0: wp.float32      # still-water plane the far field is mapped onto
    ff_depth: wp.float32     # constant offshore still depth [m]
    pix_sig: wp.float32      # pixel angular size [rad], footprint prefilter


@wp.func
def _far_shade(d: wp.vec3f, n: wp.vec3f, p: TraceParams) -> wp.vec3f:
    """One facet of open offshore water: the in-domain water shading with the
    offshore constants -- flat bed at ff_depth with the mean albedo
    (submerged => wet), no foam (nothing breaks out there), no shadow rays
    (no occluders offshore)."""
    cosi = wp.max(-wp.dot(d, n), 1.0e-4)
    R = _fresnel(cosi, p.n_water)
    refl = d + (2.0 * cosi) * n
    Lr = _sky(refl, p.zenith, p.horizon, p.sky_p,
              p.glow, p.glow_dir, p.glow_sharp)

    Lt = wp.vec3f(0.0, 0.0, 0.0)
    ir = 1.0 / p.n_water
    s2t = ir * ir * (1.0 - cosi * cosi)
    cost = wp.sqrt(wp.max(1.0 - s2t, 0.0))
    td = wp.normalize(ir * d + (ir * cosi - cost) * n)
    if td[2] < -1.0e-4:
        path = p.ff_depth / wp.max(-td[2], 1.0e-4)
        trans = _beer(p.kappa, path)
        irr = wp.cw_mul(p.e_down, _beer(p.kappa, p.ff_depth))  # flat bed, sv=1
        if p.has_sun != 0:
            # open water: lit = 1, bed normal vertical => n.l = tsun_up_z
            irr += (p.sun_trans * p.tsun_up[2]) * wp.cw_mul(
                p.sun_irr, _beer(p.kappa, p.ff_depth * p.inv_cost_sun))
        Lbed = wp.cw_mul(wp.cw_mul(p.alb_mean, p.wet_dark), irr) \
            * (1.0 / 3.14159265)
        Lt = (wp.cw_mul(Lbed, trans)
              + wp.cw_mul(p.scatter, wp.vec3f(1.0, 1.0, 1.0) - trans))
    else:
        # refracted ray bent back upward by an extreme slope: purely reflective
        R = 1.0

    L = R * Lr + (1.0 - R) * Lt
    if p.has_sun != 0:
        cg = wp.clamp(wp.dot(refl, p.sun_dir), -1.0, 1.0)
        L += (R * wp.exp(-(1.0 - cg) * p.glit_inv_sig2)) * p.glit_rad
    return L


@wp.func
def _far_field(o: wp.vec3f, d: wp.vec3f, t_edge: wp.float32,
               ff_modes: wp.array(dtype=wp.vec4f),
               ff_env: wp.array(dtype=wp.vec4f),
               p: TraceParams) -> wp.vec3f:
    """Radiance of a primary ray that left the domain through the offshore
    edge: the analytic continuation of the modelled sea.

    The surface is the still-water plane z = ff_eta0 normal-mapped with every
    mode the model asserts exists offshore -- the wavemaker's incoming
    components and the ripple closure, both exact plane waves at the constant
    offshore depth, phase-continuous with the in-domain fields at the edge by
    construction. Each mode row is (kx, ky, amp, phase(t)), the phase wrapped
    mod 2 pi on the host in float64 per frame.

    Modes finer than the pixel footprint on the plane are removed by a
    Gaussian prefilter (anisotropic: the along-view footprint stretches as
    1/sin of the grazing angle) -- displaced geometry we cannot sample is how
    water sparkles like tinfoil. The slope variance each mode loses to the
    filter is NOT discarded: it widens the reflection lobe, as a 3-point
    Gauss-Hermite quadrature over the view-plane slope with each facet
    weighted by its projected visibility (n.v). Without this the far water
    turns into a flat mirror of the bright horizon band and a hard luminance
    seam appears at the domain edge, because the marched water just inside it
    reflects a visibility-biased mix of facet angles. Cross-view slope
    variance is still dropped (the overcast sky is nearly azimuth-uniform).
    As everything filters out the far field converges to the slope-broadened
    grazing mirror, and at d_z -> 0 to the sky's own horizon radiance: the
    horizon needs no special case.
    """
    t_pl = (p.ff_eta0 - o[2]) / d[2]
    if t_pl < t_edge:  # descended below the plane inside the domain (trough)
        t_pl = t_edge
    F = o + t_pl * d
    # pixel footprint on the plane; the along-view axis is ~x for the
    # offshore-looking cameras this path serves
    sig_y = 0.7 * t_pl * p.pix_sig
    sig_x = sig_y / wp.max(-d[2], 1.0e-3)

    env = wp.float32(1.0)
    for e in range(int(p.ff_ne)):
        me = ff_env[e]
        env -= me[2] * (0.5 + 0.5 * wp.cos(me[0] * F[0] + me[1] * F[1]
                                           + me[3]))
    zx = wp.float32(0.0)   # explicit (resolved-by-the-pixel-grid) slopes
    zy = wp.float32(0.0)
    var = wp.float32(0.0)  # view-plane slope variance lost to the prefilter
    for m in range(int(p.ff_nm)):
        v = ff_modes[m]
        a = v[2]
        if m >= int(p.ff_nf):
            a = a * env
        q = 0.5 * (v[0] * sig_x * v[0] * sig_x + v[1] * sig_y * v[1] * sig_y)
        f = wp.float32(0.0)
        if q < 9.0:
            f = wp.exp(-q)
            s = a * f * wp.sin(v[0] * F[0] + v[1] * F[1] + v[3])
            zx -= v[0] * s
            zy -= v[1] * s
        var += 0.5 * a * a * v[0] * v[0] * (1.0 - f * f)
    sig = wp.sqrt(var)

    L = wp.vec3f(0.0, 0.0, 0.0)
    wsum = wp.float32(0.0)
    for i in range(3):
        dz = (wp.float32(i) - 1.0) * 1.7320508 * sig   # nodes 0, +-sqrt(3) s
        w = wp.float32(0.166666667)                    # weights 1/6, 2/3, 1/6
        if i == 1:
            w = 0.666666667
        n = wp.normalize(wp.vec3f(-(zx + dz), -zy, 1.0))
        w *= wp.max(-wp.dot(d, n), 0.0)   # projected facet visibility
        if w > 0.0:
            L += w * _far_shade(d, n, p)
            wsum += w
    if wsum > 0.0:
        L = L * (1.0 / wsum)
    else:  # every facet back-facing: the grazing limit is the horizon itself
        L = _sky(d, p.zenith, p.horizon, p.sky_p,
                 p.glow, p.glow_dir, p.glow_sharp)
    return L


@wp.func
def _surf_or_fill(hv: FLOAT, bv: FLOAT, h_wet: FLOAT, drop: FLOAT) -> FLOAT:
    """Water surface where wet; slightly below the bed where dry, so the
    interpolated surface crosses the bed near the true sub-cell shoreline
    instead of filling with a fixed reference level (which paints phantom
    gradients along the swash edge -- the diagnostic renderer's known
    artefact)."""
    if hv > h_wet:
        return hv + bv
    return bv - drop


@wp.kernel
def prep_bed(
    b: wp.array2d(dtype=FLOAT),
    z_bed: wp.array2d(dtype=wp.float32),
    ng: int, nxh: int, nyh: int,
    sx: FLOAT, sy: FLOAT,
):
    """Bed height on the render grid, bilinear from padded hydro b. Done on
    device with the same index mapping as prep_fields so the water-vs-bed
    comparison at the shoreline is exact by construction."""
    ri, rj = wp.tid()
    fx = (FLOAT(ri) + HALF) * sx - HALF
    fy = (FLOAT(rj) + HALF) * sy - HALF
    fx = wp.clamp(fx, ZERO, FLOAT(nxh - 1) - FLOAT(1.0e-3))
    fyn = FLOAT(nyh)
    fy = fy - fyn * wp.floor(fy / fyn)
    i0 = int(fx)
    ax = fx - FLOAT(i0)
    i1 = wp.min(i0 + 1, nxh - 1)
    j0 = int(fy)
    if j0 >= nyh:
        j0 = nyh - 1
    ay = fy - FLOAT(j0)
    j1 = j0 + 1
    if j1 >= nyh:
        j1 = 0
    v = (b[ng + i0, ng + j0] * (ONE - ax) * (ONE - ay)
         + b[ng + i1, ng + j0] * ax * (ONE - ay)
         + b[ng + i0, ng + j1] * (ONE - ax) * ay
         + b[ng + i1, ng + j1] * ax * ay)
    z_bed[ri, rj] = wp.float32(v)


@wp.kernel
def prep_fields(
    h: wp.array2d(dtype=FLOAT), b: wp.array2d(dtype=FLOAT),
    cf: wp.array2d(dtype=FLOAT), w: wp.array2d(dtype=FLOAT),
    zeta: wp.array2d(dtype=FLOAT),
    z_bed: wp.array2d(dtype=wp.float32),
    z_wat: wp.array2d(dtype=wp.float32),
    z_top: wp.array2d(dtype=wp.float32),
    cf_r: wp.array2d(dtype=wp.float32),
    w_r: wp.array2d(dtype=wp.float32),
    ng: int, nxh: int, nyh: int,
    sx: FLOAT, sy: FLOAT,
    h_wet: FLOAT, drop: FLOAT,
):
    """Upsample the per-frame solver fields to the render grid and assemble
    the two marchable surfaces: z_wat = eta + zeta and z_top = max(wat, bed).
    The one kernel where solver FLOAT meets render float32."""
    ri, rj = wp.tid()
    fx = (FLOAT(ri) + HALF) * sx - HALF
    fy = (FLOAT(rj) + HALF) * sy - HALF
    fx = wp.clamp(fx, ZERO, FLOAT(nxh - 1) - FLOAT(1.0e-3))
    fyn = FLOAT(nyh)
    fy = fy - fyn * wp.floor(fy / fyn)
    i0 = int(fx)
    ax = fx - FLOAT(i0)
    i1 = wp.min(i0 + 1, nxh - 1)
    j0 = int(fy)
    if j0 >= nyh:
        j0 = nyh - 1
    ay = fy - FLOAT(j0)
    j1 = j0 + 1
    if j1 >= nyh:
        j1 = 0

    I0 = ng + i0
    I1 = ng + i1
    J0 = ng + j0
    J1 = ng + j1
    w00 = (ONE - ax) * (ONE - ay)
    w10 = ax * (ONE - ay)
    w01 = (ONE - ax) * ay
    w11 = ax * ay

    e = (_surf_or_fill(h[I0, J0], b[I0, J0], h_wet, drop) * w00
         + _surf_or_fill(h[I1, J0], b[I1, J0], h_wet, drop) * w10
         + _surf_or_fill(h[I0, J1], b[I0, J1], h_wet, drop) * w01
         + _surf_or_fill(h[I1, J1], b[I1, J1], h_wet, drop) * w11)
    cfv = cf[I0, J0] * w00 + cf[I1, J0] * w10 + cf[I0, J1] * w01 + cf[I1, J1] * w11
    wv = w[I0, J0] * w00 + w[I1, J0] * w10 + w[I0, J1] * w01 + w[I1, J1] * w11

    zw = wp.float32(e + zeta[ri, rj])
    zb = z_bed[ri, rj]
    z_wat[ri, rj] = zw
    z_top[ri, rj] = wp.max(zw, zb)
    cf_r[ri, rj] = wp.float32(cfv)
    w_r[ri, rj] = wp.float32(wv)


@wp.kernel
def apply_scalar_mask(
    field: wp.array2d(dtype=FLOAT),
    mask: wp.array2d(dtype=FLOAT),
):
    i, j = wp.tid()
    field[i, j] *= mask[i, j]


@wp.kernel
def trace(
    z_top: wp.array2d(dtype=wp.float32),
    z_wat: wp.array2d(dtype=wp.float32),
    z_bed: wp.array2d(dtype=wp.float32),
    albedo: wp.array2d(dtype=wp.vec3f),
    w_r: wp.array2d(dtype=wp.float32),
    cf_r: wp.array2d(dtype=wp.float32),
    ff_modes: wp.array(dtype=wp.vec4f),
    ff_env: wp.array(dtype=wp.vec4f),
    p: TraceParams,
    img: wp.array2d(dtype=wp.vec3f),
):
    py, px = wp.tid()
    H = img.shape[0]
    W = img.shape[1]
    u = (wp.float32(px) + 0.5) / wp.float32(W) * 2.0 - 1.0
    v = 1.0 - 2.0 * (wp.float32(py) + 0.5) / wp.float32(H)
    d = wp.normalize(p.fwd + (u * p.tanx) * p.right + (v * p.tany) * p.up)
    o = p.cam_pos
    nx = int(p.nx)
    ny = int(p.ny)

    # with a far field, stop marching at the offshore edge: beyond it the
    # grids are only clamped copies of the boundary column, and the analytic
    # continuation takes over
    t_lim = p.t_max
    if p.ff_on != 0 and d[0] < -1.0e-6:
        te = (p.x0 - o[0]) / d[0]
        if te < t_lim:
            t_lim = te

    t_hit = _march(z_top, o, d, p.x0, p.y0, p.inv_dx, p.inv_dy, nx, ny,
                   p.cell, t_lim, p.z_ceiling)
    if t_hit < 0.0:
        if p.ff_on != 0 and d[0] < -1.0e-6 and d[2] < -1.0e-4:
            img[py, px] = _far_field(o, d, t_lim, ff_modes, ff_env, p)
        else:
            img[py, px] = _sky(d, p.zenith, p.horizon, p.sky_p,
                               p.glow, p.glow_dir, p.glow_sharp)
        return

    P = o + t_hit * d
    zw = _bilin(z_wat, P[0], P[1], p.x0, p.y0, p.inv_dx, p.inv_dy, nx, ny)
    zb = _bilin(z_bed, P[0], P[1], p.x0, p.y0, p.inv_dx, p.inv_dy, nx, ny)
    ex = 0.5 * p.dx
    ey = 0.5 * p.dy
    L = wp.vec3f(0.0, 0.0, 0.0)

    if zw > zb + 1.0e-4:
        # ---- water surface -------------------------------------------------
        n = _normal(z_wat, P[0], P[1], p.x0, p.y0, p.inv_dx, p.inv_dy,
                    nx, ny, ex, ey)
        cosi = wp.max(-wp.dot(d, n), 1.0e-4)
        R = _fresnel(cosi, p.n_water)
        refl = d + (2.0 * cosi) * n
        Lr = _sky(refl, p.zenith, p.horizon, p.sky_p,
                  p.glow, p.glow_dir, p.glow_sharp)

        lit = wp.float32(1.0)
        if p.has_sun != 0:
            lit = _lit(z_bed, P + wp.vec3f(0.0, 0.0, 0.01)
                       + p.shadow_step * p.sun_dir, p.sun_dir,
                       p.x0, p.y0, p.inv_dx, p.inv_dy, nx, ny,
                       p.shadow_step, p.t_max, p.z_bed_max)

        Lt = wp.vec3f(0.0, 0.0, 0.0)
        ir = 1.0 / p.n_water
        s2t = ir * ir * (1.0 - cosi * cosi)
        cost = wp.sqrt(wp.max(1.0 - s2t, 0.0))
        td = wp.normalize(ir * d + (ir * cosi - cost) * n)
        if td[2] < -1.0e-4:
            o2 = P + 1.0e-4 * td
            t2 = _march(z_bed, o2, td, p.x0, p.y0, p.inv_dx, p.inv_dy,
                        nx, ny, p.cell, p.t_max, p.z_ceiling)
            if t2 < 0.0:
                t2 = p.t_max  # bed never reached: pure water-column radiance
            Q = o2 + t2 * td
            zwQ = _bilin(z_wat, Q[0], Q[1], p.x0, p.y0, p.inv_dx, p.inv_dy,
                         nx, ny)
            depthQ = wp.max(zwQ - Q[2], 0.0)
            albQ = _bilin3(albedo, Q[0], Q[1], p.x0, p.y0, p.inv_dx, p.inv_dy,
                           nx, ny)
            nb = _normal(z_bed, Q[0], Q[1], p.x0, p.y0, p.inv_dx, p.inv_dy,
                         nx, ny, ex, ey)
            # a refracted ray can still dip up to ~1 m past the offshore edge,
            # where the bed grid is a clamped copy of the boundary column:
            # fade its stretched pebble domes and albedo to the flat mean
            fq = wp.clamp((Q[0] - p.x0 + 1.5) / 2.0, 0.0, 1.0)
            if fq < 1.0:
                nb = wp.normalize(wp.vec3f(nb[0] * fq, nb[1] * fq, nb[2]))
                albQ = p.alb_mean * (1.0 - fq) + albQ * fq
            sv = 0.5 + 0.5 * wp.max(nb[2], 0.0)
            # diffuse sky downwelling, attenuated by the local vertical depth
            irr = wp.cw_mul(p.e_down, _beer(p.kappa, depthQ)) * sv
            if p.has_sun != 0:
                # direct sun transmitted through the interface: Snell-bent
                # direction and slant-path Beer-Lambert, but NO caustic
                # focusing -- light-side transport stays deferred. The shadow
                # ray is unrefracted (occluders are emergent stones).
                litQ = _lit(z_bed, Q + wp.vec3f(0.0, 0.0, 0.01)
                            + p.shadow_step * p.sun_dir, p.sun_dir,
                            p.x0, p.y0, p.inv_dx, p.inv_dy, nx, ny,
                            p.shadow_step, p.t_max, p.z_bed_max)
                ndl = wp.max(wp.dot(nb, p.tsun_up), 0.0)
                irr += (p.sun_trans * litQ * ndl) * wp.cw_mul(
                    p.sun_irr, _beer(p.kappa, depthQ * p.inv_cost_sun))
            # submerged bed is fully wet by definition
            Lbed = wp.cw_mul(wp.cw_mul(albQ, p.wet_dark), irr) * (1.0 / 3.14159265)
            trans = _beer(p.kappa, t2)
            Lt = (wp.cw_mul(Lbed, trans)
                  + wp.cw_mul(p.scatter, wp.vec3f(1.0, 1.0, 1.0) - trans))
        else:
            # refracted ray bent back upward by an extreme slope: treat the
            # facet as purely reflective rather than invent transmission
            R = 1.0

        L = R * Lr + (1.0 - R) * Lt

        if p.has_sun != 0:
            # filtered specular glint: angular Gaussian around the mirror
            # direction. The width absorbs both the solar disc and the
            # sub-render-cell slope variance -- the temporal filtering that
            # keeps sparkle from aliasing (docs/decisions.md).
            cg = wp.clamp(wp.dot(refl, p.sun_dir), -1.0, 1.0)
            L += (lit * R * wp.exp(-(1.0 - cg) * p.glit_inv_sig2)) * p.glit_rad

        c = wp.clamp(_bilin(cf_r, P[0], P[1], p.x0, p.y0, p.inv_dx, p.inv_dy,
                            nx, ny), 0.0, 1.0)
        if c > 0.0:
            firr = p.e_down
            if p.has_sun != 0:
                firr += (lit * wp.max(p.sun_dir[2], 0.0)) * p.sun_irr
            Lfoam = firr * (p.foam_albedo / 3.14159265)
            L = (1.0 - c) * L + c * Lfoam
    else:
        # ---- bed, dry or damp ----------------------------------------------
        nb = _normal(z_bed, P[0], P[1], p.x0, p.y0, p.inv_dx, p.inv_dy,
                     nx, ny, ex, ey)
        sv = 0.5 + 0.5 * wp.max(nb[2], 0.0)
        wv = wp.clamp(_bilin(w_r, P[0], P[1], p.x0, p.y0, p.inv_dx, p.inv_dy,
                             nx, ny), 0.0, 1.0)
        albP = _bilin3(albedo, P[0], P[1], p.x0, p.y0, p.inv_dx, p.inv_dy,
                       nx, ny)
        wetf = wp.vec3f(1.0, 1.0, 1.0) * (1.0 - wv) + p.wet_dark * wv
        irr = p.e_down * sv
        if p.has_sun != 0:
            litP = _lit(z_bed, P + wp.vec3f(0.0, 0.0, 0.01)
                        + p.shadow_step * p.sun_dir, p.sun_dir,
                        p.x0, p.y0, p.inv_dx, p.inv_dy, nx, ny,
                        p.shadow_step, p.t_max, p.z_bed_max)
            irr += (litP * wp.max(wp.dot(nb, p.sun_dir), 0.0)) * p.sun_irr
        L = wp.cw_mul(wp.cw_mul(albP, wetf), irr) * (1.0 / 3.14159265)

    img[py, px] = L


# ---------------------------------------------------------------------------
# host driver
# ---------------------------------------------------------------------------
class Raytracer:
    """Owns the render-grid fields and renders one frame per call.

    `scale` is the explicit hydro-to-render scale separation: the ripple
    closure carries wavelengths down to ~1.5 cm, which the hydro grid cannot
    sample but a 2x render grid marginally can. Modes finer than the render
    grid belong in BRDF roughness, not displacement (docs/decisions.md).
    """

    def __init__(self, solver, spectrum: RippleSpectrum | None = None,
                 scale: int = 2, camera: Camera | None = None,
                 sky: OvercastSky | None = None,
                 optics: WaterOptics | None = None,
                 texture: PebbleTexture | None = None,
                 sun: Sun | None = None,
                 eta0: float = 0.0, ripples_enabled: bool = True,
                 forcing: WaveForcing | None = None,
                 offshore_field: OffshoreWaveField | None = None,
                 ripple_fade: tuple[float, float] | None = None,
                 ripple_alongshore_period: float | None = None):
        g = solver.cfg.grid
        self.solver = solver
        self.camera = camera or Camera()
        self.sky = sky or OvercastSky()
        self.optics = optics or WaterOptics()
        self.sun = sun or Sun()
        dev = solver.cfg.device

        nxr, nyr = g.nx * scale, g.ny * scale
        self.nxr, self.nyr = nxr, nyr
        self.dxr, self.dyr = g.Lx / nxr, g.Ly / nyr
        shape = (nxr, nyr)

        self.z_bed = wp.zeros(shape, dtype=wp.float32, device=dev)
        wp.launch(prep_bed, dim=shape,
                  inputs=[solver.state.b, self.z_bed, g.ng, g.nx, g.ny,
                          1.0 / scale, 1.0 / scale],
                  device=dev)

        # pebble texture: albedo, and (style "pebbles") a per-stone dome
        # displacement ADDED to the render bed -- plan.md section 9's
        # G_render = b_hydro + pebbles. The solver keeps the smooth bed; the
        # renderer marches real 1-2 cm stones, so rounded shading, a
        # pebble-ragged waterline and per-stone Beer-Lambert tinting all come
        # from the same ray marcher with no special cases.
        alb, peb_h = pebble_maps(nxr, nyr, self.dxr, self.dyr, g.Ly, texture)
        self.albedo = wp.array(alb, dtype=wp.vec3f, device=dev)
        if peb_h.any():
            self.z_bed.assign(self.z_bed.numpy() + peb_h)

        # camera sanity: inside the domain footprint and above the bed
        cx, cy, cz = self.camera.pos
        bz = float(self.z_bed.numpy()[
            min(max(int((cx - g.x0) / self.dxr), 0), nxr - 1),
            min(max(int((cy - g.y0) / self.dyr), 0), nyr - 1)])
        if cz <= bz + 0.05:
            raise ValueError(f"camera at z={cz:.2f} is inside the bed "
                             f"(bed z={bz:.2f} at x={cx:.2f}, y={cy:.2f})")

        # subgrid ripples on the render grid, alongshore-mean still depth
        self.zeta = wp.zeros(shape, dtype=FLOAT, device=dev)
        self.ripples = None
        if ripples_enabled:
            spectrum = spectrum or RippleSpectrum()
            rgrid = GridConfig(nx=nxr, ny=nyr, Lx=g.Lx, Ly=g.Ly,
                               x0=g.x0, y0=g.y0)
            zb = self.z_bed.numpy()
            still = np.maximum(eta0 - zb.mean(axis=1), 0.0)
            spectrum = self._resolvable(spectrum, float(still.max()))
            # phase from the alongshore-mean profile (WKB), amplitude taper
            # from the LOCAL depth so ripples die around stones, not along a
            # shore-parallel line where the stone band biases the mean
            self.ripples = RippleField(
                rgrid,
                still,
                spectrum,
                dev,
                still_depth_2d=np.maximum(eta0 - zb, 0.0),
                alongshore_period=ripple_alongshore_period,
            )

        self._ripple_mask = None
        if ripple_fade is not None:
            x_full, x_zero = ripple_fade
            x = g.x0 + (np.arange(nxr) + 0.5) * self.dxr
            s = np.clip((x - x_full) / (x_zero - x_full), 0.0, 1.0)
            fade = 1.0 - s * s * (3.0 - 2.0 * s)
            mask = np.broadcast_to(fade[:, None], shape).astype(np.float32)
            self._ripple_mask = wp.array(
                mask, dtype=FLOAT, device=dev,
            )

        self.z_wat = wp.zeros(shape, dtype=wp.float32, device=dev)
        self.z_top = wp.zeros(shape, dtype=wp.float32, device=dev)
        self.cf_r = wp.zeros(shape, dtype=wp.float32, device=dev)
        self.w_r = wp.zeros(shape, dtype=wp.float32, device=dev)
        self._zero_pad = None
        self.img = wp.zeros((self.camera.height, self.camera.width),
                            dtype=wp.vec3f, device=dev)

        # ---- analytic offshore far field: continue every modelled mode as
        # the exact plane wave it is at the constant offshore depth ---------
        # rows (kx, ky, amp, phase); phase gets its time term per frame
        kx_l, ky_l, a_l, ph_l, hn_l = [], [], [], [], []
        b_col = solver.state.get("b")[0, :]  # first interior column, as the
        h_off = float(eta0 - float(np.mean(b_col)))  # wavemaker defines depth
        # A caller that supplies an offshore model expects the analytic water
        # extension even when every wave amplitude is zero (or every ripple is
        # filtered out).  The mode count controls only normal displacement;
        # flat water still needs Fresnel/refraction shading rather than sky.
        self._ff_on = h_off > 0.0 and (
            forcing is not None
            or offshore_field is not None
            or self.ripples is not None
        )
        self._ff_T = None
        if forcing is not None:
            self._ff_T = float(forcing.period)
        if offshore_field is not None:
            exact_period = float(offshore_field.period)
            if self._ff_T is not None and abs(exact_period - self._ff_T) > 1e-12:
                raise ValueError(
                    "forcing and exact offshore field disagree on the loop "
                    f"period: {self._ff_T} vs {exact_period}",
                )
            self._ff_T = exact_period
        if self.ripples is not None:
            sp_T = float(self.ripples.spectrum.period)
            if self._ff_T is not None and abs(sp_T - self._ff_T) > 1e-12:
                raise ValueError("forcing and ripple spectrum disagree on the "
                                 f"loop period: {self._ff_T} vs {sp_T}")
            self._ff_T = sp_T
        if forcing is not None and h_off > 0.0:
            c0 = float(np.sqrt(solver.cfg.physics.g * h_off))
            for hn, a0, ph, mky in zip(forcing.harmonics, forcing.amplitudes,
                                       forcing.phases, forcing.ky_modes):
                om = 2.0 * np.pi * hn / forcing.period
                kyv = 2.0 * np.pi * mky / g.Ly
                kx2 = (om / c0) ** 2 - kyv * kyv
                if a0 == 0.0 or kx2 <= 0.0:  # silent or evanescent
                    continue
                kxv = float(np.sqrt(kx2))
                # boundary target a cos(om t - ky y + ph) at the face x = x0,
                # continued offshore with the SWE dispersion the solver itself
                # propagates: a cos(kx x + ky y + [-kx x0 - ph] - om t)
                kx_l.append(kxv)
                ky_l.append(kyv)
                a_l.append(float(a0))
                ph_l.append(-kxv * g.x0 - float(ph))
                hn_l.append(float(hn))
        if offshore_field is not None and h_off > 0.0:
            for hn, kxv, kyv, a0, ph in zip(
                offshore_field.harmonics,
                offshore_field.kx,
                offshore_field.ky,
                offshore_field.amplitudes,
                offshore_field.phases,
                strict=True,
            ):
                if a0 == 0.0:
                    continue
                kx_l.append(float(kxv))
                ky_l.append(float(kyv))
                a_l.append(float(a0))
                ph_l.append(float(ph))
                hn_l.append(float(hn))
        n_forcing = len(kx_l)
        ne = 0
        if self.ripples is not None and h_off > 0.0:
            rp = self.ripples
            sp = rp.spectrum
            psi, amp_r, ky_r = (rp._host["psi"], rp._host["amp"],
                                rp._host["ky"])
            xc0 = g.x0 + 0.5 * self.dxr
            for m, (hn, ph) in enumerate(zip(sp.harmonics, sp.phases)):
                a0 = float(amp_r[m, 0])
                if a0 == 0.0:  # unresolvable or dry at the boundary
                    continue
                # constant offshore kx from the first quadrature step; the
                # in-domain phase at the first column is psi[m, 0], so
                # psi(x) = psi[m, 0] + kx0 (x - xc0) is seam-continuous
                kx0 = float(psi[m, 1] - psi[m, 0]) / self.dxr
                kx_l.append(kx0)
                ky_l.append(float(ky_r[m]))
                a_l.append(a0)
                ph_l.append(float(psi[m, 0]) - kx0 * xc0 + float(ph))
                hn_l.append(float(hn))
            # gustiness envelope, same wavevectors as in-domain (taper is 1
            # offshore: the water is far deeper than taper_depth + width)
            ne = rp.n_mod
            if ne:
                self._ffe_kx = rp.mod_kx.numpy()[:ne].astype(np.float64)
                self._ffe_ky = rp.mod_ky.numpy()[:ne].astype(np.float64)
                self._ffe_amp = rp.mod_amp.numpy()[:ne].astype(np.float64)
                self._ffe_ph0 = np.asarray(sp.mod_phases, dtype=np.float64)
                self._ffe_harm = np.asarray(sp.mod_harmonics,
                                            dtype=np.float64)
        self._ff_nm = len(kx_l)
        self._ff_nf = n_forcing
        self._ff_ne = ne if self._ff_nm else 0
        if self._ff_nm:
            self._ff_kx = np.asarray(kx_l, dtype=np.float64)
            self._ff_ky = np.asarray(ky_l, dtype=np.float64)
            self._ff_amp = np.asarray(a_l, dtype=np.float64)
            self._ff_ph0 = np.asarray(ph_l, dtype=np.float64)
            self._ff_harm = np.asarray(hn_l, dtype=np.float64)
        self._ff_dev = wp.zeros(max(self._ff_nm, 1), dtype=wp.vec4f,
                                device=dev)
        self._ffe_dev = wp.zeros(max(self._ff_ne, 1), dtype=wp.vec4f,
                                 device=dev)

        o = self.optics
        s = self.sky
        fwd, right, up = self.camera.basis()
        p = TraceParams()
        p.cam_pos = wp.vec3f(*[float(q) for q in self.camera.pos])
        p.fwd = wp.vec3f(*fwd.astype(np.float32))
        p.right = wp.vec3f(*right.astype(np.float32))
        p.up = wp.vec3f(*up.astype(np.float32))
        p.tanx = self.camera.tan_half_h
        p.tany = self.camera.tan_half_v
        p.x0 = g.x0
        p.y0 = g.y0
        p.inv_dx = 1.0 / self.dxr
        p.inv_dy = 1.0 / self.dyr
        p.dx = self.dxr
        p.dy = self.dyr
        p.nx = nxr
        p.ny = nyr
        p.cell = min(self.dxr, self.dyr)
        p.t_max = float(3.0 * (g.Lx + g.Ly))
        p.z_ceiling = float(self.z_bed.numpy().max() + max(eta0, 0.0) + 0.5)
        p.n_water = o.n_water
        p.kappa = wp.vec3f(*o.kappa)
        p.scatter = wp.vec3f(*o.scatter)
        p.zenith = wp.vec3f(*s.zenith)
        p.horizon = wp.vec3f(*s.horizon)
        p.sky_p = s.p
        p.glow = wp.vec3f(*s.glow)
        p.glow_dir = wp.vec3f(*s.glow_dir().astype(np.float32))
        p.glow_sharp = s.glow_sharpness
        p.e_down = wp.vec3f(*self.sky.e_down())
        p.foam_albedo = o.foam_albedo
        p.wet_dark = wp.vec3f(*o.wet_dark)
        p.alb_mean = wp.vec3f(*alb.reshape(-1, 3).mean(axis=0))

        # directional sun (all terms off when irradiance is zero)
        sn = self.sun
        p.has_sun = 1 if sn.enabled else 0
        if sn.enabled:
            sd = sn.direction()
            if sd[2] < 0.05:
                raise ValueError("sun elevation too low: shadow and "
                                 "refraction geometry degenerate below ~3 deg")
            p.sun_dir = wp.vec3f(*sd.astype(np.float32))
            p.sun_irr = wp.vec3f(*sn.irradiance)
            p.glit_rad = wp.vec3f(*sn.glitter_radiance)
            sig = np.deg2rad(max(sn.glitter_width_deg, 0.1))
            p.glit_inv_sig2 = float(1.0 / (sig * sig))
            # refracted sun direction at a flat interface (per-facet bending
            # of the DIRECT term is part of the deferred caustics machinery)
            cosi = float(sd[2])
            sini = float(np.hypot(sd[0], sd[1]))
            sint = sini / o.n_water
            cost = float(np.sqrt(max(1.0 - sint * sint, 0.0)))
            hn = (np.array([sd[0], sd[1], 0.0]) / max(sini, 1e-9)
                  if sini > 1e-9 else np.array([1.0, 0.0, 0.0]))
            p.tsun_up = wp.vec3f(float(hn[0] * sint), float(hn[1] * sint), cost)
            p.sun_trans = float(1.0 - _fresnel_host(cosi, o.n_water))
            p.inv_cost_sun = float(1.0 / max(cost, 1e-3))
        else:
            p.sun_dir = wp.vec3f(0.0, 0.0, 1.0)
            p.sun_irr = wp.vec3f(0.0, 0.0, 0.0)
            p.glit_rad = wp.vec3f(0.0, 0.0, 0.0)
            p.glit_inv_sig2 = 1.0
            p.tsun_up = wp.vec3f(0.0, 0.0, 1.0)
            p.sun_trans = 0.0
            p.inv_cost_sun = 1.0
        p.shadow_step = 2.0 * p.cell
        p.z_bed_max = float(self.z_bed.numpy().max() + 0.02)

        # analytic offshore far field
        p.ff_on = 1 if self._ff_on else 0
        p.ff_nf = self._ff_nf
        p.ff_nm = self._ff_nm
        p.ff_ne = self._ff_ne
        p.ff_eta0 = float(eta0)
        p.ff_depth = max(h_off, 0.0)
        p.pix_sig = float(2.0 * self.camera.tan_half_v / self.camera.height)
        self.params = p

    def _resolvable(self, spectrum: RippleSpectrum, h_deep: float
                    ) -> RippleSpectrum:
        """Zero the amplitude of modes the render grid cannot sample.

        Displacing geometry we cannot sample is how water starts to sparkle
        like tinfoil (docs/decisions.md): a mode needs >= 4 render cells per
        deep-water wavelength or it aliases into salt-and-pepper normals.
        Those scales belong in BRDF roughness (not implemented yet -- so they
        are simply absent, which is honest)."""
        from shoreloop.closures.ripples import wavenumber
        lam_min = 4.0 * max(self.dxr, self.dyr)
        amps = []
        dropped = []
        for n, a in zip(spectrum.harmonics, spectrum.amplitudes):
            om = 2.0 * np.pi * n / spectrum.period
            k = wavenumber(om, h_deep, spectrum.capillary)
            lam = 2.0 * np.pi / max(k, 1e-9)
            if lam < lam_min:
                amps.append(0.0)
                dropped.append(n)
            else:
                amps.append(a)
        if dropped:
            print(f"  ripples: dropped harmonics {dropped} "
                  f"(wavelength < {100*lam_min:.1f} cm, unresolvable at "
                  f"{1000*max(self.dxr, self.dyr):.1f} mm render cells)")
        from dataclasses import replace
        return replace(spectrum, amplitudes=tuple(amps))

    def _pad_zeros(self):
        if self._zero_pad is None:
            self._zero_pad = wp.zeros(self.solver.cfg.grid.shape, dtype=FLOAT,
                                      device=self.solver.cfg.device)
        return self._zero_pad

    def _far_update(self, t: float) -> None:
        """Fold the time term into the far-field mode phases. Wrapped mod
        2 pi on the host in float64 -- the same discipline as
        RippleField.evaluate and the wavemaker, and for the same reason."""
        if not self._ff_nm:
            return
        frac = np.mod(self._ff_harm * (float(t) / self._ff_T), 1.0)
        ph = np.mod(self._ff_ph0 - 2.0 * np.pi * frac, 2.0 * np.pi)
        self._ff_dev.assign(np.stack(
            [self._ff_kx, self._ff_ky, self._ff_amp, ph],
            axis=-1).astype(np.float32))
        if self._ff_ne:
            frac = np.mod(self._ffe_harm * (float(t) / self._ff_T), 1.0)
            ph = np.mod(self._ffe_ph0 - 2.0 * np.pi * frac, 2.0 * np.pi)
            self._ffe_dev.assign(np.stack(
                [self._ffe_kx, self._ffe_ky, self._ffe_amp, ph],
                axis=-1).astype(np.float32))

    def render(self, t: float | None = None) -> np.ndarray:
        """Render the solver's current state -> (H, W, 3) float32 linear."""
        s = self.solver
        g = s.cfg.grid
        if t is None:
            t = s.state.t
        if self.ripples is not None:
            self.ripples.evaluate(t, out=self.zeta)
            if self._ripple_mask is not None:
                wp.launch(
                    apply_scalar_mask,
                    dim=(self.nxr, self.nyr),
                    inputs=[self.zeta, self._ripple_mask],
                    device=s.cfg.device,
                )
        else:
            self.zeta.zero_()
        self._far_update(t)
        cf = s.cf if getattr(s, "_closures_on", False) \
            and s.cfg.closures.foam_enabled else self._pad_zeros()
        w = s.w if getattr(s, "_wetness", False) else self._pad_zeros()
        scale_inv = 1.0 / (self.nxr // g.nx)
        wp.launch(prep_fields, dim=(self.nxr, self.nyr),
                  inputs=[s.state.h, s.state.b, cf, w, self.zeta,
                          self.z_bed, self.z_wat, self.z_top,
                          self.cf_r, self.w_r,
                          g.ng, g.nx, g.ny, scale_inv, scale_inv,
                          s.cfg.physics.h_wet_recon, 2.0e-3],
                  device=s.cfg.device)
        wp.launch(trace, dim=(self.camera.height, self.camera.width),
                  inputs=[self.z_top, self.z_wat, self.z_bed, self.albedo,
                          self.w_r, self.cf_r, self._ff_dev, self._ffe_dev,
                          self.params, self.img],
                  device=s.cfg.device)
        return self.img.numpy()

    def frame(self, t: float | None = None) -> np.ndarray:
        """Render -> display uint8 via the camera's exposure/WB/sRGB."""
        return tonemap(self.render(t), self.camera)
