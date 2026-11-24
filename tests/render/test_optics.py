"""Gate 13 (partial): Snell, Fresnel, Beer-Lambert vs analytic cases.

Two levels. The Fresnel device function is checked directly against the
textbook formulas. Then the *whole* pipeline -- camera rays, height-field
marching, the Fresnel split, Snell refraction, Beer-Lambert on both light
paths -- is checked on a flat-water scene where every pixel has a closed-form
expected radiance. That is deliberately an integration test: it is the same
kind of end-to-end check that caught real bugs in the hydro core (a per-stage
test cannot see a sign error in how stages compose).
"""

import numpy as np
import pytest
import warp as wp

from shoreloop.config import BC, BCConfig, GridConfig, PhysicsConfig, SolverConfig
from shoreloop.device import DEVICE
from shoreloop.physics.swe import SWESolver
from shoreloop.render.camera import Camera
from shoreloop.render.raytrace import Raytracer, WaterOptics, _fresnel
from shoreloop.render.sky import OvercastSky, Sun
from shoreloop.render.texture import PebbleTexture
from shoreloop.scene.forcing import WaveForcing

FLAT_TEX = PebbleTexture(style="tiles")  # zero height: analytic scenes stay flat


# ---------------------------------------------------------------------------
# analytic references (numpy, float64)
# ---------------------------------------------------------------------------
def fresnel_ref(cosi, n):
    """Unpolarized dielectric reflectance from air, exact."""
    sini = np.sqrt(np.maximum(1.0 - cosi**2, 0.0))
    sint = sini / n
    cost = np.sqrt(np.maximum(1.0 - sint**2, 0.0))
    rs = (cosi - n * cost) / (cosi + n * cost)
    rp = (n * cosi - cost) / (n * cosi + cost)
    return 0.5 * (rs**2 + rp**2)


@wp.kernel
def _eval_fresnel(cosi: wp.array(dtype=wp.float32), eta: wp.float32,
                  out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = _fresnel(cosi[i], eta)


def test_fresnel_against_analytic():
    n = 1.333
    theta = np.deg2rad(np.array([0.0, 20.0, 40.0, 53.12, 60.0, 75.0, 85.0, 89.0]))
    cosi = np.cos(theta).astype(np.float32)
    out = wp.zeros(len(cosi), dtype=wp.float32, device=DEVICE)
    wp.launch(_eval_fresnel, dim=len(cosi),
              inputs=[wp.array(cosi, dtype=wp.float32, device=DEVICE),
                      n, out], device=DEVICE)
    got = out.numpy()
    ref = fresnel_ref(np.cos(theta), n)
    assert np.allclose(got, ref, rtol=0, atol=2e-6), f"{got} vs {ref}"
    # anchor values, not just self-consistency:
    assert abs(got[0] - ((n - 1) / (n + 1)) ** 2) < 1e-6   # normal incidence
    # Brewster: Rp = 0, so R = Rs^2 / 2 exactly
    tb = np.arctan(n)
    sb = np.sin(tb) / n
    cb = np.sqrt(1 - sb**2)
    rs = (np.cos(tb) - n * cb) / (np.cos(tb) + n * cb)
    assert abs(fresnel_ref(np.cos(tb), n) - 0.5 * rs**2) < 1e-12
    assert got[-1] > 0.8, "grazing incidence must approach total reflection"


def test_sky_irradiance_matches_closed_form():
    """e_down is numerical quadrature (the glow term has no closed form);
    validate the integrator against the exact closed form of the glow-free
    base: E = 2 pi (h/2 + (z - h)/(p + 2))."""
    sky = OvercastSky()  # glow off by default
    z = np.asarray(sky.zenith)
    h = np.asarray(sky.horizon)
    closed = 2.0 * np.pi * (h / 2.0 + (z - h) / (sky.p + 2.0))
    assert np.allclose(sky.e_down(), closed, rtol=1e-3), \
        f"quadrature {sky.e_down()} vs closed form {closed}"
    # and glow must strictly add energy
    glow_sky = OvercastSky(glow=(1.0, 1.0, 1.0))
    assert (glow_sky.e_down() > sky.e_down() + 1e-4).all()


# ---------------------------------------------------------------------------
# full pipeline on a flat scene with a closed-form answer
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def flat_render():
    depth = 0.5
    g = GridConfig(nx=64, ny=32, Lx=6.4, Ly=3.2)
    cfg = SolverConfig(grid=g, physics=PhysicsConfig(friction_enabled=False),
                       bc=BCConfig(BC.WALL, BC.WALL, BC.PERIODIC, BC.PERIODIC),
                       device=DEVICE)
    s = SWESolver(cfg)
    shape = (g.nx, g.ny)
    s.set_bathymetry(np.full(shape, -depth, dtype=np.float32))
    s.set_water(np.full(shape, depth, dtype=np.float32))

    cam = Camera(pos=(4.8, 1.6, 1.0), yaw_deg=180.0, pitch_deg=40.0,
                 vfov_deg=40.0, width=64, height=48)
    rt = Raytracer(s, camera=cam, ripples_enabled=False, texture=FLAT_TEX)
    # uniform albedo: this test isolates the optics machinery; the pebble
    # texture would otherwise dominate the error budget (bilinear in-kernel
    # vs any host-side resampling of a +/-45% pattern)
    rt.albedo.assign(wp.array(np.full((rt.nxr, rt.nyr, 3), 0.3, dtype=np.float32),
                              dtype=wp.vec3f, device=DEVICE))
    img = rt.render(0.0)
    return rt, cam, img, depth


def analytic_flat(cam: Camera, sky: OvercastSky, opt: WaterOptics,
                  albedo: np.ndarray, depth: float) -> np.ndarray:
    """Closed-form radiance for flat water at z=0 over a flat bed, uniform
    albedo. Mirrors the renderer's model equation by construction -- the test
    is that the *kernel machinery* (rays, marching, splitting, path lengths)
    realises this equation."""
    d = cam.ray_dirs()                      # (H, W, 3), all pointing down
    cosi = -d[..., 2]
    n = opt.n_water
    R = fresnel_ref(cosi, n)
    refl = d.copy()
    refl[..., 2] = -refl[..., 2]
    Lr = sky.radiance(refl)

    sint = np.sqrt(1 - cosi**2) / n
    cost = np.sqrt(1 - sint**2)
    path = depth / cost
    kappa = np.asarray(opt.kappa)
    e_down = sky.e_down()
    ed_bed = e_down * np.exp(-kappa * depth)         # downwelling at the bed
    lbed = albedo * np.asarray(opt.wet_dark) * ed_bed / np.pi  # sv = 1 flat
    trans = np.exp(-kappa * path[..., None])
    Lt = lbed * trans + np.asarray(opt.scatter) * (1 - trans)
    return R[..., None] * Lr + (1 - R[..., None]) * Lt


def test_flat_scene_matches_closed_form(flat_render):
    rt, cam, img, depth = flat_render
    ref = analytic_flat(cam, rt.sky, rt.optics,
                        np.full(3, 0.3, dtype=np.float64), depth)

    err = np.abs(img - ref)
    med = np.median(err)
    assert med < 2e-3, f"median abs radiance error {med:.2e}"
    frac_bad = float((err > 0.01).mean())
    assert frac_bad < 0.01, \
        f"{100*frac_bad:.1f}% of pixels off by > 0.01 linear radiance"
    # and the Fresnel-driven vertical ramp must match the analytic one (its
    # magnitude depends on camera pitch and depth; for this 20-60 degree
    # incidence span it is modest -- the reference video's 4.2x needs grazing)
    def row_ramp(a):
        luma = 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]
        rows = luma.mean(axis=1)
        return float(rows[0] / rows[-1])

    got, want = row_ramp(img), row_ramp(ref)
    assert want > 1.05, "test scene has no ramp at all; camera badly chosen"
    assert abs(got - want) / want < 0.03, \
        f"grazing-to-steep ramp {got:.3f} vs analytic {want:.3f}"


def test_dry_scene_has_no_water_term():
    """All-dry bed: the image must be pure pebble shading -- no Fresnel, no
    scatter tint, nothing blue. Guards the water/bed classification."""
    g = GridConfig(nx=64, ny=32, Lx=6.4, Ly=3.2)
    cfg = SolverConfig(grid=g, physics=PhysicsConfig(friction_enabled=False),
                       bc=BCConfig(BC.WALL, BC.WALL, BC.PERIODIC, BC.PERIODIC),
                       device=DEVICE)
    s = SWESolver(cfg)
    shape = (g.nx, g.ny)
    s.set_bathymetry(np.full(shape, 0.2, dtype=np.float32))
    s.set_water(np.zeros(shape, dtype=np.float32))
    cam = Camera(pos=(4.8, 1.6, 1.2), yaw_deg=180.0, pitch_deg=40.0,
                 vfov_deg=40.0, width=64, height=48)
    rt = Raytracer(s, camera=cam, ripples_enabled=False, texture=FLAT_TEX)
    img = rt.render(0.0)

    e_down = rt.sky.e_down()
    alb = rt.albedo.numpy()
    lo = alb.reshape(-1, 3).min(axis=0) * 0.5 * e_down / np.pi * 0.98
    hi = alb.reshape(-1, 3).max(axis=0) * 1.0 * e_down / np.pi * 1.02
    assert (img >= lo * 0.9).all() and (img <= hi * 1.1).all(), \
        "dry-bed radiance outside pure-Lambertian bounds: water term leaked"


def test_far_field_matches_closed_form():
    """The analytic offshore far field: a grazing camera whose upper rays
    leave the domain must shade the normal-mapped still-water plane -- one
    forcing mode continued offshore, footprint-prefiltered, with the lost
    slope variance in the 3-point visibility-weighted quadrature. The
    reference mirrors the model equation by construction; the test is that
    the kernel machinery (plane intersection, mode phase, filter, quadrature
    weights) realises it."""
    depth = 0.5
    g = GridConfig(nx=64, ny=32, Lx=6.4, Ly=3.2)
    cfg = SolverConfig(grid=g, physics=PhysicsConfig(friction_enabled=False),
                       bc=BCConfig(BC.WALL, BC.WALL, BC.PERIODIC, BC.PERIODIC),
                       device=DEVICE)
    s = SWESolver(cfg)
    shape = (g.nx, g.ny)
    s.set_bathymetry(np.full(shape, -depth, dtype=np.float32))
    s.set_water(np.full(shape, depth, dtype=np.float32))

    forcing = WaveForcing(period=8.0, harmonics=(2,), amplitudes=(0.02,),
                          phases=(0.7,), eta0=0.0)
    cam = Camera(pos=(5.8, 1.6, 1.0), yaw_deg=180.0, pitch_deg=20.0,
                 vfov_deg=30.0, width=64, height=48)
    rt = Raytracer(s, camera=cam, ripples_enabled=False, texture=FLAT_TEX,
                   forcing=forcing)
    t_eval = 0.3
    img = rt.render(t_eval)

    # ---- reference, float64 -------------------------------------------
    opt, sky = rt.optics, rt.sky
    nw = opt.n_water
    kappa = np.asarray(opt.kappa)
    alb_mean = np.array([rt.params.alb_mean[i] for i in range(3)])
    om = 2.0 * np.pi * 2 / 8.0
    kx = om / np.sqrt(9.81 * depth)          # shore-normal: ky = 0
    phase = np.mod(-kx * g.x0 - 0.7 - om * t_eval, 2.0 * np.pi)

    d = cam.ray_dirs()
    o = np.asarray(cam.pos)
    down = (d[..., 0] < -1e-6) & (d[..., 2] < -1e-4)
    t_edge = (g.x0 - o[0]) / np.where(down, d[..., 0], -1.0)
    t_pl = np.maximum(-o[2] / np.where(down, d[..., 2], -1.0), t_edge)
    # a far pixel's still-water crossing lands beyond the offshore edge (with
    # a small margin: at the edge itself march bisection vs the exact plane
    # legitimately differ by a cell)
    far = down & (o[0] + t_pl * d[..., 0] < g.x0 - 0.05)
    assert far.mean() > 0.15, "test camera exercises no far-field pixels"
    F = o[None, None, :] + t_pl[..., None] * d
    pix = 2.0 * cam.tan_half_v / cam.height
    sig_y = 0.7 * t_pl * pix
    sig_x = sig_y / np.maximum(-d[..., 2], 1e-3)
    q = 0.5 * (kx * sig_x) ** 2
    f = np.where(q < 9.0, np.exp(-q), 0.0)
    zx = -kx * 0.02 * f * np.sin(kx * F[..., 0] + phase)
    sig = np.sqrt(0.5 * 0.02 ** 2 * kx ** 2 * (1.0 - f * f))

    e_down = sky.e_down()
    irr_bed = e_down * np.exp(-kappa * depth)
    lbed = alb_mean * np.asarray(opt.wet_dark) * irr_bed / np.pi
    L = np.zeros(d.shape)
    wsum = np.zeros(t_pl.shape)
    for node, w0 in ((-np.sqrt(3.0), 1.0 / 6.0), (0.0, 2.0 / 3.0),
                     (np.sqrt(3.0), 1.0 / 6.0)):
        n = np.stack([-(zx + node * sig), np.zeros_like(zx),
                      np.ones_like(zx)], axis=-1)
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
        cosi_raw = -np.einsum("...i,...i->...", d, n)
        w = w0 * np.maximum(cosi_raw, 0.0)
        cosi = np.maximum(cosi_raw, 1e-4)
        R = fresnel_ref(cosi, nw)
        refl = d + 2.0 * cosi[..., None] * n
        Lr = sky.radiance(refl)
        ir = 1.0 / nw
        cost = np.sqrt(np.maximum(1.0 - ir**2 * (1.0 - cosi**2), 0.0))
        td = ir * d + (ir * cosi - cost)[..., None] * n
        td /= np.linalg.norm(td, axis=-1, keepdims=True)
        down = td[..., 2] < -1e-4
        path = depth / np.maximum(-td[..., 2], 1e-4)
        trans = np.exp(-kappa * path[..., None])
        Lt = lbed * trans + np.asarray(opt.scatter) * (1.0 - trans)
        R = np.where(down, R, 1.0)
        L += w[..., None] * (R[..., None] * Lr
                             + (1.0 - R[..., None]) * np.where(
                                 down[..., None], Lt, 0.0))
        wsum += w
    ref = L / np.maximum(wsum, 1e-12)[..., None]

    err = np.abs(img - ref)[far]
    assert np.median(err) < 3e-3, f"median far-field error {np.median(err):.2e}"
    assert float((err > 0.02).mean()) < 0.02, \
        f"{100 * (err > 0.02).mean():.1f}% of far pixels off by > 0.02"
    # the mode must actually be visible: a shore-normal wave modulates the
    # far field across ROWS -- detrend the far-row luma profile and demand
    # residual oscillation above the noise floor
    luma = (0.2126 * img[..., 0] + 0.7152 * img[..., 1]
            + 0.0722 * img[..., 2])
    row_far = far.all(axis=1)
    assert row_far.sum() >= 8
    prof = luma[row_far].mean(axis=1)
    i = np.arange(len(prof))
    resid = prof - np.polyval(np.polyfit(i, prof, 2), i)
    assert np.ptp(resid) > 2e-3, \
        f"far field shows no wave modulation (residual ptp {np.ptp(resid):.1e})"


def test_flat_far_field_without_wave_modes_matches_closed_form():
    """An explicitly still offshore model is still water, not empty sky."""
    depth = 0.5
    g = GridConfig(nx=64, ny=32, Lx=6.4, Ly=3.2)
    cfg = SolverConfig(grid=g, physics=PhysicsConfig(friction_enabled=False),
                       bc=BCConfig(BC.WALL, BC.WALL, BC.PERIODIC, BC.PERIODIC),
                       device=DEVICE)
    s = SWESolver(cfg)
    shape = (g.nx, g.ny)
    s.set_bathymetry(np.full(shape, -depth, dtype=np.float32))
    s.set_water(np.full(shape, depth, dtype=np.float32))

    forcing = WaveForcing(period=8.0, harmonics=(1,), amplitudes=(0.0,),
                          phases=(0.0,), eta0=0.0)
    cam = Camera(pos=(5.8, 1.6, 1.0), yaw_deg=180.0, pitch_deg=20.0,
                 vfov_deg=30.0, width=64, height=48)
    rt = Raytracer(s, camera=cam, ripples_enabled=False, texture=FLAT_TEX,
                   forcing=forcing)
    img = rt.render(0.0)

    d = cam.ray_dirs()
    o = np.asarray(cam.pos)
    down = (d[..., 0] < -1e-6) & (d[..., 2] < -1e-4)
    t_pl = -o[2] / np.where(down, d[..., 2], -1.0)
    far = down & (o[0] + t_pl * d[..., 0] < g.x0 - 0.05)
    assert far.mean() > 0.15, "test camera exercises no far-field pixels"

    alb_mean = np.array([rt.params.alb_mean[i] for i in range(3)])
    ref = analytic_flat(cam, rt.sky, rt.optics, alb_mean, depth)
    err = np.abs(img - ref)[far]
    assert np.median(err) < 3e-3, f"median flat far-field error {np.median(err):.2e}"
    assert float((err > 0.02).mean()) < 0.02, \
        f"{100 * (err > 0.02).mean():.1f}% of flat far pixels off by > 0.02"


# ---------------------------------------------------------------------------
# sun: glint, transmitted direct light, shadows
# ---------------------------------------------------------------------------
def test_flat_scene_sun_matches_closed_form():
    """Flat water + directional sun: filtered glint and the transmitted
    direct term at the bed, both against closed forms. Flat bed => no
    occluders => shadow factor must be exactly 1 everywhere."""
    depth = 0.5
    g = GridConfig(nx=64, ny=32, Lx=6.4, Ly=3.2)
    cfg = SolverConfig(grid=g, physics=PhysicsConfig(friction_enabled=False),
                       bc=BCConfig(BC.WALL, BC.WALL, BC.PERIODIC, BC.PERIODIC),
                       device=DEVICE)
    s = SWESolver(cfg)
    shape = (g.nx, g.ny)
    s.set_bathymetry(np.full(shape, -depth, dtype=np.float32))
    s.set_water(np.full(shape, depth, dtype=np.float32))
    cam = Camera(pos=(4.8, 1.6, 1.0), yaw_deg=180.0, pitch_deg=40.0,
                 vfov_deg=40.0, width=64, height=48)
    sun = Sun(azimuth_deg=180.0, elevation_deg=25.0,
              irradiance=(5.0, 4.0, 3.0), glitter_radiance=(40.0, 30.0, 20.0),
              glitter_width_deg=3.0)
    rt = Raytracer(s, camera=cam, ripples_enabled=False, texture=FLAT_TEX,
                   sun=sun)
    alb0 = 0.3
    rt.albedo.assign(wp.array(np.full((rt.nxr, rt.nyr, 3), alb0,
                                      dtype=np.float32),
                              dtype=wp.vec3f, device=DEVICE))
    img = rt.render(0.0)

    # closed form
    n = rt.optics.n_water
    d = cam.ray_dirs()
    cosi = -d[..., 2]
    R = fresnel_ref(cosi, n)
    refl = d.copy()
    refl[..., 2] = -refl[..., 2]
    Lr = rt.sky.radiance(refl)
    sd = sun.direction()

    # glint: angular Gaussian around the mirror direction
    sig = np.deg2rad(sun.glitter_width_deg)
    cg = np.clip(np.tensordot(refl, sd, axes=([-1], [0])), -1.0, 1.0)
    glint = np.exp(-(1.0 - cg) / sig**2)[..., None] \
        * np.asarray(sun.glitter_radiance)

    # transmitted paths
    ir = 1.0 / n
    cost = np.sqrt(1 - ir**2 * (1 - cosi**2))
    path = depth / cost
    kappa = np.asarray(rt.optics.kappa)
    # sun through the interface
    cosi_sun = sd[2]
    R_sun = fresnel_ref(np.array([cosi_sun]), n)[0]
    sint_sun = np.sqrt(1 - cosi_sun**2) / n
    cost_sun = np.sqrt(1 - sint_sun**2)
    e_dir = (1 - R_sun) * cost_sun * np.asarray(sun.irradiance) \
        * np.exp(-kappa * depth / cost_sun)
    irr_bed = rt.sky.e_down() * np.exp(-kappa * depth) + e_dir
    lbed = alb0 * np.asarray(rt.optics.wet_dark) * irr_bed / np.pi
    trans = np.exp(-kappa * path[..., None])
    Lt = lbed * trans + np.asarray(rt.optics.scatter) * (1 - trans)
    ref = R[..., None] * (Lr + glint) + (1 - R[..., None]) * Lt

    err = np.abs(img - ref)
    assert np.median(err) < 3e-3, f"median abs error {np.median(err):.2e}"
    assert float((err > 0.02).mean()) < 0.02
    # the glint corridor must actually be present and bright
    assert img.max() > 1.5, "no glint highlight in a sun-lit flat scene"


def test_sun_shadows_are_cast_by_the_bed():
    """A raised ridge under a low sun: the strip behind it must fall back to
    sky-only irradiance while open ground gets sky + sun."""
    g = GridConfig(nx=128, ny=32, Lx=6.4, Ly=1.6)
    cfg = SolverConfig(grid=g, physics=PhysicsConfig(friction_enabled=False),
                       bc=BCConfig(BC.WALL, BC.WALL, BC.PERIODIC, BC.PERIODIC),
                       device=DEVICE)
    s = SWESolver(cfg)
    x, _ = np.meshgrid((np.arange(g.nx) + 0.5) * g.dx,
                       (np.arange(g.ny) + 0.5) * g.dy, indexing="ij")
    b = np.where((x > 3.0) & (x < 3.2), 0.7, 0.2).astype(np.float32)
    s.set_bathymetry(b)
    s.set_water(np.zeros_like(b))
    cam = Camera(pos=(5.6, 0.8, 2.2), yaw_deg=180.0, pitch_deg=55.0,
                 vfov_deg=45.0, width=96, height=64)
    sun = Sun(azimuth_deg=180.0, elevation_deg=15.0,
              irradiance=(6.0, 6.0, 6.0))
    rt = Raytracer(s, camera=cam, ripples_enabled=False, texture=FLAT_TEX,
                   sun=sun)
    alb0 = 0.3
    rt.albedo.assign(wp.array(np.full((rt.nxr, rt.nyr, 3), alb0,
                                      dtype=np.float32),
                              dtype=wp.vec3f, device=DEVICE))
    img = rt.render(0.0)
    luma = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]

    e_sky = rt.sky.e_down()
    sd = sun.direction()
    lum = lambda v: 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2]
    dark = alb0 * lum(e_sky) / np.pi                      # sky only, sv = 1
    lit = alb0 * lum(e_sky + np.asarray(sun.irradiance) * sd[2]) / np.pi

    # the image must contain two distinct flat-ground populations at the
    # predicted radiances: sunlit (sky + sun) and shadowed (sky only). A
    # quantile test would instead catch the ridge's vertical faces, whose
    # sky-view factor makes them darker than any shadow.
    # at 15 degrees elevation the 0.5 m ridge shades ~1.9 m of the visible
    # ground, so the SHADOW population dominates this camera's view
    frac_shadow = float((np.abs(luma - dark) / dark < 0.08).mean())
    frac_lit = float((np.abs(luma - lit) / lit < 0.08).mean())
    assert frac_shadow > 0.25, \
        f"no shadowed-ground population at {dark:.3f} (frac {frac_shadow:.3f})"
    assert frac_lit > 0.05, \
        f"no sunlit-ground population at {lit:.3f} (frac {frac_lit:.3f})"
    # and without the sun there must be no shadow population distinct from lit
    assert dark < 0.75 * lit
