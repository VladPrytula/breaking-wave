"""Subgrid capillary-gravity ripples: the unresolved surface roughness closure.

The rendered surface is

    z_s(x, y, t) = eta(x, y, t) + zeta(x, y, t)

with `eta` resolved hydrodynamics and `zeta` this closure. Measuring the
reference video (docs/reference-video.md) puts this second only to Fresnel
reflection in what makes the surface read as water: without ripple texture the far
water is a flat blue-grey wash and looks like plastic.

## The formulation, and why it is not the one in plan.md

`plan.md` section 6 asks for modes obeying

    omega^2 = (g k + (sigma/rho) k^3) tanh(k h)

but writes `zeta = sum a_m cos(k_m . x - omega_m t + phi_m)` with a single global
wavevector. Since `h = h(x, y)`, the `k` solving that relation is a *function of
position*; a global `k_m` either violates the dispersion relation or assumes
constant depth. Across a surf zone where `h` varies by an order of magnitude that
is not a small error.

Instead we use a precomputed WKB phase field:

    zeta(x, t) = sum_m a_m(x) cos( Psi_m(x) - omega_m t + phi_m )
    grad Psi_m = k_m(x),   k_m solved from the dispersion relation at local h(x)
    omega_m = 2 pi n_m / T,  n_m integer

Two properties fall out. The dispersion relation now holds *locally*. And
periodicity is exact for any `Psi_m`, because only `omega_m t` carries time and
`omega_m T = 2 pi n_m` -- so `zeta(x, 0) = zeta(x, T)` identically, to roundoff,
with no cross-fade.

## How Psi is obtained cheaply

For an alongshore-uniform bed, `h = h(x)`, Snell's law makes the alongshore
wavenumber `k_y` a conserved constant along each ray.  On the solver's
periodic alongshore domain it is selected from the admissible harmonics
`2 pi q / Ly`. So

    Psi_m(x, y) = integral_0^x k_x(x') dx'  +  k_y y,
    k_x = sqrt(max(0, k(x)^2 - k_y^2))

which is a 1-D quadrature per mode, not an eikonal PDE solve, and needs only two
1-D arrays per mode instead of two 2-D ones. `k_x -> 0` at the turning point where
the mode refracts to shore-parallel, which is the correct behaviour.

Approximations, stated: the bed here has a small alongshore undulation, which this
ignores; and `Psi` is built on the *still-water* depth rather than the
instantaneous one, which is the WKB assumption that the ripples are fast and short
compared with the resolved waves carrying them.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import warp as wp

from shoreloop.config import GridConfig
from shoreloop.precision import FLOAT, HALF, NP_FLOAT, ONE, ZERO

G = 9.81
#: surface tension / density for water at ~20 C [m^3/s^2]
SIGMA_OVER_RHO = 7.28e-5


def wavenumber(omega: float, h: float, capillary: bool = True,
               tol: float = 1e-12, iters: int = 60) -> float:
    """Solve omega^2 = (g k + s k^3) tanh(k h) for k > 0, by Newton.

    Initial guess is the Eckart/Guo approximation to the gravity-only root, which
    is accurate enough everywhere that Newton converges in a handful of steps.
    """
    if h <= 0.0:
        return 0.0
    s = SIGMA_OVER_RHO if capillary else 0.0
    w2 = omega * omega
    # Eckart initial guess
    k = w2 / (G * np.sqrt(np.tanh(w2 * h / G))) if w2 * h / G > 0 else w2 / G
    k = max(k, 1e-6)
    for _ in range(iters):
        th = np.tanh(k * h)
        ch = np.cosh(min(k * h, 350.0))
        sech2 = 1.0 / (ch * ch)
        f = (G * k + s * k ** 3) * th - w2
        df = (G + 3.0 * s * k * k) * th + (G * k + s * k ** 3) * h * sech2
        step = f / df
        k_new = k - step
        if k_new <= 0:
            k_new = 0.5 * k
        if abs(k_new - k) < tol * max(k, 1.0):
            k = k_new
            break
        k = k_new
    return float(k)


@dataclass
class RippleSpectrum:
    """Exactly T-periodic capillary-gravity ripple closure.

    harmonics:  integer multiples of the loop frequency; omega = 2 pi n / T
    amplitudes: deep-water amplitude per mode [m]
    directions: requested deep-water propagation angle from shore-normal
                [rad]; ``RippleField`` snaps its conserved alongshore
                wavenumber to the nearest periodic-domain harmonic
    phases:     per-mode phase offsets [rad]
    shoal_exp:  amplitude ~ (k/k_deep)^shoal_exp. 0.25 approximates energy-flux
                conservation for shoaling short waves; it is a closure knob.
    taper_depth: ripples are exactly zero below this depth [m]; short waves
                cannot exist in a film thinner than their own amplitude.
    taper_width: smoothstep band [m] over which the amplitude rises from 0 at
                taper_depth to full at taper_depth + taper_width. This width
                is what stops the rippled/flat transition being a hard line
                across the frame -- with the old 8 cm band on a steep profile
                the cutoff was visually a shore-parallel edge.

    ## Gustiness envelope

    Real ripple fields are patchy: wind gusts raise and lower the local
    amplitude on metre scales. A statistically uniform spectrum reads as
    synthetic even at the correct RMS contrast. The envelope is a closure for
    that unresolved gust field: a few slow travelling modulation waves,

        env(x, t) = 1 - sum_e A_e (1 + cos(K_e . x - 2 pi n_e t / T + phi_e))/2

    with integer temporal harmonics n_e, so zeta stays exactly T-periodic,
    and the alongshore component of each K_e snapped to a domain harmonic so
    the envelope wraps exactly at the periodic seam. sum A_e < 1, so the
    envelope is strictly positive.

    mod_amps:        modulation depth per envelope mode (sum < 1)
    mod_harmonics:   integer multiples of the loop frequency
    mod_wavelengths: envelope wavelength [m]
    mod_directions:  envelope propagation angle from shore-normal [rad]
    mod_phases:      envelope phase offsets [rad]
    """

    period: float = 8.0
    harmonics: tuple[int, ...] = (24, 31, 37, 43, 53, 61, 71, 83)
    # Amplitudes chosen for a per-mode deep-water slope a*k of ~0.13: real
    # lake ripples have RMS slopes of order 0.1-0.2, and steeper defaults (the
    # first attempt used up to 0.4) visibly destroy the grazing Fresnel
    # reflection that dominates the reference's far water. The gustiness
    # envelope then varies the local amplitude around this mean.
    amplitudes: tuple[float, ...] = (0.0036, 0.0022, 0.0015, 0.0011,
                                     0.00074, 0.00056, 0.00041, 0.0003)
    directions: tuple[float, ...] = (0.0, 0.35, -0.28, 0.55,
                                     -0.5, 0.75, -0.7, 0.2)
    phases: tuple[float, ...] = (0.0, 1.7, 3.1, 0.6, 4.4, 2.2, 5.3, 1.1)
    capillary: bool = True
    shoal_exp: float = 0.25
    taper_depth: float = 0.02
    taper_width: float = 0.25

    mod_amps: tuple[float, ...] = (0.40, 0.30)
    mod_harmonics: tuple[int, ...] = (1, 2)
    mod_wavelengths: tuple[float, ...] = (2.9, 1.7)
    mod_directions: tuple[float, ...] = (0.55, -1.05)
    mod_phases: tuple[float, ...] = (0.0, 2.1)

    @classmethod
    def sample(cls, n_modes: int = 56, period: float = 8.0,
               band: tuple[int, int] = (16, 88), rms_slope: float = 0.15,
               spread: float = 0.45, cross_frac: float = 0.25,
               cross_dir: float = 1.1, seed: int = 7,
               **kw) -> "RippleSpectrum":
        """A dense, statistically sampled spectrum.

        A handful of hand-picked modes renders as visibly *structured*
        interference -- regular crest trains and moire. A real wind-forced
        surface is a continuum: many random free modes. Sampling ~50+ modes
        with random integer harmonics in `band`, random phases, directions
        drawn from a wind-spread distribution (a fraction `cross_frac` from a
        second, crossing wind system at `cross_dir` -- the "multiple external
        forces"), and amplitudes a_m = s_m / k_m with per-mode slopes summing
        to `rms_slope` in RMS, gives a Gaussian-looking sea surface with the
        same exact T-periodicity and the same WKB refraction machinery.

        Harmonics are sampled uniformly in frequency across `band` with
        duplicates allowed (two modes on one frequency with different
        directions are still independent modes).
        """
        rng = np.random.default_rng(seed)
        harm = np.sort(rng.integers(band[0], band[1] + 1, size=n_modes))
        n_cross = int(round(cross_frac * n_modes))
        dirs = rng.normal(0.0, spread, size=n_modes)
        idx = rng.choice(n_modes, size=n_cross, replace=False)
        dirs[idx] = rng.normal(cross_dir, 0.6 * spread, size=n_cross)
        dirs = np.clip(dirs, -1.45, 1.45)  # keep every mode shoreward-going
        phases = rng.uniform(0.0, 2.0 * np.pi, size=n_modes)
        # per-mode deep-water slope, equal-energy split with mild jitter
        s_m = rms_slope * np.sqrt(2.0 / n_modes) \
            * rng.uniform(0.6, 1.4, size=n_modes)
        omegas = 2.0 * np.pi * harm / period
        amps = np.array([s / max(wavenumber(w, 10.0, True), 1e-9)
                         for s, w in zip(s_m, omegas)])
        return cls(period=period, harmonics=tuple(int(n) for n in harm),
                   amplitudes=tuple(float(a) for a in amps),
                   directions=tuple(float(d) for d in dirs),
                   phases=tuple(float(p) for p in phases), **kw)

    def __post_init__(self):
        n = len(self.harmonics)
        if not (len(self.amplitudes) == len(self.directions)
                == len(self.phases) == n):
            raise ValueError("harmonics, amplitudes, directions and phases "
                             "must be the same length")
        if any(int(h) != h or h < 1 for h in self.harmonics):
            raise ValueError("harmonics must be positive integers, or zeta is "
                             "not T-periodic")
        m = len(self.mod_amps)
        if not (len(self.mod_harmonics) == len(self.mod_wavelengths)
                == len(self.mod_directions) == len(self.mod_phases) == m):
            raise ValueError("mod_* tuples must be the same length")
        if any(int(h) != h or h < 1 for h in self.mod_harmonics):
            raise ValueError("mod_harmonics must be positive integers, or the "
                             "envelope breaks T-periodicity")
        if sum(self.mod_amps) >= 1.0:
            raise ValueError("sum(mod_amps) must be < 1 so the envelope stays "
                             "positive")

    @property
    def omegas(self) -> np.ndarray:
        return 2.0 * np.pi * np.asarray(self.harmonics, dtype=np.float64) / self.period

    def wavelengths(self, depth: float) -> np.ndarray:
        return np.array([2 * np.pi / max(wavenumber(w, depth, self.capillary), 1e-9)
                         for w in self.omegas])

    def rms_slope(self, depth: float) -> float:
        """Linear-theory RMS slope before taper and gust modulation."""
        k = 2.0 * np.pi / self.wavelengths(depth)
        mode_slopes = np.asarray(self.amplitudes) * k
        return float(np.sqrt(0.5 * np.sum(mode_slopes * mode_slopes)))


def crossing_ripple_spectrum(
    period: float,
    *,
    frequency_band_hz: tuple[float, float] = (3.0, 7.0),
    rms_slope: float = 0.20,
    spread: float = 0.36,
    cross_frac: float = 0.40,
    cross_dir: float = 1.1,
    seed: int = 707,
    direction_seed: int = 1104,
) -> RippleSpectrum:
    """Build the dense two-lobe production ripple spectrum.

    Harmonics, amplitudes, and phases retain the established sampled-spectrum
    realization.  A separate deterministic draw owns directions so changing
    lobe balance or spread cannot silently change those other fields.
    """
    band = (
        int(round(frequency_band_hz[0] * period)),
        int(round(frequency_band_hz[1] * period)),
    )
    spectrum = RippleSpectrum.sample(
        n_modes=56,
        period=period,
        band=band,
        rms_slope=rms_slope,
        seed=seed,
    )
    rng = np.random.default_rng(direction_seed)
    directions = rng.normal(0.0, spread, size=len(spectrum.harmonics))
    cross_count = int(round(cross_frac * len(directions)))
    cross_modes = rng.choice(
        len(directions), size=cross_count, replace=False,
    )
    directions[cross_modes] = rng.normal(
        cross_dir, 0.6 * spread, size=cross_count,
    )
    directions = np.clip(directions, -1.45, 1.45)
    return replace(
        spectrum,
        directions=tuple(float(direction) for direction in directions),
    )


class RippleField:
    """Precomputed WKB phase and amplitude, one 1-D profile per mode.

    The *phase* is 1-D (alongshore-uniform WKB; stated approximation above),
    with every conserved ``k_y`` on an exact harmonic of the render domain.
    The amplitude *taper*, however, is evaluated on the LOCAL 2-D still depth
    when `still_depth_2d` is given: with the alongshore-mean depth, a band of
    emergent stones biases the mean shallow and the ripples cut off along a
    shore-parallel line at the stones -- a hard, visibly wrong edge. Locally
    tapered, ripples die around each stone and persist between them, which is
    also what sheltering actually looks like.
    """

    def __init__(self, grid: GridConfig, still_depth_x: np.ndarray,
                 spectrum: RippleSpectrum, device: str,
                 still_depth_2d: np.ndarray | None = None, *,
                 alongshore_period: float | None = None):
        self.grid = grid
        self.spectrum = spectrum
        self.device = device
        self.alongshore_period = (
            grid.Ly if alongshore_period is None else alongshore_period
        )
        nx = grid.nx
        nm = len(spectrum.harmonics)
        h = np.maximum(np.asarray(still_depth_x, dtype=np.float64), 0.0)
        if h.shape != (nx,):
            raise ValueError(f"still_depth_x must have shape ({nx},)")

        h_deep = float(h.max())
        psi = np.zeros((nm, nx), dtype=np.float64)
        amp = np.zeros((nm, nx), dtype=np.float64)
        ky = np.zeros(nm, dtype=np.float64)

        for m, (om, a0, th) in enumerate(zip(spectrum.omegas,
                                             spectrum.amplitudes,
                                             spectrum.directions)):
            k_deep = wavenumber(om, h_deep, spectrum.capillary)
            requested_ky = k_deep * np.sin(th)
            ky[m] = (
                2.0 * np.pi
                * round(
                    requested_ky * self.alongshore_period / (2.0 * np.pi)
                )
                / self.alongshore_period
            )
            kx = np.zeros(nx)
            kk = np.zeros(nx)
            for i in range(nx):
                if h[i] <= spectrum.taper_depth:
                    continue
                k = wavenumber(om, h[i], spectrum.capillary)
                kk[i] = k
                kx[i] = np.sqrt(max(k * k - ky[m] ** 2, 0.0))
            # phase by quadrature along x
            psi[m] = np.cumsum(kx) * grid.dx
            # shoaling amplitude; the thin-water taper is 2-D and lives below
            with np.errstate(divide="ignore", invalid="ignore"):
                shoal = np.where(kk > 0, (kk / max(k_deep, 1e-12)) ** spectrum.shoal_exp, 0.0)
            amp[m] = a0 * shoal

        # smoothstep amplitude taper on local still depth
        if still_depth_2d is None:
            h2d = np.broadcast_to(h[:, None], (nx, grid.ny))
        else:
            h2d = np.maximum(np.asarray(still_depth_2d, dtype=np.float64), 0.0)
            if h2d.shape != (nx, grid.ny):
                raise ValueError(f"still_depth_2d must have shape "
                                 f"({nx}, {grid.ny})")
        tt = np.clip((h2d - spectrum.taper_depth)
                     / max(spectrum.taper_width, 1e-9), 0.0, 1.0)
        taper = tt * tt * (3.0 - 2.0 * tt)

        # gustiness envelope wavevectors; alongshore component snapped to a
        # domain harmonic so the envelope wraps exactly at the periodic seam
        ne = len(spectrum.mod_amps)
        mkx = np.zeros(ne)
        mky = np.zeros(ne)
        for e, (lam, th) in enumerate(zip(spectrum.mod_wavelengths,
                                          spectrum.mod_directions)):
            K = 2.0 * np.pi / max(lam, 1e-9)
            mkx[e] = K * np.cos(th)
            mky[e] = (
                2.0 * np.pi
                * round(
                    K * np.sin(th) * self.alongshore_period / (2.0 * np.pi)
                )
                / self.alongshore_period
            )

        self.n_modes = nm
        self.n_mod = ne
        self.psi = wp.array(psi.astype(NP_FLOAT), dtype=FLOAT, device=device)
        self.amp = wp.array(amp.astype(NP_FLOAT), dtype=FLOAT, device=device)
        self.ky = wp.array(ky.astype(NP_FLOAT), dtype=FLOAT, device=device)
        self.omega = wp.array(spectrum.omegas.astype(NP_FLOAT), dtype=FLOAT,
                              device=device)
        self.phase = wp.array(np.asarray(spectrum.phases, dtype=NP_FLOAT),
                              dtype=FLOAT, device=device)
        self.taper = wp.array(taper.astype(NP_FLOAT), dtype=FLOAT, device=device)
        pad = max(ne, 1)  # zero-length device arrays are not launchable

        def _mod(v):
            a = np.zeros(pad, dtype=NP_FLOAT)
            a[:ne] = np.asarray(v, dtype=NP_FLOAT)[:ne] if ne else 0.0
            return wp.array(a, dtype=FLOAT, device=device)

        self.mod_amp = _mod(spectrum.mod_amps)
        self.mod_kx = _mod(mkx)
        self.mod_ky = _mod(mky)
        self.mod_phase = _mod(spectrum.mod_phases)
        self._host = dict(psi=psi, amp=amp, ky=ky)

    def evaluate(self, t: float, out: wp.array | None = None) -> wp.array:
        """Fill (and return) a zeta field at time `t`, shape (nx, ny).

        Every time phase (ripple modes AND envelope modes) is reduced modulo
        2*pi *on the host in float64* before being passed down. It is
        algebraically identical -- `omega_m t mod 2 pi = 2 pi frac(n_m t / T)`
        exactly, since `omega_m = 2 pi n_m / T` -- but it matters numerically:
        unwrapped, `omega t` reaches ~520 rad by the end of one loop, where
        float32 resolves only ~3e-5 rad, and the field then fails its own
        periodicity gate at the 1e-5 level. Wrapped, the argument never
        exceeds 2*pi and periodicity is exact to roundoff on a small number.
        """
        g = self.grid
        if out is None:
            out = wp.zeros((g.nx, g.ny), dtype=FLOAT, device=self.device)
        sp = self.spectrum

        def _wrap(harmonics, pad=1):
            n = np.asarray(harmonics, dtype=np.float64)
            frac = np.mod(n * (float(t) / sp.period), 1.0)
            a = np.zeros(max(len(n), pad), dtype=NP_FLOAT)
            a[:len(n)] = (2.0 * np.pi * frac).astype(NP_FLOAT)
            return wp.array(a, dtype=FLOAT, device=self.device)

        self._tphase = _wrap(sp.harmonics)
        self._mod_tphase = _wrap(sp.mod_harmonics)
        wp.launch(eval_zeta, dim=(g.nx, g.ny),
                  inputs=[self.psi, self.amp, self.ky, self._tphase, self.phase,
                          self.n_modes, self.taper,
                          self.mod_amp, self.mod_kx, self.mod_ky,
                          self._mod_tphase, self.mod_phase, self.n_mod,
                          out, g.x0, g.dx, g.y0, g.dy],
                  device=self.device)
        return out

    def local_wavenumber(self) -> np.ndarray:
        """|grad Psi| per mode from the stored profiles, for the dispersion gate."""
        psi, ky = self._host["psi"], self._host["ky"]
        kx = np.gradient(psi, self.grid.dx, axis=1)
        return np.sqrt(kx ** 2 + ky[:, None] ** 2)


@wp.kernel
def eval_zeta(
    psi: wp.array2d(dtype=FLOAT), amp: wp.array2d(dtype=FLOAT),
    ky: wp.array(dtype=FLOAT), tphase: wp.array(dtype=FLOAT),
    phase: wp.array(dtype=FLOAT), n_modes: int,
    taper: wp.array2d(dtype=FLOAT),
    mod_amp: wp.array(dtype=FLOAT), mod_kx: wp.array(dtype=FLOAT),
    mod_ky: wp.array(dtype=FLOAT), mod_tphase: wp.array(dtype=FLOAT),
    mod_phase: wp.array(dtype=FLOAT), n_mod: int,
    zeta: wp.array2d(dtype=FLOAT),
    x0: FLOAT, dx: FLOAT, y0: FLOAT, dy: FLOAT,
):
    """zeta[i, j] = taper[i, j] * env(x, y) *
                    sum_m amp[m, i] cos(psi[m, i] + ky[m] y - tphase[m] + phase[m])

    with env the travelling gustiness envelope (strictly positive, exactly
    T-periodic). `tphase`/`mod_tphase` are already reduced mod 2*pi on the
    host in float64.
    """
    i, j = wp.tid()
    x = x0 + (FLOAT(i) + HALF) * dx
    y = y0 + (FLOAT(j) + HALF) * dy
    total = ZERO
    for m in range(n_modes):
        total += amp[m, i] * wp.cos(psi[m, i] + ky[m] * y
                                    - tphase[m] + phase[m])
    env = ONE
    for e in range(n_mod):
        env -= mod_amp[e] * (HALF + HALF * wp.cos(mod_kx[e] * x + mod_ky[e] * y
                                                  - mod_tphase[e] + mod_phase[e]))
    zeta[i, j] = taper[i, j] * env * total
