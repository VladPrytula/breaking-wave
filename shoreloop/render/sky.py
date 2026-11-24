"""Sky dome and (optional) sun: the light sources. All of theta_render.

## The dome

Radiance toward a direction with elevation mu = max(dir_z, 0):

    L(mu) = horizon + (zenith - horizon) * mu^p  +  glow * max(d . g_dir, 0)^q

The base defaults are HORIZON-BRIGHT, deliberately not the CIE overcast
gradation (zenith 3x horizon). The reference forces this: its far water (luma
0.502, mostly Fresnel-reflected low sky) is 3x brighter than its dry pebbles
(0.166, lit by the whole dome). A dome whose horizon is dimmer than its mean
cannot produce both numbers with any sane pebble albedo -- the sky this video
was shot under has a bright band near the horizon, which is exactly what a
low sun behind an overcast layer produces at dusk. Working the two
measurements backwards with a ~0.15 grey pebble albedo gives L(horizon) luma
~1.9 against a downward irradiance E/pi of ~0.9, and those are the defaults.

The `glow` term is an azimuthally localised bright patch (the sun's position
showing through the overcast) for the dramatic presets; zero by default.
Downward directions clamp elevation to the horizon, which is also what a ray
leaving the domain over open water should see at grazing incidence.

## The sun

`Sun` is a directional source, DEFAULT OFF (zero irradiance): the reference
scene is overcast and the default rendering must stay honest to it. When on,
the renderer adds: direct Lambertian on the dry bed with marched shadow rays,
direct sun transmitted through the interface onto the submerged bed
(Beer-Lambert, WITHOUT caustic focusing -- light-side transport stays
deferred, docs/decisions.md), and filtered specular glint on the water.
`glitter_width_deg` is the angular Gaussian that stands in for both the sun's
finite disc and the sub-render-cell slope variance -- the temporal filtering
that decisions.md demands, because unfiltered sparkle aliasing is the loudest
"this is CG" signal.

`light_preset(name)` returns coherent (sky, sun) pairs. Only "overcast" is
calibrated against the reference; the others are capability demos and say so.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _dir_from(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az),
                     np.sin(el)])


@dataclass
class OvercastSky:
    # trimmed 10% when the ripple spectrum went dense: gentler per-mode
    # slopes reflect the low sky more coherently, and far water ran 12% hot
    horizon: tuple[float, float, float] = (1.17, 1.77, 2.52)
    zenith: tuple[float, float, float] = (0.35, 0.44, 0.54)
    # p = 3: the bright band stays near-horizon-bright up to ~30 degrees of
    # elevation. This is load-bearing, not cosmetic: ripple facets tilt the
    # reflected rays 10-30 degrees above the horizon, and a band that decays
    # linearly (p = 1) makes rippled far water collapse to half the measured
    # luminance while flat far water hits the target exactly.
    p: float = 3.0
    # localised sunward glow; zero disables
    glow: tuple[float, float, float] = (0.0, 0.0, 0.0)
    glow_azimuth_deg: float = 180.0
    glow_elevation_deg: float = 8.0
    glow_sharpness: float = 40.0

    def glow_dir(self) -> np.ndarray:
        return _dir_from(self.glow_azimuth_deg, self.glow_elevation_deg)

    def radiance(self, dirs: np.ndarray) -> np.ndarray:
        """Host-side reference implementation; the kernel mirrors it exactly.

        dirs: (..., 3) unit vectors. Returns (..., 3) linear RGB radiance.
        """
        mu = np.clip(dirs[..., 2], 0.0, 1.0)[..., None] ** self.p
        out = (np.asarray(self.horizon) * (1.0 - mu)
               + np.asarray(self.zenith) * mu)
        if any(g > 0.0 for g in self.glow):
            c = np.clip(np.tensordot(dirs, self.glow_dir(), axes=([-1], [0])),
                        0.0, 1.0)
            out = out + np.asarray(self.glow) * (c ** self.glow_sharpness)[..., None]
        return out

    def e_down(self) -> np.ndarray:
        """Horizontal irradiance E = integral over the upper hemisphere of
        L(d) d_z dOmega, per channel. Numerical quadrature (the glow term has
        no tidy closed form); the base part is validated against its closed
        form in tests/render/test_optics.py."""
        n_mu, n_ph = 256, 256
        mu = (np.arange(n_mu) + 0.5) / n_mu
        ph = 2.0 * np.pi * (np.arange(n_ph) + 0.5) / n_ph
        MU, PH = np.meshgrid(mu, ph, indexing="ij")
        s = np.sqrt(1.0 - MU**2)
        dirs = np.stack([s * np.cos(PH), s * np.sin(PH), MU], axis=-1)
        L = self.radiance(dirs)
        w = (2.0 * np.pi / n_ph) * (1.0 / n_mu)
        return (L * MU[..., None] * w).sum(axis=(0, 1)).astype(np.float32)


@dataclass
class Sun:
    """Directional source. irradiance = (0,0,0) means no sun (the default)."""

    azimuth_deg: float = 180.0     # 180 = offshore, i.e. in front of the camera
    elevation_deg: float = 12.0
    # irradiance on a plane perpendicular to the sun [linear radiance units]
    irradiance: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # effective radiance of the (filtered) solar disc, for specular glint;
    # far above 1.0 by design -- glint saturates to white, as it should
    glitter_radiance: tuple[float, float, float] = (0.0, 0.0, 0.0)
    glitter_width_deg: float = 2.0

    @property
    def enabled(self) -> bool:
        return any(v > 0.0 for v in self.irradiance)

    def direction(self) -> np.ndarray:
        """Unit vector pointing FROM the scene TOWARD the sun."""
        return _dir_from(self.azimuth_deg, self.elevation_deg)


def light_preset(name: str) -> tuple[OvercastSky, Sun]:
    """Coherent (sky, sun) pairs.

    overcast  -- the reference-matched default. No sun, no glow.
    sunset    -- low warm sun dead ahead over the water: glint corridor,
                 long stone shadows, warm horizon glow. A capability demo,
                 NOT a match to IMG_4503.
    silver    -- thin-overcast late afternoon: the sun a bright smeared patch,
                 soft wide glitter, milder shadows. Also a demo.
    """
    if name == "overcast":
        return OvercastSky(), Sun()
    if name == "sunset":
        # sun 20 degrees off the view axis: with it dead ahead, the long
        # stone shadows point straight at the camera and hide behind the
        # stones that cast them; raked, they stripe across the frame
        sky = OvercastSky(horizon=(1.35, 1.15, 1.25),
                          zenith=(0.30, 0.32, 0.45),
                          glow=(14.0, 5.5, 1.6),
                          glow_azimuth_deg=160.0, glow_elevation_deg=7.0,
                          glow_sharpness=60.0)
        sun = Sun(azimuth_deg=160.0, elevation_deg=9.0,
                  irradiance=(9.0, 4.6, 1.8),
                  glitter_radiance=(60.0, 26.0, 8.0),
                  glitter_width_deg=1.8)
        return sky, sun
    if name == "silver":
        sky = OvercastSky(horizon=(1.55, 1.90, 2.45),
                          zenith=(0.45, 0.52, 0.62),
                          glow=(5.0, 4.6, 3.8),
                          glow_azimuth_deg=180.0, glow_elevation_deg=14.0,
                          glow_sharpness=25.0)
        sun = Sun(azimuth_deg=180.0, elevation_deg=16.0,
                  irradiance=(3.5, 3.2, 2.6),
                  glitter_radiance=(14.0, 12.5, 10.0),
                  glitter_width_deg=3.5)
        return sky, sun
    raise ValueError(f"unknown light preset {name!r} "
                     "(overcast | sunset | silver)")
