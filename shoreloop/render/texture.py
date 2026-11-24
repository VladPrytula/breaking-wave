"""Procedural pebble bed: albedo and render-scale displacement. Material and
render geometry, not physics.

Two styles, switchable (`PebbleTexture.style`):

  "tiles"    -- the original flat Worley mosaic: albedo only, zero height.
                Reads as a stylised mosaic; kept because it is liked and it
                is the cheapest thing that hits the calibrated mean albedo.
  "pebbles"  -- realistic: two Worley layers (pebbles + gravel between) give
                each stone a rounded dome DISPLACEMENT that is added to the
                render bed. This is exactly plan.md section 9's split:
                `G_render = b_hydro + pebbles` -- the solver keeps the smooth
                bed it can resolve, the renderer marches real 1-2 cm domes.
                Rounded shading, a pebble-ragged waterline and per-stone
                Beer-Lambert tinting then fall out of the ray tracer with no
                further code.

Honesty notes: wetness darkening, foam and ripples still come only from their
closures. The crevice darkening keyed on height is an ambient-occlusion
approximation (valleys see less sky), named here because no ray is traced for
it. Both styles are normalised to the same mean albedo, which was calibrated
against the reference's measured dry-pebble radiance -- switching style must
not silently change the radiometry.

This module remains a placeholder for stage F, where the single-source stone
field `P` supplies real per-stone geometry and material.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _hash01(i: np.ndarray, j: np.ndarray, salt: int) -> np.ndarray:
    """Deterministic per-cell uniform in [0, 1). Plain integer mix, so the
    texture is identical across runs, resolutions and machines."""
    x = (i.astype(np.uint32) * np.uint32(73856093)
         ^ j.astype(np.uint32) * np.uint32(19349663)
         ^ np.uint32((salt * 2654435761) & 0xFFFFFFFF))
    x ^= x >> np.uint32(16)
    x *= np.uint32(2246822519)
    x ^= x >> np.uint32(13)
    x *= np.uint32(3266489917)
    x ^= x >> np.uint32(16)
    return x.astype(np.float64) / 4294967296.0


@dataclass
class PebbleTexture:
    # Nearly neutral grey, worked back from the reference: dry pebble linear
    # RGB (0.120, 0.170, 0.258) under the fitted sky's per-channel irradiance
    # E = (2.65, 3.89, 5.43) gives albedo = pi * L / E, directly -- the maps
    # are mean-pinned to `base`, so no shading compensation belongs here. The
    # blue in the pebble *appearance* is the sky's, not the stone's.
    base: tuple[float, float, float] = (0.143, 0.138, 0.150)
    cell: float = 0.035        # mean pebble spacing [m]
    lum_var: float = 0.45      # per-pebble luminance spread (multiplicative)
    hue_var: float = 0.12      # per-pebble warm/cool jitter
    crevice: float = 0.45      # darkening depth between pebbles
    seed: int = 3

    style: str = "pebbles"     # "pebbles" (displaced domes) or "tiles" (flat)
    # --- "pebbles" style only ---
    height_ratio: float = 0.55   # dome height / pebble radius
    gravel_frac: float = 0.45    # gravel layer cell size, as fraction of cell
    dark_frac: float = 0.15      # fraction of stones that are dark (basalt-ish)
    pale_frac: float = 0.08      # fraction that are pale (limestone-ish)


def _worley(X, Y, cell: float, Ly: float, seed: int):
    """Nearest/second-nearest feature distances and nearest-feature cell ids,
    on a jittered grid that wraps exactly in y."""
    ncy = max(int(round(Ly / cell)), 1)
    cy = Ly / ncy
    ix = np.floor(X / cell).astype(np.int64)
    iy = np.floor(Y / cy).astype(np.int64)

    d1 = np.full(X.shape, np.inf)
    d2 = np.full(X.shape, np.inf)
    id1i = np.zeros(X.shape, dtype=np.int64)
    id1j = np.zeros(X.shape, dtype=np.int64)

    for oi in (-1, 0, 1):
        for oj in (-1, 0, 1):
            ci = ix + oi
            cj = iy + oj
            cjw = np.mod(cj, ncy)  # periodic feature lattice in y
            px = (ci + _hash01(ci, cjw, seed)) * cell
            py = (cj + _hash01(ci, cjw, seed + 1)) * cy
            d = (X - px) ** 2 + (Y - py) ** 2
            closer = d < d1
            d2 = np.where(closer, d1, np.minimum(d2, d))
            id1i = np.where(closer, ci, id1i)
            id1j = np.where(closer, cjw, id1j)
            d1 = np.where(closer, d, d1)

    return np.sqrt(d1), np.sqrt(d2), id1i, id1j


def _stone_tint(id_i, id_j, tex: PebbleTexture, salt: int):
    """Per-stone multiplicative RGB factor: luminance spread, warm/cool and
    green-grey hue jitter, plus occasional dark and pale outlier stones."""
    lum = 1.0 + tex.lum_var * 2.0 * (_hash01(id_i, id_j, salt + 2) - 0.5)
    hue = 2.0 * (_hash01(id_i, id_j, salt + 3) - 0.5)
    hue2 = 2.0 * (_hash01(id_i, id_j, salt + 4) - 0.5)
    kind = _hash01(id_i, id_j, salt + 5)
    lum = np.where(kind < tex.dark_frac, lum * 0.5, lum)
    lum = np.where(kind > 1.0 - tex.pale_frac, lum * 1.7, lum)
    tint = np.empty(id_i.shape + (3,))
    tint[..., 0] = lum * (1.0 + tex.hue_var * hue)
    tint[..., 1] = lum * (1.0 + 0.5 * tex.hue_var * hue2)
    tint[..., 2] = lum * (1.0 - tex.hue_var * hue)
    return tint


def pebble_maps(nx: int, ny: int, dx: float, dy: float, Ly: float,
                tex: PebbleTexture | None = None
                ) -> tuple[np.ndarray, np.ndarray]:
    """(albedo (nx, ny, 3) float32, height (nx, ny) float32), periodic in y.

    Both styles are normalised to mean(albedo) = tex.base, so the calibrated
    radiometry survives switching styles.
    """
    tex = tex or PebbleTexture()
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dy
    X, Y = np.meshgrid(x, y, indexing="ij")
    base = np.asarray(tex.base, dtype=np.float64)

    if tex.style == "tiles":
        d1, d2, id_i, id_j = _worley(X, Y, tex.cell, Ly, tex.seed)
        lum = 1.0 + tex.lum_var * 2.0 * (_hash01(id_i, id_j, tex.seed + 2) - 0.5)
        hue = 2.0 * (_hash01(id_i, id_j, tex.seed + 3) - 0.5)
        edge = np.clip((d2 - d1) / (0.25 * tex.cell), 0.0, 1.0)
        shade = (1.0 - tex.crevice) + tex.crevice * edge
        alb = base[None, None, :] * (lum * shade)[..., None]
        alb[..., 0] *= 1.0 + tex.hue_var * hue
        alb[..., 2] *= 1.0 - tex.hue_var * hue
        height = np.zeros((nx, ny), dtype=np.float32)

    elif tex.style == "pebbles":
        # layer A: the pebbles; layer B: finer gravel filling the gaps
        cA = tex.cell
        cB = tex.cell * tex.gravel_frac
        d1A, _, iA, jA = _worley(X, Y, cA, Ly, tex.seed)
        d1B, _, iB, jB = _worley(X, Y, cB, Ly, tex.seed + 40)

        # per-stone radius, jittered; dome height = ratio * r * sqrt(1-(d/r)^2)
        rA = 0.5 * cA * (0.80 + 0.45 * _hash01(iA, jA, tex.seed + 6))
        rB = 0.5 * cB * (0.80 + 0.45 * _hash01(iB, jB, tex.seed + 46))
        hA = tex.height_ratio * rA * np.sqrt(
            np.clip(1.0 - (d1A / rA) ** 2, 0.0, 1.0))
        hB = tex.height_ratio * rB * np.sqrt(
            np.clip(1.0 - (d1B / rB) ** 2, 0.0, 1.0))
        gravel = 0.6 * hB
        height = np.maximum(hA, gravel).astype(np.float32)

        tintA = _stone_tint(iA, jA, tex, tex.seed)
        tintB = _stone_tint(iB, jB, tex, tex.seed + 40)
        own_A = (hA >= gravel)[..., None]
        tint = np.where(own_A, tintA, tintB)

        # crevice/AO: valleys between domes see less sky. An approximation --
        # no ray is traced for it -- named per the module docstring.
        hmax = tex.height_ratio * 0.5 * cA
        occ = np.clip(height / max(0.35 * hmax, 1e-9), 0.0, 1.0)
        shade = (1.0 - tex.crevice) + tex.crevice * (occ * occ * (3 - 2 * occ))
        alb = base[None, None, :] * tint * shade[..., None]

    else:
        raise ValueError(f"unknown texture style {tex.style!r}")

    # pin the mean to the calibrated base, per channel
    mean = alb.reshape(-1, 3).mean(axis=0)
    alb = alb * (base / np.maximum(mean, 1e-9))[None, None, :]
    return (np.clip(alb, 0.005, 0.95).astype(np.float32), height)


def pebble_albedo(nx: int, ny: int, dx: float, dy: float, Ly: float,
                  tex: PebbleTexture | None = None) -> np.ndarray:
    """Back-compat: albedo only."""
    return pebble_maps(nx, ny, dx, dy, Ly, tex)[0]
