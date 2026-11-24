"""Offshore wave forcing, constructed to be exactly periodic.

The loop period `T` is the primitive: every component frequency is an integer
multiple of `2 pi / T`, so the *forcing* is exactly T-periodic by construction.
Nonlinearity in the interior can then generate harmonics, but every harmonic it
can generate still belongs to the fundamental period. That is what makes
`U(0) = U(T)` reachable at all rather than approximated by a cross-fade
(plan.md section 7).

This is only half the requirement. Exactly periodic *forcing* through a
*reflective* boundary still produces standing modes at basin periods unrelated
to T, which is why the boundary must absorb as well as generate -- see
physics.boundary.bc_wavemaker_x_lo and docs/decisions.md Correction 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import warp as wp

from shoreloop.precision import FLOAT, NP_FLOAT


@dataclass
class WaveForcing:
    """A sum of harmonics of the loop frequency, each optionally oblique.

    period:     loop period T [s]
    harmonics:  integer multiples n of the fundamental; omega_n = 2 pi n / T
    amplitudes: A_n [m]
    phases:     phi_n [rad]
    ky_modes:   alongshore INTEGER mode numbers m_n; k_y = 2 pi m_n / Ly.
                Integers are what keep the field exact across the alongshore
                periodic seam, just as integer harmonics keep it T-periodic.
                The domain quantises the available angles: sin(theta_n) =
                k_y c / omega_n at the boundary depth, so long swell in a
                narrow strip can only be shore-normal (m = 0) -- k_y > k is an
                evanescent wave, not an oblique one, and set_forcing rejects
                it. Two components on the same harmonic with opposite m are a
                crossing pair: genuine short-crested interference.
    eta0:       still-water level [m]
    """

    period: float = 8.0
    harmonics: tuple[int, ...] = (1,)
    amplitudes: tuple[float, ...] = (0.06,)
    phases: tuple[float, ...] = (0.0,)
    ky_modes: tuple[int, ...] | None = None
    eta0: float = 0.0

    def __post_init__(self):
        n = len(self.harmonics)
        if self.ky_modes is None:
            self.ky_modes = (0,) * n
        if not (len(self.amplitudes) == len(self.phases)
                == len(self.ky_modes) == n):
            raise ValueError("harmonics, amplitudes, phases and ky_modes must "
                             "be the same length")
        if any(int(h) != h or h < 1 for h in self.harmonics):
            raise ValueError(
                "harmonics must be positive integers, or the forcing is not T-periodic"
            )
        if any(int(m) != m for m in self.ky_modes):
            raise ValueError("ky_modes must be integers, or the forcing breaks "
                             "at the alongshore periodic seam")

    # -- host-side evaluation, for tests and diagnostics -------------------
    def eta(self, t: float | np.ndarray, y: float = 0.0,
            Ly: float = 1.0) -> float | np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        out = np.full(t.shape, self.eta0, dtype=np.float64)
        for n, a, p, m in zip(self.harmonics, self.amplitudes, self.phases,
                              self.ky_modes):
            out += a * np.cos(2.0 * np.pi * n * t / self.period
                              - 2.0 * np.pi * m * y / Ly + p)
        return out

    @property
    def amplitude(self) -> float:
        """Sum of component amplitudes: the worst-case surface excursion."""
        return float(sum(abs(a) for a in self.amplitudes))

    def to_device(self, device: str, Ly: float) -> "DeviceForcing":
        ky = 2.0 * np.pi * np.asarray(self.ky_modes, dtype=np.float64) / Ly
        return DeviceForcing(
            amp=wp.array(np.asarray(self.amplitudes, dtype=NP_FLOAT), dtype=FLOAT, device=device),
            harm=wp.array(np.asarray(self.harmonics, dtype=NP_FLOAT), dtype=FLOAT, device=device),
            pha=wp.array(np.asarray(self.phases, dtype=NP_FLOAT), dtype=FLOAT, device=device),
            ky=wp.array(ky.astype(NP_FLOAT), dtype=FLOAT, device=device),
            nmodes=len(self.harmonics),
            period=self.period,
            eta0=self.eta0,
        )


@dataclass
class DeviceForcing:
    amp: wp.array
    harm: wp.array
    pha: wp.array
    ky: wp.array
    nmodes: int
    period: float
    eta0: float


def still_water(eta0: float) -> WaveForcing:
    """A forcing that generates nothing: a pure absorber. Used by the gate."""
    return WaveForcing(period=1.0, harmonics=(1,), amplitudes=(0.0,), phases=(0.0,),
                       eta0=eta0)
