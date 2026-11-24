"""Solver state and scratch storage."""

from __future__ import annotations

import numpy as np
import warp as wp

from shoreloop.config import GridConfig
from shoreloop.precision import FLOAT, NP_FLOAT


def _zeros(shape, device):
    return wp.zeros(shape, dtype=FLOAT, device=device)


class State:
    """Conservative variables plus the static bed, on the padded grid.

    Arrays are indexed `[i, j]` over the padded shape; the interior is
    `[ng : ng + nx, ng : ng + ny]`.
    """

    __slots__ = ("h", "qx", "qy", "b", "t", "grid", "device")

    def __init__(self, grid: GridConfig, device: str):
        self.grid = grid
        self.device = device
        shape = grid.shape
        self.h = _zeros(shape, device)
        self.qx = _zeros(shape, device)
        self.qy = _zeros(shape, device)
        self.b = _zeros(shape, device)
        self.t = 0.0

    # -- host interop ------------------------------------------------------
    @property
    def _sl(self):
        g = self.grid
        return (slice(g.ng, g.ng + g.nx), slice(g.ng, g.ng + g.ny))

    def get(self, name: str) -> np.ndarray:
        """Interior of a field as a host array."""
        return getattr(self, name).numpy()[self._sl]

    def set(self, name: str, values: np.ndarray) -> None:
        """Write the interior of a field from a host array."""
        arr = getattr(self, name)
        host = arr.numpy()
        # Scenes build float32 arrays; cast so the same scene code works under
        # SHORELOOP_FP64 without every builder having to know the precision.
        host[self._sl] = np.asarray(values, dtype=NP_FLOAT)
        arr.assign(host)

    def eta(self) -> np.ndarray:
        """Free-surface elevation over the interior (meaningless where dry)."""
        return self.get("h") + self.get("b")

    def speed(self) -> np.ndarray:
        """|u| over the interior, zero where effectively dry.

        Dividing by raw h makes a sub-micron film with roundoff momentum
        report hundreds of m/s; that is a property of the quotient, not of
        the flow, and it has already mis-flagged one healthy run as unstable.
        """
        h = self.get("h")
        q = np.hypot(self.get("qx"), self.get("qy"))
        return np.where(h > 1e-6, q / np.maximum(h, 1e-6), 0.0)

    def clone(self) -> "State":
        other = State(self.grid, self.device)
        for f in ("h", "qx", "qy", "b"):
            wp.copy(getattr(other, f), getattr(self, f))
        other.t = self.t
        return other


class Scratch:
    """Per-step temporaries. Allocated once and reused."""

    RECON = ("etaL", "etaR", "unL", "unR", "utL", "utR", "bL", "bR")
    FACE = ("Fh", "Fn", "Ft", "Bs")
    STAGE = ("h1", "qx1", "qy1", "rh", "rqx", "rqy")

    def __init__(self, grid: GridConfig, device: str):
        shape = grid.shape
        for name in self.RECON + self.FACE + self.STAGE:
            setattr(self, name, _zeros(shape, device))
        # float64 so the mass gate is not limited by summation error
        self.clipped = wp.zeros(1, dtype=wp.float64, device=device)
        self.wave_rate = wp.zeros(1, dtype=FLOAT, device=device)
        # Separate accumulator for the viscous rate: the combined explicit
        # limit is the SUM of the advective and diffusive rates, so they must
        # not share one atomic_max slot.
        self.visc_rate = wp.zeros(1, dtype=FLOAT, device=device)
