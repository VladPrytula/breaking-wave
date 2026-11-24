"""Uniform three-dimensional MAC-grid layout and host coordinate axes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shoreloop.config import Grid3DConfig


_CELL_FIELDS = (
    "cell", "p", "alpha", "porous_storage", "foam", "solid_wetness",
)
_FIELDS = _CELL_FIELDS + ("u", "v", "w")


@dataclass(frozen=True)
class MACGrid:
    """Indexing for cell scalars and face-normal velocities.

    Logical indices address the physical domain. Storage indices add ``ng``
    on every axis, leaving a uniform halo around each staggered lattice. Array
    order is always ``[i, j, k] == [x, y, z]``.
    """

    config: Grid3DConfig

    def _location(self, field: str) -> str:
        if field not in _FIELDS:
            raise ValueError(f"unknown MAC field {field!r}; choose from {_FIELDS}")
        return "cell" if field in _CELL_FIELDS else field

    def interior_shape(self, field: str = "cell") -> tuple[int, int, int]:
        g = self.config
        location = self._location(field)
        return (
            g.nx + (location == "u"),
            g.ny + (location == "v"),
            g.nz + (location == "w"),
        )

    def padded_shape(self, field: str = "cell") -> tuple[int, int, int]:
        ng = self.config.ng
        return tuple(n + 2 * ng for n in self.interior_shape(field))

    def interior_slices(self, field: str = "cell") -> tuple[slice, slice, slice]:
        ng = self.config.ng
        return tuple(slice(ng, ng + n) for n in self.interior_shape(field))

    @property
    def cell_shape(self) -> tuple[int, int, int]:
        return self.padded_shape("cell")

    @property
    def u_shape(self) -> tuple[int, int, int]:
        return self.padded_shape("u")

    @property
    def v_shape(self) -> tuple[int, int, int]:
        return self.padded_shape("v")

    @property
    def w_shape(self) -> tuple[int, int, int]:
        return self.padded_shape("w")

    def storage_index(self, field: str, i: int, j: int, k: int
                      ) -> tuple[int, int, int]:
        """Map one physical logical index to its padded storage index."""
        shape = self.interior_shape(field)
        logical = (i, j, k)
        if any(isinstance(q, bool) or not isinstance(q, (int, np.integer))
               for q in logical):
            raise TypeError("MAC logical indices must be integers")
        if any(q < 0 or q >= n for q, n in zip(logical, shape)):
            raise IndexError(f"{field} index {logical} outside physical shape {shape}")
        ng = self.config.ng
        return (int(i) + ng, int(j) + ng, int(k) + ng)

    def boundary_face_index(self, axis: str, side: str) -> int:
        """Storage index of a physical normal-velocity boundary face."""
        if axis not in ("x", "y", "z"):
            raise ValueError("axis must be 'x', 'y', or 'z'")
        if side not in ("lo", "hi"):
            raise ValueError("side must be 'lo' or 'hi'")
        n = {"x": self.config.nx, "y": self.config.ny,
             "z": self.config.nz}[axis]
        return self.config.ng if side == "lo" else self.config.ng + n

    def coordinates(self, field: str = "cell", *, include_ghost: bool = False
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return memory-light 1-D coordinate axes for a staggered lattice.

        The three arrays broadcast to the corresponding field shape and are
        suitable for building initial conditions without allocating three full
        coordinate volumes. By default they cover only the physical domain.
        """
        g = self.config
        location = self._location(field)
        shape = self.interior_shape(location)
        offsets = (
            0.0 if location == "u" else 0.5,
            0.0 if location == "v" else 0.5,
            0.0 if location == "w" else 0.5,
        )
        origins = (g.x0, g.y0, g.z0)
        spacing = (g.dx, g.dy, g.dz)
        ng = g.ng if include_ghost else 0
        return tuple(
            origin + (np.arange(n + 2 * ng, dtype=np.float64) - ng + offset) * delta
            for n, origin, delta, offset in zip(shape, origins, spacing, offsets)
        )
