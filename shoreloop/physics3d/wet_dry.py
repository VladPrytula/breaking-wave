"""Small, measured cleanup for disconnected roundoff-scale water wisps."""

from __future__ import annotations

from dataclasses import dataclass
import math

import warp as wp

from shoreloop.config import BC, Solver3DConfig
from shoreloop.physics3d.boundary import fill_field_boundaries
from shoreloop.physics3d.extrapolation import _wrapped_cell_index
from shoreloop.physics3d.state import State3D


@dataclass(frozen=True)
class WispRemovalResult:
    removed_cells: int
    removed_volume: float
    cumulative_removed_volume: float


@wp.func
def _cell_alpha(
    alpha: wp.array3d(dtype=wp.float32),
    i: int,
    j: int,
    k: int,
    nx: int,
    ny: int,
    nz: int,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    ng: int,
) -> wp.float32:
    ci = _wrapped_cell_index(i, nx, periodic_x)
    cj = _wrapped_cell_index(j, ny, periodic_y)
    ck = _wrapped_cell_index(k, nz, periodic_z)
    if ci < 0 or cj < 0 or ck < 0:
        return wp.float32(0.0)
    return alpha[ci + ng, cj + ng, ck + ng]


@wp.kernel
def _mark_wisps(
    alpha: wp.array3d(dtype=wp.float32),
    theta: wp.array3d(dtype=wp.float32),
    solid: wp.array3d(dtype=wp.int32),
    marked: wp.array3d(dtype=wp.int32),
    removed_cells: wp.array(dtype=wp.int32),
    removed_fraction: wp.array(dtype=wp.float64),
    tolerance: wp.float32,
    nx: int,
    ny: int,
    nz: int,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    ng: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    value = alpha[I, J, K]
    if solid[I, J, K] != 0 or value <= wp.float32(0.0) or value > tolerance:
        return

    connected = int(0)
    for direction in range(6):
        di = int(0)
        dj = int(0)
        dk = int(0)
        if direction == 0:
            di = -1
        elif direction == 1:
            di = 1
        elif direction == 2:
            dj = -1
        elif direction == 3:
            dj = 1
        elif direction == 4:
            dk = -1
        else:
            dk = 1
        if _cell_alpha(
            alpha, i + di, j + dj, k + dk, nx, ny, nz,
            periodic_x, periodic_y, periodic_z, ng,
        ) > tolerance:
            connected = 1
    if connected == 0:
        marked[I, J, K] = 1
        wp.atomic_add(removed_cells, 0, 1)
        wp.atomic_add(
            removed_fraction, 0,
            wp.float64(value) * wp.float64(theta[I, J, K]),
        )


@wp.kernel
def _remove_marked_wisps(
    alpha: wp.array3d(dtype=wp.float32),
    marked: wp.array3d(dtype=wp.int32),
    ng: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    if marked[I, J, K] != 0:
        alpha[I, J, K] = wp.float32(0.0)


class WispRemover:
    """Remove only disconnected cells at or below a declared alpha tolerance."""

    __slots__ = (
        "cfg", "device", "tolerance", "total_removed_volume", "_marked",
        "_removed_cells", "_removed_fraction",
    )

    def __init__(self, cfg: Solver3DConfig, tolerance: float = 1.0e-6):
        if not math.isfinite(tolerance) or tolerance <= 0.0 or tolerance >= 0.5:
            raise ValueError("wisp tolerance must be finite in (0, 0.5)")
        self.cfg = cfg
        self.device = cfg.device
        self.tolerance = float(tolerance)
        self.total_removed_volume = 0.0
        shape = tuple(n + 2 * cfg.grid.ng for n in cfg.grid.shape)
        self._marked = wp.zeros(shape, dtype=wp.int32, device=self.device)
        self._removed_cells = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._removed_fraction = wp.zeros(1, dtype=wp.float64,
                                          device=self.device)

    def apply(self, state: State3D) -> WispRemovalResult:
        if state.cfg != self.cfg:
            raise ValueError("state configuration must match the wisp remover")
        self._marked.zero_()
        self._removed_cells.zero_()
        self._removed_fraction.zero_()
        g = self.cfg.grid
        bc = self.cfg.bc
        periodic = (
            int(bc.xlo == BC.PERIODIC), int(bc.ylo == BC.PERIODIC),
            int(bc.zlo == BC.PERIODIC),
        )
        wp.launch(
            _mark_wisps, dim=g.shape,
            inputs=[state.alpha, state.theta, state.solid, self._marked,
                    self._removed_cells, self._removed_fraction,
                    self.tolerance, g.nx, g.ny, g.nz, *periodic, g.ng],
            device=self.device,
        )
        removed_cells = int(self._removed_cells.numpy()[0])
        removed_volume = (
            float(self._removed_fraction.numpy()[0]) * g.dx * g.dy * g.dz
        )
        if removed_cells > 0:
            wp.launch(
                _remove_marked_wisps, dim=g.shape,
                inputs=[state.alpha, self._marked, g.ng], device=self.device,
            )
            fill_field_boundaries(state, "alpha")
        self.total_removed_volume += removed_volume
        return WispRemovalResult(
            removed_cells, removed_volume, self.total_removed_volume,
        )
