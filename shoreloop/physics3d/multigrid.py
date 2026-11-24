"""Geometric multigrid levels and cell-centred transfer operators.

The hierarchy is deliberately regular: every coarsening step halves all three
axes and keeps the physical box and boundary kinds unchanged. Finest grids
must therefore be even in every direction. Coarsening stops before an axis
would become smaller than two cells or when the next halving is not exact.

Pressure corrections use cell-centred trilinear prolongation. Residuals use
the volume-adjoint full-weighting restriction, ``R = P^T / 8``. Both preserve
constants exactly; the pairing also gives the later symmetric V-cycle the
transfer relation needed by PCG.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import warp as wp

from shoreloop.config import BC, Grid3DConfig, Solver3DConfig
from shoreloop.physics3d.grid import MACGrid
from shoreloop.physics3d.operators import MACOperators
from shoreloop.physics3d.pcg import (
    PCGConfig,
    PCGResult,
    PressurePCG,
    _copy_cell_interior,
    _form_dot_products,
    _form_residual,
    _reduce_sum,
    _subtract_mean,
)


@wp.func
def _transfer_index(index: int, count: int, periodic: int) -> int:
    if periodic != 0:
        if index < 0:
            return index + count
        if index >= count:
            return index - count
        return index
    return wp.max(0, wp.min(count - 1, index))


@wp.kernel
def _restrict_full_weighting(
    fine: wp.array3d(dtype=wp.float32),
    coarse: wp.array3d(dtype=wp.float32),
    coarse_nx: int,
    coarse_ny: int,
    coarse_nz: int,
    fine_ng: int,
    coarse_ng: int,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
):
    i, j, k = wp.tid()
    fine_nx = 2 * coarse_nx
    fine_ny = 2 * coarse_ny
    fine_nz = 2 * coarse_nz
    value = wp.float32(0.0)
    for di in range(4):
        fi = _transfer_index(2 * i + di - 1, fine_nx, periodic_x)
        wi = wp.float32(0.125)
        if di == 1 or di == 2:
            wi = wp.float32(0.375)
        for dj in range(4):
            fj = _transfer_index(2 * j + dj - 1, fine_ny, periodic_y)
            wj = wp.float32(0.125)
            if dj == 1 or dj == 2:
                wj = wp.float32(0.375)
            for dk in range(4):
                fk = _transfer_index(2 * k + dk - 1, fine_nz, periodic_z)
                wk = wp.float32(0.125)
                if dk == 1 or dk == 2:
                    wk = wp.float32(0.375)
                value += wi * wj * wk * fine[
                    fi + fine_ng, fj + fine_ng, fk + fine_ng
                ]
    coarse[i + coarse_ng, j + coarse_ng, k + coarse_ng] = value


@wp.kernel
def _prolong_trilinear(
    coarse: wp.array3d(dtype=wp.float32),
    fine: wp.array3d(dtype=wp.float32),
    coarse_nx: int,
    coarse_ny: int,
    coarse_nz: int,
    coarse_ng: int,
    fine_ng: int,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    add: int,
):
    i, j, k = wp.tid()
    ci = i // 2
    cj = j // 2
    ck = k // 2

    ni = ci + 1
    nj = cj + 1
    nk = ck + 1
    if i % 2 == 0:
        ni = ci - 1
    if j % 2 == 0:
        nj = cj - 1
    if k % 2 == 0:
        nk = ck - 1
    ni = _transfer_index(ni, coarse_nx, periodic_x)
    nj = _transfer_index(nj, coarse_ny, periodic_y)
    nk = _transfer_index(nk, coarse_nz, periodic_z)

    wc = wp.float32(0.75)
    wn = wp.float32(0.25)
    value = (
        wc * wc * wc * coarse[ci + coarse_ng, cj + coarse_ng, ck + coarse_ng]
        + wn * wc * wc * coarse[ni + coarse_ng, cj + coarse_ng, ck + coarse_ng]
        + wc * wn * wc * coarse[ci + coarse_ng, nj + coarse_ng, ck + coarse_ng]
        + wc * wc * wn * coarse[ci + coarse_ng, cj + coarse_ng, nk + coarse_ng]
        + wn * wn * wc * coarse[ni + coarse_ng, nj + coarse_ng, ck + coarse_ng]
        + wn * wc * wn * coarse[ni + coarse_ng, cj + coarse_ng, nk + coarse_ng]
        + wc * wn * wn * coarse[ci + coarse_ng, nj + coarse_ng, nk + coarse_ng]
        + wn * wn * wn * coarse[ni + coarse_ng, nj + coarse_ng, nk + coarse_ng]
    )
    I = i + fine_ng
    J = j + fine_ng
    K = k + fine_ng
    if add != 0:
        fine[I, J, K] += value
    else:
        fine[I, J, K] = value


@wp.kernel
def _weighted_jacobi_update(
    rhs: wp.array3d(dtype=wp.float32),
    operator_pressure: wp.array3d(dtype=wp.float32),
    diagonal: wp.array3d(dtype=wp.float32),
    pressure: wp.array3d(dtype=wp.float32),
    weight: wp.float32,
    ng: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    pressure[I, J, K] += (
        weight * (rhs[I, J, K] - operator_pressure[I, J, K])
        / diagonal[I, J, K]
    )


@dataclass
class MultigridLevel:
    """One regular cell-centred pressure level and its reusable workspace."""

    cfg: Solver3DConfig
    grid: MACGrid
    operators: MACOperators
    pressure: wp.array
    rhs: wp.array
    residual: wp.array
    operator_value: wp.array
    diagonal: wp.array
    dot_products: wp.array
    reduction: wp.array


@dataclass(frozen=True)
class VCycleConfig:
    """Fixed work performed by one deterministic symmetric V-cycle."""

    jacobi_weight: float = 2.0 / 3.0
    pre_smoothing_steps: int = 2
    post_smoothing_steps: int = 2


@dataclass(frozen=True)
class VCycleResult:
    """Correction and measured residual reduction from one V-cycle."""

    correction: wp.array
    residual_before: float
    residual_after: float
    reduction_factor: float
    level_residuals: tuple[float, ...]


class MultigridHierarchy:
    """Successively halved regular grids for one pressure solve."""

    __slots__ = ("cfg", "device", "levels", "coarsest_solver")

    def __init__(self, cfg: Solver3DConfig):
        self.cfg = cfg
        self.device = cfg.device
        finest = cfg.grid.shape
        if any(n % 2 != 0 for n in finest):
            raise ValueError(
                "multigrid requires even nx, ny, and nz on the finest grid",
            )

        configs = [cfg]
        shape = finest
        while all(n % 2 == 0 and n // 2 >= 2 for n in shape):
            g = configs[-1].grid
            coarse_grid = Grid3DConfig(
                nx=g.nx // 2,
                ny=g.ny // 2,
                nz=g.nz // 2,
                Lx=g.Lx,
                Ly=g.Ly,
                Lz=g.Lz,
                x0=g.x0,
                y0=g.y0,
                z0=g.z0,
                ng=g.ng,
            )
            configs.append(replace(cfg, grid=coarse_grid))
            shape = coarse_grid.shape

        self.levels: list[MultigridLevel] = []
        for level_cfg in configs:
            grid = MACGrid(level_cfg.grid)

            def cell_array() -> wp.array:
                return wp.zeros(
                    grid.cell_shape, dtype=wp.float32, device=self.device,
                )

            operators = MACOperators(level_cfg)
            self.levels.append(MultigridLevel(
                cfg=level_cfg,
                grid=grid,
                operators=operators,
                pressure=cell_array(),
                rhs=cell_array(),
                residual=cell_array(),
                operator_value=cell_array(),
                diagonal=operators.pressure_diagonal(),
                dot_products=wp.zeros(
                    level_cfg.grid.shape,
                    dtype=wp.float64,
                    device=self.device,
                ),
                reduction=wp.zeros(
                    2, dtype=wp.float64, device=self.device,
                ),
            ))

        self.coarsest_solver = PressurePCG(
            self.levels[-1].cfg,
            PCGConfig(relative_tolerance=1.0e-6, max_iterations=200),
        )

    @property
    def shapes(self) -> tuple[tuple[int, int, int], ...]:
        return tuple(level.cfg.grid.shape for level in self.levels)

    def restrict(
        self,
        fine_index: int,
        fine: wp.array,
        coarse: wp.array,
    ) -> None:
        """Apply volume-adjoint full weighting to adjacent levels."""
        fine_level = self.levels[fine_index]
        coarse_level = self.levels[fine_index + 1]
        coarse.zero_()
        bc = self.cfg.bc
        wp.launch(
            _restrict_full_weighting,
            dim=coarse_level.cfg.grid.shape,
            inputs=[
                fine, coarse,
                *coarse_level.cfg.grid.shape,
                fine_level.cfg.grid.ng,
                coarse_level.cfg.grid.ng,
                int(bc.xlo == BC.PERIODIC),
                int(bc.ylo == BC.PERIODIC),
                int(bc.zlo == BC.PERIODIC),
            ],
            device=self.device,
        )

    def prolong(
        self,
        coarse_index: int,
        coarse: wp.array,
        fine: wp.array,
        *,
        add: bool = False,
    ) -> None:
        """Trilinearly interpolate one correction to the next finer level."""
        coarse_level = self.levels[coarse_index]
        fine_level = self.levels[coarse_index - 1]
        if not add:
            fine.zero_()
        bc = self.cfg.bc
        wp.launch(
            _prolong_trilinear,
            dim=fine_level.cfg.grid.shape,
            inputs=[
                coarse, fine,
                *coarse_level.cfg.grid.shape,
                coarse_level.cfg.grid.ng,
                fine_level.cfg.grid.ng,
                int(bc.xlo == BC.PERIODIC),
                int(bc.ylo == BC.PERIODIC),
                int(bc.zlo == BC.PERIODIC),
                int(add),
            ],
            device=self.device,
        )

    def solve_coarsest(self) -> PCGResult:
        """Solve the terminal rediscretised pressure problem with reference PCG."""
        level = self.levels[-1]
        level.pressure.zero_()
        return self.coarsest_solver.solve(level.rhs, level.pressure)


class MultigridVCycle:
    """Weighted-Jacobi geometric V-cycle for the regular pressure operator."""

    __slots__ = ("hierarchy", "settings", "device")

    def __init__(
        self,
        cfg: Solver3DConfig,
        settings: VCycleConfig | None = None,
    ):
        self.hierarchy = MultigridHierarchy(cfg)
        self.settings = VCycleConfig() if settings is None else settings
        self.device = cfg.device

    def _copy(self, source: wp.array, target: wp.array, level_index: int) -> None:
        level = self.hierarchy.levels[level_index]
        target.zero_()
        wp.launch(
            _copy_cell_interior,
            dim=level.cfg.grid.shape,
            inputs=[source, target, level.cfg.grid.ng],
            device=self.device,
        )

    def _remove_mean(self, level_index: int, field: wp.array) -> None:
        level = self.hierarchy.levels[level_index]
        level.reduction.zero_()
        wp.launch(
            _reduce_sum,
            dim=level.cfg.grid.shape,
            inputs=[field, level.reduction, level.cfg.grid.ng],
            device=self.device,
        )
        mean = float(level.reduction.numpy()[0]) / level.cfg.grid.cell_count
        wp.launch(
            _subtract_mean,
            dim=level.cfg.grid.shape,
            inputs=[field, mean, level.cfg.grid.ng],
            device=self.device,
        )

    def _norm(self, level_index: int, field: wp.array) -> float:
        level = self.hierarchy.levels[level_index]
        g = level.cfg.grid
        wp.launch(
            _form_dot_products,
            dim=g.shape,
            inputs=[field, field, level.dot_products, g.ng,
                    g.dx * g.dy * g.dz],
            device=self.device,
        )
        wp.utils.array_sum(
            level.dot_products, out=level.reduction[0:1],
        )
        return math.sqrt(max(float(level.reduction.numpy()[0]), 0.0))

    def _residual(self, level_index: int) -> float:
        level = self.hierarchy.levels[level_index]
        level.operators.pressure_operator(
            level.pressure, out=level.operator_value,
        )
        wp.launch(
            _form_residual,
            dim=level.cfg.grid.shape,
            inputs=[level.rhs, level.operator_value, level.residual,
                    level.cfg.grid.ng],
            device=self.device,
        )
        self._remove_mean(level_index, level.residual)
        return self._norm(level_index, level.residual)

    def _smooth(self, level_index: int, steps: int) -> None:
        level = self.hierarchy.levels[level_index]
        for _ in range(steps):
            level.operators.pressure_operator(
                level.pressure, out=level.operator_value,
            )
            wp.launch(
                _weighted_jacobi_update,
                dim=level.cfg.grid.shape,
                inputs=[level.rhs, level.operator_value, level.diagonal,
                        level.pressure, self.settings.jacobi_weight,
                        level.cfg.grid.ng],
                device=self.device,
            )
            self._remove_mean(level_index, level.pressure)

    def _visit(self, level_index: int, residuals: list[float]) -> None:
        levels = self.hierarchy.levels
        if level_index == len(levels) - 1:
            self.hierarchy.solve_coarsest()
            residuals.append(self._residual(level_index))
            return

        self._smooth(level_index, self.settings.pre_smoothing_steps)
        residuals.append(self._residual(level_index))
        coarse = levels[level_index + 1]
        self.hierarchy.restrict(level_index, levels[level_index].residual,
                                coarse.rhs)
        self._remove_mean(level_index + 1, coarse.rhs)
        coarse.pressure.zero_()
        self._visit(level_index + 1, residuals)
        self.hierarchy.prolong(
            level_index + 1, coarse.pressure,
            levels[level_index].pressure, add=True,
        )
        self._remove_mean(level_index, levels[level_index].pressure)
        self._smooth(level_index, self.settings.post_smoothing_steps)

    def cycle(
        self,
        rhs: wp.array,
        pressure: wp.array | None = None,
    ) -> VCycleResult:
        """Apply one V-cycle, optionally improving an existing pressure field."""
        finest = self.hierarchy.levels[0]
        self._copy(rhs, finest.rhs, 0)
        self._remove_mean(0, finest.rhs)
        if pressure is None:
            pressure = wp.zeros(
                finest.grid.cell_shape, dtype=wp.float32, device=self.device,
            )
            finest.pressure.zero_()
        else:
            self._copy(pressure, finest.pressure, 0)
            self._remove_mean(0, finest.pressure)

        before = self._residual(0)
        level_residuals: list[float] = []
        self._visit(0, level_residuals)
        after = self._residual(0)
        self._copy(finest.pressure, pressure, 0)
        factor = after / before if before > 0.0 else 0.0
        return VCycleResult(
            pressure, before, after, factor, tuple(level_residuals),
        )

    def apply(self, rhs: wp.array, out: wp.array) -> wp.array:
        """Apply one zero-start V-cycle as a pressure preconditioner."""
        out.zero_()
        self.cycle(rhs, out)
        return out
