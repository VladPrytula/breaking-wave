"""Matrix-free PCG for the regular-grid pressure equation.

The production vectors remain float32 on CUDA.  Dot products, residual norms,
and constant-mode reductions accumulate in float64 on the same device before a
single scalar is read by the host for each reduction.

Every boundary supported by the AW-103 pressure gradient leaves constants in
the null space: wall and provisional open sides have zero normal gradient, and
periodic sides wrap.  Solves therefore project the right-hand side, Krylov
vectors, and returned pressure onto the zero-mean subspace.  In particular,
this removes the gauge mode required for closed periodic/Neumann domains.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import warp as wp

from shoreloop.config import Solver3DConfig
from shoreloop.physics3d.grid import MACGrid
from shoreloop.physics3d.operators import MACOperators


@dataclass(frozen=True)
class PCGConfig:
    """Stopping controls and preconditioner for the float32 pressure solve."""

    relative_tolerance: float = 1.0e-5
    absolute_tolerance: float = 0.0
    max_iterations: int = 200
    preconditioner: str = "jacobi"

    def __post_init__(self) -> None:
        if (not math.isfinite(self.relative_tolerance)
                or self.relative_tolerance <= 0.0):
            raise ValueError("relative_tolerance must be positive and finite")
        if (not math.isfinite(self.absolute_tolerance)
                or self.absolute_tolerance < 0.0):
            raise ValueError("absolute_tolerance must be non-negative and finite")
        if (isinstance(self.max_iterations, bool)
                or not isinstance(self.max_iterations, int)
                or self.max_iterations <= 0):
            raise ValueError("max_iterations must be a positive integer")
        if self.preconditioner not in ("jacobi", "multigrid"):
            raise ValueError("preconditioner must be 'jacobi' or 'multigrid'")


@dataclass(frozen=True)
class PCGResult:
    """Pressure and convergence diagnostics from one PCG solve."""

    pressure: wp.array
    converged: bool
    iterations: int
    absolute_residual: float
    relative_residual: float


@wp.kernel
def _copy_cell_interior(
    source: wp.array3d(dtype=wp.float32),
    target: wp.array3d(dtype=wp.float32),
    ng: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    target[I, J, K] = source[I, J, K]


@wp.kernel
def _form_residual(
    rhs: wp.array3d(dtype=wp.float32),
    operator_pressure: wp.array3d(dtype=wp.float32),
    residual: wp.array3d(dtype=wp.float32),
    ng: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    residual[I, J, K] = rhs[I, J, K] - operator_pressure[I, J, K]


@wp.kernel
def _apply_jacobi(
    residual: wp.array3d(dtype=wp.float32),
    diagonal: wp.array3d(dtype=wp.float32),
    out: wp.array3d(dtype=wp.float32),
    ng: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    value = wp.float32(0.0)
    if diagonal[I, J, K] > wp.float32(0.0):
        value = residual[I, J, K] / diagonal[I, J, K]
    out[I, J, K] = value


@wp.kernel
def _update_solution_and_residual(
    pressure: wp.array3d(dtype=wp.float32),
    residual: wp.array3d(dtype=wp.float32),
    direction: wp.array3d(dtype=wp.float32),
    operator_direction: wp.array3d(dtype=wp.float32),
    alpha: wp.float32,
    ng: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    pressure[I, J, K] += alpha * direction[I, J, K]
    residual[I, J, K] -= alpha * operator_direction[I, J, K]


@wp.kernel
def _update_direction(
    preconditioned: wp.array3d(dtype=wp.float32),
    direction: wp.array3d(dtype=wp.float32),
    beta: wp.float32,
    ng: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    direction[I, J, K] = preconditioned[I, J, K] + beta * direction[I, J, K]


@wp.kernel
def _form_dot_products(
    left: wp.array3d(dtype=wp.float32),
    right: wp.array3d(dtype=wp.float32),
    products: wp.array3d(dtype=wp.float64),
    ng: int,
    cell_volume: wp.float64,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    product = wp.float64(left[I, J, K]) * wp.float64(right[I, J, K])
    products[i, j, k] = product * cell_volume


@wp.kernel
def _reduce_sum(
    field: wp.array3d(dtype=wp.float32),
    out: wp.array(dtype=wp.float64),
    ng: int,
):
    i, j, k = wp.tid()
    wp.atomic_add(out, 0, wp.float64(field[i + ng, j + ng, k + ng]))


@wp.kernel
def _subtract_mean(
    field: wp.array3d(dtype=wp.float32),
    mean: wp.float32,
    ng: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    field[I, J, K] -= mean


class PressurePCG:
    """Reusable CUDA workspace with Jacobi or geometric-multigrid PCG."""

    __slots__ = (
        "cfg", "settings", "grid", "device", "operators", "diagonal",
        "_rhs", "_residual", "_preconditioned", "_direction",
        "_operator_vector", "_gradient", "_dot_products", "_reduction",
        "_cell_volume", "_preconditioner",
    )

    def __init__(self, cfg: Solver3DConfig, settings: PCGConfig | None = None):
        self.cfg = cfg
        self.settings = PCGConfig() if settings is None else settings
        if not isinstance(self.settings, PCGConfig):
            raise TypeError("settings must be a PCGConfig")
        self.grid = MACGrid(cfg.grid)
        self.device = cfg.device
        self.operators = MACOperators(cfg)
        self.diagonal = self.operators.pressure_diagonal()

        def cell_array() -> wp.array:
            return wp.zeros(self.grid.cell_shape, dtype=wp.float32,
                            device=self.device)

        self._rhs = cell_array()
        self._residual = cell_array()
        self._preconditioned = cell_array()
        self._direction = cell_array()
        self._operator_vector = cell_array()
        self._gradient = tuple(
            wp.zeros(self.grid.padded_shape(name), dtype=wp.float32,
                     device=self.device)
            for name in ("u", "v", "w")
        )
        self._dot_products = wp.zeros(
            cfg.grid.shape, dtype=wp.float64, device=self.device,
        )
        self._reduction = wp.zeros(1, dtype=wp.float64, device=self.device)
        g = cfg.grid
        self._cell_volume = g.dx * g.dy * g.dz
        self._preconditioner = None
        if self.settings.preconditioner == "multigrid":
            from shoreloop.physics3d.multigrid import MultigridVCycle
            self._preconditioner = MultigridVCycle(cfg)

    def apply_operator(self, pressure: wp.array, out: wp.array | None = None
                       ) -> wp.array:
        """Apply the positive matrix ``-D G`` using reusable face workspace."""
        return self.operators.pressure_operator(
            pressure, out=out, gradient_out=self._gradient,
        )

    def _copy(self, source: wp.array, target: wp.array) -> None:
        target.zero_()
        wp.launch(
            _copy_cell_interior,
            dim=self.cfg.grid.shape,
            inputs=[source, target, self.cfg.grid.ng],
            device=self.device,
        )

    def _dot(self, left: wp.array, right: wp.array) -> float:
        wp.launch(
            _form_dot_products,
            dim=self.cfg.grid.shape,
            inputs=[left, right, self._dot_products, self.cfg.grid.ng,
                    self._cell_volume],
            device=self.device,
        )
        wp.utils.array_sum(self._dot_products, out=self._reduction)
        return float(self._reduction.numpy()[0])

    def _norm(self, field: wp.array) -> float:
        return math.sqrt(max(self._dot(field, field), 0.0))

    def _remove_mean(self, field: wp.array) -> None:
        self._reduction.zero_()
        wp.launch(
            _reduce_sum,
            dim=self.cfg.grid.shape,
            inputs=[field, self._reduction, self.cfg.grid.ng],
            device=self.device,
        )
        mean = float(self._reduction.numpy()[0]) / self.cfg.grid.cell_count
        wp.launch(
            _subtract_mean,
            dim=self.cfg.grid.shape,
            inputs=[field, mean, self.cfg.grid.ng],
            device=self.device,
        )

    def _recompute_residual(self, pressure: wp.array) -> float:
        self.apply_operator(pressure, out=self._operator_vector)
        wp.launch(
            _form_residual,
            dim=self.cfg.grid.shape,
            inputs=[self._rhs, self._operator_vector, self._residual,
                    self.cfg.grid.ng],
            device=self.device,
        )
        self._remove_mean(self._residual)
        return self._norm(self._residual)

    def _apply_preconditioner(self) -> None:
        if self._preconditioner is None:
            wp.launch(
                _apply_jacobi,
                dim=self.cfg.grid.shape,
                inputs=[self._residual, self.diagonal, self._preconditioned,
                        self.cfg.grid.ng],
                device=self.device,
            )
        else:
            self._preconditioner.apply(
                self._residual, self._preconditioned,
            )
        self._remove_mean(self._preconditioned)

    @staticmethod
    def _relative_residual(absolute: float, reference: float) -> float:
        if reference > 0.0:
            return absolute / reference
        return 0.0 if absolute == 0.0 else math.inf

    def solve(self, rhs: wp.array, pressure: wp.array | None = None) -> PCGResult:
        """Solve ``-D G p = rhs`` and return a zero-mean pressure.

        ``rhs`` is copied and projected to zero mean, so the caller's array is
        unchanged.  Supplying ``pressure`` uses and updates it as a warm start;
        omitting it allocates a zero initial guess.
        """
        self.operators._check_field("cell", rhs)
        if pressure is None:
            pressure = wp.zeros(self.grid.cell_shape, dtype=wp.float32,
                                device=self.device)
        else:
            self.operators._check_field("p", pressure)

        self._copy(rhs, self._rhs)
        self._remove_mean(self._rhs)
        self._remove_mean(pressure)

        rhs_norm = self._norm(self._rhs)
        target = max(
            self.settings.absolute_tolerance,
            self.settings.relative_tolerance * rhs_norm,
        )
        residual_norm = self._recompute_residual(pressure)
        if residual_norm <= target:
            return PCGResult(
                pressure=pressure,
                converged=True,
                iterations=0,
                absolute_residual=residual_norm,
                relative_residual=self._relative_residual(residual_norm, rhs_norm),
            )

        self._apply_preconditioner()
        self._copy(self._preconditioned, self._direction)
        residual_preconditioned = self._dot(
            self._residual, self._preconditioned,
        )

        iterations = 0
        converged = False
        for iteration in range(1, self.settings.max_iterations + 1):
            if (not math.isfinite(residual_preconditioned)
                    or residual_preconditioned <= 0.0):
                break

            self.apply_operator(self._direction, out=self._operator_vector)
            curvature = self._dot(self._direction, self._operator_vector)
            if not math.isfinite(curvature) or curvature <= 0.0:
                break

            alpha = residual_preconditioned / curvature
            if not math.isfinite(alpha):
                break
            wp.launch(
                _update_solution_and_residual,
                dim=self.cfg.grid.shape,
                inputs=[pressure, self._residual, self._direction,
                        self._operator_vector, alpha, self.cfg.grid.ng],
                device=self.device,
            )
            self._remove_mean(self._residual)
            iterations = iteration
            residual_norm = self._norm(self._residual)

            restart = False
            if residual_norm <= target:
                # Recursive CG residuals can drift from b-Ap in float32.  Only
                # claim convergence after measuring the true residual.
                residual_norm = self._recompute_residual(pressure)
                if residual_norm <= target:
                    converged = True
                    break
                restart = True

            self._apply_preconditioner()
            next_residual_preconditioned = self._dot(
                self._residual, self._preconditioned,
            )
            if (not math.isfinite(next_residual_preconditioned)
                    or next_residual_preconditioned <= 0.0):
                residual_preconditioned = next_residual_preconditioned
                break

            if restart:
                self._copy(self._preconditioned, self._direction)
            else:
                beta = next_residual_preconditioned / residual_preconditioned
                if not math.isfinite(beta):
                    residual_preconditioned = next_residual_preconditioned
                    break
                wp.launch(
                    _update_direction,
                    dim=self.cfg.grid.shape,
                    inputs=[self._preconditioned, self._direction, beta,
                            self.cfg.grid.ng],
                    device=self.device,
                )
            residual_preconditioned = next_residual_preconditioned

        self._remove_mean(pressure)
        residual_norm = self._recompute_residual(pressure)
        converged = converged or residual_norm <= target
        return PCGResult(
            pressure=pressure,
            converged=converged,
            iterations=iterations,
            absolute_residual=residual_norm,
            relative_residual=self._relative_residual(residual_norm, rhs_norm),
        )
