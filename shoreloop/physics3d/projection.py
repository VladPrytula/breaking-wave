"""Regular-grid pressure projection for the three-dimensional MAC state."""

from __future__ import annotations

from dataclasses import dataclass
import math

import warp as wp

from shoreloop.config import Solver3DConfig
from shoreloop.physics3d.boundary import fill_boundaries
from shoreloop.physics3d.grid import MACGrid
from shoreloop.physics3d.operators import MACOperators
from shoreloop.physics3d.pcg import PCGConfig, PCGResult, PressurePCG
from shoreloop.physics3d.state import State3D


@dataclass(frozen=True)
class ProjectionConfig:
    """Pressure-solve controls and the accepted post-projection RMS gate."""

    pcg: PCGConfig = PCGConfig(relative_tolerance=1.0e-6)
    rms_divergence_tolerance: float = 1.0e-4

    def __post_init__(self) -> None:
        if not isinstance(self.pcg, PCGConfig):
            raise TypeError("pcg must be a PCGConfig")
        if (not math.isfinite(self.rms_divergence_tolerance)
                or self.rms_divergence_tolerance <= 0.0):
            raise ValueError(
                "rms_divergence_tolerance must be positive and finite",
            )


@dataclass(frozen=True)
class ProjectionResult:
    """Pressure convergence and divergence measurements for one projection."""

    pressure_solve: PCGResult
    converged: bool
    rms_divergence_before: float
    max_divergence_before: float
    rms_divergence_after: float
    max_divergence_after: float


@wp.kernel
def _form_pressure_rhs(
    divergence: wp.array3d(dtype=wp.float32),
    rhs: wp.array3d(dtype=wp.float32),
    scale: wp.float32,
    ng: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    rhs[I, J, K] = scale * divergence[I, J, K]


@wp.kernel
def _correct_velocity(
    velocity: wp.array3d(dtype=wp.float32),
    pressure_gradient: wp.array3d(dtype=wp.float32),
    scale: wp.float32,
    ng: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    velocity[I, J, K] -= scale * pressure_gradient[I, J, K]


@wp.kernel
def _reduce_divergence(
    divergence: wp.array3d(dtype=wp.float32),
    squared_sum: wp.array(dtype=wp.float64),
    maximum: wp.array(dtype=wp.float32),
    ng: int,
):
    i, j, k = wp.tid()
    value = divergence[i + ng, j + ng, k + ng]
    value64 = wp.float64(value)
    wp.atomic_add(squared_sum, 0, value64 * value64)
    wp.atomic_max(maximum, 0, wp.abs(value))


class PressureProjection:
    """Project a boundary-constrained ``State3D`` velocity in place."""

    __slots__ = (
        "cfg", "settings", "grid", "device", "pressure_solver", "operators",
        "_divergence", "_rhs", "_gradient", "_squared_sum", "_maximum",
    )

    def __init__(
        self,
        cfg: Solver3DConfig,
        settings: ProjectionConfig | None = None,
    ):
        self.cfg = cfg
        self.settings = ProjectionConfig() if settings is None else settings
        if not isinstance(self.settings, ProjectionConfig):
            raise TypeError("settings must be a ProjectionConfig")
        self.grid = MACGrid(cfg.grid)
        self.device = cfg.device
        self.pressure_solver = PressurePCG(cfg, self.settings.pcg)
        self.operators: MACOperators = self.pressure_solver.operators
        self._divergence = wp.zeros(
            self.grid.cell_shape, dtype=wp.float32, device=self.device,
        )
        self._rhs = wp.zeros(
            self.grid.cell_shape, dtype=wp.float32, device=self.device,
        )
        self._gradient = tuple(
            wp.zeros(self.grid.padded_shape(name), dtype=wp.float32,
                     device=self.device)
            for name in ("u", "v", "w")
        )
        self._squared_sum = wp.zeros(
            1, dtype=wp.float64, device=self.device,
        )
        self._maximum = wp.zeros(
            1, dtype=wp.float32, device=self.device,
        )

    def _measure_divergence(self) -> tuple[float, float]:
        self._squared_sum.zero_()
        self._maximum.zero_()
        wp.launch(
            _reduce_divergence,
            dim=self.cfg.grid.shape,
            inputs=[self._divergence, self._squared_sum, self._maximum,
                    self.cfg.grid.ng],
            device=self.device,
        )
        rms = math.sqrt(
            float(self._squared_sum.numpy()[0]) / self.cfg.grid.cell_count,
        )
        maximum = float(self._maximum.numpy()[0])
        return rms, maximum

    def project(self, state: State3D, dt: float) -> ProjectionResult:
        """Project ``state`` in place and return pressure/divergence diagnostics.

        The AW-104 solver uses ``A = -D G``.  Consequently the right-hand side
        is ``-(rho/dt) D u*`` and the correction is ``-(dt/rho) G p``.
        """
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"dt must be positive and finite, got {dt!r}")

        fill_boundaries(state)
        self.operators.divergence(
            state.u, state.v, state.w, out=self._divergence,
        )
        rms_before, max_before = self._measure_divergence()

        self._rhs.zero_()
        rho = self.cfg.physics.rho
        wp.launch(
            _form_pressure_rhs,
            dim=self.cfg.grid.shape,
            inputs=[self._divergence, self._rhs, -rho / dt,
                    self.cfg.grid.ng],
            device=self.device,
        )
        pressure_solve = self.pressure_solver.solve(
            self._rhs, pressure=state.p,
        )

        if pressure_solve.converged:
            gradient = self.operators.pressure_gradient(
                state.p, out=self._gradient,
            )
            correction_scale = dt / rho
            for name, velocity, component in zip(
                ("u", "v", "w"),
                (state.u, state.v, state.w),
                gradient,
            ):
                wp.launch(
                    _correct_velocity,
                    dim=self.grid.interior_shape(name),
                    inputs=[velocity, component, correction_scale,
                            self.cfg.grid.ng],
                    device=self.device,
                )
            fill_boundaries(state)
            self.operators.divergence(
                state.u, state.v, state.w, out=self._divergence,
            )
            rms_after, max_after = self._measure_divergence()
        else:
            rms_after, max_after = rms_before, max_before

        converged = (
            pressure_solve.converged
            and rms_after <= self.settings.rms_divergence_tolerance
        )
        return ProjectionResult(
            pressure_solve=pressure_solve,
            converged=converged,
            rms_divergence_before=rms_before,
            max_divergence_before=max_before,
            rms_divergence_after=rms_after,
            max_divergence_after=max_after,
        )
