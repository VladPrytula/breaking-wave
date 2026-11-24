"""Physical time-step limits for the regular three-dimensional solver."""

from __future__ import annotations

from dataclasses import dataclass
import math

import warp as wp

from shoreloop.config import Solver3DConfig
from shoreloop.physics3d.grid import MACGrid
from shoreloop.physics3d.state import State3D


@dataclass(frozen=True)
class TimeStepConfig:
    """CFL factors and switchable constraints used to select ``dt``."""

    dt_max: float = 0.05
    advective_cfl: float = 0.5
    gravity_cfl: float = 0.5
    viscous_cfl: float = 0.25
    capillary_cfl: float = 0.25
    viscous_limit_enabled: bool = False
    kinematic_viscosity: float = 1.0e-6

    def __post_init__(self) -> None:
        for name in (
            "dt_max", "advective_cfl", "gravity_cfl", "viscous_cfl",
            "capillary_cfl", "kinematic_viscosity",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("viscous_limit_enabled",):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")


@dataclass(frozen=True)
class TimeStepResult:
    """Selected step and every analytic limit used in the decision."""

    dt: float
    limiting_constraint: str
    max_face_speed: float
    advective_limit: float
    gravity_limit: float
    viscous_limit: float
    capillary_limit: float


@wp.kernel
def _reduce_max_face_speed(
    velocity: wp.array3d(dtype=wp.float32),
    maximum: wp.array(dtype=wp.float32),
    ng: int,
):
    i, j, k = wp.tid()
    wp.atomic_max(maximum, 0, wp.abs(velocity[i + ng, j + ng, k + ng]))


class TimeStepController:
    """Reusable CUDA maximum-speed reduction and host limit calculation."""

    __slots__ = ("cfg", "settings", "grid", "device", "_maximum")

    def __init__(
        self,
        cfg: Solver3DConfig,
        settings: TimeStepConfig | None = None,
    ):
        self.cfg = cfg
        self.settings = TimeStepConfig() if settings is None else settings
        if not isinstance(self.settings, TimeStepConfig):
            raise TypeError("settings must be a TimeStepConfig")
        self.grid = MACGrid(cfg.grid)
        self.device = cfg.device
        self._maximum = wp.zeros(
            1, dtype=wp.float32, device=self.device,
        )

    def _max_face_speed(self, state: State3D) -> float:
        self._maximum.zero_()
        for name in ("u", "v", "w"):
            wp.launch(
                _reduce_max_face_speed,
                dim=self.grid.interior_shape(name),
                inputs=[state.field(name), self._maximum, self.cfg.grid.ng],
                device=self.device,
            )
        return float(self._maximum.numpy()[0])

    def compute(self, state: State3D) -> TimeStepResult:
        """Measure face speed and return the minimum active physical limit."""
        speed = self._max_face_speed(state)
        g = self.cfg.grid
        p = self.cfg.physics
        s = self.settings
        spacing = min(g.dx, g.dy, g.dz)

        advective = math.inf
        if speed > 0.0:
            advective = s.advective_cfl * spacing / speed

        gravity = math.inf
        if p.g > 0.0:
            gravity = s.gravity_cfl * math.sqrt(spacing / p.g)

        viscous = math.inf
        if s.viscous_limit_enabled:
            viscous = s.viscous_cfl * spacing * spacing / s.kinematic_viscosity

        capillary = math.inf
        if p.surface_tension > 0.0:
            capillary = s.capillary_cfl * math.sqrt(
                p.rho * spacing * spacing * spacing / p.surface_tension,
            )

        limits = (
            ("maximum", s.dt_max),
            ("advective", advective),
            ("gravity", gravity),
            ("viscous", viscous),
            ("capillary", capillary),
        )
        winner, dt = min(limits, key=lambda item: item[1])
        return TimeStepResult(
            dt=dt,
            limiting_constraint=winner,
            max_face_speed=speed,
            advective_limit=advective,
            gravity_limit=gravity,
            viscous_limit=viscous,
            capillary_limit=capillary,
        )
