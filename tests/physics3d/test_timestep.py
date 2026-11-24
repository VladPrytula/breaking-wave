"""Direct analytic checks for the three-dimensional time-step controller."""

import math

import numpy as np
import pytest

from shoreloop.config import (Grid3DConfig, Physics3DConfig,
                              Solver3DConfig)
from shoreloop.device import DEVICE
from shoreloop.physics3d import (State3D, TimeStepConfig,
                                 TimeStepController)


def _state_with_speed(cfg: Solver3DConfig, speed: float) -> State3D:
    state = State3D(cfg)
    state.set_interior(
        "v",
        np.full(state.grid.interior_shape("v"), speed, dtype=np.float32),
    )
    return state


def test_timestep_limits_have_expected_velocity_and_grid_scaling():
    coarse_cfg = Solver3DConfig(
        grid=Grid3DConfig(nx=10, ny=10, nz=10, Lx=1.0, Ly=1.0, Lz=1.0),
        device=DEVICE,
    )
    fine_cfg = Solver3DConfig(
        grid=Grid3DConfig(nx=20, ny=20, nz=20, Lx=1.0, Ly=1.0, Lz=1.0),
        device=DEVICE,
    )
    base = TimeStepConfig(dt_max=10.0)
    coarse_controller = TimeStepController(coarse_cfg, base)
    fine_controller = TimeStepController(fine_cfg, base)

    coarse = coarse_controller.compute(_state_with_speed(coarse_cfg, 2.0))
    doubled = coarse_controller.compute(_state_with_speed(coarse_cfg, 4.0))
    fine = fine_controller.compute(_state_with_speed(fine_cfg, 2.0))

    assert coarse.max_face_speed == pytest.approx(2.0)
    assert doubled.max_face_speed == pytest.approx(4.0)
    assert coarse.limiting_constraint == "advective"
    assert doubled.advective_limit == pytest.approx(0.5 * coarse.advective_limit)
    assert fine.advective_limit == pytest.approx(0.5 * coarse.advective_limit)
    assert fine.gravity_limit == pytest.approx(
        math.sqrt(0.5) * coarse.gravity_limit,
    )
    assert math.isinf(coarse.viscous_limit)
    assert math.isinf(coarse.capillary_limit)

    optional = TimeStepConfig(
        dt_max=10.0,
        viscous_limit_enabled=True,
        kinematic_viscosity=0.5,
    )
    coarse_capillary_cfg = Solver3DConfig(
        grid=coarse_cfg.grid,
        physics=Physics3DConfig(surface_tension=1000.0),
        device=DEVICE,
    )
    fine_capillary_cfg = Solver3DConfig(
        grid=fine_cfg.grid,
        physics=Physics3DConfig(surface_tension=1000.0),
        device=DEVICE,
    )
    coarse_optional = TimeStepController(
        coarse_capillary_cfg, optional,
    ).compute(
        _state_with_speed(coarse_capillary_cfg, 0.0),
    )
    fine_optional = TimeStepController(
        fine_capillary_cfg, optional,
    ).compute(
        _state_with_speed(fine_capillary_cfg, 0.0),
    )

    assert coarse_optional.limiting_constraint == "viscous"
    assert fine_optional.viscous_limit == pytest.approx(
        0.25 * coarse_optional.viscous_limit,
    )
    assert fine_optional.capillary_limit == pytest.approx(
        0.5 ** 1.5 * coarse_optional.capillary_limit,
    )
