"""Short filled-box integration regression for the fractional-step driver."""

import numpy as np
import pytest

from shoreloop.config import (Grid3DConfig, Physics3DConfig, Solver3DConfig)
from shoreloop.device import DEVICE
from shoreloop.physics3d import ProjectionConfig, Solver3D


def _fill_box(solver: Solver3D) -> None:
    solver.state.set_interior(
        "alpha",
        np.ones(solver.state.grid.interior_shape("alpha"), dtype=np.float32),
    )


def test_fractional_step_preserves_uniform_flow_and_projects_gravity():
    grid = Grid3DConfig(nx=8, ny=6, nz=5, Lx=0.8, Ly=0.6, Lz=0.5)
    zero_g = Solver3DConfig(
        grid=grid,
        physics=Physics3DConfig(g=0.0),
        device=DEVICE,
    )
    uniform = Solver3D(zero_g)
    _fill_box(uniform)
    uniform.state.set_interior(
        "v",
        np.full(uniform.state.grid.interior_shape("v"), 0.4,
                dtype=np.float32),
    )

    for _ in range(4):
        result = uniform.step(0.01)
        assert result.converged
        assert (
            result.rms_divergence_after
            <= uniform.projection.settings.rms_divergence_tolerance
        )

    assert uniform.steps == 4
    assert uniform.time == pytest.approx(0.04)
    np.testing.assert_array_equal(uniform.state.get_interior("u"),
                                  np.float32(0.0))
    np.testing.assert_array_equal(uniform.state.get_interior("v"),
                                  np.float32(0.4))
    np.testing.assert_array_equal(uniform.state.get_interior("w"),
                                  np.float32(0.0))

    gravity = Solver3D(
        Solver3DConfig(grid=grid, device=DEVICE),
        ProjectionConfig(rms_divergence_tolerance=5.0e-5),
    )
    _fill_box(gravity)
    for _ in range(3):
        result = gravity.step(0.01)
        assert result.converged

    assert gravity.steps == 3
    assert gravity.time == pytest.approx(0.03)
    assert max(
        np.max(np.abs(gravity.state.get_interior(name)))
        for name in ("u", "v", "w")
    ) < 2.0e-6
    pressure = gravity.state.get_interior("p")
    vertical_difference = np.diff(pressure, axis=2)
    np.testing.assert_allclose(
        vertical_difference,
        -gravity.cfg.physics.rho * gravity.cfg.physics.g * grid.dz,
        rtol=3.0e-6,
        atol=3.0e-3,
    )


def test_gravity_does_not_accelerate_the_inactive_vof_fringe():
    grid = Grid3DConfig(nx=6, ny=4, nz=4, Lx=0.6, Ly=0.4, Lz=0.4)
    solver = Solver3D(Solver3DConfig(grid=grid, device=DEVICE))
    alpha = np.zeros(solver.state.grid.interior_shape("alpha"), dtype=np.float32)
    alpha[:, :, 0] = 1.0
    alpha[:, :, 1] = 0.25
    solver.state.set_interior("alpha", alpha)

    for _ in range(4):
        assert solver.step(0.005).converged

    # The face above the inactive partial cell is outside the pressure domain.
    # It is an extrapolation site, so body force must not leave it in free fall.
    np.testing.assert_allclose(
        solver.state.get_interior("w")[:, :, 2], 0.0, atol=2.0e-7,
    )


def test_step_reuses_an_immediately_precomputed_timestep():
    cfg = Solver3DConfig(
        grid=Grid3DConfig(nx=8, ny=6, nz=4, Lx=0.8, Ly=0.6, Lz=0.4),
        physics=Physics3DConfig(g=0.0),
        device=DEVICE,
    )
    internal = Solver3D(cfg)
    reused = Solver3D(cfg)
    for solver in (internal, reused):
        _fill_box(solver)
        solver.state.set_interior(
            "u",
            np.full(
                solver.state.grid.interior_shape("u"),
                0.2,
                dtype=np.float32,
            ),
        )

    timestep = reused.compute_dt()
    reused.step(0.01, timestep=timestep)
    internal.step(0.01)

    assert reused.last_timestep == timestep
    for name in reused.state.FIELDS:
        np.testing.assert_array_equal(
            reused.state.field(name).numpy(),
            internal.state.field(name).numpy(),
        )
