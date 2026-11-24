"""Closed-domain mechanical-energy budget gates."""

import numpy as np

from shoreloop.config import (BC, BC3DConfig, Grid3DConfig, Physics3DConfig,
                              Solver3DConfig)
from shoreloop.device import DEVICE
from shoreloop.physics3d.energy import EnergyLedger
from shoreloop.physics3d.pcg import PCGConfig
from shoreloop.physics3d.projection import ProjectionConfig
from shoreloop.physics3d.solver import Solver3D
from shoreloop.physics3d.state import State3D
from shoreloop.physics3d.viscosity import ViscosityConfig, ViscosityModel
from shoreloop.physics3d.waves import PeriodicWaveComponent


def test_closed_viscous_decay_budget_closes_to_float32_tolerance():
    periodic = BC3DConfig(
        xlo=BC.PERIODIC, xhi=BC.PERIODIC,
        ylo=BC.PERIODIC, yhi=BC.PERIODIC,
        zlo=BC.PERIODIC, zhi=BC.PERIODIC,
    )
    cfg = Solver3DConfig(
        grid=Grid3DConfig(nx=12, ny=12, nz=8, Lx=1.0, Ly=1.0, Lz=1.0),
        physics=Physics3DConfig(g=0.0), bc=periodic, device=DEVICE,
    )
    state = State3D(cfg)
    state.fill_("alpha", 1.0)
    _, y, _ = state.grid.coordinates("u")
    state.set_interior(
        "u",
        np.broadcast_to(
            np.sin(2.0 * np.pi * y)[None, :, None],
            state.grid.interior_shape("u"),
        ).astype(np.float32),
    )
    model = ViscosityModel(
        cfg, ViscosityConfig(molecular_viscosity=0.004),
    )
    ledger = EnergyLedger(cfg)
    ledger.begin_step(state)
    direct = model.apply(state, 0.002)
    budget = ledger.close_step(
        state, molecular_loss=direct.molecular_energy_loss,
    )
    assert budget.molecular_loss > 0.0
    assert abs(budget.numerical_residual) < 2.0e-4 * budget.molecular_loss


def test_signed_residual_recovery_is_not_rectified_twice():
    periodic = BC3DConfig(
        xlo=BC.PERIODIC, xhi=BC.PERIODIC,
        ylo=BC.PERIODIC, yhi=BC.PERIODIC,
        zlo=BC.PERIODIC, zhi=BC.PERIODIC,
    )
    cfg = Solver3DConfig(
        grid=Grid3DConfig(nx=4, ny=3, nz=2, Lx=0.4, Ly=0.3, Lz=0.2),
        physics=Physics3DConfig(g=0.0), bc=periodic, device=DEVICE,
    )
    state = State3D(cfg)
    state.fill_("alpha", 1.0)
    state.fill_("u", 1.0)
    ledger = EnergyLedger(cfg)

    ledger.begin_step(state)
    state.zero_("u")
    first_loss = ledger.close_step(state)
    ledger.begin_step(state)
    state.fill_("u", 1.0)
    recovery = ledger.close_step(state)
    ledger.begin_step(state)
    state.zero_("u")
    repeated_loss = ledger.close_step(state)

    assert first_loss.numerical_residual > 0.0
    assert recovery.numerical_residual < 0.0
    assert repeated_loss.numerical_residual > 0.0
    assert repeated_loss.positive_numerical_loss == 0.0
    np.testing.assert_allclose(
        ledger.positive_numerical_loss,
        first_loss.positive_numerical_loss,
        rtol=0.0, atol=1.0e-12,
    )


def test_boundary_surface_work_is_labelled_without_changing_the_residual():
    cfg = Solver3DConfig(
        grid=Grid3DConfig(nx=4, ny=3, nz=2, Lx=0.4, Ly=0.3, Lz=0.2),
        device=DEVICE,
    )
    state = State3D(cfg)
    ledger = EnergyLedger(cfg)
    ledger.begin_step(state)
    budget = ledger.close_step(
        state,
        boundary_work=1.25,
        boundary_surface_work=-0.75,
    )

    assert budget.boundary_work == 1.25
    assert budget.boundary_surface_work == -0.75
    assert budget.numerical_residual == 1.25
    assert ledger.boundary_work == 1.25
    assert ledger.boundary_surface_work == -0.75


def test_closed_periodic_wave_full_step_budget_is_small_relative_to_kinetic_energy():
    component = PeriodicWaveComponent(
        period=0.8, depth=0.28, alongshore_length=0.16, amplitude=0.003,
    )
    grid = Grid3DConfig(
        nx=32, ny=4, nz=12,
        Lx=component.wavelength, Ly=0.16, Lz=0.48,
    )
    cfg = Solver3DConfig(
        grid=grid,
        bc=BC3DConfig(
            xlo=BC.PERIODIC, xhi=BC.PERIODIC,
            ylo=BC.PERIODIC, yhi=BC.PERIODIC,
            zlo=BC.WALL, zhi=BC.WALL,
        ),
        device=DEVICE,
    )
    solver = Solver3D(
        cfg,
        ProjectionConfig(
            pcg=PCGConfig(
                relative_tolerance=2.0e-5,
                max_iterations=300,
                preconditioner="multigrid",
            ),
            rms_divergence_tolerance=8.0e-4,
        ),
    )
    x, _, z = solver.state.grid.coordinates("cell")
    surface = component.depth + component.amplitude * np.cos(component.k * x)
    alpha = np.broadcast_to(
        np.clip(
            (surface[:, None, None] - (z[None, None, :] - 0.5 * grid.dz))
            / grid.dz,
            0.0, 1.0,
        ),
        grid.shape,
    ).copy()
    solver.state.set_interior("alpha", alpha.astype(np.float32))
    for axis, name in enumerate(("u", "v", "w")):
        xs, ys, zs = solver.state.grid.coordinates(name)
        velocity = np.zeros(solver.state.grid.interior_shape(name), dtype=np.float32)
        for index in np.ndindex(velocity.shape):
            velocity[index] = component.velocity(
                float(xs[index[0]]),
                float(ys[index[1]]),
                float(zs[index[2]] - component.depth),
                0.0,
            )[axis]
        solver.state.set_interior(name, velocity)

    ledger = EnergyLedger(cfg)
    solver.set_energy_ledger(ledger)
    kinetic_scale = ledger.measure(solver.state).kinetic
    for _ in range(20):
        assert solver.step(0.001).converged

    assert kinetic_scale > 0.0
    assert abs(ledger.numerical_residual) < 0.1 * kinetic_scale
    assert ledger.positive_numerical_loss < 0.1 * kinetic_scale
