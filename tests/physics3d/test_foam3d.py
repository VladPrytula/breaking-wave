"""Passivity, interface sourcing, and decay gates for 3-D foam."""

import numpy as np
import warp as wp

from shoreloop.config import (BC, BC3DConfig, Grid3DConfig, Physics3DConfig,
                              Solver3DConfig)
from shoreloop.device import DEVICE
from shoreloop.physics3d.foam import Foam3DConfig, InterfaceFoamModel
from shoreloop.physics3d.solver import Solver3D
from shoreloop.physics3d.state import State3D
from shoreloop.physics3d.timestep import TimeStepConfig
from shoreloop.physics3d.viscosity import ViscosityConfig


def _state() -> tuple[Solver3DConfig, State3D]:
    periodic = BC3DConfig(
        xlo=BC.PERIODIC, xhi=BC.PERIODIC,
        ylo=BC.PERIODIC, yhi=BC.PERIODIC,
        zlo=BC.PERIODIC, zhi=BC.PERIODIC,
    )
    cfg = Solver3DConfig(
        grid=Grid3DConfig(nx=8, ny=6, nz=6, Lx=0.8, Ly=0.6, Lz=0.6),
        bc=periodic, device=DEVICE,
    )
    return cfg, State3D(cfg)


def test_foam_is_hydrodynamically_passive_and_sources_only_the_interface():
    cfg, state = _state()
    alpha = np.ones(cfg.grid.shape, dtype=np.float32)
    alpha[:, :, 3:] = 0.0
    alpha[:, :, 2] = 0.5
    state.set_interior("alpha", alpha)
    rng = np.random.default_rng(17)
    for name in ("u", "v", "w"):
        state.set_interior(
            name,
            rng.normal(0.0, 0.02, state.grid.interior_shape(name)).astype(np.float32),
        )
    hydro_before = {
        name: state.get_interior(name).copy()
        for name in ("alpha", "u", "v", "w")
    }
    dissipation = wp.ones(
        state.grid.cell_shape, dtype=wp.float32, device=DEVICE,
    )
    model = InterfaceFoamModel(
        cfg,
        Foam3DConfig(
            dissipation_yield=0.02, deformation_yield=0.0,
            diffusivity=0.0,
        ),
    )
    result = model.apply(
        state, 0.01, dissipation_fields=(dissipation,),
    )
    for name, expected in hydro_before.items():
        np.testing.assert_array_equal(state.get_interior(name), expected)
    foam = state.get_interior("foam")
    assert result.active_source_cells == cfg.grid.nx * cfg.grid.ny
    assert np.all(foam[:, :, 2] > 0.0)
    np.testing.assert_array_equal(foam[:, :, :2], 0.0)
    np.testing.assert_array_equal(foam[:, :, 3:], 0.0)


def test_breaker_scale_deformation_sources_more_foam_than_a_linear_wave():
    maxima = []
    for strain_value in (2.0, 20.0):
        cfg, state = _state()
        state.fill_("alpha", 0.5)
        strain = wp.full(
            state.grid.cell_shape, strain_value,
            dtype=wp.float32, device=DEVICE,
        )
        model = InterfaceFoamModel(
            cfg,
            Foam3DConfig(
                dissipation_yield=0.0, deformation_yield=0.02,
                deformation_threshold=4.0, diffusivity=0.0,
            ),
        )
        maxima.append(model.apply(
            state, 0.01, strain_rate=strain,
        ).maximum_concentration)
    assert maxima[0] == 0.0
    assert maxima[1] > 0.0


def test_foam_decay_uses_the_declared_time_scale():
    cfg, state = _state()
    state.fill_("alpha", 1.0)
    state.fill_("foam", 0.5)
    model = InterfaceFoamModel(
        cfg,
        Foam3DConfig(
            dissipation_yield=0.0, deformation_yield=0.0,
            decay_time=1.0, diffusivity=0.0,
        ),
    )
    model.apply(state, 1.0)
    np.testing.assert_allclose(state.get_interior("foam"), 0.25, rtol=2.0e-6)


def test_dissipation_yield_scales_direct_and_numerical_sources_equally():
    cfg, state = _state()
    state.fill_("alpha", 0.5)  # Uniform unit interface weight; no clipping.
    strain = wp.full(
        state.grid.cell_shape, 6.0, dtype=wp.float32, device=DEVICE,
    )
    power_density = 100.0  # W/m^3 of bulk cell volume, in either channel.
    dissipation = wp.full(
        state.grid.cell_shape, power_density, dtype=wp.float32, device=DEVICE,
    )
    dt = 0.001
    numerical_loss = (
        power_density * dt * cfg.grid.Lx * cfg.grid.Ly * cfg.grid.Lz
    )
    unchanged = {
        name: state.field(name).numpy()
        for name in state.FIELDS if name != "foam"
    }
    for foam_yield in (0.0, 1.0e-4, 2.0e-4):
        settings = Foam3DConfig(
            dissipation_yield=foam_yield, deformation_yield=0.0,
            deformation_threshold=5.0, decay_time=0.8, diffusivity=0.0,
        )
        model = InterfaceFoamModel(cfg, settings)
        expected = (
            foam_yield * power_density * dt / (1.0 + dt / settings.decay_time)
        )
        for numerical in (False, True):
            state.foam.zero_()
            result = model.apply(
                state, dt, strain_rate=strain,
                dissipation_fields=() if numerical else (dissipation,),
                numerical_loss=numerical_loss if numerical else 0.0,
            )
            np.testing.assert_allclose(
                state.get_interior("foam"), expected, rtol=2.0e-6, atol=0.0,
            )
            assert result.numerical_residual_energy_used == (
                numerical_loss if numerical and foam_yield > 0.0 else 0.0
            )
            for name, before in unchanged.items():
                np.testing.assert_array_equal(state.field(name).numpy(), before)


def test_foam_on_off_keeps_the_solver_trajectory_bitwise_identical():
    periodic = BC3DConfig(
        xlo=BC.PERIODIC, xhi=BC.PERIODIC,
        ylo=BC.PERIODIC, yhi=BC.PERIODIC,
        zlo=BC.PERIODIC, zhi=BC.PERIODIC,
    )
    cfg = Solver3DConfig(
        grid=Grid3DConfig(nx=8, ny=6, nz=6, Lx=0.8, Ly=0.6, Lz=0.6),
        physics=Physics3DConfig(g=0.0), bc=periodic, device=DEVICE,
    )
    common = dict(
        cfg=cfg,
        timestep_settings=TimeStepConfig(dt_max=0.01),
        viscosity_settings=ViscosityConfig(molecular_viscosity=0.004),
    )
    without_foam = Solver3D(**common)
    with_foam = Solver3D(
        **common,
        foam_settings=Foam3DConfig(
            dissipation_yield=0.0,
            deformation_yield=0.0,
            diffusivity=0.1,
        ),
    )
    _, y, _ = with_foam.state.grid.coordinates("u")
    velocity = np.broadcast_to(
        0.2 * np.sin(2.0 * np.pi * y / cfg.grid.Ly)[None, :, None],
        with_foam.state.grid.interior_shape("u"),
    ).astype(np.float32)
    for solver in (without_foam, with_foam):
        solver.state.fill_("alpha", 1.0)
        solver.state.set_interior("u", velocity)
    checkerboard = np.indices(cfg.grid.shape).sum(axis=0) % 2
    with_foam.state.set_interior("foam", checkerboard.astype(np.float32))

    # D_f is deliberately large enough that its explicit limit is below the
    # hydrodynamic 0.01 s step. Foam must substep itself, not slow the flow.
    assert with_foam.foam.time_step_limit() < 0.01
    for _ in range(20):
        assert without_foam.step().converged
        assert with_foam.step().converged

    assert with_foam.time == without_foam.time
    np.testing.assert_allclose(with_foam.time, 0.2, rtol=0.0, atol=1.0e-15)
    for name in ("alpha", "u", "v", "w"):
        np.testing.assert_array_equal(
            with_foam.state.get_interior(name),
            without_foam.state.get_interior(name),
        )
    foam = with_foam.state.get_interior("foam")
    assert 0.0 < float(foam.min()) <= float(foam.max()) < 1.0
