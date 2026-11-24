"""Dissipation gates for molecular and Smagorinsky MAC diffusion."""

import numpy as np

from shoreloop.config import BC, BC3DConfig, Grid3DConfig, Solver3DConfig
from shoreloop.device import DEVICE
from shoreloop.physics3d.flow_diagnostics import measure_turbulence
from shoreloop.physics3d.momentum import compute_face_liquid_mass
from shoreloop.physics3d.state import State3D
from shoreloop.physics3d.viscosity import ViscosityConfig, ViscosityModel


def _periodic_state(n: int = 12) -> tuple[Solver3DConfig, State3D]:
    periodic = BC3DConfig(
        xlo=BC.PERIODIC, xhi=BC.PERIODIC,
        ylo=BC.PERIODIC, yhi=BC.PERIODIC,
        zlo=BC.PERIODIC, zhi=BC.PERIODIC,
    )
    cfg = Solver3DConfig(
        grid=Grid3DConfig(nx=n, ny=n, nz=n, Lx=1.0, Ly=1.0, Lz=1.0),
        bc=periodic, device=DEVICE,
    )
    state = State3D(cfg)
    state.fill_("alpha", 1.0)
    return cfg, state


def _kinetic_energy(state: State3D) -> float:
    cell_volume = state.cfg.grid.dx * state.cfg.grid.dy * state.cfg.grid.dz
    total = 0.0
    for axis, name in enumerate(("u", "v", "w")):
        mass = compute_face_liquid_mass(state, name).numpy()
        velocity = state.field(name).numpy()
        interior = list(state.grid.interior_slices(name))
        if getattr(state.cfg.bc, f"{'xyz'[axis]}lo") == BC.PERIODIC:
            interior[axis] = slice(
                state.cfg.grid.ng,
                state.cfg.grid.ng + state.cfg.grid.shape[axis],
            )
        selection = tuple(interior)
        total += 0.5 * state.cfg.physics.rho * cell_volume * float(
            np.sum(
                mass[selection] * velocity[selection] * velocity[selection],
                dtype=np.float64,
            )
        )
    return total


def _shear(state: State3D) -> None:
    _, y, _ = state.grid.coordinates("u")
    u = np.broadcast_to(
        np.sin(2.0 * np.pi * y)[None, :, None],
        state.grid.interior_shape("u"),
    ).astype(np.float32)
    state.set_interior("u", u)


def test_molecular_viscosity_is_dissipative_and_monotone():
    loss_rates = []
    decay_factors = []
    for viscosity in (0.002, 0.008):
        cfg, state = _periodic_state()
        _shear(state)
        before = _kinetic_energy(state)
        model = ViscosityModel(
            cfg, ViscosityConfig(molecular_viscosity=viscosity),
        )
        dt = 0.002
        assert dt < model.time_step_limit(state)
        result = model.apply(state, dt)
        after = _kinetic_energy(state)
        assert after < before
        assert result.molecular_energy_loss > 0.0
        np.testing.assert_allclose(
            result.molecular_energy_loss, before - after, rtol=1.0e-4,
        )
        loss_rates.append(result.molecular_energy_loss / dt)
        decay_factors.append(after / before)
    assert loss_rates[1] > loss_rates[0]
    assert decay_factors[1] < decay_factors[0]


def test_smagorinsky_is_zero_for_translation_and_increases_decay_with_cs():
    cfg, rigid = _periodic_state()
    for name in ("u", "v", "w"):
        rigid.fill_(name, 0.7)
    model = ViscosityModel(cfg, ViscosityConfig(smagorinsky_cs=0.18))
    assert model.update_sgs(rigid) == 0.0
    np.testing.assert_array_equal(model.sgs_viscosity.numpy(), 0.0)

    losses = []
    enstrophies = []
    for cs in (0.08, 0.18):
        cfg, state = _periodic_state()
        _shear(state)
        model = ViscosityModel(cfg, ViscosityConfig(smagorinsky_cs=cs))
        result = model.apply(state, 0.002)
        assert result.sgs_energy_loss >= 0.0
        assert result.maximum_sgs_viscosity > 0.0
        losses.append(result.sgs_energy_loss)
        enstrophies.append(measure_turbulence(state)[1].resolved_enstrophy)
    assert losses[1] > losses[0]
    assert enstrophies[1] < enstrophies[0]


def test_viscosity_respects_closed_apertures():
    cfg, state = _periodic_state(8)
    _shear(state)
    aperture = state.get_geometry_interior("u_aperture")
    aperture[4] = 0.0
    state.set_geometry_interior("u_aperture", aperture)
    velocity = state.get_interior("u")
    velocity[4] = 0.0
    state.set_interior("u", velocity)
    model = ViscosityModel(
        cfg, ViscosityConfig(molecular_viscosity=0.004),
    )
    model.apply(state, 0.001)
    np.testing.assert_array_equal(state.get_interior("u")[4], 0.0)
