"""Liquid-volume-consistent staggered momentum transport regression."""

from dataclasses import replace

import numpy as np
import pytest

from shoreloop.config import BC, BC3DConfig, Grid3DConfig, Solver3DConfig
from shoreloop.device import DEVICE
from shoreloop.physics3d import (MomentumAdvector, State3D,
                                 compute_face_liquid_mass, initialize_box)
from shoreloop.physics3d.unsplit_advection import UnsplitVOFAdvector
from shoreloop.physics3d.vof import measure_vof


def test_transport_timestep_uses_the_sum_of_directional_courant_rates():
    periodic = BC3DConfig(
        xlo=BC.PERIODIC, xhi=BC.PERIODIC,
        ylo=BC.PERIODIC, yhi=BC.PERIODIC,
        zlo=BC.PERIODIC, zhi=BC.PERIODIC,
    )
    cfg = Solver3DConfig(
        grid=Grid3DConfig(nx=10, ny=8, nz=6, Lx=1.0, Ly=1.6, Lz=1.5),
        bc=periodic,
        device=DEVICE,
    )
    state = State3D(cfg)
    initialize_box(state, (0.0, 0.0, 0.0), (1.0, 1.6, 1.5))
    state.set_interior(
        "u", np.full(state.grid.interior_shape("u"), 0.2, dtype=np.float32),
    )
    state.set_interior(
        "v", np.full(state.grid.interior_shape("v"), 0.4, dtype=np.float32),
    )
    state.set_interior(
        "w", np.full(state.grid.interior_shape("w"), 0.5, dtype=np.float32),
    )

    limit = MomentumAdvector(cfg).transport_time_step(state, cfl=0.5)

    np.testing.assert_allclose(limit, 0.5 / 6.0, rtol=1.0e-7)


def test_residue_face_takes_the_extension_of_the_resolved_water():
    # A residue cell above a translating slab holds a 10 m/s face from an
    # earlier recovery. It owns no momentum: after one transport step the face
    # carries the extension of the slab (1 m/s) and the difference is reported.
    g = Grid3DConfig(nx=8, ny=4, nz=8, Lx=.08, Ly=.04, Lz=.08)
    cfg = Solver3DConfig(grid=g, device=DEVICE, pressure_discretization="liquid_mimetic",
                         liquid_resolved_fraction=1e-2,
                         bc=BC3DConfig(xlo=BC.PERIODIC, xhi=BC.PERIODIC,
                                       ylo=BC.PERIODIC, yhi=BC.PERIODIC))
    state = State3D(cfg)
    alpha = np.zeros(g.shape, np.float32)
    alpha[:, :, :3] = 1.0
    alpha[4, 2, 3] = 1e-5
    state.set_interior("alpha", alpha)
    state.fill_("u", 1.0)
    u = state.get_interior("u")
    u[:, :, 3:] = 0.0
    u[4, 2, 3] = 10.0            # the residue cell's low x face
    u[5, 2, 3] = 10.0            # and its high x face
    state.set_interior("u", u)
    advector = MomentumAdvector(cfg)
    advector.vof = UnsplitVOFAdvector(cfg)
    before = measure_vof(state)
    result = advector.advect(state, .2*g.dx/1.0)
    after_u = state.get_interior("u")
    assert abs(after_u[4, 2, 3]) <= 1.0+2e-6 and abs(after_u[5, 2, 3]) <= 1.0+2e-6
    np.testing.assert_allclose(after_u[:, :, :3], 1.0, atol=2e-6)
    assert result.discarded_momentum > 0.9*2*(0.5e-5)*9.0*cfg.physics.rho*g.dx*g.dy*g.dz
    after = measure_vof(state)
    assert abs(after.volume-before.volume) < 2e-6*before.volume
    assert 0 <= after.alpha_min <= after.alpha_max <= 1


def test_periodic_translating_slab_keeps_uniform_velocity_and_momentum():
    n = 16
    periodic = BC3DConfig(
        xlo=BC.PERIODIC, xhi=BC.PERIODIC,
        ylo=BC.PERIODIC, yhi=BC.PERIODIC,
        zlo=BC.PERIODIC, zhi=BC.PERIODIC,
    )
    cfg = Solver3DConfig(
        grid=Grid3DConfig(nx=n, ny=8, nz=8, Lx=1.0, Ly=0.5, Lz=0.5),
        bc=periodic,
        device=DEVICE,
    )
    state = State3D(cfg)
    initialize_box(state, (0.2, 0.0, 0.0), (0.6, 0.5, 0.5),
                   samples_per_axis=8)
    initial_mass = compute_face_liquid_mass(state, "u").numpy()[
        state.grid.interior_slices("u")
    ]
    speed = 0.18
    seeded = np.where(initial_mass > 0.0, speed, 0.0).astype(np.float32)
    state.set_interior("u", seeded)
    air_before = seeded[initial_mass == 0.0].copy()

    result = MomentumAdvector(cfg).advect(
        state, 0.3 * cfg.grid.dx / speed,
    )
    final_mass = compute_face_liquid_mass(state, "u").numpy()[
        state.grid.interior_slices("u")
    ]
    velocity = state.get_interior("u")
    water = final_mass > 1.0e-5
    np.testing.assert_allclose(velocity[water], speed,
                               rtol=0.0, atol=2.0e-5)
    np.testing.assert_array_equal(velocity[final_mass == 0.0], air_before)
    assert np.max(np.abs(velocity[water])) < 1.001 * speed

    scale = max(abs(result.momentum_before[0]), 1.0)
    assert abs(result.momentum_drift[0]) / scale < 2.0e-6
    assert abs(result.momentum_drift[1]) < 1.0e-10
    assert abs(result.momentum_drift[2]) < 1.0e-10
    assert result.discarded_momentum < 1.0e-8


def test_alongshore_transport_of_a_varying_profile_conserves_momentum():
    # Water filling a periodic box translates along y while u varies along y.
    # Both sides of every dual boundary must reconstruct the same flux velocity,
    # including stencils reaching two faces past the periodic seam, so the
    # flux-form update telescopes to zero momentum drift.
    n = 8
    periodic = BC3DConfig(
        xlo=BC.PERIODIC, xhi=BC.PERIODIC,
        ylo=BC.PERIODIC, yhi=BC.PERIODIC,
        zlo=BC.PERIODIC, zhi=BC.PERIODIC,
    )
    cfg = Solver3DConfig(
        grid=Grid3DConfig(nx=n, ny=n, nz=n, Lx=1.0, Ly=1.0, Lz=1.0),
        bc=periodic,
        device=DEVICE,
    )
    state = State3D(cfg)
    state.fill_("alpha", 1.0)
    profile = 0.2 + 0.1 * np.sin(2.0 * np.pi * np.arange(n) / n)
    state.set_interior("u", np.broadcast_to(
        profile[None, :, None].astype(np.float32),
        state.grid.interior_shape("u"),
    ).copy())
    state.fill_("v", 0.8)

    result = MomentumAdvector(cfg).advect(state, 0.045 * cfg.grid.dz / 0.2)

    assert abs(result.momentum_drift[0]) < 2.0e-6 * abs(result.momentum_before[0])
    assert result.discarded_momentum == 0.0
    velocity = state.get_interior("u")
    assert profile.min() - 1.0e-6 <= velocity.min()
    assert velocity.max() <= profile.max() + 1.0e-6


@pytest.mark.parametrize("liquid_path", [False, True])
def test_transport_step_and_fluxes_follow_liquid_faces_not_extended_air(liquid_path):
    # A sheared slab fills the lower half of a periodic box with u growing
    # upward. The four-layer extension extrapolates that gradient into the air
    # above to more than twice the wet maximum. Dry donors never move liquid,
    # so neither the transport step limit nor the flux kernel's Courant bound
    # may follow those extended values.
    n = 8
    periodic = BC3DConfig(
        xlo=BC.PERIODIC, xhi=BC.PERIODIC,
        ylo=BC.PERIODIC, yhi=BC.PERIODIC,
        zlo=BC.PERIODIC, zhi=BC.PERIODIC,
    )
    cfg = Solver3DConfig(
        grid=Grid3DConfig(nx=n, ny=n, nz=n, Lx=1.0, Ly=1.0, Lz=1.0),
        bc=periodic,
        device=DEVICE,
    )
    if liquid_path:
        cfg = replace(cfg, pressure_discretization="liquid_mimetic")
    state = State3D(cfg)
    alpha = np.zeros(cfg.grid.shape, np.float32)
    alpha[:, :, :4] = 1.0
    state.set_interior("alpha", alpha)
    u = np.zeros(state.grid.interior_shape("u"), np.float32)
    for k, speed in enumerate((0.05, 0.1, 0.2, 0.5)):
        u[:, :, k] = speed
    state.set_interior("u", u)
    advector = MomentumAdvector(cfg)
    if liquid_path:
        advector.vof = UnsplitVOFAdvector(cfg)

    limit = advector.transport_time_step(state, 0.5)
    np.testing.assert_allclose(limit, 0.5 * cfg.grid.dx / 0.5, rtol=1.0e-6)

    result = advector.advect(state, limit)

    np.testing.assert_allclose(state.get_interior("alpha"), alpha, atol=2.0e-7)
    np.testing.assert_allclose(state.get_interior("u"), u, atol=2.0e-6)
    assert result.vof.max_courant <= 0.5 * (1.0 + 1.0e-6)
    assert result.discarded_momentum == 0.0
