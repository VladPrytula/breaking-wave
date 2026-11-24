"""Conservation and capacity gates for liquid-projection-compatible transport."""

from dataclasses import replace

import numpy as np
import pytest

from shoreloop.config import BC, BC3DConfig, Grid3DConfig, Solver3DConfig
from shoreloop.device import DEVICE
from shoreloop.physics3d.cut_cells import initialize_scene_cut_cells
from shoreloop.physics3d.liquid_pressure import LiquidPressureOperator
from shoreloop.physics3d.momentum import MomentumAdvector, compute_face_liquid_mass
from shoreloop.physics3d.state import State3D
from shoreloop.physics3d.unsplit_advection import UnsplitVOFAdvector
from shoreloop.physics3d.vof import initialize_joint_plane, initialize_sphere, measure_vof
from shoreloop.scene.geometry3d import AnalyticBeach, SceneGeometry


def _periodic(n=8, flux_scheme="axis_slabs"):
    return Solver3DConfig(
        grid=Grid3DConfig(nx=n, ny=n, nz=n, Lx=1, Ly=1, Lz=1), device=DEVICE,
        vof_flux_reconstruction=flux_scheme,
        pressure_discretization=("liquid_mimetic" if flux_scheme == "swept_prism" else "ghost_fluid"),
        bc=BC3DConfig(xlo=BC.PERIODIC, xhi=BC.PERIODIC,
                      ylo=BC.PERIODIC, yhi=BC.PERIODIC,
                      zlo=BC.PERIODIC, zhi=BC.PERIODIC),
    )


@pytest.mark.parametrize("flux_scheme", ["axis_slabs", "swept_prism"])
def test_overlapping_sweeps_cannot_overdraw_thin_water(flux_scheme):
    state = State3D(_periodic(4, flux_scheme))
    alpha = np.zeros(state.cfg.grid.shape, np.float32)
    alpha[1, 1, 1] = .001
    state.set_interior("alpha", alpha)
    for q in ("u", "v", "w"):
        state.fill_(q, -.2)
    before = measure_vof(state)
    result = UnsplitVOFAdvector(state.cfg).advect(state, .2*state.cfg.grid.dx/.2)
    after = measure_vof(state)
    assert 0 <= after.alpha_min <= after.alpha_max <= 1
    if flux_scheme == "axis_slabs":
        assert result.limited_flux_volume > 0
        assert result.limiter_iterations > 0
    assert abs(after.volume-before.volume) < 2e-7*before.volume
    assert result.clipped_volume < 2e-7*before.volume


@pytest.mark.parametrize("fraction", [.37, 1.0])
@pytest.mark.parametrize("flux_scheme", ["axis_slabs", "swept_prism"])
def test_uniform_translation_preserves_fraction_without_flux_limiting(fraction, flux_scheme):
    state = State3D(_periodic(flux_scheme=flux_scheme))
    state.fill_("alpha", fraction)
    for q in ("u", "v", "w"):
        state.fill_(q, .2)
    result = UnsplitVOFAdvector(state.cfg).advect(state, .1*state.cfg.grid.dx/.2)
    np.testing.assert_allclose(state.get_interior("alpha"), fraction, atol=2e-7)
    assert result.limited_flux_volume == 0
    assert abs(result.volume_drift) < 2e-7*fraction


@pytest.mark.parametrize("fraction", [2e-6, 5e-7, 1e-8])
@pytest.mark.parametrize("flux_scheme", ["axis_slabs", "swept_prism"])
@pytest.mark.parametrize("speed_bound", ["global", "local"])
def test_retained_thin_sheet_transports_uniform_velocity_and_momentum(fraction, flux_scheme, speed_bound):
    # Sheets below the resolved fraction own no velocity; this control tests
    # thin-water transport, so every sheet here is resolved.
    cfg = replace(_periodic(flux_scheme=flux_scheme), pressure_discretization="liquid_mimetic",
                  momentum_speed_bound=speed_bound, liquid_resolved_fraction=1e-9)
    state = State3D(cfg)
    alpha = np.zeros(cfg.grid.shape, np.float32)
    alpha[2:5, :, 3] = fraction
    state.set_interior("alpha", alpha)
    slices = state.grid.interior_slices("u")
    mass = compute_face_liquid_mass(state, "u").numpy()[slices]
    speed = .2
    state.set_interior("u", np.where(mass > 0, speed, 0).astype(np.float32))
    advector = MomentumAdvector(cfg)
    advector.vof = UnsplitVOFAdvector(cfg)
    before = measure_vof(state)

    dt = .1*cfg.grid.dx/speed
    np.testing.assert_allclose(advector.transport_time_step(state, .5),
                               .5*cfg.grid.dx/speed, rtol=2e-6)
    result = advector.advect(state, dt)

    mass = compute_face_liquid_mass(state, "u").numpy()[slices]
    np.testing.assert_allclose(state.get_interior("u")[mass > 0], speed,
                               rtol=2e-6, atol=0)
    assert np.max(np.abs(state.get_interior("alpha")-alpha)) > .01*fraction
    assert abs(result.momentum_drift[0]) < 2e-6*abs(result.momentum_before[0])
    assert result.discarded_momentum < 2e-6*abs(result.momentum_before[0])
    after = measure_vof(state)
    assert abs(after.volume-before.volume) < 2e-6*before.volume
    assert 0 <= after.alpha_min <= after.alpha_max <= 1


@pytest.mark.parametrize("flux_scheme", ["axis_slabs", "swept_prism"])
def test_unsplit_sphere_translation_shares_conservative_momentum_fluxes(flux_scheme):
    state = State3D(_periodic(12, flux_scheme))
    initialize_sphere(state, (.35, .5, .5), .19)
    state.fill_("u", .2)
    advector = MomentumAdvector(state.cfg)
    advector.vof = UnsplitVOFAdvector(state.cfg)
    before = measure_vof(state)
    for _ in range(3):
        result = advector.advect(state, .2*state.cfg.grid.dx/.2)
        assert max(abs(q) for q in result.momentum_drift) < 2e-5
    after = measure_vof(state)
    assert abs(after.volume-before.volume) < 5e-7*before.volume
    assert 0 <= after.alpha_min <= after.alpha_max <= 1
    wet = state.get_interior("alpha") > .01
    u = state.get_interior("u")[:-1]
    np.testing.assert_allclose(u[wet], .2, atol=2e-6)


@pytest.mark.parametrize("flux_scheme", ["axis_slabs", "swept_prism"])
def test_liquid_projected_cut_water_advects_without_open_divergence_source(flux_scheme):
    g = Grid3DConfig(nx=12, ny=4, nz=8, Lx=.15, Ly=.05, Lz=.08)
    state = State3D(Solver3DConfig(grid=g, device=DEVICE,
        pressure_discretization=("liquid_mimetic" if flux_scheme == "swept_prism" else "ghost_fluid"),
        vof_flux_reconstruction=flux_scheme))
    initialize_scene_cut_cells(state, SceneGeometry(AnalyticBeach(-.1, .1, -.003)))
    initialize_joint_plane(state, (-.1, 0, 1), .0103)
    operator = LiquidPressureOperator(state, velocity_space="mac")
    projected = operator.project(.001, acceleration=(0, 0, -9.81))
    operator.write_mac_state(state, projected)
    before = measure_vof(state)
    result = UnsplitVOFAdvector(state.cfg).advect(state, .001)
    after = measure_vof(state)
    assert abs(after.volume-before.volume) < 2e-7*before.volume
    assert 0 <= after.alpha_min <= after.alpha_max <= 1
    assert result.clipped_volume < 2e-7*before.volume


def _surface_film_state(fraction, u_bulk, u_film, u_air, flux_scheme, speed_bound):
    """Bulk water below a partially filled surface cell, all draining downward."""
    cfg = replace(_periodic(flux_scheme=flux_scheme), pressure_discretization="liquid_mimetic",
                  momentum_speed_bound=speed_bound)
    state = State3D(cfg)
    alpha = np.zeros(cfg.grid.shape, np.float32)
    alpha[:, :, :3] = 1.0
    alpha[:, :, 3] = fraction
    state.set_interior("alpha", alpha)
    u = np.full(state.grid.interior_shape("u"), u_bulk, np.float32)
    u[:, :, 3] = u_film
    u[:, :, 4:] = u_air
    state.set_interior("u", u)
    state.fill_("w", -.2)
    return cfg, state


def _bounded_transport_checks(state, result, before, low, high, discarded=1e-6):
    slices = state.grid.interior_slices("u")
    mass = compute_face_liquid_mass(state, "u").numpy()[slices]
    u = state.get_interior("u")[mass > 0]
    assert low-1e-6 <= u.min() and u.max() <= high+1e-6
    scale = abs(result.momentum_before[0])
    assert result.discarded_momentum < discarded*scale
    assert abs(result.momentum_drift[0]) < 2e-6*scale+result.discarded_momentum
    after = measure_vof(state)
    assert abs(after.volume-before.volume) < 2e-6*before.volume
    assert 0 <= after.alpha_min <= after.alpha_max <= 1


@pytest.mark.parametrize("flux_scheme", ["axis_slabs", "swept_prism"])
@pytest.mark.parametrize("speed_bound", ["global", "local"])
def test_draining_surface_film_transport_creates_no_velocity_extremum(flux_scheme, speed_bound):
    # Ninety percent of the film's dual liquid leaves downward in one step while
    # its velocity differs from the water below and the stale air value above
    # continues the gradient. The retained momentum must stay within the wet
    # velocity range instead of being amplified by the outflow reconstruction.
    cfg, state = _surface_film_state(.05, u_bulk=.3, u_film=.1, u_air=-.1, flux_scheme=flux_scheme,
                                     speed_bound=speed_bound)
    advector = MomentumAdvector(cfg)
    advector.vof = UnsplitVOFAdvector(cfg)
    before = measure_vof(state)
    result = advector.advect(state, .045*cfg.grid.dz/.2)
    _bounded_transport_checks(state, result, before, .1, .3)


@pytest.mark.parametrize("flux_scheme", ["axis_slabs", "swept_prism"])
@pytest.mark.parametrize("speed_bound", ["global", "local"])
def test_outflow_beyond_face_liquid_carries_the_entering_velocity(flux_scheme, speed_bound):
    # Overlapping unsplit sweeps let a thin surface film send out more liquid
    # than it holds when liquid also enters alongshore. With an alongshore
    # velocity gradient the excess must carry the entering velocity rather
    # than extrapolate beyond the pre-step range.
    cfg, state = _surface_film_state(.05, u_bulk=0.0, u_film=0.0, u_air=0.0, flux_scheme=flux_scheme,
                                     speed_bound=speed_bound)
    ny = cfg.grid.ny
    profile = .2+.1*np.sin(2*np.pi*np.arange(ny)/ny)
    u = np.broadcast_to(profile[None, :, None].astype(np.float32),
                        state.grid.interior_shape("u")).copy()
    state.set_interior("u", u)
    state.fill_("v", .8)
    advector = MomentumAdvector(cfg)
    advector.vof = UnsplitVOFAdvector(cfg)
    before = measure_vof(state)
    result = advector.advect(state, .045*cfg.grid.dz/.2)
    assert result.vof.limited_flux_volume == 0
    # The overdrawn outflow mixes the raw entering mean with reconstructed
    # inflow values, amplified by 1/(remaining mass). The global bound keeps
    # that small local overshoot silently; the local bound removes and reports
    # it (2.4e-6 of the momentum here).
    _bounded_transport_checks(state, result, before, profile.min(), profile.max(),
                              discarded=1e-6 if speed_bound == "global" else 5e-6)


@pytest.mark.parametrize("speed_bound", ["global", "local"])
def test_local_bound_keeps_near_empty_faces_inside_what_they_exchanged(speed_bound):
    # A remote fast row raises the global cap to .6. The overdrawn film faces
    # then recover values above anything they exchanged with (at most .3);
    # only the local bound removes that overshoot and reports it.
    cfg, state = _surface_film_state(.05, 0.0, 0.0, 0.0, "axis_slabs", speed_bound)
    ny = cfg.grid.ny
    profile = .2+.1*np.sin(2*np.pi*np.arange(ny)/ny)
    u = np.broadcast_to(profile[None, :, None].astype(np.float32),
                        state.grid.interior_shape("u")).copy()
    u[:, 0, 0] = .6
    state.set_interior("u", u)
    state.fill_("v", 2.0)
    advector = MomentumAdvector(cfg)
    advector.vof = UnsplitVOFAdvector(cfg)
    result = advector.advect(state, .045*cfg.grid.dz/.2)
    mass = compute_face_liquid_mass(state, "u").numpy()[state.grid.interior_slices("u")]
    film = state.get_interior("u")[:, :, 3][mass[:, :, 3] > 0]
    if speed_bound == "local":
        assert film.max() <= profile.max()+1e-6
        assert result.discarded_momentum > 0
    else:
        assert film.max() > profile.max()+1e-3


@pytest.mark.parametrize("direction", [1, -1])
def test_film_translating_along_cut_beach_keeps_its_speed(direction):
    # Uniform flow parallel to a production-spacing cut beach. The dual mass
    # the VOF fluxes move is half of each cell's water; the aperture-weighted
    # mean fraction instead rescales recovered speeds on cut faces (-5.8% up
    # and -10.7% down the slope after 24 cells with the local bound).
    slope = .15
    g = Grid3DConfig(nx=40, ny=2, nz=10, Lx=.5, Ly=.025, Lz=.1)
    cfg = Solver3DConfig(grid=g, device=DEVICE, pressure_discretization="liquid_mimetic",
                         momentum_speed_bound="local", momentum_face_mass="flux_consistent",
                         bc=BC3DConfig(xlo=BC.WALL, xhi=BC.WALL, ylo=BC.PERIODIC, yhi=BC.PERIODIC,
                                       zlo=BC.WALL, zhi=BC.WALL))
    state = State3D(cfg)
    initialize_scene_cut_cells(state, SceneGeometry(AnalyticBeach(.05, slope, .002)))
    normal = np.array([-slope, 0, 1.0])/np.sqrt(1+slope*slope)
    initialize_joint_plane(state, tuple(normal), (.002-slope*.05)/np.sqrt(1+slope*slope)+.0015)
    alpha = state.get_interior("alpha")
    alpha[:14] = 0
    alpha[26:] = 0
    state.set_interior("alpha", alpha)
    speed = .5*direction
    state.fill_("u", speed)
    state.fill_("w", speed*slope)
    advector = MomentumAdvector(cfg)
    advector.vof = UnsplitVOFAdvector(cfg)
    before = measure_vof(state)
    for _ in range(20):
        advector.advect(state, .4*g.dx/.5)
    mass = compute_face_liquid_mass(state, "u", flux_consistent=True).numpy()[state.grid.interior_slices("u")]
    u = state.get_interior("u")
    wet = mass > 0
    assert abs(np.sum(mass[wet]*u[wet])/np.sum(mass[wet])/speed-1) < 1e-3
    assert np.all(np.abs(u[wet]) <= .5+1e-6)
    after = measure_vof(state)
    assert abs(after.volume-before.volume) < 2e-6*before.volume
    assert 0 <= after.alpha_min <= after.alpha_max <= 1
