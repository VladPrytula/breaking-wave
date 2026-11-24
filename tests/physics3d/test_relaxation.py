import numpy as np
import pytest

from shoreloop.config import (BC, BC3DConfig, Grid3DConfig, Solver3DConfig)
from shoreloop.device import DEVICE
from shoreloop.physics3d.vof import measure_vof
from shoreloop.physics3d import (LinearWaveInlet, PeriodicWaveComponent,
                                 Solver3D, WaveRelaxationZone, initialize_box,
                                 relaxation_weight)


def _geometric_source(depth, amplitude=0.0, phase=0.0):
    grid = Grid3DConfig(nx=8, ny=4, nz=24, Lx=.2, Ly=.4, Lz=.48)
    cfg = Solver3DConfig(grid=grid, device=DEVICE,
                         pressure_discretization="liquid_mimetic")
    solver = Solver3D(cfg)
    component = PeriodicWaveComponent(period=1.2, depth=depth,
        alongshore_length=grid.Ly, amplitude=amplitude, phase_offset=phase)
    solver.set_wave_relaxation(WaveRelaxationZone(cfg,
        LinearWaveInlet((component,)), width=.175, timescale=.048))
    initialize_box(solver.state, (0, 0, 0), (grid.Lx, grid.Ly, depth))
    return solver


@pytest.mark.parametrize("phase", [0.0, np.pi])
def test_finite_rate_source_keeps_rising_and_falling_water_on_one_interface(phase):
    solver = _geometric_source(.22, .08, phase)
    g = solver.cfg.grid
    zone = solver.wave_relaxation
    x = (np.arange(g.nx)+.5)*g.dx
    y = (np.arange(g.ny)+.5)*g.dy
    weight = 1-np.exp(-.005/zone.timescale*np.asarray([
        relaxation_weight(value, 0, zone.width) for value in x]))
    for step in range(1, 4):
        before = solver.state.get_interior("alpha").astype(float)
        height = before.sum(axis=2)*g.dz
        target = np.asarray([[.22+zone.inlet.components[0].elevation(xi, yj, step*.005)
                              for yj in y] for xi in x])
        expected_height = height+weight[:, None]*(target-height)
        source = zone.apply(solver.state, step*.005, .005)
        after = solver.state.get_interior("alpha")
        expected = np.clip((expected_height[:, :, None]-np.arange(g.nz)*g.dz)/g.dz, 0, 1)
        # The target spans several cells; fraction blending formerly created
        # several independent PLIC layers even at the very first source call.
        np.testing.assert_allclose(after, expected, atol=3e-6, rtol=0)
        assert np.all((after >= 0) & (after <= 1))
        np.testing.assert_allclose(after.sum(axis=2, dtype=float)*g.dz,
                                   expected_height, atol=6e-8, rtol=0)
        assert source.volume_source == pytest.approx(
            np.sum(after.astype(float)-before)*g.dx*g.dy*g.dz, abs=2e-12)


@pytest.mark.parametrize("target_height", [.12, .22])
def test_source_moves_folded_liquid_intervals_at_the_declared_rate(target_height):
    solver = _geometric_source(target_height)
    g = solver.cfg.grid
    # Two exactly represented liquid intervals, separated by a 10 cm gap.
    # Their volume is 16 cm per unit area. The source must keep the gap at
    # finite weight, with the same total volume target as the wave inlet.
    alpha = np.zeros(g.shape, np.float32)
    alpha[:, :, :5] = 1
    alpha[:, :, 10:13] = 1
    solver.state.set_interior("alpha", alpha)
    dt = .012
    zone = solver.wave_relaxation
    x = (np.arange(g.nx)+.5)*g.dx
    weight = 1-np.exp(-dt/zone.timescale*np.asarray([
        relaxation_weight(value, 0, zone.width) for value in x]))
    scale = 1+weight*(target_height/.16-1)
    lower_top = scale*.10
    upper_bottom = (1-weight)*.20+weight*target_height*.10/.16
    upper_top = (1-weight)*.26+weight*target_height
    np.testing.assert_allclose(upper_bottom-lower_top, .10*(1-weight), atol=1e-16)
    bottoms = np.arange(g.nz)*g.dz
    tops = bottoms+g.dz
    expected = (np.maximum(0, np.minimum(tops, lower_top[:, None])-bottoms)
        + np.maximum(0, np.minimum(tops, upper_top[:, None])
                       - np.maximum(bottoms, upper_bottom[:, None])))/g.dz
    zone.apply(solver.state, 0, dt)
    after = solver.state.get_interior("alpha")
    np.testing.assert_allclose(after, np.broadcast_to(expected[:, None, :], g.shape), atol=3e-6, rtol=0)
    assert np.all((after >= 0) & (after <= 1))


def test_source_preserves_exact_grid_face_rest_and_introduces_a_dry_column_at_the_bed():
    solver = _geometric_source(.22)
    zone = solver.wave_relaxation
    g = solver.cfg.grid
    before = solver.state.get_interior("alpha").copy()
    source = zone.apply(solver.state, 0, .005)
    np.testing.assert_array_equal(solver.state.get_interior("alpha"), before)
    assert source.volume_source == 0
    solver.state.set_interior("alpha", np.zeros(g.shape, np.float32))
    source = zone.apply(solver.state, 0, .005)
    after = solver.state.get_interior("alpha")
    x = (np.arange(g.nx)+.5)*g.dx
    weight = 1-np.exp(-.005/zone.timescale*np.asarray([
        relaxation_weight(value, 0, zone.width) for value in x]))
    expected = np.clip((weight[:, None]*.22-np.arange(g.nz)*g.dz)/g.dz, 0, 1)
    np.testing.assert_allclose(after, np.broadcast_to(expected[:, None, :], g.shape), atol=3e-6, rtol=0)
    assert source.volume_source == pytest.approx(np.sum(after.astype(float))*g.dx*g.dy*g.dz, abs=2e-12)


def _wave_solver(*, relaxation_width: float = 0.3) -> Solver3D:
    grid = Grid3DConfig(nx=24, ny=4, nz=10, Lx=1.2, Ly=0.2, Lz=0.5)
    cfg = Solver3DConfig(
        grid=grid,
        bc=BC3DConfig(
            xlo=BC.WALL, xhi=BC.WALL,
            ylo=BC.PERIODIC, yhi=BC.PERIODIC,
            zlo=BC.WALL, zhi=BC.WALL,
        ),
        device=DEVICE,
    )
    solver = Solver3D(cfg)
    depth = 0.3
    initialize_box(solver.state, (0.0, 0.0, 0.0), (grid.Lx, grid.Ly, depth))
    component = PeriodicWaveComponent(
        period=1.0,
        depth=depth,
        alongshore_length=grid.Ly,
        amplitude=0.015,
    )
    solver.set_wave_relaxation(
        WaveRelaxationZone(
            cfg,
            LinearWaveInlet((component,)),
            width=relaxation_width,
        ),
    )
    return solver


def test_relaxation_weight_is_smooth_monotone_and_finite_width():
    x = np.linspace(-0.1, 0.5, 61)
    weights = np.array([relaxation_weight(value, 0.0, 0.4) for value in x])
    assert weights[0] == 1.0
    assert weights[-1] == 0.0
    assert np.all(np.diff(weights) <= 0.0)
    assert relaxation_weight(0.2, 0.0, 0.4) == pytest.approx(0.5)


def test_relaxation_generates_a_small_wave_with_explicit_source_accounting():
    solver = _wave_solver()
    alpha_before = solver.state.get_interior("alpha").copy()

    for _ in range(4):
        result = solver.step(0.005)
        assert result.converged

    source = solver.last_relaxation
    assert source is not None
    assert source.volume_after - source.volume_before == pytest.approx(
        source.volume_source,
    )
    assert all(np.isfinite(source.momentum_source))
    assert source.momentum_source[0] > 0.0
    assert abs(source.momentum_source[1]) < 1.0e-8
    alpha_after = solver.state.get_interior("alpha")
    assert not np.array_equal(alpha_after[:6], alpha_before[:6])
    generated_change = np.max(np.abs(alpha_after[:6] - alpha_before[:6]))
    propagated_change = np.max(np.abs(alpha_after[10:] - alpha_before[10:]))
    assert propagated_change < 0.2 * generated_change
    grid = solver.cfg.grid
    generated = alpha_after[0].mean(axis=0).sum() * grid.dz - 0.3
    component = solver.wave_relaxation.inlet.components[0]
    expected = component.elevation(grid.x0 + 0.5 * grid.dx,
                                   grid.y0 + 0.5 * grid.dy, solver.time)
    assert 0.0 < generated < expected


def test_relaxation_preserves_column_source_with_rate_limited_sharpening():
    solver = _wave_solver()
    grid = solver.cfg.grid
    alpha = solver.state.get_interior("alpha")
    alpha[:6, :, 5] = 0.8
    alpha[:6, :, 6] = 0.2
    solver.state.set_interior("alpha", alpha)
    before_height = np.sum(alpha, axis=2, dtype=np.float64) * grid.dz
    dt = 0.005

    source = solver.wave_relaxation.apply(solver.state, 0.0, dt)

    after = solver.state.get_interior("alpha")
    after_height = np.sum(after, axis=2, dtype=np.float64) * grid.dz
    x = grid.x0 + (np.arange(grid.nx) + 0.5) * grid.dx
    y = grid.y0 + (np.arange(grid.ny) + 0.5) * grid.dy
    component = solver.wave_relaxation.inlet.components[0]
    target_height = np.asarray([
        [
            component.depth + component.elevation(xi, yj, 0.0)
            for yj in y
        ]
        for xi in x
    ])
    spatial_weight = np.asarray([
        relaxation_weight(xi, grid.x0, solver.wave_relaxation.width)
        for xi in x
    ])[:, None]
    source_weight = 1.0 - np.exp(
        -dt / solver.wave_relaxation.timescale * spatial_weight
    )
    expected_height = before_height + source_weight * (
        target_height - before_height
    )

    np.testing.assert_allclose(
        after_height, expected_height, rtol=0.0, atol=5.0e-8,
    )
    cell_bottom = np.arange(grid.nz) * grid.dz
    sharp = np.clip(
        (expected_height[..., None] - cell_bottom) / grid.dz,
        0.0,
        1.0,
    )
    before_sharp = np.clip(
        (before_height[..., None] - cell_bottom) / grid.dz,
        0.0,
        1.0,
    )
    before_shape_error = np.sum(np.abs(alpha - before_sharp), axis=2)
    after_shape_error = np.sum(np.abs(after - sharp), axis=2)
    assert np.all(after_shape_error[:6] < before_shape_error[:6])
    assert source.volume_after - source.volume_before == pytest.approx(
        source.volume_source,
    )


def test_relaxation_topology_change_vanishes_with_source_weight():
    solver = _wave_solver()
    grid = solver.cfg.grid
    alpha = solver.state.get_interior("alpha")
    column = 5
    alpha[column] = 0.0
    alpha[column, :, :3] = 1.0
    alpha[column, :, 4] = 1.0
    solver.state.set_interior("alpha", alpha)
    before = alpha[column].copy()
    dt = 1.0e-6

    solver.wave_relaxation.apply(solver.state, 0.0, dt)

    after = solver.state.get_interior("alpha")[column]
    x = grid.x0 + (column + 0.5) * grid.dx
    spatial_weight = relaxation_weight(
        x, grid.x0, solver.wave_relaxation.width,
    )
    source_weight = 1.0 - np.exp(
        -dt / solver.wave_relaxation.timescale * spatial_weight
    )
    assert source_weight < 1.0e-6
    assert np.max(np.abs(after - before)) <= 2.0 * source_weight + 1.0e-6


def test_relaxation_does_not_regraph_when_edge_rate_rounds_to_zero():
    solver = _wave_solver(relaxation_width=0.27501)
    grid = solver.cfg.grid
    alpha = solver.state.get_interior("alpha")
    column = 5
    alpha[column] = 0.0
    alpha[column, :, :3] = 1.0
    alpha[column, :, 4] = 1.0
    solver.state.set_interior("alpha", alpha)
    before = alpha[column].copy()
    dt = 0.005

    solver.wave_relaxation.apply(solver.state, 0.0, dt)

    x = grid.x0 + (column + 0.5) * grid.dx
    spatial_weight = relaxation_weight(
        x, grid.x0, solver.wave_relaxation.width,
    )
    exact_weight = 1.0 - np.exp(
        -dt / solver.wave_relaxation.timescale * spatial_weight,
    )
    assert 0.0 < exact_weight < np.finfo(np.float32).eps
    np.testing.assert_array_equal(
        solver.state.get_interior("alpha")[column],
        before,
    )


def test_imposed_initial_state_carries_the_incident_field_with_the_zone_taper():
    # Imposing once sets each column and face to the incident field weighted
    # by the zone's own spatial taper, leaves water beyond the zone at rest,
    # keeps every column one bottom-connected interval, and reports the
    # volume it introduced as an initial-state change.
    solver = _geometric_source(.22, .08, 0.0)
    g = solver.cfg.grid
    zone = solver.wave_relaxation
    x = (np.arange(g.nx)+.5)*g.dx
    y = (np.arange(g.ny)+.5)*g.dy
    taper = np.asarray([relaxation_weight(value, 0, zone.width) for value in x])
    before_alpha = solver.state.get_interior("alpha")
    before_height = before_alpha.astype(float).sum(axis=2)*g.dz

    result = zone.impose(solver.state, 0.0)

    alpha = solver.state.get_interior("alpha")
    height = alpha.astype(float).sum(axis=2)*g.dz
    target = np.asarray([[.22+zone.inlet.elevation(xi, yj, 0.0) for yj in y] for xi in x])
    np.testing.assert_allclose(height, before_height+taper[:, None]*(target-before_height),
                               atol=2e-6)
    assert np.max(np.abs(target-.22)) > .02
    assert 0 <= alpha.min() <= alpha.max() <= 1
    assert np.all(np.diff(alpha.astype(float), axis=2) <= 1e-6)
    assert taper[-1] == 0 and np.array_equal(alpha[-1], before_alpha[-1])
    u = solver.state.get_interior("u")
    i, j, k = 2, 1, 5
    face_x = i*g.dx
    depth_z = (k+.5)*g.dz-.22
    expected = relaxation_weight(face_x, 0, zone.width)*zone.inlet.velocity(face_x, y[j], depth_z, 0.0)[0]
    assert abs(expected) > 1e-3
    np.testing.assert_allclose(u[i, j, k], expected, atol=2e-5)
    assert np.all(u[-1] == 0)
    np.testing.assert_allclose(result.volume_source, (height-before_height).sum()*g.dx*g.dy,
                               rtol=1e-4)
    # Velocity support follows the imposed (tapered) surface, not the full
    # analytic crest: faces between the two carry no momentum.
    i = 3
    face_x = i*g.dx
    s_face = relaxation_weight(face_x, 0, zone.width)
    eta = zone.inlet.elevation(face_x, y[1], 0.0)
    assert eta > 0.05 and 0.3 < s_face < 0.8
    tapered = s_face*eta
    k_between = int(np.floor((.22+.5*(tapered+eta))/g.dz))
    k_inside = int(np.floor((.22+.5*tapered)/g.dz))
    assert (k_between+.5)*g.dz-.22 > tapered and (k_inside+.5)*g.dz-.22 < tapered
    assert u[i, 1, k_between] == 0
    assert abs(u[i, 1, k_inside]) > 1e-3

    # One projection without body force makes the imposed liquid field
    # divergence-free without moving the water.
    volume_before = measure_vof(solver.state).volume
    projected = solver.projection.project(solver.state, .005, operator=solver.liquid_operator(),
                                          acceleration=(0.0, 0.0, 0.0))
    assert projected.converged
    assert projected.rms_divergence_after < 1e-5
    assert projected.rms_divergence_after < .01*projected.rms_divergence_before
    assert measure_vof(solver.state).volume == volume_before


def test_generation_ramp_leaves_the_offshore_wall_at_rest_and_the_core_untouched():
    # With a ramp the source target rises from still water at rest at the
    # wall to the full incident field past the ramp, so a wall boundary is
    # never asked to carry the incident normal velocity; without a ramp the
    # full field reaches the wall.
    grid = Grid3DConfig(nx=8, ny=4, nz=24, Lx=.2, Ly=.4, Lz=.48)
    cfg = Solver3DConfig(grid=grid, device=DEVICE, pressure_discretization="liquid_mimetic")
    component = PeriodicWaveComponent(period=1.2, depth=.22, alongshore_length=grid.Ly,
                                      amplitude=.08, phase_offset=0.0)
    inlet = LinearWaveInlet((component,))
    heights, speeds = {}, {}
    for ramp in (0.0, .1):
        solver = Solver3D(cfg)
        initialize_box(solver.state, (0, 0, 0), (grid.Lx, grid.Ly, .22))
        zone = WaveRelaxationZone(cfg, inlet, width=.2, timescale=.048, generation_ramp=ramp)
        zone.impose(solver.state, 0.0)
        heights[ramp] = solver.state.get_interior("alpha").astype(float).sum(axis=2)*grid.dz
        speeds[ramp] = np.abs(solver.state.get_interior("u")).max(axis=(1, 2))
    x = (np.arange(grid.nx)+.5)*grid.dx
    faces = np.arange(grid.nx+1)*grid.dx
    # The wall face is at rest and the first column carries a few percent of
    # the incident amplitude only with the ramp (smoothstep: zero slope at
    # the wall, so the first cell centre sees about 3 (x/ramp)^2 of it).
    assert heights[0.0][0].max() - .22 > .02
    assert heights[.1][0].max() - .22 < .1*(heights[0.0][0].max() - .22)
    assert speeds[.1][0] == 0 and speeds[.1][1] < .2*speeds[0.0][1]
    # Beyond the ramp both targets coincide.
    core = x >= .1
    np.testing.assert_allclose(heights[.1][core], heights[0.0][core], atol=2e-6)
    np.testing.assert_allclose(speeds[.1][faces >= .1], speeds[0.0][faces >= .1], atol=2e-6)
    with pytest.raises(ValueError):
        WaveRelaxationZone(cfg, inlet, width=.2, timescale=.048, generation_ramp=.3)
