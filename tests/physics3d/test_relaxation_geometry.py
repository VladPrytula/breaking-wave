"""Source footprint accuracy, local boundedness, and separated liquid."""
import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq

from shoreloop.config import Grid3DConfig, Solver3DConfig
from shoreloop.device import DEVICE, prepare_device
from shoreloop.physics3d import Solver3D, LinearWaveInlet, PeriodicWaveComponent, WaveRelaxationZone, relaxation_weight
from shoreloop.physics3d.vof import initialize_joint_plane
from shoreloop.physics3d.relaxation_geometry import plane_fraction, fit_offsets, displace
from shoreloop.physics3d.relaxation_rays import integrate_source_rays


def source_solver(target=.26, *, order=8):
    g = Grid3DConfig(nx=24, ny=4, nz=48, Lx=.3, Ly=4*6/224, Lz=.48)
    cfg = Solver3DConfig(grid=g, device=DEVICE, pressure_discretization="liquid_mimetic",
                         interface_reconstruction="height_function",
                         wave_source_reconstruction="plic_rays", wave_source_quadrature=order)
    solver = Solver3D(cfg)
    inlet = LinearWaveInlet((PeriodicWaveComponent(period=1.2, depth=target,
                             alongshore_length=g.Ly, amplitude=0),))
    zone = WaveRelaxationZone(cfg, inlet, width=.3, timescale=.048)
    solver.set_wave_relaxation(zone)
    return solver


def integrated_intervals(g, intervals):
    """Independent adaptive integral; these controls have monotone endpoints."""
    result = np.zeros((g.nx, g.nz))
    for i in range(g.nx):
        left, right = i*g.dx, (i+1)*g.dx
        for k in range(g.nz):
            cuts = []
            for interval in range(len(intervals(left))):
                for side in (0, 1):
                    for z in (k*g.dz, (k+1)*g.dz):
                        a, b = intervals(left)[interval][side], intervals(right)[interval][side]
                        if (a-z)*(b-z) < 0:
                            cuts.append(brentq(lambda x: intervals(x)[interval][side]-z,
                                               left, right, xtol=1e-15))
            def fraction(x):
                return sum(max(0, min(top, (k+1)*g.dz)-max(bottom, k*g.dz))
                           for bottom, top in intervals(x))/g.dz
            result[i, k] = quad(fraction, left, right, points=cuts,
                                epsabs=1e-13, epsrel=1e-12)[0]/g.dx
    return result


def test_source_preserves_inclined_footprints_and_the_declared_column_volume():
    solver = source_solver()
    g, zone = solver.cfg.grid, solver.wave_relaxation
    initialize_joint_plane(solver.state, (.65, 0, 1), .36)
    before = solver.state.get_interior("alpha").copy()
    physical_fields = {name: solver.state.field(name).numpy().copy()
                       for name in ("foam", "porous_storage", "solid_wetness")}
    source = zone.apply(solver.state, 0., .005)
    after = solver.state.get_interior("alpha")
    def intervals(x):
        retained = np.exp(-.005/zone.timescale*relaxation_weight(x, 0, zone.width))
        return [(0, retained*(.36-.65*x)+(1-retained)*.26)]
    expected = integrated_intervals(g, intervals)
    # Exclude columns near the walls without complete height-function support.
    # The old lower-interval source has a 6.78% cell error in this control.
    assert np.max(abs(after[5:-5, :, :]-expected[5:-5, None, :])) < 8e-4
    # The source is now the footprint integral of the analytic taper. Normal
    # errors at the walls also perturb its spatially weighted column volume.
    np.testing.assert_allclose(after.sum(axis=2, dtype=float)*g.dz,
        np.broadcast_to(expected.sum(axis=1)[:, None]*g.dz, (g.nx, g.ny)), atol=2e-7, rtol=0)
    assert after.min() >= 0 and after.max() <= 1
    assert source.volume_source == pytest.approx(
        (after.astype(float)-before).sum()*g.dx*g.dy*g.dz, abs=2e-12)
    for name, value in physical_fields.items():
        np.testing.assert_array_equal(solver.state.field(name).numpy(), value)
    assert zone.interface_representation == "geometric_plic_ray_displacement"


def test_repeated_source_tracks_the_continuous_taper_without_column_steps():
    solver = source_solver()
    g, zone = solver.cfg.grid, solver.wave_relaxation
    initialize_joint_plane(solver.state, (.65, 0, 1), .36)
    for step in range(20):
        zone.apply(solver.state, (step+1)*.005, .005)
    def intervals(x):
        retained = np.exp(-.1/zone.timescale*relaxation_weight(x, 0, zone.width))
        return [(0, retained*(.36-.65*x)+(1-retained)*.26)]
    expected = integrated_intervals(g, intervals)
    after = solver.state.get_interior("alpha")
    assert np.max(abs(after[5:-5]-expected[5:-5, None, :])) < 2e-3
    assert after.min() >= 0 and after.max() <= 1


def test_source_refinement_reduces_the_moved_plane_overlap_error():
    alpha = np.zeros(16)
    normals = np.tile((-.7, -.3, 1.), (16, 1))
    normals /= abs(normals).sum(axis=1)[:, None]
    d = (8.2-np.arange(16))/2
    for k in range(16):
        alpha[k] = plane_fraction(normals[k], d[k])
    errors = []
    for order in (4, 16):
        after, report = displace(alpha, normals, d, lambda xy: 7., .08, order=order)
        n = np.array((-.7*.92, -.3*.92, 1.))
        scale = abs(n).sum()
        expected = np.array([plane_fraction(n/scale, (.92*8.2+.08*7-k)/scale) for k in range(16)])
        errors.append(abs(after-expected).max())
        assert abs(report["column_volume_error"]) < 3e-14
        assert after.min() >= 0 and after.max() < 1+1e-13
    assert errors[1] < errors[0]/4


def test_opposing_sheet_faces_retain_the_gap_and_all_column_water():
    nz = 24
    n = np.array((-.4, -.2, 1.)); scale = abs(n).sum(); n /= scale
    def plane(offset):
        return np.array([plane_fraction(n, (offset-k)/scale) for k in range(nz)])
    low, top, bottom = plane(7.3), plane(16.3), plane(13.3)
    alpha = low+top-bottom
    normals = np.tile(n, (nz, 1))
    normals[(bottom > 0) & (bottom < 1)] *= -1
    offsets = fit_offsets(normals, alpha)
    after, report = displace(alpha, normals, offsets, lambda xy: 11., .08)
    assert np.all(after[9:12] == 0)
    assert after[:9].sum() > 0 and after[12:].sum() > 0
    assert abs(after.sum()-(.92*alpha.sum()+.08*11)) < 3e-13
    assert report["input_volume_error"] < 1e-13
    unchanged, _ = displace(alpha, normals, offsets, lambda xy: 11., 0.)
    np.testing.assert_array_equal(unchanged, alpha)


def test_vertical_footprint_keeps_wet_and_dry_ray_source_accounting():
    # Half the footprint is a full-height vertical sheet, half is dry.
    # The same source map lowers wet rays to 15.2 cells and fills dry rays
    # to .8 cells. A column-average interval would lose this distinction.
    alpha = np.full(16, .5)
    normals = np.tile((1., 0., 0.), (16, 1))
    after, report = displace(alpha, normals, np.full(16, .5),
                            lambda xy: 8., .1)
    expected = np.full(16, .5)
    expected[0], expected[-1] = .9, .1
    np.testing.assert_allclose(after, expected, atol=8e-15, rtol=0)
    assert abs(report["column_volume_error"]) < 8e-15


@pytest.mark.parametrize("initial_height", [0., 2.5])
def test_source_can_fill_cells_above_the_initial_liquid(initial_height):
    alpha = np.clip(initial_height-np.arange(24), 0, 1)
    normals = np.tile((0., 0., 1.), (24, 1))
    after, report = displace(alpha, normals, fit_offsets(normals, alpha),
                            lambda xy: 18., .5)
    expected_height = .5*initial_height+.5*18
    expected = np.clip(expected_height-np.arange(24), 0, 1)
    np.testing.assert_allclose(after, expected, atol=8e-15, rtol=0)
    assert abs(report["column_volume_error"]) < 3e-14


@pytest.mark.parametrize("target", [.12, .22])
def test_solver_source_retains_two_existing_liquid_intervals(target):
    solver = source_solver(target)
    g = solver.cfg.grid
    alpha = np.zeros(g.shape, np.float32)
    alpha[:, :, :10] = 1
    alpha[:, :, 20:26] = 1
    solver.state.set_interior("alpha", alpha)
    solver.wave_relaxation.apply(solver.state, 0., .012)
    after = solver.state.get_interior("alpha")
    def intervals(x):
        weight = -np.expm1(-.012/.048*relaxation_weight(x, 0., .3))
        scale = 1+weight*(target/.16-1)
        return [(0, scale*.1), ((1-weight)*.2+weight*target*.1/.16,
                               (1-weight)*.26+weight*target)]
    expected = integrated_intervals(g, intervals)
    np.testing.assert_allclose(after, np.broadcast_to(expected[:, None, :], g.shape),
                               atol=2e-4, rtol=0)


def test_grid_face_rest_and_vanishing_source_leave_state_unchanged():
    solver = source_solver(.22)
    initialize_joint_plane(solver.state, (0, 0, 1), .22)
    before = solver.state.get_interior("alpha").copy()
    solver.wave_relaxation.apply(solver.state, 0., .005)
    np.testing.assert_array_equal(solver.state.get_interior("alpha"), before)
    initialize_joint_plane(solver.state, (.65, 0, 1), .36)
    before = solver.state.get_interior("alpha").copy()
    solver.wave_relaxation.apply(solver.state, 0., 1e-12)
    # With FP64 source integration, a nonzero dt has a nonzero displacement;
    # the change is bounded by nz*dt/timescale, rather than rounded to zero.
    assert np.max(abs(solver.state.get_interior("alpha")-before)) < 1e-9


def test_analytic_source_target_retains_wave_slope_without_detached_layers():
    solver = source_solver(.22)
    initialize_joint_plane(solver.state, (0, 0, 1), .22)
    inlet = LinearWaveInlet((PeriodicWaveComponent(period=1.2, depth=.22,
        alongshore_length=solver.cfg.grid.Ly, amplitude=.02, phase_offset=.8),))
    zone = WaveRelaxationZone(solver.cfg, inlet, width=.3, timescale=.048)
    before = solver.state.get_interior("alpha")
    zone.impose(solver.state, 0)
    after = solver.state.get_interior("alpha")
    assert np.all(np.diff(after, axis=2) <= 0)
    assert np.all((after >= 0) & (after <= 1))
    assert np.any(np.sum((after > 0) & (after < 1), axis=2) > 1)
    assert not np.array_equal(before, after)


@pytest.mark.parametrize("column", ["first", "beyond"])
def test_plic_ray_source_refuses_cut_columns_before_changing_carried_state(column):
    # The source skips the solid-adjacent normal fit, which can write any
    # interface cell within one cell of a cut cell: the first column beyond
    # the zone must be uncut as well.
    solver = source_solver()
    # A zone narrower than the domain leaves columns beyond it.
    solver.set_wave_relaxation(WaveRelaxationZone(
        solver.cfg, solver.wave_relaxation.inlet, width=.2, timescale=.048))
    initialize_joint_plane(solver.state, (0, 0, 1), .22)
    g = solver.cfg.grid
    source_columns = min(g.nx, math.ceil(solver.wave_relaxation.width/g.dx))
    assert source_columns < g.nx
    theta = solver.state.get_geometry_interior("theta")
    theta[0 if column == "first" else source_columns, 0, 0] = .5
    solver.state.set_geometry_interior("theta", theta)
    before = {name: solver.state.field(name).numpy().copy() for name in solver.state.FIELDS}
    with pytest.raises(ValueError, match="requires uncut source columns"):
        solver.wave_relaxation.apply(solver.state, 0., .005)
    for name, values in before.items():
        np.testing.assert_array_equal(solver.state.field(name).numpy(), values)


@pytest.mark.parametrize("normal", [(1e-10, .2, .8), (1e-10, 1e-9, 1), (-1e-9, .4, -.6)])
def test_source_offsets_preserve_thin_and_nearly_planar_cell_volumes(normal):
    n = np.array(normal, float); n /= abs(n).sum()
    fractions = np.array((1e-6, .02, .49, .9, 1-1e-6))
    normals = np.tile(n, (len(fractions), 1))
    offsets = fit_offsets(normals, fractions)
    recovered = np.array([plane_fraction(n, d) for d in offsets])
    np.testing.assert_allclose(recovered, fractions, atol=8e-14, rtol=0)


def cuda_column(alpha, normals, target, rate):
    """One physical column, flat target and the actual continuous source taper."""
    prepare_device(DEVICE)
    offsets = fit_offsets(normals, alpha)
    parameters = dict(x0=0., y0=0., z0=0., dx=.0125, dy=.02, dz=.01,
                      ny=1, nz=len(alpha), width=.3, ramp=0., level=target*.01,
                      time=0., rate=rate, impose=0)
    after, audit = integrate_source_rays(alpha[None, :], normals[None, :, :], offsets[None, :],
        parameters, np.zeros((1, 6)), order=8, device=DEVICE)
    assert audit["maximum_input_volume_error"] < 1e-12
    assert audit["maximum_column_source_residual"] < 1e-12
    return after[0]


def column_weight(x, rate):
    return -np.expm1(-rate*relaxation_weight(x*.0125, 0., .3))


def test_cuda_vertical_footprint_retains_dry_ray_birth_and_wet_ray_loss():
    alpha = np.full(16, .5)
    normals = np.tile((1., 0., 0.), (16, 1))
    rate = -np.log(.9)
    after = cuda_column(alpha, normals, 8., rate)
    expected = np.full(16, .5)
    expected[0] += quad(lambda x: 8*column_weight(x, rate), .5, 1., epsabs=1e-14)[0]
    expected[-1] -= quad(lambda x: 8*column_weight(x, rate), 0., .5, epsabs=1e-14)[0]
    np.testing.assert_allclose(after, expected, atol=3e-14, rtol=0)


@pytest.mark.parametrize("initial_height", [0., 2.5])
def test_cuda_source_fills_above_the_initial_intervals(initial_height):
    alpha = np.clip(initial_height-np.arange(24), 0, 1)
    normals = np.tile((0., 0., 1.), (24, 1))
    rate = np.log(2.)
    after = cuda_column(alpha, normals, 18., rate)
    def height(x):
        return initial_height+column_weight(x, rate)*(18-initial_height)
    expected = np.zeros(24)
    for k in range(24):
        cuts = [brentq(lambda x: height(x)-edge, 0, 1, xtol=1e-15)
                for edge in (k, k+1) if (height(0)-edge)*(height(1)-edge) < 0]
        expected[k] = quad(lambda x: np.clip(height(x)-k, 0, 1), 0, 1,
                           points=cuts, epsabs=1e-13, epsrel=1e-12)[0]
    np.testing.assert_allclose(after, expected, atol=3e-13, rtol=0)


def test_cuda_source_retains_a_rate_below_exp_subtraction_precision():
    alpha = np.zeros(24)
    normals = np.tile((0., 0., 1.), (24, 1))
    after = cuda_column(alpha, normals, 18., 1e-20)
    expected = quad(lambda x: 18*column_weight(x, 1e-20), 0, 1, epsabs=1e-32)[0]
    assert after[0] > 0
    assert np.all(after[1:] == 0)
    assert after[0] == pytest.approx(expected, rel=1e-12, abs=0)
