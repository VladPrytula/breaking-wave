"""Kinetic compatibility, support, and physical work of liquid diffusion."""

import gc
import weakref

import numpy as np
import pytest
from scipy.sparse import csr_matrix
import warp as wp
from warp.optim.linear import cg, LinearOperator
from warp.sparse import bsr_from_triplets

from shoreloop.config import BC, BC3DConfig, Grid3DConfig, Physics3DConfig, Solver3DConfig
from shoreloop.device import DEVICE
from shoreloop.physics3d.cut_cells import initialize_scene_cut_cells
from shoreloop.physics3d.energy import EnergyLedger
from shoreloop.physics3d.liquid_pressure import LiquidPressureOperator
from shoreloop.physics3d.liquid_viscosity import (
    BLOCK_LIQUID_FRACTION, LiquidVelocityMetric, LiquidViscosityOperator, _difference_matvec,
    _edge_rhs, _line_preconditioner, liquid_viscosity_operator,
)
from shoreloop.physics3d.solver import Solver3D
from shoreloop.physics3d.state import State3D
from shoreloop.physics3d.viscosity import ViscosityConfig, ViscosityModel
from shoreloop.physics3d.vof import initialize_joint_plane
from shoreloop.scene.geometry3d import AnalyticBeach, SceneGeometry


def test_thin_line_diffusion_retains_small_positive_translation_inertia():
    # The second production step has kinetic penalties about 1e10 times
    # larger than the inertia of rigid motion. A rounded assembled FP32
    # matrix loses this small positive mode; unequal inertia also makes its
    # momentum-preserving increment differ from an antisymmetric increment.
    mass = csr_matrix([[1+1e-10, -1.], [-1., 1+2e-10]])
    nodes = np.asarray([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=np.int32)
    edges = np.asarray([[0, 1]], dtype=np.int32)
    coefficients, dt = np.asarray([1e-6]), .005
    matrix = mass + .5*dt*coefficients[0]*csr_matrix([[1., -1.], [-1., 1.]])
    row_sum = np.asarray(mass.sum(axis=1)).ravel()
    def upload(values, dtype=wp.float32):
        return wp.array(values, dtype=dtype, device=DEVICE)
    starts, columns = upload(matrix.indptr, wp.int32), upload(matrix.indices, wp.int32)
    values, sums = upload(matrix.data), upload(row_sum)
    def matvec(x, y, z, alpha, beta):
        wp.launch(_difference_matvec, dim=2,
                  inputs=[starts, columns, values, sums, x, y, z, alpha, beta], device=DEVICE)
    operator = LinearOperator((2, 2), wp.float32, wp.get_device(DEVICE), matvec)
    result = wp.zeros(2, dtype=wp.float32, device=DEVICE)
    matvec(wp.ones_like(result), result, result, 1., 0.)
    np.testing.assert_allclose(result.numpy(), row_sum, rtol=2e-7)
    rhs = np.asarray([5e-12, -5e-12])
    inverse = _line_preconditioner(mass, nodes, edges, coefficients, dt, DEVICE)
    result.zero_()
    _, residual, tolerance = cg(operator, upload(rhs), result, M=inverse,
                                 tol=2e-7, atol=0., maxiter=2000, check_every=1)
    assert residual <= tolerance
    increment = result.numpy().astype(float)
    expected = np.linalg.solve(matrix.toarray(), rhs)
    np.testing.assert_allclose(increment, expected, rtol=3e-6, atol=0.)
    assert abs(row_sum @ increment) < 2e-6*np.linalg.norm(row_sum)*np.linalg.norm(increment)


def test_thin_chain_force_and_solve_preserve_metric_momentum():
    # A connected thin line amplifies force-summation and Krylov roundoff in
    # its small rigid mode. Check the actual equation and momentum, not just
    # the recursively updated CG residual that hid the production error.
    mass = csr_matrix([[.7+1e-10, -.7, 0.],
                       [-.7, 2.+2e-10, -1.3],
                       [0., -1.3, 1.3+3e-10]])
    nodes = np.asarray([[0, i, 0, 0] for i in range(3)], np.int32)
    edges = np.asarray([[0, 1], [1, 2]], np.int32)
    coefficients, dt = np.asarray([1e-6, 2e-6], np.float32), .005
    stiffness = csr_matrix([[1., -1., 0.], [-1., 3., -2.], [0., -2., 2.]])*float(coefficients[0])
    matrix = mass+.5*dt*stiffness
    row_sum = np.asarray(mass.sum(axis=1)).ravel()
    def upload(values, dtype=wp.float64):
        return wp.array(values, dtype=dtype, device=DEVICE)
    before = upload([.2, -.1, .3], wp.float32)
    rhs = wp.zeros(3, dtype=wp.float64, device=DEVICE)
    wp.launch(_edge_rhs, dim=2, inputs=[upload(edges, wp.vec2i), upload(coefficients, wp.float32),
              before, rhs, dt], device=DEVICE)
    force = rhs.numpy()
    assert abs(force.sum()) <= 4*np.finfo(float).eps*np.linalg.norm(force)
    starts, columns = upload(matrix.indptr, wp.int32), upload(matrix.indices, wp.int32)
    values, sums = upload(matrix.data), upload(row_sum)
    def matvec(x, y, z, alpha, beta):
        wp.launch(_difference_matvec, dim=3,
                  inputs=[starts, columns, values, sums, x, y, z, alpha, beta], device=DEVICE)
    operator = LinearOperator((3, 3), wp.float64, wp.get_device(DEVICE), matvec)
    inverse = _line_preconditioner(mass, nodes, edges, coefficients.astype(float), dt, DEVICE,
                                   dtype=wp.float64)
    result, residual = wp.zeros_like(rhs), wp.zeros_like(rhs)
    cg(operator, rhs, result, M=inverse, tol=2e-7, atol=0., maxiter=2000, check_every=1)
    matvec(result, rhs, residual, -1., 1.)
    assert np.linalg.norm(residual.numpy()) <= 2e-7*np.linalg.norm(force)
    increment = result.numpy()
    assert abs(row_sum@increment-force.sum()) <= 2e-6*np.linalg.norm(row_sum)*np.linalg.norm(increment)


def _periodic(fraction=1.0, nx=4, ny=8):
    cfg = Solver3DConfig(grid=Grid3DConfig(nx=nx, ny=ny, nz=4, Lx=nx/4, Ly=ny/8, Lz=1),
                         physics=Physics3DConfig(g=0), device=DEVICE,
                         pressure_discretization="liquid_mimetic", bc=BC3DConfig(
                             xlo=BC.PERIODIC, xhi=BC.PERIODIC, ylo=BC.PERIODIC,
                             yhi=BC.PERIODIC, zlo=BC.PERIODIC, zhi=BC.PERIODIC))
    state = State3D(cfg)
    alpha = np.ones(cfg.grid.shape, np.float32)
    if fraction < 1:
        alpha[:] = 0
        alpha[:, :, 1] = fraction
    state.set_interior("alpha", alpha)
    return state


def test_near_threshold_cells_with_one_heavy_face_solve_in_few_iterations_with_cell_blocks():
    # The production run's late viscosity and wall solves (AW-1118) were set by
    # resolved cells holding about 0.02 of a cell of liquid whose mimetic mass
    # couples one heavy face (2-4 cell volumes) to light faces of the other
    # components. Own-axis lines contain no such coupling; the own-cell block
    # inverse does. Give every other film cell that structure (a positive
    # semidefinite rank-one term, as the consistent mass term P P^T / V is).
    state = _periodic(.01, nx=16, ny=16)
    metric = LiquidVelocityMetric(state)
    pressure, N, volume = metric.pressure, metric.node_count, metric.volume
    assert np.all(pressure._volumes < BLOCK_LIQUID_FRACTION*volume)
    rng = np.random.default_rng(1118)
    rows, columns, entries = [], [], []
    for c in np.flatnonzero(pressure._cell_index[:, :2].sum(axis=1) % 2 == 0):
        faces = metric.node_of_dof_host[c, :pressure._widths[c]]
        direction = rng.uniform(.2, .5, len(faces))
        direction[rng.integers(len(faces))] = 1.
        block = rng.uniform(1, 4)*volume*np.outer(direction, direction)
        rows.append(np.repeat(faces, len(faces))); columns.append(np.tile(faces, len(faces))); entries.append(block.ravel())
    heavy = csr_matrix((np.concatenate(entries), (np.concatenate(rows), np.concatenate(columns))), shape=(N, N))
    system = (metric.mass_host+heavy).tocsr()
    system.sum_duplicates()
    coo = system.tocoo()
    device = bsr_from_triplets(N, N, metric.upload(coo.row, wp.int32), metric.upload(coo.col, wp.int32),
                               metric.upload(coo.data, wp.float64))
    row_sum = metric.upload(np.asarray(system.sum(axis=1)).ravel(), wp.float64)

    def matvec(x, y, z, alpha, beta):
        wp.launch(_difference_matvec, dim=N, inputs=[device.offsets, device.columns, device.values, row_sum,
                  x, y, z, alpha, beta], device=DEVICE)
    operator = LinearOperator((N, N), wp.float64, wp.get_device(DEVICE), matvec)
    exact = rng.normal(0, .02, N)
    rhs = system @ exact
    lines = _line_preconditioner(system, metric.nodes_host, np.empty((0, 2), np.int32), np.empty(0), 0., DEVICE,
                                 dtype=wp.float64)
    iterations, solutions = {}, {}
    for name, inverse in (("lines", lines), ("lines+blocks", metric.line_preconditioner(device))):
        result = wp.zeros(N, dtype=wp.float64, device=DEVICE)
        iterations[name], _, _ = cg(operator, metric.upload(rhs, wp.float64), result, M=inverse,
                                    tol=2e-7, atol=0., maxiter=2000, check_every=1)
        solutions[name] = result.numpy()
        assert np.linalg.norm(system @ solutions[name]-rhs) <= 2e-7*np.linalg.norm(rhs)
    assert iterations["lines"] >= 80 and iterations["lines+blocks"] <= 25
    # Both increments solve the same equation within its tolerance; their
    # difference is bounded by the system's conditioning times that tolerance.
    difference = solutions["lines+blocks"]-solutions["lines"]
    assert np.linalg.norm(system @ difference) <= 4e-7*np.linalg.norm(rhs)
    bound = np.linalg.cond(system.toarray())*2e-7*np.linalg.norm(exact)
    assert np.linalg.norm(difference) <= bound
    assert np.linalg.norm(solutions["lines+blocks"]-exact) <= bound


def _energy(operator, state):
    v = operator.read(state).numpy().astype(float)
    return .5*state.cfg.physics.rho*(v @ operator.mass_host @ v)


@pytest.mark.parametrize("cached", [False, True])
def test_obsolete_liquid_geometry_releases_without_cyclic_collection(cached):
    # Large pressure/viscosity arrays must be released with their geometry,
    # rather than accumulate until Python's next cyclic-GC pass (production OOM).
    state = _periodic()
    pressure = LiquidPressureOperator(state, velocity_space="mac") if cached else None
    operator = liquid_viscosity_operator(state, pressure)
    if cached:
        assert liquid_viscosity_operator(state, pressure) is operator
    else:
        pressure = operator.pressure
    references = weakref.ref(pressure), weakref.ref(operator)
    enabled = gc.isenabled()
    gc.disable()
    try:
        del operator, pressure
        assert all(reference() is None for reference in references)
    finally:
        if enabled:
            gc.enable()


def _shear(operator, state):
    values = np.zeros(len(operator.nodes_host), np.float32)
    for n, (a, _, j, _) in enumerate(operator.nodes_host):
        if a == 0:
            values[n] = .2*np.sin(2*np.pi*(j+.5)/state.cfg.grid.ny)
    operator.write(state, operator.upload(values))
    return values


@pytest.mark.parametrize("fraction", [.01, .49, .51, 1.0])
def test_rigid_liquid_translation_has_no_sgs_or_diffusive_work(fraction):
    state = _periodic(fraction)
    operator = LiquidViscosityOperator(state)
    values = np.asarray([(.3, -.2, .1)[n[0]] for n in operator.nodes_host], np.float32)
    operator.write(state, operator.upload(values))
    alpha = state.get_interior("alpha")
    model = ViscosityModel(state.cfg, ViscosityConfig(molecular_viscosity=.004, smagorinsky_cs=.12))
    result = model.apply(state, .002)
    np.testing.assert_array_equal(operator.read(state).numpy(), values)
    np.testing.assert_array_equal(model.strain_rate.numpy(), 0)
    np.testing.assert_array_equal(state.get_interior("alpha"), alpha)
    assert result.maximum_sgs_viscosity == 0
    assert result.total_energy_loss == 0


@pytest.mark.parametrize("nu", [.002, .008])
@pytest.mark.parametrize("fraction", [1.0, .01, 1e-5])
def test_periodic_shear_matches_midpoint_decay_and_energy_work(fraction, nu):
    state = _periodic(fraction)
    operator = LiquidViscosityOperator(state)
    initial = _shear(operator, state)
    before = _energy(operator, state)
    dt = .01
    model = ViscosityModel(state.cfg, ViscosityConfig(molecular_viscosity=nu))
    result = model.apply(state, dt)
    eigenvalue = 4*nu*np.sin(np.pi/state.cfg.grid.ny)**2/state.cfg.grid.dy**2
    factor = (1-.5*dt*eigenvalue)/(1+.5*dt*eigenvalue)
    np.testing.assert_allclose(operator.read(state).numpy(), factor*initial, atol=2e-7)
    assert _energy(operator, state) < before
    np.testing.assert_allclose(result.molecular_energy_loss+result.molecular_numerical_loss,
                               before-_energy(operator, state), atol=1e-11)
    field_work = np.sum(model.molecular_dissipation.numpy(), dtype=float)*operator.volume*dt
    assert field_work == pytest.approx(result.molecular_energy_loss, rel=2e-6)


def test_zero_diffusion_coefficients_leave_liquid_state_unchanged():
    state = _periodic(.01)
    operator = LiquidViscosityOperator(state)
    _shear(operator, state)
    before = {q: state.field(q).numpy().copy() for q in ("u", "v", "w", "alpha")}
    model = ViscosityModel(state.cfg, ViscosityConfig())
    result = model.apply(state, .01)
    assert result.total_energy_loss == 0 and result.maximum_sgs_viscosity == 0
    for q, values in before.items():
        np.testing.assert_array_equal(state.field(q).numpy(), values)
    for field in (model.molecular_dissipation, model.sgs_dissipation, model.sgs_viscosity):
        np.testing.assert_array_equal(field.numpy(), 0)


def test_sgs_strength_increases_shear_decay_on_retained_water():
    losses, energies = [], []
    for cs in (.08, .18):
        state = _periodic(.49)
        operator = LiquidViscosityOperator(state)
        _shear(operator, state)
        model = ViscosityModel(state.cfg, ViscosityConfig(smagorinsky_cs=cs))
        result = model.apply(state, .002)
        assert 0 < result.maximum_sgs_viscosity <= model.settings.maximum_sgs_viscosity
        losses.append(result.sgs_energy_loss)
        energies.append(_energy(operator, state))
        field_work = np.sum(model.sgs_dissipation.numpy(), dtype=float)*operator.volume*.002
        assert field_work == pytest.approx(result.sgs_energy_loss, rel=2e-6)
    assert losses[1] > losses[0] > 0
    assert energies[1] < energies[0]


@pytest.mark.parametrize("inertia", [0.0, 1.0])
@pytest.mark.parametrize("seed", [7, 49])
def test_cut_liquid_diffusion_preserves_metric_momentum_and_dissipates(seed, inertia):
    cfg = Solver3DConfig(grid=Grid3DConfig(nx=8, ny=2, nz=6, Lx=.1, Ly=.025, Lz=.06),
                         device=DEVICE, pressure_discretization="liquid_mimetic",
                         liquid_extension_inertia=inertia)
    state = State3D(cfg)
    initialize_scene_cut_cells(state, SceneGeometry(AnalyticBeach(-.1, .1, -.003)))
    initialize_joint_plane(state, (-.1, 0, 1), .0103)
    operator = LiquidViscosityOperator(state)
    spectrum = np.linalg.eigvalsh(operator.mass_host.toarray())
    # Removing extension inertia can leave unobserved cut-cell modes with
    # zero kinetic norm. Diffusion must dissipate the physical semidefinite
    # norm too; do not silently add inertia to make it positive definite.
    assert spectrum.min() >= -32*np.finfo(float).eps*spectrum.max()
    if inertia > 0:
        assert spectrum.min() > 0
    values = np.random.default_rng(seed).normal(0, .02, len(operator.nodes_host)).astype(np.float32)
    operator.write(state, operator.upload(values))
    before = _energy(operator, state)
    np.testing.assert_allclose(before, operator.pressure.kinetic_energy(operator.pressure.read_mac_state(state)), rtol=1e-12)
    translations = np.eye(3)[operator.nodes_host[:, 0]]
    initial_momentum = translations.T @ operator.mass_host @ values
    alpha = state.get_interior("alpha")
    model = ViscosityModel(cfg, ViscosityConfig(molecular_viscosity=.0002, smagorinsky_cs=.12))
    result = model.apply(state, .002)
    after = _energy(operator, state)
    assert after < before
    assert result.molecular_energy_loss > 0 and result.sgs_energy_loss > 0
    numerical = result.molecular_numerical_loss+result.sgs_numerical_loss
    assert result.total_energy_loss+numerical == pytest.approx(before-after, abs=1e-12)
    assert abs(numerical) < 2e-5*before
    for field, work in ((model.molecular_dissipation, result.molecular_energy_loss),
                        (model.sgs_dissipation, result.sgs_energy_loss)):
        density = field.numpy()
        assert density.min() >= 0
        assert np.sum(density, dtype=float)*operator.volume*.002 == pytest.approx(work, rel=2e-6)
    final_momentum = translations.T @ operator.mass_host @ operator.read(state).numpy()
    np.testing.assert_allclose(final_momentum, initial_momentum, atol=2e-6*np.linalg.norm(initial_momentum))
    np.testing.assert_array_equal(state.get_interior("alpha"), alpha)
    for q in ("u", "v", "w"):
        assert np.max(np.abs(state.get_interior(q)[state.get_geometry_interior(q+"_aperture") == 0]), initial=0) == 0


def test_disconnected_translating_sheets_do_not_share_sgs_strain():
    state = _periodic()
    alpha = np.zeros(state.cfg.grid.shape, np.float32)
    alpha[:, 1, :] = .1
    alpha[:, 3, :] = .2
    state.set_interior("alpha", alpha)
    operator = LiquidViscosityOperator(state)
    values = np.asarray([(.3 if j <= 2 else -.2) for _, _, j, _ in operator.nodes_host], np.float32)
    operator.write(state, operator.upload(values))
    model = ViscosityModel(state.cfg, ViscosityConfig(molecular_viscosity=.004, smagorinsky_cs=.12))
    result = model.apply(state, .002)
    assert result.total_energy_loss == 0 and result.maximum_sgs_viscosity == 0
    np.testing.assert_array_equal(operator.read(state).numpy(), values)
    np.testing.assert_array_equal(model.strain_rate.numpy(), 0)


def test_liquid_supported_strain_reproduces_an_affine_resolved_field():
    cfg = Solver3DConfig(grid=Grid3DConfig(nx=8, ny=4, nz=8, Lx=.1, Ly=.05, Lz=.08),
                         device=DEVICE, pressure_discretization="liquid_mimetic",
                         bc=BC3DConfig(ylo=BC.WALL, yhi=BC.WALL))
    state = State3D(cfg)
    initialize_scene_cut_cells(state, SceneGeometry(AnalyticBeach(-.1, .1, -.003)))
    initialize_joint_plane(state, (0, 0, 1), .0673)
    operator = LiquidViscosityOperator(state)
    gradient = np.asarray(((.3, -.7, 1.1), (.4, -.2, .5), (-.8, .6, .9)))
    values = []
    for a, i, j, k in operator.nodes_host:
        position = (np.asarray((i, j, k), dtype=float)+.5)*operator.spacing
        position[a] -= .5*operator.spacing[a]
        values.append((gradient @ position)[a]+.2)
    operator.write(state, operator.upload(values))
    model = ViscosityModel(cfg, ViscosityConfig(smagorinsky_cs=.12))
    model.update_sgs(state)
    symmetric = .5*(gradient+gradient.T)
    expected = np.sqrt(2*np.sum(symmetric*symmetric))
    ng = cfg.grid.ng
    np.testing.assert_allclose(model.strain_rate.numpy()[ng+2:ng+6, ng+1:ng+3, ng+3:ng+5], expected, rtol=3e-6)


def test_solver_couples_both_diffusion_channels_to_its_energy_ledger():
    state = _periodic()
    solver = Solver3D(state.cfg, viscosity_settings=ViscosityConfig(molecular_viscosity=.004, smagorinsky_cs=.12))
    solver.state.fill_("alpha", 1)
    operator = LiquidViscosityOperator(solver.state)
    _shear(operator, solver.state)
    solver.set_energy_ledger(EnergyLedger(solver.cfg))
    before = solver.energy_ledger.measure(solver.state).kinetic
    assert solver.step(.001).converged
    assert solver.last_energy.molecular_loss > 0 and solver.last_energy.sgs_loss > 0
    assert solver.last_energy.final.kinetic < before
    assert abs(solver.last_energy.numerical_residual) < 3e-6*before
