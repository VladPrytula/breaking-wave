"""Wall-force work in the same liquid kinetic metric as pressure/diffusion."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix
import warp as wp
from warp.sparse import bsr_from_triplets

from shoreloop.config import Grid3DConfig, Solver3DConfig
from shoreloop.device import DEVICE
from shoreloop.physics3d.cut_cells import initialize_scene_cut_cells
from shoreloop.physics3d.liquid_viscosity import LiquidVelocityMetric, _line_preconditioner
from shoreloop.physics3d.liquid_wall import _implicit_wall_increment
from shoreloop.physics3d.state import State3D
from shoreloop.physics3d.vof import initialize_joint_plane
from shoreloop.physics3d.wall_stress import WallStressConfig, WallStressModel
from shoreloop.scene.geometry3d import AnalyticBeach, SceneGeometry


@pytest.mark.parametrize("small_inertia", [1e-10, 1e-12])
def test_wall_force_dissipates_a_small_positive_translation_mode(small_inertia):
    # Independently damping one trace of [u,u] creates a large difference
    # penalty. The compatible force must instead act through the shared inertia.
    mass = csr_matrix([[1+small_inertia, -1.], [-1., 1+small_inertia]])
    factor = csr_matrix([[np.sqrt(2*small_inertia), 0.]])
    device = wp.get_device(DEVICE)
    def upload(values, dtype=wp.float32):
        return wp.array(values, dtype=dtype, device=device)
    entries = mass.tocoo()
    metric = SimpleNamespace(device=device, mass_host=mass, upload=upload,
        nodes_host=np.asarray([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=np.int32),
        mass_row_sum=upload(np.asarray(mass.sum(axis=1)).ravel(), wp.float64),
        mass=bsr_from_triplets(2, 2, upload(entries.row, wp.int32),
                              upload(entries.col, wp.int32), upload(entries.data, wp.float64)))

    def line_preconditioner(system):
        # The production metric factors its lines on the device; this two-node
        # stand-in factors the same matrix with the host reference.
        offsets = system.offsets.numpy(); columns = system.columns.numpy()
        values = system.values.numpy().reshape(-1); nnz = int(offsets[-1])
        rows = np.repeat(np.arange(system.nrow), np.diff(offsets))
        host = csr_matrix((values[:nnz], (rows, columns[:nnz])), shape=(system.nrow, system.ncol))
        return _line_preconditioner(host, metric.nodes_host, np.empty((0, 2), dtype=np.int32),
                                    np.empty(0), 0.0, device, dtype=wp.float64)

    metric.line_preconditioner = line_preconditioner
    f = factor.tocoo()
    factor_device = bsr_from_triplets(1, 2, upload(f.row, wp.int32), upload(f.col, wp.int32), upload(f.data, wp.float64))
    initial = np.asarray([.2, .2], np.float32)
    after, _, iterations, residual = _implicit_wall_increment(metric, factor_device, upload(initial), .1)
    actual = after.numpy().astype(float)
    # Evaluate the independent two-node solution without subtracting nearly
    # equal unit-sized terms in a dense inverse or its right-hand side.
    retained = float(mass[0, 0]-1)
    wall = .1*float(factor[0, 0])**2
    determinant = retained*(2+retained)+wall*(1+retained)
    expected = retained*float(initial[0])/determinant*np.asarray((2+retained, 2+retained+wall))
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=0)
    assert actual @ mass @ actual < initial.astype(float) @ mass @ initial.astype(float)
    assert iterations > 0 and residual <= 2e-7


def _cut_wall():
    cfg = Solver3DConfig(grid=Grid3DConfig(nx=8, ny=2, nz=6, Lx=.1, Ly=.025, Lz=.06),
                         device=DEVICE, pressure_discretization="liquid_mimetic")
    state = State3D(cfg)
    initialize_scene_cut_cells(state, SceneGeometry(AnalyticBeach(-.1, .1, -.003)))
    initialize_joint_plane(state, (-.1, 0, 1), .0103)
    return state


def test_cut_wall_reports_physical_and_numerical_work_without_changing_water():
    state = _cut_wall()
    metric = LiquidVelocityMetric(state)
    values = np.random.default_rng(1104).normal(0, .02, len(metric.nodes_host)).astype(np.float32)
    metric.write(state, metric.upload(values))
    before = metric.read(state).numpy().astype(float)
    alpha = state.get_interior("alpha")
    model = WallStressModel(state.cfg, WallStressConfig(wall_viscosity=.0008))
    result = model.apply(state, .002, liquid_operator=metric.pressure)
    after = metric.read(state).numpy().astype(float)
    loss = .5*state.cfg.physics.rho*(before @ metric.mass_host @ before-after @ metric.mass_host @ after)
    assert loss > 0 and result.energy_loss > 0
    assert result.energy_loss+result.numerical_loss == pytest.approx(loss, abs=1e-12)
    field = model.dissipation.numpy()
    assert field.min() >= 0
    assert np.sum(field, dtype=float)*metric.volume*.002 == pytest.approx(result.energy_loss, rel=2e-6)
    np.testing.assert_array_equal(state.get_interior("alpha"), alpha)
    for q in ("u", "v", "w"):
        assert np.max(np.abs(state.get_interior(q)[state.get_geometry_interior(q+"_aperture") == 0]), initial=0) == 0
