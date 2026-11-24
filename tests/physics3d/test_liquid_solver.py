"""Real Solver3D stepping with the compatible pressure/transport pair."""

import numpy as np
import pytest

from shoreloop.config import BC, BC3DConfig, Grid3DConfig, Physics3DConfig, Solver3DConfig
from shoreloop.device import DEVICE
from shoreloop.physics3d.cut_cells import initialize_scene_cut_cells
from shoreloop.physics3d.energy import EnergyLedger
from shoreloop.physics3d.liquid_pressure import LiquidPressureOperator
from shoreloop.physics3d.porous import PorousBedModel, PorousStorageConfig
from shoreloop.physics3d.relaxation import WaveRelaxationZone
from shoreloop.physics3d.solver import Solver3D
from shoreloop.physics3d.viscosity import ViscosityConfig
from shoreloop.physics3d.vof import initialize_joint_plane, measure_vof
from shoreloop.physics3d.waves import LinearWaveInlet, PeriodicWaveComponent
from shoreloop.physics3d.wet_dry import WispRemover
from shoreloop.scene.geometry3d import AnalyticBeach, SceneGeometry


@pytest.mark.parametrize("fraction", [.01, .49, .499999, .5, .500001, .51])
@pytest.mark.parametrize("extension_inertia", [0.0, 1.0])
def test_liquid_solver_advances_free_fall_without_losing_water(fraction, extension_inertia):
    g = Grid3DConfig(nx=4, ny=2, nz=6, Lx=.05, Ly=.025, Lz=.06)
    cfg = Solver3DConfig(grid=g, device=DEVICE, pressure_discretization="liquid_mimetic",
                         liquid_extension_inertia=extension_inertia,
                         bc=BC3DConfig(xlo=BC.PERIODIC, xhi=BC.PERIODIC))
    solver = Solver3D(cfg)
    alpha = np.zeros(g.shape, np.float32)
    alpha[:, :, 3] = fraction
    solver.state.set_interior("alpha", alpha)
    before = measure_vof(solver.state).volume
    for _ in range(3):
        result = solver.step(.001)
        assert result.converged
    after = measure_vof(solver.state)
    assert abs(after.volume-before) < 3e-7*before
    assert 0 <= after.alpha_min <= after.alpha_max <= 1
    np.testing.assert_allclose(solver.state.get_interior("w")[:, :, 3:5], -.02943, atol=3e-6)
    assert np.sum(solver.state.get_interior("alpha")[:, :, :3]) > 0


def test_liquid_solver_keeps_subcell_cut_beach_at_rest_with_energy_accounting():
    g = Grid3DConfig(nx=8, ny=2, nz=5, Lx=.1, Ly=.025, Lz=.05)
    cfg = Solver3DConfig(grid=g, device=DEVICE, pressure_discretization="liquid_mimetic")
    solver = Solver3D(cfg)
    initialize_scene_cut_cells(solver.state, SceneGeometry(AnalyticBeach(-.1, .1, -.003)))
    initialize_joint_plane(solver.state, (0, 0, 1), .0173)
    solver.set_energy_ledger(EnergyLedger(cfg))
    before = measure_vof(solver.state).volume
    for _ in range(3):
        result = solver.step(.001)
        assert result.converged
        assert max(np.max(np.abs(solver.state.get_interior(q))) for q in ("u", "v", "w")) < 2e-6
    assert abs(measure_vof(solver.state).volume-before) < 3e-7*before
    assert np.isfinite(solver.last_energy.numerical_residual)


def test_liquid_projection_reports_solution_and_cell_divergence():
    # The gate reads the FP64 solution's divergence and the stored field's net
    # flux per Cartesian cell volume. The liquid-relative float32 statistic is
    # bounded below by neither (it divides write-back rounding by the liquid
    # volume) and stays diagnostic.
    g = Grid3DConfig(nx=8, ny=2, nz=5, Lx=.1, Ly=.025, Lz=.05)
    cfg = Solver3DConfig(grid=g, device=DEVICE, pressure_discretization="liquid_mimetic")
    solver = Solver3D(cfg)
    initialize_scene_cut_cells(solver.state, SceneGeometry(AnalyticBeach(-.1, .1, -.003)))
    initialize_joint_plane(solver.state, (0, 0, 1), .0173)
    solver.state.fill_("u", .3)
    result = solver.step(.001)
    assert result.converged
    assert result.max_divergence_before > 1.0
    assert result.solution_max_divergence < 1e-6
    assert 0 <= result.max_cell_divergence <= result.max_divergence_after*(1+1e-12)
    operator = solver.projection.last_operator
    cell, value = operator.worst_divergence_cell(operator.read_mac_state(solver.state), cartesian=True)
    assert abs(value) == pytest.approx(result.max_cell_divergence, rel=1e-12, abs=1e-30)
    assert all(0 <= c < n for c, n in zip(cell, g.shape))


def test_stale_air_face_speed_does_not_limit_the_liquid_time_step():
    # A fast face between two dry cells moves no water. On the liquid path the
    # transport limit over positive-mass faces governs; the all-face speed is
    # only reported (a stale 81 m/s air face pinned production dt at .062 ms).
    g = Grid3DConfig(nx=4, ny=2, nz=6, Lx=.05, Ly=.025, Lz=.06)
    cfg = Solver3DConfig(grid=g, device=DEVICE, pressure_discretization="liquid_mimetic",
                         bc=BC3DConfig(xlo=BC.PERIODIC, xhi=BC.PERIODIC))
    solver = Solver3D(cfg)
    alpha = np.zeros(g.shape, np.float32)
    alpha[:, :, :2] = 1.0
    solver.state.set_interior("alpha", alpha)
    u = np.full(solver.state.grid.interior_shape("u"), 1.0, np.float32)
    u[1, 1, 5] = 80.0
    solver.state.set_interior("u", u)
    timestep = solver.compute_dt()
    assert timestep.max_face_speed == pytest.approx(80.0)
    assert timestep.limiting_constraint == "interface_advection"
    assert timestep.dt == pytest.approx(.5*g.dx/1.0, rel=1e-6)


def test_dry_liquid_solver_has_no_pressure_or_water_source():
    cfg = Solver3DConfig(grid=Grid3DConfig(nx=4, ny=2, nz=4, Lx=.05, Ly=.025, Lz=.04),
                         device=DEVICE, pressure_discretization="liquid_mimetic")
    solver = Solver3D(cfg)
    result = solver.step(.001)
    assert result.converged
    assert result.active_cells == 0
    assert measure_vof(solver.state).volume == 0


def test_liquid_solver_keeps_dry_state_with_compatible_diffusion():
    cfg = Solver3DConfig(grid=Grid3DConfig(nx=4, ny=2, nz=4, Lx=.05, Ly=.025, Lz=.04),
                         device=DEVICE, pressure_discretization="liquid_mimetic")
    solver = Solver3D(cfg, viscosity_settings=ViscosityConfig(molecular_viscosity=1e-6, smagorinsky_cs=.12))
    assert solver.step(.001).converged
    assert solver.last_viscosity.total_energy_loss == 0
    assert measure_vof(solver.state).volume == 0


def test_liquid_solver_keeps_accounted_volume_and_small_divergence_during_infiltration():
    g = Grid3DConfig(nx=8, ny=2, nz=5, Lx=.1, Ly=.025, Lz=.05)
    cfg = Solver3DConfig(grid=g, device=DEVICE, pressure_discretization="liquid_mimetic")
    solver = Solver3D(cfg)
    cut = initialize_scene_cut_cells(solver.state, SceneGeometry(AnalyticBeach(-.1, .1, -.003)))
    initialize_joint_plane(solver.state, (-.1, 0, 1), .0103)
    solver.set_porous_bed(PorousBedModel.from_geometry(cfg, cut, storage_settings=PorousStorageConfig(
        layer_depth=.08, capacity_fraction=.2, infiltration_rate=.8, drainage_time=4.0)))
    solver.set_energy_ledger(EnergyLedger(cfg))
    initial = measure_vof(solver.state).volume
    # Start with empty pores so the control exercises an actual source/sink,
    # unlike a short drainage control whose initial pore capacity is saturated.
    for _ in range(3):
        solver.step(.001)
        assert solver.last_porous_storage.infiltrated_volume > 0
        operator = LiquidPressureOperator(solver.state, velocity_space="mac")
        rms, _ = operator.divergence(operator.read_mac_state(solver.state))
        assert rms < solver.projection.settings.rms_divergence_tolerance
        accounted = measure_vof(solver.state).volume + solver.last_porous_storage.stored_volume
        assert abs(accounted-initial) < 3e-7*initial
        fresh = solver.energy_ledger.measure(solver.state)
        assert solver.last_energy.final == fresh


@pytest.mark.parametrize("timescale", [.05, 1e-5])
def test_liquid_solver_rebuilds_geometry_when_relaxation_adds_water(timescale):
    cfg = Solver3DConfig(grid=Grid3DConfig(nx=4, ny=2, nz=4, Lx=.05, Ly=.025, Lz=.04),
                         device=DEVICE, pressure_discretization="liquid_mimetic")
    solver = Solver3D(cfg, viscosity_settings=ViscosityConfig(molecular_viscosity=1e-6))
    inlet = LinearWaveInlet((PeriodicWaveComponent(period=1, depth=.025,
                                                   alongshore_length=cfg.grid.Ly, amplitude=0),))
    solver.set_wave_relaxation(WaveRelaxationZone(cfg, inlet, cfg.grid.Lx, timescale=timescale))
    solver.set_energy_ledger(EnergyLedger(cfg))
    result = solver.step(.001)
    assert result.converged and result.active_cells > 0
    water = measure_vof(solver.state)
    assert water.volume > 0 and 0 <= water.alpha_min <= water.alpha_max <= 1
    assert water.volume == pytest.approx(solver.last_relaxation.volume_source, rel=3e-7)
    assert result.active_cells == np.count_nonzero(solver.state.get_interior("alpha") > 1e-6)
    assert solver.last_energy.final == solver.energy_ledger.measure(solver.state)


def test_liquid_solver_rebuilds_geometry_after_explicit_wisp_removal():
    cfg = Solver3DConfig(grid=Grid3DConfig(nx=4, ny=2, nz=4, Lx=.05, Ly=.025, Lz=.04),
                         physics=Physics3DConfig(g=0), device=DEVICE,
                         pressure_discretization="liquid_mimetic")
    solver = Solver3D(cfg, viscosity_settings=ViscosityConfig(molecular_viscosity=1e-6))
    alpha = np.zeros(cfg.grid.shape, np.float32)
    alpha[1, 0, 2] = .01
    solver.state.set_interior("alpha", alpha)
    # A deliberately visible test-only removal exercises the change in
    # active geometry; production's wisp tolerance remains unchanged.
    solver.set_wisp_remover(WispRemover(cfg, tolerance=.02))
    solver.set_energy_ledger(EnergyLedger(cfg))
    initial_volume = measure_vof(solver.state).volume
    result = solver.step(.001)
    assert result.converged and result.active_cells == 0
    assert solver.last_wisp_removal.removed_volume == pytest.approx(initial_volume)
    assert measure_vof(solver.state).volume == 0
    assert solver.last_energy.final.total == 0


@pytest.mark.parametrize("periodic", [False, True])
def test_filled_liquid_solver_preserves_neumann_pressure_gauge(periodic):
    bc = (BC3DConfig(xlo=BC.PERIODIC, xhi=BC.PERIODIC) if periodic
          else BC3DConfig(ylo=BC.WALL, yhi=BC.WALL))
    cfg = Solver3DConfig(grid=Grid3DConfig(nx=4, ny=2, nz=4, Lx=.05, Ly=.025, Lz=.04),
                         device=DEVICE, pressure_discretization="liquid_mimetic", bc=bc)
    solver = Solver3D(cfg)
    solver.state.set_interior("alpha", np.ones(cfg.grid.shape, np.float32))
    result = solver.step(.001)
    assert result.converged
    assert max(np.max(np.abs(solver.state.get_interior(q))) for q in ("u", "v", "w")) < 2e-6
    operator = solver.projection.last_operator
    assert operator.gauges.size == 1
    assert np.linalg.eigvalsh(operator.matrix_host.toarray()).min() > 0
    rng = np.random.default_rng(1104)
    for q in ("u", "v", "w"):
        solver.state.set_interior(q, rng.normal(0, .005, solver.state.grid.interior_shape(q)).astype(np.float32))
    result = solver.projection.project(solver.state, .001)
    assert result.converged


def test_liquid_operator_is_reused_only_while_alpha_is_bitwise_unchanged(monkeypatch):
    # The operator depends on alpha and static geometry only. A velocity write
    # keeps it; the smallest alpha change rebuilds it; and a full step with a
    # source builds exactly one operator once the previous step's geometry is
    # carried over, where it built three before.
    from shoreloop.physics3d import solver as solver_module
    g = Grid3DConfig(nx=6, ny=2, nz=8, Lx=.15, Ly=.05, Lz=.2)
    cfg = Solver3DConfig(grid=g, device=DEVICE, pressure_discretization="liquid_mimetic")
    solver = Solver3D(cfg)
    alpha = np.zeros(g.shape, np.float32)
    alpha[:, :, :4] = 1.0
    alpha[:, :, 4] = .3
    solver.state.set_interior("alpha", alpha)
    first = solver.liquid_operator()
    assert solver.liquid_operator() is first
    solver.state.set_interior("u", np.full(solver.state.grid.interior_shape("u"), .05, np.float32))
    assert solver.liquid_operator() is first
    alpha[2, 1, 4] = np.float32(.3000001)
    solver.state.set_interior("alpha", alpha)
    second = solver.liquid_operator()
    assert second is not first
    assert len(second.cells) == len(first.cells)

    component = PeriodicWaveComponent(period=1.2, depth=.1, alongshore_length=g.Ly,
                                      amplitude=.004)
    solver.set_wave_relaxation(WaveRelaxationZone(cfg, LinearWaveInlet((component,)),
                                                  width=.05, timescale=.05))
    solver.set_wisp_remover(WispRemover(cfg))
    solver.set_energy_ledger(EnergyLedger(cfg))
    built = []
    real = solver_module.LiquidPressureOperator

    class Counting(real):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            built.append(self)

    monkeypatch.setattr(solver_module, "LiquidPressureOperator", Counting)
    solver.step(.002)
    first_step = len(built)
    solver.step(.002)
    # Only the alpha-changing stages (advection, source) rebuild; the start of
    # the second step reuses the end of the first. The old path built four per
    # step: initial, after relaxation, after wisp removal, and final energy.
    assert 2 <= first_step <= 3
    assert 1 <= len(built) - first_step <= 2
    assert solver.liquid_operator() is built[-1]
    assert solver.last_energy is not None
