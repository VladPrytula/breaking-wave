"""Balanced single-phase surface-tension physical gates."""

import math

import numpy as np
import pytest

from shoreloop.config import Grid3DConfig, Physics3DConfig, Solver3DConfig
from shoreloop.device import DEVICE
from shoreloop.physics3d.capillary_validation import (
    run_dispersion_pair,
    run_planar_capillary_case,
    run_static_drop_case,
)
from shoreloop.physics3d.energy import EnergyLedger
from shoreloop.physics3d.free_surface import build_free_surface_geometry
from shoreloop.physics3d.state import State3D
from shoreloop.physics3d.surface_tension import (
    MAX_RESOLVED_GRID_SCALED_CURVATURE,
)
from shoreloop.physics3d.vof import initialize_plane


def test_planar_interface_has_zero_capillary_correction():
    metrics = run_planar_capillary_case(DEVICE)

    assert abs(metrics.relative_surface_area_error) < 1.0e-6
    assert metrics.max_abs_curvature == 0.0
    assert metrics.max_abs_capillary_pressure == 0.0
    assert metrics.max_abs_pressure == 0.0
    assert metrics.max_abs_velocity == 0.0
    assert metrics.max_divergence == 0.0
    assert metrics.pressure_iterations == 0


def test_grid_aligned_interface_contributes_surface_energy():
    grid = Grid3DConfig(nx=12, ny=10, nz=8, Lx=0.18, Ly=0.14, Lz=0.20)
    cfg = Solver3DConfig(
        grid=grid,
        physics=Physics3DConfig(g=0.0, surface_tension=0.072),
        device=DEVICE,
    )
    state = State3D(cfg)
    initialize_plane(state, (0.0, 0.0, 1.0), 0.10, samples_per_axis=8)

    energy = EnergyLedger(cfg).measure(state)

    assert energy.surface == pytest.approx(
        cfg.physics.surface_tension * grid.Lx * grid.Ly,
        rel=1.0e-6,
    )


def test_near_pure_cut_cell_layer_uses_float32_geometry_scale():
    grid = Grid3DConfig(nx=12, ny=10, nz=8, Lx=0.18, Ly=0.14, Lz=0.20)
    cfg = Solver3DConfig(
        grid=grid,
        physics=Physics3DConfig(g=0.0, surface_tension=0.072),
        device=DEVICE,
    )
    state = State3D(cfg)
    alpha = state.get_interior("alpha")
    alpha[:, :, :4] = 1.0
    alpha[:, :, 4] = 1.0 - 2.0e-6
    state.set_interior("alpha", alpha)
    theta = state.get_geometry_interior("theta")
    theta[:, :, 4] = 0.5
    state.set_geometry_interior("theta", theta)
    open_normals = np.zeros((*grid.shape, 3), dtype=np.float32)
    open_offsets = np.zeros(grid.shape, dtype=np.float32)
    open_normals[:, :, 4, 2] = -1
    open_offsets[:, :, 4] = -.5
    state.set_open_planes(open_normals, open_offsets)

    geometry = build_free_surface_geometry(state)
    capillary = geometry.surface_tension

    assert capillary is not None
    assert capillary.mixed_cells == 0
    assert capillary.unresolved_cells == 0
    assert capillary.max_abs_curvature == 0.0
    assert capillary.surface_area == pytest.approx(
        grid.Lx * grid.Ly, rel=1.0e-6,
    )


def test_height_function_rejects_a_multi_interface_column():
    grid = Grid3DConfig(nx=12, ny=10, nz=20, Lx=0.24, Ly=0.20, Lz=0.40)
    cfg = Solver3DConfig(
        grid=grid,
        physics=Physics3DConfig(g=0.0, surface_tension=0.072),
        device=DEVICE,
    )
    state = State3D(cfg)
    alpha = state.get_interior("alpha")
    alpha[:, :, :5] = 1.0
    alpha[:, :, 5] = 0.5
    alpha[:, :, 8] = 0.5
    alpha[:, :, 9] = 1.0
    alpha[:, :, 10] = 0.5
    state.set_interior("alpha", alpha)

    geometry = build_free_surface_geometry(state)
    capillary = geometry.surface_tension
    i, j, k = grid.nx // 2, grid.ny // 2, 10
    index = i + grid.ng, j + grid.ng, k + grid.ng

    assert capillary is not None
    assert capillary.height_function_valid.numpy()[index] == 0
    assert capillary.plic_fit_valid.numpy()[index] == 1
    assert capillary.curvature_owner.numpy()[index] == 2
    assert capillary.curvature.numpy()[index] == pytest.approx(0.0, abs=1.0e-6)


def test_subcell_sawtooth_curvature_is_unresolved_not_forced():
    grid = Grid3DConfig(nx=14, ny=12, nz=20, Lx=0.28, Ly=0.24, Lz=0.40)
    cfg = Solver3DConfig(
        grid=grid,
        physics=Physics3DConfig(g=0.0, surface_tension=0.072),
        device=DEVICE,
    )
    state = State3D(cfg)
    alpha = state.get_interior("alpha")
    alpha[:, :, :6] = 1.0
    alpha[:, :, 6] = 0.5
    i, j, k = grid.nx // 2, grid.ny // 2, 11
    alpha[i, j, :11] = 1.0
    alpha[i, j, 11] = 0.5
    state.set_interior("alpha", alpha)

    geometry = build_free_surface_geometry(state)
    capillary = geometry.surface_tension
    index = i + grid.ng, j + grid.ng, k + grid.ng

    assert capillary is not None
    assert capillary.height_function_valid.numpy()[index] == 0
    assert capillary.plic_fit_valid.numpy()[index] == 0
    assert capillary.curvature.numpy()[index] == 0.0
    assert capillary.unresolved_cells > 0
    assert (
        capillary.max_abs_curvature * min(grid.dx, grid.dy, grid.dz)
        <= MAX_RESOLVED_GRID_SCALED_CURVATURE
    )


def test_laplace_jump_static_drop_and_surface_energy_refine():
    coarse = run_static_drop_case(32, DEVICE)
    fine = run_static_drop_case(48, DEVICE)

    assert abs(fine.relative_curvature_error) < abs(
        coarse.relative_curvature_error
    )
    assert abs(fine.relative_pressure_jump_error) < abs(
        coarse.relative_pressure_jump_error
    )
    assert abs(fine.relative_pressure_jump_error) < 0.002
    assert fine.height_function_cells > 0
    assert fine.unresolved_curvature_cells == 0

    assert fine.peak_parasitic_speed < coarse.peak_parasitic_speed
    assert fine.peak_parasitic_speed < 2.0e-5
    for metrics in (coarse, fine):
        assert abs(metrics.relative_volume_residual) < 5.0e-9
        assert metrics.alpha_min == 0.0
        assert metrics.alpha_max == 1.0
        assert metrics.max_divergence < 2.0e-4
        assert metrics.max_pressure_iterations < 100

    assert abs(fine.relative_energy_residual) < abs(
        coarse.relative_energy_residual
    )
    assert abs(fine.relative_energy_residual) < 2.0e-5
    assert abs(fine.capillary_exchange) < abs(coarse.capillary_exchange)
    assert fine.capillary_exchange != 0.0
    assert fine.numerical_energy_residual != fine.capillary_exchange
    assert fine.capillary_limit == pytest.approx(
        (fine.spacing / coarse.spacing) ** 1.5
        * coarse.capillary_limit,
    )


@pytest.mark.integration
def test_gravity_capillary_dispersion_increment_refines():
    coarse = run_dispersion_pair(24, DEVICE)
    fine = run_dispersion_pair(48, DEVICE)

    assert abs(fine.relative_capillary_dispersion_error) < abs(
        coarse.relative_capillary_dispersion_error
    )
    assert abs(fine.relative_capillary_dispersion_error) < 0.002
    for pair in (coarse, fine):
        assert pair.measured_capillary_frequency_squared > 0.0
        assert pair.gravity_capillary.measured_angular_frequency > (
            pair.gravity_only.measured_angular_frequency
        )
        assert pair.gravity_only.capillary_limit is None
        assert math.isfinite(pair.gravity_capillary.capillary_limit)
        for metrics in (pair.gravity_only, pair.gravity_capillary):
            assert abs(metrics.relative_volume_residual) < 5.0e-9
            assert metrics.alpha_min == 0.0
            assert metrics.alpha_max == 1.0
            assert metrics.max_divergence < 1.0e-4
            assert metrics.max_pressure_iterations <= 100
