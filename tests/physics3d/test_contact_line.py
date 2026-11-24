"""Advancing/receding contact-band regularisation gates."""

import numpy as np

from shoreloop.config import Grid3DConfig, Solver3DConfig
from shoreloop.device import DEVICE
from shoreloop.physics3d.contact_line import ContactLineConfig, ContactLineModel
from shoreloop.physics3d.cut_cells import initialize_scene_cut_cells
from shoreloop.physics3d.state import State3D
from shoreloop.scene.geometry3d import AnalyticBeach, SceneGeometry


def test_contact_band_smooths_motion_conservatively_without_a_dry_film():
    grid = Grid3DConfig(nx=20, ny=4, nz=6, Lx=1.0, Ly=0.2, Lz=0.3)
    cfg = Solver3DConfig(grid=grid, device=DEVICE)
    state = State3D(cfg)
    initialize_scene_cut_cells(
        state,
        SceneGeometry(AnalyticBeach(2.0, 0.0, base_elevation=0.05)),
    )
    alpha = np.zeros(grid.shape, dtype=np.float32)
    alpha[:8, :, 1:3] = 1.0
    alpha[8:11, :, 1] = np.asarray((0.8, 0.5, 0.2))[:, None]
    state.set_interior("alpha", alpha)
    model = ContactLineModel(
        cfg,
        ContactLineConfig(
            contact_length=0.06,
            advancing_diffusivity=1.0e-4,
            receding_diffusivity=5.0e-5,
            hysteresis_fraction=0.005,
        ),
    )
    model.prime(state)

    moved = alpha.copy()
    moved[8:11, :, 1] = np.asarray((0.7, 0.55, 0.3))[:, None]
    state.set_interior("alpha", moved)
    before_mass = float(np.sum(
        moved * state.get_geometry_interior("theta"), dtype=np.float64,
    ))
    result = model.apply(state, 0.01)
    after = state.get_interior("alpha")
    after_mass = float(np.sum(
        after * state.get_geometry_interior("theta"), dtype=np.float64,
    ))

    assert result.active_cells > 0
    assert result.exchanged_volume > 0.0
    assert abs(result.volume_residual) < 2.0e-11
    np.testing.assert_allclose(after_mass, before_mass, atol=2.0e-6)
    assert 0.0 <= float(after.min()) <= float(after.max()) <= 1.0
    np.testing.assert_array_equal(after[12:], 0.0)


def _resolved_contact_exchange(nx: int):
    grid = Grid3DConfig(nx=nx, ny=4, nz=6, Lx=1.0, Ly=0.2, Lz=0.3)
    cfg = Solver3DConfig(grid=grid, device=DEVICE)
    state = State3D(cfg)
    initialize_scene_cut_cells(
        state,
        SceneGeometry(AnalyticBeach(2.0, 0.0, base_elevation=0.05)),
    )
    x, _, _ = state.grid.coordinates("cell")
    previous = np.zeros(grid.shape, dtype=np.float32)
    previous[:, :, 1] = np.clip((0.55 - x[:, None]) / 0.15, 0.0, 1.0)
    state.set_interior("alpha", previous)
    model = ContactLineModel(
        cfg,
        ContactLineConfig(
            contact_length=0.06,
            advancing_diffusivity=1.0e-4,
            receding_diffusivity=5.0e-5,
            hysteresis_fraction=0.002,
        ),
    )
    model.prime(state)
    advanced = np.zeros_like(previous)
    advanced[:, :, 1] = np.clip((0.565 - x[:, None]) / 0.15, 0.0, 1.0)
    state.set_interior("alpha", advanced)
    return model.apply(state, 0.01)


def test_contact_exchange_has_a_continuous_two_resolution_trend():
    coarse = _resolved_contact_exchange(20)
    fine = _resolved_contact_exchange(40)
    assert abs(coarse.volume_residual) < 1.0e-12
    assert abs(fine.volume_residual) < 1.0e-12
    np.testing.assert_allclose(
        fine.exchanged_volume, coarse.exchanged_volume, rtol=5.0e-6,
    )
    assert fine.active_cells == 2 * coarse.active_cells
