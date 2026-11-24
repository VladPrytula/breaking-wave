"""Finite porous storage and Darcy-Forchheimer dissipation gates."""

import numpy as np
import pytest

from shoreloop.config import BC, BC3DConfig, Grid3DConfig, Solver3DConfig
from shoreloop.device import DEVICE
from shoreloop.physics3d.porous import (PorousBedModel, PorousDragConfig,
                                        PorousStorageConfig)
from shoreloop.physics3d.state import State3D


def _state() -> tuple[Solver3DConfig, State3D, np.ndarray]:
    periodic = BC3DConfig(
        xlo=BC.PERIODIC, xhi=BC.PERIODIC,
        ylo=BC.PERIODIC, yhi=BC.PERIODIC,
        zlo=BC.PERIODIC, zhi=BC.PERIODIC,
    )
    grid = Grid3DConfig(nx=6, ny=5, nz=4, Lx=0.6, Ly=0.5, Lz=0.4)
    cfg = Solver3DConfig(grid=grid, bc=periodic, device=DEVICE)
    state = State3D(cfg)
    capacity = np.full(grid.shape, 0.2, dtype=np.float32)
    return cfg, state, capacity


def test_porous_patch_relaxes_to_contact_equilibrium_and_drains_when_exposed():
    cfg, state, capacity = _state()
    state.fill_("alpha", 0.5)
    settings = PorousStorageConfig(
        layer_depth=0.1, capacity_fraction=0.2,
        infiltration_rate=100.0, drainage_time=1.0,
    )
    model = PorousBedModel(
        cfg, capacity, storage_settings=settings,
    )
    cell_volume = cfg.grid.dx * cfg.grid.dy * cfg.grid.dz
    water_before = float(np.sum(
        state.get_interior("alpha"), dtype=np.float64,
    )) * cell_volume
    filled = None
    for _ in range(8):
        filled = model.apply_storage(state, 0.1)
    assert filled is not None
    water_after = float(np.sum(
        state.get_interior("alpha"), dtype=np.float64,
    )) * cell_volume
    assert filled.infiltrated_volume > 0.0
    np.testing.assert_allclose(
        water_before - water_after,
        filled.stored_volume,
        rtol=2.0e-6, atol=2.0e-9,
    )
    assert abs(filled.mass_residual) < 2.0e-11
    np.testing.assert_allclose(
        state.get_interior("porous_storage"), capacity, rtol=2.0e-6,
    )

    equilibrium_volume = model.initialize_storage_equilibrium(state)
    saturated = model.apply_storage(state, 1.0)
    assert saturated.infiltrated_volume == 0.0
    assert saturated.drained_volume == 0.0
    assert saturated.stored_volume == pytest.approx(equilibrium_volume)

    state.zero_("alpha")
    drained = model.apply_storage(state, 1.0)
    np.testing.assert_allclose(
        state.get_interior("porous_storage"), 0.1, rtol=2.0e-6,
    )
    np.testing.assert_allclose(
        drained.drained_volume, 0.5 * 0.2 * cfg.grid.Lx * cfg.grid.Ly * cfg.grid.Lz,
        rtol=2.0e-6,
    )


def test_porous_drag_is_dissipative_and_orders_with_inverse_permeability():
    final_speeds = []
    losses = []
    for permeability in (2.0e-6, 2.0e-7):
        cfg, state, capacity = _state()
        state.fill_("alpha", 1.0)
        state.fill_("u", 0.8)
        model = PorousBedModel(
            cfg, capacity,
            drag_settings=PorousDragConfig(
                permeability=permeability,
                forchheimer_coefficient=0.2,
            ),
        )
        result = model.apply_drag(state, 0.002)
        final_speeds.append(float(np.mean(state.get_interior("u"))))
        losses.append(result.energy_loss)
        assert result.energy_loss > 0.0
        assert float(np.min(model.dissipation.numpy())) >= 0.0
    assert final_speeds[1] < final_speeds[0] < 0.8
    assert losses[1] > losses[0]


def test_porous_bed_rejects_an_empty_capacity_mask():
    cfg, _, capacity = _state()
    capacity.fill(0.0)
    with pytest.raises(ValueError, match="positive cell"):
        PorousBedModel(
            cfg, capacity,
            drag_settings=PorousDragConfig(),
        )
