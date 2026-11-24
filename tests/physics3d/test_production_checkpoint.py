import numpy as np
import pytest

from shoreloop.config import Grid3DConfig, Solver3DConfig
from shoreloop.device import DEVICE
from shoreloop.physics3d.contact_line import ContactLineConfig
from shoreloop.physics3d.energy import EnergyLedger
from shoreloop.physics3d.production_checkpoint import (
    load_production_checkpoint,
    write_production_checkpoint,
)
from shoreloop.physics3d.solver import Solver3D


def _solver() -> Solver3D:
    cfg = Solver3DConfig(
        grid=Grid3DConfig(nx=4, ny=3, nz=2, Lx=0.4, Ly=0.3, Lz=0.2),
        device=DEVICE,
    )
    solver = Solver3D(cfg, contact_line_settings=ContactLineConfig())
    solver.set_energy_ledger(EnergyLedger(cfg))
    return solver


def test_production_checkpoint_restores_complete_carried_state(tmp_path):
    source = _solver()
    expected = {}
    for index, name in enumerate(source.state.FIELDS, start=1):
        values = np.full(
            source.state.field(name).shape,
            0.01 * index,
            dtype=np.float32,
        )
        source.state.field(name).assign(values)
        expected[name] = values
    previous_alpha = np.full(
        source.contact_line.previous_alpha.shape,
        0.37,
        dtype=np.float32,
    )
    source.contact_line.previous_alpha.assign(previous_alpha)
    source.contact_line.primed = True
    source.time = 12.0
    source.steps = 346
    source.advection.vof.steps = source.steps
    energy = (
        "molecular_loss",
        "sgs_loss",
        "wall_loss",
        "porous_loss",
        "numerical_residual",
        "positive_numerical_loss",
        "boundary_work",
        "boundary_surface_work",
    )
    for index, name in enumerate(energy, start=1):
        setattr(source.energy_ledger, name, 0.25 * index)
    source.energy_ledger.steps = 344

    path = write_production_checkpoint(source, tmp_path / "state.npz")
    restored = _solver()
    load_production_checkpoint(restored, path)

    for name, values in expected.items():
        np.testing.assert_array_equal(restored.state.field(name).numpy(), values)
    np.testing.assert_array_equal(
        restored.contact_line.previous_alpha.numpy(),
        previous_alpha,
    )
    assert restored.contact_line.primed
    assert restored.time == 12.0
    assert restored.steps == 346
    assert restored.advection.vof.steps == source.advection.vof.steps
    for name in energy:
        assert getattr(restored.energy_ledger, name) == getattr(
            source.energy_ledger,
            name,
        )
    assert restored.energy_ledger.steps == 344

    source_advection = source.advection.vof.advect(source.state, 1.0e-3)
    restored_advection = restored.advection.vof.advect(restored.state, 1.0e-3)
    assert restored_advection.sweep_order == source_advection.sweep_order
    for name in ("alpha", "u", "v", "w"):
        np.testing.assert_array_equal(
            restored.state.field(name).numpy(),
            source.state.field(name).numpy(),
        )


def test_interrupted_checkpoint_write_preserves_previous_complete_state(tmp_path, monkeypatch):
    source = _solver()
    source.time = .04
    source.steps = 8
    path = write_production_checkpoint(source, tmp_path / "latest_state.npz")
    before = path.read_bytes()
    source.time = .08

    def interrupted(stream, **payload):
        stream.write(b"incomplete checkpoint")
        raise OSError("injected write interruption")

    monkeypatch.setattr(np, "savez", interrupted)
    with pytest.raises(OSError, match="injected write interruption"):
        write_production_checkpoint(source, path)
    assert path.read_bytes() == before
    restored = _solver()
    load_production_checkpoint(restored, path)
    assert restored.time == .04
    assert restored.steps == 8
