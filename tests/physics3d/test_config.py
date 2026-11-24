from dataclasses import FrozenInstanceError

import pytest

from shoreloop.config import (Grid3DConfig, Physics3DConfig,
                              SOLVER3D_PRESET_NAMES, Solver3DConfig,
                              solver3d_preset)


def test_solver3d_presets_are_immutable_and_validate_physical_sizes():
    presets = [solver3d_preset(name) for name in SOLVER3D_PRESET_NAMES]
    assert SOLVER3D_PRESET_NAMES == ("tiny", "development", "validation")
    for cfg in presets:
        g = cfg.grid
        assert min(g.shape) > 0
        assert min(g.dx, g.dy, g.dz) > 0.0
        assert g.cell_count == g.nx * g.ny * g.nz
        assert cfg.physics.g > 0.0
        assert cfg.physics.rho > 0.0

    development, validation = presets[1:]
    assert (development.grid.Lx, development.grid.Ly, development.grid.Lz) == \
        (validation.grid.Lx, validation.grid.Ly, validation.grid.Lz)
    assert validation.grid.dx == pytest.approx(0.5 * development.grid.dx)
    with pytest.raises(FrozenInstanceError):
        presets[0].grid.nx = 1

    with pytest.raises(ValueError, match="nx"):
        Grid3DConfig(nx=0)
    with pytest.raises(ValueError, match="nz"):
        Grid3DConfig(nz=-1)
    with pytest.raises(ValueError, match="ng"):
        Grid3DConfig(ng=0)
    with pytest.raises(ValueError, match="smallest"):
        Grid3DConfig(nx=1, ng=2)
    with pytest.raises(ValueError, match="Ly"):
        Grid3DConfig(Ly=0.0)
    with pytest.raises(ValueError, match="Lz"):
        Grid3DConfig(Lz=float("nan"))
    with pytest.raises(ValueError, match="g"):
        Physics3DConfig(g=-1.0)
    with pytest.raises(ValueError, match="rho"):
        Physics3DConfig(rho=0.0)
    with pytest.raises(ValueError, match="surface_tension"):
        Physics3DConfig(surface_tension=-0.01)
    with pytest.raises(ValueError, match="CUDA"):
        Solver3DConfig(device="cpu")
    with pytest.raises(ValueError, match="unknown"):
        solver3d_preset("hero")
