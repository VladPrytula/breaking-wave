import numpy as np
import pytest

from shoreloop.config import Grid3DConfig, Solver3DConfig
from shoreloop.device import DEVICE
from shoreloop.physics3d.state import State3D
from shoreloop.physics3d.wet_dry import WispRemover


def test_wisp_removal_is_limited_to_disconnected_sub_tolerance_water():
    cfg = Solver3DConfig(
        grid=Grid3DConfig(nx=4, ny=4, nz=4,
                          Lx=0.4, Ly=0.4, Lz=0.4),
        device=DEVICE,
    )
    state = State3D(cfg)
    alpha = np.zeros(cfg.grid.shape, dtype=np.float32)
    alpha[1, 1, 1] = 5.0e-7
    alpha[2, 2, 2] = 5.0e-7
    alpha[2, 2, 1] = 0.2
    state.set_interior("alpha", alpha)

    result = WispRemover(cfg, tolerance=1.0e-6).apply(state)
    cleaned = state.get_interior("alpha")
    assert cleaned[1, 1, 1] == 0.0
    assert cleaned[2, 2, 2] == alpha[2, 2, 2]
    assert result.removed_cells == 1
    assert result.removed_volume == pytest.approx(
        5.0e-7 * cfg.grid.dx * cfg.grid.dy * cfg.grid.dz,
    )
