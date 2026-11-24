"""Planar-interface normal reconstruction regressions."""

import numpy as np
import pytest

from shoreloop.config import Grid3DConfig, Solver3DConfig
from shoreloop.device import DEVICE
from shoreloop.physics3d import (State3D, estimate_interface_normals,
                                 initialize_plane)


def _planar_error(n: int, normal: tuple[float, float, float]) -> tuple[float, int]:
    cfg = Solver3DConfig(
        grid=Grid3DConfig(nx=n, ny=n, nz=n, Lx=1.0, Ly=1.0, Lz=1.0),
        device=DEVICE,
    )
    state = State3D(cfg)
    expected = np.asarray(normal, dtype=np.float64)
    expected /= np.linalg.norm(expected)
    offset = float(np.dot(expected, (0.47, 0.53, 0.49)))
    initialize_plane(state, expected, offset, samples_per_axis=8)
    result = estimate_interface_normals(state)

    alpha = state.get_interior("alpha")
    normals = result.normals.numpy()[state.grid.interior_slices("cell")]
    mixed = (alpha > 1.0e-6) & (alpha < 1.0 - 1.0e-6)
    reconstructed = normals[mixed].astype(np.float64)
    cosine = np.clip(reconstructed @ expected, -1.0, 1.0)
    angular_rms = float(np.sqrt(np.mean(np.arccos(cosine) ** 2)))
    assert result.mixed_cells == int(np.count_nonzero(mixed))
    return angular_rms, result.ambiguous_cells


@pytest.mark.parametrize("normal", [
    (1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 1.7, 0.8),
])
def test_planar_normals_are_water_to_air_and_improve_under_refinement(normal):
    coarse, coarse_ambiguous = _planar_error(12, normal)
    fine, fine_ambiguous = _planar_error(24, normal)

    assert fine < 0.20
    if np.count_nonzero(normal) > 1:
        assert fine < coarse
    else:
        assert max(coarse, fine) < 2.0e-4
    assert coarse_ambiguous == 0
    assert fine_ambiguous == 0
