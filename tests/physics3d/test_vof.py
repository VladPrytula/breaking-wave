"""Analytic initialisation and volume diagnostics for the VOF field."""

import math

import numpy as np
import pytest

from shoreloop.config import Grid3DConfig, Solver3DConfig
from shoreloop.device import DEVICE
from shoreloop.physics3d import (State3D, initialize_box, initialize_plane,
                                 initialize_sphere, measure_vof)


def _state(n: int) -> State3D:
    return State3D(Solver3DConfig(
        grid=Grid3DConfig(nx=n, ny=n, nz=n, Lx=1.0, Ly=1.0, Lz=1.0),
        device=DEVICE,
    ))


def test_full_box_has_exact_volume_and_classification():
    state = _state(9)
    initialize_box(state, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    diagnostics = measure_vof(state, clipped_volume=2.5e-8)

    assert diagnostics.volume == pytest.approx(1.0, abs=2.0e-15)
    assert diagnostics.alpha_min == 1.0
    assert diagnostics.alpha_max == 1.0
    assert diagnostics.clipped_volume == 2.5e-8
    assert diagnostics.air_cells == 0
    assert diagnostics.mixed_cells == 0
    assert diagnostics.water_cells == 9 ** 3

    initialize_plane(state, (0.0, 0.0, 2.0), 1.0)
    plane = measure_vof(state)
    assert plane.volume == pytest.approx(0.5, abs=2.0e-7)
    assert plane.air_cells + plane.mixed_cells + plane.water_cells == 9 ** 3


def test_sampled_sphere_volume_converges_and_stays_bounded():
    exact = 4.0 * math.pi * 0.27 ** 3 / 3.0
    errors = []
    for n in (10, 20):
        state = _state(n)
        initialize_sphere(state, (0.5, 0.5, 0.5), 0.27)
        diagnostics = measure_vof(state)
        errors.append(abs(diagnostics.volume - exact))
        assert 0.0 <= diagnostics.alpha_min <= diagnostics.alpha_max <= 1.0
        assert diagnostics.air_cells > 0
        assert diagnostics.mixed_cells > 0
        assert diagnostics.water_cells > 0

    assert errors[1] < errors[0]
    assert errors[1] / exact < 5.0e-3

    with pytest.raises(ValueError, match="radius"):
        initialize_sphere(_state(4), (0.5, 0.5, 0.5), np.nan)
