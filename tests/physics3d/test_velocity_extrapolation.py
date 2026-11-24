"""Narrow air-band velocity-extension regressions."""

import numpy as np
import pytest

from shoreloop.config import Grid3DConfig, Solver3DConfig
from shoreloop.device import DEVICE
from shoreloop.physics3d import (State3D, extrapolate_velocity,
                                 initialize_plane)


def _planar_state() -> State3D:
    cfg = Solver3DConfig(
        grid=Grid3DConfig(nx=8, ny=6, nz=8, Lx=1.0, Ly=0.75, Lz=1.0),
        device=DEVICE,
    )
    state = State3D(cfg)
    initialize_plane(state, (0.0, 0.0, 1.0), 0.5)
    return state


def _water_seeded_field(state: State3D, name: str, values: np.ndarray) -> np.ndarray:
    _, _, z = state.grid.coordinates(name)
    seeded = np.where(z[None, None, :] <= 0.5, values, 0.0).astype(np.float32)
    seeded = np.broadcast_to(seeded, state.grid.interior_shape(name)).copy()
    state.set_interior(name, seeded)
    return seeded


def test_constant_velocity_extends_only_the_requested_air_band():
    state = _planar_state()
    authoritative = _water_seeded_field(state, "u", np.array(0.42))
    before = state.get_interior("u").copy()
    extension = extrapolate_velocity(state, layers=2)

    extended = extension.u.numpy()[state.grid.interior_slices("u")]
    valid = extension.valid_masks[0].numpy()[state.grid.interior_slices("u")]
    np.testing.assert_array_equal(state.get_interior("u"), before)
    np.testing.assert_allclose(extended[valid != 0], 0.42,
                               rtol=0.0, atol=1.0e-7)
    assert np.count_nonzero(valid) > np.count_nonzero(authoritative)
    assert np.count_nonzero(valid == 0) > 0
    np.testing.assert_array_equal(extended[valid == 0], np.float32(0.0))
    assert extension.extended_faces[0] > 0


@pytest.mark.parametrize("order", [0, 1])
def test_planar_extension_is_width_independent_in_required_band(order):
    state = _planar_state()
    x, y, z = state.grid.coordinates("u")
    exact = (0.25 + 0.3 * x[:, None, None]
             - 0.15 * y[None, :, None]
             + 0.1 * z[None, None, :])
    authoritative = _water_seeded_field(state, "u", exact)

    narrow = extrapolate_velocity(state, layers=2, order=order)
    wide = extrapolate_velocity(state, layers=4, order=order)
    slices = state.grid.interior_slices("u")
    narrow_values = narrow.u.numpy()[slices]
    wide_values = wide.u.numpy()[slices]
    narrow_valid = narrow.valid_masks[0].numpy()[slices] != 0
    if order == 1:
        np.testing.assert_allclose(
            narrow_values[narrow_valid], exact[narrow_valid],
            rtol=0.0, atol=2.0e-6,
        )
    else:
        # Constant extension obeys a maximum principle: the air band never
        # leaves the range of the water values it is built from.
        water = np.broadcast_to(z[None, None, :] <= 0.5, authoritative.shape)
        known = authoritative[water]
        assert narrow_values[narrow_valid].min() >= known.min() - 1.0e-7
        assert narrow_values[narrow_valid].max() <= known.max() + 1.0e-7
        assert wide.u.numpy()[slices][wide.valid_masks[0].numpy()[slices] != 0].max() <= known.max() + 1.0e-7
    np.testing.assert_allclose(
        wide_values[narrow_valid], narrow_values[narrow_valid],
        rtol=0.0, atol=2.0e-6,
    )
