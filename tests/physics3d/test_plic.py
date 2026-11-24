"""Geometric unit-cube PLIC reconstruction gates."""

import numpy as np
import pytest
import warp as wp

from shoreloop.config import Grid3DConfig, Solver3DConfig
from shoreloop.device import DEVICE
from shoreloop.physics3d import (State3D, estimate_interface_normals,
                                 initialize_plane, reconstruct_plic,
                                 solve_plic_offset, unit_cube_volume)
from shoreloop.physics3d.plic import _positive_cube_volume


@wp.kernel
def _evaluate_cube_volume(
    coefficients: wp.vec3,
    offset: wp.float32,
    result: wp.array(dtype=wp.float32),
):
    result[0] = _positive_cube_volume(coefficients, offset)


@pytest.mark.parametrize("normal", [
    (1.0, 0.0, 0.0),
    (0.3, 0.7, 0.0),
    (0.2, 0.3, 0.5),
    (-0.2, 0.3, -0.5),
])
@pytest.mark.parametrize("fraction", [1.0e-4, 0.05, 0.5, 0.93, 0.9999])
def test_plic_offset_reconstructs_requested_unit_cube_fraction(normal, fraction):
    offset = solve_plic_offset(normal, fraction)
    reconstructed = unit_cube_volume(normal, offset)
    assert reconstructed == pytest.approx(fraction, abs=2.0e-10)


def test_gpu_reconstruction_handles_mixed_cells_only_and_is_deterministic():
    cfg = Solver3DConfig(
        grid=Grid3DConfig(nx=12, ny=10, nz=8,
                          Lx=1.2, Ly=0.8, Lz=0.6),
        device=DEVICE,
    )
    state = State3D(cfg)
    initialize_plane(state, (0.4, -0.3, 0.8), 0.31, samples_per_axis=8)
    normals = estimate_interface_normals(state)
    first = reconstruct_plic(state, normals.normals)
    second = reconstruct_plic(state, normals.normals)

    alpha = state.get_interior("alpha")
    mixed = (alpha > 1.0e-6) & (alpha < 1.0 - 1.0e-6)
    assert first.mixed_cells == int(np.count_nonzero(mixed))
    assert first.max_volume_error < 2.0e-6
    assert second.max_volume_error < 2.0e-6
    np.testing.assert_array_equal(first.offsets.numpy(), second.offsets.numpy())
    np.testing.assert_array_equal(first.normals.numpy(), second.normals.numpy())
    np.testing.assert_array_equal(
        first.offsets.numpy()[state.grid.interior_slices("cell")][~mixed],
        np.float32(0.0),
    )


@pytest.mark.parametrize("coefficients", [
    (2.26604856e-8, 0.128672174, 0.871327803),
    (0.128672174, 2.26604856e-8, 0.871327803),
    (0.128672174, 0.871327803, 2.26604856e-8),
])
def test_gpu_cube_volume_is_accurate_for_nearly_two_dimensional_planes(
    coefficients,
):
    offset = 0.46052731312667416
    result = wp.zeros(1, dtype=wp.float32, device=DEVICE)

    wp.launch(
        _evaluate_cube_volume,
        dim=1,
        inputs=[wp.vec3(*coefficients), offset, result],
        device=DEVICE,
    )

    expected = unit_cube_volume(coefficients, offset)
    assert float(result.numpy()[0]) == pytest.approx(expected, abs=1.0e-6)


@pytest.mark.parametrize("offset", [2e-6, 1e-5, 2.5e-5, .4])
def test_nearly_horizontal_plane_retains_components_larger_than_film_depth(offset):
    # Fresh production wave input exposed this orientation at (61,30,11).
    # Its small coefficients exceed a retained film's depth, so treating
    # the plane as one-dimensional loses almost all of its true volume.
    coefficients = np.asarray((4.547205662674969e-6, 2.146575934602879e-5,
                               .9999740123748779), dtype=np.float32)
    # Integrate along z: an interior linear column or analytic (truncated)
    # tetrahedra, evaluated directly at the rounded input offset.
    x = float(np.float32(offset))
    a, b, c = coefficients.astype(float)
    if x >= a+b:
        expected = (x-.5*(a+b))/c
    else:
        expected = (x**3-max(x-a, 0)**3-max(x-b, 0)**3)/(6*a*b*c)
    result = wp.zeros(1, dtype=wp.float32, device=DEVICE)
    wp.launch(_evaluate_cube_volume, dim=1,
              inputs=[wp.vec3(*coefficients), x, result], device=DEVICE)
    assert float(result.numpy()[0]) == pytest.approx(expected, rel=3e-7)
