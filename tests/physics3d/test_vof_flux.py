"""Directional geometric liquid-flux regressions."""

import numpy as np
import pytest
import warp as wp

from shoreloop.config import BC, BC3DConfig, Grid3DConfig, Solver3DConfig
from shoreloop.device import DEVICE
from shoreloop.physics3d.polyhedron import _Planes, convex_box_moments
from shoreloop.physics3d.plic import unit_cube_volume
from shoreloop.physics3d import (State3D, compute_directional_liquid_fluxes,
                                 directional_liquid_flux,
                                 estimate_interface_normals, reconstruct_plic)


@wp.kernel
def _box_volumes(planes: wp.array2d(dtype=wp.vec3), offsets: wp.array2d(dtype=wp.float32),
                 counts: wp.array(dtype=wp.int32), volumes: wp.array(dtype=wp.float32)):
    t = wp.tid()
    p = _Planes()
    d = wp.vec4(wp.float32(0.0))
    for q in range(counts[t]):
        p[q] = planes[t, q]
        d[q] = offsets[t, q]
    volumes[t] = convex_box_moments(wp.vec3(wp.float32(0.0)), wp.vec3(wp.float32(1.0)), p, d, counts[t])[0]


def _box_volume(constraints):
    planes = np.zeros((len(constraints), 4, 3), np.float32)
    offsets = np.zeros((len(constraints), 4), np.float32)
    counts = np.zeros(len(constraints), np.int32)
    for row, halfspaces in enumerate(constraints):
        for q, (normal, offset) in enumerate(halfspaces):
            planes[row, q] = normal
            offsets[row, q] = offset
        counts[row] = len(halfspaces)
    volumes = wp.zeros(len(constraints), dtype=wp.float32, device=DEVICE)
    wp.launch(_box_volumes, dim=len(constraints), inputs=[
        wp.array(planes, dtype=wp.vec3, device=DEVICE), wp.array(offsets, dtype=wp.float32, device=DEVICE),
        wp.array(counts, dtype=wp.int32, device=DEVICE), volumes], device=DEVICE)
    return volumes.numpy()


def test_duplicate_constraint_plane_does_not_change_the_box_volume():
    """The axis-slab flux passes bitwise-identical solid planes on a translation-
    invariant beach; the FP32 clipper must treat the twin as absent."""
    rng = np.random.default_rng(20261009)
    n = 2000
    normals = rng.standard_normal((n, 3)).astype(np.float32)
    normals /= np.abs(normals).sum(axis=1, keepdims=True)
    low = np.minimum(normals, 0).sum(1)
    high = np.maximum(normals, 0).sum(1)
    offsets = (low+rng.random(n)*(high-low)).astype(np.float32)
    once = _box_volume([[(m, d)] for m, d in zip(normals, offsets)])
    twice = _box_volume([[(m, d), (m, d)] for m, d in zip(normals, offsets)])
    exact = np.array([unit_cube_volume(m.astype(float), float(d)) for m, d in zip(normals, offsets)])
    np.testing.assert_allclose(once, exact, atol=2e-6, rtol=0)
    np.testing.assert_array_equal(twice, once)
    beach = np.array([np.float32(.15789473), 0., np.float32(-.84210527)], np.float32)
    with_twin, without_twin = [], []
    for _ in range(n):
        offset = np.float32(rng.uniform(-.8, .15))
        partition = np.array([0., -1., 0.], np.float32)
        without_twin.append([(partition, np.float32(0.)), (beach, offset)])
        with_twin.append([(partition, np.float32(0.)), (beach, offset), (beach, offset)])
    np.testing.assert_array_equal(_box_volume(with_twin), _box_volume(without_twin))


@pytest.mark.parametrize("courant", [0.3, -0.4])
def test_full_and_empty_cells_reduce_to_upwind_flux(courant):
    normal = (1.0, 0.0, 0.0)
    assert directional_liquid_flux(normal, 0.5, 1.0, courant, "x") == courant
    assert directional_liquid_flux(normal, 0.5, 0.0, courant, "x") == 0.0


def test_axis_aligned_planar_slab_flux_matches_geometry_in_both_directions():
    # Water occupies x <= 0.6.  The positive 0.5 sweep intersects [0.5, 0.6],
    # while the negative 0.4 sweep is entirely water.
    assert directional_liquid_flux(
        (1.0, 0.0, 0.0), 0.6, 0.6, 0.5, 0,
    ) == pytest.approx(0.1, abs=2.0e-14)
    assert directional_liquid_flux(
        (1.0, 0.0, 0.0), 0.6, 0.6, -0.4, 0,
    ) == pytest.approx(-0.4, abs=2.0e-14)

    # Reflected plane: water occupies x >= 0.4.
    assert directional_liquid_flux(
        (-1.0, 0.0, 0.0), -0.4, 0.6, 0.3, "x",
    ) == pytest.approx(0.3, abs=2.0e-14)
    assert directional_liquid_flux(
        (-1.0, 0.0, 0.0), -0.4, 0.6, -0.6, "x",
    ) == pytest.approx(-0.2, abs=2.0e-14)


def test_gpu_directional_flux_uses_periodic_donors_and_face_courant():
    periodic_x = BC3DConfig(
        xlo=BC.PERIODIC, xhi=BC.PERIODIC,
        ylo=BC.WALL, yhi=BC.WALL,
        zlo=BC.WALL, zhi=BC.WALL,
    )
    cfg = Solver3DConfig(
        grid=Grid3DConfig(nx=4, ny=2, nz=2, Lx=4.0, Ly=2.0, Lz=2.0),
        bc=periodic_x,
        device=DEVICE,
    )
    state = State3D(cfg)
    state.set_interior("alpha", np.ones((4, 2, 2), dtype=np.float32))
    state.set_interior("u", np.full((5, 2, 2), 0.25, dtype=np.float32))
    normals = estimate_interface_normals(state)
    reconstruction = reconstruct_plic(state, normals.normals)
    result = compute_directional_liquid_fluxes(state, reconstruction, "x", 1.0)

    physical = result.fluxes.numpy()[state.grid.interior_slices("u")]
    np.testing.assert_allclose(physical, 0.25, rtol=0.0, atol=1.0e-7)
    np.testing.assert_array_equal(physical[0], physical[-1])
    assert result.max_courant == pytest.approx(0.25)
