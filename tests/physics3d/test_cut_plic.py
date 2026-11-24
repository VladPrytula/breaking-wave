"""Joint liquid/solid volume, location, and swept-volume properties."""

import numpy as np
import pytest
import warp as wp

from shoreloop.config import BC, BC3DConfig, Grid3DConfig, Solver3DConfig
from shoreloop.device import DEVICE, prepare_device
from shoreloop.physics3d.cut_cells import initialize_scene_cut_cells
from shoreloop.physics3d.advection import VOFAdvector
from shoreloop.physics3d.interface import estimate_interface_normals
from shoreloop.physics3d.plic import reconstruct_plic
from shoreloop.physics3d.polyhedron import clipped_box_moments
from shoreloop.physics3d.state import State3D
from shoreloop.physics3d.vof_flux import compute_directional_liquid_fluxes
from shoreloop.physics3d.vof import initialize_joint_plane, measure_vof
from shoreloop.scene.geometry3d import AnalyticBeach, SceneGeometry


@wp.kernel
def _moments(
    solid_normals: wp.array(dtype=wp.vec3),
    solid_offsets: wp.array(dtype=wp.float32),
    water_normals: wp.array(dtype=wp.vec3),
    water_offsets: wp.array(dtype=wp.float32),
    lower: wp.vec3,
    upper: wp.vec3,
    out: wp.array(dtype=wp.vec4),
):
    i = wp.tid()
    out[i] = clipped_box_moments(
        lower, upper, solid_normals[i], solid_offsets[i],
        water_normals[i], water_offsets[i],
    )


def _evaluate(solids, solid_offsets, water, water_offsets, lower=(0, 0, 0),
              upper=(1, 1, 1)):
    prepare_device(DEVICE)
    result = wp.zeros(len(solids), dtype=wp.vec4, device=DEVICE)
    wp.launch(
        _moments, dim=len(solids),
        inputs=[wp.array(solids, dtype=wp.vec3, device=DEVICE),
                wp.array(solid_offsets, dtype=wp.float32, device=DEVICE),
                wp.array(water, dtype=wp.vec3, device=DEVICE),
                wp.array(water_offsets, dtype=wp.float32, device=DEVICE),
                wp.vec3(*lower), wp.vec3(*upper), result],
        device=DEVICE,
    )
    return result.numpy().astype(np.float64)


def test_joint_thin_sheet_volume_centroid_and_empty_intersection():
    # Bed at z=.25, liquid top at z=.25+h. Fractions describe the actual
    # intersection, including films too thin for subcell point quadrature.
    depths = np.asarray((0.0, 1e-5, .01, .49, .51, .75), dtype=np.float32)
    top = np.float32(.25) + depths
    result = _evaluate(
        [(0, 0, -1)] * len(depths), [-.25] * len(depths),
        [(0, 0, 1)] * len(depths), top,
    )
    exact_depth = top.astype(float) - .25
    np.testing.assert_allclose(result[:, 0], exact_depth, rtol=2e-5, atol=2e-8)
    wet = exact_depth > 0
    centroids = result[wet, 1:] / result[wet, :1]
    expected = np.column_stack((np.full(wet.sum(), .5), np.full(wet.sum(), .5),
                                .25 + .5 * exact_depth[wet]))
    np.testing.assert_allclose(centroids, expected, atol=2e-6)
    empty = _evaluate([(0, 0, -1)], [-.75], [(0, 0, 1)], [.25])
    np.testing.assert_array_equal(empty, 0.0)


def test_oblique_joint_geometry_partitions_without_volume_or_moment_loss():
    solids = np.asarray(((.4, 0, -.6), (-.3, .2, -.5), (.2, -.7, .1)), np.float32)
    solid_offsets = np.asarray((-.1, -.2, -.1), np.float32)
    water = np.asarray(((.1, .2, .7), (-.2, .1, .7), (.6, .3, -.1)), np.float32)
    water_offsets = np.asarray((.4, .3, .25), np.float32)
    whole = _evaluate(solids, solid_offsets, water, water_offsets)
    assert np.all(whole[:, 0] > 0)
    for axis in range(3):
        upper = np.ones(3); upper[axis] = .37
        lower = np.zeros(3); lower[axis] = .37
        pieces = (_evaluate(solids, solid_offsets, water, water_offsets, upper=upper)
                  + _evaluate(solids, solid_offsets, water, water_offsets, lower=lower))
        np.testing.assert_allclose(pieces, whole, rtol=4e-6, atol=3e-7)
    centroid = whole[:, 1:] / whole[:, :1]
    assert np.all(np.einsum("ij,ij->i", solids, centroid) <= solid_offsets + 2e-6)
    assert np.all(np.einsum("ij,ij->i", water, centroid) <= water_offsets + 2e-6)


@pytest.mark.parametrize("courant", [-.2, -1e-8, 1e-8, .02, .2])
def test_cut_plic_matches_retained_volume_location_and_face_transport(courant):
    grid = Grid3DConfig(nx=4, ny=4, nz=4, Lx=.4, Ly=.4, Lz=.4)
    cfg = Solver3DConfig(grid=grid, device=DEVICE, bc=BC3DConfig(
        xlo=BC.PERIODIC, xhi=BC.PERIODIC,
        ylo=BC.PERIODIC, yhi=BC.PERIODIC,
    ))
    state = State3D(cfg)
    # Four-point volume quadrature exactly gives .75 open at this flat bed.
    cut = initialize_scene_cut_cells(
        state, SceneGeometry(AnalyticBeach(1, 0, .025)),
    )
    normals = np.zeros((*state.grid.cell_shape, 3), np.float32)
    normals[..., 2] = 1
    normals = wp.array(normals, dtype=wp.vec3, device=DEVICE)
    alpha = np.zeros(grid.shape, np.float32)
    alpha[:, :, 0] = .49
    state.set_interior("alpha", alpha)
    state.fill_("u", courant * grid.dx / .01)
    before = state.get_interior("alpha").copy()
    plic = reconstruct_plic(state, normals)
    cell_slices = state.grid.interior_slices("cell")
    offsets = plic.offsets.numpy()[cell_slices]
    centroids = plic.centroids.numpy()[cell_slices]
    # h=alpha*theta, and both the water centroid and plane are above the bed.
    expected_top = .25 + .49 * .75
    np.testing.assert_allclose(offsets[:, :, 0], expected_top, atol=2e-6)
    np.testing.assert_allclose(centroids[:, :, 0, 2], .25 + .5 * .49 * .75, atol=2e-6)
    assert plic.max_volume_error < 3e-6
    flux = compute_directional_liquid_fluxes(state, plic, "x", .01)
    values = flux.fluxes.numpy()[state.grid.interior_slices("u")]
    np.testing.assert_allclose(values[:, :, 0], .49 * .75 * courant, rtol=3e-6, atol=1e-14)
    np.testing.assert_array_equal(values[0], values[-1])
    np.testing.assert_array_equal(before, state.get_interior("alpha"))
    np.testing.assert_allclose(state.get_geometry_interior("theta"), cut.theta)
    assert cut.max_plane_displacement < 1e-8


def test_cached_build_after_a_change_matches_a_fresh_cold_fit():
    # The second build of a run reuses cached normals where the stencil is
    # unchanged and searches the rest; after the water moves the result must
    # equal a search on a fresh cache.
    from shoreloop.physics3d.cut_normals import CutNormalCache
    grid = Grid3DConfig(nx=12, ny=4, nz=8, Lx=.15, Ly=.05, Lz=.08)
    state = State3D(Solver3DConfig(grid=grid, device=DEVICE, interface_reconstruction="youngs",
                                   bc=BC3DConfig(ylo=BC.WALL, yhi=BC.WALL)))
    initialize_scene_cut_cells(state, SceneGeometry(AnalyticBeach(-.1, .1, -.003)))
    initialize_joint_plane(state, (.1, .07, 1), .0273)
    estimate_interface_normals(state)  # cold first build fills the cache
    initialize_joint_plane(state, (.12, .05, 1), .0281)  # the sheet tilts and rises
    cached = estimate_interface_normals(state).normals.numpy()[state.grid.interior_slices("cell")]
    assert state.cut_normal_cache.misses > 0
    state.cut_normal_cache = CutNormalCache(grid.shape, DEVICE)
    cold = estimate_interface_normals(state).normals.numpy()[state.grid.interior_slices("cell")]
    alpha = state.get_interior("alpha")
    theta = state.get_geometry_interior("theta")
    mixed_cut = (alpha > 1e-6) & (alpha < 1-1e-6) & (theta > 0) & (theta < 1)
    assert mixed_cut.sum() > 10
    np.testing.assert_array_equal(cached[mixed_cut], cold[mixed_cut])
    expected = np.asarray((.12, .05, 1)) / np.linalg.norm((.12, .05, 1))
    np.testing.assert_allclose(cold[mixed_cut], np.broadcast_to(expected, (mixed_cut.sum(), 3)), atol=3e-3)


@pytest.mark.parametrize("water_normal", [(0, 0, 1), (.1, .07, 1)])
@pytest.mark.parametrize("method", ["youngs", "height_function"])
def test_cut_normal_fit_recovers_planar_water_from_retained_volumes(water_normal, method):
    grid = Grid3DConfig(nx=12, ny=4, nz=8, Lx=.15, Ly=.05, Lz=.08)
    state = State3D(Solver3DConfig(grid=grid, device=DEVICE, interface_reconstruction=method, bc=BC3DConfig(
        ylo=BC.WALL, yhi=BC.WALL,
    )))
    initialize_scene_cut_cells(state, SceneGeometry(AnalyticBeach(-.1, .1, -.003)))
    initialize_joint_plane(state, water_normal, .0273)
    alpha = state.get_interior("alpha")
    theta = state.get_geometry_interior("theta")
    recovered = estimate_interface_normals(state).normals
    mixed_cut = (alpha > 1e-6) & (alpha < 1-1e-6) & (theta > 0) & (theta < 1)
    assert np.any(mixed_cut)
    expected = np.asarray(water_normal) / np.linalg.norm(water_normal)
    np.testing.assert_allclose(recovered.numpy()[state.grid.interior_slices("cell")][mixed_cut],
                               np.broadcast_to(expected, (mixed_cut.sum(), 3)), atol=3e-3)
    plic = reconstruct_plic(state, recovered)
    assert plic.max_volume_error < 5e-6
    np.testing.assert_array_equal(alpha, state.get_interior("alpha"))


def test_cut_water_translation_uses_joint_geometry_conservatively():
    grid = Grid3DConfig(nx=8, ny=12, nz=8, Lx=.1, Ly=.15, Lz=.08)
    state = State3D(Solver3DConfig(grid=grid, device=DEVICE))
    initialize_scene_cut_cells(state, SceneGeometry(AnalyticBeach(-.1, .1, -.003)))
    initialize_joint_plane(state, (0, 0, 1), .0223)
    alpha = state.get_interior("alpha")
    alpha[:, :3, :] = 0
    alpha[:, 9:, :] = 0
    state.set_interior("alpha", alpha)
    state.fill_("v", .2)
    initial = measure_vof(state)
    advector = VOFAdvector(state.cfg)
    for _ in range(3):
        result = advector.advect(state, .2 * grid.dy / .2)
    final = measure_vof(state)
    assert 0 <= final.alpha_min <= final.alpha_max <= 1
    assert abs(final.volume - initial.volume) < 2e-7 * initial.volume
    assert abs(result.clipped_volume) < 2e-7 * initial.volume
    assert np.sum(state.get_interior("alpha")[:, 9:, :]) > 0
