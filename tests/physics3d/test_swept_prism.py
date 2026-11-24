"""Geometric translation accuracy of the full-vector unsplit flux."""

import numpy as np
import pytest
import warp as wp

from shoreloop.config import BC, BC3DConfig, Grid3DConfig, Solver3DConfig
from shoreloop.device import DEVICE
from shoreloop.physics3d.plic import unit_cube_volume
from shoreloop.physics3d.momentum import MomentumAdvector, compute_face_liquid_mass
from shoreloop.physics3d.extrapolation import extrapolate_velocity
from shoreloop.physics3d.state import State3D
from shoreloop.physics3d.swept_prism import _TraceOffsets, _TracePlanes, _volume
from shoreloop.physics3d.unsplit_advection import UnsplitVOFAdvector


@wp.kernel
def _prism_volumes(planes: wp.array2d(dtype=wp.vec3d), offsets: wp.array2d(dtype=wp.float64),
                   counts: wp.array(dtype=wp.int32), volumes: wp.array(dtype=wp.float64)):
    t = wp.tid()
    p = _TracePlanes()
    d = _TraceOffsets()
    for q in range(counts[t]):
        p[q] = planes[t, q]
        d[q] = offsets[t, q]
    volumes[t] = _volume(p, d, counts[t])


def _prism_volume(constraints):
    """Traced-prism unit-cube volume for each list of (normal, offset) halfspaces."""
    planes = np.zeros((len(constraints), 6, 3))
    offsets = np.zeros((len(constraints), 6))
    counts = np.zeros(len(constraints), np.int32)
    for row, halfspaces in enumerate(constraints):
        for q, (normal, offset) in enumerate(halfspaces):
            planes[row, q] = normal
            offsets[row, q] = offset
        counts[row] = len(halfspaces)
    volumes = wp.zeros(len(constraints), dtype=wp.float64, device=DEVICE)
    wp.launch(_prism_volumes, dim=len(constraints), inputs=[
        wp.array(planes, dtype=wp.vec3d, device=DEVICE),
        wp.array(offsets, dtype=wp.float64, device=DEVICE),
        wp.array(counts, dtype=wp.int32, device=DEVICE), volumes], device=DEVICE)
    return volumes.numpy()


def test_duplicate_constraint_plane_does_not_change_the_prism_volume():
    """A plane passed twice must clip exactly as once: the production y-faces of a
    translation-invariant beach carry bitwise-identical low/high common solids."""
    rng = np.random.default_rng(20261009)
    n = 2000
    normals = rng.standard_normal((n, 3)).astype(np.float32).astype(np.float64)
    normals /= np.abs(normals).sum(axis=1, keepdims=True)
    normals = normals.astype(np.float32).astype(np.float64)
    low = np.minimum(normals, 0).sum(1)
    high = np.maximum(normals, 0).sum(1)
    offsets = (low+rng.random(n)*(high-low)).astype(np.float32).astype(np.float64)
    once = _prism_volume([[(m, d)] for m, d in zip(normals, offsets)])
    twice = _prism_volume([[(m, d), (m, d)] for m, d in zip(normals, offsets)])
    exact = np.array([unit_cube_volume(m, d) for m, d in zip(normals, offsets)])
    np.testing.assert_allclose(once, exact, atol=1e-6, rtol=0)
    np.testing.assert_array_equal(twice, once)
    # Production structure of a beach y-face: two partition planes with small
    # transverse Courants, a donor solid nearly parallel to the common solids,
    # and the common solid itself once or twice.
    beach = np.array([np.float32(.15789473), 0., np.float32(-.84210527)])
    with_twin, without_twin = [], []
    for _ in range(n):
        cx, cz = rng.uniform(-5e-3, 5e-3, 2)
        halfspaces = []
        for axis, courant in ((2, cz), (0, cx)):
            normal = np.zeros(3)
            normal[1] = courant
            normal[axis] = -1.
            halfspaces.append((normal, 0.) if courant >= 0 else (-normal, 1.))
        donor = beach.copy()
        donor[1] = -(beach[0]*cx+beach[2]*cz)
        offset = float(np.float32(rng.uniform(-.8, .15)))
        halfspaces.append((donor, offset))
        halfspaces.append((beach, offset))
        without_twin.append(list(halfspaces))
        with_twin.append(halfspaces+[(beach, offset)])
    np.testing.assert_allclose(_prism_volume(with_twin), _prism_volume(without_twin),
                               atol=1e-12, rtol=0)


def _periodic_plane(signs, displacement=0.):
    """Exact periodic slab; each nonzero cell-coordinate increment is 1 cm."""
    g = Grid3DConfig(nx=32, ny=32, nz=32, Lx=.4, Ly=32*6/224, Lz=.32)
    cfg = Solver3DConfig(grid=g, device=DEVICE, pressure_discretization="liquid_mimetic",
        interface_reconstruction="height_function", vof_flux_reconstruction="swept_prism",
        bc=BC3DConfig(**{a+s: BC.PERIODIC for a in "xyz" for s in ("lo", "hi")}))
    coordinates = np.indices(g.shape)
    index = np.einsum("i,ijkl->jkl", signs, coordinates).astype(int)
    lower, upper = int(index.min()), int(index.max())
    normal = .01*np.asarray(signs, float)
    magnitude = abs(normal).sum()
    values = []
    for cell in range(lower, upper+1):
        volume = 0.
        for period in range(-4, 5):
            bottom = period*.32+.037+displacement-.01*cell
            volume += unit_cube_volume(normal, (bottom+.16)/magnitude)
            volume -= unit_cube_volume(normal, bottom/magnitude)
        values.append(volume)
    return cfg, np.asarray(values, np.float32)[index-lower]


@pytest.mark.parametrize("signs", [(1, 0, 1), (1, 1, 1), (-1, 1, -1)])
@pytest.mark.parametrize("motion", ["axis", "tangent", "oblique"])
@pytest.mark.parametrize("direction", [-1, 1])
def test_periodic_plane_translation_keeps_shape_and_volume(signs, motion, direction):
    cfg, initial = _periodic_plane(signs)
    state = State3D(cfg)
    state.set_interior("alpha", initial)
    g = cfg.grid
    normal = .01*np.asarray(signs)/np.array((g.dx, g.dy, g.dz))
    velocity = np.array((.2, 0., 0.), np.float32)
    if motion != "axis":
        velocity[1] = .07
        velocity[2] = (.1 if motion == "oblique" else
            -(normal[0]*velocity[0]+normal[1]*velocity[1])/normal[2])
    velocity *= direction
    for name, speed in zip("uvw", velocity, strict=True):
        state.fill_(name, float(speed))
    advector = UnsplitVOFAdvector(cfg)
    initial_volume = initial.astype(float).sum()*g.dx*g.dy*g.dz
    limited = 0.
    for step in range(1, 21):
        result = advector.advect(state, .005)
        limited += result.limited_flux_volume
        if step in (1, 20):
            _, expected = _periodic_plane(signs, .005*step*(normal@velocity))
            after = state.get_interior("alpha")
            np.testing.assert_allclose(after, expected, atol=3e-5, rtol=0)
            assert 0 <= after.min() <= after.max() <= 1
            volume = after.astype(float).sum()*g.dx*g.dy*g.dz
            assert abs(volume-initial_volume) < 5e-7*initial_volume
    # This control requires no substantive donor/capacity correction.
    assert limited < 1e-6*initial_volume
    assert advector.total_clipped_volume < 1e-6*initial_volume


def test_transport_timestep_includes_air_band_faces_reached_diagonally():
    g = Grid3DConfig(nx=8, ny=8, nz=8, Lx=1, Ly=1, Lz=1)
    cfg = Solver3DConfig(grid=g, device=DEVICE, pressure_discretization="liquid_mimetic",
        vof_flux_reconstruction="swept_prism",
        bc=BC3DConfig(**{a+s: BC.PERIODIC for a in "xyz" for s in ("lo", "hi")}))
    state = State3D(cfg)
    alpha = np.zeros(g.shape, np.float32)
    alpha[:, :, 2:5] = 1
    alpha[:, :, 5] = .4
    state.set_interior("alpha", alpha)
    mass = compute_face_liquid_mass(state, "u").numpy()[state.grid.interior_slices("u")]
    profile = (.2+.3*(np.arange(g.nz)-2)).astype(np.float32)
    state.set_interior("u", np.where(mass > 0, profile[None, None, :], 0).astype(np.float32))
    state.fill_("w", .01)
    advector = MomentumAdvector(cfg)
    advector.vof = UnsplitVOFAdvector(cfg)
    dt = advector.transport_time_step(state, .5)
    result = advector.advect(state, dt)
    assert result.vof.max_courant <= .5*(1+2e-6)


@pytest.mark.parametrize("direction", [-1, 1])
def test_periodic_plane_keeps_shape_with_actual_air_velocity_extension(direction):
    cfg, initial = _periodic_plane((1, 1, 1))
    state = State3D(cfg)
    state.set_interior("alpha", initial)
    g = cfg.grid
    normal = .01/np.array((g.dx, g.dy, g.dz))
    velocity = np.array((.2, .07, 0.))
    velocity[2] = -(normal[:2]@velocity[:2])/normal[2]
    velocity *= direction
    for name, speed in zip("uvw", velocity):
        state.fill_(name, float(speed))
    extension = extrapolate_velocity(state, layers=4, water_threshold=0.)
    UnsplitVOFAdvector(cfg).advect(state, .005, transport_velocity=extension.fields)
    after = state.get_interior("alpha")
    # This includes all periodic seams: tangent motion leaves the exact slab unchanged.
    np.testing.assert_allclose(after, initial, atol=3e-5, rtol=0)
