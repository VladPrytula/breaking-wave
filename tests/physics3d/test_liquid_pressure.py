"""Physical and adjoint gates for the retained-liquid pressure operator."""

from itertools import product
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix
import warp as wp

from shoreloop.config import BC, BC3DConfig, Grid3DConfig, Solver3DConfig
from shoreloop.device import DEVICE
from shoreloop.physics3d.cut_cells import initialize_scene_cut_cells
from shoreloop.physics3d.liquid_pressure import (
    LiquidPressureOperator, LiquidPressureProjection, _cells, _local_projection,
    _pressure_coordinates,
)
from shoreloop.physics3d.plic import PLICReconstruction
from shoreloop.physics3d.relaxation import WaveRelaxationZone
from shoreloop.physics3d.state import State3D
from shoreloop.physics3d.resolved import resolved_liquid_mask
from shoreloop.physics3d.vof import initialize_joint_plane
from shoreloop.physics3d.waves import LinearWaveInlet, PeriodicWaveComponent
from shoreloop.scene.geometry3d import AnalyticBeach, SceneGeometry


def _beach(eta=.0201):
    g = Grid3DConfig(nx=12, ny=2, nz=8, Lx=.15, Ly=.025, Lz=.08)
    state = State3D(Solver3DConfig(grid=g, device=DEVICE))
    initialize_scene_cut_cells(state, SceneGeometry(AnalyticBeach(-.1, .1, -.003)))
    initialize_joint_plane(state, (0, 0, 1), eta)
    return state


def _velocity(operator, flux):
    if operator.velocity_space == "mac":
        return flux
    return flux / np.concatenate([c.areas[a] for c, a in zip(operator.cells, operator.active)])


def test_pressure_source_preserves_solenoidal_motion_in_a_thin_metric():
    # Outward coordinates: opposite velocities carry rigid motion. Applying
    # independently rounded M and E erases this small positive kinetic mode.
    mass = np.asarray([[1+1e-10, 1.], [1., 1+1e-10]])
    divergence = np.ones(2)
    _, _, projector, _ = _local_projection(mass, divergence)
    initial = np.asarray([-.2, .2], dtype=np.float32)
    np.testing.assert_array_equal(projector.astype(np.float32) @ initial, initial)


def test_pressure_pair_coordinates_retain_a_small_positive_mode_in_fp32():
    # The second production stage has pressure pairs whose small positive
    # eigenvalue disappears when the large coefficients are rounded first.
    matrix = csr_matrix([[1e11+1000, -1e11], [-1e11, 1e11+1000]])
    basis = _pressure_coordinates(matrix)
    np.testing.assert_allclose((basis.T @ basis).toarray(), np.eye(2), atol=3e-16)
    transformed = (basis.T @ matrix @ basis).toarray().astype(np.float32).astype(float)
    assert np.linalg.eigvalsh(transformed).min() == pytest.approx(1000., rel=2e-7)
    solution = basis @ np.linalg.solve(transformed, basis.T @ np.ones(2))
    np.testing.assert_allclose(solution, [.001, .001], rtol=2e-7, atol=0.)


def test_enclosed_boulder_pocket_has_positive_finite_pressure_metric():
    # The production pocket retained five MAC velocities but only one active
    # facet. Three opposite-face difference penalties left a rigid wall-normal
    # mode without inertia, so the local inverse generated NaNs.
    from shoreloop.physics3d.liquid_assembly import assemble_local_systems

    g = Grid3DConfig(nx=5, ny=5, nz=5, Lx=5*4.8/384, Ly=5*6/224, Lz=.05)
    state = State3D(Solver3DConfig(grid=g, device=DEVICE, liquid_resolved_fraction=.01))
    with np.load(Path(__file__).parent / "data/boulder_pressure_pocket.npz") as saved:
        for name in ("alpha", "theta", "open_normals", "open_offsets",
                     "u_aperture", "v_aperture", "w_aperture"):
            getattr(state, name).assign(saved[name])
        before = saved["alpha"].copy()
        plic = PLICReconstruction(
            wp.array(saved["liquid_normals"], dtype=wp.vec3, device=DEVICE),
            wp.array(saved["liquid_offsets"], device=DEVICE),
            wp.array(saved["liquid_centroids"], dtype=wp.vec3, device=DEVICE), 0, 0,
        )
    # Resolved liquid is now measured per cell volume; this pocket holds less
    # than 1% of a cell, so the regression keeps it resolved with a lower
    # threshold to exercise the metric it was recorded for.
    local = assemble_local_systems(state, plic, extension_inertia=1., wet_threshold=1e-4).host()
    row = int(np.flatnonzero(np.all(local.cell_index == (2, 2, 2), axis=1))[0])
    n = local.widths[row]
    mass = local.mass[row, :n, :n]
    np.testing.assert_allclose(mass, mass.T, rtol=0, atol=1e-20)
    assert np.linalg.eigvalsh(mass).min() > 1e-8*np.linalg.norm(mass, ord=2)
    for name in ("weights", "projector", "response", "gravity", "pressure_gravity"):
        assert np.isfinite(getattr(local, name)).all(), name
    divergence = local.divergence[row, :n]
    projector = local.projector[row, :n, :n]
    np.testing.assert_allclose(divergence @ projector, 0, atol=1e-17)
    rng = np.random.default_rng(739)
    initial = rng.normal(size=n)
    projected = projector @ initial
    assert projected @ mass @ projected <= initial @ mass @ initial*(1+1e-12)
    np.testing.assert_array_equal(state.alpha.numpy(), before)


def test_tiny_exposed_neighbour_facet_still_closes_the_liquid_boundary():
    # Two planes from fresh production wave input at (58,38,11). Their
    # positive-area difference is below the former absolute facet cutoff,
    # but omitting it violates the original thin-volume affine identity.
    g = Grid3DConfig(nx=3, ny=3, nz=3, Lx=.075, Ly=3*6/112, Lz=.06)
    state = State3D(Solver3DConfig(grid=g, device=DEVICE))
    state.set_interior("alpha", np.full(g.shape, .000336192, np.float32))
    n = np.empty((*state.grid.cell_shape, 3), np.float32)
    n[:] = (.00020603160373866558, -3.2782665471131622e-9, .9997939467430115)
    d = np.full(state.grid.cell_shape, .00043913687113672495, np.float32)
    n[2, 1, 2] = (.00020163696899544448, -4.395393716549734e-6, .9997939467430115)
    d[2, 1, 2] = .0004347386711742729
    centre = np.empty_like(n)
    centre[:] = (.44891640543937683, .5000036358833313, .00017335479788016528)
    plic = PLICReconstruction(wp.array(n, dtype=wp.vec3, device=DEVICE),
                              wp.array(d, device=DEVICE),
                              wp.array(centre, dtype=wp.vec3, device=DEVICE), 27, 0)
    cells, _, _ = _cells(state, plic)
    cell = next(c for c in cells if c.index == (1, 1, 1))
    np.testing.assert_allclose(cell.displacements.T @ (cell.areas[:, None]*cell.normals)/cell.volume,
                               np.eye(3), rtol=0, atol=2e-10)


def test_nearly_horizontal_production_film_preserves_affine_moments():
    # Production cell (62,30,11), including the exact grid-spacing rounding.
    # Broad opposing facets enclose alpha=1.23e-6; FP64-only quadrature loses
    # the original off-diagonal affine bound through moment cancellation.
    g = Grid3DConfig(nx=192, ny=112, nz=3, Lx=4.8, Ly=6, Lz=.06)
    state = State3D(Solver3DConfig(grid=g, device=DEVICE))
    alpha = np.zeros(g.shape, np.float32)
    alpha[1, 1, 1] = 1.2321665963099804e-6
    state.set_interior("alpha", alpha)
    n = np.zeros((*state.grid.cell_shape, 3), np.float32)
    d = np.zeros(state.grid.cell_shape, np.float32)
    centre = np.zeros_like(n)
    index = (g.ng+1,)*3
    n[index] = (3.3875044209707994e-6, -6.3917732404661365e-6, .9999902248382568)
    d[index] = -8.516841489836224e-7
    centre[index] = (.3608938753604889, .77638840675354, 1.429083340553916e-6)
    plic = PLICReconstruction(wp.array(n, dtype=wp.vec3, device=DEVICE),
                              wp.array(d, device=DEVICE),
                              wp.array(centre, dtype=wp.vec3, device=DEVICE), 1, 0)
    cells, _, _ = _cells(state, plic)
    cell, = cells
    np.testing.assert_allclose(cell.displacements.T @ (cell.areas[:, None]*cell.normals)/cell.volume,
                               np.eye(3), rtol=0, atol=2e-10)


def test_second_step_production_film_preserves_affine_pressure():
    # Production cell (65,102,11), alpha=1.37e-6. Rounding area and first
    # moment separately made the opposing faces' y centroids differ by one
    # FP64 rounding unit, enough to break the thin-volume affine identity.
    g = Grid3DConfig(nx=192, ny=112, nz=3, Lx=4.8, Ly=6, Lz=.06)
    state = State3D(Solver3DConfig(grid=g, device=DEVICE))
    alpha = np.zeros(g.shape, np.float32)
    alpha[1, 1, 1] = 1.370273363665575e-6
    state.set_interior("alpha", alpha)
    normals = np.zeros((*state.grid.cell_shape, 3), np.float32)
    offsets = np.zeros(state.grid.cell_shape, np.float32)
    centroids = np.zeros_like(normals)
    index = (g.ng+1,)*3
    normals[index] = (1.975563606038122e-7, -2.306214810232632e-6, .9999974966049194)
    offsets[index] = 3.1604014338881825e-7
    centroids[index] = (.48828017711639404, .6410495042800903, 8.501428965246305e-7)
    plic = PLICReconstruction(wp.array(normals, dtype=wp.vec3, device=DEVICE),
                              wp.array(offsets, device=DEVICE),
                              wp.array(centroids, dtype=wp.vec3, device=DEVICE), 1, 0)
    cells, _, _ = _cells(state, plic)
    cell, = cells
    n, d = normals[index].astype(float), float(offsets[index])
    # The plane is above z=0 and below z=1 over the full x/y footprint.
    exact_volume = (d-.5*(n[0]+n[1]))/n[2]*g.dx*g.dy*g.dz
    assert cell.volume == pytest.approx(exact_volume, rel=1e-9)
    np.testing.assert_allclose(cell.displacements.T @ (cell.areas[:, None]*cell.normals)/cell.volume,
                               np.eye(3), rtol=0, atol=2e-10)
    np.testing.assert_array_equal(state.get_interior("alpha"), alpha)


def test_partitioned_third_step_film_preserves_affine_pressure():
    # Production (87,33,11): the lower solid plane partitions the broad
    # bottom facet. Rounding its centroids before forming displacements loses
    # the original affine bound even when each centroid divides before rounding.
    g = Grid3DConfig(nx=192, ny=112, nz=3, Lx=4.8, Ly=6, Lz=.06)
    # Films of 1e-6 of an open cell are resolved for this geometry test; the
    # production rule measures liquid per cell volume.
    state = State3D(Solver3DConfig(grid=g, device=DEVICE, liquid_resolved_fraction=1e-9))
    alpha = np.zeros(g.shape, np.float32)
    theta = np.ones(g.shape, np.float32)
    alpha[1, 1, 1], alpha[1, 1, 0] = 1.0398058520877385e-6, 1.
    theta[1, 1, 1] = theta[1, 1, 0] = .75
    state.set_interior("alpha", alpha)
    state.set_geometry_interior("theta", theta)
    sn = np.zeros((*state.grid.cell_shape, 3), np.float32)
    sd = np.zeros(state.grid.cell_shape, np.float32)
    wn, wd, centres = np.zeros_like(sn), np.zeros_like(sd), np.zeros_like(sn)
    cell = (g.ng+1,)*3
    below = (g.ng+1, g.ng+1, g.ng)
    sn[cell] = (-.10597056150436401, -.8821896314620972, .01183980144560337)
    sd[cell] = -.2676127851009369
    sn[below] = (-.10496778041124344, -.8627211451530457, .032311078161001205)
    sd[below] = -.2520086467266083
    wn[cell], wd[cell] = (0, 0, 1), 1.0282965376973152e-6
    centres[cell] = (.5128350257873535, .6185535192489624, 5.126667019794695e-7)
    centres[below] = (.5135189890861511, .6240996718406677, .49583861231803894)
    state.open_normals.assign(sn)
    state.open_offsets.assign(sd)
    plic = PLICReconstruction(wp.array(wn, dtype=wp.vec3, device=DEVICE),
                              wp.array(wd, device=DEVICE),
                              wp.array(centres, dtype=wp.vec3, device=DEVICE), 1, 0)
    cells, _, _ = _cells(state, plic)
    film = next(c for c in cells if c.index == (1, 1, 1))
    np.testing.assert_allclose(film.displacements.T @ (film.areas[:, None]*film.normals)/film.volume,
                               np.eye(3), rtol=0, atol=2e-10)
    assert film.displacements.dtype == np.dtype(np.longdouble)
    assert film.mass.dtype == np.dtype(float)
    np.testing.assert_array_equal(state.get_interior("alpha"), alpha)


def test_tiny_boulder_cut_film_preserves_clipped_affine_geometry():
    # Fourth production step (96,73,11): a 1/64-open cell with a film only
    # 1.64e-7 cells high. FP64 clipping loses the affine identity before
    # extended moment/displacement accumulation has a chance to preserve it.
    g = Grid3DConfig(nx=192, ny=112, nz=3, Lx=4.8, Ly=6, Lz=.06)
    # Films of 1e-6 of an open cell are resolved for this geometry test; the
    # production rule measures liquid per cell volume.
    state = State3D(Solver3DConfig(grid=g, device=DEVICE, liquid_resolved_fraction=1e-9))
    alpha = np.zeros(g.shape, np.float32)
    theta = np.ones(g.shape, np.float32)
    alpha[1, 1, 1], theta[1, 1, 1] = 1.1790234566433355e-6, .015625
    state.set_interior("alpha", alpha)
    state.set_geometry_interior("theta", theta)
    sn = np.zeros((*state.grid.cell_shape, 3), np.float32)
    sd = np.zeros(state.grid.cell_shape, np.float32)
    wn, wd, centres = np.zeros_like(sn), np.zeros_like(sd), np.zeros_like(sn)
    index = (g.ng+1,)*3
    sn[index] = (.5198972821235657, .16266775131225586, .31743499636650085)
    sd[index] = .13602383434772491
    wn[index], wd[index] = (0, 0, 1), 1.6379635781049728e-7
    centres[index] = (.08777906000614166, .2784019410610199, 7.918785627225589e-8)
    state.open_normals.assign(sn)
    state.open_offsets.assign(sd)
    plic = PLICReconstruction(wp.array(wn, dtype=wp.vec3, device=DEVICE),
                              wp.array(wd, device=DEVICE),
                              wp.array(centres, dtype=wp.vec3, device=DEVICE), 1, 0)
    cells, _, _ = _cells(state, plic)
    cell, = cells
    a, b, c = sn[index].astype(float)
    d, h = float(sd[index]), float(wd[index])
    # Integrate the triangular footprint (d-c*z)^2/(2*a*b), z=0..h.
    volume = h*(d*d-d*c*h+c*c*h*h/3)/(2*a*b)*g.dx*g.dy*g.dz
    assert cell.volume == pytest.approx(volume, rel=2e-10)
    np.testing.assert_allclose(cell.displacements.T @ (cell.areas[:, None]*cell.normals)/cell.volume,
                               np.eye(3), rtol=0, atol=2e-10)
    np.testing.assert_array_equal(state.get_interior("alpha"), alpha)


@pytest.mark.parametrize("fraction, normal, offset, centre", [
    (1.0911952585956897e-6, (0., -4.7683693082944956e-7, .9999995231628418),
     2.386533637377397e-8, (.06872431188821793, .3949294686317444, 1.076911289032978e-7)),
    (1.0911953722825274e-6, (2.9802308176840597e-8, -5.364411208574893e-7, .999999463558197),
     1.4785394242267103e-8, (.06076456978917122, .4389773905277252, 1.1536305066783825e-7)),
])
def test_tilted_boulder_film_preserves_affine_geometry(fraction, normal, offset, centre):
    # Fresh steps ten and eleven, cell (96,73,11), with the three neighbours
    # that partition its facets. Rounding normals or displacements to FP64
    # before assembly loses the original thin-volume affine identity.
    g = Grid3DConfig(nx=192, ny=112, nz=3, Lx=4.8, Ly=6, Lz=.06)
    # Films of 1e-6 of an open cell are resolved for this geometry test; the
    # production rule measures liquid per cell volume.
    state = State3D(Solver3DConfig(grid=g, device=DEVICE, liquid_resolved_fraction=1e-9))
    alpha, theta = np.zeros(g.shape, np.float32), np.zeros(g.shape, np.float32)
    alpha[1, 1, 1], theta[1, 1, 1] = fraction, .015625
    alpha[0, 1, 1], theta[0, 1, 1] = 8.623777034699742e-7, .828125
    alpha[1, 0, 1], theta[1, 0, 1] = 7.340540264522133e-7, .546875
    alpha[1, 1, 0], theta[1, 1, 0] = 1., .46875
    state.set_interior("alpha", alpha)
    state.set_geometry_interior("theta", theta)
    sn = np.zeros((*state.grid.cell_shape, 3), np.float32)
    sd = np.zeros(state.grid.cell_shape, np.float32)
    wn, wd, centres = np.zeros_like(sn), np.zeros_like(sd), np.zeros_like(sn)
    index = (g.ng+1,)*3
    for local, solid_normal, solid_offset in (
        ((1,1,1), (.5198972821235657, .16266775131225586, .31743499636650085), .13602383434772491),
        ((0,1,1), (.5580409169197083, .16424742341041565, .2777116596698761), .6918640732765198),
        ((1,0,1), (.3741341829299927, .41010087728500366, .21576493978500366), .5214071869850159),
        ((1,1,0), (.4557563364505768, .17068885266780853, .3735547959804535), .4852893650531769),
    ):
        cell = tuple(g.ng+i for i in local)
        sn[cell], sd[cell] = solid_normal, solid_offset
    wn[index], wd[index], centres[index] = normal, offset, centre
    centres[g.ng+1, g.ng+1, g.ng] = (.3051312565803528, .4346000552177429, .3555927276611328)
    state.open_normals.assign(sn)
    state.open_offsets.assign(sd)
    plic = PLICReconstruction(wp.array(wn, dtype=wp.vec3, device=DEVICE),
                              wp.array(wd, device=DEVICE),
                              wp.array(centres, dtype=wp.vec3, device=DEVICE), 2, 0)
    cells, _, _ = _cells(state, plic)
    film = next(c for c in cells if c.index == (1, 1, 1))
    a, b, c = map(Fraction, sn[index].astype(float).tolist())
    d = Fraction(float(sd[index]))
    nx, ny, nz = map(Fraction, wn[index].astype(float).tolist())
    offset = Fraction(float(wd[index]))
    # The solid plane cuts a tetrahedron from the positive octant. Subtract
    # the dry tetrahedron above the liquid plane using exact rational volumes,
    # independently of polygon clipping/quadrature and without cancellation.
    top = nz*d/c
    assert max(0, nx*d/a, ny*d/b) < offset < top
    dry_fraction = (top-offset)**3/(top*(top-nx*d/a)*(top-ny*d/b))
    volume = float(d**3/(6*a*b*c)*(1-dry_fraction))*g.dx*g.dy*g.dz
    assert film.volume == pytest.approx(volume, rel=2e-10, abs=0)
    np.testing.assert_allclose(film.displacements.T @ (film.areas[:, None]*film.normals)/film.volume,
                               np.eye(3), rtol=0, atol=2e-10)
    assert film.displacements.dtype == film.areas.dtype == film.normals.dtype == np.dtype(np.longdouble)
    assert film.mass.dtype == np.dtype(float)
    np.testing.assert_array_equal(state.get_interior("alpha"), alpha)


@pytest.mark.parametrize("signs", list(product((-1, 1), repeat=3)))
def test_corner_tetrahedron_geometry_preserves_affine_pressure(signs):
    # The actual 64x64x32 saved-water failure, reflected into every corner.
    # These are stored FP32 planes, so derive each analytic tetrahedron from
    # its rounded plane rather than assuming reflection preserves its offset.
    original = np.asarray((-.9511614441871643, .0018352462211623788,
                           -.04700330272316933), dtype=np.float32)
    height = -.9972317218780518 - original[original < 0].astype(float).sum()
    normal = (np.abs(original)*signs).astype(np.float32)
    offset = np.float32(height + normal[normal < 0].astype(float).sum())
    intercepts = (float(offset)-normal[normal < 0].astype(float).sum())/np.abs(normal.astype(float))
    fraction = float(np.prod(intercepts)/6)
    local_centroid = np.where(normal < 0, 1-intercepts/4, intercepts/4)
    g = Grid3DConfig(nx=3, ny=3, nz=3, Lx=3*4.8/384, Ly=3*6/224, Lz=.03)
    state = State3D(Solver3DConfig(grid=g, device=DEVICE))
    alpha = np.zeros(g.shape, np.float32)
    alpha[1, 1, 1] = fraction
    state.set_interior("alpha", alpha)
    index = (g.ng+1,)*3
    normals = np.zeros((*state.grid.cell_shape, 3), np.float32)
    offsets = np.zeros(state.grid.cell_shape, np.float32)
    centroids = np.zeros_like(normals)
    normals[index], offsets[index], centroids[index] = normal, offset, local_centroid
    plic = PLICReconstruction(wp.array(normals, dtype=wp.vec3, device=DEVICE),
                              wp.array(offsets, device=DEVICE),
                              wp.array(centroids, dtype=wp.vec3, device=DEVICE), 1, 0)
    cells, _, _ = _cells(state, plic)
    assert len(cells) == 1
    cell = cells[0]
    assert len(cell.areas) == 4
    assert cell.volume == pytest.approx(fraction*g.dx*g.dy*g.dz, rel=2e-10)
    N, R = cell.areas[:, None]*cell.normals, cell.displacements
    np.testing.assert_allclose(R.T @ N / cell.volume, np.eye(3), rtol=0, atol=2e-10)
    assert np.linalg.eigvalsh(cell.mass).min() > 0
    np.testing.assert_allclose(cell.mass @ N, R, rtol=0, atol=1e-9*np.max(np.abs(R)))
    np.testing.assert_array_equal(state.get_interior("alpha"), alpha)


@pytest.mark.parametrize("space", ["facets", "mac"])
def test_liquid_pressure_affine_reproduction_symmetry_and_adjoint_identity(space):
    operator = LiquidPressureOperator(_beach(), velocity_space=space)
    direction = np.asarray((.31, -.7, 1.2))
    rng = np.random.default_rng(1204)
    for cell in operator.cells:
        N = cell.areas[:, None] * cell.normals
        R = cell.displacements
        np.testing.assert_allclose(cell.mass @ N, R, atol=3e-11)
        assert np.linalg.eigvalsh(cell.mass).min() > 0
        # Linear face/cell pressure samples produce exactly A*n.grad(p).
        gradient_flux = np.linalg.solve(cell.mass, np.asarray(R @ direction, dtype=float))
        np.testing.assert_allclose(gradient_flux, N @ direction, rtol=1e-8, atol=1e-12)
        q = cell.areas * rng.normal(size=len(cell.areas))
        pcell = 1.7
        pface = pcell + R @ direction
        gradient = np.linalg.solve(cell.mass, np.asarray(pface-pcell, dtype=float))
        # Discrete integration by parts, including boundary pressure work.
        assert q @ cell.mass @ gradient == pytest.approx(q @ pface - pcell*q.sum(), abs=1e-12)
    A = operator.matrix_host
    x, y = rng.normal(size=(2, A.shape[0]))
    assert x @ A @ y == pytest.approx(y @ A @ x, rel=1e-12, abs=1e-12)
    assert x @ A @ x > 0


@pytest.mark.parametrize("fraction", [.01, .49, .499999, .5, .500001, .51])
@pytest.mark.parametrize("space", ["facets", "mac"])
def test_retained_sheet_freefall_has_no_half_cell_jump(fraction, space):
    g = Grid3DConfig(nx=4, ny=2, nz=4, Lx=.05, Ly=.025, Lz=.04)
    state = State3D(Solver3DConfig(grid=g, device=DEVICE, bc=BC3DConfig(
        xlo=BC.PERIODIC, xhi=BC.PERIODIC,
        ylo=BC.PERIODIC, yhi=BC.PERIODIC,
    )))
    alpha = np.zeros(g.shape, np.float32)
    alpha[:, :, 2] = fraction
    state.set_interior("alpha", alpha)
    operator = LiquidPressureOperator(state, velocity_space=space)
    dt = .001
    result = operator.project(dt, acceleration=(0, 0, -9.81))
    normal_z = np.concatenate([c.normals[a, 2] for c, a in zip(operator.cells, operator.active)])
    if space == "mac":
        normal_z = np.asarray([sign if axis == 2 else 0 for faces in operator.dof_faces
                               for axis, i, j, k, sign in faces])
    np.testing.assert_allclose(_velocity(operator, result.flux.numpy()), -9.81*dt*normal_z, atol=2e-7)
    np.testing.assert_array_equal(alpha, state.get_interior("alpha"))


@pytest.mark.parametrize("eta", [.0201, .0273])
@pytest.mark.parametrize("space", ["facets", "mac"])
def test_subcell_cut_beach_pressure_and_physical_gravity_balance(eta, space):
    state = _beach(eta)
    before = state.get_interior("alpha")
    operator = LiquidPressureOperator(state, velocity_space=space)
    result = operator.project(.001, acceleration=(0, 0, -9.81), tolerance=2e-7)
    velocity = _velocity(operator, result.flux.numpy())
    assert np.max(np.abs(velocity)) < 2e-6
    expected = np.asarray([state.cfg.physics.rho*9.81*(eta-c.centroid[2]) for c in operator.cells])
    np.testing.assert_allclose(result.pressure.numpy(), expected, atol=.003)
    np.testing.assert_array_equal(before, state.get_interior("alpha"))
    if space == "mac":
        operator.write_mac_state(state, result)
        actual = operator.read_mac_state(state).numpy()
        assert np.max(np.abs(actual)) < 2e-6
        for i, cell in enumerate(operator.cells):
            values = actual[operator.starts_host[i]:operator.starts_host[i+1]]
            assert abs(operator.divergence_vectors[i] @ values) / cell.volume < 5e-4


@pytest.mark.parametrize("space", ["facets", "mac"])
@pytest.mark.parametrize("relative_tolerance", [1e-6, 1e-10])
def test_liquid_projection_does_not_inject_its_kinetic_energy(space, relative_tolerance):
    state = _beach()
    operator = LiquidPressureOperator(state, velocity_space=space)
    areas = np.concatenate([c.areas[a] for c, a in zip(operator.cells, operator.active)])
    if space == "mac":
        areas = np.ones(operator.starts_host[-1])
    initial = (areas * np.random.default_rng(72).normal(0, .01, len(areas))).astype(np.float32)
    result = operator.project(.001, initial_flux=wp.array(initial, device=DEVICE),
                              tolerance=relative_tolerance)
    projected = result.flux.numpy()
    shared = operator.shared_host >= 0
    jumps = np.bincount(operator.shared_host[shared], weights=projected[shared].astype(float),
                        minlength=operator.matrix_host.shape[0])
    jumps[operator.gauges.numpy()] = 0
    assert np.linalg.norm(jumps) <= result.tolerance
    before = after = 0.0
    for i, (cell, active) in enumerate(zip(operator.cells, operator.active)):
        section = slice(operator.starts_host[i], operator.starts_host[i+1])
        M = operator.mass_blocks[i]
        before += initial[section] @ M @ initial[section]
        after += projected[section] @ M @ projected[section]
        assert abs(operator.divergence_vectors[i] @ projected[section]) / cell.volume < 2e-5
    assert after <= before * (1+2e-6)


@pytest.mark.parametrize("reflect_x", [False, True])
def test_weak_relaxation_at_wall_corners_has_positive_pressure_metric(reflect_x):
    g = Grid3DConfig(nx=4, ny=2, nz=4, Lx=.05, Ly=.025, Lz=.04)
    state = State3D(Solver3DConfig(grid=g, device=DEVICE,
                                  pressure_discretization="liquid_mimetic"))
    inlet = LinearWaveInlet((PeriodicWaveComponent(period=1, depth=.025,
                             alongshore_length=g.Ly, amplitude=0),))
    WaveRelaxationZone(state.cfg, inlet, g.Lx, timescale=.05).apply(state, .001, .001)
    if reflect_x:
        state.set_interior("alpha", state.get_interior("alpha")[::-1].copy())
    alpha = state.get_interior("alpha")
    operator = LiquidPressureOperator(state, velocity_space="mac")
    # The corner's free-surface mean flux cannot observe every admissible
    # MAC velocity. The declared extension inertia must include differences
    # from the fixed wall velocities so these modes still have positive energy.
    for mass in operator.mass_blocks:
        assert np.linalg.eigvalsh(mass).min() > 0
    matrix = operator.matrix_host.toarray()
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-12*np.max(np.abs(matrix)))
    assert np.linalg.eigvalsh(matrix).min() > 0
    result = operator.project(.001, acceleration=(0, 0, -9.81),
                              initial_flux=operator.read_mac_state(state), tolerance=1e-6,
                              max_iterations=200)
    operator.write_mac_state(state, result)
    assert operator.divergence(operator.read_mac_state(state))[0] < 1e-4
    np.testing.assert_array_equal(state.get_interior("alpha"), alpha)
    np.testing.assert_array_equal(state.get_interior("u")[[0, -1]], 0)
    np.testing.assert_array_equal(state.get_interior("w")[:, :, [0, -1]], 0)


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("fraction", [.01, 1.0])
def test_periodic_freefall_survives_mac_transfer_with_only_upper_side_wet(axis, fraction):
    g = Grid3DConfig(nx=4, ny=4, nz=4, Lx=.05, Ly=.05, Lz=.05)
    state = State3D(Solver3DConfig(grid=g, device=DEVICE, bc=BC3DConfig(
        xlo=BC.PERIODIC, xhi=BC.PERIODIC, ylo=BC.PERIODIC, yhi=BC.PERIODIC,
        zlo=BC.PERIODIC, zhi=BC.PERIODIC,
    )))
    alpha = np.zeros(g.shape, np.float32)
    wet = [slice(None)]*3
    wet[axis] = -1
    alpha[tuple(wet)] = fraction
    state.set_interior("alpha", alpha)
    operator = LiquidPressureOperator(state, velocity_space="mac")
    acceleration = np.zeros(3)
    acceleration[axis] = -9.81
    result = operator.project(.001, acceleration=acceleration)
    assert operator.divergence(result.flux)[0] < 1e-4
    operator.write_mac_state(state, result)
    actual = operator.read_mac_state(state)
    np.testing.assert_allclose(actual.numpy(), result.flux.numpy(), atol=2e-7)
    assert operator.divergence(actual)[0] < 1e-4
    # The high face is a periodic duplicate: the canonical low face must be
    # written even when the only incident liquid cell is at the upper side.
    seam = np.take(state.get_interior("uvw"[axis]), [0, -1], axis=axis)
    np.testing.assert_allclose(seam, -.00981, atol=2e-7)
    np.testing.assert_array_equal(state.get_interior("alpha"), alpha)


def test_resolved_liquid_is_measured_per_cell_volume_in_cut_cells():
    # The saved production pocket beside a boulder: two of its cells hold more
    # than 1% of their open volume but less than 1% of a cell. Such slivers
    # carried pressure unknowns and 5-8 m/s faces in production.
    g = Grid3DConfig(nx=5, ny=5, nz=5, Lx=5*4.8/384, Ly=5*6/224, Lz=.05)
    state = State3D(Solver3DConfig(grid=g, device=DEVICE, pressure_discretization="liquid_mimetic",
                                   liquid_resolved_fraction=.02))
    with np.load(Path(__file__).parent / "data/boulder_pressure_pocket.npz") as saved:
        for name in ("alpha", "theta", "open_normals", "open_offsets",
                     "u_aperture", "v_aperture", "w_aperture"):
            getattr(state, name).assign(saved[name])
    alpha = state.get_interior("alpha")
    theta = state.get_geometry_interior("theta")
    # One sliver 90% full of a 1/64-open cell becomes a full, wall-trapped
    # sliver: it holds .0156 of a cell, below the .02 threshold, but has no
    # free surface and stays liquid.
    full = (3, 4, 1)
    assert theta[full] == pytest.approx(.015625) and .85 < alpha[full] < .95
    alpha[full] = 1.0
    state.set_interior("alpha", alpha)
    cell_rule = resolved_liquid_mask(alpha, theta, .02)
    film = (2, 2, 2)                      # alpha .014 of a quarter-open cell
    assert 0 < alpha[film] < .02 and theta[film] == pytest.approx(.25)
    assert cell_rule[full] and not (alpha * theta > .02)[full]
    assert not cell_rule[film] and (alpha > .01)[film]
    assert 0 < cell_rule.sum() < ((alpha > 0) & (theta > 0)).sum()
    operator = LiquidPressureOperator(state, velocity_space="mac")
    assert operator.cell_count == int(cell_rule.sum())
    assert operator.residue_cells == int(((alpha > 0) & (theta > 0) & ~cell_rule).sum())


def test_liquid_below_the_resolved_fraction_carries_no_pressure_unknown():
    """A cell holding .4% of its open volume is residue: it gets no pressure unknown,
    the resolved cell below sees a free surface there, and its liquid and the
    velocity on its own faces are left to the transport."""
    g = Grid3DConfig(nx=4, ny=3, nz=6, Lx=.04, Ly=.03, Lz=.06)
    cell_volume = g.dx * g.dy * g.dz
    rho, gravity = 1000.0, 9.81
    counts = {}
    for fraction in (1e-6, 1e-2):
        cfg = Solver3DConfig(grid=g, device=DEVICE, pressure_discretization="liquid_mimetic",
                             liquid_resolved_fraction=fraction,
                             bc=BC3DConfig(ylo=BC.PERIODIC, yhi=BC.PERIODIC))
        state = State3D(cfg)
        alpha = np.zeros(g.shape, dtype=np.float32)
        alpha[:, :, :3] = 1.0
        alpha[:, :, 3] = 4e-3
        state.set_interior("alpha", alpha)
        w = np.zeros(state.grid.interior_shape("w"), dtype=np.float32)
        w[:, :, 4] = 0.5          # the residue layer's top faces
        state.set_interior("w", w)
        operator = LiquidPressureOperator(state, velocity_space="mac")
        counts[fraction] = (operator.cell_count, operator.residue_cells, operator.residue_volume)
        if fraction == 1e-2:
            top = [c for c in operator.cells if c.index[2] == 2]
            assert len(top) == 12 and all("free" in c.kinds for c in top)
            result = LiquidPressureProjection(cfg).project(state, 1e-3, operator=operator)
            assert result.pressure_solve.converged
            pressure = state.get_interior("p")
            np.testing.assert_allclose(pressure[:, :, 2], rho * gravity * .5 * g.dz, rtol=2e-3)
            np.testing.assert_array_equal(state.get_interior("alpha")[:, :, 3], np.float32(4e-3))
            np.testing.assert_array_equal(state.get_interior("w")[:, :, 4], np.float32(0.5))
    assert counts[1e-6][0] == 4 * 3 * 4 and counts[1e-6][1] == 0
    assert counts[1e-2][0] == 4 * 3 * 3 and counts[1e-2][1] == 12
    assert counts[1e-2][2] == pytest.approx(12 * 4e-3 * cell_volume, rel=1e-6)
