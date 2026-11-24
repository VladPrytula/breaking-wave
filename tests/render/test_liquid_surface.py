"""Retained thin-water volume, contact, separation, and state passivity."""

import numpy as np
import pytest
import warp as wp
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from shoreloop.config import BC, BC3DConfig, Grid3DConfig, Solver3DConfig
from shoreloop.device import DEVICE
from shoreloop.physics3d.cut_cells import initialize_scene_cut_cells
from shoreloop.physics3d.plic import reconstruct_plic
from shoreloop.physics3d.state import State3D
from shoreloop.render.liquid_surface import extract_liquid_surface_mesh, _split_edges, _triangulate
from shoreloop.render.offshore_join import _fundamental_period_mesh, _tile_periodic_y
from shoreloop.render.surface_mesh import extract_free_surface_mesh, extract_state_free_surface_mesh
from shoreloop.scene.geometry3d import AnalyticBeach, SceneGeometry


def _sheet(fraction, *, bed=None, separated=False):
    g = Grid3DConfig(nx=8, ny=6, nz=7, Lx=.08, Ly=.06, Lz=.07)
    s = State3D(Solver3DConfig(grid=g, device=DEVICE, pressure_discretization="liquid_mimetic"))
    if bed is not None:
        initialize_scene_cut_cells(s, SceneGeometry(AnalyticBeach(-1, 0, bed)))
    alpha = np.zeros(g.shape, np.float32)
    alpha[1:3, 1:5, 3] = fraction
    if separated:
        alpha[4:6, 1:5, 3] = fraction
    s.set_interior("alpha", alpha)
    for q, value in zip(("u", "v", "w"), (.03, -.02, .01), strict=True):
        s.set_interior(q, np.full(s.grid.interior_shape(q), value, np.float32))
    s.set_interior("foam", np.where(alpha > 0, .2, 0).astype(np.float32))
    # Prescribe an exact geometric plane only in this analytic extraction
    # control; reconstruct_plic still solves its offset from retained water.
    normals = wp.full(s.grid.cell_shape, wp.vec3(0, 0, 1), dtype=wp.vec3, device=DEVICE)
    return s, reconstruct_plic(s, normals)


def _volume(mesh, origin=(0, 0, 0)):
    p = mesh.vertices[mesh.triangles].astype(float)-origin
    return np.sum(np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2])))/6


def test_periodic_plic_step_matches_the_full_neighbourhood_and_owns_faces_once():
    # The first column is lower than the last: a real vertical water-air face
    # lies on the period cut. Compare with the interior of an extended tank,
    # where the same complete vertex neighbourhood needs no periodic pooling.
    meshes = []
    for repeats in (1, 3):
        g = Grid3DConfig(nx=4, ny=4*repeats, nz=4, Lx=1, Ly=repeats, Lz=1,
                         y0=0 if repeats == 1 else -1)
        s = State3D(Solver3DConfig(grid=g, device=DEVICE,
            pressure_discretization="liquid_mimetic",
            bc=BC3DConfig(xlo=BC.OPEN, xhi=BC.OPEN)))
        alpha = np.zeros(g.shape, np.float32)
        alpha[:, :, :2] = 1
        alpha[:, :, 2] = np.tile((.25, .75, .5, .75), repeats)
        s.set_interior("alpha", alpha)
        for name in ("u", "v", "w", "foam"):
            shape = s.grid.interior_shape(name)
            values = .02*(np.arange(shape[1]) % 4)
            s.set_interior(name, np.broadcast_to(values[None, :, None], shape).astype(np.float32))
        before = {name: s.field(name).numpy().copy() for name in s.FIELDS}
        normals = wp.full(s.grid.cell_shape, wp.vec3(0, 0, 1), dtype=wp.vec3, device=DEVICE)
        mesh = extract_liquid_surface_mesh(s, reconstruction=reconstruct_plic(s, normals))
        assert all(np.array_equal(values, s.field(name).numpy()) for name, values in before.items())
        meshes.append(mesh)
    mesh, extended = meshes
    extended_ids = {tuple(position): i for i, position in enumerate(extended.vertices)}
    ids = [extended_ids[tuple(position)] for position in mesh.vertices]
    for name in ("normals", "velocity", "foam"):
        np.testing.assert_allclose(getattr(mesh, name), getattr(extended, name)[ids], atol=1e-7)
    # A cut-face triangulation genuinely has different vertex sets at the ends.
    assert np.count_nonzero(mesh.vertices[:, 1] == 0) != np.count_nonzero(mesh.vertices[:, 1] == 1)
    tiled = _tile_periodic_y(mesh, 1)
    fundamental = _fundamental_period_mesh(tiled, 0, 1)
    retiled = _tile_periodic_y(fundamental, 1)
    for result in (tiled, retiled):
        p = result.vertices[result.triangles]
        assert np.all(np.linalg.norm(np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0]), axis=1) > 0)
        assert len(np.unique(np.sort(result.triangles, axis=1), axis=0)) == len(result.triangles)
        edges = np.sort(np.concatenate([result.triangles[:, pair] for pair in ((0, 1), (1, 2), (2, 0))]), axis=1)
        edges, counts = np.unique(edges, axis=0, return_counts=True)
        p = result.vertices[edges]
        open_x = np.all(p[:, :, 0] == 0, axis=1) | np.all(p[:, :, 0] == 1, axis=1)
        exterior = open_x | np.all(p[:, :, 1] == -1, axis=1) | np.all(p[:, :, 1] == 2, axis=1)
        assert np.all(counts[~exterior] == 2)
        assert np.all(counts[open_x] == 1)
        assert np.all(counts <= 2)  # outer cut faces can have internal edges
    # Half-open ownership changes which instance supplies a cut face, but its
    # area and number of triangles per period must remain unchanged.
    assert len(fundamental.triangles) == len(mesh.triangles)
    def area(surface):
        p = surface.vertices[surface.triangles].astype(float)
        return np.linalg.norm(np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0]), axis=1).sum()/2
    assert area(fundamental) == area(mesh)


@pytest.mark.parametrize("reverse", [False, True])
def test_coplanar_clipping_roundoff_does_not_invent_a_shared_edge_split(reverse):
    # Cartesian facet from the 10 ms drainage state: its lower edge belongs
    # to the neighbour's liquid plane. FP64 residuals formerly inserted y=.5
    # on this facet alone, disconnecting seven pieces after FP32 export.
    polygon = np.asarray(((1, 0, .6550999975122078),
                          (1, 0, .6550827818166166),
                          (1, 1, .655082691715165),
                          (1, 1, .6550999975122078)))
    if reverse:
        polygon = polygon[::-1]
    normals = np.zeros((2, 1, 1, 3))
    offsets = np.ones((2, 1, 1))
    normals[1, 0, 0] = (-.11101879924535751, .000000080098487842, .8889811038970947)
    offsets[1, 0, 0] = .5823562145233154
    split = _split_edges(polygon, (0, 0, 0), (normals,), (offsets,),
                         (False, False, False), offsets.shape)
    np.testing.assert_array_equal(split, polygon)
    # A resolved crossing must still split both traversals of the polygon.
    normals[1, 0, 0] = (0, 1, 0)
    offsets[1, 0, 0] = .5
    split = _split_edges(polygon, (0, 0, 0), (normals,), (offsets,),
                         (False, False, False), offsets.shape)
    assert np.count_nonzero(split[:, 1] == .5) == 2


def test_one_ulp_wide_facet_keeps_its_split_boundary_edges():
    low = np.float32(.01655)
    high = np.nextafter(low, np.float32(np.inf))
    points = np.asarray(((.0625, 0, low), (.0625, .00625, low),
                         (.0625, .0125, low), (.0625, .0125, high),
                         (.0625, 0, high)), dtype=np.float32).astype(float)
    triangles = np.asarray(_triangulate(points, np.asarray((1, 0, 0))))
    edges = np.concatenate([triangles[:, pair] for pair in ((0, 1), (1, 2), (2, 0))])
    edges.sort(axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    expected = np.sort(np.asarray(((0, 1), (1, 2), (2, 3), (3, 4), (4, 0))), axis=1)
    assert set(map(tuple, unique[counts == 1])) == set(map(tuple, expected))


def test_crossings_that_round_to_one_corner_do_not_drop_polygon_area():
    points = np.asarray(((0, 0, 0), (1, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)), dtype=float)
    triangles = np.asarray(_triangulate(points, np.asarray((0, 0, 1))))
    p = points[triangles]
    assert np.sum(np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0])[:, 2])/2 == 1


@pytest.mark.parametrize("fraction", [.01, .49])
def test_thin_sheet_keeps_closed_volume_and_passive_attributes(fraction):
    s, plic = _sheet(fraction)
    alpha = s.get_interior("alpha")
    assert not len(extract_free_surface_mesh(alpha, s.cfg.grid).triangles)
    mesh = extract_liquid_surface_mesh(s, reconstruction=plic)
    g = s.cfg.grid
    expected = float(alpha.astype(float).sum())*g.dx*g.dy*g.dz
    assert _volume(mesh) == pytest.approx(expected, rel=5e-5)
    edges = np.concatenate([mesh.triangles[:, pair] for pair in ((0, 1), (1, 2), (2, 0))])
    edges.sort(axis=1)
    assert np.all(np.unique(edges, axis=0, return_counts=True)[1] == 2)
    assert np.all(np.isfinite(mesh.normals))
    np.testing.assert_allclose(mesh.velocity, np.tile((.03, -.02, .01), (len(mesh.vertices), 1)), atol=1e-8)
    np.testing.assert_allclose(mesh.foam, .2, atol=1e-7)
    np.testing.assert_array_equal(s.get_interior("alpha"), alpha)


def test_cut_bed_film_keeps_volume_and_contact_without_a_solid_air_face():
    bed = .0325
    s, plic = _sheet(.02, bed=bed)
    before = s.get_interior("alpha")
    mesh = extract_liquid_surface_mesh(s, reconstruction=plic)
    g = s.cfg.grid
    expected = float(np.sum(before.astype(float)*s.get_geometry_interior("theta")))*g.dx*g.dy*g.dz
    # The omitted planar solid face has zero signed volume about this origin.
    assert _volume(mesh, (0, 0, bed)) == pytest.approx(expected, rel=5e-5)
    assert mesh.vertices[:, 2].min() == pytest.approx(bed, abs=2e-8)
    assert mesh.vertices[:, 2].max() == pytest.approx(bed+.02*.75*g.dz, abs=2e-8)
    points = mesh.vertices[mesh.triangles]
    face_normals = np.cross(points[:, 1]-points[:, 0], points[:, 2]-points[:, 0])
    assert np.all(face_normals[:, 2] >= 0)  # no water-air face along the bed
    np.testing.assert_array_equal(s.get_interior("alpha"), before)


def test_extraction_does_not_bridge_a_dry_cell_between_thin_sheets():
    s, plic = _sheet(.01, separated=True)
    mesh = extract_liquid_surface_mesh(s, reconstruction=plic)
    edges = np.concatenate([mesh.triangles[:, pair] for pair in ((0, 1), (1, 2), (2, 0))])
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                       shape=(len(mesh.vertices), len(mesh.vertices))).tocsr()
    assert connected_components(graph, directed=False)[0] == 2
    x = mesh.vertices[mesh.triangles, 0]
    assert np.all((x.max(axis=1) <= np.float32(.03)) | (x.min(axis=1) >= np.float32(.04)))


def test_different_neighbour_planes_form_a_closed_volume_without_cracks():
    s, _ = _sheet(.1)
    normals = np.empty((*s.grid.cell_shape, 3), np.float32)
    normals[:] = (.15, .1, 1)
    normals[3, :, :, 1] = -.1
    plic = reconstruct_plic(s, wp.array(normals, dtype=wp.vec3, device=DEVICE))
    mesh = extract_liquid_surface_mesh(s, reconstruction=plic)
    edges = np.concatenate([mesh.triangles[:, pair] for pair in ((0, 1), (1, 2), (2, 0))])
    edges.sort(axis=1)
    assert np.all(np.unique(edges, axis=0, return_counts=True)[1] == 2)
    g = s.cfg.grid
    expected = float(s.get_interior("alpha").astype(float).sum())*g.dx*g.dy*g.dz
    assert _volume(mesh) == pytest.approx(expected, rel=5e-5)


def test_state_extraction_selects_the_liquid_boundary_and_leaves_state_unchanged():
    s, _ = _sheet(.01)
    before = {q: s.get_interior(q) for q in ("alpha", "u", "v", "w", "p", "foam")}
    mesh = extract_state_free_surface_mesh(s)
    assert len(mesh.triangles) > 0
    for q, values in before.items():
        np.testing.assert_array_equal(s.get_interior(q), values)
    with pytest.raises(ValueError, match="unsmoothed"):
        extract_state_free_surface_mesh(s, smoothing_iterations=1)
    with pytest.raises(ValueError, match="unsmoothed"):
        extract_state_free_surface_mesh(s, smoothing_iterations=1, render_surface="nodal_isosurface")
    with pytest.raises(ValueError, match="unknown render surface"):
        extract_state_free_surface_mesh(s, render_surface="smoothed_plic")
    legacy = State3D(Solver3DConfig(grid=s.cfg.grid, device=DEVICE))
    with pytest.raises(ValueError, match="liquid_mimetic"):
        extract_state_free_surface_mesh(legacy, render_surface="plic_boundary")


def _tilted_plane_state():
    """Bulk water under z = .0503+.2(x-.08)+.1(y-.06): 2.8-7.2 cells deep, walls on x and y."""
    g = Grid3DConfig(nx=16, ny=12, nz=12, Lx=.16, Ly=.12, Lz=.12)
    s = State3D(Solver3DConfig(grid=g, device=DEVICE, pressure_discretization="liquid_mimetic",
                               bc=BC3DConfig(ylo=BC.WALL, yhi=BC.WALL)))
    q = (np.arange(24)+.5)/24
    x = (np.arange(g.nx)[:, None]+q)*g.dx
    y = (np.arange(g.ny)[:, None]+q)*g.dy
    height = .0503+.2*(x[:, None, :, None]-.08)+.1*(y[None, :, None, :]-.06)
    z = np.arange(g.nz)*g.dz
    alpha = np.clip((height[..., None]-z)/g.dz, 0, 1).mean(axis=(2, 3))
    s.set_interior("alpha", alpha.astype(np.float32))
    for name, value in zip(("u", "v", "w"), (.03, -.02, .01), strict=True):
        s.set_interior(name, np.full(s.grid.interior_shape(name), value, np.float32))
    return s, alpha


def _column_top_heights(mesh, grid, samples=4):
    """Topmost mesh height above a samples x samples footprint of every column; NaN if uncovered."""
    q = (np.arange(samples)+.5)/samples
    x = ((np.arange(grid.nx)[:, None]+q)*grid.dx).ravel()
    y = ((np.arange(grid.ny)[:, None]+q)*grid.dy).ravel()
    px, py = np.meshgrid(x, y, indexing="ij")
    a, b, c = np.moveaxis(mesh.vertices[mesh.triangles].astype(float), 1, 0)
    det = (b[:, 1]-c[:, 1])*(a[:, 0]-c[:, 0])+(c[:, 0]-b[:, 0])*(a[:, 1]-c[:, 1])
    top = np.full(px.shape, np.nan)
    for t in np.flatnonzero(det != 0):
        u = ((b[t, 1]-c[t, 1])*(px-c[t, 0])+(c[t, 0]-b[t, 0])*(py-c[t, 1]))/det[t]
        v = ((c[t, 1]-a[t, 1])*(px-c[t, 0])+(a[t, 0]-c[t, 0])*(py-c[t, 1]))/det[t]
        inside = (u >= 0) & (v >= 0) & (u+v <= 1)
        z = u*a[t, 2]+v*b[t, 2]+(1-u-v)*c[t, 2]
        top[inside] = np.fmax(top[inside], z[inside])
    return top.reshape(grid.nx, samples, grid.ny, samples)


def test_nodal_isosurface_of_a_tilted_plane_has_no_grid_facets_and_keeps_column_volume():
    s, alpha = _tilted_plane_state()
    g = s.cfg.grid
    before = {name: s.field(name).numpy().copy() for name in s.FIELDS}
    before.update({name: getattr(s, name).numpy().copy()
                   for name in ("theta", "u_aperture", "v_aperture", "w_aperture",
                                "open_normals", "open_offsets")})
    mesh = extract_state_free_surface_mesh(s, render_surface="nodal_isosurface")
    for name, values in before.items():
        np.testing.assert_array_equal(getattr(s, name).numpy(), values)
    points = mesh.vertices[mesh.triangles]
    assert len(points) and not np.any(np.ptp(points, axis=1) == 0)  # no facet in an axis plane
    top = _column_top_heights(mesh, g)
    assert np.all(np.isfinite(top))  # every footprint sample sees the surface from above
    error = (top.mean(axis=(1, 3))*g.dx*g.dy-alpha.sum(axis=2)*g.dx*g.dy*g.dz)/(g.dx*g.dy*g.dz)
    assert np.max(np.abs(error[1:-1, 1:-1])) < .02
    # A wall node averages one-sidedly, so the first column is biased by half
    # a cell of the plane's slope (.2 and .1 here), never more.
    assert np.max(np.abs(error)) < .1
    np.testing.assert_allclose(mesh.velocity, np.tile((.03, -.02, .01), (len(mesh.vertices), 1)), atol=1e-6)
    # The default stays the retained PLIC boundary, bitwise, whichever name selects it.
    retained = extract_liquid_surface_mesh(s)
    for selected in (extract_state_free_surface_mesh(s),
                     extract_state_free_surface_mesh(s, render_surface="plic_boundary")):
        for field in ("vertices", "triangles", "normals", "velocity", "foam"):
            np.testing.assert_array_equal(getattr(selected, field), getattr(retained, field))
    assert np.any(np.ptp(retained.vertices[retained.triangles], axis=1) == 0)  # PLIC keeps its cell-face patches


@pytest.mark.parametrize("fraction, bed", [(1.0, .0325), (.5, None)])
def test_nodal_isosurface_cannot_show_a_one_cell_sheet_that_the_plic_boundary_keeps(fraction, bed):
    # A theta-weighted nodal average of a one-cell sheet never reaches .5: the
    # sheet is below the isosurface's resolution, while the PLIC boundary is exact.
    s, plic = _sheet(fraction, bed=bed)
    before = s.get_interior("alpha")
    g = s.cfg.grid
    assert not len(extract_state_free_surface_mesh(s, render_surface="nodal_isosurface").triangles)
    mesh = extract_liquid_surface_mesh(s, reconstruction=plic)
    expected = float(np.sum(before.astype(float)*s.get_geometry_interior("theta")))*g.dx*g.dy*g.dz
    assert _volume(mesh, (0, 0, bed or 0)) == pytest.approx(expected, rel=5e-5)
    np.testing.assert_array_equal(s.get_interior("alpha"), before)
