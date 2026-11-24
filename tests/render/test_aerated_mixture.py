"""The aerated-mixture composite: what it replaces, keeps, paints and reports."""

import numpy as np
import pytest

from shoreloop.render.aerated_mixture import (
    AERATION_SWITCH, band_totals, boulder_film_triangles, classify_frame, compose_aerated_surface,
    persist_indicator, represented_volume, residue_triangles, swash_line_columns,
)
from shoreloop.render.mixture_mask import excluded_columns
from shoreloop.scene.geometry3d import RoundedBoulder
from shoreloop.render.mixture_mask import PlanGrid
from shoreloop.render.surface_mesh import WaterSurfaceMesh

GRID = PlanGrid(nx=12, ny=8, nz=10, dx=0.0125, dy=0.025, dz=0.01)


def _sheet(height, columns=None, foam=0.0, cells=1):
    """A horizontal sheet triangulated per column (``cells`` subdivisions)."""
    g = GRID
    vertices, triangles, foams = [], [], []
    for i in range(g.nx):
        for j in range(g.ny):
            if columns is not None and not columns[i, j]:
                continue
            base = len(vertices)
            for a in range(cells + 1):
                for b in range(cells + 1):
                    vertices.append((g.x0 + (i + a / cells) * g.dx, g.y0 + (j + b / cells) * g.dy, height))
                    foams.append(foam)
            for a in range(cells):
                for b in range(cells):
                    v = base + a * (cells + 1) + b
                    triangles.append((v, v + cells + 1, v + 1))
                    triangles.append((v + 1, v + cells + 1, v + cells + 2))
    vertices = np.asarray(vertices, np.float32)
    return WaterSurfaceMesh(
        vertices=vertices, triangles=np.asarray(triangles, np.int32),
        normals=np.tile(np.float32([0, 0, 1]), (len(vertices), 1)),
        velocity=np.zeros((len(vertices), 3), np.float32), foam=np.asarray(foams, np.float32),
        vertex_open_boundaries=np.zeros(len(vertices), np.uint8),
        open_boundary_edges=np.zeros((0, 2), np.int32), open_boundary_labels=np.zeros(0, np.uint8),
    )


def _theta():
    return np.ones((GRID.nx, GRID.ny, GRID.nz), np.float32)


def test_closure_off_returns_the_plain_mesh_unchanged():
    plic, nodal = _sheet(0.055, foam=0.3), _sheet(0.052, cells=2)
    none = np.zeros(GRID.shape, bool)
    out, report = compose_aerated_surface(plic, nodal, none, GRID, aeration_weight=0.0, foam_gate=False)
    assert out is plic and report["closure"] == "off"
    out, report = compose_aerated_surface(plic, nodal, none, GRID, composite="nodal_body",
                                          aeration_weight=0.0, foam_gate=False)
    assert out is nodal


def test_plic_resolved_composite_replaces_mixture_columns_and_keeps_films():
    plic = _sheet(0.055, foam=0.4)
    nodal_columns = np.ones(GRID.shape, bool)
    nodal_columns[6, 3] = False  # a film column the isosurface misses
    nodal = _sheet(0.052, columns=nodal_columns, cells=2)
    mixture = np.zeros(GRID.shape, bool)
    mixture[5:8, 2:5] = True
    out, report = compose_aerated_surface(plic, nodal, mixture, GRID, theta=_theta(), aeration_weight=1.0)
    column_area = GRID.dx * GRID.dy
    assert report["mixture_columns"] == 9
    assert report["plic_film_triangles_kept_in_mixture"] == 2
    np.testing.assert_allclose(report["plic_film_area_kept_m2"], column_area, rtol=1e-5)
    # Eight mixture columns lose their PLIC facets; the ring around the block keeps both.
    np.testing.assert_allclose(report["plic_area_dropped_m2"], 8 * column_area, rtol=1e-5)
    assert report["nodal_columns"] == 8
    assert report["ring_columns"] == 16  # the 5x5 frame around the block: nodal extends outward
    np.testing.assert_allclose(report["overlap_area_m2"], 16 * column_area, rtol=1e-5)
    np.testing.assert_allclose(report["aerated_area_m2"], 8 * column_area, rtol=1e-5)
    assert len(out.triangles) == len(plic.triangles) - 16 + 24 * 8
    # The aeration indicator paints the nodal surface of mixture columns only;
    # the gate keeps the solver's foam in mixture and swash-line columns.
    assert np.isclose(report["foam_mean_plic_in"], 0.4)
    assert report["gate_columns"] == 9 + 3 * GRID.ny
    x, y = out.vertices[:, 0], out.vertices[:, 1]
    plic_vertices = np.arange(len(out.vertices)) < report["plic_vertices"]
    assert np.all(out.foam[plic_vertices & (x < 4.5 * GRID.dx)] == 0.0)
    np.testing.assert_allclose(out.foam[plic_vertices & (x > 10.5 * GRID.dx)], 0.4, rtol=1e-6)
    inside = ~plic_vertices & (x > 5.2 * GRID.dx) & (x < 5.8 * GRID.dx) & (y > 2.2 * GRID.dy) & (y < 2.8 * GRID.dy)
    assert inside.any() and np.all(out.foam[inside] == 1.0)
    # The ring's nodal surface outside the block is not aerated and not gated in.
    ring_interior = ~plic_vertices & (x > 4.2 * GRID.dx) & (x < 4.8 * GRID.dx) & (y > 2.2 * GRID.dy) & (y < 2.8 * GRID.dy)
    assert ring_interior.any() and np.all(out.foam[ring_interior] == 0.0)
    volumes = report["represented_volume_m3"]
    # Represented water: the composite differs from the PLIC mesh only in mixture columns.
    per_column_plic = represented_volume(plic, GRID, _theta())
    per_column_out = represented_volume(out, GRID, _theta())
    assert np.array_equal(per_column_plic[~mixture], per_column_out[~mixture])
    assert abs(report["represented_volume_error_vs_plic"]["offshore"]) < 0.02
    assert volumes["plic"]["offshore"] == pytest.approx(0.055 // GRID.dz * GRID.dz * GRID.nx * GRID.ny * GRID.dx * GRID.dy, rel=1e-6)


def test_nodal_body_composite_keeps_plic_only_where_the_isosurface_is_absent():
    plic = _sheet(0.055)
    nodal_columns = np.ones(GRID.shape, bool)
    nodal_columns[9:, :] = False  # the isosurface stops short of the swash edge
    nodal = _sheet(0.052, columns=nodal_columns)
    mixture = np.zeros(GRID.shape, bool)
    out, report = compose_aerated_surface(plic, nodal, mixture, GRID, composite="nodal_body")
    assert report["nodal_columns"] == 8 * GRID.ny  # nodal alone
    assert report["plic_columns"] == 3 * GRID.ny  # films alone
    assert report["ring_columns"] == GRID.ny  # both: the films extend one column under the body
    assert report["aerated_area_m2"] == 0.0
    assert len(out.triangles) == report["nodal_triangles_used"] + (len(plic.triangles) - report["plic_triangles_dropped"])


def test_persisted_indicator_outlives_the_classification_and_gates_the_foam():
    plic, nodal = _sheet(0.055, foam=0.6), _sheet(0.052, cells=2, foam=0.6)
    mixture = np.zeros(GRID.shape, bool)
    mixture[5:8, 2:5] = True
    indicator = persist_indicator(None, mixture, 0.04, 0.8)
    assert np.array_equal(indicator, mixture.astype(float))
    gone = np.zeros(GRID.shape, bool)
    for _ in range(5):
        indicator = persist_indicator(indicator, gone, 0.04, 0.8)
    np.testing.assert_allclose(indicator[mixture], np.exp(-0.2 / 0.8))
    assert np.all(indicator[mixture] >= AERATION_SWITCH)
    out, report = compose_aerated_surface(plic, nodal, gone, GRID, indicator=indicator)
    # Geometry still follows the decayed indicator; the gated foam is the solver's foam times it.
    assert report["mixture_columns"] == 0 and report["aerated_columns"] == 9
    assert report["nodal_columns"] == 9
    x, y = out.vertices[:, 0], out.vertices[:, 1]
    nodal_vertices = np.arange(len(out.vertices)) >= report["plic_vertices"]
    inside = nodal_vertices & (x > 6.2 * GRID.dx) & (x < 6.8 * GRID.dx) & (y > 3.2 * GRID.dy) & (y < 3.8 * GRID.dy)
    np.testing.assert_allclose(out.foam[inside], 0.6 * np.exp(-0.2 / 0.8), rtol=1e-5)
    for _ in range(10):
        indicator = persist_indicator(indicator, gone, 0.04, 0.8)
    assert np.all(indicator < AERATION_SWITCH)
    out, report = compose_aerated_surface(plic, nodal, gone, GRID, indicator=indicator)
    assert report["aerated_columns"] == 0 and report["nodal_columns"] == 0


def test_residue_facets_and_footprint_columns_are_not_drawn():
    # A sheet at 0.055 m fills half of cell 5; in two columns it sits 0.1 mm
    # above the cell face, leaving 1% of a cell: residue by the solver's rule.
    heights = np.full(GRID.shape, 0.055)
    heights[2, 3] = heights[2, 4] = 0.0501
    heights[6, 4] = heights[6, 6] = 0.056  # a 3 mm film on a solid filling 30% of cell 5
    g = GRID
    vertices, triangles = [], []
    for i in range(g.nx):
        for j in range(g.ny):
            base = len(vertices)
            for a in (0, 1):
                for b in (0, 1):
                    vertices.append((g.x0 + (i + a) * g.dx, g.y0 + (j + b) * g.dy, heights[i, j]))
            triangles += [(base, base + 2, base + 1), (base + 1, base + 2, base + 3)]
    vertices = np.asarray(vertices, np.float32)
    plic = WaterSurfaceMesh(
        vertices=vertices, triangles=np.asarray(triangles, np.int32),
        normals=np.tile(np.float32([0, 0, 1]), (len(vertices), 1)),
        velocity=np.zeros((len(vertices), 3), np.float32), foam=np.zeros(len(vertices), np.float32),
        vertex_open_boundaries=np.zeros(len(vertices), np.uint8),
        open_boundary_edges=np.zeros((0, 2), np.int32), open_boundary_labels=np.zeros(0, np.uint8))
    theta = _theta()
    theta[9, :, 7:] = 0.0  # an overhang: solid above open space, a boulder-footprint column row
    # Two bed-like cut cells (solid below, 30% of cell 5) holding the film: one
    # against a boulder, one against the bed.
    open_normals = np.zeros(GRID.shape + (GRID.nz, 3), np.float32)
    open_offsets = np.zeros(GRID.shape + (GRID.nz,), np.float32)
    for i, j in ((6, 4), (6, 6)):
        theta[i, j, :5] = 0.0
        theta[i, j, 5] = 0.7
        open_normals[i, j, 5] = (0.0, 0.0, -1.0)
        open_offsets[i, j, 5] = -0.3
    boulder = RoundedBoulder(centre=(6.5 * GRID.dx, 4.5 * GRID.dy, 0.03), radii=(0.02, 0.02, 0.02))
    nodal_columns = np.ones(GRID.shape, bool)
    nodal_columns[9:, :] = False  # the isosurface has nothing in the footprint row or beyond
    nodal = _sheet(0.052, columns=nodal_columns)
    result = classify_frame(plic, GRID, theta, open_normals, open_offsets)
    residue = residue_triangles(plic, GRID, result, theta)
    assert residue.sum() == 4
    film = boulder_film_triangles(plic, GRID, result, theta, [boulder])
    assert film.sum() == 2
    x, y = plic.vertices[plic.triangles[film]].mean(axis=1)[:, 0], plic.vertices[plic.triangles[film]].mean(axis=1)[:, 1]
    assert np.all((x > 6 * GRID.dx) & (x < 7 * GRID.dx) & (y > 4 * GRID.dy) & (y < 5 * GRID.dy))
    footprint = excluded_columns(theta)
    assert footprint.sum() == GRID.ny and footprint[9].all()
    out, report = compose_aerated_surface(
        plic, nodal, np.zeros(GRID.shape, bool), GRID, composite="nodal_body",
        residue=residue, footprint=footprint, boulder_film=film, theta=theta)
    assert report["residue_triangles_not_drawn"] == 4
    assert report["boulder_film_triangles_not_drawn"] == 2
    assert "upper_beach" in report["represented_volume_error_vs_plic_outside_footprint"]
    assert report["footprint_triangles_not_drawn"] == 2 * GRID.ny
    column_area = GRID.dx * GRID.dy
    np.testing.assert_allclose(report["residue_area_m2"], 2 * column_area, rtol=1e-5)
    np.testing.assert_allclose(report["footprint_area_m2"], GRID.ny * column_area, rtol=1e-5)
    x = out.vertices[:, 0]
    plic_vertices = np.arange(len(out.vertices)) < report["plic_vertices"]
    assert not np.any(plic_vertices & (x > 9.1 * GRID.dx) & (x < 9.9 * GRID.dx))
    assert np.any(plic_vertices & (x > 10.1 * GRID.dx))  # films beyond the footprint stay


def test_swash_line_marks_the_up_beach_edge():
    wet = np.zeros((6, 2), bool)
    wet[:4, 0] = True
    wet[1:3, 1] = True
    line = swash_line_columns(wet, 1)
    assert line[:, 0].tolist() == [False, False, True, True, False, False]
    assert line[:, 1].tolist() == [False, True, True, False, False, False]


def test_band_totals_split_columns_by_centre():
    per_column = np.ones(GRID.shape)
    totals = band_totals(per_column, GRID, {"low": (0.0, 0.05), "high": (0.05, 1.0)})
    assert totals["low"] == 4 * GRID.ny and totals["high"] == 8 * GRID.ny
