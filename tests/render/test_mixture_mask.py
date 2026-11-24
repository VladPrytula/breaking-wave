"""Governing properties of the mesh-based mixture mask (AW-1113 part B).

A single tilted sheet gives one crossing per covered column and no mixture
column; a folded sheet (an overturning lip crossed three times by the ray)
is flagged; the ray-cast crossing counts are exact on both meshes.
"""

from __future__ import annotations

import numpy as np

from shoreloop.render.mixture_mask import (
    PlanGrid,
    cast_columns,
    classify_columns,
    facet_columns,
    stacked_partial_columns,
)

GRID = PlanGrid(nx=24, ny=12, nz=40, dx=0.0125, dy=6.0 / 224, dz=0.01)


def _extrude(profile_x: np.ndarray, profile_z: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate the strip swept by an x-z polyline along y, consistently oriented."""
    ns, ny = len(profile_x), len(y)
    X, Y = np.meshgrid(profile_x, y, indexing="ij")
    Z = np.broadcast_to(profile_z[:, None], (ns, ny))
    vertices = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    index = np.arange(ns * ny).reshape(ns, ny)
    a, b, c, d = index[:-1, :-1].ravel(), index[1:, :-1].ravel(), index[1:, 1:].ravel(), index[:-1, 1:].ravel()
    triangles = np.concatenate([np.column_stack((a, b, c)), np.column_stack((a, c, d))])
    return vertices.astype(np.float32), triangles.astype(np.int32)


def _polyline_crossings(profile_x: np.ndarray, x: float) -> int:
    """Number of times the polyline x(s) passes the abscissa x (the reference count)."""
    left, right = profile_x[:-1], profile_x[1:]
    return int(np.sum(((left <= x) & (x < right)) | ((right <= x) & (x < left))))


def test_single_tilted_sheet_has_one_crossing_per_column_and_no_mixture():
    x = np.linspace(0.02, 0.28, 9)            # a plane z = 0.12 + 0.3 x, open at both ends
    z = 0.12 + 0.3 * x
    y = np.linspace(0.0, GRID.ny * GRID.dy, 7)   # covers every y column
    vertices, triangles = _extrude(x, z, y)
    crossings = cast_columns(vertices, triangles, GRID)
    counts = crossings.counts()
    centres_x = GRID.x0 + (np.arange(GRID.nx) + 0.5) * GRID.dx
    covered = (centres_x >= x[0]) & (centres_x < x[-1])
    assert np.array_equal(counts[covered], np.ones((covered.sum(), GRID.ny), int))
    assert np.all(counts[~covered] == 0)
    assert np.all(crossings.sign == 1)
    expected_z = 0.12 + 0.3 * centres_x[crossings.column // GRID.ny]
    assert np.allclose(crossings.z, expected_z, atol=1e-6)
    mask = classify_columns(crossings)
    assert mask.mixture.sum() == 0
    assert mask.resolved.sum() == covered.sum() * GRID.ny
    assert np.all(mask.stacked_facets == 0)


def test_folded_sheet_is_flagged_with_exact_crossing_counts():
    s = np.linspace(0.0, 1.0, 161)
    x = 0.04 + 0.2 * s + 0.06 * np.sin(2.0 * np.pi * s)   # a Z-fold: dx/ds < 0 for s in (0.34, 0.66)
    z = 0.15 + 0.04 * s                                     # the whole sheet rises four cells
    y = np.linspace(0.0, GRID.ny * GRID.dy, 5)
    vertices, triangles = _extrude(x, z, y)
    crossings = cast_columns(vertices, triangles, GRID)
    counts = crossings.counts()
    centres_x = GRID.x0 + (np.arange(GRID.nx) + 0.5) * GRID.dx
    expected = np.array([_polyline_crossings(x, cx) for cx in centres_x])
    assert expected.max() == 3 and (expected == 1).any()
    assert np.array_equal(counts, np.repeat(expected[:, None], GRID.ny, axis=1))
    # The lip's underside faces down between two upward-facing crossings.
    folded = np.flatnonzero(expected == 3)[0] * GRID.ny
    signs = crossings.sign[crossings.column == folded]
    assert signs.tolist() == [1, -1, 1]
    mask = classify_columns(crossings)
    assert np.array_equal(mask.mixture, np.repeat((expected == 3)[:, None], GRID.ny, axis=1))
    assert np.array_equal(mask.window_max, counts)
    assert np.array_equal(mask.resolved, np.repeat((expected == 1)[:, None], GRID.ny, axis=1))
    # A window shorter than the lip no longer sees three crossings together.
    thin = classify_columns(crossings, window=0.005)
    assert thin.mixture.sum() == 0


def test_stacked_facets_count_covering_triangles_with_overlapping_z_extents():
    column_x, column_y = 2.5 * GRID.dx, 1.5 * GRID.dy
    steep = np.array([[column_x - 0.005, column_y - 0.01, 0.10], [column_x + 0.005, column_y - 0.01, 0.13],
                      [column_x, column_y + 0.01, 0.115]], np.float32)           # z extent 0.10..0.13
    flat = np.array([[column_x - 0.005, column_y - 0.01, 0.12], [column_x + 0.005, column_y - 0.01, 0.12],
                     [column_x, column_y + 0.01, 0.12]], np.float32)             # inside that extent
    high = np.array([[column_x - 0.005, column_y - 0.01, 0.20], [column_x + 0.005, column_y - 0.01, 0.20],
                     [column_x, column_y + 0.01, 0.21]], np.float32)             # clear of both
    vertices = np.concatenate([steep, flat, high])
    triangles = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], np.int32)
    mask = classify_columns(cast_columns(vertices, triangles, GRID))
    assert mask.crossings[2, 1] == 3 and mask.crossings.sum() == 3
    assert mask.stacked_facets[2, 1] == 2
    assert mask.window_max[2, 1] == 2 and not mask.mixture[2, 1] and mask.resolved[2, 1]


def test_sliver_triple_merges_below_the_separation_and_interface_cells_are_counted():
    column_x, column_y = 2.5 * GRID.dx, 1.5 * GRID.dy
    tri = lambda z: [[column_x - 0.005, column_y - 0.01, z[0]], [column_x + 0.005, column_y - 0.01, z[1]],
                     [column_x, column_y + 0.01, z[2]]]
    # A plane-mismatch sliver: facet 0.2 mm under the cell face, a ceiling patch on the face, the next facet above.
    vertices = np.array(tri([0.0998, 0.0998, 0.0998]) + tri([0.1, 0.1, 0.1])[::-1] + tri([0.103, 0.103, 0.103]), np.float32)
    triangles = np.arange(9).reshape(3, 3)
    crossings = cast_columns(vertices, triangles, GRID)
    assert crossings.sign.tolist() == [1, -1, 1]
    exact = classify_columns(crossings)
    assert exact.crossings[2, 1] == 3 and exact.events[2, 1] == 3 and exact.mixture[2, 1]
    merged = classify_columns(crossings, min_separation=1e-3)
    assert merged.crossings[2, 1] == 3 and merged.events[2, 1] == 2 and merged.window_max[2, 1] == 2
    assert not merged.mixture[2, 1] and merged.resolved[2, 1]
    # Interface cells: three sloped facets in cells k = 9, 10, 12 of one column, none in the grid planes.
    from shoreloop.render.mixture_mask import interface_cells_per_column
    sloped = np.array(tri([0.092, 0.095, 0.097]) + tri([0.102, 0.105, 0.107]) + tri([0.122, 0.125, 0.127]), np.float32)
    cells = interface_cells_per_column(sloped, triangles, GRID)
    assert cells[2, 1] == 3 and cells.sum() == 3
    assert interface_cells_per_column(vertices, triangles, GRID)[2, 1] == 2   # the face patch is a grid plane


def test_excluded_columns_follow_theta_and_facets_map_to_columns():
    theta = np.ones(GRID.shape + (GRID.nz,), np.float32)
    theta[3, :, :] = 0.0                 # a solid column
    theta[5, :, 20:] = 0.0               # solid above open space: an overhang
    theta[7, :, 10:15] = 0.0             # open below solid below open: an overhang
    x = np.linspace(0.0, GRID.nx * GRID.dx, 5)
    vertices, triangles = _extrude(x, np.full(5, 0.05), np.linspace(0.0, GRID.ny * GRID.dy, 3))
    mask = classify_columns(cast_columns(vertices, triangles, GRID), theta=theta)
    assert mask.excluded[3].all() and mask.excluded[5].all() and mask.excluded[7].all()
    assert mask.excluded.sum() == 3 * GRID.ny
    assert not mask.excluded[0].any() and mask.wet[0].all()
    columns = facet_columns(vertices, triangles, GRID)
    centroid = vertices.astype(np.float64)[triangles].mean(axis=1)
    assert np.array_equal(columns[:, 0], columns[:, 1])
    assert np.array_equal(columns[:, 0] // GRID.ny, np.floor(centroid[:, 0] / GRID.dx).astype(int))
    # A facet in the x grid plane between columns 4 and 5 separates those two columns.
    riser = np.array([[5 * GRID.dx, 0.0, 0.0], [5 * GRID.dx, GRID.dy, 0.0], [5 * GRID.dx, 0.0, 0.01]], np.float32)
    pair = facet_columns(riser, np.array([[0, 1, 2]]), GRID, axis=np.array([0]))
    assert pair.tolist() == [[4 * GRID.ny, 5 * GRID.ny]]


def test_facet_planes_give_the_cell_water_fractions():
    from shoreloop.render.mixture_mask import cell_fractions_from_facets, stacked_partial_from_mesh
    x = np.linspace(0.0, 0.3, 25)                    # z = 0.1 + 0.3 x, one facet plane per cell crossed
    z = 0.1 + 0.3 * x
    vertices, triangles = _extrude(x, z, np.linspace(0.0, GRID.ny * GRID.dy, 3))
    cells, alpha = cell_fractions_from_facets(vertices, triangles, GRID)
    # Reference: the fraction of each cell under the plane, integrated over x with a fine midpoint rule.
    for (i, j, k), a in zip(cells, alpha):
        xs = i * GRID.dx + (np.arange(2000) + 0.5) / 2000 * GRID.dx
        expected = np.clip((0.1 + 0.3 * xs - k * GRID.dz) / GRID.dz, 0.0, 1.0).mean()
        assert abs(a - expected) < 2e-4, (i, k, a, expected)
    # A level facet at 30% of a cell's height, water below, gives alpha 0.3; the same facet flipped (water above) gives 0.7.
    tri = np.array([[0.05, 0.05, 0.103], [0.06, 0.05, 0.103], [0.055, 0.06, 0.103]], np.float32)
    _, below = cell_fractions_from_facets(tri, np.array([[0, 1, 2]]), GRID)
    _, above = cell_fractions_from_facets(tri, np.array([[0, 2, 1]]), GRID)
    assert abs(below[0] - 0.3) < 1e-6 and abs(above[0] - 0.7) < 1e-6
    # The stacked-partial rule on the mesh: a sheet rising 0.375 dz per cell holds at most two partial cells per column.
    mask, count = stacked_partial_from_mesh(vertices, triangles, GRID)
    assert not mask.any() and 1 <= count.max() <= 2


def _extrude_per_cell(profile_x: np.ndarray, profile_z: np.ndarray, grid: PlanGrid) -> tuple[np.ndarray, np.ndarray]:
    """Sweep an x-z polyline along y with every triangle inside one cell, as the PLIC boundary is."""
    V, T = [], []
    ys = np.arange(grid.ny + 1) * grid.dy
    for (x0, z0), (x1, z1) in zip(zip(profile_x[:-1], profile_z[:-1]), zip(profile_x[1:], profile_z[1:])):
        cuts = {0.0, 1.0}
        for lo, hi, d in ((x0, x1, grid.dx), (z0, z1, grid.dz)):
            if hi != lo:
                for m in range(int(np.floor(min(lo, hi) / d)) + 1, int(np.ceil(max(lo, hi) / d))):
                    cuts.add((m * d - lo) / (hi - lo))
        ts = sorted(t for t in cuts if 0.0 <= t <= 1.0)
        for ta, tb in zip(ts[:-1], ts[1:]):
            if tb - ta < 1e-9:
                continue
            xa, za = x0 + ta * (x1 - x0), z0 + ta * (z1 - z0)
            xb, zb = x0 + tb * (x1 - x0), z0 + tb * (z1 - z0)
            for j in range(grid.ny):
                base = len(V)
                V += [(xa, ys[j], za), (xb, ys[j], zb), (xb, ys[j + 1], zb), (xa, ys[j + 1], za)]
                T += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
    return np.array(V, np.float32), np.array(T, np.int32)


def test_mesh_resolvedness_reproduces_the_kernel_rule_on_synthetic_sheets():
    from shoreloop.render.mixture_mask import mesh_resolvedness, water_at_cell_centres
    # One tilted sheet: every interface cell's neighbours share its plane, so angle and residual vanish.
    x = np.linspace(0.0, GRID.nx * GRID.dx, 49)
    vertices, triangles = _extrude_per_cell(x, 0.12 + 0.3 * x, GRID)
    sheet = mesh_resolvedness(vertices, triangles, GRID)
    interior = (sheet.cells[:, 0] > 0) & (sheet.cells[:, 0] < GRID.nx - 1)
    assert np.all(sheet.angle_deg[interior] < 1e-3) and np.all(sheet.residual[interior] < 1e-3)
    assert not sheet.unresolved[interior].any() and sheet.mixture_columns(2).sum() == 0
    assert sheet.stacked_partial.max() <= 2 and sheet.stacked_unresolved.max() == 0
    # The mesh state at cell centres: below the sheet water, above it air.
    crossings = cast_columns(vertices, triangles, GRID)
    column = np.array([5 * GRID.ny + 2, 5 * GRID.ny + 2])
    z_sheet = 0.12 + 0.3 * (5.5 * GRID.dx)
    k = np.array([int(z_sheet / GRID.dz) - 2, int(z_sheet / GRID.dz) + 2])
    assert water_at_cell_centres(crossings, column, k).tolist() == [True, False]
    # Two level sheets two cells apart (k = 10 at 40% and k = 12 at 60%): the normals agree (angle 0),
    # the lower plane continued upward calls the cell between them air while the mesh puts water there
    # (under the upper sheet), so the lower cells are unresolved by the residual and every column holds
    # two partial cells with an unresolved one: a mixture column under the kernel's rule at 2, not at 3.
    lower, tri_lower = _extrude_per_cell(x, np.full(len(x), 0.104), GRID)
    upper, tri_upper = _extrude_per_cell(x, np.full(len(x), 0.126), GRID)
    stacked = mesh_resolvedness(np.concatenate([lower, upper]), np.concatenate([tri_lower, tri_upper + len(lower)]), GRID)
    assert np.all(stacked.angle_deg[stacked.angle_deg >= 0] < 1e-6)
    low_cells, high_cells = stacked.cells[:, 2] == 10, stacked.cells[:, 2] == 12
    assert low_cells.sum() == high_cells.sum() == GRID.nx * GRID.ny
    assert np.allclose(stacked.fraction[low_cells], 0.4, atol=1e-5) and np.allclose(stacked.fraction[high_cells], 0.6, atol=1e-5)
    assert stacked.residual[low_cells].min() > 0.1 and stacked.unresolved[low_cells].all()
    assert stacked.residual[high_cells].max() < 1e-3 and not stacked.unresolved[high_cells].any()
    assert np.all(stacked.stacked_partial == 2) and np.all(stacked.stacked_unresolved == 2)
    assert stacked.mixture_columns(2).all() and not stacked.mixture_columns(3).any()
    # The folded sheet of the crossing test is flagged along its lip and nowhere else.
    s = np.linspace(0.0, 1.0, 161)
    xf = 0.04 + 0.2 * s + 0.06 * np.sin(2.0 * np.pi * s)
    vf, tf = _extrude_per_cell(xf, 0.15 + 0.04 * s, GRID)
    fold = mesh_resolvedness(vf, tf, GRID)
    centres_x = GRID.x0 + (np.arange(GRID.nx) + 0.5) * GRID.dx
    triple = np.array([_polyline_crossings(xf, cx) for cx in centres_x]) == 3
    flagged = fold.mixture_columns(2).any(axis=1)
    assert flagged[triple].all()
    assert not flagged[centres_x < xf.min() + 0.03].any() and not flagged[centres_x > xf.max() - 0.03].any()


def test_stacked_partial_reference_counts_partial_cells_in_a_window():
    alpha = np.zeros((2, 1, 12))
    theta = np.ones_like(alpha)
    alpha[0, 0, :4] = 1.0
    alpha[0, 0, 4] = 0.5                 # one partial cell: a resolved sheet
    alpha[1, 0, :2] = 1.0
    alpha[1, 0, 2:7] = [0.9, 0.3, 0.6, 0.01, 0.5]   # four mid-range cells within five, three within four
    mask, count = stacked_partial_columns(alpha, theta)
    assert mask.tolist() == [[False], [True]]
    assert count.tolist() == [[1], [3]]
