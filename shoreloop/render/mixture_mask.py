"""Geometric mixture classification of a PLIC-boundary mesh per plan column.

The saved frames of a run carry the exact PLIC-boundary mesh but not the
water fraction, so the per-cell resolvedness of AW-1113 part A (which needs
``alpha``) cannot be evaluated on them.  This module provides the geometric
equivalent on the solver's plan grid (``nx x ny`` columns of ``dx x dy``
metres, cells of height ``dz``; origin ``x0, y0, z0``; y periodic).

Definition.  For every plan column a vertical ray is cast through the column
centre ``(x0 + (i + 1/2) dx, y0 + (j + 1/2) dy)`` and its crossings with the
mesh are collected: one crossing per triangle whose plan footprint covers the
centre.  A half-open edge rule makes the footprints of a triangulated sheet a
partition of the plane, so one sheet yields exactly one crossing; triangles
that are edge-on in plan (the vertical stair risers) cover nothing.  Each
crossing carries the height ``z`` of the triangle at the centre, the sign
``+1`` for an upward-facing facet (water below) or ``-1`` for a
downward-facing one, the triangle index and the z extent of that triangle.

A column is *mixture* when the surface crosses it ``min_crossings`` (default
three) or more times within any window of length ``window`` (default four
cells, ``4 dz = 0.04 m``) along z, i.e. the water is not a single sheet there
but a fold, a lip, a sheet over a pocket, or stacked fragments.  The window
slides continuously; it is not aligned to cell faces.  Consecutive crossings
of one column closer than ``min_separation`` are one *event* for this count:
the union-of-polyhedra PLIC boundary carries sub-millimetre slivers wherever
two neighbouring planes miss each other at a cell face (facet, face patch,
facet within 0.1-0.3 mm), which are the plane-mismatch stairs themselves and
not a second sheet.  ``min_separation = 0`` counts every crossing.  Per
column the raw counts are exposed so the thresholds can be changed later:
the number of crossings, the number of events, the maximum number of events
in any window, the number of *stacked facets* (covering triangles whose z
extent overlaps the z extent of another covering triangle of the same
column: two sheets within one facet height), and optionally the maximum
number of interface cells (cells owning a sloped PLIC facet) in any
``window`` of the column, the geometric analogue of the stacked-partial-cell
count.  Columns whose cut-cell geometry is solid throughout, or unsupported
(solid above open space, the boulder overhangs, where a vertical ray meets
water that has no sky-facing surface by construction) are *excluded*.  The
remaining columns with at least one crossing are *wet*; wet columns that are
not mixture are *resolved*.

The reference this definition is validated against is the alpha-based column
criterion of part A, ``stacked_partial_columns``: three or more partial cells
(``low < alpha < high``, default ``0.02 < alpha < 0.98``, ``alpha`` the water
fraction of the open volume) within four consecutive cells of a column.  The
same count can be taken from the mesh alone: ``cell_fractions_from_facets``
recovers each interface cell's water fraction from its PLIC plane (exact to
float32 for uncut cells) and ``stacked_partial_from_mesh`` applies the rule to
it; the stair census carries it beside the crossing rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlanGrid:
    """The solver's Cartesian grid as seen from above; z only sets the window."""

    nx: int
    ny: int
    nz: int
    dx: float
    dy: float
    dz: float
    x0: float = 0.0
    y0: float = 0.0
    z0: float = 0.0

    @classmethod
    def from_configuration(cls, configuration: dict) -> PlanGrid:
        """Read a saved ``configuration.json`` (``scene.solver.grid``)."""
        grid = configuration["scene"]["solver"]["grid"]
        return cls(
            nx=int(grid["nx"]), ny=int(grid["ny"]), nz=int(grid["nz"]),
            dx=float(grid["Lx"]) / int(grid["nx"]),
            dy=float(grid["Ly"]) / int(grid["ny"]),
            dz=float(grid["Lz"]) / int(grid["nz"]),
            x0=float(grid["x0"]), y0=float(grid["y0"]), z0=float(grid["z0"]),
        )

    @property
    def shape(self) -> tuple[int, int]:
        return (self.nx, self.ny)

    @property
    def columns(self) -> int:
        return self.nx * self.ny

    def column_index(self, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        """Flat column index with periodic y."""
        return np.asarray(i) * self.ny + np.mod(np.asarray(j), self.ny)


@dataclass(frozen=True)
class ColumnCrossings:
    """All crossings of the column-centre rays, sorted by (column, z)."""

    grid: PlanGrid
    column: np.ndarray    # flat column index i*ny + j, int64
    z: np.ndarray         # height of the crossing (m)
    sign: np.ndarray      # +1 upward-facing facet (water below), -1 downward-facing
    triangle: np.ndarray  # index of the crossed triangle
    z_low: np.ndarray     # z extent of the crossed triangle
    z_high: np.ndarray

    def __len__(self) -> int:
        return int(len(self.z))

    def counts(self) -> np.ndarray:
        """Crossings per column, shape ``(nx, ny)``."""
        return np.bincount(self.column, minlength=self.grid.columns).reshape(self.grid.shape)


def cast_columns(vertices: np.ndarray, triangles: np.ndarray, grid: PlanGrid) -> ColumnCrossings:
    """Cast one vertical ray per column centre and collect every mesh crossing.

    Plan coordinates are measured in cells from the grid origin.  A triangle
    covers a centre when the centre is strictly inside its plan footprint or
    on an edge that the half-open rule assigns to it (for the footprint
    oriented counter-clockwise: an edge pointing in +y, or a horizontal edge
    pointing in -x).  Two triangles sharing an edge with consistent 3-D
    orientation traverse it in opposite directions in plan, so a centre on
    the shared edge belongs to exactly one of them.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    P = vertices[triangles]
    gx = (P[:, :, 0] - grid.x0) / grid.dx
    gy = (P[:, :, 1] - grid.y0) / grid.dy
    gz = P[:, :, 2]
    # Plan orientation: positive determinant <=> counter-clockwise <=> n_z > 0.
    det = (gx[:, 1] - gx[:, 0]) * (gy[:, 2] - gy[:, 0]) - (gx[:, 2] - gx[:, 0]) * (gy[:, 1] - gy[:, 0])
    sign = np.sign(det).astype(np.int8)
    # Orient every footprint counter-clockwise by swapping b and c where needed.
    swap = det < 0
    bx, cx = np.where(swap, gx[:, 2], gx[:, 1]), np.where(swap, gx[:, 1], gx[:, 2])
    by, cy = np.where(swap, gy[:, 2], gy[:, 1]), np.where(swap, gy[:, 1], gy[:, 2])
    bz, cz = np.where(swap, gz[:, 2], gz[:, 1]), np.where(swap, gz[:, 1], gz[:, 2])
    ax, ay, az = gx[:, 0], gy[:, 0], gz[:, 0]
    area2 = np.abs(det)

    # Candidate centres (i + 1/2, j + 1/2) inside the plan bounding box.
    i0 = np.ceil(np.minimum.reduce([ax, bx, cx]) - 0.5).astype(np.int64)
    i1 = np.floor(np.maximum.reduce([ax, bx, cx]) - 0.5).astype(np.int64)
    j0 = np.ceil(np.minimum.reduce([ay, by, cy]) - 0.5).astype(np.int64)
    j1 = np.floor(np.maximum.reduce([ay, by, cy]) - 0.5).astype(np.int64)
    i0 = np.maximum(i0, 0)
    i1 = np.minimum(i1, grid.nx - 1)
    ni = np.where(area2 > 0.0, np.maximum(i1 - i0 + 1, 0), 0)
    nj = np.where(area2 > 0.0, np.maximum(j1 - j0 + 1, 0), 0)
    total = ni * nj
    tri = np.repeat(np.arange(len(triangles)), total)
    if len(tri) == 0:
        empty = np.zeros(0)
        return ColumnCrossings(grid, empty.astype(np.int64), empty, empty.astype(np.int8),
                               empty.astype(np.int64), empty, empty)
    offsets = np.arange(len(tri)) - np.repeat(np.cumsum(total) - total, total)
    ii = i0[tri] + offsets // nj[tri]
    jj = j0[tri] + offsets % nj[tri]
    px, py = ii + 0.5, jj + 0.5

    def edge(x_from, y_from, x_to, y_to):
        """Edge function value and half-open inclusion of the edge itself."""
        dx_, dy_ = x_to - x_from, y_to - y_from
        value = dx_ * (py - y_from) - dy_ * (px - x_from)
        include = (dy_ > 0.0) | ((dy_ == 0.0) & (dx_ < 0.0))
        return value, (value > 0.0) | ((value == 0.0) & include)

    e_ab, in_ab = edge(ax[tri], ay[tri], bx[tri], by[tri])
    e_bc, in_bc = edge(bx[tri], by[tri], cx[tri], cy[tri])
    e_ca, in_ca = edge(cx[tri], cy[tri], ax[tri], ay[tri])
    inside = in_ab & in_bc & in_ca
    tri, ii, jj = tri[inside], ii[inside], jj[inside]
    w_a = e_bc[inside] / area2[tri]
    w_b = e_ca[inside] / area2[tri]
    w_c = e_ab[inside] / area2[tri]
    z = w_a * az[tri] + w_b * bz[tri] + w_c * cz[tri]
    column = grid.column_index(ii, jj)
    order = np.lexsort((z, column))
    tri, column, z = tri[order], column[order], z[order]
    return ColumnCrossings(
        grid=grid, column=column, z=z, sign=sign[tri], triangle=tri,
        z_low=gz.min(axis=1)[tri], z_high=gz.max(axis=1)[tri],
    )


@dataclass(frozen=True)
class ColumnMixture:
    """Per-column classification and the raw counts behind it, all ``(nx, ny)``."""

    grid: PlanGrid
    window: float
    min_crossings: int
    min_separation: float
    crossings: np.ndarray      # number of ray crossings
    events: np.ndarray         # crossings after merging neighbours closer than min_separation
    window_max: np.ndarray     # maximum events inside any z window of length ``window``
    stacked_facets: np.ndarray # covering triangles whose z extent overlaps another's in the column
    interface_cells: np.ndarray  # maximum interface cells in any window (zeros unless supplied)
    excluded: np.ndarray       # solid or unsupported (overhang) columns, never classified
    wet: np.ndarray            # at least one crossing and not excluded
    mixture: np.ndarray        # wet and window_max >= min_crossings
    resolved: np.ndarray       # wet and not mixture


def _events(crossings: ColumnCrossings, min_separation: float) -> tuple[np.ndarray, np.ndarray]:
    """Merge consecutive crossings of a column closer than ``min_separation``; returns (column, z)."""
    col, z = crossings.column, crossings.z
    if len(col) == 0:
        return col, z
    new = np.ones(len(col), bool)
    new[1:] = (col[1:] != col[:-1]) | (z[1:] - z[:-1] >= min_separation)
    event = np.cumsum(new) - 1
    size = np.bincount(event)
    return col[new], np.bincount(event, weights=z) / size


def _window_max(column: np.ndarray, z: np.ndarray, grid: PlanGrid, window: float) -> np.ndarray:
    """Maximum number of (column, z) events of one column within any z window; sorted input."""
    if len(column) == 0:
        return np.zeros(grid.shape, np.int32)
    span = 10.0 * max(1.0, abs(grid.z0) + grid.nz * grid.dz + window)
    key = column * span + z
    upper = np.searchsorted(key, key + window, side="right")
    per_event = upper - np.arange(len(key))
    out = np.zeros(grid.columns, np.int64)
    np.maximum.at(out, column, per_event)
    return out.reshape(grid.shape).astype(np.int32)


def grid_plane_axis(vertices: np.ndarray, triangles: np.ndarray, grid: PlanGrid, tolerance: float = 2e-4) -> np.ndarray:
    """Axis (0, 1, 2) of the grid plane a triangle lies in, -1 for a sloped facet.

    The rule of the stair census: all three vertices share one coordinate
    that is a whole number of cells from the origin, within ``tolerance``
    cells.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    P = vertices[triangles]
    origin = np.array([grid.x0, grid.y0, grid.z0])
    spacing = np.array([grid.dx, grid.dy, grid.dz])
    g = (P - origin) / spacing
    axis = np.full(len(triangles), -1)
    for q in range(3):
        r = np.round(g[:, :, q])
        on = np.all(np.abs(g[:, :, q] - r) < tolerance, axis=1) & (np.ptp(r, axis=1) == 0)
        axis[on & (axis < 0)] = q
    return axis


def interface_cells_per_column(vertices: np.ndarray, triangles: np.ndarray, grid: PlanGrid, window_cells: int = 4) -> np.ndarray:
    """Maximum number of interface cells within ``window_cells`` consecutive cells of each column.

    An interface cell is a cell owning at least one sloped (non-grid-plane)
    facet, located by the facet centroid.  This is the geometric analogue of
    the stacked-partial-cell count, but it counts every cut cell, including
    the ones below 2% and above 98% water.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    axis = grid_plane_axis(vertices, triangles, grid)
    centroid = vertices[triangles[axis < 0]].mean(axis=1)
    i = np.clip(np.floor((centroid[:, 0] - grid.x0) / grid.dx).astype(np.int64), 0, grid.nx - 1)
    j = np.mod(np.floor((centroid[:, 1] - grid.y0) / grid.dy).astype(np.int64), grid.ny)
    k = np.clip(np.floor((centroid[:, 2] - grid.z0) / grid.dz).astype(np.int64), 0, grid.nz - 1)
    cells = np.zeros(grid.shape + (grid.nz,), np.int32)
    cells[i, j, k] = 1
    cumulative = np.concatenate([np.zeros(grid.shape + (1,), np.int32), np.cumsum(cells, axis=2)], axis=2)
    width = min(window_cells, grid.nz)
    return (cumulative[:, :, width:] - cumulative[:, :, :grid.nz + 1 - width]).max(axis=2)


def _stacked_facets(crossings: ColumnCrossings) -> np.ndarray:
    """Covering triangles whose z extent strictly overlaps another one of the same column."""
    grid = crossings.grid
    n = len(crossings)
    if n == 0:
        return np.zeros(grid.shape, np.int32)
    # Crossings are sorted by column; offset keys keep every column's z range disjoint.
    span = 10.0 * max(1.0, abs(grid.z0) + grid.nz * grid.dz)
    col = crossings.column
    start = np.searchsorted(col, col, side="left")
    end = np.searchsorted(col, col, side="right")
    key_low = col * span + crossings.z_low
    key_high = col * span + crossings.z_high
    sorted_low, sorted_high = np.sort(key_low), np.sort(key_high)
    flat = (crossings.z_high <= crossings.z_low).astype(np.int64)   # a flat facet counts itself below and above
    below = np.searchsorted(sorted_high, key_low, side="right") - start - flat    # others entirely below me
    above = end - np.searchsorted(sorted_low, key_high, side="left") - flat      # others entirely above me
    overlapping = (end - start - 1) - below - above
    assert np.all(overlapping >= 0)
    stacked = overlapping > 0
    return np.bincount(col[stacked], minlength=grid.columns).reshape(grid.shape).astype(np.int32)


def excluded_columns(theta: np.ndarray) -> np.ndarray:
    """Columns that are solid throughout or unsupported (solid above open space)."""
    theta = np.asarray(theta)
    solid = np.all(theta <= 0.0, axis=2)
    unsupported = np.any(np.diff(theta, axis=2) < 0.0, axis=2)
    return solid | unsupported


def classify_columns(
    crossings: ColumnCrossings,
    theta: np.ndarray | None = None,
    window: float | None = None,
    min_crossings: int = 3,
    min_separation: float = 0.0,
    interface_cells: np.ndarray | None = None,
) -> ColumnMixture:
    """Apply the mixture definition of the module docstring to the ray crossings.

    ``theta`` is the open-volume fraction per cell ``(nx, ny, nz)``; without it
    no column is excluded.  ``window`` defaults to four cells.
    ``interface_cells`` (from ``interface_cells_per_column``) is carried as a
    raw count only; it does not enter the classification.
    """
    grid = crossings.grid
    if window is None:
        window = 4.0 * grid.dz
    counts = crossings.counts().astype(np.int32)
    event_column, event_z = _events(crossings, min_separation)
    events = np.bincount(event_column, minlength=grid.columns).reshape(grid.shape).astype(np.int32)
    window_max = _window_max(event_column, event_z, grid, window)
    stacked = _stacked_facets(crossings)
    if theta is None:
        excluded = np.zeros(grid.shape, bool)
    else:
        if tuple(np.shape(theta)[:2]) != grid.shape:
            raise ValueError(f"theta plan shape {np.shape(theta)[:2]} does not match the grid {grid.shape}")
        excluded = excluded_columns(theta)
    if interface_cells is None:
        interface_cells = np.zeros(grid.shape, np.int32)
    wet = (counts > 0) & ~excluded
    mixture = wet & (window_max >= min_crossings)
    return ColumnMixture(
        grid=grid, window=float(window), min_crossings=int(min_crossings), min_separation=float(min_separation),
        crossings=counts, events=events, window_max=window_max, stacked_facets=stacked,
        interface_cells=np.asarray(interface_cells, dtype=np.int32),
        excluded=excluded, wet=wet, mixture=mixture, resolved=wet & ~mixture,
    )


def _stacked_count(partial: np.ndarray, window_cells: int) -> np.ndarray:
    """Maximum number of flagged cells within ``window_cells`` consecutive cells of each column."""
    partial = partial.astype(np.int32)
    cumulative = np.concatenate([np.zeros(partial.shape[:2] + (1,), np.int32), np.cumsum(partial, axis=2)], axis=2)
    nz = partial.shape[2]
    width = min(window_cells, nz)
    return (cumulative[:, :, width:] - cumulative[:, :, :nz + 1 - width]).max(axis=2)


def stacked_partial_columns(
    alpha: np.ndarray,
    theta: np.ndarray,
    low: float = 0.02,
    high: float = 0.98,
    window_cells: int = 4,
    min_partial: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """The alpha-based reference: columns with ``min_partial`` or more partial
    cells (``low < alpha < high`` in open cells) within ``window_cells``
    consecutive cells.  Returns the mask and the per-column maximum count."""
    alpha = np.asarray(alpha, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    if alpha.shape != theta.shape:
        raise ValueError(f"alpha {alpha.shape} and theta {theta.shape} differ in shape")
    count = _stacked_count((alpha > low) & (alpha < high) & (theta > 0.0), window_cells)
    return count >= min_partial, count


def _box_fraction(m: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Volume of ``{xi in [0,1]^3 : m . xi <= a}`` for ``m >= 0`` (rows), by inclusion-exclusion
    over the corner simplices.  Components below ``1e-4`` are dropped (the plane barely
    depends on that axis; the error is at most the dropped component) so the formula never
    divides by a vanishing product."""
    pos = lambda x: np.maximum(x, 0.0)
    V = np.zeros(len(a))
    big = m >= 1e-4
    count = big.sum(axis=1)
    s = count == 3
    if s.any():
        m3, a3 = m[s], pos(a[s])
        M = m3.sum(axis=1)
        V[s] = (a3 ** 3 - (pos(a3[:, None] - m3) ** 3).sum(axis=1) + (pos(a3[:, None] - (M[:, None] - m3)) ** 3).sum(axis=1)
                - pos(a3 - M) ** 3) / (6.0 * m3.prod(axis=1))
    s = count == 2
    if s.any():
        m2 = np.sort(np.where(big[s], m[s], 0.0), axis=1)[:, 1:]
        a2 = pos(a[s])
        V[s] = (a2 ** 2 - (pos(a2[:, None] - m2) ** 2).sum(axis=1) + pos(a2 - m2.sum(axis=1)) ** 2) / (2.0 * m2.prod(axis=1))
    s = count == 1
    if s.any():
        V[s] = pos(a[s]) / m[s].max(axis=1)
    s = count == 0
    V[s] = np.where(a[s] >= 0.0, 1.0, 0.0)
    return np.clip(V, 0.0, 1.0)


def _fraction_under(l1_normal: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Fraction of a unit cell on the water side ``l1_normal . xi <= offset`` (rows)."""
    offset = offset - np.where(l1_normal < 0.0, l1_normal, 0.0).sum(axis=1)   # reflect the negative axes
    return _box_fraction(np.abs(l1_normal), offset)


def _cell_planes(vertices: np.ndarray, triangles: np.ndarray, grid: PlanGrid):
    """Recover one PLIC plane per interface cell from its sloped facets.

    Returns ``(cells, normal, l1_normal, offset, facet_area)``: the ``(n, 3)``
    cell indices owning a sloped facet (sorted by flat index), the physical
    unit water-to-air normal, the cell-local L1-normalised normal and offset
    (water where ``l1_normal . xi <= offset``, ``xi`` in cell units from the
    cell origin) taken from the cell's largest triangle, and the summed
    physical area of the cell's sloped triangles.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    axis = grid_plane_axis(vertices, triangles, grid)
    P = vertices[triangles[axis < 0]]
    origin = np.array([grid.x0, grid.y0, grid.z0])
    spacing = np.array([grid.dx, grid.dy, grid.dz])
    centroid = P.mean(axis=1)
    ijk = np.floor((centroid - origin) / spacing).astype(np.int64)
    ijk[:, 0] = np.clip(ijk[:, 0], 0, grid.nx - 1)
    ijk[:, 1] = np.mod(ijk[:, 1], grid.ny)
    ijk[:, 2] = np.clip(ijk[:, 2], 0, grid.nz - 1)
    n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    area2 = np.linalg.norm(n, axis=1)
    n /= np.maximum(area2, 1e-300)[:, None]
    # Plane in cell units: m . xi = d with xi = (x - cell origin) / spacing; water where m . xi <= d.
    m = n * spacing
    xi = (centroid - origin) / spacing - ijk
    d = (m * xi).sum(axis=1)
    scale = np.abs(m).sum(axis=1)
    l1_normal, offset = m / scale[:, None], d / scale
    flat = (ijk[:, 0] * grid.ny + ijk[:, 1]) * grid.nz + ijk[:, 2]
    order = np.lexsort((-area2, flat))
    flat = flat[order]
    first = np.r_[True, flat[1:] != flat[:-1]]
    group = np.cumsum(first) - 1
    facet_area = np.bincount(group, weights=0.5 * area2[order])
    keep = order[first]
    return ijk[keep], n[keep], l1_normal[keep], offset[keep], facet_area


def _clipped_fraction(
    l1_normal: np.ndarray, offset: np.ndarray, open_normal: np.ndarray, open_offset: np.ndarray,
    samples: int = 16, chunk: int = 20000,
) -> np.ndarray:
    """Volume of the unit cell under the water plane and inside the open half-space
    ``open_normal . xi <= open_offset`` (the solver's cut-cell convention), by a
    ``samples x samples`` plan midpoint rule along z.  Against the cell's own
    open volume this reproduces theta to 1e-3 (p99) on the production geometry."""
    s = (np.arange(samples) + 0.5) / samples
    X, Y = (axis.ravel() for axis in np.meshgrid(s, s, indexing="ij"))

    def bounds(normal, d):
        r = d[:, None] - normal[:, 0:1] * X[None, :] - normal[:, 1:2] * Y[None, :]
        nz = normal[:, 2:3]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = r / nz
        upper = np.where(nz > 0.0, t, np.where(nz < 0.0, 1.0, np.where(r >= 0.0, 1.0, 0.0)))
        lower = np.where(nz < 0.0, t, np.where(nz > 0.0, 0.0, np.where(r >= 0.0, 0.0, 1.0)))
        return lower, upper

    out = np.empty(len(offset))
    for start in range(0, len(offset), chunk):
        sl = slice(start, start + chunk)
        lo_w, up_w = bounds(l1_normal[sl], offset[sl])
        lo_o, up_o = bounds(open_normal[sl], open_offset[sl])
        length = np.minimum(np.minimum(up_w, up_o), 1.0) - np.maximum(np.maximum(lo_w, lo_o), 0.0)
        out[sl] = np.clip(length, 0.0, 1.0).mean(axis=1)
    return out


def _water_volume(
    l1_normal: np.ndarray, offset: np.ndarray, theta_cells: np.ndarray,
    open_normal: np.ndarray | None = None, open_offset: np.ndarray | None = None,
) -> np.ndarray:
    """Water volume (fraction of the full cell) under the plane: the box fraction in an uncut cell;
    in a cut cell the clipped volume when the open half-space is given, else the box fraction
    minus the solid share (which assumes the solid lies under the water plane)."""
    box = _fraction_under(l1_normal, offset)
    cut = theta_cells < 1.0
    if not cut.any():
        return box
    if open_normal is None:
        return np.where(cut, np.maximum(box - (1.0 - theta_cells), 0.0), box)
    volume = box.copy()
    volume[cut] = _clipped_fraction(l1_normal[cut], offset[cut], open_normal[cut], open_offset[cut])
    return np.minimum(volume, theta_cells)


def cell_fractions_from_facets(
    vertices: np.ndarray, triangles: np.ndarray, grid: PlanGrid, theta: np.ndarray | None = None,
    open_normals: np.ndarray | None = None, open_offsets: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Water fraction of every interface cell implied by its PLIC facet plane.

    Returns ``(cells, alpha)``: the ``(n, 3)`` cell indices owning a sloped
    facet and the fraction of the cell's open volume under that facet's plane
    (water lies on the side opposite the outward facet normal).  Exact up to
    float32 for uncut cells, whose PLIC plane is volume-matched to alpha.  Cut
    cells need ``theta`` and the solver's open half-space per cell
    (``open_normals``, ``open_offsets``, cell-local L1 convention) for the
    clipped volume; without the half-space the solid is assumed to lie under
    the water plane.
    """
    cells, _, l1_normal, offset, _ = _cell_planes(vertices, triangles, grid)
    if theta is None:
        return cells, _fraction_under(l1_normal, offset)
    i, j, k = cells.T
    theta_cells = np.asarray(theta, dtype=np.float64)[i, j, k]
    on = None if open_normals is None else np.asarray(open_normals, dtype=np.float64)[i, j, k]
    od = None if open_offsets is None else np.asarray(open_offsets, dtype=np.float64)[i, j, k]
    volume = _water_volume(l1_normal, offset, theta_cells, on, od)
    return cells, np.where(theta_cells > 0.0, np.clip(volume / np.maximum(theta_cells, 1e-12), 0.0, 1.0), 0.0)


def stacked_partial_from_mesh(
    vertices: np.ndarray,
    triangles: np.ndarray,
    grid: PlanGrid,
    theta: np.ndarray | None = None,
    low: float = 0.02,
    high: float = 0.98,
    window_cells: int = 4,
    min_partial: int = 3,
    open_normals: np.ndarray | None = None,
    open_offsets: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``stacked_partial_columns`` evaluated on the facet-implied cell fractions of a mesh."""
    cells, alpha = cell_fractions_from_facets(vertices, triangles, grid, theta, open_normals, open_offsets)
    partial = np.zeros(grid.shape + (grid.nz,), bool)
    keep = (alpha > low) & (alpha < high)
    partial[cells[keep, 0], cells[keep, 1], cells[keep, 2]] = True
    count = _stacked_count(partial, window_cells)
    return count >= min_partial, count


# The kernel's thresholds (shoreloop.physics3d.resolvedness): plicRDF's resolvedness cut-off and the
# per-neighbour RMS plane mismatch above which no single plane describes the stencil.
UNRESOLVED_ANGLE_DEG = 15.0
UNRESOLVED_RESIDUAL = 0.1


@dataclass(frozen=True)
class MeshResolvedness:
    """Per-interface-cell resolvedness recovered from the mesh, and the column rule built on it.

    ``cells`` are the cells owning a sloped facet, sorted by flat index
    ``(i * ny + j) * nz + k``.  ``angle_deg`` is the facet-area-weighted mean
    angle to the interface cells among the 26 neighbours (-1 without one);
    ``residual`` the RMS over the 27-stencil, divided by 26, of the cell's
    plane continued into each neighbour minus the neighbour's recovered water
    (in cell fractions); ``near_cut`` marks cells whose stencil holds a cut
    cell, where the recovery assumes the solid lies under the water plane.
    ``stacked_partial`` and ``stacked_unresolved`` follow the kernel's column
    rule on the recovered fractions and flags.
    """

    grid: PlanGrid
    cells: np.ndarray            # (n, 3) i, j, k
    fraction: np.ndarray         # (n,) recovered water fraction of the open volume
    normal: np.ndarray           # (n, 3) physical unit water-to-air normal
    offset: np.ndarray           # (n,) cell-local L1 plane offset
    facet_area: np.ndarray       # (n,) m^2
    angle_deg: np.ndarray        # (n,)
    residual: np.ndarray         # (n,)
    near_cut: np.ndarray         # (n,) bool
    partial: np.ndarray          # (n,) bool: low < fraction < high in an open cell
    unresolved: np.ndarray       # (n,) bool: angle or residual above its limit
    stacked_partial: np.ndarray      # (nx, ny)
    stacked_unresolved: np.ndarray   # (nx, ny)

    @property
    def flat(self) -> np.ndarray:
        g = self.grid
        return (self.cells[:, 0] * g.ny + self.cells[:, 1]) * g.nz + self.cells[:, 2]

    def mixture_columns(self, min_stacked: int = 2) -> np.ndarray:
        return self.stacked_unresolved >= min_stacked


def _stacked_unresolved_count(partial: np.ndarray, unresolved_partial: np.ndarray, window_cells: int) -> np.ndarray:
    """The kernel's ``stacked_unresolved``: the largest partial count over windows holding an unresolved partial cell."""
    nz = partial.shape[2]
    width = min(window_cells, nz)
    pad = np.zeros(partial.shape[:2] + (1,), np.int32)
    cp = np.concatenate([pad, np.cumsum(partial.astype(np.int32), axis=2)], axis=2)
    cu = np.concatenate([pad, np.cumsum(unresolved_partial.astype(np.int32), axis=2)], axis=2)
    count = cp[:, :, width:] - cp[:, :, :nz + 1 - width]
    flagged = (cu[:, :, width:] - cu[:, :, :nz + 1 - width]) > 0
    return np.where(flagged, count, 0).max(axis=2)


def water_at_cell_centres(crossings: ColumnCrossings, column: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Whether the mesh puts the centre of cell ``(column, k)`` in water: the nearest ray crossing
    above the centre is an upward-facing facet (water below it); no crossing above means air."""
    grid = crossings.grid
    span = 10.0 * max(1.0, abs(grid.z0) + grid.nz * grid.dz)
    keys = crossings.column * span + crossings.z
    centre = grid.z0 + (np.asarray(k) + 0.5) * grid.dz
    idx = np.searchsorted(keys, np.asarray(column) * span + centre, side="right")
    idx_c = np.minimum(idx, max(len(keys) - 1, 0))
    if len(keys) == 0:
        return np.zeros(len(idx), bool)
    return (idx < len(keys)) & (crossings.column[idx_c] == column) & (crossings.sign[idx_c] > 0)


def mesh_resolvedness(
    vertices: np.ndarray,
    triangles: np.ndarray,
    grid: PlanGrid,
    theta: np.ndarray | None = None,
    open_normals: np.ndarray | None = None,
    open_offsets: np.ndarray | None = None,
    crossings: ColumnCrossings | None = None,
    low: float = 0.02,
    high: float = 0.98,
    window_cells: int = 4,
    angle_limit: float = UNRESOLVED_ANGLE_DEG,
    residual_limit: float = UNRESOLVED_RESIDUAL,
    periodic_y: bool = True,
) -> MeshResolvedness:
    """The kernel's resolvedness measures and column rule, from the mesh alone.

    Each interface cell's plane comes from its largest sloped triangle.  A
    neighbour's water is its recovered volume when it owns a facet; a cell
    without a facet is full or empty according to the mesh itself at the cell
    centre (``water_at_cell_centres``: the nearest crossing of the column-
    centre ray above the centre is an upward-facing facet), which is exact
    for the mesh and differs from the state only by the sub-2% residue the
    solver keeps without a plane.  Solid neighbours and neighbours outside
    the domain in x or z contribute nothing, as in the kernel; y wraps.  Cut
    cells use the volume clipped by the solver's open half-space when
    ``open_normals``/``open_offsets`` are given (plan quadrature, 1e-3), else
    the box fraction minus the solid share; ``near_cut`` flags the cells
    whose stencil holds a cut cell.
    """
    cells, normal, l1_normal, offset, facet_area = _cell_planes(vertices, triangles, grid)
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    theta = np.ones((nx, ny, nz)) if theta is None else np.asarray(theta, dtype=np.float64)
    on = None if open_normals is None else np.asarray(open_normals, dtype=np.float64)
    od = None if open_offsets is None else np.asarray(open_offsets, dtype=np.float64)
    i, j, k = cells.T
    theta_cells = theta[i, j, k]
    volume_cells = _water_volume(l1_normal, offset, theta_cells,
                                 None if on is None else on[i, j, k], None if od is None else od[i, j, k])
    fraction = np.where(theta_cells > 0.0, np.clip(volume_cells / np.maximum(theta_cells, 1e-12), 0.0, 1.0), 0.0)
    flat = (i * ny + j) * nz + k
    n = len(cells)
    if crossings is None:
        crossings = cast_columns(vertices, triangles, grid)

    # The 27-stencil of every interface cell: neighbour cell indices, validity, and whether it is an interface cell.
    di, dj, dk = (np.indices((3, 3, 3)).reshape(3, -1) - 1)
    a = i[:, None] + di[None, :]
    b = j[:, None] + dj[None, :]
    c = k[:, None] + dk[None, :]
    valid = (a >= 0) & (a < nx) & (c >= 0) & (c < nz)
    if periodic_y:
        b = np.mod(b, ny)
    else:
        valid &= (b >= 0) & (b < ny)
    a, b, c = np.clip(a, 0, nx - 1), np.clip(b, 0, ny - 1), np.clip(c, 0, nz - 1)
    q_flat = (a * ny + b) * nz + c
    pos_c = np.minimum(np.searchsorted(flat, q_flat), n - 1)
    is_interface = valid & (flat[pos_c] == q_flat)
    q_theta = np.where(valid, theta[a, b, c], 0.0)
    live = valid & (q_theta > 0.0)

    # Water of the neighbours as a fraction of the full cell: recovered volumes for interface cells,
    # the mesh's in/out state at the centre times theta otherwise.
    q_water = np.where(is_interface, volume_cells[pos_c], 0.0)
    other = live & ~is_interface
    q_water[other] = water_at_cell_centres(crossings, a[other] * ny + b[other], c[other]) * q_theta[other]

    # Residual: the cell's own plane continued into every live stencil cell against that cell's water.
    shift = np.stack([di, dj, dk], axis=1).astype(np.float64)          # (27, 3)
    cont_offset = offset[:, None] - l1_normal @ shift.T                 # (n, 27)
    cont_normal = np.repeat(l1_normal, 27, axis=0)
    cut = live & (q_theta < 1.0)
    volume = _fraction_under(cont_normal, cont_offset.ravel()).reshape(n, 27)
    if cut.any():
        if on is None:
            volume[cut] = np.maximum(volume[cut] - (1.0 - q_theta[cut]), 0.0)
        else:
            volume[cut] = _clipped_fraction(cont_normal.reshape(n, 27, 3)[cut], cont_offset[cut],
                                            on[a[cut], b[cut], c[cut]], od[a[cut], b[cut], c[cut]])
    term = np.where(live, (volume - q_water) ** 2, 0.0)
    residual = np.sqrt(term.sum(axis=1) / 26.0)
    near_cut = cut.any(axis=1)

    # Angle: facet-area-weighted mean angle to the interface neighbours (self excluded).
    nb = is_interface.copy()
    nb[:, 13] = False
    w = np.where(nb, facet_area[pos_c], 0.0)
    nq = normal[pos_c]                                                  # (n, 27, 3)
    cross = np.linalg.norm(np.cross(normal[:, None, :], nq), axis=2)
    dot = (normal[:, None, :] * nq).sum(axis=2)
    angles = np.arctan2(cross, dot)
    weight = w.sum(axis=1)
    angle_deg = np.where(weight > 0.0, np.degrees((w * angles).sum(axis=1) / np.maximum(weight, 1e-300)), -1.0)

    partial = (fraction > low) & (fraction < high) & (theta_cells > 0.0)
    unresolved = (angle_deg > angle_limit) | (residual > residual_limit)
    partial_grid = np.zeros((nx, ny, nz), bool)
    partial_grid[cells[partial, 0], cells[partial, 1], cells[partial, 2]] = True
    unresolved_grid = np.zeros((nx, ny, nz), bool)
    sel = partial & unresolved
    unresolved_grid[cells[sel, 0], cells[sel, 1], cells[sel, 2]] = True
    return MeshResolvedness(
        grid=grid, cells=cells, fraction=fraction, normal=normal, offset=offset, facet_area=facet_area,
        angle_deg=angle_deg, residual=residual, near_cut=near_cut, partial=partial, unresolved=unresolved,
        stacked_partial=_stacked_count(partial_grid, window_cells),
        stacked_unresolved=_stacked_unresolved_count(partial_grid, unresolved_grid, window_cells),
    )


def facet_columns(vertices: np.ndarray, triangles: np.ndarray, grid: PlanGrid, axis: np.ndarray | None = None) -> np.ndarray:
    """The plan columns each triangle lies on, shape ``(n, 2)`` of flat indices.

    Facets lying in an x or y grid plane (``axis`` 0 or 1 as classified by the
    stair census) separate the two columns on either side of the plane; every
    other facet, including z shelves and ceilings, belongs to the column of
    its centroid and the second entry repeats the first.  Columns outside the
    domain in x are ``-1``; y is periodic.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    centroid = vertices[triangles].mean(axis=1)
    gx = (centroid[:, 0] - grid.x0) / grid.dx
    gy = (centroid[:, 1] - grid.y0) / grid.dy
    i = np.clip(np.floor(gx).astype(np.int64), 0, grid.nx - 1)
    j = np.mod(np.floor(gy).astype(np.int64), grid.ny)
    first = grid.column_index(i, j)
    second = first.copy()
    if axis is not None:
        axis = np.asarray(axis)
        on_x = axis == 0
        m = np.rint(gx[on_x]).astype(np.int64)
        left = np.where(m - 1 >= 0, grid.column_index(np.maximum(m - 1, 0), j[on_x]), -1)
        right = np.where(m <= grid.nx - 1, grid.column_index(np.minimum(m, grid.nx - 1), j[on_x]), -1)
        first[on_x], second[on_x] = left, right
        on_y = axis == 1
        m = np.rint(gy[on_y]).astype(np.int64)
        first[on_y] = grid.column_index(i[on_y], m - 1)
        second[on_y] = grid.column_index(i[on_y], m)
    return np.column_stack((first, second))
