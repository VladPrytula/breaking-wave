"""Aerated-mixture presentation of a saved frame (AW-1114). Rendered only.

The frame's exact PLIC boundary mesh and its theta-weighted nodal isosurface
are combined per plan column by the mixture classification of AW-1113. Two
named composites:

* ``plic_resolved``: resolved water keeps the exact PLIC boundary; mixture
  columns are drawn through the nodal isosurface; PLIC facets stay in mixture
  columns where the nodal mesh has no surface (sub-half-cell films, spray).
* ``nodal_body``: the nodal isosurface everywhere it exists; PLIC facets only
  in columns the nodal mesh does not cover (the leading swash films).

At a class boundary the guest surface (the nodal isosurface in
``plic_resolved``, the PLIC films in ``nodal_body``) extends one column into
the host's columns, so the composite has no hole where the two surfaces meet
at different heights; the overlap area is reported.

The aeration indicator is the mixture mask carried in time: a column's value
is ``max(mixture, previous * exp(-dt / persistence))`` (``persist_indicator``),
so entrained air outlives the instant the planes disagree, as it does
physically; the persistence time is a declared closure parameter and the
geometry follows the indicator above ``AERATION_SWITCH``. The indicator gates
the existing foam shading: the composite's ``foam`` attribute is the solver's
passive foam times the indicator (plus the swash line, where the foam field
is its own evidence), plus ``aeration_weight * indicator`` (zero by default:
the whiteness is the solver's foam where the solver holds a mixture). Every
weight and count is returned in the report; with the closure off and an empty
mixture mask the plain mesh is returned unchanged. The hydrodynamic state is
never read or changed.

Two facet rules apply before composition, both from the solver's own
definitions. Residue rule: a PLIC facet whose cell holds less than
``RESIDUE_FRACTION`` of a cell of water is not drawn; that liquid is the
unresolved residue of the ownership rule (no face velocity, no pressure
unknown), and the mesh builder's plane for it is a picture of nothing the
solver governs. Footprint rule: in boulder-footprint columns (solid above
open space) no PLIC facet is kept where the isosurface has nothing; the
sub-half-cell water on the rock is shown by the solid-wetness shading.
Boulder-film rule: a facet of a cut cell against a boulder holding less than
``BOULDER_FILM_FRACTION`` of a cell is likewise left to the wetness shading
(a draining film on the rock, not a free surface); films on the bed keep
their facets, since the swash edge is made of them. Every rule reports the
facets and the water it leaves undrawn.

Represented water is measured the same way for every mesh: a column cell
holds water when the mesh puts its centre in water (``water_at_cell_centres``)
and contributes ``theta * dx * dy * dz``; this is bias-free for a smooth
surface and lets the composite be compared with the exact PLIC mesh per band.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from shoreloop.render.mixture_mask import (
    MeshResolvedness, PlanGrid, cast_columns, excluded_columns, mesh_resolvedness,
    water_at_cell_centres,
)
from shoreloop.render.surface_mesh import WaterSurfaceMesh
from shoreloop.scene.geometry3d import RoundedBoulder

COMPOSITES = ("plic_resolved", "nodal_body")
# Closure weights: extra foam coverage painted on aerated columns (off: the
# solver's gated foam is the whiteness) and the indicator level above which a
# column is drawn through the nodal isosurface (the persistence half-life).
AERATION_WEIGHT = 0.0
AERATION_SWITCH = 0.5
# Swash line: wet columns within this many columns of the up-beach wet edge.
SWASH_COLUMNS = 2
# Mixture columns: the AW-1113 rule, two stacked partial cells one unresolved.
MIXTURE_STACKED = 2
# Residue: water below this share of a Cartesian cell has no governing surface
# (the production resolved fraction of the ownership rule).
RESIDUE_FRACTION = 0.02
# Water below this share of a cut cell against a boulder is a film on the rock.
BOULDER_FILM_FRACTION = 0.5
BANDS = {"offshore": (0.0, 0.8), "forcing": (0.8, 1.58), "bore": (1.58, 2.6), "upper_beach": (2.6, 1e9)}


def triangle_columns(mesh: WaterSurfaceMesh, grid: PlanGrid) -> tuple[np.ndarray, np.ndarray]:
    """Column (i, j) of every triangle by its centroid; j periodic, i clipped."""
    centroid = mesh.vertices[mesh.triangles].astype(np.float64).mean(axis=1)
    i = np.clip(np.floor((centroid[:, 0] - grid.x0) / grid.dx).astype(int), 0, grid.nx - 1)
    j = np.mod(np.floor((centroid[:, 1] - grid.y0) / grid.dy).astype(int), grid.ny)
    return i, j


def triangle_areas(mesh: WaterSurfaceMesh) -> np.ndarray:
    points = mesh.vertices[mesh.triangles].astype(np.float64)
    return 0.5 * np.linalg.norm(np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=1)


def covered_columns(mesh: WaterSurfaceMesh, grid: PlanGrid) -> np.ndarray:
    """Columns holding at least one triangle centroid."""
    i, j = triangle_columns(mesh, grid)
    covered = np.zeros(grid.shape, bool)
    covered[i, j] = True
    return covered


def dilate(mask: np.ndarray) -> np.ndarray:
    """8-neighbour dilation, periodic in y, clipped in x."""
    out = mask.copy()
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            shifted = np.roll(mask, dj, axis=1)
            if di > 0:
                shifted = np.concatenate([np.zeros((di, mask.shape[1]), bool), shifted[:-di]])
            elif di < 0:
                shifted = np.concatenate([shifted[-di:], np.zeros((-di, mask.shape[1]), bool)])
            out |= shifted
    return out


def swash_line_columns(wet: np.ndarray, width: int = SWASH_COLUMNS) -> np.ndarray:
    """Wet columns within ``width`` columns of the up-beach wet edge of their row."""
    out = np.zeros_like(wet)
    for j in range(wet.shape[1]):
        rows = np.flatnonzero(wet[:, j])
        if len(rows):
            edge = rows.max()
            out[max(edge - width, 0):edge + 1, j] = wet[max(edge - width, 0):edge + 1, j]
    return out


def classify_frame(plic: WaterSurfaceMesh, grid: PlanGrid, theta: np.ndarray,
                   open_normals: np.ndarray | None = None,
                   open_offsets: np.ndarray | None = None) -> MeshResolvedness:
    """Resolvedness of a saved frame from its PLIC mesh (AW-1113 mesh rule).

    ``result.mixture_columns(MIXTURE_STACKED)`` is the frame's mixture mask;
    ``result.fraction`` the recovered water fraction of every facet cell.
    """
    return mesh_resolvedness(plic.vertices, plic.triangles, grid, theta, open_normals, open_offsets)


def triangle_cells(mesh: WaterSurfaceMesh, grid: PlanGrid) -> np.ndarray:
    """Cell (i, j, k) of every triangle by its centroid; j periodic, i and k clipped."""
    centroid = mesh.vertices[mesh.triangles].astype(np.float64).mean(axis=1)
    origin = np.array([grid.x0, grid.y0, grid.z0]); spacing = np.array([grid.dx, grid.dy, grid.dz])
    ijk = np.floor((centroid - origin) / spacing).astype(np.int64)
    ijk[:, 0] = np.clip(ijk[:, 0], 0, grid.nx - 1)
    ijk[:, 1] = np.mod(ijk[:, 1], grid.ny)
    ijk[:, 2] = np.clip(ijk[:, 2], 0, grid.nz - 1)
    return ijk


def cell_water_share(resolvedness: MeshResolvedness, theta: np.ndarray) -> np.ndarray:
    """Water of every facet cell as a share of its Cartesian cell."""
    c = resolvedness.cells
    return resolvedness.fraction * theta[c[:, 0], c[:, 1], c[:, 2]]


def _cell_flag_on_triangles(plic: WaterSurfaceMesh, grid: PlanGrid, resolvedness: MeshResolvedness,
                            flag: np.ndarray) -> np.ndarray:
    """Carry a per-facet-cell flag onto every PLIC triangle of that cell."""
    ijk = triangle_cells(plic, grid)
    flat = (ijk[:, 0] * grid.ny + ijk[:, 1]) * grid.nz + ijk[:, 2]
    known = resolvedness.flat
    index = np.minimum(np.searchsorted(known, flat), max(len(known) - 1, 0))
    found = (len(known) > 0) & (known[index] == flat)
    out = np.zeros(len(flat), bool)
    out[found] = flag[index[found]]
    return out


def residue_triangles(plic: WaterSurfaceMesh, grid: PlanGrid, resolvedness: MeshResolvedness,
                      theta: np.ndarray, fraction: float = RESIDUE_FRACTION) -> np.ndarray:
    """PLIC triangles whose cell holds less than ``fraction`` of a cell of water."""
    return _cell_flag_on_triangles(plic, grid, resolvedness, cell_water_share(resolvedness, theta) < fraction)


def boulder_film_cells(resolvedness: MeshResolvedness, grid: PlanGrid, theta: np.ndarray,
                       boulders: Sequence[RoundedBoulder],
                       fraction: float = BOULDER_FILM_FRACTION) -> np.ndarray:
    """Facet cells that are cut against a boulder and hold less than ``fraction`` of a cell."""
    c = resolvedness.cells
    cut = theta[c[:, 0], c[:, 1], c[:, 2]] < 1.0
    film = np.zeros(len(c), bool)
    if not boulders or not cut.any():
        return film
    spacing = np.array([grid.dx, grid.dy, grid.dz])
    centres = np.array([grid.x0, grid.y0, grid.z0]) + (c[cut] + 0.5) * spacing
    distance = np.min([np.asarray(b.signed_distance(centres)) for b in boulders], axis=0)
    share = cell_water_share(resolvedness, theta)[cut]
    film[cut] = (distance < spacing.max()) & (share < fraction)
    return film


def boulder_film_triangles(plic: WaterSurfaceMesh, grid: PlanGrid, resolvedness: MeshResolvedness,
                           theta: np.ndarray, boulders: Sequence[RoundedBoulder],
                           fraction: float = BOULDER_FILM_FRACTION) -> np.ndarray:
    """PLIC triangles of draining films on the boulders (shown by wetness instead)."""
    return _cell_flag_on_triangles(plic, grid, resolvedness,
                                   boulder_film_cells(resolvedness, grid, theta, boulders, fraction))


def persist_indicator(previous: np.ndarray | None, mixture: np.ndarray, dt: float,
                      persistence: float) -> np.ndarray:
    """Carry the aeration indicator forward with an exponential memory."""
    current = mixture.astype(np.float64)
    if previous is None or persistence <= 0.0:
        return current
    return np.maximum(current, previous * np.exp(-dt / persistence))


def represented_volume(mesh: WaterSurfaceMesh, grid: PlanGrid, theta: np.ndarray) -> np.ndarray:
    """Water volume per column (m^3) the mesh represents, by the cell-centre rule."""
    crossings = cast_columns(mesh.vertices, mesh.triangles, grid)
    column = np.repeat(np.arange(grid.columns), grid.nz)
    k = np.tile(np.arange(grid.nz), grid.columns)
    water = water_at_cell_centres(crossings, column, k).reshape(grid.nx, grid.ny, grid.nz)
    return (water * theta).sum(axis=2) * grid.dx * grid.dy * grid.dz


def band_totals(per_column: np.ndarray, grid: PlanGrid, bands: dict = BANDS) -> dict[str, float]:
    x = grid.x0 + (np.arange(grid.nx) + 0.5) * grid.dx
    return {name: float(per_column[(x >= lo) & (x < hi)].sum()) for name, (lo, hi) in bands.items()}


def _select(mesh: WaterSurfaceMesh, keep: np.ndarray) -> WaterSurfaceMesh:
    if keep.all():
        return mesh
    used = np.zeros(len(mesh.vertices), bool)
    used[mesh.triangles[keep]] = True
    index = np.cumsum(used) - 1
    return WaterSurfaceMesh(
        vertices=mesh.vertices[used], triangles=index[mesh.triangles[keep]].astype(np.int32),
        normals=mesh.normals[used], velocity=mesh.velocity[used], foam=mesh.foam[used],
        vertex_open_boundaries=mesh.vertex_open_boundaries[used],
        open_boundary_edges=index[mesh.open_boundary_edges].astype(np.int32)
        if len(mesh.open_boundary_edges) else mesh.open_boundary_edges,
        open_boundary_labels=mesh.open_boundary_labels,
    )


def _concatenate(first: WaterSurfaceMesh, second: WaterSurfaceMesh) -> WaterSurfaceMesh:
    offset = len(first.vertices)
    return WaterSurfaceMesh(
        vertices=np.concatenate([first.vertices, second.vertices]),
        triangles=np.concatenate([first.triangles, second.triangles + offset]).astype(np.int32),
        normals=np.concatenate([first.normals, second.normals]),
        velocity=np.concatenate([first.velocity, second.velocity]),
        foam=np.concatenate([first.foam, second.foam]),
        vertex_open_boundaries=np.concatenate([first.vertex_open_boundaries, second.vertex_open_boundaries]),
        open_boundary_edges=np.concatenate([first.open_boundary_edges, second.open_boundary_edges + offset]).astype(np.int32),
        open_boundary_labels=np.concatenate([first.open_boundary_labels, second.open_boundary_labels]),
    )


def _vertex_mean(mesh: WaterSurfaceMesh, per_triangle: np.ndarray) -> np.ndarray:
    """Average a per-triangle value onto vertices (soft edges between classes)."""
    total = np.zeros(len(mesh.vertices)); count = np.zeros(len(mesh.vertices))
    for corner in range(3):
        np.add.at(total, mesh.triangles[:, corner], per_triangle)
        np.add.at(count, mesh.triangles[:, corner], 1.0)
    return np.divide(total, count, out=np.zeros_like(total), where=count > 0)


def compose_aerated_surface(
    plic: WaterSurfaceMesh, nodal: WaterSurfaceMesh, mixture: np.ndarray, grid: PlanGrid,
    *, composite: str = "plic_resolved", indicator: np.ndarray | None = None,
    residue: np.ndarray | None = None, footprint: np.ndarray | None = None,
    boulder_film: np.ndarray | None = None,
    aeration_weight: float = AERATION_WEIGHT, aeration_switch: float = AERATION_SWITCH,
    foam_gate: bool = True, swash_columns: int = SWASH_COLUMNS, theta: np.ndarray | None = None,
) -> tuple[WaterSurfaceMesh, dict]:
    """Assemble the named composite and report every replaced, kept and painted part.

    ``mixture`` is the frame's instantaneous classification; ``indicator`` the
    persisted aeration indicator in ``[0, 1]`` (defaults to the mask itself);
    ``residue`` marks PLIC triangles of residue cells (not drawn),
    ``footprint`` the boulder-footprint columns (no PLIC facets kept there)
    and ``boulder_film`` the PLIC triangles of draining films on the boulders.
    """
    if composite not in COMPOSITES:
        raise ValueError(f"unknown composite {composite!r}; choose from {COMPOSITES}")
    if mixture.shape != grid.shape:
        raise ValueError("mixture mask must have the plan-grid shape")
    if indicator is None:
        indicator = mixture.astype(np.float64)
    if residue is None:
        residue = np.zeros(len(plic.triangles), bool)
    if footprint is None:
        footprint = np.zeros(grid.shape, bool)
    if boulder_film is None:
        boulder_film = np.zeros(len(plic.triangles), bool)
    aerated = indicator >= aeration_switch
    pi, pj = triangle_columns(plic, grid)
    ni, nj = triangle_columns(nodal, grid)
    has_nodal = np.zeros(grid.shape, bool); has_nodal[ni, nj] = True
    has_plic = np.zeros(grid.shape, bool); has_plic[pi, pj] = True
    wet = has_plic | has_nodal
    if composite == "plic_resolved":
        nodal_columns = (aerated | footprint) & has_nodal
        plic_columns = ~nodal_columns & ~footprint
        # The guest surface extends one column into the host: no hole at the seam.
        ring = dilate(nodal_columns) & ~nodal_columns & has_nodal
        nodal_columns = nodal_columns | ring
    else:
        nodal_columns = has_nodal
        plic_columns = ~has_nodal & ~footprint
        ring = dilate(plic_columns) & ~plic_columns & has_plic
        plic_columns = plic_columns | ring
    hidden = residue | boulder_film
    plic_keep = plic_columns[pi, pj] & ~hidden
    nodal_keep = nodal_columns[ni, nj]
    plic_area = triangle_areas(plic); nodal_area = triangle_areas(nodal)
    footprint_dropped = footprint[pi, pj] & ~hidden
    film = plic_keep & aerated[pi, pj] & ~has_nodal[pi, pj]
    guest_area = nodal_area[nodal_keep & ring[ni, nj]] if composite == "plic_resolved" \
        else plic_area[plic_keep & ring[pi, pj]]
    off = aeration_weight == 0.0 and not foam_gate
    if off and plic_keep.all() and not nodal_keep.any():
        return plic, {"composite": composite, "closure": "off", "unchanged": "plic"}
    if off and nodal_keep.all() and not plic_keep.any():
        return nodal, {"composite": composite, "closure": "off", "unchanged": "nodal"}
    plic_part = _select(plic, plic_keep)
    nodal_part = _select(nodal, nodal_keep)
    # The indicator gates the solver's foam (the swash line keeps its own) and
    # carries the optional extra coverage on the nodal surface of aerated columns.
    gate = np.maximum(indicator, swash_line_columns(wet, swash_columns))
    gate_plic = _vertex_mean(plic_part, gate[pi, pj][plic_keep]) if foam_gate else 1.0
    gate_nodal = _vertex_mean(nodal_part, gate[ni, nj][nodal_keep]) if foam_gate else 1.0
    painted = _vertex_mean(nodal_part, (indicator * aerated)[ni, nj][nodal_keep])
    plic_foam = plic_part.foam * gate_plic
    nodal_foam = np.clip(nodal_part.foam * gate_nodal + aeration_weight * painted, 0.0, 1.0)
    plic_part = WaterSurfaceMesh(**{**plic_part.__dict__, "foam": plic_foam.astype(np.float32)})
    nodal_part = WaterSurfaceMesh(**{**nodal_part.__dict__, "foam": nodal_foam.astype(np.float32)})
    result = _concatenate(plic_part, nodal_part)
    report = {
        "composite": composite, "closure": "on", "aeration_weight": float(aeration_weight),
        "aeration_switch": float(aeration_switch),
        "foam_gate": bool(foam_gate), "swash_columns": int(swash_columns),
        "mixture_columns": int(mixture.sum()), "aerated_columns": int(aerated.sum()),
        "indicator_sum": float(indicator.sum()), "wet_columns": int(wet.sum()),
        "nodal_columns": int((nodal_columns & ~ring).sum()),
        "plic_columns": int((plic_columns & ~ring & has_plic).sum()),
        "ring_columns": int(ring.sum()), "gate_columns": int(((gate > 0.05) & wet).sum()),
        "plic_triangles_dropped": int((~plic_keep).sum()),
        "plic_area_dropped_m2": float(plic_area[~plic_keep].sum()),
        "plic_film_triangles_kept_in_mixture": int(film.sum()),
        "plic_film_area_kept_m2": float(plic_area[film].sum()),
        "residue_triangles_not_drawn": int(residue.sum()),
        "residue_area_m2": float(plic_area[residue].sum()),
        "boulder_film_triangles_not_drawn": int((boulder_film & ~residue).sum()),
        "boulder_film_area_m2": float(plic_area[boulder_film & ~residue].sum()),
        "footprint_columns": int(footprint.sum()),
        "footprint_triangles_not_drawn": int(footprint_dropped.sum()),
        "footprint_area_m2": float(plic_area[footprint_dropped].sum()),
        "nodal_triangles_used": int(nodal_keep.sum()),
        "nodal_area_used_m2": float(nodal_area[nodal_keep].sum()),
        "overlap_area_m2": float(guest_area.sum()),
        "aerated_area_m2": float(nodal_area[nodal_keep & aerated[ni, nj]].sum()),
        "foam_mean_plic_in": float(plic.foam.mean()), "foam_mean_out": float(result.foam.mean()),
        "vertices": int(len(result.vertices)), "triangles": int(len(result.triangles)),
        "plic_vertices": int(len(plic_part.vertices)),
    }
    if theta is not None:
        volumes = {name: represented_volume(mesh, grid, theta)
                   for name, mesh in (("plic", plic), ("nodal", nodal), ("composite", result))}
        report["represented_volume_m3"] = {name: band_totals(v, grid) for name, v in volumes.items()}
        report["represented_volume_error_vs_plic"] = {
            band: (report["represented_volume_m3"]["composite"][band] - value) / value if value else 0.0
            for band, value in report["represented_volume_m3"]["plic"].items()}
        # Columns under a boulder overhang hide their surface from the rays;
        # compare the drawn surfaces where the rule can see them.
        outside = {name: band_totals(np.where(footprint, 0.0, v), grid) for name, v in volumes.items()}
        report["represented_volume_error_vs_plic_outside_footprint"] = {
            band: (outside["composite"][band] - value) / value if value else 0.0
            for band, value in outside["plic"].items()}
    return result, report
