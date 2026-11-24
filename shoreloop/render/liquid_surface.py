"""Boundary of the solver's cell-local cut/PLIC liquid polyhedra.

No alpha is averaged and no vertices are moved to thicken or join water.
Neighbouring liquid footprints cancel on Cartesian faces; their differences
are genuine exposed faces of this piecewise planar representation. Solid
faces and non-periodic domain caps are omitted. This preserves the solver's
faceted contact geometry, including its measured offset from analytic solids.

This is the ``plic_boundary`` render surface. The ``nodal_isosurface``
alternative in :mod:`shoreloop.render.surface_mesh` reconstructs the same
cell averages a second way (theta-weighted nodal ``alpha = 0.5``); it is not
derived from, and does not smooth, this mesh.
"""

from itertools import product

import numpy as np

from shoreloop.config import BC
from shoreloop.physics3d.interface import estimate_interface_normals
from shoreloop.physics3d.liquid_pressure import (
    _BOX, _BOX_SECTIONS, _clip, _outside, _section, _area_moment, _facet_area_tolerance,
)
from shoreloop.physics3d.plic import reconstruct_plic
from shoreloop.render.surface_mesh import (
    WaterSurfaceMesh, _boundary_labels, _labelled_boundary_edges, _empty_mesh,
)


def _split_edges(polygon, cell, normals, offsets, periodic, shape):
    """Insert adjacent plane crossings so shared edges have matching segments.

    A cell-face edge sees the neighbour across that face. A box edge also
    sees the diagonal cell. Inserting these points changes only triangulation,
    not the polygon. Final welding uses exact stored FP32 coordinates.
    """
    points = []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
        shifts = [((0, int(2*a[axis]-1)) if a[axis] == b[axis] and a[axis] in (0, 1)
                   else (0,)) for axis in range(3)]
        crossings = [0.0]
        for shift in product(*shifts):
            if not any(shift):
                continue
            neighbour = np.asarray(cell)+shift
            if any(not periodic[q] and not 0 <= neighbour[q] < shape[q] for q in range(3)):
                continue
            neighbour %= shape
            index = tuple(neighbour)
            for field, distances in zip(normals, offsets, strict=True):
                n = field[index]
                d = distances[index]+n @ shift
                before, after = n @ a-d, n @ b-d
                # An edge constructed on this plane can straddle it by
                # FP64 clipping roundoff. Dividing those residuals invents
                # an arbitrary interior split (often the midpoint) which
                # the adjacent polygon does not have. Classify only signs
                # resolved beyond the local dot/interpolation roundoff.
                error = 16*np.finfo(float).eps*(
                    np.abs(n) @ np.maximum(np.abs(a), np.abs(b))+abs(d))
                if before < -error and after > error or after < -error and before > error:
                    crossings.append(before/(before-after))
        points.extend(a+t*(b-a) for t in sorted(set(crossings)))
    return np.asarray(points)


def _surface_polygons(alpha, theta, sn, sd, wn, wd, periodic):
    shape = alpha.shape
    wet = (alpha > 1e-6) & (theta > 0)
    full = wet & ~np.any(wn, axis=-1)
    candidates = ~full
    for axis in range(3):
        for sign in (-1, 1):
            candidates |= ~np.roll(full, -sign, axis=axis)
    for raw in np.argwhere(wet & candidates):
        cell = tuple(raw)
        solid = [(sn[cell], sd[cell])] if np.any(sn[cell]) else []
        liquid = [(wn[cell], wd[cell])] if np.any(wn[cell]) else []
        if liquid:
            yield cell, _section(*liquid[0], solid), wn[cell]
        for face, (n, _) in enumerate(_BOX):
            axis, sign = face//2, 2*(face % 2)-1
            neighbour = raw.copy()
            neighbour[axis] += sign
            if not 0 <= neighbour[axis] < shape[axis]:
                if not periodic[axis]:
                    # Wall contact and open-domain intersections are mesh
                    # boundaries, not artificial water-air caps.
                    continue
                neighbour[axis] %= shape[axis]
            neighbour = tuple(neighbour)
            if theta[neighbour] == 0 or full[neighbour]:
                continue
            polygon = _BOX_SECTIONS[face]
            for normal, offset in solid+liquid:
                polygon = _clip(polygon, normal, offset)
            polygon = _clip(polygon, sn[neighbour], sd[neighbour]+sign*sn[neighbour][axis])
            if wet[neighbour]:
                polygon = _outside(polygon, wn[neighbour], wd[neighbour]+sign*wn[neighbour][axis])
            yield cell, polygon, n


def _triangulate(points, normal):
    """Convex polygon ears, retaining collinear boundary splits in FP32.

    A centre fan can put its rounded centre on an edge of a one-ulp-wide
    facet, creating T junctions. Ears use only the existing boundary vertices.
    Removing the smallest positive ear keeps area for the remaining polygon.
    A polygon collapsed to a line at output precision has no surface area.
    """
    axes = [q for q in range(3) if q != np.argmax(np.abs(normal))]
    p = points[:, axes].astype(float)
    # Different analytic crossings can become the same output point. Remove
    # consecutive duplicates before clipping ears, or they can strand a
    # positive-area remainder behind a run of zero-angle corners.
    remaining = list(np.flatnonzero(np.any(points != np.roll(points, 1, axis=0), axis=1)))
    result = []
    while len(remaining) >= 3:
        ids = np.asarray(remaining)
        a, b, c = p[np.roll(ids, 1)], p[ids], p[np.roll(ids, -1)]
        cross = (b[:, 0]-a[:, 0])*(c[:, 1]-a[:, 1])-(b[:, 1]-a[:, 1])*(c[:, 0]-a[:, 0])
        if not np.any(cross):
            break
        orientation = np.sign(cross[np.argmax(np.abs(cross))])
        positive = np.where(cross*orientation > 0, cross*orientation, np.inf)
        j = int(np.argmin(positive))
        tri = [remaining[j-1], remaining[j], remaining[(j+1) % len(remaining)]]
        if np.cross(points[tri[1]]-points[tri[0]], points[tri[2]]-points[tri[0]]) @ normal < 0:
            tri[1], tri[2] = tri[2], tri[1]
        result.append(tri)
        remaining.pop(j)
    return result


def extract_liquid_surface_mesh(state, *, reconstruction=None):
    """Extract the current governing liquid geometry and passive attributes.

    A supplied reconstruction must belong to this unchanged alpha/solid state.
    Geometry is unsmoothed. Normals and passive attributes are area-averaged
    at identical stored positions, including their periodic images; dry gaps
    stay open. Periodic averaging changes attributes, never mesh coordinates.
    """
    g = state.cfg.grid
    spacing = np.asarray((g.dx, g.dy, g.dz))
    origin = np.asarray((g.x0, g.y0, g.z0))
    if reconstruction is None:
        reconstruction = reconstruct_plic(state, estimate_interface_normals(state).normals)
    slices = state.grid.interior_slices("cell")
    sn = state.open_normals.numpy()[slices].astype(float)
    sd = state.open_offsets.numpy()[slices].astype(float)
    wn = reconstruction.normals.numpy()[slices].astype(float)
    wd = reconstruction.offsets.numpy()[slices].astype(float)
    alpha = state.get_interior("alpha")
    theta = state.get_geometry_interior("theta")
    velocity = [state.get_interior(q) for q in ("u", "v", "w")]
    foam = state.get_interior("foam")
    periodic = tuple(getattr(state.cfg.bc, a+"lo") == BC.PERIODIC for a in "xyz")
    open_sides = tuple(a+s for a in "xyz" for s in ("lo", "hi")
                       if getattr(state.cfg.bc, a+s) == BC.OPEN)
    vertices, triangles, velocities, foams = [], [], [], []
    cell_volume = float(np.prod(spacing))
    for cell, polygon, normal in _surface_polygons(alpha, theta, sn, sd, wn, wd, periodic):
        threshold = _facet_area_tolerance(float(alpha[cell]*theta[cell])*cell_volume, spacing)
        if _area_moment(polygon, spacing)[0] <= threshold:
            continue
        polygon = _split_edges(polygon, cell, (sn, wn), (sd, wd), periodic, alpha.shape)
        local = polygon
        base = len(vertices)
        world = (origin+(np.asarray(cell)+local)*spacing).astype(np.float32).astype(float)
        vertices.extend(world)
        values = np.empty_like(local)
        for axis in range(3):
            high = list(cell); high[axis] += 1
            values[:, axis] = ((1-local[:, axis])*velocity[axis][cell]
                               + local[:, axis]*velocity[axis][tuple(high)])
        velocities.extend(values)
        foams.extend([foam[cell]]*len(local))
        triangles.extend([base+i for i in tri] for tri in _triangulate(world, normal/spacing))
    if not triangles:
        return _empty_mesh()
    vertices, inverse = np.unique(np.asarray(vertices, dtype=np.float32), axis=0, return_inverse=True)
    original_triangles = np.asarray(triangles)
    triangles = inverse[original_triangles].astype(np.int32)
    points = vertices[triangles].astype(float)
    area_vectors = np.cross(points[:, 1]-points[:, 0], points[:, 2]-points[:, 0])
    area = np.linalg.norm(area_vectors, axis=1)
    keep = area > 0
    triangles, original_triangles, area_vectors, area = triangles[keep], original_triangles[keep], area_vectors[keep], area[keep]
    if not len(triangles):
        return _empty_mesh()
    attribute_positions = vertices.copy()
    for axis, enabled in enumerate(periodic):
        if enabled:
            high = np.float32(origin[axis]+alpha.shape[axis]*spacing[axis])
            attribute_positions[vertices[:, axis] == high, axis] = origin[axis]
    _, attribute_ids = np.unique(attribute_positions, axis=0, return_inverse=True)
    normals = np.zeros_like(vertices, dtype=float)
    sampled_velocity = np.zeros_like(normals)
    sampled_foam = np.zeros(len(vertices))
    weights = np.zeros(len(vertices))
    velocities, foams = np.asarray(velocities), np.asarray(foams)
    for q in range(3):
        ids = attribute_ids[triangles[:, q]]
        np.add.at(normals, ids, area_vectors)
        np.add.at(weights, ids, area)
        np.add.at(sampled_velocity, ids, area[:, None]*velocities[original_triangles[:, q]])
        np.add.at(sampled_foam, ids, area*foams[original_triangles[:, q]])
    used = np.unique(triangles)
    remap = np.empty(len(vertices), dtype=np.int32)
    remap[used] = np.arange(len(used))
    vertices, triangles = vertices[used], remap[triangles]
    attributes = attribute_ids[used]
    normals = normals[attributes]
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    labels = _boundary_labels(vertices, g, open_sides)
    edges, edge_labels = _labelled_boundary_edges(triangles, labels)
    return WaterSurfaceMesh(vertices, triangles, normals.astype(np.float32),
                            (sampled_velocity[attributes]/weights[attributes, None]).astype(np.float32),
                            (sampled_foam[attributes]/weights[attributes]).astype(np.float32),
                            labels, edges, edge_labels)
