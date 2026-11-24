"""Device assembly of the liquid pressure operator's cell geometry and local systems.

Every retained liquid polyhedron is the cell box clipped by its solid plane and its
liquid (PLIC) plane; each Cartesian face piece is split by the neighbour's planes
into solid, free and shared parts, exactly as the host construction in
``liquid_pressure._cells``. One kernel computes facet areas, unit normals and
centroids, the liquid volume, the liquid centroid and the affine residual; a
second forms the MAC-restricted mimetic local systems (mass, divergence, the
local projection response, weights, projector, gravity couplings) per cell.

Precision boundary: facet moments are accumulated in float64 here, where the
host path used quadrature precision. The affine identity residual
``|R^T N - V I| / V`` is reported per assembly instead of asserted; films whose
residual exceeds the host tolerance are the small liquid cells that AW-1110
agglomerates. Units: facet geometry is physical (metres) and cell-local.

Sliver inertia. The mimetic stabilisation is scaled by the facet-space
consistency trace, ``sum_f |A_f R_f|^2 / (3 V)``, which for a full cell is
half its volume. A near-empty cell keeps cell-sized facets while its liquid
volume vanishes, so that scale grows as ``1 / V`` and hands the cell's
non-affine MAC modes an inertia far above the liquid's (measured on a saved
production state, M11 "AW-1120": median 9, p90 190, extreme 4e5 liquid
volumes for cells under 3% full); the wall, viscous and drag solves and the
pressure response then barely touch those faces. ``SLIVER_STABILISATION_CAP``
bounds the scale by that multiple of the MAC consistency trace per kept
face, ``sum_col |P_col|^2 / (V n)``, which equals the uncapped scale for a
full cell and shrinks with the liquid for a sliver. Affine traces are
unaffected: the stabilisation annihilates them.
"""

from dataclasses import dataclass

import numpy as np
import warp as wp

from shoreloop.config import BC
from shoreloop.physics3d.resolved import resolved_liquid, resolved_liquid_mask

MAX_FACETS = 20
MAX_MAC = 6
SLIVER_STABILISATION_CAP = 2.0

_Poly = wp.types.matrix(shape=(12, 3), dtype=wp.float64)
_Points = wp.types.matrix(shape=(24, 3), dtype=wp.float64)
_Mat6 = wp.types.matrix(shape=(6, 6), dtype=wp.float64)
_Vec6 = wp.types.vector(length=6, dtype=wp.float64)
_Vec6i = wp.types.vector(length=6, dtype=wp.int32)
_Facets = wp.types.matrix(shape=(MAX_FACETS, 3), dtype=wp.float64)
_FacetScalars = wp.types.vector(length=MAX_FACETS, dtype=wp.float64)
_FacetInts = wp.types.vector(length=MAX_FACETS, dtype=wp.int32)

KIND_SHARED = 0
KIND_SOLID = 1
KIND_FREE = 2


@wp.func
def _clip(points: _Poly, count: int, normal: wp.vec3d, offset: wp.float64) -> tuple[_Poly, int]:
    """Points with normal.p - offset <= 0; a zero normal keeps all iff offset >= 0."""
    result = _Poly()
    size = int(0)
    if count == 0:
        return result, size
    if wp.dot(normal, normal) == wp.float64(0.0):
        if offset >= wp.float64(0.0):
            return points, count
        return result, size
    previous = points[count - 1]
    before = wp.dot(normal, previous) - offset
    for index in range(count):
        point = points[index]
        value = wp.dot(normal, point) - offset
        if (value <= wp.float64(0.0)) != (before <= wp.float64(0.0)):
            # Symmetric interpolation, as the host path.
            result[size] = (before * point - value * previous) / (before - value)
            size += 1
        if value <= wp.float64(0.0):
            result[size] = point
            size += 1
        previous = point
        before = value
    return result, size


@wp.func
def _outside(points: _Poly, count: int, normal: wp.vec3d, offset: wp.float64) -> tuple[_Poly, int]:
    """Positive-area part strictly outside the halfspace, excluding coplanarity."""
    empty = _Poly()
    if count == 0:
        return empty, 0
    inside = int(1)
    for index in range(count):
        if wp.dot(points[index], normal) > offset:
            inside = 0
    if inside == 1:
        return empty, 0
    return _clip(points, count, -normal, -offset)


@wp.func
def _box_vertex(index: int) -> wp.vec3d:
    return wp.vec3d(wp.float64((index >> 2) & 1), wp.float64((index >> 1) & 1), wp.float64(index & 1))


@wp.func
def _section(normal: wp.vec3d, offset: wp.float64, other: wp.vec3d, other_offset: wp.float64,
             has_other: int) -> tuple[_Poly, int]:
    """Plane section of the unit box from its edges, then clipped by the other plane."""
    raw = _Points()
    size = int(0)
    for bit in range(3):
        step = 1 << bit
        for a in range(8):
            if (a & step) == 0:
                b = a | step
                pa = _box_vertex(a)
                pb = _box_vertex(b)
                da = wp.dot(pa, normal) - offset
                db = wp.dot(pb, normal) - offset
                if da == wp.float64(0.0):
                    raw[size] = pa
                    size += 1
                if db == wp.float64(0.0):
                    raw[size] = pb
                    size += 1
                if (da < wp.float64(0.0) and db > wp.float64(0.0)) or (db < wp.float64(0.0) and da > wp.float64(0.0)):
                    raw[size] = (da * pb - db * pa) / (da - db)
                    size += 1
    # Unique points (exact duplicates arise from shared edge endpoints).
    points = _Poly()
    count = int(0)
    for index in range(size):
        candidate = raw[index]
        duplicate = int(0)
        for known in range(count):
            delta = points[known] - candidate
            if wp.dot(delta, delta) == wp.float64(0.0):
                duplicate = 1
        if duplicate == 0 and count < 12:
            points[count] = candidate
            count += 1
    if count == 0:
        return points, count
    # Angular order about the mean in the plane's dominant projection.
    dominant = int(0)
    if wp.abs(normal[1]) > wp.abs(normal[dominant]):
        dominant = 1
    if wp.abs(normal[2]) > wp.abs(normal[dominant]):
        dominant = 2
    first = int(0)
    second = int(1)
    if dominant == 0:
        first = 1
        second = 2
    elif dominant == 1:
        first = 0
        second = 2
    mean = wp.vec3d(wp.float64(0.0))
    for index in range(count):
        mean += points[index]
    mean = mean / wp.float64(count)
    angles = _FacetScalars()
    for index in range(count):
        p = points[index] - mean
        angles[index] = wp.atan2(p[second], p[first])
    # Insertion sort (stable) on the angle.
    for i in range(1, count):
        key_angle = angles[i]
        key_point = points[i]
        j = i - 1
        while j >= 0 and angles[j] > key_angle:
            angles[j + 1] = angles[j]
            points[j + 1] = points[j]
            j -= 1
        angles[j + 1] = key_angle
        points[j + 1] = key_point
    if has_other == 1:
        points, count = _clip(points, count, other, other_offset)
    return points, count


@wp.func
def _moments(points: _Poly, count: int, spacing: wp.vec3d) -> tuple[wp.float64, wp.vec3d]:
    """Physical area and area-weighted centroid moment of a convex polygon."""
    area = wp.float64(0.0)
    moment = wp.vec3d(wp.float64(0.0))
    if count < 3:
        return area, moment
    a = wp.cw_mul(points[0], spacing)
    for j in range(1, count - 1):
        b = wp.cw_mul(points[j], spacing)
        c = wp.cw_mul(points[j + 1], spacing)
        cross = wp.cross(b - a, c - a)
        weight = wp.sqrt(wp.dot(cross, cross)) / wp.float64(2.0)
        area += weight
        moment += weight * (a + b + c) / wp.float64(3.0)
    return area, moment


@wp.func
def _box_face(face: int) -> _Poly:
    """The unit-box face polygon in the host's vertex order."""
    p = _Poly()
    if face == 0:
        p[0] = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
        p[1] = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(1.0))
        p[2] = wp.vec3d(wp.float64(0.0), wp.float64(1.0), wp.float64(1.0))
        p[3] = wp.vec3d(wp.float64(0.0), wp.float64(1.0), wp.float64(0.0))
    elif face == 1:
        p[0] = wp.vec3d(wp.float64(1.0), wp.float64(0.0), wp.float64(1.0))
        p[1] = wp.vec3d(wp.float64(1.0), wp.float64(0.0), wp.float64(0.0))
        p[2] = wp.vec3d(wp.float64(1.0), wp.float64(1.0), wp.float64(0.0))
        p[3] = wp.vec3d(wp.float64(1.0), wp.float64(1.0), wp.float64(1.0))
    elif face == 2:
        p[0] = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(1.0))
        p[1] = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
        p[2] = wp.vec3d(wp.float64(1.0), wp.float64(0.0), wp.float64(0.0))
        p[3] = wp.vec3d(wp.float64(1.0), wp.float64(0.0), wp.float64(1.0))
    elif face == 3:
        p[0] = wp.vec3d(wp.float64(0.0), wp.float64(1.0), wp.float64(0.0))
        p[1] = wp.vec3d(wp.float64(0.0), wp.float64(1.0), wp.float64(1.0))
        p[2] = wp.vec3d(wp.float64(1.0), wp.float64(1.0), wp.float64(1.0))
        p[3] = wp.vec3d(wp.float64(1.0), wp.float64(1.0), wp.float64(0.0))
    elif face == 4:
        p[0] = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
        p[1] = wp.vec3d(wp.float64(0.0), wp.float64(1.0), wp.float64(0.0))
        p[2] = wp.vec3d(wp.float64(1.0), wp.float64(1.0), wp.float64(0.0))
        p[3] = wp.vec3d(wp.float64(1.0), wp.float64(0.0), wp.float64(0.0))
    else:
        p[0] = wp.vec3d(wp.float64(0.0), wp.float64(1.0), wp.float64(1.0))
        p[1] = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(1.0))
        p[2] = wp.vec3d(wp.float64(1.0), wp.float64(0.0), wp.float64(1.0))
        p[3] = wp.vec3d(wp.float64(1.0), wp.float64(1.0), wp.float64(1.0))
    return p


@wp.func
def _box_plane(face: int) -> tuple[wp.vec3d, wp.float64]:
    axis = face // 2
    normal = wp.vec3d(wp.float64(0.0))
    offset = wp.float64(0.0)
    if face % 2 == 1:
        normal[axis] = wp.float64(1.0)
        offset = wp.float64(1.0)
    else:
        normal[axis] = wp.float64(-1.0)
    return normal, offset


@wp.func
def _coincident(n: wp.vec3d, d: wp.float64, other: wp.vec3d, other_d: wp.float64) -> int:
    cross = wp.cross(n, other)
    if (wp.sqrt(wp.dot(cross, cross)) < wp.float64(1.0e-14) and wp.dot(n, other) > wp.float64(0.0)
            and wp.abs(d * wp.dot(n, other) - other_d * wp.dot(n, n)) < wp.float64(1.0e-14)):
        return 1
    return 0


@wp.func
def _neighbour_index(i: int, j: int, k: int, axis: int, sign: int, nx: int, ny: int, nz: int,
                     periodic: wp.vec3i) -> wp.vec4i:
    """Neighbour cell index and an inside flag (1) after periodic wrapping."""
    n = wp.vec3i(i, j, k)
    n[axis] = n[axis] + sign
    size = nx
    if axis == 1:
        size = ny
    elif axis == 2:
        size = nz
    inside = int(1)
    if n[axis] < 0 or n[axis] >= size:
        if periodic[axis] == 1:
            n[axis] = (n[axis] + size) % size
        else:
            inside = 0
    return wp.vec4i(n[0], n[1], n[2], inside)


@wp.kernel
def _liquid_facets(
    cells: wp.array(dtype=wp.vec3i),
    alpha: wp.array3d(dtype=wp.float32),
    theta: wp.array3d(dtype=wp.float32),
    open_normals: wp.array3d(dtype=wp.vec3),
    open_offsets: wp.array3d(dtype=wp.float32),
    liquid_normals: wp.array3d(dtype=wp.vec3),
    liquid_offsets: wp.array3d(dtype=wp.float32),
    liquid_centroids: wp.array3d(dtype=wp.vec3),
    nx: int, ny: int, nz: int, ng: int,
    periodic: wp.vec3i,
    boundary_kind: _Vec6i,
    spacing: wp.vec3d,
    wet_threshold: wp.float32,
    facet_area: wp.array2d(dtype=wp.float64),
    facet_normal: wp.array2d(dtype=wp.vec3d),
    facet_centre: wp.array2d(dtype=wp.vec3d),
    facet_kind: wp.array2d(dtype=wp.int32),
    facet_face: wp.array2d(dtype=wp.int32),
    facet_count: wp.array(dtype=wp.int32),
    volume: wp.array(dtype=wp.float64),
    centroid: wp.array(dtype=wp.vec3d),
    volume_error: wp.array(dtype=wp.float64),
    affine_error: wp.array(dtype=wp.float64),
    has_free: wp.array(dtype=wp.int32),
    maxima: wp.array(dtype=wp.float64),
    free_facets: wp.array(dtype=wp.int32),
):
    c = wp.tid()
    cell = cells[c]
    i = cell[0]
    j = cell[1]
    k = cell[2]
    I = i + ng
    J = j + ng
    K = k + ng
    cell_volume = spacing[0] * spacing[1] * spacing[2]
    max_spacing = wp.max(spacing[0], wp.max(spacing[1], spacing[2]))
    target = wp.float64(alpha[I, J, K]) * wp.float64(theta[I, J, K]) * cell_volume
    area_tolerance = wp.float64(1.0e-12) * target / max_spacing
    sn = wp.vec3d(open_normals[I, J, K])
    sd = wp.float64(open_offsets[I, J, K])
    wn = wp.vec3d(liquid_normals[I, J, K])
    wd = wp.float64(liquid_offsets[I, J, K])
    has_solid = int(0)
    if wp.dot(sn, sn) > wp.float64(0.0):
        has_solid = 1
    has_liquid = int(0)
    if wp.dot(wn, wn) > wp.float64(0.0):
        has_liquid = 1

    count = int(0)
    areas = _FacetScalars()
    normals = _Facets()
    centres = _Facets()
    kinds = _FacetInts()
    faces = _FacetInts()

    # Cartesian face pieces.
    for face in range(6):
        axis = face // 2
        sign = 2 * (face % 2) - 1
        polygon = _box_face(face)
        pcount = int(4)
        if has_solid == 1:
            polygon, pcount = _clip(polygon, pcount, sn, sd)
        if has_liquid == 1:
            polygon, pcount = _clip(polygon, pcount, wn, wd)
        box_normal = wp.vec3d(wp.float64(0.0))
        box_normal[axis] = wp.float64(sign)
        if pcount < 3:
            continue
        nb = _neighbour_index(i, j, k, axis, sign, nx, ny, nz, periodic)
        if nb[3] == 0:
            area, moment = _moments(polygon, pcount, spacing)
            if area > area_tolerance and count < MAX_FACETS:
                areas[count] = area
                normals[count] = box_normal
                centres[count] = moment / area
                kinds[count] = boundary_kind[face]
                faces[count] = face
                count += 1
            continue
        NI = nb[0] + ng
        NJ = nb[1] + ng
        NK = nb[2] + ng
        if theta[NI, NJ, NK] == wp.float32(0.0):
            area, moment = _moments(polygon, pcount, spacing)
            if area > area_tolerance and count < MAX_FACETS:
                areas[count] = area
                normals[count] = box_normal
                centres[count] = moment / area
                kinds[count] = KIND_SOLID
                faces[count] = face
                count += 1
            continue
        nsn = wp.vec3d(open_normals[NI, NJ, NK])
        nsd = wp.float64(open_offsets[NI, NJ, NK]) + wp.float64(sign) * nsn[axis]
        opened, ocount = _clip(polygon, pcount, nsn, nsd)
        blocked, bcount = _outside(polygon, pcount, nsn, nsd)
        area, moment = _moments(blocked, bcount, spacing)
        if area > area_tolerance and count < MAX_FACETS:
            areas[count] = area
            normals[count] = box_normal
            centres[count] = moment / area
            kinds[count] = KIND_SOLID
            faces[count] = face
            count += 1
        neighbour_wet = int(0)
        if resolved_liquid(alpha[NI, NJ, NK], theta[NI, NJ, NK], wet_threshold) != 0:
            neighbour_wet = 1
        if neighbour_wet == 0:
            area, moment = _moments(opened, ocount, spacing)
            if area > area_tolerance and count < MAX_FACETS:
                areas[count] = area
                normals[count] = box_normal
                centres[count] = moment / area
                kinds[count] = KIND_FREE
                faces[count] = face
                count += 1
            continue
        nwn = wp.vec3d(liquid_normals[NI, NJ, NK])
        nwd = wp.float64(liquid_offsets[NI, NJ, NK]) + wp.float64(sign) * nwn[axis]
        common, ccount = _clip(opened, ocount, nwn, nwd)
        exposed, ecount = _outside(opened, ocount, nwn, nwd)
        area, moment = _moments(exposed, ecount, spacing)
        if area > area_tolerance and count < MAX_FACETS:
            areas[count] = area
            normals[count] = box_normal
            centres[count] = moment / area
            kinds[count] = KIND_FREE
            faces[count] = face
            count += 1
        area, moment = _moments(common, ccount, spacing)
        if area > area_tolerance and count < MAX_FACETS:
            areas[count] = area
            normals[count] = box_normal
            centres[count] = moment / area
            kinds[count] = KIND_SHARED
            faces[count] = face
            count += 1

    # The cell's own solid and liquid plane sections.
    for extra in range(2):
        present = has_solid
        n = sn
        d = sd
        other = wn
        other_d = wd
        has_other = has_liquid
        kind = KIND_SOLID
        if extra == 1:
            present = has_liquid
            n = wn
            d = wd
            other = sn
            other_d = sd
            has_other = has_solid
            kind = KIND_FREE
        if present == 0:
            continue
        skip = int(0)
        for face in range(6):
            bn, bd = _box_plane(face)
            if _coincident(n, d, bn, bd) == 1:
                skip = 1
        if extra == 1 and has_solid == 1 and _coincident(n, d, sn, sd) == 1:
            skip = 1
        if skip == 1:
            continue
        polygon, pcount = _section(n, d, other, other_d, has_other)
        area, moment = _moments(polygon, pcount, spacing)
        if area > area_tolerance and count < MAX_FACETS:
            physical = wp.vec3d(n[0] / spacing[0], n[1] / spacing[1], n[2] / spacing[2])
            physical = physical / wp.sqrt(wp.dot(physical, physical))
            areas[count] = area
            normals[count] = physical
            centres[count] = moment / area
            kinds[count] = kind
            faces[count] = -1
            count += 1

    # Volume by the divergence theorem about the liquid centroid (from PLIC). Any
    # reference point is exact in exact arithmetic; the centroid lies inside a
    # film, so opposing facets contribute h/2 each instead of cancelling two
    # half-cell terms, which keeps float64 volumes accurate for thin films.
    local_centroid = wp.cw_mul(wp.vec3d(liquid_centroids[I, J, K]), spacing)
    liquid_volume = wp.float64(0.0)
    for f in range(count):
        liquid_volume += areas[f] * wp.dot(centres[f] - local_centroid, normals[f])
    liquid_volume = liquid_volume / wp.float64(3.0)
    residual = wp.float64(0.0)
    for a in range(3):
        for b in range(3):
            value = wp.float64(0.0)
            for f in range(count):
                value += (centres[f][a] - local_centroid[a]) * areas[f] * normals[f][b]
            if a == b:
                value -= liquid_volume
            residual = wp.max(residual, wp.abs(value))
    for f in range(count):
        facet_area[c, f] = areas[f]
        facet_normal[c, f] = normals[f]
        facet_centre[c, f] = centres[f]
        facet_kind[c, f] = kinds[f]
        facet_face[c, f] = faces[f]
    facet_count[c] = count
    volume[c] = liquid_volume
    centroid[c] = local_centroid
    volume_error[c] = wp.abs(liquid_volume / target - wp.float64(1.0))
    affine_error[c] = residual / liquid_volume
    wp.atomic_max(maxima, 0, volume_error[c])
    wp.atomic_max(maxima, 1, affine_error[c])
    free = int(0)
    for f in range(count):
        if kinds[f] == KIND_FREE:
            free += 1
    if free > 0:
        has_free[c] = 1
        wp.atomic_add(free_facets, 0, free)


@wp.func
def _invert6(m: _Mat6, n: int) -> _Mat6:
    """Gauss-Jordan inverse of the leading n x n block with partial pivoting."""
    a = _Mat6()
    inv = _Mat6()
    for i in range(n):
        for j in range(n):
            a[i, j] = m[i, j]
        inv[i, i] = wp.float64(1.0)
    for col in range(n):
        pivot = col
        best = wp.abs(a[col, col])
        for row in range(col + 1, n):
            if wp.abs(a[row, col]) > best:
                best = wp.abs(a[row, col])
                pivot = row
        if pivot != col:
            for j in range(n):
                t = a[col, j]
                a[col, j] = a[pivot, j]
                a[pivot, j] = t
                t = inv[col, j]
                inv[col, j] = inv[pivot, j]
                inv[pivot, j] = t
        scale = wp.float64(1.0) / a[col, col]
        for j in range(n):
            a[col, j] = a[col, j] * scale
            inv[col, j] = inv[col, j] * scale
        for row in range(n):
            if row != col:
                factor = a[row, col]
                if factor != wp.float64(0.0):
                    for j in range(n):
                        a[row, j] = a[row, j] - factor * a[col, j]
                        inv[row, j] = inv[row, j] - factor * inv[col, j]
    return inv


@wp.kernel
def _liquid_local_systems(
    cells: wp.array(dtype=wp.vec3i),
    alpha: wp.array3d(dtype=wp.float32),
    theta: wp.array3d(dtype=wp.float32),
    u_aperture: wp.array3d(dtype=wp.float32),
    v_aperture: wp.array3d(dtype=wp.float32),
    w_aperture: wp.array3d(dtype=wp.float32),
    facet_area: wp.array2d(dtype=wp.float64),
    facet_normal: wp.array2d(dtype=wp.vec3d),
    facet_centre: wp.array2d(dtype=wp.vec3d),
    facet_kind: wp.array2d(dtype=wp.int32),
    facet_count: wp.array(dtype=wp.int32),
    volume: wp.array(dtype=wp.float64),
    centroid: wp.array(dtype=wp.vec3d),
    nx: int, ny: int, nz: int, ng: int,
    periodic: wp.vec3i,
    wall_side: _Vec6i,
    spacing: wp.vec3d,
    extension_inertia: wp.float64,
    wet_threshold: wp.float32,
    widths: wp.array(dtype=wp.int32),
    mass: wp.array3d(dtype=wp.float64),
    divergence: wp.array2d(dtype=wp.float64),
    response: wp.array3d(dtype=wp.float64),
    weights: wp.array2d(dtype=wp.float64),
    projector: wp.array3d(dtype=wp.float64),
    initial: wp.array2d(dtype=wp.float64),
    gravity: wp.array2d(dtype=wp.vec3d),
    pressure_gravity: wp.array(dtype=wp.vec3d),
    dof_axis: wp.array2d(dtype=wp.int32),
    dof_position: wp.array2d(dtype=wp.vec3i),
    dof_sign: wp.array2d(dtype=wp.int32),
    dof_shared: wp.array2d(dtype=wp.int32),
):
    c = wp.tid()
    cell = cells[c]
    i = cell[0]
    j = cell[1]
    k = cell[2]
    count = facet_count[c]
    liquid_volume = volume[c]
    liquid_centroid = centroid[c]
    cell_volume = spacing[0] * spacing[1] * spacing[2]

    # Kept MAC faces: open aperture and not a wall boundary.
    kept_axis = _Vec6i()
    kept_sign = _Vec6i()
    kept_face = _Vec6i()
    n = int(0)
    for face in range(6):
        axis = face // 2
        sign = 2 * (face % 2) - 1
        position = wp.vec3i(i, j, k)
        if sign > 0:
            position[axis] = position[axis] + 1
        size = nx
        if axis == 1:
            size = ny
        elif axis == 2:
            size = nz
        boundary = int(0)
        if position[axis] == 0 or position[axis] == size:
            boundary = 1
        aperture = wp.float32(0.0)
        PI = position[0] + ng
        PJ = position[1] + ng
        PK = position[2] + ng
        if axis == 0:
            aperture = u_aperture[PI, PJ, PK]
        elif axis == 1:
            aperture = v_aperture[PI, PJ, PK]
        else:
            aperture = w_aperture[PI, PJ, PK]
        if aperture == wp.float32(0.0):
            continue
        if boundary == 1 and periodic[axis] == 0 and wall_side[face] == 1:
            continue
        kept_axis[n] = axis
        kept_sign[n] = sign
        kept_face[n] = face
        n += 1
    widths[c] = n
    if n == 0:
        return

    # Facet sums over all facets (null projector and stabilisation trace).
    NtN = wp.mat33d()
    trace = wp.float64(0.0)
    surface_area = wp.float64(0.0)
    for f in range(count):
        nf = facet_normal[c, f]
        NtN += wp.outer(nf, nf)
        vR = facet_area[c, f] * (facet_centre[c, f] - liquid_centroid)
        trace += wp.dot(vR, vR)
        surface_area += facet_area[c, f]
    G = wp.inverse(NtN)
    s = trace / (wp.float64(3.0) * liquid_volume)

    # Restricted sums over active facets: P = B^T (A R) (mass and displacement
    # couplings), Q = B'^T n (null projector), S = B'^T B' and the divergence d.
    P = _Facets()
    Q = _Facets()
    S = _Mat6()
    wall_metric = _Mat6()
    d = _Vec6()
    for f in range(count):
        area = facet_area[c, f]
        nf = facet_normal[c, f]
        centre = facet_centre[c, f]
        xi = wp.cw_div(centre, spacing)
        R = centre - liquid_centroid
        weight_vec = _Vec6()
        for col in range(n):
            axis = kept_axis[col]
            w = xi[axis]
            if kept_sign[col] < 0:
                w = wp.float64(1.0) - xi[axis]
            b = wp.float64(kept_sign[col]) * nf[axis] * w        # B / A
            weight_vec[col] = b
        if facet_kind[c, f] == KIND_SOLID:
            # An open MAC face can still border a liquid piece closed by a
            # neighbouring solid plane. Opposite-face differences alone then
            # miss rigid modes normal to those walls. Complete the existing
            # extension metric with the RT0 normal traces at solid facets.
            # This is positive and vanishes for wall-tangent rigid motion.
            # Area weights avoid giving a tiny wall facet excessive inertia.
            for col in range(n):
                for other in range(n):
                    wall_metric[col, other] += (area / surface_area
                        * weight_vec[col] * weight_vec[other])
            continue
        for col in range(n):
            b = weight_vec[col]
            P[col] = P[col] + b * area * R
            Q[col] = Q[col] + b * nf
            d[col] = d[col] + b * area
        for col in range(n):
            for other in range(n):
                S[col, other] = S[col, other] + weight_vec[col] * weight_vec[other]
    # Bound the stabilisation scale by the MAC consistency trace per kept face
    # (see the module docstring): inactive for full cells, it keeps a sliver's
    # non-affine inertia of the order of its liquid volume.
    mac_trace = wp.float64(0.0)
    for col in range(n):
        mac_trace += wp.dot(P[col], P[col])
    s = wp.min(s, wp.float64(SLIVER_STABILISATION_CAP) * mac_trace / (liquid_volume * wp.float64(n)))
    M = _Mat6()
    for col in range(n):
        for other in range(n):
            value = wp.dot(P[col], P[other]) / liquid_volume
            value += s * (S[col, other] - wp.dot(Q[col], G * Q[other]))
            if extension_inertia > wp.float64(0.0):
                deficit = wp.max(wp.float64(0.0), wp.float64(1.0) - liquid_volume / cell_volume)
                extension = wall_metric[col, other]
                if kept_axis[col] == kept_axis[other]:
                    extension += wp.float64(0.5)
                value += extension_inertia * liquid_volume * wp.float64(0.5) * deficit * extension
            M[col, other] = value
    H = _invert6(M, n)
    h = _Vec6()
    for col in range(n):
        value = wp.float64(0.0)
        for other in range(n):
            value += H[col, other] * d[other]
        h[col] = value
    denominator = wp.float64(0.0)
    for col in range(n):
        denominator += d[col] * h[col]
    pg = wp.vec3d(wp.float64(0.0))
    for col in range(n):
        wgt = h[col] / denominator
        weights[c, col] = wgt
        initial[c, col] = d[col] / denominator
        divergence[c, col] = d[col]
        pg += wgt * P[col]
        grav = wp.vec3d(wp.float64(0.0))
        for other in range(n):
            e = H[col, other] - h[col] * h[other] / denominator
            response[c, col, other] = e
            mass[c, col, other] = M[col, other]
            proj = -wgt * d[other]
            if col == other:
                proj += wp.float64(1.0)
            projector[c, col, other] = proj
            grav += e * P[other]
        gravity[c, col] = grav
        axis = kept_axis[col]
        sign = kept_sign[col]
        position = wp.vec3i(i, j, k)
        if sign > 0:
            position[axis] = position[axis] + 1
        dof_axis[c, col] = axis
        dof_position[c, col] = position
        dof_sign[c, col] = sign
        nb = _neighbour_index(i, j, k, axis, sign, nx, ny, nz, periodic)
        shared = int(0)
        if nb[3] == 1:
            if resolved_liquid(alpha[nb[0] + ng, nb[1] + ng, nb[2] + ng],
                               theta[nb[0] + ng, nb[1] + ng, nb[2] + ng], wet_threshold) != 0:
                shared = 1
        dof_shared[c, col] = shared
    pressure_gravity[c] = pg


@dataclass
class LiquidLocalSystems:
    """Host copies of the device assembly (tests and diagnostics)."""
    cell_index: np.ndarray
    facet_area: np.ndarray
    facet_normal: np.ndarray
    facet_centre: np.ndarray
    facet_kind: np.ndarray
    facet_face: np.ndarray
    facet_count: np.ndarray
    volume: np.ndarray
    centroid: np.ndarray
    volume_error: np.ndarray
    affine_error: np.ndarray
    widths: np.ndarray
    mass: np.ndarray
    divergence: np.ndarray
    response: np.ndarray
    weights: np.ndarray
    projector: np.ndarray
    initial: np.ndarray
    gravity: np.ndarray
    pressure_gravity: np.ndarray
    dof_axis: np.ndarray
    dof_position: np.ndarray
    dof_sign: np.ndarray
    dof_shared: np.ndarray
    residue_cells: int
    residue_volume: float


class DeviceLocalSystems:
    """Device arrays of one liquid geometry's facets and MAC local systems."""

    def __init__(self, **arrays):
        self.__dict__.update(arrays)
        self._host = None

    def host(self):
        if self._host is None:
            names = [f.name for f in LiquidLocalSystems.__dataclass_fields__.values()]
            values = {}
            for name in names:
                value = getattr(self, name)
                values[name] = value.numpy() if isinstance(value, wp.array) else value
            self._host = LiquidLocalSystems(**values)
        return self._host


def assemble_local_systems(state, plic, *, extension_inertia, wet_threshold=1e-6):
    """Run the device assembly for the wet cells of ``state``; arrays stay on the device.

    Resolved liquid follows :mod:`shoreloop.physics3d.resolved`: more than
    ``wet_threshold`` of a cell of liquid, or a full cell however small its
    open part.
    """
    g = state.cfg.grid
    device = state.device
    alpha = state.get_interior("alpha")
    theta = state.get_geometry_interior("theta")
    wet = resolved_liquid_mask(alpha, theta, wet_threshold)
    cell_index = np.argwhere(wet).astype(np.int32)
    residue = (alpha > 0) & (theta > 0) & ~wet
    residue_cells = int(np.count_nonzero(residue))
    residue_volume = float(np.sum((alpha * theta)[residue])) * g.dx * g.dy * g.dz
    count = len(cell_index)
    periodic = wp.vec3i(*[int(getattr(state.cfg.bc, a + "lo") == BC.PERIODIC) for a in "xyz"])
    sides = [getattr(state.cfg.bc, "xyz"[face // 2] + ("hi" if face % 2 else "lo")) for face in range(6)]
    boundary_kind = _Vec6i(*[KIND_SOLID if side == BC.WALL else KIND_FREE for side in sides])
    wall_side = _Vec6i(*[int(side == BC.WALL) for side in sides])
    spacing = wp.vec3d(g.dx, g.dy, g.dz)

    def zeros(shape, dtype):
        return wp.zeros(shape, dtype=dtype, device=device)

    cells = wp.array(cell_index, dtype=wp.vec3i, device=device)
    d = dict(
        cell_index=cell_index.astype(np.int64), count=count, cells=cells,
        facet_area=zeros((count, MAX_FACETS), wp.float64), facet_normal=zeros((count, MAX_FACETS), wp.vec3d),
        facet_centre=zeros((count, MAX_FACETS), wp.vec3d), facet_kind=zeros((count, MAX_FACETS), wp.int32),
        facet_face=zeros((count, MAX_FACETS), wp.int32), facet_count=zeros(count, wp.int32),
        volume=zeros(count, wp.float64), centroid=zeros(count, wp.vec3d),
        volume_error=zeros(count, wp.float64), affine_error=zeros(count, wp.float64),
        has_free=zeros(count, wp.int32), maxima=zeros(2, wp.float64), free_facets=zeros(1, wp.int32),
        widths=zeros(count, wp.int32), mass=zeros((count, MAX_MAC, MAX_MAC), wp.float64),
        divergence=zeros((count, MAX_MAC), wp.float64), response=zeros((count, MAX_MAC, MAX_MAC), wp.float64),
        weights=zeros((count, MAX_MAC), wp.float64), projector=zeros((count, MAX_MAC, MAX_MAC), wp.float64),
        initial=zeros((count, MAX_MAC), wp.float64), gravity=zeros((count, MAX_MAC), wp.vec3d),
        pressure_gravity=zeros(count, wp.vec3d), dof_axis=zeros((count, MAX_MAC), wp.int32),
        dof_position=zeros((count, MAX_MAC), wp.vec3i), dof_sign=zeros((count, MAX_MAC), wp.int32),
        dof_shared=zeros((count, MAX_MAC), wp.int32),
        residue_cells=residue_cells, residue_volume=residue_volume,
    )
    if count:
        wp.launch(_liquid_facets, dim=count, device=device, inputs=[
            cells, state.alpha, state.theta, state.open_normals, state.open_offsets,
            plic.normals, plic.offsets, plic.centroids,
            g.nx, g.ny, g.nz, g.ng, periodic, boundary_kind, spacing, wp.float32(wet_threshold),
            d["facet_area"], d["facet_normal"], d["facet_centre"], d["facet_kind"], d["facet_face"],
            d["facet_count"], d["volume"], d["centroid"], d["volume_error"], d["affine_error"],
            d["has_free"], d["maxima"], d["free_facets"],
        ])
        wp.launch(_liquid_local_systems, dim=count, device=device, inputs=[
            cells, state.alpha, state.theta, state.u_aperture, state.v_aperture, state.w_aperture,
            d["facet_area"], d["facet_normal"], d["facet_centre"], d["facet_kind"], d["facet_count"],
            d["volume"], d["centroid"], g.nx, g.ny, g.nz, g.ng, periodic, wall_side, spacing,
            wp.float64(extension_inertia), wp.float32(wet_threshold),
            d["widths"], d["mass"], d["divergence"], d["response"], d["weights"], d["projector"],
            d["initial"], d["gravity"], d["pressure_gravity"], d["dof_axis"], d["dof_position"],
            d["dof_sign"], d["dof_shared"],
        ])
    return DeviceLocalSystems(**d)


# ---------------------------------------------------------------------------
# Hybrid-pressure sparse assembly, MAC transfers and reductions on the device.

@wp.kernel
def _dof_codes(widths: wp.array(dtype=wp.int32), dof_axis: wp.array2d(dtype=wp.int32),
               dof_position: wp.array2d(dtype=wp.vec3i), dof_shared: wp.array2d(dtype=wp.int32),
               shape: wp.vec3i, periodic: wp.vec3i, codes: wp.array2d(dtype=wp.int32)):
    """Face code of every shared DOF (canonical after periodic wrap), else -1."""
    c = wp.tid()
    for i in range(widths[c]):
        code = int(-1)
        if dof_shared[c, i] == 1:
            axis = dof_axis[c, i]
            position = dof_position[c, i]
            if periodic[axis] == 1:
                position[axis] = position[axis] % shape[axis]
            code = ((axis * (shape[0] + 1) + position[0]) * (shape[1] + 1) + position[1]) * (shape[2] + 1) + position[2]
        codes[c, i] = code


@wp.kernel
def _mark_codes(widths: wp.array(dtype=wp.int32), codes: wp.array2d(dtype=wp.int32), mark: wp.array(dtype=wp.int32)):
    c = wp.tid()
    for i in range(widths[c]):
        if codes[c, i] >= 0:
            mark[codes[c, i]] = 1


@wp.kernel
def _assign_keys(widths: wp.array(dtype=wp.int32), codes: wp.array2d(dtype=wp.int32),
                 scan: wp.array(dtype=wp.int32), keys: wp.array2d(dtype=wp.int32),
                 keyed: wp.array(dtype=wp.int32)):
    c = wp.tid()
    n = int(0)
    for i in range(widths[c]):
        key = int(-1)
        if codes[c, i] >= 0:
            key = scan[codes[c, i]] - 1
            n += 1
        keys[c, i] = key
    keyed[c] = n


@wp.kernel
def _flat_keys(widths: wp.array(dtype=wp.int32), starts: wp.array(dtype=wp.int32),
               keys: wp.array2d(dtype=wp.int32), flat: wp.array(dtype=wp.int32)):
    c = wp.tid()
    for i in range(widths[c]):
        flat[starts[c] + i] = keys[c, i]


@wp.kernel
def _ground_cells(widths: wp.array(dtype=wp.int32), keys: wp.array2d(dtype=wp.int32),
                  face_grounded: wp.array(dtype=wp.int32), cell_grounded: wp.array(dtype=wp.int32),
                  changed: wp.array(dtype=wp.int32)):
    c = wp.tid()
    if cell_grounded[c] == 1:
        return
    for i in range(widths[c]):
        key = keys[c, i]
        if key >= 0 and face_grounded[key] == 1:
            cell_grounded[c] = 1
            changed[0] = 1
            return


@wp.kernel
def _ground_faces(widths: wp.array(dtype=wp.int32), keys: wp.array2d(dtype=wp.int32),
                  cell_grounded: wp.array(dtype=wp.int32), face_grounded: wp.array(dtype=wp.int32),
                  changed: wp.array(dtype=wp.int32)):
    c = wp.tid()
    if cell_grounded[c] == 0:
        return
    for i in range(widths[c]):
        key = keys[c, i]
        if key >= 0 and face_grounded[key] == 0:
            face_grounded[key] = 1
            changed[0] = 1


@wp.kernel
def _matrix_triplets(widths: wp.array(dtype=wp.int32), keys: wp.array2d(dtype=wp.int32),
                     response: wp.array3d(dtype=wp.float64), offsets: wp.array(dtype=wp.int32),
                     gauge: wp.array(dtype=wp.int32), rows: wp.array(dtype=wp.int32),
                     cols: wp.array(dtype=wp.int32), values: wp.array(dtype=wp.float64)):
    """Response entries between keyed DOFs; gauge rows/columns are zeroed here and
    receive a unit diagonal separately."""
    c = wp.tid()
    n = offsets[c]
    for i in range(widths[c]):
        ki = keys[c, i]
        if ki < 0:
            continue
        for j in range(widths[c]):
            kj = keys[c, j]
            if kj < 0:
                continue
            value = response[c, i, j]
            if gauge[ki] == 1 or gauge[kj] == 1:
                value = wp.float64(0.0)
            rows[n] = ki
            cols[n] = kj
            values[n] = value
            n += 1


@wp.kernel
def _gauge_diagonal(gauges: wp.array(dtype=wp.int32), base: int, rows: wp.array(dtype=wp.int32),
                    cols: wp.array(dtype=wp.int32), values: wp.array(dtype=wp.float64)):
    i = wp.tid()
    rows[base + i] = gauges[i]
    cols[base + i] = gauges[i]
    values[base + i] = wp.float64(1.0)


@wp.kernel
def _flux_response_triplets(widths: wp.array(dtype=wp.int32), starts: wp.array(dtype=wp.int32),
                            keys: wp.array2d(dtype=wp.int32), response: wp.array3d(dtype=wp.float64),
                            offsets: wp.array(dtype=wp.int32), rows: wp.array(dtype=wp.int32),
                            cols: wp.array(dtype=wp.int32), values: wp.array(dtype=wp.float64)):
    c = wp.tid()
    n = offsets[c]
    for i in range(widths[c]):
        for j in range(widths[c]):
            kj = keys[c, j]
            if kj < 0:
                continue
            rows[n] = starts[c] + i
            cols[n] = kj
            values[n] = response[c, i, j]
            n += 1


@wp.kernel
def _pressure_response_triplets(widths: wp.array(dtype=wp.int32), keys: wp.array2d(dtype=wp.int32),
                                weights: wp.array2d(dtype=wp.float64), offsets: wp.array(dtype=wp.int32),
                                rows: wp.array(dtype=wp.int32), cols: wp.array(dtype=wp.int32),
                                values: wp.array(dtype=wp.float64)):
    c = wp.tid()
    n = offsets[c]
    for i in range(widths[c]):
        ki = keys[c, i]
        if ki < 0:
            continue
        rows[n] = c
        cols[n] = ki
        values[n] = weights[c, i]
        n += 1


@wp.kernel
def _read_mac(widths: wp.array(dtype=wp.int32), starts: wp.array(dtype=wp.int32),
              dof_axis: wp.array2d(dtype=wp.int32), dof_position: wp.array2d(dtype=wp.vec3i),
              dof_sign: wp.array2d(dtype=wp.int32), u: wp.array3d(dtype=wp.float32),
              v: wp.array3d(dtype=wp.float32), w: wp.array3d(dtype=wp.float32), ng: int,
              values: wp.array(dtype=wp.float64)):
    """Outward normal MAC velocities of every DOF (unwrapped positions, padded fields)."""
    c = wp.tid()
    for i in range(widths[c]):
        p = dof_position[c, i]
        axis = dof_axis[c, i]
        value = wp.float32(0.0)
        if axis == 0:
            value = u[p[0] + ng, p[1] + ng, p[2] + ng]
        elif axis == 1:
            value = v[p[0] + ng, p[1] + ng, p[2] + ng]
        else:
            value = w[p[0] + ng, p[1] + ng, p[2] + ng]
        values[starts[c] + i] = wp.float64(dof_sign[c, i]) * wp.float64(value)


@wp.kernel
def _scatter_mac(widths: wp.array(dtype=wp.int32), starts: wp.array(dtype=wp.int32),
                 dof_axis: wp.array2d(dtype=wp.int32), dof_position: wp.array2d(dtype=wp.vec3i),
                 dof_sign: wp.array2d(dtype=wp.int32), values: wp.array(dtype=wp.float64),
                 shape: wp.vec3i, periodic: wp.vec3i,
                 sum_u: wp.array3d(dtype=wp.float64), sum_v: wp.array3d(dtype=wp.float64),
                 sum_w: wp.array3d(dtype=wp.float64), count_u: wp.array3d(dtype=wp.int32),
                 count_v: wp.array3d(dtype=wp.int32), count_w: wp.array3d(dtype=wp.int32)):
    """Accumulate the two views of each face on its canonical (wrapped) position."""
    c = wp.tid()
    for i in range(widths[c]):
        p = dof_position[c, i]
        axis = dof_axis[c, i]
        if periodic[axis] == 1:
            p[axis] = p[axis] % shape[axis]
        value = wp.float64(dof_sign[c, i]) * values[starts[c] + i]
        if axis == 0:
            wp.atomic_add(sum_u, p[0], p[1], p[2], value)
            wp.atomic_add(count_u, p[0], p[1], p[2], 1)
        elif axis == 1:
            wp.atomic_add(sum_v, p[0], p[1], p[2], value)
            wp.atomic_add(count_v, p[0], p[1], p[2], 1)
        else:
            wp.atomic_add(sum_w, p[0], p[1], p[2], value)
            wp.atomic_add(count_w, p[0], p[1], p[2], 1)


@wp.kernel
def _write_mac_component(sums: wp.array3d(dtype=wp.float64), counts: wp.array3d(dtype=wp.int32),
                         field: wp.array3d(dtype=wp.float32), ng: int):
    i, j, k = wp.tid()
    n = counts[i, j, k]
    if n > 0:
        field[i + ng, j + ng, k + ng] = wp.float32(sums[i, j, k] / wp.float64(n))


@wp.kernel
def _write_pressure(cells: wp.array(dtype=wp.vec3i), pressure: wp.array(dtype=wp.float64),
                    field: wp.array3d(dtype=wp.float32), ng: int):
    c = wp.tid()
    cell = cells[c]
    field[cell[0] + ng, cell[1] + ng, cell[2] + ng] = wp.float32(pressure[c])


@wp.kernel
def _divergence_statistics(widths: wp.array(dtype=wp.int32), starts: wp.array(dtype=wp.int32),
                           divergence: wp.array2d(dtype=wp.float64), volume: wp.array(dtype=wp.float64),
                           dofs: wp.array(dtype=wp.float64), sum_squares: wp.array(dtype=wp.float64),
                           maximum: wp.array(dtype=wp.float64)):
    c = wp.tid()
    value = wp.float64(0.0)
    for i in range(widths[c]):
        value += divergence[c, i] * dofs[starts[c] + i]
    value = value / volume[c]
    wp.atomic_add(sum_squares, 0, value * value)
    wp.atomic_max(maximum, 0, wp.abs(value))


@wp.kernel
def _divergence_per_cell(widths: wp.array(dtype=wp.int32), starts: wp.array(dtype=wp.int32),
                         divergence: wp.array2d(dtype=wp.float64), volume: wp.array(dtype=wp.float64),
                         dofs: wp.array(dtype=wp.float64), values: wp.array(dtype=wp.float64)):
    c = wp.tid()
    value = wp.float64(0.0)
    for i in range(widths[c]):
        value += divergence[c, i] * dofs[starts[c] + i]
    values[c] = value / volume[c]


@wp.kernel
def _kinetic_energy(widths: wp.array(dtype=wp.int32), starts: wp.array(dtype=wp.int32),
                    mass: wp.array3d(dtype=wp.float64), dofs: wp.array(dtype=wp.float64),
                    total: wp.array(dtype=wp.float64)):
    c = wp.tid()
    energy = wp.float64(0.0)
    for i in range(widths[c]):
        row = wp.float64(0.0)
        for j in range(widths[c]):
            row += mass[c, i, j] * dofs[starts[c] + j]
        energy += dofs[starts[c] + i] * row
    wp.atomic_add(total, 0, energy)


def exclusive_scan(values, device):
    """Exclusive prefix sum of an int32 device array and its total (host int)."""
    inclusive = wp.zeros_like(values)
    if values.size:
        wp.utils.array_scan(values, inclusive, inclusive=True)
        total = int(inclusive.numpy()[-1])
    else:
        total = 0
    exclusive = wp.zeros(values.size + 1, dtype=wp.int32, device=device)
    if values.size:
        wp.copy(exclusive, inclusive, dest_offset=1, count=values.size)
    return exclusive, total
