"""Convex cell-local water/solid geometry shared by PLIC and swept volumes.

Both halfspaces use ``dot(n, xi) <= d`` in Cartesian cell coordinates. The
solid plane points from open volume into solid; the liquid plane points from
water into air. Zero normals omit a constraint (an uncut or full-water box).
All geometry returned here is in units of a full Cartesian cell.
"""

import numpy as np
import warp as wp


_Polygon = wp.types.matrix(shape=(12, 3), dtype=wp.float32)
_Planes = wp.types.matrix(shape=(4, 3), dtype=wp.float32)
_Faces = wp.types.matrix(shape=(120, 3), dtype=wp.float32)
_Counts = wp.types.vector(length=10, dtype=wp.int32)


def open_face_area_host(axis, low_normal, low_offset, high_normal, high_offset):
    """Common open area on the high cell's xi[axis]=0 face, on the host."""
    transverse = [q for q in range(3) if q != axis]
    points = np.zeros((4, 3))
    points[:, transverse] = ((0, 0), (1, 0), (1, 1), (0, 1))
    for normal, offset in ((low_normal, low_offset), (high_normal, high_offset)):
        if len(points) == 0:
            return 0.0
        result = []
        previous = points[-1]
        previous_value = np.dot(normal, previous) - offset
        for point in points:
            value = np.dot(normal, point) - offset
            if (value <= 0) != (previous_value <= 0):
                weight = previous_value / (previous_value - value)
                result.append(previous + weight * (point - previous))
            if value <= 0:
                result.append(point)
            previous, previous_value = point, value
        points = np.asarray(result)
    area = 0.0
    for index in range(1, len(points) - 1):
        area += .5 * abs(np.cross(points[index] - points[0],
                                  points[index + 1] - points[0])[axis])
    return area


@wp.func
def _clip_polygon(
    points: _Polygon,
    count: int,
    normal: wp.vec3,
    offset: wp.float32,
) -> tuple[_Polygon, int]:
    clipped = _Polygon()
    size = int(0)
    if count == 0:
        return clipped, size
    previous = points[count - 1]
    previous_value = wp.dot(normal, previous) - offset
    for index in range(count):
        point = points[index]
        value = wp.dot(normal, point) - offset
        if (value <= wp.float32(0.0)) != (previous_value <= wp.float32(0.0)):
            weight = previous_value / (previous_value - value)
            clipped[size] = previous + weight * (point - previous)
            size += 1
        if value <= wp.float32(0.0):
            clipped[size] = point
            size += 1
        previous = point
        previous_value = value
    return clipped, size


@wp.func
def _box_constraint(
    face: int,
    lower: wp.vec3,
    upper: wp.vec3,
    planes: _Planes,
    offsets: wp.vec4,
) -> tuple[wp.vec3, wp.float32]:
    normal = wp.vec3(wp.float32(0.0))
    offset = wp.float32(0.0)
    if face < 6:
        axis = face // 2
        normal = wp.vec3(wp.float32(0.0))
        normal[axis] = wp.float32(-1.0)
        offset = -lower[axis]
        if face % 2 == 1:
            normal[axis] = wp.float32(1.0)
            offset = upper[axis]
    else:
        normal = planes[face - 6]
        offset = offsets[face - 6]
    return normal, offset


@wp.func
def _plane_quad(normal: wp.vec3, offset: wp.float32) -> _Polygon:
    """An oriented square covering the unit cube's section by this plane."""
    n = wp.normalize(normal)
    reference = wp.vec3(wp.float32(1.0), wp.float32(0.0), wp.float32(0.0))
    if wp.abs(n[0]) > wp.float32(0.8):
        reference = wp.vec3(wp.float32(0.0), wp.float32(1.0), wp.float32(0.0))
    tangent = wp.normalize(wp.cross(n, reference)) * wp.float32(2.0)
    bitangent = wp.cross(n, tangent)
    centre = wp.vec3(wp.float32(0.5))
    centre += normal * ((offset - wp.dot(normal, centre)) / wp.dot(normal, normal))
    points = _Polygon()
    points[0] = centre - tangent - bitangent
    points[1] = centre + tangent - bitangent
    points[2] = centre + tangent + bitangent
    points[3] = centre - tangent + bitangent
    return points


@wp.func
def _redundant_constraint(
    face: int,
    lower: wp.vec3,
    upper: wp.vec3,
    planes: _Planes,
    offsets: wp.vec4,
    constraint_count: int,
) -> int:
    """Whether a constraint is implied by an exactly parallel, same-direction one.

    The axis-slab flux passes bitwise-identical solid planes for a
    translation-invariant beach. Clipping a polygon by its own plane is decided
    by roundoff, so a redundant plane is dropped as a face and as a clipper;
    the tighter parallel plane is kept and the lower index wins an exact tie.
    """
    normal, offset = _box_constraint(face, lower, upper, planes, offsets)
    if wp.dot(normal, normal) == wp.float32(0.0):
        return 1
    for other in range(constraint_count):
        if other != face:
            n, d = _box_constraint(other, lower, upper, planes, offsets)
            # Identical components, not a zero cross product: under fused
            # multiply-add a*b - b*a is not exactly zero in float32.
            if n[0] == normal[0] and n[1] == normal[1] and n[2] == normal[2]:
                if d < offset or (d == offset and other < face):
                    return 1
    return 0


@wp.func
def convex_box_moments(
    lower: wp.vec3,
    upper: wp.vec3,
    planes: _Planes,
    offsets: wp.vec4,
    plane_count: int,
) -> wp.vec4:
    """Return volume and first moment (V, integral x, integral y, integral z).

    Clip the boundary polygons. Their vertex mean is inside
    the intersection, so tetrahedral volumes have one sign even for a thin
    sheet. This avoids cancellation of signed tetrahedra about an exterior
    origin. No alpha, mass, or pressure is changed by geometric clipping.
    """
    if (upper[0] <= lower[0] or upper[1] <= lower[1]
            or upper[2] <= lower[2]):
        return wp.vec4(wp.float32(0.0))
    for constraint in range(plane_count):
        n = planes[constraint]
        if wp.dot(n, n) == wp.float32(0.0) and offsets[constraint] < wp.float32(0.0):
            return wp.vec4(wp.float32(0.0))
    faces = _Faces()
    counts = _Counts()
    vertex_sum = wp.vec3(wp.float32(0.0))
    vertex_count = int(0)
    # These nested polygon loops must stay loops, rather than expanding 64
    # matrix-valued branches into each calling reconstruction kernel.
    constraint_count = 6 + plane_count
    redundant = _Counts()
    for face in range(constraint_count):
        redundant[face] = _redundant_constraint(face, lower, upper, planes, offsets, constraint_count)
    for face in range(constraint_count):
        if redundant[face] == 0:
            normal, offset = _box_constraint(
                face, lower, upper, planes, offsets,
            )
            polygon = _plane_quad(normal, offset)
            count = int(4)
            for constraint in range(constraint_count):
                if constraint != face and redundant[constraint] == 0:
                    n, d = _box_constraint(
                        constraint, lower, upper, planes, offsets,
                    )
                    polygon, count = _clip_polygon(polygon, count, n, d)
            counts[face] = count
            for vertex in range(count):
                point = polygon[vertex]
                faces[12 * face + vertex] = point
                vertex_sum += point
                vertex_count += 1
    if vertex_count == 0:
        return wp.vec4(wp.float32(0.0))
    centre = vertex_sum / wp.float32(vertex_count)
    volume = wp.float32(0.0)
    moment = wp.vec3(wp.float32(0.0))
    for face in range(constraint_count):
        count = counts[face]
        if count >= 3:
            a = faces[12 * face]
            for triangle in range(1, count - 1):
                b = faces[12 * face + triangle]
                c = faces[12 * face + triangle + 1]
                tetra_volume = wp.abs(wp.dot(
                    a - centre, wp.cross(b - centre, c - centre),
                )) / wp.float32(6.0)
                volume += tetra_volume
                moment += tetra_volume * (centre + a + b + c) / wp.float32(4.0)
    return wp.vec4(volume, moment[0], moment[1], moment[2])


@wp.func
def clipped_box_moments(
    lower: wp.vec3,
    upper: wp.vec3,
    solid_normal: wp.vec3,
    solid_offset: wp.float32,
    water_normal: wp.vec3,
    water_offset: wp.float32,
) -> wp.vec4:
    planes = _Planes()
    planes[0] = solid_normal
    planes[1] = water_normal
    return convex_box_moments(
        lower, upper, planes,
        wp.vec4(solid_offset, water_offset, wp.float32(0.0), wp.float32(0.0)), 2,
    )
