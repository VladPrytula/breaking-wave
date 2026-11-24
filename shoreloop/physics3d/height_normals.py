"""Volume-column interface normals on supported graph patches.

A complete column integral is its mean interface height. Centred transverse
differences reproduce planar normals, including on anisotropic grids. Only a
single monotone water/air transition in every 3x3 column is admissible; solid
contact, unresolved sheets and incomplete columns retain their existing
orientation. Alpha is never changed, and PLIC still fits each cell's volume.
"""

import warp as wp

from shoreloop.physics3d.extrapolation import _wrapped_cell_index


@wp.func
def _normal_height_column(
    alpha: wp.array3d(dtype=wp.float32),
    theta: wp.array3d(dtype=wp.float32),
    cell: wp.vec3i, axis: int, a: int, b: int, sign: wp.float32,
    shape: wp.vec3i, periodic: wp.vec3i, tolerance: wp.float32, ng: int,
) -> wp.float32:
    total = wp.float32(0.0)
    previous = wp.float32(0.0)
    # Nine samples cover all transverse columns of a planar 3x3 patch when
    # the dominant normal component is chosen in cell coordinates.
    for sample in range(9):
        shift = wp.vec3i(sample - 4, a, b)
        if axis == 1:
            shift = wp.vec3i(a, sample - 4, b)
        elif axis == 2:
            shift = wp.vec3i(a, b, sample - 4)
        q = cell + shift
        i = _wrapped_cell_index(q[0], shape[0], periodic[0])
        j = _wrapped_cell_index(q[1], shape[1], periodic[1])
        k = _wrapped_cell_index(q[2], shape[2], periodic[2])
        if i < 0 or j < 0 or k < 0:
            return wp.float32(-1.0)
        if theta[i + ng, j + ng, k + ng] != wp.float32(1.0):
            return wp.float32(-1.0)
        value = alpha[i + ng, j + ng, k + ng]
        # Orient all columns with liquid below and air above the graph.
        if sign < wp.float32(0.0):
            value = wp.float32(1.0) - value
        if sample == 0:
            if value < wp.float32(1.0) - tolerance:
                return wp.float32(-1.0)
        elif value > previous + tolerance:
            return wp.float32(-1.0)
        if sample == 8 and value > tolerance:
            return wp.float32(-1.0)
        total += value
        previous = value
    return total


@wp.kernel
def apply_height_function_normals(
    alpha: wp.array3d(dtype=wp.float32),
    theta: wp.array3d(dtype=wp.float32),
    normals: wp.array3d(dtype=wp.vec3),
    counts: wp.array(dtype=wp.int32),
    spacing: wp.vec3, shape: wp.vec3i, periodic: wp.vec3i,
    tolerance: wp.float32, ng: int,
):
    i, j, k = wp.tid()
    I, J, K = i + ng, j + ng, k + ng
    fraction = alpha[I, J, K]
    if fraction <= tolerance or fraction >= wp.float32(1.0) - tolerance:
        return
    seed = normals[I, J, K]
    local = wp.cw_mul(seed, spacing)
    axis = int(0)
    if wp.abs(local[1]) > wp.abs(local[0]):
        axis = 1
    if wp.abs(local[2]) > wp.abs(local[axis]):
        axis = 2
    sign = wp.float32(1.0)
    if local[axis] < wp.float32(0.0):
        sign = -wp.float32(1.0)
    heights = wp.mat33(wp.float32(0.0))
    for a in range(3):
        for b in range(3):
            height = _normal_height_column(
                alpha, theta, wp.vec3i(i, j, k), axis, a - 1, b - 1, sign,
                shape, periodic, tolerance, ng,
            )
            if height < wp.float32(0.0):
                return
            heights[a, b] = height
    # Height is measured in cells along axis; derivatives use physical lengths.
    transverse_a = int(0)
    transverse_b = int(1)
    if axis == 0:
        transverse_a = 1
        transverse_b = 2
    elif axis == 1:
        transverse_b = 2
    slope_a = (heights[2, 1] - heights[0, 1]) * spacing[axis] / (
        wp.float32(2.0) * spacing[transverse_a])
    slope_b = (heights[1, 2] - heights[1, 0]) * spacing[axis] / (
        wp.float32(2.0) * spacing[transverse_b])
    normal = wp.vec3(wp.float32(0.0))
    normal[axis] = sign
    normal[transverse_a] = -sign * slope_a
    normal[transverse_b] = -sign * slope_b
    normals[I, J, K] = wp.normalize(normal)
    wp.atomic_add(counts, 2, 1)
