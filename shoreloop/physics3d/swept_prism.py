"""Unsplit PLIC fluxes through a face traced with its complete MAC velocity.

Velocity is constant on each swept face during a step, with centred MAC
interpolation for transverse components. The face-area/time cube is split
among the cells reached by those characteristics. Each part uses its own
retained liquid and solid planes. At most one cell of travel per component
is supported. This resolves constant-vector translation without independent
axis-slab overlap; deformation of the swept face remains a first-order
approximation. Prism coordinates, clipping and volume sums use CUDA FP64:
FP32 clipping fails the retained thin-water conservation/momentum controls.
Input state and the shared stored fluxes remain FP32.
Uncut offsets are refitted to retained alpha in that work precision; rounded
FP32 offsets otherwise misrepresent the volume of very thin donor planes.
The existing joint solid/liquid planes are retained in cut cells.

At cut faces, the common open footprint and donor solids constrain the prism.
As in the existing cut flux, aperture * normal Courant owns the open flux;
the traced prism supplies its liquid saturation. Variable open area along a
straight characteristic remains an explicit geometric approximation.
"""

import math

import warp as wp

from shoreloop.config import BC
from shoreloop.physics3d.liquid_assembly import _box_face, _section, _clip
from shoreloop.physics3d.vof_flux import DirectionalFluxResult

_TracePlanes = wp.types.matrix(shape=(6, 3), dtype=wp.float64)
_TraceOffsets = wp.types.vector(length=6, dtype=wp.float64)
_TraceFaces = wp.types.matrix(shape=(144, 3), dtype=wp.float64)
_TraceCounts = wp.types.vector(length=12, dtype=wp.int32)


@wp.func
def _plane_volume(normal: wp.vec3d, offset: wp.float64) -> wp.float64:
    """Factored cube volume, retaining tiny nonzero normal components."""
    a, b, c = wp.abs(normal[0]), wp.abs(normal[1]), wp.abs(normal[2])
    if a > b:
        a, b = b, a
    if b > c:
        b, c = c, b
    if a > b:
        a, b = b, a
    d = offset-(wp.min(normal[0], wp.float64(0.0))
        +wp.min(normal[1], wp.float64(0.0))+wp.min(normal[2], wp.float64(0.0)))
    total = a+b+c
    if d <= wp.float64(0.0):
        return wp.float64(0.0)
    if d >= total:
        return wp.float64(1.0)
    complement = d > wp.float64(0.5)*total
    if complement:
        d = total-d
    value = wp.float64(0.0)
    if b == wp.float64(0.0):
        value = d/c
    elif a == wp.float64(0.0):
        value = (d-wp.float64(0.5)*b)/c
        if d < b:
            value = wp.float64(0.5)*(d/b)*(d/c)
    elif d < a:
        value = (d/a)*(d/b)*(d/c)/wp.float64(6.0)
    elif d >= a+b:
        value = (d-wp.float64(0.5)*(a+b))/c
    else:
        pb, pc = wp.max(d-b, wp.float64(0.0)), wp.max(d-c, wp.float64(0.0))
        value = (wp.float64(3.0)*d*(d-a)+a*a-(pb/a)*pb*pb-(pc/a)*pc*pc)/(wp.float64(6.0)*b*c)
    if complement:
        value = wp.float64(1.0)-value
    return value


@wp.kernel
def _uncut_offsets(alpha: wp.array3d(dtype=wp.float32), theta: wp.array3d(dtype=wp.float32),
                   normals: wp.array3d(dtype=wp.vec3), original: wp.array3d(dtype=wp.float32),
                   offsets: wp.array3d(dtype=wp.float64), ng: int):
    i, j, k = wp.tid()
    I, J, K = i+ng, j+ng, k+ng
    offsets[I, J, K] = wp.float64(original[I, J, K])
    fraction = wp.float64(alpha[I, J, K])
    n = wp.vec3d(normals[I, J, K])
    if (theta[I, J, K] == wp.float32(1.0) and fraction > wp.float64(0.0)
            and fraction < wp.float64(1.0) and wp.dot(n, n) > wp.float64(0.0)):
        low = wp.min(n[0], wp.float64(0.0))+wp.min(n[1], wp.float64(0.0))+wp.min(n[2], wp.float64(0.0))
        high = wp.max(n[0], wp.float64(0.0))+wp.max(n[1], wp.float64(0.0))+wp.max(n[2], wp.float64(0.0))
        for iteration in range(48):
            midpoint = wp.float64(0.5)*(low+high)
            if _plane_volume(n, midpoint) < fraction:
                low = midpoint
            else:
                high = midpoint
        offsets[I, J, K] = wp.float64(0.5)*(low+high)


@wp.func
def _constraint(face: int, planes: _TracePlanes,
                offsets: _TraceOffsets) -> tuple[wp.vec3d, wp.float64]:
    normal = wp.vec3d(wp.float64(0.0))
    offset = wp.float64(0.0)
    if face < 6:
        normal[face//2] = wp.float64(-1.0)
        if face%2 == 1:
            normal[face//2] = wp.float64(1.0)
            offset = wp.float64(1.0)
    else:
        normal = planes[face-6]
        offset = offsets[face-6]
    return normal, offset


@wp.func
def _redundant(q: int, planes: _TracePlanes, offsets: _TraceOffsets, count: int) -> int:
    """Whether constraint q is implied by an exactly parallel, same-direction one.

    Bitwise twins are common: both common-face solids of a translation-invariant
    beach are the same plane. Clipping a polygon by its own plane is decided by
    roundoff, so a redundant plane is dropped both as a face and as a clipper.
    The tighter parallel plane is kept; the lower index wins an exact tie.
    """
    normal, offset = _constraint(q, planes, offsets)
    if wp.dot(normal, normal) == wp.float64(0.0):
        return 1
    for p in range(6+count):
        if p != q:
            n, d = _constraint(p, planes, offsets)
            if n[0] == normal[0] and n[1] == normal[1] and n[2] == normal[2]:
                # Identical components: the twins are copies of one FP32 plane.
                if d < offset or (d == offset and p < q):
                    return 1
            elif wp.dot(n, n) > wp.float64(0.0) and wp.dot(n, normal) > wp.float64(0.0):
                parallel = wp.cross(n, normal)
                if wp.dot(parallel, parallel) == wp.float64(0.0):
                    tighter = d*wp.dot(normal, normal)
                    own = offset*wp.dot(n, normal)
                    if tighter < own or (tighter == own and p < q):
                        return 1
    return 0


@wp.func
def _volume(planes: _TracePlanes, offsets: _TraceOffsets, count: int) -> wp.float64:
    """Positive tetrahedral volume in the face-area/time unit cube.

    Six extra halfspaces are needed for two donor partitions, the donor solid,
    both common-face solids and liquid. The existing polygon clipper is shared;
    this accumulator needs no first moments and stores up to twelve faces.
    """
    all_inside = bool(True)
    for q in range(count):
        n = planes[q]
        low = wp.min(n[0], wp.float64(0.0))+wp.min(n[1], wp.float64(0.0))+wp.min(n[2], wp.float64(0.0))
        high = wp.max(n[0], wp.float64(0.0))+wp.max(n[1], wp.float64(0.0))+wp.max(n[2], wp.float64(0.0))
        if low > offsets[q]:
            return wp.float64(0.0)
        if high > offsets[q]:
            all_inside = False
    if all_inside:
        return wp.float64(1.0)
    redundant = _TraceCounts()
    for face in range(6+count):
        redundant[face] = _redundant(face, planes, offsets, count)
    faces = _TraceFaces()
    counts = _TraceCounts()
    vertex_sum = wp.vec3d(wp.float64(0.0))
    vertex_count = int(0)
    for face in range(6+count):
        if redundant[face] == 0:
            normal, offset = _constraint(face, planes, offsets)
            polygon = _box_face(face)
            size = int(4)
            if face >= 6:
                polygon, size = _section(normal, offset, wp.vec3d(wp.float64(0.0)), wp.float64(0.0), 0)
            for other in range(6, 6+count):
                if other != face and redundant[other] == 0:
                    n, d = _constraint(other, planes, offsets)
                    polygon, size = _clip(polygon, size, n, d)
            counts[face] = size
            for v in range(size):
                point = polygon[v]
                faces[12*face+v] = point
                vertex_sum += point
                vertex_count += 1
    if vertex_count == 0:
        return wp.float64(0.0)
    centre = vertex_sum/wp.float64(vertex_count)
    volume = wp.float64(0.0)
    for face in range(6+count):
        a = faces[12*face]
        for t in range(1, counts[face]-1):
            b, c = faces[12*face+t], faces[12*face+t+1]
            volume += wp.abs(wp.dot(a-centre, wp.cross(b-centre, c-centre)))/wp.float64(6.0)
    return volume


@wp.func
def _at_face(field: wp.array3d(dtype=wp.float32), face: wp.vec3i,
             axis: int, component: int, shape: wp.vec3i,
             periodic: wp.vec3i, ng: int) -> wp.float64:
    if component == axis:
        return wp.float64(field[face[0], face[1], face[2]])
    lo = face
    lo[axis] -= 1
    hi = face
    # Air-band extension owns only interior faces. Its ghost cells are empty;
    # sample the actual periodic neighbour across the normal-axis seam.
    if periodic[axis] != 0:
        if lo[axis] < ng:
            lo[axis] += shape[axis]
        if hi[axis] >= ng+shape[axis]:
            hi[axis] -= shape[axis]
    upper = hi
    upper[component] += 1
    corner = lo
    corner[component] += 1
    return wp.float64(0.25)*(wp.float64(field[lo[0], lo[1], lo[2]])
        +wp.float64(field[hi[0], hi[1], hi[2]])+wp.float64(field[upper[0], upper[1], upper[2]])
        +wp.float64(field[corner[0], corner[1], corner[2]]))


@wp.func
def _donor(face: wp.vec3i, delta: wp.vec3i, courant: wp.vec3d,
           axis: int, shape: wp.vec3i, periodic: wp.vec3i) -> wp.vec3i:
    cell = face+delta
    if courant[axis] > wp.float64(0.0):
        cell[axis] -= 1
    for q in range(3):
        if cell[q] < 0 or cell[q] >= shape[q]:
            if periodic[q] == 0:
                return wp.vec3i(-1)
            cell[q] = (cell[q]+shape[q])%shape[q]
    return cell


@wp.func
def _water_support(alpha: wp.array3d(dtype=wp.float32), theta: wp.array3d(dtype=wp.float32),
                   face: wp.vec3i, velocity: wp.vec3d, shape: wp.vec3i,
                   periodic: wp.vec3i, axis: int, ng: int) -> wp.vec2i:
    """Potential donors depend on velocity signs, shared with the CFL bound."""
    a, b = (axis+1)%3, (axis+2)%3
    count_a = 1+int(velocity[a] != wp.float64(0.0))
    count_b = 1+int(velocity[b] != wp.float64(0.0))
    shift_a, shift_b = -int(wp.sign(velocity[a])), -int(wp.sign(velocity[b]))
    any_water, all_water = int(0), int(1)
    for sa in range(count_a):
        for sb in range(count_b):
            delta = wp.vec3i(0)
            delta[a] = sa*shift_a
            delta[b] = sb*shift_b
            cell = _donor(face, delta, velocity, axis, shape, periodic)
            if cell[0] >= 0:
                D = cell+wp.vec3i(ng)
                if theta[D[0], D[1], D[2]] > wp.float32(0.0):
                    fraction = alpha[D[0], D[1], D[2]]
                    if fraction > wp.float32(0.0):
                        any_water = 1
                    if fraction < wp.float32(1.0):
                        all_water = 0
    return wp.vec2i(any_water, all_water)


@wp.kernel
def _trace_rate(alpha: wp.array3d(dtype=wp.float32), theta: wp.array3d(dtype=wp.float32),
                aperture: wp.array3d(dtype=wp.float32),
                u: wp.array3d(dtype=wp.float32), v: wp.array3d(dtype=wp.float32), w: wp.array3d(dtype=wp.float32),
                rates: wp.array(dtype=wp.float64), inverse_spacing: wp.vec3d,
                shape: wp.vec3i, periodic: wp.vec3i, axis: int, ng: int):
    i, j, k = wp.tid()
    face = wp.vec3i(i, j, k)
    F = face+wp.vec3i(ng)
    if aperture[F[0], F[1], F[2]] <= wp.float32(0.0):
        return
    if periodic[axis] == 0 and (face[axis] == 0 or face[axis] == shape[axis]):
        return
    rate = wp.cw_mul(wp.vec3d(_at_face(u, F, axis, 0, shape, periodic, ng),
        _at_face(v, F, axis, 1, shape, periodic, ng),
        _at_face(w, F, axis, 2, shape, periodic, ng)), inverse_spacing)
    if rate[axis] == wp.float64(0.0):
        return
    if _water_support(alpha, theta, face, rate, shape, periodic, axis, ng)[0] == 0:
        return
    for q in range(3):
        wp.atomic_max(rates, q, wp.abs(rate[q]))


@wp.func
def _record(invalid: wp.array(dtype=wp.int32), detail: wp.array(dtype=wp.float64),
            code: int, face: wp.vec3i, normal_courant: wp.float64,
            open_amount: wp.float64, water_amount: wp.float64):
    """Flag a rejected face; the first rejecting thread also keeps its numbers."""
    wp.atomic_max(invalid, 0, code)
    if wp.atomic_cas(invalid, 1, 0, 1) == 0:
        detail[0] = wp.float64(face[0])
        detail[1] = wp.float64(face[1])
        detail[2] = wp.float64(face[2])
        detail[3] = normal_courant
        detail[4] = open_amount
        detail[5] = water_amount


@wp.kernel
def _compute(
    alpha: wp.array3d(dtype=wp.float32), theta: wp.array3d(dtype=wp.float32),
    aperture: wp.array3d(dtype=wp.float32),
    open_normals: wp.array3d(dtype=wp.vec3), open_offsets: wp.array3d(dtype=wp.float32),
    normals: wp.array3d(dtype=wp.vec3), offsets: wp.array3d(dtype=wp.float64),
    u: wp.array3d(dtype=wp.float32), v: wp.array3d(dtype=wp.float32), w: wp.array3d(dtype=wp.float32),
    flux: wp.array3d(dtype=wp.float32), maximum: wp.array(dtype=wp.float32),
    invalid: wp.array(dtype=wp.int32), detail: wp.array(dtype=wp.float64),
    inverse_spacing_dt: wp.vec3d, shape: wp.vec3i, periodic: wp.vec3i, axis: int, ng: int,
):
    i, j, k = wp.tid()
    face = wp.vec3i(i, j, k)
    F = face+wp.vec3i(ng)
    area = wp.float64(aperture[F[0], F[1], F[2]])
    if area <= wp.float64(0.0):
        return
    # The current closed/nonperiodic normal-boundary transport contract.
    if periodic[axis] == 0 and (face[axis] == 0 or face[axis] == shape[axis]):
        return
    courant = wp.cw_mul(wp.vec3d(_at_face(u, F, axis, 0, shape, periodic, ng),
        _at_face(v, F, axis, 1, shape, periodic, ng),
        _at_face(w, F, axis, 2, shape, periodic, ng)), inverse_spacing_dt)
    if courant[axis] == wp.float64(0.0):
        return
    a, b = (axis+1)%3, (axis+2)%3
    count_a, count_b = int(1), int(1)
    shift_a, shift_b = int(0), int(0)
    if courant[a] != wp.float64(0.0):
        count_a = 2
        shift_a = 1
        if courant[a] > wp.float64(0.0):
            shift_a = -1
    if courant[b] != wp.float64(0.0):
        count_b = 2
        shift_b = 1
        if courant[b] > wp.float64(0.0):
            shift_b = -1
    support = _water_support(alpha, theta, face, courant, shape, periodic, axis, ng)
    if support[0] == 0:
        return
    size = wp.max(wp.abs(courant[0]), wp.max(wp.abs(courant[1]), wp.abs(courant[2])))
    wp.atomic_max(maximum, 0, wp.float32(size))
    if not wp.isfinite(size) or size > wp.float64(1.000001):
        _record(invalid, detail, 1, face, courant[axis], size, wp.float64(0.0))
        return
    if support[1] == 1:
        flux[F[0], F[1], F[2]] = wp.float32(area*courant[axis])
        return
    low = face
    high = face
    low[axis] -= 1
    if low[axis] < 0:
        low[axis] += shape[axis]
    if high[axis] >= shape[axis]:
        high[axis] -= shape[axis]
    L, H = low+wp.vec3i(ng), high+wp.vec3i(ng)
    ln = wp.vec3d(open_normals[L[0], L[1], L[2]])
    hn = wp.vec3d(open_normals[H[0], H[1], H[2]])
    ld = wp.float64(open_offsets[L[0], L[1], L[2]])-ln[axis]
    hd = wp.float64(open_offsets[H[0], H[1], H[2]])
    ln[axis] = wp.float64(0.0)
    hn[axis] = wp.float64(0.0)
    face_axis = wp.float64(0.0)
    if courant[axis] > wp.float64(0.0):
        face_axis = wp.float64(1.0)
    open_amount, water_amount = wp.float64(0.0), wp.float64(0.0)
    for sa in range(count_a):
        for sb in range(count_b):
            delta = wp.vec3i(0)
            delta[a] = sa*shift_a
            delta[b] = sb*shift_b
            cell = _donor(face, delta, courant, axis, shape, periodic)
            if cell[0] >= 0:
                D = cell+wp.vec3i(ng)
                if theta[D[0], D[1], D[2]] > wp.float32(0.0):
                    planes = _TracePlanes()
                    ds = _TraceOffsets()
                    count = int(0)
                    for index in range(2):
                        q, side = a, sa
                        if index == 1:
                            q, side = b, sb
                        if courant[q] != wp.float64(0.0):
                            n = wp.vec3d(wp.float64(0.0))
                            n[axis] = courant[q]
                            n[q] = wp.float64(-1.0)
                            d = wp.float64(0.0)
                            if courant[q] < wp.float64(0.0):
                                n, d = -n, wp.float64(1.0)
                            if side == 1:
                                n, d = -n, -d
                            planes[count] = n
                            ds[count] = d
                            count += 1
                    sn = wp.vec3d(open_normals[D[0], D[1], D[2]])
                    if wp.dot(sn, sn) > wp.float64(0.0):
                        transformed = sn
                        transformed[axis] = -wp.dot(sn, courant)
                        planes[count] = transformed
                        ds[count] = (wp.float64(open_offsets[D[0], D[1], D[2]])-sn[axis]*face_axis
                            +sn[a]*wp.float64(delta[a])+sn[b]*wp.float64(delta[b]))
                        count += 1
                    if wp.dot(ln, ln) > wp.float64(0.0):
                        planes[count] = ln
                        ds[count] = ld
                        count += 1
                    if wp.dot(hn, hn) > wp.float64(0.0):
                        planes[count] = hn
                        ds[count] = hd
                        count += 1
                    opened = _volume(planes, ds, count)
                    open_amount += opened
                    fraction = wp.float64(alpha[D[0], D[1], D[2]])
                    n = wp.vec3d(normals[D[0], D[1], D[2]])
                    if wp.dot(n, n) == wp.float64(0.0):
                        # Match retained sub-tolerance/unoriented fractions to
                        # their existing uniform saturation transport contract.
                        water_amount += fraction*opened
                    elif fraction > wp.float64(0.0) and opened > wp.float64(0.0):
                        transformed = n
                        transformed[axis] = -wp.dot(n, courant)
                        planes[count] = transformed
                        ds[count] = (wp.float64(offsets[D[0], D[1], D[2]])-n[axis]*face_axis
                            +n[a]*wp.float64(delta[a])+n[b]*wp.float64(delta[b]))
                        water_amount += _volume(planes, ds, count+1)
    if open_amount <= wp.float64(0.0):
        _record(invalid, detail, 2, face, courant[axis], open_amount, water_amount)
        return
    saturation = water_amount/open_amount
    if saturation < wp.float64(-5e-6) or saturation > wp.float64(1.000005):
        _record(invalid, detail, 3, face, courant[axis], open_amount, water_amount)
        return
    # Leave geometric roundoff visible to the shared, accounted capacity update.
    flux[F[0], F[1], F[2]] = wp.float32(area*courant[axis]*saturation)


def compute_swept_prism_fluxes(state, reconstruction, dt, transport_velocity):
    """Return the three shared face fluxes consumed by VOF and momentum."""
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite")
    if tuple(reconstruction.normals.shape) != state.grid.cell_shape:
        raise ValueError("PLIC normal field shape does not match state")
    if tuple(reconstruction.offsets.shape) != state.grid.cell_shape:
        raise ValueError("PLIC offset field shape does not match state")
    for name, field in zip("uvw", transport_velocity, strict=True):
        if tuple(field.shape) != state.grid.padded_shape(name):
            raise ValueError(f"{name} transport velocity has an incompatible padded shape")
        if field.dtype != wp.float32:
            raise TypeError(f"{name} transport velocity must use float32")
    g = state.cfg.grid
    bc = state.cfg.bc
    periodic = wp.vec3i(*(int(getattr(bc, q+"lo") == BC.PERIODIC) for q in "xyz"))
    offsets = wp.empty(state.grid.cell_shape, dtype=wp.float64, device=state.device)
    wp.launch(_uncut_offsets, dim=g.shape, inputs=[state.alpha, state.theta,
        reconstruction.normals, reconstruction.offsets, offsets, g.ng], device=state.device)
    results = []
    for axis, name in enumerate("uvw"):
        flux = wp.zeros(state.grid.padded_shape(name), dtype=wp.float32, device=state.device)
        maximum = wp.zeros(1, dtype=wp.float32, device=state.device)
        invalid = wp.zeros(2, dtype=wp.int32, device=state.device)
        detail = wp.zeros(6, dtype=wp.float64, device=state.device)
        wp.launch(_compute, dim=state.grid.interior_shape(name), inputs=[
            state.alpha, state.theta, getattr(state, name+"_aperture"),
            state.open_normals, state.open_offsets, reconstruction.normals, offsets,
            *transport_velocity, flux, maximum, invalid, detail,
            wp.vec3d(dt/g.dx, dt/g.dy, dt/g.dz), wp.vec3i(*g.shape), periodic, axis, g.ng],
            device=state.device)
        maximum_courant = float(maximum.numpy()[0])
        error = int(invalid.numpy()[0])
        if error:
            reason = {1: "non-finite or super-unit traced Courant number",
                      2: "open face has no traced donor open volume",
                      3: "traced water saturation exceeds geometric roundoff"}[error]
            fi, fj, fk, normal_courant, open_amount, water_amount = detail.numpy()
            raise RuntimeError(
                f"{reason} in {'xyz'[axis]} flux at face ({int(fi)}, {int(fj)}, {int(fk)}): "
                f"normal Courant {normal_courant:.6g}, open {open_amount:.9g}, "
                f"water {water_amount:.9g} (max Courant {maximum_courant:.6g})")
        results.append(DirectionalFluxResult("xyz"[axis], flux, maximum_courant))
    return tuple(results)


def swept_prism_transport_rate(state, transport_velocity):
    """Sum of component maxima over the actual face-tracing support, in 1/s.

    An initially dry normal donor can receive liquid along a transverse trace.
    Its extended velocity must participate in the time-step bound as well.
    """
    g = state.cfg.grid
    periodic = wp.vec3i(*(int(getattr(state.cfg.bc, q+"lo") == BC.PERIODIC) for q in "xyz"))
    rates = wp.zeros(3, dtype=wp.float64, device=state.device)
    for axis, name in enumerate("uvw"):
        wp.launch(_trace_rate, dim=state.grid.interior_shape(name), inputs=[
            state.alpha, state.theta, getattr(state, name+"_aperture"), *transport_velocity,
            rates, wp.vec3d(1/g.dx, 1/g.dy, 1/g.dz), wp.vec3i(*g.shape), periodic, axis, g.ng],
            device=state.device)
    return float(rates.numpy().sum())
