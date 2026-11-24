"""Mimetic pressure/body-force operator on the retained liquid polyhedra.

The reference space keeps normal fluxes on liquid and free-surface facets.
The MAC space restricts those traces to Cartesian face velocities before the
solve. Solver3D selects it with pressure_discretization="liquid_mimetic",
paired with conservative unsplit transport and its kinetic energy metric.

For outward integrated flux q, Dq=sum(q)/V. With face areas A, normals n,
and centroid displacements r, N=A*n and R=r obey R.T*N=V*I. The symmetric
positive mass matrix satisfies M*N=R. Consequently the adjoint gradient
reproduces every affine pressure field, including vertical hydrostatics.
Gravity enters as the physical vector in dt*R*g; solid fluxes are constrained
to zero. Hybrid face pressures impose continuity between neighbouring cells.

Reference: Brezzi, Lipnikov, Simoncini (2005), equations (3.7)-(3.10),
https://mate.unipv.it/brezzi/papers/strike1-9.pdf . Facet geometry retains host
quadrature precision through assembly; local factors, the assembled solve and
force/flux updates use float64 (CUDA for the solve/updates). Carried state
remains float32 and post-transfer divergence is measured on that stored state.
"""

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import warp as wp
from warp.optim.linear import cg, preconditioner
from warp.sparse import bsr_from_triplets, bsr_mv

from shoreloop.config import BC
from shoreloop.physics3d.boundary import fill_boundaries
from shoreloop.physics3d.interface import estimate_interface_normals
from shoreloop.physics3d.resolved import resolved_liquid_mask
from shoreloop.physics3d.liquid_assembly import (
    KIND_FREE, KIND_SHARED, KIND_SOLID, MAX_FACETS, MAX_MAC, _assign_keys, _divergence_per_cell, _divergence_statistics,
    _dof_codes, _flat_keys, _flux_response_triplets, _gauge_diagonal, _ground_cells, _ground_faces,
    _kinetic_energy, _mark_codes, _matrix_triplets, _pressure_response_triplets, _read_mac,
    _scatter_mac, _write_mac_component, _write_pressure, assemble_local_systems, exclusive_scan,
)
from shoreloop.physics3d.plic import reconstruct_plic
from shoreloop.physics3d.state import State3D
from shoreloop.physics3d.solids import enforce_voxel_solids


def _clip(points, normal, offset):
    """Clip without mutating input; preserve identity when no point is removed."""
    if len(points) == 0:
        return points
    if not (normal[0] or normal[1] or normal[2]):
        return points if offset >= 0 else np.empty((0, 3))
    result = []
    unchanged = True
    previous = points[-1]
    before = normal @ previous - offset
    for point in points:
        value = normal @ point - offset
        if (value <= 0) != (before <= 0):
            # Symmetric interpolation avoids subtracting a nearly unit edge
            # parameter from a box corner when the retained film is tiny.
            result.append((before*point-value*previous) / (before-value))
        if value <= 0:
            result.append(point)
        else:
            unchanged = False
        previous, before = point, value
    if unchanged:
        return points
    return np.asarray(result).reshape(-1, 3)


def _cross3(a, b):
    # Individual host 3-vectors, with no axis/broadcast work.
    return np.asarray((a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2],
                       a[0]*b[1]-a[1]*b[0]))


def _outside(points, normal, offset):
    """Positive-area part strictly outside a halfspace, excluding coplanarity."""
    if len(points) == 0 or np.all(points @ normal <= offset):
        return np.empty((0, 3))
    return _clip(points, -normal, -offset)


def _facet_area_tolerance(retained_volume, spacing):
    # Omitted area times any cell-local displacement must be small compared
    # with liquid volume in R.T N = V I. A full-cell area bound cannot ensure
    # that for a film. This equals the old 1e-12*min(face area) in a full cell
    # and tightens with retained volume, while excluding FP64 clipping dust.
    return 1e-12*retained_volume/max(spacing)


def _section(normal, offset, planes):
    """Intersect a plane with the unit box, then the other solid/liquid plane.

    Start on actual box edges. Clipping an oversized tangent square introduces
    independent vertex errors, amplified by the inverse volume of a corner sliver.
    The symmetric interpolation also gives one point for either edge direction.
    """
    distances = _BOX_VERTICES @ normal - offset
    points = []
    for a, b in _BOX_EDGES:
        before, after = distances[a], distances[b]
        if before == 0:
            points.append(_BOX_VERTICES[a])
        if after == 0:
            points.append(_BOX_VERTICES[b])
        if before < 0 < after or after < 0 < before:
            points.append((before*_BOX_VERTICES[b]-after*_BOX_VERTICES[a]) / (before-after))
    if not points:
        return np.empty((0, 3))
    points = np.unique(points, axis=0)
    # Dropping the dominant normal coordinate gives a nonsingular 2-D
    # projection. Angular order is the boundary order of this convex section.
    dominant = int(np.argmax(np.abs(normal)))
    axes = [axis for axis in range(3) if axis != dominant]
    projected = points[:, axes]-points[:, axes].mean(axis=0)
    points = points[np.argsort(np.arctan2(projected[:, 1], projected[:, 0]))]
    for other, d in planes:
        points = _clip(points, other, d)
    return points


def _polygon_moments(points, spacing):
    """Physical area and first moment, retaining quadrature precision."""
    # Opposing broad facets can enclose a film only 1e-6 cells deep. Their
    # first moments cancel in the affine identity, so accumulate geometric
    # quadrature above FP64 precision before storing FP64 operator metrics.
    points = points.astype(np.longdouble) * spacing
    area = np.longdouble(0)
    moment = np.zeros(3, dtype=np.longdouble)
    for j in range(1, len(points)-1):
        a, b, c = points[0], points[j], points[j+1]
        cross = _cross3(b-a, c-a)
        weight = np.sqrt(cross @ cross) / 2
        area += weight
        moment += weight * (a+b+c) / 3
    return area, moment


def _area_moment(points, spacing):
    """Physical area and area times cell-local physical centroid."""
    area, moment = _polygon_moments(points, spacing)
    return float(area), moment.astype(float)


def _area_centroid(points, spacing):
    area, moment = _polygon_moments(points, spacing)
    # Retain area and centroid precision through displacement and assembly.
    # Independent rounding before opposing moments cancel can lose the
    # thin-volume affine identity even with an accurate ratio.
    return area, moment / area if area > 0 else moment


_BOX = tuple((np.eye(3)[axis] * sign, float(sign > 0))
             for axis in range(3) for sign in (-1, 1))
# Retain clipping coordinates at quadrature precision too: a tiny liquid
# film through a small solid aperture can lose its affine identity before
# moment accumulation if intersections are rounded to FP64. The stored planes
# stay unchanged; operator metrics and exported vertices still round below.
# Fixed Cartesian faces retain their original vertex order, and sloping
# sections use these same box edges without an auxiliary tangent square.
_BOX_SECTIONS = tuple(np.asarray(points, dtype=np.longdouble) for points in (
    ((0,0,0), (0,0,1), (0,1,1), (0,1,0)),
    ((1,0,1), (1,0,0), (1,1,0), (1,1,1)),
    ((0,0,1), (0,0,0), (1,0,0), (1,0,1)),
    ((0,1,0), (0,1,1), (1,1,1), (1,1,0)),
    ((0,0,0), (0,1,0), (1,1,0), (1,0,0)),
    ((0,1,1), (0,0,1), (1,0,1), (1,1,1)),
))
_BOX_VERTICES = np.asarray([(x, y, z) for x in (0., 1.)
                            for y in (0., 1.) for z in (0., 1.)], dtype=np.longdouble)
_BOX_EDGES = tuple((i, i | bit) for bit in (1, 2, 4) for i in range(8) if not i & bit)
for _polygon in _BOX_SECTIONS:
    _polygon.flags.writeable = False


@dataclass
class LiquidPressureCell:
    # Facet geometry retains quadrature precision. The kinetic matrix and
    # assembled GPU operators use FP64; carried fields remain FP32.
    index: tuple
    volume: float
    centroid: np.ndarray
    areas: np.ndarray
    normals: np.ndarray
    displacements: np.ndarray
    kinds: tuple  # shared, free, or solid
    keys: tuple
    mass: np.ndarray  # integrated-flux kinetic inner product, including walls
    template: bool = False  # fixed box geometry shared by plain full cells


def _neighbour(values, axis, sign, periodic, fill):
    """Value of the neighbour cell along +/-axis for every cell."""
    if periodic:
        return np.roll(values, -sign, axis=axis)
    result = np.full_like(values, fill)
    source = [slice(None)] * 3
    destination = [slice(None)] * 3
    if sign > 0:
        source[axis] = slice(1, None)
        destination[axis] = slice(0, -1)
    else:
        source[axis] = slice(0, -1)
        destination[axis] = slice(1, None)
    result[tuple(destination)] = values[tuple(source)]
    return result


_FACE_KINDS = ("shared", "solid", "free")


def _finish_cell(cell, raw, areas, normals, centres, kinds, keys, target,
                 centroid_local, spacing, reference, identity, origin0):
    """Volume, affine identity, and the kinetic inner product of one cell."""
    areas = np.asarray(areas)
    normals = np.asarray(normals)
    centres = np.asarray(centres)
    # The divergence theorem gives the volume. Any reference point gives
    # the same result because the oriented boundary closes.
    volume = float(np.sum(areas * np.einsum("ij,ij->i", centres-reference, normals)) / 3)
    centroid = centroid_local * spacing
    R = centres-centroid
    N = areas[:, None] * normals
    # Same elementwise bound as assert_allclose (including its default
    # rtol), without constructing testing diagnostics for every good cell.
    affine = R.T @ N
    expected = volume*identity
    if not np.all(np.abs(affine-expected) <= 2e-10*volume + 1e-7*np.abs(expected)):
        np.testing.assert_allclose(affine, expected, rtol=1e-7, atol=2e-10*volume,
                                   err_msg=f"liquid cell {cell}")
    # Start with the velocity form and convert to integrated-flux units.
    velocity_R = areas[:, None] * R
    consistent = velocity_R @ velocity_R.T / volume
    normal64 = normals.astype(float)
    null_projector = np.eye(len(areas)) - normal64 @ np.linalg.solve(normal64.T @ normal64, normal64.T)
    mass = ((consistent + np.trace(consistent)/3 * null_projector)
            / (areas[:, None]*areas[None, :])).astype(float)
    origin = origin0 + raw*spacing
    return (LiquidPressureCell(cell, volume, origin+centroid, areas, normals, R,
                               tuple(kinds), tuple(keys), mass),
            abs(volume/target-1))


def _cells(state, plic):
    """Retained liquid cells in C order of the wet mask.

    A full cell without its own interface or solid plane whose six faces are
    whole (every neighbour is a wall, blocked, dry without a solid plane, or
    wet without any plane) has the fixed box geometry: one template supplies
    its facets and kinetic matrix and only its face kinds and keys are set
    per cell. Every other wet cell keeps the general clipped construction at
    host quadrature precision. Returns the cells, the largest relative volume
    error and the liquid centroids; template cells carry ``template=True``.
    """
    g = state.cfg.grid
    spacing = np.asarray((g.dx, g.dy, g.dz))
    # Unclipped Cartesian polygons have fixed physical metrics on this grid.
    # Compute them with the same triangulation as clipped facets, once per
    # assembly. Identity from _clip proves when these metrics still apply.
    box_metrics = tuple(_area_centroid(polygon, spacing) for polygon in _BOX_SECTIONS)
    box_normals = np.asarray([n / spacing for n, _ in _BOX], dtype=np.longdouble)
    box_normals /= np.linalg.norm(box_normals, axis=1)[:, None]
    cell_volume = np.prod(spacing)
    reference = np.full(3, .5) * spacing
    identity = np.eye(3)
    origin0 = np.asarray((g.x0, g.y0, g.z0))
    alpha = state.get_interior("alpha")
    theta = state.get_geometry_interior("theta")
    slices = state.grid.interior_slices("cell")
    sn = state.open_normals.numpy()[slices].astype(float)
    sd = state.open_offsets.numpy()[slices].astype(float)
    wn = plic.normals.numpy()[slices].astype(float)
    wd = plic.offsets.numpy()[slices].astype(float)
    liquid_centroids = plic.centroids.numpy()[slices].astype(float)
    wet = resolved_liquid_mask(alpha, theta, state.cfg.liquid_resolved_fraction)
    periodic = tuple(getattr(state.cfg.bc, a+"lo") == BC.PERIODIC for a in "xyz")
    shape = g.shape

    # Classify template cells and their six face kinds without per-cell work.
    plane_free = (np.einsum("...i,...i", sn, sn) == 0) & (np.einsum("...i,...i", wn, wn) == 0)
    plain = wet & plane_free
    whole = plain.copy()
    face_kind = np.zeros(shape + (6,), dtype=np.int8)
    for face in range(6):
        axis, sign = face // 2, 2*(face % 2)-1
        neighbour_wet = _neighbour(wet, axis, sign, periodic[axis], False)
        neighbour_blocked = _neighbour(theta == 0, axis, sign, periodic[axis], False)
        neighbour_plane_free = _neighbour(plane_free, axis, sign, periodic[axis], False)
        kind = np.where(neighbour_blocked, 1, np.where(neighbour_wet, 0, 2)).astype(np.int8)
        face_whole = neighbour_blocked | neighbour_plane_free
        if not periodic[axis]:
            edge = [slice(None)] * 3
            edge[axis] = -1 if sign > 0 else 0
            boundary = getattr(state.cfg.bc, "xyz"[axis] + ("lo" if sign < 0 else "hi"))
            kind[tuple(edge)] = 1 if boundary == BC.WALL else 2
            face_whole[tuple(edge)] = True
        face_kind[..., face] = kind
        whole &= face_whole
    template_mask_grid = plain & whole
    if template_mask_grid.any():
        template, template_error = _finish_cell(
            None, np.zeros(3, dtype=int), [m[0] for m in box_metrics],
            list(box_normals), [m[1] for m in box_metrics], ("shared",)*6, (None,)*6,
            cell_volume, np.full(3, .5), spacing, reference, identity, origin0,
        )
    result = []
    maximum_volume_error = 0.0
    for raw in np.argwhere(wet):
        cell = tuple(raw)
        if template_mask_grid[cell]:
            kinds = []
            keys = []
            for face in range(6):
                code = int(face_kind[cell + (face,)])
                kinds.append(_FACE_KINDS[code])
                if code == 0:
                    axis, sign = face // 2, 2*(face % 2)-1
                    face_index = raw.copy()
                    face_index[axis] += int(sign > 0)
                    if periodic[axis]:
                        face_index[axis] %= shape[axis]
                    keys.append((axis, *face_index))
                else:
                    keys.append(None)
            result.append(LiquidPressureCell(
                cell, template.volume, origin0 + raw*spacing + np.full(3, .5)*spacing,
                template.areas, template.normals, template.displacements,
                tuple(kinds), tuple(keys), template.mass, True,
            ))
            maximum_volume_error = max(maximum_volume_error, template_error)
            continue
        target = float(alpha[cell] * theta[cell]) * cell_volume
        area_tolerance = _facet_area_tolerance(target, spacing)
        planes = list(_BOX)
        for n, d in ((sn[cell], sd[cell]), (wn[cell], wd[cell])):
            if n @ n > 0:
                planes.append((n, d))
        areas, centres, normals, kinds, keys = [], [], [], [], []

        def add(area, centre, normal, kind, key=None):
            # Integrate actual polygon pieces below rather than subtracting
            # almost equal moments. Keep facets needed to close the boundary
            # relative to this cell's retained volume, excluding roundoff dust.
            if area > area_tolerance:
                areas.append(area)
                centres.append(centre)
                normals.append(normal)
                kinds.append(kind)
                keys.append(key)

        for face, (n, d) in enumerate(planes):
            if face >= 6 and any(
                np.linalg.norm(_cross3(n, other)) < 1e-14
                and n @ other > 0
                and abs(d * (n @ other) - other_d * (n @ n)) < 1e-14
                for other, other_d in planes[:face]
            ):
                continue
            if face < 6:
                polygon = _BOX_SECTIONS[face]
                for other, other_d in planes[6:]:
                    polygon = _clip(polygon, other, other_d)
                area, centre = (box_metrics[face] if polygon is _BOX_SECTIONS[face]
                                else _area_centroid(polygon, spacing))
                normal = box_normals[face]
            else:
                polygon = _section(n, d, [p for f, p in enumerate(planes[6:], 6) if f != face])
                area, centre = _area_centroid(polygon, spacing)
                # Normalize the physical covector at quadrature precision too.
                # One rounding unit in a broad facet's normal can exceed a
                # tiny cut film's affine bound after opposing moments cancel.
                normal = n.astype(np.longdouble) / spacing
                normal /= np.linalg.norm(normal)
            if face >= 6:
                kind = "solid" if sn[cell] @ sn[cell] > 0 and face == 6 else "free"
                add(area, centre, normal, kind)
                continue
            axis, sign = face // 2, 2*(face % 2)-1
            neighbour = raw.copy()
            neighbour[axis] += sign
            inside = 0 <= neighbour[axis] < g.shape[axis]
            if periodic[axis]:
                neighbour[axis] %= g.shape[axis]
                inside = True
            if not inside:
                boundary = getattr(state.cfg.bc, "xyz"[axis] + ("lo" if sign < 0 else "hi"))
                add(area, centre, normal, "solid" if boundary == BC.WALL else "free")
                continue
            neighbour = tuple(neighbour)
            if theta[neighbour] == 0:
                add(area, centre, normal, "solid")
                continue
            # Express the neighbour's plane in this cell's coordinates.
            opened = _clip(polygon, sn[neighbour], sd[neighbour]+sign*sn[neighbour][axis])
            open_area, open_centre = ((area, centre) if opened is polygon
                                      else _area_centroid(opened, spacing))
            blocked = _outside(polygon, sn[neighbour], sd[neighbour]+sign*sn[neighbour][axis])
            add(*_area_centroid(blocked, spacing), normal, "solid")
            if not wet[neighbour]:
                add(open_area, open_centre, normal, "free")
                continue
            common = _clip(opened, wn[neighbour], wd[neighbour]+sign*wn[neighbour][axis])
            common_area, common_centre = ((open_area, open_centre) if common is opened
                                          else _area_centroid(common, spacing))
            exposed = _outside(opened, wn[neighbour], wd[neighbour]+sign*wn[neighbour][axis])
            add(*_area_centroid(exposed, spacing), normal, "free")
            face_index = raw.copy()
            face_index[axis] += int(sign > 0)
            if periodic[axis]:
                face_index[axis] %= g.shape[axis]
            add(common_area, common_centre, normal, "shared", (axis, *face_index))
        finished, error = _finish_cell(cell, raw, areas, normals, centres, kinds, keys, target,
                                       liquid_centroids[cell], spacing, reference, identity, origin0)
        result.append(finished)
        maximum_volume_error = max(maximum_volume_error, error)
    return result, maximum_volume_error, liquid_centroids


@wp.kernel
def _fix_pressure_gauges(rhs: wp.array(dtype=wp.float64), gauges: wp.array(dtype=int)):
    rhs[gauges[wp.tid()]] = wp.float64(0.0)


@wp.kernel
def _source_and_rhs(
    widths: wp.array(dtype=int), starts: wp.array(dtype=int),
    projector: wp.array3d(dtype=wp.float64), gravity: wp.array2d(dtype=wp.vec3d),
    pressure_initial: wp.array2d(dtype=wp.float64), pressure_gravity: wp.array(dtype=wp.vec3d),
    initial: wp.array(dtype=wp.float64),
    keys: wp.array2d(dtype=int), source: wp.array(dtype=wp.float64),
    rhs: wp.array(dtype=wp.float64), pressure: wp.array(dtype=wp.float64),
    acceleration_dt: wp.vec3d, rho_over_dt: wp.float64,
):
    cell = wp.tid()
    start = starts[cell]
    size = widths[cell]
    pc = wp.dot(pressure_gravity[cell], acceleration_dt)
    for i in range(size):
        value = wp.dot(gravity[cell, i], acceleration_dt)
        for j in range(size):
            value += projector[cell, i, j] * initial[start+j]
        source[start+i] = value
        key = keys[cell, i]
        if key >= 0:
            wp.atomic_add(rhs, key, value)
        pc += pressure_initial[cell, i]*initial[start+i]
    pressure[cell] = -rho_over_dt*pc


@wp.kernel
def _constraint_residual(shared: wp.array(dtype=int), flux: wp.array(dtype=wp.float64),
                         rhs: wp.array(dtype=wp.float64)):
    i = wp.tid()
    key = shared[i]
    if key >= 0:
        wp.atomic_add(rhs, key, flux[i])


def _local_projection(mass, divergence):
    H = np.linalg.inv(mass)
    h = H @ divergence
    denominator = divergence @ h if len(h) else 1.0
    weights = h/denominator
    response = H - np.outer(h, h)/denominator
    # E M = I - (H d) d.T / (d.T H d). Form this identity before rounding:
    # applying a badly conditioned M and then E in FP32 can erase rigid motion.
    projector = np.eye(len(mass)) - np.outer(weights, divergence)
    return response, weights, projector, divergence/denominator


def _pressure_coordinates(matrix):
    """Orthogonal pair coordinates retain small modes before FP32 storage.

    Greedily match disjoint pairs with normalized coupling above .9. This is
    a conditioning choice, not a dropped coupling or physical cutoff: every
    matrix entry survives the congruence, with at most four times the nonzeros.
    """
    entries = matrix.tocoo()
    diagonal = matrix.diagonal()
    strength = np.abs(entries.data)/np.sqrt(diagonal[entries.row]*diagonal[entries.col])
    candidates = np.flatnonzero((entries.row < entries.col) & (strength > .9))
    candidates = candidates[np.argsort(-strength[candidates], kind="stable")]
    used = np.zeros(matrix.shape[0], dtype=bool)
    diagonal = np.ones(matrix.shape[0])
    rows, columns, values = [], [], []
    for entry in candidates:
        i, j = int(entries.row[entry]), int(entries.col[entry])
        if used[i] or used[j]:
            continue
        used[i] = used[j] = True
        _, basis = np.linalg.eigh(matrix[[i, j], :][:, [i, j]].toarray())
        diagonal[i], diagonal[j] = basis[0, 0], basis[1, 1]
        rows.extend((i, j)); columns.extend((j, i))
        values.extend((basis[0, 1], basis[1, 0]))
    identity = np.arange(matrix.shape[0])
    return coo_matrix((np.r_[diagonal, values],
                       (np.r_[identity, rows], np.r_[identity, columns])), shape=matrix.shape).tocsr()


def _mac_basis(cell, active, kept, spacing):
    """RT0 restriction of the active facets to the kept MAC faces.

    The same arithmetic as the sequential construction: each component is
    affine in its own coordinate, so a facet weights the low face by
    ``1-xi`` and the high face by ``xi`` of its centroid.
    """
    origin = cell.centroid - np.full(3, .5)*spacing
    xi = (cell.displacements[active]+cell.centroid-origin) / spacing
    cartesian = np.zeros((len(active), 6))
    for face in kept:
        axis, sign = face//2, 2*(face % 2)-1
        weight = xi[:, axis] if sign > 0 else 1-xi[:, axis]
        cartesian[:, face] = sign*cell.areas[active]*cell.normals[active, axis]*weight
    return cartesian[:, list(kept)]


@dataclass(frozen=True)
class LiquidPressureResult:
    """Normal flux coordinates (facets) or outward velocities (MAC), and p."""
    flux: wp.array
    pressure: wp.array
    iterations: int
    residual: float
    tolerance: float
    rhs_norm: float


class LiquidPressureOperator:
    """Construct a fixed-geometry CUDA pressure operator from retained water.

    In the reference facet space, shared fluxes occur once per adjacent cell
    with opposite outward signs. In the MAC space, the basis restriction is
    applied before constructing the mass/divergence/force operators. Each
    Cartesian face then has one constrained velocity, with two local views.
    """

    def __init__(self, state: State3D, *, velocity_space="facets", extension_inertia=None):
        if extension_inertia is None:
            extension_inertia = state.cfg.liquid_extension_inertia
        if state.cfg.physics.surface_tension != 0:
            raise ValueError("liquid_mimetic currently requires zero surface tension")
        if velocity_space not in ("facets", "mac"):
            raise ValueError("velocity_space must be facets or mac")
        if not np.isfinite(extension_inertia) or extension_inertia < 0:
            raise ValueError("extension_inertia must be finite and nonnegative")
        self.velocity_space = velocity_space
        self.grid = state.cfg.grid
        self.device = state.device
        self.rho = state.cfg.physics.rho
        self.reconstruction = reconstruct_plic(state, estimate_interface_normals(state).normals)
        g = state.cfg.grid
        spacing = np.asarray((g.dx, g.dy, g.dz))
        mac = velocity_space == "mac"
        periodic = tuple(getattr(state.cfg.bc, a+"lo") == BC.PERIODIC for a in "xyz")
        self._periodic = periodic
        key_shape = np.asarray(g.shape) + 1
        self._cells_host = None
        self._active_host = None
        self._local = None
        self._host_views = {}
        device = self.device

        def key_code(axis, position):
            position = np.asarray(position)
            return ((axis * key_shape[0] + position[..., 0]) * key_shape[1]
                    + position[..., 1]) * key_shape[2] + position[..., 2]

        def upload(data, dtype=wp.float64):
            return wp.array(np.ascontiguousarray(data), dtype=dtype, device=device)

        if mac:
            # Device assembly (AW-1111): facet geometry, MAC local systems and the
            # hybrid sparse system are built by kernels; nothing large is downloaded.
            local = assemble_local_systems(state, self.reconstruction, extension_inertia=extension_inertia,
                                           wet_threshold=state.cfg.liquid_resolved_fraction)
            self._local = local
            count = local.count
            nmax = MAX_MAC
            widths_d, mass_d, divergence_d = local.widths, local.mass, local.divergence
            response_d, weights_d, projector_d, initial_d = local.response, local.weights, local.projector, local.initial
            gravity_d, pressure_gravity_d, volume_d, has_free_d = local.gravity, local.pressure_gravity, local.volume, local.has_free
            self._cells_d = local.cells
            self._cell_index = local.cell_index
            self._dof_axis_d, self._dof_position_d, self._dof_sign_d = local.dof_axis, local.dof_position, local.dof_sign
            codes_d = wp.zeros((count, nmax), dtype=wp.int32, device=device)
            if count:
                wp.launch(_dof_codes, dim=count, device=device, inputs=[
                    widths_d, local.dof_axis, local.dof_position, local.dof_shared,
                    wp.vec3i(*g.shape), wp.vec3i(*[int(v) for v in periodic]), codes_d])
            maxima = local.maxima.numpy()
            self.max_volume_error = float(maxima[0])
            self.max_affine_residual = float(maxima[1])
            self.centroids = self.reconstruction.centroids.numpy()[state.grid.interior_slices("cell")].astype(float)
            self.cell_count = count
            self.free_facet_count = int(local.free_facets.numpy()[0])
            self.residue_cells, self.residue_volume = local.residue_cells, local.residue_volume
            self.bases = None
        else:
            self._cells_host, self.max_volume_error, self.centroids = _cells(state, self.reconstruction)
            self._active_host = [np.flatnonzero(np.asarray(c.kinds) != "solid") for c in self.cells]
            apertures = [state.get_geometry_interior(q+"_aperture") for q in ("u", "v", "w")]
            wet_cells = {c.index for c in self.cells}
            count = len(self.cells)
            widths = np.asarray([len(active) for active in self.active], dtype=np.int64)
            nmax = 6 if mac else int(widths.max(initial=0))
            # Every cell's local system, padded to the widest cell. Template cells
            # share one geometry, so their systems are formed once per pattern of
            # kept faces; the general cells keep the per-cell construction.
            widths_host = np.zeros(count, dtype=np.int64)
            mass_host = np.zeros((count, nmax, nmax))
            divergence_host = np.zeros((count, nmax))
            response = np.zeros((count, nmax, nmax))
            weights_all = np.zeros((count, nmax))
            projectors_all = np.zeros((count, nmax, nmax))
            initial_all = np.zeros((count, nmax))
            gravity_all = np.zeros((count, nmax, 3))
            pressure_gravity_all = np.zeros((count, 3))
            codes = np.full((count, nmax), -1, dtype=np.int64)
            dofs_host = np.zeros((count, nmax, 5), dtype=np.int64)
            bases = [None] * count
            general_cells = np.arange(count)
            for cell_index in general_cells:
                cell, active = self.cells[cell_index], self.active[cell_index]
                basis = np.eye(len(active))
                local_keys = [cell.keys[face] for face in active]
                dof_faces = []
                if velocity_space == "mac":
                    # Restrict the facet field to the six MAC face velocities.
                    # Each component is affine in its own coordinate (RT0). Wall
                    # facet flux is prescribed zero; its reaction remains in the
                    # variational force balance. Shared MAC traces are constrained
                    # before projection, so no averaging of a richer solution is
                    # used to manufacture a velocity state afterwards.
                    origin = np.asarray((g.x0, g.y0, g.z0)) + np.asarray(cell.index)*spacing
                    xi = (cell.displacements[active]+cell.centroid-origin) / spacing
                    cartesian = np.zeros((len(active), 6))
                    local_keys = []
                    columns_kept = []
                    for face in range(6):
                        axis, sign = face//2, 2*(face % 2)-1
                        position = np.asarray(cell.index)
                        position[axis] += int(sign > 0)
                        boundary = int(position[axis]) in (0, g.shape[axis])
                        periodic = getattr(state.cfg.bc, "xyz"[axis]+"lo") == BC.PERIODIC
                        bc = getattr(state.cfg.bc, "xyz"[axis]+("lo" if sign < 0 else "hi"))
                        if apertures[axis][tuple(position)] == 0 or (boundary and not periodic and bc == BC.WALL):
                            continue
                        columns_kept.append(face)
                        weight = xi[:, axis] if sign > 0 else 1-xi[:, axis]
                        cartesian[:, face] = sign*cell.areas[active]*cell.normals[active, axis]*weight
                        dof_faces.append((axis, *position, sign))
                        neighbour = np.asarray(cell.index)
                        neighbour[axis] += sign
                        canonical = position.copy()
                        if periodic:
                            neighbour[axis] %= g.shape[axis]
                            canonical[axis] %= g.shape[axis]
                        local_keys.append((axis, *canonical) if tuple(neighbour) in wet_cells else None)
                    basis = cartesian[:, columns_kept]
                M = basis.T @ cell.mass[np.ix_(active, active)] @ basis
                if velocity_space == "mac" and extension_inertia > 0 and len(columns_kept):
                    # Unresolved variation across an almost empty cell otherwise
                    # leaves exterior MAC velocities with vanishing inertia. This
                    # positive stabilization penalises velocity differences and
                    # solid-normal traces: it annihilates admissible uniform
                    # vectors, preserving free fall. It vanishes in full cells
                    # and contributes O(h^2) kinetic work
                    # on smooth fields. Its coefficient is independently zeroable.
                    # Form differences on all six MAC faces, with impermeable
                    # face velocities fixed at zero, then restrict to free DOFs.
                    # In outward coordinates each opposite-face block is
                    # [[1, 1], [1, 1]]/2. Dropping a wall face before forming this
                    # projector would mistake its opposite velocity for a rigid
                    # translation, leaving a zero-energy mode at a wet corner.
                    axes = np.asarray(columns_kept)//2
                    Q = .5*(axes[:, None] == axes[None, :])
                    wall = np.flatnonzero(np.asarray(cell.kinds) == "solid")
                    if len(wall):
                        xi_wall = (cell.displacements[wall]+cell.centroid-origin) / spacing
                        signs = 2*(np.asarray(columns_kept) % 2)-1
                        weights_wall = np.where(signs > 0, xi_wall[:, axes], 1-xi_wall[:, axes])
                        traces = signs*cell.normals[wall][:, axes]*weights_wall
                        Q += np.asarray(traces.T @ (cell.areas[wall, None]*traces)
                                        / np.sum(cell.areas), dtype=float)
                    M += extension_inertia * cell.volume*.5*max(0, 1-cell.volume/np.prod(spacing))*Q
                divergence = basis.sum(axis=0)
                E, weights, projector, initial_weights = _local_projection(M, divergence)
                displacement = (basis.T @ cell.displacements[active]).astype(float)
                n = len(divergence)
                widths_host[cell_index] = n
                mass_host[cell_index, :n, :n] = M
                divergence_host[cell_index, :n] = divergence
                response[cell_index, :n, :n] = E
                weights_all[cell_index, :n] = weights
                projectors_all[cell_index, :n, :n] = projector
                initial_all[cell_index, :n] = initial_weights
                gravity_all[cell_index, :n] = E @ displacement
                pressure_gravity_all[cell_index] = weights @ displacement
                if mac:
                    for q, face in enumerate(dof_faces):
                        dofs_host[cell_index, q] = face
                codes[cell_index, :n] = [-1 if key is None else key_code(key[0], key[1:]) for key in local_keys]
                bases[cell_index] = basis
            self.bases = bases
            cell_index_array = np.asarray([c.index for c in self.cells], dtype=np.int64).reshape(-1, 3)
            volumes_array = np.asarray([c.volume for c in self.cells])
            has_free = np.asarray(["free" in cell.kinds for cell in self.cells], dtype=bool)
            self.max_affine_residual = 0.0
            self.residue_cells, self.residue_volume = 0, 0.0
            self.cell_count = count
            self.free_facet_count = sum(c.kinds.count("free") for c in self.cells)
            widths_d = upload(widths_host, wp.int32)
            mass_d, divergence_d = upload(mass_host), upload(divergence_host)
            response_d, weights_d = upload(response), upload(weights_all)
            projector_d, initial_d = upload(projectors_all), upload(initial_all)
            gravity_d = upload(gravity_all, wp.vec3d)
            pressure_gravity_d = upload(pressure_gravity_all, wp.vec3d)
            volume_d = upload(volumes_array)
            has_free_d = upload(has_free.astype(np.int32), wp.int32)
            codes_d = upload(np.where(codes >= 0, codes, -1).astype(np.int32), wp.int32)
            self._cells_d = upload(cell_index_array.astype(np.int32), wp.vec3i)
            self._cell_index = cell_index_array
            self._host_views.update(_widths=widths_host, _mass=mass_host, _divergence=divergence_host,
                                    _dofs=dofs_host, _volumes=volumes_array)

        # ---- hybrid pressure system on the device --------------------------------
        self.count, self.nmax = count, nmax
        self._widths_d, self._mass_d, self._divergence_d, self._volume_d = widths_d, mass_d, divergence_d, volume_d
        self._response_d, self._weights_d = response_d, weights_d
        self.projectors, self.gravity, self.pressure_initial, self.pressure_gravity = projector_d, gravity_d, initial_d, pressure_gravity_d
        self.starts, self.total_dofs = exclusive_scan(widths_d, device)
        # Hybrid pressure unknowns: one per shared face, numbered by a scan over
        # the marked face codes.
        code_span = int(3 * np.prod(key_shape))
        mark = wp.zeros(code_span, dtype=wp.int32, device=device)
        keys_d = wp.full((count, nmax), -1, dtype=wp.int32, device=device)
        keyed_d = wp.zeros(count, dtype=wp.int32, device=device)
        if count:
            wp.launch(_mark_codes, dim=count, device=device, inputs=[widths_d, codes_d, mark])
        scan = wp.zeros_like(mark)
        wp.utils.array_scan(mark, scan, inclusive=True)
        key_count = int(scan.numpy()[-1]) if code_span else 0
        if count:
            wp.launch(_assign_keys, dim=count, device=device, inputs=[widths_d, codes_d, scan, keys_d, keyed_d])
        self.keys, self.key_count = keys_d, key_count
        self.shared = wp.zeros(self.total_dofs, dtype=wp.int32, device=device)
        if count:
            wp.launch(_flat_keys, dim=count, device=device, inputs=[widths_d, self.starts, keys_d, self.shared])
        # Every closed liquid component has one arbitrary pressure constant. A
        # component is grounded when any of its cells has a free facet; the
        # flag is propagated across shared faces, and one unknown per
        # ungrounded component is pinned (row/column removed, unit diagonal).
        face_grounded = wp.zeros(max(key_count, 1), dtype=wp.int32, device=device)
        cell_grounded = wp.clone(has_free_d)
        changed = wp.zeros(1, dtype=wp.int32, device=device)
        if count and key_count:
            for iteration in range(10 * (g.nx + g.ny + g.nz) + 8):
                if iteration % 8 == 0:
                    changed.zero_()
                wp.launch(_ground_faces, dim=count, device=device, inputs=[widths_d, keys_d, cell_grounded, face_grounded, changed])
                wp.launch(_ground_cells, dim=count, device=device, inputs=[widths_d, keys_d, face_grounded, cell_grounded, changed])
                if iteration % 8 == 7 and int(changed.numpy()[0]) == 0:
                    break
        gauges = self._pocket_gauges(face_grounded, keys_d, widths_d, cell_grounded) if key_count else np.zeros(0, dtype=np.int32)
        self.gauges = upload(gauges, wp.int32)
        gauge_mask = wp.zeros(max(key_count, 1), dtype=wp.int32, device=device)
        if len(gauges):
            gauge_mask_host = np.zeros(max(key_count, 1), dtype=np.int32); gauge_mask_host[gauges] = 1
            gauge_mask = upload(gauge_mask_host, wp.int32)
        # Triplets of the local responses between keyed DOFs, and the flux and
        # pressure responses to the hybrid increments.
        keyed_host = keyed_d.numpy().astype(np.int64) if count else np.zeros(0, dtype=np.int64)
        widths_host_now = self._host_views.get("_widths")
        if widths_host_now is None:
            widths_host_now = widths_d.numpy().astype(np.int64)
        pair_offsets, pair_total = exclusive_scan(upload((keyed_host * keyed_host).astype(np.int32), wp.int32), device)
        flux_offsets, flux_total = exclusive_scan(upload((widths_host_now * keyed_host).astype(np.int32), wp.int32), device)
        keyed_offsets, keyed_total = exclusive_scan(keyed_d, device)
        rows = wp.zeros(pair_total + len(gauges), dtype=wp.int32, device=device)
        cols = wp.zeros_like(rows)
        values = wp.zeros(pair_total + len(gauges), dtype=wp.float64, device=device)
        if count:
            wp.launch(_matrix_triplets, dim=count, device=device, inputs=[widths_d, keys_d, response_d, pair_offsets, gauge_mask, rows, cols, values])
        if len(gauges):
            wp.launch(_gauge_diagonal, dim=len(gauges), device=device, inputs=[self.gauges, pair_total, rows, cols, values])
        self.matrix = bsr_from_triplets(key_count, key_count, rows, cols, values)
        frows = wp.zeros(flux_total, dtype=wp.int32, device=device); fcols = wp.zeros_like(frows)
        fvalues = wp.zeros(flux_total, dtype=wp.float64, device=device)
        if count:
            wp.launch(_flux_response_triplets, dim=count, device=device, inputs=[widths_d, self.starts, keys_d, response_d, flux_offsets, frows, fcols, fvalues])
        self.flux_response = bsr_from_triplets(self.total_dofs, key_count, frows, fcols, fvalues)
        prows = wp.zeros(keyed_total, dtype=wp.int32, device=device); pcols = wp.zeros_like(prows)
        pvalues = wp.zeros(keyed_total, dtype=wp.float64, device=device)
        if count:
            wp.launch(_pressure_response_triplets, dim=count, device=device, inputs=[widths_d, keys_d, weights_d, keyed_offsets, prows, pcols, pvalues])
        self.pressure_response = bsr_from_triplets(count, key_count, prows, pcols, pvalues)

    def _pocket_gauges(self, face_grounded, keys_d, widths_d, cell_grounded):
        """One hybrid unknown per closed liquid pocket (component without a free facet)."""
        grounded = face_grounded.numpy()
        ungrounded = np.flatnonzero(grounded == 0)
        if not len(ungrounded):
            return np.zeros(0, dtype=np.int32)
        keys = keys_d.numpy(); widths = widths_d.numpy()
        cells = np.flatnonzero(cell_grounded.numpy() == 0)
        # Edges between ungrounded cells through their (ungrounded) keyed faces.
        column = np.arange(keys.shape[1])[None, :] < widths[:, None]
        cell_of = np.repeat(np.arange(len(widths)), widths)
        flat_keys = keys[column]
        keep = np.isin(cell_of, cells) & (flat_keys >= 0)
        pairs = coo_matrix((np.ones(keep.sum()), (cell_of[keep], flat_keys[keep])),
                           shape=(len(widths), max(int(flat_keys.max()) + 1, 1)))
        adjacency = pairs @ pairs.T
        components, labels = connected_components(adjacency, directed=False)
        gauges = []
        for component in np.unique(labels[cells]):
            for cell in cells[labels[cells] == component]:
                row = keys[cell, :widths[cell]]
                row = row[row >= 0]
                if len(row):
                    # A pocket enclosed entirely by solids has no unknown to pin.
                    gauges.append(int(row[0]))
                    break
        return np.asarray(sorted(set(gauges)), dtype=np.int32)

    def project(self, dt, *, acceleration=(0, 0, 0), initial_flux=None, tolerance=1e-6,
                absolute_tolerance=1e-14, max_iterations=2000):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        count = self.total_dofs
        if initial_flux is None:
            initial_flux = wp.zeros(count, dtype=wp.float64, device=self.device)
        elif initial_flux.dtype != wp.float64:
            converted = wp.empty(count, dtype=wp.float64, device=self.device)
            wp.utils.array_cast(initial_flux, converted)
            initial_flux = converted
        flux = wp.zeros(count, dtype=wp.float64, device=self.device)
        pressure = wp.zeros(self.cell_count, dtype=wp.float64, device=self.device)
        rhs = wp.zeros(self.matrix.nrow, dtype=wp.float64, device=self.device)
        if self.cell_count:
            wp.launch(_source_and_rhs, dim=self.cell_count, inputs=[
                self._widths_d, self.starts, self.projectors, self.gravity, self.pressure_initial,
                self.pressure_gravity, initial_flux, self.keys, flux, rhs, pressure,
                wp.vec3d(*(dt*np.asarray(acceleration))), self.rho/dt], device=self.device)
        wp.launch(_fix_pressure_gauges, dim=self.gauges.size, inputs=[rhs, self.gauges], device=self.device)
        rhs_norm = float(np.linalg.norm(rhs.numpy().astype(float)))
        target = max(absolute_tolerance, tolerance*rhs_norm)
        iterations, residual = 0, rhs_norm
        if self.matrix.nrow:
            increment = wp.zeros_like(rhs)
            inverse = preconditioner(self.matrix)
            while residual > target and iterations < max_iterations:
                increment.zero_()
                steps, _, _ = cg(self.matrix, rhs, increment, tol=tolerance, atol=target,
                                  maxiter=max_iterations-iterations, M=inverse, check_every=1)
                iterations += steps
                bsr_mv(self.flux_response, increment, flux, alpha=-1.0, beta=1.0)
                bsr_mv(self.pressure_response, increment, pressure, alpha=self.rho/dt, beta=1.0)
                # Use the actual shared-velocity jump, not a recursively
                # updated Krylov residual. Refinement solves the same paired
                # constraints, within the original total iteration budget.
                rhs.zero_()
                wp.launch(_constraint_residual, dim=count, inputs=[self.shared, flux, rhs], device=self.device)
                wp.launch(_fix_pressure_gauges, dim=self.gauges.size, inputs=[rhs, self.gauges], device=self.device)
                residual = float(np.linalg.norm(rhs.numpy().astype(float)))
                if steps == 0:
                    break
        if not np.isfinite(residual) or residual > target:
            raise RuntimeError(f"liquid pressure failed after {iterations} iterations: "
                               f"actual residual {residual} > {target}")
        return LiquidPressureResult(flux, pressure, iterations, residual, target, rhs_norm)

    # ---- host views (tests, tools, the viscosity metric) -------------------------
    def _view(self, name, produce):
        if name not in self._host_views:
            self._host_views[name] = produce()
        return self._host_views[name]

    @property
    def _widths(self):
        return self._view("_widths", lambda: self._widths_d.numpy().astype(np.int64))

    @property
    def _mass(self):
        return self._view("_mass", lambda: self._mass_d.numpy())

    @property
    def _divergence(self):
        return self._view("_divergence", lambda: self._divergence_d.numpy())

    @property
    def _volumes(self):
        return self._view("_volumes", lambda: self._volume_d.numpy())

    @property
    def _dofs(self):
        def produce():
            local = self._local
            return np.concatenate([local.dof_axis.numpy()[..., None], local.dof_position.numpy(),
                                   local.dof_sign.numpy()[..., None]], axis=-1).astype(np.int64)
        return self._view("_dofs", produce)

    @property
    def starts_host(self):
        return self._view("starts_host", lambda: self.starts.numpy().astype(np.int64))

    @property
    def shared_host(self):
        return self._view("shared_host", lambda: self.shared.numpy().astype(np.int64))

    @property
    def matrix_host(self):
        def produce():
            offsets = self.matrix.offsets.numpy(); columns = self.matrix.columns.numpy()
            values = self.matrix.values.numpy().reshape(-1)
            nnz = int(offsets[-1])
            rows = np.repeat(np.arange(self.matrix.nrow), np.diff(offsets))
            matrix = coo_matrix((values[:nnz], (rows, columns[:nnz])), shape=(self.matrix.nrow, self.matrix.ncol)).tocsr()
            matrix.sum_duplicates()
            return matrix
        return self._view("matrix_host", produce)

    @property
    def cells(self):
        """Host cell records; materialised from the device facets on first use."""
        if self._cells_host is None:
            self._cells_host = self._materialise_cells()
        return self._cells_host

    @property
    def active(self):
        if self._active_host is None:
            self._active_host = [np.flatnonzero(np.asarray(c.kinds) != "solid") for c in self.cells]
        return self._active_host

    def _materialise_cells(self):
        local = self._local.host()
        g = self.grid
        spacing = np.asarray((g.dx, g.dy, g.dz))
        origin0 = np.asarray((g.x0, g.y0, g.z0))
        periodic = [self._periodic[axis] for axis in range(3)]
        cells = []
        for c in range(len(local.cell_index)):
            n = int(local.facet_count[c])
            index = tuple(int(v) for v in local.cell_index[c])
            areas = local.facet_area[c, :n].copy()
            normals = local.facet_normal[c, :n].copy()
            centres = local.facet_centre[c, :n]
            R = centres - local.centroid[c]
            kinds, keys = [], []
            for f in range(n):
                kinds.append(_FACE_KINDS[int(local.facet_kind[c, f])])
                face = int(local.facet_face[c, f])
                if local.facet_kind[c, f] == KIND_SHARED and face >= 0:
                    axis, sign = face // 2, 2 * (face % 2) - 1
                    position = list(index)
                    position[axis] += int(sign > 0)
                    if periodic[axis]:
                        position[axis] %= g.shape[axis]
                    keys.append((axis, *position))
                else:
                    keys.append(None)
            volume = float(local.volume[c])
            velocity_R = areas[:, None] * R
            consistent = velocity_R @ velocity_R.T / volume
            null_projector = np.eye(n) - normals @ np.linalg.solve(normals.T @ normals, normals.T)
            mass = (consistent + np.trace(consistent)/3 * null_projector) / (areas[:, None]*areas[None, :])
            cells.append(LiquidPressureCell(
                index, volume, origin0 + np.asarray(index)*spacing + local.centroid[c],
                areas, normals, R, tuple(kinds), tuple(keys), mass,
            ))
        return cells

    def shared_high_areas(self):
        """Shared facet area on each cell's high face per direction, (count, 3)."""
        if self._local is not None:
            local = self._local.host()
            valid = np.arange(MAX_FACETS)[None, :] < local.facet_count[:, None]
            result = np.zeros((len(local.cell_index), 3))
            for axis in range(3):
                mask = valid & (local.facet_kind == KIND_SHARED) & (local.facet_face == 2*axis + 1)
                result[:, axis] = (local.facet_area * mask).sum(axis=1)
            return result
        shape = np.asarray(self.grid.shape)
        result = np.zeros((len(self.cells), 3))
        for c, cell in enumerate(self.cells):
            for key, area, kind in zip(cell.keys, cell.areas, cell.kinds, strict=True):
                if kind == "shared" and key[1+key[0]] == (cell.index[key[0]] + 1) % (shape[key[0]] if self._periodic[key[0]] else shape[key[0]]+1):
                    result[c, key[0]] = area
        return result

    @cached_property
    def mass_blocks(self):
        return [self._mass[c, :n, :n] for c, n in enumerate(self._widths)]

    @cached_property
    def divergence_vectors(self):
        return [self._divergence[c, :n] for c, n in enumerate(self._widths)]

    @cached_property
    def dof_faces(self):
        return [[tuple(int(v) for v in face) for face in self._dofs[c, :n]]
                for c, n in enumerate(self._widths)]

    def read_mac_state(self, state):
        """Read the outward normal MAC velocities of every DOF (float64, device)."""
        if self.velocity_space != "mac" or state.cfg.grid != self.grid:
            raise ValueError("MAC state transfer requires the matching MAC operator")
        values = wp.zeros(self.total_dofs, dtype=wp.float64, device=self.device)
        if self.cell_count:
            wp.launch(_read_mac, dim=self.cell_count, device=self.device, inputs=[
                self._widths_d, self.starts, self._dof_axis_d, self._dof_position_d, self._dof_sign_d,
                state.u, state.v, state.w, self.grid.ng, values])
        return values

    def write_mac_state(self, state, result):
        """Write the constrained MAC solution and cell-centroid pressures.

        The two views of a shared velocity agree to the solve residual. Their
        mean gives one stored value; post-transfer divergence must be measured
        using read_mac_state, rather than claiming the pre-transfer constraint.
        Periodic images accumulate on the canonical low face before halo
        filling copies that value to the high face.
        """
        if self.velocity_space != "mac" or state.cfg.grid != self.grid:
            raise ValueError("MAC state transfer requires the matching MAC operator")
        g = self.grid
        sums = [wp.zeros(state.grid.interior_shape(q), dtype=wp.float64, device=self.device) for q in ("u", "v", "w")]
        counts = [wp.zeros(state.grid.interior_shape(q), dtype=wp.int32, device=self.device) for q in ("u", "v", "w")]
        if self.cell_count:
            wp.launch(_scatter_mac, dim=self.cell_count, device=self.device, inputs=[
                self._widths_d, self.starts, self._dof_axis_d, self._dof_position_d, self._dof_sign_d,
                result.flux, wp.vec3i(*g.shape), wp.vec3i(*[int(v) for v in self._periodic]),
                sums[0], sums[1], sums[2], counts[0], counts[1], counts[2]])
        for q, total, number in zip(("u", "v", "w"), sums, counts, strict=True):
            wp.launch(_write_mac_component, dim=state.grid.interior_shape(q), device=self.device,
                      inputs=[total, number, state.field(q), g.ng])
        state.p.zero_()
        if self.cell_count:
            wp.launch(_write_pressure, dim=self.cell_count, device=self.device,
                      inputs=[self._cells_d, result.pressure, state.p, g.ng])
        fill_boundaries(state)
        enforce_voxel_solids(state)

    @cached_property
    def _cartesian_volume_d(self):
        return wp.full(self.cell_count, float(self.grid.dx*self.grid.dy*self.grid.dz),
                       dtype=wp.float64, device=self.device)

    def divergence(self, dofs, *, cartesian=False):
        """RMS and maximum net flux per cell in 1/s.

        Relative to each cell's liquid volume by default; ``cartesian`` divides
        by the cell volume instead, the rate at which the stored field changes
        the water of that cell. The liquid-relative value of a nearly empty cell
        amplifies the float32 write-back rounding of its faces.
        """
        if not self.cell_count:
            return 0.0, 0.0
        volume = self._cartesian_volume_d if cartesian else self._volume_d
        sum_squares = wp.zeros(1, dtype=wp.float64, device=self.device)
        maximum = wp.zeros(1, dtype=wp.float64, device=self.device)
        wp.launch(_divergence_statistics, dim=self.cell_count, device=self.device, inputs=[
            self._widths_d, self.starts, self._divergence_d, volume, dofs, sum_squares, maximum])
        return (float(np.sqrt(sum_squares.numpy()[0] / self.cell_count)), float(maximum.numpy()[0]))

    def worst_divergence_cell(self, dofs, *, cartesian=False):
        """Cell index (i, j, k) and signed value of the largest |net flux / volume|."""
        if not self.cell_count:
            return None, 0.0
        volume = self._cartesian_volume_d if cartesian else self._volume_d
        values = wp.zeros(self.cell_count, dtype=wp.float64, device=self.device)
        wp.launch(_divergence_per_cell, dim=self.cell_count, device=self.device, inputs=[
            self._widths_d, self.starts, self._divergence_d, volume, dofs, values])
        host = values.numpy()
        c = int(np.argmax(np.abs(host)))
        cell = self._cells_d.numpy()[c]
        return tuple(int(v) for v in cell), float(host[c])

    def kinetic_energy(self, dofs):
        if not self.cell_count:
            return 0.0
        total = wp.zeros(1, dtype=wp.float64, device=self.device)
        wp.launch(_kinetic_energy, dim=self.cell_count, device=self.device, inputs=[
            self._widths_d, self.starts, self._mass_d, dofs, total])
        return .5*self.rho*float(total.numpy()[0])


class LiquidPressureProjection:
    """Solver3D's combined physical gravity and liquid-domain projection.

    The before-divergence diagnostic describes incoming velocity, before this
    combined force/projection step. After-divergence includes MAC transfer.
    """

    def __init__(self, cfg, settings=None):
        from shoreloop.physics3d.projection import ProjectionConfig
        self.cfg = cfg
        self.settings = ProjectionConfig() if settings is None else settings
        if cfg.physics.surface_tension != 0:
            raise ValueError("liquid_mimetic currently requires zero surface tension")
        if self.settings.pcg.preconditioner != "jacobi":
            raise ValueError("liquid_mimetic currently uses the jacobi preconditioner")
        self.last_operator = None

    def project(self, state, dt, *, operator=None, acceleration=None):
        """Project velocity using supplied geometry only while alpha/solids are fixed.

        ``acceleration`` defaults to gravity; pass zero to make an imposed
        initial field divergence-free without advancing it.
        """
        if acceleration is None:
            acceleration = (0, 0, -self.cfg.physics.g)
        from shoreloop.physics3d.free_surface import FreeSurfaceProjectionResult
        from shoreloop.physics3d.pcg import PCGResult
        fill_boundaries(state)
        enforce_voxel_solids(state)
        if operator is None:
            operator = LiquidPressureOperator(state, velocity_space="mac")
        initial = operator.read_mac_state(state)
        before = operator.divergence(initial)
        result = operator.project(dt, acceleration=tuple(acceleration),
                                  initial_flux=initial,
                                  tolerance=self.settings.pcg.relative_tolerance,
                                  absolute_tolerance=max(1e-14, self.settings.pcg.absolute_tolerance),
                                  max_iterations=self.settings.pcg.max_iterations)
        solution = operator.divergence(result.flux)
        operator.write_mac_state(state, result)
        stored = operator.read_mac_state(state)
        after = operator.divergence(stored)
        per_cell_volume = operator.divergence(stored, cartesian=True)
        self.last_operator = operator
        solve = PCGResult(state.p, True, result.iterations, result.residual,
                          result.residual/result.rhs_norm if result.rhs_norm else 0.0)
        return FreeSurfaceProjectionResult(
            solve, after[0] <= self.settings.rms_divergence_tolerance,
            *before, *after, operator.cell_count,
            operator.free_facet_count,
            0.0, 0.0, 0.0, 0, 0,
            solution_rms_divergence=solution[0],
            solution_max_divergence=solution[1],
            max_cell_divergence=per_cell_volume[1],
        )
