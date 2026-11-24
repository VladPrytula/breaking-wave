"""Symmetric graph diffusion in the retained-liquid kinetic metric.

Unique MAC velocities assemble M=T.T M_local T from the pressure operator.
The componentwise graph approximation supplies K=B.T diag(c) B. Along a
component's own axis, c=nu*Vwater/h**2 integrates its linear MAC derivative.
Transverse edges use c=nu*Adual/h, with Adual the half-sum of shared liquid
facets incident on the MAC face. Missing liquid connections have zero flux.
Wall tangential stress remains a separate closure.

Implicit midpoint solves (M+dt*K/2) delta=-dt*K*u. Its physical work is
rho*dt*sum(c*(B*u_mid)**2). Actual kinetic loss minus that work is reported
as signed numerical loss, including the FP32 state write and linear residual.
Geometry/topology and local least-squares stencils are built on the host.
State, strain and SGS coefficients use CUDA FP32; the diffusion system,
edge-force accumulation and Krylov vectors use CUDA FP64 to retain thin-liquid
kinetic modes. The final state write and its numerical work remain explicit.
"""

from dataclasses import dataclass
from functools import cached_property
import weakref
import math
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix
import warp as wp
from warp.optim.linear import cg, LinearOperator
from warp.sparse import bsr_axpy, bsr_from_triplets, bsr_mv

from shoreloop.config import BC
from shoreloop.physics3d.boundary import fill_boundaries
from shoreloop.physics3d.liquid_assembly import KIND_SHARED, _Mat6, _invert6, exclusive_scan
from shoreloop.physics3d.liquid_pressure import LiquidPressureOperator
from shoreloop.physics3d.solids import enforce_voxel_solids

# Cells holding less liquid than this fraction of a cell also get the inverse
# of their own-cell block of the solved system added to the line
# preconditioner. Measured 2026-10-13 (out/review_20261013/cg_growth): the
# extreme eigenvectors of the line-preconditioned diffusion and wall systems
# sat entirely on cells with 0.020-0.026 of a cell of liquid, whose mimetic
# mass couples one heavy face (2-4 cell volumes) to light faces (< 0.03) of
# other components, a coupling no own-axis line contains; the preconditioned
# condition number grew 132 -> 1224 over the production run with their number.
# 0.1 covers them with about 1% of the cells; all cells gave the same counts.
BLOCK_LIQUID_FRACTION = 0.1


@wp.kernel
def _difference_matvec(starts: wp.array(dtype=int), columns: wp.array(dtype=int),
                      entries: wp.array(dtype=Any), row_sum: wp.array(dtype=Any),
                      x: wp.array(dtype=Any), y: wp.array(dtype=Any),
                      z: wp.array(dtype=Any), alpha: Any, beta: Any):
    i = wp.tid()
    xi = x[i]
    value = row_sum[i]*xi
    for q in range(starts[i], starts[i+1]):
        value += entries[q]*(x[columns[q]]-xi)
    z[i] = x.dtype(alpha)*value + x.dtype(beta)*y[i]


@wp.kernel
def _solve_lines(nodes: wp.array2d(dtype=int), lower: wp.array2d(dtype=Any),
                 lower_error: wp.array2d(dtype=Any),
                 pivots: wp.array2d(dtype=Any), work: wp.array2d(dtype=Any),
                 x: wp.array(dtype=Any), y: wp.array(dtype=Any),
                 z: wp.array(dtype=Any), alpha: Any, beta: Any):
    line = wp.tid()
    previous = x.dtype(0.0)
    for j in range(nodes.shape[0]):
        node = nodes[j, line]
        value = x.dtype(0.0)
        if node >= 0:
            value = x[node]
        value -= lower[j, line]*previous
        value -= lower_error[j, line]*previous
        work[j, line] = value
        previous = value
    previous = x.dtype(0.0)
    for offset in range(nodes.shape[0]):
        j = nodes.shape[0]-1-offset
        value = work[j, line]/pivots[j, line]
        if j+1 < nodes.shape[0]:
            value -= lower[j+1, line]*previous
            value -= lower_error[j+1, line]*previous
        node = nodes[j, line]
        if node >= 0:
            z[node] = x.dtype(alpha)*value + x.dtype(beta)*y[node]
        previous = value


def _line_preconditioner(mass, nodes, edges, coefficients, dt, device, *, dtype=wp.float32):
    """Open tridiagonal velocity lines; all other couplings stay in the system.

    Near-empty cells can have large kinetic penalties on differences between
    opposite MAC velocities, beside tiny translational inertia. Jacobi cannot
    resolve that contrast. Factor each component's own-coordinate line in
    host metric precision, then apply its positive LDL.T inverse on CUDA in the
    solve's precision. Wall stress retains its existing FP32 application.
    Periodic wrap edges remain in the system and contribute only their diagonal
    diffusion coefficients to this open-line preconditioner.
    """
    positions = nodes[np.arange(len(nodes)), nodes[:, 0]+1]
    keys = nodes.copy()
    keys[np.arange(len(nodes)), nodes[:, 0]+1] = 0
    _, lines = np.unique(keys, axis=0, return_inverse=True)
    shape = (int(positions.max()+1), int(lines.max()+1))
    indices = positions*shape[1]+lines
    pivots = np.ones(shape)
    pivots.ravel()[indices] = mass.diagonal()
    lower = np.zeros(shape)
    matrix = mass.tocoo()
    keep = ((lines[matrix.row] == lines[matrix.col])
            & (positions[matrix.row] == positions[matrix.col]+1))
    lower.ravel()[indices[matrix.row[keep]]] = matrix.data[keep]
    weights = .5*dt*coefficients
    for side in range(2):
        np.add.at(pivots.ravel(), indices[edges[:, side]], weights)
    keep = ((lines[edges[:, 0]] == lines[edges[:, 1]])
            & (positions[edges[:, 1]] == positions[edges[:, 0]]+1))
    np.add.at(lower.ravel(), indices[edges[keep, 1]], -weights[keep])
    for j in range(1, shape[0]):
        edge = lower[j].copy()
        lower[j] /= pivots[j-1]
        pivots[j] -= lower[j]*edge
    if not np.all(pivots > 0):
        raise FloatingPointError("liquid diffusion line metric is not positive definite")
    lookup = np.full(shape, -1, dtype=np.int32)
    lookup.ravel()[indices] = np.arange(len(nodes))
    node_array = wp.array(lookup, dtype=wp.int32, device=device)
    lower_array = wp.array(lower, dtype=dtype, device=device)
    # A factor near -1 still carries the ratio of tiny rigid inertia to a
    # large difference penalty. Preserve its rounding remainder separately:
    # otherwise forward cancellation can erase the momentum-carrying mode.
    numpy_dtype = np.float64 if dtype == wp.float64 else np.float32
    lower_error = wp.array(lower-lower.astype(numpy_dtype).astype(float),
                           dtype=dtype, device=device)
    pivot_array = wp.array(pivots, dtype=dtype, device=device)
    work = wp.empty(shape, dtype=dtype, device=device)

    def matvec(x, y, z, alpha, beta):
        wp.launch(_solve_lines, dim=shape[1], inputs=[node_array, lower_array, lower_error, pivot_array,
                  work, x, y, z, alpha, beta], device=device)

    return LinearOperator(mass.shape, dtype, wp.get_device(device), matvec)


@wp.kernel
def _gather_velocity(nodes: wp.array(dtype=wp.vec4i), u: wp.array3d(dtype=wp.float32),
                     v: wp.array3d(dtype=wp.float32), w: wp.array3d(dtype=wp.float32),
                     values: wp.array(dtype=wp.float32), ng: int):
    n = wp.tid()
    q = nodes[n]
    value = u[q[1]+ng, q[2]+ng, q[3]+ng]
    if q[0] == 1:
        value = v[q[1]+ng, q[2]+ng, q[3]+ng]
    elif q[0] == 2:
        value = w[q[1]+ng, q[2]+ng, q[3]+ng]
    values[n] = value


@wp.kernel
def _scatter_velocity(nodes: wp.array(dtype=wp.vec4i), values: wp.array(dtype=wp.float32),
                      u: wp.array3d(dtype=wp.float32), v: wp.array3d(dtype=wp.float32),
                      w: wp.array3d(dtype=wp.float32), ng: int):
    n = wp.tid()
    q = nodes[n]
    if q[0] == 0:
        u[q[1]+ng, q[2]+ng, q[3]+ng] = values[n]
    elif q[0] == 1:
        v[q[1]+ng, q[2]+ng, q[3]+ng] = values[n]
    else:
        w[q[1]+ng, q[2]+ng, q[3]+ng] = values[n]


@wp.kernel
def _liquid_strain(starts: wp.array(dtype=int), samples: wp.array(dtype=int),
                   weights: wp.array(dtype=wp.vec3), cells: wp.array(dtype=wp.vec3i),
                   velocity: wp.array(dtype=wp.float32), strain: wp.array3d(dtype=wp.float32),
                   viscosity: wp.array3d(dtype=wp.float32), maximum: wp.array(dtype=wp.float32),
                   cs_delta_squared: wp.float32, cap: wp.float32, ng: int):
    cell = wp.tid()
    gradient = wp.mat33(0.0)
    for component in range(3):
        start = int(starts[3*cell+component])
        end = int(starts[3*cell+component+1])
        if end > start:
            anchor = velocity[samples[start]]
            row = wp.vec3(0.0)
            for q in range(start, end):
                # Subtracting an observed anchor makes constant translation
                # exactly strain-free, even with rounded stencil coefficients.
                row += weights[q] * (velocity[samples[q]]-anchor)
            for axis in range(3):
                gradient[component, axis] = row[axis]
    squared = wp.float32(0.0)
    for a in range(3):
        for b in range(3):
            symmetric = wp.float32(.5)*(gradient[a, b]+gradient[b, a])
            squared += wp.float32(2.0)*symmetric*symmetric
    magnitude = wp.sqrt(squared)
    value = wp.min(cap, cs_delta_squared*magnitude)
    pos = cells[cell]
    strain[pos[0]+ng, pos[1]+ng, pos[2]+ng] = magnitude
    viscosity[pos[0]+ng, pos[1]+ng, pos[2]+ng] = value
    wp.atomic_max(maximum, 0, value)


@wp.func
def _node_viscosity(node: int, starts: wp.array(dtype=int), owners: wp.array(dtype=int),
                    cells: wp.array(dtype=wp.vec3i), viscosity: wp.array3d(dtype=wp.float32),
                    ng: int):
    value = wp.float32(0.0)
    for q in range(starts[node], starts[node+1]):
        c = cells[owners[q]]
        value += viscosity[c[0]+ng, c[1]+ng, c[2]+ng]
    return value/wp.float32(starts[node+1]-starts[node])


@wp.kernel
def _edge_coefficients(edges: wp.array(dtype=wp.vec2i), geometry: wp.array(dtype=wp.float32),
                       starts: wp.array(dtype=int), owners: wp.array(dtype=int),
                       cells: wp.array(dtype=wp.vec3i), viscosity: wp.array3d(dtype=wp.float32),
                       coefficients: wp.array(dtype=wp.float32), entries: wp.array(dtype=wp.float64),
                       base: wp.float32, ng: int):
    e = wp.tid()
    edge = edges[e]
    nu = base + wp.float32(.5)*(
        _node_viscosity(edge[0], starts, owners, cells, viscosity, ng)
        + _node_viscosity(edge[1], starts, owners, cells, viscosity, ng))
    c = geometry[e]*nu
    coefficients[e] = c
    entries[4*e] = wp.float64(c)
    entries[4*e+1] = wp.float64(c)
    entries[4*e+2] = -wp.float64(c)
    entries[4*e+3] = -wp.float64(c)


@wp.kernel
def _edge_rhs(edges: wp.array(dtype=wp.vec2i), coefficients: wp.array(dtype=wp.float32),
              before: wp.array(dtype=wp.float32), rhs: wp.array(dtype=wp.float64), dt: wp.float64):
    e = wp.tid()
    edge = edges[e]
    transfer = dt*wp.float64(coefficients[e])*(wp.float64(before[edge[1]])-wp.float64(before[edge[0]]))
    wp.atomic_add(rhs, edge[0], transfer)
    wp.atomic_add(rhs, edge[1], -transfer)


@wp.kernel
def _add_increment(before: wp.array(dtype=wp.float32), delta: wp.array(dtype=Any),
                   after: wp.array(dtype=wp.float32)):
    n = wp.tid()
    after[n] = wp.float32(delta.dtype(before[n])+delta[n])


@wp.kernel
def _edge_work(edges: wp.array(dtype=wp.vec2i), coefficients: wp.array(dtype=wp.float32),
               before: wp.array(dtype=wp.float32), after: wp.array(dtype=wp.float32),
               starts: wp.array(dtype=int), owners: wp.array(dtype=int), cells: wp.array(dtype=wp.vec3i),
               field: wp.array3d(dtype=wp.float32), total: wp.array(dtype=wp.float64),
               rho: wp.float64, dt: wp.float64, inverse_volume: wp.float64, ng: int):
    e = wp.tid()
    edge = edges[e]
    midpoint_i = wp.float64(.5)*(wp.float64(before[edge[0]])+wp.float64(after[edge[0]]))
    midpoint_j = wp.float64(.5)*(wp.float64(before[edge[1]])+wp.float64(after[edge[1]]))
    difference = midpoint_j-midpoint_i
    power = rho*wp.float64(coefficients[e])*difference*difference
    wp.atomic_add(total, 0, dt*power)
    # Each edge's power is split between endpoints, then equally among their
    # incident liquid cells. Fields are W/m^3 of full grid volume, as foam
    # expects; integrating the field gives the physical edge work exactly up
    # to its FP32 accumulation, without a normalization or repair.
    for side in range(2):
        node = edge[side]
        density = wp.float32(wp.float64(.5)*power*inverse_volume
                            /wp.float64(starts[node+1]-starts[node]))
        for q in range(starts[node], starts[node+1]):
            pos = cells[owners[q]]
            wp.atomic_add(field, pos[0]+ng, pos[1]+ng, pos[2]+ng, density)


@dataclass(frozen=True)
class LiquidDiffusionResult:
    physical_loss: float
    numerical_loss: float
    iterations: int
    relative_residual: float


def _code(shape_plus_one, axis, position):
    """Integer key of a face (axis, i, j, k) with positions in [0, shape]."""
    position = np.asarray(position, dtype=np.int64)
    return ((np.asarray(axis, dtype=np.int64) * shape_plus_one[0] + position[..., 0]) * shape_plus_one[1]
            + position[..., 1]) * shape_plus_one[2] + position[..., 2]


def _lookup(sorted_codes, ranks, codes):
    """Rank of each code in a sorted unique code table, or -1 when absent."""
    codes = np.asarray(codes, dtype=np.int64)
    if len(sorted_codes) == 0:
        return np.full(codes.shape, -1, dtype=np.int64)
    place = np.minimum(np.searchsorted(sorted_codes, codes), len(sorted_codes) - 1)
    return np.where(sorted_codes[place] == codes, ranks[place], -1)


# ---------------------------------------------------------------------------
# Device construction of the metric and the diffusion operator.

_OFFSET_DIRECTION = wp.types.vector(length=7, dtype=wp.int32)(0, 0, 0, 1, 1, 2, 2)
_OFFSET_SHIFT = wp.types.vector(length=7, dtype=wp.int32)(0, -1, 1, -1, 1, -1, 1)
_Positions14 = wp.types.matrix(shape=(14, 3), dtype=wp.int32)
_Ints14 = wp.types.vector(length=14, dtype=wp.int32)


@wp.func
def _face_code(axis: int, p: wp.vec3i, shape: wp.vec3i) -> int:
    """Code of a face (axis, position) with positions in [0, shape]; -1 outside."""
    for a in range(3):
        if p[a] < 0 or p[a] > shape[a]:
            return -1
    return ((axis * (shape[0] + 1) + p[0]) * (shape[1] + 1) + p[1]) * (shape[2] + 1) + p[2]


@wp.kernel
def _all_dof_codes(widths: wp.array(dtype=wp.int32), dof_axis: wp.array2d(dtype=wp.int32),
                   dof_position: wp.array2d(dtype=wp.vec3i), shape: wp.vec3i, periodic: wp.vec3i,
                   codes: wp.array2d(dtype=wp.int32)):
    c = wp.tid()
    for i in range(widths[c]):
        axis = dof_axis[c, i]
        p = dof_position[c, i]
        if periodic[axis] == 1:
            p[axis] = p[axis] % shape[axis]
        codes[c, i] = _face_code(axis, p, shape)


@wp.kernel
def _mark_all(widths: wp.array(dtype=wp.int32), codes: wp.array2d(dtype=wp.int32), mark: wp.array(dtype=wp.int32)):
    c = wp.tid()
    for i in range(widths[c]):
        mark[codes[c, i]] = 1


@wp.kernel
def _node_table(mark: wp.array(dtype=wp.int32), scan: wp.array(dtype=wp.int32), table: wp.array(dtype=wp.int32)):
    i = wp.tid()
    if mark[i] == 1:
        table[i] = scan[i] - 1
    else:
        table[i] = -1


@wp.kernel
def _assign_nodes(widths: wp.array(dtype=wp.int32), codes: wp.array2d(dtype=wp.int32),
                  table: wp.array(dtype=wp.int32), dof_axis: wp.array2d(dtype=wp.int32),
                  dof_position: wp.array2d(dtype=wp.vec3i), shape: wp.vec3i, periodic: wp.vec3i,
                  node_of_dof: wp.array2d(dtype=wp.int32), nodes: wp.array(dtype=wp.vec4i),
                  owner_count: wp.array(dtype=wp.int32)):
    c = wp.tid()
    for i in range(widths[c]):
        node = table[codes[c, i]]
        node_of_dof[c, i] = node
        axis = dof_axis[c, i]
        p = dof_position[c, i]
        if periodic[axis] == 1:
            p[axis] = p[axis] % shape[axis]
        nodes[node] = wp.vec4i(axis, p[0], p[1], p[2])
        wp.atomic_add(owner_count, node, 1)


@wp.kernel
def _fill_owners(widths: wp.array(dtype=wp.int32), node_of_dof: wp.array2d(dtype=wp.int32),
                 owner_starts: wp.array(dtype=wp.int32), cursor: wp.array(dtype=wp.int32),
                 owners: wp.array(dtype=wp.int32)):
    c = wp.tid()
    for i in range(widths[c]):
        node = node_of_dof[c, i]
        slot = wp.atomic_add(cursor, node, 1)
        owners[owner_starts[node] + slot] = c


@wp.kernel
def _sort_owners(owner_starts: wp.array(dtype=wp.int32), owners: wp.array(dtype=wp.int32)):
    n = wp.tid()
    start = owner_starts[n]
    end = owner_starts[n + 1]
    for a in range(start + 1, end):
        key = owners[a]
        b = a - 1
        while b >= start and owners[b] > key:
            owners[b + 1] = owners[b]
            b -= 1
        owners[b + 1] = key


@wp.kernel
def _mass_triplets(widths: wp.array(dtype=wp.int32), node_of_dof: wp.array2d(dtype=wp.int32),
                   dof_sign: wp.array2d(dtype=wp.int32), mass: wp.array3d(dtype=wp.float64),
                   offsets: wp.array(dtype=wp.int32), rows: wp.array(dtype=wp.int32),
                   cols: wp.array(dtype=wp.int32), values: wp.array(dtype=wp.float64)):
    c = wp.tid()
    n = offsets[c]
    for i in range(widths[c]):
        for j in range(widths[c]):
            rows[n] = node_of_dof[c, i]
            cols[n] = node_of_dof[c, j]
            values[n] = wp.float64(dof_sign[c, i] * dof_sign[c, j]) * mass[c, i, j]
            n += 1


@wp.kernel
def _bsr_row_sums(offsets: wp.array(dtype=int), values: wp.array(dtype=wp.float64), sums: wp.array(dtype=wp.float64)):
    i = wp.tid()
    total = wp.float64(0.0)
    for q in range(offsets[i], offsets[i + 1]):
        total += values[q]
    sums[i] = total


@wp.func
def _bsr_entry(offsets: wp.array(dtype=int), columns: wp.array(dtype=int),
               values: wp.array(dtype=wp.float64), row: int, column: int) -> wp.float64:
    # Triplet-built matrices may keep several slots for one column; sum them.
    total = wp.float64(0.0)
    for q in range(offsets[row], offsets[row + 1]):
        if columns[q] == column:
            total += values[q]
    return total


@wp.kernel
def _cell_table(cells: wp.array(dtype=wp.vec3i), table: wp.array3d(dtype=wp.int32)):
    c = wp.tid()
    q = cells[c]
    table[q[0], q[1], q[2]] = c


@wp.kernel
def _shared_high_areas(facet_count: wp.array(dtype=wp.int32), facet_kind: wp.array2d(dtype=wp.int32),
                       facet_face: wp.array2d(dtype=wp.int32), facet_area: wp.array2d(dtype=wp.float64),
                       shared_high: wp.array2d(dtype=wp.float64)):
    c = wp.tid()
    for f in range(facet_count[c]):
        face = facet_face[c, f]
        if facet_kind[c, f] == KIND_SHARED and face >= 0 and face % 2 == 1:
            shared_high[c, face // 2] = shared_high[c, face // 2] + facet_area[c, f]


@wp.func
def _owns(owner_starts: wp.array(dtype=wp.int32), owners: wp.array(dtype=wp.int32), node: int, cell: int) -> int:
    for q in range(owner_starts[node], owner_starts[node + 1]):
        if owners[q] == cell:
            return 1
    return 0


@wp.kernel
def _edge_candidates(nodes: wp.array(dtype=wp.vec4i), table: wp.array(dtype=wp.int32), shape: wp.vec3i,
                     periodic: wp.vec3i, cell_table: wp.array3d(dtype=wp.int32),
                     owner_starts: wp.array(dtype=wp.int32), owners: wp.array(dtype=wp.int32),
                     cells: wp.array(dtype=wp.vec3i), volumes: wp.array(dtype=wp.float64),
                     shared_high: wp.array2d(dtype=wp.float64), spacing: wp.vec3d,
                     kept: wp.array(dtype=wp.int32), neighbour: wp.array(dtype=wp.int32),
                     conductance: wp.array(dtype=wp.float64)):
    """Edge from each node to the node one cell further along each direction.

    Same-direction conductance is the owning cell's volume over the squared
    spacing; transverse conductance is half the shared high-face area over the
    spacing for every owner whose high-face neighbour also owns the other node.
    """
    n = wp.tid()
    q = nodes[n]
    axis = q[0]
    p = wp.vec3i(q[1], q[2], q[3])
    for d in range(3):
        slot = 3 * n + d
        other = wp.vec3i(p[0], p[1], p[2])
        other[d] = other[d] + 1
        if periodic[d] == 1:
            other[d] = other[d] % shape[d]
        m = int(-1)
        code = _face_code(axis, other, shape)
        if code >= 0:
            m = table[code]
        kept[slot] = 0
        neighbour[slot] = m
        conductance[slot] = wp.float64(0.0)
        if m < 0 or m == n:
            continue
        value = wp.float64(0.0)
        if d == axis:
            if p[0] < shape[0] and p[1] < shape[1] and p[2] < shape[2]:
                cell = cell_table[p[0], p[1], p[2]]
                if cell >= 0:
                    value += volumes[cell] / (spacing[d] * spacing[d])
        else:
            for o in range(owner_starts[n], owner_starts[n + 1]):
                owner = owners[o]
                face = cells[owner]
                face[d] = face[d] + 1
                if periodic[d] == 1:
                    face[d] = face[d] % shape[d]
                if face[d] >= shape[d]:
                    continue
                other_cell = cell_table[face[0], face[1], face[2]]
                if other_cell >= 0 and _owns(owner_starts, owners, m, other_cell) == 1:
                    value += wp.float64(0.5) * shared_high[owner, d] / spacing[d]
        if value > wp.float64(0.0):
            kept[slot] = 1
            conductance[slot] = value


@wp.kernel
def _compact_edges(kept: wp.array(dtype=wp.int32), offsets: wp.array(dtype=wp.int32),
                   neighbour: wp.array(dtype=wp.int32), conductance: wp.array(dtype=wp.float64),
                   edges: wp.array(dtype=wp.vec2i), geometry: wp.array(dtype=wp.float32),
                   edge_of_slot: wp.array(dtype=wp.int32), rows: wp.array(dtype=wp.int32),
                   columns: wp.array(dtype=wp.int32)):
    slot = wp.tid()
    edge_of_slot[slot] = -1
    if kept[slot] == 1:
        e = offsets[slot]
        a = slot / 3
        b = neighbour[slot]
        edges[e] = wp.vec2i(a, b)
        geometry[e] = wp.float32(conductance[slot])
        edge_of_slot[slot] = e
        rows[4 * e] = a
        rows[4 * e + 1] = b
        rows[4 * e + 2] = a
        rows[4 * e + 3] = b
        columns[4 * e] = a
        columns[4 * e + 1] = b
        columns[4 * e + 2] = b
        columns[4 * e + 3] = a


@wp.kernel
def _strain_stencils(widths: wp.array(dtype=wp.int32), node_of_dof: wp.array2d(dtype=wp.int32),
                     dof_axis: wp.array2d(dtype=wp.int32), dof_sign: wp.array2d(dtype=wp.int32),
                     dof_position: wp.array2d(dtype=wp.vec3i), table: wp.array(dtype=wp.int32),
                     shape: wp.vec3i, periodic: wp.vec3i, edge_of_slot: wp.array(dtype=wp.int32),
                     neighbour: wp.array(dtype=wp.int32), pattern: wp.array2d(dtype=wp.int32),
                     kept_count: wp.array2d(dtype=wp.int32), samples: wp.array3d(dtype=wp.int32)):
    """Affine least-squares stencils: each face of a component plus its
    immediate same-component neighbours connected through liquid edges. The
    first accepted slot at each distinct local position is kept."""
    c = wp.tid()
    for component in range(3):
        positions = _Positions14()
        accepted = _Ints14()
        for face_slot in range(2):
            sign = 2 * face_slot - 1
            has = int(0)
            source = int(-1)
            face = wp.vec3i(0, 0, 0)
            for i in range(widths[c]):
                if dof_axis[c, i] == component and dof_sign[c, i] == sign:
                    has = 1
                    source = node_of_dof[c, i]
                    face = dof_position[c, i]
            for offset_slot in range(7):
                slot = 7 * face_slot + offset_slot
                sample = int(-1)
                local = wp.vec3i(face[0], face[1], face[2])
                if has == 1:
                    direction = _OFFSET_DIRECTION[offset_slot]
                    shift = _OFFSET_SHIFT[offset_slot]
                    local[direction] = local[direction] + shift
                    wrapped = wp.vec3i(local[0], local[1], local[2])
                    for a in range(3):
                        if periodic[a] == 1:
                            wrapped[a] = ((wrapped[a] % shape[a]) + shape[a]) % shape[a]
                    code = _face_code(component, wrapped, shape)
                    if code >= 0:
                        s = table[code]
                        if s >= 0:
                            connected = int(0)
                            if s == source:
                                connected = 1
                            elif shift > 0:
                                e = 3 * source + direction
                                if edge_of_slot[e] >= 0 and neighbour[e] == s:
                                    connected = 1
                            else:
                                e = 3 * s + direction
                                if edge_of_slot[e] >= 0 and neighbour[e] == source:
                                    connected = 1
                            if connected == 1:
                                sample = s
                positions[slot] = local
                accepted[slot] = sample
        bits = int(0)
        count = int(0)
        for slot in range(14):
            if accepted[slot] < 0:
                continue
            duplicate = int(0)
            for earlier in range(slot):
                if accepted[earlier] >= 0 and positions[earlier][0] == positions[slot][0] and positions[earlier][1] == positions[slot][1] and positions[earlier][2] == positions[slot][2]:
                    duplicate = 1
            if duplicate == 1:
                accepted[slot] = -1
                continue
            bits |= 1 << slot
            count += 1
        pattern[c, component] = bits | (component << 14)
        kept_count[c, component] = count
        for slot in range(14):
            samples[c, component, slot] = accepted[slot]


@wp.kernel
def _fill_stencils(pattern: wp.array2d(dtype=wp.int32), samples_table: wp.array3d(dtype=wp.int32),
                   starts: wp.array(dtype=wp.int32), pattern_row: wp.array2d(dtype=wp.int32),
                   weight_table: wp.array2d(dtype=wp.vec3), samples: wp.array(dtype=wp.int32),
                   weights: wp.array(dtype=wp.vec3)):
    c = wp.tid()
    for component in range(3):
        n = starts[3 * c + component]
        row = pattern_row[c, component]
        for slot in range(14):
            if samples_table[c, component, slot] >= 0:
                samples[n] = samples_table[c, component, slot]
                weights[n] = weight_table[row, slot]
                n += 1


@wp.kernel
def _line_keys(nodes: wp.array(dtype=wp.vec4i), shape: wp.vec3i, codes: wp.array(dtype=wp.int32),
               positions: wp.array(dtype=wp.int32), mark: wp.array(dtype=wp.int32)):
    n = wp.tid()
    q = nodes[n]
    key = wp.vec3i(q[1], q[2], q[3])
    positions[n] = key[q[0]]
    key[q[0]] = 0
    code = _face_code(q[0], key, shape)
    codes[n] = code
    mark[code] = 1


@wp.kernel
def _line_table(codes: wp.array(dtype=wp.int32), positions: wp.array(dtype=wp.int32),
                scan: wp.array(dtype=wp.int32), table: wp.array2d(dtype=wp.int32)):
    n = wp.tid()
    table[positions[n], scan[codes[n]] - 1] = n


@wp.kernel
def _factor_lines(table: wp.array2d(dtype=wp.int32),
                  offsets: wp.array(dtype=int), columns: wp.array(dtype=int), values: wp.array(dtype=wp.float64),
                  pivots: wp.array2d(dtype=wp.float64), lower: wp.array2d(dtype=wp.float64),
                  failure: wp.array(dtype=wp.int32)):
    """Open tridiagonal LDL^T factor of each own-coordinate velocity line of a matrix."""
    line = wp.tid()
    previous = int(-1)
    for j in range(table.shape[0]):
        node = table[j, line]
        pivot = wp.float64(1.0)
        low = wp.float64(0.0)
        if node >= 0:
            pivot = _bsr_entry(offsets, columns, values, node, node)
            if previous >= 0:
                low = _bsr_entry(offsets, columns, values, node, previous)
        if j > 0:
            original = low
            low = low / pivots[j - 1, line]
            pivot -= low * original
        pivots[j, line] = pivot
        lower[j, line] = low
        if pivot <= wp.float64(0.0):
            failure[0] = 1
        previous = node


@wp.kernel
def _mark_block_cells(volume: wp.array(dtype=wp.float64), widths: wp.array(dtype=wp.int32),
                      limit: wp.float64, mark: wp.array(dtype=wp.int32)):
    c = wp.tid()
    if volume[c] < limit and widths[c] > 0:
        mark[c] = 1


@wp.kernel
def _block_inverses(mark: wp.array(dtype=wp.int32), slots: wp.array(dtype=wp.int32),
                    widths: wp.array(dtype=wp.int32), node_of_dof: wp.array2d(dtype=wp.int32),
                    offsets: wp.array(dtype=int), columns: wp.array(dtype=int), values: wp.array(dtype=wp.float64),
                    cells: wp.array(dtype=wp.int32), inverse: wp.array(dtype=_Mat6)):
    """Inverse of a marked cell's own-node block of the assembled matrix."""
    c = wp.tid()
    if mark[c] == 0:
        return
    n = widths[c]
    block = _Mat6()
    for i in range(n):
        for j in range(n):
            block[i, j] = _bsr_entry(offsets, columns, values, node_of_dof[c, i], node_of_dof[c, j])
    cells[slots[c]] = c
    inverse[slots[c]] = _invert6(block, n)


@wp.kernel
def _apply_blocks(cells: wp.array(dtype=wp.int32), widths: wp.array(dtype=wp.int32),
                  node_of_dof: wp.array2d(dtype=wp.int32), inverse: wp.array(dtype=_Mat6),
                  x: wp.array(dtype=wp.float64), alpha: wp.float64, z: wp.array(dtype=wp.float64)):
    s = wp.tid()
    c = cells[s]
    block = inverse[s]
    for i in range(widths[c]):
        value = wp.float64(0.0)
        for j in range(widths[c]):
            value += block[i, j] * x[node_of_dof[c, j]]
        wp.atomic_add(z, node_of_dof[c, i], alpha * value)


@wp.kernel
def _cast_to_double(source: wp.array(dtype=wp.float32), target: wp.array(dtype=wp.float64)):
    i = wp.tid()
    target[i] = wp.float64(source[i])


class LiquidVelocityMetric:
    """The liquid MAC kinetic inner product on the operator's faces (nodes)."""

    def __init__(self, state, *, pressure=None):
        self.cfg, self.device = state.cfg, state.device
        self.pressure = (LiquidPressureOperator(state, velocity_space="mac")
                         if pressure is None else pressure)
        g = state.cfg.grid
        self.spacing = np.asarray((g.dx, g.dy, g.dz))
        self.volume = float(np.prod(self.spacing))
        self.periodic = tuple(getattr(state.cfg.bc, a+"lo") == BC.PERIODIC for a in "xyz")
        self._shape_plus_one = np.asarray(g.shape, dtype=np.int64) + 1
        self._host = {}
        pressure = self.pressure
        device = self.device
        count, nmax = pressure.cell_count, pressure.nmax
        shape = wp.vec3i(*g.shape)
        periodic = wp.vec3i(*[int(v) for v in self.periodic])
        widths = pressure._widths_d
        self._shape, self._periodic_vec = shape, periodic
        # Nodes: every kept MAC face, numbered by a scan over its face code.
        code_span = int(3 * np.prod(self._shape_plus_one))
        codes = wp.full((count, nmax), -1, dtype=wp.int32, device=device)
        mark = wp.zeros(code_span, dtype=wp.int32, device=device)
        if count:
            wp.launch(_all_dof_codes, dim=count, device=device,
                      inputs=[widths, pressure._dof_axis_d, pressure._dof_position_d, shape, periodic, codes])
            wp.launch(_mark_all, dim=count, device=device, inputs=[widths, codes, mark])
        scan = wp.zeros_like(mark)
        wp.utils.array_scan(mark, scan, inclusive=True)
        node_count = int(scan.numpy()[-1])
        self.node_count = node_count
        self.node_table = wp.zeros(code_span, dtype=wp.int32, device=device)
        wp.launch(_node_table, dim=code_span, device=device, inputs=[mark, scan, self.node_table])
        self.node_of_dof = wp.full((count, nmax), -1, dtype=wp.int32, device=device)
        self.nodes = wp.zeros(max(node_count, 1), dtype=wp.vec4i, device=device)
        owner_count = wp.zeros(max(node_count, 1), dtype=wp.int32, device=device)
        if count:
            wp.launch(_assign_nodes, dim=count, device=device, inputs=[
                widths, codes, self.node_table, pressure._dof_axis_d, pressure._dof_position_d,
                shape, periodic, self.node_of_dof, self.nodes, owner_count])
        self.owner_starts, owner_total = exclusive_scan(owner_count, device)
        self.owners = wp.zeros(max(owner_total, 1), dtype=wp.int32, device=device)
        cursor = wp.zeros(max(node_count, 1), dtype=wp.int32, device=device)
        if count:
            wp.launch(_fill_owners, dim=count, device=device, inputs=[widths, self.node_of_dof, self.owner_starts, cursor, self.owners])
            wp.launch(_sort_owners, dim=node_count, device=device, inputs=[self.owner_starts, self.owners])
        # Kinetic metric M = T^T M_local T over the nodes.
        widths_host = pressure._widths
        offsets, total = exclusive_scan(wp.array((widths_host * widths_host).astype(np.int32), dtype=wp.int32, device=device), device)
        rows = wp.zeros(total, dtype=wp.int32, device=device)
        cols = wp.zeros_like(rows)
        values = wp.zeros(total, dtype=wp.float64, device=device)
        if count:
            wp.launch(_mass_triplets, dim=count, device=device, inputs=[
                widths, self.node_of_dof, pressure._dof_sign_d, pressure._mass_d, offsets, rows, cols, values])
        self.mass = bsr_from_triplets(node_count, node_count, rows, cols, values)
        self.mass_row_sum = wp.zeros(max(node_count, 1), dtype=wp.float64, device=device)
        if node_count:
            wp.launch(_bsr_row_sums, dim=node_count, device=device, inputs=[self.mass.offsets, self.mass.values, self.mass_row_sum])
        self.cells = pressure._cells_d
        # Own-coordinate velocity lines (one per component and transverse position)
        # for the tridiagonal preconditioners of the diffusion and wall solves.
        line_codes = wp.zeros(max(node_count, 1), dtype=wp.int32, device=device)
        positions = wp.zeros(max(node_count, 1), dtype=wp.int32, device=device)
        line_mark = wp.zeros(code_span, dtype=wp.int32, device=device)
        if node_count:
            wp.launch(_line_keys, dim=node_count, device=device, inputs=[self.nodes, shape, line_codes, positions, line_mark])
        line_scan = wp.zeros_like(line_mark)
        wp.utils.array_scan(line_mark, line_scan, inclusive=True)
        line_count = int(line_scan.numpy()[-1])
        self.line_table = wp.full((max(g.shape) + 1, max(line_count, 1)), -1, dtype=wp.int32, device=device)
        if node_count:
            wp.launch(_line_table, dim=node_count, device=device, inputs=[line_codes, positions, line_scan, self.line_table])
        self.line_count = line_count

    def line_preconditioner(self, system):
        """Open tridiagonal LDL^T of ``system`` along the own-coordinate lines,
        plus the inverse own-cell blocks of the cells below BLOCK_LIQUID_FRACTION.

        The two parts are added (each is symmetric positive definite on its
        own nodes, so their sum is); every other coupling stays in the full
        matrix action of the solve.
        """
        device = self.device
        shape = self.line_table.shape
        pivots = wp.zeros(shape, dtype=wp.float64, device=device)
        lower = wp.zeros(shape, dtype=wp.float64, device=device)
        failure = wp.zeros(1, dtype=wp.int32, device=device)
        wp.launch(_factor_lines, dim=shape[1], device=device, inputs=[
            self.line_table, system.offsets, system.columns, system.values, pivots, lower, failure])
        if int(failure.numpy()[0]):
            raise FloatingPointError("liquid line metric is not positive definite")
        lower_error = wp.zeros(shape, dtype=wp.float64, device=device)
        work = wp.empty(shape, dtype=wp.float64, device=device)
        pressure = self.pressure
        count, widths = pressure.cell_count, pressure._widths_d
        mark = wp.zeros(max(count, 1), dtype=wp.int32, device=device)
        if count:
            wp.launch(_mark_block_cells, dim=count, device=device, inputs=[
                pressure._volume_d, widths, wp.float64(BLOCK_LIQUID_FRACTION * self.volume), mark])
        slots, block_count = exclusive_scan(mark, device)
        cells = wp.zeros(max(block_count, 1), dtype=wp.int32, device=device)
        inverse = wp.zeros(max(block_count, 1), dtype=_Mat6, device=device)
        if block_count:
            wp.launch(_block_inverses, dim=count, device=device, inputs=[
                mark, slots, widths, self.node_of_dof, system.offsets, system.columns, system.values, cells, inverse])

        def matvec(x, y, z, alpha, beta):
            wp.launch(_solve_lines, dim=shape[1], inputs=[self.line_table, lower, lower_error, pivots,
                      work, x, y, z, alpha, beta], device=device)
            if block_count:
                wp.launch(_apply_blocks, dim=block_count, inputs=[cells, widths, self.node_of_dof, inverse,
                          x, wp.float64(alpha), z], device=device)

        return LinearOperator(system.shape, wp.float64, wp.get_device(device), matvec)

    # ---- host views --------------------------------------------------------------
    def _view(self, name, produce):
        if name not in self._host:
            self._host[name] = produce()
        return self._host[name]

    @property
    def nodes_host(self):
        return self._view("nodes_host", lambda: self.nodes.numpy()[:self.node_count].astype(np.int32).reshape(-1, 4))

    @property
    def mass_host(self):
        def produce():
            offsets = self.mass.offsets.numpy(); columns = self.mass.columns.numpy(); values = self.mass.values.numpy().reshape(-1)
            nnz = int(offsets[-1]); rows = np.repeat(np.arange(self.node_count), np.diff(offsets))
            matrix = coo_matrix((values[:nnz], (rows, columns[:nnz])), shape=(self.node_count, self.node_count)).tocsr()
            matrix.sum_duplicates(); return matrix
        return self._view("mass_host", produce)

    @property
    def node_of_dof_host(self):
        return self._view("node_of_dof_host", lambda: self.node_of_dof.numpy().astype(np.int64))

    @property
    def owners_host(self):
        def produce():
            starts = self.owner_starts.numpy(); ids = self.owners.numpy()
            return [ids[starts[n]:starts[n+1]].tolist() for n in range(self.node_count)]
        return self._view("owners_host", produce)

    @property
    def node_map(self):
        return self._view("node_map", lambda: {tuple(int(v) for v in key): index for index, key in enumerate(self.nodes_host)})

    def canonical(self, key):
        result = list(key)
        for axis in range(3):
            if self.periodic[axis]:
                result[axis+1] %= self.cfg.grid.shape[axis]
        return tuple(result)

    def node_index(self, key):
        """Node of a face key after periodic wrapping, or -1 if absent."""
        return self.node_map.get(self.canonical(key), -1)

    def upload(self, values, dtype=wp.float32):
        return wp.array(np.ascontiguousarray(values), dtype=dtype, device=self.device)

    def read(self, state):
        result = wp.empty(self.node_count, dtype=wp.float32, device=self.device)
        if self.node_count:
            wp.launch(_gather_velocity, dim=result.size,
                      inputs=[self.nodes, state.u, state.v, state.w, result, self.cfg.grid.ng], device=self.device)
        return result

    def write(self, state, values):
        if values.size:
            wp.launch(_scatter_velocity, dim=values.size,
                      inputs=[self.nodes, values, state.u, state.v, state.w, self.cfg.grid.ng], device=self.device)
        fill_boundaries(state)
        enforce_voxel_solids(state)

    def quadratic(self, values):
        """values^T M values for a float32 node vector, on the device."""
        if not self.node_count:
            return 0.0
        doubled = wp.empty(self.node_count, dtype=wp.float64, device=self.device)
        wp.launch(_cast_to_double, dim=self.node_count, device=self.device, inputs=[values, doubled])
        product = wp.zeros_like(doubled)
        bsr_mv(self.mass, doubled, product)
        return float(wp.utils.array_inner(doubled, product))


def liquid_viscosity_operator(state, pressure=None):
    """The viscosity operator for one liquid geometry, built once per pressure operator.

    The operator depends only on the pressure operator's geometry, so the
    diffusion pass, the SGS update, the viscous time-step cap and the wall
    stress metric share one construction while that geometry is current.
    A cached metric borrows its owning pressure operator; a standalone metric
    owns its pressure geometry. Neither path creates an ownership cycle.
    """
    if pressure is None:
        return LiquidViscosityOperator(state)
    cached = getattr(pressure, "_viscosity_operator", None)
    if cached is None:
        cached = LiquidViscosityOperator(state, pressure=weakref.proxy(pressure))
        pressure._viscosity_operator = cached
    return cached


_STENCIL_OFFSETS = ((0, 0), (0, -1), (0, 1), (1, -1), (1, 1), (2, -1), (2, 1))


def _stencil_weight_table(patterns, spacing):
    """Least-squares derivative weights per (component, kept-slot) pattern, (P, 14, 3)."""
    table = np.zeros((len(patterns), 14, 3), dtype=np.float32)
    for row, value in enumerate(patterns):
        component = int(value) >> 14
        slots = np.flatnonzero((int(value) >> np.arange(14)) & 1)
        if not len(slots):
            continue
        positions = np.zeros((len(slots), 3))
        for k, slot in enumerate(slots):
            face_slot, offset_slot = divmod(int(slot), 7)
            positions[k, component] += face_slot
            direction, shift = _STENCIL_OFFSETS[offset_slot]
            positions[k, direction] += shift
        positions -= positions.mean(axis=0)
        X = np.column_stack((np.ones(len(positions)), positions))
        derivative = np.linalg.pinv(X)[1:] / spacing[:, None]
        table[row, slots] = derivative.T
    return table


class LiquidViscosityOperator(LiquidVelocityMetric):
    """One fixed liquid geometry for molecular and frozen-SGS midpoint passes."""

    def __init__(self, state, *, pressure=None):
        super().__init__(state, pressure=pressure)
        pressure = self.pressure
        g = state.cfg.grid
        device = self.device
        count, node_count = pressure.cell_count, self.node_count
        shape, periodic = self._shape, self._periodic_vec
        cell_table = wp.full(tuple(g.shape), -1, dtype=wp.int32, device=device)
        if count:
            wp.launch(_cell_table, dim=count, device=device, inputs=[self.cells, cell_table])
        local = pressure._local
        shared_high = wp.zeros((count, 3), dtype=wp.float64, device=device)
        if local is not None and count:
            wp.launch(_shared_high_areas, dim=count, device=device, inputs=[
                local.facet_count, local.facet_kind, local.facet_face, local.facet_area, shared_high])
        elif count:
            shared_high = self.upload(pressure.shared_high_areas(), wp.float64)
        # Edges from each node to its neighbour node along +direction.
        slots = 3 * max(node_count, 1)
        kept = wp.zeros(slots, dtype=wp.int32, device=device)
        neighbour = wp.full(slots, -1, dtype=wp.int32, device=device)
        conductance = wp.zeros(slots, dtype=wp.float64, device=device)
        if node_count:
            wp.launch(_edge_candidates, dim=node_count, device=device, inputs=[
                self.nodes, self.node_table, shape, periodic, cell_table, self.owner_starts, self.owners,
                self.cells, pressure._volume_d, shared_high, wp.vec3d(*self.spacing), kept, neighbour, conductance])
        edge_offsets, edge_count = exclusive_scan(kept, device)
        self.edges = wp.zeros(max(edge_count, 1), dtype=wp.vec2i, device=device)
        self.geometry = wp.zeros(max(edge_count, 1), dtype=wp.float32, device=device)
        self.edge_of_slot = wp.full(slots, -1, dtype=wp.int32, device=device)
        self.neighbour = neighbour
        self.rows = wp.zeros(4 * max(edge_count, 1), dtype=wp.int32, device=device)
        self.columns = wp.zeros_like(self.rows)
        self.edge_count = edge_count
        if node_count:
            wp.launch(_compact_edges, dim=slots, device=device, inputs=[
                kept, edge_offsets, neighbour, conductance, self.edges, self.geometry, self.edge_of_slot, self.rows, self.columns])
        # Affine least-squares strain stencils per (cell, component).
        pattern = wp.zeros((count, 3), dtype=wp.int32, device=device)
        kept_count = wp.zeros((count, 3), dtype=wp.int32, device=device)
        samples_table = wp.full((count, 3, 14), -1, dtype=wp.int32, device=device)
        if count:
            wp.launch(_strain_stencils, dim=count, device=device, inputs=[
                pressure._widths_d, self.node_of_dof, pressure._dof_axis_d, pressure._dof_sign_d,
                pressure._dof_position_d, self.node_table, shape, periodic, self.edge_of_slot, neighbour,
                pattern, kept_count, samples_table])
        pattern_host = pattern.numpy().reshape(-1) if count else np.zeros(0, dtype=np.int32)
        unique_patterns, inverse = np.unique(pattern_host, return_inverse=True)
        weight_table = self.upload(_stencil_weight_table(unique_patterns, self.spacing).reshape(len(unique_patterns), 14, 3), wp.vec3).reshape((max(len(unique_patterns), 1), 14)) if len(unique_patterns) else wp.zeros((1, 14), dtype=wp.vec3, device=device)
        pattern_row = self.upload(inverse.astype(np.int32).reshape(count, 3), wp.int32) if count else wp.zeros((1, 3), dtype=wp.int32, device=device)
        starts, sample_total = exclusive_scan(kept_count.reshape(3 * count) if count else wp.zeros(0, dtype=wp.int32, device=device), device)
        self.strain_starts = starts
        self.samples = wp.zeros(max(sample_total, 1), dtype=wp.int32, device=device)
        self.weights = wp.zeros(max(sample_total, 1), dtype=wp.vec3, device=device)
        if count:
            wp.launch(_fill_stencils, dim=count, device=device, inputs=[
                pattern, samples_table, starts, pattern_row, weight_table, self.samples, self.weights])
    @property
    def edges_host(self):
        return self._view("edges_host", lambda: self.edges.numpy()[:self.edge_count].astype(np.int32).reshape(-1, 2))

    def update_sgs(self, state, settings, strain, viscosity):
        strain.zero_(); viscosity.zero_()
        maximum = wp.zeros(1, dtype=wp.float32, device=self.device)
        wp.launch(_liquid_strain, dim=self.pressure.cell_count,
                  inputs=[self.strain_starts, self.samples, self.weights, self.cells, self.read(state),
                          strain, viscosity, maximum, (settings.smagorinsky_cs*self.volume**(1/3))**2,
                          settings.maximum_sgs_viscosity, self.cfg.grid.ng], device=self.device)
        return float(maximum.numpy()[0])

    def diffuse(self, state, dt, base_viscosity, cell_viscosity, dissipation):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        dissipation.zero_()
        before = self.read(state)
        if before.size == 0:
            return LiquidDiffusionResult(0, 0, 0, 0)
        coefficients = wp.zeros(max(self.edge_count, 1), dtype=wp.float32, device=self.device)
        entries = wp.zeros(4 * max(self.edge_count, 1), dtype=wp.float64, device=self.device)
        if self.edge_count:
            wp.launch(_edge_coefficients, dim=self.edge_count,
                      inputs=[self.edges, self.geometry, self.owner_starts, self.owners, self.cells,
                              cell_viscosity, coefficients, entries, base_viscosity, self.cfg.grid.ng], device=self.device)
        stiffness = bsr_from_triplets(before.size, before.size, self.rows, self.columns, entries)
        # bsr_axpy accumulates in place into its second argument; keep the metric
        # intact by accumulating into the per-call stiffness matrix instead.
        system = bsr_axpy(self.mass, stiffness, alpha=1.0, beta=.5*dt)
        row_sum = self.mass_row_sum
        rhs, delta = (wp.zeros(before.size, dtype=wp.float64, device=self.device) for _ in range(2))
        after = wp.zeros_like(before)
        if self.edge_count:
            wp.launch(_edge_rhs, dim=self.edge_count, inputs=[self.edges, coefficients, before, rhs, dt], device=self.device)

        def matvec(x, y, z, alpha, beta):
            # A x = (M 1)*x + sum_j A_ij*(x_j-x_i), since K 1 = 0.
            # Preserve M 1 before rounding: storing a large diagonal and its
            # cancelling off-diagonals separately in FP32 can erase a film's
            # positive inertia. This is the same quadratic form, not a shift.
            wp.launch(_difference_matvec, dim=before.size, inputs=[system.offsets, system.columns,
                      system.values, row_sum, x, y, z, alpha, beta], device=self.device)

        operator = LinearOperator(system.shape, wp.float64, self.device, matvec)
        inverse = self.line_preconditioner(system)
        iterations, _, tolerance = cg(operator, rhs, delta, M=inverse,
                                             tol=2e-7, atol=0.0, maxiter=2000, check_every=1)
        # Accept the actual equation residual; recursive CG convergence can
        # conceal a much larger residual in an ill-conditioned thin film.
        residual_values = wp.empty_like(rhs)
        matvec(delta, rhs, residual_values, -1.0, 1.0)
        residual = float(np.sqrt(wp.utils.array_inner(residual_values, residual_values)))
        if not math.isfinite(residual) or residual > tolerance:
            raise RuntimeError(f"liquid diffusion failed after {iterations} iterations: "
                               f"actual residual {residual} > {tolerance}")
        wp.launch(_add_increment, dim=before.size, inputs=[before, delta, after], device=self.device)
        self.write(state, after)
        actual = self.read(state)
        work = wp.zeros(1, dtype=wp.float64, device=self.device)
        if self.edge_count:
            wp.launch(_edge_work, dim=self.edge_count,
                      inputs=[self.edges, coefficients, before, actual, self.owner_starts, self.owners,
                              self.cells, dissipation, work, self.cfg.physics.rho, dt, 1/self.volume,
                              self.cfg.grid.ng], device=self.device)
        initial = .5*self.cfg.physics.rho*self.quadratic(before)
        final = .5*self.cfg.physics.rho*self.quadratic(actual)
        if final-initial > 64*np.finfo(np.float32).eps*max(initial, final):
            raise FloatingPointError(f"liquid diffusion injected {final-initial:.6g} J")
        physical = float(work.numpy()[0])
        rhs_norm = float(np.sqrt(wp.utils.array_inner(rhs, rhs)))
        return LiquidDiffusionResult(physical, initial-final-physical, iterations,
                                     residual/rhs_norm if rhs_norm else 0.0)
