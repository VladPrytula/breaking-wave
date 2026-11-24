"""Navier wall force paired with the retained-liquid kinetic metric.

For cell velocity B*u and the existing wall normal/rate, let
F=sqrt(V*rate)*(I-n*n.T)*B. The wall force is -F.T*F*u, so its work
is nonpositive in the shared kinetic metric M. Backward Euler solves
(M+dt*F.T*F)*delta=-dt*F.T*F*u. Physical work uses the final velocity;
implicit-time, solve and state-write remainders stay in numerical work.
The coupled system/force work uses CUDA FP64 with a separately retained
residual correction; carried fields and wall parameters remain FP32.
"""

import math

import numpy as np
import warp as wp
from warp.optim.linear import cg, LinearOperator
from warp.sparse import bsr_axpy, bsr_from_triplets, bsr_mm, bsr_mv, bsr_transposed

from shoreloop.config import BC
from shoreloop.physics3d.liquid_viscosity import LiquidVelocityMetric, _difference_matvec
from shoreloop.physics3d.wall_stress import WallStressResult, _cell_centre_velocity, _wall_geometry


@wp.kernel
def _wall_parameters(
    cells: wp.array(dtype=wp.vec3i), u: wp.array3d(dtype=wp.float32),
    v: wp.array3d(dtype=wp.float32), w: wp.array3d(dtype=wp.float32),
    theta: wp.array3d(dtype=wp.float32),
    material: wp.array3d(dtype=wp.int32), parameters: wp.array(dtype=wp.vec4),
    viscosity: wp.float32, beach_slip: wp.float32, boulder_slip: wp.float32,
    law: int, roughness: wp.float32,
    inverse_spacing: wp.vec3, distance: wp.float32,
    shape: wp.vec3i, periodic: wp.vec3i, ng: int,
):
    c = wp.tid()
    p = cells[c]
    velocity = _cell_centre_velocity(u, v, w, p[0]+ng, p[1]+ng, p[2]+ng)
    parameters[c] = _wall_geometry(theta, material, p[0], p[1], p[2], velocity, viscosity,
        beach_slip, boulder_slip, law, roughness,
        inverse_spacing[0], inverse_spacing[1], inverse_spacing[2],
        distance, shape[0], shape[1], shape[2], periodic[0], periodic[1], periodic[2], ng)


@wp.kernel
def _wall_work(cells: wp.array(dtype=wp.vec3i), tangent: wp.array(dtype=wp.float64),
               field: wp.array3d(dtype=wp.float32), total: wp.array(dtype=wp.float64),
               rho: wp.float64, dt: wp.float64, inverse_volume: wp.float64, ng: int):
    c = wp.tid()
    squared = wp.float64(0.0)
    for axis in range(3):
        value = wp.float64(tangent[3*c+axis])
        squared += value*value
    p = cells[c]
    field[p[0]+ng, p[1]+ng, p[2]+ng] = wp.float32(rho*squared*inverse_volume)
    wp.atomic_add(total, 0, rho*dt*squared)


@wp.kernel
def _accumulate_wall_correction(low: wp.array(dtype=wp.float64),
                                correction: wp.array(dtype=wp.float64)):
    i = wp.tid()
    low[i] += correction[i]


@wp.kernel
def _write_wall_velocity(before: wp.array(dtype=wp.float32),
                         high: wp.array(dtype=wp.float64), low: wp.array(dtype=wp.float64),
                         after: wp.array(dtype=wp.float32)):
    i = wp.tid()
    after[i] = wp.float32(wp.float64(before[i])+high[i]+low[i])


def _implicit_wall_increment(metric, factor, before, dt):
    """CUDA FP64 backward Euler with the actual equation residual.

    A nearly uniform increment can require differences smaller than one FP64
    increment rounding unit. Keep refinement separately until the final FP32
    state write, rather than losing it when adding it to the first solution.
    """
    mass, row_sum = metric.mass, metric.mass_row_sum
    transpose = bsr_transposed(factor)
    before64 = wp.empty(before.size, dtype=wp.float64, device=metric.device)
    wp.utils.array_cast(before, before64)
    work = wp.zeros(factor.nrow, dtype=wp.float64, device=metric.device)
    rhs, delta = wp.zeros_like(before64), wp.zeros_like(before64)
    after = wp.zeros_like(before)
    bsr_mv(factor, before64, work)
    bsr_mv(transpose, work, rhs, alpha=-dt)

    def matvec(x, y, z, alpha, beta):
        bsr_mv(factor, x, work)
        wp.launch(_difference_matvec, dim=before.size, inputs=[mass.offsets,
            mass.columns, mass.values, row_sum,
            x, y, z, alpha, beta], device=metric.device)
        bsr_mv(transpose, work, z, alpha=alpha*dt, beta=1.0)

    # Principal component lines of M + dt F^T F precondition the solve; all
    # other couplings remain in the full matrix action above.
    system = bsr_axpy(mass, bsr_mm(transpose, factor), alpha=1.0, beta=dt)
    inverse = metric.line_preconditioner(system)
    operator = LinearOperator(system.shape, wp.float64, metric.device, matvec)
    iterations, recursive, target = cg(operator, rhs, delta, M=inverse,
        tol=2e-7, atol=0.0, maxiter=2000, check_every=1)
    # Both parts solve the same equation within the original total CG budget.
    # Recompute its residual instead of trusting the recurrence.
    low, correction, remainder = (wp.zeros_like(rhs) for _ in range(3))
    while True:
        matvec(delta, rhs, remainder, -1.0, 1.0)
        matvec(low, remainder, remainder, -1.0, 1.0)
        residual = float(np.sqrt(wp.utils.array_inner(remainder, remainder)))
        if math.isfinite(residual) and residual <= target:
            break
        if not math.isfinite(residual) or iterations >= 2000:
            raise RuntimeError(f"liquid wall stress failed after {iterations} iterations: "
                               f"actual residual {residual} > {target}; recursive {recursive}")
        correction.zero_()
        steps, recursive, _ = cg(operator, remainder, correction, M=inverse,
            tol=2e-7, atol=target, maxiter=2000-iterations, check_every=1)
        iterations += steps
        if steps == 0:
            raise RuntimeError(f"wall correction made no progress: {residual} > {target}")
        wp.launch(_accumulate_wall_correction, dim=before.size,
                  inputs=[low, correction], device=metric.device)
    wp.launch(_write_wall_velocity, dim=before.size,
              inputs=[before, delta, low, after], device=metric.device)
    rhs_norm = float(np.sqrt(wp.utils.array_inner(rhs, rhs)))
    return after, factor, iterations, residual/rhs_norm if rhs_norm else 0.0


def apply_liquid_wall_stress(model, state, dt, *, pressure=None):
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite")
    model.wall_delta.zero_()
    model.dissipation.zero_()
    # The viscosity operator is a velocity metric on the same geometry; reuse it
    # when the step has already built it for this pressure operator.
    metric = (getattr(pressure, "_viscosity_operator", None) if pressure is not None else None)
    if metric is None:
        metric = LiquidVelocityMetric(state, pressure=pressure)
    if metric.pressure.cell_count == 0:
        return WallStressResult(0, 0, 0)
    g, settings = state.cfg.grid, model.settings
    parameters = wp.empty(metric.pressure.cell_count, dtype=wp.vec4, device=state.device)
    periodic = tuple(int(getattr(state.cfg.bc, a+"lo") == BC.PERIODIC) for a in "xyz")
    wp.launch(_wall_parameters, dim=parameters.size, inputs=[metric.cells, state.u, state.v, state.w,
        state.theta, state.material, parameters, settings.wall_viscosity, settings.beach_slip_length,
        settings.boulder_slip_length, settings.law_code, settings.roughness_length,
        wp.vec3(*(1/metric.spacing)), .5*min(metric.spacing),
        wp.vec3i(*g.shape), wp.vec3i(*periodic), g.ng], device=state.device)
    geometry = parameters.numpy().astype(float)
    active = np.flatnonzero(geometry[:, 3] > 0)
    if not len(active):
        return WallStressResult(0, 0, 0)
    pressure = metric.pressure
    # Tangential-velocity factor of every active cell: the cell-centre average
    # of its liquid MAC faces (each component from its own faces, weight .5),
    # projected onto the wall tangent plane and scaled by sqrt(V * geometry).
    widths = pressure._widths[active]
    slots = np.arange(pressure.nmax)[None, :]
    column = slots < widths[:, None]
    face_axis = pressure._dofs[active][:, :, 0]
    nodes = metric.node_of_dof_host[active]
    normal = geometry[active, :3] / np.linalg.norm(geometry[active, :3], axis=1, keepdims=True)
    tangent = np.eye(3)[None] - normal[:, :, None] * normal[:, None, :]
    scale = np.sqrt(pressure._volumes[active] * geometry[active, 3])
    # local[cell, axis, slot] = scale * tangent[axis, face_axis] * .5 on kept slots
    local = .5 * scale[:, None, None] * np.take_along_axis(
        np.broadcast_to(tangent, (len(active), 3, 3)), np.broadcast_to(face_axis[:, None, :], (len(active), 3, pressure.nmax)).clip(0, 2), axis=2)
    keep = np.broadcast_to(column[:, None, :], local.shape)
    rows = (3 * np.arange(len(active))[:, None, None] + np.arange(3)[None, :, None])
    rows = np.broadcast_to(rows, local.shape)[keep]
    columns = np.broadcast_to(nodes[:, None, :], local.shape)[keep]
    factor = bsr_from_triplets(3 * len(active), metric.node_count,
        metric.upload(rows.astype(np.int32), wp.int32), metric.upload(columns.astype(np.int32), wp.int32),
        metric.upload(local[keep], wp.float64))
    before = metric.read(state)
    after, factor, iterations, residual = _implicit_wall_increment(metric, factor, before, dt)
    metric.write(state, after)
    actual = metric.read(state)
    tangent = wp.empty(3*len(active), dtype=wp.float64, device=state.device)
    actual64 = wp.empty(actual.size, dtype=wp.float64, device=state.device)
    wp.utils.array_cast(actual, actual64)
    bsr_mv(factor, actual64, tangent)
    cells = metric.upload(metric.pressure._cell_index[active], wp.vec3i)
    work = wp.zeros(1, dtype=wp.float64, device=state.device)
    wp.launch(_wall_work, dim=len(active), inputs=[cells, tangent, model.dissipation, work,
        state.cfg.physics.rho, dt, 1/metric.volume, g.ng], device=state.device)
    initial = .5*state.cfg.physics.rho*metric.quadratic(before)
    final = .5*state.cfg.physics.rho*metric.quadratic(actual)
    if final-initial > 2e-6*max(initial, final)+1e-15:
        raise FloatingPointError(f"wall stress injected {final-initial:.6g} J in the liquid kinetic metric")
    physical = float(work.numpy()[0])
    return WallStressResult(physical, len(active), float(geometry[active, 3].max()),
        initial-final-physical, iterations, residual)
