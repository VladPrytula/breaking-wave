"""Volume-fitting interface normals in the neighbourhood of a cut solid.

The central retained volume fixes the plane offset. Its orientation minimises
the squared retained-volume errors in the 3x3x3 open-cell neighbourhood (an
LVIRA objective). Youngs and axis seeds precede a local direction search.
The fit changes geometry, never alpha or theta; curved/underresolved surfaces
retain a nonzero fitting error rather than receiving a prescribed orientation.

The search is a coordinate descent with step halving from the Youngs seed.
A lane ends when its perturbation falls below ``STEP_FLOOR``, and the active
lanes are compacted every few rounds so the trial launches shrink with them.
Seeding a lane from its previous fitted normal was measured and rejected
(decisions 2026-10-13): the objective has several minima for a few percent
of cells and the seed decides which one is reached.
"""

import numpy as np
import warp as wp

from shoreloop.physics3d.extrapolation import _wrapped_cell_index
from shoreloop.physics3d.plic import (
    _solve_cut_plane_offset, _solve_plane_offset_l1, _unit_cube_volume_l1,
)
from shoreloop.physics3d.polyhedron import clipped_box_moments

# Lanes perturb by a quarter of a unit vector and halve on failure. A
# perturbation below STEP_FLOOR moves the normal by less than 0.006 deg, below
# the float32 resolution of the objective: it ends the lane (measured
# 2026-10-13 on the 6.8 s production state: the 1e-4 floor keeps every fitted
# normal within 0.13 deg, p99 0.02 deg, of the former 1e-7 floor's result and
# cuts the fit from 17.2 s to 12.2 s with the compaction).
INITIAL_STEP = 0.25
STEP_FLOOR = 1.0e-4
COMPACTION_ROUNDS = 4


@wp.func
def _plane_offset(
    n: wp.vec3,
    alpha: wp.array3d(dtype=wp.float32),
    theta: wp.array3d(dtype=wp.float32),
    open_normals: wp.array3d(dtype=wp.vec3),
    open_offsets: wp.array3d(dtype=wp.float32),
    I: int, J: int, K: int,
) -> wp.float32:
    """Offset of the L1-normalised plane reproducing the central retained volume."""
    d = _solve_plane_offset_l1(n, alpha[I, J, K])
    if theta[I, J, K] < wp.float32(1.0):
        d = _solve_cut_plane_offset(n, alpha[I, J, K] * theta[I, J, K],
                                   open_normals[I, J, K], open_offsets[I, J, K])
    return d


@wp.func
def _neighbour_residual(
    n: wp.vec3, d: wp.float32,
    alpha: wp.array3d(dtype=wp.float32),
    theta: wp.array3d(dtype=wp.float32),
    open_normals: wp.array3d(dtype=wp.vec3),
    open_offsets: wp.array3d(dtype=wp.float32),
    i: int, j: int, k: int, q: int,
    shape: wp.vec3i, periodic: wp.vec3i, ng: int,
) -> wp.float32:
    """Squared retained-volume error of neighbour ``q`` (0..26) under the plane."""
    di = q // 9 - 1
    dj = q // 3 % 3 - 1
    dk = q % 3 - 1
    a = _wrapped_cell_index(i + di, shape[0], periodic[0])
    b = _wrapped_cell_index(j + dj, shape[1], periodic[1])
    c = _wrapped_cell_index(k + dk, shape[2], periodic[2])
    if a < 0 or b < 0 or c < 0:
        return wp.float32(0.0)
    A = a + ng
    B = b + ng
    C = c + ng
    capacity = theta[A, B, C]
    if capacity <= wp.float32(0.0):
        return wp.float32(0.0)
    offset = d - wp.dot(n, wp.vec3(wp.float32(di), wp.float32(dj), wp.float32(dk)))
    volume = _unit_cube_volume_l1(n, offset)
    if capacity < wp.float32(1.0):
        volume = clipped_box_moments(
            wp.vec3(wp.float32(0.0)), wp.vec3(wp.float32(1.0)),
            open_normals[A, B, C], open_offsets[A, B, C], n, offset,
        )[0]
    residual = volume - alpha[A, B, C] * capacity
    return residual * residual


@wp.func
def _volume_fit_error(
    candidate: wp.vec3,
    alpha: wp.array3d(dtype=wp.float32),
    theta: wp.array3d(dtype=wp.float32),
    open_normals: wp.array3d(dtype=wp.vec3),
    open_offsets: wp.array3d(dtype=wp.float32),
    i: int, j: int, k: int,
    shape: wp.vec3i, periodic: wp.vec3i, ng: int,
) -> wp.float32:
    n = candidate / (wp.abs(candidate[0]) + wp.abs(candidate[1]) + wp.abs(candidate[2]))
    d = _plane_offset(n, alpha, theta, open_normals, open_offsets, i + ng, j + ng, k + ng)
    error = wp.float32(0.0)
    neighbours = int(27)
    for q in range(neighbours):
        error += _neighbour_residual(n, d, alpha, theta, open_normals, open_offsets,
                                     i, j, k, q, shape, periodic, ng)
    return error


@wp.kernel
def _collect_cut_interface_cells(
    alpha: wp.array3d(dtype=wp.float32),
    theta: wp.array3d(dtype=wp.float32),
    cells: wp.array(dtype=int), count: wp.array(dtype=int),
    shape: wp.vec3i, periodic: wp.vec3i,
    tolerance: wp.float32, ng: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    fraction = alpha[I, J, K]
    if (fraction <= tolerance or fraction >= wp.float32(1.0) - tolerance
            or theta[I, J, K] <= wp.float32(0.0)):
        return
    near_cut = int(0)
    neighbours = int(27)
    for q in range(neighbours):
        a = _wrapped_cell_index(i + q // 9 - 1, shape[0], periodic[0])
        b = _wrapped_cell_index(j + q // 3 % 3 - 1, shape[1], periodic[1])
        c = _wrapped_cell_index(k + q % 3 - 1, shape[2], periodic[2])
        if a >= 0 and b >= 0 and c >= 0:
            capacity = theta[a + ng, b + ng, c + ng]
            if capacity > wp.float32(0.0) and capacity < wp.float32(1.0):
                near_cut = 1
    if near_cut == 0:
        return
    slot = wp.atomic_add(count, 0, 1)
    cells[slot] = (i * shape[1] + j) * shape[2] + k


@wp.func
def _candidate(best: wp.vec3, step: wp.float32, q: int, phase: int) -> wp.vec3:
    """Axis seed (phase 0) or a renormalised Cartesian perturbation of ``best``."""
    candidate = wp.vec3(wp.float32(0.0))
    if phase == 0:
        candidate[q // 2] = wp.float32(1 - 2 * (q % 2))
        return candidate
    candidate = best
    candidate[q // 2] += step * wp.float32(1 - 2 * (q % 2))
    return wp.normalize(candidate)


@wp.kernel
def _seed_errors(
    cells: wp.array(dtype=int),
    alpha: wp.array3d(dtype=wp.float32),
    theta: wp.array3d(dtype=wp.float32),
    open_normals: wp.array3d(dtype=wp.vec3),
    open_offsets: wp.array3d(dtype=wp.float32),
    normals: wp.array3d(dtype=wp.vec3),
    spacing: wp.vec3, shape: wp.vec3i, periodic: wp.vec3i, ng: int,
    best: wp.array(dtype=wp.vec3), error: wp.array(dtype=wp.float32),
):
    """Start every fit from its Youngs normal."""
    t = wp.tid()
    cell = cells[t]
    i = cell // (shape[1] * shape[2])
    j = cell // shape[2] % shape[1]
    k = cell % shape[2]
    seed = wp.normalize(wp.cw_mul(normals[i + ng, j + ng, k + ng], spacing))
    best[t] = seed
    error[t] = _volume_fit_error(seed, alpha, theta, open_normals, open_offsets,
                                 i, j, k, shape, periodic, ng)


@wp.func
def _lane_done(error: wp.float32, step: wp.float32, phase: int, floor: wp.float32) -> int:
    if phase == 1 and (error <= wp.float32(1.0e-12) or step < floor):
        return 1
    return 0


@wp.kernel
def _trial_offsets(
    lanes: wp.array(dtype=int), cells: wp.array(dtype=int),
    alpha: wp.array3d(dtype=wp.float32),
    theta: wp.array3d(dtype=wp.float32),
    open_normals: wp.array3d(dtype=wp.vec3),
    open_offsets: wp.array3d(dtype=wp.float32),
    shape: wp.vec3i, ng: int,
    best: wp.array(dtype=wp.vec3), error: wp.array(dtype=wp.float32),
    step: wp.array(dtype=wp.float32), phase: int, floor: wp.float32,
    offsets: wp.array2d(dtype=wp.float32),
):
    """One lane per (active cell, candidate): the candidate plane's offset, solved once."""
    l, q = wp.tid()
    t = lanes[l]
    if _lane_done(error[t], step[t], phase, floor) == 1:
        return
    cell = cells[t]
    i = cell // (shape[1] * shape[2])
    j = cell // shape[2] % shape[1]
    k = cell % shape[2]
    candidate = _candidate(best[t], step[t], q, phase)
    n = candidate / (wp.abs(candidate[0]) + wp.abs(candidate[1]) + wp.abs(candidate[2]))
    offsets[l, q] = _plane_offset(n, alpha, theta, open_normals, open_offsets, i + ng, j + ng, k + ng)


@wp.kernel
def _trial_residuals(
    lanes: wp.array(dtype=int), cells: wp.array(dtype=int),
    alpha: wp.array3d(dtype=wp.float32),
    theta: wp.array3d(dtype=wp.float32),
    open_normals: wp.array3d(dtype=wp.vec3),
    open_offsets: wp.array3d(dtype=wp.float32),
    shape: wp.vec3i, periodic: wp.vec3i, ng: int,
    best: wp.array(dtype=wp.vec3), error: wp.array(dtype=wp.float32),
    step: wp.array(dtype=wp.float32), phase: int, floor: wp.float32,
    offsets: wp.array2d(dtype=wp.float32),
    residuals: wp.array3d(dtype=wp.float32),
):
    """One lane per (active cell, candidate, neighbour): one clipped-volume residual."""
    l, q, m = wp.tid()
    t = lanes[l]
    if _lane_done(error[t], step[t], phase, floor) == 1:
        return
    cell = cells[t]
    i = cell // (shape[1] * shape[2])
    j = cell // shape[2] % shape[1]
    k = cell % shape[2]
    candidate = _candidate(best[t], step[t], q, phase)
    n = candidate / (wp.abs(candidate[0]) + wp.abs(candidate[1]) + wp.abs(candidate[2]))
    residuals[l, q, m] = _neighbour_residual(n, offsets[l, q], alpha, theta, open_normals, open_offsets,
                                             i, j, k, m, shape, periodic, ng)


@wp.kernel
def _select_trials(
    lanes: wp.array(dtype=int),
    best: wp.array(dtype=wp.vec3), error: wp.array(dtype=wp.float32),
    step: wp.array(dtype=wp.float32), phase: int, floor: wp.float32,
    residuals: wp.array3d(dtype=wp.float32),
):
    """Greedy acceptance in candidate order, as the sequential search, summing the
    neighbour residuals in the sequential order; halve the step when a
    perturbation round brings no improvement."""
    l = wp.tid()
    t = lanes[l]
    if _lane_done(error[t], step[t], phase, floor) == 1:
        return
    previous = error[t]
    origin = best[t]
    for q in range(6):
        trial = wp.float32(0.0)
        for m in range(27):
            trial += residuals[l, q, m]
        if trial < error[t]:
            best[t] = _candidate(origin, step[t], q, phase)
            error[t] = trial
    if phase == 1 and error[t] >= previous:
        step[t] *= wp.float32(0.5)


@wp.kernel
def _lane_flags(
    lanes: wp.array(dtype=int), error: wp.array(dtype=wp.float32),
    step: wp.array(dtype=wp.float32), floor: wp.float32, flags: wp.array(dtype=wp.int32),
):
    l = wp.tid()
    t = lanes[l]
    flags[l] = 1 - _lane_done(error[t], step[t], 1, floor)


@wp.kernel
def _compact_lanes(lanes: wp.array(dtype=int), flags: wp.array(dtype=wp.int32),
                   offsets: wp.array(dtype=wp.int32), out: wp.array(dtype=int)):
    l = wp.tid()
    if flags[l] == 1:
        out[offsets[l]] = lanes[l]


@wp.kernel
def _write_fitted_normals(
    cells: wp.array(dtype=int), best: wp.array(dtype=wp.vec3),
    normals: wp.array3d(dtype=wp.vec3), spacing: wp.vec3, shape: wp.vec3i, ng: int,
):
    t = wp.tid()
    cell = cells[t]
    i = cell // (shape[1] * shape[2])
    j = cell // shape[2] % shape[1]
    k = cell % shape[2]
    normals[i + ng, j + ng, k + ng] = wp.normalize(wp.cw_div(best[t], spacing))


@wp.func
def _stencil_value(alpha: wp.array3d(dtype=wp.float32), i: int, j: int, k: int, q: int,
                   shape: wp.vec3i, periodic: wp.vec3i, ng: int) -> wp.float32:
    a = _wrapped_cell_index(i + q // 9 - 1, shape[0], periodic[0])
    b = _wrapped_cell_index(j + q // 3 % 3 - 1, shape[1], periodic[1])
    c = _wrapped_cell_index(k + q % 3 - 1, shape[2], periodic[2])
    if a < 0 or b < 0 or c < 0:
        return wp.float32(-1.0)
    return alpha[a + ng, b + ng, c + ng]


@wp.kernel
def _lookup_cache(
    cells: wp.array(dtype=int), alpha: wp.array3d(dtype=wp.float32),
    slot: wp.array3d(dtype=wp.int32), stencil: wp.array2d(dtype=wp.float32),
    cached: wp.array(dtype=wp.vec3), normals: wp.array3d(dtype=wp.vec3),
    shape: wp.vec3i, periodic: wp.vec3i, ng: int, miss: wp.array(dtype=wp.int32),
):
    """Reuse a fitted normal when the 27-cell water stencil is bitwise unchanged."""
    t = wp.tid()
    cell = cells[t]
    i = cell // (shape[1] * shape[2])
    j = cell // shape[2] % shape[1]
    k = cell % shape[2]
    entry = slot[i, j, k]
    miss[t] = 1
    if entry < 0:
        return
    for q in range(27):
        if stencil[entry, q] != _stencil_value(alpha, i, j, k, q, shape, periodic, ng):
            return
    normals[i + ng, j + ng, k + ng] = cached[entry]
    miss[t] = 0


@wp.kernel
def _compact_misses(cells: wp.array(dtype=int), miss: wp.array(dtype=wp.int32),
                    offsets: wp.array(dtype=wp.int32), misses: wp.array(dtype=int)):
    t = wp.tid()
    if miss[t] == 1:
        misses[offsets[t]] = cells[t]


@wp.kernel
def _clear_slots(cells: wp.array(dtype=int), slot: wp.array3d(dtype=wp.int32), shape: wp.vec3i):
    t = wp.tid()
    cell = cells[t]
    slot[cell // (shape[1] * shape[2]), cell // shape[2] % shape[1], cell % shape[2]] = -1


@wp.kernel
def _store_cache(
    cells: wp.array(dtype=int), alpha: wp.array3d(dtype=wp.float32),
    normals: wp.array3d(dtype=wp.vec3), slot: wp.array3d(dtype=wp.int32),
    stencil: wp.array2d(dtype=wp.float32), cached: wp.array(dtype=wp.vec3),
    shape: wp.vec3i, periodic: wp.vec3i, ng: int,
):
    t = wp.tid()
    cell = cells[t]
    i = cell // (shape[1] * shape[2])
    j = cell // shape[2] % shape[1]
    k = cell % shape[2]
    slot[i, j, k] = t
    for q in range(27):
        stencil[t, q] = _stencil_value(alpha, i, j, k, q, shape, periodic, ng)
    cached[t] = normals[i + ng, j + ng, k + ng]


class CutNormalCache:
    """Fitted normals of solid-adjacent interface cells, keyed by their water stencil.

    The fit depends only on the 27-cell water fractions around a cell and on
    the static solid geometry, so a cell whose stencil is bitwise unchanged
    since its last fit keeps its normal. Between the operator constructions of
    one step most solid-adjacent cells are unchanged.
    """

    def __init__(self, shape, device):
        self.shape = tuple(int(v) for v in shape)
        self.device = device
        self.slot = wp.full(self.shape, -1, dtype=wp.int32, device=device)
        self.cells = wp.zeros(0, dtype=int, device=device)
        self.stencil = wp.zeros((0, 27), dtype=wp.float32, device=device)
        self.normal = wp.zeros(0, dtype=wp.vec3, device=device)
        self.hits = 0
        self.misses = 0


def fit_cut_interface_normals(alpha, theta, open_normals, open_offsets,
                              normals, spacing, shape, periodic, tolerance, ng,
                              *, mixed_cells, iterations=40, cache=None):
    """Fit the solid-adjacent interface cells with one lane per (cell, candidate, neighbour).

    ``mixed_cells`` is the Youngs pass's count at the same alpha/tolerance;
    solid-adjacent candidates are a subset. The search is the sequential
    LVIRA descent (Youngs seed, six axis seeds, then up to 40 rounds of six
    renormalised Cartesian perturbations with step halving), evaluated round
    by round across lanes; the selection sums the 27 residuals in the
    sequential order and rounds whose perturbation is below float32
    resolution are skipped, so the fitted normals equal the sequential
    search to rounding. With a ``CutNormalCache`` cells whose 27-cell water
    stencil is unchanged reuse their previous normal.
    """
    if mixed_cells == 0:
        return 0
    device = alpha.device
    candidates = wp.empty(mixed_cells, dtype=int, device=device)
    count = wp.zeros(1, dtype=int, device=device)
    wp.launch(_collect_cut_interface_cells, dim=tuple(shape),
              inputs=[alpha, theta, candidates, count, shape, periodic, tolerance, ng],
              device=device)
    fitted_cells = int(count.numpy()[0])
    if not fitted_cells:
        return 0
    candidates = candidates[:fitted_cells]
    cells = candidates
    if cache is not None and cache.normal.size:
        miss = wp.zeros(fitted_cells, dtype=wp.int32, device=device)
        wp.launch(_lookup_cache, dim=fitted_cells, device=device, inputs=[
            candidates, alpha, cache.slot, cache.stencil, cache.normal, normals, shape, periodic, ng, miss])
        inclusive = wp.zeros_like(miss)
        wp.utils.array_scan(miss, inclusive, inclusive=True)
        missing = int(inclusive.numpy()[-1])
        offsets = wp.zeros_like(miss)
        if fitted_cells > 1:
            wp.copy(offsets, inclusive, dest_offset=1, count=fitted_cells - 1)
        cells = wp.zeros(max(missing, 1), dtype=int, device=device)
        wp.launch(_compact_misses, dim=fitted_cells, device=device, inputs=[candidates, miss, offsets, cells])
        cache.hits += fitted_cells - missing
        cache.misses += missing
        cells = cells[:missing]
    if cells.size:
        _search(cells, alpha, theta, open_normals, open_offsets, normals, spacing, shape, periodic, ng, iterations)
    if cache is not None:
        if cache.cells.size:
            wp.launch(_clear_slots, dim=cache.cells.size, device=device, inputs=[cache.cells, cache.slot, shape])
        cache.cells = wp.clone(candidates)
        cache.stencil = wp.zeros((fitted_cells, 27), dtype=wp.float32, device=device)
        cache.normal = wp.zeros(fitted_cells, dtype=wp.vec3, device=device)
        wp.launch(_store_cache, dim=fitted_cells, device=device, inputs=[
            candidates, alpha, normals, cache.slot, cache.stencil, cache.normal, shape, periodic, ng])
    return fitted_cells


def _search(cells, alpha, theta, open_normals, open_offsets, normals, spacing, shape, periodic, ng, iterations):
    """Descend every lane from its Youngs seed; compact the active lanes as they finish."""
    device = alpha.device
    fitted_cells = cells.size
    best = wp.empty(fitted_cells, dtype=wp.vec3, device=device)
    error = wp.empty(fitted_cells, dtype=wp.float32, device=device)
    step = wp.full(fitted_cells, INITIAL_STEP, dtype=wp.float32, device=device)
    residuals = wp.zeros((fitted_cells, 6, 27), dtype=wp.float32, device=device)
    offsets = wp.zeros((fitted_cells, 6), dtype=wp.float32, device=device)
    lanes = wp.array(np.arange(fitted_cells, dtype=np.int32), dtype=int, device=device)
    floor = float(STEP_FLOOR)
    geometry = [alpha, theta, open_normals, open_offsets]
    wp.launch(_seed_errors, dim=fitted_cells, device=device,
              inputs=[cells, *geometry, normals, spacing, shape, periodic, ng, best, error])
    for phase, rounds in ((0, 1), (1, iterations)):
        for round_index in range(rounds):
            wp.launch(_trial_offsets, dim=(lanes.size, 6), device=device,
                      inputs=[lanes, cells, *geometry, shape, ng, best, error, step, phase, floor, offsets])
            wp.launch(_trial_residuals, dim=(lanes.size, 6, 27), device=device,
                      inputs=[lanes, cells, *geometry, shape, periodic, ng, best, error, step, phase, floor,
                              offsets, residuals])
            wp.launch(_select_trials, dim=lanes.size, device=device,
                      inputs=[lanes, best, error, step, phase, floor, residuals])
            if phase == 1 and round_index % COMPACTION_ROUNDS == COMPACTION_ROUNDS - 1:
                # Keep only the lanes still moving, so the next launches shrink.
                flags = wp.zeros(lanes.size, dtype=wp.int32, device=device)
                wp.launch(_lane_flags, dim=lanes.size, device=device, inputs=[lanes, error, step, floor, flags])
                inclusive = wp.zeros_like(flags)
                wp.utils.array_scan(flags, inclusive, inclusive=True)
                remaining = int(inclusive.numpy()[-1])
                if remaining == 0:
                    break
                exclusive = wp.zeros_like(flags)
                if lanes.size > 1:
                    wp.copy(exclusive, inclusive, dest_offset=1, count=lanes.size - 1)
                compacted = wp.zeros(remaining, dtype=int, device=device)
                wp.launch(_compact_lanes, dim=lanes.size, device=device, inputs=[lanes, flags, exclusive, compacted])
                lanes = compacted
    wp.launch(_write_fitted_normals, dim=fitted_cells, device=device,
              inputs=[cells, best, normals, spacing, shape, ng])
