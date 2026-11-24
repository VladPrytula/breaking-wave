"""Conservative unsplit PLIC transport for a liquid-divergence projection.

All directional fluxes use the same alpha and interface geometry. Their sum
updates retained volume directly, without a whole-open-cell dilation source.
Overlapping swept slabs can overdraw a thin donor. Shared fluxes are reduced
against donor volume and recipient capacity until the simultaneous update is
bounded. The reduction is measured; no cell-volume repair or redistribution
is used. Momentum consumes these same limited fluxes.
"""

from dataclasses import dataclass
import math

import warp as wp

from shoreloop.config import BC
from shoreloop.physics3d.advection import (
    VOFAdvectionResult, VOFAdvector, _finalize_directional_vof,
)
from shoreloop.physics3d.boundary import fill_field_boundaries
from shoreloop.physics3d.extrapolation import _wrapped_cell_index
from shoreloop.physics3d.interface import estimate_interface_normals
from shoreloop.physics3d.plic import reconstruct_plic
from shoreloop.physics3d.vof import measure_vof
from shoreloop.physics3d.vof_flux import compute_directional_liquid_fluxes
from shoreloop.physics3d.swept_prism import compute_swept_prism_fluxes

_SixFlux = wp.types.vector(length=6, dtype=wp.float32)


@dataclass(frozen=True)
class UnsplitVOFAdvectionResult(VOFAdvectionResult):
    limited_flux_volume: float
    limiter_iterations: int


@wp.kernel
def _capacity_factors(
    alpha: wp.array3d(dtype=wp.float32), theta: wp.array3d(dtype=wp.float32),
    flux_x: wp.array3d(dtype=wp.float32), flux_y: wp.array3d(dtype=wp.float32),
    flux_z: wp.array3d(dtype=wp.float32), outgoing_factor: wp.array3d(dtype=wp.float32),
    incoming_factor: wp.array3d(dtype=wp.float32), change: wp.array3d(dtype=wp.float32),
    maximum: wp.array(dtype=wp.float32), ng: int,
):
    i, j, k = wp.tid()
    I, J, K = i+ng, j+ng, k+ng
    capacity = theta[I, J, K]
    if capacity <= wp.float32(0.0):
        return
    faces = _SixFlux(
        -flux_x[I, J, K], flux_x[I+1, J, K],
        -flux_y[I, J, K], flux_y[I, J+1, K],
        -flux_z[I, J, K], flux_z[I, J, K+1],
    )
    incoming, outgoing = wp.float32(0.0), wp.float32(0.0)
    for face in range(6):
        outgoing += wp.max(faces[face], wp.float32(0.0))
        incoming += wp.max(-faces[face], wp.float32(0.0))
    water = alpha[I, J, K]*capacity
    change[I, J, K] = incoming-outgoing
    raw = (water+incoming-outgoing)/capacity
    violation = wp.max(wp.max(-raw, raw-wp.float32(1.0)), wp.float32(0.0))
    wp.atomic_max(maximum, 0, violation)
    outgoing_factor[I, J, K] = wp.float32(1.0)
    incoming_factor[I, J, K] = wp.float32(1.0)
    if outgoing > wp.float32(0.0):
        outgoing_factor[I, J, K] = wp.min(wp.float32(1.0), (water+incoming)/outgoing)
    if incoming > wp.float32(0.0):
        incoming_factor[I, J, K] = wp.min(wp.float32(1.0), (capacity-water+outgoing)/incoming)


@wp.kernel
def _limit_shared_flux(
    flux: wp.array3d(dtype=wp.float32),
    outgoing: wp.array3d(dtype=wp.float32), incoming: wp.array3d(dtype=wp.float32),
    reduction: wp.array(dtype=wp.float64), axis: int, shape: wp.vec3i,
    periodic: int, ng: int,
):
    i, j, k = wp.tid()
    position = wp.vec3i(i, j, k)
    low = position
    low[axis] = _wrapped_cell_index(position[axis]-1, shape[axis], periodic)
    high = position
    high[axis] = _wrapped_cell_index(position[axis], shape[axis], periodic)
    value = flux[i+ng, j+ng, k+ng]
    donor, receiver = low, high
    if value < wp.float32(0.0):
        donor, receiver = high, low
    factor = wp.float32(1.0)
    if donor[axis] >= 0:
        factor = wp.min(factor, outgoing[donor[0]+ng, donor[1]+ng, donor[2]+ng])
    if receiver[axis] >= 0:
        factor = wp.min(factor, incoming[receiver[0]+ng, receiver[1]+ng, receiver[2]+ng])
    limited = value*factor
    flux[i+ng, j+ng, k+ng] = limited
    if periodic == 0 or position[axis] < shape[axis]:
        wp.atomic_add(reduction, 0, wp.float64(wp.abs(value)-wp.abs(limited)))


class UnsplitVOFAdvector(VOFAdvector):
    """Shared geometric flux update without an open-divergence source."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self._outgoing_factor = wp.zeros_like(self._reference)
        self._incoming_factor = wp.zeros_like(self._reference)
        self._maximum_violation = wp.zeros(1, dtype=wp.float32, device=self.device)
        self._limited = wp.zeros(1, dtype=wp.float64, device=self.device)

    def advect(self, state, dt, *, transport_velocity=None, reconstruction=None):
        if state.cfg.grid != self.cfg.grid or state.device != self.device:
            raise ValueError("state grid and device must match the VOF advector")
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        before = measure_vof(state)
        if before.alpha_min < 0 or before.alpha_max > 1:
            raise ValueError("alpha must be bounded before advection")
        self._copy(state.alpha, self._reference)
        if transport_velocity is None:
            for name in self._FIELDS:
                fill_field_boundaries(state, name)
            transport_velocity = tuple(state.field(q) for q in self._FIELDS)
        if len(transport_velocity) != 3:
            raise ValueError("transport_velocity must contain u, v, and w")
        # A supplied reconstruction belongs to alpha at the start of this
        # unsplit update. All three directional fluxes consume that same state.
        if reconstruction is None:
            reconstruction = reconstruct_plic(state, estimate_interface_normals(state).normals)
        if self.cfg.vof_flux_reconstruction == "swept_prism":
            directional = compute_swept_prism_fluxes(state, reconstruction, dt, transport_velocity)
        else:
            directional = tuple(compute_directional_liquid_fluxes(
                state, reconstruction, axis, dt, velocity=transport_velocity[axis],
            ) for axis in range(3))
        g = self.cfg.grid
        self._limited.zero_()
        iterations = 0
        # Reducing one face can tighten another cell's budget; iterate the
        # simultaneous shared-flux constraints. Failure is explicit, never a
        # fallback to clipping water. The tolerance is float32 roundoff in alpha.
        for iteration in range(2*sum(g.shape)+1):
            self._maximum_violation.zero_()
            self._change.zero_()
            wp.launch(_capacity_factors, dim=g.shape, inputs=[state.alpha, state.theta,
                      *(d.fluxes for d in directional), self._outgoing_factor,
                      self._incoming_factor, self._change, self._maximum_violation, g.ng],
                      device=self.device)
            if float(self._maximum_violation.numpy()[0]) <= 5e-7:
                break
            for axis, item in enumerate(directional):
                periodic = int(getattr(self.cfg.bc, "xyz"[axis]+"lo") == BC.PERIODIC)
                wp.launch(_limit_shared_flux, dim=state.grid.interior_shape(self._FIELDS[axis]),
                          inputs=[item.fluxes, self._outgoing_factor, self._incoming_factor,
                                  self._limited, axis, wp.vec3i(*g.shape), periodic, g.ng],
                          device=self.device)
            iterations += 1
        else:
            raise RuntimeError("unsplit flux capacity iteration did not reach float32 boundedness")
        self._updated.zero_()
        self._redistributed.zero_()
        self._clipped.zero_()
        self._invalid.zero_()
        wp.launch(_finalize_directional_vof, dim=g.shape,
                  inputs=[state.alpha, state.theta, state.solid, self._change,
                          self._redistributed, self._updated, self._clipped, self._invalid,
                          g.dx*g.dy*g.dz, g.ng], device=self.device)
        if int(self._invalid.numpy()[0]):
            raise FloatingPointError("non-finite unsplit water update")
        self._copy(self._updated, state.alpha)
        fill_field_boundaries(state, "alpha")
        after = measure_vof(state)
        clipped = float(self._clipped.numpy()[0])
        self.steps += 1
        self.total_clipped_volume += clipped
        return UnsplitVOFAdvectionResult(
            sweep_order=("x", "y", "z"), volume_before=before.volume,
            volume_after=after.volume, volume_drift=after.volume-before.volume,
            clipped_volume=clipped, cumulative_clipped_volume=self.total_clipped_volume,
            max_courant=max(d.max_courant for d in directional), directional_fluxes=directional,
            limited_flux_volume=float(self._limited.numpy()[0])*g.dx*g.dy*g.dz,
            limiter_iterations=iterations,
        )
