"""Conservative directionally split PLIC advection of water fraction."""

from __future__ import annotations

from dataclasses import dataclass
import math

import warp as wp

from shoreloop.config import BC, Solver3DConfig
from shoreloop.physics3d.boundary import fill_field_boundaries
from shoreloop.physics3d.interface import estimate_interface_normals
from shoreloop.physics3d.extrapolation import _wrapped_cell_index
from shoreloop.physics3d.plic import reconstruct_plic
from shoreloop.physics3d.state import State3D
from shoreloop.physics3d.vof import measure_vof
from shoreloop.physics3d.vof_flux import (DirectionalFluxResult,
                                          compute_directional_liquid_fluxes)


@dataclass(frozen=True)
class VOFAdvectionResult:
    """Mass, boundedness, and sweep diagnostics for one accepted VOF step."""

    sweep_order: tuple[str, str, str]
    volume_before: float
    volume_after: float
    volume_drift: float
    clipped_volume: float
    cumulative_clipped_volume: float
    max_courant: float
    directional_fluxes: tuple[DirectionalFluxResult, DirectionalFluxResult,
                               DirectionalFluxResult]


@wp.kernel
def _copy_cell_field(
    source: wp.array3d(dtype=wp.float32),
    target: wp.array3d(dtype=wp.float32),
    ng: int,
):
    i, j, k = wp.tid()
    target[i + ng, j + ng, k + ng] = source[i + ng, j + ng, k + ng]


@wp.kernel
def _directional_mass_change(
    alpha: wp.array3d(dtype=wp.float32),
    solid: wp.array3d(dtype=wp.int32),
    reference_alpha: wp.array3d(dtype=wp.float32),
    fluxes: wp.array3d(dtype=wp.float32),
    velocity: wp.array3d(dtype=wp.float32),
    aperture: wp.array3d(dtype=wp.float32),
    change: wp.array3d(dtype=wp.float32),
    invalid: wp.array(dtype=wp.int32),
    axis: int,
    uniform_fraction: int,
    inverse_spacing_dt: wp.float32,
    ng: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    if solid[I, J, K] != 0:
        change[I, J, K] = wp.float32(0.0)
        return
    HI = I
    HJ = J
    HK = K
    if axis == 0:
        HI += 1
    elif axis == 1:
        HJ += 1
    else:
        HK += 1

    low_flux = fluxes[I, J, K]
    high_flux = fluxes[HI, HJ, HK]
    low_courant = (aperture[I, J, K] * velocity[I, J, K]
                   * inverse_spacing_dt)
    high_courant = (aperture[HI, HJ, HK] * velocity[HI, HJ, HK]
                    * inverse_spacing_dt)
    # Weymouth-Yue conservative dilation: hold one binary cell colour fixed
    # through all three sweeps.  Using fractional alpha here lets low-volume
    # interface wisps create or destroy water when their extension is not
    # divergence-free.
    colour = wp.float32(0.0)
    if uniform_fraction != 0:
        colour = reference_alpha[I, J, K]
    elif reference_alpha[I, J, K] >= wp.float32(0.5):
        colour = wp.float32(1.0)
    value = (-(high_flux - low_flux)
             + colour * (high_courant - low_courant))
    if not wp.isfinite(value):
        wp.atomic_max(invalid, 0, 1)
        change[I, J, K] = wp.float32(0.0)
        return
    change[I, J, K] = value


@wp.kernel
def _redistribute_small_cell_change(
    alpha: wp.array3d(dtype=wp.float32),
    change: wp.array3d(dtype=wp.float32),
    theta: wp.array3d(dtype=wp.float32),
    solid: wp.array3d(dtype=wp.int32),
    redistributed: wp.array3d(dtype=wp.float32),
    threshold: wp.float32,
    nx: int,
    ny: int,
    nz: int,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    ng: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    fraction = theta[I, J, K]
    if solid[I, J, K] != 0 or fraction <= wp.float32(0.0) or fraction >= threshold:
        return

    original = change[I, J, K]
    retained = original * fraction / threshold
    excess = original - retained
    if excess == wp.float32(0.0):
        return
    recipients = int(0)
    for neighbor in range(6):
        ni = i
        nj = j
        nk = k
        if neighbor == 0:
            ni -= 1
        elif neighbor == 1:
            ni += 1
        elif neighbor == 2:
            nj -= 1
        elif neighbor == 3:
            nj += 1
        elif neighbor == 4:
            nk -= 1
        else:
            nk += 1
        ci = _wrapped_cell_index(ni, nx, periodic_x)
        cj = _wrapped_cell_index(nj, ny, periodic_y)
        ck = _wrapped_cell_index(nk, nz, periodic_z)
        if ci >= 0 and cj >= 0 and ck >= 0:
            NI = ci + ng
            NJ = cj + ng
            NK = ck + ng
            neighbor_theta = theta[NI, NJ, NK]
            neighbor_mass = neighbor_theta * alpha[NI, NJ, NK] + change[NI, NJ, NK]
            accepts = int(0)
            if excess > wp.float32(0.0) and neighbor_mass < neighbor_theta:
                accepts = 1
            elif excess < wp.float32(0.0) and neighbor_mass > wp.float32(0.0):
                accepts = 1
            if (solid[NI, NJ, NK] == 0 and neighbor_theta >= threshold
                    and accepts != 0):
                recipients += 1
    if recipients == 0:
        return

    share = excess / wp.float32(recipients)
    change[I, J, K] = retained
    for neighbor in range(6):
        ni = i
        nj = j
        nk = k
        if neighbor == 0:
            ni -= 1
        elif neighbor == 1:
            ni += 1
        elif neighbor == 2:
            nj -= 1
        elif neighbor == 3:
            nj += 1
        elif neighbor == 4:
            nk -= 1
        else:
            nk += 1
        ci = _wrapped_cell_index(ni, nx, periodic_x)
        cj = _wrapped_cell_index(nj, ny, periodic_y)
        ck = _wrapped_cell_index(nk, nz, periodic_z)
        if ci >= 0 and cj >= 0 and ck >= 0:
            NI = ci + ng
            NJ = cj + ng
            NK = ck + ng
            neighbor_theta = theta[NI, NJ, NK]
            neighbor_mass = neighbor_theta * alpha[NI, NJ, NK] + change[NI, NJ, NK]
            accepts = int(0)
            if excess > wp.float32(0.0) and neighbor_mass < neighbor_theta:
                accepts = 1
            elif excess < wp.float32(0.0) and neighbor_mass > wp.float32(0.0):
                accepts = 1
            if (solid[NI, NJ, NK] == 0 and neighbor_theta >= threshold
                    and accepts != 0):
                wp.atomic_add(redistributed, NI, NJ, NK, share)


@wp.kernel
def _finalize_directional_vof(
    alpha: wp.array3d(dtype=wp.float32),
    theta: wp.array3d(dtype=wp.float32),
    solid: wp.array3d(dtype=wp.int32),
    change: wp.array3d(dtype=wp.float32),
    redistributed: wp.array3d(dtype=wp.float32),
    updated: wp.array3d(dtype=wp.float32),
    clipped_volume: wp.array(dtype=wp.float64),
    invalid: wp.array(dtype=wp.int32),
    cell_volume: wp.float64,
    ng: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    fraction = theta[I, J, K]
    if solid[I, J, K] != 0 or fraction <= wp.float32(0.0):
        updated[I, J, K] = wp.float32(0.0)
        return
    raw_mass = (fraction * alpha[I, J, K] + change[I, J, K]
                + redistributed[I, J, K])
    raw = raw_mass / fraction
    if not wp.isfinite(raw):
        wp.atomic_max(invalid, 0, 1)
        updated[I, J, K] = alpha[I, J, K]
        return
    bounded = wp.min(wp.float32(1.0), wp.max(wp.float32(0.0), raw))
    wp.atomic_add(
        clipped_volume, 0,
        wp.float64(wp.abs(bounded * fraction - raw_mass)) * cell_volume,
    )
    updated[I, J, K] = bounded


class VOFAdvector:
    """Advance ``alpha`` with cyclic x/y/z split ordering on CUDA."""

    __slots__ = (
        "cfg", "device", "steps", "total_clipped_volume", "_reference",
        "_updated", "_change", "_redistributed", "_clipped", "_invalid",
    )

    _ORDERS = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
    _NAMES = ("x", "y", "z")
    _FIELDS = ("u", "v", "w")

    def __init__(self, cfg: Solver3DConfig):
        if not isinstance(cfg, Solver3DConfig):
            raise TypeError("cfg must be a Solver3DConfig")
        self.cfg = cfg
        self.device = cfg.device
        shape = tuple(n + 2 * cfg.grid.ng for n in cfg.grid.shape)
        self._reference = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self._updated = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self._change = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self._redistributed = wp.zeros(
            shape, dtype=wp.float32, device=self.device,
        )
        self._clipped = wp.zeros(1, dtype=wp.float64, device=self.device)
        self._invalid = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.steps = 0
        self.total_clipped_volume = 0.0

    def _copy(self, source: wp.array, target: wp.array) -> None:
        target.zero_()
        wp.launch(
            _copy_cell_field,
            dim=self.cfg.grid.shape,
            inputs=[source, target, self.cfg.grid.ng],
            device=self.device,
        )

    def _restore(self, state: State3D) -> None:
        self._copy(self._reference, state.alpha)
        fill_field_boundaries(state, "alpha")

    def advect(
        self,
        state: State3D,
        dt: float,
        *,
        transport_velocity: tuple[wp.array, wp.array, wp.array] | None = None,
    ) -> VOFAdvectionResult:
        """Advance one full three-direction cycle and update ``state.alpha``."""
        if state.device != self.device or state.cfg.grid != self.cfg.grid:
            raise ValueError("state grid and device must match the VOF advector")
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive and finite")
        if transport_velocity is None:
            velocity_fields = tuple(state.field(name) for name in self._FIELDS)
        else:
            if len(transport_velocity) != 3:
                raise ValueError("transport_velocity must contain u, v, and w")
            velocity_fields = tuple(transport_velocity)
            for name, field in zip(self._FIELDS, velocity_fields):
                expected = state.grid.padded_shape(name)
                if tuple(field.shape) != expected or field.dtype != wp.float32:
                    raise ValueError(
                        f"transport {name} must be a float32 field of shape "
                        f"{expected}",
                    )

        before = measure_vof(state)
        if before.alpha_min < 0.0 or before.alpha_max > 1.0:
            raise ValueError(
                f"alpha must be bounded before advection, got "
                f"[{before.alpha_min:.6g}, {before.alpha_max:.6g}]",
            )
        self._copy(state.alpha, self._reference)
        uniform_fraction = int(before.alpha_min == before.alpha_max)
        self._clipped.zero_()
        self._invalid.zero_()
        order = self._ORDERS[self.steps % len(self._ORDERS)]
        max_courant = 0.0
        g = self.cfg.grid
        spacings = (g.dx, g.dy, g.dz)
        cell_volume = g.dx * g.dy * g.dz
        directional_fluxes: list[DirectionalFluxResult] = []
        periodic = (
            int(self.cfg.bc.xlo == BC.PERIODIC),
            int(self.cfg.bc.ylo == BC.PERIODIC),
            int(self.cfg.bc.zlo == BC.PERIODIC),
        )
        apertures = (
            state.u_aperture, state.v_aperture, state.w_aperture,
        )

        try:
            if transport_velocity is None:
                for field_name in self._FIELDS:
                    fill_field_boundaries(state, field_name)
            for axis in order:
                interface = estimate_interface_normals(state)
                reconstruction = reconstruct_plic(state, interface.normals)
                directional = compute_directional_liquid_fluxes(
                    state, reconstruction, axis, dt,
                    velocity=velocity_fields[axis],
                )
                directional_fluxes.append(directional)
                max_courant = max(max_courant, directional.max_courant)
                self._updated.zero_()
                self._change.zero_()
                self._redistributed.zero_()
                wp.launch(
                    _directional_mass_change,
                    dim=g.shape,
                    inputs=[state.alpha, state.solid, self._reference,
                            directional.fluxes,
                            velocity_fields[axis], apertures[axis],
                            self._change, self._invalid, axis,
                            uniform_fraction,
                            dt / spacings[axis], g.ng],
                    device=self.device,
                )
                if state.small_cell_threshold > 0.0:
                    wp.launch(
                        _redistribute_small_cell_change,
                        dim=g.shape,
                        inputs=[state.alpha, self._change,
                                state.theta, state.solid,
                                self._redistributed,
                                state.small_cell_threshold,
                                g.nx, g.ny, g.nz, *periodic, g.ng],
                        device=self.device,
                    )
                wp.launch(
                    _finalize_directional_vof,
                    dim=g.shape,
                    inputs=[state.alpha, state.theta, state.solid,
                            self._change, self._redistributed,
                            self._updated, self._clipped, self._invalid,
                            cell_volume, g.ng],
                    device=self.device,
                )
                if int(self._invalid.numpy()[0]) != 0:
                    raise FloatingPointError("non-finite alpha during VOF sweep")
                self._copy(self._updated, state.alpha)
                fill_field_boundaries(state, "alpha")
        except Exception:
            self._restore(state)
            raise

        clipped = float(self._clipped.numpy()[0])
        after = measure_vof(state, clipped_volume=clipped)
        self.total_clipped_volume += clipped
        self.steps += 1
        return VOFAdvectionResult(
            sweep_order=tuple(self._NAMES[axis] for axis in order),
            volume_before=before.volume,
            volume_after=after.volume,
            volume_drift=after.volume - before.volume,
            clipped_volume=clipped,
            cumulative_clipped_volume=self.total_clipped_volume,
            max_courant=max_courant,
            directional_fluxes=tuple(directional_fluxes),  # type: ignore[arg-type]
        )
