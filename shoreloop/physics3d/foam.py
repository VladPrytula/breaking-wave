"""Passive interface-band foam sourced by measured irreversible work."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Sequence

import warp as wp

from shoreloop.config import BC, Solver3DConfig
from shoreloop.physics3d.advection import _copy_cell_field
from shoreloop.physics3d.boundary import fill_field_boundaries
from shoreloop.physics3d.extrapolation import _wrapped_cell_index
from shoreloop.physics3d.state import State3D


@dataclass(frozen=True)
class Foam3DConfig:
    # m^3/J: dissipated power density [W/m^3] -> concentration rate [1/s].
    dissipation_yield: float = 2.0e-4
    deformation_yield: float = 0.012
    deformation_threshold: float = 4.0
    decay_time: float = 2.5
    diffusivity: float = 2.0e-5
    maximum_concentration: float = 1.0

    def __post_init__(self) -> None:
        if self.dissipation_yield < 0.0 or self.deformation_yield < 0.0:
            raise ValueError("foam yields must be non-negative")
        if self.deformation_threshold < 0.0:
            raise ValueError("deformation_threshold must be non-negative")
        if self.decay_time <= 0.0:
            raise ValueError("decay_time must be positive")
        if self.diffusivity < 0.0:
            raise ValueError("diffusivity must be non-negative")
        if self.maximum_concentration <= 0.0:
            raise ValueError("maximum_concentration must be positive")


@dataclass(frozen=True)
class Foam3DResult:
    mass_before: float
    mass_after: float
    maximum_concentration: float
    active_source_cells: int
    numerical_residual_energy_used: float


@wp.func
def _cell_scalar(
    field: wp.array3d(dtype=wp.float32),
    i: int,
    j: int,
    k: int,
    nx: int,
    ny: int,
    nz: int,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    ng: int,
) -> wp.float32:
    ci = _wrapped_cell_index(i, nx, periodic_x)
    cj = _wrapped_cell_index(j, ny, periodic_y)
    ck = _wrapped_cell_index(k, nz, periodic_z)
    if ci < 0 or cj < 0 or ck < 0:
        return wp.float32(0.0)
    return field[ci + ng, cj + ng, ck + ng]


@wp.func
def _foam_face_flux(
    foam: wp.array3d(dtype=wp.float32),
    alpha: wp.array3d(dtype=wp.float32),
    theta: wp.array3d(dtype=wp.float32),
    velocity: wp.float32,
    aperture: wp.float32,
    lower_i: int,
    lower_j: int,
    lower_k: int,
    upper_i: int,
    upper_j: int,
    upper_k: int,
    inverse_spacing_dt: wp.float32,
    nx: int,
    ny: int,
    nz: int,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    ng: int,
) -> wp.float32:
    lower_liquid = _cell_scalar(
        alpha, lower_i, lower_j, lower_k,
        nx, ny, nz, periodic_x, periodic_y, periodic_z, ng,
    ) * _cell_scalar(
        theta, lower_i, lower_j, lower_k,
        nx, ny, nz, periodic_x, periodic_y, periodic_z, ng,
    )
    upper_liquid = _cell_scalar(
        alpha, upper_i, upper_j, upper_k,
        nx, ny, nz, periodic_x, periodic_y, periodic_z, ng,
    ) * _cell_scalar(
        theta, upper_i, upper_j, upper_k,
        nx, ny, nz, periodic_x, periodic_y, periodic_z, ng,
    )
    liquid = aperture * wp.min(lower_liquid, upper_liquid)
    donor = _cell_scalar(
        foam, upper_i, upper_j, upper_k,
        nx, ny, nz, periodic_x, periodic_y, periodic_z, ng,
    )
    if velocity >= wp.float32(0.0):
        donor = _cell_scalar(
            foam, lower_i, lower_j, lower_k,
            nx, ny, nz, periodic_x, periodic_y, periodic_z, ng,
        )
    return velocity * inverse_spacing_dt * liquid * donor


@wp.kernel
def _transport_foam(
    foam: wp.array3d(dtype=wp.float32),
    alpha: wp.array3d(dtype=wp.float32),
    theta: wp.array3d(dtype=wp.float32),
    u: wp.array3d(dtype=wp.float32),
    v: wp.array3d(dtype=wp.float32),
    w: wp.array3d(dtype=wp.float32),
    u_aperture: wp.array3d(dtype=wp.float32),
    v_aperture: wp.array3d(dtype=wp.float32),
    w_aperture: wp.array3d(dtype=wp.float32),
    transported: wp.array3d(dtype=wp.float32),
    dt_dx: wp.float32,
    dt_dy: wp.float32,
    dt_dz: wp.float32,
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
    liquid = alpha[I, J, K] * theta[I, J, K]
    if liquid <= wp.float32(1.0e-6):
        transported[I, J, K] = wp.float32(0.0)
        return
    mass = liquid * foam[I, J, K]
    xlo = _foam_face_flux(
        foam, alpha, theta, u[I, J, K], u_aperture[I, J, K],
        i - 1, j, k, i, j, k, dt_dx,
        nx, ny, nz, periodic_x, periodic_y, periodic_z, ng,
    )
    xhi = _foam_face_flux(
        foam, alpha, theta, u[I + 1, J, K], u_aperture[I + 1, J, K],
        i, j, k, i + 1, j, k, dt_dx,
        nx, ny, nz, periodic_x, periodic_y, periodic_z, ng,
    )
    ylo = _foam_face_flux(
        foam, alpha, theta, v[I, J, K], v_aperture[I, J, K],
        i, j - 1, k, i, j, k, dt_dy,
        nx, ny, nz, periodic_x, periodic_y, periodic_z, ng,
    )
    yhi = _foam_face_flux(
        foam, alpha, theta, v[I, J + 1, K], v_aperture[I, J + 1, K],
        i, j, k, i, j + 1, k, dt_dy,
        nx, ny, nz, periodic_x, periodic_y, periodic_z, ng,
    )
    zlo = _foam_face_flux(
        foam, alpha, theta, w[I, J, K], w_aperture[I, J, K],
        i, j, k - 1, i, j, k, dt_dz,
        nx, ny, nz, periodic_x, periodic_y, periodic_z, ng,
    )
    zhi = _foam_face_flux(
        foam, alpha, theta, w[I, J, K + 1], w_aperture[I, J, K + 1],
        i, j, k, i, j, k + 1, dt_dz,
        nx, ny, nz, periodic_x, periodic_y, periodic_z, ng,
    )
    next_mass = mass + xlo - xhi + ylo - yhi + zlo - zhi
    transported[I, J, K] = wp.max(wp.float32(0.0), next_mass / liquid)


@wp.kernel
def _source_weights(
    alpha: wp.array3d(dtype=wp.float32),
    strain: wp.array3d(dtype=wp.float32),
    weight: wp.array3d(dtype=wp.float32),
    total: wp.array(dtype=wp.float64),
    threshold: wp.float32,
    cell_volume: wp.float64,
    ng: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    fraction = alpha[I, J, K]
    interface = wp.float32(4.0) * fraction * (wp.float32(1.0) - fraction)
    value = interface * wp.max(wp.float32(0.0), strain[I, J, K] - threshold)
    weight[I, J, K] = value
    wp.atomic_add(total, 0, wp.float64(value) * cell_volume)


@wp.kernel
def _source_decay_diffuse_foam(
    transported: wp.array3d(dtype=wp.float32),
    alpha: wp.array3d(dtype=wp.float32),
    theta: wp.array3d(dtype=wp.float32),
    strain: wp.array3d(dtype=wp.float32),
    molecular: wp.array3d(dtype=wp.float32),
    sgs: wp.array3d(dtype=wp.float32),
    wall: wp.array3d(dtype=wp.float32),
    porous: wp.array3d(dtype=wp.float32),
    residual_weight: wp.array3d(dtype=wp.float32),
    updated: wp.array3d(dtype=wp.float32),
    active_source_cells: wp.array(dtype=wp.int32),
    maximum: wp.array(dtype=wp.float32),
    dt: wp.float32,
    dissipation_yield: wp.float32,
    deformation_yield: wp.float32,
    deformation_threshold: wp.float32,
    residual_power_scale: wp.float32,
    decay_time: wp.float32,
    diffusivity: wp.float32,
    inverse_dx_squared: wp.float32,
    inverse_dy_squared: wp.float32,
    inverse_dz_squared: wp.float32,
    maximum_concentration: wp.float32,
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
    liquid = alpha[I, J, K] * theta[I, J, K]
    if liquid <= wp.float32(1.0e-6):
        updated[I, J, K] = wp.float32(0.0)
        return
    fraction = alpha[I, J, K]
    interface = wp.float32(4.0) * fraction * (wp.float32(1.0) - fraction)
    direct_power = (
        molecular[I, J, K] + sgs[I, J, K]
        + wall[I, J, K] + porous[I, J, K]
    )
    deformation = wp.max(
        wp.float32(0.0), strain[I, J, K] - deformation_threshold,
    )
    # The residual is power too, already weighted into the interface band.
    # Both energy channels use the same conversion yield exactly once.
    source = interface * (
        dissipation_yield * direct_power
        + deformation_yield * deformation
    ) + dissipation_yield * residual_power_scale * residual_weight[I, J, K]
    value = transported[I, J, K]
    laplacian = wp.float32(0.0)
    for direction in range(6):
        ni = i
        nj = j
        nk = k
        inverse_spacing_squared = inverse_dx_squared
        if direction == 0:
            ni -= 1
        elif direction == 1:
            ni += 1
        elif direction == 2:
            nj -= 1
            inverse_spacing_squared = inverse_dy_squared
        elif direction == 3:
            nj += 1
            inverse_spacing_squared = inverse_dy_squared
        elif direction == 4:
            nk -= 1
            inverse_spacing_squared = inverse_dz_squared
        else:
            nk += 1
            inverse_spacing_squared = inverse_dz_squared
        neighbor = _cell_scalar(
            transported, ni, nj, nk,
            nx, ny, nz, periodic_x, periodic_y, periodic_z, ng,
        )
        laplacian += (neighbor - value) * inverse_spacing_squared
    next_value = (
        value + dt * (source + diffusivity * laplacian)
    ) / (wp.float32(1.0) + dt / decay_time)
    next_value = wp.min(maximum_concentration, wp.max(wp.float32(0.0), next_value))
    updated[I, J, K] = next_value
    if source > wp.float32(0.0):
        wp.atomic_add(active_source_cells, 0, 1)
    wp.atomic_max(maximum, 0, next_value)


@wp.kernel
def _sum_foam_mass(
    foam: wp.array3d(dtype=wp.float32),
    alpha: wp.array3d(dtype=wp.float32),
    theta: wp.array3d(dtype=wp.float32),
    total: wp.array(dtype=wp.float64),
    cell_volume: wp.float64,
    ng: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    wp.atomic_add(
        total, 0,
        wp.float64(foam[I, J, K] * alpha[I, J, K] * theta[I, J, K])
        * cell_volume,
    )


class InterfaceFoamModel:
    """Advance a passive concentration without writing hydrodynamic fields."""

    __slots__ = (
        "cfg", "settings", "device", "_transported", "_updated",
        "_zero", "_strain", "_residual_weight", "_weight_total",
        "_mass", "_maximum", "_active_source_cells",
    )

    def __init__(
        self,
        cfg: Solver3DConfig,
        settings: Foam3DConfig | None = None,
    ):
        self.cfg = cfg
        self.settings = Foam3DConfig() if settings is None else settings
        self.device = cfg.device
        shape = tuple(n + 2 * cfg.grid.ng for n in cfg.grid.shape)
        self._transported = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self._updated = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self._zero = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self._strain = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self._residual_weight = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self._weight_total = wp.zeros(1, dtype=wp.float64, device=self.device)
        self._mass = wp.zeros(1, dtype=wp.float64, device=self.device)
        self._maximum = wp.zeros(1, dtype=wp.float32, device=self.device)
        self._active_source_cells = wp.zeros(
            1, dtype=wp.int32, device=self.device,
        )

    def time_step_limit(self, cfl: float = 0.25) -> float:
        """Return the diffusion limit used only for internal foam substeps."""
        if self.settings.diffusivity == 0.0:
            return math.inf
        g = self.cfg.grid
        return cfl / (
            self.settings.diffusivity
            * (1.0 / g.dx**2 + 1.0 / g.dy**2 + 1.0 / g.dz**2)
        )

    def _foam_mass(self, state: State3D) -> float:
        self._mass.zero_()
        g = self.cfg.grid
        wp.launch(
            _sum_foam_mass,
            dim=g.shape,
            inputs=[
                state.foam, state.alpha, state.theta, self._mass,
                g.dx * g.dy * g.dz, g.ng,
            ],
            device=self.device,
        )
        return float(self._mass.numpy()[0])

    def apply(
        self,
        state: State3D,
        dt: float,
        *,
        strain_rate: wp.array | None = None,
        dissipation_fields: Sequence[wp.array] = (),
        numerical_loss: float = 0.0,
    ) -> Foam3DResult:
        g = self.cfg.grid
        periodic = (
            int(self.cfg.bc.xlo == BC.PERIODIC),
            int(self.cfg.bc.ylo == BC.PERIODIC),
            int(self.cfg.bc.zlo == BC.PERIODIC),
        )
        mass_before = self._foam_mass(state)
        strain = self._strain if strain_rate is None else strain_rate
        if strain_rate is None:
            strain.zero_()
        direct = list(dissipation_fields[:4])
        direct.extend([self._zero] * (4 - len(direct)))
        self._residual_weight.zero_()
        self._weight_total.zero_()
        cell_volume = g.dx * g.dy * g.dz
        wp.launch(
            _source_weights,
            dim=g.shape,
            inputs=[
                state.alpha, strain, self._residual_weight,
                self._weight_total, self.settings.deformation_threshold,
                cell_volume, g.ng,
            ],
            device=self.device,
        )
        weight_total = float(self._weight_total.numpy()[0])
        residual_power_scale = (
            numerical_loss / (dt * weight_total) if weight_total > 0.0 else 0.0
        )
        diffusion_limit = self.time_step_limit()
        substeps = max(1, math.ceil(dt / diffusion_limit))
        substep_dt = dt / substeps
        for _ in range(substeps):
            self._transported.zero_()
            self._updated.zero_()
            wp.launch(
                _transport_foam,
                dim=g.shape,
                inputs=[
                    state.foam, state.alpha, state.theta,
                    state.u, state.v, state.w,
                    state.u_aperture, state.v_aperture, state.w_aperture,
                    self._transported,
                    substep_dt / g.dx, substep_dt / g.dy, substep_dt / g.dz,
                    g.nx, g.ny, g.nz, *periodic, g.ng,
                ],
                device=self.device,
            )
            self._maximum.zero_()
            self._active_source_cells.zero_()
            wp.launch(
                _source_decay_diffuse_foam,
                dim=g.shape,
                inputs=[
                    self._transported, state.alpha, state.theta, strain,
                    direct[0], direct[1], direct[2], direct[3],
                    self._residual_weight, self._updated,
                    self._active_source_cells, self._maximum, substep_dt,
                    self.settings.dissipation_yield,
                    self.settings.deformation_yield,
                    self.settings.deformation_threshold,
                    residual_power_scale, self.settings.decay_time,
                    self.settings.diffusivity,
                    1.0 / g.dx**2, 1.0 / g.dy**2, 1.0 / g.dz**2,
                    self.settings.maximum_concentration,
                    g.nx, g.ny, g.nz, *periodic, g.ng,
                ],
                device=self.device,
            )
            wp.launch(
                _copy_cell_field,
                dim=g.shape,
                inputs=[self._updated, state.foam, g.ng],
                device=self.device,
            )
            fill_field_boundaries(state, "foam")
        return Foam3DResult(
            mass_before=mass_before,
            mass_after=self._foam_mass(state),
            maximum_concentration=float(self._maximum.numpy()[0]),
            active_source_cells=int(self._active_source_cells.numpy()[0]),
            numerical_residual_energy_used=(
                numerical_loss
                if weight_total > 0.0 and self.settings.dissipation_yield > 0.0
                else 0.0
            ),
        )
