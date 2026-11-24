"""Aperture-aware molecular and Smagorinsky momentum diffusion.

Both paths use a symmetric graph Laplacian on MAC velocities. The legacy
explicit path uses diagonal face inertia and minimum-face-mass conductance.
The liquid path uses the pressure kinetic matrix, shared liquid facets, and
implicit midpoint; see ``liquid_viscosity`` for its geometry and work contract.
Both use the componentwise approximation
``div(nu_eff grad(u))``, not the full variable-viscosity
``div(2 nu_eff D(u))``; the two agree for constant viscosity and
divergence-free flow.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import warp as wp

from shoreloop.config import BC, Solver3DConfig
from shoreloop.physics3d.advection import _copy_cell_field
from shoreloop.physics3d.boundary import fill_field_boundaries
from shoreloop.physics3d.extrapolation import (_face_neighbor_index,
                                               _wrapped_cell_index)
from shoreloop.physics3d.momentum import compute_face_liquid_mass
from shoreloop.physics3d.state import State3D


@dataclass(frozen=True)
class ViscosityConfig:
    """Switchable physical and sub-grid kinematic viscosities."""

    molecular_viscosity: float = 0.0
    smagorinsky_cs: float = 0.0
    maximum_sgs_viscosity: float = 0.05

    def __post_init__(self) -> None:
        if self.molecular_viscosity < 0.0:
            raise ValueError("molecular_viscosity must be non-negative")
        if self.smagorinsky_cs < 0.0:
            raise ValueError("smagorinsky_cs must be non-negative")
        if self.maximum_sgs_viscosity <= 0.0:
            raise ValueError("maximum_sgs_viscosity must be positive")


@dataclass(frozen=True)
class ViscosityResult:
    """Molecular/SGS work, with midpoint solve/roundoff loss reported separately.

    The legacy explicit path retains its before/after direct-work convention.
    The liquid midpoint path reports nonnegative edge work as direct loss;
    its signed numerical remainder belongs to EnergyLedger's residual.
    """

    molecular_energy_loss: float
    sgs_energy_loss: float
    maximum_sgs_viscosity: float
    molecular_numerical_loss: float = 0.0
    sgs_numerical_loss: float = 0.0
    # Liquid midpoint path: CG iterations summed over the molecular and SGS solves.
    iterations: int = 0

    @property
    def total_energy_loss(self) -> float:
        return self.molecular_energy_loss + self.sgs_energy_loss


@wp.func
def _bounded_cell_index(index: int, count: int, periodic: int) -> int:
    result = _wrapped_cell_index(index, count, periodic)
    if result < 0:
        if index < 0:
            result = 0
        else:
            result = count - 1
    return result


@wp.func
def _cell_velocity(
    u: wp.array3d(dtype=wp.float32),
    v: wp.array3d(dtype=wp.float32),
    w: wp.array3d(dtype=wp.float32),
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
) -> wp.vec3:
    ci = _bounded_cell_index(i, nx, periodic_x)
    cj = _bounded_cell_index(j, ny, periodic_y)
    ck = _bounded_cell_index(k, nz, periodic_z)
    I = ci + ng
    J = cj + ng
    K = ck + ng
    return wp.vec3(
        wp.float32(0.5) * (u[I, J, K] + u[I + 1, J, K]),
        wp.float32(0.5) * (v[I, J, K] + v[I, J + 1, K]),
        wp.float32(0.5) * (w[I, J, K] + w[I, J, K + 1]),
    )


@wp.kernel
def _compute_smagorinsky(
    u: wp.array3d(dtype=wp.float32),
    v: wp.array3d(dtype=wp.float32),
    w: wp.array3d(dtype=wp.float32),
    alpha: wp.array3d(dtype=wp.float32),
    theta: wp.array3d(dtype=wp.float32),
    strain: wp.array3d(dtype=wp.float32),
    viscosity: wp.array3d(dtype=wp.float32),
    maximum: wp.array(dtype=wp.float32),
    cs_delta_squared: wp.float32,
    viscosity_cap: wp.float32,
    inverse_dx: wp.float32,
    inverse_dy: wp.float32,
    inverse_dz: wp.float32,
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
    if alpha[I, J, K] <= wp.float32(0.0) or theta[I, J, K] <= wp.float32(0.0):
        strain[I, J, K] = wp.float32(0.0)
        viscosity[I, J, K] = wp.float32(0.0)
        return

    xlo = _cell_velocity(
        u, v, w, i - 1, j, k, nx, ny, nz,
        periodic_x, periodic_y, periodic_z, ng,
    )
    xhi = _cell_velocity(
        u, v, w, i + 1, j, k, nx, ny, nz,
        periodic_x, periodic_y, periodic_z, ng,
    )
    ylo = _cell_velocity(
        u, v, w, i, j - 1, k, nx, ny, nz,
        periodic_x, periodic_y, periodic_z, ng,
    )
    yhi = _cell_velocity(
        u, v, w, i, j + 1, k, nx, ny, nz,
        periodic_x, periodic_y, periodic_z, ng,
    )
    zlo = _cell_velocity(
        u, v, w, i, j, k - 1, nx, ny, nz,
        periodic_x, periodic_y, periodic_z, ng,
    )
    zhi = _cell_velocity(
        u, v, w, i, j, k + 1, nx, ny, nz,
        periodic_x, periodic_y, periodic_z, ng,
    )

    du_dx = (u[I + 1, J, K] - u[I, J, K]) * inverse_dx
    dv_dy = (v[I, J + 1, K] - v[I, J, K]) * inverse_dy
    dw_dz = (w[I, J, K + 1] - w[I, J, K]) * inverse_dz
    du_dy = wp.float32(0.5) * (yhi[0] - ylo[0]) * inverse_dy
    du_dz = wp.float32(0.5) * (zhi[0] - zlo[0]) * inverse_dz
    dv_dx = wp.float32(0.5) * (xhi[1] - xlo[1]) * inverse_dx
    dv_dz = wp.float32(0.5) * (zhi[1] - zlo[1]) * inverse_dz
    dw_dx = wp.float32(0.5) * (xhi[2] - xlo[2]) * inverse_dx
    dw_dy = wp.float32(0.5) * (yhi[2] - ylo[2]) * inverse_dy

    sxy = wp.float32(0.5) * (du_dy + dv_dx)
    sxz = wp.float32(0.5) * (du_dz + dw_dx)
    syz = wp.float32(0.5) * (dv_dz + dw_dy)
    magnitude = wp.sqrt(wp.float32(2.0) * (
        du_dx * du_dx + dv_dy * dv_dy + dw_dz * dw_dz
        + wp.float32(2.0) * (sxy * sxy + sxz * sxz + syz * syz)
    ))
    value = wp.min(viscosity_cap, cs_delta_squared * magnitude)
    strain[I, J, K] = magnitude
    viscosity[I, J, K] = value
    wp.atomic_max(maximum, 0, value)


@wp.func
def _face_viscosity(
    cell_viscosity: wp.array3d(dtype=wp.float32),
    i: int,
    j: int,
    k: int,
    component_axis: int,
    base_viscosity: wp.float32,
    nx: int,
    ny: int,
    nz: int,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    ng: int,
) -> wp.float32:
    lower_i = i
    lower_j = j
    lower_k = k
    if component_axis == 0:
        lower_i -= 1
    elif component_axis == 1:
        lower_j -= 1
    else:
        lower_k -= 1
    li = _wrapped_cell_index(lower_i, nx, periodic_x)
    lj = _wrapped_cell_index(lower_j, ny, periodic_y)
    lk = _wrapped_cell_index(lower_k, nz, periodic_z)
    ui = _wrapped_cell_index(i, nx, periodic_x)
    uj = _wrapped_cell_index(j, ny, periodic_y)
    uk = _wrapped_cell_index(k, nz, periodic_z)
    total = wp.float32(0.0)
    count = wp.float32(0.0)
    if li >= 0 and lj >= 0 and lk >= 0:
        total += cell_viscosity[li + ng, lj + ng, lk + ng]
        count += wp.float32(1.0)
    if ui >= 0 and uj >= 0 and uk >= 0:
        total += cell_viscosity[ui + ng, uj + ng, uk + ng]
        count += wp.float32(1.0)
    value = base_viscosity
    if count > wp.float32(0.0):
        value += total / count
    return value


@wp.kernel
def _diffuse_faces(
    velocity: wp.array3d(dtype=wp.float32),
    mass: wp.array3d(dtype=wp.float32),
    cell_viscosity: wp.array3d(dtype=wp.float32),
    updated: wp.array3d(dtype=wp.float32),
    local_dissipation: wp.array3d(dtype=wp.float32),
    energy_loss: wp.array(dtype=wp.float64),
    component_axis: int,
    base_viscosity: wp.float32,
    dt: wp.float32,
    inverse_dx_squared: wp.float32,
    inverse_dy_squared: wp.float32,
    inverse_dz_squared: wp.float32,
    rho_cell_volume: wp.float64,
    rho: wp.float32,
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
    centre_mass = mass[I, J, K]
    value = velocity[I, J, K]
    if centre_mass <= wp.float32(1.0e-6):
        updated[I, J, K] = value
        return

    sx = nx
    sy = ny
    sz = nz
    if component_axis == 0:
        sx += 1
    elif component_axis == 1:
        sy += 1
    else:
        sz += 1
    centre_viscosity = _face_viscosity(
        cell_viscosity, i, j, k, component_axis, base_viscosity,
        nx, ny, nz, periodic_x, periodic_y, periodic_z, ng,
    )
    change = wp.float32(0.0)
    power = wp.float32(0.0)
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
        ni = _face_neighbor_index(ni, sx, periodic_x, int(component_axis == 0))
        nj = _face_neighbor_index(nj, sy, periodic_y, int(component_axis == 1))
        nk = _face_neighbor_index(nk, sz, periodic_z, int(component_axis == 2))
        if ni >= 0 and nj >= 0 and nk >= 0:
            NI = ni + ng
            NJ = nj + ng
            NK = nk + ng
            neighbor_mass = mass[NI, NJ, NK]
            if neighbor_mass > wp.float32(1.0e-6):
                neighbor_viscosity = _face_viscosity(
                    cell_viscosity, ni, nj, nk, component_axis,
                    base_viscosity, nx, ny, nz,
                    periodic_x, periodic_y, periodic_z, ng,
                )
                conductance = (
                    wp.min(centre_mass, neighbor_mass)
                    * wp.float32(0.5)
                    * (centre_viscosity + neighbor_viscosity)
                    * inverse_spacing_squared
                )
                difference = velocity[NI, NJ, NK] - value
                change += conductance * difference
                power += (
                    wp.float32(0.5) * rho * conductance
                    * difference * difference
                )

    next_value = value + dt * change / centre_mass
    updated[I, J, K] = next_value
    q = i
    count = nx
    periodic = periodic_x
    if component_axis == 1:
        q = j
        count = ny
        periodic = periodic_y
    elif component_axis == 2:
        q = k
        count = nz
        periodic = periodic_z
    if periodic == 0 or q != count:
        wp.atomic_add(
            energy_loss, 0,
            wp.float64(wp.float32(0.5) * centre_mass
                       * (value * value - next_value * next_value))
            * rho_cell_volume,
        )
        ci = wp.min(nx - 1, wp.max(0, i - int(component_axis == 0)))
        cj = wp.min(ny - 1, wp.max(0, j - int(component_axis == 1)))
        ck = wp.min(nz - 1, wp.max(0, k - int(component_axis == 2)))
        wp.atomic_add(local_dissipation, ci + ng, cj + ng, ck + ng, power)


class ViscosityModel:
    """Apply molecular viscosity and optional Smagorinsky LES on CUDA."""

    __slots__ = (
        "cfg", "settings", "device", "strain_rate", "sgs_viscosity",
        "molecular_dissipation", "sgs_dissipation", "_maximum_sgs",
        "_updated", "_energy_loss", "_zero_viscosity", "_last_maximum_sgs",
    )

    def __init__(
        self,
        cfg: Solver3DConfig,
        settings: ViscosityConfig | None = None,
    ):
        self.cfg = cfg
        self.settings = ViscosityConfig() if settings is None else settings
        self.device = cfg.device
        cell_shape = tuple(n + 2 * cfg.grid.ng for n in cfg.grid.shape)
        self.strain_rate = wp.zeros(cell_shape, dtype=wp.float32, device=self.device)
        self.sgs_viscosity = wp.zeros(cell_shape, dtype=wp.float32, device=self.device)
        self.molecular_dissipation = wp.zeros(
            cell_shape, dtype=wp.float32, device=self.device,
        )
        self.sgs_dissipation = wp.zeros(
            cell_shape, dtype=wp.float32, device=self.device,
        )
        self._zero_viscosity = wp.zeros(
            cell_shape, dtype=wp.float32, device=self.device,
        )
        self._maximum_sgs = wp.zeros(1, dtype=wp.float32, device=self.device)
        self._last_maximum_sgs = None
        self._updated = tuple(
            wp.zeros(
                tuple(n + 2 * cfg.grid.ng for n in shape),
                dtype=wp.float32, device=self.device,
            )
            for shape in (
                (cfg.grid.nx + 1, cfg.grid.ny, cfg.grid.nz),
                (cfg.grid.nx, cfg.grid.ny + 1, cfg.grid.nz),
                (cfg.grid.nx, cfg.grid.ny, cfg.grid.nz + 1),
            )
        )
        self._energy_loss = wp.zeros(1, dtype=wp.float64, device=self.device)

    @property
    def enabled(self) -> bool:
        return (
            self.settings.molecular_viscosity > 0.0
            or self.settings.smagorinsky_cs > 0.0
        )

    def update_sgs(self, state: State3D, *, liquid_operator=None) -> float:
        """Evaluate ``|S|`` and the bounded cell-centred SGS viscosity."""
        self.strain_rate.zero_()
        self.sgs_viscosity.zero_()
        self._maximum_sgs.zero_()
        if self.settings.smagorinsky_cs == 0.0:
            return 0.0
        if self.cfg.pressure_discretization == "liquid_mimetic":
            from shoreloop.physics3d.liquid_viscosity import liquid_viscosity_operator
            return liquid_viscosity_operator(state, liquid_operator).update_sgs(
                state, self.settings, self.strain_rate, self.sgs_viscosity)
        g = self.cfg.grid
        periodic = (
            int(self.cfg.bc.xlo == BC.PERIODIC),
            int(self.cfg.bc.ylo == BC.PERIODIC),
            int(self.cfg.bc.zlo == BC.PERIODIC),
        )
        delta = (g.dx * g.dy * g.dz) ** (1.0 / 3.0)
        wp.launch(
            _compute_smagorinsky,
            dim=g.shape,
            inputs=[
                state.u, state.v, state.w, state.alpha, state.theta,
                self.strain_rate, self.sgs_viscosity, self._maximum_sgs,
                (self.settings.smagorinsky_cs * delta) ** 2,
                self.settings.maximum_sgs_viscosity,
                1.0 / g.dx, 1.0 / g.dy, 1.0 / g.dz,
                g.nx, g.ny, g.nz, *periodic, g.ng,
            ],
            device=self.device,
        )
        return float(self._maximum_sgs.numpy()[0])

    def maximum_effective_viscosity(self, state: State3D, *, liquid_operator=None) -> float:
        if (self.cfg.pressure_discretization == "liquid_mimetic"
                and self._last_maximum_sgs is not None):
            # The time-step cap reuses the SGS maximum measured by the last
            # diffusion pass instead of rebuilding a viscosity operator on the
            # current geometry; the cap is an accuracy bound far from binding.
            return self.settings.molecular_viscosity + self._last_maximum_sgs
        return self.settings.molecular_viscosity + self.update_sgs(state, liquid_operator=liquid_operator)

    def time_step_limit(self, state: State3D, cfl: float = 0.25, *, liquid_operator=None) -> float:
        # On the liquid midpoint path this is an accuracy cap. Energy stability
        # does not require the legacy explicit diagonal-mass CFL estimate.
        viscosity = self.maximum_effective_viscosity(state, liquid_operator=liquid_operator)
        if viscosity == 0.0:
            return math.inf
        g = self.cfg.grid
        inverse_sum = 1.0 / g.dx**2 + 1.0 / g.dy**2 + 1.0 / g.dz**2
        return cfl / (viscosity * inverse_sum)

    def _apply_part(
        self,
        state: State3D,
        dt: float,
        base_viscosity: float,
        cell_viscosity: wp.array,
        local_dissipation: wp.array,
    ) -> float:
        g = self.cfg.grid
        periodic = (
            int(self.cfg.bc.xlo == BC.PERIODIC),
            int(self.cfg.bc.ylo == BC.PERIODIC),
            int(self.cfg.bc.zlo == BC.PERIODIC),
        )
        local_dissipation.zero_()
        self._energy_loss.zero_()
        rho_cell_volume = self.cfg.physics.rho * g.dx * g.dy * g.dz
        for axis, name in enumerate(("u", "v", "w")):
            mass = compute_face_liquid_mass(state, name)
            target = self._updated[axis]
            target.zero_()
            wp.launch(
                _diffuse_faces,
                dim=state.grid.interior_shape(name),
                inputs=[
                    state.field(name), mass, cell_viscosity, target,
                    local_dissipation, self._energy_loss, axis,
                    base_viscosity, dt,
                    1.0 / g.dx**2, 1.0 / g.dy**2, 1.0 / g.dz**2,
                    rho_cell_volume, self.cfg.physics.rho,
                    g.nx, g.ny, g.nz, *periodic, g.ng,
                ],
                device=self.device,
            )
            wp.launch(
                _copy_cell_field,
                dim=state.grid.interior_shape(name),
                inputs=[target, state.field(name), g.ng],
                device=self.device,
            )
            fill_field_boundaries(state, name)
        return float(self._energy_loss.numpy()[0])

    def apply(self, state: State3D, dt: float, *, liquid_operator=None) -> ViscosityResult:
        """Diffuse velocity and return separately measured energy work."""
        if self.cfg.pressure_discretization == "liquid_mimetic" and self.enabled:
            from shoreloop.physics3d.liquid_viscosity import liquid_viscosity_operator
            operator = liquid_viscosity_operator(state, liquid_operator)
            maximum_sgs = operator.update_sgs(
                state, self.settings, self.strain_rate, self.sgs_viscosity)
            self._last_maximum_sgs = maximum_sgs
            self.molecular_dissipation.zero_()
            self.sgs_dissipation.zero_()
            molecular = molecular_numerical = sgs = sgs_numerical = 0.0
            iterations = 0
            if self.settings.molecular_viscosity > 0:
                result = operator.diffuse(state, dt, self.settings.molecular_viscosity,
                                          self._zero_viscosity, self.molecular_dissipation)
                molecular, molecular_numerical = result.physical_loss, result.numerical_loss
                iterations += result.iterations
            if self.settings.smagorinsky_cs > 0:
                result = operator.diffuse(state, dt, 0.0, self.sgs_viscosity, self.sgs_dissipation)
                sgs, sgs_numerical = result.physical_loss, result.numerical_loss
                iterations += result.iterations
            return ViscosityResult(molecular, sgs, maximum_sgs, molecular_numerical, sgs_numerical,
                                   iterations=iterations)
        maximum_sgs = self.update_sgs(state)
        molecular = 0.0
        if self.settings.molecular_viscosity > 0.0:
            molecular = self._apply_part(
                state, dt, self.settings.molecular_viscosity,
                self._zero_viscosity, self.molecular_dissipation,
            )
        sgs = 0.0
        if self.settings.smagorinsky_cs > 0.0:
            sgs = self._apply_part(
                state, dt, 0.0, self.sgs_viscosity,
                self.sgs_dissipation,
            )
        return ViscosityResult(molecular, sgs, maximum_sgs)
