"""Dissipative Navier-slip stress at beach and boulder cut boundaries."""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from shoreloop.config import BC, Solver3DConfig
from shoreloop.physics3d.advection import _copy_cell_field
from shoreloop.physics3d.boundary import fill_field_boundaries
from shoreloop.physics3d.extrapolation import _wrapped_cell_index
from shoreloop.physics3d.momentum import compute_face_liquid_mass
from shoreloop.physics3d.state import State3D
from shoreloop.scene.geometry3d import BOULDER_MATERIAL_ID


# Face energies are formed in float32 and only their reduction is float64.
# Sixty-four ULPs covers the interpolation/projection arithmetic without making
# a scale-independent joule threshold part of the physical model.
_FLOAT32_ENERGY_TOLERANCE = 64.0 * 2.0**-23


WALL_LAWS = ("slip", "log")
KARMAN = 0.41


@dataclass(frozen=True)
class WallStressConfig:
    """Material-specific wall-stress parameters.

    ``law`` "slip" is the regularised Navier slip on every solid; "log" puts the
    fully rough log-law wall function on the beach (``roughness_length`` is its
    z0, the stress is quadratic in the tangential speed at the first cell's
    centre height) while boulders keep their slip length.
    """

    wall_viscosity: float = 2.0e-4
    beach_slip_length: float = 0.012
    boulder_slip_length: float = 0.006
    law: str = "slip"
    roughness_length: float = 0.0

    def __post_init__(self) -> None:
        if self.wall_viscosity < 0.0:
            raise ValueError("wall_viscosity must be non-negative")
        if self.beach_slip_length <= 0.0 or self.boulder_slip_length <= 0.0:
            raise ValueError("wall slip lengths must be positive")
        if self.law not in WALL_LAWS:
            raise ValueError(f"wall law must be one of {WALL_LAWS}")
        if self.law == "log" and self.roughness_length <= 0.0:
            raise ValueError("the log law needs a positive roughness length")

    @property
    def law_code(self) -> int:
        return WALL_LAWS.index(self.law)


@dataclass(frozen=True)
class WallStressResult:
    """Macroscopic wall work from one split step."""

    energy_loss: float
    active_cells: int
    maximum_damping_rate: float
    numerical_loss: float = 0.0
    iterations: int = 0
    relative_residual: float = 0.0


@wp.func
def _wall_neighbor(
    index: int,
    count: int,
    periodic: int,
) -> int:
    return _wrapped_cell_index(index, count, periodic)


@wp.func
def _cell_centre_velocity(
    u: wp.array3d(dtype=wp.float32),
    v: wp.array3d(dtype=wp.float32),
    w: wp.array3d(dtype=wp.float32),
    I: int,
    J: int,
    K: int,
) -> wp.vec3:
    return wp.vec3(
        wp.float32(0.5) * (u[I, J, K] + u[I + 1, J, K]),
        wp.float32(0.5) * (v[I, J, K] + v[I, J + 1, K]),
        wp.float32(0.5) * (w[I, J, K] + w[I, J, K + 1]),
    )


@wp.func
def _wall_geometry(
    theta: wp.array3d(dtype=wp.float32), material: wp.array3d(dtype=wp.int32),
    i: int, j: int, k: int, velocity: wp.vec3,
    wall_viscosity: wp.float32, beach_slip: wp.float32, boulder_slip: wp.float32,
    law: int, roughness: wp.float32,
    inverse_dx: wp.float32, inverse_dy: wp.float32, inverse_dz: wp.float32,
    wall_distance: wp.float32, nx: int, ny: int, nz: int,
    periodic_x: int, periodic_y: int, periodic_z: int, ng: int,
) -> wp.vec4:
    I, J, K = i+ng, j+ng, k+ng
    txlo = theta[I, J, K]
    txhi = txlo
    tylo = txlo
    tyhi = txlo
    tzlo = txlo
    tzhi = txlo
    selected_material = material[I, J, K]
    proximity = wp.float32(1.0) - theta[I, J, K]
    for direction in range(6):
        ni = i
        nj = j
        nk = k
        if direction == 0:
            ni -= 1
        elif direction == 1:
            ni += 1
        elif direction == 2:
            nj -= 1
        elif direction == 3:
            nj += 1
        elif direction == 4:
            nk -= 1
        else:
            nk += 1
        ni = _wall_neighbor(ni, nx, periodic_x)
        nj = _wall_neighbor(nj, ny, periodic_y)
        nk = _wall_neighbor(nk, nz, periodic_z)
        if ni >= 0 and nj >= 0 and nk >= 0:
            value = theta[ni + ng, nj + ng, nk + ng]
            if direction == 0:
                txlo = value
            elif direction == 1:
                txhi = value
            elif direction == 2:
                tylo = value
            elif direction == 3:
                tyhi = value
            elif direction == 4:
                tzlo = value
            else:
                tzhi = value
            solid_weight = wp.float32(1.0) - value
            if solid_weight > proximity:
                proximity = solid_weight
                selected_material = material[ni + ng, nj + ng, nk + ng]

    gradient = wp.vec3(
        wp.float32(0.5) * (txhi - txlo) * inverse_dx,
        wp.float32(0.5) * (tyhi - tylo) * inverse_dy,
        wp.float32(0.5) * (tzhi - tzlo) * inverse_dz,
    )
    length = wp.length(gradient)
    if proximity <= wp.float32(0.0) or length <= wp.float32(1.0e-8):
        return wp.vec4(wp.float32(0.0))

    normal = gradient / length
    slip = beach_slip
    if selected_material == BOULDER_MATERIAL_ID:
        slip = boulder_slip
    rate = wall_viscosity * proximity / (slip * wall_distance)
    if law == 1 and selected_material != BOULDER_MATERIAL_ID:
        # Fully rough log law at the first cell's centre height: the damping
        # rate of a sheet one wall distance thick under tau = rho (kappa U / ln(z1/z0))^2,
        # capped at kappa^2 U / z1 where the cell centre sits inside the roughness.
        tangent = velocity - wp.dot(velocity, normal) * normal
        log = wp.max(wp.log(wall_distance / roughness), wp.float32(1.0))
        kappa = wp.float32(KARMAN)
        rate = kappa * kappa * wp.length(tangent) * proximity / (log * log * wall_distance)
    return wp.vec4(normal[0], normal[1], normal[2], rate)


@wp.kernel
def _compute_wall_delta(
    u: wp.array3d(dtype=wp.float32),
    v: wp.array3d(dtype=wp.float32),
    w: wp.array3d(dtype=wp.float32),
    alpha: wp.array3d(dtype=wp.float32),
    theta: wp.array3d(dtype=wp.float32),
    material: wp.array3d(dtype=wp.int32),
    delta: wp.array3d(dtype=wp.vec3),
    dissipation: wp.array3d(dtype=wp.float32),
    active_cells: wp.array(dtype=wp.int32),
    maximum_rate: wp.array(dtype=wp.float32),
    dt: wp.float32,
    wall_viscosity: wp.float32,
    beach_slip: wp.float32,
    boulder_slip: wp.float32,
    law: int,
    roughness: wp.float32,
    inverse_dx: wp.float32,
    inverse_dy: wp.float32,
    inverse_dz: wp.float32,
    wall_distance: wp.float32,
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
    if alpha[I, J, K] <= wp.float32(0.0) or theta[I, J, K] <= wp.float32(0.0):
        delta[I, J, K] = wp.vec3(wp.float32(0.0))
        dissipation[I, J, K] = wp.float32(0.0)
        return

    velocity = _cell_centre_velocity(u, v, w, I, J, K)
    geometry = _wall_geometry(theta, material, i, j, k, velocity, wall_viscosity,
        beach_slip, boulder_slip, law, roughness, inverse_dx, inverse_dy, inverse_dz,
        wall_distance, nx, ny, nz, periodic_x, periodic_y, periodic_z, ng)
    normal = wp.vec3(geometry[0], geometry[1], geometry[2])
    rate = geometry[3]
    if rate == wp.float32(0.0):
        delta[I, J, K] = wp.vec3(wp.float32(0.0))
        dissipation[I, J, K] = wp.float32(0.0)
        return
    factor = dt * rate / (wp.float32(1.0) + dt * rate)
    tangent = velocity - wp.dot(velocity, normal) * normal
    change = -factor * tangent
    delta[I, J, K] = change
    updated = velocity + change
    loss = wp.float32(0.5) * rho * alpha[I, J, K] * theta[I, J, K] * (
        wp.dot(velocity, velocity) - wp.dot(updated, updated)
    )
    dissipation[I, J, K] = loss / dt
    wp.atomic_add(active_cells, 0, 1)
    wp.atomic_max(maximum_rate, 0, rate)


@wp.func
def _cell_delta_component(
    delta: wp.array3d(dtype=wp.vec3),
    i: int,
    j: int,
    k: int,
    component: int,
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
    return delta[ci + ng, cj + ng, ck + ng][component]


@wp.kernel
def _apply_wall_delta_to_faces(
    velocity: wp.array3d(dtype=wp.float32),
    mass: wp.array3d(dtype=wp.float32),
    delta: wp.array3d(dtype=wp.vec3),
    updated: wp.array3d(dtype=wp.float32),
    energy_loss: wp.array(dtype=wp.float64),
    kinetic_energy_before: wp.array(dtype=wp.float64),
    component: int,
    rho_cell_volume: wp.float64,
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
    face_mass = mass[I, J, K]
    before = velocity[I, J, K]
    if face_mass <= wp.float32(1.0e-6):
        updated[I, J, K] = before
        return
    lower_i = i - int(component == 0)
    lower_j = j - int(component == 1)
    lower_k = k - int(component == 2)
    change = wp.float32(0.5) * (
        _cell_delta_component(
            delta, lower_i, lower_j, lower_k, component,
            nx, ny, nz, periodic_x, periodic_y, periodic_z, ng,
        )
        + _cell_delta_component(
            delta, i, j, k, component,
            nx, ny, nz, periodic_x, periodic_y, periodic_z, ng,
        )
    )
    after = before + change
    updated[I, J, K] = after
    q = i
    count = nx
    periodic = periodic_x
    if component == 1:
        q = j
        count = ny
        periodic = periodic_y
    elif component == 2:
        q = k
        count = nz
        periodic = periodic_z
    if periodic == 0 or q != count:
        before_energy = (
            wp.float64(wp.float32(0.5) * face_mass * before * before)
            * rho_cell_volume
        )
        wp.atomic_add(
            energy_loss, 0,
            wp.float64(wp.float32(0.5) * face_mass
                       * (before * before - after * after)) * rho_cell_volume,
        )
        wp.atomic_add(kinetic_energy_before, 0, before_energy)


class WallStressModel:
    """Apply material-dependent tangential stress near embedded solids."""

    __slots__ = (
        "cfg", "settings", "device", "wall_delta", "dissipation",
        "_active_cells", "_maximum_rate", "_energy_loss",
        "_kinetic_energy_before", "_updated",
    )

    def __init__(
        self,
        cfg: Solver3DConfig,
        settings: WallStressConfig | None = None,
    ):
        self.cfg = cfg
        self.settings = WallStressConfig() if settings is None else settings
        self.device = cfg.device
        cell_shape = tuple(n + 2 * cfg.grid.ng for n in cfg.grid.shape)
        self.wall_delta = wp.zeros(cell_shape, dtype=wp.vec3, device=self.device)
        self.dissipation = wp.zeros(
            cell_shape, dtype=wp.float32, device=self.device,
        )
        self._active_cells = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._maximum_rate = wp.zeros(1, dtype=wp.float32, device=self.device)
        self._energy_loss = wp.zeros(1, dtype=wp.float64, device=self.device)
        self._kinetic_energy_before = wp.zeros(
            1, dtype=wp.float64, device=self.device,
        )
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

    @property
    def enabled(self) -> bool:
        return self.settings.wall_viscosity > 0.0 or self.settings.law == "log"

    def apply(self, state: State3D, dt: float, *, liquid_operator=None) -> WallStressResult:
        if self.cfg.pressure_discretization == "liquid_mimetic" and self.enabled:
            from shoreloop.physics3d.liquid_wall import apply_liquid_wall_stress
            return apply_liquid_wall_stress(self, state, dt, pressure=liquid_operator)
        self.wall_delta.zero_()
        self.dissipation.zero_()
        self._active_cells.zero_()
        self._maximum_rate.zero_()
        self._energy_loss.zero_()
        self._kinetic_energy_before.zero_()
        g = self.cfg.grid
        periodic = (
            int(self.cfg.bc.xlo == BC.PERIODIC),
            int(self.cfg.bc.ylo == BC.PERIODIC),
            int(self.cfg.bc.zlo == BC.PERIODIC),
        )
        wp.launch(
            _compute_wall_delta,
            dim=g.shape,
            inputs=[
                state.u, state.v, state.w, state.alpha, state.theta,
                state.material, self.wall_delta, self.dissipation,
                self._active_cells, self._maximum_rate, dt,
                self.settings.wall_viscosity,
                self.settings.beach_slip_length,
                self.settings.boulder_slip_length,
                self.settings.law_code, self.settings.roughness_length,
                1.0 / g.dx, 1.0 / g.dy, 1.0 / g.dz,
                0.5 * min(g.dx, g.dy, g.dz), self.cfg.physics.rho,
                g.nx, g.ny, g.nz, *periodic, g.ng,
            ],
            device=self.device,
        )
        rho_cell_volume = self.cfg.physics.rho * g.dx * g.dy * g.dz
        for component, name in enumerate(("u", "v", "w")):
            mass = compute_face_liquid_mass(state, name)
            target = self._updated[component]
            target.zero_()
            wp.launch(
                _apply_wall_delta_to_faces,
                dim=state.grid.interior_shape(name),
                inputs=[
                    state.field(name), mass, self.wall_delta, target,
                    self._energy_loss, self._kinetic_energy_before,
                    component, rho_cell_volume,
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
        energy_loss = float(self._energy_loss.numpy()[0])
        kinetic_energy_before = float(self._kinetic_energy_before.numpy()[0])
        tolerance = _FLOAT32_ENERGY_TOLERANCE * kinetic_energy_before
        if energy_loss < -tolerance:
            raise FloatingPointError(
                f"wall stress injected {-energy_loss:.6g} J of kinetic energy",
            )
        energy_loss = max(0.0, energy_loss)
        return WallStressResult(
            energy_loss=energy_loss,
            active_cells=int(self._active_cells.numpy()[0]),
            maximum_damping_rate=float(self._maximum_rate.numpy()[0]),
        )
