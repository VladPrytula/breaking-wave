"""Mechanical-energy accounting for the local three-dimensional solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from shoreloop.config import BC, Solver3DConfig
from shoreloop.physics3d.momentum import compute_face_liquid_mass
from shoreloop.physics3d.state import State3D
from shoreloop.physics3d.surface_tension import measure_plic_surface_area


@dataclass(frozen=True)
class MechanicalEnergy:
    kinetic: float
    gravitational: float
    surface: float

    @property
    def total(self) -> float:
        return self.kinetic + self.gravitational + self.surface


@dataclass(frozen=True)
class EnergyBudgetStep:
    """Resolved change, direct work, and labelled discrete remainder."""

    initial: MechanicalEnergy
    final: MechanicalEnergy
    boundary_work: float
    boundary_surface_work: float
    molecular_loss: float
    sgs_loss: float
    wall_loss: float
    porous_loss: float
    capillary_exchange: float
    numerical_residual: float
    # New cumulative signed-loss high exposed to foam by this step.
    positive_numerical_loss: float

    @property
    def direct_irreversible_loss(self) -> float:
        return (
            self.molecular_loss + self.sgs_loss
            + self.wall_loss + self.porous_loss
        )


@wp.kernel
def _form_face_kinetic_energy(
    velocity: wp.array3d(dtype=wp.float32),
    mass: wp.array3d(dtype=wp.float32),
    energy: wp.array3d(dtype=wp.float64),
    component: int,
    periodic: int,
    count: int,
    rho_cell_volume: wp.float64,
    ng: int,
):
    i, j, k = wp.tid()
    q = i
    if component == 1:
        q = j
    elif component == 2:
        q = k
    if periodic != 0 and q == count:
        energy[i, j, k] = wp.float64(0.0)
        return
    I = i + ng
    J = j + ng
    K = k + ng
    value = velocity[I, J, K]
    energy[i, j, k] = (
        wp.float64(wp.float32(0.5) * mass[I, J, K] * value * value)
        * rho_cell_volume
    )


@wp.kernel
def _form_gravitational_energy(
    alpha: wp.array3d(dtype=wp.float32),
    theta: wp.array3d(dtype=wp.float32),
    energy: wp.array3d(dtype=wp.float64),
    rho_g_cell_volume: wp.float64,
    z0: wp.float64,
    dz: wp.float64,
    ng: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    height = z0 + (wp.float64(k) + wp.float64(0.5)) * dz
    energy[i, j, k] = (
        wp.float64(alpha[I, J, K] * theta[I, J, K])
        * rho_g_cell_volume * height
    )


class EnergyLedger:
    """Measure direct closure work and the remaining signed discrete loss."""

    __slots__ = (
        "cfg", "device", "_kinetic_fields", "_gravitational_field",
        "_totals", "_initial",
        "molecular_loss", "sgs_loss", "wall_loss", "porous_loss",
        "capillary_exchange", "numerical_residual",
        "positive_numerical_loss", "boundary_work",
        "boundary_surface_work", "steps",
    )

    def __init__(self, cfg: Solver3DConfig):
        self.cfg = cfg
        self.device = cfg.device
        self._kinetic_fields = tuple(
            wp.zeros(
                shape,
                dtype=wp.float64,
                device=self.device,
            )
            for shape in (
                (cfg.grid.nx + 1, cfg.grid.ny, cfg.grid.nz),
                (cfg.grid.nx, cfg.grid.ny + 1, cfg.grid.nz),
                (cfg.grid.nx, cfg.grid.ny, cfg.grid.nz + 1),
            )
        )
        self._gravitational_field = wp.zeros(
            cfg.grid.shape, dtype=wp.float64, device=self.device,
        )
        self._totals = wp.zeros(4, dtype=wp.float64, device=self.device)
        self._initial: MechanicalEnergy | None = None
        self.molecular_loss = 0.0
        self.sgs_loss = 0.0
        self.wall_loss = 0.0
        self.porous_loss = 0.0
        self.capillary_exchange = 0.0
        self.numerical_residual = 0.0
        self.positive_numerical_loss = 0.0
        self.boundary_work = 0.0
        self.boundary_surface_work = 0.0
        self.steps = 0

    def measure(self, state: State3D, *, liquid_operator=None) -> MechanicalEnergy:
        """Measure current velocity; a supplied operator must match its geometry."""
        g = self.cfg.grid
        if self.cfg.pressure_discretization == "liquid_mimetic":
            from shoreloop.physics3d.liquid_pressure import LiquidPressureOperator
            operator = (LiquidPressureOperator(state, velocity_space="mac")
                        if liquid_operator is None else liquid_operator)
            kinetic = operator.kinetic_energy(operator.read_mac_state(state))
            # Retained alpha*theta owns mass; the reconstructed water centroid
            # locates gravitational energy. Sub-tolerance traces retain the
            # reconstruction's declared cell-centre location and are counted.
            centroid_z = operator.centroids[..., 2]
            z = g.z0 + (np.arange(g.nz)[None, None, :] + centroid_z) * g.dz
            gravitational = (self.cfg.physics.rho*self.cfg.physics.g*g.dx*g.dy*g.dz
                             *float(np.sum(state.get_interior("alpha").astype(float)
                                           *state.get_geometry_interior("theta")*z)))
            return MechanicalEnergy(kinetic, gravitational, 0.0)
        periodic = (
            int(self.cfg.bc.xlo == BC.PERIODIC),
            int(self.cfg.bc.ylo == BC.PERIODIC),
            int(self.cfg.bc.zlo == BC.PERIODIC),
        )
        rho_cell_volume = self.cfg.physics.rho * g.dx * g.dy * g.dz
        for component, name in enumerate(("u", "v", "w")):
            mass = compute_face_liquid_mass(state, name)
            wp.launch(
                _form_face_kinetic_energy,
                dim=state.grid.interior_shape(name),
                inputs=[
                    state.field(name), mass, self._kinetic_fields[component],
                    component, periodic[component], g.shape[component],
                    rho_cell_volume, g.ng,
                ],
                device=self.device,
            )
            wp.utils.array_sum(
                self._kinetic_fields[component],
                out=self._totals[component:component + 1],
            )
        if self.cfg.physics.g > 0.0:
            wp.launch(
                _form_gravitational_energy,
                dim=g.shape,
                inputs=[
                    state.alpha, state.theta, self._gravitational_field,
                    rho_cell_volume * self.cfg.physics.g,
                    g.z0, g.dz, g.ng,
                ],
                device=self.device,
            )
            wp.utils.array_sum(
                self._gravitational_field,
                out=self._totals[3:4],
            )
        totals = self._totals.numpy()
        surface = 0.0
        if self.cfg.physics.surface_tension > 0.0:
            surface = (
                self.cfg.physics.surface_tension
                * measure_plic_surface_area(state)
            )
        return MechanicalEnergy(
            kinetic=float(totals[:3].sum()),
            gravitational=float(totals[3]),
            surface=surface,
        )

    def begin_step(self, state: State3D, *, liquid_operator=None) -> MechanicalEnergy:
        self._initial = self.measure(state, liquid_operator=liquid_operator)
        return self._initial

    def close_step(
        self,
        state: State3D,
        *,
        boundary_work: float = 0.0,
        boundary_surface_work: float = 0.0,
        molecular_loss: float = 0.0,
        sgs_loss: float = 0.0,
        wall_loss: float = 0.0,
        porous_loss: float = 0.0,
        final_energy: MechanicalEnergy | None = None,
    ) -> EnergyBudgetStep:
        initial = self._initial
        final = self.measure(state) if final_energy is None else final_energy
        direct = molecular_loss + sgs_loss + wall_loss + porous_loss
        capillary_exchange = initial.surface - final.surface
        residual = initial.total + boundary_work - final.total - direct
        cumulative_residual = self.numerical_residual + residual
        positive_loss = max(
            0.0, cumulative_residual - self.positive_numerical_loss,
        )
        result = EnergyBudgetStep(
            initial=initial,
            final=final,
            boundary_work=boundary_work,
            boundary_surface_work=boundary_surface_work,
            molecular_loss=molecular_loss,
            sgs_loss=sgs_loss,
            wall_loss=wall_loss,
            porous_loss=porous_loss,
            capillary_exchange=capillary_exchange,
            numerical_residual=residual,
            positive_numerical_loss=positive_loss,
        )
        self.molecular_loss += molecular_loss
        self.sgs_loss += sgs_loss
        self.wall_loss += wall_loss
        self.porous_loss += porous_loss
        self.capillary_exchange += capillary_exchange
        self.numerical_residual = cumulative_residual
        self.positive_numerical_loss += positive_loss
        self.boundary_work += boundary_work
        self.boundary_surface_work += boundary_surface_work
        self.steps += 1
        self._initial = None
        return result
