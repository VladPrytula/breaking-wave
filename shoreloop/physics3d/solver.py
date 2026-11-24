"""First fractional-step driver for a filled regular-grid box."""

from __future__ import annotations

import math
from dataclasses import replace

import warp as wp

from shoreloop.config import Solver3DConfig
from shoreloop.physics3d.energy import EnergyBudgetStep, EnergyLedger
from shoreloop.physics3d.foam import (Foam3DConfig, Foam3DResult,
                                      InterfaceFoamModel)
from shoreloop.physics3d.free_surface import (FreeSurfaceProjection,
                                              FreeSurfaceProjectionResult)
from shoreloop.physics3d.contact_line import (ContactLineConfig,
                                              ContactLineModel,
                                              ContactLineResult)
from shoreloop.physics3d.momentum import (MomentumAdvectionResult,
                                          MomentumAdvector)
from shoreloop.physics3d.liquid_pressure import LiquidPressureOperator, LiquidPressureProjection
from shoreloop.physics3d.unsplit_advection import UnsplitVOFAdvector
from shoreloop.physics3d.projection import ProjectionConfig
from shoreloop.physics3d.porous import (PorousBedModel, PorousDragResult,
                                        PorousStorageResult)
from shoreloop.physics3d.relaxation import (WaveRelaxationResult,
                                            WaveRelaxationZone)
from shoreloop.physics3d.solids import enforce_voxel_solids
from shoreloop.physics3d.solid_wetness import (
    SolidWetnessConfig, SolidWetnessModel, SolidWetnessResult,
)
from shoreloop.physics3d.state import State3D
from shoreloop.physics3d.timestep import (TimeStepConfig, TimeStepController,
                                          TimeStepResult)
from shoreloop.physics3d.viscosity import (ViscosityConfig, ViscosityModel,
                                           ViscosityResult)
from shoreloop.physics3d.wall_stress import (WallStressConfig, WallStressModel,
                                             WallStressResult)
from shoreloop.physics3d.wet_dry import WispRemover, WispRemovalResult


@wp.kernel
def _add_active_liquid_acceleration(
    velocity: wp.array3d(dtype=wp.float32),
    alpha: wp.array3d(dtype=wp.float32),
    increment: wp.float32,
    active_threshold: wp.float32,
    nz: int,
    ng: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    active = int(0)
    if k > 0 and alpha[I, J, K - 1] >= active_threshold:
        active = 1
    if k < nz and alpha[I, J, K] >= active_threshold:
        active = 1
    # Apply body force on the same water faces corrected by the pressure solve.
    # Velocities beyond that domain are supplied by narrow-band extension.
    if active != 0:
        velocity[I, J, K] += increment


@wp.kernel
def _mark_non_finite(
    p: wp.array3d(dtype=wp.float32),
    alpha: wp.array3d(dtype=wp.float32),
    u: wp.array3d(dtype=wp.float32),
    v: wp.array3d(dtype=wp.float32),
    w: wp.array3d(dtype=wp.float32),
    flag: wp.array(dtype=wp.int32),
    nx: int,
    ny: int,
    nz: int,
    ng: int,
):
    i, j, k = wp.tid()
    invalid = int(0)
    if i < nx and j < ny and k < nz:
        if not wp.isfinite(p[i + ng, j + ng, k + ng]):
            invalid = 1
        if not wp.isfinite(alpha[i + ng, j + ng, k + ng]):
            invalid = 1
    if i <= nx and j < ny and k < nz:
        if not wp.isfinite(u[i + ng, j + ng, k + ng]):
            invalid = 1
    if i < nx and j <= ny and k < nz:
        if not wp.isfinite(v[i + ng, j + ng, k + ng]):
            invalid = 1
    if i < nx and j < ny and k <= nz:
        if not wp.isfinite(w[i + ng, j + ng, k + ng]):
            invalid = 1
    if invalid != 0:
        wp.atomic_max(flag, 0, 1)


@wp.kernel
def _mark_alpha_difference(
    alpha: wp.array3d(dtype=wp.float32),
    reference: wp.array3d(dtype=wp.float32),
    flag: wp.array(dtype=wp.int32),
    ng: int,
):
    i, j, k = wp.tid()
    if alpha[i + ng, j + ng, k + ng] != reference[i + ng, j + ng, k + ng]:
        wp.atomic_max(flag, 0, 1)


class Solver3D:
    """Advance body force and pressure projection in a filled closed box."""

    __slots__ = (
        "cfg", "device", "state", "projection", "time", "steps",
        "timestep", "advection", "wave_relaxation", "last_timestep",
        "wisp_remover", "last_advection", "last_relaxation",
        "last_wisp_removal", "viscosity", "last_viscosity", "wall_stress",
        "last_wall_stress", "contact_line", "last_contact_line", "_non_finite",
        "porous_bed", "last_porous_storage", "last_porous_drag",
        "energy_ledger", "last_energy",
        "foam", "last_foam",
        "solid_wetness", "last_solid_wetness",
        "_liquid_operator_cache", "_liquid_operator_alpha", "_alpha_changed",
    )

    def __init__(
        self,
        cfg: Solver3DConfig | None = None,
        projection_settings: ProjectionConfig | None = None,
        timestep_settings: TimeStepConfig | None = None,
        viscosity_settings: ViscosityConfig | None = None,
        wall_stress_settings: WallStressConfig | None = None,
        contact_line_settings: ContactLineConfig | None = None,
        foam_settings: Foam3DConfig | None = None,
        solid_wetness_settings: SolidWetnessConfig | None = None,
    ):
        self.cfg = Solver3DConfig() if cfg is None else cfg
        self.device = self.cfg.device
        self.state = State3D(self.cfg)
        self.projection = (
            LiquidPressureProjection(self.cfg, projection_settings)
            if self.cfg.pressure_discretization == "liquid_mimetic"
            else FreeSurfaceProjection(self.cfg, projection_settings)
        )
        self.timestep = TimeStepController(self.cfg, timestep_settings)
        self.advection = MomentumAdvector(self.cfg)
        if self.cfg.pressure_discretization == "liquid_mimetic":
            self.advection.vof = UnsplitVOFAdvector(self.cfg)
        self.wave_relaxation: WaveRelaxationZone | None = None
        self.wisp_remover: WispRemover | None = None
        self.viscosity = (
            None if viscosity_settings is None
            else ViscosityModel(self.cfg, viscosity_settings)
        )
        self.wall_stress = (
            None if wall_stress_settings is None
            else WallStressModel(self.cfg, wall_stress_settings)
        )
        self.contact_line = (
            None if contact_line_settings is None
            else ContactLineModel(self.cfg, contact_line_settings)
        )
        self.porous_bed: PorousBedModel | None = None
        self.energy_ledger: EnergyLedger | None = None
        self.foam = (
            None if foam_settings is None
            else InterfaceFoamModel(self.cfg, foam_settings)
        )
        self.solid_wetness = (
            None if solid_wetness_settings is None
            else SolidWetnessModel(self.cfg, solid_wetness_settings)
        )
        self.last_timestep: TimeStepResult | None = None
        self.last_advection: MomentumAdvectionResult | None = None
        self.last_relaxation: WaveRelaxationResult | None = None
        self.last_wisp_removal: WispRemovalResult | None = None
        self.last_viscosity: ViscosityResult | None = None
        self.last_wall_stress: WallStressResult | None = None
        self.last_contact_line: ContactLineResult | None = None
        self.last_porous_storage: PorousStorageResult | None = None
        self.last_porous_drag: PorousDragResult | None = None
        self.last_energy: EnergyBudgetStep | None = None
        self.last_foam: Foam3DResult | None = None
        self.last_solid_wetness: SolidWetnessResult | None = None
        self.time = 0.0
        self.steps = 0
        self._non_finite = wp.zeros(
            1, dtype=wp.int32, device=self.device,
        )
        self._liquid_operator_cache: LiquidPressureOperator | None = None
        self._liquid_operator_alpha: wp.array | None = None
        self._alpha_changed = wp.zeros(1, dtype=wp.int32, device=self.device)

    def liquid_operator(self) -> LiquidPressureOperator:
        """Return the liquid operator assembled on the current water geometry.

        The operator depends only on alpha and the static solid geometry;
        velocities are read when it is used. It is rebuilt exactly when alpha
        differs bitwise from the field it was assembled on, which also covers
        writes to the state between steps. Assembly is host work of about
        45 s on the production grid, so this is the measured bottleneck.
        """
        g = self.cfg.grid
        if self._liquid_operator_cache is not None:
            self._alpha_changed.zero_()
            wp.launch(
                _mark_alpha_difference, dim=g.shape,
                inputs=[self.state.alpha, self._liquid_operator_alpha,
                        self._alpha_changed, g.ng],
                device=self.device,
            )
            if int(self._alpha_changed.numpy()[0]) == 0:
                return self._liquid_operator_cache
        operator = LiquidPressureOperator(self.state, velocity_space="mac")
        if self._liquid_operator_alpha is None:
            self._liquid_operator_alpha = wp.empty_like(self.state.alpha)
        wp.copy(self._liquid_operator_alpha, self.state.alpha)
        self._liquid_operator_cache = operator
        return operator

    def _state_is_finite(self) -> bool:
        self._non_finite.zero_()
        g = self.cfg.grid
        wp.launch(
            _mark_non_finite,
            dim=(g.nx + 1, g.ny + 1, g.nz + 1),
            inputs=[self.state.p, self.state.alpha, self.state.u, self.state.v,
                    self.state.w, self._non_finite, g.nx, g.ny, g.nz, g.ng],
            device=self.device,
        )
        return int(self._non_finite.numpy()[0]) == 0

    def _apply_gravity(self, dt: float) -> None:
        if self.cfg.physics.g == 0.0:
            return
        wp.launch(
            _add_active_liquid_acceleration,
            dim=self.state.grid.interior_shape("w"),
            inputs=[self.state.w, self.state.alpha,
                    -self.cfg.physics.g * dt, 0.5, self.cfg.grid.nz,
                    self.cfg.grid.ng],
            device=self.device,
        )

    def compute_dt(self, *, liquid_operator=None) -> TimeStepResult:
        """Return the current physical time-step limits and their winner."""
        result = self.timestep.compute(self.state)
        transport_limit = self.advection.transport_time_step(
            self.state, self.timestep.settings.advective_cfl,
        )
        advective_limit = min(result.advective_limit, transport_limit)
        if self.cfg.pressure_discretization == "liquid_mimetic":
            # Dry donors carry no VOF flux and the transport limit covers every
            # face the flux Courant check enforces. The all-face scan also reads
            # stale air faces with the smallest spacing on every axis, so it is
            # only the logged max_face_speed diagnostic on this path.
            winner, dt = min((
                ("maximum", self.timestep.settings.dt_max),
                ("interface_advection", transport_limit),
                ("gravity", result.gravity_limit),
                ("viscous", result.viscous_limit),
                ("capillary", result.capillary_limit),
            ), key=lambda item: item[1])
            result = TimeStepResult(
                dt, winner, result.max_face_speed, transport_limit,
                result.gravity_limit, result.viscous_limit,
                result.capillary_limit,
            )
        elif transport_limit < result.dt:
            result = TimeStepResult(
                transport_limit, "interface_advection", result.max_face_speed,
                advective_limit, result.gravity_limit, result.viscous_limit,
                result.capillary_limit,
            )
        elif advective_limit != result.advective_limit:
            result = TimeStepResult(
                result.dt, result.limiting_constraint, result.max_face_speed,
                advective_limit, result.gravity_limit, result.viscous_limit,
                result.capillary_limit,
            )
        viscous_limit = result.viscous_limit
        if (liquid_operator is None and self.viscosity is not None and self.viscosity.enabled
                and self.cfg.pressure_discretization == "liquid_mimetic"):
            # The cached operator is current between steps; do not rebuild for the cap.
            liquid_operator = self.liquid_operator()
        if self.viscosity is not None and self.viscosity.enabled:
            viscous_limit = min(
                viscous_limit,
                self.viscosity.time_step_limit(
                    self.state, self.timestep.settings.viscous_cfl,
                    liquid_operator=liquid_operator,
                ),
            )
        if self.contact_line is not None and self.contact_line.enabled:
            viscous_limit = min(
                viscous_limit,
                self.contact_line.time_step_limit(
                    self.timestep.settings.viscous_cfl,
                ),
            )
        if math.isinf(viscous_limit):
            return result
        if viscous_limit < result.dt:
            return TimeStepResult(
                viscous_limit, "viscous", result.max_face_speed,
                result.advective_limit, result.gravity_limit, viscous_limit,
                result.capillary_limit,
            )
        return TimeStepResult(
            result.dt, result.limiting_constraint, result.max_face_speed,
            result.advective_limit, result.gravity_limit, viscous_limit,
            result.capillary_limit,
        )

    def set_wave_relaxation(self, zone: WaveRelaxationZone) -> None:
        if zone.cfg != self.cfg:
            raise ValueError("wave relaxation configuration must match the solver")
        self.wave_relaxation = zone

    def set_wisp_remover(self, remover: WispRemover) -> None:
        if remover.cfg != self.cfg:
            raise ValueError("wisp-removal configuration must match the solver")
        self.wisp_remover = remover

    def set_viscosity(self, model: ViscosityModel) -> None:
        if model.cfg != self.cfg:
            raise ValueError("viscosity configuration must match the solver")
        self.viscosity = model

    def set_wall_stress(self, model: WallStressModel) -> None:
        if model.cfg != self.cfg:
            raise ValueError("wall-stress configuration must match the solver")
        self.wall_stress = model

    def set_contact_line(self, model: ContactLineModel) -> None:
        if model.cfg != self.cfg:
            raise ValueError("contact-line configuration must match the solver")
        self.contact_line = model

    def set_porous_bed(self, model: PorousBedModel) -> None:
        if model.cfg != self.cfg:
            raise ValueError("porous-bed configuration must match the solver")
        self.porous_bed = model

    def set_energy_ledger(self, ledger: EnergyLedger) -> None:
        if ledger.cfg != self.cfg:
            raise ValueError("energy-ledger configuration must match the solver")
        self.energy_ledger = ledger

    def set_foam(self, model: InterfaceFoamModel) -> None:
        if model.cfg != self.cfg:
            raise ValueError("foam configuration must match the solver")
        self.foam = model

    def set_solid_wetness(self, model: SolidWetnessModel) -> None:
        if model.cfg != self.cfg:
            raise ValueError("solid-wetness configuration must match the solver")
        self.solid_wetness = model

    def step(
        self,
        dt: float | None = None,
        *,
        timestep: TimeStepResult | None = None,
    ) -> FreeSurfaceProjectionResult:
        """Advance one step, optionally reusing its immediately computed limit."""
        if not self._state_is_finite():
            raise FloatingPointError("non-finite 3-D state before step")
        enforce_voxel_solids(self.state)

        # Reuse geometry already needed by the initial energy/SGS measurement
        # for the initial PLIC transport as well. Reuse stays inside this call.
        initial_operator = None
        if (self.cfg.pressure_discretization == "liquid_mimetic"
                and (self.energy_ledger is not None
                     or (timestep is None and self.viscosity is not None
                         and self.viscosity.settings.smagorinsky_cs > 0))):
            initial_operator = self.liquid_operator()
        if timestep is None:
            timestep = self.compute_dt(liquid_operator=initial_operator)
        if dt is None:
            dt = timestep.dt
        elif not math.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"dt must be positive and finite, got {dt!r}")
        elif dt > timestep.dt * (1.0 + 1.0e-6):
            raise ValueError(
                f"dt={dt:.6g} exceeds {timestep.limiting_constraint} limit "
                f"{timestep.dt:.6g}",
            )

        self.last_energy = None
        if self.energy_ledger is not None:
            self.energy_ledger.begin_step(self.state, liquid_operator=initial_operator)
        boundary_work = 0.0
        boundary_surface_work = 0.0

        reconstruction = initial_operator.reconstruction if initial_operator is not None else None
        initial_operator = None
        self.last_advection = self.advection.advect(self.state, dt, reconstruction=reconstruction)
        reconstruction = None
        self.last_contact_line = None
        if self.contact_line is not None and self.contact_line.enabled:
            self.last_contact_line = self.contact_line.apply(self.state, dt)
        self.last_porous_storage = None
        # Diffusion, drag and projection change velocity on fixed water
        # geometry. The operator is reused while alpha is bitwise unchanged and
        # rebuilt after any alpha-changing operation.
        liquid_operator = None
        if (self.cfg.pressure_discretization == "liquid_mimetic"
                and ((self.viscosity is not None and self.viscosity.enabled)
                     or (self.wall_stress is not None and self.wall_stress.enabled)
                     or (self.porous_bed is not None and self.porous_bed.drag_enabled)
                     or (self.energy_ledger is not None
                         and (self.wave_relaxation is not None
                              or self.wisp_remover is not None)))):
            liquid_operator = self.liquid_operator()
        self.last_viscosity = None
        if self.viscosity is not None and self.viscosity.enabled:
            self.last_viscosity = self.viscosity.apply(self.state, dt, liquid_operator=liquid_operator)
        closure_operator = liquid_operator
        self.last_wall_stress = None
        if self.wall_stress is not None and self.wall_stress.enabled:
            before_wall = (closure_operator.kinetic_energy(closure_operator.read_mac_state(self.state))
                           if closure_operator is not None else 0.0)
            self.last_wall_stress = self.wall_stress.apply(self.state, dt, liquid_operator=closure_operator)
            if closure_operator is not None:
                after_wall = closure_operator.kinetic_energy(closure_operator.read_mac_state(self.state))
                loss = before_wall-after_wall
                if loss < -2e-6*max(before_wall, after_wall)-1e-15:
                    raise FloatingPointError(f"wall stress injected {-loss:.6g} J in the liquid kinetic metric")
        self.last_porous_drag = None
        if self.porous_bed is not None and self.porous_bed.drag_enabled:
            before_porous = (closure_operator.kinetic_energy(closure_operator.read_mac_state(self.state))
                             if closure_operator is not None else 0.0)
            self.last_porous_drag = self.porous_bed.apply_drag(self.state, dt)
            if closure_operator is not None:
                after_porous = closure_operator.kinetic_energy(closure_operator.read_mac_state(self.state))
                loss = before_porous-after_porous
                if loss < -2e-6*max(before_porous, after_porous)-1e-15:
                    raise FloatingPointError(f"porous drag injected {-loss:.6g} J in the liquid kinetic metric")
                self.last_porous_drag = replace(self.last_porous_drag, energy_loss=loss)
        closure_operator = None
        if self.cfg.pressure_discretization != "liquid_mimetic":
            self._apply_gravity(dt)
        if not self._state_is_finite():
            raise FloatingPointError("non-finite 3-D state after body force")

        # The wave source and wisp removal both change alpha and both count as
        # boundary work, so one energy difference across the pair is the same
        # sum as measuring each stage separately.
        self.last_relaxation = None
        self.last_wisp_removal = None
        source_before = None
        if (self.energy_ledger is not None
                and (self.wave_relaxation is not None or self.wisp_remover is not None)):
            source_before = self.energy_ledger.measure(self.state, liquid_operator=liquid_operator)
        if self.wave_relaxation is not None:
            self.last_relaxation = self.wave_relaxation.apply(
                self.state, self.time + dt, dt,
                gravity_increment=(-self.cfg.physics.g * dt
                                   if self.cfg.pressure_discretization != "liquid_mimetic"
                                   else 0.0),
            )
        if self.wisp_remover is not None:
            self.last_wisp_removal = self.wisp_remover.apply(self.state)
        liquid_operator = None
        if self.cfg.pressure_discretization == "liquid_mimetic":
            liquid_operator = self.liquid_operator()
        if source_before is not None:
            after_source = self.energy_ledger.measure(self.state, liquid_operator=liquid_operator)
            boundary_work += after_source.total - source_before.total
            boundary_surface_work += after_source.surface - source_before.surface

        if self.cfg.pressure_discretization == "liquid_mimetic":
            result = self.projection.project(self.state, dt, operator=liquid_operator)
            liquid_operator = self.projection.last_operator
        else:
            result = self.projection.project(self.state, dt)
        if not result.pressure_solve.converged:
            raise RuntimeError(
                "pressure solve failed: "
                f"iterations={result.pressure_solve.iterations}, "
                f"relative_residual={result.pressure_solve.relative_residual:.3e}",
            )
        if not result.converged:
            raise RuntimeError(
                "projection failed RMS divergence gate: "
                f"{result.rms_divergence_after:.3e} > "
                f"{self.projection.settings.rms_divergence_tolerance:.3e}",
            )
        if not self._state_is_finite():
            raise FloatingPointError("non-finite 3-D state after projection")

        final_energy = None
        if self.porous_bed is not None and self.porous_bed.storage_enabled:
            before_exchange = (
                self.energy_ledger.measure(self.state, liquid_operator=liquid_operator)
                if self.energy_ledger is not None else None
            )
            self.last_porous_storage = self.porous_bed.apply_storage(
                self.state, dt,
            )
            liquid_operator = (self.liquid_operator()
                               if self.cfg.pressure_discretization == "liquid_mimetic" else None)
            if before_exchange is not None:
                final_energy = self.energy_ledger.measure(self.state, liquid_operator=liquid_operator)
                boundary_work += (
                    final_energy.total
                    - before_exchange.total
                )
                boundary_surface_work += (
                    final_energy.surface
                    - before_exchange.surface
                )

        if self.energy_ledger is not None:
            if final_energy is None:
                final_energy = self.energy_ledger.measure(self.state, liquid_operator=liquid_operator)
            self.last_energy = self.energy_ledger.close_step(
                self.state,
                boundary_work=boundary_work,
                boundary_surface_work=boundary_surface_work,
                molecular_loss=(
                    self.last_viscosity.molecular_energy_loss
                    if self.last_viscosity is not None else 0.0
                ),
                sgs_loss=(
                    self.last_viscosity.sgs_energy_loss
                    if self.last_viscosity is not None else 0.0
                ),
                wall_loss=(
                    self.last_wall_stress.energy_loss
                    if self.last_wall_stress is not None else 0.0
                ),
                porous_loss=(
                    self.last_porous_drag.energy_loss
                    if self.last_porous_drag is not None else 0.0
                ),
                final_energy=final_energy,
            )

        self.last_foam = None
        if self.foam is not None:
            dissipation_fields = []
            strain_rate = None
            if self.viscosity is not None:
                strain_rate = self.viscosity.strain_rate
                dissipation_fields.extend((
                    self.viscosity.molecular_dissipation,
                    self.viscosity.sgs_dissipation,
                ))
            if self.wall_stress is not None:
                dissipation_fields.append(self.wall_stress.dissipation)
            if self.porous_bed is not None:
                dissipation_fields.append(self.porous_bed.dissipation)
            self.last_foam = self.foam.apply(
                self.state, dt,
                strain_rate=strain_rate,
                dissipation_fields=dissipation_fields,
                numerical_loss=(
                    self.last_energy.positive_numerical_loss
                    if self.last_energy is not None else 0.0
                ),
            )

        self.last_solid_wetness = None
        if self.solid_wetness is not None:
            self.last_solid_wetness = self.solid_wetness.apply(
                self.state, dt,
            )

        self.time += dt
        self.steps += 1
        self.last_timestep = timestep
        return result
