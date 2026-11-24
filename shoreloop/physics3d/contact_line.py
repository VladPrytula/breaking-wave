"""Conservative grid-scale regularisation of a moving solid contact region.

The VOF interface remains authoritative.  This closure only exchanges liquid
between already-mixed cells that lie within a declared distance of a solid,
so it smooths cell-to-cell contact chatter without creating a universal film.
Advancing and receding cells use separate diffusivities selected from the
change since the preceding accepted step.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import warp as wp

from shoreloop.config import BC, Solver3DConfig
from shoreloop.physics3d.advection import _copy_cell_field
from shoreloop.physics3d.boundary import fill_field_boundaries
from shoreloop.physics3d.extrapolation import _wrapped_cell_index
from shoreloop.physics3d.state import State3D


@dataclass(frozen=True)
class ContactLineConfig:
    """Resolved contact-band length and advancing/receding mobility."""

    contact_length: float = 0.04
    advancing_diffusivity: float = 8.0e-5
    receding_diffusivity: float = 4.0e-5
    hysteresis_fraction: float = 0.01
    mixed_tolerance: float = 1.0e-4

    def __post_init__(self) -> None:
        if self.contact_length <= 0.0:
            raise ValueError("contact_length must be positive")
        if self.advancing_diffusivity < 0.0 or self.receding_diffusivity < 0.0:
            raise ValueError("contact diffusivities must be non-negative")
        if self.hysteresis_fraction < 0.0:
            raise ValueError("hysteresis_fraction must be non-negative")


@dataclass(frozen=True)
class ContactLineResult:
    active_cells: int
    exchanged_volume: float
    volume_residual: float


@wp.kernel
def _mark_contact_band(
    alpha: wp.array3d(dtype=wp.float32),
    theta: wp.array3d(dtype=wp.float32),
    material: wp.array3d(dtype=wp.int32),
    contact: wp.array3d(dtype=wp.int32),
    active_cells: wp.array(dtype=wp.int32),
    layers: int,
    mixed_tolerance: wp.float32,
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
    fraction = alpha[I, J, K]
    if (theta[I, J, K] <= wp.float32(0.0)
            or fraction <= mixed_tolerance
            or fraction >= wp.float32(1.0) - mixed_tolerance):
        contact[I, J, K] = 0
        return
    near_solid = int(material[I, J, K] != 0 or theta[I, J, K] < wp.float32(1.0))
    for direction in range(6):
        for distance in range(1, 5):
            if distance <= layers:
                ni = i
                nj = j
                nk = k
                if direction == 0:
                    ni -= distance
                elif direction == 1:
                    ni += distance
                elif direction == 2:
                    nj -= distance
                elif direction == 3:
                    nj += distance
                elif direction == 4:
                    nk -= distance
                else:
                    nk += distance
                ni = _wrapped_cell_index(ni, nx, periodic_x)
                nj = _wrapped_cell_index(nj, ny, periodic_y)
                nk = _wrapped_cell_index(nk, nz, periodic_z)
                if ni >= 0 and nj >= 0 and nk >= 0:
                    NI = ni + ng
                    NJ = nj + ng
                    NK = nk + ng
                    if (material[NI, NJ, NK] != 0
                            or theta[NI, NJ, NK] < wp.float32(1.0)):
                        near_solid = 1
    contact[I, J, K] = near_solid
    if near_solid != 0:
        wp.atomic_add(active_cells, 0, 1)


@wp.func
def _contact_mobility(
    alpha: wp.float32,
    previous: wp.float32,
    advancing: wp.float32,
    receding: wp.float32,
    hysteresis: wp.float32,
) -> wp.float32:
    change = alpha - previous
    if change > hysteresis:
        return advancing
    if change < -hysteresis:
        return receding
    return wp.float32(0.0)


@wp.kernel
def _regularise_contact(
    alpha: wp.array3d(dtype=wp.float32),
    previous: wp.array3d(dtype=wp.float32),
    theta: wp.array3d(dtype=wp.float32),
    contact: wp.array3d(dtype=wp.int32),
    updated: wp.array3d(dtype=wp.float32),
    exchanged: wp.array(dtype=wp.float64),
    residual: wp.array(dtype=wp.float64),
    advancing: wp.float32,
    receding: wp.float32,
    hysteresis: wp.float32,
    dt: wp.float32,
    inverse_dx_squared: wp.float32,
    inverse_dy_squared: wp.float32,
    inverse_dz_squared: wp.float32,
    cell_volume: wp.float64,
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
    centre = alpha[I, J, K]
    centre_theta = theta[I, J, K]
    if contact[I, J, K] == 0:
        updated[I, J, K] = centre
        return
    centre_mobility = _contact_mobility(
        centre, previous[I, J, K], advancing, receding, hysteresis,
    )
    mass_change = wp.float32(0.0)
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
        ni = _wrapped_cell_index(ni, nx, periodic_x)
        nj = _wrapped_cell_index(nj, ny, periodic_y)
        nk = _wrapped_cell_index(nk, nz, periodic_z)
        if ni >= 0 and nj >= 0 and nk >= 0:
            NI = ni + ng
            NJ = nj + ng
            NK = nk + ng
            if contact[NI, NJ, NK] != 0:
                neighbor = alpha[NI, NJ, NK]
                neighbor_mobility = _contact_mobility(
                    neighbor, previous[NI, NJ, NK],
                    advancing, receding, hysteresis,
                )
                mobility = wp.float32(0.5) * (
                    centre_mobility + neighbor_mobility
                )
                conductance = (
                    wp.min(centre_theta, theta[NI, NJ, NK])
                    * mobility * inverse_spacing_squared
                )
                mass_change += dt * conductance * (neighbor - centre)
    next_fraction = centre + mass_change / centre_theta
    updated[I, J, K] = next_fraction
    wp.atomic_add(
        exchanged, 0,
        wp.float64(wp.float32(0.5) * wp.abs(mass_change)) * cell_volume,
    )
    wp.atomic_add(residual, 0, wp.float64(mass_change) * cell_volume)


class ContactLineModel:
    """Smooth accepted VOF contact motion without an optical wetness state."""

    __slots__ = (
        "cfg", "settings", "device", "previous_alpha", "contact_mask",
        "_updated", "_active_cells", "_exchanged", "_residual", "primed",
    )

    def __init__(
        self,
        cfg: Solver3DConfig,
        settings: ContactLineConfig | None = None,
    ):
        self.cfg = cfg
        self.settings = ContactLineConfig() if settings is None else settings
        self.device = cfg.device
        shape = tuple(n + 2 * cfg.grid.ng for n in cfg.grid.shape)
        self.previous_alpha = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self.contact_mask = wp.zeros(shape, dtype=wp.int32, device=self.device)
        self._updated = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self._active_cells = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._exchanged = wp.zeros(1, dtype=wp.float64, device=self.device)
        self._residual = wp.zeros(1, dtype=wp.float64, device=self.device)
        self.primed = False

    @property
    def enabled(self) -> bool:
        return max(
            self.settings.advancing_diffusivity,
            self.settings.receding_diffusivity,
        ) > 0.0

    def _copy_alpha(self, state: State3D, target: wp.array) -> None:
        wp.launch(
            _copy_cell_field,
            dim=self.cfg.grid.shape,
            inputs=[state.alpha, target, self.cfg.grid.ng],
            device=self.device,
        )

    def prime(self, state: State3D) -> None:
        self.previous_alpha.zero_()
        self._copy_alpha(state, self.previous_alpha)
        self.primed = True

    def time_step_limit(self, cfl: float = 0.25) -> float:
        diffusivity = max(
            self.settings.advancing_diffusivity,
            self.settings.receding_diffusivity,
        )
        if diffusivity == 0.0:
            return math.inf
        g = self.cfg.grid
        return cfl / (
            diffusivity * (1.0 / g.dx**2 + 1.0 / g.dy**2 + 1.0 / g.dz**2)
        )

    def apply(self, state: State3D, dt: float) -> ContactLineResult:
        if not self.primed:
            self.prime(state)
            return ContactLineResult(0, 0.0, 0.0)
        g = self.cfg.grid
        periodic = (
            int(self.cfg.bc.xlo == BC.PERIODIC),
            int(self.cfg.bc.ylo == BC.PERIODIC),
            int(self.cfg.bc.zlo == BC.PERIODIC),
        )
        layers = min(
            4,
            max(1, math.ceil(
                self.settings.contact_length / min(g.dx, g.dy, g.dz)
            )),
        )
        self.contact_mask.zero_()
        self._updated.zero_()
        self._active_cells.zero_()
        self._exchanged.zero_()
        self._residual.zero_()
        wp.launch(
            _mark_contact_band,
            dim=g.shape,
            inputs=[
                state.alpha, state.theta, state.material,
                self.contact_mask, self._active_cells, layers,
                self.settings.mixed_tolerance,
                g.nx, g.ny, g.nz, *periodic, g.ng,
            ],
            device=self.device,
        )
        wp.launch(
            _regularise_contact,
            dim=g.shape,
            inputs=[
                state.alpha, self.previous_alpha, state.theta,
                self.contact_mask, self._updated, self._exchanged,
                self._residual, self.settings.advancing_diffusivity,
                self.settings.receding_diffusivity,
                self.settings.hysteresis_fraction, dt,
                1.0 / g.dx**2, 1.0 / g.dy**2, 1.0 / g.dz**2,
                g.dx * g.dy * g.dz,
                g.nx, g.ny, g.nz, *periodic, g.ng,
            ],
            device=self.device,
        )
        wp.launch(
            _copy_cell_field,
            dim=g.shape,
            inputs=[self._updated, state.alpha, g.ng],
            device=self.device,
        )
        fill_field_boundaries(state, "alpha")
        self.previous_alpha.zero_()
        self._copy_alpha(state, self.previous_alpha)
        return ContactLineResult(
            active_cells=int(self._active_cells.numpy()[0]),
            exchanged_volume=float(self._exchanged.numpy()[0]),
            volume_residual=float(self._residual.numpy()[0]),
        )
