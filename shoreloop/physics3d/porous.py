"""Finite pebble-bed storage, drainage, and Darcy-Forchheimer drag."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import warp as wp

from shoreloop.config import BC, Solver3DConfig
from shoreloop.physics3d.advection import _copy_cell_field
from shoreloop.physics3d.boundary import fill_field_boundaries
from shoreloop.physics3d.cut_cells import CutCellGeometry
from shoreloop.physics3d.extrapolation import _wrapped_cell_index
from shoreloop.physics3d.momentum import compute_face_liquid_mass
from shoreloop.physics3d.state import State3D, _copy_into_interior
from shoreloop.scene.geometry3d import BEACH_MATERIAL_ID


@dataclass(frozen=True)
class PorousStorageConfig:
    """Finite surface-column capacity and exchange time scales.

    With ``water_table`` unset every bed column stores ``capacity_fraction`` of
    the ``layer_depth`` above the bed. With it set (an elevation), a column
    stores ``capacity_fraction`` of its unsaturated thickness, the bed's height
    above the water table capped at ``layer_depth``: nothing at or below the
    water table, where the bed is saturated.
    """

    layer_depth: float = 0.08
    capacity_fraction: float = 0.22
    infiltration_rate: float = 1.5
    drainage_time: float = 6.0
    water_table: float | None = None

    def __post_init__(self) -> None:
        if self.layer_depth <= 0.0:
            raise ValueError("layer_depth must be positive")
        if not 0.0 < self.capacity_fraction <= 1.0:
            raise ValueError("capacity_fraction must be in (0, 1]")
        if self.infiltration_rate < 0.0:
            raise ValueError("infiltration_rate must be non-negative")
        if self.drainage_time <= 0.0:
            raise ValueError("drainage_time must be positive")


DRAG_LAWS = ("darcy_forchheimer", "canopy")


@dataclass(frozen=True)
class PorousDragConfig:
    """Switchable drag on the water next to the bed.

    ``darcy_forchheimer`` applies ``nu / k + c_F |u| / sqrt(k)`` over the
    storage layer's cells. ``canopy`` applies the form drag of the grain tops,
    ``1/2 C_d a |u|`` with the frontal area density ``a`` of the grains, to the
    water within ``layer_depth`` (one grain) of the bed, weighted by the part
    of each cell inside that layer.
    """

    permeability: float = 4.0e-7
    forchheimer_coefficient: float = 0.55
    kinematic_viscosity: float = 1.0e-6
    law: str = "darcy_forchheimer"
    drag_coefficient: float = 0.0
    frontal_area_density: float = 0.0
    layer_depth: float = 0.0

    def __post_init__(self) -> None:
        if self.permeability <= 0.0:
            raise ValueError("permeability must be positive")
        if self.forchheimer_coefficient < 0.0:
            raise ValueError("forchheimer_coefficient must be non-negative")
        if self.kinematic_viscosity < 0.0:
            raise ValueError("kinematic_viscosity must be non-negative")
        if self.law not in DRAG_LAWS:
            raise ValueError(f"drag law must be one of {DRAG_LAWS}")
        if self.law == "canopy" and not (
            self.drag_coefficient > 0.0 and self.frontal_area_density > 0.0 and self.layer_depth > 0.0
        ):
            raise ValueError("the canopy law needs a positive drag coefficient, frontal area density and layer depth")

    @property
    def law_code(self) -> int:
        return DRAG_LAWS.index(self.law)

    @property
    def canopy_rate(self) -> float:
        """Damping rate per unit speed inside the grain layer, 1/m."""
        return 0.5 * self.drag_coefficient * self.frontal_area_density


@dataclass(frozen=True)
class PorousStorageResult:
    infiltrated_volume: float
    drained_volume: float
    stored_volume: float
    mass_residual: float


@dataclass(frozen=True)
class PorousDragResult:
    energy_loss: float
    maximum_damping_rate: float


def _beach_surface(geometry: CutCellGeometry, cfg: Solver3DConfig) -> tuple[np.ndarray, np.ndarray]:
    """The lowest fluid cell of every beach column and the bed elevation there.

    Returns the surface mask (cells) and, per column, the bed elevation (NaN
    where the column has no beach surface). A cut cell holds the planar bed at
    ``(1 - theta)`` of its height; a full cell above a solid beach cell has it at
    its bottom face.
    """
    theta = geometry.theta
    material = geometry.material_ids
    surface = (
        (material == BEACH_MATERIAL_ID) & (theta > 0.0) & (theta < 1.0)
    )
    surface[:, :, 1:] |= (
        (theta[:, :, 1:] > 0.0)
        & (theta[:, :, :-1] == 0.0)
        & (material[:, :, :-1] == BEACH_MATERIAL_ID)
    )
    g = cfg.grid
    has = surface.any(axis=2)
    k_surface = np.argmax(surface, axis=2)
    columns = np.nonzero(has)
    bed = np.full(has.shape, np.nan)
    k = k_surface[columns]
    theta_surface = theta[columns[0], columns[1], k]
    bed[columns] = g.z0 + k * g.dz + (1.0 - theta_surface) * g.dz
    return surface, bed


def _stack_above(surface: np.ndarray, layers: int) -> list[np.ndarray]:
    stack = []
    for offset in range(layers):
        if offset == 0:
            stack.append(surface)
        else:
            shifted = np.zeros_like(surface)
            shifted[:, :, offset:] = surface[:, :, :-offset]
            stack.append(shifted)
    return stack


def porous_capacity_from_geometry(
    geometry: CutCellGeometry,
    cfg: Solver3DConfig,
    settings: PorousStorageConfig,
) -> np.ndarray:
    """Pore capacity, as a fraction of each cell, in the layer above beach cells."""
    theta = geometry.theta
    surface, bed = _beach_surface(geometry, cfg)
    g = cfg.grid
    layers = max(1, math.ceil(settings.layer_depth / g.dz))
    capacity = np.zeros(g.shape, dtype=np.float32)
    if settings.water_table is None:
        column_fraction = np.full(bed.shape, settings.capacity_fraction, dtype=np.float32)
    else:
        # Unsaturated thickness of the column, capped at the layer, spread over
        # the layer's cells so the column total is capacity_fraction * thickness.
        unsaturated = np.clip(np.nan_to_num(bed - settings.water_table, nan=0.0), 0.0, settings.layer_depth)
        column_fraction = (settings.capacity_fraction * unsaturated / (layers * g.dz)).astype(np.float32)
    for shifted in _stack_above(surface, layers):
        cells = shifted & (theta > 0.0)
        capacity[cells] = np.broadcast_to(column_fraction[:, :, None], g.shape)[cells]
    return capacity


def canopy_weight_from_geometry(
    geometry: CutCellGeometry,
    cfg: Solver3DConfig,
    settings: PorousDragConfig,
) -> np.ndarray:
    """Fraction of each cell's water lying within ``layer_depth`` above the bed.

    The open part of a cell runs from the bed (in the cut cell) or the cell's
    bottom face to its top; the weight is the share of that open height inside
    the canopy, so a cut cell whose water all sits among the grain tops gets 1.
    """
    theta = geometry.theta
    surface, bed = _beach_surface(geometry, cfg)
    g = cfg.grid
    layers = max(1, math.ceil(settings.layer_depth / g.dz)) + 1
    weight = np.zeros(g.shape, dtype=np.float32)
    z_bottom = g.z0 + np.arange(g.nz) * g.dz
    for shifted in _stack_above(surface, layers):
        cells = shifted & (theta > 0.0)
        i, j, k = np.nonzero(cells)
        low = np.maximum(z_bottom[k], bed[i, j])
        top = z_bottom[k] + g.dz
        high = np.minimum(top, bed[i, j] + settings.layer_depth)
        weight[i, j, k] = np.clip((high - low) / (top - low), 0.0, 1.0)
    return weight


@wp.kernel
def _exchange_storage(
    alpha: wp.array3d(dtype=wp.float32),
    theta: wp.array3d(dtype=wp.float32),
    storage: wp.array3d(dtype=wp.float32),
    capacity: wp.array3d(dtype=wp.float32),
    updated_alpha: wp.array3d(dtype=wp.float32),
    updated_storage: wp.array3d(dtype=wp.float32),
    infiltrated: wp.array(dtype=wp.float64),
    drained: wp.array(dtype=wp.float64),
    residual: wp.array(dtype=wp.float64),
    dt: wp.float32,
    infiltration_rate: wp.float32,
    drainage_time: wp.float32,
    cell_volume: wp.float64,
    ng: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    fraction = theta[I, J, K]
    water = fraction * alpha[I, J, K]
    before_storage = storage[I, J, K]
    available = capacity[I, J, K]
    if available <= wp.float32(0.0) or fraction <= wp.float32(0.0):
        updated_alpha[I, J, K] = alpha[I, J, K]
        updated_storage[I, J, K] = before_storage
        return

    equilibrium = wp.float32(0.0)
    if water > wp.float32(0.0):
        equilibrium = available

    infiltration = wp.float32(0.0)
    drainage = wp.float32(0.0)
    if before_storage < equilibrium:
        fill_ratio = dt * infiltration_rate
        infiltration = (
            (equilibrium - before_storage) * fill_ratio
            / (wp.float32(1.0) + fill_ratio)
        )
        infiltration = wp.min(water, infiltration)
    elif before_storage > equilibrium:
        drainage = (
            (before_storage - equilibrium) * dt
            / (drainage_time + dt)
        )

    after_water = water - infiltration
    after_storage = before_storage + infiltration - drainage
    updated_alpha[I, J, K] = alpha[I, J, K]
    if infiltration > wp.float32(0.0):
        updated_alpha[I, J, K] = after_water / fraction
    updated_storage[I, J, K] = after_storage
    wp.atomic_add(infiltrated, 0, wp.float64(infiltration) * cell_volume)
    wp.atomic_add(drained, 0, wp.float64(drainage) * cell_volume)
    wp.atomic_add(
        residual, 0,
        wp.float64(after_water + after_storage + drainage
                   - water - before_storage) * cell_volume,
    )


@wp.kernel
def _initialize_storage_equilibrium(
    alpha: wp.array3d(dtype=wp.float32),
    theta: wp.array3d(dtype=wp.float32),
    capacity: wp.array3d(dtype=wp.float32),
    storage: wp.array3d(dtype=wp.float32),
    ng: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    value = wp.float32(0.0)
    if theta[I, J, K] * alpha[I, J, K] > wp.float32(0.0):
        value = capacity[I, J, K]
    storage[I, J, K] = value


@wp.kernel
def _sum_storage(
    storage: wp.array3d(dtype=wp.float32),
    total: wp.array(dtype=wp.float64),
    cell_volume: wp.float64,
    ng: int,
):
    i, j, k = wp.tid()
    wp.atomic_add(
        total, 0,
        wp.float64(storage[i + ng, j + ng, k + ng]) * cell_volume,
    )


@wp.func
def _capacity_cell(
    capacity: wp.array3d(dtype=wp.float32),
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
    return capacity[ci + ng, cj + ng, ck + ng]


@wp.kernel
def _apply_porous_drag(
    velocity: wp.array3d(dtype=wp.float32),
    mass: wp.array3d(dtype=wp.float32),
    capacity: wp.array3d(dtype=wp.float32),
    weight: wp.array3d(dtype=wp.float32),
    updated: wp.array3d(dtype=wp.float32),
    local_dissipation: wp.array3d(dtype=wp.float32),
    energy_loss: wp.array(dtype=wp.float64),
    maximum_rate: wp.array(dtype=wp.float32),
    component: int,
    law: int,
    canopy_rate: wp.float32,
    capacity_fraction: wp.float32,
    permeability: wp.float32,
    forchheimer: wp.float32,
    kinematic_viscosity: wp.float32,
    dt: wp.float32,
    rho: wp.float32,
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
    if law == 1:
        layer = wp.float32(0.5) * (
            _capacity_cell(
                weight, lower_i, lower_j, lower_k,
                nx, ny, nz, periodic_x, periodic_y, periodic_z, ng,
            )
            + _capacity_cell(
                weight, i, j, k,
                nx, ny, nz, periodic_x, periodic_y, periodic_z, ng,
            )
        )
    else:
        layer = wp.float32(0.5) * (
            _capacity_cell(
                capacity, lower_i, lower_j, lower_k,
                nx, ny, nz, periodic_x, periodic_y, periodic_z, ng,
            )
            + _capacity_cell(
                capacity, i, j, k,
                nx, ny, nz, periodic_x, periodic_y, periodic_z, ng,
            )
        ) / capacity_fraction
    if layer <= wp.float32(0.0):
        updated[I, J, K] = before
        return
    if law == 1:
        rate = layer * canopy_rate * wp.abs(before)
    else:
        rate = layer * (
            kinematic_viscosity / permeability
            + forchheimer * wp.abs(before) / wp.sqrt(permeability)
        )
    after = before / (wp.float32(1.0) + dt * rate)
    updated[I, J, K] = after
    loss = wp.float32(0.5) * face_mass * (before * before - after * after)
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
        wp.atomic_add(energy_loss, 0, wp.float64(loss) * rho_cell_volume)
        wp.atomic_max(maximum_rate, 0, rate)
        ci = wp.min(nx - 1, wp.max(0, lower_i))
        cj = wp.min(ny - 1, wp.max(0, lower_j))
        ck = wp.min(nz - 1, wp.max(0, lower_k))
        wp.atomic_add(
            local_dissipation, ci + ng, cj + ng, ck + ng,
            rho * loss / dt,
        )


class PorousBedModel:
    """Share one shallow-layer mask between independent storage and drag."""

    __slots__ = (
        "cfg", "storage_settings", "drag_settings", "device", "capacity",
        "capacity_fraction", "drag_weight",
        "dissipation", "_updated_alpha", "_updated_storage", "_updated_faces",
        "_infiltrated", "_drained", "_residual", "_stored", "_energy_loss",
        "_maximum_rate",
    )

    def __init__(
        self,
        cfg: Solver3DConfig,
        capacity: np.ndarray,
        *,
        storage_settings: PorousStorageConfig | None = None,
        drag_settings: PorousDragConfig | None = None,
        drag_weight: np.ndarray | None = None,
    ):
        self.cfg = cfg
        self.storage_settings = storage_settings
        self.drag_settings = drag_settings
        self.device = cfg.device
        self.capacity_fraction = float(np.max(capacity))
        if not self.capacity_fraction > 0.0:
            raise ValueError("porous capacity must contain a positive cell")
        if drag_settings is not None and drag_settings.law == "canopy":
            if drag_weight is None or not np.max(drag_weight) > 0.0:
                raise ValueError("the canopy drag needs a positive weight field")
        shape = tuple(n + 2 * cfg.grid.ng for n in cfg.grid.shape)
        self.capacity = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self.drag_weight = wp.zeros(shape, dtype=wp.float32, device=self.device)
        for field, target in ((capacity, self.capacity), (drag_weight, self.drag_weight)):
            if field is None:
                continue
            source = wp.array(
                np.ascontiguousarray(field, dtype=np.float32),
                dtype=wp.float32, device=self.device,
            )
            wp.launch(
                _copy_into_interior,
                dim=cfg.grid.shape,
                inputs=[source, target, cfg.grid.ng],
                device=self.device,
            )
        self.dissipation = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self._updated_alpha = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self._updated_storage = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self._updated_faces = tuple(
            wp.zeros(
                tuple(n + 2 * cfg.grid.ng for n in face_shape),
                dtype=wp.float32, device=self.device,
            )
            for face_shape in (
                (cfg.grid.nx + 1, cfg.grid.ny, cfg.grid.nz),
                (cfg.grid.nx, cfg.grid.ny + 1, cfg.grid.nz),
                (cfg.grid.nx, cfg.grid.ny, cfg.grid.nz + 1),
            )
        )
        self._infiltrated = wp.zeros(1, dtype=wp.float64, device=self.device)
        self._drained = wp.zeros(1, dtype=wp.float64, device=self.device)
        self._residual = wp.zeros(1, dtype=wp.float64, device=self.device)
        self._stored = wp.zeros(1, dtype=wp.float64, device=self.device)
        self._energy_loss = wp.zeros(1, dtype=wp.float64, device=self.device)
        self._maximum_rate = wp.zeros(1, dtype=wp.float32, device=self.device)

    @classmethod
    def from_geometry(
        cls,
        cfg: Solver3DConfig,
        geometry: CutCellGeometry,
        *,
        storage_settings: PorousStorageConfig | None = None,
        drag_settings: PorousDragConfig | None = None,
    ) -> "PorousBedModel":
        capacity_settings = (
            PorousStorageConfig() if storage_settings is None
            else storage_settings
        )
        capacity = porous_capacity_from_geometry(
            geometry, cfg, capacity_settings,
        )
        drag_weight = (
            canopy_weight_from_geometry(geometry, cfg, drag_settings)
            if drag_settings is not None and drag_settings.law == "canopy" else None
        )
        return cls(
            cfg, capacity, storage_settings=storage_settings,
            drag_settings=drag_settings, drag_weight=drag_weight,
        )

    @property
    def storage_enabled(self) -> bool:
        return self.storage_settings is not None

    @property
    def drag_enabled(self) -> bool:
        return self.drag_settings is not None

    def initialize_storage_equilibrium(self, state: State3D) -> float:
        """Saturate initially submerged pores without changing free water."""
        g = self.cfg.grid
        wp.launch(
            _initialize_storage_equilibrium,
            dim=g.shape,
            inputs=[
                state.alpha, state.theta, self.capacity,
                state.porous_storage, g.ng,
            ],
            device=self.device,
        )
        fill_field_boundaries(state, "porous_storage")
        self._stored.zero_()
        wp.launch(
            _sum_storage,
            dim=g.shape,
            inputs=[
                state.porous_storage, self._stored,
                g.dx * g.dy * g.dz, g.ng,
            ],
            device=self.device,
        )
        return float(self._stored.numpy()[0])

    def apply_storage(self, state: State3D, dt: float) -> PorousStorageResult:
        settings = self.storage_settings
        if settings is None:
            return PorousStorageResult(0.0, 0.0, 0.0, 0.0)
        for value in (
            self._infiltrated, self._drained, self._residual, self._stored,
        ):
            value.zero_()
        self._updated_alpha.zero_()
        self._updated_storage.zero_()
        g = self.cfg.grid
        cell_volume = g.dx * g.dy * g.dz
        wp.launch(
            _exchange_storage,
            dim=g.shape,
            inputs=[
                state.alpha, state.theta, state.porous_storage,
                self.capacity, self._updated_alpha, self._updated_storage,
                self._infiltrated, self._drained, self._residual,
                dt, settings.infiltration_rate, settings.drainage_time,
                cell_volume, g.ng,
            ],
            device=self.device,
        )
        for source, target in (
            (self._updated_alpha, state.alpha),
            (self._updated_storage, state.porous_storage),
        ):
            wp.launch(
                _copy_cell_field,
                dim=g.shape,
                inputs=[source, target, g.ng],
                device=self.device,
            )
        fill_field_boundaries(state, "alpha")
        fill_field_boundaries(state, "porous_storage")
        wp.launch(
            _sum_storage,
            dim=g.shape,
            inputs=[state.porous_storage, self._stored, cell_volume, g.ng],
            device=self.device,
        )
        return PorousStorageResult(
            infiltrated_volume=float(self._infiltrated.numpy()[0]),
            drained_volume=float(self._drained.numpy()[0]),
            stored_volume=float(self._stored.numpy()[0]),
            mass_residual=float(self._residual.numpy()[0]),
        )

    def apply_drag(self, state: State3D, dt: float) -> PorousDragResult:
        settings = self.drag_settings
        if settings is None:
            return PorousDragResult(0.0, 0.0)
        self.dissipation.zero_()
        self._energy_loss.zero_()
        self._maximum_rate.zero_()
        g = self.cfg.grid
        periodic = (
            int(self.cfg.bc.xlo == BC.PERIODIC),
            int(self.cfg.bc.ylo == BC.PERIODIC),
            int(self.cfg.bc.zlo == BC.PERIODIC),
        )
        rho_cell_volume = self.cfg.physics.rho * g.dx * g.dy * g.dz
        for component, name in enumerate(("u", "v", "w")):
            mass = compute_face_liquid_mass(state, name)
            target = self._updated_faces[component]
            target.zero_()
            wp.launch(
                _apply_porous_drag,
                dim=state.grid.interior_shape(name),
                inputs=[
                    state.field(name), mass, self.capacity, self.drag_weight, target,
                    self.dissipation, self._energy_loss, self._maximum_rate,
                    component, settings.law_code, settings.canopy_rate,
                    self.capacity_fraction, settings.permeability,
                    settings.forchheimer_coefficient,
                    settings.kinematic_viscosity, dt, self.cfg.physics.rho,
                    rho_cell_volume, g.nx, g.ny, g.nz, *periodic, g.ng,
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
        return PorousDragResult(
            energy_loss=float(self._energy_loss.numpy()[0]),
            maximum_damping_rate=float(self._maximum_rate.numpy()[0]),
        )
