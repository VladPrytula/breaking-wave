"""Geometry-only diagnostics for resolved overturning and topology change."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import warp as wp

from shoreloop.config import BC, Solver3DConfig
from shoreloop.physics3d.interface import estimate_interface_normals
from shoreloop.physics3d.momentum import compute_face_liquid_mass
from shoreloop.physics3d.state import State3D
from shoreloop.physics3d.vof import measure_vof


_TOPOLOGY_INTERFACE_TOLERANCE = 0.05


@dataclass(frozen=True)
class BreakingTopologyDiagnostics:
    time: float
    interface_fraction_tolerance: float
    normal_vertical_tolerance: float
    component_downsample_factor: int
    mixed_interface_cells: int
    ambiguous_interface_cells: int
    downward_interface_cells: int
    downward_interface_x_centroid: float | None
    downward_interface_x_shoreward_p95: float | None
    multi_transition_columns: int
    maximum_vertical_transitions: int
    supported_downward_interface_cells: int
    supported_downward_interface_x_centroid: float | None
    supported_downward_interface_x_shoreward_p95: float | None
    largest_connected_multi_transition_columns: int
    connected_multi_transition_x_centroid: float | None
    connected_multi_transition_x_span: float
    connected_multi_transition_y_span: float
    upper_sheet_thickness_minimum: float
    upper_sheet_thickness_median: float
    upper_sheet_thickness_maximum: float
    water_components: int
    component_mergers: int
    component_splits: int
    water_volume: float
    kinetic_energy: float
    rms_divergence: float
    max_divergence: float
    pressure_iterations: int


@wp.kernel
def _count_downward_normals_by_column(
    alpha: wp.array3d(dtype=wp.float32),
    solid: wp.array3d(dtype=wp.int32),
    normals: wp.array3d(dtype=wp.vec3),
    count_by_column: wp.array2d(dtype=wp.int32),
    mixed_tolerance: wp.float32,
    vertical_tolerance: wp.float32,
    i_start: int,
    i_stop: int,
    ng: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    fraction = alpha[I, J, K]
    if (i >= i_start and i < i_stop
            and solid[I, J, K] == 0
            and fraction > mixed_tolerance
            and fraction < wp.float32(1.0) - mixed_tolerance
            and normals[I, J, K][2] < -vertical_tolerance):
        wp.atomic_add(count_by_column, i - i_start, j, 1)


@wp.kernel
def _count_vertical_transitions(
    alpha: wp.array3d(dtype=wp.float32),
    solid: wp.array3d(dtype=wp.int32),
    multi_columns: wp.array(dtype=wp.int32),
    maximum: wp.array(dtype=wp.int32),
    nz: int,
    wet_threshold: wp.float32,
    i_start: int,
    ng: int,
):
    local_i, j = wp.tid()
    i = local_i + i_start
    previous = int(-1)
    transitions = int(0)
    for k in range(nz):
        I = i + ng
        J = j + ng
        K = k + ng
        if solid[I, J, K] != 0:
            previous = -1
            continue
        wet = int(alpha[I, J, K] >= wet_threshold)
        if previous >= 0 and wet != previous:
            transitions += 1
        previous = wet
    if transitions > 1:
        wp.atomic_add(multi_columns, 0, 1)
    wp.atomic_max(maximum, 0, transitions)


@wp.kernel
def _reduce_kinetic_energy(
    velocity: wp.array3d(dtype=wp.float32),
    mass: wp.array3d(dtype=wp.float32),
    total: wp.array(dtype=wp.float64),
    component_axis: int,
    n: int,
    periodic: int,
    scale: wp.float64,
    ng: int,
):
    i, j, k = wp.tid()
    q = i
    if component_axis == 1:
        q = j
    elif component_axis == 2:
        q = k
    if periodic != 0 and q == n:
        return
    I = i + ng
    J = j + ng
    K = k + ng
    speed = wp.float64(velocity[I, J, K])
    wp.atomic_add(
        total, 0,
        wp.float64(0.5) * wp.float64(mass[I, J, K]) * speed * speed * scale,
    )


def measure_liquid_kinetic_energy(state: State3D) -> float:
    """Return staggered liquid kinetic energy in joules."""
    g = state.cfg.grid
    bc = state.cfg.bc
    periodic = (
        bc.xlo == BC.PERIODIC,
        bc.ylo == BC.PERIODIC,
        bc.zlo == BC.PERIODIC,
    )
    total = wp.zeros(1, dtype=wp.float64, device=state.device)
    scale = state.cfg.physics.rho * g.dx * g.dy * g.dz
    for axis, name in enumerate(("u", "v", "w")):
        mass = compute_face_liquid_mass(state, name)
        wp.launch(
            _reduce_kinetic_energy,
            dim=state.grid.interior_shape(name),
            inputs=[state.field(name), mass, total, axis, g.shape[axis],
                    int(periodic[axis]), scale, g.ng],
            device=state.device,
        )
    return float(total.numpy()[0])


def _downsample_water_mask(
    alpha: np.ndarray,
    solid: np.ndarray,
    factor: int,
    i_start: int,
    i_stop: int,
) -> np.ndarray:
    water = alpha[i_start:i_stop] >= 0.5
    water &= solid[i_start:i_stop] == 0
    output_shape = tuple((size + factor - 1) // factor for size in water.shape)
    coarse = np.zeros(output_shape, dtype=bool)
    for oi in range(factor):
        for oj in range(factor):
            for ok in range(factor):
                block = water[oi::factor, oj::factor, ok::factor]
                coarse[:block.shape[0], :block.shape[1], :block.shape[2]] |= block
    return coarse


def _column_components(mask: np.ndarray, periodic_y: bool) -> list[np.ndarray]:
    """Return eight-neighbour column components, wrapping only alongshore."""
    visited = np.zeros(mask.shape, dtype=bool)
    components: list[np.ndarray] = []
    nx, ny = mask.shape
    for seed_array in np.argwhere(mask):
        seed = tuple(int(value) for value in seed_array)
        if visited[seed]:
            continue
        visited[seed] = True
        pending = [seed]
        component: list[tuple[int, int]] = []
        while pending:
            cell = pending.pop()
            component.append(cell)
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    neighbour_i = cell[0] + di
                    neighbour_j = cell[1] + dj
                    if neighbour_i < 0 or neighbour_i >= nx:
                        continue
                    if neighbour_j < 0 or neighbour_j >= ny:
                        if not periodic_y:
                            continue
                        neighbour_j %= ny
                    neighbour = (neighbour_i, neighbour_j)
                    if mask[neighbour] and not visited[neighbour]:
                        visited[neighbour] = True
                        pending.append(neighbour)
        components.append(np.asarray(component, dtype=np.int32))
    return components


def _upper_sheet_thickness(
    alpha: np.ndarray,
    fluid: np.ndarray,
    columns: np.ndarray,
    dz: float,
) -> np.ndarray:
    """Measure the highest liquid band in each multi-transition column."""
    thickness = np.empty(len(columns), dtype=np.float64)
    for output, (i, j) in enumerate(columns):
        wet = (alpha[i, j] >= 0.5) & fluid[i, j]
        boundaries = np.flatnonzero(np.diff(np.r_[False, wet, False]))
        start, stop = boundaries[-2:]
        thickness[output] = float(np.sum(
            alpha[i, j, start:stop], dtype=np.float64,
        )) * dz
    return thickness


def _breaking_column_support(
    alpha: np.ndarray,
    solid: np.ndarray,
    downward_by_column: np.ndarray,
    cfg: Solver3DConfig,
    i_start: int,
    i_stop: int,
) -> dict[str, float | int | None]:
    """Summarize spatially supported folds carrying downward interface cells."""
    local_alpha = alpha[i_start:i_stop]
    fluid = solid[i_start:i_stop] == 0
    wet = (local_alpha >= 0.5) & fluid
    valid_pairs = fluid[:, :, 1:] & fluid[:, :, :-1]
    transitions = np.count_nonzero(
        (wet[:, :, 1:] != wet[:, :, :-1]) & valid_pairs,
        axis=2,
    )
    multi = transitions > 1
    periodic_y = (
        cfg.bc.ylo == BC.PERIODIC and cfg.bc.yhi == BC.PERIODIC
    )
    components = _column_components(multi, periodic_y)
    supported = [
        component for component in components
        if (
            len(component) >= 4
            and np.any(downward_by_column[
                component[:, 0], component[:, 1]
            ] > 0)
        )
    ]
    if not supported:
        return {
            "supported_downward_interface_cells": 0,
            "supported_downward_interface_x_centroid": None,
            "supported_downward_interface_x_shoreward_p95": None,
            "largest_connected_multi_transition_columns": 0,
            "connected_multi_transition_x_centroid": None,
            "connected_multi_transition_x_span": 0.0,
            "connected_multi_transition_y_span": 0.0,
            "upper_sheet_thickness_minimum": 0.0,
            "upper_sheet_thickness_median": 0.0,
            "upper_sheet_thickness_maximum": 0.0,
        }

    largest = max(supported, key=len)
    grid = cfg.grid
    x = grid.x0 + (i_start + largest[:, 0] + 0.5) * grid.dx
    supported_columns = np.concatenate(supported, axis=0)
    downward_counts = downward_by_column[
        supported_columns[:, 0], supported_columns[:, 1]
    ]
    downward_i = np.repeat(supported_columns[:, 0], downward_counts)
    downward_x = grid.x0 + (i_start + downward_i + 0.5) * grid.dx
    thickness = _upper_sheet_thickness(
        local_alpha, fluid, largest, grid.dz,
    )
    return {
        "supported_downward_interface_cells": len(downward_x),
        "supported_downward_interface_x_centroid": float(np.mean(downward_x)),
        "supported_downward_interface_x_shoreward_p95": float(np.quantile(
            downward_x, 0.95,
        )),
        "largest_connected_multi_transition_columns": len(largest),
        "connected_multi_transition_x_centroid": float(np.mean(x)),
        "connected_multi_transition_x_span": (
            float(np.ptp(x)) + grid.dx
        ),
        "connected_multi_transition_y_span": (
            len(np.unique(largest[:, 1])) * grid.dy
        ),
        "upper_sheet_thickness_minimum": float(np.min(thickness)),
        "upper_sheet_thickness_median": float(np.median(thickness)),
        "upper_sheet_thickness_maximum": float(np.max(thickness)),
    }


def _component_count(mask: np.ndarray, periodic: tuple[bool, bool, bool]) -> int:
    visited = np.zeros(mask.shape, dtype=bool)
    components = 0
    for seed_array in np.argwhere(mask):
        seed = tuple(int(value) for value in seed_array)
        if visited[seed]:
            continue
        components += 1
        visited[seed] = True
        stack = [seed]
        while stack:
            cell = stack.pop()
            for axis in range(3):
                for direction in (-1, 1):
                    neighbour = list(cell)
                    neighbour[axis] += direction
                    if neighbour[axis] < 0 or neighbour[axis] >= mask.shape[axis]:
                        if not periodic[axis]:
                            continue
                        neighbour[axis] %= mask.shape[axis]
                    index = tuple(neighbour)
                    if mask[index] and not visited[index]:
                        visited[index] = True
                        stack.append(index)
    return components


class TopologyTracker:
    """Measure overturning signals and component-count merger/split events."""

    __slots__ = (
        "cfg", "downsample_factor", "normal_vertical_tolerance",
        "i_start", "i_stop", "_previous_components",
    )

    def __init__(
        self,
        cfg: Solver3DConfig,
        *,
        downsample_factor: int = 2,
        normal_vertical_tolerance: float = 0.1,
        x_bounds: tuple[float, float] | None = None,
    ):
        if (isinstance(downsample_factor, bool)
                or not isinstance(downsample_factor, int)
                or downsample_factor <= 0):
            raise ValueError("downsample factor must be a positive integer")
        if (not math.isfinite(normal_vertical_tolerance)
                or not 0.0 <= normal_vertical_tolerance < 1.0):
            raise ValueError("normal vertical tolerance must be finite in [0, 1)")
        self.cfg = cfg
        self.downsample_factor = downsample_factor
        self.normal_vertical_tolerance = float(normal_vertical_tolerance)
        if x_bounds is None:
            self.i_start = 0
            self.i_stop = cfg.grid.nx
        else:
            x = (cfg.grid.x0
                 + (np.arange(cfg.grid.nx) + 0.5) * cfg.grid.dx)
            selected = np.flatnonzero((x >= x_bounds[0]) & (x < x_bounds[1]))
            if selected.size == 0:
                raise ValueError("topology x bounds must contain cell centres")
            self.i_start = int(selected[0])
            self.i_stop = int(selected[-1]) + 1
        self._previous_components: int | None = None

    def sample(
        self,
        state: State3D,
        *,
        time: float = 0.0,
        rms_divergence: float = 0.0,
        max_divergence: float = 0.0,
        pressure_iterations: int = 0,
    ) -> BreakingTopologyDiagnostics:
        if state.cfg != self.cfg:
            raise ValueError("state configuration must match the topology tracker")
        interface = estimate_interface_normals(state)
        downward_by_column = wp.zeros(
            (self.i_stop - self.i_start, self.cfg.grid.ny),
            dtype=wp.int32,
            device=state.device,
        )
        wp.launch(
            _count_downward_normals_by_column, dim=self.cfg.grid.shape,
            inputs=[state.alpha, state.solid, interface.normals,
                    downward_by_column,
                    _TOPOLOGY_INTERFACE_TOLERANCE,
                    self.normal_vertical_tolerance,
                    self.i_start, self.i_stop,
                    self.cfg.grid.ng],
            device=state.device,
        )
        downward_columns = downward_by_column.numpy()
        downward_counts = np.sum(downward_columns, axis=1)
        downward_i = np.repeat(
            np.arange(self.i_start, self.i_stop), downward_counts,
        )
        downward_x = (
            self.cfg.grid.x0
            + (downward_i + 0.5) * self.cfg.grid.dx
        )
        multi_columns = wp.zeros(1, dtype=wp.int32, device=state.device)
        maximum_transitions = wp.zeros(1, dtype=wp.int32, device=state.device)
        wp.launch(
            _count_vertical_transitions,
            dim=(self.i_stop - self.i_start, self.cfg.grid.ny),
            inputs=[state.alpha, state.solid, multi_columns,
                    maximum_transitions, self.cfg.grid.nz, 0.5,
                    self.i_start, self.cfg.grid.ng],
            device=state.device,
        )
        alpha = state.get_interior("alpha")
        solid = state.get_solid_interior()
        column_support = _breaking_column_support(
            alpha, solid, downward_columns, self.cfg,
            self.i_start, self.i_stop,
        )
        bc = self.cfg.bc
        periodic = (
            (bc.xlo == BC.PERIODIC
             and self.i_start == 0 and self.i_stop == self.cfg.grid.nx),
            bc.ylo == BC.PERIODIC,
            bc.zlo == BC.PERIODIC,
        )
        components = _component_count(
            _downsample_water_mask(
                alpha, solid, self.downsample_factor,
                self.i_start, self.i_stop,
            ),
            periodic,
        )
        mergers = 0
        splits = 0
        if self._previous_components is not None:
            mergers = max(self._previous_components - components, 0)
            splits = max(components - self._previous_components, 0)
        self._previous_components = components
        vof = measure_vof(state)
        return BreakingTopologyDiagnostics(
            time=float(time),
            interface_fraction_tolerance=_TOPOLOGY_INTERFACE_TOLERANCE,
            normal_vertical_tolerance=self.normal_vertical_tolerance,
            component_downsample_factor=self.downsample_factor,
            mixed_interface_cells=interface.mixed_cells,
            ambiguous_interface_cells=interface.ambiguous_cells,
            downward_interface_cells=len(downward_i),
            downward_interface_x_centroid=(
                None if len(downward_x) == 0
                else float(np.mean(downward_x))
            ),
            downward_interface_x_shoreward_p95=(
                None if len(downward_x) == 0
                else float(np.quantile(downward_x, 0.95))
            ),
            multi_transition_columns=int(multi_columns.numpy()[0]),
            maximum_vertical_transitions=int(maximum_transitions.numpy()[0]),
            **column_support,
            water_components=components,
            component_mergers=mergers,
            component_splits=splits,
            water_volume=vof.volume,
            kinetic_energy=measure_liquid_kinetic_energy(state),
            rms_divergence=float(rms_divergence),
            max_divergence=float(max_divergence),
            pressure_iterations=int(pressure_iterations),
        )
