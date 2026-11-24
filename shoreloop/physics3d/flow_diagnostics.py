"""Resolved three-dimensional wake and turbulence measurements."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from shoreloop.config import BC
from shoreloop.physics3d.state import State3D


@dataclass(frozen=True)
class TurbulenceFields:
    """Cell-centred resolved velocity-gradient observables."""

    velocity: tuple[np.ndarray, np.ndarray, np.ndarray]
    vorticity: tuple[np.ndarray, np.ndarray, np.ndarray]
    vorticity_magnitude: np.ndarray
    q_criterion: np.ndarray
    strain_magnitude: np.ndarray


@dataclass(frozen=True)
class TurbulenceMetrics:
    """Volume-weighted summaries used by boulder wake gates."""

    maximum_vorticity: float
    rms_vorticity: float
    resolved_enstrophy: float
    positive_q_volume: float
    maximum_q: float
    resolved_kinetic_energy: float


@dataclass(frozen=True)
class RecirculationMetrics:
    cells: int
    volume: float
    streamwise_length: float
    maximum_reverse_speed: float


def cell_velocity(state: State3D) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate the three staggered velocity components to cell centres."""
    u = state.get_interior("u")
    v = state.get_interior("v")
    w = state.get_interior("w")
    return (
        0.5 * (u[:-1] + u[1:]),
        0.5 * (v[:, :-1] + v[:, 1:]),
        0.5 * (w[:, :, :-1] + w[:, :, 1:]),
    )


def _derivative(
    field: np.ndarray,
    axis: int,
    spacing: float,
    periodic: bool,
) -> np.ndarray:
    if periodic:
        return (
            np.roll(field, -1, axis=axis) - np.roll(field, 1, axis=axis)
        ) / (2.0 * spacing)
    return np.gradient(field, spacing, axis=axis, edge_order=2)


def velocity_gradient(
    velocity: tuple[np.ndarray, np.ndarray, np.ndarray],
    spacing: tuple[float, float, float],
    periodic: tuple[bool, bool, bool] = (False, False, False),
) -> np.ndarray:
    """Return ``gradient[component, derivative_axis, i, j, k]``."""
    shape = velocity[0].shape
    gradient = np.empty((3, 3, *shape), dtype=np.float64)
    for component in range(3):
        for axis in range(3):
            gradient[component, axis] = _derivative(
                velocity[component], axis, spacing[axis], periodic[axis],
            )
    return gradient


def turbulence_fields(
    velocity: tuple[np.ndarray, np.ndarray, np.ndarray],
    spacing: tuple[float, float, float],
    periodic: tuple[bool, bool, bool] = (False, False, False),
) -> TurbulenceFields:
    """Compute curl, strain magnitude, and the Q vortex criterion."""
    gradient = velocity_gradient(velocity, spacing, periodic)
    symmetric = 0.5 * (gradient + np.swapaxes(gradient, 0, 1))
    antisymmetric = 0.5 * (gradient - np.swapaxes(gradient, 0, 1))
    strain_squared = np.sum(symmetric * symmetric, axis=(0, 1))
    rotation_squared = np.sum(antisymmetric * antisymmetric, axis=(0, 1))
    q_criterion = 0.5 * (rotation_squared - strain_squared)
    vorticity = (
        gradient[2, 1] - gradient[1, 2],
        gradient[0, 2] - gradient[2, 0],
        gradient[1, 0] - gradient[0, 1],
    )
    vorticity_magnitude = np.sqrt(sum(value * value for value in vorticity))
    return TurbulenceFields(
        velocity=velocity,
        vorticity=vorticity,
        vorticity_magnitude=vorticity_magnitude,
        q_criterion=q_criterion,
        strain_magnitude=np.sqrt(2.0 * strain_squared),
    )


def measure_turbulence(
    state: State3D,
    *,
    wet_threshold: float = 0.05,
) -> tuple[TurbulenceFields, TurbulenceMetrics]:
    g = state.cfg.grid
    periodic = (
        state.cfg.bc.xlo == state.cfg.bc.xhi == BC.PERIODIC,
        state.cfg.bc.ylo == state.cfg.bc.yhi == BC.PERIODIC,
        state.cfg.bc.zlo == state.cfg.bc.zhi == BC.PERIODIC,
    )
    fields = turbulence_fields(
        cell_velocity(state), (g.dx, g.dy, g.dz), periodic,
    )
    liquid = (
        state.get_interior("alpha")
        * state.get_geometry_interior("theta")
    )
    wet = liquid > wet_threshold
    cell_volume = g.dx * g.dy * g.dz
    weight = np.where(wet, liquid, 0.0)
    weighted_cells = float(np.sum(weight, dtype=np.float64))
    omega_squared = fields.vorticity_magnitude**2
    speed_squared = sum(value * value for value in fields.velocity)
    metrics = TurbulenceMetrics(
        maximum_vorticity=float(np.max(fields.vorticity_magnitude[wet], initial=0.0)),
        rms_vorticity=(
            float(np.sqrt(np.sum(weight * omega_squared, dtype=np.float64)
                          / weighted_cells))
            if weighted_cells > 0.0 else 0.0
        ),
        resolved_enstrophy=(
            0.5 * float(np.sum(weight * omega_squared, dtype=np.float64))
            * cell_volume
        ),
        positive_q_volume=(
            float(np.count_nonzero(wet & (fields.q_criterion > 0.0)))
            * cell_volume
        ),
        maximum_q=float(np.max(fields.q_criterion[wet], initial=0.0)),
        resolved_kinetic_energy=(
            0.5 * state.cfg.physics.rho
            * float(np.sum(weight * speed_squared, dtype=np.float64))
            * cell_volume
        ),
    )
    return fields, metrics


def measure_recirculation(
    streamwise_velocity: np.ndarray,
    wet_mask: np.ndarray,
    x: np.ndarray,
    cell_volume: float,
    *,
    reverse_threshold: float = 0.02,
) -> RecirculationMetrics:
    reversed_flow = wet_mask & (streamwise_velocity < -reverse_threshold)
    cells = int(np.count_nonzero(reversed_flow))
    active_x = np.flatnonzero(np.any(reversed_flow, axis=(1, 2)))
    length = 0.0
    if active_x.size:
        dx = float(np.mean(np.diff(x))) if x.size > 1 else 0.0
        length = float(x[active_x[-1]] - x[active_x[0]] + dx)
    return RecirculationMetrics(
        cells=cells,
        volume=cells * cell_volume,
        streamwise_length=length,
        maximum_reverse_speed=(
            float(-np.min(streamwise_velocity[reversed_flow])) if cells else 0.0
        ),
    )


def render_turbulence_slice(
    fields: TurbulenceFields,
    path: str | Path,
    *,
    plane: str = "xz",
    index: int | None = None,
) -> Path:
    """Write paired vorticity/Q slices for an integration run."""
    import matplotlib.pyplot as plt

    axes = {"yz": 0, "xz": 1, "xy": 2}
    axis = axes[plane]
    shape = fields.vorticity_magnitude.shape
    selected = shape[axis] // 2 if index is None else index
    selector = [slice(None), slice(None), slice(None)]
    selector[axis] = selected
    selector_tuple = tuple(selector)
    omega = fields.vorticity_magnitude[selector_tuple].T
    q_value = fields.q_criterion[selector_tuple].T
    figure, plot_axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    first = plot_axes[0].imshow(omega, origin="lower", cmap="magma")
    second = plot_axes[1].imshow(q_value, origin="lower", cmap="coolwarm")
    plot_axes[0].set_title("vorticity magnitude [s⁻¹]")
    plot_axes[1].set_title("Q criterion [s⁻²]")
    figure.colorbar(first, ax=plot_axes[0], shrink=0.8)
    figure.colorbar(second, ax=plot_axes[1], shrink=0.8)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    return destination
