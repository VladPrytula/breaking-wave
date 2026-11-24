"""Plain field-slice PNG diagnostics for the local 3-D solver."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from shoreloop.physics3d.interface import estimate_interface_normals
from shoreloop.physics3d.state import State3D


@dataclass(frozen=True)
class SliceDiagnostic:
    """Location of one rendered field slice."""

    path: Path
    plane: str
    index: int


def _cell_velocity(state: State3D) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = state.get_interior("u")
    v = state.get_interior("v")
    w = state.get_interior("w")
    return (
        0.5 * (u[:-1] + u[1:]),
        0.5 * (v[:, :-1] + v[:, 1:]),
        0.5 * (w[:, :, :-1] + w[:, :, 1:]),
    )


def _validate_index(index: int | None, count: int, plane: str) -> int:
    if index is None:
        return count // 2
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError(f"{plane} slice index must be an integer")
    if not 0 <= index < count:
        raise IndexError(f"{plane} slice index {index} outside [0, {count})")
    return index


def render_field_slice(
    state: State3D,
    path: str | Path,
    *,
    plane: str,
    index: int | None = None,
    vector_stride: int | None = None,
) -> SliceDiagnostic:
    """Render alpha/velocity and pressure/normal panels to one PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if plane not in ("xz", "xy"):
        raise ValueError("plane must be 'xz' or 'xy'")
    output = Path(path)
    if output.suffix.lower() != ".png":
        raise ValueError("diagnostic output must use a .png suffix")
    output.parent.mkdir(parents=True, exist_ok=True)

    g = state.cfg.grid
    alpha = state.get_interior("alpha").astype(np.float64)
    pressure = state.get_interior("p").astype(np.float64)
    solid = state.get_solid_interior().astype(bool)
    if not np.all(np.isfinite(alpha)) or not np.all(np.isfinite(pressure)):
        raise FloatingPointError("non-finite alpha or pressure in diagnostic slice")
    u, v, w = _cell_velocity(state)
    normal_result = estimate_interface_normals(state)
    normals = normal_result.normals.numpy()[
        state.grid.interior_slices("cell")
    ].astype(np.float64)
    x, y, z = state.grid.coordinates("cell")

    if plane == "xz":
        selected = _validate_index(index, g.ny, plane)
        horizontal, vertical = x, z
        alpha_slice = alpha[:, selected, :]
        pressure_slice = pressure[:, selected, :]
        solid_slice = solid[:, selected, :]
        velocity_horizontal = u[:, selected, :]
        velocity_vertical = w[:, selected, :]
        normal_horizontal = normals[:, selected, :, 0]
        normal_vertical = normals[:, selected, :, 2]
        horizontal_label, vertical_label = "x [m]", "z [m]"
        extent = (g.x0, g.x0 + g.Lx, g.z0, g.z0 + g.Lz)
    else:
        selected = _validate_index(index, g.nz, plane)
        horizontal, vertical = x, y
        alpha_slice = alpha[:, :, selected]
        pressure_slice = pressure[:, :, selected]
        solid_slice = solid[:, :, selected]
        velocity_horizontal = u[:, :, selected]
        velocity_vertical = v[:, :, selected]
        normal_horizontal = normals[:, :, selected, 0]
        normal_vertical = normals[:, :, selected, 1]
        horizontal_label, vertical_label = "x [m]", "y [m]"
        extent = (g.x0, g.x0 + g.Lx, g.y0, g.y0 + g.Ly)

    if vector_stride is None:
        vector_stride = max(1, min(alpha_slice.shape) // 14)
    if (isinstance(vector_stride, bool) or not isinstance(vector_stride, int)
            or vector_stride <= 0):
        raise ValueError("vector_stride must be a positive integer")
    hs = slice(None, None, vector_stride)
    vs = slice(None, None, vector_stride)

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), constrained_layout=True)
    alpha_image = axes[0].imshow(
        alpha_slice.T, origin="lower", extent=extent, vmin=0.0, vmax=1.0,
        cmap="Blues", interpolation="nearest", aspect="auto",
    )
    axes[0].imshow(
        np.ma.masked_where(~solid_slice.T, solid_slice.T),
        origin="lower", extent=extent, vmin=0.0, vmax=1.0,
        cmap="Greys", interpolation="nearest", aspect="auto",
    )
    if float(np.min(alpha_slice)) < 0.5 < float(np.max(alpha_slice)):
        axes[0].contour(horizontal, vertical, alpha_slice.T,
                        levels=[0.5], colors="white", linewidths=1.2)
    horizontal_grid, vertical_grid = np.meshgrid(
        horizontal[hs], vertical[vs], indexing="ij",
    )
    sampled_velocity_horizontal = velocity_horizontal[hs, vs]
    sampled_velocity_vertical = velocity_vertical[hs, vs]
    visible_velocity = np.hypot(
        sampled_velocity_horizontal, sampled_velocity_vertical,
    ) >= 1.0e-6
    if np.any(visible_velocity):
        axes[0].quiver(
            horizontal_grid[visible_velocity], vertical_grid[visible_velocity],
            sampled_velocity_horizontal[visible_velocity],
            sampled_velocity_vertical[visible_velocity],
            color="black", alpha=0.75, pivot="mid",
        )
    axes[0].set_title("water fraction and velocity")
    figure.colorbar(alpha_image, ax=axes[0], label="alpha")

    pressure_image = axes[1].imshow(
        pressure_slice.T, origin="lower", extent=extent, cmap="coolwarm",
        interpolation="nearest", aspect="auto",
    )
    axes[1].imshow(
        np.ma.masked_where(~solid_slice.T, solid_slice.T),
        origin="lower", extent=extent, vmin=0.0, vmax=1.0,
        cmap="Greys", interpolation="nearest", aspect="auto", alpha=0.55,
    )
    mixed = (alpha_slice > 1.0e-6) & (alpha_slice < 1.0 - 1.0e-6)
    sampled_mixed = mixed[hs, vs]
    if np.any(sampled_mixed):
        axes[1].quiver(
            horizontal_grid[sampled_mixed], vertical_grid[sampled_mixed],
            normal_horizontal[hs, vs][sampled_mixed],
            normal_vertical[hs, vs][sampled_mixed],
            color="lime", pivot="mid", scale=18.0,
        )
    axes[1].set_title("pressure and water-to-air normals")
    figure.colorbar(pressure_image, ax=axes[1], label="pressure [Pa]")
    for axis_plot in axes:
        axis_plot.set_xlabel(horizontal_label)
        axis_plot.set_ylabel(vertical_label)
    figure.suptitle(f"3-D solver {plane} slice {selected}")
    figure.savefig(output, dpi=140)
    plt.close(figure)
    return SliceDiagnostic(path=output, plane=plane, index=selected)


def render_diagnostic_slices(
    state: State3D,
    output_dir: str | Path,
    *,
    prefix: str = "frame",
    xz_y: int | None = None,
    xy_z: int | None = None,
) -> tuple[SliceDiagnostic, SliceDiagnostic]:
    """Write the standard x-z and x-y diagnostic pair."""
    if not prefix or Path(prefix).name != prefix:
        raise ValueError("prefix must be one non-empty path component")
    directory = Path(output_dir)
    return (
        render_field_slice(state, directory / f"{prefix}_xz.png",
                           plane="xz", index=xz_y),
        render_field_slice(state, directory / f"{prefix}_xy.png",
                           plane="xy", index=xy_z),
    )
