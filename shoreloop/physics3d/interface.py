"""Interface-normal reconstruction from the authoritative VOF field."""

from __future__ import annotations

from dataclasses import dataclass
import math

import warp as wp

from shoreloop.config import BC
from shoreloop.physics3d.boundary import fill_field_boundaries
from shoreloop.physics3d.cut_normals import fit_cut_interface_normals
from shoreloop.physics3d.height_normals import apply_height_function_normals
from shoreloop.physics3d.state import State3D


@dataclass(frozen=True)
class InterfaceNormalResult:
    """Cell-centred water-to-air normals and reconstruction diagnostics."""

    normals: wp.array
    mixed_cells: int
    ambiguous_cells: int
    height_function_cells: int = 0


@wp.kernel
def _estimate_youngs_normals(
    alpha: wp.array3d(dtype=wp.float32),
    normals: wp.array3d(dtype=wp.vec3),
    counts: wp.array(dtype=wp.int32),
    inverse_dx: wp.float32,
    inverse_dy: wp.float32,
    inverse_dz: wp.float32,
    mixed_tolerance: wp.float32,
    gradient_epsilon: wp.float32,
    ng: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    fraction = alpha[I, J, K]
    zero = wp.vec3(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    if fraction <= mixed_tolerance or fraction >= wp.float32(1.0) - mixed_tolerance:
        normals[I, J, K] = zero
        return

    wp.atomic_add(counts, 0, 1)

    # Youngs' 3x3 transverse weighting [1, 2, 1] damps the grid-direction
    # bias of a single centred difference.  The common factor of 1/32 does not
    # affect the normalised direction and is intentionally omitted.
    gx = wp.float32(0.0)
    gy = wp.float32(0.0)
    gz = wp.float32(0.0)
    for a in range(3):
        da = a - 1
        wa = wp.float32(1.0)
        if a == 1:
            wa = wp.float32(2.0)
        for b in range(3):
            db = b - 1
            wb = wp.float32(1.0)
            if b == 1:
                wb = wp.float32(2.0)
            weight = wa * wb
            gx += weight * (alpha[I + 1, J + da, K + db]
                            - alpha[I - 1, J + da, K + db]) * inverse_dx
            gy += weight * (alpha[I + da, J + 1, K + db]
                            - alpha[I + da, J - 1, K + db]) * inverse_dy
            gz += weight * (alpha[I + da, J + db, K + 1]
                            - alpha[I + da, J + db, K - 1]) * inverse_dz

    # alpha rises from air to water, hence -grad(alpha) points water to air.
    candidate = -wp.vec3(gx, gy, gz)
    magnitude = wp.length(candidate)
    if magnitude > gradient_epsilon and wp.isfinite(magnitude):
        normals[I, J, K] = candidate / magnitude
    else:
        # A mixed but locally symmetric field has no recoverable orientation.
        # Keep the result finite and deterministic, and make the ambiguity
        # explicit rather than allowing an uninitialised or NaN normal.
        normals[I, J, K] = wp.vec3(
            wp.float32(0.0), wp.float32(0.0), wp.float32(1.0),
        )
        wp.atomic_add(counts, 1, 1)


def estimate_interface_normals(
    state: State3D,
    *,
    mixed_tolerance: float = 1.0e-6,
    gradient_epsilon: float = 1.0e-12,
    out: wp.array | None = None,
    fit_cut_cells: bool = True,
) -> InterfaceNormalResult:
    """Reconstruct robust cell-centred normals oriented from water to air.

    ``fit_cut_cells=False`` leaves the solid-adjacent cells with their Youngs or
    height-function normal instead of the LVIRA fit. A caller that reads only
    cells away from every solid may skip the fit; it is the expensive part.
    """
    if (not math.isfinite(mixed_tolerance) or mixed_tolerance < 0.0
            or mixed_tolerance >= 0.5):
        raise ValueError("mixed_tolerance must be finite in [0, 0.5)")
    if not math.isfinite(gradient_epsilon) or gradient_epsilon <= 0.0:
        raise ValueError("gradient_epsilon must be positive and finite")

    grid = state.grid
    if out is None:
        out = wp.zeros(grid.cell_shape, dtype=wp.vec3, device=state.device)
    else:
        if tuple(out.shape) != grid.cell_shape:
            raise ValueError(
                f"normal output must have padded shape {grid.cell_shape}, "
                f"got {tuple(out.shape)}",
            )
        if out.dtype != wp.vec3:
            raise TypeError(f"normal output must use vec3, got {out.dtype}")
        out.zero_()

    fill_field_boundaries(state, "alpha")
    counts = wp.zeros(3, dtype=wp.int32, device=state.device)
    g = state.cfg.grid
    wp.launch(
        _estimate_youngs_normals,
        dim=g.shape,
        inputs=[state.alpha, out, counts, 1.0 / g.dx, 1.0 / g.dy,
                1.0 / g.dz, mixed_tolerance, gradient_epsilon, g.ng],
        device=state.device,
    )
    if state.cfg.interface_reconstruction == "height_function":
        wp.launch(
            apply_height_function_normals, dim=g.shape, device=state.device,
            inputs=[state.alpha, state.theta, out, counts,
                    wp.vec3(g.dx, g.dy, g.dz), wp.vec3i(*g.shape),
                    wp.vec3i(*(int(getattr(state.cfg.bc, axis + "lo") == BC.PERIODIC)
                               for axis in "xyz")), mixed_tolerance, g.ng],
        )
    counts_host = counts.numpy()
    if fit_cut_cells:
        fit_cut_interface_normals(
            state.alpha, state.theta, state.open_normals, state.open_offsets,
            out, wp.vec3(g.dx, g.dy, g.dz), wp.vec3i(*g.shape),
            wp.vec3i(*(int(getattr(state.cfg.bc, axis + "lo") == BC.PERIODIC)
                       for axis in "xyz")), mixed_tolerance, g.ng,
            mixed_cells=int(counts_host[0]), cache=state.cut_normal_cache,
        )
    return InterfaceNormalResult(
        normals=out,
        mixed_cells=int(counts_host[0]),
        ambiguous_cells=int(counts_host[1]),
        height_function_cells=int(counts_host[2]),
    )
