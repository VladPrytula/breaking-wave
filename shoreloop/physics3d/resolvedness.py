"""Interface resolvedness of the retained PLIC planes. Diagnostic only.

Nothing governing reads these numbers. Per interface cell (a cell holding a
retained plane) two measures are made:

* ``angle_deg``: the facet-area-weighted mean angle between the cell's
  water-to-air normal and the normals of the interface cells among its 26
  neighbours (the resolvedness angle of Scheufler and Roenby's plicRDF, which
  treats a cell as unresolved above ``UNRESOLVED_ANGLE_DEG``). A cell without
  an interface neighbour reports ``-1``.
* ``residual``: the per-neighbour RMS retained-volume mismatch, in cell
  fractions, of the cell's own plane continued into its 27-cell stencil (the
  square root of the LVIRA objective of ``cut_normals._neighbour_residual``
  over the 26 neighbours). A locally planar interface gives about zero; water
  smeared over several cells of a column cannot be matched by any single plane
  and keeps a residual of the order of the smeared fraction whatever the
  normal.

A cell is unresolved above ``UNRESOLVED_ANGLE_DEG`` or above
``UNRESOLVED_RESIDUAL``: the angle catches sheets that fold or meet, the
residual catches a transition smeared over several cells whose normals still
agree.

Per plan column, ``stacked_partial`` is the largest number of partial cells
(``partial_tolerance < alpha < 1 - partial_tolerance`` in open cells) within
any window of ``MIXTURE_WINDOW_CELLS`` consecutive cells, and
``stacked_unresolved`` the same count over windows that also hold an
unresolved cell. A column with ``stacked_unresolved`` of
``MIXTURE_STACKED_PARTIAL`` or more is a mixture column: the water there is not
one sheet, and no single plane per cell describes it. Two stacked partial
cells whose planes disagree are the smallest such stack. A steep single sheet
also stacks partial cells, but its planes agree, so it stays resolved.
``partial_volume_m3`` holds the water volume of a column's partial cells.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import warp as wp

from shoreloop.config import BC
from shoreloop.physics3d.cut_normals import _neighbour_residual
from shoreloop.physics3d.extrapolation import _wrapped_cell_index
from shoreloop.physics3d.plic import PLICReconstruction
from shoreloop.physics3d.polyhedron import _clip_polygon, _plane_quad
from shoreloop.physics3d.state import State3D

# Column classification: two or more partial cells within four consecutive
# cells (40 mm at dz 10 mm), one of them unresolved, is water in more than one
# sheet. Measured 2026-10-10 on state_1.52s: this stack carries 78% of the
# bore-band stair area on mixture pairs where three cells carry 44%, and
# flags no forcing-band column.
MIXTURE_WINDOW_CELLS = 4
MIXTURE_STACKED_PARTIAL = 2
# Cells below 2% water are residue, not a sheet (the production resolved fraction).
MIXTURE_PARTIAL_FRACTION = 0.02
# plicRDF's resolvedness cut-off.
UNRESOLVED_ANGLE_DEG = 15.0
# Per-neighbour RMS plane mismatch (cell fractions) above which no single plane
# describes the cell's stencil; height-function cells measure 0.004-0.01.
UNRESOLVED_RESIDUAL = 0.1


@wp.func
def _facet_area(
    theta: wp.float32, open_normal: wp.vec3, open_offset: wp.float32,
    normal: wp.vec3, offset: wp.float32, spacing: wp.vec3,
) -> wp.float32:
    """Physical area of the plane's section of the open part of the unit cell."""
    polygon = _plane_quad(normal, offset)
    count = int(4)
    for face in range(6):
        n = wp.vec3(wp.float32(0.0))
        n[face // 2] = wp.float32(2 * (face % 2) - 1)
        polygon, count = _clip_polygon(polygon, count, n, wp.float32(face % 2))
    if theta < wp.float32(1.0):
        polygon, count = _clip_polygon(polygon, count, open_normal, open_offset)
    area = wp.float32(0.0)
    if count >= 3:
        a = wp.cw_mul(polygon[0], spacing)
        for vertex in range(1, count - 1):
            b = wp.cw_mul(polygon[vertex], spacing)
            c = wp.cw_mul(polygon[vertex + 1], spacing)
            area += wp.float32(0.5) * wp.length(wp.cross(b - a, c - a))
    return area


@wp.kernel
def _facet_areas(
    theta: wp.array3d(dtype=wp.float32),
    open_normals: wp.array3d(dtype=wp.vec3),
    open_offsets: wp.array3d(dtype=wp.float32),
    normals: wp.array3d(dtype=wp.vec3),
    offsets: wp.array3d(dtype=wp.float32),
    area: wp.array3d(dtype=wp.float32),
    spacing: wp.vec3, ng: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    n = normals[I, J, K]
    if wp.dot(n, n) == wp.float32(0.0):
        return
    area[I, J, K] = _facet_area(theta[I, J, K], open_normals[I, J, K], open_offsets[I, J, K],
                                n, offsets[I, J, K], spacing)


@wp.kernel
def _resolvedness(
    alpha: wp.array3d(dtype=wp.float32),
    theta: wp.array3d(dtype=wp.float32),
    open_normals: wp.array3d(dtype=wp.vec3),
    open_offsets: wp.array3d(dtype=wp.float32),
    normals: wp.array3d(dtype=wp.vec3),
    offsets: wp.array3d(dtype=wp.float32),
    area: wp.array3d(dtype=wp.float32),
    angle: wp.array3d(dtype=wp.float32),
    residual: wp.array3d(dtype=wp.float32),
    inverse_spacing: wp.vec3, shape: wp.vec3i, periodic: wp.vec3i, ng: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    n = normals[I, J, K]
    if wp.dot(n, n) == wp.float32(0.0):
        return
    d = offsets[I, J, K]
    # The retained plane is L1-normalised in cell-local coordinates; its
    # physical direction follows from dividing by the spacing.
    m = wp.normalize(wp.cw_mul(n, inverse_spacing))
    weight = wp.float32(0.0)
    weighted_angle = wp.float32(0.0)
    error = wp.float32(0.0)
    neighbours = int(27)
    for q in range(neighbours):
        error += _neighbour_residual(n, d, alpha, theta, open_normals, open_offsets,
                                     i, j, k, q, shape, periodic, ng)
        if q == 13:
            continue
        a = _wrapped_cell_index(i + q // 9 - 1, shape[0], periodic[0])
        b = _wrapped_cell_index(j + q // 3 % 3 - 1, shape[1], periodic[1])
        c = _wrapped_cell_index(k + q % 3 - 1, shape[2], periodic[2])
        if a < 0 or b < 0 or c < 0:
            continue
        nq = normals[a + ng, b + ng, c + ng]
        if wp.dot(nq, nq) == wp.float32(0.0):
            continue
        mq = wp.normalize(wp.cw_mul(nq, inverse_spacing))
        w = area[a + ng, b + ng, c + ng]
        # atan2 of the cross and dot products stays accurate for the small
        # angles of a resolved surface, where float32 acos quantises at 0.02 deg.
        weighted_angle += w * wp.atan2(wp.length(wp.cross(m, mq)), wp.dot(m, mq))
        weight += w
    residual[I, J, K] = wp.sqrt(error / wp.float32(26.0))
    angle[I, J, K] = wp.float32(-1.0)
    if weight > wp.float32(0.0):
        angle[I, J, K] = wp.degrees(weighted_angle / weight)


@wp.kernel
def _column_mixture(
    alpha: wp.array3d(dtype=wp.float32),
    theta: wp.array3d(dtype=wp.float32),
    angle: wp.array3d(dtype=wp.float32),
    residual: wp.array3d(dtype=wp.float32),
    stacked: wp.array2d(dtype=wp.int32),
    stacked_unresolved: wp.array2d(dtype=wp.int32),
    partial_volume: wp.array2d(dtype=wp.float32),
    nz: int, tolerance: wp.float32, window: int,
    angle_limit: wp.float32, residual_limit: wp.float32, cell_volume: wp.float32, ng: int,
):
    i, j = wp.tid()
    partial_history = int(0)
    unresolved_history = int(0)
    best = int(0)
    best_unresolved = int(0)
    volume = wp.float32(0.0)
    for k in range(nz):
        I = i + ng
        J = j + ng
        K = k + ng
        partial = int(0)
        unresolved = int(0)
        if (theta[I, J, K] > wp.float32(0.0) and alpha[I, J, K] > tolerance
                and alpha[I, J, K] < wp.float32(1.0) - tolerance):
            partial = 1
            volume += alpha[I, J, K] * theta[I, J, K] * cell_volume
            if angle[I, J, K] > angle_limit or residual[I, J, K] > residual_limit:
                unresolved = 1
        mask = (1 << window) - 1
        partial_history = ((partial_history << 1) | partial) & mask
        unresolved_history = ((unresolved_history << 1) | unresolved) & mask
        count = int(0)
        for bit in range(window):
            count += (partial_history >> bit) & 1
        best = wp.max(best, count)
        if unresolved_history != 0:
            best_unresolved = wp.max(best_unresolved, count)
    stacked[i, j] = best
    stacked_unresolved[i, j] = best_unresolved
    partial_volume[i, j] = volume


@dataclass(frozen=True)
class InterfaceResolvedness:
    """Per-interface-cell resolvedness and per-column mixture classification."""

    shape: tuple[int, int, int]
    cells: np.ndarray  # (n,) int32 flat interior index (i * ny + j) * nz + k
    angle_deg: np.ndarray  # (n,) float32; -1 without an interface neighbour
    residual: np.ndarray  # (n,) float32 cell fractions
    facet_area_m2: np.ndarray  # (n,) float32
    stacked_partial: np.ndarray  # (nx, ny) int8
    stacked_unresolved: np.ndarray  # (nx, ny) int8
    partial_volume_m3: np.ndarray  # (nx, ny) float32

    @property
    def cell_indices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return np.unravel_index(self.cells, self.shape)

    @property
    def mixture_columns(self) -> np.ndarray:
        return self.stacked_unresolved >= MIXTURE_STACKED_PARTIAL

    @property
    def in_mixture_column(self) -> np.ndarray:
        i, j, _ = self.cell_indices
        return self.mixture_columns[i, j]

    @property
    def unresolved(self) -> np.ndarray:
        return (self.angle_deg > UNRESOLVED_ANGLE_DEG) | (self.residual > UNRESOLVED_RESIDUAL)

    def summary(self) -> dict:
        mixture = self.in_mixture_column
        return {
            "interface_cells": int(len(self.cells)),
            "unresolved_cells": int(np.count_nonzero(self.unresolved)),
            "mixture_columns": int(np.count_nonzero(self.mixture_columns)),
            "mixture_interface_cells": int(np.count_nonzero(mixture)),
            "mixture_partial_volume_m3": float(self.partial_volume_m3[self.mixture_columns].sum()),
            "angle_deg_p50": float(np.median(self.angle_deg)) if len(self.cells) else 0.0,
            "residual_p50": float(np.median(self.residual)) if len(self.cells) else 0.0,
        }

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        np.savez_compressed(
            path, shape=np.asarray(self.shape, dtype=np.int32), cells=self.cells,
            angle_deg=self.angle_deg, residual=self.residual, facet_area_m2=self.facet_area_m2,
            stacked_partial=self.stacked_partial, stacked_unresolved=self.stacked_unresolved,
            partial_volume_m3=self.partial_volume_m3,
        )
        return path

    @classmethod
    def read(cls, path: str | Path) -> InterfaceResolvedness:
        with np.load(path) as saved:
            return cls(
                shape=tuple(int(n) for n in saved["shape"]), cells=saved["cells"],
                angle_deg=saved["angle_deg"], residual=saved["residual"],
                facet_area_m2=saved["facet_area_m2"], stacked_partial=saved["stacked_partial"],
                stacked_unresolved=saved["stacked_unresolved"],
                partial_volume_m3=saved["partial_volume_m3"],
            )


def measure_interface_resolvedness(
    state: State3D,
    reconstruction: PLICReconstruction,
    *,
    partial_tolerance: float = MIXTURE_PARTIAL_FRACTION,
) -> InterfaceResolvedness:
    """Measure the resolvedness of ``reconstruction``, the planes of this state.

    The state and the reconstruction are only read.
    """
    if not 0.0 < partial_tolerance < 0.5:
        raise ValueError("partial_tolerance must lie in (0, 0.5)")
    g = state.cfg.grid
    padded = state.grid.cell_shape
    device = state.device
    area = wp.zeros(padded, dtype=wp.float32, device=device)
    angle = wp.zeros(padded, dtype=wp.float32, device=device)
    residual = wp.zeros(padded, dtype=wp.float32, device=device)
    stacked = wp.zeros((g.nx, g.ny), dtype=wp.int32, device=device)
    stacked_unresolved = wp.zeros((g.nx, g.ny), dtype=wp.int32, device=device)
    partial_volume = wp.zeros((g.nx, g.ny), dtype=wp.float32, device=device)
    spacing = wp.vec3(g.dx, g.dy, g.dz)
    periodic = wp.vec3i(*(int(getattr(state.cfg.bc, axis + "lo") == BC.PERIODIC) for axis in "xyz"))
    wp.launch(_facet_areas, dim=g.shape, device=device,
              inputs=[state.theta, state.open_normals, state.open_offsets,
                      reconstruction.normals, reconstruction.offsets, area, spacing, g.ng])
    wp.launch(_resolvedness, dim=g.shape, device=device,
              inputs=[state.alpha, state.theta, state.open_normals, state.open_offsets,
                      reconstruction.normals, reconstruction.offsets, area, angle, residual,
                      wp.vec3(1.0 / g.dx, 1.0 / g.dy, 1.0 / g.dz), wp.vec3i(*g.shape),
                      periodic, g.ng])
    wp.launch(_column_mixture, dim=(g.nx, g.ny), device=device,
              inputs=[state.alpha, state.theta, angle, residual, stacked, stacked_unresolved,
                      partial_volume, g.nz, float(partial_tolerance), MIXTURE_WINDOW_CELLS,
                      UNRESOLVED_ANGLE_DEG, UNRESOLVED_RESIDUAL, float(g.dx * g.dy * g.dz), g.ng])
    interior = state.grid.interior_slices("cell")
    normals = reconstruction.normals.numpy()[interior]
    cells = np.flatnonzero(np.any(normals != 0.0, axis=-1)).astype(np.int32)
    return InterfaceResolvedness(
        shape=tuple(g.shape), cells=cells,
        angle_deg=angle.numpy()[interior].reshape(-1)[cells],
        residual=residual.numpy()[interior].reshape(-1)[cells],
        facet_area_m2=area.numpy()[interior].reshape(-1)[cells],
        stacked_partial=stacked.numpy().astype(np.int8),
        stacked_unresolved=stacked_unresolved.numpy().astype(np.int8),
        partial_volume_m3=partial_volume.numpy(),
    )
