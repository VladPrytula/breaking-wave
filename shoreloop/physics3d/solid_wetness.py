"""Persistent optical wetness attached to analytic solid surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
import warp as wp

from shoreloop.config import BC, Solver3DConfig
from shoreloop.physics3d.advection import _copy_cell_field
from shoreloop.physics3d.boundary import fill_field_boundaries
from shoreloop.physics3d.extrapolation import _wrapped_cell_index
from shoreloop.physics3d.state import State3D
from shoreloop.scene.geometry3d import SceneGeometry, TriangleMesh


@dataclass(frozen=True)
class SolidWetnessConfig:
    wetting_time: float = 0.25
    drying_time: float = 30.0
    contact_threshold: float = 0.05

    def __post_init__(self) -> None:
        if self.wetting_time <= 0.0 or self.drying_time <= 0.0:
            raise ValueError("wetness time scales must be positive")
        if not 0.0 <= self.contact_threshold <= 1.0:
            raise ValueError("contact_threshold must be in [0, 1]")


@dataclass(frozen=True)
class SolidWetnessResult:
    contacted_cells: int
    maximum_wetness: float


@dataclass(frozen=True)
class SurfaceWetnessSample:
    material_id: int
    primitive_id: int
    coordinates: np.ndarray
    wetness: np.ndarray


@wp.kernel
def _update_solid_wetness(
    alpha: wp.array3d(dtype=wp.float32),
    theta: wp.array3d(dtype=wp.float32),
    material: wp.array3d(dtype=wp.int32),
    wetness: wp.array3d(dtype=wp.float32),
    updated: wp.array3d(dtype=wp.float32),
    contacted_cells: wp.array(dtype=wp.int32),
    maximum: wp.array(dtype=wp.float32),
    dt: wp.float32,
    wetting_time: wp.float32,
    drying_time: wp.float32,
    contact_threshold: wp.float32,
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
    is_surface = int(
        material[I, J, K] != 0
        and theta[I, J, K] < wp.float32(1.0)
    )
    contact = alpha[I, J, K] * theta[I, J, K]
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
        ni = _wrapped_cell_index(ni, nx, periodic_x)
        nj = _wrapped_cell_index(nj, ny, periodic_y)
        nk = _wrapped_cell_index(nk, nz, periodic_z)
        if ni >= 0 and nj >= 0 and nk >= 0:
            NI = ni + ng
            NJ = nj + ng
            NK = nk + ng
            if material[I, J, K] != 0 and theta[NI, NJ, NK] > wp.float32(0.0):
                is_surface = 1
            contact = wp.max(
                contact, alpha[NI, NJ, NK] * theta[NI, NJ, NK],
            )
    if is_surface == 0:
        updated[I, J, K] = wp.float32(0.0)
        return
    target = wp.float32(0.0)
    timescale = drying_time
    if contact >= contact_threshold:
        target = wp.float32(1.0)
        timescale = wetting_time
        wp.atomic_add(contacted_cells, 0, 1)
    ratio = dt / timescale
    value = (wetness[I, J, K] + ratio * target) / (wp.float32(1.0) + ratio)
    updated[I, J, K] = value
    wp.atomic_max(maximum, 0, value)


class SolidWetnessModel:
    """Update surface memory independently of fluid mass and momentum."""

    __slots__ = (
        "cfg", "settings", "device", "_updated", "_contacted", "_maximum",
    )

    def __init__(
        self,
        cfg: Solver3DConfig,
        settings: SolidWetnessConfig | None = None,
    ):
        self.cfg = cfg
        self.settings = SolidWetnessConfig() if settings is None else settings
        self.device = cfg.device
        shape = tuple(n + 2 * cfg.grid.ng for n in cfg.grid.shape)
        self._updated = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self._contacted = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._maximum = wp.zeros(1, dtype=wp.float32, device=self.device)

    def apply(self, state: State3D, dt: float) -> SolidWetnessResult:
        self._updated.zero_()
        self._contacted.zero_()
        self._maximum.zero_()
        g = self.cfg.grid
        periodic = (
            int(self.cfg.bc.xlo == BC.PERIODIC),
            int(self.cfg.bc.ylo == BC.PERIODIC),
            int(self.cfg.bc.zlo == BC.PERIODIC),
        )
        wp.launch(
            _update_solid_wetness,
            dim=g.shape,
            inputs=[
                state.alpha, state.theta, state.material,
                state.solid_wetness, self._updated,
                self._contacted, self._maximum, dt,
                self.settings.wetting_time, self.settings.drying_time,
                self.settings.contact_threshold,
                g.nx, g.ny, g.nz, *periodic, g.ng,
            ],
            device=self.device,
        )
        wp.launch(
            _copy_cell_field,
            dim=g.shape,
            inputs=[self._updated, state.solid_wetness, g.ng],
            device=self.device,
        )
        fill_field_boundaries(state, "solid_wetness")
        return SolidWetnessResult(
            contacted_cells=int(self._contacted.numpy()[0]),
            maximum_wetness=float(self._maximum.numpy()[0]),
        )


def _mesh_wetness(
    state: State3D,
    mesh: TriangleMesh,
    wetness: np.ndarray,
    material: np.ndarray,
) -> np.ndarray:
    g = state.cfg.grid
    origin = np.asarray((g.x0, g.y0, g.z0))
    spacing = np.asarray((g.dx, g.dy, g.dz))
    logical = np.floor((mesh.vertices - origin) / spacing).astype(np.int64)
    sampled = np.zeros(len(mesh.vertices), dtype=np.float32)
    best_distance = np.full(len(mesh.vertices), np.inf)
    maximum = np.asarray(g.shape) - 1
    periodic_y = state.cfg.bc.ylo == BC.PERIODIC
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for dk in (-1, 0, 1):
                neighbour = logical + (di, dj, dk)
                index = np.clip(neighbour, 0, maximum)
                if periodic_y:
                    index[:, 1] = np.mod(neighbour[:, 1], g.ny)
                index_tuple = tuple(index[:, axis] for axis in range(3))
                centre_index = index.copy()
                if periodic_y:
                    centre_index[:, 1] = neighbour[:, 1]
                cell_centre = origin + (centre_index + 0.5) * spacing
                distance = np.sum(
                    (cell_centre - mesh.vertices) ** 2, axis=1,
                )
                selected = (
                    (material[index_tuple] == mesh.material_id)
                    & (distance < best_distance)
                )
                best_distance[selected] = distance[selected]
                sampled[selected] = wetness[index_tuple][selected]
    return sampled


def sample_scene_surface_wetness(
    state: State3D,
    scene: SceneGeometry,
    beach_bounds: tuple[float, float, float, float],
    *,
    beach_segments: tuple[int, int] = (64, 32),
    boulder_segments: tuple[int, int] = (64, 32),
) -> tuple[SurfaceWetnessSample, ...]:
    """Sample the 3-D memory field onto stable analytic render primitives."""
    meshes = scene.render_meshes(
        beach_bounds,
        beach_segments=beach_segments,
        boulder_segments=boulder_segments,
    )
    wetness = state.get_interior("solid_wetness")
    material = state.get_material_interior()
    samples = []
    for primitive_id, mesh in enumerate(meshes):
        primitive = scene.beach if primitive_id == 0 else scene.boulders[primitive_id - 1]
        samples.append(SurfaceWetnessSample(
            material_id=mesh.material_id,
            primitive_id=primitive_id,
            coordinates=primitive.surface_coordinates(mesh.vertices),
            wetness=_mesh_wetness(state, mesh, wetness, material),
        ))
    return tuple(samples)
