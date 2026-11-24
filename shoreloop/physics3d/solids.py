"""Voxel solids sampled from the shared analytic scene geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from shoreloop.config import BC
from shoreloop.physics3d.boundary import fill_field_boundaries
from shoreloop.physics3d.extrapolation import _wrapped_cell_index
from shoreloop.physics3d.state import State3D
from shoreloop.physics3d.grid import MACGrid
from shoreloop.scene.geometry3d import AnalyticBeach, SceneGeometry


PlaneBeach = AnalyticBeach


@dataclass(frozen=True)
class VoxelSolidGeometry:
    """Cell and face samples of a full-cell scene approximation."""

    cell_sdf: np.ndarray
    face_sdf: tuple[np.ndarray, np.ndarray, np.ndarray]
    solid: np.ndarray
    material_ids: np.ndarray
    apertures: tuple[np.ndarray, np.ndarray, np.ndarray]


@dataclass(frozen=True)
class SolidFluxDiagnostics:
    """Resolved normal flow through full-cell solid boundaries."""

    net_flux: float
    absolute_flux: float
    maximum_normal_velocity: float
    boundary_faces: int


def _sample_scene(
    scene: SceneGeometry,
    grid: MACGrid,
    field: str,
) -> np.ndarray:
    axes = grid.coordinates(field)
    coordinates = np.meshgrid(*axes, indexing="ij")
    points = np.stack(coordinates, axis=-1)
    return np.asarray(
        scene.signed_distance(points), dtype=np.float64,
    ).reshape(points.shape[:-1])


def _voxel_apertures(
    solid: np.ndarray,
    face_sdf: tuple[np.ndarray, np.ndarray, np.ndarray],
    periodic: tuple[bool, bool, bool],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    apertures: list[np.ndarray] = []
    for axis, sdf in enumerate(face_sdf):
        open_face = sdf >= 0.0
        interior = [slice(None), slice(None), slice(None)]
        interior[axis] = slice(1, -1)
        lower = [slice(None), slice(None), slice(None)]
        lower[axis] = slice(None, -1)
        upper = [slice(None), slice(None), slice(None)]
        upper[axis] = slice(1, None)
        open_face[tuple(interior)] &= ~(
            solid[tuple(lower)] | solid[tuple(upper)]
        )

        low_face = [slice(None), slice(None), slice(None)]
        low_face[axis] = 0
        high_face = [slice(None), slice(None), slice(None)]
        high_face[axis] = -1
        low_cell = [slice(None), slice(None), slice(None)]
        low_cell[axis] = 0
        high_cell = [slice(None), slice(None), slice(None)]
        high_cell[axis] = -1
        if periodic[axis]:
            seam = (
                open_face[tuple(low_face)]
                & open_face[tuple(high_face)]
                & ~solid[tuple(low_cell)]
                & ~solid[tuple(high_cell)]
            )
            open_face[tuple(low_face)] = seam
            open_face[tuple(high_face)] = seam
        else:
            open_face[tuple(low_face)] &= ~solid[tuple(low_cell)]
            open_face[tuple(high_face)] &= ~solid[tuple(high_cell)]
        apertures.append(open_face.astype(np.float32))
    return (apertures[0], apertures[1], apertures[2])


def rasterize_voxel_solids(
    scene: SceneGeometry,
    state: State3D,
) -> VoxelSolidGeometry:
    """Sample one analytic scene into binary cell and face geometry."""
    cell_sdf = _sample_scene(scene, state.grid, "cell")
    face_sdf = tuple(
        _sample_scene(scene, state.grid, name) for name in ("u", "v", "w")
    )
    solid = cell_sdf < 0.0
    cell_axes = state.grid.coordinates("cell")
    cell_points = np.stack(np.meshgrid(*cell_axes, indexing="ij"), axis=-1)
    material = np.asarray(
        scene.material_ids(cell_points), dtype=np.int32,
    ).reshape(cell_points.shape[:-1])
    bc = state.cfg.bc
    periodic = (
        bc.xlo == BC.PERIODIC,
        bc.ylo == BC.PERIODIC,
        bc.zlo == BC.PERIODIC,
    )
    apertures = _voxel_apertures(solid, face_sdf, periodic)
    return VoxelSolidGeometry(
        cell_sdf, face_sdf, solid, material, apertures,
    )


def initialize_scene_voxels(
    state: State3D,
    scene: SceneGeometry,
) -> VoxelSolidGeometry:
    """Install the scene's full-cell approximation in a solver state."""
    geometry = rasterize_voxel_solids(scene, state)
    state.set_solid_interior(geometry.solid.astype(np.int32))
    state.set_material_interior(geometry.material_ids)
    state.set_geometry_interior("theta", (~geometry.solid).astype(np.float32))
    state.open_normals.zero_()
    state.open_offsets.zero_()
    for name, aperture in zip(
        ("u_aperture", "v_aperture", "w_aperture"),
        geometry.apertures,
        strict=True,
    ):
        state.set_geometry_interior(name, aperture)
    state.small_cell_threshold = 0.0
    state.small_cell_count = 0
    state.minimum_open_fraction = 1.0
    enforce_voxel_solids(state)
    return geometry


@wp.func
def _solid_cell(
    solid: wp.array3d(dtype=wp.int32),
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
) -> int:
    ci = _wrapped_cell_index(i, nx, periodic_x)
    cj = _wrapped_cell_index(j, ny, periodic_y)
    ck = _wrapped_cell_index(k, nz, periodic_z)
    if ci < 0 or cj < 0 or ck < 0:
        return 0
    return solid[ci + ng, cj + ng, ck + ng]


@wp.kernel
def _clear_solid_cells(
    pressure: wp.array3d(dtype=wp.float32),
    alpha: wp.array3d(dtype=wp.float32),
    solid: wp.array3d(dtype=wp.int32),
    ng: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    if solid[I, J, K] != 0:
        pressure[I, J, K] = wp.float32(0.0)
        alpha[I, J, K] = wp.float32(0.0)


@wp.kernel
def _close_solid_faces(
    velocity: wp.array3d(dtype=wp.float32),
    aperture: wp.array3d(dtype=wp.float32),
    solid: wp.array3d(dtype=wp.int32),
    axis: int,
    nx: int,
    ny: int,
    nz: int,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    ng: int,
):
    i, j, k = wp.tid()
    low_i = i
    low_j = j
    low_k = k
    if axis == 0:
        low_i -= 1
    elif axis == 1:
        low_j -= 1
    else:
        low_k -= 1
    if (aperture[i + ng, j + ng, k + ng] <= wp.float32(0.0)
            or _solid_cell(solid, low_i, low_j, low_k, nx, ny, nz,
                    periodic_x, periodic_y, periodic_z, ng) != 0
            or _solid_cell(solid, i, j, k, nx, ny, nz,
                           periodic_x, periodic_y, periodic_z, ng) != 0):
        velocity[i + ng, j + ng, k + ng] = wp.float32(0.0)


def enforce_voxel_solids(state: State3D) -> None:
    """Clear solid cells and close every face touching a solid voxel."""
    if state.solid_cells == 0:
        return
    g = state.cfg.grid
    bc = state.cfg.bc
    periodic = (
        int(bc.xlo == BC.PERIODIC), int(bc.ylo == BC.PERIODIC),
        int(bc.zlo == BC.PERIODIC),
    )
    wp.launch(
        _clear_solid_cells, dim=g.shape,
        inputs=[state.p, state.alpha, state.solid, g.ng], device=state.device,
    )
    for axis, name in enumerate(("u", "v", "w")):
        wp.launch(
            _close_solid_faces, dim=state.grid.interior_shape(name),
            inputs=[state.field(name), getattr(state, f"{name}_aperture"),
                    state.solid, axis,
                    g.nx, g.ny, g.nz, *periodic, g.ng],
            device=state.device,
        )
        fill_field_boundaries(state, name)
    fill_field_boundaries(state, "p")
    fill_field_boundaries(state, "alpha")


def measure_solid_flux(state: State3D) -> SolidFluxDiagnostics:
    """Integrate normal velocity on every fluid/voxel-solid interface."""
    solid = state.get_solid_interior().astype(bool)
    velocities = tuple(state.get_interior(name) for name in ("u", "v", "w"))
    g = state.cfg.grid
    periodic = (
        state.cfg.bc.xlo == BC.PERIODIC,
        state.cfg.bc.ylo == BC.PERIODIC,
        state.cfg.bc.zlo == BC.PERIODIC,
    )
    areas = (g.dy * g.dz, g.dx * g.dz, g.dx * g.dy)
    signed_fluxes: list[np.ndarray] = []
    for axis, (velocity, area) in enumerate(zip(velocities, areas, strict=True)):
        low = [slice(None), slice(None), slice(None)]
        low[axis] = slice(None, -1)
        high = [slice(None), slice(None), slice(None)]
        high[axis] = slice(1, None)
        face = [slice(None), slice(None), slice(None)]
        face[axis] = slice(1, -1)
        low_solid = solid[tuple(low)]
        high_solid = solid[tuple(high)]
        direction = low_solid.astype(np.int8) - high_solid.astype(np.int8)
        selected = direction != 0
        signed_fluxes.append(
            velocity[tuple(face)][selected] * direction[selected] * area,
        )
        if periodic[axis]:
            low_cell = [slice(None), slice(None), slice(None)]
            low_cell[axis] = 0
            high_cell = [slice(None), slice(None), slice(None)]
            high_cell[axis] = -1
            seam_direction = (
                solid[tuple(high_cell)].astype(np.int8)
                - solid[tuple(low_cell)].astype(np.int8)
            )
            seam_face = [slice(None), slice(None), slice(None)]
            seam_face[axis] = 0
            selected = seam_direction != 0
            signed_fluxes.append(
                velocity[tuple(seam_face)][selected]
                * seam_direction[selected] * area,
            )
    values = np.concatenate(signed_fluxes) if signed_fluxes else np.empty(0)
    normal_velocities = np.concatenate([
        values / area for values, area in zip(
            signed_fluxes,
            [areas[q] for q in range(3) for _ in range(1 + int(periodic[q]))],
            strict=True,
        )
    ]) if signed_fluxes else np.empty(0)
    return SolidFluxDiagnostics(
        net_flux=float(values.sum()),
        absolute_flux=float(np.abs(values).sum()),
        maximum_normal_velocity=float(np.max(np.abs(normal_velocities), initial=0.0)),
        boundary_faces=len(values),
    )


def initialize_plane_beach(state: State3D, beach: PlaneBeach) -> int:
    """Compatibility wrapper around the shared analytic scene rasteriser."""
    geometry = initialize_scene_voxels(state, SceneGeometry(beach))
    return int(geometry.solid.sum())
