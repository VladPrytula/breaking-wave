"""Shared scene sampling, wetness, look and offshore-join helpers of the production
capture and render pipeline, moved from the original evaluation tools."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from shoreloop.closures.ripples import RippleField
from shoreloop.config import GridConfig
from shoreloop.physics3d.boulder_cluster import _section_flow
from shoreloop.physics3d.boulder_wash import _cell_velocity
from shoreloop.physics3d.production_scene import ProductionCamera
from shoreloop.physics3d.solid_wetness import SurfaceWetnessSample
from shoreloop.render.camera import Camera
from shoreloop.render.mitsuba_water import MitsubaWaterRenderer
from shoreloop.render.offshore_join import periodic_render_y_bounds
from shoreloop.render.surface_mesh import WaterSurfaceMesh


COLLAR_WIDTH = 0.4
OFFSHORE_MESH_LENGTH = 4.0


@dataclass(frozen=True)
class CameraCandidate:
    name: str
    position: tuple[float, float, float]
    target: tuple[float, float, float]
    vertical_fov_degrees: float

    def production_camera(self) -> ProductionCamera:
        direction = np.asarray(self.target) - np.asarray(self.position)
        yaw = math.degrees(math.atan2(direction[1], direction[0])) % 360.0
        pitch = math.degrees(math.atan2(
            -direction[2],
            np.linalg.norm(direction[:2]),
        ))
        return ProductionCamera(
            position=self.position,
            yaw_degrees=yaw,
            pitch_degrees=pitch,
            vertical_fov_degrees=self.vertical_fov_degrees,
            exposure=1.12,
            white_balance=(1.0, 1.0, 1.0),
        )


CAMERA_CANDIDATES = (
    CameraCandidate(
        "low_oblique",
        position=(6.75, 5.05, 1.30),
        target=(2.65, 3.85, 0.32),
        vertical_fov_degrees=37.0,
    ),
    CameraCandidate(
        "balanced_oblique",
        position=(6.75, 5.15, 1.55),
        target=(2.70, 3.85, 0.34),
        vertical_fov_degrees=38.0,
    ),
    CameraCandidate(
        "high_oblique",
        position=(6.75, 5.05, 1.85),
        target=(2.70, 3.85, 0.36),
        vertical_fov_degrees=38.0,
    ),
    CameraCandidate(
        "crest_offset_oblique",
        position=(6.75, 5.15, 1.55),
        target=(2.70, 3.95, 0.34),
        vertical_fov_degrees=38.0,
    ),
)


def _stored_volume(solver) -> float:
    grid = solver.cfg.grid
    return float(np.sum(
        solver.state.get_interior("porous_storage"), dtype=np.float64,
    )) * grid.dx * grid.dy * grid.dz


def _mean_shoreline(solver, *, depth_fraction: float = 0.25) -> float:
    grid = solver.cfg.grid
    x, _, _ = solver.state.grid.coordinates("cell")
    alpha = solver.state.get_interior("alpha")
    theta = solver.state.get_geometry_interior("theta")
    depth = np.sum(alpha * theta, axis=2) * grid.dz
    return float(np.mean([
        x[np.flatnonzero(depth[:, j] > depth_fraction * grid.dz)[-1]]
        + 0.5 * grid.dx
        for j in range(grid.ny)
    ]))


def _terminal_sheet_metrics(solver) -> dict[str, float]:
    grid = solver.cfg.grid
    x, _, _ = solver.state.grid.coordinates("cell")
    alpha = solver.state.get_interior("alpha")
    theta = solver.state.get_geometry_interior("theta")
    weight = alpha * theta
    depth = np.sum(weight, axis=2) * grid.dz
    u, v, w = _cell_velocity(solver)
    speed = np.sqrt(u * u + v * v + w * w)
    sheet = x[:, None, None] >= 2.90
    active = sheet & (weight > 0.05)
    sheet_weight = np.where(sheet, weight, 0.0)
    return {
        "terminal_sheet_mean_u_m_per_s": float(
            np.sum(sheet_weight * u, dtype=np.float64)
            / np.sum(sheet_weight, dtype=np.float64)
        ),
        "terminal_sheet_p95_speed_m_per_s": float(
            np.quantile(speed[active], 0.95)
        ),
        "terminal_sheet_moving_fraction": float(
            np.sum(sheet_weight * (speed > 0.02), dtype=np.float64)
            / np.sum(sheet_weight, dtype=np.float64)
        ),
        "terminal_sheet_wet_area_m2": float(np.count_nonzero(
            (x[:, None] >= 2.90) & (depth > 0.25 * grid.dz)
        )) * grid.dx * grid.dy,
    }


def _sample_state(
    solver,
    tracker,
    gaps,
    cluster_bounds: tuple[float, float, float, float],
    time_s: float,
) -> dict[str, object]:
    grid = solver.cfg.grid
    x, y, _ = solver.state.grid.coordinates("cell")
    alpha = solver.state.get_interior("alpha")
    theta = solver.state.get_geometry_interior("theta")
    depth = np.sum(alpha * theta, axis=2) * grid.dz
    xlo, xhi, ylo, yhi = cluster_bounds
    cluster = (
        (x[:, None] >= xlo)
        & (x[:, None] <= xhi)
        & (y[None, :] >= ylo)
        & (y[None, :] <= yhi)
    )
    topology = tracker.sample(solver.state, time=time_s)
    flows = [_section_flow(solver, gap, alpha) for gap in gaps]
    return {
        "time_s": time_s,
        "downward_interface_cells": topology.downward_interface_cells,
        "supported_downward_interface_cells": (
            topology.supported_downward_interface_cells
        ),
        "multi_transition_columns": topology.multi_transition_columns,
        "largest_connected_multi_transition_columns": (
            topology.largest_connected_multi_transition_columns
        ),
        "connected_multi_transition_x_span_m": (
            topology.connected_multi_transition_x_span
        ),
        "connected_multi_transition_y_span_m": (
            topology.connected_multi_transition_y_span
        ),
        "maximum_vertical_transitions": topology.maximum_vertical_transitions,
        "mean_shoreline_x": _mean_shoreline(solver),
        "mean_one_cell_shoreline_x": _mean_shoreline(
            solver,
            depth_fraction=1.0,
        ),
        **_terminal_sheet_metrics(solver),
        "maximum_cluster_water_depth": float(np.max(depth[cluster])),
        "gap_discharges": [float(flow[0]) for flow in flows],
        "gap_mean_velocities": [float(flow[1]) for flow in flows],
        "gap_maximum_speeds": [float(flow[2]) for flow in flows],
        "contacted_solid_cells": (
            solver.last_solid_wetness.contacted_cells
            if solver.last_solid_wetness is not None else 0
        ),
        "maximum_solid_wetness": float(np.max(
            solver.state.get_interior("solid_wetness"),
        )),
        "maximum_foam": float(np.max(
            solver.state.get_interior("foam"),
        )),
    }


def _ripple_field(
    mesh,
    scene,
    eta0: float,
    spectrum,
    spacing: float,
    device: str,
    handoff_x: float,
    y0: float,
    y1: float,
) -> RippleField:
    maximum = mesh.vertices.max(axis=0)
    grid = GridConfig(
        nx=int(np.ceil((maximum[0] - handoff_x) / spacing)),
        ny=int(np.ceil((y1 - y0) / spacing)),
        Lx=float(maximum[0] - handoff_x),
        Ly=float(y1 - y0),
        x0=handoff_x,
        y0=y0,
    )
    x = grid.x0 + (np.arange(grid.nx) + 0.5) * grid.dx
    depth = np.maximum(eta0 - scene.beach.elevation(x), 0.0)
    return RippleField(grid, depth, spectrum, device)


def _project(camera: Camera, points: np.ndarray) -> np.ndarray:
    forward, right, up = camera.basis()
    relative = points - np.asarray(camera.pos)
    depth = relative @ forward
    u = (relative @ right) / (depth * camera.tan_half_h)
    v = (relative @ up) / (depth * camera.tan_half_v)
    return np.stack((
        0.5 * camera.width * (u + 1.0),
        0.5 * camera.height * (1.0 - v),
    ), axis=-1)


def _render_bounds(scene) -> tuple[float, float, float, float]:
    grid = scene.solver.grid
    handoff_x = scene.offshore_handoff_x
    ylo, yhi = periodic_render_y_bounds(grid.y0, grid.Ly)
    return (
        handoff_x - COLLAR_WIDTH - OFFSHORE_MESH_LENGTH,
        max(
            grid.x0 + grid.Lx,
            *(candidate.position[0] for candidate in CAMERA_CANDIDATES),
        ) + 1.0,
        ylo,
        yhi,
    )


def _write_wetness(
    path: Path,
    samples: tuple[SurfaceWetnessSample, ...],
) -> Path:
    arrays: dict[str, np.ndarray] = {
        "material_ids": np.asarray(
            [sample.material_id for sample in samples], dtype=np.int32,
        ),
        "primitive_ids": np.asarray(
            [sample.primitive_id for sample in samples], dtype=np.int32,
        ),
    }
    for index, sample in enumerate(samples):
        arrays[f"coordinates_{index:02d}"] = sample.coordinates
        arrays[f"wetness_{index:02d}"] = sample.wetness
    np.savez(path, **arrays)
    return path


def _read_wetness(path: Path) -> tuple[SurfaceWetnessSample, ...]:
    with np.load(path) as saved:
        material_ids = saved["material_ids"]
        primitive_ids = saved["primitive_ids"]
        return tuple(
            SurfaceWetnessSample(
                material_id=int(material_id),
                primitive_id=int(primitive_id),
                coordinates=saved[f"coordinates_{index:02d}"].copy(),
                wetness=saved[f"wetness_{index:02d}"].copy(),
            )
            for index, (material_id, primitive_id) in enumerate(zip(
                material_ids,
                primitive_ids,
                strict=True,
            ))
        )


def _arrays_unchanged(
    before: tuple[np.ndarray, ...],
    after: tuple[np.ndarray, ...],
) -> bool:
    return all(
        np.array_equal(first, second)
        for first, second in zip(before, after, strict=True)
    )


def _water_arrays(surface: WaterSurfaceMesh) -> tuple[np.ndarray, ...]:
    return (
        surface.vertices,
        surface.triangles,
        surface.normals,
        surface.velocity,
        surface.foam,
        surface.vertex_open_boundaries,
        surface.open_boundary_edges,
        surface.open_boundary_labels,
    )


def _wetness_arrays(
    wetness: tuple[SurfaceWetnessSample, ...],
) -> tuple[np.ndarray, ...]:
    return tuple(
        values
        for sample in wetness
        for values in (sample.coordinates, sample.wetness)
    )


def _visible_domain_side_samples(camera, scene) -> dict[str, int]:
    grid = scene.solver.grid
    x = np.linspace(
        scene.offshore_handoff_x,
        grid.x0 + grid.Lx,
        512,
    )
    counts = {}
    tiled_y_minimum, tiled_y_maximum = periodic_render_y_bounds(
        grid.y0, grid.Ly,
    )
    for name, y in (
        ("ylo", tiled_y_minimum),
        ("yhi", tiled_y_maximum),
    ):
        points = np.column_stack((
            x,
            np.full_like(x, y),
            scene.geometry.beach.elevation(x),
        ))
        forward = camera.basis()[0]
        depth = (points - np.asarray(camera.pos)) @ forward
        pixels = _project(camera, points)
        visible = (
            (depth > 0.0)
            & (pixels[:, 0] >= 0.0)
            & (pixels[:, 0] < camera.width)
            & (pixels[:, 1] >= 0.0)
            & (pixels[:, 1] < camera.height)
        )
        counts[name] = int(np.count_nonzero(visible))
    return counts


def _solid_arrays(renderer: MitsubaWaterRenderer) -> tuple[np.ndarray, ...]:
    return tuple(
        values
        for mesh in renderer.solid_meshes
        for values in (mesh.vertices, mesh.triangles)
    )


def _look_arrays(renderer: MitsubaWaterRenderer) -> tuple[np.ndarray, ...]:
    arrays = []
    for look in renderer.solid_looks:
        if look is None:
            continue
        arrays.extend((
            look.dry_albedo,
            look.wet_albedo,
            look.roughness_weight,
        ))
        if look.texcoords is not None:
            arrays.append(look.texcoords)
        if look.normal_map is not None:
            arrays.append(look.normal_map)
        if look.relief_height_m is not None:
            arrays.append(look.relief_height_m)
        if look.vertex_normals is not None:
            arrays.append(look.vertex_normals)
        if look.base_vertex_normals is not None:
            arrays.append(look.base_vertex_normals)
    return tuple(arrays)
