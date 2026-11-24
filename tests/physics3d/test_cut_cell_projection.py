import numpy as np
import warp as wp

from shoreloop.config import Grid3DConfig, Solver3DConfig
from shoreloop.device import DEVICE
from shoreloop.physics3d.cut_cells import initialize_scene_cut_cells
from shoreloop.physics3d.free_surface import (
    FreeSurfacePressurePCG,
    FreeSurfaceProjection,
    build_free_surface_geometry,
)
from shoreloop.physics3d.pcg import PCGConfig
from shoreloop.physics3d.projection import ProjectionConfig
from shoreloop.physics3d.solids import enforce_voxel_solids
from shoreloop.physics3d.state import State3D
from shoreloop.scene.geometry3d import AnalyticBeach, RoundedBoulder, SceneGeometry


def _state(scene: SceneGeometry) -> State3D:
    cfg = Solver3DConfig(
        grid=Grid3DConfig(
            nx=16, ny=12, nz=12, Lx=1.6, Ly=1.2, Lz=1.2,
            x0=0.0, y0=0.0, z0=0.0,
        ),
        device=DEVICE,
    )
    state = State3D(cfg)
    state.fill_("alpha", 1.0)
    initialize_scene_cut_cells(state, scene)
    return state


def _padded(state: State3D, values: np.ndarray) -> wp.array:
    padded = np.zeros(state.grid.cell_shape, dtype=np.float32)
    padded[state.grid.interior_slices("cell")] = values
    return wp.array(padded, dtype=wp.float32, device=state.device)


def _operator_symmetry(state: State3D) -> float:
    geometry = build_free_surface_geometry(state)
    solver = FreeSurfacePressurePCG(
        state.cfg, PCGConfig(relative_tolerance=1.0e-6),
    )
    solver.configure(geometry)
    active = geometry.active.numpy()[state.grid.interior_slices("cell")]
    rng = np.random.default_rng(505)
    p_host = rng.normal(size=state.cfg.grid.shape).astype(np.float32) * active
    q_host = rng.normal(size=state.cfg.grid.shape).astype(np.float32) * active
    p = _padded(state, p_host)
    q = _padded(state, q_host)
    ap = wp.zeros(state.grid.cell_shape, dtype=wp.float32, device=state.device)
    aq = wp.zeros(state.grid.cell_shape, dtype=wp.float32, device=state.device)
    solver.apply_operator(p, ap)
    solver.apply_operator(q, aq)
    slices = state.grid.interior_slices("cell")
    left = float(np.sum(
        p_host.astype(np.float64) * aq.numpy()[slices].astype(np.float64),
    ))
    right = float(np.sum(
        q_host.astype(np.float64) * ap.numpy()[slices].astype(np.float64),
    ))
    return abs(left - right) / max(abs(left), abs(right))


def _projection_gate(state: State3D) -> tuple[float, float]:
    rng = np.random.default_rng(506)
    for name in ("u", "v", "w"):
        state.set_interior(
            name,
            rng.normal(0.0, 0.04, state.grid.interior_shape(name)),
        )
    enforce_voxel_solids(state)
    projection = FreeSurfaceProjection(
        state.cfg,
        ProjectionConfig(
            pcg=PCGConfig(
                relative_tolerance=1.0e-6,
                max_iterations=300,
                preconditioner="multigrid",
            ),
            rms_divergence_tolerance=8.0e-4,
        ),
    )
    result = projection.project(state, 0.01)
    assert result.converged
    for name in ("u", "v", "w"):
        aperture = state.get_geometry_interior(f"{name}_aperture")
        velocity = state.get_interior(name)
        assert np.all(velocity[aperture == 0.0] == 0.0)
    return result.rms_divergence_after, result.max_divergence_after


def test_cut_cell_operators_project_on_sloped_plane_and_sphere():
    scenes = (
        SceneGeometry(AnalyticBeach(toe_x=0.2, slope=0.31,
                                    base_elevation=0.12)),
        SceneGeometry(
            AnalyticBeach(toe_x=3.0, slope=0.0, base_elevation=-1.0),
            (RoundedBoulder((0.82, 0.60, 0.56), (0.28, 0.23, 0.25),
                            rotation_xyz=(0.10, -0.16, 0.23),
                            exponent=2.4),),
        ),
    )
    for scene in scenes:
        state = _state(scene)
        assert _operator_symmetry(state) < 2.0e-6
        rms, maximum = _projection_gate(state)
        assert rms < 8.0e-4
        assert maximum < 2.0e-2
