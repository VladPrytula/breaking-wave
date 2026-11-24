import numpy as np

from shoreloop.config import BC, BC3DConfig, Grid3DConfig, Solver3DConfig
from shoreloop.device import DEVICE
from shoreloop.physics3d.advection import VOFAdvector
from shoreloop.physics3d.cut_cells import initialize_scene_cut_cells
from shoreloop.physics3d.free_surface import FreeSurfaceProjection
from shoreloop.physics3d.pcg import PCGConfig
from shoreloop.physics3d.projection import ProjectionConfig
from shoreloop.physics3d.solids import enforce_voxel_solids
from shoreloop.physics3d.state import State3D
from shoreloop.physics3d.timestep import TimeStepController
from shoreloop.scene.geometry3d import AnalyticBeach, RoundedBoulder, SceneGeometry


def _sliver_advection(threshold: float):
    periodic = BC3DConfig(
        xlo=BC.PERIODIC, xhi=BC.PERIODIC,
        ylo=BC.PERIODIC, yhi=BC.PERIODIC,
        zlo=BC.PERIODIC, zhi=BC.PERIODIC,
    )
    cfg = Solver3DConfig(
        grid=Grid3DConfig(nx=8, ny=4, nz=4,
                          Lx=0.8, Ly=0.4, Lz=0.4),
        bc=periodic,
        device=DEVICE,
    )
    state = State3D(cfg)
    theta = np.ones(cfg.grid.shape, dtype=np.float32)
    theta[3, 2, 2] = 1.0 / 64.0
    state.set_geometry_interior("theta", theta)
    normals = np.zeros((*cfg.grid.shape, 3), dtype=np.float32)
    offsets = np.zeros(cfg.grid.shape, dtype=np.float32)
    normals[3, 2, 2, 2] = -1
    offsets[3, 2, 2] = -63 / 64
    state.set_open_planes(normals, offsets)
    state.small_cell_threshold = threshold
    alpha = np.zeros(cfg.grid.shape, dtype=np.float32)
    alpha[:3] = 1.0
    state.set_interior("alpha", alpha)
    state.fill_("u", 1.0)
    result = VOFAdvector(cfg).advect(state, 0.01)
    return result, state.get_interior("alpha")[3, 2, 2]


def test_small_cell_flux_redistribution_is_conservative():
    unstabilized, unstabilized_alpha = _sliver_advection(0.0)
    stabilized, stabilized_alpha = _sliver_advection(0.125)
    assert unstabilized_alpha == 1.0
    assert unstabilized.clipped_volume > 8.0e-5
    assert stabilized_alpha == np.float32(0.8)
    assert stabilized.clipped_volume == 0.0
    assert abs(stabilized.volume_drift) < 5.0e-10


def test_subcell_boulder_motion_does_not_spike_dt_or_pressure_iterations():
    cfg = Solver3DConfig(
        grid=Grid3DConfig(nx=16, ny=12, nz=12,
                          Lx=1.6, Ly=1.2, Lz=1.2),
        device=DEVICE,
    )
    time_steps = []
    pressure_iterations = []
    stabilized_counts = []
    minimum_fractions = []
    for shift in (0.0, 0.02, 0.04, 0.06):
        state = State3D(cfg)
        state.fill_("alpha", 1.0)
        geometry = initialize_scene_cut_cells(
            state,
            SceneGeometry(
                AnalyticBeach(3.0, 0.0, -1.0),
                (RoundedBoulder(
                    (0.8 + shift, 0.6, 0.6), (0.29, 0.24, 0.26),
                    exponent=2.4,
                ),),
            ),
        )
        rng = np.random.default_rng(506)
        for name in ("u", "v", "w"):
            state.set_interior(
                name,
                rng.normal(0.0, 0.02, state.grid.interior_shape(name)),
            )
        enforce_voxel_solids(state)
        time_steps.append(TimeStepController(cfg).compute(state).dt)
        projection = FreeSurfaceProjection(
            cfg,
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
        pressure_iterations.append(result.pressure_solve.iterations)
        stabilized_counts.append(geometry.stabilized_cells)
        minimum_fractions.append(geometry.minimum_open_fraction)

    assert min(stabilized_counts) > 0
    assert min(minimum_fractions) == 1.0 / 16.0
    assert max(time_steps) / min(time_steps) < 1.01
    assert max(pressure_iterations) - min(pressure_iterations) <= 2
