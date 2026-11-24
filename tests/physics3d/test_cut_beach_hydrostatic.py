import numpy as np

from shoreloop.config import Grid3DConfig, Solver3DConfig
from shoreloop.device import DEVICE
from shoreloop.physics3d.cut_cells import initialize_scene_cut_cells
from shoreloop.physics3d.pcg import PCGConfig
from shoreloop.physics3d.porous import (PorousBedModel,
                                        PorousStorageConfig)
from shoreloop.physics3d.projection import ProjectionConfig
from shoreloop.physics3d.relaxation import WaveRelaxationZone
from shoreloop.physics3d.solver import Solver3D
from shoreloop.physics3d.timestep import TimeStepConfig
from shoreloop.physics3d.vof import initialize_box, measure_vof
from shoreloop.physics3d.waves import LinearWaveInlet, PeriodicWaveComponent
from shoreloop.scene.geometry3d import AnalyticBeach, SceneGeometry


def test_sloped_cut_beach_remains_at_hydrostatic_rest():
    cfg = Solver3DConfig(
        grid=Grid3DConfig(nx=32, ny=8, nz=16,
                          Lx=3.2, Ly=0.8, Lz=0.8),
        device=DEVICE,
    )
    solver = Solver3D(
        cfg,
        ProjectionConfig(
            pcg=PCGConfig(
                relative_tolerance=2.0e-6,
                max_iterations=300,
                preconditioner="multigrid",
            ),
            rms_divergence_tolerance=5.0e-4,
        ),
        TimeStepConfig(dt_max=0.005),
    )
    initialize_box(solver.state, (0.0, 0.0, 0.0), (3.2, 0.8, 0.45))
    geometry = initialize_scene_cut_cells(
        solver.state, SceneGeometry(AnalyticBeach(0.8, 0.18, 0.0)),
    )
    initial_volume = measure_vof(solver.state).volume
    maximum_velocity = 0.0
    maximum_divergence = 0.0
    for _ in range(40):
        result = solver.step(0.0025)
        maximum_velocity = max(
            maximum_velocity,
            *(float(np.max(np.abs(solver.state.get_interior(name))))
              for name in ("u", "v", "w")),
        )
        maximum_divergence = max(
            maximum_divergence, result.max_divergence_after,
        )

    assert np.count_nonzero(geometry.cut_mask) > 0
    assert maximum_velocity < 2.0e-7
    assert maximum_divergence < 1.0e-5
    assert abs(measure_vof(solver.state).volume - initial_volume) < 2.0e-9


def test_submerged_porous_beach_has_no_drainage_or_relaxation_current():
    grid = Grid3DConfig(
        nx=32, ny=8, nz=16, Lx=3.2, Ly=0.8, Lz=0.8,
    )
    cfg = Solver3DConfig(grid=grid, device=DEVICE)
    depth = 0.45
    component = PeriodicWaveComponent(
        period=1.0,
        depth=depth,
        alongshore_length=grid.Ly,
        amplitude=0.0,
    )

    def make_solver(*, with_porosity: bool) -> tuple[Solver3D, float]:
        solver = Solver3D(
            cfg,
            ProjectionConfig(
                pcg=PCGConfig(
                    relative_tolerance=2.0e-6,
                    max_iterations=300,
                    preconditioner="multigrid",
                ),
                rms_divergence_tolerance=5.0e-4,
            ),
            TimeStepConfig(dt_max=0.005),
        )
        initialize_box(
            solver.state, (0.0, 0.0, 0.0),
            (grid.Lx, grid.Ly, depth),
        )
        geometry = initialize_scene_cut_cells(
            solver.state, SceneGeometry(AnalyticBeach(0.8, 0.18, 0.0)),
        )
        initial_storage = 0.0
        if with_porosity:
            porous = PorousBedModel.from_geometry(
                cfg,
                geometry,
                storage_settings=PorousStorageConfig(
                    infiltration_rate=0.8, drainage_time=0.05,
                ),
            )
            solver.set_porous_bed(porous)
            initial_storage = porous.initialize_storage_equilibrium(solver.state)
        solver.set_wave_relaxation(WaveRelaxationZone(
            cfg,
            LinearWaveInlet((component,)),
            width=0.4,
            timescale=0.05,
        ))
        return solver, initial_storage

    baseline, _ = make_solver(with_porosity=False)
    solver, initial_storage = make_solver(with_porosity=True)
    drained_volume = 0.0
    source_difference = 0.0
    maximum_velocity = 0.0
    for _ in range(40):
        baseline.step(0.0025)
        solver.step(0.0025)
        drained_volume += solver.last_porous_storage.drained_volume
        source_difference += (
            solver.last_relaxation.volume_source
            - baseline.last_relaxation.volume_source
        )
        maximum_velocity = max(
            maximum_velocity,
            *(float(np.max(np.abs(solver.state.get_interior(name))))
              for name in ("u", "v", "w")),
        )

    final_storage = float(np.sum(
        solver.state.get_interior("porous_storage"), dtype=np.float64,
    )) * grid.dx * grid.dy * grid.dz
    assert drained_volume < 1.0e-9 * initial_storage
    assert abs(source_difference) < 2.0e-9
    assert maximum_velocity < 2.0e-7
    assert abs(final_storage - initial_storage) < 2.0e-9
    assert abs(
        measure_vof(solver.state).volume
        - measure_vof(baseline.state).volume
    ) < 2.0e-9
