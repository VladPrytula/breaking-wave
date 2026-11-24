import numpy as np

from shoreloop.config import BC, BC3DConfig, Grid3DConfig, Solver3DConfig
from shoreloop.device import DEVICE
from shoreloop.physics3d.free_surface import FreeSurfaceProjection
from shoreloop.physics3d.pcg import PCGConfig
from shoreloop.physics3d.projection import ProjectionConfig
from shoreloop.physics3d.solids import (
    enforce_voxel_solids,
    initialize_scene_voxels,
    measure_solid_flux,
)
from shoreloop.physics3d.state import State3D
from shoreloop.scene.geometry3d import AnalyticBeach, RoundedBoulder, SceneGeometry


def test_projected_uniform_flow_diverts_without_entering_voxel_boulder():
    periodic = BC3DConfig(
        xlo=BC.PERIODIC, xhi=BC.PERIODIC,
        ylo=BC.PERIODIC, yhi=BC.PERIODIC,
        zlo=BC.PERIODIC, zhi=BC.PERIODIC,
    )
    cfg = Solver3DConfig(
        grid=Grid3DConfig(
            nx=24, ny=20, nz=16, Lx=2.4, Ly=2.0, Lz=1.6,
            x0=-1.2, y0=-1.0, z0=-0.8,
        ),
        bc=periodic,
        device=DEVICE,
    )
    state = State3D(cfg)
    state.fill_("alpha", 1.0)
    scene = SceneGeometry(
        AnalyticBeach(toe_x=4.0, slope=0.0, base_elevation=-2.0),
        (RoundedBoulder((0.0, 0.0, 0.0), (0.42, 0.34, 0.30),
                        rotation_xyz=(0.08, -0.16, 0.24), exponent=2.4),),
    )
    initialize_scene_voxels(state, scene)
    state.fill_("u", 0.25)
    enforce_voxel_solids(state)

    projection = FreeSurfaceProjection(
        cfg,
        ProjectionConfig(
            pcg=PCGConfig(relative_tolerance=2.0e-6, max_iterations=500),
            rms_divergence_tolerance=2.0e-4,
        ),
    )
    result = projection.project(state, 0.01)
    leakage = measure_solid_flux(state)

    assert result.converged
    assert leakage.boundary_faces > 0
    assert leakage.absolute_flux == 0.0
    assert leakage.maximum_normal_velocity == 0.0
    transverse_speed = max(
        float(np.max(np.abs(state.get_interior("v")))),
        float(np.max(np.abs(state.get_interior("w")))),
    )
    assert transverse_speed > 1.0e-3
