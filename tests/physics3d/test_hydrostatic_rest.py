"""Repeated hydrostatic free-surface rest regression."""

import numpy as np

from shoreloop.config import Grid3DConfig, Solver3DConfig
from shoreloop.device import DEVICE
from shoreloop.physics3d import (PCGConfig, ProjectionConfig, Solver3D,
                                 initialize_box, measure_vof)


def _rest_run(relative_tolerance: float) -> tuple[float, float, float]:
    cfg = Solver3DConfig(
        grid=Grid3DConfig(nx=8, ny=6, nz=10,
                          Lx=0.8, Ly=0.6, Lz=1.0),
        device=DEVICE,
    )
    solver = Solver3D(
        cfg,
        ProjectionConfig(
            pcg=PCGConfig(relative_tolerance=relative_tolerance,
                          max_iterations=300),
            rms_divergence_tolerance=1.5e-4,
        ),
    )
    initialize_box(
        solver.state, (0.0, 0.0, 0.0), (0.8, 0.6, 0.6),
    )
    alpha_before = solver.state.get_interior("alpha").copy()
    volume_before = measure_vof(solver.state).volume

    for _ in range(12):
        result = solver.step(0.005)
        assert result.converged

    alpha_after = solver.state.get_interior("alpha")
    volume_after = measure_vof(solver.state).volume
    max_velocity = max(
        float(np.max(np.abs(solver.state.get_interior(name))))
        for name in ("u", "v", "w")
    )
    surface_drift = float(np.max(np.abs(alpha_after - alpha_before)))
    return max_velocity, surface_drift, volume_after - volume_before


def test_flat_free_surface_remains_at_hydrostatic_rest():
    loose = _rest_run(2.0e-5)
    tight = _rest_run(1.0e-6)

    assert loose[0] < 5.0e-7
    assert tight[0] <= loose[0] + 2.0e-8
    # Coupled geometric transport moves an aligned surface by at most one
    # float32 alpha ULP under the projection's roundoff velocity.
    assert loose[1] < 1.0e-7
    assert tight[1] < 1.0e-7
    assert abs(loose[2]) < 2.0e-9
    assert abs(tight[2]) < 2.0e-9
