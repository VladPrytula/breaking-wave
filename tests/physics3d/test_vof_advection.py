"""Conservative split-PLIC VOF advection gates."""

import numpy as np
import pytest

from shoreloop.config import BC, BC3DConfig, Grid3DConfig, Solver3DConfig
from shoreloop.device import DEVICE
from shoreloop.physics3d import (State3D, VOFAdvector, initialize_sphere,
                                 fill_boundaries, measure_vof)


def _periodic_cfg(n: int) -> Solver3DConfig:
    periodic = BC3DConfig(
        xlo=BC.PERIODIC, xhi=BC.PERIODIC,
        ylo=BC.PERIODIC, yhi=BC.PERIODIC,
        zlo=BC.PERIODIC, zhi=BC.PERIODIC,
    )
    return Solver3DConfig(
        grid=Grid3DConfig(nx=n, ny=n, nz=n, Lx=1.0, Ly=1.0, Lz=1.0),
        bc=periodic,
        device=DEVICE,
    )


@pytest.mark.parametrize("n", [12, 24])
def test_periodic_sphere_translation_is_bounded_and_conservative(n):
    cfg = _periodic_cfg(n)
    state = State3D(cfg)
    initialize_sphere(state, (0.35, 0.5, 0.5), 0.19,
                      samples_per_axis=4)
    speed = 0.2
    state.set_interior(
        "u", np.full(state.grid.interior_shape("u"), speed, dtype=np.float32),
    )
    advector = VOFAdvector(cfg)
    initial = measure_vof(state)
    dt = 0.35 * cfg.grid.dx / speed

    for _ in range(3):
        result = advector.advect(state, dt)
        assert result.max_courant == pytest.approx(0.35, rel=2.0e-7)

    final = measure_vof(state)
    assert 0.0 <= final.alpha_min <= final.alpha_max <= 1.0
    assert abs(final.volume - initial.volume) < 2.0e-7
    assert advector.total_clipped_volume < 2.0e-7
    assert advector.steps == 3
    assert result.sweep_order == ("z", "x", "y")


def test_uniform_fraction_survives_discrete_divergence_free_velocity():
    n = 14
    cfg = _periodic_cfg(n)
    state = State3D(cfg)
    transport = State3D(cfg)
    state.set_interior(
        "alpha", np.full((n, n, n), 0.37, dtype=np.float32),
    )
    g = cfg.grid
    ux, uy, _ = state.grid.coordinates("u")
    vx, vy, _ = state.grid.coordinates("v")
    amplitude = 0.12
    u2 = (amplitude * np.sin(2.0 * np.pi * ux)[:, None]
          * np.cos(2.0 * np.pi * uy)[None, :])
    v2 = (-amplitude * np.cos(2.0 * np.pi * vx)[:, None]
          * np.sin(2.0 * np.pi * vy)[None, :])
    transport.set_interior("u", np.repeat(u2[:, :, None], n, axis=2))
    transport.set_interior("v", np.repeat(v2[:, :, None], n, axis=2))
    fill_boundaries(transport)

    result = VOFAdvector(cfg).advect(
        state,
        0.2 * g.dx / amplitude,
        transport_velocity=(transport.u, transport.v, transport.w),
    )
    alpha = state.get_interior("alpha")
    np.testing.assert_allclose(alpha, 0.37, rtol=0.0, atol=2.5e-6)
    assert abs(result.volume_drift) < 2.0e-7
    assert result.clipped_volume == pytest.approx(0.0, abs=1.0e-12)
