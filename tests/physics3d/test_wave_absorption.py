import numpy as np

from shoreloop.config import Grid3DConfig, Solver3DConfig
from shoreloop.device import DEVICE
from shoreloop.physics3d import (LinearWaveInlet, PeriodicWaveComponent,
                                 State3D, WaveRelaxationZone, initialize_box)


def test_offshore_relaxation_damps_an_outgoing_surface_and_velocity_pulse():
    grid = Grid3DConfig(nx=40, ny=4, nz=20, Lx=1.0, Ly=0.2, Lz=0.5)
    cfg = Solver3DConfig(grid=grid, device=DEVICE)
    state = State3D(cfg)
    depth = 0.3
    initialize_box(state, (0.0, 0.0, 0.0), (grid.Lx, grid.Ly, depth))

    x = grid.x0 + (np.arange(grid.nx) + 0.5) * grid.dx
    pulse = 0.015 * np.exp(-0.5 * ((x - 0.055) / 0.025) ** 2)
    alpha = state.get_interior("alpha")
    surface_cell = int(round(depth / grid.dz))
    alpha[:, :, surface_cell] = pulse[:, None] / grid.dz
    state.set_interior("alpha", alpha)

    ux = grid.x0 + np.arange(grid.nx + 1) * grid.dx
    outgoing = -0.08 * np.exp(-0.5 * ((ux - 0.055) / 0.025) ** 2)
    u = state.get_interior("u")
    u[:, :, :surface_cell] = outgoing[:, None, None]
    state.set_interior("u", u)

    flat = PeriodicWaveComponent(
        period=1.0,
        depth=depth,
        alongshore_length=grid.Ly,
        amplitude=0.0,
    )
    zone = WaveRelaxationZone(cfg, LinearWaveInlet((flat,)), width=0.3)
    before_surface = state.get_interior("alpha").sum(axis=2).mean(axis=1) * grid.dz - depth
    before_speed = np.max(np.abs(state.get_interior("u")[:8]))
    result = zone.apply(state, 0.0, 0.2)
    after_surface = state.get_interior("alpha").sum(axis=2).mean(axis=1) * grid.dz - depth
    after_speed = np.max(np.abs(state.get_interior("u")[:8]))

    assert np.linalg.norm(after_surface[:8]) < 0.2 * np.linalg.norm(before_surface[:8])
    assert after_speed < 0.2 * before_speed
    assert result.volume_source < 0.0
    assert result.momentum_source[0] > 0.0
