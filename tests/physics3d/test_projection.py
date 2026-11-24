"""Projection and idempotence regression for the regular three-dimensional grid."""

import numpy as np

from shoreloop.config import Grid3DConfig, Solver3DConfig
from shoreloop.device import DEVICE
from shoreloop.physics3d import (PCGConfig, PressureProjection,
                                 ProjectionConfig, State3D)


def _seed_smooth_velocity(state: State3D) -> None:
    rng = np.random.default_rng(105)
    coefficients = 0.35 * rng.standard_normal(6)
    g = state.cfg.grid

    x, y, z = state.grid.coordinates("u")
    X = (x - g.x0) / g.Lx
    Y = (y - g.y0) / g.Ly
    Z = (z - g.z0) / g.Lz
    u = (
        coefficients[0] * np.sin(np.pi * X)[:, None, None]
        * np.cos(2.0 * np.pi * Y)[None, :, None]
        * np.cos(np.pi * Z)[None, None, :]
        + coefficients[1] * np.sin(2.0 * np.pi * X)[:, None, None]
        * np.sin(2.0 * np.pi * Y)[None, :, None]
        * np.cos(2.0 * np.pi * Z)[None, None, :]
    )

    x, y, z = state.grid.coordinates("v")
    X = (x - g.x0) / g.Lx
    Y = (y - g.y0) / g.Ly
    Z = (z - g.z0) / g.Lz
    v = (
        coefficients[2] * np.cos(np.pi * X)[:, None, None]
        * np.sin(2.0 * np.pi * Y)[None, :, None]
        * np.cos(np.pi * Z)[None, None, :]
        + coefficients[3] * np.sin(np.pi * X)[:, None, None]
        * np.cos(4.0 * np.pi * Y)[None, :, None]
        * np.sin(np.pi * Z)[None, None, :]
    )

    x, y, z = state.grid.coordinates("w")
    X = (x - g.x0) / g.Lx
    Y = (y - g.y0) / g.Ly
    Z = (z - g.z0) / g.Lz
    w = (
        coefficients[4] * np.cos(np.pi * X)[:, None, None]
        * np.cos(2.0 * np.pi * Y)[None, :, None]
        * np.sin(np.pi * Z)[None, None, :]
        + coefficients[5] * np.sin(2.0 * np.pi * X)[:, None, None]
        * np.sin(2.0 * np.pi * Y)[None, :, None]
        * np.sin(2.0 * np.pi * Z)[None, None, :]
    )

    state.set_interior("u", u)
    state.set_interior("v", v)
    state.set_interior("w", w)


def test_projection_reaches_divergence_gate_and_is_idempotent():
    cfg = Solver3DConfig(
        grid=Grid3DConfig(
            nx=12, ny=10, nz=8,
            Lx=1.2, Ly=1.0, Lz=0.8,
        ),
        device=DEVICE,
    )
    state = State3D(cfg)
    _seed_smooth_velocity(state)
    settings = ProjectionConfig(
        pcg=PCGConfig(relative_tolerance=2.0e-6, max_iterations=200),
        rms_divergence_tolerance=5.0e-5,
    )
    projection = PressureProjection(cfg, settings)
    assert projection.device.startswith("cuda")

    first = projection.project(state, dt=0.025)
    assert first.pressure_solve.converged
    assert first.converged
    assert first.rms_divergence_before > 1.0e-2
    assert first.rms_divergence_after <= settings.rms_divergence_tolerance
    assert first.max_divergence_after < first.max_divergence_before

    u = state.get_interior("u")
    v = state.get_interior("v")
    w = state.get_interior("w")
    np.testing.assert_array_equal(u[[0, -1]], np.float32(0.0))
    np.testing.assert_array_equal(w[:, :, [0, -1]], np.float32(0.0))
    np.testing.assert_array_equal(v[:, 0], v[:, -1])
    projected = (u.copy(), v.copy(), w.copy())

    second = projection.project(state, dt=0.025)
    assert second.pressure_solve.converged
    assert second.converged
    np.testing.assert_allclose(
        second.rms_divergence_before,
        first.rms_divergence_after,
        rtol=2.0e-6,
        atol=1.0e-8,
    )

    projected_twice = (
        state.get_interior("u"),
        state.get_interior("v"),
        state.get_interior("w"),
    )
    squared_change = sum(
        np.sum((after.astype(np.float64) - before) ** 2)
        for before, after in zip(projected, projected_twice)
    )
    face_count = sum(field.size for field in projected)
    rms_change = np.sqrt(squared_change / face_count)
    assert rms_change < 5.0e-6
