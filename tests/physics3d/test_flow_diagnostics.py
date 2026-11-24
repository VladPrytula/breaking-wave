"""Analytic formula checks for resolved turbulence diagnostics."""

import numpy as np

from shoreloop.physics3d.flow_diagnostics import turbulence_fields


def _grid(n: int = 13):
    axis = np.linspace(-1.0, 1.0, n)
    return (*np.meshgrid(axis, axis, axis, indexing="ij"), axis[1] - axis[0])


def test_q_and_vorticity_distinguish_rotation_from_strain():
    x, y, z, spacing = _grid()
    omega = 1.7
    rotation = turbulence_fields(
        (-omega * y, omega * x, np.zeros_like(z)),
        (spacing, spacing, spacing),
    )
    interior = np.s_[2:-2, 2:-2, 2:-2]
    np.testing.assert_allclose(
        rotation.vorticity_magnitude[interior], 2.0 * omega,
        rtol=2.0e-14, atol=2.0e-14,
    )
    np.testing.assert_allclose(
        rotation.q_criterion[interior], omega**2,
        rtol=2.0e-14, atol=2.0e-14,
    )

    rate = 0.8
    strain = turbulence_fields(
        (rate * x, -rate * y, np.zeros_like(z)),
        (spacing, spacing, spacing),
    )
    np.testing.assert_allclose(strain.vorticity_magnitude[interior], 0.0,
                               atol=2.0e-14)
    np.testing.assert_allclose(strain.q_criterion[interior], -rate**2,
                               rtol=2.0e-14, atol=2.0e-14)
