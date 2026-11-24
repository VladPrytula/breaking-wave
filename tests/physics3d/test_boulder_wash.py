import numpy as np
import pytest

from shoreloop.device import DEVICE
from shoreloop.physics3d import run_one_boulder_wash_gate
from shoreloop.physics3d.boulder_wash import _vorticity


def test_vorticity_recovers_solid_body_rotation():
    shape = (9, 8, 7)
    spacing = (0.2, 0.25, 0.3)
    x = (np.arange(shape[0]) + 0.5) * spacing[0]
    y = (np.arange(shape[1]) + 0.5) * spacing[1]
    xx, yy, zz = np.meshgrid(
        x, y, (np.arange(shape[2]) + 0.5) * spacing[2], indexing="ij",
    )
    velocity = (-1.7 * yy, 1.7 * xx, np.zeros_like(zz))

    omega = _vorticity(velocity, spacing)

    np.testing.assert_allclose(omega[:, 1:-1], 3.4, atol=1.0e-6)


@pytest.mark.integration
def test_breaking_wash_splits_and_rejoins_around_one_boulder():
    metrics = run_one_boulder_wash_gate(
        device=DEVICE,
        output_dir="/tmp/shoreloop-one-boulder-wash-test",
    )

    assert metrics.maximum_surface_head_difference > 0.03
    assert metrics.maximum_side_velocity > 0.5
    assert metrics.maximum_boulder_pressure > 100.0
    assert metrics.maximum_boulder_contact_fraction > 0.4
    assert metrics.maximum_boulder_flux_residual < 2.0e-3
    assert metrics.maximum_vorticity > 20.0
    assert metrics.maximum_reverse_flow_cells > 50
    assert metrics.maximum_reverse_speed > 0.1
    assert metrics.maximum_recirculation_length > 0.2
    assert metrics.maximum_split_depth_contrast > 0.03
    assert metrics.maximum_downstream_rejoin_depth > 0.015
    assert metrics.backwash_start_time is not None
    assert metrics.maximum_backwash_speed > 0.02
    assert metrics.closed_face_absolute_flux == 0.0
    assert metrics.alpha_min >= 0.0
    assert metrics.alpha_max <= 1.0
    assert metrics.solid_water_volume == 0.0
    assert abs(metrics.relative_volume_residual) < 1.0e-3
    assert metrics.max_divergence < 2.0e-3
    assert metrics.max_pressure_iterations < 100
