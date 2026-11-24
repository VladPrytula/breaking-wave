import pytest

from shoreloop.device import DEVICE
from shoreloop.physics3d import run_boulder_cluster_gate


pytestmark = pytest.mark.integration


def test_breaking_wash_and_backwash_form_gap_jets_and_interacting_wakes():
    metrics = run_boulder_cluster_gate(
        device=DEVICE,
        output_dir="/tmp/shoreloop-boulder-cluster-test",
    )

    assert min(metrics.gap_open_areas) > 0.02
    assert min(metrics.maximum_forward_gap_discharges) > 0.005
    assert min(metrics.maximum_reverse_gap_discharges) > 2.0e-4
    assert min(metrics.maximum_gap_speeds) > 1.0
    assert metrics.maximum_gap_jet_amplification > 2.0
    assert metrics.reversing_gaps == 2
    assert metrics.first_gap_reversal_time is not None
    assert metrics.maximum_reverse_flow_cells > 500
    assert metrics.maximum_reverse_speed > 0.5
    assert metrics.maximum_recirculation_length > 0.5
    assert metrics.maximum_vorticity > 50.0
    assert metrics.maximum_interacting_wake_cells > 100
    assert metrics.maximum_interacting_wake_span > 0.5
    assert metrics.maximum_sheltered_water_volume > 1.0e-3
    assert metrics.minimum_shelter_velocity_ratio < 0.8
    assert metrics.maximum_boulder_pressure > 100.0
    assert metrics.maximum_boulder_contact_fraction > 0.3
    assert metrics.maximum_boulder_flux_residual < 1.0e-3
    assert metrics.closed_face_absolute_flux == 0.0
    assert metrics.alpha_min >= 0.0
    assert metrics.alpha_max <= 1.0
    assert metrics.solid_water_volume == 0.0
    assert abs(metrics.relative_volume_residual) < 1.0e-3
    assert metrics.max_divergence < 2.0e-3
    assert metrics.max_pressure_iterations < 100
    assert metrics.minimum_dt > 1.0e-4
