import pytest

from shoreloop.device import DEVICE
from shoreloop.physics3d import run_plane_beach_gate


pytestmark = pytest.mark.integration


def test_gentle_wave_runs_up_and_recedes_on_the_cut_cell_beach():
    metrics = run_plane_beach_gate(
        device=DEVICE, output_dir="/tmp/shoreloop-plane-beach-test",
    )

    assert metrics.shoreline_advance >= 2.0 * metrics.dx
    assert metrics.shoreline_recession >= 0.9 * metrics.dx
    assert metrics.maximum_runup > 0.0
    assert metrics.post_peak_minimum_time > metrics.maximum_shoreline_time
    assert metrics.final_dry_beach_fraction > 0.75
    assert metrics.final_thin_film_fraction < 0.5
    assert metrics.alpha_min >= 0.0
    assert metrics.alpha_max <= 1.0
    assert metrics.solid_water_volume == 0.0
    assert abs(metrics.relative_volume_residual) < 1.0e-3
    assert metrics.max_divergence < 1.0e-3
    assert metrics.max_pressure_iterations < 200
    assert metrics.minimum_dt > 1.0e-4
