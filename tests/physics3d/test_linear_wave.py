import pytest

from shoreloop.device import DEVICE
from shoreloop.physics3d import run_linear_wave_gate


pytestmark = pytest.mark.integration


def test_small_linear_wave_propagates_with_bounded_vof_and_low_reflection():
    metrics = run_linear_wave_gate(refinement=1, device=DEVICE,
                                   output_dir="/tmp/shoreloop-linear-wave-test")

    assert metrics.cells_per_wavelength >= 20.0
    assert abs(metrics.amplitude_error) < 0.25
    assert abs(metrics.phase_speed_error) < 0.1
    assert metrics.reflection_coefficient < 0.05
    assert metrics.wave_fit_residual < 0.05
    assert metrics.alpha_min >= 0.0
    assert metrics.alpha_max <= 1.0
    assert abs(metrics.volume_conservation_residual) < 2.0e-6
    assert metrics.max_divergence < 1.0e-4
    assert metrics.max_downward_interface_cells == 0
    assert metrics.max_multi_transition_columns == 0
    assert metrics.max_vertical_transitions <= 1
    assert metrics.max_water_components == 1
    assert metrics.topology_events == 0
