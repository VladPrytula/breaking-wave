import math

import numpy as np
import pytest

from shoreloop.physics3d.waves import (LinearWaveInlet, PeriodicWaveComponent,
                                       fit_probe_wave_modes,
                                       finite_depth_wavenumber,
                                       measure_two_probe_reflection)


def test_finite_depth_dispersion_spans_shallow_intermediate_and_deep_water():
    gravity = 9.81
    cases = (
        (0.05, 2.0 * math.pi / 8.0),
        (0.50, 2.0 * math.pi / 1.5),
        (5.00, 2.0 * math.pi / 0.8),
    )
    solved = []
    for depth, omega in cases:
        k = finite_depth_wavenumber(omega, depth, gravity)
        relative_residual = abs(gravity * k * math.tanh(k * depth) - omega**2) / omega**2
        assert relative_residual < 2.0e-15
        solved.append((depth, omega, k))

    shallow_depth, shallow_omega, shallow_k = solved[0]
    assert shallow_k * shallow_depth < 0.1
    shallow_speed = shallow_omega / shallow_k
    assert shallow_speed == pytest.approx(math.sqrt(gravity * shallow_depth), rel=2.0e-3)

    deep_depth, deep_omega, deep_k = solved[-1]
    assert deep_k * deep_depth > 3.0
    assert deep_k == pytest.approx(deep_omega**2 / gravity, rel=2.0e-6)


def test_periodic_component_has_consistent_modes_and_float64_phase_reduction():
    component = PeriodicWaveComponent(
        period=4.0,
        depth=0.8,
        alongshore_length=8.0,
        amplitude=0.03,
        temporal_mode=2,
        alongshore_mode=-1,
        phase_offset=0.4,
    )

    assert component.omega == pytest.approx(math.pi)
    assert component.ky == pytest.approx(-2.0 * math.pi / 8.0)
    assert component.kx**2 + component.ky**2 == pytest.approx(component.k**2)
    assert component.phase(0.7, 1.1, 0.37 + component.period) \
        == pytest.approx(component.phase(0.7, 1.1, 0.37), abs=2.0e-15)


def test_periodic_component_rejects_non_integer_and_evanescent_modes():
    common = dict(
        period=4.0,
        depth=0.5,
        alongshore_length=4.0,
        amplitude=0.02,
    )
    with pytest.raises(ValueError, match="positive integer"):
        PeriodicWaveComponent(**common, temporal_mode=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="integer"):
        PeriodicWaveComponent(**common, alongshore_mode=0.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="evanescent"):
        PeriodicWaveComponent(**common, alongshore_mode=2)


def test_shore_normal_profile_matches_linear_theory_and_integrated_discharge():
    component = PeriodicWaveComponent(
        period=2.0,
        depth=0.7,
        alongshore_length=2.0,
        amplitude=0.025,
    )
    x = y = time = 0.0

    assert component.elevation(x, y, time) == pytest.approx(component.amplitude)
    assert component.velocity(x, y, -component.depth, time)[2] == pytest.approx(0.0)
    assert component.velocity(x, y, -0.3, time)[1] == pytest.approx(0.0)

    n = 20_001
    dz = component.depth / (n - 1)
    numerical_qx = 0.0
    for index in range(n):
        z = -component.depth + index * dz
        weight = 0.5 if index in (0, n - 1) else 1.0
        numerical_qx += weight * component.velocity(x, y, z, time)[0]
    numerical_qx *= dz
    expected_qx, expected_qy = component.discharge(x, y, time)
    assert numerical_qx == pytest.approx(expected_qx, rel=2.0e-9)
    assert expected_qy == pytest.approx(0.0)

    assert component.velocity(x, y, -component.depth - 0.01, time) == (0.0, 0.0, 0.0)
    assert component.velocity(x, y, component.amplitude + 0.01, time) == (0.0, 0.0, 0.0)


def test_second_order_stokes_component_adds_bound_surface_and_velocity_harmonics():
    component = PeriodicWaveComponent(
        period=1.1,
        depth=0.4,
        alongshore_length=2.0,
        amplitude=0.025,
        stokes_order=2,
    )
    tangent = math.tanh(component.k * component.depth)
    expected_surface_coefficient = (
        0.25 * component.k * component.amplitude**2
        * (3.0 - tangent**2) / tangent**3
    )

    assert component.second_order_elevation_amplitude == pytest.approx(
        expected_surface_coefficient,
    )
    assert component.elevation(0.0, 0.0, 0.0) == pytest.approx(
        component.amplitude + expected_surface_coefficient,
    )

    z = -0.2
    linear = PeriodicWaveComponent(
        period=component.period,
        depth=component.depth,
        alongshore_length=component.alongshore_length,
        amplitude=component.amplitude,
    )
    expected_second_u = (
        2.0 * component.k * component.second_order_potential_amplitude
        * math.cosh(2.0 * component.k * (z + component.depth))
    )
    assert component.velocity(0.0, 0.0, z, 0.0)[0] \
        - linear.velocity(0.0, 0.0, z, 0.0)[0] \
        == pytest.approx(expected_second_u)


def test_oblique_components_sum_with_the_expected_direction_and_bed_condition():
    first = PeriodicWaveComponent(
        period=1.2,
        depth=0.8,
        alongshore_length=8.0,
        amplitude=0.015,
        alongshore_mode=1,
    )
    second = PeriodicWaveComponent(
        period=1.2,
        depth=0.8,
        alongshore_length=8.0,
        amplitude=0.006,
        temporal_mode=2,
        alongshore_mode=-1,
        phase_offset=0.3,
    )
    inlet = LinearWaveInlet((first, second))
    point = (0.2, 0.4, -0.35, 0.17)

    combined = inlet.velocity(*point)
    expected = tuple(
        a + b
        for a, b in zip(first._velocity(*point), second._velocity(*point), strict=True)
    )
    assert combined == pytest.approx(expected)

    horizontal = first._velocity(0.0, 0.0, -0.3, 0.0)
    assert horizontal[1] / horizontal[0] == pytest.approx(first.ky / first.kx)
    assert inlet.velocity(0.0, 0.0, -inlet.depth, 0.25)[2] == pytest.approx(0.0)


def test_sampled_maximum_local_wave_height_recovers_a_linear_crest_to_trough():
    component = PeriodicWaveComponent(
        period=1.2,
        depth=0.8,
        alongshore_length=8.0,
        amplitude=0.015,
        alongshore_mode=1,
    )
    inlet = LinearWaveInlet((component,))

    assert inlet.sampled_maximum_local_wave_height(
        0.0,
        alongshore_samples=64,
        time_samples=64,
    ) == pytest.approx(2.0 * component.amplitude, rel=2.0e-4)
    with pytest.raises(ValueError, match="at least two"):
        inlet.sampled_maximum_local_wave_height(0.0, time_samples=1)


def test_two_probes_separate_incident_and_reflected_wave_amplitudes():
    component = PeriodicWaveComponent(
        period=1.4,
        depth=0.6,
        alongshore_length=1.0,
        amplitude=0.03,
    )
    times = np.linspace(0.0, 4.0 * component.period, 801, endpoint=False)
    incident = 0.03 * np.exp(0.2j)
    reflected = 0.004 * np.exp(-0.6j)
    probes = (0.35, 0.77)

    def series(x: float) -> np.ndarray:
        spatial = (incident * np.exp(1j * component.kx * x)
                   + reflected * np.exp(-1j * component.kx * x))
        return np.real(spatial * np.exp(-1j * component.omega * times))

    measured = measure_two_probe_reflection(
        times, series(probes[0]), series(probes[1]), *probes, component,
    )
    assert measured.incident_amplitude == pytest.approx(abs(incident), rel=1.0e-12)
    assert measured.reflected_amplitude == pytest.approx(abs(reflected), rel=1.0e-12)
    assert measured.reflection_coefficient == pytest.approx(
        abs(reflected / incident), rel=1.0e-12,
    )


def test_multi_probe_fit_separates_dispersion_error_from_reflection():
    component = PeriodicWaveComponent(
        period=1.4,
        depth=0.6,
        alongshore_length=1.0,
        amplitude=0.03,
    )
    times = np.linspace(0.0, 4.0 * component.period, 801, endpoint=False)
    positions = np.linspace(0.2, 1.1, 7)
    actual_k = 1.12 * component.kx
    incident = 0.03 * np.exp(0.2j)
    reflected = 0.003 * np.exp(-0.6j)
    spatial = (incident * np.exp(1j * actual_k * positions)
               + reflected * np.exp(-1j * actual_k * positions))
    elevations = np.real(
        np.exp(-1j * component.omega * times[:, None]) * spatial[None, :],
    )

    fitted = fit_probe_wave_modes(times, elevations, positions, component)
    assert fitted.wavenumber == pytest.approx(actual_k, rel=2.0e-5)
    assert fitted.incident_amplitude == pytest.approx(abs(incident), rel=2.0e-5)
    assert fitted.reflection_coefficient == pytest.approx(0.1, rel=2.0e-4)
    assert fitted.relative_residual < 2.0e-5
