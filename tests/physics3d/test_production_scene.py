from dataclasses import FrozenInstanceError
import math

import numpy as np
import pytest

from shoreloop.physics3d.production_scene import (
    PRODUCTION_INCIDENT_SCALE_FROM_AW1103,
    PRODUCTION_MAXIMUM_INCIDENT_BOUNDARY_HEIGHT_TO_DEPTH_RATIO,
    PRODUCTION_LIQUID_RESOLVED_FRACTION,
    AW1101_OBLIQUE_AMPLITUDES,
    PRODUCTION_OBLIQUE_MAXIMUM_ANGLE,
    PRODUCTION_OBLIQUE_SPREAD_EXPONENT,
    PRODUCTION_OFFSHORE_HANDOFF_X,
    ProductionCamera,
    production_scene,
)
from shoreloop.physics3d.relaxation import relaxation_weight


def test_reference_incident_spectrum_fits_its_offshore_stretch(tmp_path):
    # AW-1119: the reference-calibrated forcing rides a 4.0 s loop, stays under the
    # depth-limited guard, and its relaxation zone ends offshore of the scene.
    from shoreloop.physics3d.production_scene import (
        REFERENCE_AMPLITUDES, REFERENCE_LOOP_PERIOD, REFERENCE_NORMAL_SHARE, REFERENCE_X0,
    )
    scene = production_scene(output_dir=tmp_path, incident="reference")
    production = production_scene(output_dir=tmp_path)
    grid = scene.solver.grid
    assert grid.x0 == REFERENCE_X0 and grid.x0 + grid.Lx == pytest.approx(4.8)
    assert (grid.dx, grid.dy, grid.dz) == pytest.approx((production.solver.grid.dx,
                                                         production.solver.grid.dy,
                                                         production.solver.grid.dz))
    assert all(c.period == REFERENCE_LOOP_PERIOD for c in scene.inlet.components)
    assert {c.temporal_mode for c in scene.inlet.components} == set(REFERENCE_AMPLITUDES)
    assert grid.x0 + scene.relaxation_width < 0.0 < scene.geometry.beach.toe_x
    crest_trough = 2.0 * sum(c.amplitude for c in scene.inlet.components if c.alongshore_mode == 0)
    assert crest_trough < 0.78 * scene.water_depth
    normal_energy = sum(c.amplitude ** 2 for c in scene.inlet.components if c.alongshore_mode == 0)
    total_energy = sum(c.amplitude ** 2 for c in scene.inlet.components)
    assert normal_energy / total_energy == pytest.approx(REFERENCE_NORMAL_SHARE, abs=0.01)
    assert scene.incident == "reference" and production.incident == "aw1103"
    # Everything inshore is the production scene: beach, boulders, cameras.
    assert scene.geometry == production.geometry
    assert scene.sunset_camera == production.sunset_camera
    with pytest.raises(ValueError):
        production_scene(output_dir=tmp_path, incident="spectral")


def test_production_scene_configuration_is_fixed_and_self_consistent(tmp_path):
    scene = production_scene(output_dir=tmp_path)
    unit_scale = production_scene(output_dir=tmp_path, incident_scale=1.0)
    grid = scene.solver.grid

    assert grid.shape == (192, 112, 48)
    assert (grid.dx, grid.dy, grid.dz) == pytest.approx((
        0.025,
        6.0 / 112.0,
        0.020,
    ))
    assert scene.input_mode == "standalone"
    assert scene.solver.physics.surface_tension == 0.0
    assert scene.solver.pressure_discretization == "liquid_mimetic"
    assert scene.projection.pcg.preconditioner == "jacobi"
    assert scene.incident_scale_from_aw1103 == pytest.approx(
        PRODUCTION_INCIDENT_SCALE_FROM_AW1103,
    )
    assert scene.porous_drag.permeability == pytest.approx(2.0e-5)
    assert scene.porous_drag.forchheimer_coefficient == pytest.approx(0.08)
    assert scene.relaxation_width < scene.geometry.beach.toe_x
    assert scene.offshore_handoff_x == PRODUCTION_OFFSHORE_HANDOFF_X
    assert grid.x0 < scene.offshore_handoff_x < scene.relaxation_width
    assert scene.offshore_handoff_x < scene.topology_x_bounds[0]
    assert relaxation_weight(
        scene.offshore_handoff_x,
        grid.x0,
        scene.relaxation_width,
    ) > 0.45
    assert scene.preview_frame_times[0] == 0.0
    assert scene.preview_frame_times[-1] == scene.preview_duration

    # Liquid per cell volume; measured 2026-10-09 (docs/decisions.md).
    assert scene.solver.liquid_resolved_fraction == PRODUCTION_LIQUID_RESOLVED_FRACTION == 2.0e-2

    components = scene.inlet.components
    normal = tuple(c for c in components if c.alongshore_mode == 0)
    oblique = tuple(c for c in components if c.alongshore_mode != 0)
    assert components[:3] == normal
    assert tuple(c.temporal_mode for c in normal) == (2, 3, 4)
    aw1103_normal_amplitudes = (
        0.015126144304132297,
        0.020420294810578603,
        0.010588301012892608,
    )
    assert tuple(
        c.amplitude for c in unit_scale.inlet.components if c.alongshore_mode == 0
    ) == pytest.approx(aw1103_normal_amplitudes)
    assert tuple(c.amplitude for c in normal) == pytest.approx(tuple(
        PRODUCTION_INCIDENT_SCALE_FROM_AW1103 * amplitude
        for amplitude in aw1103_normal_amplitudes
    ))
    assert tuple(c.phase_offset for c in normal) == pytest.approx((
        1.1176160321499085,
        0.0,
        2.591987353072888,
    ))
    energy = sum(component.amplitude**2 for component in components)
    assert energy == pytest.approx(
        0.00111456 * PRODUCTION_INCIDENT_SCALE_FROM_AW1103**2,
    )
    # Each period's oblique energy is the AW-1101 pair energy, spread over every
    # propagating alongshore mode inside the maximum angle with a cos^(2s)
    # weight; the loop's integer modes are the only directions available.
    for temporal_mode, pair in AW1101_OBLIQUE_AMPLITUDES.items():
        spread = [c for c in oblique if c.temporal_mode == temporal_mode]
        assert sum(c.amplitude**2 for c in spread) == pytest.approx(
            PRODUCTION_INCIDENT_SCALE_FROM_AW1103**2 * sum(a * a for a in pair),
        )
        assert len({abs(c.alongshore_mode) for c in spread}) >= 4
        for c in spread:
            angle = math.atan2(abs(c.ky), c.kx)
            assert angle <= PRODUCTION_OBLIQUE_MAXIMUM_ANGLE
            same_period = [d for d in spread if d.alongshore_mode == c.alongshore_mode]
            assert len(same_period) == 1
        weights = {
            c.alongshore_mode: c.amplitude**2 / math.cos(math.atan2(abs(c.ky), c.kx))
            ** (2 * PRODUCTION_OBLIQUE_SPREAD_EXPONENT)
            for c in spread
        }
        positive = [weights[m] for m in sorted(weights) if m > 0]
        negative = [weights[m] for m in sorted(weights) if m < 0]
        assert positive == pytest.approx([positive[0]] * len(positive))
        assert negative == pytest.approx([negative[0]] * len(negative))
    assert tuple(c.alongshore_mode for c in oblique) == tuple(
        c.alongshore_mode for c in production_scene(device="cuda:0").inlet.components
        if c.alongshore_mode != 0
    )
    assert tuple(c.phase_offset for c in oblique) == tuple(
        c.phase_offset for c in production_scene(device="cuda:0").inlet.components
        if c.alongshore_mode != 0
    )
    direction_shares = tuple(
        sum(
            component.amplitude**2 for component in components
            if predicate(component.alongshore_mode)
        ) / energy
        for predicate in (
            lambda mode: mode == 0,
            lambda mode: mode > 0,
            lambda mode: mode < 0,
        )
    )
    assert direction_shares == pytest.approx((0.68, 0.20, 0.12))
    assert max(direction_shares) <= 0.70
    direction_bins = {
        round(math.degrees(math.atan2(c.ky, c.kx)) / 10.0) for c in components
    }
    assert len(direction_bins) >= 5

    for component in components:
        assert component.kx > 0.0
        assert 2.0 * math.pi / component.kx >= 12.0 * grid.dx
        if component.alongshore_mode:
            assert (
                grid.Ly / abs(component.alongshore_mode)
                >= 12.0 * grid.dy
            )

    probe = (2.1, 0.7, -0.5 * scene.water_depth, 0.371)
    assert scene.inlet.elevation(probe[0], probe[1], probe[3]) == pytest.approx(
        scene.inlet.elevation(
            probe[0], probe[1], probe[3] + scene.inlet.period,
        ),
        abs=1.0e-14,
    )
    np.testing.assert_allclose(
        scene.inlet.velocity(*probe),
        scene.inlet.velocity(
            probe[0], probe[1], probe[2], probe[3] + scene.inlet.period,
        ),
        rtol=0.0,
        atol=1.0e-14,
    )
    assert scene.inlet.elevation(
        probe[0], 0.0, probe[3],
    ) == pytest.approx(
        scene.inlet.elevation(
            probe[0], grid.Ly, probe[3],
        ),
        abs=1.0e-14,
    )
    np.testing.assert_allclose(
        scene.inlet.velocity(probe[0], 0.0, probe[2], probe[3]),
        scene.inlet.velocity(probe[0], grid.Ly, probe[2], probe[3]),
        rtol=0.0,
        atol=1.0e-14,
    )

    assert len(scene.geometry.boulders) == 5
    still_water = scene.still_water_elevation
    production_spacing = np.asarray((grid.dx, grid.dy, grid.dz)) / 2.0
    for boulder in scene.geometry.boulders:
        lower = np.asarray(boulder.centre) - np.asarray(boulder.radii)
        upper = np.asarray(boulder.centre) + np.asarray(boulder.radii)
        assert lower[1] > grid.y0
        assert upper[1] < grid.y0 + grid.Ly
        assert upper[2] < grid.z0 + grid.Lz
        assert np.all(
            2.0 * np.asarray(boulder.radii) >= 4.0 * production_spacing
        )

        vertices = boulder.render_mesh(
            longitude_segments=128,
            latitude_segments=64,
        ).vertices
        beach_residual = (
            vertices[:, 2] - scene.geometry.beach.elevation(vertices[:, 0])
        )
        assert -0.030 <= np.min(beach_residual) <= -0.020
        assert np.min(vertices[:, 2]) < still_water < np.max(vertices[:, 2])

    boulders = scene.geometry.boulders
    minimum_corridor = 4.0 * production_spacing[1]
    for first_index, first in enumerate(boulders):
        first_vertices = first.render_mesh(
            longitude_segments=128,
            latitude_segments=64,
        ).vertices
        for second in boulders[first_index + 1:]:
            second_vertices = second.render_mesh(
                longitude_segments=128,
                latitude_segments=64,
            ).vertices
            clearance = min(
                np.min(first.signed_distance(second_vertices)),
                np.min(second.signed_distance(first_vertices)),
            )
            assert clearance >= minimum_corridor
    assert {material.name for material in scene.geometry.materials} == {
        "beach", "boulder",
    }
    assert scene.camera == ProductionCamera(
        position=(7.20, 3.00, 1.55),
        yaw_degrees=180.0,
        pitch_degrees=14.0,
        vertical_fov_degrees=32.0,
        exposure=1.12,
        white_balance=(1.0, 1.0, 1.0),
    )
    assert scene.hero_camera == ProductionCamera(
        position=(6.75, 5.15, 1.55),
        yaw_degrees=196.50436138175502,
        pitch_degrees=15.984663800879124,
        vertical_fov_degrees=38.0,
        exposure=1.12,
        white_balance=(1.0, 1.0, 1.0),
    )
    assert scene.camera.build(640, 384).width == 640
    assert scene.hero_camera.build(*scene.hero_preview_resolution).height == 480
    assert scene.sunset_camera == ProductionCamera(
        position=(6.50, 4.80, 1.25),
        yaw_degrees=196.2323506611562,
        pitch_degrees=12.472758324663351,
        vertical_fov_degrees=30.0,
        exposure=0.62,
        white_balance=(1.0, 1.0, 1.0),
    )
    assert scene.sunset_light_preset == "sunset"
    assert scene.sunset_probe_resolution == (960, 540)
    assert scene.sunset_final_resolution == (1920, 1080)
    assert scene.sunset_render_seed == 1001

    with pytest.raises(FrozenInstanceError):
        scene.water_depth = 0.3  # type: ignore[misc]

    capillary = production_scene(output_dir=tmp_path, surface_tension=0.072)
    assert capillary.solver.physics.surface_tension == pytest.approx(0.072)


def test_incident_boundary_guard_rejects_the_overdriven_aw1104_scale(tmp_path):
    # The withdrawn 1.60x AW-1104 scale sampled .851 of depth with the AW-1101
    # oblique pairs; the spread phases sample .694 at 1.45x, so the guard now
    # binds at 1.63x and the same-energy 1.70x source is rejected.
    bounded = production_scene(output_dir=tmp_path)
    overdriven = production_scene(output_dir=tmp_path, incident_scale=1.70)

    bounded_ratio = bounded.inlet.sampled_maximum_local_wave_height(
        bounded.solver.grid.x0,
    ) / bounded.water_depth
    overdriven_ratio = overdriven.inlet.sampled_maximum_local_wave_height(
        overdriven.solver.grid.x0,
    ) / overdriven.water_depth
    assert bounded_ratio <= (
        PRODUCTION_MAXIMUM_INCIDENT_BOUNDARY_HEIGHT_TO_DEPTH_RATIO
    )
    assert overdriven_ratio > (
        PRODUCTION_MAXIMUM_INCIDENT_BOUNDARY_HEIGHT_TO_DEPTH_RATIO
    )
