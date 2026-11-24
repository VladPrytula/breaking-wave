"""AW-1104 connected-sheet and ordered-flow diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from shoreloop.physics3d.swash import (
    MAXIMUM_SHALLOW_FILM_DEPTH_M,
    MINIMUM_VISIBLE_RETREAT_PLAN_AREA_M2,
    corridor_events,
    retreating_surface_component,
    retreat_events,
    sea_connected_wet_mask,
    visible_retreat_events,
)


def test_visible_sheet_support_is_fixed_in_physical_units():
    assert MAXIMUM_SHALLOW_FILM_DEPTH_M == pytest.approx(0.04)
    assert MINIMUM_VISIBLE_RETREAT_PLAN_AREA_M2 == pytest.approx(0.008)


def test_wet_beach_connectivity_wraps_alongshore_but_rejects_a_puddle():
    wet = np.zeros((5, 6), dtype=bool)
    wet[:4, 0] = True
    wet[3, -1] = True
    wet[4, -1] = True
    wet[4, 3] = True

    connected = sea_connected_wet_mask(wet)

    assert np.all(connected[:4, 0])
    assert connected[3, -1]
    assert connected[4, -1]
    assert not connected[4, 3]


def test_retreat_duration_starts_at_the_runup_maximum_inside_negative_flow():
    times = np.arange(11, dtype=np.float64) * 0.1
    shoreline = (3.00, 3.02, 3.05, 3.08, 3.10, 3.09,
                 3.07, 3.05, 3.03, 3.01, 3.00)
    sheet_u = (0.02, 0.01, -0.01, -0.02, -0.03, -0.03,
               -0.02, -0.02, -0.01, -0.01, 0.01)
    history = [
        {
            "time_s": float(time),
            "mean_shoreline_x": x,
            "shallow_film_mean_u_m_per_s": u,
        }
        for time, x, u in zip(times, shoreline, sheet_u, strict=True)
    ]

    event = retreat_events(history)[0]

    assert event["runup_maximum_time_s"] == 0.4
    assert event[
        "negative_sheet_duration_after_runup_maximum_s"
    ] == 0.5
    assert event["minimum_shallow_film_mean_u_m_per_s"] == -0.03


def test_corridor_event_requires_wash_before_reverse():
    discharges = (
        (0.0, -0.001, 0.0, 0.0),
        (0.002, -0.001, 0.0, 0.003),
        (0.001, 0.002, 0.0, 0.002),
        (-0.003, 0.001, 0.0, -0.002),
    )
    history = [
        {"time_s": 0.1 * index, "gap_discharges": values}
        for index, values in enumerate(discharges)
    ]

    events = corridor_events(history)

    assert [event["corridor"] for event in events] == [0, 3]
    assert events[0]["wash_time_s"] == 0.1
    assert events[0]["reverse_time_s"] == pytest.approx(0.3)


def test_retreating_surface_measurement_selects_largest_shallow_patch():
    vertices = np.asarray((
        (3.20, 0.00, 0.01),
        (3.30, 0.00, 0.01),
        (3.20, 0.10, 0.01),
        (3.30, 0.10, 0.01),
        (3.40, 0.00, 0.01),
        (3.45, 0.00, 0.01),
        (3.40, 0.05, 0.01),
    ))
    triangles = np.asarray(((0, 1, 2), (1, 3, 2), (4, 5, 6)))
    velocity = np.zeros_like(vertices)
    velocity[:, 0] = -0.02

    result = retreating_surface_component(
        vertices,
        triangles,
        velocity,
        vertices[:, 2],
        still_water_shoreline=3.15,
        maximum_depth=0.04,
    )

    assert result[
        "largest_connected_retreating_surface_plan_area_m2"
    ] == pytest.approx(0.01)
    assert result["largest_connected_retreating_surface_triangles"] == 2
    assert result["largest_connected_retreating_surface_vertices"] == 4
    assert result["largest_connected_retreating_surface_x_bounds_m"] \
        == pytest.approx((3.2, 3.3))
    assert result["largest_connected_retreating_surface_y_bounds_m"] \
        == pytest.approx((0.0, 0.1))


def test_visible_retreat_duration_uses_connected_patch_shoreward_extent():
    times = np.arange(10, dtype=np.float64) * 0.1
    area = (0.0, 0.0, 0.01, 0.02, 0.03,
            0.03, 0.02, 0.02, 0.01, 0.0)
    shoreward_x = (0.0, 0.0, 3.08, 3.10, 3.08,
                   3.07, 3.06, 3.04, 3.02, 0.0)
    history = [
        {
            "time_s": float(time),
            "largest_connected_retreating_surface_plan_area_m2": patch,
            "largest_connected_retreating_surface_x_bounds_m": [
                3.0 if patch > 0.0 else 0.0,
                patch_x,
            ],
            "largest_connected_retreating_surface_y_bounds_m": [0.0, 1.0],
        }
        for time, patch, patch_x in zip(times, area, shoreward_x, strict=True)
    ]

    event = visible_retreat_events(
        history,
        minimum_plan_area=0.008,
    )[0]

    assert event["runup_maximum_time_s"] == pytest.approx(0.3)
    assert event[
        "visible_retreat_duration_after_runup_maximum_s"
    ] == pytest.approx(0.5)
