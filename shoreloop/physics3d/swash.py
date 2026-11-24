"""Connected-sheet diagnostics for physical swash and backwash."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


RETREAT_SPEED_M_PER_S = -0.005
CORRIDOR_WASH_M3_PER_S = 1.0e-3
CORRIDOR_REVERSE_M3_PER_S = 2.0e-4
# Fixed physical supports keep the development/validation comparison causal.
MAXIMUM_SHALLOW_FILM_DEPTH_M = 0.04
# Approximately six development-grid plan cells, held in physical units so a
# refined validation mesh does not make the visibility gate easier.
MINIMUM_VISIBLE_RETREAT_PLAN_AREA_M2 = 8.0e-3


def sea_connected_wet_mask(wet: np.ndarray) -> np.ndarray:
    """Return face-connected wet columns joined to the offshore boundary."""
    labels, count = ndimage.label(
        wet,
        structure=np.asarray(((0, 1, 0), (1, 1, 1), (0, 1, 0))),
    )
    parent = np.arange(count + 1, dtype=np.int32)

    def root(label: int) -> int:
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = int(parent[label])
        return label

    for lower, upper in zip(labels[:, 0], labels[:, -1], strict=True):
        if lower == 0 or upper == 0:
            continue
        lower_root = root(int(lower))
        upper_root = root(int(upper))
        if lower_root != upper_root:
            parent[upper_root] = lower_root

    roots = np.asarray([root(label) for label in range(count + 1)])
    offshore = labels[0]
    sea_roots = np.unique(roots[offshore[offshore > 0]])
    return wet & np.isin(roots[labels], sea_roots)


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1) - 1
    return list(zip(starts.tolist(), stops.tolist(), strict=True))


def retreat_events(history: list[dict[str, object]]) -> list[dict[str, float]]:
    """Measure intervals whose whole connected wet-beach mean is negative."""
    times = np.asarray([record["time_s"] for record in history])
    shoreline = np.asarray([record["mean_shoreline_x"] for record in history])
    sheet_u = np.asarray([
        record["shallow_film_mean_u_m_per_s"] for record in history
    ])
    retreating = sheet_u <= RETREAT_SPEED_M_PER_S
    washing = sheet_u >= -RETREAT_SPEED_M_PER_S
    events = []
    for start, stop in _true_runs(retreating):
        prior = np.flatnonzero(washing[:start])
        if prior.size == 0:
            continue
        peak = start + int(np.argmax(shoreline[start:stop + 1]))
        events.append({
            "preceding_wash_time_s": float(times[prior[-1]]),
            "runup_maximum_time_s": float(times[peak]),
            "runup_maximum_mean_shoreline_x_m": float(shoreline[peak]),
            "retreat_end_time_s": float(times[stop]),
            "negative_sheet_duration_after_runup_maximum_s": float(
                times[stop] - times[peak]
            ),
            "minimum_shallow_film_mean_u_m_per_s": float(np.min(
                sheet_u[peak:stop + 1],
            )),
        })
    return events


def retreating_surface_component(
    vertices: np.ndarray,
    triangles: np.ndarray,
    velocity: np.ndarray,
    surface_depth: np.ndarray,
    *,
    still_water_shoreline: float,
    maximum_depth: float,
) -> dict[str, float | int | list[float]]:
    """Measure the largest connected shallow surface patch moving offshore."""
    candidate = (
        (vertices[:, 0] >= still_water_shoreline)
        & (surface_depth > 0.0)
        & (surface_depth <= maximum_depth)
        & (velocity[:, 0] <= RETREAT_SPEED_M_PER_S)
    )
    points = vertices[triangles].astype(np.float64)
    projected_twice_area = np.cross(
        points[:, 1] - points[:, 0],
        points[:, 2] - points[:, 0],
    )[:, 2]
    supported = np.all(candidate[triangles], axis=1) & (
        projected_twice_area > 0.0
    )
    supported_triangles = triangles[supported]
    if len(supported_triangles) == 0:
        return {
            "largest_connected_retreating_surface_plan_area_m2": 0.0,
            "largest_connected_retreating_surface_triangles": 0,
            "largest_connected_retreating_surface_vertices": 0,
            "largest_connected_retreating_surface_x_span_m": 0.0,
            "largest_connected_retreating_surface_y_span_m": 0.0,
            "largest_connected_retreating_surface_x_bounds_m": [0.0, 0.0],
            "largest_connected_retreating_surface_y_bounds_m": [0.0, 0.0],
            "largest_connected_retreating_surface_mean_u_m_per_s": 0.0,
        }

    parent = np.arange(len(vertices), dtype=np.int32)

    def root(vertex: int) -> int:
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = int(parent[vertex])
        return vertex

    for a, b, c in supported_triangles:
        roots = (root(int(a)), root(int(b)), root(int(c)))
        parent[roots[1]] = roots[0]
        parent[roots[2]] = roots[0]

    triangle_roots = np.asarray([
        root(int(triangle[0])) for triangle in supported_triangles
    ])
    triangle_area = 0.5 * projected_twice_area[supported]
    component_area = np.bincount(
        triangle_roots,
        weights=triangle_area,
        minlength=len(vertices),
    )
    largest = int(np.argmax(component_area))
    in_largest = triangle_roots == largest
    component_triangles = supported_triangles[in_largest]
    component_vertices = np.unique(component_triangles)
    points = vertices[component_vertices].astype(np.float64)
    return {
        "largest_connected_retreating_surface_plan_area_m2": float(
            component_area[largest]
        ),
        "largest_connected_retreating_surface_triangles": len(
            component_triangles
        ),
        "largest_connected_retreating_surface_vertices": len(
            component_vertices
        ),
        "largest_connected_retreating_surface_x_span_m": float(np.ptp(
            points[:, 0]
        )),
        "largest_connected_retreating_surface_y_span_m": float(np.ptp(
            points[:, 1]
        )),
        "largest_connected_retreating_surface_x_bounds_m": [
            float(np.min(points[:, 0])),
            float(np.max(points[:, 0])),
        ],
        "largest_connected_retreating_surface_y_bounds_m": [
            float(np.min(points[:, 1])),
            float(np.max(points[:, 1])),
        ],
        "largest_connected_retreating_surface_mean_u_m_per_s": float(np.mean(
            velocity[component_vertices, 0]
        )),
    }


def visible_retreat_events(
    history: list[dict[str, object]],
    minimum_plan_area: float,
) -> list[dict[str, float]]:
    """Measure visible retreat after the same component's shoreward maximum."""
    times = np.asarray([record["time_s"] for record in history])
    component_shoreward_x = np.asarray([
        record["largest_connected_retreating_surface_x_bounds_m"][1]
        for record in history
    ])
    area = np.asarray([
        record["largest_connected_retreating_surface_plan_area_m2"]
        for record in history
    ])
    visible = area >= minimum_plan_area
    events = []
    for start, stop in _true_runs(visible):
        representative = start + int(np.argmax(area[start:stop + 1]))
        y_lower, y_upper = history[representative][
            "largest_connected_retreating_surface_y_bounds_m"
        ]
        window = component_shoreward_x[start:stop + 1]
        peak = start + int(np.flatnonzero(
            window == np.max(window)
        )[-1])
        events.append({
            "runup_maximum_time_s": float(times[peak]),
            "runup_maximum_component_shoreward_x_m": float(
                component_shoreward_x[peak]
            ),
            "retreat_support_y_lower_m": float(y_lower),
            "retreat_support_y_upper_m": float(y_upper),
            "visible_retreat_start_time_s": float(times[start]),
            "visible_retreat_end_time_s": float(times[stop]),
            "visible_retreat_duration_after_runup_maximum_s": float(
                times[stop] - times[peak]
            ),
            "maximum_connected_retreat_plan_area_m2": float(np.max(
                area[start:stop + 1]
            )),
        })
    return events


def corridor_events(
    history: list[dict[str, object]],
) -> list[dict[str, float | int]]:
    """Return corridor events in which measurable wash precedes reverse flux."""
    times = np.asarray([record["time_s"] for record in history])
    discharges = np.asarray([record["gap_discharges"] for record in history])
    events = []
    for corridor in range(discharges.shape[1]):
        values = discharges[:, corridor]
        reverse_indices = np.flatnonzero(
            values <= -CORRIDOR_REVERSE_M3_PER_S
        )
        eligible = [
            index for index in reverse_indices
            if np.any(values[:index] >= CORRIDOR_WASH_M3_PER_S)
        ]
        if not eligible:
            continue
        reverse = min(eligible, key=lambda index: values[index])
        wash = int(np.argmax(values[:reverse]))
        events.append({
            "corridor": corridor,
            "wash_time_s": float(times[wash]),
            "wash_discharge_m3_per_s": float(values[wash]),
            "reverse_time_s": float(times[reverse]),
            "reverse_discharge_m3_per_s": float(values[reverse]),
        })
    return events
