"""Capture and present the accepted advanced-water production sequence."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import numpy as np

from production_render import (
    COLLAR_WIDTH,
    OFFSHORE_MESH_LENGTH,
    _arrays_unchanged,
    _look_arrays,
    _read_wetness,
    _render_bounds,
    _ripple_field,
    _sample_state,
    _solid_arrays,
    _stored_volume,
    _visible_domain_side_samples,
    _water_arrays,
    _wetness_arrays,
    _write_wetness,
)
from shoreloop.device import prepare_device
from shoreloop.physics3d.boulder_cluster import _gap_section
from shoreloop.physics3d.production_checkpoint import (
    load_production_checkpoint,
    write_production_checkpoint,
)
from shoreloop.physics3d.production_scene import (
    PRODUCTION_SURFACE_SMOOTHING_ITERATIONS,
    build_production_solver,
    production_scene,
)
from shoreloop.physics3d.solid_wetness import sample_scene_surface_wetness
from shoreloop.physics3d.topology import TopologyTracker
from shoreloop.physics3d.vof import measure_vof
from shoreloop.render.camera import tonemap
from shoreloop.render.mitsuba_water import MitsubaWaterRenderer, SharedSolidScene
from shoreloop.render.offshore_join import render_offshore_view
from shoreloop.render.shoreline_look import BEACH_TEXTURE_RESOLUTION
from shoreloop.render.sky import light_preset
from shoreloop.render.surface_mesh import (
    WaterSurfaceMesh,
    extract_state_free_surface_mesh,
)
from shoreloop.render.video import VideoWriter, write_png


ACCEPTED_START_TIME = 28.8
FRAME_INTERVAL = 0.04
VIDEO_FPS = 25.0
HERO_SPP = 128
WIDE_SPP = 32
RENDER_SEED = 910
TEXTURE_RESOLUTION = BEACH_TEXTURE_RESOLUTION
BOULDER_SUBDIVISION_LEVELS = 2
EXPECTED_TOPOLOGY = {
    23: (27, 1, 3),
    42: (25, 27, 3),
    56: (23, 1, 3),
}
HERO_SHEET_INDICES = (0, 10, 20, 23, 30, 37, 40, 42, 45, 49, 55, 60)
WIDE_INDICES = (0, 10, 20, 30, 37, 41, 45, 49, 55, 60)
EXTENDED_HERO_SHEET_INDICES = (
    0, 20, 37, 42, 49, 59,
    60, 80, 97, 102, 109, 119,
    120, 140, 157, 163, 169, 180,
)
REVIEW_PERIOD_INDICES = (0, 10, 20, 30, 37, 41, 45, 49, 54, 59)
CAPTURE_STORIES = {
    1: "AW-910 capture",
    3: "AW-1002 capture",
}


def _frame_times(period: float, periods: int) -> np.ndarray:
    return (
        ACCEPTED_START_TIME
        + np.arange(round(periods * period / FRAME_INTERVAL) + 1)
        * FRAME_INTERVAL
    )


def _period_behavior(
    records: list[dict[str, object]],
    period_index: int,
    frames_per_period: int,
) -> dict[str, object]:
    start = period_index * frames_per_period
    segment = records[start:start + frames_per_period + 1]
    peak_offset, peak = max(
        enumerate(segment),
        key=lambda item: (
            item[1]["state"]["downward_interface_cells"]
            + 4 * item[1]["state"]["multi_transition_columns"]
        ),
    )
    reconnection = next((
        record for record in segment[peak_offset + 1:]
        if record["state"]["multi_transition_columns"] == 0
        and record["state"]["maximum_vertical_transitions"] == 1
    ), None)
    shoreline = np.asarray([
        record["state"]["mean_shoreline_x"] for record in segment
    ])
    discharge = np.asarray([
        record["state"]["gap_discharges"] for record in segment
    ])
    return {
        "period": period_index + 1,
        "interval_s": [segment[0]["time_s"], segment[-1]["time_s"]],
        "maximum_downward_interface_cells": max(
            record["state"]["downward_interface_cells"] for record in segment
        ),
        "maximum_multi_transition_columns": max(
            record["state"]["multi_transition_columns"] for record in segment
        ),
        "maximum_vertical_transitions": max(
            record["state"]["maximum_vertical_transitions"] for record in segment
        ),
        "maximum_downward_surface_vertices": max(
            record["downward_surface_vertices"] for record in segment
        ),
        "breaker_peak_time_s": peak["time_s"],
        "breaker_peak_sequence_phase_s": peak["phase_s"],
        "breaker_peak_period_phase_s": (
            peak["time_s"] - segment[0]["time_s"]
        ),
        "reconnection_time_s": (
            None if reconnection is None else reconnection["time_s"]
        ),
        "reconnection_sequence_phase_s": (
            None if reconnection is None else reconnection["phase_s"]
        ),
        "reconnection_period_phase_s": (
            None if reconnection is None
            else reconnection["time_s"] - segment[0]["time_s"]
        ),
        "minimum_mean_shoreline_x": float(np.min(shoreline)),
        "maximum_mean_shoreline_x": float(np.max(shoreline)),
        "mean_shoreline_excursion": float(np.ptp(shoreline)),
        "maximum_forward_gap_discharges": np.max(
            discharge, axis=0,
        ).tolist(),
        "maximum_reverse_gap_discharges": np.maximum(
            0.0, -np.min(discharge, axis=0),
        ).tolist(),
        "maximum_cluster_water_depth": max(
            record["state"]["maximum_cluster_water_depth"]
            for record in segment
        ),
        "maximum_contacted_solid_cells": max(
            record["state"]["contacted_solid_cells"] for record in segment
        ),
        "maximum_solid_wetness": max(
            record["solid_wetness_maximum"] for record in segment
        ),
        "maximum_surface_foam": max(
            record["surface_foam_maximum"] for record in segment
        ),
        "maximum_surface_foam_fraction_over_0_05": max(
            record["surface_foam_fraction_over_0_05"] for record in segment
        ),
    }


def _foam_metrics(surface: WaterSurfaceMesh) -> dict[str, float]:
    foam = surface.foam.astype(np.float64)
    return {
        "surface_foam_maximum": float(np.max(foam)),
        "surface_foam_fraction_over_0_05": float(np.mean(foam > 0.05)),
    }


def _wetness_metrics(wetness) -> dict[str, float]:
    values = np.concatenate([
        sample.wetness.astype(np.float64) for sample in wetness
    ])
    return {
        "solid_wetness_maximum": float(np.max(values)),
        "solid_wetness_fraction_over_0_05": float(np.mean(values > 0.05)),
    }


def _capture_boundary(solver) -> dict[str, object]:
    vof = measure_vof(solver.state)
    storage = _stored_volume(solver)
    alpha = solver.state.get_interior("alpha")
    solid = solver.state.get_solid_interior().astype(bool)
    grid = solver.cfg.grid
    return {
        "time_s": float(solver.time),
        "free_water_volume": vof.volume,
        "porous_storage": storage,
        "alpha_bounds": [vof.alpha_min, vof.alpha_max],
        "solid_water_volume": float(np.sum(
            alpha[solid], dtype=np.float64,
        )) * grid.dx * grid.dy * grid.dz,
    }


def _capture(args: argparse.Namespace) -> int:
    surfaces_dir = args.output_dir / "payloads" / "surfaces"
    wetness_dir = args.output_dir / "payloads" / "wetness"
    surfaces_dir.mkdir(parents=True, exist_ok=True)
    wetness_dir.mkdir(parents=True, exist_ok=True)

    prepare_device(args.device)
    scene = production_scene(device=args.device, output_dir=args.output_dir)
    solver = build_production_solver(scene)
    load_production_checkpoint(solver, args.checkpoint)
    if not math.isclose(
        solver.time, ACCEPTED_START_TIME, rel_tol=0.0, abs_tol=1.0e-12,
    ):
        raise ValueError(
            f"capture requires the accepted {ACCEPTED_START_TIME:.1f} s "
            f"checkpoint, got {solver.time:.16g} s"
        )

    bounds = _render_bounds(scene)
    dry_solids = SharedSolidScene(scene.geometry, bounds)
    tracker = TopologyTracker(scene.solver, x_bounds=scene.topology_x_bounds)
    boulders = scene.geometry.boulders
    gaps = tuple(
        _gap_section(solver, boulders[lo], boulders[hi], x_position)
        for lo, hi, x_position in scene.gap_sections
    )
    frames_per_period = round(scene.inlet.period / FRAME_INTERVAL)
    targets = _frame_times(scene.inlet.period, args.periods)
    initial = _capture_boundary(solver)
    start_steps = solver.steps
    source_volume = 0.0
    wisp_volume = 0.0
    infiltrated_volume = 0.0
    drained_volume = 0.0
    period_health = [{
        "accepted_steps": 0,
        "maximum_divergence": 0.0,
        "maximum_pressure_iterations": 0,
        "minimum_dt": math.inf,
        "maximum_courant": 0.0,
    } for _ in range(args.periods)]
    boundaries = [{
        **initial,
        "relaxation_volume_source": 0.0,
        "wisp_volume_removed": 0.0,
        "infiltrated_volume": 0.0,
        "drained_volume": 0.0,
    }]
    records = []
    started = time.perf_counter()

    for index, target in enumerate(targets):
        while solver.time < target - 1.0e-12:
            health = period_health[(index - 1) // frames_per_period]
            timestep = solver.compute_dt()
            remaining = target - solver.time
            substeps = max(
                1,
                int(np.ceil(remaining / timestep.dt - 1.0e-9)),
            )
            dt = remaining / substeps
            projection = solver.step(dt, timestep=timestep)
            source_volume += solver.last_relaxation.volume_source
            wisp_volume += solver.last_wisp_removal.removed_volume
            infiltrated_volume += solver.last_porous_storage.infiltrated_volume
            drained_volume += solver.last_porous_storage.drained_volume
            health["accepted_steps"] += 1
            health["maximum_divergence"] = max(
                health["maximum_divergence"],
                projection.max_divergence_after,
            )
            health["maximum_pressure_iterations"] = max(
                health["maximum_pressure_iterations"],
                projection.pressure_solve.iterations,
            )
            health["minimum_dt"] = min(health["minimum_dt"], dt)
            health["maximum_courant"] = max(
                health["maximum_courant"],
                solver.last_advection.vof.max_courant,
            )

        surface_path = surfaces_dir / f"frame_{index:03d}.npz"
        wetness_path = wetness_dir / f"frame_{index:03d}.npz"
        surface = extract_state_free_surface_mesh(
            solver.state,
            smoothing_iterations=PRODUCTION_SURFACE_SMOOTHING_ITERATIONS,
        )
        surface.write(surface_path)
        wetness = sample_scene_surface_wetness(
            solver.state,
            scene.geometry,
            bounds,
            beach_segments=dry_solids.beach_segments,
            boulder_segments=dry_solids.boulder_segments,
        )
        _write_wetness(wetness_path, wetness)
        state = _sample_state(
            solver,
            tracker,
            gaps,
            scene.cluster_bounds,
            float(target),
        )
        observed_topology = (
            state["downward_interface_cells"],
            state["multi_transition_columns"],
            state["maximum_vertical_transitions"],
        )
        if index in EXPECTED_TOPOLOGY \
                and observed_topology != EXPECTED_TOPOLOGY[index]:
            phase = target - ACCEPTED_START_TIME
            raise RuntimeError(
                f"phase {phase:.2f} s left the accepted AW-905 trajectory: "
                f"expected {EXPECTED_TOPOLOGY[index]}, got {observed_topology}"
            )
        records.append({
            "frame": index,
            "phase_s": float(target - ACCEPTED_START_TIME),
            "time_s": float(target),
            "surface_path": str(surface_path),
            "wetness_path": str(wetness_path),
            "surface_vertices": len(surface.vertices),
            "surface_triangles": len(surface.triangles),
            "downward_surface_vertices": int(np.count_nonzero(
                surface.normals[:, 2] < -0.1,
            )),
            **_foam_metrics(surface),
            **_wetness_metrics(wetness),
            "state": state,
        })
        if index > 0 and index % frames_per_period == 0:
            boundaries.append({
                **_capture_boundary(solver),
                "relaxation_volume_source": source_volume,
                "wisp_volume_removed": wisp_volume,
                "infiltrated_volume": infiltrated_volume,
                "drained_volume": drained_volume,
            })
        print(
            f"capture {index + 1:03d}/{len(targets)}: "
            f"phase={target - ACCEPTED_START_TIME:.2f}s, "
            f"topology={observed_topology[0]}/{observed_topology[1]}, "
            f"triangles={len(surface.triangles)}",
            flush=True,
        )

    final = boundaries[-1]
    mass_residual = (
        final["free_water_volume"] + final["porous_storage"] + drained_volume
        - initial["free_water_volume"] - initial["porous_storage"]
        - source_volume + wisp_volume
    )
    grid = scene.solver.grid
    shoreline = np.asarray([
        record["state"]["mean_shoreline_x"] for record in records
    ])
    discharge = np.asarray([
        record["state"]["gap_discharges"] for record in records
    ])
    forward = np.max(discharge, axis=0)
    reverse = np.maximum(0.0, -np.min(discharge, axis=0))
    periods = []
    for period_index, health in enumerate(period_health):
        previous = boundaries[period_index]
        current = boundaries[period_index + 1]
        period_mass_residual = (
            current["free_water_volume"] + current["porous_storage"]
            - previous["free_water_volume"] - previous["porous_storage"]
            + current["drained_volume"] - previous["drained_volume"]
            - current["relaxation_volume_source"]
            + previous["relaxation_volume_source"]
            + current["wisp_volume_removed"]
            - previous["wisp_volume_removed"]
        )
        periods.append({
            **_period_behavior(records, period_index, frames_per_period),
            "initial_free_water_volume": previous["free_water_volume"],
            "final_free_water_volume": current["free_water_volume"],
            "initial_porous_storage": previous["porous_storage"],
            "final_porous_storage": current["porous_storage"],
            "infiltrated_volume": (
                current["infiltrated_volume"]
                - previous["infiltrated_volume"]
            ),
            "drained_volume": (
                current["drained_volume"] - previous["drained_volume"]
            ),
            "relaxation_volume_source": (
                current["relaxation_volume_source"]
                - previous["relaxation_volume_source"]
            ),
            "wisp_volume_removed": (
                current["wisp_volume_removed"]
                - previous["wisp_volume_removed"]
            ),
            "relative_mass_residual": (
                period_mass_residual / previous["free_water_volume"]
            ),
            "alpha_bounds": current["alpha_bounds"],
            "solid_water_volume": current["solid_water_volume"],
            **health,
        })
    final_checkpoint = write_production_checkpoint(
        solver,
        args.output_dir / "final_state.npz",
    )
    report = {
        "story": CAPTURE_STORIES[args.periods],
        "source_checkpoint": str(args.checkpoint),
        "final_checkpoint": str(final_checkpoint),
        "sequence_mode": "finite contiguous non-looping interval",
        "capture_periods": args.periods,
        "interval_s": [float(targets[0]), float(targets[-1])],
        "frame_interval_s": FRAME_INTERVAL,
        "frame_count": len(targets),
        "render_bounds_m": list(bounds),
        "grid": list(grid.shape),
        "spacing_m": [grid.dx, grid.dy, grid.dz],
        "maximum_downward_interface_cells": max(
            record["state"]["downward_interface_cells"] for record in records
        ),
        "maximum_multi_transition_columns": max(
            record["state"]["multi_transition_columns"] for record in records
        ),
        "maximum_vertical_transitions": max(
            record["state"]["maximum_vertical_transitions"] for record in records
        ),
        "maximum_downward_surface_vertices": max(
            record["downward_surface_vertices"] for record in records
        ),
        "minimum_mean_shoreline_x": float(np.min(shoreline)),
        "maximum_mean_shoreline_x": float(np.max(shoreline)),
        "mean_shoreline_excursion": float(np.ptp(shoreline)),
        "maximum_cluster_water_depth": max(
            record["state"]["maximum_cluster_water_depth"] for record in records
        ),
        "maximum_forward_gap_discharges": forward.tolist(),
        "maximum_reverse_gap_discharges": reverse.tolist(),
        "maximum_contacted_solid_cells": max(
            record["state"]["contacted_solid_cells"] for record in records
        ),
        "maximum_solid_wetness": max(
            record["solid_wetness_maximum"] for record in records
        ),
        "maximum_surface_foam_fraction_over_0_05": max(
            record["surface_foam_fraction_over_0_05"] for record in records
        ),
        "maximum_surface_foam": max(
            record["surface_foam_maximum"] for record in records
        ),
        "initial_free_water_volume": initial["free_water_volume"],
        "final_free_water_volume": final["free_water_volume"],
        "initial_porous_storage": initial["porous_storage"],
        "final_porous_storage": final["porous_storage"],
        "infiltrated_volume": infiltrated_volume,
        "drained_volume": drained_volume,
        "relaxation_volume_source": source_volume,
        "wisp_volume_removed": wisp_volume,
        "relative_mass_residual": mass_residual / initial["free_water_volume"],
        "alpha_bounds": final["alpha_bounds"],
        "solid_water_volume": final["solid_water_volume"],
        "maximum_divergence": max(
            period["maximum_divergence"] for period in periods
        ),
        "maximum_pressure_iterations": max(
            period["maximum_pressure_iterations"] for period in periods
        ),
        "minimum_dt": min(period["minimum_dt"] for period in periods),
        "maximum_courant": max(
            period["maximum_courant"] for period in periods
        ),
        "accepted_steps_this_capture": solver.steps - start_steps,
        "wall_seconds": time.perf_counter() - started,
        "periods": periods,
        "frames": records,
    }
    destination = args.output_dir / "capture_report.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")

    story = report["story"]
    if any(
        period["maximum_downward_interface_cells"] == 0
        or period["maximum_multi_transition_columns"] == 0
        or period["maximum_vertical_transitions"] < 3
        for period in periods
    ):
        raise RuntimeError(f"{story} lacks a resolved breaker in one period")
    if any(period["reconnection_time_s"] is None for period in periods):
        raise RuntimeError(f"{story} lacks reconnection after a period peak")
    if min(forward) <= 0.0 or min(reverse) <= 0.0:
        raise RuntimeError(f"{story} lacks wash/backwash through both gaps")
    if report["maximum_contacted_solid_cells"] == 0:
        raise RuntimeError(f"{story} contains no boulder contact")
    if any(
        not 0.0 <= period["alpha_bounds"][0]
        <= period["alpha_bounds"][1] <= 1.0
        for period in periods
    ):
        raise RuntimeError(f"{story} produced unbounded VOF")
    if any(period["solid_water_volume"] != 0.0 for period in periods):
        raise RuntimeError(f"{story} placed water in solid cells")
    if report["maximum_divergence"] >= 4.0e-3:
        raise RuntimeError(f"{story} exceeded the divergence gate")

    print(f"wrote {destination}", flush=True)
    return 0


def _render_payload(
    scene,
    surface: WaterSurfaceMesh,
    wetness,
    bounds: tuple[float, float, float, float],
    camera,
    target: float,
    *,
    variant: str,
    spp: int,
    sky=None,
    sun=None,
    seed: int = RENDER_SEED,
    spectrum=None,
    ripples_enabled: bool = True,
    foam_enabled: bool = True,
    pebble_microgeometry_enabled: bool = False,
    pebble_microgeometry=None,
    boulder_subdivision_levels: int = BOULDER_SUBDIVISION_LEVELS,
    simulated_water_only: bool = False,
):
    if simulated_water_only:
        # Render the retained interface directly. No offshore cut, fitting,
        # seam weld, analytic water extension or ripple map; the foam material
        # follows ``foam_enabled`` so a surface's own foam attribute still shows.
        mesh = surface
        join = None
        join_fade = None
        background = None
        period_y = scene.solver.grid.Ly
        ripple_field = None
    else:
        outer = render_offshore_view(
            surface,
            scene.inlet,
            target,
            camera,
            eta0=scene.still_water_elevation,
            device=scene.solver.device,
            handoff_x=scene.offshore_handoff_x,
            collar_width=COLLAR_WIDTH,
            offshore_mesh_length=OFFSHORE_MESH_LENGTH,
            cell_size=scene.solver.grid.dz,
            render_scale=4,
            sky=sky,
            sun=sun,
            spectrum=spectrum,
            ripples_enabled=ripples_enabled,
        )
        mesh = outer.periodic_instance_mesh
        join = outer.metrics
        join_fade = (join.handoff_x, join.handoff_x + join.collar_width)
        background = outer.image
        period_y = outer.periodic_y_period
        ripple_field = _ripple_field(
            mesh,
            scene.geometry,
            scene.still_water_elevation,
            outer.ripple_spectrum,
            scene.ripple_spacing,
            scene.solver.device,
            join.handoff_x,
            scene.solver.grid.y0,
            scene.solver.grid.y0 + scene.solver.grid.Ly,
        )
    solids = SharedSolidScene(
        scene.geometry,
        bounds,
        surface_wetness=wetness,
        beach_detail_enabled=True,
        boulder_detail_enabled=True,
        beach_texture_resolution=TEXTURE_RESOLUTION,
        beach_microgeometry_enabled=pebble_microgeometry_enabled,
        beach_microgeometry=pebble_microgeometry,
        boulder_subdivision_levels=boulder_subdivision_levels,
    )
    renderer = MitsubaWaterRenderer(
        mesh,
        solids,
        camera=camera,
        sky=sky,
        sun=sun,
        ripple_field=ripple_field,
        ripple_time=target,
        join_fade=join_fade,
        periodic_y_period=period_y,
        background_image=background,
        foam_enabled=foam_enabled,
        ripples_enabled=ripples_enabled and not simulated_water_only,
        variant=variant,
    )
    started = time.perf_counter()
    linear = renderer.render(spp=spp, seed=seed)
    return linear, join, renderer, time.perf_counter() - started


def _write_sheet(
    path: Path,
    records: list[dict[str, object]],
    indices: tuple[int, ...],
    shape: tuple[int, int],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        *shape,
        figsize=(3.2 * shape[1], 2.5 * shape[0]),
        constrained_layout=True,
        squeeze=False,
    )
    for axis, index in zip(axes.ravel(), indices, strict=True):
        record = records[index]
        axis.imshow(mpimg.imread(record["path"]))
        axis.set_title(f"phase {record['phase_s']:.2f} s", fontsize=9)
        axis.axis("off")
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _review_indices(capture: dict[str, object]) -> tuple[int, ...]:
    if capture["capture_periods"] != 3:
        raise ValueError("AW-1002 wide review requires a three-period capture")
    frames_per_period = (
        capture["frame_count"] - 1
    ) // capture["capture_periods"]
    indices = []
    for period_index in range(capture["capture_periods"]):
        local = list(REVIEW_PERIOD_INDICES)
        if period_index == capture["capture_periods"] - 1:
            local[-1] = frames_per_period
        indices.extend(
            period_index * frames_per_period + index for index in local
        )
    return tuple(indices)


def _review(args: argparse.Namespace) -> int:
    capture_path = args.output_dir / "capture_report.json"
    capture = json.loads(capture_path.read_text())
    frame_dir = args.output_dir / "frames" / "wide_review"
    linear_dir = args.output_dir / "linear" / "wide_review"
    frame_dir.mkdir(parents=True, exist_ok=True)
    linear_dir.mkdir(parents=True, exist_ok=True)

    prepare_device(args.device)
    scene = production_scene(device=args.device, output_dir=args.output_dir)
    camera = scene.camera.build(640, 384)
    bounds = _render_bounds(scene)
    records = []
    solid_reference = None
    look_reference = None
    solid_geometry_stable = True
    material_fields_stable = True
    payloads_unchanged = True
    started = time.perf_counter()

    for index in _review_indices(capture):
        payload = capture["frames"][index]
        target = float(payload["time_s"])
        surface = WaterSurfaceMesh.read(payload["surface_path"])
        wetness = _read_wetness(Path(payload["wetness_path"]))
        water_before = tuple(array.copy() for array in _water_arrays(surface))
        wetness_before = tuple(
            array.copy() for array in _wetness_arrays(wetness)
        )
        linear, join, renderer, render_seconds = _render_payload(
            scene,
            surface,
            wetness,
            bounds,
            camera,
            target,
            variant=args.variant,
            spp=args.spp,
        )
        if not np.all(np.isfinite(linear)):
            raise RuntimeError(f"wide review frame {index} is non-finite")
        current_solids = _solid_arrays(renderer)
        current_look = _look_arrays(renderer)
        if solid_reference is None:
            solid_reference = tuple(array.copy() for array in current_solids)
            look_reference = tuple(array.copy() for array in current_look)
        else:
            solid_geometry_stable &= _arrays_unchanged(
                solid_reference, current_solids,
            )
            material_fields_stable &= _arrays_unchanged(
                look_reference, current_look,
            )

        frame_path = frame_dir / f"frame_{index:03d}.png"
        linear_path = linear_dir / f"frame_{index:03d}.npy"
        write_png(frame_path, tonemap(linear, camera))
        np.save(linear_path, linear)
        records.append({
            "frame": index,
            "phase_s": float(payload["phase_s"]),
            "time_s": target,
            "path": str(frame_path),
            "linear_path": str(linear_path),
            "render_seconds": render_seconds,
            "linear_rgb_minimum": float(np.min(linear)),
            "linear_rgb_maximum": float(np.max(linear)),
            **_foam_metrics(renderer.mesh),
            "offshore_join": asdict(join),
        })
        payloads_unchanged &= _arrays_unchanged(
            water_before, _water_arrays(surface),
        )
        payloads_unchanged &= _arrays_unchanged(
            wetness_before, _wetness_arrays(wetness),
        )
        print(
            f"wide review {len(records):02d}/30: "
            f"phase={payload['phase_s']:.2f}s in {render_seconds:.2f}s",
            flush=True,
        )

    sheet = args.output_dir / "wide_review_contact_sheet.png"
    _write_sheet(sheet, records, tuple(range(len(records))), (3, 10))
    joins = [record["offshore_join"] for record in records]
    report = {
        "story": "AW-1002 scientific wide review",
        "source_capture_report": str(capture_path),
        "simulation_steps": 0,
        "sequence_mode": capture["sequence_mode"],
        "interval_s": capture["interval_s"],
        "resolution": [camera.width, camera.height],
        "frame_count": len(records),
        "samples_per_pixel": args.spp,
        "render_seed": RENDER_SEED,
        "camera": asdict(scene.camera),
        "solid_geometry_identical_across_frames": solid_geometry_stable,
        "material_fields_identical_across_frames": material_fields_stable,
        "surface_foam_and_wetness_payloads_unchanged": payloads_unchanged,
        "maximum_offshore_height_jump": max(
            record["maximum_height_jump"] for record in joins
        ),
        "maximum_offshore_slope_jump": max(
            record["maximum_slope_jump"] for record in joins
        ),
        "maximum_offshore_overlap_width": max(
            record["overlap_width"] for record in joins
        ),
        "maximum_joined_surface_foam_fraction_over_0_05": max(
            record["surface_foam_fraction_over_0_05"] for record in records
        ),
        "contact_sheet": str(sheet),
        "wall_seconds": time.perf_counter() - started,
        "frames": records,
    }
    destination = args.output_dir / "wide_review_report.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")

    if not solid_geometry_stable or not material_fields_stable \
            or not payloads_unchanged:
        raise RuntimeError("AW-1002 review changed frozen geometry or payload state")

    print(f"wrote {sheet}", flush=True)
    print(f"wrote {destination}", flush=True)
    return 0


def _render(args: argparse.Namespace) -> int:
    capture_path = args.output_dir / "capture_report.json"
    capture = json.loads(capture_path.read_text())
    extended = capture["capture_periods"] == 3
    prepare_device(args.device)
    scene = production_scene(device=args.device, output_dir=args.output_dir)
    if extended:
        story = "AW-1003 render"
        camera_config = scene.sunset_camera
        resolution = scene.sunset_final_resolution
        sky, sun = light_preset(scene.sunset_light_preset)
        render_seed = scene.sunset_render_seed
        look = "frozen AW-1001a beach relief and six-stone material fields"
        hero_sheet_indices = EXTENDED_HERO_SHEET_INDICES
        hero_sheet_shape = (3, 6)
        hero_dir = args.output_dir / "frames" / "sunset"
        linear_dir = args.output_dir / "linear" / "sunset"
    else:
        story = "AW-910 render"
        camera_config = scene.hero_camera
        resolution = scene.hero_preview_resolution
        sky = sun = None
        render_seed = RENDER_SEED
        look = "frozen AW-907 beach and six-stone material fields"
        hero_sheet_indices = HERO_SHEET_INDICES
        hero_sheet_shape = (3, 4)
        hero_dir = args.output_dir / "frames" / "hero"
        linear_dir = args.output_dir / "linear"
    wide_dir = args.output_dir / "frames" / "wide"
    for directory in (hero_dir, linear_dir):
        directory.mkdir(parents=True, exist_ok=True)
    if not extended:
        wide_dir.mkdir(parents=True, exist_ok=True)

    hero_camera = camera_config.build(*resolution)
    wide_camera = scene.camera.build(640, 384)
    bounds = _render_bounds(scene)
    visible_sides = _visible_domain_side_samples(hero_camera, scene)
    hero_records = []
    wide_records = []
    solid_reference = None
    look_reference = None
    solid_geometry_stable = True
    material_fields_stable = True
    payloads_unchanged = True
    started = time.perf_counter()

    for payload in capture["frames"]:
        index = int(payload["frame"])
        target = float(payload["time_s"])
        surface = WaterSurfaceMesh.read(payload["surface_path"])
        wetness = _read_wetness(Path(payload["wetness_path"]))
        water_before = tuple(array.copy() for array in _water_arrays(surface))
        wetness_before = tuple(
            array.copy() for array in _wetness_arrays(wetness)
        )
        linear, join, renderer, render_seconds = _render_payload(
            scene,
            surface,
            wetness,
            bounds,
            hero_camera,
            target,
            variant=args.variant,
            spp=args.spp,
            sky=sky,
            sun=sun,
            seed=render_seed,
        )
        if not np.all(np.isfinite(linear)):
            raise RuntimeError(f"hero frame {index} contains non-finite radiance")
        current_solids = _solid_arrays(renderer)
        current_look = _look_arrays(renderer)
        if solid_reference is None:
            solid_reference = tuple(array.copy() for array in current_solids)
            look_reference = tuple(array.copy() for array in current_look)
        else:
            solid_geometry_stable &= _arrays_unchanged(
                solid_reference, current_solids,
            )
            material_fields_stable &= _arrays_unchanged(
                look_reference, current_look,
            )

        frame_path = hero_dir / f"frame_{index:03d}.png"
        linear_path = linear_dir / f"frame_{index:03d}.npy"
        write_png(frame_path, tonemap(linear, hero_camera))
        np.save(linear_path, linear)
        hero_records.append({
            "frame": index,
            "phase_s": float(payload["phase_s"]),
            "time_s": target,
            "path": str(frame_path),
            "linear_path": str(linear_path),
            "render_seconds": render_seconds,
            "linear_rgb_minimum": float(np.min(linear)),
            "linear_rgb_maximum": float(np.max(linear)),
            **_foam_metrics(renderer.mesh),
            "offshore_join": asdict(join),
        })

        if not extended and index in WIDE_INDICES:
            wide_linear, wide_join, _, wide_seconds = _render_payload(
                scene,
                surface,
                wetness,
                bounds,
                wide_camera,
                target,
                variant=args.variant,
                spp=args.wide_spp,
            )
            wide_path = wide_dir / f"frame_{index:03d}.png"
            write_png(wide_path, tonemap(wide_linear, wide_camera))
            wide_records.append({
                "frame": index,
                "phase_s": float(payload["phase_s"]),
                "time_s": target,
                "path": str(wide_path),
                "render_seconds": wide_seconds,
                "offshore_join": asdict(wide_join),
            })

        payloads_unchanged &= _arrays_unchanged(
            water_before, _water_arrays(surface),
        )
        payloads_unchanged &= _arrays_unchanged(
            wetness_before, _wetness_arrays(wetness),
        )
        print(
            f"render {index + 1:02d}/{len(capture['frames'])}: "
            f"phase={payload['phase_s']:.2f}s in {render_seconds:.2f}s",
            flush=True,
        )

    hero_sheet = args.output_dir / (
        "sunset_contact_sheet.png" if extended else "hero_contact_sheet.png"
    )
    _write_sheet(
        hero_sheet,
        hero_records,
        hero_sheet_indices,
        hero_sheet_shape,
    )
    if extended:
        wide_report_path = args.output_dir / "wide_review_report.json"
        wide_report = json.loads(wide_report_path.read_text())
        wide_sheet = Path(wide_report["contact_sheet"])
        wide_records = wide_report["frames"]
        wide_resolution = wide_report["resolution"]
        wide_samples_per_pixel = wide_report["samples_per_pixel"]
    else:
        wide_report_path = None
        wide_sheet = args.output_dir / "wide_contact_sheet.png"
        _write_sheet(
            wide_sheet,
            wide_records,
            tuple(range(len(wide_records))),
            (2, 5),
        )
        wide_resolution = [640, 384]
        wide_samples_per_pixel = args.wide_spp
    joins = [record["offshore_join"] for record in hero_records]
    report = {
        "story": story,
        "source_capture_report": str(capture_path),
        "simulation_steps": 0,
        "sequence_mode": "finite contiguous non-looping interval",
        "interval_s": capture["interval_s"],
        "resolution": [hero_camera.width, hero_camera.height],
        "frame_count": len(hero_records),
        "samples_per_pixel": args.spp,
        "render_seed": render_seed,
        "camera": asdict(camera_config),
        "light": scene.sunset_light_preset if extended else "overcast",
        "look": look,
        "texture_resolution": TEXTURE_RESOLUTION,
        "visible_local_domain_side_samples": visible_sides,
        "solid_geometry_identical_across_frames": solid_geometry_stable,
        "material_fields_identical_across_frames": material_fields_stable,
        "surface_foam_and_wetness_payloads_unchanged": payloads_unchanged,
        "maximum_offshore_height_jump": max(
            record["maximum_height_jump"] for record in joins
        ),
        "maximum_offshore_slope_jump": max(
            record["maximum_slope_jump"] for record in joins
        ),
        "maximum_offshore_overlap_width": max(
            record["overlap_width"] for record in joins
        ),
        "maximum_joined_surface_foam_fraction_over_0_05": max(
            record["surface_foam_fraction_over_0_05"]
            for record in hero_records
        ),
        "mean_hero_render_seconds": float(np.mean([
            record["render_seconds"] for record in hero_records
        ])),
        "wide_resolution": wide_resolution,
        "wide_frame_count": len(wide_records),
        "wide_samples_per_pixel": wide_samples_per_pixel,
        "wide_source_report": (
            None if wide_report_path is None else str(wide_report_path)
        ),
        "hero_contact_sheet": str(hero_sheet),
        "wide_contact_sheet": str(wide_sheet),
        "wall_seconds": time.perf_counter() - started,
        "frames": hero_records,
        "wide_frames": wide_records,
    }
    destination = args.output_dir / "render_report.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")

    if any(visible_sides.values()):
        raise RuntimeError(
            f"production camera exposes a local-domain side: {visible_sides}"
        )
    if not solid_geometry_stable or not material_fields_stable \
            or not payloads_unchanged:
        raise RuntimeError(f"{story} changed frozen geometry or material state")

    print(f"wrote {hero_sheet}", flush=True)
    print(f"wrote {wide_sheet}", flush=True)
    print(f"wrote {destination}", flush=True)
    return 0


def _read_display_png(path: Path) -> np.ndarray:
    import matplotlib.image as mpimg

    image = mpimg.imread(path)[..., :3]
    return (255.0 * image + 0.5).astype(np.uint8)


def _encode(args: argparse.Namespace) -> int:
    capture_path = args.output_dir / "capture_report.json"
    render_path = args.output_dir / "render_report.json"
    capture = json.loads(capture_path.read_text())
    render = json.loads(render_path.read_text())
    extended = capture["capture_periods"] == 3
    story = "AW-1003" if extended else "AW-910"
    video_name = (
        "advanced_water_sunset.mp4" if extended else "advanced_water_hero.mp4"
    )
    video_path = args.output_dir / video_name
    width, height = render["resolution"]
    started = time.perf_counter()
    with VideoWriter(
        video_path,
        width,
        height,
        fps=VIDEO_FPS,
        cq=args.cq,
    ) as video:
        for frame in render["frames"]:
            video.add(_read_display_png(Path(frame["path"])))
        codec = video.codec

    report = {
        "story": story,
        "sequence_mode": "finite contiguous non-looping interval",
        "source_checkpoint": capture["source_checkpoint"],
        "interval_s": capture["interval_s"],
        "frame_interval_s": capture["frame_interval_s"],
        "frame_count": render["frame_count"],
        "video_fps": VIDEO_FPS,
        "video_duration_s": render["frame_count"] / VIDEO_FPS,
        "video_path": str(video_path),
        "video_codec": codec,
        "video_quality": args.cq,
        "endpoint_connected_to_start": False,
        "physical_crossfade_frames": 0,
        "capture_report": str(capture_path),
        "render_report": str(render_path),
        "hero_contact_sheet": render["hero_contact_sheet"],
        "wide_contact_sheet": render["wide_contact_sheet"],
        "encoding_wall_seconds": time.perf_counter() - started,
    }
    destination = args.output_dir / "report.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {video_path}", flush=True)
    print(f"wrote {destination}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture, review, render, and encode advanced water.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/final_production_sequence"),
    )
    parser.add_argument("--device", default="cuda:0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser(
        "capture",
        help="resume the accepted state and extract all final-frame payloads",
    )
    capture.add_argument("--checkpoint", type=Path, required=True)
    capture.add_argument(
        "--periods",
        type=int,
        choices=tuple(CAPTURE_STORIES),
        default=1,
        help="capture one AW-910 period or the three-period AW-1002 interval",
    )

    review = subparsers.add_parser(
        "review",
        help="render only the AW-1002 low-cadence scientific wide sheet",
    )
    review.add_argument("--variant", default="cuda_ad_rgb")
    review.add_argument("--spp", type=int, default=WIDE_SPP)

    render = subparsers.add_parser(
        "render",
        help="render lossless hero frames and the wide diagnostic sheet",
    )
    render.add_argument("--variant", default="cuda_ad_rgb")
    render.add_argument("--spp", type=int, default=HERO_SPP)
    render.add_argument("--wide-spp", type=int, default=WIDE_SPP)

    encode = subparsers.add_parser(
        "encode",
        help="encode the inspected lossless hero frames as a non-looping MP4",
    )
    encode.add_argument("--cq", type=int, default=18)

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "capture":
        return _capture(args)
    if args.command == "review":
        return _review(args)
    if args.command == "render":
        return _render(args)
    return _encode(args)


if __name__ == "__main__":
    raise SystemExit(main())
