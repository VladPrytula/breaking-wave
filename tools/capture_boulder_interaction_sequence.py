"""Capture or continue a current-physics boulder and swash sequence."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import time

import numpy as np
import warp as wp

from production_render import _sample_state, _stored_volume
from shoreloop.config import BC
from shoreloop.device import prepare_device
from shoreloop.physics3d.boulder_cluster import _gap_section
from shoreloop.physics3d.boulder_wash import _cell_velocity
from shoreloop.physics3d.interface import estimate_interface_normals
from shoreloop.physics3d.plic import reconstruct_plic
from shoreloop.physics3d.production_checkpoint import (
    load_production_checkpoint, write_production_checkpoint,
)
from shoreloop.physics3d.production_scene import (
    BEDS, INCIDENT_SPECTRA,
    PRODUCTION_INCIDENT_SCALE_FROM_AW1103,
    PRODUCTION_MAXIMUM_INCIDENT_BOUNDARY_HEIGHT_TO_DEPTH_RATIO,
    build_production_solver,
    production_scene,
)
from shoreloop.physics3d.resolvedness import measure_interface_resolvedness
from shoreloop.physics3d.swash import (
    CORRIDOR_REVERSE_M3_PER_S,
    CORRIDOR_WASH_M3_PER_S,
    MAXIMUM_SHALLOW_FILM_DEPTH_M,
    MINIMUM_VISIBLE_RETREAT_PLAN_AREA_M2,
    RETREAT_SPEED_M_PER_S,
    corridor_events,
    retreating_surface_component,
    retreat_events,
    sea_connected_wet_mask,
    visible_retreat_events,
)
from shoreloop.physics3d.topology import TopologyTracker
from shoreloop.physics3d.vof import measure_vof
from shoreloop.render.offshore_join import MultiValuedOffshoreSeamError
from shoreloop.render.surface_mesh import RENDER_SURFACES, extract_state_free_surface_mesh


def _capture_wetness(solver, scene, output_dir, index):
    """Save the state-dependent solid appearance for immediate or later rendering."""
    from production_render import _render_bounds, _write_wetness
    from shoreloop.physics3d.solid_wetness import sample_scene_surface_wetness
    from shoreloop.render.mitsuba_water import SharedSolidScene

    bounds = _render_bounds(scene)
    dry_solids = SharedSolidScene(scene.geometry, bounds)
    wetness = sample_scene_surface_wetness(
        solver.state, scene.geometry, bounds,
        beach_segments=dry_solids.beach_segments,
        boulder_segments=dry_solids.boulder_segments,
    )
    label = f"frame_{index:04d}"
    wetness_dir = output_dir / "wetness"
    wetness_dir.mkdir(exist_ok=True)
    wetness_path = _write_wetness(wetness_dir / f"{label}.npz", wetness)
    return bounds, wetness, wetness_path


def _render_capture(
    solver, scene, surface, target, output_dir, index, args, *,
    simulated_water_only=False,
    ripples_enabled=True,
    foam_enabled=True,
):
    """Render this state with the existing sunset presentation."""
    from render_production_sequence import BOULDER_SUBDIVISION_LEVELS, _render_payload
    from shoreloop.closures.ripples import crossing_ripple_spectrum
    from shoreloop.render.camera import tonemap
    from shoreloop.render.sky import light_preset
    from shoreloop.render.video import write_png

    bounds, wetness, wetness_path = _capture_wetness(solver, scene, output_dir, index)
    label = f"frame_{index:04d}"
    camera = scene.sunset_camera.build(args.width, args.height)
    sky, sun = light_preset(scene.sunset_light_preset)
    linear, join, renderer, render_seconds = _render_payload(
        scene, surface, wetness, bounds, camera, target,
        variant=args.variant, spp=args.spp, sky=sky, sun=sun,
        seed=scene.sunset_render_seed,
        spectrum=crossing_ripple_spectrum(period=scene.inlet.period),
        pebble_microgeometry_enabled=True,
        simulated_water_only=simulated_water_only,
        ripples_enabled=ripples_enabled,
        foam_enabled=foam_enabled,
    )
    if not np.all(np.isfinite(linear)):
        raise RuntimeError(f"non-finite radiance at t={target:.6f}s")
    linear_dir = output_dir / "linear"
    linear_dir.mkdir(exist_ok=True)
    np.save(linear_dir / f"{label}.npy", linear)
    write_png(linear_dir / f"{label}.png", np.clip(linear, 0, 1))
    path = output_dir / "frames" / f"{label}.png"
    write_png(path, tonemap(linear, camera))
    return dict(path=str(path), wetness_path=str(wetness_path),
                linear_path=str(linear_dir / f"{label}.npy"),
                boulder_subdivision_levels=BOULDER_SUBDIVISION_LEVELS,
                render_seconds=render_seconds,
                ripples_enabled=ripples_enabled and not simulated_water_only,
                foam_shading_enabled=foam_enabled,
                analytic_offshore_enabled=not simulated_water_only,
                offshore_join=None if join is None else asdict(join))


def _periodic_distance(values: np.ndarray, centre: float, length: float) -> np.ndarray:
    distance = np.abs(values - centre)
    return np.minimum(distance, length - distance)


def _depth_averaged_flow(solver):
    grid = solver.cfg.grid
    alpha = solver.state.get_interior("alpha")
    theta = solver.state.get_geometry_interior("theta")
    weight = alpha * theta
    u, v, _ = _cell_velocity(solver)
    depth = np.sum(weight, axis=2, dtype=np.float64) * grid.dz
    transport_u = np.sum(weight * u, axis=2, dtype=np.float64) * grid.dz
    transport_v = np.sum(weight * v, axis=2, dtype=np.float64) * grid.dz
    mean_u = np.zeros_like(depth)
    mean_v = np.zeros_like(depth)
    np.divide(transport_u, depth, out=mean_u, where=depth > 0.0)
    np.divide(transport_v, depth, out=mean_v, where=depth > 0.0)
    dv_dx = np.gradient(mean_v, grid.dx, axis=0)
    du_dy = (
        np.roll(mean_u, -1, axis=1) - np.roll(mean_u, 1, axis=1)
    ) / (2.0 * grid.dy)
    return depth, mean_u, mean_v, dv_dx - du_dy


def _swash_metrics(solver, scene) -> dict[str, object]:
    grid = solver.cfg.grid
    x, _, _ = solver.state.grid.coordinates("cell")
    depth, mean_u, mean_v, _ = _depth_averaged_flow(solver)
    wet = depth > 0.25 * grid.dz
    connected = sea_connected_wet_mask(wet)
    shoreline = np.asarray([
        x[np.flatnonzero(connected[:, j])[-1]] + 0.5 * grid.dx
        for j in range(grid.ny)
    ])
    still_water_shoreline = (
        scene.geometry.beach.toe_x
        + scene.water_depth / scene.geometry.beach.slope
    )
    wet_beach = connected & (
        x[:, None] + 0.5 * grid.dx >= still_water_shoreline
    )
    shallow = wet_beach & (depth <= MAXIMUM_SHALLOW_FILM_DEPTH_M)
    shallow_depth = depth[shallow]
    weight = np.where(shallow, depth, 0.0)
    total_weight = np.sum(weight, dtype=np.float64)
    speed = np.hypot(mean_u, mean_v)
    if np.any(shallow):
        shallow_values = {
            "shallow_film_mean_depth_m": float(np.mean(shallow_depth)),
            "shallow_film_p95_depth_m": float(np.quantile(
                shallow_depth, 0.95,
            )),
            "shallow_film_mean_u_m_per_s": float(
                np.sum(weight * mean_u, dtype=np.float64) / total_weight
            ),
            "shallow_film_mean_v_m_per_s": float(
                np.sum(weight * mean_v, dtype=np.float64) / total_weight
            ),
            "shallow_film_p95_speed_m_per_s": float(np.quantile(
                speed[shallow], 0.95,
            )),
            "shallow_film_reverse_area_fraction": float(np.mean(
                mean_u[shallow] < RETREAT_SPEED_M_PER_S,
            )),
        }
    else:
        shallow_values = {
            "shallow_film_mean_depth_m": 0.0,
            "shallow_film_p95_depth_m": 0.0,
            "shallow_film_mean_u_m_per_s": 0.0,
            "shallow_film_mean_v_m_per_s": 0.0,
            "shallow_film_p95_speed_m_per_s": 0.0,
            "shallow_film_reverse_area_fraction": 0.0,
        }
    return {
        "shoreline_x_by_alongshore_column_m": shoreline.tolist(),
        "connected_wet_beach_area_m2": (
            float(np.count_nonzero(wet_beach)) * grid.dx * grid.dy
        ),
        "shallow_film_area_m2": (
            float(np.count_nonzero(shallow)) * grid.dx * grid.dy
        ),
        **shallow_values,
    }


def _weighted_mean(
    field: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
) -> float:
    weight = np.where(mask, depth, 0.0)
    return float(
        np.sum(weight * field, dtype=np.float64)
        / np.sum(weight, dtype=np.float64)
    )


def _boulder_flow_metrics(solver, scene) -> list[dict[str, float | int]]:
    grid = solver.cfg.grid
    x, y, _ = solver.state.grid.coordinates("cell")
    xx = x[:, None]
    depth, mean_u, mean_v, vertical_vorticity = _depth_averaged_flow(solver)
    result = []
    for index, boulder in enumerate(scene.geometry.boulders):
        cx, cy, _ = boulder.centre
        rx, ry, _ = boulder.radii
        y_distance = _periodic_distance(y, cy, grid.Ly)[None, :]
        wet = depth > 0.25 * grid.dz
        approach = (
            (xx >= cx - rx - 0.30)
            & (xx <= cx - rx - 0.05)
            & (y_distance <= ry + 0.10)
            & wet
        )
        lee = (
            (xx >= cx + rx + 0.03)
            & (xx <= cx + rx + 0.40)
            & (y_distance <= ry + 0.15)
            & wet
        )
        side_x = (xx >= cx - rx) & (xx <= cx + rx)
        lower = (
            side_x
            & (y[None, :] >= cy - ry - 0.25)
            & (y[None, :] <= cy - ry - 0.02)
            & wet
        )
        upper = (
            side_x
            & (y[None, :] >= cy + ry + 0.02)
            & (y[None, :] <= cy + ry + 0.25)
            & wet
        )
        result.append({
            "boulder": index,
            "approach_mean_u_m_per_s": _weighted_mean(
                mean_u, depth, approach,
            ),
            "lee_mean_u_m_per_s": _weighted_mean(mean_u, depth, lee),
            "lee_mean_v_m_per_s": _weighted_mean(mean_v, depth, lee),
            "lee_maximum_vertical_vorticity_s^-1": float(np.max(
                np.abs(vertical_vorticity[lee]),
            )),
            "lower_side_mean_v_m_per_s": _weighted_mean(
                mean_v, depth, lower,
            ),
            "upper_side_mean_v_m_per_s": _weighted_mean(
                mean_v, depth, upper,
            ),
        })
    return result


def _targets(end_time: float, interval: float) -> np.ndarray:
    return np.arange(round(end_time / interval) + 1, dtype=np.float64) * interval


def _write_progress(output_dir: Path, progress: dict) -> None:
    temporary = output_dir / "progress.json.tmp"
    temporary.write_text(json.dumps(progress, indent=2) + "\n")
    temporary.replace(output_dir / "progress.json")


DIVERGENCE_GATE_PER_S = 4.0e-3
RELATIVE_MASS_GATE = 5.0e-5


def _numerical_ledger(solver, grid, initial_volume, initial_storage,
                      source_volume, wisp_volume, drained_volume):
    """Water bounds, water in solids and the accounted mass residual of the state."""
    vof = measure_vof(solver.state)
    storage = _stored_volume(solver)
    alpha = solver.state.get_interior("alpha")
    solid = solver.state.get_solid_interior().astype(bool)
    solid_water = float(np.sum(alpha[solid], dtype=np.float64)) * (grid.dx * grid.dy * grid.dz)
    residual = (vof.volume + storage + drained_volume
                - initial_volume - initial_storage - source_volume + wisp_volume)
    return vof, storage, solid_water, residual


def _cell_record(solver, cell, value):
    i, j, k = cell
    state = solver.state
    u, v, w = (state.get_interior(q) for q in ("u", "v", "w"))
    return {
        "cell": [i, j, k], "divergence_per_s": value,
        "alpha": float(state.get_interior("alpha")[i, j, k]),
        "theta": float(state.get_geometry_interior("theta")[i, j, k]),
        "pressure": float(state.get_interior("p")[i, j, k]),
        "u_faces": [float(u[i, j, k]), float(u[i + 1, j, k])],
        "v_faces": [float(v[i, j, k]), float(v[i, j + 1, k])],
        "w_faces": [float(w[i, j, k]), float(w[i, j, k + 1])],
    }


def _check_numerical_gates(solver, output_dir, time_s, maximum_divergence,
                           vof, solid_water, relative_residual):
    """Stop at the first capture whose ledger can no longer pass the final gates.

    The gated divergence is the larger of the FP64 solution's maximum
    liquid-relative divergence and the stored float32 field's maximum net flux
    per Cartesian cell volume. The float32 liquid-relative maximum is logged
    only: it divides write-back rounding by a possibly tiny liquid volume.
    """
    failures = []
    if not 0.0 <= vof.alpha_min <= vof.alpha_max <= 1.0:
        failures.append(f"unbounded VOF [{vof.alpha_min:.9g}, {vof.alpha_max:.9g}]")
    if solid_water != 0.0:
        failures.append(f"water in solids {solid_water:.6g} m3")
    if maximum_divergence >= DIVERGENCE_GATE_PER_S:
        failures.append(f"maximum divergence {maximum_divergence:.6g}/s >= {DIVERGENCE_GATE_PER_S}/s")
    if abs(relative_residual) >= RELATIVE_MASS_GATE:
        failures.append(f"relative mass residual {relative_residual:.6g} >= {RELATIVE_MASS_GATE}")
    if not failures:
        return
    record = {"time_s": time_s, "steps": solver.steps, "failures": failures,
              "maximum_divergence": maximum_divergence, "solid_water_m3": solid_water,
              "relative_mass_residual": relative_residual,
              "alpha_bounds": [vof.alpha_min, vof.alpha_max]}
    operator = getattr(solver.projection, "last_operator", None)
    if operator is not None and operator.cell_count:
        stored = operator.read_mac_state(solver.state)
        for label, cartesian in (("cell_relative", True), ("liquid_relative", False)):
            cell, value = operator.worst_divergence_cell(stored, cartesian=cartesian)
            record[f"worst_{label}"] = _cell_record(solver, cell, value)
    path = output_dir / "failure.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    raise RuntimeError(f"numerical gate failed at t={time_s:.4f} s: "
                       + "; ".join(failures) + f" (record {path})")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture fresh distributed-boulder interaction surfaces.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path,
                        help="continue a captured case's latest checkpoint and numerical ledger into a new directory")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--grid-scale", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--interface-reconstruction", choices=("youngs", "height_function"),
                        help="override the governing interface reconstruction for a paired control")
    parser.add_argument("--wave-source-reconstruction", choices=("column", "plic_rays"),
                        help="select the geometric wave-source reconstruction for a paired control")
    parser.add_argument("--wave-source-quadrature", type=int, default=8,
                        help="Gauss order per footprint coordinate for PLIC-ray forcing")
    parser.add_argument("--vof-flux-reconstruction", choices=("axis_slabs", "swept_prism"),
                        help="select the unsplit liquid flux geometry for a paired control")
    parser.add_argument("--end-time", type=float, default=6.80)
    parser.add_argument("--capture-start", type=float, default=1.20)
    parser.add_argument("--interval", type=float, default=0.04)
    parser.add_argument("--surface-tension", type=float, default=0.0)
    parser.add_argument("--render", action="store_true",
                        help="render each captured fresh state with the sunset camera")
    parser.add_argument("--save-wetness", action="store_true",
                        help="save solid wetness with each surface for offline rendering")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--spp", type=int, default=128)
    parser.add_argument("--variant", default="cuda_ad_rgb")
    parser.add_argument("--no-ripples", action="store_true",
                        help="disable ripple shading without changing the camera or water geometry")
    parser.add_argument("--no-foam", action="store_true",
                        help="disable foam shading without changing the passive foam state")
    parser.add_argument("--render-surface", choices=RENDER_SURFACES, default="plic_boundary",
                        help="render-surface reconstruction of the liquid_mimetic state: the "
                             "exact cut/PLIC boundary (default) or the theta-weighted nodal "
                             "alpha=0.5 isosurface; the solver state is unchanged either way")
    parser.add_argument("--save-nodal-surface", action="store_true",
                        help="also save the nodal_isosurface mesh of every capture beside the "
                             "selected render surface, so the presentation can be chosen later")
    parser.add_argument("--offshore-handoff-x", type=float,
                        help="render-only location of the offshore attachment in metres")
    parser.add_argument(
        "--simulated-water-only", action="store_true",
        help="render the retained water directly, without ripple/foam shading or analytic offshore water; retain the configured camera",
    )
    parser.add_argument(
        "--incident-scale",
        type=float,
        default=PRODUCTION_INCIDENT_SCALE_FROM_AW1103,
    )
    parser.add_argument(
        "--porous-permeability",
        type=float,
        help="override only the effective shallow-bed permeability in m^2",
    )
    parser.add_argument("--incident", choices=INCIDENT_SPECTRA, default="aw1103",
                        help="incident spectrum: the AW-1103 group or the reference-calibrated one")
    parser.add_argument("--bed", choices=BEDS, default="aw1104",
                        help="bed closures: the first capture's slip/storage/drag or the gravel laws")
    parser.add_argument(
        "--receding-contact-diffusivity",
        type=float,
        help=(
            "override only the production contact-line receding diffusivity "
            "in m^2/s"
        ),
    )
    parser.add_argument("--require-aw1104", action="store_true")
    parser.add_argument("--diagnostic", action="store_true",
                        help="require numerical health while reporting, but not requiring, the late-time interaction gates")
    args = parser.parse_args()
    if args.resume_from is not None and args.resume_from.resolve() == args.output_dir.resolve():
        parser.error("--resume-from and --output-dir must differ to preserve the input case")
    if args.diagnostic and args.require_aw1104:
        parser.error("--diagnostic cannot require the full AW-1104 interaction gate")
    if (
        args.porous_permeability is not None
        and args.receding_contact_diffusivity is not None
    ):
        parser.error("override one AW-1104 comparison mechanism at a time")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    surfaces_dir = args.output_dir / "surfaces"
    surfaces_dir.mkdir(parents=True, exist_ok=True)
    prepare_device(args.device)
    scene = production_scene(
        device=args.device,
        output_dir=args.output_dir,
        surface_tension=args.surface_tension,
        incident_scale=args.incident_scale,
        incident=args.incident,
        bed=args.bed,
    )
    if args.offshore_handoff_x is not None:
        scene = replace(scene, offshore_handoff_x=args.offshore_handoff_x)
    if args.porous_permeability is not None:
        if scene.porous_drag is None or scene.porous_drag.law != "darcy_forchheimer":
            parser.error("only the Darcy-Forchheimer shallow drag has a permeability to override")
        scene = replace(
            scene,
            porous_drag=replace(
                scene.porous_drag,
                permeability=args.porous_permeability,
            ),
        )
    if args.receding_contact_diffusivity is not None:
        scene = replace(
            scene,
            contact_line=replace(
                scene.contact_line,
                receding_diffusivity=args.receding_contact_diffusivity,
            ),
        )
    retained = scene.solver.grid
    grid = replace(
        retained,
        nx=args.grid_scale * retained.nx,
        ny=args.grid_scale * retained.ny,
        nz=args.grid_scale * retained.nz,
    )
    scene = replace(
        scene,
        solver=replace(
            scene.solver,
            grid=grid,
            interface_reconstruction=(args.interface_reconstruction
                                      or scene.solver.interface_reconstruction),
            wave_source_reconstruction=(args.wave_source_reconstruction
                                        or scene.solver.wave_source_reconstruction),
            wave_source_quadrature=args.wave_source_quadrature,
            vof_flux_reconstruction=(args.vof_flux_reconstruction
                                     or scene.solver.vof_flux_reconstruction),
            output_dir=str(args.output_dir),
        ),
    )
    previous = None
    if args.resume_from is not None:
        saved_config = json.loads((args.resume_from / "configuration.json").read_text())
        previous = json.loads((args.resume_from / "progress.json").read_text())
        physical_scene = json.loads(json.dumps(asdict(scene)))
        # Camera, offshore attachment and output location do not enter dynamics.
        physical_scene["solver"]["output_dir"] = saved_config["scene"]["solver"]["output_dir"]
        for key in ("sunset_camera", "offshore_handoff_x"):
            physical_scene[key] = saved_config["scene"][key]
        if physical_scene != saved_config["scene"]:
            parser.error("continuation must use the saved physical configuration")
        if (args.interval != saved_config["interval_s"]
                or args.capture_start != saved_config["capture_start_s"]):
            parser.error("continuation must preserve the saved capture schedule")
        if args.end_time <= previous["time_s"]:
            parser.error("--end-time must be after the saved checkpoint")
        if previous["history"][-1]["time_s"] != previous["time_s"]:
            parser.error("continuation requires a completed checkpoint capture")

    solver = build_production_solver(scene)
    if previous is not None:
        load_production_checkpoint(solver, args.resume_from / "latest_state.npz")
        if solver.time != previous["time_s"] or solver.steps != previous["steps"]:
            raise ValueError("checkpoint and progress ledger refer to different states")
    tracker = TopologyTracker(scene.solver, x_bounds=scene.topology_x_bounds)
    boulders = scene.geometry.boulders
    gaps = tuple(
        _gap_section(solver, boulders[lo], boulders[hi], x_position)
        for lo, hi, x_position in scene.gap_sections
    )

    initial_volume = measure_vof(solver.state).volume
    initial_storage = _stored_volume(solver)
    source_volume = 0.0
    wisp_volume = 0.0
    infiltrated_volume = 0.0
    drained_volume = 0.0
    maximum_divergence = 0.0
    maximum_stored_divergence = 0.0
    maximum_pressure_iterations = 0
    maximum_courant = 0.0
    maximum_face_speed = 0.0
    minimum_dt = math.inf
    maximum_residue_volume = 0.0
    minimum_capillary_limit = math.inf
    history: list[dict[str, object]] = []
    surfaces: list[dict[str, object]] = []
    unrendered_frames: list[int] = []
    if previous is not None:
        initial_volume = previous["initial_free_water_volume_m3"]
        initial_storage = previous["initial_porous_storage_m3"]
        source_volume = previous["relaxation_volume_source_m3"]
        wisp_volume = previous["wisp_volume_removed_m3"]
        infiltrated_volume = previous["infiltrated_volume_m3"]
        drained_volume = previous["drained_volume_m3"]
        if "maximum_stored_divergence" in previous:
            maximum_divergence = previous["maximum_divergence"]
            maximum_stored_divergence = previous["maximum_stored_divergence"]
        else:
            # Ledgers before 2026-10-09 gated the float32 liquid-relative
            # statistic; the gated FP64/cell-volume maximum starts here.
            maximum_stored_divergence = previous["maximum_divergence"]
        maximum_pressure_iterations = previous["maximum_pressure_iterations"]
        maximum_courant = previous["maximum_courant"]
        maximum_face_speed = previous["maximum_face_speed_m_per_s"]
        maximum_residue_volume = previous["maximum_residue_volume_m3"]
        minimum_dt = previous["minimum_dt_s"]
        # The capillary limit is constant for this unchanged grid/physics.
        history = previous["history"]
        surfaces = previous["surfaces"]
        unrendered_frames = [i for i, frame in enumerate(surfaces) if "render_failed" in frame]
        tracker.sample(solver.state, time=solver.time)
    start_time = solver.time
    started = time.perf_counter()
    last_projection = None
    (args.output_dir / "configuration.json").write_text(json.dumps(dict(
        scene=asdict(scene), end_time_s=args.end_time,
        capture_start_s=args.capture_start, interval_s=args.interval,
        rendering=args.render, resolution=[args.width, args.height], spp=args.spp,
        render_surface=args.render_surface,
        save_nodal_surface=args.save_nodal_surface,
        save_wetness=args.save_wetness,
        resumed_from=None if args.resume_from is None else str(args.resume_from),
        start_time_s=start_time,
        diagnostic=args.diagnostic,
        camera=asdict(scene.sunset_camera), light=scene.sunset_light_preset,
        seed=scene.sunset_render_seed, pebble_microgeometry_enabled=True,
        simulated_water_only=args.simulated_water_only,
        ripples_enabled=not (args.no_ripples or args.simulated_water_only),
        foam_shading_enabled=not (args.no_foam or args.simulated_water_only),
        analytic_offshore_enabled=not args.simulated_water_only,
    ), indent=2)+"\n")
    print(f"{grid.shape} {scene.solver.pressure_discretization} capture: "
          f"{start_time}..{args.end_time}s, interval={args.interval}s, render={args.render}", flush=True)

    for target in _targets(args.end_time, args.interval):
        if previous is not None and target <= start_time + 1.0e-12:
            continue
        while solver.time < target - 1.0e-12:
            timestep = solver.compute_dt()
            remaining = target - solver.time
            substeps = max(1, int(np.ceil(remaining / timestep.dt - 1.0e-9)))
            dt = remaining / substeps
            step_started = time.perf_counter()
            last_projection = solver.step(dt, timestep=timestep)
            source_volume += solver.last_relaxation.volume_source
            wisp_volume += solver.last_wisp_removal.removed_volume
            infiltrated_volume += solver.last_porous_storage.infiltrated_volume
            drained_volume += solver.last_porous_storage.drained_volume
            maximum_divergence = max(
                maximum_divergence,
                last_projection.solution_max_divergence,
                last_projection.max_cell_divergence,
            )
            maximum_stored_divergence = max(
                maximum_stored_divergence,
                last_projection.max_divergence_after,
            )
            maximum_pressure_iterations = max(
                maximum_pressure_iterations,
                last_projection.pressure_solve.iterations,
            )
            maximum_courant = max(
                maximum_courant,
                solver.last_advection.vof.max_courant,
            )
            maximum_face_speed = max(
                maximum_face_speed,
                timestep.max_face_speed,
            )
            minimum_dt = min(minimum_dt, dt)
            minimum_capillary_limit = min(
                minimum_capillary_limit,
                timestep.capillary_limit,
            )
            operator = solver.projection.last_operator
            advection = solver.last_advection
            print(f"step={solver.steps} t={solver.time:.6f}s dt={dt:.6g}s "
                  f"limit={timestep.limiting_constraint} "
                  f"wall={time.perf_counter()-step_started:.2f}s "
                  f"cg={last_projection.pressure_solve.iterations} "
                  f"vof-passes={advection.vof.limiter_iterations} "
                  f"rms-div={last_projection.rms_divergence_after:.6g}/s "
                  f"max-div={last_projection.solution_max_divergence:.3g}/s "
                  f"cell-div={last_projection.max_cell_divergence:.3g}/s "
                  f"stored-div={last_projection.max_divergence_after:.3g}/s "
                  f"visc-cg={solver.last_viscosity.iterations if solver.last_viscosity else 0} "
                  f"wall-cg={solver.last_wall_stress.iterations if solver.last_wall_stress else 0} "
                  f"speed={advection.max_water_speed:.3g}m/s "
                  f"discarded={advection.discarded_momentum:.3g}kg*m/s "
                  f"residue={operator.residue_cells}cells/{operator.residue_volume*1e6:.2f}mL", flush=True)
            maximum_residue_volume = max(maximum_residue_volume, operator.residue_volume)

        # Preserve the reached solver state before diagnostics, surface export,
        # wetness sampling or rendering can refuse it. Ordinary programmer
        # errors still propagate; the integration no longer loses its progress.
        latest_checkpoint = write_production_checkpoint(
            solver, args.output_dir / "latest_state.npz",
        )
        progress = {
            "time_s": solver.time, "steps": solver.steps,
            "checkpoint": str(latest_checkpoint),
            "initial_free_water_volume_m3": initial_volume,
            "initial_porous_storage_m3": initial_storage,
            "relaxation_volume_source_m3": source_volume,
            "wisp_volume_removed_m3": wisp_volume,
            "infiltrated_volume_m3": infiltrated_volume,
            "drained_volume_m3": drained_volume,
            "maximum_divergence": maximum_divergence,
            "maximum_stored_divergence": maximum_stored_divergence,
            "maximum_pressure_iterations": maximum_pressure_iterations,
            "maximum_courant": maximum_courant,
            "maximum_face_speed_m_per_s": maximum_face_speed,
            "maximum_residue_volume_m3": maximum_residue_volume,
            "minimum_dt_s": minimum_dt if math.isfinite(minimum_dt) else None,
            "history": history, "surfaces": surfaces,
        }
        _write_progress(args.output_dir, progress)
        # A ledger that cannot pass the final gates stops the run here, with
        # the state and the worst cells recorded, instead of days later.
        vof, _, solid_water, residual = _numerical_ledger(
            solver, grid, initial_volume, initial_storage, source_volume, wisp_volume, drained_volume)
        _check_numerical_gates(solver, args.output_dir, float(target), maximum_divergence,
                               vof, solid_water, residual / initial_volume)
        state = _sample_state(
            solver,
            tracker,
            gaps,
            scene.cluster_bounds,
            float(target),
        )
        state.update(_swash_metrics(solver, scene))
        state["boulder_flows"] = _boulder_flow_metrics(solver, scene)
        surface_retreat = {
            "largest_connected_retreating_surface_plan_area_m2": 0.0,
            "largest_connected_retreating_surface_triangles": 0,
            "largest_connected_retreating_surface_vertices": 0,
            "largest_connected_retreating_surface_x_span_m": 0.0,
            "largest_connected_retreating_surface_y_span_m": 0.0,
            "largest_connected_retreating_surface_x_bounds_m": [0.0, 0.0],
            "largest_connected_retreating_surface_y_bounds_m": [0.0, 0.0],
            "largest_connected_retreating_surface_mean_u_m_per_s": 0.0,
        }
        if target >= args.capture_start - 1.0e-12:
            # The frame's retained planes, reconstructed once: the PLIC surface
            # and the resolvedness diagnostic describe the same geometry.
            reconstruction = reconstruct_plic(
                solver.state, estimate_interface_normals(solver.state).normals)
            surface = extract_state_free_surface_mesh(
                solver.state,
                smoothing_iterations=0,
                render_surface=args.render_surface,
                reconstruction=reconstruction if args.render_surface == "plic_boundary" else None,
            )
            surface_depth = (
                surface.vertices[:, 2]
                - scene.geometry.beach.elevation(surface.vertices[:, 0])
            )
            surface_retreat = retreating_surface_component(
                surface.vertices,
                surface.triangles,
                surface.velocity,
                surface_depth,
                still_water_shoreline=(
                    scene.geometry.beach.toe_x
                    + scene.water_depth / scene.geometry.beach.slope
                ),
                maximum_depth=MAXIMUM_SHALLOW_FILM_DEPTH_M,
            )
            label = f"{target:05.2f}".replace(".", "_")
            path = surface.write(surfaces_dir / f"surface_{label}s_raw.npz")
            surfaces.append({
                "time_s": float(target),
                "mesh_paths": [str(path)],
                "render_surface": args.render_surface,
                "vertices": len(surface.vertices),
                "triangles": len(surface.triangles),
                **surface_retreat,
            })
            if args.save_nodal_surface and args.render_surface != "nodal_isosurface":
                nodal = extract_state_free_surface_mesh(
                    solver.state, smoothing_iterations=0, render_surface="nodal_isosurface",
                )
                surfaces[-1]["nodal_mesh_path"] = str(
                    nodal.write(surfaces_dir / f"surface_{label}s_nodal.npz"))
            # Resolvedness of the frame's planes (AW-1113): read-only diagnostic,
            # exported beside the surfaces with its summary and cost.
            resolvedness_started = time.perf_counter()
            resolvedness = measure_interface_resolvedness(solver.state, reconstruction)
            surfaces[-1]["resolvedness_path"] = str(
                resolvedness.write(surfaces_dir / f"resolvedness_{label}s.npz"))
            surfaces[-1]["resolvedness"] = resolvedness.summary()
            surfaces[-1]["resolvedness_seconds"] = time.perf_counter() - resolvedness_started
            index = len(surfaces)-1
            if args.save_wetness and not args.render:
                _, _, wetness_path = _capture_wetness(solver, scene, args.output_dir, index)
                surfaces[-1]["wetness_path"] = str(wetness_path)
            if args.render:
                try:
                    rendered = _render_capture(
                        solver, scene, surface, float(target), args.output_dir,
                        index, args, simulated_water_only=args.simulated_water_only,
                        ripples_enabled=not args.no_ripples,
                        foam_enabled=not args.no_foam,
                    )
                except MultiValuedOffshoreSeamError as error:
                    # The join refuses a surface whose cut at the handoff plane
                    # is not single-valued. The water is unaffected; the raw
                    # surface and sampled wetness are already saved, so the
                    # frame is recorded as refused and rendered offline later.
                    rendered = dict(
                        render_failed=str(error),
                        wetness_path=str(args.output_dir / "wetness" / f"frame_{index:04d}.npz"),
                    )
                    unrendered_frames.append(index)
                surfaces[-1].update(rendered)
                if "path" in rendered:
                    print(f"render t={target:.2f}s: {rendered['path']} "
                          f"in {rendered['render_seconds']:.2f}s", flush=True)
                else:
                    print(f"render t={target:.2f}s: refused by the offshore join "
                          f"({rendered['render_failed']}); frame {index} kept for offline rendering",
                          flush=True)
            frame_report = args.output_dir / "frames" / f"frame_{index:04d}.json"
            frame_report.parent.mkdir(exist_ok=True)
            frame_report.write_text(json.dumps(surfaces[-1], indent=2)+"\n")
        state.update(surface_retreat)
        history.append(state)
        _write_progress(args.output_dir, progress)
        print(
            f"t={target:.2f}s shoreline={state['mean_shoreline_x']:.3f}m "
            f"sheet-u={state['terminal_sheet_mean_u_m_per_s']:+.3f}m/s "
            f"topology={state['downward_interface_cells']}/"
            f"{state['multi_transition_columns']}",
            flush=True,
        )

    wp.synchronize()
    final_vof, final_storage, solid_water, mass_residual = _numerical_ledger(
        solver, grid, initial_volume, initial_storage, source_volume, wisp_volume, drained_volume)
    checkpoint_path = write_production_checkpoint(
        solver,
        args.output_dir / "final_state.npz",
    )
    ledger = solver.energy_ledger
    final_energy = ledger.measure(solver.state)
    discharges = np.asarray([
        record["gap_discharges"] for record in history
    ])
    forward = np.max(discharges, axis=0)
    reverse = np.maximum(0.0, -np.min(discharges, axis=0))
    wrap_counts = []
    for record in history:
        wrap_counts.append(sum(
            flow["lower_side_mean_v_m_per_s"]
            * flow["upper_side_mean_v_m_per_s"] < 0.0
            and abs(flow["lower_side_mean_v_m_per_s"]) >= 0.005
            and abs(flow["upper_side_mean_v_m_per_s"]) >= 0.005
            for flow in record["boulder_flows"]
        ))
    shoreline = np.asarray([
        record["mean_shoreline_x"] for record in history
    ])
    shoreline_by_column = np.asarray([
        record["shoreline_x_by_alongshore_column_m"] for record in history
    ])
    whole_beach_mean_retreat_events = retreat_events(history)
    measured_visible_retreat_events = visible_retreat_events(
        history,
        MINIMUM_VISIBLE_RETREAT_PLAN_AREA_M2,
    )
    measured_corridor_events = corridor_events(history)
    longest_retreat = max(
        (
            event["visible_retreat_duration_after_runup_maximum_s"]
            for event in measured_visible_retreat_events
        ),
        default=0.0,
    )
    ordered_corridors = {
        int(event["corridor"]) for event in measured_corridor_events
    }
    localized_breaker = any(
        record["largest_connected_multi_transition_columns"] > 0
        and 0.0 < record["connected_multi_transition_y_span_m"]
        < 0.75 * grid.Ly
        for record in history
    )
    dissipative_work = {
        "molecular_loss": ledger.molecular_loss,
        "sgs_loss": ledger.sgs_loss,
        "wall_loss": ledger.wall_loss,
        "porous_loss": ledger.porous_loss,
    }
    incident_wave_height_samples = 512
    incident_boundary_wave_height = scene.inlet.sampled_maximum_local_wave_height(
        grid.x0,
        alongshore_samples=incident_wave_height_samples,
        time_samples=incident_wave_height_samples,
    )
    incident_boundary_height_to_depth = (
        incident_boundary_wave_height / scene.water_depth
    )
    surface_periodic_axes = [
        axis for axis in "xyz"
        if (
            getattr(scene.solver.bc, f"{axis}lo") == BC.PERIODIC
            and getattr(scene.solver.bc, f"{axis}hi") == BC.PERIODIC
        )
    ]
    aw1104_acceptance = {
        "mean_shoreline_excursion_at_least_0_15_m": (
            float(np.ptp(shoreline)) >= 0.15
        ),
        "negative_connected_sheet_at_least_0_5_s_after_runup_maximum": (
            longest_retreat >= 0.5
        ),
        "separated_outer_corridors_wash_then_reverse": (
            {0, len(gaps) - 1} <= ordered_corridors
        ),
        "localized_breaker_component_present": localized_breaker,
        "dissipative_work_nonnegative": all(
            value >= 0.0 for value in dissipative_work.values()
        ),
        "incident_boundary_height_to_depth_at_most_0_78": (
            incident_boundary_height_to_depth
            <= PRODUCTION_MAXIMUM_INCIDENT_BOUNDARY_HEIGHT_TO_DEPTH_RATIO
        ),
    }
    report = {
        "story": "AW-1104 physical swash/backwash capture",
        "diagnostic": args.diagnostic,
        "case": {1: "retained", 2: "half_spacing_xyz", 4: "quarter_spacing_xyz"}[args.grid_scale],
        "grid": list(grid.shape),
        "pressure_discretization": scene.solver.pressure_discretization,
        "interface_reconstruction": scene.solver.interface_reconstruction,
        "vof_flux_reconstruction": scene.solver.vof_flux_reconstruction,
        "spacing_m": [grid.dx, grid.dy, grid.dz],
        "surface_tension_n_per_m": scene.solver.physics.surface_tension,
        "relaxation_interface_representation": solver.wave_relaxation.interface_representation,
        "wave_source_quadrature": scene.solver.wave_source_quadrature,
        "relaxation_width_m": scene.relaxation_width,
        "relaxation_timescale_s": solver.wave_relaxation.timescale,
        "offshore_handoff_x_m": scene.offshore_handoff_x,
        "surface_periodic_axes": surface_periodic_axes,
        "incident_group_scale_from_aw1103": (
            scene.incident_scale_from_aw1103
        ),
        "incident_component_amplitudes_m": [
            component.amplitude for component in scene.inlet.components
        ],
        "incident_group_linear_energy_proxy_m2": sum(
            component.amplitude**2 for component in scene.inlet.components
        ),
        "incident_group_maximum_component_steepness": max(
            component.steepness for component in scene.inlet.components
        ),
        "incident_boundary_maximum_local_wave_height_m": (
            incident_boundary_wave_height
        ),
        "incident_boundary_maximum_wave_height_to_depth_ratio": (
            incident_boundary_height_to_depth
        ),
        "incident_boundary_maximum_wave_height_to_depth_ratio_limit": (
            PRODUCTION_MAXIMUM_INCIDENT_BOUNDARY_HEIGHT_TO_DEPTH_RATIO
        ),
        "incident_boundary_wave_height_sampling": {
            "x_m": grid.x0,
            "alongshore_samples": incident_wave_height_samples,
            "time_samples": incident_wave_height_samples,
            "period_s": scene.inlet.period,
        },
        "contact_line": asdict(scene.contact_line),
        "porous_storage": asdict(scene.porous_storage),
        "porous_drag": None if scene.porous_drag is None else asdict(scene.porous_drag),
        "integration_interval_s": [0.0, args.end_time],
        "capture_interval_s": [args.capture_start, args.end_time],
        "ordinary_interval_s": args.interval,
        "frame_count": len(surfaces),
        "unrendered_frames": unrendered_frames,
        "initial_state": "tapered incident field imposed geometrically, then projected without body force",
        "boulders": [asdict(boulder) for boulder in boulders],
        "gap_sections": [list(section) for section in scene.gap_sections],
        "gap_open_areas_m2": [gap.open_area for gap in gaps],
        "maximum_forward_gap_discharges_m3_per_s": forward.tolist(),
        "maximum_reverse_gap_discharges_m3_per_s": reverse.tolist(),
        "reversing_corridors": int(np.count_nonzero(
            (forward >= 1.0e-3) & (reverse >= 2.0e-4),
        )),
        "maximum_simultaneous_boulder_wrap_structures": max(wrap_counts),
        "mean_shoreline_range_m": [
            float(np.min(shoreline)),
            float(np.max(shoreline)),
        ],
        "mean_shoreline_excursion_m": float(np.ptp(shoreline)),
        "alongshore_column_shoreline_excursions_m": np.ptp(
            shoreline_by_column, axis=0,
        ).tolist(),
        "minimum_alongshore_column_shoreline_excursion_m": float(np.min(
            np.ptp(shoreline_by_column, axis=0),
        )),
        "maximum_alongshore_column_shoreline_excursion_m": float(np.max(
            np.ptp(shoreline_by_column, axis=0),
        )),
        "connected_wet_beach_area_range_m2": [
            min(record["connected_wet_beach_area_m2"] for record in history),
            max(record["connected_wet_beach_area_m2"] for record in history),
        ],
        "shallow_film_area_range_m2": [
            min(record["shallow_film_area_m2"] for record in history),
            max(record["shallow_film_area_m2"] for record in history),
        ],
        "shallow_film_mean_depth_range_m": [
            min(record["shallow_film_mean_depth_m"] for record in history),
            max(record["shallow_film_mean_depth_m"] for record in history),
        ],
        "shallow_film_mean_u_range_m_per_s": [
            min(record["shallow_film_mean_u_m_per_s"] for record in history),
            max(record["shallow_film_mean_u_m_per_s"] for record in history),
        ],
        "retreat_speed_threshold_m_per_s": RETREAT_SPEED_M_PER_S,
        "maximum_shallow_film_depth_m": MAXIMUM_SHALLOW_FILM_DEPTH_M,
        "whole_connected_wet_beach_mean_retreat_events": (
            whole_beach_mean_retreat_events
        ),
        "minimum_visible_retreat_component_plan_area_m2": (
            MINIMUM_VISIBLE_RETREAT_PLAN_AREA_M2
        ),
        "visible_retreat_events": measured_visible_retreat_events,
        "longest_visible_connected_retreat_duration_after_runup_maximum_s": (
            longest_retreat
        ),
        "maximum_connected_retreating_surface_plan_area_m2": max(
            record["largest_connected_retreating_surface_plan_area_m2"]
            for record in history
        ),
        "corridor_wash_threshold_m3_per_s": CORRIDOR_WASH_M3_PER_S,
        "corridor_reverse_threshold_m3_per_s": CORRIDOR_REVERSE_M3_PER_S,
        "ordered_corridor_wash_then_reverse_events": measured_corridor_events,
        "ordered_wash_then_reverse_corridors": sorted(ordered_corridors),
        "maximum_largest_connected_multi_transition_columns": max(
            record["largest_connected_multi_transition_columns"]
            for record in history
        ),
        "maximum_connected_multi_transition_x_span_m": max(
            record["connected_multi_transition_x_span_m"]
            for record in history
        ),
        "maximum_connected_multi_transition_y_span_m": max(
            record["connected_multi_transition_y_span_m"]
            for record in history
        ),
        "terminal_sheet_mean_u_range_m_per_s": [
            min(record["terminal_sheet_mean_u_m_per_s"] for record in history),
            max(record["terminal_sheet_mean_u_m_per_s"] for record in history),
        ],
        "maximum_terminal_sheet_moving_fraction": max(
            record["terminal_sheet_moving_fraction"] for record in history
        ),
        "maximum_contacted_solid_cells": max(
            record["contacted_solid_cells"] for record in history
        ),
        "initial_free_water_volume_m3": initial_volume,
        "initial_porous_storage_m3": initial_storage,
        "final_free_water_volume_m3": final_vof.volume,
        "final_porous_storage_m3": final_storage,
        "relaxation_volume_source_m3": source_volume,
        "wisp_volume_removed_m3": wisp_volume,
        "infiltrated_volume_m3": infiltrated_volume,
        "drained_volume_m3": drained_volume,
        "relative_mass_residual": mass_residual / initial_volume,
        "alpha_bounds": [final_vof.alpha_min, final_vof.alpha_max],
        "solid_water_volume_m3": solid_water,
        "maximum_divergence": maximum_divergence,
        "maximum_stored_divergence": maximum_stored_divergence,
        "maximum_pressure_iterations": maximum_pressure_iterations,
        "maximum_courant": maximum_courant,
        "maximum_face_speed_m_per_s": maximum_face_speed,
        "minimum_dt_s": minimum_dt,
        "liquid_resolved_fraction": scene.solver.liquid_resolved_fraction,
        "maximum_residue_volume_m3": maximum_residue_volume,
        "minimum_capillary_limit_s": minimum_capillary_limit,
        "accepted_steps": solver.steps,
        "final_energy_j": asdict(final_energy),
        "cumulative_energy_j": {
            "boundary_work": ledger.boundary_work,
            "boundary_surface_work": ledger.boundary_surface_work,
            "capillary_exchange": ledger.capillary_exchange,
            "molecular_loss": ledger.molecular_loss,
            "sgs_loss": ledger.sgs_loss,
            "wall_loss": ledger.wall_loss,
            "porous_loss": ledger.porous_loss,
            "numerical_residual": ledger.numerical_residual,
        },
        "aw1104_acceptance": aw1104_acceptance,
        "final_checkpoint": str(checkpoint_path),
        "wall_seconds": time.perf_counter() - started,
        "surfaces": surfaces,
        "history": history,
    }
    report_path = args.output_dir / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    _check_numerical_gates(solver, args.output_dir, solver.time, maximum_divergence,
                           final_vof, solid_water, report["relative_mass_residual"])
    if not args.diagnostic and report["maximum_terminal_sheet_moving_fraction"] < 0.10:
        raise RuntimeError("terminal shoreline sheet did not move")
    if not args.diagnostic and report["reversing_corridors"] < 3:
        raise RuntimeError("fewer than three boulder corridors reversed")
    if not aw1104_acceptance["dissipative_work_nonnegative"]:
        raise RuntimeError("a dissipative mechanism injected energy")
    if args.require_aw1104 and not all(aw1104_acceptance.values()):
        failed = [
            name for name, passed in aw1104_acceptance.items() if not passed
        ]
        raise RuntimeError(f"AW-1104 acceptance failed: {', '.join(failed)}")

    print(json.dumps({
        key: report[key] for key in (
            "grid",
            "frame_count",
            "mean_shoreline_excursion_m",
            "longest_visible_connected_retreat_duration_after_runup_maximum_s",
            "incident_boundary_maximum_wave_height_to_depth_ratio",
            "ordered_wash_then_reverse_corridors",
            "aw1104_acceptance",
            "terminal_sheet_mean_u_range_m_per_s",
            "maximum_forward_gap_discharges_m3_per_s",
            "maximum_reverse_gap_discharges_m3_per_s",
            "maximum_simultaneous_boulder_wrap_structures",
            "relative_mass_residual",
            "maximum_divergence",
            "accepted_steps",
            "wall_seconds",
        )
    }, indent=2), flush=True)
    print(f"wrote {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
