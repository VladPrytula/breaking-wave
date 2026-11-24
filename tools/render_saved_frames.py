"""Render saved water-surface meshes of a production run through its presentation.

Each requested mesh (a saved PLIC boundary, nodal isosurface, or an AW-1114
composite) is rendered with the run's camera, light, solids and the saved
solid wetness of the same capture, through the production offshore join
unless ``--simulated-water-only`` is given (then the retained mesh alone, no
analytic water beyond the handoff; a comparison render, not the production
presentation). Foam shading is on by default so a composite's aeration
indicator shows; ``--no-foam`` renders the bare surface. Linear PNGs are the
judgement artefacts; tonemapped copies are written beside them.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np

from shoreloop.device import prepare_device


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, nargs="+", required=True, help="saved mesh npz files")
    parser.add_argument("--time", type=float, nargs="+", required=True,
                        help="capture time of each mesh (selects the saved wetness)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", nargs="*", default=None, help="output name per mesh")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--variant", default="cuda_ad_rgb")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--spp", type=int, default=512)
    parser.add_argument("--no-foam", action="store_true")
    parser.add_argument("--simulated-water-only", action="store_true")
    args = parser.parse_args()
    if len(args.mesh) != len(args.time) or (args.label and len(args.label) != len(args.mesh)):
        parser.error("--mesh, --time and --label must have the same length")

    prepare_device(args.device)
    from render_production_sequence import BOULDER_SUBDIVISION_LEVELS, _render_payload
    from production_render import _read_wetness, _render_bounds
    from shoreloop.closures.ripples import crossing_ripple_spectrum
    from shoreloop.physics3d.production_scene import production_scene
    from shoreloop.render.camera import tonemap
    from shoreloop.render.offshore_join import MultiValuedOffshoreSeamError
    from shoreloop.render.sky import light_preset
    from shoreloop.render.surface_mesh import WaterSurfaceMesh
    from shoreloop.render.video import write_png

    sequence = args.run / "sequence"
    saved = json.loads((sequence / "configuration.json").read_text())
    scene = production_scene(device=args.device, output_dir=str(args.out / "scene_tmp"),
                             incident=saved["scene"].get("incident", "aw1103"))
    scene = replace(scene, offshore_handoff_x=float(saved["scene"]["offshore_handoff_x"]))
    retained = scene.solver.grid
    g = saved["scene"]["solver"]["grid"]
    scale = g["nx"] // retained.nx
    grid = replace(retained, nx=scale * retained.nx, ny=scale * retained.ny, nz=scale * retained.nz)
    scene = replace(scene, solver=replace(scene.solver, grid=grid, output_dir=str(args.out / "scene_tmp")))
    if json.loads(json.dumps(asdict(scene.sunset_camera))) != saved["camera"]:
        raise ValueError("the scene's camera differs from the run's saved camera")
    camera = scene.sunset_camera.build(args.width, args.height)
    sky, sun = light_preset(saved["light"])
    bounds = _render_bounds(scene)
    spectrum = crossing_ripple_spectrum(period=scene.inlet.period)
    args.out.mkdir(parents=True, exist_ok=True)
    report_path = args.out / "render_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    for index, (mesh_path, target) in enumerate(zip(args.mesh, args.time, strict=True)):
        label = args.label[index] if args.label else mesh_path.stem
        if args.simulated_water_only:
            label += "_simonly"
        # Wetness frames are numbered from the run's first capture, not from t = 0.
        frame = int(round((target - float(saved["capture_start_s"])) / float(saved["interval_s"])))
        wetness = _read_wetness(sequence / "wetness" / f"frame_{frame:04d}.npz")
        mesh = WaterSurfaceMesh.read(mesh_path)
        started = time.perf_counter()
        try:
            linear, join, renderer, seconds = _render_payload(
                scene, mesh, wetness, bounds, camera, target, variant=args.variant, spp=args.spp,
                sky=sky, sun=sun, seed=scene.sunset_render_seed, spectrum=spectrum,
                pebble_microgeometry_enabled=bool(saved["pebble_microgeometry_enabled"]),
                simulated_water_only=args.simulated_water_only,
                ripples_enabled=False, foam_enabled=not args.no_foam)
        except MultiValuedOffshoreSeamError as error:
            report[label] = dict(render_failed=str(error), mesh=str(mesh_path), time_s=target)
            print(f"{label}: refused by the offshore join: {error}", flush=True)
            report_path.write_text(json.dumps(report, indent=1, default=str) + "\n")
            continue
        if not np.all(np.isfinite(linear)):
            raise RuntimeError(f"{label}: non-finite radiance")
        np.save(args.out / f"{label}_linear.npy", linear)
        write_png(args.out / f"{label}_linear.png", np.clip(linear, 0, 1))
        write_png(args.out / f"{label}_tonemapped.png", tonemap(linear, camera))
        report[label] = dict(
            time_s=target, wetness_frame=frame, mesh=str(mesh_path), spp=args.spp,
            resolution=[args.width, args.height], foam_shading_enabled=not args.no_foam,
            simulated_water_only=args.simulated_water_only,
            mesh_vertices=int(len(mesh.vertices)), mesh_triangles=int(len(mesh.triangles)),
            render_seconds=seconds, join=None if join is None else asdict(join),
            joined_vertices=int(len(renderer.mesh.vertices)), joined_triangles=int(len(renderer.mesh.triangles)),
            boulder_subdivision_levels=BOULDER_SUBDIVISION_LEVELS)
        print(f"{label}: rendered in {seconds:.1f}s (total {time.perf_counter() - started:.1f}s)", flush=True)
        report_path.write_text(json.dumps(report, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
