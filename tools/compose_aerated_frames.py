"""Build the aerated-mixture composites of saved production frames (AW-1114).

For each requested time the frame's PLIC boundary mesh and nodal isosurface are
read, the mixture columns are classified from the PLIC mesh (the AW-1113 mesh
rule) or from the frame's kernel export when one exists, the aeration
indicator is carried forward from the preceding captures (warm-up frames are
classified first so a late start sees the same memory as a full pass), and
both named composites are written beside a report of everything replaced,
kept, painted and the represented water per band. The persistence time
defaults to the run's foam closure decay time. Rendering is a separate step;
nothing in the run directory is modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from shoreloop.physics3d.resolvedness import MIXTURE_STACKED_PARTIAL, InterfaceResolvedness
from shoreloop.render.aerated_mixture import (
    AERATION_SWITCH, AERATION_WEIGHT, BOULDER_FILM_FRACTION, COMPOSITES, MIXTURE_STACKED,
    RESIDUE_FRACTION, SWASH_COLUMNS, boulder_film_cells, cell_water_share, classify_frame,
    compose_aerated_surface, persist_indicator, residue_triangles, _cell_flag_on_triangles,
)
from shoreloop.render.mixture_mask import PlanGrid, excluded_columns
from shoreloop.scene.geometry3d import RoundedBoulder
from shoreloop.render.surface_mesh import WaterSurfaceMesh


def _label(time_s: float) -> str:
    return f"{time_s:05.2f}".replace(".", "_")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="run directory holding sequence/")
    parser.add_argument("--times", type=float, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True,
                        help="npz with theta, open_normals, open_offsets of the run's grid")
    parser.add_argument("--composite", choices=COMPOSITES + ("both",), default="both")
    parser.add_argument("--aeration-weight", type=float, default=AERATION_WEIGHT)
    parser.add_argument("--no-foam-gate", action="store_true")
    parser.add_argument("--keep-residue", action="store_true", help="draw the facets of residue cells")
    parser.add_argument("--keep-footprint", action="store_true",
                        help="keep PLIC facets in boulder-footprint columns")
    parser.add_argument("--keep-boulder-films", action="store_true",
                        help="draw the facets of sub-half-cell films on the boulders")
    parser.add_argument("--swash-columns", type=int, default=SWASH_COLUMNS)
    parser.add_argument("--aeration-switch", type=float, default=AERATION_SWITCH,
                        help="indicator level above which a column is drawn through the nodal isosurface")
    parser.add_argument("--persistence", type=float, default=None,
                        help="aeration persistence time [s]; default: the run's foam decay time")
    parser.add_argument("--warmup-frames", type=int, default=20,
                        help="captures classified before the first requested time to seed the memory")
    args = parser.parse_args()

    sequence = args.run / "sequence"
    configuration = json.loads((sequence / "configuration.json").read_text())
    grid = PlanGrid.from_configuration(configuration)
    interval = float(configuration["interval_s"])
    persistence = float(configuration["scene"]["foam"]["decay_time"]) if args.persistence is None else args.persistence
    with np.load(args.geometry) as saved:
        theta = saved["theta"].astype(np.float32)
        open_normals, open_offsets = saved["open_normals"], saved["open_offsets"]
    args.out.mkdir(parents=True, exist_ok=True)
    composites = COMPOSITES if args.composite == "both" else (args.composite,)
    requested = sorted(args.times)
    first = int(round(requested[0] / interval))
    last = int(round(requested[-1] / interval))
    # Every capture from the warm-up to the last requested time is classified so
    # the memory is the same as in a full pass; composites are written for the
    # requested times only.
    start = max(first - args.warmup_frames, 0) if persistence > 0 else first
    visited = [frame * interval for frame in range(start, last + 1)]
    wanted = {int(round(time_s / interval)) for time_s in requested}
    indicator = None
    previous_aerated = None
    warmup = 0

    footprint = np.zeros(grid.shape, bool) if args.keep_footprint else excluded_columns(theta)
    boulders = [] if args.keep_boulder_films else [
        RoundedBoulder(**b) for b in configuration["scene"]["geometry"]["boulders"]]

    def classify(time_s):
        label = _label(time_s)
        plic = WaterSurfaceMesh.read(sequence / "surfaces" / f"surface_{label}s_raw.npz")
        # The mesh recovery supplies the facet cells' water for the residue
        # rule; the kernel export, when a frame has one, supplies the mask.
        result = classify_frame(plic, grid, theta, open_normals, open_offsets)
        export = sequence / "surfaces" / f"resolvedness_{label}s.npz"
        if export.exists():
            return plic, result, InterfaceResolvedness.read(export).stacked_unresolved >= MIXTURE_STACKED_PARTIAL, str(export)
        return plic, result, result.mixture_columns(MIXTURE_STACKED), "mesh rule (mixture_mask.mesh_resolvedness)"

    for time_s in visited:
        label = _label(time_s)
        started = time.perf_counter()
        plic, result, mixture, source = classify(time_s)
        classification_seconds = time.perf_counter() - started
        indicator = persist_indicator(indicator, mixture, interval, persistence)
        aerated = indicator >= args.aeration_switch
        flips = int((aerated != previous_aerated).sum()) if previous_aerated is not None else None
        previous_aerated = aerated
        if int(round(time_s / interval)) not in wanted:
            warmup += 1
            continue
        nodal = WaterSurfaceMesh.read(sequence / "surfaces" / f"surface_{label}s_nodal.npz")
        np.savez_compressed(args.out / f"indicator_{label}s.npz", mixture=mixture, indicator=indicator)
        residue = None if args.keep_residue else residue_triangles(plic, grid, result, theta)
        share = cell_water_share(result, theta)
        cell_volume = grid.dx * grid.dy * grid.dz
        residue_water = float(share[share < RESIDUE_FRACTION].sum() * cell_volume) if not args.keep_residue else 0.0
        in_footprint = footprint[result.cells[:, 0], result.cells[:, 1]]
        footprint_water = float(share[in_footprint].sum() * cell_volume)
        film_cells = boulder_film_cells(result, grid, theta, boulders)
        boulder_film = _cell_flag_on_triangles(plic, grid, result, film_cells) if boulders else None
        boulder_film_water = float(share[film_cells & (share >= RESIDUE_FRACTION)].sum() * cell_volume)
        for composite in composites:
            started = time.perf_counter()
            mesh, report = compose_aerated_surface(
                plic, nodal, mixture, grid, composite=composite, indicator=indicator,
                residue=residue, footprint=footprint, boulder_film=boulder_film,
                aeration_weight=args.aeration_weight, aeration_switch=args.aeration_switch,
                foam_gate=not args.no_foam_gate,
                swash_columns=args.swash_columns, theta=theta)
            path = mesh.write(args.out / f"surface_{label}s_{composite}.npz")
            report.update(time_s=time_s, mesh_path=str(path), classification=source,
                          residue_fraction=RESIDUE_FRACTION if not args.keep_residue else None,
                          residue_water_not_drawn_m3=residue_water,
                          footprint_facet_cell_water_m3=footprint_water,
                          boulder_film_fraction=BOULDER_FILM_FRACTION if boulders else None,
                          boulder_film_water_not_drawn_m3=boulder_film_water,
                          classification_seconds=classification_seconds,
                          compose_seconds=time.perf_counter() - started,
                          persistence_s=persistence, captures_classified_before=warmup,
                          aerated_column_flips_since_previous_capture=flips)
            (args.out / f"composite_{label}s_{composite}.json").write_text(json.dumps(report, indent=2) + "\n")
            errors = report["represented_volume_error_vs_plic"]
            print(f"t={time_s:.2f}s {composite}: mixture {report['mixture_columns']} cols, "
                  f"aerated {report['aerated_columns']} cols, flips {flips}, "
                  f"residue {report['residue_triangles_not_drawn']} tris/{residue_water * 1e3:.2f} L, "
                  f"footprint {report['footprint_triangles_not_drawn']} tris/{footprint_water * 1e3:.2f} L, "
                  f"boulder films {report['boulder_film_triangles_not_drawn']} tris/{boulder_film_water * 1e3:.2f} L, "
                  f"dropped {report['plic_area_dropped_m2']:.4f} m2, films {report['plic_film_area_kept_m2']:.4f} m2, "
                  f"aerated {report['aerated_area_m2']:.4f} m2, overlap {report['overlap_area_m2']:.4f} m2, "
                  f"volume error " + ", ".join(f"{b} {e * 100:+.3f}%" for b, e in errors.items())
                  + " | outside footprint " + ", ".join(
                      f"{b} {e * 100:+.3f}%" for b, e in report["represented_volume_error_vs_plic_outside_footprint"].items()),
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
