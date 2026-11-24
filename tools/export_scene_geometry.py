"""Export the static cut-cell geometry of a saved production run.

``compose_aerated_frames.py`` needs the solver's open fractions ``theta`` and
the solid-plane normals and offsets of every cut cell to classify mixture
columns and footprint columns. They are a property of the scene and the grid,
not of the saved frames, so they are rebuilt here from the run's
``sequence/configuration.json`` and written once as ``geometry.npz``.
Requires the CUDA device the production solver was built for.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from shoreloop.device import prepare_device
from shoreloop.physics3d.production_scene import build_production_solver, production_scene


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="run directory holding sequence/")
    parser.add_argument("--out", type=Path, required=True, help="output geometry.npz")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    saved = json.loads((args.run / "sequence" / "configuration.json").read_text())["scene"]
    prepare_device(args.device)
    scratch = args.out.parent / "solver_scratch"
    scene = production_scene(
        device=args.device,
        output_dir=scratch,
        surface_tension=saved["solver"]["physics"]["surface_tension"],
        incident_scale=saved["incident_scale_from_aw1103"],
        incident=saved.get("incident", "aw1103"),
        bed=saved.get("bed", "aw1104"),
    )
    solver_cfg = saved["solver"]
    scene = replace(
        scene,
        solver=replace(
            scene.solver,
            grid=replace(scene.solver.grid, **{k: solver_cfg["grid"][k] for k in ("nx", "ny", "nz")}),
            interface_reconstruction=solver_cfg["interface_reconstruction"],
            wave_source_reconstruction=solver_cfg["wave_source_reconstruction"],
            wave_source_quadrature=solver_cfg["wave_source_quadrature"],
            vof_flux_reconstruction=solver_cfg["vof_flux_reconstruction"],
        ),
    )
    # The cut-cell geometry is fixed by the grid, the solids and the water depth.
    rebuilt = json.loads(json.dumps(asdict(scene)))
    if (rebuilt["solver"]["grid"] != saved["solver"]["grid"]
            or rebuilt["geometry"] != saved["geometry"]
            or rebuilt["water_depth"] != saved["water_depth"]):
        raise SystemExit("rebuilt grid, geometry or water depth differ from the run's configuration.json")

    solver = build_production_solver(scene)
    state = solver.state
    interior = state.grid.interior_slices("cell")
    theta = state.get_geometry_interior("theta")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        theta=theta,
        open_normals=state.open_normals.numpy()[interior],
        open_offsets=state.open_offsets.numpy()[interior],
    )
    cut = int(((theta > 0) & (theta < 1)).sum())
    print(f"{args.out}: grid {theta.shape}, solid cells {int((theta == 0).sum())}, cut cells {cut}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
