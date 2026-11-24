"""AW-1125: break a pushed wave in a flat box.

A quasi-two-dimensional tank with a uniform bottom and the production numerics
(liquid-mimetic pressure, height-function PLIC, swept-prism transport, the
resolved-liquid ownership rule, molecular and Smagorinsky viscosity, the energy
ledger), driven from the x-low side three named ways:

- ``front``: one periodic component whose crest-to-trough height at the zone
  exit lies over the shallow-water depth limit, so the resolved steepening over
  the flat bed breaks it (the depth limit ``H / d = 0.78`` is the reference);
- ``focus``: several harmonics of one loop period with phases chosen so every
  crest passes the focus point at the focus time (``phi_i = omega_i t_f -
  k_i x_f``), the dispersive focusing flat flumes use to make one breaker;
- ``dam``: no inlet, a slab of water released on the x-low side (a dam break),
  giving a bore whose speed the hydraulic-jump relation
  ``U = sqrt(g h2 (h1 + h2) / (2 h1))`` predicts.

Every capture records the alongshore-mean surface elevation, the crest, the
front-face slope, the overturning cells (interface normals pointing down), the
fastest face, the liquid volume and the energy ledger; the run ends with a
summary against the limits. Outputs: ``progress.json``, ``surface.npz`` (eta(x,t)),
``sections.png``, ``spacetime.png``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from shoreloop.config import BC, BC3DConfig, Grid3DConfig, Physics3DConfig, Solver3DConfig
from shoreloop.device import prepare_device
from shoreloop.physics3d.energy import EnergyLedger
from shoreloop.physics3d.interface import estimate_interface_normals
from shoreloop.physics3d.pcg import PCGConfig
from shoreloop.physics3d.plic import reconstruct_plic
from shoreloop.physics3d.projection import ProjectionConfig
from shoreloop.physics3d.relaxation import WaveRelaxationZone
from shoreloop.physics3d.solver import Solver3D
from shoreloop.physics3d.timestep import TimeStepConfig
from shoreloop.physics3d.viscosity import ViscosityConfig
from shoreloop.physics3d.vof import initialize_box, measure_vof
from shoreloop.physics3d.waves import LinearWaveInlet, PeriodicWaveComponent
from shoreloop.physics3d.wet_dry import WispRemover

DRIVES = ("front", "focus", "dam")
DEPTH = 0.22                 # m, the production scene's still water
DEPTH_LIMIT = 0.78           # H / d of a solitary wave on a flat bed
TANK_LENGTH = 8.0            # m, long enough that the far wall's reflection stays out of the window
TANK_HEIGHT = 0.6            # m
PRIMARY_PERIOD = 2.0         # s, the reference's front period
FOCUS_LOOP_PERIOD = 6.0      # s, loop carrying the focused group as its 3rd-6th harmonics
FOCUS_MODES = (3, 4, 5, 6)   # 2.0, 1.5, 1.2, 1.0 s
GENERATION_RAMP = 0.40       # m, the production zone's ramp
SLAB_LENGTH = 1.0            # m, the dam-break slab
SLAB_HEIGHT = 0.15           # m above still water
JUDGE_MARGIN = 0.3           # m past the drive before the wave is judged (the zone's edge seam)
WALL_MARGIN = 0.5            # m short of the far wall
WALL_ARRIVAL = 0.03          # m of elevation within the wall margin that ends the judged window
BREAKING_HEIGHT = 0.3        # H / d a capture needs before its overturning counts as the breaker


def build(drive: str, *, spacing: float, amplitude: float, focus_x: float, focus_time: float,
          device: str, output_dir: Path) -> tuple[Solver3D, dict]:
    nx = round(TANK_LENGTH / spacing)
    nz = round(TANK_HEIGHT / (0.8 * spacing))          # the production dz / dx ratio
    grid = Grid3DConfig(nx=nx, ny=2, nz=nz, Lx=TANK_LENGTH, Ly=2 * spacing, Lz=TANK_HEIGHT)
    cfg = Solver3DConfig(
        grid=grid, physics=Physics3DConfig(surface_tension=0.0),
        bc=BC3DConfig(xlo=BC.WALL, xhi=BC.WALL, ylo=BC.PERIODIC, yhi=BC.PERIODIC, zlo=BC.WALL, zhi=BC.WALL),
        device=device, output_dir=str(output_dir / "solver"),
        pressure_discretization="liquid_mimetic", liquid_resolved_fraction=2.0e-2,
        momentum_speed_bound="local", momentum_face_mass="flux_consistent",
        interface_reconstruction="height_function", wave_source_reconstruction="plic_rays",
        wave_source_quadrature=8, vof_flux_reconstruction="swept_prism",
        velocity_extension_order=0, liquid_extension_inertia=1.0,
    )
    solver = Solver3D(
        cfg,
        ProjectionConfig(pcg=PCGConfig(relative_tolerance=2.0e-7, max_iterations=10000, preconditioner="jacobi"),
                         rms_divergence_tolerance=6.0e-4),
        TimeStepConfig(dt_max=0.005, advective_cfl=0.5, gravity_cfl=0.5, viscous_cfl=0.25, capillary_cfl=0.25,
                       viscous_limit_enabled=False, kinematic_viscosity=1.0e-6),
        viscosity_settings=ViscosityConfig(molecular_viscosity=1.0e-6, smagorinsky_cs=0.12, maximum_sgs_viscosity=0.02),
    )
    initialize_box(solver.state, (grid.x0, grid.y0, grid.z0), (grid.x0 + grid.Lx, grid.y0 + grid.Ly, grid.z0 + DEPTH))
    record: dict = dict(drive=drive, spacing=spacing, depth=DEPTH, grid=asdict(grid))
    if drive == "dam":
        # The slab stands on the still water: the union of the two sampled boxes.
        still = solver.state.get_interior("alpha").copy()
        initialize_box(solver.state, (grid.x0, grid.y0, grid.z0 + DEPTH - spacing),
                       (grid.x0 + SLAB_LENGTH, grid.y0 + grid.Ly, grid.z0 + DEPTH + SLAB_HEIGHT))
        solver.state.set_interior("alpha", np.maximum(still, solver.state.get_interior("alpha")))
        record.update(slab_length=SLAB_LENGTH, slab_height=SLAB_HEIGHT,
                      jump_relation="U = sqrt(g h2 (h1 + h2) / (2 h1))")
    else:
        shared = dict(depth=DEPTH, alongshore_length=grid.Ly, gravity=cfg.physics.g)
        if drive == "front":
            components = (PeriodicWaveComponent(period=PRIMARY_PERIOD, amplitude=amplitude, stokes_order=2, **shared),)
        else:
            components = []
            for mode in FOCUS_MODES:
                c = PeriodicWaveComponent(period=FOCUS_LOOP_PERIOD, temporal_mode=mode, amplitude=amplitude, **shared)
                # Every crest passes (focus_x, focus_time): k x_f - omega t_f + phi = 0.
                phi = c.omega * focus_time - c.k * focus_x
                components.append(PeriodicWaveComponent(period=FOCUS_LOOP_PERIOD, temporal_mode=mode,
                                                        amplitude=amplitude, phase_offset=phi, **shared))
            components = tuple(components)
        inlet = LinearWaveInlet(components)
        width = max(c.wavelength for c in components)
        solver.set_wave_relaxation(WaveRelaxationZone(cfg, inlet, width, timescale=0.02 * inlet.period,
                                                      generation_ramp=GENERATION_RAMP))
        record.update(
            components=[dict(period=c.period / c.temporal_mode, amplitude=c.amplitude, wavelength=c.wavelength,
                             phase_offset=c.phase_offset, stokes_order=c.stokes_order) for c in components],
            relaxation_width=width, zone_exit_x=grid.x0 + width,
            linear_height_sum=2.0 * sum(c.amplitude for c in components),
            focus_x=focus_x if drive == "focus" else None, focus_time=focus_time if drive == "focus" else None)
    solver.set_wisp_remover(WispRemover(cfg))
    solver.set_energy_ledger(EnergyLedger(cfg))
    return solver, record


def surface_profile(solver: Solver3D) -> np.ndarray:
    """Alongshore-mean free-surface elevation above still water from the column water."""
    g = solver.cfg.grid
    alpha = solver.state.get_interior("alpha")
    return alpha.sum(axis=2).mean(axis=1) * g.dz - DEPTH


def overturning_cells(solver: Solver3D, first_cell: int, last_cell: int) -> int:
    """Interface cells in the judged stretch whose reconstructed normal points downward: a lip's overhang."""
    normals_result = estimate_interface_normals(solver.state)
    reconstruction = reconstruct_plic(solver.state, normals_result.normals)
    sl = solver.state.grid.interior_slices("cell")
    normals = reconstruction.normals.numpy()[sl][first_cell:last_cell]
    alpha = solver.state.get_interior("alpha")[first_cell:last_cell]
    mixed = (alpha > 0.02) & (alpha < 0.98)
    return int(np.sum(mixed & (normals[..., 2] < -0.3)))


def fastest_face(solver: Solver3D) -> float:
    s = solver.state
    return float(max(np.abs(s.get_interior(name)).max() for name in ("u", "v", "w")))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--drive", choices=DRIVES, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--spacing", type=float, default=0.025, help="dx in m (0.025 development, 0.0125 production)")
    parser.add_argument("--amplitude", type=float, default=None,
                        help="component amplitude in m (front: default 0.055, just under the depth limit at the zone exit; focus: default 0.04 per harmonic)")
    parser.add_argument("--focus-x", type=float, default=5.0)
    parser.add_argument("--focus-time", type=float, default=6.5)
    parser.add_argument("--end-time", type=float, default=None)
    parser.add_argument("--interval", type=float, default=0.04)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    amplitude = args.amplitude if args.amplitude is not None else (0.055 if args.drive == "front" else 0.04)
    end_time = args.end_time if args.end_time is not None else {"front": 4.6, "focus": 7.5, "dam": 3.0}[args.drive]
    args.out.mkdir(parents=True, exist_ok=True)
    prepare_device(args.device)
    solver, record = build(args.drive, spacing=args.spacing, amplitude=amplitude, focus_x=args.focus_x,
                           focus_time=args.focus_time, device=args.device, output_dir=args.out)
    g = solver.cfg.grid
    x = g.x0 + (np.arange(g.nx) + 0.5) * g.dx
    ledger = solver.energy_ledger
    initial_energy = ledger.measure(solver.state, liquid_operator=solver.liquid_operator())
    initial_volume = measure_vof(solver.state).volume
    record.update(end_time=end_time, interval=args.interval, initial_volume_m3=initial_volume,
                  initial_energy_j=initial_energy.total)
    if args.drive == "dam":
        # The slab's excess over still water of the same volume: rho g A (h^2 - h0^2) / 2 per slab area.
        area = SLAB_LENGTH * g.Ly
        level = DEPTH + SLAB_HEIGHT * SLAB_LENGTH / g.Lx
        record["still_energy_j"] = initial_energy.total - 0.5 * 1000.0 * solver.cfg.physics.g * (
            area * ((DEPTH + SLAB_HEIGHT) ** 2 - level ** 2) + (g.Lx - SLAB_LENGTH) * g.Ly * (DEPTH ** 2 - level ** 2))
    # Judge the wave beyond the drive (past the relaxation zone's edge seam, or past the slab) and
    # short of the far wall, whose run-up is not a breaker.
    first_cell = int(np.searchsorted(x, record.get("zone_exit_x", SLAB_LENGTH) + JUDGE_MARGIN))
    last_cell = int(np.searchsorted(x, TANK_LENGTH - WALL_MARGIN))
    record["judged_from_x"] = float(x[first_cell]); record["judged_to_x"] = float(x[last_cell - 1])
    captures, profiles, times = [], [], []
    started = time.perf_counter()
    steps = 0
    minimum_dt = math.inf
    source_volume = wisp_volume = 0.0
    targets = np.arange(0.0, end_time + 1e-9, args.interval)
    for target in targets:
        while solver.time < target - 1.0e-12:
            timestep = solver.compute_dt()
            remaining = target - solver.time
            substeps = max(1, int(np.ceil(remaining / timestep.dt - 1.0e-9)))
            dt = remaining / substeps
            solver.step(dt, timestep=timestep)
            steps += 1
            minimum_dt = min(minimum_dt, dt)
            if solver.last_relaxation is not None:
                source_volume += solver.last_relaxation.volume_source
            wisp_volume += solver.last_wisp_removal.removed_volume
        eta = surface_profile(solver)
        crest = first_cell + int(np.argmax(eta[first_cell:last_cell]))
        slope = np.gradient(eta, g.dx)
        front = slope[crest:last_cell] if crest < last_cell - 1 else slope[crest:crest + 1]
        wall_reached = bool((eta[last_cell:] > WALL_ARRIVAL).any())
        energy = ledger.measure(solver.state, liquid_operator=solver.liquid_operator())
        volume = measure_vof(solver.state).volume
        losses = ledger.molecular_loss + ledger.sgs_loss + ledger.wall_loss + ledger.porous_loss
        capture = dict(
            time_s=round(float(solver.time), 6), steps=steps, crest_x=float(x[crest]), crest_m=float(eta[crest]),
            trough_m=float(eta[first_cell:crest + 1].min()),
            front_face_slope=float(-front.min()), overturning_cells=overturning_cells(solver, first_cell, last_cell),
            wall_reached=wall_reached,
            fastest_face_m_per_s=fastest_face(solver), minimum_dt_s=minimum_dt,
            volume_m3=volume, volume_residual_m3=volume - initial_volume - source_volume + wisp_volume,
            energy_total_j=energy.total, boundary_work_j=ledger.boundary_work, molecular_loss_j=ledger.molecular_loss,
            sgs_loss_j=ledger.sgs_loss,
            numerical_residual_j=initial_energy.total + ledger.boundary_work - losses - energy.total,
            wall_s=time.perf_counter() - started,
        )
        captures.append(capture)
        profiles.append(eta.astype(np.float32))
        times.append(float(solver.time))
        print("t=%.2fs crest %.3f m at x=%.2f (H %.3f, H/d %.2f) face slope %.2f overturning %d fastest %.2f m/s "
              "dt_min %.2f ms vol residual %.2e m3 energy %.2f J (in %.2f, visc %.2f, numerical %.2f) wall %.0fs" % (
                  capture["time_s"], capture["crest_m"], capture["crest_x"], capture["crest_m"] - capture["trough_m"],
                  (capture["crest_m"] - capture["trough_m"]) / DEPTH, capture["front_face_slope"],
                  capture["overturning_cells"], capture["fastest_face_m_per_s"], 1e3 * minimum_dt,
                  capture["volume_residual_m3"], energy.total, ledger.boundary_work,
                  ledger.molecular_loss + ledger.sgs_loss, capture["numerical_residual_j"], capture["wall_s"]), flush=True)
        json.dump(dict(record, captures=captures), open(args.out / "progress.json", "w"), indent=1)
        np.savez_compressed(args.out / "surface.npz", x=x, times=np.array(times), eta=np.array(profiles))
    summary = summarise(args.drive, record, captures, np.array(profiles), np.array(times), x)
    record["summary"] = summary
    json.dump(dict(record, captures=captures), open(args.out / "progress.json", "w"), indent=1)
    plot(args.out, x, np.array(times), np.array(profiles), captures, summary)
    print(json.dumps(summary, indent=1), flush=True)
    return 0


def summarise(drive: str, record: dict, captures: list[dict], eta: np.ndarray, times: np.ndarray, x: np.ndarray) -> dict:
    # The reflection from the far wall is not part of the experiment: judge before the first arrival.
    reached = [c["time_s"] for c in captures if c.get("wall_reached")]
    window_end = reached[0] if reached else math.inf
    captures = [c for c in captures if c["time_s"] < window_end]
    # A breaker is a tall front: the zone's edge seam and the first judged cell can show
    # downward normals or a steep slope on a near-flat surface.
    tall = [c for c in captures if (c["crest_m"] - c["trough_m"]) / DEPTH >= BREAKING_HEIGHT
            and c["crest_x"] > record["judged_from_x"] + 0.1]
    over = [c for c in tall if c["overturning_cells"] > 0]
    steep = [c for c in tall if c["front_face_slope"] > 1.0]
    first_break = over[0] if over else (steep[0] if steep else None)
    g = 9.81
    summary = dict(
        judged_until_s=window_end,
        first_overturning=None if not over else dict(time_s=over[0]["time_s"], crest_x=over[0]["crest_x"],
                                                      height_over_depth=(over[0]["crest_m"] - over[0]["trough_m"]) / DEPTH),
        first_vertical_face=None if not steep else dict(time_s=steep[0]["time_s"], crest_x=steep[0]["crest_x"],
                                                         height_over_depth=(steep[0]["crest_m"] - steep[0]["trough_m"]) / DEPTH),
        maximum_height_over_depth=max((c["crest_m"] - c["trough_m"]) / DEPTH for c in captures),
        maximum_overturning_cells=max(c["overturning_cells"] for c in captures),
        fastest_face_m_per_s=max(c["fastest_face_m_per_s"] for c in captures),
        minimum_dt_s=min(c["minimum_dt_s"] for c in captures),
        volume_residual_m3=captures[-1]["volume_residual_m3"],
        energy=dict(boundary_work_j=captures[-1]["boundary_work_j"], viscous_loss_j=captures[-1]["molecular_loss_j"] + captures[-1]["sgs_loss_j"],
                    numerical_residual_j=captures[-1]["numerical_residual_j"],
                    # Share of the energy that entered: the boundary work for the driven tanks, the slab's
                    # excess potential energy for the dam break (its boundary work is zero).
                    numerical_share=captures[-1]["numerical_residual_j"] / max(
                        abs(captures[-1]["boundary_work_j"]), record["initial_energy_j"] - record.get("still_energy_j", record["initial_energy_j"]), 1e-12)),
    )
    if drive == "front":
        summary["depth_limit"] = DEPTH_LIMIT
        summary["breaks_within_depth_limit_band"] = (
            first_break is not None and 0.6 <= (first_break["crest_m"] - first_break["trough_m"]) / DEPTH <= 1.0)
    elif drive == "focus":
        summary["predicted_focus_x"] = record["focus_x"]
        summary["predicted_focus_time"] = record["focus_time"]
        longest = max(c["wavelength"] for c in record["components"])
        summary["breaks_within_one_wavelength_of_focus"] = (
            first_break is not None and abs(first_break["crest_x"] - record["focus_x"]) <= longest)
    else:
        # Bore front: the shoreward-most x where the surface stands more than 1 cm above still water,
        # tracked over the captures after the slab has collapsed; U from its slope in time.
        fronts = []
        for t, profile in zip(times, eta):
            above = np.flatnonzero(profile > 0.01)
            fronts.append(x[above].max() if len(above) else np.nan)
        fronts = np.array(fronts)
        window = (times > 0.5) & np.isfinite(fronts)
        if window.sum() > 3:
            speed = float(np.polyfit(times[window], fronts[window], 1)[0])
            # Depth behind the front: the mean elevation 0.3-0.6 m behind it at the window's midpoint.
            mid = int(np.flatnonzero(window)[len(np.flatnonzero(window)) // 2])
            behind = (x < fronts[mid] - 0.3) & (x > fronts[mid] - 0.6)
            h2 = DEPTH + float(eta[mid][behind].mean()) if behind.any() else float("nan")
            predicted = math.sqrt(g * h2 * (DEPTH + h2) / (2.0 * DEPTH))
            summary.update(bore_speed_m_per_s=speed, bore_depth_behind_m=h2, jump_relation_speed_m_per_s=predicted,
                           bore_speed_within_10_percent=abs(speed - predicted) <= 0.1 * predicted)
    return summary


def plot(out: Path, x: np.ndarray, times: np.ndarray, eta: np.ndarray, captures: list[dict], summary: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 4))
    picks = np.linspace(0, len(times) - 1, 9).astype(int)
    for i in picks:
        ax.plot(x, eta[i], lw=0.8, label="%.2f s" % times[i])
    ax.axhline(0, color="gray", lw=0.4); ax.set_xlabel("x [m]"); ax.set_ylabel("eta [m]"); ax.legend(fontsize=7, ncol=3)
    fig.savefig(out / "sections.png", dpi=90); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(eta, origin="lower", aspect="auto", extent=(x[0], x[-1], times[0], times[-1]), cmap="RdBu_r",
                   vmin=-0.1, vmax=0.2)
    ax.set_xlabel("x [m]"); ax.set_ylabel("t [s]"); fig.colorbar(im, ax=ax, label="eta [m]")
    over = [(c["crest_x"], c["time_s"]) for c in captures if c["overturning_cells"] > 0]
    if over:
        ax.scatter([o[0] for o in over], [o[1] for o in over], s=6, c="k", label="overturning")
        ax.legend(fontsize=8)
    fig.savefig(out / "spacetime.png", dpi=90); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
