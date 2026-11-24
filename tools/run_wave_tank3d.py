"""Run local 3-D solver gates and field-diagnostic experiments."""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np

from shoreloop.config import (BC, BC3DConfig, SOLVER3D_PRESET_NAMES,
                              Solver3DConfig, solver3d_preset)
from shoreloop.physics3d import (MomentumAdvector, Solver3D,
                                 compute_face_liquid_mass, initialize_box,
                                 initialize_sphere, measure_vof,
                                 render_diagnostic_slices,
                                 run_boulder_cluster_gate,
                                 run_linear_wave_gate,
                                 run_natural_breaker_gate,
                                 run_one_boulder_wash_gate,
                                 run_plane_beach_gate,
                                 run_shoaling_wave_gate)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or configure a local 3-D MAC/VOF experiment.",
    )
    parser.add_argument(
        "--preset",
        choices=SOLVER3D_PRESET_NAMES,
        default="tiny",
        help="provisional resolution tier (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        help="Warp CUDA device override, for example cuda:0",
    )
    parser.add_argument(
        "--output-dir",
        help="output-directory override",
    )
    parser.add_argument(
        "--demo",
        choices=("ready", "hydrostatic", "sphere", "linear-wave",
                 "plane-beach", "shoaling-wave", "natural-breaker",
                 "one-boulder-wash", "boulder-cluster"),
        default="ready",
        help="optional runnable field diagnostic (default: %(default)s)",
    )
    parser.add_argument(
        "--steps", type=int, default=1,
        help="accepted steps for a runnable demo (default: %(default)s)",
    )
    parser.add_argument(
        "--refine", action="store_true",
        help="use the x-z refinement for wave integration gates",
    )
    parser.add_argument(
        "--surface-meshes", action="store_true",
        help="write constrained meshes during the natural-breaker event",
    )
    return parser


def configured_experiment(args: argparse.Namespace) -> Solver3DConfig:
    """Build the immutable configuration selected on the command line."""
    cfg = solver3d_preset(args.preset)
    if args.device is not None:
        cfg = replace(cfg, device=args.device)
    if args.output_dir is not None:
        cfg = replace(cfg, output_dir=args.output_dir)
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    cfg = configured_experiment(args)
    if args.demo == "linear-wave":
        metrics = run_linear_wave_gate(
            refinement=2 if args.refine else 1,
            device=cfg.device,
            output_dir=cfg.output_dir,
        )
        report = f"{cfg.output_dir}/linear_wave_metrics.json"
        metrics.write(report)
        print(f"cells per wavelength: {metrics.cells_per_wavelength:.1f}")
        print(f"incident amplitude: {metrics.incident_amplitude:.6f} m "
              f"(error {metrics.amplitude_error:+.1%})")
        print(f"phase speed: {metrics.measured_phase_speed:.4f} m/s "
              f"(error {metrics.phase_speed_error:+.1%})")
        print(f"reflection: {metrics.reflection_coefficient:.1%}")
        print(f"volume conservation residual: "
              f"{metrics.volume_conservation_residual:.3e} m^3")
        print(f"diagnostic: {report}")
        return 0
    if args.demo == "plane-beach":
        metrics = run_plane_beach_gate(
            device=cfg.device, output_dir=cfg.output_dir,
        )
        report = f"{cfg.output_dir}/plane_beach_metrics.json"
        metrics.write(report)
        print(f"shoreline advance: {metrics.shoreline_advance:.3f} m")
        print(f"shoreline recession: {metrics.shoreline_recession:.3f} m")
        print(f"maximum run-up: {metrics.maximum_runup:.3f} m")
        print(f"final dry beach: {metrics.final_dry_beach_fraction:.1%}")
        print(f"wisp volume removed: {metrics.wisp_volume_removed:.3e} m^3")
        print(f"volume conservation residual: "
              f"{metrics.volume_conservation_residual:.3e} m^3")
        print(f"diagnostic: {report}")
        return 0
    if args.demo == "shoaling-wave":
        metrics = run_shoaling_wave_gate(
            refinement=2 if args.refine else 1,
            device=cfg.device,
            output_dir=cfg.output_dir,
        )
        report = f"{cfg.output_dir}/shoaling_wave_metrics.json"
        metrics.write(report)
        print(f"incident crest: {metrics.incident_crest:.3f} m")
        print(f"shoaling crest: {metrics.maximum_shoaling_crest:.3f} m "
              f"({metrics.crest_amplification:.2f}x)")
        print(f"maximum slope: {metrics.maximum_shoaling_slope:.2f} "
              f"({metrics.slope_amplification:.2f}x offshore)")
        print(f"steepest point: x={metrics.steepest_position:.3f} m at "
              f"t={metrics.steepest_time:.3f} s")
        print(f"measured celerity: {metrics.measured_celerity:.3f} m/s")
        print(f"relative volume residual: "
              f"{metrics.relative_volume_residual:.3%}")
        print(f"diagnostic: {report}")
        return 0
    if args.demo == "natural-breaker":
        metrics = run_natural_breaker_gate(
            refinement=2 if args.refine else 1,
            device=cfg.device,
            output_dir=cfg.output_dir,
            write_frames=True,
            write_surface_meshes=args.surface_meshes,
        )
        report = f"{cfg.output_dir}/natural_breaker_metrics.json"
        metrics.write(report)
        print(f"topology window: {metrics.first_topology_time} to "
              f"{metrics.last_topology_time} s")
        print(f"consecutive topology frames: "
              f"{metrics.maximum_consecutive_topology_frames}")
        print(f"downward interface cells: "
              f"{metrics.maximum_downward_interface_cells}")
        print(f"multi-crossing columns: "
              f"{metrics.maximum_multi_transition_columns}")
        print(f"breaker: x={metrics.breaker_position} m at "
              f"t={metrics.breaker_time} s")
        print(f"shoreline advance/recession: "
              f"{metrics.shoreline_advance:.3f} / "
              f"{metrics.shoreline_recession:.3f} m")
        print(f"backwash onset: {metrics.backwash_start_time} s "
              f"({metrics.minimum_post_wash_swash_velocity:.3f} m/s)")
        print(f"relative volume residual: "
              f"{metrics.relative_volume_residual:.3%}")
        print(f"diagnostic: {report}")
        return 0
    if args.demo == "one-boulder-wash":
        metrics = run_one_boulder_wash_gate(
            device=cfg.device,
            output_dir=cfg.output_dir,
            write_frames=True,
        )
        report = f"{cfg.output_dir}/one_boulder_wash_metrics.json"
        metrics.write(report)
        print(f"impact head: {metrics.maximum_surface_head_difference:.3f} m")
        print(f"side-wrap speed: {metrics.maximum_side_velocity:.3f} m/s")
        print(f"surface contact: "
              f"{metrics.maximum_boulder_contact_fraction:.1%}")
        print(f"recirculation: {metrics.maximum_recirculation_length:.3f} m "
              f"at {metrics.maximum_reverse_speed:.3f} m/s")
        print(f"backwash: {metrics.backwash_start_time} s "
              f"at {metrics.maximum_backwash_speed:.3f} m/s")
        print(f"relative volume residual: "
              f"{metrics.relative_volume_residual:.3%}")
        print(f"diagnostic: {report}")
        return 0
    if args.demo == "boulder-cluster":
        metrics = run_boulder_cluster_gate(
            device=cfg.device,
            output_dir=cfg.output_dir,
            write_frames=True,
        )
        report = f"{cfg.output_dir}/boulder_cluster_metrics.json"
        metrics.write(report)
        print(f"gap areas: {metrics.gap_open_areas} m²")
        print(f"forward gap discharges: "
              f"{metrics.maximum_forward_gap_discharges} m³/s")
        print(f"reverse gap discharges: "
              f"{metrics.maximum_reverse_gap_discharges} m³/s")
        print(f"peak gap speeds: {metrics.maximum_gap_speeds} m/s")
        print(f"interacting wake cells: "
              f"{metrics.maximum_interacting_wake_cells}")
        print(f"recirculation: {metrics.maximum_recirculation_length:.3f} m")
        print(f"shelter velocity ratio: "
              f"{metrics.minimum_shelter_velocity_ratio:.2f}")
        print(f"relative volume residual: "
              f"{metrics.relative_volume_residual:.3%}")
        print(f"diagnostic: {report}")
        return 0
    if args.demo == "sphere":
        periodic = BC3DConfig(
            xlo=BC.PERIODIC, xhi=BC.PERIODIC,
            ylo=BC.PERIODIC, yhi=BC.PERIODIC,
            zlo=BC.PERIODIC, zhi=BC.PERIODIC,
        )
        cfg = replace(cfg, bc=periodic,
                      physics=replace(cfg.physics, g=0.0))
    g = cfg.grid
    solver = Solver3D(cfg)
    state = solver.state
    timestep = solver.compute_dt()
    print(f"wave-tank3d preset: {args.preset}")
    print(f"grid: {g.nx} x {g.ny} x {g.nz} ({g.cell_count:,} cells)")
    print(f"extent: {g.Lx:g} x {g.Ly:g} x {g.Lz:g} m")
    print(f"spacing: {g.dx * 1000:g} x {g.dy * 1000:g} x {g.dz * 1000:g} mm")
    print(f"gravity: {cfg.physics.g:g} m/s^2")
    print(f"device: {cfg.device}")
    print(f"output: {cfg.output_dir}")
    print(f"allocated: p/alpha {state.p.shape}, u {state.u.shape}, "
          f"v {state.v.shape}, w {state.w.shape}")
    print(f"initial dt: {timestep.dt:g} s ({timestep.limiting_constraint})")
    print("free-surface fractional-step solver ready")

    if args.demo == "hydrostatic":
        initialize_box(
            state,
            (g.x0, g.y0, g.z0),
            (g.x0 + g.Lx, g.y0 + g.Ly, g.z0 + 0.6 * g.Lz),
        )
        for _ in range(args.steps):
            timestep = solver.compute_dt()
            solver.step(min(0.01, timestep.dt), timestep=timestep)
    elif args.demo == "sphere":
        radius = 0.18 * min(g.Lx, g.Ly, g.Lz)
        initialize_sphere(
            state,
            (g.x0 + 0.35 * g.Lx, g.y0 + 0.5 * g.Ly,
             g.z0 + 0.5 * g.Lz),
            radius,
        )
        face_mass = compute_face_liquid_mass(state, "u").numpy()[
            state.grid.interior_slices("u")
        ]
        speed = 0.15
        state.set_interior(
            "u", np.where(face_mass > 0.0, speed, 0.0).astype(np.float32),
        )
        advector = MomentumAdvector(cfg)
        for _ in range(args.steps):
            advector.advect(state, min(0.3 * g.dx / speed,
                                      solver.compute_dt().dt))

    if args.demo != "ready":
        diagnostics = measure_vof(state)
        frames = render_diagnostic_slices(
            state, cfg.output_dir, prefix=args.demo,
        )
        print(f"water volume: {diagnostics.volume:.9g} m^3")
        print(f"alpha bounds: [{diagnostics.alpha_min:.6g}, "
              f"{diagnostics.alpha_max:.6g}]")
        for frame in frames:
            print(f"diagnostic: {frame.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
