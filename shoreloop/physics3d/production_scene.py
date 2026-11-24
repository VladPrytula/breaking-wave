"""Immutable advanced-water physical and rendering configuration.

The scene targets the reference's calm grouped short-crested water, one
localized natural breaker, shallow wash and backwash through a shore-edge rock
cluster, and persistent wet stone.  People, spray, entrained air, bubbles, and
the distant coastline in ``IMG_4543.MOV`` are deliberately outside the model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from shoreloop.config import (BC, BC3DConfig, Grid3DConfig, Physics3DConfig,
                              Solver3DConfig)
from shoreloop.physics3d.contact_line import ContactLineConfig
from shoreloop.physics3d.cut_cells import initialize_scene_cut_cells
from shoreloop.physics3d.energy import EnergyLedger
from shoreloop.physics3d.foam import Foam3DConfig
from shoreloop.physics3d.free_surface import initialize_hydrostatic_pressure
from shoreloop.physics3d.pcg import PCGConfig
from shoreloop.physics3d.porous import (
    PorousBedModel,
    PorousDragConfig,
    PorousStorageConfig,
    canopy_weight_from_geometry,
    porous_capacity_from_geometry,
)
from shoreloop.physics3d.projection import ProjectionConfig
from shoreloop.physics3d.relaxation import WaveRelaxationZone
from shoreloop.physics3d.solid_wetness import SolidWetnessConfig
from shoreloop.physics3d.solver import Solver3D
from shoreloop.physics3d.timestep import TimeStepConfig
from shoreloop.physics3d.viscosity import ViscosityConfig
from shoreloop.physics3d.vof import initialize_box
from shoreloop.physics3d.wall_stress import WallStressConfig
from shoreloop.physics3d.gravel import gravel_drag, gravel_storage
from shoreloop.physics3d.waves import (LinearWaveInlet, PeriodicWaveComponent,
                                       finite_depth_wavenumber)
from shoreloop.physics3d.wet_dry import WispRemover
from shoreloop.render.camera import Camera
from shoreloop.scene.geometry3d import (
    AnalyticBeach,
    BEACH_MATERIAL_ID,
    BOULDER_MATERIAL_ID,
    SceneGeometry,
    SceneMaterial,
    production_boulders,
)


PRODUCTION_SURFACE_SMOOTHING_ITERATIONS = 0
PRODUCTION_INCIDENT_SCALE_FROM_AW1103 = 1.45
# The render collar requires a graph-valued seam.  Keep it well inside the
# incident zone, upstream of resolved breaker topology; the renderer tiles the
# solver's exact periodic-y surface before attaching the offshore field.
PRODUCTION_OFFSHORE_HANDOFF_X = 0.80
# The incident group is generated from rest over this distance from the
# offshore wall (16 cells), so the zone's target at the wall is still water
# and consistent with the wall; the zone core still imposes the full field.
PRODUCTION_SOURCE_GENERATION_RAMP = 0.40
# The relaxation target uses linear/Stokes kinematics.  Keep its fully weighted
# boundary below the project's conservative depth-limited forcing guard; the
# resolved solver, rather than the source, owns subsequent shoaling and breaking.
PRODUCTION_MAXIMUM_INCIDENT_BOUNDARY_HEIGHT_TO_DEPTH_RATIO = 0.78
# Liquid per cell volume (theta*alpha) for cells with a free surface; full
# cells stay liquid (shoreloop/physics3d/resolved.py). Measured on the
# 1.52/1.92/2.32 s states of restart_20260925b
# (out/review_20261009/residue_threshold/): 0.02 is the smallest value that
# demotes both boulder slivers whose faces set the time step at 2.32 s and 11
# of the 12 cells with mass-block conditioning below 1e-4; among cells with a
# free surface it demotes only beach columns shallower than .5 mm. 0.01 of
# the open volume, the former rule, saw neither sliver; the first rule without
# the full-cell clause turned submerged beach slivers into free surfaces.
PRODUCTION_LIQUID_RESOLVED_FRACTION = 2.0e-2


# AW-1101 oblique pair amplitudes at unit incident scale by temporal mode.  Their
# a^2 sums fix the oblique energy of each period; 62.5 percent of it travels
# toward +y, as in AW-1101, so the alongshore modulation is not a symmetric
# standing pattern.  A random-phase sum with a cos^(2s) directional weight is
# the standard linear model of a directionally spread sea; the exactly
# periodic loop only admits the integer alongshore modes sampled here.
AW1101_OBLIQUE_AMPLITUDES = {
    3: (0.011785216293175237, 0.009410756352979029),
    4: (0.009166279339136295, 0.006721968823556448),
}
PRODUCTION_OBLIQUE_POSITIVE_SHARE = 0.625
PRODUCTION_OBLIQUE_SPREAD_EXPONENT = 3
PRODUCTION_OBLIQUE_MAXIMUM_ANGLE = math.radians(40.0)
# Phases are one fixed uniform draw.  Seed 31 gives the lowest sampled inlet
# crest-trough height (.694 of depth) of seeds 0 to 63, whose ratios spanned
# .69 to 1.00 with a median of .83, so the source stays below the .78
# depth-limited forcing guard with the same energy.
PRODUCTION_OBLIQUE_PHASE_SEED = 31

# Reference-calibrated incident spectrum (AW-1119). Measured on IMG_4543.MOV
# on 2026-10-13 (out/review_20261013/aw1119_reference): fronts every 2.0 s
# (1.7-2.3) with every second one the larger, 6-10 cm typical and 12-16 cm
# largest at breaking, breaking 0.25-0.7 m seaward of the waterline, a
# collapsing to weakly plunging breaker. A 4.0 s loop carries the 2.0 s
# primary as its second harmonic and the alternation as the beat with the
# third (1.33 s); the amplitudes are inlet values in 0.22 m depth, before
# shoaling. The 2.83 m wavelength of the primary needs an offshore stretch
# the 4.8 m scene does not have, so the reference scene starts at
# x0 = -3.2 m: the relaxation zone ends 0.4 m offshore of x = 0 and the wave
# travels 2.1 m of flat bed to the toe. Everything inshore is the production
# scene unchanged.
#
# The first probe (M11, AW-1119) fixed the amplitudes and the directional
# split: with 0.030 / 0.015 / 0.006 m and AW-1101's 68% shore-normal share
# the beat-maximum front left the relaxation zone 14.5 cm high at the focus
# of the oblique pairs (twice its alongshore median), steepened to 17 cm on
# the flat bed (0.79 of the depth) and broke there, 2.7 m seaward of the
# waterline, while the beat-minimum front (sum 0.021 m) collapsed at the
# waterline with a 13 cm swash, the reference's type. The reference's crests
# are nearly shore-parallel, so the oblique share drops to 10% (focus 1.4x
# the median instead of 2x), and the amplitudes keep the measured
# largest-to-typical ratio of about 1.6 (in-phase sums 0.042 / 0.026 m)
# with the focus height of the largest front predicted at 10.7 cm, half
# the depth, from the first probe's measured response.
REFERENCE_LOOP_PERIOD = 4.0
REFERENCE_AMPLITUDES = {2: 0.030, 3: 0.008, 4: 0.004}  # metres, by temporal mode
REFERENCE_NORMAL_SHARE = 0.90
REFERENCE_X0 = -3.2
INCIDENT_SPECTRA = ("aw1103", "reference")
# The bed closures: "aw1104" is the first capture's Navier slip, relaxation
# storage and grid-scale shallow drag; "gravel" derives the grain canopy drag
# within one grain of the bed and the storage layer's capacity (bound to the
# water table), fill rate and drainage from the reference's 3 cm grain, under
# the same Navier slip (shoreloop/physics3d/gravel.py).
BEDS = ("aw1104", "gravel")


def _spread_oblique_components(
    shared_wave: dict[str, float],
    incident_scale: float,
    oblique_amplitudes: dict[int, tuple[float, float]] = AW1101_OBLIQUE_AMPLITUDES,
) -> tuple[PeriodicWaveComponent, ...]:
    """Spread each period's oblique pair energy over its resolved oblique modes."""
    phases = np.random.default_rng(PRODUCTION_OBLIQUE_PHASE_SEED)
    components = []
    for temporal_mode, pair_amplitudes in oblique_amplitudes.items():
        energy = incident_scale**2 * sum(a * a for a in pair_amplitudes)
        omega = 2.0 * math.pi * temporal_mode / shared_wave["period"]
        k = finite_depth_wavenumber(omega, shared_wave["depth"], shared_wave["gravity"])
        weights = {}
        for mode in range(1, 1000):
            sine = 2.0 * math.pi * mode / (shared_wave["alongshore_length"] * k)
            if sine > math.sin(PRODUCTION_OBLIQUE_MAXIMUM_ANGLE):
                break
            weights[mode] = math.cos(math.asin(sine)) ** (2 * PRODUCTION_OBLIQUE_SPREAD_EXPONENT)
        total_weight = sum(weights.values())
        for mode, weight in weights.items():
            for sign, share in (
                (1, PRODUCTION_OBLIQUE_POSITIVE_SHARE),
                (-1, 1.0 - PRODUCTION_OBLIQUE_POSITIVE_SHARE),
            ):
                components.append(PeriodicWaveComponent(
                    **shared_wave,
                    temporal_mode=temporal_mode,
                    alongshore_mode=sign * mode,
                    amplitude=math.sqrt(energy * share * weight / total_weight),
                    phase_offset=float(phases.uniform(-math.pi, math.pi)),
                ))
    return tuple(components)


@dataclass(frozen=True)
class ProductionCamera:
    """One fixed production-scene camera and image response."""

    position: tuple[float, float, float]
    yaw_degrees: float
    pitch_degrees: float
    vertical_fov_degrees: float
    exposure: float
    white_balance: tuple[float, float, float]

    def build(self, width: int, height: int) -> Camera:
        return Camera(
            pos=self.position,
            yaw_deg=self.yaw_degrees,
            pitch_deg=self.pitch_degrees,
            vfov_deg=self.vertical_fov_degrees,
            width=width,
            height=height,
            exposure=self.exposure,
            wb=self.white_balance,
        )


@dataclass(frozen=True)
class ProductionScene:
    """Every physical choice held fixed during the production-scene run."""

    solver: Solver3DConfig
    geometry: SceneGeometry
    water_depth: float
    inlet: LinearWaveInlet
    incident_scale_from_aw1103: float
    incident: str
    bed: str
    projection: ProjectionConfig
    timestep: TimeStepConfig
    viscosity: ViscosityConfig
    wall_stress: WallStressConfig
    contact_line: ContactLineConfig
    porous_storage: PorousStorageConfig
    porous_drag: PorousDragConfig | None
    foam: Foam3DConfig
    solid_wetness: SolidWetnessConfig
    camera: ProductionCamera
    hero_camera: ProductionCamera
    hero_preview_resolution: tuple[int, int]
    sunset_camera: ProductionCamera
    sunset_light_preset: str
    sunset_probe_resolution: tuple[int, int]
    sunset_final_resolution: tuple[int, int]
    sunset_render_seed: int
    input_mode: str
    input_decision: str
    preview_duration: float
    preview_frame_times: tuple[float, ...]
    topology_x_bounds: tuple[float, float]
    cluster_bounds: tuple[float, float, float, float]
    gap_sections: tuple[tuple[int, int, float], ...]
    offshore_handoff_x: float
    ripple_spacing: float
    target_features: tuple[str, ...]
    deferred_features: tuple[str, ...]

    @property
    def relaxation_width(self) -> float:
        return max(component.wavelength for component in self.inlet.components)

    @property
    def still_water_elevation(self) -> float:
        return self.solver.grid.z0 + self.water_depth


def production_scene(
    *,
    device: str = "cuda:0",
    output_dir: str | Path = "out/production_scene_preview",
    surface_tension: float = 0.0,
    incident_scale: float = PRODUCTION_INCIDENT_SCALE_FROM_AW1103,
    incident: str = "aw1103",
    bed: str = "aw1104",
) -> ProductionScene:
    """Construct the fixed development-resolution production scene.

    ``incident`` selects the forcing: the AW-1103 group on the 2.4 s loop, or
    the reference-calibrated spectrum on a 4.0 s loop with the offshore
    stretch it needs (see ``REFERENCE_LOOP_PERIOD``). ``bed`` selects the bed
    closures (see ``BEDS``).
    """
    if incident not in INCIDENT_SPECTRA:
        raise ValueError(f"unknown incident spectrum {incident!r}; choose from {INCIDENT_SPECTRA}")
    if bed not in BEDS:
        raise ValueError(f"unknown bed {bed!r}; choose from {BEDS}")
    grid = Grid3DConfig(
        nx=192 if incident == "aw1103" else 320,
        ny=112,
        nz=48,
        Lx=4.8 if incident == "aw1103" else 4.8 - REFERENCE_X0,
        Ly=6.0,
        Lz=0.96,
        x0=0.0 if incident == "aw1103" else REFERENCE_X0,
    )
    solver = Solver3DConfig(
        grid=grid,
        pressure_discretization="liquid_mimetic",
        # AW-1110: liquid below 1% of a cell's open volume (.2 mm here, .1 mm at
        # the production grid) is unresolved residue with no pressure unknown;
        # a drained boulder cut cell otherwise drives the projection to 10 m/s.
        liquid_resolved_fraction=PRODUCTION_LIQUID_RESOLVED_FRACTION,
        # The global transport clamp let near-empty faces hold 81 m/s and pin
        # dt at .062 ms (2026-09-25 review); bound each face locally instead,
        # with the dual mass the VOF fluxes move so cut-beach films are not
        # rescaled into that bound.
        momentum_speed_bound="local",
        momentum_face_mass="flux_consistent",
        physics=Physics3DConfig(surface_tension=surface_tension),
        bc=BC3DConfig(
            xlo=BC.WALL,
            xhi=BC.WALL,
            ylo=BC.PERIODIC,
            yhi=BC.PERIODIC,
            zlo=BC.WALL,
            zhi=BC.WALL,
        ),
        device=device,
        output_dir=str(output_dir),
    )
    water_depth = 0.22
    base_period = 2.4
    shared_wave = {
        "period": base_period,
        "depth": water_depth,
        "alongshore_length": grid.Ly,
        "gravity": solver.physics.g,
    }
    # The shore-normal modes constructively focus at the beach toe at
    # t=1.5413041 s.  AW-1101 keeps that group timing and total sum(a^2) while
    # assigning 68/20/12 percent to normal/positive/negative directions.  The
    # shore-normal components keep their AW-1101 amplitudes and phases; each
    # period's oblique energy is the AW-1101 pair energy, spread by
    # _spread_oblique_components over every propagating alongshore mode inside
    # PRODUCTION_OBLIQUE_MAXIMUM_ANGLE so the alongshore interference is
    # irregular instead of a repeating two-mode pattern.  The corrected
    # AW-1104 inlet scales every amplitude together, leaving periods, phases,
    # and direction shares unchanged; its one effective shallow-bed response
    # bracket is declared separately below.
    normal_components = (
        PeriodicWaveComponent(
            **shared_wave,
            temporal_mode=2,
            amplitude=incident_scale * 0.015126144304132297,
            phase_offset=1.1176160321499085,
            stokes_order=2,
        ),
        PeriodicWaveComponent(
            **shared_wave,
            temporal_mode=3,
            amplitude=incident_scale * 0.020420294810578603,
        ),
        PeriodicWaveComponent(
            **shared_wave,
            temporal_mode=4,
            amplitude=incident_scale * 0.010588301012892608,
            phase_offset=2.591987353072888,
        ),
    )
    if incident == "reference":
        base_period = REFERENCE_LOOP_PERIOD
        shared_wave["period"] = base_period
        normal_components = tuple(
            PeriodicWaveComponent(
                **shared_wave, temporal_mode=mode,
                amplitude=math.sqrt(REFERENCE_NORMAL_SHARE) * amplitude,
                stokes_order=2 if mode == 2 else 1,
            )
            for mode, amplitude in REFERENCE_AMPLITUDES.items()
        )
        oblique = {
            mode: (math.sqrt((1.0 - REFERENCE_NORMAL_SHARE) * PRODUCTION_OBLIQUE_POSITIVE_SHARE) * amplitude,
                   math.sqrt((1.0 - REFERENCE_NORMAL_SHARE) * (1.0 - PRODUCTION_OBLIQUE_POSITIVE_SHARE)) * amplitude)
            for mode, amplitude in REFERENCE_AMPLITUDES.items()
        }
        inlet = LinearWaveInlet(
            normal_components + _spread_oblique_components(shared_wave, 1.0, oblique),
        )
    else:
        inlet = LinearWaveInlet(
            normal_components + _spread_oblique_components(shared_wave, incident_scale),
        )
    materials = (
        SceneMaterial(
            BEACH_MATERIAL_ID,
            "beach",
            0.80,
            (0.143, 0.140, 0.149),
            (0.108, 0.107, 0.113),
            0.34,
        ),
        SceneMaterial(
            BOULDER_MATERIAL_ID,
            "boulder",
            0.68,
            (0.165, 0.170, 0.160),
            (0.116, 0.124, 0.111),
            0.30,
        ),
    )
    geometry = SceneGeometry(
        AnalyticBeach(toe_x=1.75, slope=0.15, base_elevation=grid.z0),
        production_boulders(),
        materials,
    )
    return ProductionScene(
        solver=solver,
        geometry=geometry,
        water_depth=water_depth,
        inlet=inlet,
        incident_scale_from_aw1103=incident_scale,
        incident=incident,
        bed=bed,
        projection=ProjectionConfig(
            pcg=PCGConfig(
                relative_tolerance=2.0e-7,
                max_iterations=10000,
                preconditioner="jacobi",
            ),
            rms_divergence_tolerance=6.0e-4,
        ),
        timestep=TimeStepConfig(dt_max=0.005),
        viscosity=ViscosityConfig(
            molecular_viscosity=1.0e-6,
            smagorinsky_cs=0.12,
            maximum_sgs_viscosity=0.02,
        ),
        wall_stress=WallStressConfig(
            wall_viscosity=2.0e-4,
            beach_slip_length=0.012,
            boulder_slip_length=0.006,
        ),
        contact_line=ContactLineConfig(
            contact_length=0.05,
            advancing_diffusivity=8.0e-5,
            receding_diffusivity=4.0e-5,
            hysteresis_fraction=0.005,
        ),
        porous_storage=PorousStorageConfig(
            layer_depth=0.08,
            capacity_fraction=0.20,
            infiltration_rate=0.8,
            drainage_time=4.0,
        ) if bed == "aw1104" else gravel_storage(water_table=grid.z0 + water_depth),
        porous_drag=PorousDragConfig(
            # Effective resistance of the unresolved shallow bed layer, not a
            # claim of pore-scale permeability at the hydrodynamic grid size.
            permeability=2.0e-5,
            forchheimer_coefficient=0.08,
        ) if bed == "aw1104" else gravel_drag(),
        foam=Foam3DConfig(
            dissipation_yield=1.0e-4,
            deformation_yield=0.006,
            deformation_threshold=5.0,
            decay_time=0.8,
            diffusivity=5.0e-6,
        ),
        solid_wetness=SolidWetnessConfig(),
        camera=ProductionCamera(
            position=(7.20, 3.00, 1.55),
            yaw_degrees=180.0,
            pitch_degrees=14.0,
            vertical_fov_degrees=32.0,
            exposure=1.12,
            white_balance=(1.0, 1.0, 1.0),
        ),
        hero_camera=ProductionCamera(
            position=(6.75, 5.15, 1.55),
            yaw_degrees=196.50436138175502,
            pitch_degrees=15.984663800879124,
            vertical_fov_degrees=38.0,
            exposure=1.12,
            white_balance=(1.0, 1.0, 1.0),
        ),
        hero_preview_resolution=(640, 480),
        sunset_camera=ProductionCamera(
            position=(6.50, 4.80, 1.25),
            yaw_degrees=196.2323506611562,
            pitch_degrees=12.472758324663351,
            vertical_fov_degrees=30.0,
            exposure=0.62,
            white_balance=(1.0, 1.0, 1.0),
        ),
        sunset_light_preset="sunset",
        sunset_probe_resolution=(960, 540),
        sunset_final_resolution=(1920, 1080),
        sunset_render_seed=1001,
        input_mode="standalone",
        input_decision=(
            "AW-708 produced the required event and AW-707 measured no "
            "physical or visible offshore seam, so conditional M8 remains "
            "deferred."
        ),
        preview_duration=4.8,
        preview_frame_times=(
            0.0, 0.6, 1.2, 1.48, 1.96, 2.4, 3.0, 3.6, 4.2, 4.8,
        ),
        topology_x_bounds=(1.65, 2.85),
        cluster_bounds=(1.70, 3.10, 1.20, 5.60),
        gap_sections=(
            (3, 0, 2.28),
            (0, 1, 2.55),
            (1, 2, 2.48),
            (2, 4, 2.52),
        ),
        offshore_handoff_x=PRODUCTION_OFFSHORE_HANDOFF_X,
        ripple_spacing=0.005,
        target_features=(
            "calm grouped multi-scale short-crested incident waves",
            "a localized resolved natural breaker",
            "shallow wash and backwash through the boulder field",
            "resolved wrap, split flow, wakes, and boulder wetting",
        ),
        deferred_features=(
            "people and distant coastline",
            "resolved air, spray, bubbles, and entrainment",
            "individual pebble geometry in the hydrodynamic grid",
        ),
    )


def build_production_solver(
    scene: ProductionScene,
    *,
    porosity_enabled: bool = True,
    porous_storage_enabled: bool = True,
) -> Solver3D:
    """Instantiate authoritative state from one immutable scene."""
    solver = Solver3D(
        scene.solver,
        scene.projection,
        scene.timestep,
        viscosity_settings=scene.viscosity,
        wall_stress_settings=scene.wall_stress,
        contact_line_settings=scene.contact_line,
        foam_settings=scene.foam,
        solid_wetness_settings=scene.solid_wetness,
    )
    grid = scene.solver.grid
    initialize_box(
        solver.state,
        (grid.x0, grid.y0, grid.z0),
        (grid.x0 + grid.Lx, grid.y0 + grid.Ly,
         scene.still_water_elevation),
    )
    cut_cells = initialize_scene_cut_cells(solver.state, scene.geometry)
    initialize_hydrostatic_pressure(
        solver.state, scene.still_water_elevation,
    )
    if porosity_enabled:
        capacity = porous_capacity_from_geometry(
            cut_cells,
            scene.solver,
            scene.porous_storage,
        )
        porous_bed = PorousBedModel(
            scene.solver,
            capacity,
            storage_settings=(
                scene.porous_storage if porous_storage_enabled else None
            ),
            drag_settings=scene.porous_drag,
            drag_weight=(
                canopy_weight_from_geometry(cut_cells, scene.solver, scene.porous_drag)
                if scene.porous_drag is not None and scene.porous_drag.law == "canopy" else None
            ),
        )
        solver.set_porous_bed(porous_bed)
        if porous_storage_enabled:
            porous_bed.initialize_storage_equilibrium(solver.state)
    zone = WaveRelaxationZone(
        scene.solver,
        scene.inlet,
        scene.relaxation_width,
        timescale=0.02 * scene.inlet.period,
        generation_ramp=PRODUCTION_SOURCE_GENERATION_RAMP,
    )
    # The incident group already stands several centimetres high at the wall
    # at t=0. Starting from that field inside the zone, tapered like the zone
    # itself, is the initial state; building it from rest made the source
    # rebuild a collapsing step against the offshore wall every step.
    zone.impose(solver.state, 0.0)
    # The tapered field carries the taper's own divergence and the staircase
    # of its wet support, and transport precedes projection inside a step.
    # One projection without body force makes the liquid field discretely
    # divergence-free before the first step; gravity acts from step one.
    solver.projection.project(
        solver.state, scene.timestep.dt_max,
        operator=solver.liquid_operator(), acceleration=(0.0, 0.0, 0.0),
    )
    solver.set_wave_relaxation(zone)
    solver.set_wisp_remover(WispRemover(scene.solver))
    solver.set_energy_ledger(EnergyLedger(scene.solver))
    return solver
