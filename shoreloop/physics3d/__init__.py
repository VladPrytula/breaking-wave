"""Local three-dimensional MAC/VOF solver components."""

from shoreloop.physics3d.advection import VOFAdvectionResult, VOFAdvector
from shoreloop.physics3d.boundary import fill_boundaries, fill_field_boundaries
from shoreloop.physics3d.breaker_tank import (NaturalBreakerMetrics,
                                              ShoalingWaveMetrics,
                                              run_natural_breaker_gate,
                                              run_shoaling_wave_gate)
from shoreloop.physics3d.boulder_cluster import (BoulderClusterMetrics,
                                                 run_boulder_cluster_gate)
from shoreloop.physics3d.boulder_wash import (OneBoulderWashMetrics,
                                              run_one_boulder_wash_gate)
from shoreloop.physics3d.cut_cells import (CutCellGeometry,
                                           generate_cut_cell_geometry,
                                           initialize_scene_cut_cells)
from shoreloop.physics3d.contact_line import (ContactLineConfig,
                                              ContactLineModel,
                                              ContactLineResult)
from shoreloop.physics3d.diagnostic_render import (SliceDiagnostic,
                                                   render_diagnostic_slices,
                                                   render_field_slice)
from shoreloop.physics3d.extrapolation import (VelocityExtension,
                                               extrapolate_velocity)
from shoreloop.physics3d.energy import (EnergyBudgetStep, EnergyLedger,
                                        MechanicalEnergy)
from shoreloop.physics3d.flow_diagnostics import (
    RecirculationMetrics, TurbulenceFields, TurbulenceMetrics,
    cell_velocity, measure_recirculation, measure_turbulence,
    render_turbulence_slice, turbulence_fields, velocity_gradient,
)
from shoreloop.physics3d.foam import (Foam3DConfig, Foam3DResult,
                                      InterfaceFoamModel)
from shoreloop.physics3d.free_surface import (FreeSurfaceGeometry,
                                              FreeSurfaceProjection,
                                              FreeSurfaceProjectionResult,
                                              build_free_surface_geometry,
                                              initialize_hydrostatic_pressure)
from shoreloop.physics3d.grid import MACGrid
from shoreloop.physics3d.interface import (InterfaceNormalResult,
                                           estimate_interface_normals)
from shoreloop.physics3d.momentum import (MomentumAdvectionResult,
                                          MomentumAdvector,
                                          compute_face_liquid_mass,
                                          measure_liquid_momentum)
from shoreloop.physics3d.multigrid import (MultigridHierarchy,
                                           MultigridLevel,
                                           MultigridVCycle,
                                           VCycleConfig,
                                           VCycleResult)
from shoreloop.physics3d.operators import MACOperators
from shoreloop.physics3d.pcg import PCGConfig, PCGResult, PressurePCG
from shoreloop.physics3d.plic import (PLICReconstruction, reconstruct_plic,
                                     solve_plic_offset, unit_cube_volume)
from shoreloop.physics3d.porous import (
    PorousBedModel, PorousDragConfig, PorousDragResult,
    PorousStorageConfig, PorousStorageResult, porous_capacity_from_geometry,
)
from shoreloop.physics3d.projection import (PressureProjection, ProjectionConfig,
                                           ProjectionResult)
from shoreloop.physics3d.relaxation import (WaveRelaxationResult,
                                            WaveRelaxationZone,
                                            relaxation_weight)
from shoreloop.physics3d.solver import Solver3D
from shoreloop.physics3d.solids import (PlaneBeach, SolidFluxDiagnostics,
                                        VoxelSolidGeometry,
                                        enforce_voxel_solids,
                                        initialize_plane_beach,
                                        initialize_scene_voxels,
                                        measure_solid_flux,
                                        rasterize_voxel_solids)
from shoreloop.physics3d.solid_wetness import (
    SolidWetnessConfig, SolidWetnessModel, SolidWetnessResult,
    SurfaceWetnessSample, sample_scene_surface_wetness,
)
from shoreloop.physics3d.state import State3D
from shoreloop.physics3d.surface_tension import (
    SurfaceTensionGeometry, gravity_capillary_angular_frequency,
    measure_plic_surface_area, reconstruct_surface_tension_geometry,
)
from shoreloop.physics3d.timestep import (TimeStepConfig, TimeStepController,
                                          TimeStepResult)
from shoreloop.physics3d.topology import (BreakingTopologyDiagnostics,
                                          TopologyTracker,
                                          measure_liquid_kinetic_energy)
from shoreloop.physics3d.vof import (VOFDiagnostics, initialize_box,
                                     initialize_plane, initialize_sphere,
                                     measure_vof)
from shoreloop.physics3d.vof_flux import (DirectionalFluxResult,
                                          compute_directional_liquid_fluxes,
                                          directional_liquid_flux)
from shoreloop.physics3d.viscosity import (ViscosityConfig, ViscosityModel,
                                           ViscosityResult)
from shoreloop.physics3d.wave_tank import (LinearWaveMetrics,
                                           PlaneBeachMetrics,
                                           run_linear_wave_gate,
                                           run_plane_beach_gate)
from shoreloop.physics3d.waves import (LinearWaveInlet, PeriodicWaveComponent,
                                       ReflectionMeasurement,
                                       WaveModeFit, fit_probe_wave_modes,
                                       finite_depth_wavenumber,
                                       measure_two_probe_reflection)
from shoreloop.physics3d.wet_dry import WispRemover, WispRemovalResult
from shoreloop.physics3d.wall_stress import (WallStressConfig, WallStressModel,
                                             WallStressResult)

__all__ = [
    "BoulderClusterMetrics",
    "BreakingTopologyDiagnostics",
    "ContactLineConfig", "ContactLineModel", "ContactLineResult",
    "CutCellGeometry", "DirectionalFluxResult",
    "FreeSurfaceGeometry", "FreeSurfaceProjection",
    "Foam3DConfig", "Foam3DResult", "InterfaceFoamModel",
    "EnergyBudgetStep", "EnergyLedger", "MechanicalEnergy",
    "FreeSurfaceProjectionResult", "InterfaceNormalResult", "MACGrid",
    "RecirculationMetrics", "TurbulenceFields", "TurbulenceMetrics",
    "MomentumAdvectionResult", "MomentumAdvector",
    "MultigridHierarchy", "MultigridLevel", "MultigridVCycle",
    "NaturalBreakerMetrics", "OneBoulderWashMetrics",
    "MACOperators", "PCGConfig",
    "PCGResult", "PLICReconstruction", "PressurePCG", "SliceDiagnostic",
    "PorousBedModel", "PorousDragConfig", "PorousDragResult",
    "PorousStorageConfig", "PorousStorageResult",
    "LinearWaveInlet", "LinearWaveMetrics", "PeriodicWaveComponent",
    "PlaneBeach", "PlaneBeachMetrics", "PressureProjection",
    "ProjectionConfig", "ReflectionMeasurement",
    "ProjectionResult", "State3D",
    "ShoalingWaveMetrics",
    "SolidFluxDiagnostics",
    "SolidWetnessConfig", "SolidWetnessModel", "SolidWetnessResult",
    "SurfaceWetnessSample",
    "SurfaceTensionGeometry",
    "Solver3D", "TimeStepConfig", "TimeStepController", "TimeStepResult",
    "TopologyTracker",
    "VOFAdvectionResult", "VOFAdvector", "VOFDiagnostics",
    "ViscosityConfig", "ViscosityModel", "ViscosityResult",
    "VelocityExtension",
    "VoxelSolidGeometry",
    "VCycleConfig", "VCycleResult",
    "WaveModeFit", "WaveRelaxationResult", "WaveRelaxationZone",
    "WispRemover", "WispRemovalResult",
    "WallStressConfig", "WallStressModel", "WallStressResult",
    "estimate_interface_normals", "fill_boundaries",
    "fill_field_boundaries",
    "cell_velocity", "measure_recirculation", "measure_turbulence",
    "render_turbulence_slice", "turbulence_fields", "velocity_gradient",
    "generate_cut_cell_geometry", "initialize_scene_cut_cells",
    "initialize_hydrostatic_pressure",
    "gravity_capillary_angular_frequency", "measure_plic_surface_area",
    "enforce_voxel_solids", "initialize_plane_beach",
    "initialize_scene_voxels", "rasterize_voxel_solids",
    "compute_directional_liquid_fluxes", "directional_liquid_flux",
    "compute_face_liquid_mass",
    "measure_liquid_momentum",
    "measure_liquid_kinetic_energy",
    "measure_solid_flux",
    "sample_scene_surface_wetness",
    "measure_two_probe_reflection",
    "fit_probe_wave_modes",
    "build_free_surface_geometry",
    "render_diagnostic_slices", "render_field_slice",
    "finite_depth_wavenumber",
    "initialize_box", "initialize_plane", "initialize_sphere", "measure_vof",
    "extrapolate_velocity",
    "reconstruct_plic", "solve_plic_offset", "unit_cube_volume",
    "reconstruct_surface_tension_geometry",
    "porous_capacity_from_geometry",
    "relaxation_weight",
    "run_linear_wave_gate", "run_natural_breaker_gate",
    "run_boulder_cluster_gate",
    "run_one_boulder_wash_gate",
    "run_plane_beach_gate", "run_shoaling_wave_gate",
]
