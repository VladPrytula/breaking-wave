"""Configuration dataclasses. Everything the solver needs, no hidden globals."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math


class BC(IntEnum):
    """Boundary condition kinds. Values are passed into Warp kernels as ints."""

    WALL = 0      # reflective, zero normal flux
    PERIODIC = 1  # wrap-around (natural for the along-shore direction)
    OPEN = 2      # zero-gradient outflow (reflective for long waves; see BCConfig docs)
    WAVEMAKER = 3  # generating-absorbing, on Riemann invariants (xlo only)


@dataclass(frozen=True)
class GridConfig:
    """Cartesian cell-centred grid with a ghost halo.

    ng=2 is the minimum for MUSCL: slopes need 1 neighbour, faces need
    reconstructed values from cells one further out.
    """

    nx: int = 256
    ny: int = 128
    Lx: float = 12.0
    Ly: float = 6.0
    x0: float = 0.0
    y0: float = 0.0
    ng: int = 2

    @property
    def dx(self) -> float:
        return self.Lx / self.nx

    @property
    def dy(self) -> float:
        return self.Ly / self.ny

    @property
    def shape(self) -> tuple[int, int]:
        """Padded array shape, including the ghost halo."""
        return (self.nx + 2 * self.ng, self.ny + 2 * self.ng)

    @property
    def cell_area(self) -> float:
        return self.dx * self.dy


@dataclass(frozen=True)
class PhysicsConfig:
    g: float = 9.81
    # Manning roughness; ~0.02-0.04 for a gravel/pebble bed.
    manning_n: float = 0.025
    # Depth below which a *face* is treated as dry in the Riemann solver.
    h_dry: float = 1.0e-10
    # Kurganov-Petrova velocity desingularisation scale [m]. u = q/h is exact
    # for h >= desing_eps and smoothly rolls off to 0 below it.
    desing_eps: float = 1.0e-3
    # Depth below which a cell counts as dry for *reconstruction*: MUSCL slopes
    # are zeroed in it and its neighbours. See numerics.update.reconstruct for
    # why this is required for well-balancedness at a shoreline.
    h_wet_recon: float = 1.0e-6
    # CFL for MUSCL + SSP-RK2 in 2D. The unsplit 2D condition is
    # dt * (a_x/dx + a_y/dy) <= 1; 0.4 leaves headroom for the source terms.
    cfl: float = 0.4
    dt_max: float = 0.05
    # Manning friction (semi-implicit operator split; see numerics.update).
    friction_enabled: bool = True

    # --- horizontal eddy viscosity (a CLOSURE, see closures/viscosity.py) ---
    # Constant part [m^2/s]. 0.0 disables viscosity entirely. Molecular water is
    # ~1e-6 and irrelevant at these scales; useful values are eddy viscosities,
    # O(1e-3) to O(1e-1) for a nearshore strip.
    nu_const: float = 0.0
    # Smagorinsky coefficient: adds (cs*delta)^2 |S|, delta = sqrt(dx*dy).
    # 0.0 disables. Typical 0.1-0.2 if used.
    smagorinsky_cs: float = 0.0


@dataclass(frozen=True)
class BCConfig:
    """Per-side boundary conditions.

    NOTE: OPEN is zero-gradient, which is *not* absorbing for gravity waves --
    it reflects a significant fraction of an outgoing wave. For a radiating
    offshore boundary use BC.WAVEMAKER (generating-absorbing, on Riemann
    invariants; xlo only), which with zero amplitudes is a pure absorber.
    """

    xlo: BC = BC.WALL
    xhi: BC = BC.WALL
    ylo: BC = BC.PERIODIC
    yhi: BC = BC.PERIODIC


@dataclass(frozen=True)
class ClosureConfig:
    """Explicit closures. All default OFF, so the resolved physics runs alone.

    Being able to switch each of these off independently and see which visual
    phenomenon disappears is the project's conceptual identity (plan.md end).
    """

    # --- breaking: detect, and measure what the shock-capturing scheme already
    # dissipates. It does not ADD dissipation; see closures/breaking.py.
    breaking_enabled: bool = False
    # Front-steepness threshold: breaking where d(eta)/dt >= gamma sqrt(g h).
    # Literature range 0.35-0.65.
    breaking_gamma: float = 0.45
    # 0 = hard indicator; > 0 = logistic of that relative width (differentiable
    # mode). The two are NOT claimed to be the same operator.
    breaking_smooth: float = 0.0

    # --- foam: transported, sourced only by measured breaking power ---
    foam_enabled: bool = False
    foam_alpha: float = 0.5     # foam per unit dissipated power
    foam_tau: float = 3.0       # decay time [s]
    foam_kappa: float = 0.004   # diffusivity [m^2/s]
    foam_max: float = 1.0       # cap; 0 disables the cap

    # --- optical wetness of the bed (plan.md section 10) ---
    # w has memory (tau_dry >> a wave period), so when enabled it belongs in
    # the periodic-orbit state vector or the loop will not close visually.
    wetness_enabled: bool = False
    tau_wet: float = 0.3        # wetting time once submerged [s]
    tau_dry: float = 35.0       # drying time once exposed [s]


@dataclass(frozen=True)
class SolverConfig:
    grid: GridConfig = GridConfig()
    physics: PhysicsConfig = PhysicsConfig()
    bc: BCConfig = BCConfig()
    closures: ClosureConfig = ClosureConfig()
    device: str = "cuda:0"


# ---------------------------------------------------------------------------
# Local 3-D solver configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Grid3DConfig:
    """Uniform Cartesian cell grid for the local MAC/VOF solver.

    The extents in the named presets are provisional experiment boxes.  The
    production domain is selected from the wave and scene requirements later
    in the advanced-water plan; these values do not make that choice early.
    """

    nx: int = 16
    ny: int = 12
    nz: int = 8
    Lx: float = 0.4
    Ly: float = 0.3
    Lz: float = 0.2
    x0: float = 0.0
    y0: float = 0.0
    z0: float = 0.0
    ng: int = 1

    def __post_init__(self) -> None:
        for name in ("nx", "ny", "nz"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if isinstance(self.ng, bool) or not isinstance(self.ng, int) or self.ng < 1:
            raise ValueError(f"ng must be a positive integer, got {self.ng!r}")
        if self.ng > min(self.nx, self.ny, self.nz):
            raise ValueError("ng cannot exceed the smallest physical cell count")
        for name in ("Lx", "Ly", "Lz"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a positive finite size, got {value!r}")
        for name in ("x0", "y0", "z0"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")

    @property
    def dx(self) -> float:
        return self.Lx / self.nx

    @property
    def dy(self) -> float:
        return self.Ly / self.ny

    @property
    def dz(self) -> float:
        return self.Lz / self.nz

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.nx, self.ny, self.nz)

    @property
    def cell_count(self) -> int:
        return self.nx * self.ny * self.nz


@dataclass(frozen=True)
class Physics3DConfig:
    """Physical constants used by the first 3-D solver stories."""

    g: float = 9.81
    rho: float = 1000.0
    surface_tension: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.g) or self.g < 0.0:
            raise ValueError(f"g must be non-negative and finite, got {self.g!r}")
        if not math.isfinite(self.rho) or self.rho <= 0.0:
            raise ValueError(f"rho must be positive and finite, got {self.rho!r}")
        if (not math.isfinite(self.surface_tension)
                or self.surface_tension < 0.0):
            raise ValueError(
                "surface_tension must be non-negative and finite, "
                f"got {self.surface_tension!r}",
            )


@dataclass(frozen=True)
class BC3DConfig:
    """Boundary kinds for the six sides of the local 3-D box.

    Periodicity is an axis property, so it must be selected on both sides.
    ``BC.OPEN`` is only the zero-gradient bring-up condition from AW-102; the
    later wave relaxation and absorption machinery replaces it for production.
    """

    xlo: BC = BC.WALL
    xhi: BC = BC.WALL
    ylo: BC = BC.PERIODIC
    yhi: BC = BC.PERIODIC
    zlo: BC = BC.WALL
    zhi: BC = BC.WALL

    def __post_init__(self) -> None:
        for name in ("xlo", "xhi", "ylo", "yhi", "zlo", "zhi"):
            kind = getattr(self, name)
            if not isinstance(kind, BC) or kind == BC.WAVEMAKER:
                raise ValueError(f"{name} must be BC.WALL, BC.PERIODIC, or BC.OPEN")
        for axis in ("x", "y", "z"):
            lo = getattr(self, f"{axis}lo")
            hi = getattr(self, f"{axis}hi")
            if (lo == BC.PERIODIC) != (hi == BC.PERIODIC):
                raise ValueError(f"{axis} periodic boundary must be paired")


@dataclass(frozen=True)
class Solver3DConfig:
    """Shared grid, physics, device, and output settings for 3-D runs."""

    grid: Grid3DConfig = Grid3DConfig()
    physics: Physics3DConfig = Physics3DConfig()
    bc: BC3DConfig = BC3DConfig()
    device: str = "cuda:0"
    output_dir: str = "out/wave_tank3d/tiny"
    # The retained-liquid path selects its matching unsplit transport and
    # kinetic metric. Its current contract requires zero surface tension;
    # production keeps the existing default until the performance boundary.
    pressure_discretization: str = "ghost_fluid"
    # Positive kinetic completion for unresolved opposite-face differences and
    # solid-normal traces on retained liquid facets; zero disables both terms.
    liquid_extension_inertia: float = 1.0
    # Liquid below this fraction of a cell's volume (theta*alpha) in a cell that
    # is not full (alpha < 1 - fraction) is unresolved residue: it carries no
    # pressure unknown (the resolved liquid sees it as free surface) and owns no
    # face velocity; its faces take the bounded extension of the resolved
    # faces, its volume is transported conservatively with that velocity, and
    # the momentum difference is reported as discarded. A full cell is liquid
    # however small its open part (shoreloop/physics3d/resolved.py).
    liquid_resolved_fraction: float = 1.0e-6
    # Height functions replace the Youngs seed only on complete, single-branch
    # liquid/air columns away from solids. This changes governing PLIC geometry.
    interface_reconstruction: str = "youngs"
    # PLIC-ray forcing integrates inclined footprints and the analytic source
    # across each column. The previous column-centre source remains selectable.
    wave_source_reconstruction: str = "column"
    wave_source_quadrature: int = 8
    # The unsplit comparison traces the whole MAC velocity through donor PLICs.
    vof_flux_reconstruction: str = "axis_slabs"
    # Numerical bound on velocity recovered as momentum/mass after transport.
    # "global" clamps to the pre-step maximum over every positive-mass face;
    # "local" clamps each face to the range of its own pre-step value, the
    # transported values it exchanged and its wet axial neighbours (and zero),
    # so transport cannot create or ratchet a spurious speed on near-empty faces.
    momentum_speed_bound: str = "global"
    # Dual control-volume mass carried by momentum transport. "aperture" is the
    # aperture-weighted mean cell fraction; "flux_consistent" is half of each
    # adjacent cell's water (theta*alpha), which the averaged VOF fluxes move.
    # They differ only on cut faces, where "aperture" rescales recovered speeds.
    momentum_face_mass: str = "aperture"
    # Air-band velocity extension used by transport and its time step. Order 0
    # averages the valid neighbours (a maximum principle: no extended value
    # exceeds the water values it is built from); order 1 continues the
    # gradient (2a - b) and can exceed every water speed by several times.
    velocity_extension_order: int = 0

    def __post_init__(self) -> None:
        if self.velocity_extension_order not in (0, 1):
            raise ValueError("velocity_extension_order must be 0 or 1")
        if self.vof_flux_reconstruction not in ("axis_slabs", "swept_prism"):
            raise ValueError("vof_flux_reconstruction must be axis_slabs or swept_prism")
        if self.momentum_speed_bound not in ("global", "local"):
            raise ValueError("momentum_speed_bound must be global or local")
        if self.momentum_speed_bound == "local" and self.pressure_discretization != "liquid_mimetic":
            raise ValueError("the local momentum speed bound requires liquid_mimetic transport")
        if self.momentum_face_mass not in ("aperture", "flux_consistent"):
            raise ValueError("momentum_face_mass must be aperture or flux_consistent")
        if self.momentum_face_mass == "flux_consistent" and self.pressure_discretization != "liquid_mimetic":
            raise ValueError("the flux-consistent face mass requires liquid_mimetic transport")
        if self.vof_flux_reconstruction == "swept_prism" and self.pressure_discretization != "liquid_mimetic":
            raise ValueError("swept_prism requires liquid_mimetic unsplit transport")
        if self.interface_reconstruction not in ("youngs", "height_function"):
            raise ValueError("interface_reconstruction must be youngs or height_function")
        if self.wave_source_reconstruction not in ("column", "plic_rays"):
            raise ValueError("wave_source_reconstruction must be column or plic_rays")
        if (isinstance(self.wave_source_quadrature, bool)
                or not isinstance(self.wave_source_quadrature, int)
                or self.wave_source_quadrature < 2):
            raise ValueError("wave_source_quadrature must be an integer of at least two")
        if self.pressure_discretization not in ("ghost_fluid", "liquid_mimetic"):
            raise ValueError("pressure_discretization must be ghost_fluid or liquid_mimetic")
        if not math.isfinite(self.liquid_extension_inertia) or self.liquid_extension_inertia < 0:
            raise ValueError("liquid_extension_inertia must be finite and nonnegative")
        if (not math.isfinite(self.liquid_resolved_fraction)
                or not 0.0 < self.liquid_resolved_fraction < 0.5):
            raise ValueError("liquid_resolved_fraction must lie in (0, 0.5)")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty string")
        if self.device != "cuda" and not self.device.startswith("cuda:"):
            raise ValueError("the 3-D solver requires a CUDA device")
        if not isinstance(self.output_dir, str) or not self.output_dir.strip():
            raise ValueError("output_dir must be a non-empty string")


SOLVER3D_PRESET_NAMES = ("tiny", "development", "validation")


def solver3d_preset(name: str) -> Solver3DConfig:
    """Construct one provisional, validated 3-D resolution preset."""

    if name == "tiny":
        grid = Grid3DConfig(nx=16, ny=12, nz=8, Lx=0.4, Ly=0.3, Lz=0.2)
    elif name == "development":
        # AW-404 measured tier: 25 mm, 49,152 cells, productive launch-bound
        # iteration while retaining three useful multigrid coarsenings.
        grid = Grid3DConfig(nx=64, ny=32, nz=24, Lx=1.6, Ly=0.8, Lz=0.6)
    elif name == "validation":
        # AW-404 measured tier: the same box at 12.5 mm and one additional
        # multigrid level. It remains small enough for frequent integration.
        grid = Grid3DConfig(nx=128, ny=64, nz=48, Lx=1.6, Ly=0.8, Lz=0.6)
    else:
        choices = ", ".join(SOLVER3D_PRESET_NAMES)
        raise ValueError(f"unknown 3-D solver preset {name!r}; choose from {choices}")
    return Solver3DConfig(grid=grid, output_dir=f"out/wave_tank3d/{name}")
