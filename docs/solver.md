# The solver

What the code in `shoreloop/physics3d/` computes, in the order one time step
runs, and the measurements each piece had to pass before it was trusted. The
numbers quoted are the ones recorded when the corresponding story was
accepted; they came from the DGX Spark GB10 unless stated otherwise. The
[post](https://vladyslavprytula.substack.com/p/one-shore-multiple-representations)
tells the story around them; this document is the reference.

## 1. The model

### Why three dimensions

A depth-averaged heightmap describes the surface as a single-valued graph
`z = η(x, y)`. A plunging crest folds over air and crosses one vertical line
three times, which that state space cannot hold. It also moves every wave at
`sqrt(g h)`: at the 0.8 m offshore depth the 1.14 s harmonic actually travels
59% slower than that. Both limits are properties of the representation, not
of the scheme, so the representation changed.

### Equations

One-phase incompressible Navier–Stokes for the water; the air is not solved.
With density `ρ`, kinematic viscosity `ν`, a sub-grid viscosity `ν_t`, gravity
`g`, and a porous resistance `f_porous` that is nonzero only inside the bed,

```
∂u/∂t + ∇·(u ⊗ u) = −(1/ρ) ∇p + ∇·( 2 (ν + ν_t) D(u) ) + g + f_porous
∇·u = 0,     D(u) = ( ∇u + ∇uᵀ ) / 2
```

The free surface carries atmospheric pressure, optionally corrected by surface
tension, `p = p_atm + σ κ`. The water region is a volume fraction
`α(x, t) ∈ [0, 1]` per cell transported conservatively,

```
∂α/∂t + ∇·(α u) = 0 .
```

This is volume of fluid (VOF): water volume is conserved exactly and topology
changes (overturning, reconnection, splitting) need no special event. The
price is that the interface is one plane per cell, which section 7 returns to.

### What is resolved and what is a closure

Resolved: three velocity components, incompressible pressure, gravity-wave
propagation and nonlinear interaction inside the box, free-surface advection
including overturning and reconnection, run-up and backwash, flow exclusion
and large-scale separation around the boulders, resolved eddies and jets, the
exchange flux with the porous bed.

Modelled, each as a named and switchable closure: sub-grid turbulence,
rough-wall stress, grid-scale contact-line regularisation, sub-grid pebble-bed
storage, drainage and drag, foam generation and transport, persistent optical
wetness, surface tension at unresolved curvature.

Rendered only: ripples smaller than the grid, microstructure inside a foam
field, films thinner than a cell, stone albedo and normal detail, camera
exposure. No rendered-only effect may touch mass or momentum.

## 2. Discretisation

### The grid

A uniform staggered MAC grid (`physics3d/grid.py`, `state.py`): pressure,
water fraction `α`, open fraction `θ` and scalars at cell centres; `u`, `v`,
`w` on the `x`, `y`, `z` faces; solid apertures on faces. Uniform because
geometric multigrid, predictable memory access and deterministic surface
extraction all want it. Production state is float32.

The beach is an analytic plane and the boulders are rotated superquadrics
(`scene/geometry3d.py`). One scene object owns the implicit solids and their
render tessellation, and everything derives from it: the solver's open
fractions and face apertures (sampled at `4³` and `4²` sub-cells,
`physics3d/cut_cells.py`), the cut-cell centroids and normals from the signed
distance, the diagnostics, and the triangles the renderer draws. Mesh
vertices satisfy the implicit equation to `2e-14`. This single-source rule is
what stops the water from flowing as if the bed were smooth while the picture
shows boulders.

### Compatible operators

The discrete divergence and gradient are built as an adjoint pair
(`physics3d/operators.py`), so the pressure matrix `−D G` is symmetric
positive definite and the projection cannot create energy. With cut cells the
divergence uses the open face areas `A_f` and the open cell volume `V_i`:

```
(D u)_i = (1/V_i) Σ_{f ∈ ∂i} A_f u_f s_if .
```

Symmetry error below `2e-6` on a sloped plane and a sphere, with cut geometry
on every multigrid level. Solid faces are exactly impermeable (measured
closed-face flux exactly zero). Cells below one eighth open retain the stable
part of their update and conservatively redistribute the rest to neighbours,
because clipping the time step to the smallest fractional cell is not
acceptable at production size: an adversarial `θ = 1/64` cell that clipped
`8e-5 m³` without stabilisation closed to `5e-10 m³` with it.

### Time step

`physics3d/timestep.py` takes the minimum of four analytic limits, each with
its own CFL factor (defaults 0.5, 0.5, 0.25, 0.25):

```
dt_adv = C_a  dx / max|u|          dt_g = C_g  sqrt(dx / g)
dt_ν   = C_ν  dx² / max ν_eff       dt_σ = C_σ  sqrt(ρ dx³ / σ)
```

The advective maximum is taken over faces that hold liquid only. It used to
include the velocity extension into the air band, where linear extrapolation
reached three to nine times the water speed; no liquid ever crosses a face
whose donor is dry, so those faces no longer set the step. A step is accepted
only if `α` stays bounded, the pressure solve converged and the RMS divergence
gate holds; otherwise the solver raises.

## 3. One step, in order

`Solver3D.step` in `physics3d/solver.py`:

1. **Geometry and ledger.** Enforce solid cells (no water in solids). Build
   the liquid operator of section 4 from the current `α`, and open the energy
   ledger on it.
2. **Transport.** Reconstruct the interface (section 5), compute geometric
   fluxes of `α` through faces, and advect momentum with the same fluxes so
   water and velocity cannot decouple at the interface
   (`physics3d/advection.py`, `unsplit_advection.py`, `swept_prism.py`,
   `momentum.py`).
3. **Contact line.** Conservative exchange of volume between already-mixed
   cells within a physical band of the solid, with hysteresis
   (`contact_line.py`).
4. **Dissipative closures on fixed geometry.** Viscosity and sub-grid stress
   (`viscosity.py`, `liquid_viscosity.py`), rough-wall stress (`wall_stress.py`,
   `liquid_wall.py`), porous drag (`porous.py`, `gravel.py`). After each, the
   liquid kinetic energy is measured and the step aborts if a dissipative term
   added energy beyond float32 roundoff.
5. **Wave source and wisps.** The relaxation zone imposes the incident field
   and absorbs outgoing waves (`relaxation.py`, `waves.py`); disconnected
   roundoff-scale water is removed and counted (`wet_dry.py`). Both change
   `α` and both are booked as boundary work.
6. **Projection.** Solve the pressure equation on the rebuilt liquid geometry
   and correct the velocity to divergence-free (`projection.py`,
   `free_surface.py`, `liquid_pressure.py`, `pcg.py`, `multigrid.py`).
   Gravity enters here through the same operator in the thin-liquid path.
7. **Porous storage.** Infiltration into and drainage out of the bed, each
   exactly the volume it removes from or returns to the water (`porous.py`).
8. **Ledger, foam, wetness.** Close the energy budget (`energy.py`); update
   the passive foam field from the measured dissipation (`foam.py`) and the
   solid wetness (`solid_wetness.py`), neither of which touches `α` or `u`.

## 4. Pressure

### Regular grid

Matrix-free preconditioned conjugate gradients with geometric multigrid
(`pcg.py`, `multigrid.py`). The hierarchy halves all three axes; restriction
is the volume adjoint of trilinear prolongation, `R = Pᵀ / 8`; two weighted
Jacobi smooths (`ω = 2/3`) each side of a reference solve on the coarsest
grid. One V-cycle reduces the residual by 0.32 on `16 × 8 × 8` and 0.043 on
`64 × 32 × 32`; on a broadband right-hand side the iteration count fell from
45–69 (Jacobi) to 6. The moving free surface and the solids are masks, and the
hierarchy coarsens the masks and rediscretises the atmospheric and solid
coefficients at every level. Treating the problem as an unmasked Laplacian
stalled at 240 iterations on the 12.5 mm benchmark; with masks it converged in
11.

### Thin water: the mimetic liquid operator

A swash sheet over pebbles is a few millimetres thick on a 10 mm grid. With
cell-centred hydrostatic pressure `p_i = ρ g (η − z_i)`, the two-point
horizontal gradient between neighbours at different heights is
`−ρ g (z_j − z_i) / dx`: a spurious horizontal force where the physical one is
zero. Measured as a control on a cut beach at rest with a flat 34.9 mm
surface, it produced a 6 mm/s current from nothing within 1 ms, surviving two
refinements. Lowering the activation threshold does not fix a geometric error.

The replacement (`liquid_pressure.py`, `liquid_assembly.py`) follows Brezzi,
Lipnikov and Simoncini (2005). For a liquid cell with retained polyhedron
volume `V`, facet areas `A`, outward normals `n`, facet-centroid displacements
`r` from the water centroid, and integrated normal fluxes `q`:

```
D q = Σ q / V
N = A n,   R = r,   Rᵀ N = V I
M N = R                       (symmetric positive mass matrix)
```

`M N = R` makes the adjoint gradient reproduce any affine pressure exactly,
and gravity enters through the same volume integral. The MAC form restricts
the velocity space first (each component affine between its two faces, with
`B` mapping MAC velocities to facet fluxes): kinetic matrix `Bᵀ M B`, liquid
divergence `Bᵀ 1`, gravity `Bᵀ R g`. A stabilisation
`s V/2 · max(0, 1 − V/V_cube) Q` on the projector away from uniform
translation gives exterior velocity variation some inertia on tiny liquid
support; it vanishes in full cells and is independently zeroable, and its
scale is capped at twice the MAC consistency trace per face so that a sliver
cannot acquire a mass thousands of times its liquid volume. Cell pressure is
eliminated locally and continuity gives a symmetric hybrid system solved by
conjugate gradients.

Gates: six sheets at fractions 0.01 to 0.51 all fall at `w = −0.00981 m/s`
after 1 ms to `1e-9`; three sub-cell cut-beach surfaces at rest stay below
`2e-8 m/s`. A consequence: the operator constrains *liquid* divergence while
the old split VOF update used whole-aperture divergence, and those differ
(`2.4e-7 /s` against `0.247 /s` on the same patch), so transport had to
change too (section 5).

### Precision

The thin-liquid pressure and diffusion systems are assembled and solved in
float64 on the GPU; the carried state stays float32 and the float64 work is
accounted. The reason is measured: on the sixth production step, compensated
float32 CG reported a relative residual of `1.7e-7` while the true equation
residual was `7.8e-4`; the float64 control reached a true residual of
`1.9e-7`. Facet centroids, unit-box intersections and the kinetic-matrix
remainder are computed in quadrature precision before storing, because a
`3.7e-11 m³` film loses its affine identity to a single float64 rounding of
an area-over-moment ratio. `precision.py` provides the typed constants Warp
needs: it does not promote literals, and `x = float(0.0)` silently pins `x` to
float32.

### Residue

A cell is resolved liquid when it holds more than 2% of a cell of water or is
full; everything else holding water is residue (`resolved.py`). Residue keeps
its volume, its momentum and its transported velocities; only the pressure
ignores it. The number came from the saved states that set the time step: a
boulder sliver 6% open and 18% full, 1% of a cell, carried 7.8 m/s with a
full pressure unknown on 0.035 mL of water (−11.2 kPa, then −34.7 kPa). The
full-cell clause came from a fresh start from rest: a cell 1/64 open and full
is wall-trapped water with no free surface, and demoting it made the cell
above it see a free surface at 20 cm depth and accelerate into it.

## 5. Interface and transport

### PLIC

Each mixed cell holds a plane whose offset is solved so that it cuts off
exactly `α θ` of the cell (`plic.py`, `polyhedron.py`). Normals come from the
fractions: Youngs where nothing better is supported; signed three-dimensional
height functions on resolved column patches (`height_normals.py`); a
least-squares plane fit (LVIRA) in the neighbourhood of a cut solid
(`cut_normals.py`), ended per lane when its perturbation drops below the
float32 resolution of the objective. On synthetic planes the height-function
and LVIRA normals are exact where Youngs errs 4.7° on this anisotropic grid.
The three-coefficient unit-cube volume formula cancels catastrophically in
float32 when one normal component is negligible; below that scale the kernel
switches to the lower-dimensional closed form.

### Fluxes

Transport is unsplit (`unsplit_advection.py`, `swept_prism.py`): the fluxes
through all faces of a cell are reconstructed from one PLIC geometry, with
each face's swept prism traced by its complete MAC velocity, shared by water
and momentum. Donor-water and receiver-capacity budgets keep overlapping
swept slabs from overdrawing a thin donor; this is a positivity limiter, and
its diagnostics say so. Momentum flux velocities come from wet neighbours
only and blend toward the donor's own velocity, because a slope-limited
reconstruction that differs from the donor left thin interface water with
`u + F (u − û) / (m − F)`: faces amplified by up to 7.8 in one flux. The
directionally split path (`advection.py`, `vof_flux.py`) remains for the
regular-grid projection.

### Velocity extension

Velocities are extended into a narrow air band so the swept prisms have a
field to trace (`extrapolation.py`). The extension is constant from the owner
faces, not linear: the linear form reached 22 m/s in the air above 1.6 m/s
water and set the time step.

## 6. Waves, solids and closures

### Wave generation and absorption

Each inlet component is finite-depth potential theory (`waves.py`): the
wavenumber from `ω² = g k tanh(k h₀)`, the alongshore wavenumber from the
periodic strip `k_y = 2π m / L_y` (an integer `m`, so the field is exact
across the seam), `k_x = sqrt(k² − k_y²)`, evanescent requests rejected,
second-order Stokes corrections on the steep long components:

```
η    = a cos φ
u_h  = a ω cosh(k (z + h₀)) / sinh(k h₀) · k̂ cos φ
w    = a ω sinh(k (z + h₀)) / sinh(k h₀) · sin φ
```

The target is imposed through a relaxation zone,
`Q ← (1 − χ) Q_solved + χ Q_target`, with a smooth weight `χ` that is zero at
the interior edge and one at the wall; the same zone absorbs outgoing waves
(`relaxation.py`, with PLIC-footprint quadrature of the target in
`relaxation_geometry.py` and `relaxation_rays.py`). Gates: a linear wave
through the tank against the finite-depth relation; an oblique pair at
±36.5° measured at 35.2° and 34.2° with 3.8% of alongshore power outside the
requested modes. The relaxation target keeps its crest-to-trough height below
`H / d = 0.78` of the depth, a guard against injecting an already-broken wave
and crediting the breaker to resolved physics.

### Closures

All default off, each with its own gate and its own line in the ledger.

- **Viscosity and sub-grid stress** (`viscosity.py`, `liquid_viscosity.py`).
  Molecular viscosity and a bounded Smagorinsky viscosity (`C_s = 0.12` in
  production) as one symmetric graph Laplacian per MAC lattice, each edge
  conductance the smaller neighbouring liquid face mass, so a closed aperture
  or the free surface cannot acquire momentum. Gates: closed periodic shear
  loses kinetic energy, more viscosity loses it faster, rigid translation gives
  exactly zero sub-grid viscosity, a larger `C_s` reduces resolved enstrophy.
- **Wall stress** (`wall_stress.py`, `liquid_wall.py`). Regularised
  Navier-slip damping of tangential velocity only, separate slip lengths for
  beach and boulder. The force is paired with the velocity sample through its
  adjoint, solving `(M + dt Fᵀ F) δ = −dt Fᵀ F u`, so the work
  `ρ dt |F u_after|²` is a positive squared norm; cell-centred damping
  averaged to faces was measured to inject `1.06e-5 J` into a closed metric.
- **Contact line** (`contact_line.py`). Volume exchange between already-mixed
  cells within a physical band of the solid, with hysteresis; closes volume
  to `4.4e-16 m³`.
- **Porous bed** (`porous.py`, `gravel.py`). A finite storage `S` with
  declared capacity bound to the water table, infiltration that subtracts
  exactly what it adds, drainage reported as an open-system flux, and drag.
  For the gravel bed the laws are derived from a 3 cm grain at porosity 0.38:
  Ergun permeability `k = d² φ³ / (150 (1 − φ)²) = 8.6e-7 m²`, Forchheimer
  coefficient 0.61, gravity percolation at 0.121 m/s; and because a 10–15 mm
  sheet flows *between* 30 mm grains rather than over them, a canopy drag
  `½ C_d a |U|` with `a = 3 (1 − φ) / (2 d₅₀) = 31 /m`, `C_d = 1`, acting
  within one grain of the bed. The fill–saturate–drain gate closes the
  combined budget below `2e-11 m³`.
- **Surface tension** (`surface_tension.py`). Curvature from the PLIC
  interface through signed height functions on supported patches and local
  normal fitting elsewhere, entering both the pressure right-hand side and the
  same face gradient of the correction. Estimators refuse `|κ| h > 4`;
  refused cells apply no invented curvature and are counted. Gates: a static
  25 mm drop against Young–Laplace (`κ = 80 /m`, `Δp = 5.76 Pa`) with mean
  curvature error 0.17% at `32³` and 0.088% at `48³`, parasitic currents
  halving from 34 to 16 µm/s under refinement; an exactly planar interface
  with exactly zero curvature; a standing wave isolating the added term
  `σ k³ tanh(k h) / ρ` with error moving from −3.5% to +0.06% under
  refinement. The capillary time-step limit scales as `dx^{3/2}`.
- **Foam** (`foam.py`) and **wetness** (`solid_wetness.py`). Passive. Foam is
  sourced in the interface band from the measured direct dissipation and from
  positive numerical loss above a running high-water mark, advected by the
  flow, decayed on 0.8 s. Wetness is attached to the solid surfaces. A gate
  compares `α`, `u`, `v`, `w` and `p` bit for bit with each closure on and
  off.

### The energy ledger

`energy.py` books resolved kinetic plus potential energy, the direct work of
each closure, the boundary work of the wave source and porous exchange, and a
signed `numerical_residual` that is never repaired. A closed viscous decay
closes the ledger within `2e-4` of its direct loss. On the third capture the
ledger reads 940 J of boundary work, 48 J wall, 61 J canopy drag, 37 J
sub-grid stress and 169 J unexplained numerical work, 18% of the boundary
work; that number is open, see `results.md`.

## 7. Presentation

### Surfaces

Two surfaces are extracted from every capture (`render/surface_mesh.py`,
`render/liquid_surface.py`). The exact PLIC boundary is the boundary of the
union of the per-cell water polyhedra; it is the reference surface and holds
every millilitre, down to one-cell films. The nodal isosurface is the
`α = 0.5` level set of an open-fraction-weighted nodal field, extracted with
Lewiner marching cubes; it has no cell-face patches by construction and holds
no surface where water is thinner than about half a cell.

### Why the bore has stairs, and what is done about it

On a curved surface, two neighbouring volume-exact planes do not meet on their
shared face, so the exact boundary closes each gap with a patch of the cell
face. A census of those patches on the saved production states found three
populations: a floor on smooth water (0.5–0.9% of the forcing band's pixels,
p90 mismatch 0.05–0.12 mm, inherent to one plane per cell), the breaking bore
(3.3–6.2% of visible pixels, columns with two to four stacked partial cells
whose planes disagree by 24° in the median), and thin films on the swash edge
and boulder flanks. Fourteen reconstruction variants from the literature
(Youngs, MYC, ELVIRA, LVIRA, plicRDF, Mosso–Swartz, isoAlpha) were refitted on
the saved states; none lowered the bore stairs by more than a tenth of a
percentage point, because in those columns no single plane per cell fits the
stencil. Reference codes do not obtain meeting planes either: isoAlpha is
documented as discontinuous across faces, and Basilisk's viewer enlarges every
facet by 1.1 to close the gaps.

`physics3d/resolvedness.py` therefore measures, for every cell with a plane,
the facet-area-weighted angle between its normal and its interface
neighbours' and the RMS mismatch between its plane continued into the 27-cell
stencil and the water those cells hold. A cell is unresolved above 15° or
0.1; a plan column is a mixture column when some four-cell window holds two
or more partial cells of which at least one is unresolved. Because entrained
air outlives the instant the planes disagree, the indicator carries the foam
closure's time constant: `indicator = max(mask, previous · exp(−dt / 0.8 s))`,
aerated at or above 0.5.

`render/aerated_mixture.py` and `render/mixture_mask.py` build one composite
surface per column: the nodal isosurface wherever it exists, the exact PLIC
facets where it does not (so the swash edge and thin films remain the
solver's), the solver's own passive foam field gated by the indicator in
mixture columns, and three named facet rules that leave residue cells,
boulder-footprint columns and sub-half-cell boulder films to the wetness
shading, each with the water it leaves undrawn reported per frame (at most
0.21 L, 4.9 L and 1.6 L per frame over the third capture). No extra
whiteness is painted. The exact PLIC boundary is saved with every capture.

### Rendering

`render/mitsuba_water.py` puts the composite mesh into Mitsuba as a smooth
dielectric (`n = 1.333`) bounding a purely absorbing medium, so Fresnel, Snell
and Beer–Lambert apply at every actual crossing and along every actual
in-water segment; a folded sheet is entered, exited and re-entered with no
height-field rule. Gates: an emitter through a planar mesh at 2048 samples per
pixel against the closed form, and a vertical ray through a C-shaped volume
observing four ordered crossings with alternating normal signs. The beach and
boulders are the same analytic solids as the solver's, with measured alignment
between cut cells and tessellation (`render/geometry_alignment.py`).
Offshore of the box the far water is the 2-D shallow-water model's surface,
joined to the local mesh at a camera-phase seam (`render/offshore_join.py`);
that is why `shoreloop/physics/` is still in the tree.

## 8. Verification

The quick suite (`tests/`) holds the governing-property tests: compatible
operators and their symmetry, PLIC volume exactness, bounded and conservative
transport, hydrostatic rest on cut beaches, no penetration of solids, the
multigrid contraction, the dissipativity of every closure, ledger closure,
relaxation-zone absorption, resolvedness on synthetic planes and stacked
columns, surface extraction topology, and the Mitsuba optics against closed
forms. Two small saved fixtures (`tests/physics3d/data`, `tests/render/data`)
pin regressions found on production states: a rank-deficient kinetic matrix
in a five-facet boulder sliver, and a 30 µm periodic edge contact in a raw
PLIC surface.

The named integrations are the scenes in `tools/run_wave_tank3d.py`: a linear
wave through the tank, run-up on a plane beach, shoaling, the natural breaker
on a quasi-2-D tank (first overturning with no breaker force, no crest
clipping and no prescribed collapse; halving the cell size moved the breaker
by 5 cm and 60 ms and kept it), wash around one boulder (5.6 cm upstream head,
2.45 m/s side wrap, a 1.2 m separated region, vorticity 95 /s, no wake
prescribed), and the boulder cluster. `tools/run_flat_box_breaker.py` checks
the free dynamics against closed forms: a dam-break bore at 1.83 m/s against
1.87 from the hydraulic-jump relation, numerical energy share 1.1%.

Convergence of a breaking wave is not pointwise. Across a three-grid
hierarchy at 25/20, 16.7/13.3 and 12.5/10 mm the impact path shifted by 49
and 72 mm RMS and the thinnest supported sheet reached 5 mm, then 2.5 mm at
quarter spacing, where the capillary length `sqrt(σ / ρ g) = 2.7 mm` sits.
Surface tension went in for that reason and did not stabilise the path (56
and 149 mm with it against 56 and 78 without), while quarter-grid runs with
and without it both showed the same 38–41 mL sealed air pockets that a
one-phase solver cannot compress. What did converge, at every one of 17
compared states, was the coarse-grained liquid volume field: fine-to-coarse
L1 error ratios of 0.53 to 0.73. That is the convergence of the model class,
and the half-spacing grid was selected on it.

## References

- Brezzi, F., Lipnikov, K. and Simoncini, V. (2005). A family of mimetic
  finite difference methods on polygonal and polyhedral meshes.
- Deike, L., Melville, W. K. and Popinet, S. (2016). Air entrainment and
  bubble statistics in breaking waves. *J. Fluid Mech.* 801.
- Hinze, J. O. (1955). Fundamentals of the hydrodynamic mechanism of
  splitting in dispersion processes. *AIChE J.* 1(3).
- Lewiner, T., Lopes, H., Vieira, A. W. and Tavares, G. (2003). Efficient
  implementation of marching cubes' cases with topological guarantees.
  *J. Graphics Tools* 8(2).
- Mostert, W., Popinet, S. and Deike, L. (2022). High-resolution direct
  simulation of deep water breaking waves. *J. Fluid Mech.* 942.
- Scheufler, H. and Roenby, J. (2019). Accurate and efficient surface
  reconstruction from volume fraction data on general meshes. *J. Comput.
  Phys.* 383.
- US Army Corps of Engineers, CERC (1993). Technical Report CERC-93-11, on
  measured breaking ratios behind the `H / d ≤ 0.78` guard.
- NVIDIA Warp, [github.com/NVIDIA/warp](https://github.com/NVIDIA/warp);
  Mitsuba 3, [mitsuba-renderer.org](https://www.mitsuba-renderer.org).
