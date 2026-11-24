# Results

The two accepted captures, where they came from, what their ledgers say, and
what is still open. Everything numerical here is read from the files in
[`media/`](../media/) or from the measurements recorded when the runs were
accepted.

## The scene

A `4.8 × 6.0 × 0.96 m` box, periodic alongshore, 0.22 m of still water, a
beach of slope 0.15 from a toe at `x = 1.75 m`, five rotated superquadric
boulders at the shore edge, a sunset camera. The incident field is the AW-1103
group: 23 exactly periodic components on a 2.4 s loop (temporal modes 2, 3
and 4, alongshore modes up to ±6, 62.5% of the oblique energy travelling
toward `+y`, one fixed random phase draw), scaled by 1.45 from its original
calibration and held under the `H / d ≤ 0.78` admissibility guard at the
relaxation boundary (measured 0.69). Production grid `384 × 224 × 96`, cells
`12.5 × 26.8 × 10 mm`, 8.3 million cells, liquid-mimetic pressure,
height-function interface normals, PLIC-ray wave source, swept-prism
transport, resolved-liquid fraction 2%, molecular and Smagorinsky viscosity,
Navier-slip wall stress, contact-line regularisation, porous storage and
drag, foam and wetness on. Surface tension off (the production grid sits
above the capillary scale; see `solver.md` §8).

The two captures differ only in the bed:

| | First capture | Third capture |
| --- | --- | --- |
| Bed | AW-1104: slip-length plane, uniform porous layer with tuned permeability, grid-scale drag bracket | Gravel: Ergun–Forchheimer storage bound to the water table and grain-canopy drag, both derived from a 3 cm pebble at porosity 0.38 |
| Accepted | 2026-10-13 | 2026-10-22 |
| Source commit | `cee4a08a1f4d0300cd6d7bec2acec2079cffcb92` | `0dba9ac2b78d1afd0e3c351b1522c68401f92a92` (this tree) |
| Steps, wall time | 6,656, 77.9 h | 6,632, 62.0 h |
| Minimum `dt` | 0.31 ms | 0.24 ms |
| Maximum Courant | 0.351 | 0.365 |
| Maximum pressure iterations | 449 | 491 |
| Relative mass residual | `−2.81e-7` | `−3.28e-7` |
| Maximum divergence (float64 solution; stored net flux per cell volume) | `2.1e-5 /s` | `2.1e-5 /s` |
| Water in solid cells | 0 | 0 |
| Fastest face | 9.4 m/s | 13.3 m/s |
| Mean shoreline excursion | 0.181 m | 0.270 m |
| Infiltrated / drained | 17.9 L / 2.3 L | 5.2 L / 2.9 L |
| Longest connected retreat after a run-up maximum | 0.04 s | 1.12 s |

Both ran 0–6.8 s with captures every 40 ms (171 frames) and both are
presented through the same resolvedness-gated `nodal_body` composite with
the offshore join and foam shading. Neither is a loop: the breaking,
reconnection, swash and porous exchange carry memory from one wave to the
next, and the periodicity probes said so; the project recorded a non-closing
physical response as a valid result and dropped the loop.

## What the captures show

**Third capture** (`media/third-capture-0-6p8s-25fps.mp4`). Three plunging
events, peaking at 1.3 s, 3.9 s and 6.2 s with 610, 1,167 and 985
downward-normal interface cells, crest water at 10–13 m/s in the plunges,
secondary overturning as each roller runs up. The first wave's uprush ran
24 cm against the first capture's 18; the later swashes added 8–10 cm on a
beach that stayed wet, and the backwash drained as a sheet thinning in place
rather than a retreating edge, which is what the gravel closures were derived
to do. The lip at 3.9 s is the clearest; `media/third-capture-03_88s-linear.png`
is that frame in linear light.

**First capture** (`media/first-capture-0-6p8s-25fps.mp4`). The same spectrum
on the earlier bed: plunges peaking at 3.88 s and 6.28 s (1,136 and 1,162
downward-normal cells), a shorter swash, a backwash that retreats as an edge.
Accepted as the first deliverable before the gravel physics existed; kept
because it is the other half of a controlled comparison of bed closures.
`media/first-capture-03_96s-linear.png` is its 3.96 s frame.

The second capture, between these two, is not here. It ran the spectrum
calibrated frame by frame to a measured gentle pebble-beach reference
(fronts every 2 s, 6–10 cm high, collapsing at the waterline) on the gravel
bed, 55 h, every gate clean, and matched the measured swash excursion and
front speed more closely than anything else the project produced. It was
rejected by eye: 202 downward-normal cells, no plunging jet, nothing that
read as breaking. "Physically faithful to a measured reference" and
"visually legible breaking" turned out to be different targets.

## Energy ledgers

From `cumulative_energy_j` in the metrics files, joules over the whole run:

| | First | Third |
| --- | --- | --- |
| Boundary work (wave source and porous exchange) | 932.9 | 939.6 |
| Molecular loss | 0.4 | 0.5 |
| Sub-grid stress | 31.8 | 37.2 |
| Wall stress | 37.5 | 48.1 |
| Porous drag (first: Darcy–Forchheimer; third: canopy) | 110.1 | 60.7 |
| Numerical residual | 149.6 (16%) | 169.0 (18%) |

The residual is the part of the boundary work that no closure booked. It is
a loss, not a gain, in both runs, and no gate covers it. The post quotes 2.7%
for the first capture; that figure belongs to a development-grid probe of the
first capture's bed laws under the reference spectrum, not to the capture
itself, whose own ledger says 16%. Whether the plunging jets lose energy the
operators do not account for, or the sliver-capped metric books their splash
as numerical loss, is the first open question below.

## Gates

Each run's `aw1104_acceptance` block records the six named criteria: mean
shoreline excursion at least 0.15 m, a negative connected sheet for at least
0.5 s after the run-up maximum, separated outer corridors that wash then
reverse, a localised breaker component, non-negative dissipative work, and
the incident boundary height at most 0.78 of the depth. The third capture
passes all six. The first records `false` on the connected-retreat criterion:
its longest visible connected retreating sheet after a run-up maximum lasted
one capture interval, 0.04 s, against 1.12 s on the gravel bed. That is the
measured difference between the two beds, and the reason the gravel closures
were derived.

Numerical gates over every step of both runs: relative mass residual below
`1e-6`, float64 solution divergence below `1e-12 /s`, the stored float32
field's net flux per Cartesian cell volume at most `2.1e-5 /s`, `α` in
`[0, 1]`, zero water in solids, every dissipative term non-negative beyond
float32 roundoff.

## Open items

1. **The 16–18% numerical energy share.** Whether it is physical loss in the
   plunging impact that the resolved operators cannot represent, or an
   accounting artefact of the capped sliver metric, is undecided.
2. **Phantom droplets.** A 0.6 mL droplet 2.2% full, just over the resolved
   fraction, whose water lies away from its exit face so the swept-prism flux
   carries nothing, while its air faces carry the extension at the local
   speed bound: 4.5 m/s, stationary for 240 ms, touching no gate, bounding
   the time step.
3. **The swash-edge facet crust.** On the leading 25–50 mm of a swash tongue
   the nodal isosurface has no support and the exact PLIC facets are drawn.
   They are small grid-aligned boxes if you look closely.
4. **Uniform whitewater tone.** The aerated composite is a closure on the
   solver's passive foam field with one time constant; the indicator still
   flips about a thousand columns per capture at breaking onsets.
5. **Sealed air pockets.** Quarter-spacing runs resolved 38–41 mL closed
   cavities inside the breaker that a one-phase solver assigns atmospheric
   pressure. A low-Mach pocket pressure or an air phase is the next physics,
   and the air phase is a different solver.

## Reproduce

The third capture, from this tree, on the DGX Spark. Days of GPU time;
launch it detached and check the log for device initialisation before
leaving it.

```bash
mkdir -p out/third_capture
tmux new-session -d -s third-capture \
  'PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=.:tools \
   .venv/bin/python tools/capture_boulder_interaction_sequence.py \
     --output-dir out/third_capture/sequence \
     --incident aw1103 --bed gravel \
     --grid-scale 2 --end-time 6.8 --capture-start 0 --interval .04 \
     --interface-reconstruction height_function \
     --wave-source-reconstruction plic_rays --wave-source-quadrature 8 \
     --vof-flux-reconstruction swept_prism \
     --render-surface plic_boundary --save-nodal-surface \
     --save-wetness --width 1920 --height 1080 --spp 512 \
     --no-ripples --no-foam --offshore-handoff-x 0 --diagnostic \
     > out/third_capture/run.log 2>&1'
tail -n 30 out/third_capture/run.log
```

Compare `out/third_capture/sequence/configuration.json` with
`media/third-capture-configuration.json`: the `scene` blocks must agree apart
from `output_dir`. The GPU trajectory is not bitwise deterministic across
runs, so metrics agree to the gates, not to the digit.

Then the static geometry, the composites and the frames:

```bash
export PYTHONPATH=.:tools
.venv/bin/python tools/export_scene_geometry.py --run out/third_capture --out out/third_capture/geometry.npz
.venv/bin/python tools/compose_aerated_frames.py --run out/third_capture \
    --geometry out/third_capture/geometry.npz --composite nodal_body \
    --times $(seq -f %.2f 0 0.04 6.8) --out out/third_capture/composites
.venv/bin/python tools/render_saved_frames.py --run out/third_capture --out out/third_capture/renders \
    --mesh out/third_capture/composites/surface_*_nodal_body.npz \
    --time $(seq -f %.2f 0 0.04 6.8) --width 1920 --height 1080 --spp 512
ffmpeg -framerate 25 -pattern_type glob -i 'out/third_capture/renders/*_tonemapped.png' \
    -c:v libx265 -pix_fmt yuv420p out/third_capture/third_capture_0_6p8s_25fps.mp4
```

Composition takes minutes on the CPU; rendering took 4.3 h on the GPU for
171 frames (about 67 s of Mitsuba per frame plus the join). Judge the linear
PNGs, not the video: 4:2:0 chroma subsampling discards the specular detail
first.

The first capture needs the development repository at its source commit; its
bed closures were superseded in this tree.
