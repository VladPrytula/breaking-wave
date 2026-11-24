# Pressure-pocket regression

`boulder_pressure_pocket.npz` is a five-cell-wide crop around production cell
`(176, 125, 24)` immediately before the failed pressure projection after step
739, at `t = 1.224473274519979 s`, from the September 23 replay of the
September 13 production run. Spacing is `(4.8/384, 6/224, .96/96) m`.

It stores the water fraction, solid geometry/apertures, and retained PLIC
planes/centroids. These are inputs to the local pressure assembly, not a
stored solution. The central cell has five retained MAC velocities, four
solid facets and one free facet. Its former kinetic matrix had rank four;
inverting it produced NaNs. Tests use the retained planes directly to isolate
the metric from changes to interface reconstruction.
