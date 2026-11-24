`periodic_plic_edge_contact.npz` is an unchanged-coordinate, unchanged-attribute
subset of the production `.76 s` raw PLIC surface from 2026-09-12: triangles
fully inside `x=[1.0, 1.05]` and `y<.06` or `y>5.94`, followed by index
compaction (181 vertices, 251 triangles). It retains the 29.832 micrometre
edge contact at `x=1.024999976`, `y=0/6`, `z=.239970163 .. .239999995`.

The source passes the original incidence-one periodic check. After tiling and
assigning y-coplanar facets to the lower period cut, the contact has three
incident faces at the lower edge and one at the upper edge. Their signed
boundary coefficients are +1 and -1. The old count-only check refuses the
owned period, although every facet and boundary attribute corresponds.
