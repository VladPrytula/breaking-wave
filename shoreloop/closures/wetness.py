"""Optical wetness of the bed -- an explicit closure state (plan.md section 10).

    w(x, y, t) in [0, 1]
    submerged:  dw/dt = (1 - w) / tau_wet     (fast: seconds)
    exposed:    dw/dt = -w / tau_dry          (slow: tens of seconds)

The moving dark band of wet pebbles at the swash edge is one of the strongest
cues in the reference video, and it is *memory*: the bed stays dark long after
the water has left. That memory is why w must be part of the periodic-orbit
state vector -- with tau_dry ~ tens of seconds the transient decays at only
exp(-T/tau_dry) per loop period, much slower than the hydrodynamics contract.

Measured target (docs/reference-video.md): wet pebbles are 1.26x darker than
dry, i.e. an albedo ratio of ~0.79 -- far less than dry-vs-wet asphalt
intuition suggests, because under an overcast sky the dry stones are not
bright either. The albedo mapping lives in the renderer; this module owns only
the state w and its dynamics.

Both branches are integrated implicitly, so the update is unconditionally
stable and w stays in [0, 1] by construction:

    submerged:  w <- (w + dt/tau_wet) / (1 + dt/tau_wet)
    exposed:    w <- w / (1 + dt/tau_dry)
"""

from __future__ import annotations

import warp as wp

from shoreloop.precision import FLOAT, ONE


@wp.kernel
def advance_wetness(
    w: wp.array2d(dtype=FLOAT),
    h: wp.array2d(dtype=FLOAT),
    dt: FLOAT, tau_wet: FLOAT, tau_dry: FLOAT,
    h_wet: FLOAT, ng: int,
):
    i, j = wp.tid()
    I = i + ng
    J = j + ng
    if h[I, J] > h_wet:
        a = dt / tau_wet
        w[I, J] = (w[I, J] + a) / (ONE + a)
    else:
        w[I, J] = w[I, J] / (ONE + dt / tau_dry)
