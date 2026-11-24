"""Build-time float precision switch.

    SHORELOOP_FP64=1 python ...

float32 is the production setting: a ~1e-5 m error floor on 0.1-0.3 m waves is
invisible to any renderer. float64 is the *verification* setting, needed because
three gates cannot be met in float32 at all -- convergence order on smooth
solutions, linear dispersion vs the analytic relation, and AD vs centred finite
differences. See docs/decisions.md, "Precision".

Cost is ~2x, not the ~1/64 that consumer Blackwell's FP64 compute ratio would
suggest, because the solver is memory-bandwidth-bound rather than flop-bound.

## Why every literal in a kernel is wrapped

Warp does **not** promote float literals. Inside a kernel, a bare `2.0` is
float32, so with `a: float64` the expression `a * 2.0` is a hard error:

    RuntimeError: Input types must be the same, got ['float64', 'float32']

Generic kernels (`dtype=Any`) do not help -- the literal stays float32 there
too, verified. So kernel code must write `FLOAT(2.0)` or use one of the typed
constants below. It is verbose, but it is a compile-time error when forgotten
rather than a silent precision loss, which is the right failure mode.

The switch is read at import time and fixes the precision for the whole process.
Mixing precisions in one process is deliberately not supported: it would mean two
copies of every kernel and a way to get them confused.
"""

from __future__ import annotations

import os

import numpy as np
import warp as wp


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0").lower() not in ("0", "", "false", "no", "off")


FP64: bool = _env_flag("SHORELOOP_FP64")

if FP64:
    FLOAT = wp.float64
    NP_FLOAT = np.float64
    vec2, vec3, vec4 = wp.vec2d, wp.vec3d, wp.vec4d
else:
    FLOAT = wp.float32
    NP_FLOAT = np.float32
    vec2, vec3, vec4 = wp.vec2f, wp.vec3f, wp.vec4f

#: Machine epsilon of the active type, for tolerance scaling in tests.
EPS: float = float(np.finfo(NP_FLOAT).eps)

# Typed constants. Named where the name carries meaning, so kernel code reads as
# mathematics rather than as casts.
ZERO = wp.constant(FLOAT(0.0))
ONE = wp.constant(FLOAT(1.0))
TWO = wp.constant(FLOAT(2.0))
FOUR = wp.constant(FLOAT(4.0))
HALF = wp.constant(FLOAT(0.5))
QUARTER = wp.constant(FLOAT(0.25))
SQRT2 = wp.constant(FLOAT(1.4142135623730951))
TWO_PI = wp.constant(FLOAT(6.283185307179586))
FOUR_THIRDS = wp.constant(FLOAT(4.0 / 3.0))


def describe() -> str:
    return f"{'float64' if FP64 else 'float32'} (eps={EPS:.3e})"
