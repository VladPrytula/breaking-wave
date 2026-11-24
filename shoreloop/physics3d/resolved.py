"""The one definition of resolved liquid shared by pressure, assembly and transport.

A cell is resolved liquid when it holds more than ``fraction`` of a cell of
liquid (``alpha*theta``) or when it is full (``alpha >= 1 - fraction``): a
full cut cell is wall-trapped liquid with no free surface, however small its
open part, and must keep its pressure unknown or the cell above it would see
a free surface at depth. Everything else that holds liquid is residue: a film
with a free surface and too little liquid to govern a face velocity.
"""

import numpy as np
import warp as wp


@wp.func
def resolved_liquid(alpha: wp.float32, theta: wp.float32, fraction: wp.float32) -> int:
    if theta <= wp.float32(0.0):
        return 0
    if alpha * theta > fraction or alpha >= wp.float32(1.0) - fraction:
        return 1
    return 0


def resolved_liquid_mask(alpha, theta, fraction):
    """Host form of :func:`resolved_liquid` on interior arrays."""
    alpha = np.asarray(alpha, np.float32)
    theta = np.asarray(theta, np.float32)
    fraction = np.float32(fraction)
    return (theta > 0) & ((alpha * theta > fraction) | (alpha >= np.float32(1.0) - fraction))
