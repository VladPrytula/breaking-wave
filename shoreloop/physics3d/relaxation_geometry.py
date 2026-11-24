"""Finite-rate column displacement retaining 3-D PLIC footprints.

Partition the XY footprint at every plane/cell-face crossing. Positive
quadrature on those polygons integrates all initial PLIC interval lengths
exactly. Each ray then uses the existing liquid-coordinate map. Final
cell overlaps are quadrature approximations; no volume repair is applied.
The source geometry uses host FP64 through its positive quadrature and
interval integrals, followed by one FP32 state write. Only uncut source
columns are supported; the production forcing zone ends before the beach.
"""

from collections.abc import Callable
from functools import lru_cache

import numpy as np


def plane_fraction(normal: np.ndarray, offset: float) -> float:
    """Stable host counterpart of the factored PLIC volume formula.

    Retain all nonzero normal components, including near-planar footprints.
    The input normal is L1 normalized in unit-cell coordinates.
    """
    a, b, c = sorted(abs(normal))
    offset -= normal[normal < 0].sum()
    total = a+b+c
    if offset <= 0:
        return 0.
    if offset >= total:
        return 1.
    complement = offset > .5*total
    if complement:
        offset = total-offset
    if b == 0:
        value = offset/c
    elif a == 0:
        value = .5*(offset/b)*(offset/c) if offset < b else (offset-.5*b)/c
    elif offset < a:
        value = (offset/a)*(offset/b)*(offset/c)/6
    elif offset >= a+b:
        value = (offset-.5*(a+b))/c
    else:
        pb, pc = max(offset-b, 0), max(offset-c, 0)
        value = (3*offset*(offset-a)+a*a-(pb/a)*pb*pb-(pc/a)*pc*pc)/(6*b*c)
    return 1-value if complement else value


def fit_offsets(normals: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Fit retained volumes in FP64, batching the same 48 scalar bisections."""
    result = np.zeros(len(alpha))
    mixed = (alpha > 0) & (alpha < 1)
    n, target = normals[mixed], alpha[mixed]
    if np.any(~np.any(n, axis=1)):
        raise ValueError("a mixed source cell needs an interface normal")
    a, b, c = np.sort(np.abs(n), axis=1).T
    origin = np.minimum(n, 0).sum(axis=1)
    low, high = origin.copy(), np.maximum(n, 0).sum(axis=1)
    total = a+b+c
    for _ in range(48):
        midpoint = (low+high)/2
        d = midpoint-origin
        complement = d > .5*total
        z = np.where(complement, total-d, d)
        value = np.zeros(len(d))
        active = z > 0
        mask = active & (b == 0)
        value[mask] = z[mask]/c[mask]
        mask = active & (b > 0) & (a == 0) & (z < b)
        value[mask] = .5*(z[mask]/b[mask])*(z[mask]/c[mask])
        mask = active & (b > 0) & (a == 0) & (z >= b)
        value[mask] = (z[mask]-.5*b[mask])/c[mask]
        mask = active & (a > 0) & (z < a)
        value[mask] = (z[mask]/a[mask])*(z[mask]/b[mask])*(z[mask]/c[mask])/6
        mask = active & (a > 0) & (z >= a+b)
        value[mask] = (z[mask]-.5*(a[mask]+b[mask]))/c[mask]
        mask = active & (a > 0) & (z >= a) & (z < a+b)
        za, aa, ba, ca = z[mask], a[mask], b[mask], c[mask]
        pb, pc = np.maximum(za-ba, 0), np.maximum(za-ca, 0)
        value[mask] = (3*za*(za-aa)+aa*aa-(pb/aa)*pb*pb-(pc/aa)*pc*pc)/(6*ba*ca)
        value[complement] = 1-value[complement]
        below = value < target
        low = np.where(below, midpoint, low)
        high = np.where(below, high, midpoint)
    result[mixed] = (low+high)/2
    return result


def _clip(polygon, normal, offset):
    result = []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        da, db = a@normal-offset, b@normal-offset
        if da <= 0:
            result.append(a)
        if da*db < 0:
            result.append((da*b-db*a)/(da-db))
    return np.asarray(result)


@lru_cache(maxsize=8)
def _duffy_rule(order):
    """Shared positive quadrature table; callers only read these arrays."""
    x, w = np.polynomial.legendre.leggauss(order)
    x, w = (x+1)/2, w/2
    u, v = np.meshgrid(x, x, indexing="ij")
    return u, v, w[:, None]*w[None, :]*(1-u)


def _footprint_quadrature(normals, offsets, alpha, order):
    """Positive Duffy/Gauss quadrature on the common plane arrangement."""
    polygons = [np.array(((0., 0.), (1., 0.), (1., 1.), (0., 1.)))]
    for k in np.flatnonzero((alpha > 0) & (alpha < 1)):
        normal = normals[k, :2]
        if not np.any(normal):
            continue
        for z in (0, 1):
            distance = offsets[k]-normals[k, 2]*z
            pieces = []
            for polygon in polygons:
                signs = polygon@normal-distance
                if np.min(signs) < 0 < np.max(signs):
                    pieces.extend((_clip(polygon, normal, distance),
                                   _clip(polygon, -normal, -distance)))
                else:
                    pieces.append(polygon)
            polygons = pieces
    u, v, weight = _duffy_rule(order)
    points, weights = [], []
    for polygon in polygons:
        for a in range(1, len(polygon)-1):
            p, q, r = polygon[[0, a, a+1]]
            area2 = abs(np.linalg.det(np.stack((q-p, r-p))))
            if area2 == 0:
                continue
            samples = p+u[:, :, None]*(q-p)+(1-u[:, :, None])*v[:, :, None]*(r-p)
            points.append(samples.reshape(-1, 2))
            weights.append((area2*weight).ravel())
    return np.concatenate(points), np.concatenate(weights)


def displace(
    alpha: np.ndarray,
    normals: np.ndarray,
    offsets: np.ndarray,
    target_height: Callable[[np.ndarray], np.ndarray | float],
    source_weight: float | Callable[[np.ndarray], np.ndarray],
    *,
    order: int = 8,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """One uncut source column, with height and offsets in cell units.

    target_height and source_weight are sampled at local XY positions. The
    source is their footprint integral, not a column-centre approximation.
    Normals and offsets must fit the supplied alpha in verification precision.
    """
    if source_weight == 0:
        return alpha.copy(), {"ray_count": 0, "input_volume_error": 0., "column_volume_error": 0.}
    xy, weights = _footprint_quadrature(normals, offsets, alpha, order)
    capacity = len(alpha)
    wet_cells = np.flatnonzero(alpha > 0)
    height = int(wet_cells[-1])+1 if len(wet_cells) else 1
    # Dry cells above the last retained interval have identical lower/upper
    # endpoints and contribute nothing to the scatter. Keep full-width volume
    # reductions below, preserving their FP64 summation order exactly.
    lower = np.broadcast_to(np.arange(height), (len(xy), height)).astype(float).copy()
    upper = lower.copy()
    for k, fraction in enumerate(alpha[:height]):
        if fraction == 1:
            upper[:, k] += 1
        elif fraction > 0:
            n, d = normals[k], offsets[k]
            if n[2] == 0:
                upper[:, k] += (xy@n[:2] <= d)
            else:
                cut = np.clip((d-xy@n[:2])/n[2], 0, 1)
                if n[2] > 0:
                    upper[:, k] += cut
                else:
                    lower[:, k] += cut
                    upper[:, k] += 1
    interval_length = upper-lower
    length = np.zeros((len(xy), capacity))
    length[:, :height] = interval_length
    before = weights@length
    # Every interval is affine on each polygon, so this checks the actual
    # input geometric contract, independently of the after-source integral.
    input_error = np.max(abs(before-alpha))
    if input_error > 5e-11:
        raise ValueError(f"PLIC footprints do not reproduce source alpha: {input_error:g}")
    total = length.sum(axis=1)
    rate_weight = np.broadcast_to(
        source_weight(xy) if callable(source_weight) else source_weight, total.shape,
    )
    assert np.all((rate_weight >= 0) & (rate_weight <= 1))
    target = np.broadcast_to(target_height(xy), total.shape)
    assert np.all((target >= 0) & (target <= capacity))
    cumulative = (np.cumsum(length, axis=1)-length)[:, :height]
    ratio = np.zeros_like(total)
    np.divide(target, total, out=ratio, where=total > 0)
    # Increment form preserves an exactly resting, connected ray without
    # subtracting or adding rounded fractions of its unchanged endpoints.
    bottom = lower+rate_weight[:, None]*(ratio[:, None]*cumulative-lower)
    top = upper+rate_weight[:, None]*(ratio[:, None]*(cumulative+interval_length)-upper)
    dry = total == 0
    # The first interval receives newly introduced water on a dry ray.
    top[dry, 0] = rate_weight[dry]*target[dry]
    # Scatter only the cells crossed by nonempty displaced intervals. This
    # keeps every contribution positive while avoiding an nz-by-nz scan.
    wet = top > bottom
    ray_weights = np.broadcast_to(weights[:, None], top.shape)[wet]
    bottom, top = bottom[wet], top[wet]
    first = np.floor(bottom).astype(int)
    stop = np.ceil(top).astype(int)
    after = np.zeros(capacity)
    for shift in range(np.max(stop-first, initial=0)):
        cell = first+shift
        crossed = (cell < stop) & (cell < capacity)
        k = cell[crossed]
        overlap = np.minimum(top[crossed], k+1)-np.maximum(bottom[crossed], k)
        after += np.bincount(k, weights=ray_weights[crossed]*overlap, minlength=capacity)
    expected = weights@((1-rate_weight)*total+rate_weight*target)
    return after, dict(ray_count=len(xy), input_volume_error=float(input_error),
                       column_volume_error=float(after.sum()-expected))
