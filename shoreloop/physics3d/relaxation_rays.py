"""CUDA ray integration of the unchanged host footprint quadrature.

All ray arithmetic and accumulations are FP64, with contraction disabled.
Atomic reduction order differs from NumPy; compare before the FP32 state write.
No saturation clamp, redistribution or volume correction follows integration.
"""
import numpy as np
import warp as wp
from shoreloop.physics3d.relaxation_geometry import _footprint_quadrature

wp.set_module_options({"enable_backward": False, "fuse_fp": False})


@wp.func_native("return ::expm1(x);")
def _expm1(x: wp.float64) -> wp.float64:
    ...


@wp.struct
class Source:
    x0: wp.float64
    y0: wp.float64
    z0: wp.float64
    dx: wp.float64
    dy: wp.float64
    dz: wp.float64
    width: wp.float64
    ramp: wp.float64
    level: wp.float64
    time: wp.float64
    rate: wp.float64
    ny: int
    nz: int
    impose: int


@wp.func
def _interval(k: int, fraction: wp.float64, n: wp.vec3d, d: wp.float64, xy: wp.vec2d) -> tuple[wp.float64, wp.float64]:
    lo = wp.float64(k)
    hi = lo
    if fraction == wp.float64(1.0):
        hi += wp.float64(1.0)
    elif fraction > wp.float64(0.0):
        dot = xy[0]*n[0]+xy[1]*n[1]
        if n[2] == wp.float64(0.0):
            if dot <= d:
                hi += wp.float64(1.0)
        else:
            cut = wp.clamp((d-dot)/n[2], wp.float64(0.0), wp.float64(1.0))
            if n[2] > wp.float64(0.0):
                hi += cut
            else:
                lo += cut
                hi += wp.float64(1.0)
    return lo, hi


@wp.kernel
def _integrate(alpha: wp.array2d(dtype=wp.float64), normals: wp.array2d(dtype=wp.vec3d),
               offsets: wp.array2d(dtype=wp.float64), heights: wp.array(dtype=int),
               xy_values: wp.array(dtype=wp.vec2d), weights: wp.array(dtype=wp.float64),
               columns: wp.array(dtype=int), components: wp.array2d(dtype=wp.float64),
               source: Source, before: wp.array2d(dtype=wp.float64),
               after: wp.array2d(dtype=wp.float64), expected: wp.array(dtype=wp.float64)):
    ray = wp.tid()
    col = columns[ray]
    xy, weight = xy_values[ray], weights[ray]
    total = wp.float64(0.0)
    for k in range(heights[col]):
        lo, hi = _interval(k, alpha[col, k], normals[col, k], offsets[col, k], xy)
        length = hi-lo
        total += length
        if length > wp.float64(0.0):
            wp.atomic_add(before, col, k, weight*length)
    i, j = col//source.ny, col%source.ny
    x = (source.x0+wp.float64(i)*source.dx)+xy[0]*source.dx
    y = source.y0+(wp.float64(j)+xy[1])*source.dy
    s = wp.clamp((source.width-(wp.float64(i)+xy[0])*source.dx)/source.width, wp.float64(0.0), wp.float64(1.0))
    spatial = s*s*(wp.float64(3.0)-wp.float64(2.0)*s)
    w = spatial
    if source.impose == 0:
        w = -_expm1(-source.rate*spatial)
    elevation = wp.float64(0.0)
    for mode in range(components.shape[0]):
        phase = components[mode, 0]*x+components[mode, 1]*y-components[mode, 2]*source.time+components[mode, 3]
        elevation += components[mode, 4]*wp.cos(phase)+components[mode, 5]*wp.cos(wp.float64(2.0)*phase)
    ramp = wp.float64(1.0)
    if source.ramp > wp.float64(0.0):
        ramp = wp.clamp((x-source.x0)/source.ramp, wp.float64(0.0), wp.float64(1.0))
    envelope = ramp*ramp*(wp.float64(3.0)-wp.float64(2.0)*ramp)
    target = wp.clamp((source.level+envelope*elevation-source.z0)/source.dz, wp.float64(0.0), wp.float64(source.nz))
    wp.atomic_add(expected, col, weight*((wp.float64(1.0)-w)*total+w*target))
    if total == wp.float64(0.0):
        top = w*target
        for k in range(int(wp.ceil(top))):
            wp.atomic_add(after, col, k, weight*(wp.min(top, wp.float64(k+1))-wp.float64(k)))
    else:
        ratio = target/total
        cumulative = wp.float64(0.0)
        for k in range(heights[col]):
            lo, hi = _interval(k, alpha[col, k], normals[col, k], offsets[col, k], xy)
            length = hi-lo
            if length > wp.float64(0.0):
                bottom = lo+w*(ratio*cumulative-lo)
                top = hi+w*(ratio*(cumulative+length)-hi)
                for cell in range(int(wp.floor(bottom)), int(wp.ceil(top))):
                    if cell < source.nz:
                        overlap = wp.min(top, wp.float64(cell+1))-wp.max(bottom, wp.float64(cell))
                        wp.atomic_add(after, col, cell, weight*overlap)
            cumulative += length


def integrate_source_rays(alpha, normals, offsets, parameters, components, *, order, device):
    footprints = [_footprint_quadrature(n, d, a, order) for a, n, d in zip(alpha, normals, offsets)]
    count = np.array([len(xy) for xy, weight in footprints])
    columns = np.repeat(np.arange(len(alpha)), count).astype(np.int32)
    xy = np.concatenate([q[0] for q in footprints])
    weights = np.concatenate([q[1] for q in footprints])
    heights = np.array([int(np.flatnonzero(a > 0)[-1])+1 if np.any(a > 0) else 1 for a in alpha], np.int32)
    p = Source()
    for key, value in parameters.items():
        setattr(p, key, value)
    before = wp.zeros(alpha.shape, dtype=wp.float64, device=device)
    after = wp.zeros_like(before)
    expected = wp.zeros(len(alpha), dtype=wp.float64, device=device)
    wp.launch(_integrate, dim=len(xy), inputs=[wp.array(alpha, dtype=wp.float64, device=device),
        wp.array(normals, dtype=wp.vec3d, device=device), wp.array(offsets, dtype=wp.float64, device=device),
        wp.array(heights, dtype=int, device=device), wp.array(xy, dtype=wp.vec2d, device=device),
        wp.array(weights, dtype=wp.float64, device=device), wp.array(columns, dtype=int, device=device),
        wp.array(components, dtype=wp.float64, device=device), p, before, after, expected], device=device)
    result = after.numpy()
    input_error = float(abs(before.numpy()-alpha).max())
    residual = float(abs(result.sum(axis=1)-expected.numpy()).max())
    if input_error > 5e-11:
        raise ValueError(f"PLIC footprints do not reproduce source alpha: {input_error:g}")
    return result, dict(rays=len(xy), maximum_input_volume_error=input_error,
                        maximum_column_source_residual=residual)
