"""One-phase atmospheric free-surface pressure projection.

Pressure unknowns live only at water cell centres.  Water/air faces impose
atmospheric pressure through a ghost-fluid distance estimated from the local
PLIC planes; air/air faces have no pressure action.  A completely filled box
falls back to the closed Neumann null-space treatment.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import warp as wp

from shoreloop.config import BC, Solver3DConfig
from shoreloop.physics3d.boundary import fill_boundaries
from shoreloop.physics3d.extrapolation import _wrapped_cell_index
from shoreloop.physics3d.interface import estimate_interface_normals
from shoreloop.physics3d.pcg import (PCGConfig, PCGResult, _apply_jacobi,
                                     _copy_cell_interior, _form_dot_products,
                                     _form_residual,
                                     _update_direction,
                                     _update_solution_and_residual)
from shoreloop.physics3d.plic import reconstruct_plic
from shoreloop.physics3d.projection import ProjectionConfig, _correct_velocity
from shoreloop.physics3d.solids import (_solid_cell,
                                        enforce_voxel_solids)
from shoreloop.physics3d.state import State3D
from shoreloop.physics3d.surface_tension import (
    SurfaceTensionGeometry,
    reconstruct_surface_tension_geometry,
)


@dataclass(frozen=True)
class FreeSurfaceGeometry:
    """Active pressure cells and inverse cell-to-interface face distances."""

    active: wp.array
    solid: wp.array
    theta: wp.array
    aperture_u: wp.array
    aperture_v: wp.array
    aperture_w: wp.array
    inverse_u_distance: wp.array
    inverse_v_distance: wp.array
    inverse_w_distance: wp.array
    active_cells: int
    dirichlet_faces: int
    surface_tension: SurfaceTensionGeometry | None


@dataclass(frozen=True)
class FreeSurfaceProjectionResult:
    """Pressure convergence and water-only divergence diagnostics."""

    pressure_solve: PCGResult
    converged: bool
    rms_divergence_before: float
    max_divergence_before: float
    rms_divergence_after: float
    max_divergence_after: float
    active_cells: int
    dirichlet_faces: int
    surface_area: float
    max_abs_curvature: float
    max_abs_capillary_pressure: float
    height_function_curvature_cells: int
    unresolved_curvature_cells: int
    # Liquid-mimetic path only. The FP64 solution's divergence before the
    # float32 write-back measures the solve; the stored field's net flux per
    # Cartesian cell volume measures the state. The liquid-relative "after"
    # statistics above divide the write-back rounding by a possibly tiny
    # liquid volume and are diagnostic.
    solution_rms_divergence: float = 0.0
    solution_max_divergence: float = 0.0
    max_cell_divergence: float = 0.0


@wp.kernel
def _initialize_hydrostatic_pressure(
    pressure: wp.array3d(dtype=wp.float32),
    alpha: wp.array3d(dtype=wp.float32),
    theta: wp.array3d(dtype=wp.float32),
    solid: wp.array3d(dtype=wp.int32),
    surface_elevation: wp.float32,
    rho_g: wp.float32,
    z0: wp.float32,
    dz: wp.float32,
    ng: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    if (solid[I, J, K] == 0 and theta[I, J, K] > wp.float32(0.0)
            and alpha[I, J, K] >= wp.float32(0.5)):
        z = z0 + (wp.float32(k) + wp.float32(0.5)) * dz
        pressure[I, J, K] = rho_g * wp.max(
            surface_elevation - z, wp.float32(0.0),
        )
    else:
        pressure[I, J, K] = wp.float32(0.0)


@wp.func
def _active_cell(
    active: wp.array3d(dtype=wp.int32),
    i: int,
    j: int,
    k: int,
    nx: int,
    ny: int,
    nz: int,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    ng: int,
) -> int:
    ci = _wrapped_cell_index(i, nx, periodic_x)
    cj = _wrapped_cell_index(j, ny, periodic_y)
    ck = _wrapped_cell_index(k, nz, periodic_z)
    if ci < 0 or cj < 0 or ck < 0:
        return 0
    return active[ci + ng, cj + ng, ck + ng]


@wp.func
def _cell_pressure(
    pressure: wp.array3d(dtype=wp.float32),
    active: wp.array3d(dtype=wp.int32),
    i: int,
    j: int,
    k: int,
    nx: int,
    ny: int,
    nz: int,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    ng: int,
) -> wp.float32:
    ci = _wrapped_cell_index(i, nx, periodic_x)
    cj = _wrapped_cell_index(j, ny, periodic_y)
    ck = _wrapped_cell_index(k, nz, periodic_z)
    if ci < 0 or cj < 0 or ck < 0:
        return wp.float32(0.0)
    if active[ci + ng, cj + ng, ck + ng] == 0:
        return wp.float32(0.0)
    return pressure[ci + ng, cj + ng, ck + ng]


@wp.kernel
def _build_active_mask(
    alpha: wp.array3d(dtype=wp.float32),
    theta: wp.array3d(dtype=wp.float32),
    solid: wp.array3d(dtype=wp.int32),
    active: wp.array3d(dtype=wp.int32),
    count: wp.array(dtype=wp.int32),
    threshold: wp.float32,
    ng: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    if (solid[I, J, K] == 0 and theta[I, J, K] > wp.float32(0.0)
            and alpha[I, J, K] >= threshold):
        active[I, J, K] = 1
        wp.atomic_add(count, 0, 1)


@wp.func
def _mixed_plane_distance(
    normals: wp.array3d(dtype=wp.vec3),
    offsets: wp.array3d(dtype=wp.float32),
    i: int,
    j: int,
    k: int,
    axis: int,
    direction: wp.float32,
    spacing: wp.float32,
    ng: int,
) -> wp.float32:
    normal = normals[i + ng, j + ng, k + ng]
    denominator = normal[axis] * direction
    if denominator <= wp.float32(1.0e-8):
        return wp.float32(-1.0)
    centre_value = wp.float32(0.5) * (
        normal[0] + normal[1] + normal[2]
    )
    distance_cells = (offsets[i + ng, j + ng, k + ng]
                      - centre_value) / denominator
    if (not wp.isfinite(distance_cells) or distance_cells <= wp.float32(0.0)
            or distance_cells > wp.float32(0.55)):
        return wp.float32(-1.0)
    return distance_cells * spacing


@wp.kernel
def _build_face_inverse_distance(
    active: wp.array3d(dtype=wp.int32),
    solid: wp.array3d(dtype=wp.int32),
    aperture: wp.array3d(dtype=wp.float32),
    normals: wp.array3d(dtype=wp.vec3),
    offsets: wp.array3d(dtype=wp.float32),
    inverse_distance: wp.array3d(dtype=wp.float32),
    dirichlet_count: wp.array(dtype=wp.int32),
    axis: int,
    nx: int,
    ny: int,
    nz: int,
    spacing: wp.float32,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    ng: int,
):
    i, j, k = wp.tid()
    if aperture[i + ng, j + ng, k + ng] <= wp.float32(0.0):
        return
    low_i = i
    low_j = j
    low_k = k
    if axis == 0:
        low_i -= 1
    elif axis == 1:
        low_j -= 1
    else:
        low_k -= 1
    high_i = i
    high_j = j
    high_k = k

    wrapped_low_i = _wrapped_cell_index(low_i, nx, periodic_x)
    wrapped_low_j = _wrapped_cell_index(low_j, ny, periodic_y)
    wrapped_low_k = _wrapped_cell_index(low_k, nz, periodic_z)
    wrapped_high_i = _wrapped_cell_index(high_i, nx, periodic_x)
    wrapped_high_j = _wrapped_cell_index(high_j, ny, periodic_y)
    wrapped_high_k = _wrapped_cell_index(high_k, nz, periodic_z)
    if (wrapped_low_i < 0 or wrapped_low_j < 0 or wrapped_low_k < 0
            or wrapped_high_i < 0 or wrapped_high_j < 0
            or wrapped_high_k < 0):
        return

    if (_solid_cell(solid, low_i, low_j, low_k, nx, ny, nz,
                    periodic_x, periodic_y, periodic_z, ng) != 0
            or _solid_cell(solid, high_i, high_j, high_k, nx, ny, nz,
                           periodic_x, periodic_y, periodic_z, ng) != 0):
        return

    low_active = active[wrapped_low_i + ng, wrapped_low_j + ng,
                        wrapped_low_k + ng]
    high_active = active[wrapped_high_i + ng, wrapped_high_j + ng,
                         wrapped_high_k + ng]
    I = i + ng
    J = j + ng
    K = k + ng
    if low_active != 0 and high_active != 0:
        inverse_distance[I, J, K] = wp.float32(1.0) / spacing
        return
    if low_active == 0 and high_active == 0:
        return

    # Default to an interface on the geometric face.  When either adjacent
    # mixed-cell PLIC plane crosses the centre-to-centre segment, use that
    # subcell location instead.
    active_is_low = int(low_active != 0)
    direction = wp.float32(1.0)
    if active_is_low == 0:
        direction = -wp.float32(1.0)
    active_i = wrapped_low_i
    active_j = wrapped_low_j
    active_k = wrapped_low_k
    inactive_i = wrapped_high_i
    inactive_j = wrapped_high_j
    inactive_k = wrapped_high_k
    if active_is_low == 0:
        active_i = wrapped_high_i
        active_j = wrapped_high_j
        active_k = wrapped_high_k
        inactive_i = wrapped_low_i
        inactive_j = wrapped_low_j
        inactive_k = wrapped_low_k

    active_distance = _mixed_plane_distance(
        normals, offsets, active_i, active_j, active_k,
        axis, direction, spacing, ng,
    )
    inactive_to_plane = _mixed_plane_distance(
        normals, offsets, inactive_i, inactive_j, inactive_k,
        axis, -direction, spacing, ng,
    )
    inactive_distance = wp.float32(-1.0)
    if inactive_to_plane > wp.float32(0.0):
        inactive_distance = spacing - inactive_to_plane

    distance = wp.float32(0.5) * spacing
    if active_distance > wp.float32(0.0):
        distance = active_distance
    if inactive_distance > wp.float32(0.0):
        if active_distance > wp.float32(0.0):
            distance = wp.float32(0.5) * (distance + inactive_distance)
        else:
            distance = inactive_distance
    distance = wp.min(spacing, wp.max(wp.float32(0.05) * spacing, distance))
    inverse_distance[I, J, K] = wp.float32(1.0) / distance
    wp.atomic_add(dirichlet_count, 0, 1)


@wp.kernel
def _masked_diagonal(
    active: wp.array3d(dtype=wp.int32),
    aperture_u: wp.array3d(dtype=wp.float32),
    aperture_v: wp.array3d(dtype=wp.float32),
    aperture_w: wp.array3d(dtype=wp.float32),
    inverse_u: wp.array3d(dtype=wp.float32),
    inverse_v: wp.array3d(dtype=wp.float32),
    inverse_w: wp.array3d(dtype=wp.float32),
    diagonal: wp.array3d(dtype=wp.float32),
    inverse_dx: wp.float32,
    inverse_dy: wp.float32,
    inverse_dz: wp.float32,
    ng: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    if active[I, J, K] == 0:
        return
    diagonal[I, J, K] = (
        (aperture_u[I, J, K] * inverse_u[I, J, K]
         + aperture_u[I + 1, J, K] * inverse_u[I + 1, J, K]) * inverse_dx
        + (aperture_v[I, J, K] * inverse_v[I, J, K]
           + aperture_v[I, J + 1, K] * inverse_v[I, J + 1, K]) * inverse_dy
        + (aperture_w[I, J, K] * inverse_w[I, J, K]
           + aperture_w[I, J, K + 1] * inverse_w[I, J, K + 1]) * inverse_dz
    )


@wp.kernel
def _masked_operator(
    pressure: wp.array3d(dtype=wp.float32),
    active: wp.array3d(dtype=wp.int32),
    aperture_u: wp.array3d(dtype=wp.float32),
    aperture_v: wp.array3d(dtype=wp.float32),
    aperture_w: wp.array3d(dtype=wp.float32),
    inverse_u: wp.array3d(dtype=wp.float32),
    inverse_v: wp.array3d(dtype=wp.float32),
    inverse_w: wp.array3d(dtype=wp.float32),
    out: wp.array3d(dtype=wp.float32),
    nx: int,
    ny: int,
    nz: int,
    inverse_dx: wp.float32,
    inverse_dy: wp.float32,
    inverse_dz: wp.float32,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    ng: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    if active[I, J, K] == 0:
        out[I, J, K] = wp.float32(0.0)
        return
    centre = pressure[I, J, K]
    value = (
        (centre - _cell_pressure(pressure, active, i - 1, j, k,
                                 nx, ny, nz, periodic_x, periodic_y,
                                 periodic_z, ng))
        * aperture_u[I, J, K] * inverse_u[I, J, K] * inverse_dx
        + (centre - _cell_pressure(pressure, active, i + 1, j, k,
                                   nx, ny, nz, periodic_x, periodic_y,
                                   periodic_z, ng))
        * aperture_u[I + 1, J, K] * inverse_u[I + 1, J, K] * inverse_dx
        + (centre - _cell_pressure(pressure, active, i, j - 1, k,
                                   nx, ny, nz, periodic_x, periodic_y,
                                   periodic_z, ng))
        * aperture_v[I, J, K] * inverse_v[I, J, K] * inverse_dy
        + (centre - _cell_pressure(pressure, active, i, j + 1, k,
                                   nx, ny, nz, periodic_x, periodic_y,
                                   periodic_z, ng))
        * aperture_v[I, J + 1, K] * inverse_v[I, J + 1, K] * inverse_dy
        + (centre - _cell_pressure(pressure, active, i, j, k - 1,
                                   nx, ny, nz, periodic_x, periodic_y,
                                   periodic_z, ng))
        * aperture_w[I, J, K] * inverse_w[I, J, K] * inverse_dz
        + (centre - _cell_pressure(pressure, active, i, j, k + 1,
                                   nx, ny, nz, periodic_x, periodic_y,
                                   periodic_z, ng))
        * aperture_w[I, J, K + 1] * inverse_w[I, J, K + 1] * inverse_dz
    )
    out[I, J, K] = value


@wp.kernel
def _masked_pressure_gradient(
    pressure: wp.array3d(dtype=wp.float32),
    active: wp.array3d(dtype=wp.int32),
    inverse_distance: wp.array3d(dtype=wp.float32),
    gradient: wp.array3d(dtype=wp.float32),
    axis: int,
    nx: int,
    ny: int,
    nz: int,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    ng: int,
):
    i, j, k = wp.tid()
    low_i = i
    low_j = j
    low_k = k
    if axis == 0:
        low_i -= 1
    elif axis == 1:
        low_j -= 1
    else:
        low_k -= 1
    low = _cell_pressure(pressure, active, low_i, low_j, low_k,
                         nx, ny, nz, periodic_x, periodic_y, periodic_z, ng)
    high = _cell_pressure(pressure, active, i, j, k,
                          nx, ny, nz, periodic_x, periodic_y, periodic_z, ng)
    I = i + ng
    J = j + ng
    K = k + ng
    gradient[I, J, K] = (high - low) * inverse_distance[I, J, K]


@wp.func
def _cell_scalar(
    field: wp.array3d(dtype=wp.float32),
    i: int,
    j: int,
    k: int,
    nx: int,
    ny: int,
    nz: int,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    ng: int,
) -> wp.float32:
    ci = _wrapped_cell_index(i, nx, periodic_x)
    cj = _wrapped_cell_index(j, ny, periodic_y)
    ck = _wrapped_cell_index(k, nz, periodic_z)
    if ci < 0 or cj < 0 or ck < 0:
        return wp.float32(0.0)
    return field[ci + ng, cj + ng, ck + ng]


@wp.kernel
def _add_capillary_boundary_gradient(
    active: wp.array3d(dtype=wp.int32),
    curvature: wp.array3d(dtype=wp.float32),
    inverse_distance: wp.array3d(dtype=wp.float32),
    gradient: wp.array3d(dtype=wp.float32),
    surface_tension: wp.float32,
    axis: int,
    nx: int,
    ny: int,
    nz: int,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    ng: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    if inverse_distance[I, J, K] <= wp.float32(0.0):
        return
    low_i = i
    low_j = j
    low_k = k
    if axis == 0:
        low_i -= 1
    elif axis == 1:
        low_j -= 1
    else:
        low_k -= 1
    low_active = _active_cell(
        active, low_i, low_j, low_k, nx, ny, nz,
        periodic_x, periodic_y, periodic_z, ng,
    )
    high_active = _active_cell(
        active, i, j, k, nx, ny, nz,
        periodic_x, periodic_y, periodic_z, ng,
    )
    if low_active == high_active:
        return
    face_curvature = wp.float32(0.5) * (
        _cell_scalar(
            curvature, low_i, low_j, low_k, nx, ny, nz,
            periodic_x, periodic_y, periodic_z, ng,
        )
        + _cell_scalar(
            curvature, i, j, k, nx, ny, nz,
            periodic_x, periodic_y, periodic_z, ng,
        )
    )
    boundary_gradient = (
        surface_tension * face_curvature * inverse_distance[I, J, K]
    )
    if low_active == 0:
        boundary_gradient = -boundary_gradient
    gradient[I, J, K] += boundary_gradient


@wp.kernel
def _mask_pressure(
    pressure: wp.array3d(dtype=wp.float32),
    active: wp.array3d(dtype=wp.int32),
    ng: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    if active[I, J, K] == 0:
        pressure[I, J, K] = wp.float32(0.0)


@wp.kernel
def _accumulate_pressure_correction(
    correction: wp.array3d(dtype=wp.float32),
    pressure: wp.array3d(dtype=wp.float32),
    ng: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    pressure[I, J, K] += correction[I, J, K]


@wp.kernel
def _reduce_active_sum(
    field: wp.array3d(dtype=wp.float32),
    active: wp.array3d(dtype=wp.int32),
    reduction: wp.array(dtype=wp.float64),
    ng: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    if active[I, J, K] != 0:
        wp.atomic_add(reduction, 0, wp.float64(field[I, J, K]))
        wp.atomic_add(reduction, 1, wp.float64(1.0))


@wp.kernel
def _subtract_active_mean(
    field: wp.array3d(dtype=wp.float32),
    active: wp.array3d(dtype=wp.int32),
    mean: wp.float32,
    ng: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    if active[I, J, K] != 0:
        field[I, J, K] -= mean
    else:
        field[I, J, K] = wp.float32(0.0)


@wp.kernel
def _form_masked_rhs(
    divergence: wp.array3d(dtype=wp.float32),
    active: wp.array3d(dtype=wp.int32),
    rhs: wp.array3d(dtype=wp.float32),
    scale: wp.float32,
    ng: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    if active[I, J, K] != 0:
        rhs[I, J, K] = scale * divergence[I, J, K]


@wp.kernel
def _add_masked_rhs(
    source: wp.array3d(dtype=wp.float32),
    active: wp.array3d(dtype=wp.int32),
    rhs: wp.array3d(dtype=wp.float32),
    ng: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    if active[I, J, K] != 0:
        rhs[I, J, K] += source[I, J, K]


@wp.kernel
def _aperture_divergence(
    u: wp.array3d(dtype=wp.float32),
    v: wp.array3d(dtype=wp.float32),
    w: wp.array3d(dtype=wp.float32),
    aperture_u: wp.array3d(dtype=wp.float32),
    aperture_v: wp.array3d(dtype=wp.float32),
    aperture_w: wp.array3d(dtype=wp.float32),
    out: wp.array3d(dtype=wp.float32),
    inverse_dx: wp.float32,
    inverse_dy: wp.float32,
    inverse_dz: wp.float32,
    ng: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    out[I, J, K] = (
        (aperture_u[I + 1, J, K] * u[I + 1, J, K]
         - aperture_u[I, J, K] * u[I, J, K]) * inverse_dx
        + (aperture_v[I, J + 1, K] * v[I, J + 1, K]
           - aperture_v[I, J, K] * v[I, J, K]) * inverse_dy
        + (aperture_w[I, J, K + 1] * w[I, J, K + 1]
           - aperture_w[I, J, K] * w[I, J, K]) * inverse_dz
    )


@wp.kernel
def _reduce_masked_divergence(
    divergence: wp.array3d(dtype=wp.float32),
    active: wp.array3d(dtype=wp.int32),
    theta: wp.array3d(dtype=wp.float32),
    squared_sum: wp.array(dtype=wp.float64),
    maximum: wp.array(dtype=wp.float32),
    ng: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    if active[I, J, K] == 0:
        return
    value = divergence[I, J, K] / theta[I, J, K]
    wp.atomic_add(squared_sum, 0, wp.float64(value) * wp.float64(value))
    wp.atomic_max(maximum, 0, wp.abs(value))


def build_free_surface_geometry(
    state: State3D,
    *,
    active_threshold: float = 0.5,
) -> FreeSurfaceGeometry:
    """Build the water pressure mask and ghost-fluid face distances."""
    if (not math.isfinite(active_threshold) or active_threshold <= 0.0
            or active_threshold >= 1.0):
        raise ValueError("active_threshold must be finite in (0, 1)")
    g = state.cfg.grid
    active = wp.zeros(state.grid.cell_shape, dtype=wp.int32, device=state.device)
    active_count = wp.zeros(1, dtype=wp.int32, device=state.device)
    wp.launch(
        _build_active_mask, dim=g.shape,
        inputs=[state.alpha, state.theta, state.solid, active, active_count,
                active_threshold, g.ng],
        device=state.device,
    )
    interface = estimate_interface_normals(state)
    plic = reconstruct_plic(state, interface.normals)
    surface_tension = None
    if state.cfg.physics.surface_tension > 0.0:
        surface_tension = reconstruct_surface_tension_geometry(
            state, interface.normals, plic,
        )
    inverse = tuple(
        wp.zeros(state.grid.padded_shape(name), dtype=wp.float32,
                 device=state.device)
        for name in ("u", "v", "w")
    )
    dirichlet_count = wp.zeros(1, dtype=wp.int32, device=state.device)
    bc = state.cfg.bc
    periodic = (
        int(bc.xlo == BC.PERIODIC), int(bc.ylo == BC.PERIODIC),
        int(bc.zlo == BC.PERIODIC),
    )
    apertures = (state.u_aperture, state.v_aperture, state.w_aperture)
    for axis, (name, spacing) in enumerate(zip(
        ("u", "v", "w"), (g.dx, g.dy, g.dz),
    )):
        wp.launch(
            _build_face_inverse_distance,
            dim=state.grid.interior_shape(name),
            inputs=[active, state.solid, apertures[axis],
                    plic.normals, plic.offsets,
                    inverse[axis],
                    dirichlet_count, axis, g.nx, g.ny, g.nz, spacing,
                    *periodic, g.ng],
            device=state.device,
        )
    return FreeSurfaceGeometry(
        active=active,
        solid=state.solid,
        theta=state.theta,
        aperture_u=state.u_aperture,
        aperture_v=state.v_aperture,
        aperture_w=state.w_aperture,
        inverse_u_distance=inverse[0],
        inverse_v_distance=inverse[1],
        inverse_w_distance=inverse[2],
        active_cells=int(active_count.numpy()[0]),
        dirichlet_faces=int(dirichlet_count.numpy()[0]),
        surface_tension=surface_tension,
    )


def initialize_hydrostatic_pressure(
    state: State3D,
    surface_elevation: float,
) -> None:
    """Set atmospheric-reference pressure for an initially still water body.

    Pressure is a projection constraint rather than a transported state.  This
    supplies its physically balanced initial iterate; every subsequent
    projection still solves the unchanged discrete pressure equation.
    """
    g = state.cfg.grid
    state.p.zero_()
    wp.launch(
        _initialize_hydrostatic_pressure,
        dim=g.shape,
        inputs=[state.p, state.alpha, state.theta, state.solid,
                surface_elevation,
                state.cfg.physics.rho * state.cfg.physics.g,
                g.z0, g.dz, g.ng],
        device=state.device,
    )


@wp.kernel
def _coarsen_pressure_masks(
    fine_active: wp.array3d(dtype=wp.int32),
    fine_solid: wp.array3d(dtype=wp.int32),
    fine_theta: wp.array3d(dtype=wp.float32),
    coarse_active: wp.array3d(dtype=wp.int32),
    coarse_solid: wp.array3d(dtype=wp.int32),
    coarse_theta: wp.array3d(dtype=wp.float32),
    fine_ng: int,
    coarse_ng: int,
):
    i, j, k = wp.tid()
    active_count = int(0)
    solid_count = int(0)
    theta_sum = wp.float32(0.0)
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                I = 2 * i + di + fine_ng
                J = 2 * j + dj + fine_ng
                K = 2 * k + dk + fine_ng
                active_count += fine_active[I, J, K]
                solid_count += int(fine_solid[I, J, K] != 0)
                theta_sum += fine_theta[I, J, K]
    I = i + coarse_ng
    J = j + coarse_ng
    K = k + coarse_ng
    coarse_theta[I, J, K] = theta_sum / wp.float32(8.0)
    if solid_count == 8:
        coarse_solid[I, J, K] = 1
    elif active_count >= 4:
        coarse_active[I, J, K] = 1


@wp.kernel
def _coarsen_face_aperture(
    fine: wp.array3d(dtype=wp.float32),
    coarse: wp.array3d(dtype=wp.float32),
    axis: int,
    fine_ng: int,
    coarse_ng: int,
):
    i, j, k = wp.tid()
    value = wp.float32(0.0)
    for first in range(2):
        for second in range(2):
            fi = 2 * i
            fj = 2 * j
            fk = 2 * k
            if axis == 0:
                fj += first
                fk += second
            elif axis == 1:
                fi += first
                fk += second
            else:
                fi += first
                fj += second
            value += fine[fi + fine_ng, fj + fine_ng, fk + fine_ng]
    coarse[i + coarse_ng, j + coarse_ng, k + coarse_ng] = (
        value / wp.float32(4.0)
    )


@wp.kernel
def _build_coarse_inverse_distance(
    active: wp.array3d(dtype=wp.int32),
    solid: wp.array3d(dtype=wp.int32),
    inverse_distance: wp.array3d(dtype=wp.float32),
    axis: int,
    nx: int,
    ny: int,
    nz: int,
    spacing: wp.float32,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
    ng: int,
):
    i, j, k = wp.tid()
    low_i = i
    low_j = j
    low_k = k
    if axis == 0:
        low_i -= 1
    elif axis == 1:
        low_j -= 1
    else:
        low_k -= 1

    li = _wrapped_cell_index(low_i, nx, periodic_x)
    lj = _wrapped_cell_index(low_j, ny, periodic_y)
    lk = _wrapped_cell_index(low_k, nz, periodic_z)
    hi = _wrapped_cell_index(i, nx, periodic_x)
    hj = _wrapped_cell_index(j, ny, periodic_y)
    hk = _wrapped_cell_index(k, nz, periodic_z)
    if li < 0 or lj < 0 or lk < 0 or hi < 0 or hj < 0 or hk < 0:
        return
    if (solid[li + ng, lj + ng, lk + ng] != 0
            or solid[hi + ng, hj + ng, hk + ng] != 0):
        return

    low_active = active[li + ng, lj + ng, lk + ng]
    high_active = active[hi + ng, hj + ng, hk + ng]
    value = wp.float32(0.0)
    if low_active != 0 and high_active != 0:
        value = wp.float32(1.0) / spacing
    elif low_active != high_active:
        value = wp.float32(2.0) / spacing
    inverse_distance[i + ng, j + ng, k + ng] = value


@wp.kernel
def _masked_weighted_jacobi_update(
    rhs: wp.array3d(dtype=wp.float32),
    operator_pressure: wp.array3d(dtype=wp.float32),
    diagonal: wp.array3d(dtype=wp.float32),
    active: wp.array3d(dtype=wp.int32),
    pressure: wp.array3d(dtype=wp.float32),
    weight: wp.float32,
    ng: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    if active[I, J, K] != 0 and diagonal[I, J, K] > wp.float32(0.0):
        pressure[I, J, K] += (
            weight * (rhs[I, J, K] - operator_pressure[I, J, K])
            / diagonal[I, J, K]
        )


@dataclass
class _MaskedMultigridGeometry:
    active: wp.array
    solid: wp.array
    theta: wp.array
    aperture: tuple[wp.array, wp.array, wp.array]
    inverse: tuple[wp.array, wp.array, wp.array]
    diagonal: wp.array


class FreeSurfaceMultigridVCycle:
    """Symmetric V-cycle rediscretised on the moving water/solid masks."""

    __slots__ = (
        "hierarchy", "device", "geometries", "pure_neumann",
        "jacobi_weight", "pre_smoothing_steps", "post_smoothing_steps",
        "coarse_smoothing_steps",
    )

    def __init__(self, cfg: Solver3DConfig):
        from shoreloop.physics3d.multigrid import MultigridHierarchy
        self.hierarchy = MultigridHierarchy(cfg)
        self.device = cfg.device
        self.pure_neumann = False
        self.jacobi_weight = 2.0 / 3.0
        self.pre_smoothing_steps = 2
        self.post_smoothing_steps = 2
        self.coarse_smoothing_steps = 40
        self.geometries: list[_MaskedMultigridGeometry] = []
        for level in self.hierarchy.levels:
            active = wp.zeros(
                level.grid.cell_shape, dtype=wp.int32, device=self.device,
            )
            solid = wp.zeros(
                level.grid.cell_shape, dtype=wp.int32, device=self.device,
            )
            theta = wp.zeros(
                level.grid.cell_shape, dtype=wp.float32, device=self.device,
            )
            aperture = tuple(
                wp.zeros(level.grid.padded_shape(name), dtype=wp.float32,
                         device=self.device)
                for name in ("u", "v", "w")
            )
            inverse = tuple(
                wp.zeros(level.grid.padded_shape(name), dtype=wp.float32,
                         device=self.device)
                for name in ("u", "v", "w")
            )
            diagonal = wp.zeros(
                level.grid.cell_shape, dtype=wp.float32, device=self.device,
            )
            self.geometries.append(_MaskedMultigridGeometry(
                active, solid, theta, aperture, inverse, diagonal,
            ))

    def configure(
        self,
        geometry: FreeSurfaceGeometry,
        diagonal: wp.array,
    ) -> None:
        self.pure_neumann = geometry.dirichlet_faces == 0
        self.geometries[0] = _MaskedMultigridGeometry(
            geometry.active,
            geometry.solid,
            geometry.theta,
            (geometry.aperture_u, geometry.aperture_v,
             geometry.aperture_w),
            (geometry.inverse_u_distance,
             geometry.inverse_v_distance,
             geometry.inverse_w_distance),
            diagonal,
        )
        bc = self.hierarchy.cfg.bc
        periodic = (
            int(bc.xlo == BC.PERIODIC),
            int(bc.ylo == BC.PERIODIC),
            int(bc.zlo == BC.PERIODIC),
        )
        for index in range(1, len(self.hierarchy.levels)):
            fine_level = self.hierarchy.levels[index - 1]
            coarse_level = self.hierarchy.levels[index]
            fine_geometry = self.geometries[index - 1]
            coarse_geometry = self.geometries[index]
            coarse_geometry.active.zero_()
            coarse_geometry.solid.zero_()
            coarse_geometry.theta.zero_()
            for field in coarse_geometry.aperture:
                field.zero_()
            for field in coarse_geometry.inverse:
                field.zero_()
            coarse_geometry.diagonal.zero_()
            wp.launch(
                _coarsen_pressure_masks,
                dim=coarse_level.cfg.grid.shape,
                inputs=[fine_geometry.active, fine_geometry.solid,
                        fine_geometry.theta,
                        coarse_geometry.active, coarse_geometry.solid,
                        coarse_geometry.theta,
                        fine_level.cfg.grid.ng, coarse_level.cfg.grid.ng],
                device=self.device,
            )
            g = coarse_level.cfg.grid
            for axis, (name, spacing) in enumerate(zip(
                ("u", "v", "w"), (g.dx, g.dy, g.dz), strict=True,
            )):
                wp.launch(
                    _coarsen_face_aperture,
                    dim=coarse_level.grid.interior_shape(name),
                    inputs=[fine_geometry.aperture[axis],
                            coarse_geometry.aperture[axis], axis,
                            fine_level.cfg.grid.ng, g.ng],
                    device=self.device,
                )
                wp.launch(
                    _build_coarse_inverse_distance,
                    dim=coarse_level.grid.interior_shape(name),
                    inputs=[coarse_geometry.active, coarse_geometry.solid,
                            coarse_geometry.inverse[axis], axis,
                            g.nx, g.ny, g.nz, spacing, *periodic, g.ng],
                    device=self.device,
                )
            wp.launch(
                _masked_diagonal,
                dim=g.shape,
                inputs=[coarse_geometry.active,
                        *coarse_geometry.aperture,
                        *coarse_geometry.inverse,
                        coarse_geometry.diagonal,
                        1.0 / g.dx, 1.0 / g.dy, 1.0 / g.dz, g.ng],
                device=self.device,
            )

    def _copy(self, level_index: int, source: wp.array, target: wp.array) -> None:
        level = self.hierarchy.levels[level_index]
        target.zero_()
        wp.launch(
            _copy_cell_interior,
            dim=level.cfg.grid.shape,
            inputs=[source, target, level.cfg.grid.ng],
            device=self.device,
        )

    def _remove_mean(self, level_index: int, field: wp.array) -> None:
        level = self.hierarchy.levels[level_index]
        level.reduction.zero_()
        wp.launch(
            _reduce_active_sum,
            dim=level.cfg.grid.shape,
            inputs=[field, self.geometries[level_index].active,
                    level.reduction, level.cfg.grid.ng],
            device=self.device,
        )
        totals = level.reduction.numpy()
        mean = float(totals[0] / totals[1])
        wp.launch(
            _subtract_active_mean,
            dim=level.cfg.grid.shape,
            inputs=[field, self.geometries[level_index].active,
                    mean, level.cfg.grid.ng],
            device=self.device,
        )

    def _mask(self, level_index: int, field: wp.array) -> None:
        level = self.hierarchy.levels[level_index]
        wp.launch(
            _mask_pressure,
            dim=level.cfg.grid.shape,
            inputs=[field, self.geometries[level_index].active,
                    level.cfg.grid.ng],
            device=self.device,
        )

    def _operator(self, level_index: int) -> None:
        level = self.hierarchy.levels[level_index]
        geometry = self.geometries[level_index]
        g = level.cfg.grid
        bc = level.cfg.bc
        level.operator_value.zero_()
        wp.launch(
            _masked_operator,
            dim=g.shape,
            inputs=[level.pressure, geometry.active, *geometry.aperture,
                    *geometry.inverse,
                    level.operator_value, g.nx, g.ny, g.nz,
                    1.0 / g.dx, 1.0 / g.dy, 1.0 / g.dz,
                    int(bc.xlo == BC.PERIODIC),
                    int(bc.ylo == BC.PERIODIC),
                    int(bc.zlo == BC.PERIODIC), g.ng],
            device=self.device,
        )

    def _residual(self, level_index: int) -> None:
        level = self.hierarchy.levels[level_index]
        self._operator(level_index)
        wp.launch(
            _form_residual,
            dim=level.cfg.grid.shape,
            inputs=[level.rhs, level.operator_value, level.residual,
                    level.cfg.grid.ng],
            device=self.device,
        )
        self._mask(level_index, level.residual)
        if self.pure_neumann:
            self._remove_mean(level_index, level.residual)

    def _smooth(self, level_index: int, steps: int) -> None:
        level = self.hierarchy.levels[level_index]
        geometry = self.geometries[level_index]
        for _ in range(steps):
            self._operator(level_index)
            wp.launch(
                _masked_weighted_jacobi_update,
                dim=level.cfg.grid.shape,
                inputs=[level.rhs, level.operator_value, geometry.diagonal,
                        geometry.active, level.pressure,
                        self.jacobi_weight, level.cfg.grid.ng],
                device=self.device,
            )
            if self.pure_neumann:
                self._remove_mean(level_index, level.pressure)

    def _visit(self, level_index: int) -> None:
        levels = self.hierarchy.levels
        if level_index == len(levels) - 1:
            self._smooth(level_index, self.coarse_smoothing_steps)
            return
        self._smooth(level_index, self.pre_smoothing_steps)
        self._residual(level_index)
        coarse = levels[level_index + 1]
        self.hierarchy.restrict(
            level_index, levels[level_index].residual, coarse.rhs,
        )
        self._mask(level_index + 1, coarse.rhs)
        if self.pure_neumann:
            self._remove_mean(level_index + 1, coarse.rhs)
        coarse.pressure.zero_()
        self._visit(level_index + 1)
        self.hierarchy.prolong(
            level_index + 1, coarse.pressure,
            levels[level_index].pressure, add=True,
        )
        self._mask(level_index, levels[level_index].pressure)
        if self.pure_neumann:
            self._remove_mean(level_index, levels[level_index].pressure)
        self._smooth(level_index, self.post_smoothing_steps)

    def apply(self, rhs: wp.array, out: wp.array) -> wp.array:
        finest = self.hierarchy.levels[0]
        self._copy(0, rhs, finest.rhs)
        self._mask(0, finest.rhs)
        if self.pure_neumann:
            self._remove_mean(0, finest.rhs)
        for level in self.hierarchy.levels:
            level.pressure.zero_()
        self._visit(0)
        self._copy(0, finest.pressure, out)
        self._mask(0, out)
        return out


class FreeSurfacePressurePCG:
    """Jacobi- or masked-multigrid PCG for ghost-fluid pressure.

    With an atmospheric free-surface Dirichlet condition, a non-zero initial
    pressure is retained as ``p0`` while PCG solves the equivalent defect
    equation ``A delta_p = b - A p0``.  The reported residual is the explicitly
    recomputed defect residual, normalized by the original right-hand-side
    norm; ``delta_p`` is accumulated into ``p0`` only after the solve.  A pure
    Neumann domain has no absolute pressure background, so it retains the
    direct zero-mean pressure solve.
    """

    __slots__ = (
        "cfg", "settings", "device", "geometry", "diagonal", "_rhs",
        "_residual", "_preconditioned", "_direction", "_operator_vector",
        "_correction", "_dot_products", "_reduction", "_cell_volume",
        "_preconditioner",
    )

    def __init__(self, cfg: Solver3DConfig, settings: PCGConfig):
        self.cfg = cfg
        self.settings = settings
        self.device = cfg.device
        self.geometry: FreeSurfaceGeometry | None = None
        shape = tuple(n + 2 * cfg.grid.ng for n in cfg.grid.shape)

        def cell_array() -> wp.array:
            return wp.zeros(shape, dtype=wp.float32, device=self.device)

        self.diagonal = cell_array()
        self._rhs = cell_array()
        self._residual = cell_array()
        self._preconditioned = cell_array()
        self._direction = cell_array()
        self._operator_vector = cell_array()
        self._correction = cell_array()
        self._dot_products = wp.zeros(
            cfg.grid.shape, dtype=wp.float64, device=self.device,
        )
        self._reduction = wp.zeros(2, dtype=wp.float64, device=self.device)
        g = cfg.grid
        self._cell_volume = g.dx * g.dy * g.dz
        self._preconditioner = None
        if self.settings.preconditioner == "multigrid":
            self._preconditioner = FreeSurfaceMultigridVCycle(cfg)

    def configure(self, geometry: FreeSurfaceGeometry) -> None:
        self.geometry = geometry
        self.diagonal.zero_()
        g = self.cfg.grid
        wp.launch(
            _masked_diagonal, dim=g.shape,
            inputs=[geometry.active,
                    geometry.aperture_u, geometry.aperture_v,
                    geometry.aperture_w,
                    geometry.inverse_u_distance,
                    geometry.inverse_v_distance, geometry.inverse_w_distance,
                    self.diagonal, 1.0 / g.dx, 1.0 / g.dy, 1.0 / g.dz, g.ng],
            device=self.device,
        )
        if self._preconditioner is not None:
            self._preconditioner.configure(geometry, self.diagonal)

    def _copy(self, source: wp.array, target: wp.array) -> None:
        target.zero_()
        wp.launch(
            _copy_cell_interior, dim=self.cfg.grid.shape,
            inputs=[source, target, self.cfg.grid.ng], device=self.device,
        )

    def _dot(self, left: wp.array, right: wp.array) -> float:
        wp.launch(
            _form_dot_products, dim=self.cfg.grid.shape,
            inputs=[left, right, self._dot_products, self.cfg.grid.ng,
                    self._cell_volume],
            device=self.device,
        )
        wp.utils.array_sum(
            self._dot_products, out=self._reduction[0:1],
        )
        return float(self._reduction.numpy()[0])

    def _norm(self, field: wp.array) -> float:
        return math.sqrt(max(self._dot(field, field), 0.0))

    def _remove_mean(self, field: wp.array) -> None:
        self._reduction.zero_()
        wp.launch(
            _reduce_active_sum, dim=self.cfg.grid.shape,
            inputs=[field, self.geometry.active, self._reduction,
                    self.cfg.grid.ng],
            device=self.device,
        )
        totals = self._reduction.numpy()
        mean = float(totals[0] / totals[1])
        wp.launch(
            _subtract_active_mean, dim=self.cfg.grid.shape,
            inputs=[field, self.geometry.active, mean, self.cfg.grid.ng],
            device=self.device,
        )

    def apply_operator(self, pressure: wp.array, out: wp.array) -> None:
        if self.geometry is None:
            raise RuntimeError("free-surface pressure geometry is not configured")
        out.zero_()
        g = self.cfg.grid
        bc = self.cfg.bc
        wp.launch(
            _masked_operator, dim=g.shape,
            inputs=[pressure, self.geometry.active,
                    self.geometry.aperture_u,
                    self.geometry.aperture_v,
                    self.geometry.aperture_w,
                    self.geometry.inverse_u_distance,
                    self.geometry.inverse_v_distance,
                    self.geometry.inverse_w_distance, out,
                    g.nx, g.ny, g.nz, 1.0 / g.dx, 1.0 / g.dy, 1.0 / g.dz,
                    int(bc.xlo == BC.PERIODIC),
                    int(bc.ylo == BC.PERIODIC),
                    int(bc.zlo == BC.PERIODIC), g.ng],
            device=self.device,
        )

    def _recompute_residual(self, pressure: wp.array, pure_neumann: bool) -> float:
        self.apply_operator(pressure, self._operator_vector)
        wp.launch(
            _form_residual, dim=self.cfg.grid.shape,
            inputs=[self._rhs, self._operator_vector, self._residual,
                    self.cfg.grid.ng],
            device=self.device,
        )
        if pure_neumann:
            self._remove_mean(self._residual)
        return self._norm(self._residual)

    def _apply_preconditioner(self, pure_neumann: bool) -> None:
        if self._preconditioner is None:
            wp.launch(
                _apply_jacobi, dim=self.cfg.grid.shape,
                inputs=[self._residual, self.diagonal, self._preconditioned,
                        self.cfg.grid.ng], device=self.device,
            )
        else:
            self._preconditioner.apply(
                self._residual, self._preconditioned,
            )
        wp.launch(
            _mask_pressure, dim=self.cfg.grid.shape,
            inputs=[self._preconditioned, self.geometry.active,
                    self.cfg.grid.ng], device=self.device,
        )
        if pure_neumann:
            self._remove_mean(self._preconditioned)

    @staticmethod
    def _relative(absolute: float, reference: float) -> float:
        if reference > 0.0:
            return absolute / reference
        return 0.0 if absolute == 0.0 else math.inf

    def solve(self, rhs: wp.array, pressure: wp.array) -> PCGResult:
        if self.geometry is None:
            raise RuntimeError("free-surface pressure geometry is not configured")
        pure_neumann = self.geometry.dirichlet_faces == 0
        self._copy(rhs, self._rhs)
        wp.launch(
            _mask_pressure, dim=self.cfg.grid.shape,
            inputs=[pressure, self.geometry.active, self.cfg.grid.ng],
            device=self.device,
        )
        if pure_neumann:
            self._remove_mean(self._rhs)
            self._remove_mean(pressure)
        rhs_norm = self._norm(self._rhs)
        target = max(self.settings.absolute_tolerance,
                     self.settings.relative_tolerance * rhs_norm)
        residual_norm = self._recompute_residual(pressure, pure_neumann)
        if residual_norm <= target:
            return PCGResult(pressure, True, 0, residual_norm,
                             self._relative(residual_norm, rhs_norm))

        # Atmospheric free surfaces carry a physical hydrostatic background.
        # Solve A delta_p = b - A p0 without repeatedly forming A(p0+delta_p),
        # preserving the small defect in float32.  A pure Neumann pressure has
        # only a zero-mean gauge and therefore solves its original equation
        # directly, as there is no absolute background to preserve.
        solution = pressure
        if not pure_neumann:
            self._copy(self._residual, self._rhs)
            self._correction.zero_()
            solution = self._correction
        self._apply_preconditioner(pure_neumann)
        self._copy(self._preconditioned, self._direction)
        rz = self._dot(self._residual, self._preconditioned)
        iterations = 0
        converged = False
        for iteration in range(1, self.settings.max_iterations + 1):
            if not math.isfinite(rz) or rz <= 0.0:
                break
            self.apply_operator(self._direction, self._operator_vector)
            curvature = self._dot(self._direction, self._operator_vector)
            if not math.isfinite(curvature) or curvature <= 0.0:
                break
            step = rz / curvature
            wp.launch(
                _update_solution_and_residual, dim=self.cfg.grid.shape,
                inputs=[solution, self._residual, self._direction,
                        self._operator_vector, step, self.cfg.grid.ng],
                device=self.device,
            )
            if pure_neumann:
                self._remove_mean(self._residual)
            iterations = iteration
            residual_norm = self._norm(self._residual)
            restart = False
            if residual_norm <= target:
                residual_norm = self._recompute_residual(
                    solution, pure_neumann,
                )
                if residual_norm <= target:
                    converged = True
                    break
                restart = True
            self._apply_preconditioner(pure_neumann)
            next_rz = self._dot(self._residual, self._preconditioned)
            if not math.isfinite(next_rz) or next_rz <= 0.0:
                rz = next_rz
                break
            if restart:
                self._copy(self._preconditioned, self._direction)
            else:
                beta = next_rz / rz
                wp.launch(
                    _update_direction, dim=self.cfg.grid.shape,
                    inputs=[self._preconditioned, self._direction, beta,
                            self.cfg.grid.ng], device=self.device,
                )
            rz = next_rz
        if pure_neumann:
            self._remove_mean(solution)
        wp.launch(
            _mask_pressure, dim=self.cfg.grid.shape,
            inputs=[solution, self.geometry.active, self.cfg.grid.ng],
            device=self.device,
        )
        residual_norm = self._recompute_residual(
            solution, pure_neumann,
        )
        converged = converged or residual_norm <= target
        if not pure_neumann:
            wp.launch(
                _accumulate_pressure_correction, dim=self.cfg.grid.shape,
                inputs=[self._correction, pressure, self.cfg.grid.ng],
                device=self.device,
            )
        wp.launch(
            _mask_pressure, dim=self.cfg.grid.shape,
            inputs=[pressure, self.geometry.active, self.cfg.grid.ng],
            device=self.device,
        )
        return PCGResult(
            pressure, converged, iterations, residual_norm,
            self._relative(residual_norm, rhs_norm),
        )


class FreeSurfaceProjection:
    """Project velocity in active water cells with atmospheric interface pressure."""

    __slots__ = (
        "cfg", "settings", "device", "pressure_solver", "_divergence",
        "_rhs", "_gradient", "_squared_sum", "_maximum",
    )

    def __init__(self, cfg: Solver3DConfig, settings: ProjectionConfig | None = None):
        self.cfg = cfg
        self.settings = ProjectionConfig() if settings is None else settings
        if not isinstance(self.settings, ProjectionConfig):
            raise TypeError("settings must be a ProjectionConfig")
        self.device = cfg.device
        self.pressure_solver = FreeSurfacePressurePCG(cfg, self.settings.pcg)
        shape = tuple(n + 2 * cfg.grid.ng for n in cfg.grid.shape)
        self._divergence = wp.zeros(shape, dtype=wp.float32, device=self.device)
        self._rhs = wp.zeros(shape, dtype=wp.float32, device=self.device)
        grid_shapes = (
            (cfg.grid.nx + 1, cfg.grid.ny, cfg.grid.nz),
            (cfg.grid.nx, cfg.grid.ny + 1, cfg.grid.nz),
            (cfg.grid.nx, cfg.grid.ny, cfg.grid.nz + 1),
        )
        self._gradient = tuple(
            wp.zeros(tuple(n + 2 * cfg.grid.ng for n in field_shape),
                     dtype=wp.float32, device=self.device)
            for field_shape in grid_shapes
        )
        self._squared_sum = wp.zeros(1, dtype=wp.float64, device=self.device)
        self._maximum = wp.zeros(1, dtype=wp.float32, device=self.device)

    def _divergence_launch(self, state: State3D) -> None:
        g = self.cfg.grid
        self._divergence.zero_()
        wp.launch(
            _aperture_divergence, dim=g.shape,
            inputs=[state.u, state.v, state.w,
                    state.u_aperture, state.v_aperture, state.w_aperture,
                    self._divergence,
                    1.0 / g.dx, 1.0 / g.dy, 1.0 / g.dz, g.ng],
            device=self.device,
        )

    def _measure(self, geometry: FreeSurfaceGeometry) -> tuple[float, float]:
        if geometry.active_cells == 0:
            return (0.0, 0.0)
        self._squared_sum.zero_()
        self._maximum.zero_()
        wp.launch(
            _reduce_masked_divergence, dim=self.cfg.grid.shape,
            inputs=[self._divergence, geometry.active, geometry.theta,
                    self._squared_sum,
                    self._maximum, self.cfg.grid.ng], device=self.device,
        )
        return (
            math.sqrt(float(self._squared_sum.numpy()[0])
                      / geometry.active_cells),
            float(self._maximum.numpy()[0]),
        )

    def project(self, state: State3D, dt: float) -> FreeSurfaceProjectionResult:
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"dt must be positive and finite, got {dt!r}")
        fill_boundaries(state)
        enforce_voxel_solids(state)
        geometry = build_free_surface_geometry(state)
        if geometry.active_cells == 0:
            state.p.zero_()
            solve = PCGResult(state.p, True, 0, 0.0, 0.0)
            return FreeSurfaceProjectionResult(
                solve, True, 0.0, 0.0, 0.0, 0.0, 0,
                geometry.dirichlet_faces,
                0.0, 0.0, 0.0, 0, 0,
            )

        self._divergence_launch(state)
        rms_before, max_before = self._measure(geometry)
        self._rhs.zero_()
        wp.launch(
            _form_masked_rhs, dim=self.cfg.grid.shape,
            inputs=[self._divergence, geometry.active, self._rhs,
                    -self.cfg.physics.rho / dt, self.cfg.grid.ng],
            device=self.device,
        )
        capillary = geometry.surface_tension
        if capillary is not None:
            g = self.cfg.grid
            bc = self.cfg.bc
            periodic = (
                int(bc.xlo == BC.PERIODIC), int(bc.ylo == BC.PERIODIC),
                int(bc.zlo == BC.PERIODIC),
            )
            inverse = (
                geometry.inverse_u_distance,
                geometry.inverse_v_distance,
                geometry.inverse_w_distance,
            )
            for axis, name in enumerate(("u", "v", "w")):
                self._gradient[axis].zero_()
                wp.launch(
                    _add_capillary_boundary_gradient,
                    dim=state.grid.interior_shape(name),
                    inputs=[
                        geometry.active, capillary.curvature, inverse[axis],
                        self._gradient[axis],
                        self.cfg.physics.surface_tension,
                        axis, g.nx, g.ny, g.nz, *periodic, g.ng,
                    ],
                    device=self.device,
                )
            self._divergence.zero_()
            wp.launch(
                _aperture_divergence, dim=g.shape,
                inputs=[
                    *self._gradient,
                    geometry.aperture_u, geometry.aperture_v,
                    geometry.aperture_w, self._divergence,
                    1.0 / g.dx, 1.0 / g.dy, 1.0 / g.dz, g.ng,
                ],
                device=self.device,
            )
            wp.launch(
                _add_masked_rhs, dim=g.shape,
                inputs=[self._divergence, geometry.active, self._rhs, g.ng],
                device=self.device,
            )
        self.pressure_solver.configure(geometry)
        pressure_solve = self.pressure_solver.solve(self._rhs, state.p)

        if pressure_solve.converged:
            g = self.cfg.grid
            bc = self.cfg.bc
            periodic = (
                int(bc.xlo == BC.PERIODIC), int(bc.ylo == BC.PERIODIC),
                int(bc.zlo == BC.PERIODIC),
            )
            inverse = (geometry.inverse_u_distance,
                       geometry.inverse_v_distance,
                       geometry.inverse_w_distance)
            for axis, name in enumerate(("u", "v", "w")):
                self._gradient[axis].zero_()
                wp.launch(
                    _masked_pressure_gradient,
                    dim=state.grid.interior_shape(name),
                    inputs=[state.p, geometry.active, inverse[axis],
                            self._gradient[axis], axis, g.nx, g.ny, g.nz,
                            *periodic, g.ng],
                    device=self.device,
                )
                if capillary is not None:
                    wp.launch(
                        _add_capillary_boundary_gradient,
                        dim=state.grid.interior_shape(name),
                        inputs=[
                            geometry.active, capillary.curvature,
                            inverse[axis], self._gradient[axis],
                            self.cfg.physics.surface_tension,
                            axis, g.nx, g.ny, g.nz, *periodic, g.ng,
                        ],
                        device=self.device,
                    )
                wp.launch(
                    _correct_velocity,
                    dim=state.grid.interior_shape(name),
                    inputs=[state.field(name), self._gradient[axis],
                            dt / self.cfg.physics.rho, g.ng],
                    device=self.device,
                )
            fill_boundaries(state)
            enforce_voxel_solids(state)
            self._divergence_launch(state)
            rms_after, max_after = self._measure(geometry)
        else:
            rms_after, max_after = rms_before, max_before
        converged = (pressure_solve.converged
                     and rms_after <= self.settings.rms_divergence_tolerance)
        return FreeSurfaceProjectionResult(
            pressure_solve, converged, rms_before, max_before,
            rms_after, max_after, geometry.active_cells,
            geometry.dirichlet_faces,
            0.0 if capillary is None else capillary.surface_area,
            0.0 if capillary is None else capillary.max_abs_curvature,
            (
                0.0 if capillary is None
                else self.cfg.physics.surface_tension
                * capillary.max_abs_curvature
            ),
            0 if capillary is None else capillary.height_function_cells,
            0 if capillary is None else capillary.unresolved_cells,
        )
