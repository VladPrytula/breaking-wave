"""Plane accuracy, graph support and volume ownership of height normals."""

from dataclasses import replace

import numpy as np
import pytest

from shoreloop.config import BC, BC3DConfig, Grid3DConfig, Solver3DConfig
from shoreloop.device import DEVICE
from shoreloop.physics3d.interface import estimate_interface_normals
from shoreloop.physics3d.plic import reconstruct_plic, unit_cube_volume
from shoreloop.physics3d.state import State3D


def _state(n=24):
    return State3D(Solver3DConfig(
        grid=Grid3DConfig(nx=n, ny=n, nz=n, Lx=.3, Ly=9/14, Lz=.24),
        bc=BC3DConfig(**{a+s: BC.OPEN for a in "xyz" for s in ("lo", "hi")}),
        device=DEVICE, pressure_discretization="liquid_mimetic",
        interface_reconstruction="height_function",
    ))


def _plane(state, normal, translation):
    g = state.cfg.grid
    n = np.asarray(normal, float)
    n /= np.linalg.norm(n)
    spacing = np.array([g.dx, g.dy, g.dz])
    origin = np.array([g.x0, g.y0, g.z0])
    local = n*spacing
    scale = np.abs(local).sum()
    offset = n @ (origin+np.array([g.Lx, g.Ly, g.Lz])/2)+translation
    cells = np.moveaxis(np.indices(g.shape), 0, -1)
    d = (offset-(origin+cells*spacing)@n)/scale
    alpha = np.empty(g.shape, np.float32)
    for cell in np.ndindex(g.shape):
        alpha[cell] = unit_cube_volume(local, d[cell])
    state.set_interior("alpha", alpha)
    return n, alpha


@pytest.mark.parametrize("normal", [(1.1, .07, 1), (.07, 1, 1.1), (1, 1.1, .07), (-1.1, .07, -.9)])
@pytest.mark.parametrize("translation", [0.0031, -0.0067])
def test_anisotropic_planar_normals_preserve_direction_and_cell_volume(normal, translation):
    state = _state()
    expected, before = _plane(state, normal, translation)
    result = estimate_interface_normals(state)
    sl = state.grid.interior_slices("cell")
    normals = result.normals.numpy()[sl]
    interior = np.zeros(before.shape, bool)
    interior[5:-5, 5:-5, 5:-5] = True
    mixed = interior & (before > 1e-6) & (before < 1-1e-6)
    assert mixed.sum() > 30
    assert result.height_function_cells >= mixed.sum()
    assert np.max(np.linalg.norm(normals[mixed]-expected, axis=1)) < 2e-5
    np.testing.assert_array_equal(state.get_interior("alpha"), before)
    plic = reconstruct_plic(state, result.normals)
    local = plic.normals.numpy()[sl][mixed]
    offsets = plic.offsets.numpy()[sl][mixed]
    volume = np.array([unit_cube_volume(n, float(d)) for n, d in zip(local, offsets, strict=True)])
    np.testing.assert_allclose(volume, before[mixed], atol=1.5e-6, rtol=0)


@pytest.mark.parametrize("obstruction", ["second_sheet", "cut_solid", "incomplete"])
def test_unsupported_column_retains_existing_orientation(obstruction):
    state = _state()
    alpha = np.zeros(state.cfg.grid.shape, np.float32)
    alpha[:, :, :12] = 1
    alpha[:, :, 12] = .35
    i = j = k = 12
    if obstruction == "second_sheet":
        alpha[:, :, 14] = .2
    elif obstruction == "cut_solid":
        # A solid inside one column makes alpha a saturation, not a cell volume.
        theta = state.get_geometry_interior("theta")
        theta[i+1, j, k-2] = 0
        state.set_geometry_interior("theta", theta)
        alpha[i+1, j, k-2] = 0
    else:
        # Water reaches the end of the candidate column: no air endpoint.
        alpha[:, :, 12:] = .35
    state.set_interior("alpha", alpha)
    measured = estimate_interface_normals(state).normals.numpy()
    state.cfg = replace(state.cfg, interface_reconstruction="youngs")
    baseline = estimate_interface_normals(state).normals.numpy()
    index = (i+state.cfg.grid.ng, j+state.cfg.grid.ng, k+state.cfg.grid.ng)
    np.testing.assert_array_equal(measured[index], baseline[index])
    np.testing.assert_array_equal(state.get_interior("alpha"), alpha)


def test_curved_graph_normals_converge_on_anisotropic_grids():
    errors = []
    # The entire surface footprint fits inside each column; saturation is integrated by
    # 8x8 Gauss quadrature, independent of the reconstruction's stencil.
    nodes, weights = np.polynomial.legendre.leggauss(8)
    for n in (24, 48):
        state = _state(n)
        g = state.cfg.grid
        x = (np.arange(n)+.5)*g.dx
        y = (np.arange(n)+.5)*g.dy
        z = np.arange(n)*g.dz
        alpha = np.zeros(g.shape, float)
        for a, wa in zip(nodes, weights, strict=True):
            for b, wb in zip(nodes, weights, strict=True):
                xx = x[:, None]+a*g.dx/2-.15
                yy = y[None, :]+b*g.dy/2-g.Ly/2
                height = .12+.025*np.cos(10*xx)*np.cos(5*yy)
                alpha += wa*wb/4*np.clip((height[:, :, None]-z)/g.dz, 0, 1)
        state.set_interior("alpha", alpha.astype(np.float32))
        result = estimate_interface_normals(state)
        normals = result.normals.numpy()[state.grid.interior_slices("cell")]
        xx, yy = np.meshgrid(x-.15, y-g.Ly/2, indexing="ij")
        nx = .25*np.sin(10*xx)*np.cos(5*yy)
        ny = .125*np.cos(10*xx)*np.sin(5*yy)
        exact = np.stack([nx, ny, np.ones_like(nx)], axis=-1)
        exact /= np.linalg.norm(exact, axis=-1)[:, :, None]
        # Fixed physical patch well away from the open domain endpoints.
        support = (np.abs(xx) < .07) & (np.abs(yy) < .15)
        mixed = (alpha > 1e-5) & (alpha < 1-1e-5) & support[:, :, None]
        diff = normals-np.broadcast_to(exact[:, :, None, :], normals.shape)
        errors.append(np.sqrt(np.mean(np.sum(diff[mixed]**2, axis=1))))
        assert result.height_function_cells > mixed.sum()
    assert errors[1] < .4*errors[0]
    assert errors[1] < 5e-4


def test_two_thin_sheets_keep_the_same_separated_volume_boundary():
    from shoreloop.render.surface_mesh import extract_state_free_surface_mesh

    state = _state()
    alpha = np.zeros(state.cfg.grid.shape, np.float32)
    alpha[6:18, 6:18, 9] = .03
    alpha[6:18, 6:18, 12] = .08
    state.set_interior("alpha", alpha)
    after = extract_state_free_surface_mesh(state)
    state.cfg = replace(state.cfg, interface_reconstruction="youngs")
    before = extract_state_free_surface_mesh(state)
    for field in ("vertices", "triangles", "normals", "velocity", "foam"):
        np.testing.assert_array_equal(getattr(after, field), getattr(before, field))
    g = state.cfg.grid
    points = after.vertices[after.triangles].astype(float)
    volume = np.sum(np.einsum('ij,ij->i', points[:, 0],
                              np.cross(points[:, 1], points[:, 2])))/6
    assert volume == pytest.approx(alpha.astype(float).sum()*g.dx*g.dy*g.dz, rel=5e-5)
    assert np.all((points[:, :, 2].max(axis=1) <= float(np.float32(10*g.dz)))
                  | (points[:, :, 2].min(axis=1) >= float(np.float32(12*g.dz))))
