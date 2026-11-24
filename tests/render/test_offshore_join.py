"""AW-707 geometric and image-transfer gates for the offshore join."""

from __future__ import annotations

from pathlib import Path

import mitsuba as mi
import numpy as np
import pytest

from shoreloop.closures.ripples import RippleSpectrum
from shoreloop.device import DEVICE
from shoreloop.physics3d.waves import LinearWaveInlet, PeriodicWaveComponent
from shoreloop.render.camera import Camera
from shoreloop.render.mitsuba_water import _background_plate, _sensor
from shoreloop.render.offshore_join import (
    MultiValuedOffshoreSeamError,
    _attach_offshore_mesh,
    _fundamental_period_mesh,
    _ordered_seam,
    _parametric_collar,
    _periodic_seam,
    _periodic_y_boundary_indices,
    _tile_periodic_y,
    clip_local_mesh,
    exact_offshore_field,
    render_offshore_view,
)
from shoreloop.render.sky import light_preset
from shoreloop.render.surface_mesh import WaterSurfaceMesh


def _local_surface(
    inlet: LinearWaveInlet,
    time: float,
    eta0: float,
) -> WaterSurfaceMesh:
    x = np.linspace(0.0, 2.0, 17)
    y = np.linspace(0.0, 1.0, 9)
    X, Y = np.meshgrid(x, y, indexing="ij")
    component = inlet.components[0]
    phase = (
        component.kx * X + component.ky * Y
        - component.omega * time + component.phase_offset
    )
    resolved = 0.018 * np.cos(2.0 * np.pi * Y) * np.exp(-X / 0.7)
    z = eta0 + component.amplitude * np.cos(phase) + resolved
    dz_dx = (
        -component.kx * component.amplitude * np.sin(phase)
        - resolved / 0.7
    )
    dz_dy = (
        -component.ky * component.amplitude * np.sin(phase)
        - 0.018 * 2.0 * np.pi * np.sin(2.0 * np.pi * Y)
        * np.exp(-X / 0.7)
    )
    normals = np.stack((-dz_dx, -dz_dy, np.ones_like(z)), axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    normals[:, -1] = normals[:, 0]
    vertices = np.stack((X, Y, z), axis=-1).reshape(-1, 3)
    triangles = []
    for i in range(len(x) - 1):
        for j in range(len(y) - 1):
            a = i * len(y) + j
            b = a + len(y)
            triangles.extend(((a, b, b + 1), (a, b + 1, a + 1)))
    triangles = np.asarray(triangles, dtype=np.int32)
    return WaterSurfaceMesh(
        vertices=vertices.astype(np.float32),
        triangles=triangles,
        normals=normals.reshape(-1, 3).astype(np.float32),
        velocity=np.zeros_like(vertices, dtype=np.float32),
        foam=np.zeros(len(vertices), dtype=np.float32),
        vertex_open_boundaries=np.zeros(len(vertices), dtype=np.uint8),
        open_boundary_edges=np.empty((0, 2), dtype=np.int32),
        open_boundary_labels=np.empty(0, dtype=np.uint8),
    )


def _inlet() -> LinearWaveInlet:
    return LinearWaveInlet((PeriodicWaveComponent(
        period=0.8,
        depth=0.28,
        alongshore_length=1.0,
        amplitude=0.035,
        phase_offset=0.4,
        stokes_order=2,
    ),))


def _stepped_surface(profile, slope=0.07):
    """Extrude an explicitly ordered y-z boundary, including vertical facets."""
    profile = np.asarray(profile, dtype=np.float32)
    vertices = np.asarray([
        (x, y, height + slope*x) for x in (0.0, 1.0)
        for y, height in profile
    ], dtype=np.float32)
    count = len(profile)
    triangles = np.asarray([
        triangle for j in range(count-1)
        for triangle in ((j, count+j, count+j+1), (j, count+j+1, j+1))
    ], dtype=np.int32)
    points = vertices[triangles].astype(float)
    area = np.cross(points[:, 1]-points[:, 0], points[:, 2]-points[:, 0])
    normals = np.zeros_like(vertices, dtype=float)
    for corner in range(3):
        np.add.at(normals, triangles[:, corner], area)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    velocity = vertices.copy()
    velocity[:, 1] = np.cos(2*np.pi*vertices[:, 1].astype(float))
    return WaterSurfaceMesh(
        vertices, triangles, normals.astype(np.float32),
        velocity, (.2+.1*velocity[:, 1]).astype(np.float32),
        np.zeros(len(vertices), np.uint8), np.empty((0, 2), np.int32),
        np.empty(0, np.uint8),
    )


def test_offshore_seam_preserves_vertical_steps_and_retained_surface_tangents():
    profile = ((0, .2), (.25, .2), (.25, .24), (.25, .3),
               (.5, .3), (.5, .2), (1, .2))
    mesh = _stepped_surface(profile)
    originals = [value.copy() for value in (mesh.vertices, mesh.triangles,
                 mesh.normals, mesh.velocity, mesh.foam)]
    boundary, y, height, slope = _ordered_seam(mesh, 0.0)
    np.testing.assert_array_equal(boundary, np.arange(len(profile)))
    np.testing.assert_array_equal(np.c_[y, height], np.asarray(profile, np.float32))
    np.testing.assert_allclose(slope, .07, rtol=0, atol=2e-8)
    for before, after in zip(originals, (mesh.vertices, mesh.triangles,
                            mesh.normals, mesh.velocity, mesh.foam), strict=True):
        np.testing.assert_array_equal(before, after)


def test_offshore_seam_rejects_a_vertical_backtrack():
    mesh = _stepped_surface(((0, .2), (.25, .2), (.25, .3),
                             (.25, .25), (.25, .4), (1, .4)))
    with pytest.raises(MultiValuedOffshoreSeamError, match="multi-valued water surface"):
        _ordered_seam(mesh, 0.0)


def test_open_handoff_omits_only_facets_entirely_on_the_cut():
    from dataclasses import replace

    mesh = _stepped_surface(((0, .2), (.25, .2), (.25, .3), (1, .3)))
    # An x-normal cap on the newly introduced open boundary. Both cap
    # orientations have the same domain ownership; this is no branch filter.
    for cap in ((0, 1, 2), (2, 1, 0)):
        capped = replace(mesh, triangles=np.r_[mesh.triangles, [cap]])
        clipped = clip_local_mesh(capped, 0.0)
        np.testing.assert_array_equal(
            clipped.vertices[clipped.triangles], mesh.vertices[mesh.triangles],
        )
        _ordered_seam(clipped, 0.0)


@pytest.mark.parametrize("step_height", [0.1, float(np.spacing(np.float32(.2)))])
def test_parametric_collar_keeps_steps_without_holes_or_overlapping_triangles(step_height):
    from dataclasses import replace

    profile = ((0, .2), (.25, .2), (.25, .2+step_height),
               (.5, .2+step_height), (.5, .2), (1, .2))
    source = _stepped_surface(profile)
    mesh = _tile_periodic_y(source, 1.0)
    inlet = LinearWaveInlet((replace(_inlet().components[0], alongshore_mode=1),))
    seam = _periodic_seam(mesh, inlet, 0.0)
    at_seam, normal, slope = _parametric_collar(
        seam, inlet, np.zeros_like(seam.parameter), seam.parameter,
        .23, .30, 0.0, .4, .05,
    )
    np.testing.assert_allclose(at_seam, mesh.vertices[seam.indices], atol=2e-14, rtol=0)
    np.testing.assert_allclose(slope, seam.slope, atol=2e-14, rtol=0)
    np.testing.assert_allclose(normal[:, 0]+slope*normal[:, 2], 0, atol=2e-14, rtol=0)
    if step_height < 1e-7:
        # A vanishing vertical step must not turn the adjacent broad faces'
        # interpolated reflection normals sideways. Its own vertical facet
        # remains present, with its actual (vanishing) area.
        assert np.all(normal[:, 2] > .99)
    outer, _, outer_slope = _parametric_collar(
        seam, inlet, np.full_like(seam.parameter, -.4), seam.parameter,
        .23, .30, 0.0, .4, .05,
    )
    from shoreloop.render.offshore_join import _inlet_surface
    expected_height, expected_slope = _inlet_surface(
        inlet, outer[:, 0], outer[:, 1], .23, .30,
    )
    np.testing.assert_array_equal(outer[:, 2], expected_height)
    np.testing.assert_array_equal(outer_slope, expected_slope)

    joined = _attach_offshore_mesh(mesh, inlet, .23, .30, 0.0, .4, .2, .05)
    np.testing.assert_array_equal(joined.vertices[:len(mesh.vertices)], mesh.vertices)
    np.testing.assert_array_equal(joined.triangles[:len(mesh.triangles)], mesh.triangles)
    np.testing.assert_array_equal(joined.velocity[:len(mesh.vertices)], mesh.velocity)
    np.testing.assert_array_equal(joined.foam[:len(mesh.vertices)], mesh.foam)
    assert np.isfinite(joined.normals).all()
    new_triangles = joined.triangles[len(mesh.triangles):]
    points = joined.vertices[new_triangles]
    areas = np.cross(points[:, 1]-points[:, 0], points[:, 2]-points[:, 0])
    assert np.all(np.any(areas != 0, axis=1))
    assert np.all(areas[:, 2] >= 0)
    # The projected strip tiles its rectangle once, with only the genuine
    # vertical seam facets having zero projected area.
    assert np.sum(areas[:, 2])/2 == pytest.approx(3*float(np.float32(.6)), abs=1e-12)
    assert np.count_nonzero(areas[:, 2] == 0) == 6
    period = _fundamental_period_mesh(joined, 0.0, 1.0)
    retiled = _tile_periodic_y(period, 1.0)
    edges, counts = np.unique(np.sort(np.concatenate([
        retiled.triangles[:, pair] for pair in ((0, 1), (1, 2), (2, 0))
    ]), axis=1), axis=0, return_counts=True)
    points = retiled.vertices[edges]
    exterior = (
        np.all(points[:, :, 0] == retiled.vertices[:, 0].min(), axis=1)
        | np.all(points[:, :, 0] == retiled.vertices[:, 0].max(), axis=1)
        | np.all(points[:, :, 1] == -1, axis=1)
        | np.all(points[:, :, 1] == 2, axis=1)
    )
    assert np.all(counts[~exterior] == 2)
    assert np.all(counts[exterior] == 1)


def test_discrete_seam_tangents_do_not_carve_the_full_height_collar():
    """Frame 14: alternating PLIC tangents must stay local, with their exact end jet."""
    from dataclasses import replace

    inlet = LinearWaveInlet((replace(_inlet().components[0], amplitude=0.0),))
    profile = tuple((float(y), .3) for y in np.linspace(0, 1, 65))
    mesh = _tile_periodic_y(_stepped_surface(profile, slope=0.0), 1.0)
    seam = _periodic_seam(mesh, inlet, 0.0)
    seam = replace(seam, slope=.6*np.cos(2*np.pi*16*seam.y))
    width = .01
    q = seam.parameter
    at_seam, normals, slope = _parametric_collar(
        seam, inlet, np.zeros_like(q), q, 0.0, float(np.float32(.3)), 0.0, .4, width,
    )
    np.testing.assert_allclose(at_seam, mesh.vertices[seam.indices], rtol=0, atol=2e-14)
    np.testing.assert_allclose(slope, seam.slope, rtol=0, atol=2e-14)
    np.testing.assert_allclose(normals[:, 0]+slope*normals[:, 2], 0, rtol=0, atol=2e-14)

    x = np.linspace(-.4, 0, 401)[:, None]
    positions, _, _ = _parametric_collar(
        seam, inlet, x, q[None, :], 0.0, float(np.float32(.3)), 0.0, .4, width,
    )
    displacement = positions[:, :, 2]-float(np.float32(.3))
    # Exactly untouched outside the support, even for this highly varying jet.
    np.testing.assert_array_equal(displacement[x[:, 0] <= -width], 0)
    assert np.max(np.abs(displacement)) <= 16/81*width*.6 + 2e-14
    assert np.max(np.abs(displacement)) > .001
    # Geometric x derivatives agree with the matched jet inside its support.
    location = -.004
    epsilon = 1e-7
    before, _, _ = _parametric_collar(
        seam, inlet, np.full_like(q, location-epsilon), q,
        0.0, float(np.float32(.3)), 0.0, .4, width,
    )
    after, _, _ = _parametric_collar(
        seam, inlet, np.full_like(q, location+epsilon), q,
        0.0, float(np.float32(.3)), 0.0, .4, width,
    )
    _, _, derivative = _parametric_collar(
        seam, inlet, np.full_like(q, location), q,
        0.0, float(np.float32(.3)), 0.0, .4, width,
    )
    np.testing.assert_allclose((after[:, 2]-before[:, 2])/(2*epsilon), derivative,
                               rtol=0, atol=5e-9)


def test_collar_shading_uses_rendered_chords_for_subcell_seam_steps():
    """A tiny steep seam segment must not turn a broad collar face backward."""
    from dataclasses import replace

    inlet = LinearWaveInlet((replace(_inlet().components[0], amplitude=0.0),))
    profile = ((0, .3), (.25, .3), (.250001, .300001),
               (.250002, .300002), (.26, .3), (1, .3))
    mesh = _tile_periodic_y(_stepped_surface(profile, slope=0.0), 1.0)
    seam = _periodic_seam(mesh, inlet, 0.0)
    samples = (seam.parameter[seam.canonical[2]] + .01*np.arange(-4, 5))[None, :]
    x = np.array([[-.05]])
    args = (seam, inlet, x, samples, 0.0, float(np.float32(.3)), 0.0, .4, .01)
    positions, original_normals, original_x = _parametric_collar(*args)
    sampled, normals, derivative_x = _parametric_collar(*args, sampled_normals=True)
    np.testing.assert_array_equal(sampled, positions)
    np.testing.assert_array_equal(derivative_x, original_x)

    epsilon = 1e-7
    before = _parametric_collar(seam, inlet, x-epsilon, *args[3:])[0]
    after = _parametric_collar(seam, inlet, x+epsilon, *args[3:])[0]
    tangent_x = (after[0, 4]-before[0, 4])/(2*epsilon)
    tangent_q = positions[0, 5]-positions[0, 3]
    expected = np.cross(tangent_x, tangent_q)
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(normals[0, 4], expected, atol=2e-9, rtol=0)
    assert normals[0, 4, 2] > .999
    assert np.linalg.norm(original_normals[0, 4]-expected) > .2


def test_offshore_mesh_uses_the_exact_clipping_plane():
    vertices = np.asarray((
        (-0.1, 0.0, 0.0),
        (0.1, 0.0, 0.0),
        (0.1, 1.0, 0.0),
        (-0.1, 1.0, 0.0),
        (9.0e-7, 0.5, 5.0e-7),
        (0.1, 0.4, 5.0e-7),
        (0.1, 0.6, 5.0e-7),
    ), dtype=np.float32)
    triangles = np.asarray((
        (0, 1, 2),
        (0, 2, 3),
        (4, 5, 6),
    ), dtype=np.int32)
    mesh = WaterSurfaceMesh(
        vertices=vertices,
        triangles=triangles,
        normals=np.tile((0.0, 0.0, 1.0), (len(vertices), 1)).astype(
            np.float32
        ),
        velocity=np.zeros_like(vertices),
        foam=np.zeros(len(vertices), dtype=np.float32),
        vertex_open_boundaries=np.zeros(len(vertices), dtype=np.uint8),
        open_boundary_edges=np.empty((0, 2), dtype=np.int32),
        open_boundary_labels=np.empty(0, dtype=np.uint8),
    )

    clipped = clip_local_mesh(mesh, 0.0)
    assert np.any(clipped.vertices[:, 0] == np.float32(9.0e-7))
    with np.errstate(divide="raise", invalid="raise"):
        joined = _attach_offshore_mesh(
            clipped,
            _inlet(),
            0.23,
            0.30,
            0.0,
            0.4,
            0.2,
            0.05,
        )

    assert np.all(np.isfinite(joined.vertices))
    assert np.all(np.isfinite(joined.normals))


def test_periodic_clipping_rounds_opposite_edge_traversals_to_the_same_point():
    # Inclined source-control edge: reversed traversal formerly rounded the
    # two periodic intersection heights to adjacent FP32 values.
    a = (.07500000298023224, .21475766599178314)
    b = (.0995156317949295, .2199999988079071)
    vertices = np.asarray(((a[0], 0, a[1]), (b[0], 0, b[1]),
                           (a[0], 1, a[1]), (b[0], 1, b[1])), dtype=np.float32)
    mesh = WaterSurfaceMesh(vertices,
        np.asarray(((0, 1, 3), (0, 3, 2)), dtype=np.int32),
        np.tile((0, 0, 1), (4, 1)).astype(np.float32),
        np.zeros((4, 3), np.float32), np.zeros(4, np.float32),
        np.zeros(4, np.uint8), np.empty((0, 2), np.int32), np.empty(0, np.uint8))
    cut = np.float32(.0925)
    clipped = clip_local_mesh(mesh, float(cut))
    expected_z = np.float32(a[1]+(float(cut)-a[0])/(b[0]-a[0])*(b[1]-a[1]))
    seam = clipped.vertices[clipped.vertices[:, 0] == cut]
    np.testing.assert_array_equal(seam[:, 2], np.full(len(seam), expected_z))
    _tile_periodic_y(clipped, 1)


def test_offshore_seam_rejects_offset_samples_from_overlapping_branches():
    mesh = _local_surface(_inlet(), 0.23, eta0=0.30)
    first = len(mesh.vertices)
    branch_vertices = np.asarray((
        (0.0, 0.2001, 0.60),
        (0.0, 0.8001, 0.60),
        (0.1, 0.5001, 0.60),
    ), dtype=np.float32)
    mesh = WaterSurfaceMesh(
        vertices=np.concatenate((mesh.vertices, branch_vertices)),
        triangles=np.concatenate((
            mesh.triangles,
            np.asarray(((first, first + 2, first + 1),), dtype=np.int32),
        )),
        normals=np.concatenate((
            mesh.normals,
            np.tile(np.asarray(((0.0, 0.0, 1.0),), dtype=np.float32), (3, 1)),
        )),
        velocity=np.concatenate((
            mesh.velocity,
            np.zeros((3, 3), dtype=np.float32),
        )),
        foam=np.concatenate((mesh.foam, np.zeros(3, dtype=np.float32))),
        vertex_open_boundaries=np.concatenate((
            mesh.vertex_open_boundaries,
            np.zeros(3, dtype=np.uint8),
        )),
        open_boundary_edges=mesh.open_boundary_edges,
        open_boundary_labels=mesh.open_boundary_labels,
    )

    with pytest.raises(
        MultiValuedOffshoreSeamError,
        match="multi-valued water surface",
    ):
        _ordered_seam(mesh, 0.0)


def test_offshore_seam_rejects_a_missing_interval():
    mesh = _local_surface(_inlet(), 0.23, eta0=0.30)
    mesh = WaterSurfaceMesh(
        vertices=mesh.vertices,
        triangles=np.delete(mesh.triangles, 7, axis=0),
        normals=mesh.normals,
        velocity=mesh.velocity,
        foam=mesh.foam,
        vertex_open_boundaries=mesh.vertex_open_boundaries,
        open_boundary_edges=mesh.open_boundary_edges,
        open_boundary_labels=mesh.open_boundary_labels,
    )

    with pytest.raises(
        MultiValuedOffshoreSeamError,
        match="continuous water graph",
    ):
        _ordered_seam(mesh, 0.0)


def test_periodic_local_mesh_is_welded_without_an_internal_side():
    mesh = _local_surface(_inlet(), 0.23, eta0=0.30)
    tiled = _tile_periodic_y(mesh, period=1.0)

    assert float(np.min(tiled.vertices[:, 1])) == -1.0
    assert float(np.max(tiled.vertices[:, 1])) == 2.0
    assert len(tiled.triangles) == 3 * len(mesh.triangles)
    edges = np.sort(np.concatenate((
        tiled.triangles[:, (0, 1)],
        tiled.triangles[:, (1, 2)],
        tiled.triangles[:, (2, 0)],
    )), axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    endpoints = tiled.vertices[unique_edges]
    for y in (0.0, 1.0):
        seam = np.all(endpoints[:, :, 1] == np.float32(y), axis=1)
        assert np.count_nonzero(seam) == 16
        assert np.all(counts[seam] == 2)
    for y in (-1.0, 2.0):
        outer_side = np.all(endpoints[:, :, 1] == np.float32(y), axis=1)
        assert np.count_nonzero(outer_side) == 16
        assert np.all(counts[outer_side] == 1)


@pytest.mark.parametrize("attribute", ["vertices", "normals", "velocity", "foam"])
def test_periodic_local_mesh_rejects_mismatched_boundary_data(attribute):
    mesh = _local_surface(_inlet(), 0.23, eta0=0.30)
    high = np.flatnonzero(mesh.vertices[:, 1] == np.float32(1.0))
    values = getattr(mesh, attribute)
    index = high[len(high) // 2]
    if values.ndim == 1:
        values[index] += np.float32(1.0e-3)
    else:
        values[index, 2] += np.float32(1.0e-3)

    # A moved vertex leaves its cut edges without a partner; moved attributes
    # keep the geometric partner but break the weld.
    message = ("unmatched periodic y boundaries" if attribute == "vertices"
               else "not continuous across periodic y")
    with pytest.raises(ValueError, match=message):
        _tile_periodic_y(mesh, period=1.0)


def _periodic_edge_contact():
    return WaterSurfaceMesh.read(
        Path(__file__).parent / "data" / "periodic_plic_edge_contact.npz",
    )


@pytest.mark.parametrize("renumber", [False, True])
def test_periodic_edge_contact_preserves_each_owned_facet_once(renumber):
    from dataclasses import replace

    source = _periodic_edge_contact()
    if renumber:
        order = np.random.default_rng(123).permutation(len(source.vertices))
        inverse = np.argsort(order)
        source = replace(source, vertices=source.vertices[order],
                         triangles=inverse[source.triangles[::-1]],
                         normals=source.normals[order], velocity=source.velocity[order],
                         foam=source.foam[order],
                         vertex_open_boundaries=source.vertex_open_boundaries[order])
    original = [a.copy() for a in (source.vertices, source.triangles, source.normals,
                                   source.velocity, source.foam)]
    tiled = _tile_periodic_y(source, 6.0)
    period = _fundamental_period_mesh(tiled, 0.0, 6.0)
    retiled = _tile_periodic_y(period, 6.0)

    def owned_facets(mesh):
        points = mesh.vertices[mesh.triangles]
        y = points[:, :, 1]
        owned = np.all((y >= 0) & (y <= 6), axis=1) & ~np.all(y == 6, axis=1)
        return sorted(tuple(sorted(tuple(v) for v in triangle)) for triangle in points[owned])

    expected, actual = owned_facets(tiled), owned_facets(retiled)
    assert actual == expected
    assert len(actual) == len(set(actual))
    edges, counts = np.unique(np.sort(np.concatenate([
        retiled.triangles[:, p] for p in ((0, 1), (1, 2), (2, 0))
    ]), axis=1), axis=0, return_counts=True)
    points = retiled.vertices[edges]
    contact = (np.all(points[:, :, 0] == float(np.float32(1.025)), axis=1)
               & np.all(points[:, :, 2] >= float(np.float32(.239970163)), axis=1)
               & np.all(points[:, :, 2] <= float(np.float32(.24)), axis=1))
    for y in (0, 6):
        selected = contact & np.all(points[:, :, 1] == y, axis=1)
        assert np.count_nonzero(selected) == 1
        assert counts[selected][0] == 4
    for before, after in zip(original, (source.vertices, source.triangles, source.normals,
                                       source.velocity, source.foam), strict=True):
        np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("damage", ["reverse", "remove", "duplicate"])
def test_periodic_edge_contact_rejects_a_damaged_oriented_boundary(damage):
    from dataclasses import replace

    period = _fundamental_period_mesh(_tile_periodic_y(_periodic_edge_contact(), 6), 0, 6)
    vertices = period.vertices
    edge = np.flatnonzero((vertices[:, 0] == float(np.float32(1.025)))
                          & (vertices[:, 1] == 6)
                          & (vertices[:, 2] >= float(np.float32(.239970163)))
                          & (vertices[:, 2] <= float(np.float32(.24))))
    assert len(edge) == 2
    incident = np.flatnonzero(np.sum(np.isin(period.triangles, edge), axis=1) == 2)
    assert len(incident) == 1
    index = incident[0]
    triangles = period.triangles.copy()
    if damage == "reverse":
        triangles[index] = triangles[index, ::-1]
    elif damage == "remove":
        triangles = np.delete(triangles, index, axis=0)
    else:
        triangles = np.r_[triangles, triangles[index:index+1]]
    with pytest.raises(ValueError, match="periodic y"):
        _tile_periodic_y(replace(period, triangles=triangles), 6)


def _one_sided_seam_patch_surface(*, patch: bool = True) -> WaterSurfaceMesh:
    """Periodic surface plus a down-facing facet ending on the y=1 cut.

    The cut trace of that facet is closed only by a cut-face patch owned by
    the y=0 side, as a full cell closes the gap to a PLIC neighbour across
    the period. The patch's other two edges end on an omitted solid face.
    """
    base = _local_surface(_inlet(), 0.23, eta0=0.30)
    facet = [(1.0, 0.75, 0.10), (1.5, 0.75, 0.15), (1.5, 1.0, 0.15), (1.0, 1.0, 0.10)]
    seam_patch = [(1.0, 0.0, 0.10), (1.5, 0.0, 0.15), (1.5, 0.0, 0.10)]
    points = facet + (seam_patch if patch else [])
    n = len(base.vertices)
    triangles = [(n, n+2, n+1), (n, n+3, n+2)]
    if patch:
        triangles.append((n+4, n+6, n+5))
    down = np.asarray((0.1, 0.0, -1.0)) / np.hypot(0.1, 1.0)
    normals = np.tile(down, (len(points), 1))
    if patch:
        normals[-1] = (0.0, -1.0, 0.0)
    vertices = np.concatenate((base.vertices, np.asarray(points, dtype=np.float32)))
    return WaterSurfaceMesh(
        vertices,
        np.concatenate((base.triangles, np.asarray(triangles, dtype=np.int32))),
        np.concatenate((base.normals, normals.astype(np.float32))),
        np.zeros_like(vertices), np.zeros(len(vertices), dtype=np.float32),
        np.zeros(len(vertices), dtype=np.uint8), np.empty((0, 2), dtype=np.int32),
        np.empty(0, dtype=np.uint8),
    )


def _is_exact_translate(tiled: WaterSurfaceMesh, source: WaterSurfaceMesh, period: float) -> bool:
    positions = {tuple(v) for v in source.vertices.astype(np.float64).tolist()}
    return all(
        any((x, y - copy * period, z) in positions for copy in (-1, 0, 1))
        for x, y, z in tiled.vertices.tolist()
    )


def test_one_sided_seam_patch_welds_its_trace_and_leaves_solid_contact_edges_open():
    mesh = _one_sided_seam_patch_surface()
    n = len(mesh.vertices) - 7
    boundary = _periodic_y_boundary_indices(mesh, 1.0)
    assert boundary.unwelded_edges == 0 and boundary.unwelded_x_min is None
    pairs = dict(zip(boundary.low.tolist(), boundary.high.tolist(), strict=True))
    assert pairs[n+4] == n+3 and pairs[n+5] == n+2
    assert n+6 not in boundary.low and n+6 not in boundary.high

    tiled = _tile_periodic_y(mesh, 1.0, boundary)
    assert _is_exact_translate(tiled, mesh, 1.0)
    edges, counts = np.unique(np.sort(np.concatenate([
        tiled.triangles[:, p] for p in ((0, 1), (1, 2), (2, 0))
    ]), axis=1), axis=0, return_counts=True)
    points = tiled.vertices[edges]
    z_low, z_high = float(np.float32(0.10)), float(np.float32(0.15))
    on_cut = np.all(points[:, :, 1] == 1.0, axis=1)
    low_z = np.all(points[:, :, 2] <= z_high, axis=1)
    # The welded trace is the diagonal edge; the open solid edges are axis-aligned.
    trace = on_cut & low_z & (points[:, 0, 0] != points[:, 1, 0]) & (points[:, 0, 2] != points[:, 1, 2])
    solid = on_cut & low_z & ~trace & np.any(points[:, :, 2] == z_low, axis=1)
    assert np.count_nonzero(trace) == 1 and counts[trace][0] == 2
    assert np.count_nonzero(solid) == 2 and np.all(counts[solid] == 1)

    period = _fundamental_period_mesh(tiled, 0.0, 1.0)
    retiled = _tile_periodic_y(period, 1.0)

    def owned_facets(result):
        corners = result.vertices[result.triangles]
        y = corners[:, :, 1]
        owned = np.all((y >= 0) & (y <= 1), axis=1) & ~np.all(y == 1, axis=1)
        return sorted(tuple(sorted(tuple(v) for v in triangle)) for triangle in corners[owned])

    assert owned_facets(retiled) == owned_facets(tiled)
    assert len(owned_facets(tiled)) == len(mesh.triangles)

    with pytest.raises(ValueError, match="unmatched periodic y boundaries"):
        _periodic_y_boundary_indices(_one_sided_seam_patch_surface(patch=False), 1.0)


@pytest.mark.parametrize("attribute, message", [
    ("vertices", "unmatched periodic y boundaries"),
    ("foam", "not continuous across periodic y"),
])
def test_inshore_seam_mismatch_is_counted_only_beyond_the_periodic_x_limit(attribute, message):
    mesh = _local_surface(_inlet(), 0.23, eta0=0.30)
    index = np.flatnonzero(
        (mesh.vertices[:, 1] == np.float32(1.0)) & (mesh.vertices[:, 0] == np.float32(1.0)),
    )
    assert len(index) == 1
    if attribute == "vertices":
        mesh.vertices[index[0], 2] += np.float32(1.0e-3)
    else:
        mesh.foam[index[0]] = np.float32(0.5)

    with pytest.raises(ValueError, match=message):
        _periodic_y_boundary_indices(mesh, 1.0)
    # The nearer affected cut edge starts at x = 0.875.
    with pytest.raises(ValueError, match=message):
        _periodic_y_boundary_indices(mesh, 1.0, periodic_x_limit=0.875)

    boundary = _periodic_y_boundary_indices(mesh, 1.0, periodic_x_limit=0.8)
    assert boundary.unwelded_edges == 4
    assert boundary.unwelded_x_min == 0.875
    assert len(boundary.low) == 16
    tiled = _tile_periodic_y(mesh, 1.0, boundary)
    assert len(tiled.vertices) == 3 * len(mesh.vertices) - 2 * 16
    assert _is_exact_translate(tiled, mesh, 1.0)
    _fundamental_period_mesh(tiled, 0.0, 1.0, 0.8)
    with pytest.raises(ValueError, match=message):
        _fundamental_period_mesh(tiled, 0.0, 1.0)


def test_offshore_join_is_geometrically_continuous_and_exactly_periodic():
    inlet = _inlet()
    time = 0.23
    mesh = _local_surface(inlet, time, eta0=0.30)
    camera = Camera(
        pos=(1.7, 0.5, 0.75),
        yaw_deg=180.0,
        pitch_deg=22.0,
        vfov_deg=34.0,
        width=48,
        height=32,
    )
    quiet = RippleSpectrum(
        period=inlet.period,
        harmonics=(8,),
        amplitudes=(0.0,),
        directions=(0.0,),
        phases=(0.0,),
        mod_amps=(),
        mod_harmonics=(),
        mod_wavelengths=(),
        mod_directions=(),
        mod_phases=(),
    )
    sky, sun = light_preset("sunset")
    view = render_offshore_view(
        mesh,
        inlet,
        time,
        camera,
        eta0=0.30,
        device=DEVICE,
        handoff_x=0.07,
        offshore_mesh_length=0.2,
        cell_size=0.05,
        render_scale=1,
        sky=sky,
        sun=sun,
        spectrum=quiet,
    )

    assert view.metrics.maximum_height_jump < 2.0e-8
    assert view.metrics.maximum_slope_jump < 2.0e-8
    assert view.metrics.maximum_outer_height_jump < 2.0e-8
    assert view.metrics.maximum_outer_slope_jump < 2.0e-8
    assert view.metrics.maximum_resolved_analytic_offset > 0.015
    assert view.metrics.maximum_resolved_analytic_slope_offset > 0.0
    assert view.metrics.overlap_width == 0.0
    assert view.periodic_y_period == 1.0
    assert float(np.min(view.local_mesh.vertices[:, 1])) == -1.0
    assert float(np.max(view.local_mesh.vertices[:, 1])) == 2.0
    assert float(np.min(view.periodic_instance_mesh.vertices[:, 1])) == 0.0
    assert float(np.max(view.periodic_instance_mesh.vertices[:, 1])) == 1.0
    instance_points = view.periodic_instance_mesh.vertices[
        view.periodic_instance_mesh.triangles
    ]
    assert np.all(np.any(np.cross(
        instance_points[:, 1] - instance_points[:, 0],
        instance_points[:, 2] - instance_points[:, 0],
    ) != 0.0, axis=1))
    assert float(np.min(view.local_mesh.vertices[:, 0])) == np.float32(-0.53)
    assert np.any(view.local_mesh.vertices[:, 0] == np.float32(0.07))
    assert np.all(
        view.local_mesh.vertices[view.local_mesh.triangles, 0] >= -0.53
    )
    edges = np.sort(np.concatenate((
        view.local_mesh.triangles[:, (0, 1)],
        view.local_mesh.triangles[:, (1, 2)],
        view.local_mesh.triangles[:, (2, 0)],
    )), axis=1)
    unique_edges, edge_counts = np.unique(edges, axis=0, return_counts=True)
    points = view.local_mesh.vertices[unique_edges]
    for seam_x in (view.metrics.handoff_x, view.metrics.collar_outer_x):
        seam_edges = (
            np.all(np.isclose(points[:, :, 0], seam_x, atol=1.0e-6), axis=1)
            & np.all((points[:, :, 1] >= 0.0) & (points[:, :, 1] <= 1.0), axis=1)
        )
        seam_vertices = np.isclose(
            view.local_mesh.vertices[:, 0], seam_x, atol=1.0e-6,
        ) & (
            (view.local_mesh.vertices[:, 1] >= 0.0)
            & (view.local_mesh.vertices[:, 1] <= 1.0)
        )
        assert np.count_nonzero(seam_edges) == (
            len(np.unique(view.local_mesh.vertices[seam_vertices, 1])) - 1
        )
        assert np.all(edge_counts[seam_edges] == 2)
    assert np.all(np.isfinite(view.image))
    assert float(np.ptp(view.image)) > 0.02
    assert view.renderer.sky is sky
    assert view.renderer.sun is sun
    np.testing.assert_allclose(
        view.renderer._ff_kx[:2],
        (inlet.components[0].kx, 2.0 * inlet.components[0].kx),
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        view.renderer.render(time + inlet.period),
        view.image,
        rtol=0.0,
        atol=2.0e-6,
    )
    ripple_mask = view.renderer._ripple_mask.numpy()
    assert np.all(ripple_mask[-1] == 0.0)
    assert np.all(ripple_mask[0] > 0.0)


def test_exact_offshore_interchange_keeps_bound_harmonic_dispersion():
    inlet = _inlet()
    component = inlet.components[0]
    field = exact_offshore_field(inlet)
    assert field.harmonics == (1, 2)
    assert field.kx == (component.kx, 2.0 * component.kx)
    assert field.ky == (component.ky, 2.0 * component.ky)
    assert field.phases == (component.phase_offset, 2.0 * component.phase_offset)


def test_camera_plate_transfers_linear_pixels_without_mirroring():
    mi.set_variant("llvm_ad_rgb")
    camera = Camera(
        pos=(1.0, 2.0, 3.0),
        yaw_deg=137.0,
        pitch_deg=19.0,
        vfov_deg=37.0,
        width=32,
        height=24,
    )
    image = np.empty((camera.height, camera.width, 3), dtype=np.float32)
    image[:12, :16] = (0.8, 0.1, 0.1)
    image[:12, 16:] = (0.1, 0.8, 0.1)
    image[12:, :16] = (0.1, 0.1, 0.8)
    image[12:, 16:] = (0.7, 0.7, 0.1)
    scene = mi.load_dict({
        "type": "scene",
        "integrator": {"type": "path", "max_depth": 2},
        "sensor": _sensor(camera),
        "sky": {
            "type": "constant",
            "radiance": {"type": "rgb", "value": (0.01, 0.01, 0.01)},
        },
        "plate": _background_plate(camera, image),
    })
    rendered = np.asarray(mi.render(scene, spp=1, seed=707))
    for ys, xs in (
        (slice(2, 10), slice(2, 14)),
        (slice(2, 10), slice(18, 30)),
        (slice(14, 22), slice(2, 14)),
        (slice(14, 22), slice(18, 30)),
    ):
        np.testing.assert_allclose(
            rendered[ys, xs].mean(axis=(0, 1)),
            image[ys, xs].mean(axis=(0, 1)),
            rtol=0.0,
            atol=2.0e-3,
        )


def test_subcell_seam_fold_is_welded_and_larger_folds_stay_refused():
    from shoreloop.render.offshore_join import _weld_subcell_seam_folds
    # A 5 mm double-back within .4 mm of height: below the resolved geometry.
    folded = _stepped_surface(((0, .2), (.25, .2), (.26, .2004), (.255, .2002),
                               (.27, .2), (.5, .2), (1, .2)))
    with pytest.raises(MultiValuedOffshoreSeamError):
        _ordered_seam(folded, 0.0)
    welded, count, shift = _weld_subcell_seam_folds(folded, 0.0, .02, .005)
    assert count == 1
    assert shift == pytest.approx(np.hypot(.005, .0002), rel=1e-3)
    assert len(welded.vertices) == len(folded.vertices)-1
    assert len(welded.triangles) < len(folded.triangles)
    boundary, y, height, _ = _ordered_seam(welded, 0.0)
    assert np.all(np.diff(y) >= 0)
    assert y[0] == 0 and y[-1] == 1
    assert len(boundary) == 6

    # The same double-back with a 3 cm rise is an overhang: refused, unwelded.
    overhang = _stepped_surface(((0, .2), (.25, .2), (.26, .23), (.255, .2),
                                 (.27, .2), (.5, .2), (1, .2)))
    same, count, shift = _weld_subcell_seam_folds(overhang, 0.0, .02, .005)
    assert count == 0 and shift == 0.0 and same is overhang
    with pytest.raises(MultiValuedOffshoreSeamError):
        _ordered_seam(overhang, 0.0)

    # A 3 cm double-back exceeds a cell: refused, unwelded.
    wide = _stepped_surface(((0, .2), (.25, .2), (.30, .2004), (.27, .2002),
                             (.31, .2), (.5, .2), (1, .2)))
    same, count, _ = _weld_subcell_seam_folds(wide, 0.0, .02, .005)
    assert count == 0 and same is wide
    with pytest.raises(MultiValuedOffshoreSeamError):
        _ordered_seam(wide, 0.0)


def test_zero_thickness_seam_sliver_is_removed_and_a_real_bubble_is_refused():
    from dataclasses import replace
    from shoreloop.render.offshore_join import _remove_seam_slivers

    def with_loop(mesh, height, thickness):
        # A closed three-vertex contour on the cut plane, capped by one vertex
        # just inside the local half-space: the cut of a tiny closed sheet.
        base = len(mesh.vertices)
        extra = np.asarray([(0.0, .4, height), (0.0, .41, height+thickness),
                            (0.0, .405, height), (.01, .405, height)], np.float32)
        vertices = np.concatenate((mesh.vertices, extra))
        triangles = np.concatenate((mesh.triangles, np.asarray(
            [(base, base+1, base+3), (base+1, base+2, base+3), (base+2, base, base+3)], np.int32)))
        pad = lambda a, fill: np.concatenate((a, np.full((4, *a.shape[1:]), fill, a.dtype)))
        return replace(mesh, vertices=vertices, triangles=triangles,
                       normals=pad(mesh.normals, 0), velocity=pad(mesh.velocity, 0),
                       foam=pad(mesh.foam, 0), vertex_open_boundaries=pad(mesh.vertex_open_boundaries, 0))

    smooth = _stepped_surface(((0, .2), (.5, .2), (1, .2)))
    sliver = with_loop(smooth, .2002, .0001)
    with pytest.raises(MultiValuedOffshoreSeamError):
        _ordered_seam(sliver, 0.0)
    cleaned, removed = _remove_seam_slivers(sliver, 0.0, .005)
    assert removed == 1
    assert len(cleaned.vertices) == len(smooth.vertices)
    _ordered_seam(cleaned, 0.0)

    bubble = with_loop(smooth, .17, .0001)
    same, removed = _remove_seam_slivers(bubble, 0.0, .005)
    assert removed == 0 and same is bubble
    with pytest.raises(MultiValuedOffshoreSeamError):
        _ordered_seam(bubble, 0.0)
