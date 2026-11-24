"""Camera-phase join between local mesh water and the analytic offshore view.

The local 3-D surface is clipped at a calm plane, then welded to a narrow
quintic correction collar and several wavelengths of the exact analytic
incoming field.  Only the genuinely distant, pixel-footprint-filtered horizon
comes from the existing height-field renderer.  Its linear image is used as a
camera plate behind the mesh, so a primary ray sees at most one geometric
water surface while the established outer renderer retains its far-field
prefilter and lost-slope-variance model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shoreloop.closures.ripples import (
    RippleSpectrum,
    crossing_ripple_spectrum,
)
from shoreloop.config import (
    BC,
    BCConfig,
    GridConfig,
    PhysicsConfig,
    SolverConfig,
)
from shoreloop.physics.swe import SWESolver
from shoreloop.physics3d.waves import LinearWaveInlet
from shoreloop.render.camera import Camera
from shoreloop.render.raytrace import (
    OffshoreWaveField,
    Raytracer,
    WaterOptics,
)
from shoreloop.render.sky import OvercastSky, Sun
from shoreloop.render.surface_mesh import WaterSurfaceMesh
from shoreloop.render.texture import PebbleTexture


OFFSHORE_ALONGSHORE_REPEATS = 1


def periodic_render_y_bounds(y0: float, period: float) -> tuple[float, float]:
    """Alongshore solid extent matching the instantiated water footprint."""
    return (
        y0 - OFFSHORE_ALONGSHORE_REPEATS * period,
        y0 + (OFFSHORE_ALONGSHORE_REPEATS + 1) * period,
    )


@dataclass(frozen=True)
class OffshoreJoinMetrics:
    """Geometric measurements at the local/outer render handoff."""

    handoff_x: float
    collar_outer_x: float
    outer_edge_x: float
    collar_width: float
    tangent_width: float
    tangent_height_bound_at_seam_samples_m: float
    maximum_height_jump: float
    maximum_slope_jump: float
    maximum_outer_height_jump: float
    maximum_outer_slope_jump: float
    maximum_resolved_analytic_offset: float
    maximum_resolved_analytic_slope_offset: float
    overlap_width: float
    maximum_collar_alongshore_displacement: float
    minimum_collar_triangle_area: float
    maximum_boundary_tangent_residual: float
    welded_seam_folds: int
    maximum_weld_shift_m: float
    removed_seam_slivers: int
    unwelded_periodic_seam_edges: int
    unwelded_periodic_seam_x_min: float | None


@dataclass(frozen=True)
class OffshoreView:
    """One outer image, exact tiled diagnostic, and local render period."""

    local_mesh: WaterSurfaceMesh
    periodic_instance_mesh: WaterSurfaceMesh
    periodic_y_period: float
    image: np.ndarray
    ripple_spectrum: RippleSpectrum
    metrics: OffshoreJoinMetrics
    renderer: Raytracer


class MultiValuedOffshoreSeamError(ValueError):
    """The handoff is not a single monotone polyline, including vertical steps."""


@dataclass(frozen=True)
class _PeriodicBoundary:
    """Welded vertex pairs of the y period cut and the cut edges left open."""

    low: np.ndarray
    high: np.ndarray
    unwelded_edges: int
    unwelded_x_min: float | None


def _periodic_y_boundary_indices(
    mesh: WaterSurfaceMesh,
    period: float,
    periodic_x_limit: float = np.inf,
) -> _PeriodicBoundary:
    """Pair the vertices of the oriented periodic boundary for welding.

    A PLIC footprint jump can expose a vertical face on the period cut.
    Its internal vertices/edges belong to that face, not to the sewing edge.
    Edges on the independent open x ends also remain open after tiling.
    At an edge contact, cut ownership can leave three incident faces on one
    side and one on the other. Opposite directed incidences cancel; their
    remaining boundary coefficients must match with opposite orientations,
    and the paired vertices must carry identical attributes.

    Two kinds of cut edge have no partner by construction and stay open:

    * An open edge of a cut-face patch that bounds nothing else. The patch
      closes the gap between the two neighbouring liquid sections and is
      owned by one side; where it meets an omitted solid face its boundary
      lies in the cut plane and no water-air facet continues across it.
    * Inshore of ``periodic_x_limit``, any cut edge without an exact partner
      (a composite classifies the two sides of the cut differently). The
      join welds, samples and shades the retained mesh only at the handoff
      plane and its fade band, so such seams stay unwelded and are counted.

    No vertex is moved: an unwelded cut vertex keeps its own copy per period.
    """
    vertices = mesh.vertices
    y0 = float(np.min(vertices[:, 1]))
    y1 = float(np.max(vertices[:, 1]))
    coordinate_type = vertices.dtype
    represented_y1 = np.asarray(y0, dtype=coordinate_type) + np.asarray(
        period, dtype=coordinate_type,
    )
    if np.asarray(y1, dtype=coordinate_type) != represented_y1:
        raise ValueError("local water mesh does not span one alongshore period")
    triangles = mesh.triangles
    directed = np.concatenate((
        triangles[:, (0, 1)],
        triangles[:, (1, 2)],
        triangles[:, (2, 0)],
    ))
    owner = np.tile(np.arange(len(triangles)), 3)
    # Only edges on the two y cuts contribute to the periodic boundary.
    edge_y = vertices[directed, 1]
    on_cut = np.all(edge_y == y0, axis=1) | np.all(edge_y == y1, axis=1)
    directed, owner = directed[on_cut], owner[on_cut]
    edges, inverse = np.unique(
        np.sort(directed, axis=1), axis=0, return_inverse=True,
    )
    incidence = np.bincount(inverse, weights=np.where(
        directed[:, 0] < directed[:, 1], 1, -1,
    )).astype(np.int64)
    count = np.bincount(inverse)
    triangle_y = vertices[triangles, 1]
    in_cut = np.all(triangle_y == y0, axis=1) | np.all(triangle_y == y1, axis=1)
    open_patch = (count == 1) & (np.bincount(inverse, weights=in_cut[owner]) == 1)
    edge_x = vertices[edges, 0]
    x_ends = (
        np.all(edge_x == np.min(vertices[:, 0]), axis=1)
        | np.all(edge_x == np.max(vertices[:, 0]), axis=1)
    )
    boundary = (incidence != 0) & ~x_ends
    low_side = np.all(vertices[edges, 1] == y0, axis=1)
    low_ids = np.flatnonzero(boundary & low_side)
    high_ids = np.flatnonzero(boundary & ~low_side)

    # Order each edge by (x, z) so both cuts describe it by the same key and
    # a circulation sign that must be opposite on the two sides.
    xz = vertices[:, (0, 2)].astype(np.float64)
    a, b = edges[:, 0], edges[:, 1]
    forward = (xz[a, 0] < xz[b, 0]) | ((xz[a, 0] == xz[b, 0]) & (xz[a, 1] < xz[b, 1]))
    first = np.where(forward, a, b)
    second = np.where(forward, b, a)
    circulation = np.where(forward, incidence, -incidence)
    keys = np.concatenate((xz[first], xz[second]), axis=1)
    unique_keys, key_of = np.unique(
        np.concatenate((keys[low_ids], keys[high_ids])),
        axis=0, return_inverse=True,
    )
    low_key, high_key = key_of[:len(low_ids)], key_of[len(low_ids):]
    paired = (
        (np.bincount(low_key, minlength=len(unique_keys)) == 1)
        & (np.bincount(high_key, minlength=len(unique_keys)) == 1)
    )
    low_of_key = np.full(len(unique_keys), -1, dtype=np.int64)
    high_of_key = np.full(len(unique_keys), -1, dtype=np.int64)
    low_of_key[low_key] = low_ids
    high_of_key[high_key] = high_ids
    low_edge = low_of_key[paired]
    high_edge = high_of_key[paired]
    low_v = np.stack((first[low_edge], second[low_edge]), axis=1)
    high_v = np.stack((first[high_edge], second[high_edge]), axis=1)
    continuous = circulation[low_edge] == -circulation[high_edge]
    for name in ("normals", "velocity", "foam"):
        values = getattr(mesh, name)
        continuous &= np.all(
            (values[low_v] == values[high_v]).reshape(len(low_edge), -1), axis=1,
        )

    welded = np.zeros(len(edges), dtype=bool)
    welded[low_edge[continuous]] = True
    welded[high_edge[continuous]] = True
    geometric = np.zeros(len(edges), dtype=bool)
    geometric[low_edge] = True
    geometric[high_edge] = True
    unwelded = np.flatnonzero(boundary & ~welded & ~(open_patch & ~geometric))
    inside = unwelded[np.min(edge_x[unwelded], axis=1) <= periodic_x_limit]
    if np.any(~geometric[inside]):
        raise ValueError("local water mesh has unmatched periodic y boundaries")
    if len(inside):
        raise ValueError("local water mesh is not continuous across periodic y")

    pairs = np.unique(np.concatenate((
        np.stack((low_v[continuous, 0], high_v[continuous, 0]), axis=1),
        np.stack((low_v[continuous, 1], high_v[continuous, 1]), axis=1),
    )), axis=0)
    low, high = pairs[:, 0], pairs[:, 1]
    if len(low) == 0:
        raise ValueError("local water mesh has unmatched periodic y boundaries")
    if len(np.unique(low)) != len(low) or len(np.unique(high)) != len(high):
        raise ValueError("local water mesh is not continuous across periodic y")
    return _PeriodicBoundary(
        low, high, int(len(unwelded)),
        float(np.min(edge_x[unwelded])) if len(unwelded) else None,
    )


def _tile_periodic_y(
    mesh: WaterSurfaceMesh,
    period: float,
    boundary: _PeriodicBoundary | None = None,
) -> WaterSurfaceMesh:
    """Build a float64 welded diagnostic from exact periodic copies."""
    if boundary is None:
        boundary = _periodic_y_boundary_indices(mesh, period)
    low, high = boundary.low, boundary.high

    offsets = np.arange(
        -OFFSHORE_ALONGSHORE_REPEATS,
        OFFSHORE_ALONGSHORE_REPEATS + 1,
    )
    vertex_count = len(mesh.vertices)
    source_vertices = mesh.vertices.astype(np.float64, copy=False)
    vertices = np.concatenate(tuple(
        source_vertices + np.asarray((0.0, offset * period, 0.0))
        for offset in offsets
    ))
    triangles = np.concatenate(tuple(
        mesh.triangles + copy * vertex_count
        for copy in range(len(offsets))
    ))
    normals = np.tile(mesh.normals, (len(offsets), 1))
    velocity = np.tile(mesh.velocity, (len(offsets), 1))
    foam = np.tile(mesh.foam, len(offsets))
    boundary_labels = np.tile(mesh.vertex_open_boundaries, len(offsets))
    open_edges = np.concatenate(tuple(
        mesh.open_boundary_edges + copy * vertex_count
        for copy in range(len(offsets))
    )) if len(mesh.open_boundary_edges) else mesh.open_boundary_edges.copy()
    open_labels = np.tile(mesh.open_boundary_labels, len(offsets))

    vertex_map = np.arange(len(vertices), dtype=np.int64)
    for copy in range(len(offsets) - 1):
        vertex_map[(copy + 1) * vertex_count + low] = (
            copy * vertex_count + high
        )
    triangles = vertex_map[triangles]
    if len(open_edges):
        open_edges = vertex_map[open_edges]
    used = np.unique(triangles)
    compact = np.empty(len(vertices), dtype=np.int64)
    compact[used] = np.arange(len(used))
    triangles = compact[triangles].astype(np.int32)
    if len(open_edges):
        open_edges = compact[open_edges].astype(np.int32)
    return WaterSurfaceMesh(
        vertices=vertices[used],
        triangles=triangles,
        normals=normals[used],
        velocity=velocity[used],
        foam=foam[used],
        vertex_open_boundaries=boundary_labels[used],
        open_boundary_edges=open_edges,
        open_boundary_labels=open_labels,
    )


def _fundamental_period_mesh(
    mesh: WaterSurfaceMesh,
    y0: float,
    period: float,
    periodic_x_limit: float = np.inf,
) -> WaterSurfaceMesh:
    """Select the local-period mesh used by translated render instances."""
    y1 = y0 + period
    triangle_y = mesh.vertices[mesh.triangles, 1]
    selected = np.all(
        (triangle_y >= y0) & (triangle_y <= y1),
        axis=1,
    ) & ~np.all(triangle_y == y1, axis=1)
    # A face lying on the cut belongs to the lower end of exactly one period.
    # Non-coplanar triangles still retain their upper-boundary vertices.
    triangles = mesh.triangles[selected]
    used = np.unique(triangles)
    compact = np.empty(len(mesh.vertices), dtype=np.int64)
    compact[used] = np.arange(len(used))
    result = WaterSurfaceMesh(
        vertices=mesh.vertices[used],
        triangles=compact[triangles].astype(np.int32),
        normals=mesh.normals[used],
        velocity=mesh.velocity[used],
        foam=mesh.foam[used],
        vertex_open_boundaries=mesh.vertex_open_boundaries[used],
        open_boundary_edges=np.empty((0, 2), dtype=np.int32),
        open_boundary_labels=np.empty(0, dtype=np.uint8),
    )
    _periodic_y_boundary_indices(result, period, periodic_x_limit)
    return result


def clip_local_mesh(
    mesh: WaterSurfaceMesh,
    handoff_x: float,
) -> WaterSurfaceMesh:
    """Restrict water-air facets to the open local half-space x > handoff.

    Intersection edges are retained, but a facet entirely on the new open
    boundary is not a water-air cap. This is the same open-domain convention
    used by the governing PLIC exporter; no off-plane facet is selected away.
    """
    # Intersections are stored in FP32. Use that represented plane throughout
    # the FP64 edge interpolation, then round the resulting point only once.
    handoff_x = float(np.float32(handoff_x))
    vertices: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    velocity: list[np.ndarray] = []
    foam: list[float] = []
    boundary: list[int] = []
    triangles: list[tuple[int, int, int]] = []
    originals: dict[int, int] = {}
    crossings: dict[tuple[int, int], int] = {}

    def append_vertex(index: int, other: int | None = None) -> int:
        if other is None:
            if index not in originals:
                originals[index] = len(vertices)
                vertices.append(mesh.vertices[index])
                normals.append(mesh.normals[index])
                velocity.append(mesh.velocity[index])
                foam.append(float(mesh.foam[index]))
                boundary.append(int(mesh.vertex_open_boundaries[index]))
            return originals[index]

        if mesh.vertices[index, 0] == handoff_x:
            return append_vertex(index)
        if mesh.vertices[other, 0] == handoff_x:
            return append_vertex(other)
        key = (min(index, other), max(index, other))
        if key not in crossings:
            # Periodic copies traverse their matching boundary edges in
            # opposite directions. A fixed x ordering also makes their
            # interpolated coordinates and attributes bitwise identical.
            if mesh.vertices[index, 0] > mesh.vertices[other, 0]:
                index, other = other, index
            first = mesh.vertices[index].astype(np.float64)
            second = mesh.vertices[other].astype(np.float64)
            fraction = (
                (handoff_x - first[0]) / (second[0] - first[0])
            )
            crossings[key] = len(vertices)
            point = first + fraction * (second - first)
            point[0] = handoff_x
            vertices.append(point)
            normals.append(
                mesh.normals[index].astype(np.float64)
                + fraction * (mesh.normals[other].astype(np.float64) - mesh.normals[index])
            )
            velocity.append(
                mesh.velocity[index].astype(np.float64)
                + fraction * (mesh.velocity[other].astype(np.float64) - mesh.velocity[index])
            )
            foam.append(float(
                float(mesh.foam[index])
                + fraction * (float(mesh.foam[other]) - float(mesh.foam[index]))
            ))
            boundary.append(0)
        return crossings[key]

    for triangle in mesh.triangles:
        if np.all(mesh.vertices[triangle, 0] == handoff_x):
            continue
        polygon: list[int] = []
        for first, second in zip(triangle, np.roll(triangle, -1), strict=True):
            first = int(first)
            second = int(second)
            first_inside = mesh.vertices[first, 0] >= handoff_x
            second_inside = mesh.vertices[second, 0] >= handoff_x
            if first_inside:
                index = append_vertex(first)
                if not polygon or polygon[-1] != index:
                    polygon.append(index)
            if first_inside != second_inside:
                index = append_vertex(first, second)
                if not polygon or polygon[-1] != index:
                    polygon.append(index)
        if len(polygon) > 1 and polygon[0] == polygon[-1]:
            polygon.pop()
        for index in range(1, len(polygon) - 1):
            triangles.append((polygon[0], polygon[index], polygon[index + 1]))

    normal_array = np.asarray(normals, dtype=np.float64)
    normal_array /= np.linalg.norm(normal_array, axis=1, keepdims=True)
    return WaterSurfaceMesh(
        vertices=np.asarray(vertices, dtype=np.float32),
        triangles=np.asarray(triangles, dtype=np.int32),
        normals=normal_array.astype(np.float32),
        velocity=np.asarray(velocity, dtype=np.float32),
        foam=np.asarray(foam, dtype=np.float32),
        vertex_open_boundaries=np.asarray(boundary, dtype=np.uint8),
        open_boundary_edges=np.empty((0, 2), dtype=np.int32),
        open_boundary_labels=np.empty(0, dtype=np.uint8),
    )


def exact_offshore_field(inlet: LinearWaveInlet) -> OffshoreWaveField:
    """Convert the finite-depth inlet without changing its dispersion."""
    harmonics = []
    kx = []
    ky = []
    amplitudes = []
    phases = []
    for component in inlet.components:
        harmonics.append(component.temporal_mode)
        kx.append(component.kx)
        ky.append(component.ky)
        amplitudes.append(component.amplitude)
        phases.append(component.phase_offset)
        second = component.second_order_elevation_amplitude
        if second != 0.0:
            harmonics.append(2 * component.temporal_mode)
            kx.append(2.0 * component.kx)
            ky.append(2.0 * component.ky)
            amplitudes.append(second)
            phases.append(2.0 * component.phase_offset)
    return OffshoreWaveField(
        period=inlet.period,
        harmonics=tuple(harmonics),
        kx=tuple(kx),
        ky=tuple(ky),
        amplitudes=tuple(amplitudes),
        phases=tuple(phases),
    )


def _inlet_surface_gradient(
    inlet: LinearWaveInlet,
    x: np.ndarray,
    y: np.ndarray,
    time: float,
    eta0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height = np.full(np.broadcast_shapes(x.shape, y.shape), eta0)
    slope_x = np.zeros_like(height)
    slope_y = np.zeros_like(height)
    slope_xy = np.zeros_like(height)
    canonical_y = np.remainder(
        y,
        inlet.components[0].alongshore_length,
    )
    for component in inlet.components:
        phase = (
            component.kx * x
            + component.ky * canonical_y
            - component.omega * np.remainder(time, component.period)
            + component.phase_offset
        )
        height += component.amplitude * np.cos(phase)
        slope_x -= component.kx * component.amplitude * np.sin(phase)
        slope_y -= component.ky * component.amplitude * np.sin(phase)
        second = component.second_order_elevation_amplitude
        height += second * np.cos(2.0 * phase)
        slope_x -= 2.0 * component.kx * second * np.sin(2.0 * phase)
        slope_y -= 2.0 * component.ky * second * np.sin(2.0 * phase)
        slope_xy -= component.kx * component.ky * (
            component.amplitude * np.cos(phase) + 4.0 * second * np.cos(2.0 * phase)
        )
    return height, slope_x, slope_y, slope_xy


def _inlet_surface(inlet, x, y, time, eta0):
    height, slope_x, _, _ = _inlet_surface_gradient(inlet, x, y, time, eta0)
    return height, slope_x


def _seam_edges(mesh: WaterSurfaceMesh, handoff_x: float) -> np.ndarray:
    # Only incidence-one edges bound the retained open half-space. Internal
    # edges of vertical facets on other planes are not additional contours.
    triangle_edges = np.concatenate(tuple(
        mesh.triangles[:, pair] for pair in ((0, 1), (1, 2), (2, 0))
    ))
    edges, counts = np.unique(
        np.sort(triangle_edges, axis=1), axis=0, return_counts=True,
    )
    seam_edges = edges[(counts == 1) & np.all(
        mesh.vertices[edges, 0] == handoff_x, axis=1,
    )]
    if len(seam_edges) == 0:
        raise ValueError("offshore handoff plane does not intersect the mesh")
    return seam_edges


def _seam_path(seam_edges: np.ndarray, mesh: WaterSurfaceMesh) -> np.ndarray | None:
    """Order a simple seam path from its low-y end, or None if it branches."""
    boundary = np.unique(seam_edges)
    adjacent = {int(index): [] for index in boundary}
    for first, second in seam_edges:
        adjacent[first].append(int(second))
        adjacent[second].append(int(first))
    ends = [index for index in boundary if len(adjacent[index]) == 1]
    if len(ends) != 2 or any(len(neighbours) > 2 for neighbours in adjacent.values()):
        return None
    path = [int(min(ends, key=lambda index: mesh.vertices[index, 1]))]
    previous = -1
    while True:
        following = [index for index in adjacent[path[-1]] if index != previous]
        if not following:
            break
        previous = path[-1]
        path.append(following[0])
    if len(path) != len(adjacent):
        return None
    return np.asarray(path, dtype=np.int64)


def _compact_mesh(mesh: WaterSurfaceMesh, triangles: np.ndarray,
                  open_edges: np.ndarray, open_labels: np.ndarray) -> WaterSurfaceMesh:
    used = np.unique(triangles)
    compact = np.empty(len(mesh.vertices), dtype=np.int64)
    compact[used] = np.arange(len(used))
    return WaterSurfaceMesh(
        vertices=mesh.vertices[used],
        triangles=compact[triangles].astype(np.int32),
        normals=mesh.normals[used],
        velocity=mesh.velocity[used],
        foam=mesh.foam[used],
        vertex_open_boundaries=mesh.vertex_open_boundaries[used],
        open_boundary_edges=(compact[open_edges].astype(np.int32)
                             if len(open_edges) else open_edges),
        open_boundary_labels=open_labels,
    )


def _remove_seam_slivers(
    mesh: WaterSurfaceMesh,
    handoff_x: float,
    height_tolerance: float,
) -> tuple[WaterSurfaceMesh, int]:
    """Drop closed seam contours thinner than the tolerance and hugging the surface.

    Marching cubes leaves zero-thickness slivers exactly on cell-face planes
    where nodal values sit on its level. Cut by the handoff plane they form a
    closed contour a fraction of a millimetre thick within a millimetre of the
    water surface. Such a contour is not a resolved bubble or droplet; the
    triangles touching it are removed. Contours with real vertical extent or
    offset stay and are refused downstream.
    """
    edges = _seam_edges(mesh, handoff_x)
    boundary = np.unique(edges)
    parent = {int(i): int(i) for i in boundary}

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    degree = {int(i): 0 for i in boundary}
    for first, second in edges:
        parent[root(int(first))] = root(int(second))
        degree[int(first)] += 1
        degree[int(second)] += 1
    components: dict[int, list[int]] = {}
    for index in boundary:
        components.setdefault(root(int(index)), []).append(int(index))
    if len(components) < 2:
        return mesh, 0
    main_key = max(components, key=lambda key: len(components[key]))
    main = np.asarray(components[main_key])
    order = np.argsort(mesh.vertices[main, 1])
    main_y = mesh.vertices[main, 1].astype(np.float64)[order]
    main_z = mesh.vertices[main, 2].astype(np.float64)[order]
    remove = np.zeros(len(mesh.vertices), dtype=bool)
    removed = 0
    for key, members in components.items():
        if key == main_key:
            continue
        members = np.asarray(members)
        if any(degree[int(i)] != 2 for i in members):
            continue
        z = mesh.vertices[members, 2].astype(np.float64)
        surface = np.interp(mesh.vertices[members, 1].astype(np.float64), main_y, main_z)
        if (z.max() - z.min() <= height_tolerance
                and np.all(np.abs(z - surface) <= height_tolerance)):
            remove[members] = True
            removed += 1
    if removed == 0:
        return mesh, 0
    keep = ~np.any(remove[mesh.triangles], axis=1)
    open_edges = mesh.open_boundary_edges
    open_labels = mesh.open_boundary_labels
    if len(open_edges):
        keep_edges = ~np.any(remove[open_edges], axis=1)
        open_edges = open_edges[keep_edges]
        open_labels = open_labels[keep_edges]
    return _compact_mesh(mesh, mesh.triangles[keep], open_edges, open_labels), removed


def _weld_subcell_seam_folds(
    mesh: WaterSurfaceMesh,
    handoff_x: float,
    back_tolerance: float,
    height_tolerance: float,
) -> tuple[WaterSurfaceMesh, int, float]:
    """Collapse sub-cell folds of the seam path onto the running-maximum vertex.

    A marching-cubes surface is single-valued only at grid resolution. Where
    a trough bottoms exactly on a cell face, its cut along the handoff plane
    can double back by a few millimetres within an almost flat patch. Such a
    fold is below the resolved geometry, so its vertices are merged into the
    vertex they fold back from and the degenerate triangles are dropped. Any
    fold wider than ``back_tolerance`` in y or taller than
    ``height_tolerance`` is left for :func:`_ordered_seam` to refuse.
    """
    path = _seam_path(_seam_edges(mesh, handoff_x), mesh)
    if path is None:
        return mesh, 0, 0.0
    y = mesh.vertices[path, 1].astype(np.float64)
    height = mesh.vertices[path, 2].astype(np.float64)
    remap = np.arange(len(mesh.vertices), dtype=np.int64)
    running = 0
    shifts = []
    for position in range(1, len(path)):
        if y[position] > y[running]:
            running = position
            continue
        back = y[running] - y[position]
        rise = abs(height[position] - height[running])
        if 0.0 < back <= back_tolerance and rise <= height_tolerance:
            remap[path[position]] = path[running]
            shifts.append(float(np.hypot(back, rise)))
    if not shifts:
        return mesh, 0, 0.0
    triangles = remap[mesh.triangles]
    keep = (
        (triangles[:, 0] != triangles[:, 1])
        & (triangles[:, 1] != triangles[:, 2])
        & (triangles[:, 2] != triangles[:, 0])
    )
    triangles = triangles[keep]
    open_edges = mesh.open_boundary_edges
    open_labels = mesh.open_boundary_labels
    if len(open_edges):
        open_edges = remap[open_edges]
        keep_edges = open_edges[:, 0] != open_edges[:, 1]
        open_edges = open_edges[keep_edges]
        open_labels = open_labels[keep_edges]
    return _compact_mesh(mesh, triangles, open_edges, open_labels), len(shifts), max(shifts)


def _ordered_seam(
    mesh: WaterSurfaceMesh,
    handoff_x: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seam_edges = _seam_edges(mesh, handoff_x)
    boundary = np.unique(seam_edges)
    unique_y = np.unique(mesh.vertices[boundary, 1].astype(np.float64))
    segment_y = mesh.vertices[seam_edges, 1].astype(np.float64)
    segment_low = np.min(segment_y, axis=1)
    segment_high = np.max(segment_y, axis=1)
    for value in .5 * (unique_y[:-1] + unique_y[1:]):
        active = np.count_nonzero((segment_low < value) & (segment_high > value))
        if active == 0:
            raise MultiValuedOffshoreSeamError(
                "offshore handoff intersection is not a continuous water graph",
            )
        if active > 1:
            raise MultiValuedOffshoreSeamError(
                "offshore handoff plane intersects a multi-valued water surface",
            )

    adjacent = {int(index): [] for index in boundary}
    for first, second in seam_edges:
        adjacent[first].append(int(second))
        adjacent[second].append(int(first))
    ends = [index for index in boundary if len(adjacent[index]) == 1]
    if len(ends) != 2 or any(len(neighbours) > 2 for neighbours in adjacent.values()):
        raise MultiValuedOffshoreSeamError(
            "offshore handoff plane intersects a multi-valued water surface",
        )
    path = [int(min(ends, key=lambda index: mesh.vertices[index, 1]))]
    previous = -1
    while True:
        following = [index for index in adjacent[path[-1]] if index != previous]
        if not following:
            break
        previous = path[-1]
        path.append(following[0])
    boundary = np.asarray(path, dtype=np.int64)
    y, height = mesh.vertices[boundary, 1:].astype(np.float64).T
    if (
        len(path) != len(adjacent)
        or y[0] != np.min(mesh.vertices[:, 1])
        or y[-1] != np.max(mesh.vertices[:, 1])
    ):
        raise MultiValuedOffshoreSeamError(
            "offshore handoff intersection is not a continuous water graph",
        )
    if np.any(np.diff(y) < 0):
        raise MultiValuedOffshoreSeamError(
            "offshore handoff plane intersects a multi-valued water surface",
        )
    for value in unique_y:
        vertical_height = height[y == value]
        if len(vertical_height) > 1:
            differences = np.diff(vertical_height)
            if not (np.all(differences > 0) or np.all(differences < 0)):
                raise MultiValuedOffshoreSeamError(
                    "offshore handoff plane intersects a multi-valued water surface",
                )

    # Shading normals from the uncut surface include discarded facets. The
    # x tangent must instead lie in the retained incident surface's plane.
    points = mesh.vertices[mesh.triangles].astype(np.float64)
    area_vectors = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    normals = np.zeros((len(mesh.vertices), 3))
    for corner in range(3):
        np.add.at(normals, mesh.triangles[:, corner], area_vectors)
    normal = normals[boundary]
    determined = normal[:, 2] > 0
    if np.any(normal[:, 2] < 0) or np.any((normal[:, 2] == 0) & (normal[:, 0] != 0)):
        raise MultiValuedOffshoreSeamError(
            "offshore handoff plane intersects a multi-valued water surface",
        )
    # Pure alongshore facets contain every x-z tangent. Interpolate that free
    # jet by boundary arc length between the adjacent determined endpoints.
    arc = np.r_[0.0, np.cumsum(np.hypot(np.diff(y), np.diff(height)))]
    if not np.any(determined):
        raise MultiValuedOffshoreSeamError(
            "offshore handoff plane intersects a multi-valued water surface",
        )
    slope = np.interp(arc, arc[determined], -normal[determined, 0] / normal[determined, 2])
    return boundary, y, height, slope


def _seam_profile(
    mesh: WaterSurfaceMesh,
    handoff_x: float,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    _, boundary_y, height, slope = _ordered_seam(mesh, handoff_x)
    return (
        np.interp(y, boundary_y, height),
        np.interp(y, boundary_y, slope),
    )


def _collar_surface(
    inlet: LinearWaveInlet,
    x: np.ndarray,
    y: np.ndarray,
    time: float,
    eta0: float,
    handoff_x: float,
    collar_width: float,
    seam_height: np.ndarray,
    seam_slope: np.ndarray,
    tangent_width: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    outer_x = handoff_x - collar_width
    analytic, analytic_slope = _inlet_surface(inlet, x, y, time, eta0)
    analytic_at_seam, analytic_slope_at_seam = _inlet_surface(
        inlet,
        np.full_like(y, handoff_x),
        y,
        time,
        eta0,
    )
    s = np.clip((x - outer_x) / collar_width, 0.0, 1.0)
    height_delta = seam_height - analytic_at_seam
    slope_delta = seam_slope - analytic_slope_at_seam
    tangent, tangent_x = _tangent_extension(x, handoff_x, tangent_width)
    correction = s**3*(10.0+s*(-15.0+6.0*s))*height_delta + tangent*slope_delta
    correction_slope = (
        30.0*s**2*(1.0-s)**2/collar_width*height_delta + tangent_x*slope_delta
    )
    collar = analytic + correction
    collar_slope = analytic_slope + correction_slope
    surface = np.where(x < outer_x, analytic, collar)
    surface_slope = np.where(x < outer_x, analytic_slope, collar_slope)
    return surface, surface_slope, analytic


@dataclass(frozen=True)
class _PeriodicSeam:
    """An ordered boundary and its arc coordinate in one complete period."""

    indices: np.ndarray
    y: np.ndarray
    height: np.ndarray
    slope: np.ndarray
    parameter: np.ndarray
    canonical: np.ndarray
    period: float

    def interpolate(self, values, samples, jump=0.0):
        q = self.parameter[self.canonical]
        cycles = np.floor((samples-q[0])/self.period)
        return np.interp(
            samples-cycles*self.period,
            np.r_[q, q[0]+self.period], np.r_[values, values[0]+jump],
        ) + cycles*jump

    def derivative(self, values, jump=0.0):
        q = self.parameter[self.canonical]
        before = q-np.r_[q[-1]-self.period, q[:-1]]
        after = np.r_[q[1:], q[0]+self.period]-q
        left = values-np.r_[values[-1]-jump, values[:-1]]
        right = np.r_[values[1:], values[0]+jump]-values
        # Chord tangent of the piecewise-linear boundary. Weight its incident
        # segments by length: a vanishing PLIC step must not rotate a broad
        # face's reflection normal by ninety degrees. The nonuniform
        # quadratic-derivative formula instead gives the shortest edge the
        # greatest weight and has no continuous small-step limit here.
        return (left+right)/(before+after)


def _periodic_seam(
    mesh: WaterSurfaceMesh, inlet: LinearWaveInlet, handoff_x: float,
) -> _PeriodicSeam:
    indices, y, height, slope = _ordered_seam(mesh, handoff_x)
    period = inlet.components[0].alongshore_length
    periods = round((y[-1]-y[0])/period)
    # Use the middle copy's complete retained neighbourhood when tiled. For
    # a single period, the path already supplies both ends of its closure.
    start_y = y[0] + (period if periods >= 3 else 0.0)
    start = int(np.flatnonzero((y == start_y) & (height == height[0]))[0])
    end = int(np.flatnonzero((y == start_y+period) & (height == height[0]))[0])
    canonical = np.arange(start, end)
    arc = np.r_[0.0, np.cumsum(np.hypot(
        np.diff(y[start:end+1]), np.diff(height[start:end+1]),
    ))]
    q = start_y + period*arc/arc[-1]
    keys = {
        (float(np.remainder(y[i]-start_y, period)), float(height[i])): j
        for j, i in enumerate(canonical)
    }
    mapping = np.asarray([
        keys[(float(np.remainder(yy-start_y, period)), float(hh))]
        for yy, hh in zip(y, height, strict=True)
    ])
    parameter = y + (q[:-1]-y[canonical])[mapping]
    return _PeriodicSeam(
        indices, y, height, slope[canonical][mapping], parameter, canonical, period,
    )


def _tangent_extension(x, handoff_x, width):
    """Compact Hermite jet: zero value at both ends, unit x derivative at the seam.

    Its maximum absolute displacement per unit slope is 16*width/81.
    Restricting the support to a mesh-scale strip prevents a discrete PLIC
    tangent jump from becoming a groove across the entire height collar.
    """
    t = np.clip((x-(handoff_x-width))/width, 0.0, 1.0)
    return width*t**3*(-4.0+t*(7.0-3.0*t)), t**2*(-12.0+t*(28.0-15.0*t))


def _parametric_collar(seam, inlet, x, samples, time, eta0, handoff_x, collar_width,
                      tangent_width, *, sampled_normals=False):
    """Evaluate the quintic surface, averaged shading normal and x derivative.

    Y=(1-S)q+S*y_seam(q), S=10s^3-15s^4+6s^5. Arc q is strictly
    increasing even on a vertical PLIC step. Thus Y is strictly increasing
    inside the collar, without moving the local boundary or clipping jets.
    Both Y_x and the height correction vanish at the analytic outer edge;
    Y_x also vanishes at the seam, preserving its retained x tangent.
    Along q, chord tangents provide the vertex shading interpolation for
    the piecewise-linear boundary; they are not a higher-order curve fit.
    Height matching spans the full collar. Tangent matching is confined to
    ``tangent_width`` next to the local mesh, with the same exact end jet.
    On the regular interior mesh, shading uses chords of the sampled surface.
    A subcell seam derivative must not tilt a broad rendered face sideways.
    The caller supplies ghost samples for central chords at the periodic ends.
    """
    canonical_y = seam.y[seam.canonical]
    seam_y = seam.interpolate(canonical_y, samples, seam.period)
    seam_y_q = seam.interpolate(seam.derivative(canonical_y, seam.period), samples)
    analytic_seam, analytic_seam_x, analytic_seam_y, analytic_seam_xy = _inlet_surface_gradient(
        inlet, np.full_like(seam_y, handoff_x), seam_y, time, eta0,
    )
    delta = seam.interpolate(seam.height[seam.canonical], samples)-analytic_seam
    jet = seam.interpolate(seam.slope[seam.canonical], samples)-analytic_seam_x
    delta_q = (
        seam.interpolate(seam.derivative(seam.height[seam.canonical]), samples)
        - analytic_seam_y*seam_y_q
    )
    jet_q = (
        seam.interpolate(seam.derivative(seam.slope[seam.canonical]), samples)
        - analytic_seam_xy*seam_y_q
    )
    s = np.clip((x-(handoff_x-collar_width))/collar_width, 0.0, 1.0)
    blend = s**3*(10.0+s*(-15.0+6.0*s))
    blend_x = 30.0*s**2*(1.0-s)**2/collar_width
    tangent, tangent_x = _tangent_extension(x, handoff_x, tangent_width)
    y = samples+blend*(seam_y-samples)
    y_x = blend_x*(seam_y-samples)
    y_q = 1.0+blend*(seam_y_q-1.0)
    analytic, analytic_x, analytic_y, _ = _inlet_surface_gradient(
        inlet, x, y, time, eta0,
    )
    height = (
        analytic+blend*delta+tangent*jet
    )
    height_x = (
        analytic_x+analytic_y*y_x+blend_x*delta+tangent_x*jet
    )
    height_q = (
        analytic_y*y_q+blend*delta_q+tangent*jet_q
    )
    if sampled_normals:
        y_q = np.gradient(y, samples.ravel(), axis=-1)
        height_q = np.gradient(height, samples.ravel(), axis=-1)
    normals = np.stack((y_x*height_q-height_x*y_q, -height_q, y_q), axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    positions = np.stack((np.broadcast_to(x, height.shape), y, height), axis=-1)
    return positions, normals, height_x


def _attach_offshore_mesh(
    mesh: WaterSurfaceMesh,
    inlet: LinearWaveInlet,
    time: float,
    eta0: float,
    handoff_x: float,
    collar_width: float,
    offshore_length: float,
    spacing: float,
    *,
    tangent_width: float | None = None,
) -> WaterSurfaceMesh:
    """Sew a fixed-resolution parametric collar onto the exact PLIC edge."""
    seam = _periodic_seam(mesh, inlet, handoff_x)
    periods = round((seam.y[-1]-seam.y[0])/seam.period)
    rows_per_period = max(2, int(np.ceil(seam.period/spacing)))
    samples = np.linspace(seam.y[0], seam.y[-1], periods*rows_per_period+1)
    outer = handoff_x-collar_width
    if tangent_width is None:
        tangent_width = min(collar_width, spacing)
    if not 0.0 < tangent_width <= collar_width:
        raise ValueError("tangent_width must be positive and no larger than collar_width")
    tangent_start = handoff_x-tangent_width
    x = np.r_[
        np.linspace(outer-offshore_length, outer,
                    max(1, int(np.ceil(offshore_length/spacing)))+1)[:-1],
        np.linspace(outer, tangent_start,
                    int(np.ceil((collar_width-tangent_width)/spacing))+1)[:-1],
        # Resolve the interior of the compact slope correction.
        np.linspace(tangent_start, handoff_x,
                    max(4, int(np.ceil(tangent_width/spacing)))+1)[:-1],
    ]
    sample_spacing = seam.period/rows_per_period
    padded_samples = np.r_[samples[0]-sample_spacing, samples,
                           samples[-1]+sample_spacing]
    positions, normals, _ = _parametric_collar(
        seam, inlet, x[:, None], padded_samples[None, :],
        time, eta0, handoff_x, collar_width, tangent_width,
        sampled_normals=True,
    )
    positions, normals = positions[:, 1:-1], normals[:, 1:-1]
    # Reuse one complete periodic neighbourhood after chord differencing;
    # translated coordinates can otherwise leave different roundoff in normals.
    first = (periods//2)*rows_per_period
    normal_period = normals[:, first:first+rows_per_period]
    normals = np.concatenate((np.tile(normal_period, (1, periods, 1)),
                              normal_period[:, :1]), axis=1)
    positions[:, :, (0, 2)] = positions[:, :, (0, 2)].astype(np.float32)
    # Store one FP32 y period and translate copies in FP64, as for the local
    # mesh. This retains exactly matching coordinates on rendered instances.
    cycles = np.floor(positions[:, :, 1]/seam.period)
    positions[:, :, 1] = (
        (positions[:, :, 1]-cycles*seam.period).astype(np.float32).astype(float)
        + cycles*seam.period
    )
    if np.any(np.diff(positions[:, :, 1], axis=1) <= 0.0):
        raise ValueError("parametric collar rows collapse at render precision")

    vertices = np.concatenate((mesh.vertices, positions.reshape(-1, 3)))
    retained_normals = mesh.normals.copy()
    retained_normals[seam.indices] = _parametric_collar(
        seam, inlet, np.full_like(seam.parameter, handoff_x), seam.parameter,
        time, eta0, handoff_x, collar_width, tangent_width,
    )[1]
    normals = np.concatenate((retained_normals, normals.reshape(-1, 3))).astype(np.float32)
    indices = len(mesh.vertices)+np.arange(len(x)*len(samples)).reshape(len(x), len(samples))
    triangles = []
    for left, right in zip(indices[:-1], indices[1:], strict=True):
        for j in range(len(samples)-1):
            triangles.extend(((left[j], right[j], right[j+1]),
                              (left[j], right[j+1], left[j+1])))
    # Both chains have monotonically increasing projected y. A zipper in arc
    # order triangulates the strip without intersections, retaining every
    # original vertical edge. No repeated rows continue into the collar.
    left, right = indices[-1], seam.indices
    i = j = 0
    while i < len(left)-1 or j < len(right)-1:
        if j == len(right)-1 or (i < len(left)-1 and samples[i+1] < seam.parameter[j+1]):
            triangles.append((left[i], right[j], left[i+1]))
            i += 1
        else:
            triangles.append((left[i], right[j], right[j+1]))
            j += 1
    triangles = np.asarray(triangles, dtype=np.int32)
    points = vertices[triangles]
    area = np.cross(points[:, 1]-points[:, 0], points[:, 2]-points[:, 0])
    if np.any(~np.any(area != 0.0, axis=1)) or np.any(area[:, 2] < 0.0):
        raise ValueError("parametric collar has a collapsed or reversed triangle")
    extra = len(vertices)-len(mesh.vertices)
    return WaterSurfaceMesh(
        vertices=vertices,
        triangles=np.concatenate((mesh.triangles, triangles)),
        normals=normals,
        velocity=np.concatenate((mesh.velocity, np.zeros((extra, 3), dtype=np.float32))),
        foam=np.r_[mesh.foam, np.zeros(extra, dtype=np.float32)],
        vertex_open_boundaries=np.r_[mesh.vertex_open_boundaries, np.zeros(extra, dtype=np.uint8)],
        open_boundary_edges=np.empty((0, 2), dtype=np.int32),
        open_boundary_labels=np.empty(0, dtype=np.uint8),
    )


def render_offshore_view(
    mesh: WaterSurfaceMesh,
    inlet: LinearWaveInlet,
    time: float,
    camera: Camera,
    *,
    eta0: float,
    device: str,
    handoff_x: float | None = None,
    collar_width: float = 0.4,
    offshore_mesh_length: float = 4.0,
    cell_size: float = 0.02,
    tangent_width: float | None = None,
    render_scale: int = 4,
    sky: OvercastSky | None = None,
    sun: Sun | None = None,
    optics: WaterOptics | None = None,
    spectrum: RippleSpectrum | None = None,
    ripples_enabled: bool = True,
) -> OffshoreView:
    """Render the stitched outer water for one camera and loop phase."""
    if tangent_width is None:
        tangent_width = min(collar_width, cell_size)
    periodic_mesh = (
        mesh if handoff_x is None else clip_local_mesh(mesh, handoff_x)
    )
    period_y = inlet.components[0].alongshore_length
    periodic_y0 = float(np.min(periodic_mesh.vertices[:, 1]))
    # The join welds, samples and shades the retained mesh at the handoff
    # plane and across its fade band; the y cut must be exactly periodic there.
    periodic_x_limit = float(np.min(periodic_mesh.vertices[:, 0])) + collar_width
    seam_boundary = _periodic_y_boundary_indices(
        periodic_mesh, period_y, periodic_x_limit,
    )
    clipped_mesh = _tile_periodic_y(periodic_mesh, period_y, seam_boundary)
    handoff_x = float(np.min(clipped_mesh.vertices[:, 0]))
    clipped_mesh, removed_seam_slivers = _remove_seam_slivers(
        clipped_mesh, handoff_x, 0.25 * cell_size,
    )
    clipped_mesh, welded_seam_folds, maximum_weld_shift = _weld_subcell_seam_folds(
        clipped_mesh, handoff_x, cell_size, 0.25 * cell_size,
    )
    local_mesh = _attach_offshore_mesh(
        clipped_mesh,
        inlet,
        time,
        eta0,
        handoff_x,
        collar_width,
        offshore_mesh_length,
        cell_size,
        tangent_width=tangent_width,
    )
    periodic_instance_mesh = _fundamental_period_mesh(
        local_mesh,
        periodic_y0,
        period_y,
        periodic_x_limit,
    )
    render_edge_x = float(np.min(local_mesh.vertices[:, 0]))
    y0 = float(np.min(clipped_mesh.vertices[:, 1]))
    y1 = float(np.max(clipped_mesh.vertices[:, 1]))
    collar_outer_x = handoff_x - collar_width
    x1 = max(float(np.max(local_mesh.vertices[:, 0])), camera.pos[0]) + 1.0
    nx = int(np.ceil((x1 - render_edge_x) / cell_size))
    ny = int(np.ceil((y1 - y0) / cell_size))
    grid = GridConfig(
        nx=nx,
        ny=ny,
        Lx=nx * cell_size,
        Ly=y1 - y0,
        x0=render_edge_x,
        y0=y0,
    )
    x = grid.x0 + (np.arange(grid.nx) + 0.5) * grid.dx
    y = grid.y0 + (np.arange(grid.ny) + 0.5) * grid.dy
    X, Y = np.meshgrid(x, y, indexing="ij")
    seam_height, seam_slope = _seam_profile(clipped_mesh, handoff_x, y)
    surface, _, _ = _collar_surface(
        inlet,
        X,
        Y,
        time,
        eta0,
        handoff_x,
        collar_width,
        seam_height[None, :],
        seam_slope[None, :],
        tangent_width,
    )

    bed_level = eta0 - inlet.depth
    wet = X < handoff_x
    bed = np.where(wet, bed_level, bed_level - 5.0).astype(np.float32)
    depth = np.where(wet, surface - bed_level, 0.0).astype(np.float32)
    config = SolverConfig(
        grid=grid,
        physics=PhysicsConfig(friction_enabled=False),
        bc=BCConfig(BC.WALL, BC.WALL, BC.PERIODIC, BC.PERIODIC),
        device=device,
    )
    solver = SWESolver(config)
    solver.set_bathymetry(bed)
    solver.set_water(depth)

    if spectrum is None:
        spectrum = crossing_ripple_spectrum(inlet.period)
    clear_optics = optics or WaterOptics(scatter=(0.0, 0.0, 0.0))
    flat_bed = PebbleTexture(
        style="tiles",
        lum_var=0.0,
        hue_var=0.0,
        crevice=0.0,
        dark_frac=0.0,
        pale_frac=0.0,
    )
    renderer = Raytracer(
        solver,
        spectrum=spectrum,
        scale=render_scale,
        camera=camera,
        sky=sky or OvercastSky(),
        sun=sun or Sun(),
        optics=clear_optics,
        texture=flat_bed,
        eta0=eta0,
        offshore_field=exact_offshore_field(inlet),
        ripple_fade=(collar_outer_x, handoff_x),
        ripples_enabled=ripples_enabled,
        ripple_alongshore_period=(
            inlet.components[0].alongshore_length
        ),
    )
    image = renderer.render(time)

    # Measure the actual parametric collar at every retained seam vertex,
    # including vertical steps, rather than a resampled height-graph surrogate.
    seam = _periodic_seam(clipped_mesh, inlet, handoff_x)
    collar_at_seam, _, collar_slope_at_seam = _parametric_collar(
        seam, inlet, np.full_like(seam.parameter, handoff_x), seam.parameter,
        time, eta0, handoff_x, collar_width, tangent_width,
    )
    analytic_at_seam, analytic_slope_at_seam = _inlet_surface(
        inlet, np.full_like(seam.y, handoff_x), seam.y, time, eta0,
    )
    collar_at_outer, _, collar_slope_at_outer = _parametric_collar(
        seam, inlet, np.full_like(seam.parameter, collar_outer_x), seam.parameter,
        time, eta0, handoff_x, collar_width, tangent_width,
    )
    analytic_at_outer, analytic_slope_at_outer = _inlet_surface(
        inlet, collar_at_outer[:, 0], collar_at_outer[:, 1], time, eta0,
    )
    new_points = local_mesh.vertices[local_mesh.triangles[len(clipped_mesh.triangles):]]
    area = .5*np.linalg.norm(np.cross(
        new_points[:, 1]-new_points[:, 0], new_points[:, 2]-new_points[:, 0],
    ), axis=1)
    boundary_normals = local_mesh.normals[seam.indices].astype(float)
    metrics = OffshoreJoinMetrics(
        handoff_x=handoff_x,
        collar_outer_x=collar_outer_x,
        outer_edge_x=render_edge_x,
        collar_width=collar_width,
        tangent_width=tangent_width,
        tangent_height_bound_at_seam_samples_m=(16.0/81.0*tangent_width
            * float(np.max(np.abs(seam.slope-analytic_slope_at_seam)))),
        maximum_height_jump=float(np.max(np.abs(collar_at_seam[:, 2]-seam.height))),
        maximum_slope_jump=float(np.max(np.abs(collar_slope_at_seam-seam.slope))),
        maximum_outer_height_jump=float(np.max(np.abs(collar_at_outer[:, 2]-analytic_at_outer))),
        maximum_outer_slope_jump=float(np.max(np.abs(collar_slope_at_outer-analytic_slope_at_outer))),
        maximum_resolved_analytic_offset=float(np.max(np.abs(seam.height-analytic_at_seam))),
        maximum_resolved_analytic_slope_offset=float(np.max(np.abs(seam.slope-analytic_slope_at_seam))),
        overlap_width=max(0.0, render_edge_x-float(np.min(local_mesh.vertices[:, 0]))),
        maximum_collar_alongshore_displacement=float(np.max(np.abs(seam.parameter-seam.y))),
        minimum_collar_triangle_area=float(np.min(area)),
        maximum_boundary_tangent_residual=float(np.max(np.abs(
            boundary_normals[:, 0]+seam.slope*boundary_normals[:, 2],
        ))),
        welded_seam_folds=welded_seam_folds,
        maximum_weld_shift_m=maximum_weld_shift,
        removed_seam_slivers=removed_seam_slivers,
        unwelded_periodic_seam_edges=seam_boundary.unwelded_edges,
        unwelded_periodic_seam_x_min=seam_boundary.unwelded_x_min,
    )
    return OffshoreView(
        local_mesh=local_mesh,
        periodic_instance_mesh=periodic_instance_mesh,
        periodic_y_period=period_y,
        image=image,
        ripple_spectrum=spectrum,
        metrics=metrics,
        renderer=renderer,
    )
