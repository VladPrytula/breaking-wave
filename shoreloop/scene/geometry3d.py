"""Shared analytic geometry for the local 3-D solver and renderer.

The beach and boulders are defined once as implicit solids.  Solver geometry is
sampled from those implicit functions; rendering consumes triangle meshes made
from the same primitive transforms.  Negative signed distance denotes solid.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Sequence

import numpy as np


BEACH_MATERIAL_ID = 1
BOULDER_MATERIAL_ID = 2


def _points(
    values: np.ndarray | Sequence[float],
) -> tuple[np.ndarray, tuple[int, ...]]:
    points = np.asarray(values, dtype=np.float64)
    return points.reshape(-1, 3), points.shape[:-1]


def _signed_power(value: np.ndarray, power: float) -> np.ndarray:
    return np.sign(value) * np.abs(value) ** power


def _rotation_xyz(angles: tuple[float, float, float]) -> np.ndarray:
    rx, ry, rz = angles
    sx, cx = math.sin(rx), math.cos(rx)
    sy, cy = math.sin(ry), math.cos(ry)
    sz, cz = math.sin(rz), math.cos(rz)
    x = np.array(((1.0, 0.0, 0.0), (0.0, cx, -sx), (0.0, sx, cx)))
    y = np.array(((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy)))
    z = np.array(((cz, -sz, 0.0), (sz, cz, 0.0), (0.0, 0.0, 1.0)))
    return z @ y @ x


@dataclass(frozen=True)
class SceneMaterial:
    """Material identity shared by simulation diagnostics and rendering."""

    material_id: int
    name: str
    roughness: float
    dry_albedo: tuple[float, float, float]
    wet_albedo: tuple[float, float, float]
    wet_roughness: float


@dataclass(frozen=True)
class TriangleMesh:
    """One closed primitive mesh in world coordinates."""

    vertices: np.ndarray
    triangles: np.ndarray
    material_id: int

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def contains(self, values: np.ndarray | Sequence[float]) -> np.ndarray | bool:
        """Classify points by the closed mesh's solid-angle winding number."""
        points, shape = _points(values)
        winding = np.zeros(len(points), dtype=np.float64)
        vertices = self.vertices[self.triangles]
        for triangle in vertices:
            a = triangle[0] - points
            b = triangle[1] - points
            c = triangle[2] - points
            determinant = np.einsum("ij,ij->i", a, np.cross(b, c))
            la = np.linalg.norm(a, axis=1)
            lb = np.linalg.norm(b, axis=1)
            lc = np.linalg.norm(c, axis=1)
            denominator = (
                la * lb * lc
                + np.einsum("ij,ij->i", a, b) * lc
                + np.einsum("ij,ij->i", b, c) * la
                + np.einsum("ij,ij->i", c, a) * lb
            )
            winding += 2.0 * np.arctan2(determinant, denominator)
        inside = np.abs(winding) > 2.0 * math.pi
        return bool(inside[0]) if not shape else inside.reshape(shape)


@dataclass(frozen=True)
class AnalyticBeach:
    """Plane beach with a flat offshore continuation before its toe."""

    toe_x: float
    slope: float
    base_elevation: float = 0.0
    material_id: int = BEACH_MATERIAL_ID

    def elevation(self, x: np.ndarray | float) -> np.ndarray | float:
        values = np.asarray(x, dtype=np.float64)
        elevation = self.base_elevation + self.slope * np.maximum(
            values - self.toe_x, 0.0,
        )
        return float(elevation) if values.ndim == 0 else elevation

    def signed_distance(
        self,
        values: np.ndarray | Sequence[float],
    ) -> np.ndarray | float:
        points, shape = _points(values)
        rising = points[:, 0] > self.toe_x
        distance = points[:, 2] - self.base_elevation
        distance[rising] = (
            points[rising, 2]
            - self.base_elevation
            - self.slope * (points[rising, 0] - self.toe_x)
        ) / math.sqrt(1.0 + self.slope * self.slope)
        return float(distance[0]) if not shape else distance.reshape(shape)

    def surface_coordinates(
        self,
        values: np.ndarray | Sequence[float],
    ) -> np.ndarray:
        """Stable world-space coordinates on the analytic beach plane."""
        points, shape = _points(values)
        coordinates = points[:, :2]
        return coordinates.reshape((*shape, 2))

    def render_mesh(
        self,
        bounds: tuple[float, float, float, float],
        *,
        nx: int = 64,
        ny: int = 32,
    ) -> TriangleMesh:
        """Triangulate the beach over ``(xlo, xhi, ylo, yhi)``."""
        xlo, xhi, ylo, yhi = bounds
        x = np.linspace(xlo, xhi, nx + 1)
        y = np.linspace(ylo, yhi, ny + 1)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        vertices = np.column_stack((
            xx.ravel(), yy.ravel(),
            np.broadcast_to(self.elevation(x)[:, None], xx.shape).ravel(),
        ))
        triangles = []
        stride = ny + 1
        for i in range(nx):
            for j in range(ny):
                a = i * stride + j
                b = a + stride
                triangles.append((a, b, a + 1))
                triangles.append((a + 1, b, b + 1))
        return TriangleMesh(
            vertices, np.asarray(triangles, dtype=np.int32), self.material_id,
        )


@dataclass(frozen=True)
class RoundedBoulder:
    """A transformed rounded ellipsoid represented by an Lp superquadric."""

    centre: tuple[float, float, float]
    radii: tuple[float, float, float]
    rotation_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    exponent: float = 2.5
    material_id: int = BOULDER_MATERIAL_ID

    @property
    def rotation(self) -> np.ndarray:
        return _rotation_xyz(self.rotation_xyz)

    def local_coordinates(
        self,
        values: np.ndarray | Sequence[float],
    ) -> np.ndarray:
        points, shape = _points(values)
        local = (points - np.asarray(self.centre)) @ self.rotation
        return local.reshape((*shape, 3))

    def implicit(
        self,
        values: np.ndarray | Sequence[float],
    ) -> np.ndarray | float:
        points, shape = _points(values)
        local = (points - np.asarray(self.centre)) @ self.rotation
        scaled = np.abs(local / np.asarray(self.radii))
        value = np.sum(scaled ** self.exponent, axis=1) ** (
            1.0 / self.exponent
        ) - 1.0
        return float(value[0]) if not shape else value.reshape(shape)

    def signed_distance(
        self,
        values: np.ndarray | Sequence[float],
    ) -> np.ndarray | float:
        value = self.implicit(values)
        return value * min(self.radii)

    def surface_coordinates(
        self,
        values: np.ndarray | Sequence[float],
    ) -> np.ndarray:
        """Stable longitude/latitude coordinates on the superquadric."""
        points, shape = _points(values)
        local = (points - np.asarray(self.centre)) @ self.rotation
        scaled = local / np.asarray(self.radii)
        longitude = np.mod(np.arctan2(scaled[:, 1], scaled[:, 0]), 2.0 * math.pi)
        latitude = np.arctan2(
            scaled[:, 2], np.sqrt(scaled[:, 0] ** 2 + scaled[:, 1] ** 2),
        )
        coordinates = np.column_stack((
            longitude / (2.0 * math.pi),
            latitude / math.pi + 0.5,
        ))
        return coordinates.reshape((*shape, 2))

    def project_to_surface(self, values: np.ndarray) -> np.ndarray:
        """Project noncentral points radially onto this analytic primitive."""
        local = self.local_coordinates(values)
        radius = np.sum(
            np.abs(local / np.asarray(self.radii)) ** self.exponent, axis=-1,
        ) ** (1.0 / self.exponent)
        return (local / radius[..., None]) @ self.rotation.T + np.asarray(self.centre)

    def render_mesh(
        self,
        *,
        longitude_segments: int = 64,
        latitude_segments: int = 32,
    ) -> TriangleMesh:
        """Generate the transformed superquadric's closed render mesh."""
        power = 2.0 / self.exponent
        vertices = [np.array((0.0, 0.0, self.radii[2]))]
        for ring in range(1, latitude_segments):
            phi = 0.5 * math.pi - math.pi * ring / latitude_segments
            theta = 2.0 * math.pi * np.arange(longitude_segments) / longitude_segments
            cos_phi = _signed_power(np.full_like(theta, math.cos(phi)), power)
            sin_phi = _signed_power(np.full_like(theta, math.sin(phi)), power)
            vertices.extend(np.column_stack((
                self.radii[0] * cos_phi * _signed_power(np.cos(theta), power),
                self.radii[1] * cos_phi * _signed_power(np.sin(theta), power),
                self.radii[2] * sin_phi,
            )))
        south = len(vertices)
        vertices.append(np.array((0.0, 0.0, -self.radii[2])))

        triangles: list[tuple[int, int, int]] = []
        first = 1
        for j in range(longitude_segments):
            triangles.append((0, first + j, first + (j + 1) % longitude_segments))
        for ring in range(latitude_segments - 2):
            upper = 1 + ring * longitude_segments
            lower = upper + longitude_segments
            for j in range(longitude_segments):
                next_j = (j + 1) % longitude_segments
                triangles.append((upper + j, lower + j, upper + next_j))
                triangles.append((upper + next_j, lower + j, lower + next_j))
        last = south - longitude_segments
        for j in range(longitude_segments):
            triangles.append((south, last + (j + 1) % longitude_segments,
                              last + j))

        local = np.asarray(vertices, dtype=np.float64)
        world = local @ self.rotation.T + np.asarray(self.centre)
        return TriangleMesh(
            world, np.asarray(triangles, dtype=np.int32), self.material_id,
        )


@dataclass(frozen=True)
class SceneGeometry:
    """Union of the true beach and deterministic resolved boulders."""

    beach: AnalyticBeach
    boulders: tuple[RoundedBoulder, ...] = ()
    materials: tuple[SceneMaterial, ...] = (
        SceneMaterial(
            BEACH_MATERIAL_ID,
            "beach",
            0.78,
            (0.143, 0.138, 0.150),
            (0.11297, 0.10902, 0.11850),
            0.34,
        ),
        SceneMaterial(
            BOULDER_MATERIAL_ID,
            "boulder",
            0.62,
            (0.180, 0.155, 0.135),
            (0.14220, 0.12245, 0.10665),
            0.28,
        ),
    )

    def signed_distance(
        self,
        values: np.ndarray | Sequence[float],
    ) -> np.ndarray | float:
        points, shape = _points(values)
        distance = np.asarray(self.beach.signed_distance(points))
        for boulder in self.boulders:
            distance = np.minimum(distance, boulder.signed_distance(points))
        return float(distance[0]) if not shape else distance.reshape(shape)

    def material_ids(
        self,
        values: np.ndarray | Sequence[float],
    ) -> np.ndarray | int:
        points, shape = _points(values)
        best = np.asarray(self.beach.signed_distance(points))
        material = np.full(len(points), self.beach.material_id, dtype=np.int32)
        for boulder in self.boulders:
            distance = np.asarray(boulder.signed_distance(points))
            selected = distance < best
            best[selected] = distance[selected]
            material[selected] = boulder.material_id
        material[best >= 0.0] = 0
        return int(material[0]) if not shape else material.reshape(shape)

    def render_meshes(
        self,
        beach_bounds: tuple[float, float, float, float],
        *,
        beach_segments: tuple[int, int] = (64, 32),
        boulder_segments: tuple[int, int] = (64, 32),
    ) -> tuple[TriangleMesh, ...]:
        return (
            self.beach.render_mesh(
                beach_bounds, nx=beach_segments[0], ny=beach_segments[1],
            ),
            *(boulder.render_mesh(
                longitude_segments=boulder_segments[0],
                latitude_segments=boulder_segments[1],
            ) for boulder in self.boulders),
        )


def deterministic_boulders() -> tuple[RoundedBoulder, ...]:
    """The resolved M5 cluster; placements are fixed across resolutions."""
    return (
        RoundedBoulder(
            centre=(2.45, 0.55, 0.22), radii=(0.28, 0.20, 0.22),
            rotation_xyz=(0.08, -0.12, 0.32), exponent=2.6,
        ),
        RoundedBoulder(
            centre=(2.72, 1.22, 0.27), radii=(0.34, 0.24, 0.26),
            rotation_xyz=(-0.06, 0.10, -0.25), exponent=2.8,
        ),
        RoundedBoulder(
            centre=(3.02, 0.66, 0.34), radii=(0.30, 0.23, 0.28),
            rotation_xyz=(0.12, 0.05, 0.18), exponent=2.4,
        ),
    )


def production_boulders() -> tuple[RoundedBoulder, ...]:
    """The AW-1102 low, beach-seated field, fixed across resolutions."""
    return (
        RoundedBoulder(
            centre=(2.50, 3.35, 0.303269), radii=(0.30, 0.24, 0.21),
            rotation_xyz=(0.08, -0.12, 0.20), exponent=2.55,
        ),
        RoundedBoulder(
            centre=(2.65, 3.98, 0.322411), radii=(0.27, 0.22, 0.19),
            rotation_xyz=(-0.06, 0.10, -0.25), exponent=2.80,
        ),
        RoundedBoulder(
            centre=(2.28, 4.52, 0.234002), radii=(0.24, 0.20, 0.17),
            rotation_xyz=(0.10, -0.08, 0.32), exponent=2.45,
        ),
        RoundedBoulder(
            centre=(2.05, 1.55, 0.230508), radii=(0.28, 0.23, 0.19),
            rotation_xyz=(-0.07, 0.13, 0.18), exponent=2.70,
        ),
        RoundedBoulder(
            centre=(2.78, 5.25, 0.313350), radii=(0.26, 0.22, 0.18),
            rotation_xyz=(0.06, -0.11, -0.30), exponent=2.60,
        ),
    )
