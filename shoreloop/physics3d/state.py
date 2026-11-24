"""Primary device-resident state on the local three-dimensional MAC grid."""

from __future__ import annotations

import numpy as np
import warp as wp

from shoreloop.config import Solver3DConfig
from shoreloop.device import prepare_device
from shoreloop.physics3d.grid import MACGrid


@wp.kernel
def _copy_into_interior(
    source: wp.array3d(dtype=wp.float32),
    target: wp.array3d(dtype=wp.float32),
    ng: int,
):
    i, j, k = wp.tid()
    target[ng + i, ng + j, ng + k] = source[i, j, k]


@wp.kernel
def _copy_from_interior(
    source: wp.array3d(dtype=wp.float32),
    target: wp.array3d(dtype=wp.float32),
    ng: int,
):
    i, j, k = wp.tid()
    target[i, j, k] = source[ng + i, ng + j, ng + k]


@wp.kernel
def _copy_int_into_interior(
    source: wp.array3d(dtype=wp.int32),
    target: wp.array3d(dtype=wp.int32),
    ng: int,
):
    i, j, k = wp.tid()
    target[ng + i, ng + j, ng + k] = source[i, j, k]


@wp.kernel
def _copy_int_from_interior(
    source: wp.array3d(dtype=wp.int32),
    target: wp.array3d(dtype=wp.int32),
    ng: int,
):
    i, j, k = wp.tid()
    target[i, j, k] = source[ng + i, ng + j, ng + k]


@wp.kernel
def _copy_vector_into_interior(
    source: wp.array3d(dtype=wp.vec3),
    target: wp.array3d(dtype=wp.vec3),
    ng: int,
):
    i, j, k = wp.tid()
    target[ng + i, ng + j, ng + k] = source[i, j, k]


class State3D:
    """Primary flow fields and static embedded-solid geometry."""

    FIELDS = (
        "p", "alpha", "u", "v", "w",
        "porous_storage", "foam", "solid_wetness",
    )
    __slots__ = (
        "cfg", "grid", "device", "p", "alpha", "u", "v", "w", "solid",
        "theta", "u_aperture", "v_aperture", "w_aperture", "material",
        "porous_storage", "foam", "solid_wetness",
        "solid_cells", "small_cell_threshold", "small_cell_count",
        "minimum_open_fraction",
        "open_normals", "open_offsets", "cut_normal_cache",
    )

    def __init__(self, cfg: Solver3DConfig):
        self.cfg = cfg
        self.grid = MACGrid(cfg.grid)
        prepare_device(cfg.device)
        from shoreloop.physics3d.cut_normals import CutNormalCache
        self.cut_normal_cache = CutNormalCache(cfg.grid.shape, cfg.device)
        self.device = cfg.device
        self.p = wp.zeros(self.grid.padded_shape("p"), dtype=wp.float32,
                          device=self.device)
        self.alpha = wp.zeros(self.grid.padded_shape("alpha"), dtype=wp.float32,
                              device=self.device)
        self.u = wp.zeros(self.grid.u_shape, dtype=wp.float32, device=self.device)
        self.v = wp.zeros(self.grid.v_shape, dtype=wp.float32, device=self.device)
        self.w = wp.zeros(self.grid.w_shape, dtype=wp.float32, device=self.device)
        self.porous_storage = wp.zeros(
            self.grid.cell_shape, dtype=wp.float32, device=self.device,
        )
        self.foam = wp.zeros(
            self.grid.cell_shape, dtype=wp.float32, device=self.device,
        )
        self.solid_wetness = wp.zeros(
            self.grid.cell_shape, dtype=wp.float32, device=self.device,
        )
        self.solid = wp.zeros(
            self.grid.cell_shape, dtype=wp.int32, device=self.device,
        )
        self.theta = wp.ones(
            self.grid.cell_shape, dtype=wp.float32, device=self.device,
        )
        # Cell-local open halfspace n.xi <= d. Only cut cells have a normal;
        # its offset reconstructs the authoritative theta without a mass remap.
        self.open_normals = wp.zeros(
            self.grid.cell_shape, dtype=wp.vec3, device=self.device,
        )
        self.open_offsets = wp.zeros(
            self.grid.cell_shape, dtype=wp.float32, device=self.device,
        )
        self.u_aperture = wp.ones(
            self.grid.u_shape, dtype=wp.float32, device=self.device,
        )
        self.v_aperture = wp.ones(
            self.grid.v_shape, dtype=wp.float32, device=self.device,
        )
        self.w_aperture = wp.ones(
            self.grid.w_shape, dtype=wp.float32, device=self.device,
        )
        self.material = wp.zeros(
            self.grid.cell_shape, dtype=wp.int32, device=self.device,
        )
        self.solid_cells = 0
        self.small_cell_threshold = 0.0
        self.small_cell_count = 0
        self.minimum_open_fraction = 1.0

    def field(self, name: str) -> wp.array:
        if name not in self.FIELDS:
            raise ValueError(f"unknown 3-D state field {name!r}; choose from {self.FIELDS}")
        return getattr(self, name)

    def zero_(self, name: str | None = None) -> None:
        """Zero one field or the complete primary state on the device."""
        names = self.FIELDS if name is None else (name,)
        for field_name in names:
            self.field(field_name).zero_()

    def fill_(self, name: str, value: float) -> None:
        """Fill one complete padded field on the device."""
        self.field(name).fill_(float(value))

    def set_interior(self, name: str, values: np.ndarray) -> None:
        """Upload one physical field without copying its ghost halo via host."""
        shape = self.grid.interior_shape(name)
        host = np.ascontiguousarray(values, dtype=np.float32)
        if host.shape != shape:
            raise ValueError(f"{name} interior must have shape {shape}, got {host.shape}")
        source = wp.array(host, dtype=wp.float32, device=self.device)
        wp.launch(_copy_into_interior, dim=shape,
                  inputs=[source, self.field(name), self.cfg.grid.ng],
                  device=self.device)

    def get_interior(self, name: str) -> np.ndarray:
        """Download only the physical part of one field for diagnostics."""
        shape = self.grid.interior_shape(name)
        target = wp.empty(shape, dtype=wp.float32, device=self.device)
        wp.launch(_copy_from_interior, dim=shape,
                  inputs=[self.field(name), target, self.cfg.grid.ng],
                  device=self.device)
        return target.numpy()

    def set_solid_interior(self, values: np.ndarray) -> None:
        """Upload the static cell mask; non-zero values denote solid cells."""
        shape = self.grid.interior_shape("cell")
        host = np.ascontiguousarray(values, dtype=np.int32)
        if host.shape != shape:
            raise ValueError(f"solid interior must have shape {shape}, got {host.shape}")
        source = wp.array(host, dtype=wp.int32, device=self.device)
        self.solid.zero_()
        wp.launch(
            _copy_int_into_interior, dim=shape,
            inputs=[source, self.solid, self.cfg.grid.ng], device=self.device,
        )
        self.solid_cells = int(np.count_nonzero(host))

    def get_solid_interior(self) -> np.ndarray:
        """Download the physical voxel-solid mask."""
        shape = self.grid.interior_shape("cell")
        target = wp.empty(shape, dtype=wp.int32, device=self.device)
        wp.launch(
            _copy_int_from_interior, dim=shape,
            inputs=[self.solid, target, self.cfg.grid.ng], device=self.device,
        )
        return target.numpy()

    def set_geometry_interior(self, name: str, values: np.ndarray) -> None:
        """Upload one cell-volume or face-aperture geometry field."""
        field = getattr(self, name)
        location = "cell" if name == "theta" else name.removesuffix("_aperture")
        shape = self.grid.interior_shape(location)
        host = np.ascontiguousarray(values, dtype=np.float32)
        if host.shape != shape:
            raise ValueError(f"{name} interior must have shape {shape}, got {host.shape}")
        source = wp.array(host, dtype=wp.float32, device=self.device)
        field.zero_()
        wp.launch(
            _copy_into_interior, dim=shape,
            inputs=[source, field, self.cfg.grid.ng], device=self.device,
        )

    def get_geometry_interior(self, name: str) -> np.ndarray:
        """Download one physical cell-volume or face-aperture field."""
        field = getattr(self, name)
        location = "cell" if name == "theta" else name.removesuffix("_aperture")
        shape = self.grid.interior_shape(location)
        target = wp.empty(shape, dtype=wp.float32, device=self.device)
        wp.launch(
            _copy_from_interior, dim=shape,
            inputs=[field, target, self.cfg.grid.ng], device=self.device,
        )
        return target.numpy()

    def set_open_planes(self, normals: np.ndarray, offsets: np.ndarray) -> None:
        """Upload the volume-matched planar representation of static cut cells."""
        shape = self.cfg.grid.shape
        normals = np.ascontiguousarray(normals, dtype=np.float32)
        offsets = np.ascontiguousarray(offsets, dtype=np.float32)
        if normals.shape != (*shape, 3) or offsets.shape != shape:
            raise ValueError("open planes must have cell-shaped normals and offsets")
        self.open_normals.zero_()
        self.open_offsets.zero_()
        wp.launch(
            _copy_vector_into_interior, dim=shape,
            inputs=[wp.array(normals, dtype=wp.vec3, device=self.device),
                    self.open_normals, self.cfg.grid.ng], device=self.device,
        )
        wp.launch(
            _copy_into_interior, dim=shape,
            inputs=[wp.array(offsets, dtype=wp.float32, device=self.device),
                    self.open_offsets, self.cfg.grid.ng], device=self.device,
        )

    def set_material_interior(self, values: np.ndarray) -> None:
        """Upload per-cell scene material IDs."""
        shape = self.grid.interior_shape("cell")
        host = np.ascontiguousarray(values, dtype=np.int32)
        source = wp.array(host, dtype=wp.int32, device=self.device)
        self.material.zero_()
        wp.launch(
            _copy_int_into_interior, dim=shape,
            inputs=[source, self.material, self.cfg.grid.ng], device=self.device,
        )

    def get_material_interior(self) -> np.ndarray:
        """Download the physical scene material IDs."""
        shape = self.grid.interior_shape("cell")
        target = wp.empty(shape, dtype=wp.int32, device=self.device)
        wp.launch(
            _copy_int_from_interior, dim=shape,
            inputs=[self.material, target, self.cfg.grid.ng], device=self.device,
        )
        return target.numpy()
