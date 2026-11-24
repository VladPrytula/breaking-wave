"""Compatible regular-grid MAC divergence, gradient, and pressure operator.

The production fields remain float32.  Diagnostic inner products accumulate
on the configured CUDA device in float64 so operator compatibility is measured
without adding float32 summation error.

Boundary conventions match :mod:`shoreloop.physics3d.boundary`: wall and open
cell fields have zero normal continuation, while a periodic pressure gradient
crosses the seam.  The duplicated high periodic face is not a second degree of
freedom and is therefore omitted from the face inner product.

The pressure matrix exposed here is ``A = -D G``.  The minus sign is required
because compatibility makes ``D G`` negative semidefinite; ``-D G`` is the
positive-semidefinite form expected by conjugate gradient.
"""

from __future__ import annotations

from collections.abc import Sequence

import warp as wp

from shoreloop.config import BC, Solver3DConfig
from shoreloop.physics3d.grid import MACGrid


FaceVector = tuple[wp.array, wp.array, wp.array]


@wp.kernel
def _divergence(
    u: wp.array3d(dtype=wp.float32),
    v: wp.array3d(dtype=wp.float32),
    w: wp.array3d(dtype=wp.float32),
    out: wp.array3d(dtype=wp.float32),
    ng: int,
    inv_dx: wp.float32,
    inv_dy: wp.float32,
    inv_dz: wp.float32,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    out[I, J, K] = (
        (u[I + 1, J, K] - u[I, J, K]) * inv_dx
        + (v[I, J + 1, K] - v[I, J, K]) * inv_dy
        + (w[I, J, K + 1] - w[I, J, K]) * inv_dz
    )


@wp.kernel
def _gradient_axis(
    p: wp.array3d(dtype=wp.float32),
    out: wp.array3d(dtype=wp.float32),
    axis: int,
    n: int,
    ng: int,
    inverse_spacing: wp.float32,
    periodic: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    q = i
    if axis == 1:
        q = j
    elif axis == 2:
        q = k

    value = wp.float32(0.0)
    if q > 0 and q < n:
        if axis == 0:
            value = (p[I, J, K] - p[I - 1, J, K]) * inverse_spacing
        elif axis == 1:
            value = (p[I, J, K] - p[I, J - 1, K]) * inverse_spacing
        else:
            value = (p[I, J, K] - p[I, J, K - 1]) * inverse_spacing
    elif periodic != 0:
        # Both physical seam faces carry the same value.  AW-102 declares the
        # low face canonical and the high face its duplicate.
        if axis == 0:
            value = (p[ng, J, K] - p[ng + n - 1, J, K]) * inverse_spacing
        elif axis == 1:
            value = (p[I, ng, K] - p[I, ng + n - 1, K]) * inverse_spacing
        else:
            value = (p[I, J, ng] - p[I, J, ng + n - 1]) * inverse_spacing

    out[I, J, K] = value


@wp.kernel
def _cell_inner_product(
    left: wp.array3d(dtype=wp.float32),
    right: wp.array3d(dtype=wp.float32),
    out: wp.array(dtype=wp.float64),
    ng: int,
    cell_volume: wp.float64,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    product = wp.float64(left[I, J, K]) * wp.float64(right[I, J, K])
    wp.atomic_add(out, 0, product * cell_volume)


@wp.kernel
def _face_inner_product(
    left: wp.array3d(dtype=wp.float32),
    right: wp.array3d(dtype=wp.float32),
    out: wp.array(dtype=wp.float64),
    axis: int,
    n: int,
    ng: int,
    cell_volume: wp.float64,
    periodic: int,
):
    i, j, k = wp.tid()
    q = i
    if axis == 1:
        q = j
    elif axis == 2:
        q = k
    if periodic != 0 and q == n:
        return

    I = i + ng
    J = j + ng
    K = k + ng
    product = wp.float64(left[I, J, K]) * wp.float64(right[I, J, K])
    wp.atomic_add(out, 0, product * cell_volume)


@wp.kernel
def _negate_cell_interior(
    field: wp.array3d(dtype=wp.float32),
    ng: int,
):
    i, j, k = wp.tid()
    I = i + ng
    J = j + ng
    K = k + ng
    field[I, J, K] = -field[I, J, K]


@wp.kernel
def _pressure_diagonal(
    out: wp.array3d(dtype=wp.float32),
    nx: int,
    ny: int,
    nz: int,
    ng: int,
    inv_dx2: wp.float32,
    inv_dy2: wp.float32,
    inv_dz2: wp.float32,
    periodic_x: int,
    periodic_y: int,
    periodic_z: int,
):
    i, j, k = wp.tid()
    diagonal = wp.float32(0.0)

    if nx > 1:
        if periodic_x != 0:
            diagonal += wp.float32(2.0) * inv_dx2
        else:
            if i > 0:
                diagonal += inv_dx2
            if i + 1 < nx:
                diagonal += inv_dx2

    if ny > 1:
        if periodic_y != 0:
            diagonal += wp.float32(2.0) * inv_dy2
        else:
            if j > 0:
                diagonal += inv_dy2
            if j + 1 < ny:
                diagonal += inv_dy2

    if nz > 1:
        if periodic_z != 0:
            diagonal += wp.float32(2.0) * inv_dz2
        else:
            if k > 0:
                diagonal += inv_dz2
            if k + 1 < nz:
                diagonal += inv_dz2

    out[i + ng, j + ng, k + ng] = diagonal


class MACOperators:
    """CUDA launches for the first regular-grid three-dimensional operators."""

    __slots__ = ("cfg", "grid", "device")

    def __init__(self, cfg: Solver3DConfig):
        self.cfg = cfg
        self.grid = MACGrid(cfg.grid)
        self.device = cfg.device

    def _check_field(self, name: str, field: wp.array) -> None:
        expected = self.grid.padded_shape(name)
        if tuple(field.shape) != expected:
            raise ValueError(f"{name} field must have padded shape {expected}, "
                             f"got {tuple(field.shape)}")
        if field.dtype != wp.float32:
            raise TypeError(f"{name} field must use float32, got {field.dtype}")

    def divergence(
        self,
        u: wp.array,
        v: wp.array,
        w: wp.array,
        out: wp.array | None = None,
    ) -> wp.array:
        """Return ``D(u, v, w)`` at padded cell centres.

        Only physical face values enter the stencil.  Callers apply velocity
        boundary ownership before this launch when constrained sides are used.
        """
        self._check_field("u", u)
        self._check_field("v", v)
        self._check_field("w", w)
        if out is None:
            out = wp.zeros(self.grid.cell_shape, dtype=wp.float32,
                           device=self.device)
        else:
            self._check_field("cell", out)
            out.zero_()

        g = self.cfg.grid
        wp.launch(
            _divergence,
            dim=g.shape,
            inputs=[u, v, w, out, g.ng, 1.0 / g.dx, 1.0 / g.dy, 1.0 / g.dz],
            device=self.device,
        )
        return out

    def pressure_gradient(
        self,
        p: wp.array,
        out: FaceVector | None = None,
    ) -> FaceVector:
        """Return ``G(p)`` on padded x-, y-, and z-face lattices.

        Wall/open boundary gradients are zero; periodic seam gradients use the
        last and first physical cells.  Both periodic seam faces are written so
        the result obeys the same duplicated-face convention as velocity.
        """
        self._check_field("p", p)
        if out is None:
            result = tuple(
                wp.zeros(self.grid.padded_shape(name), dtype=wp.float32,
                         device=self.device)
                for name in ("u", "v", "w")
            )
        else:
            if len(out) != 3:
                raise ValueError("pressure-gradient output must contain u, v, and w")
            result = tuple(out)
            for name, field in zip(("u", "v", "w"), result):
                self._check_field(name, field)
                field.zero_()

        g = self.cfg.grid
        bc = self.cfg.bc
        axes = (
            ("u", 0, g.nx, g.dx, bc.xlo == BC.PERIODIC),
            ("v", 1, g.ny, g.dy, bc.ylo == BC.PERIODIC),
            ("w", 2, g.nz, g.dz, bc.zlo == BC.PERIODIC),
        )
        for field, (name, axis, n, spacing, periodic) in zip(result, axes):
            wp.launch(
                _gradient_axis,
                dim=self.grid.interior_shape(name),
                inputs=[p, field, axis, n, g.ng, 1.0 / spacing, int(periodic)],
                device=self.device,
            )
        return result  # type: ignore[return-value]

    def pressure_operator(
        self,
        p: wp.array,
        out: wp.array | None = None,
        gradient_out: FaceVector | None = None,
    ) -> wp.array:
        """Return the matrix-free pressure action ``-D(G(p))``.

        ``D G`` is negative semidefinite under the compatible AW-103 inner
        products, so this positive-semidefinite sign is the one consumed by
        PCG.  The intermediate face gradient can be supplied by a caller that
        owns reusable workspace; no sparse matrix is assembled.
        """
        gradient = self.pressure_gradient(p, out=gradient_out)
        result = self.divergence(*gradient, out=out)
        wp.launch(
            _negate_cell_interior,
            dim=self.cfg.grid.shape,
            inputs=[result, self.cfg.grid.ng],
            device=self.device,
        )
        return result

    def pressure_diagonal(self, out: wp.array | None = None) -> wp.array:
        """Return the Jacobi diagonal of the positive pressure operator."""
        if out is None:
            out = wp.zeros(self.grid.cell_shape, dtype=wp.float32,
                           device=self.device)
        else:
            self._check_field("cell", out)
            out.zero_()

        g = self.cfg.grid
        bc = self.cfg.bc
        wp.launch(
            _pressure_diagonal,
            dim=g.shape,
            inputs=[
                out, g.nx, g.ny, g.nz, g.ng,
                1.0 / (g.dx * g.dx),
                1.0 / (g.dy * g.dy),
                1.0 / (g.dz * g.dz),
                int(bc.xlo == BC.PERIODIC),
                int(bc.ylo == BC.PERIODIC),
                int(bc.zlo == BC.PERIODIC),
            ],
            device=self.device,
        )
        return out

    def cell_inner_product(self, left: wp.array, right: wp.array) -> float:
        """Measure a volume-weighted cell inner product with float64 reduction."""
        self._check_field("cell", left)
        self._check_field("cell", right)
        g = self.cfg.grid
        total = wp.zeros(1, dtype=wp.float64, device=self.device)
        wp.launch(
            _cell_inner_product,
            dim=g.shape,
            inputs=[left, right, total, g.ng, g.dx * g.dy * g.dz],
            device=self.device,
        )
        return float(total.numpy()[0])

    def face_inner_product(
        self,
        left: Sequence[wp.array],
        right: Sequence[wp.array],
    ) -> float:
        """Measure the volume-weighted inner product of two MAC vectors.

        Periodic high faces duplicate their canonical low faces and are skipped.
        Other physical boundary faces are included; compatible wall fields make
        their normal contribution zero.
        """
        if len(left) != 3 or len(right) != 3:
            raise ValueError("face vectors must contain u, v, and w components")
        names = ("u", "v", "w")
        for fields in (left, right):
            for name, field in zip(names, fields):
                self._check_field(name, field)

        g = self.cfg.grid
        bc = self.cfg.bc
        cell_volume = g.dx * g.dy * g.dz
        total = wp.zeros(1, dtype=wp.float64, device=self.device)
        axes = (
            (0, g.nx, bc.xlo == BC.PERIODIC),
            (1, g.ny, bc.ylo == BC.PERIODIC),
            (2, g.nz, bc.zlo == BC.PERIODIC),
        )
        for name, a, b, (axis, n, periodic) in zip(names, left, right, axes):
            wp.launch(
                _face_inner_product,
                dim=self.grid.interior_shape(name),
                inputs=[a, b, total, axis, n, g.ng, cell_volume, int(periodic)],
                device=self.device,
            )
        return float(total.numpy()[0])
