"""Smooth wave generation and absorption at the low-x side of a 3-D tank."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import warp as wp

from shoreloop.config import Solver3DConfig
from shoreloop.physics3d.boundary import fill_boundaries
from shoreloop.physics3d.momentum import measure_liquid_momentum
from shoreloop.physics3d.state import State3D
from shoreloop.physics3d.vof import measure_vof
from shoreloop.physics3d.waves import LinearWaveInlet


RELAXATION_INTERFACE_REPRESENTATION = "geometric_column_displacement"


def relaxation_weight(x: float, start: float, width: float) -> float:
    """Cubic smoothstep from one at ``start`` to zero at ``start + width``."""
    s = min(1.0, max(0.0, (start + width - x) / width))
    return s * s * (3.0 - 2.0 * s)


@wp.func
def _zone_weight(x: wp.float32, start: wp.float32, width: wp.float32) -> wp.float32:
    s = wp.clamp((start + width - x) / width, wp.float32(0.0), wp.float32(1.0))
    return s * s * (wp.float32(3.0) - wp.float32(2.0) * s)


@wp.func
def _generation_envelope(x: wp.float32, start: wp.float32, ramp: wp.float32) -> wp.float32:
    """Incident amplitude factor: zero at the offshore boundary, one past the ramp.

    A wall cannot carry the incident field's normal velocity. Generating the
    wave from rest over a short ramp keeps the target consistent with the
    boundary, while the zone core still imposes the full field.
    """
    if ramp <= wp.float32(0.0):
        return wp.float32(1.0)
    return wp.float32(1.0) - _zone_weight(x, start, ramp)


@wp.func
def _source_weight(
    spatial_weight: wp.float32,
    relaxation_rate: wp.float32,
    impose: int,
) -> wp.float32:
    """Blend toward the incident field: per step by rate, or once with the zone taper."""
    if impose != 0:
        return spatial_weight
    return wp.float32(1.0) - wp.exp(-relaxation_rate * spatial_weight)


@wp.kernel
def _relax_alpha(
    before: wp.array3d(dtype=wp.float32),
    alpha: wp.array3d(dtype=wp.float32),
    amplitude: wp.array(dtype=wp.float32),
    second_elevation_amplitude: wp.array(dtype=wp.float32),
    omega: wp.array(dtype=wp.float32),
    kx: wp.array(dtype=wp.float32),
    ky: wp.array(dtype=wp.float32),
    phase_offset: wp.array(dtype=wp.float32),
    mode_count: int,
    time: wp.float32,
    start: wp.float32,
    width: wp.float32,
    ramp: wp.float32,
    relaxation_rate: wp.float32,
    still_water_level: wp.float32,
    x0: wp.float32,
    y0: wp.float32,
    z0: wp.float32,
    dx: wp.float32,
    dy: wp.float32,
    dz: wp.float32,
    nz: int,
    ng: int,
    impose: int,
):
    i, j = wp.tid()
    x = x0 + (wp.float32(i) + wp.float32(0.5)) * dx
    spatial_weight = _zone_weight(x, start, width)
    if spatial_weight == wp.float32(0.0):
        return
    weight = _source_weight(spatial_weight, relaxation_rate, impose)
    if weight == wp.float32(0.0):
        return
    y = y0 + (wp.float32(j) + wp.float32(0.5)) * dy
    elevation = wp.float32(0.0)
    for mode in range(mode_count):
        angle = kx[mode] * x + ky[mode] * y - omega[mode] * time + phase_offset[mode]
        elevation += amplitude[mode] * wp.cos(angle)
        elevation += second_elevation_amplitude[mode] * wp.cos(
            wp.float32(2.0) * angle,
        )
    surface = still_water_level + _generation_envelope(x, start, ramp) * elevation
    I = i + ng
    J = j + ng
    # Work in cell-height units to avoid subtracting nearly equal world
    # coordinates at a grid-face interface. State remains FP32; geometric
    # interval endpoints/overlaps accumulate in FP64 before the final write.
    current_height = wp.float64(0.0)
    for k in range(nz):
        current_height += wp.float64(before[I, J, k + ng])
    target_height = wp.float64(wp.clamp(
        (surface - z0) / dz, wp.float32(0.0), wp.float32(nz),
    ))
    source_weight = wp.float64(weight)
    retained = wp.float64(1.0)-source_weight
    target_scale = wp.float64(0.0)
    if current_height > wp.float64(0.0):
        target_scale = source_weight*target_height/current_height
    for k in range(nz):
        bottom = wp.float64(k)
        top = bottom+wp.float64(1.0)
        occupied = wp.float64(0.0)
        if current_height == wp.float64(0.0):
            # A dry source column has no existing parcels. Introduce exactly
            # its declared volume as one bottom-connected liquid interval.
            occupied = wp.clamp(source_weight*target_height-bottom,
                                wp.float64(0.0), wp.float64(1.0))
        else:
            cumulative_water = wp.float64(0.0)
            for old_k in range(nz):
                fraction = wp.float64(before[I, J, old_k + ng])
                if fraction > wp.float64(0.0):
                    # Named source approximation: each cell's liquid occupies
                    # its lower vertical interval [old_k, old_k+alpha]. Move
                    # its volume coordinate q by z'=(1-w)z+w*(H_target/H)q.
                    # Liquid lengths scale by H_new/H; gaps shrink by 1-w.
                    # Thus a graph stays sharp, while finite weights retain
                    # folds instead of resetting the whole column topology.
                    interval_bottom = retained*wp.float64(old_k)+target_scale*cumulative_water
                    interval_top = (retained*(wp.float64(old_k)+fraction)
                           + target_scale*(cumulative_water+fraction))
                    occupied += wp.max(wp.float64(0.0),
                        wp.min(top, interval_top)-wp.max(bottom, interval_bottom))
                cumulative_water += fraction
        # Ordered disjoint intervals imply 0 <= occupied <= 1. No clipping or
        # volume repair is applied after conservative interval intersection.
        alpha[I, J, k + ng] = wp.float32(occupied)


@wp.kernel
def _relax_velocity(
    velocity: wp.array3d(dtype=wp.float32),
    alpha: wp.array3d(dtype=wp.float32),
    amplitude: wp.array(dtype=wp.float32),
    second_elevation_amplitude: wp.array(dtype=wp.float32),
    second_potential_amplitude: wp.array(dtype=wp.float32),
    omega: wp.array(dtype=wp.float32),
    wave_number: wp.array(dtype=wp.float32),
    kx: wp.array(dtype=wp.float32),
    ky: wp.array(dtype=wp.float32),
    phase_offset: wp.array(dtype=wp.float32),
    mode_count: int,
    component_axis: int,
    time: wp.float32,
    start: wp.float32,
    width: wp.float32,
    ramp: wp.float32,
    relaxation_rate: wp.float32,
    depth: wp.float32,
    still_water_level: wp.float32,
    x0: wp.float32,
    y0: wp.float32,
    z0: wp.float32,
    dx: wp.float32,
    dy: wp.float32,
    dz: wp.float32,
    gravity_increment: wp.float32,
    nz: int,
    ng: int,
    impose: int,
):
    i, j, k = wp.tid()
    ox = wp.float32(0.5)
    oy = wp.float32(0.5)
    oz = wp.float32(0.5)
    if component_axis == 0:
        ox = wp.float32(0.0)
    elif component_axis == 1:
        oy = wp.float32(0.0)
    else:
        oz = wp.float32(0.0)
    x = x0 + (wp.float32(i) + ox) * dx
    spatial_weight = _zone_weight(x, start, width)
    if spatial_weight == wp.float32(0.0):
        return
    weight = _source_weight(spatial_weight, relaxation_rate, impose)
    y = y0 + (wp.float32(j) + oy) * dy
    z = z0 + (wp.float32(k) + oz) * dz - still_water_level

    envelope = _generation_envelope(x, start, ramp)
    elevation = wp.float32(0.0)
    for mode in range(mode_count):
        angle = kx[mode] * x + ky[mode] * y - omega[mode] * time + phase_offset[mode]
        elevation += amplitude[mode] * wp.cos(angle)
        elevation += second_elevation_amplitude[mode] * wp.cos(
            wp.float32(2.0) * angle,
        )
    elevation *= envelope

    I = i + ng
    J = j + ng
    K = k + ng
    target = wp.float32(0.0)
    orbital = wp.float32(0.0)
    if component_axis == 2:
        active = int(0)
        if k > 0 and alpha[I, J, K - 1] >= wp.float32(0.5):
            active = 1
        if k < nz and alpha[I, J, K] >= wp.float32(0.5):
            active = 1
        if active != 0:
            target = gravity_increment
    # Imposing places the water surface at weight * elevation (the displaced
    # column), so the velocity support must follow that surface, not the
    # full analytic one; otherwise dry faces above a tapered crest carry
    # momentum and wet faces in a tapered trough carry none.
    surface = elevation
    if impose != 0:
        surface = weight * elevation
    if z >= -depth and z <= surface:
        for mode in range(mode_count):
            angle = kx[mode] * x + ky[mode] * y - omega[mode] * time + phase_offset[mode]
            denominator = wp.sinh(wave_number[mode] * depth)
            if component_axis < 2:
                direction = kx[mode] / wave_number[mode]
                if component_axis == 1:
                    direction = ky[mode] / wave_number[mode]
                orbital += (
                    amplitude[mode] * omega[mode]
                    * wp.cosh(wave_number[mode] * (z + depth)) / denominator
                    * direction * wp.cos(angle)
                )
            else:
                orbital += (
                    amplitude[mode] * omega[mode]
                    * wp.sinh(wave_number[mode] * (z + depth)) / denominator
                    * wp.sin(angle)
                )
            second_scale = (
                wp.float32(2.0) * wave_number[mode]
                * second_potential_amplitude[mode]
            )
            if component_axis < 2:
                direction = kx[mode] / wave_number[mode]
                if component_axis == 1:
                    direction = ky[mode] / wave_number[mode]
                orbital += (
                    second_scale
                    * wp.cosh(wp.float32(2.0) * wave_number[mode]
                              * (z + depth))
                    * direction * wp.cos(wp.float32(2.0) * angle)
                )
            else:
                orbital += (
                    second_scale
                    * wp.sinh(wp.float32(2.0) * wave_number[mode]
                              * (z + depth))
                    * wp.sin(wp.float32(2.0) * angle)
                )

    target += envelope * orbital
    velocity[I, J, K] += weight * (target - velocity[I, J, K])


@dataclass(frozen=True)
class WaveRelaxationResult:
    """The source contribution from one relaxation application."""

    volume_before: float
    volume_after: float
    volume_source: float
    momentum_source: tuple[float, float, float]


class WaveRelaxationZone:
    """Blend toward the incoming wave while damping outgoing deviations."""

    __slots__ = (
        "cfg", "inlet", "width", "still_water_level", "device",
        "timescale", "generation_ramp",
        "_amplitude", "_second_elevation", "_second_potential", "_omega",
        "_wave_number", "_kx", "_ky", "_phase",
        "_alpha_before",
    )

    def __init__(
        self,
        cfg: Solver3DConfig,
        inlet: LinearWaveInlet,
        width: float,
        *,
        still_water_level: float | None = None,
        timescale: float | None = None,
        generation_ramp: float = 0.0,
    ):
        if width <= 0.0 or width > cfg.grid.Lx:
            raise ValueError("relaxation width must lie within the x extent")
        if not 0.0 <= generation_ramp <= width:
            raise ValueError("generation ramp must lie within the relaxation width")
        # Incident amplitude rises from zero at the offshore boundary to one
        # over this distance, so a wall there sees a target at rest. Zero
        # keeps the full incident field up to the boundary.
        self.generation_ramp = float(generation_ramp)
        component = inlet.components[0]
        if component.alongshore_length != cfg.grid.Ly:
            raise ValueError("wave alongshore length must match the periodic tank width")
        if component.gravity != cfg.physics.g:
            raise ValueError("wave and solver gravity must match")
        self.cfg = cfg
        self.inlet = inlet
        self.width = float(width)
        self.timescale = (
            0.1 * inlet.period if timescale is None else float(timescale)
        )
        if self.timescale <= 0.0:
            raise ValueError("relaxation timescale must be positive")
        self.still_water_level = (
            cfg.grid.z0 + inlet.depth
            if still_water_level is None else float(still_water_level)
        )
        self.device = cfg.device
        components = inlet.components

        def device_array(values: list[float]) -> wp.array:
            return wp.array(np.asarray(values, dtype=np.float32),
                            dtype=wp.float32, device=self.device)

        self._amplitude = device_array([component.amplitude for component in components])
        self._second_elevation = device_array([
            component.second_order_elevation_amplitude for component in components
        ])
        self._second_potential = device_array([
            component.second_order_potential_amplitude for component in components
        ])
        self._omega = device_array([component.omega for component in components])
        self._wave_number = device_array([component.k for component in components])
        self._kx = device_array([component.kx for component in components])
        self._ky = device_array([component.ky for component in components])
        self._phase = device_array([component.phase_offset for component in components])
        # Read the old column throughout the remap: a displaced interval can
        # cross cells whose old fractions have not yet been visited.
        self._alpha_before = wp.empty(
            tuple(size+2*cfg.grid.ng for size in cfg.grid.shape),
            dtype=wp.float32, device=self.device,
        )

    def apply(
        self,
        state: State3D,
        time: float,
        dt: float,
        *,
        gravity_increment: float = 0.0,
    ) -> WaveRelaxationResult:
        """Apply the source at a period-reduced time and report its accounting."""
        if state.cfg != self.cfg:
            raise ValueError("state configuration must match the relaxation zone")
        if dt <= 0.0:
            raise ValueError("relaxation dt must be positive")
        return self._blend(state, time, dt / self.timescale, gravity_increment, 0)

    @property
    def interface_representation(self) -> str:
        if self.cfg.wave_source_reconstruction == "plic_rays":
            return "geometric_plic_ray_displacement"
        return RELAXATION_INTERFACE_REPRESENTATION

    def _relax_plic_alpha(
        self, state: State3D, time: float, relaxation_rate: float, impose: int,
    ) -> None:
        """Apply the same volume-coordinate source on actual PLIC intervals.

        Sample the analytic target and smooth source taper at the same positive
        footprint quadrature points. Thus the column source is the integral of
        w(x)*(H_target(x,y)-H(x,y)), replacing the column-centre approximation.
        Existing dry gaps contract by 1-w on each ray; no volume repair follows.
        """
        from shoreloop.physics3d.interface import estimate_interface_normals
        from shoreloop.physics3d.relaxation_geometry import fit_offsets
        from shoreloop.physics3d.relaxation_rays import integrate_source_rays

        g = self.cfg.grid
        alpha = state.get_interior("alpha").astype(float)
        theta = state.get_geometry_interior("theta")
        source_columns = min(g.nx, math.ceil(self.width/g.dx))
        # The solid-adjacent normal fit writes only interface cells within one
        # cell of a cut cell. With the first column beyond the zone uncut as
        # well, no fitted cell lies in the source columns, so the fit (the
        # expensive part of the whole-grid estimate) is skipped exactly.
        if np.any(theta[:source_columns+1] != 1):
            raise ValueError("PLIC-ray wave forcing requires uncut source columns "
                             "and an uncut column beyond them")
        slices = state.grid.interior_slices("cell")
        physical = estimate_interface_normals(
            state, fit_cut_cells=False).normals.numpy()[slices].astype(float)
        # The source moves every retained partial amount, including tails
        # below the governing PLIC cutoff. Give only those tails the same
        # gradient estimator at zero cutoff. Keep the ordinary height-function
        # endpoint tolerance for supported resolved interfaces.
        tails = (alpha > 0) & (alpha < 1) & ~np.any(physical, axis=-1)
        if np.any(tails):
            complete = estimate_interface_normals(
                state, mixed_tolerance=0., fit_cut_cells=False).normals.numpy()[slices]
            physical[tails] = complete[tails]
        normals = physical*np.array((g.dx, g.dy, g.dz))
        length = np.abs(normals).sum(axis=-1)
        np.divide(normals, length[..., None], out=normals, where=length[..., None] > 0)
        offsets = fit_offsets(
            normals[:source_columns].reshape(-1, 3), alpha[:source_columns].ravel(),
        ).reshape(source_columns, g.ny, g.nz)
        parameters = dict(x0=g.x0, y0=g.y0, z0=g.z0, dx=g.dx, dy=g.dy, dz=g.dz,
            ny=g.ny, nz=g.nz, width=self.width, ramp=self.generation_ramp,
            level=self.still_water_level, time=time, rate=relaxation_rate, impose=impose)
        components = np.array([
            (c.kx, c.ky, c.omega, c.phase_offset, c.amplitude, c.second_order_elevation_amplitude)
            for c in self.inlet.components
        ])
        # Host FP64 footprints retain the original positive quadrature. CUDA
        # FP64 ray integrals/atomic sums precede the single FP32 state write;
        # their reduction order is measured against the host reference.
        after, _ = integrate_source_rays(
            alpha[:source_columns].reshape(-1, g.nz),
            normals[:source_columns].reshape(-1, g.nz, 3),
            offsets.reshape(-1, g.nz), parameters, components,
            order=self.cfg.wave_source_quadrature, device=self.cfg.device,
        )
        alpha[:source_columns] = after.reshape(source_columns, g.ny, g.nz)
        # Source volumes and endpoints used FP64; governing state remains FP32.
        state.set_interior("alpha", alpha.astype(np.float32))

    def impose(self, state: State3D, time: float) -> WaveRelaxationResult:
        """Set the incident field inside the zone with its spatial taper.

        This is an initial condition, not a source: it replaces the water and
        velocity the zone would otherwise have to build from rest against the
        offshore wall. The returned accounting is the change from the prior
        state. The tapered field is not discretely divergence-free in the
        liquid; project it once without body force before stepping.
        """
        if state.cfg != self.cfg:
            raise ValueError("state configuration must match the relaxation zone")
        return self._blend(state, time, 0.0, 0.0, 1)

    def _blend(
        self,
        state: State3D,
        time: float,
        relaxation_rate: float,
        gravity_increment: float,
        impose: int,
    ) -> WaveRelaxationResult:
        before = measure_vof(state)
        momentum_before = measure_liquid_momentum(state)
        g = self.cfg.grid
        reduced_time = math.remainder(time, self.inlet.period)
        common = [
            self._amplitude, self._second_elevation,
            self._omega, self._kx, self._ky, self._phase,
            len(self.inlet.components), reduced_time, g.x0, self.width,
            self.generation_ramp, relaxation_rate, self.still_water_level, g.x0, g.y0, g.z0,
            g.dx, g.dy, g.dz, g.nz, g.ng,
        ]
        if self.cfg.wave_source_reconstruction == "plic_rays":
            self._relax_plic_alpha(state, reduced_time, relaxation_rate, impose)
        else:
            wp.copy(self._alpha_before, state.alpha)
            wp.launch(
                _relax_alpha, dim=(g.nx, g.ny),
                inputs=[self._alpha_before, state.alpha, *common, impose],
                device=self.device,
            )
        for axis, name in enumerate(("u", "v", "w")):
            wp.launch(
                _relax_velocity,
                dim=state.grid.interior_shape(name),
                inputs=[state.field(name), state.alpha, self._amplitude,
                        self._second_elevation, self._second_potential, self._omega,
                        self._wave_number, self._kx, self._ky, self._phase,
                        len(self.inlet.components), axis, reduced_time,
                        g.x0, self.width, self.generation_ramp, relaxation_rate, self.inlet.depth,
                        self.still_water_level, g.x0, g.y0, g.z0,
                        g.dx, g.dy, g.dz, gravity_increment, g.nz, g.ng, impose],
                device=self.device,
            )
        fill_boundaries(state)
        after = measure_vof(state)
        momentum_after = measure_liquid_momentum(state)
        return WaveRelaxationResult(
            volume_before=before.volume,
            volume_after=after.volume,
            volume_source=after.volume - before.volume,
            momentum_source=tuple(
                final - initial
                for final, initial in zip(momentum_after, momentum_before, strict=True)
            ),
        )
