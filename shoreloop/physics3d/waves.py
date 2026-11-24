"""Exactly periodic finite-depth wave components for the 3-D inlet."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np


def finite_depth_wavenumber(omega: float, depth: float, gravity: float = 9.81) -> float:
    """Solve ``omega**2 = gravity * k * tanh(k * depth)`` for ``k``."""
    if omega <= 0.0 or depth <= 0.0 or gravity <= 0.0:
        raise ValueError("omega, depth, and gravity must be positive")

    def residual(k: float) -> float:
        return gravity * k * math.tanh(k * depth) - omega * omega

    lo = 0.0
    hi = max(omega * omega / gravity, omega / math.sqrt(gravity * depth))
    while residual(hi) < 0.0:
        hi *= 2.0

    for _ in range(64):
        mid = 0.5 * (lo + hi)
        if residual(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class PeriodicWaveComponent:
    """One periodic component tied to integer temporal and alongshore modes."""

    period: float
    depth: float
    alongshore_length: float
    amplitude: float
    temporal_mode: int = 1
    alongshore_mode: int = 0
    phase_offset: float = 0.0
    stokes_order: int = 1
    gravity: float = 9.81
    omega: float = field(init=False)
    k: float = field(init=False)
    kx: float = field(init=False)
    ky: float = field(init=False)

    def __post_init__(self) -> None:
        if self.period <= 0.0 or self.depth <= 0.0 or self.alongshore_length <= 0.0:
            raise ValueError("period, depth, and alongshore length must be positive")
        if self.gravity <= 0.0:
            raise ValueError("gravity must be positive")
        if isinstance(self.temporal_mode, bool) or not isinstance(self.temporal_mode, int) \
                or self.temporal_mode < 1:
            raise ValueError("temporal mode must be a positive integer")
        if isinstance(self.alongshore_mode, bool) \
                or not isinstance(self.alongshore_mode, int):
            raise ValueError("alongshore mode must be an integer")
        if self.stokes_order not in (1, 2):
            raise ValueError("Stokes order must be one or two")
        if not math.isfinite(self.amplitude) or not math.isfinite(self.phase_offset):
            raise ValueError("amplitude and phase must be finite")

        omega = 2.0 * math.pi * self.temporal_mode / self.period
        k = finite_depth_wavenumber(omega, self.depth, self.gravity)
        ky = 2.0 * math.pi * self.alongshore_mode / self.alongshore_length
        if abs(ky) >= k:
            raise ValueError(
                f"alongshore mode is evanescent: |ky|={abs(ky):.6g} >= k={k:.6g}",
            )

        object.__setattr__(self, "omega", omega)
        object.__setattr__(self, "k", k)
        object.__setattr__(self, "kx", math.sqrt(k * k - ky * ky))
        object.__setattr__(self, "ky", ky)

    @property
    def wavelength(self) -> float:
        return 2.0 * math.pi / self.k

    @property
    def phase_speed(self) -> float:
        return self.omega / self.k

    @property
    def steepness(self) -> float:
        return abs(self.amplitude) * self.k

    @property
    def second_order_elevation_amplitude(self) -> float:
        """Finite-depth Stokes coefficient multiplying ``cos(2 phase)``."""
        if self.stokes_order == 1:
            return 0.0
        tangent = math.tanh(self.k * self.depth)
        return (
            0.25 * self.k * self.amplitude**2
            * (3.0 - tangent * tangent) / tangent**3
        )

    @property
    def second_order_potential_amplitude(self) -> float:
        """Finite-depth Stokes coefficient multiplying ``sin(2 phase)``."""
        if self.stokes_order == 1:
            return 0.0
        sine = math.sinh(self.k * self.depth)
        return 3.0 * self.amplitude**2 * self.omega / (8.0 * sine**4)

    def phase(self, x: float, y: float, time: float) -> float:
        """Evaluate phase after reducing time by the base period in float64."""
        reduced_time = math.remainder(time, self.period)
        return (
            self.kx * x + self.ky * y
            - self.omega * reduced_time + self.phase_offset
        )

    def elevation(self, x: float, y: float, time: float) -> float:
        """Free-surface displacement from the still-water level."""
        phase = self.phase(x, y, time)
        return (
            self.amplitude * math.cos(phase)
            + self.second_order_elevation_amplitude * math.cos(2.0 * phase)
        )

    def velocity(
        self,
        x: float,
        y: float,
        z: float,
        time: float,
    ) -> tuple[float, float, float]:
        """Wave velocity at ``z``, measured upward from still water."""
        if z < -self.depth or z > self.elevation(x, y, time):
            return (0.0, 0.0, 0.0)
        return self._velocity(x, y, z, time)

    def _velocity(
        self,
        x: float,
        y: float,
        z: float,
        time: float,
    ) -> tuple[float, float, float]:
        phase = self.phase(x, y, time)
        denominator = math.sinh(self.k * self.depth)
        horizontal = (
            self.amplitude * self.omega
            * math.cosh(self.k * (z + self.depth)) / denominator
            * math.cos(phase)
        )
        vertical = (
            self.amplitude * self.omega
            * math.sinh(self.k * (z + self.depth)) / denominator
            * math.sin(phase)
        )
        second_potential = self.second_order_potential_amplitude
        second_horizontal = (
            2.0 * self.k * second_potential
            * math.cosh(2.0 * self.k * (z + self.depth))
            * math.cos(2.0 * phase)
        )
        second_vertical = (
            2.0 * self.k * second_potential
            * math.sinh(2.0 * self.k * (z + self.depth))
            * math.sin(2.0 * phase)
        )
        horizontal += second_horizontal
        vertical += second_vertical
        return (
            horizontal * self.kx / self.k,
            horizontal * self.ky / self.k,
            vertical,
        )

    def discharge(self, x: float, y: float, time: float) -> tuple[float, float]:
        """Horizontal discharge integrated from bed to mean surface."""
        magnitude = (
            self.amplitude * self.omega / self.k
            * math.cos(self.phase(x, y, time))
        )
        magnitude += (
            self.second_order_potential_amplitude
            * math.sinh(2.0 * self.k * self.depth)
            * math.cos(2.0 * self.phase(x, y, time))
        )
        return (magnitude * self.kx / self.k, magnitude * self.ky / self.k)


@dataclass(frozen=True)
class LinearWaveInlet:
    """A sum of compatible periodic components in mean-surface coordinates."""

    components: tuple[PeriodicWaveComponent, ...]

    def __post_init__(self) -> None:
        if not self.components:
            raise ValueError("a linear wave inlet needs at least one component")
        reference = self.components[0]
        for component in self.components[1:]:
            if (
                component.period != reference.period
                or component.depth != reference.depth
                or component.alongshore_length != reference.alongshore_length
                or component.gravity != reference.gravity
            ):
                raise ValueError("wave components must share period, depth, domain, and gravity")

    @property
    def period(self) -> float:
        return self.components[0].period

    @property
    def depth(self) -> float:
        return self.components[0].depth

    def elevation(self, x: float, y: float, time: float) -> float:
        return sum(component.elevation(x, y, time) for component in self.components)

    def velocity(
        self,
        x: float,
        y: float,
        z: float,
        time: float,
    ) -> tuple[float, float, float]:
        if z < -self.depth or z > self.elevation(x, y, time):
            return (0.0, 0.0, 0.0)
        velocities = [component._velocity(x, y, z, time) for component in self.components]
        return tuple(sum(values) for values in zip(*velocities, strict=True))

    def discharge(self, x: float, y: float, time: float) -> tuple[float, float]:
        discharges = [component.discharge(x, y, time) for component in self.components]
        return tuple(sum(values) for values in zip(*discharges, strict=True))

    def sampled_maximum_local_wave_height(
        self,
        x: float,
        *,
        alongshore_samples: int = 512,
        time_samples: int = 512,
    ) -> float:
        """Sample the largest local crest-to-trough height over one period."""
        if alongshore_samples < 2 or time_samples < 2:
            raise ValueError("wave-height sampling needs at least two samples per axis")
        alongshore = np.linspace(
            0.0,
            self.components[0].alongshore_length,
            alongshore_samples,
            endpoint=False,
            dtype=np.float64,
        )[None, :]
        times = np.linspace(
            0.0,
            self.period,
            time_samples,
            endpoint=False,
            dtype=np.float64,
        )[:, None]
        elevation = np.zeros((time_samples, alongshore_samples), dtype=np.float64)
        for component in self.components:
            phase = (
                component.kx * x
                + component.ky * alongshore
                - component.omega * times
                + component.phase_offset
            )
            elevation += component.amplitude * np.cos(phase)
            elevation += (
                component.second_order_elevation_amplitude * np.cos(2.0 * phase)
            )
        return float(np.max(np.ptp(elevation, axis=0)))


@dataclass(frozen=True)
class ReflectionMeasurement:
    """Incident and reflected complex surface amplitudes from two probes."""

    incident_amplitude: float
    reflected_amplitude: float
    reflection_coefficient: float


@dataclass(frozen=True)
class WaveModeFit:
    """Spatial harmonic fit that separates dispersion error from reflection."""

    wavenumber: float
    incident_amplitude: float
    reflected_amplitude: float
    reflection_coefficient: float
    relative_residual: float


def measure_two_probe_reflection(
    times: np.ndarray,
    elevation_a: np.ndarray,
    elevation_b: np.ndarray,
    x_a: float,
    x_b: float,
    component: PeriodicWaveComponent,
) -> ReflectionMeasurement:
    """Separate right- and left-running amplitudes at one known frequency."""
    times = np.asarray(times, dtype=np.float64)
    elevation_a = np.asarray(elevation_a, dtype=np.float64)
    elevation_b = np.asarray(elevation_b, dtype=np.float64)
    if times.shape != elevation_a.shape or times.shape != elevation_b.shape:
        raise ValueError("times and both probe series must have the same shape")

    temporal_basis = np.exp(1j * component.omega * times)
    measured = 2.0 * np.array([
        np.mean(elevation_a * temporal_basis),
        np.mean(elevation_b * temporal_basis),
    ])
    propagation = np.array([
        [np.exp(1j * component.kx * x_a), np.exp(-1j * component.kx * x_a)],
        [np.exp(1j * component.kx * x_b), np.exp(-1j * component.kx * x_b)],
    ])
    incident, reflected = np.linalg.solve(propagation, measured)
    incident_amplitude = float(abs(incident))
    reflected_amplitude = float(abs(reflected))
    coefficient = (
        reflected_amplitude / incident_amplitude
        if incident_amplitude > 0.0 else math.inf
    )
    return ReflectionMeasurement(
        incident_amplitude=incident_amplitude,
        reflected_amplitude=reflected_amplitude,
        reflection_coefficient=coefficient,
    )


def fit_probe_wave_modes(
    times: np.ndarray,
    elevations: np.ndarray,
    probe_positions: np.ndarray,
    component: PeriodicWaveComponent,
) -> WaveModeFit:
    """Fit ``Ai exp(ikx) + Ar exp(-ikx)`` and the propagating ``k``."""
    times = np.asarray(times, dtype=np.float64)
    elevations = np.asarray(elevations, dtype=np.float64)
    positions = np.asarray(probe_positions, dtype=np.float64)
    if elevations.shape != (times.size, positions.size):
        raise ValueError("elevations must have shape (times, probe positions)")
    temporal_basis = np.exp(1j * component.omega * times)
    measured = 2.0 * np.mean(elevations * temporal_basis[:, None], axis=0)

    centre = component.kx
    half_width = 0.5 * component.kx
    best_k = centre
    best_coefficients = np.zeros(2, dtype=np.complex128)
    best_residual = np.inf
    for _ in range(3):
        candidates = np.linspace(centre - half_width, centre + half_width, 401)
        for wave_number in candidates:
            modes = np.column_stack((
                np.exp(1j * wave_number * positions),
                np.exp(-1j * wave_number * positions),
            ))
            coefficients, *_ = np.linalg.lstsq(modes, measured, rcond=None)
            residual = float(np.linalg.norm(modes @ coefficients - measured))
            if residual < best_residual:
                best_residual = residual
                best_k = float(wave_number)
                best_coefficients = coefficients
        centre = best_k
        half_width *= 0.05

    incident = float(abs(best_coefficients[0]))
    reflected = float(abs(best_coefficients[1]))
    scale = float(np.linalg.norm(measured))
    return WaveModeFit(
        wavenumber=best_k,
        incident_amplitude=incident,
        reflected_amplitude=reflected,
        reflection_coefficient=reflected / incident if incident > 0.0 else math.inf,
        relative_residual=best_residual / scale if scale > 0.0 else 0.0,
    )
