"""Oblique pinhole camera plus the image-formation half of theta_camera.

`theta_camera` deliberately includes exposure, white balance and the transfer
function, not only pose (docs/decisions.md, "Stack"): the reference is ordinary
phone video, and fitting it without a camera response model is hopeless.

Conventions: world z is up, angles in degrees at the interface, radians
internally. `yaw = 0` looks along +x, `yaw = 180` along -x (offshore, for the
default beach whose berm is at high x). `pitch > 0` tilts below the horizontal.
Pixel (0, 0) is the top-left corner of the image.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Camera:
    """Pose from docs/reference-video.md: ~1.0-1.5 m above the stones, tilted
    30-40 degrees below horizontal, vertical FOV 40-50 degrees, waterline about
    2/3 down the frame. Weakly constrained -- pick a plausible value now, fit
    later (plan.md section 14 makes camera the first thing fitted)."""

    pos: tuple[float, float, float] = (8.6, 3.0, 1.5)
    yaw_deg: float = 180.0
    pitch_deg: float = 35.0
    vfov_deg: float = 45.0
    width: int = 910
    height: int = 512

    # image formation
    exposure: float = 1.0                       # linear gain before encoding
    wb: tuple[float, float, float] = (1.0, 1.0, 1.0)  # per-channel gains

    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(forward, right, up) unit vectors of the camera frame."""
        yaw = np.deg2rad(self.yaw_deg)
        pitch = np.deg2rad(self.pitch_deg)
        fwd = np.array([np.cos(pitch) * np.cos(yaw),
                        np.cos(pitch) * np.sin(yaw),
                        -np.sin(pitch)])
        right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        return fwd, right, up

    @property
    def tan_half_v(self) -> float:
        return float(np.tan(0.5 * np.deg2rad(self.vfov_deg)))

    @property
    def tan_half_h(self) -> float:
        return self.tan_half_v * self.width / self.height

    def ray_dirs(self) -> np.ndarray:
        """(height, width, 3) unit ray directions, host side.

        The renderer regenerates these in-kernel; this exists for tests and
        for analytic cross-checks of the kernel's geometry.
        """
        fwd, right, up = self.basis()
        u = (np.arange(self.width) + 0.5) / self.width * 2.0 - 1.0
        v = 1.0 - 2.0 * (np.arange(self.height) + 0.5) / self.height
        U, V = np.meshgrid(u * self.tan_half_h, v * self.tan_half_v)
        d = (fwd[None, None, :] + U[..., None] * right[None, None, :]
             + V[..., None] * up[None, None, :])
        return d / np.linalg.norm(d, axis=-1, keepdims=True)


def srgb_encode(x: np.ndarray) -> np.ndarray:
    """The sRGB transfer function. NOT a 2.2 power: the linear toe matters in
    the dark water rows where the reference ramp bottoms out."""
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x,
                    1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def srgb_decode(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4))


def tonemap(linear: np.ndarray, cam: Camera) -> np.ndarray:
    """Linear HWC radiance -> display uint8, via exposure, WB and sRGB."""
    out = linear * cam.exposure * np.asarray(cam.wb, dtype=np.float32)
    return (srgb_encode(out) * 255.0 + 0.5).astype(np.uint8)
