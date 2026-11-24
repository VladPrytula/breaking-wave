"""Device selection. Prefers CUDA, falls back to CPU so tests can still run."""

from __future__ import annotations

import time

import warp as wp


_PRIMARY_CONTEXT_ERROR = "Failed to acquire primary context"
_CONTEXT_RETRY_DELAYS = (0.25, 0.5, 1.0, 2.0)


def prepare_device(device: str) -> wp.Device:
    """Return a ready device, bridging the DGX CUDA context hand-off window."""
    wp.init()
    selected = wp.get_device(device)
    if not selected.is_cuda:
        return selected

    for delay in _CONTEXT_RETRY_DELAYS:
        try:
            wp.set_device(selected)
            return selected
        except RuntimeError as exc:
            if _PRIMARY_CONTEXT_ERROR not in str(exc):
                raise
            time.sleep(delay)

    wp.set_device(selected)
    return selected


def default_device() -> str:
    wp.init()
    device = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
    prepare_device(device)
    return device


DEVICE = default_device()
