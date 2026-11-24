"""Offscreen frame output: PNG for inspection, MP4 via NVENC for playback.

Offscreen is the primary output path (docs/headless-setup.md): it needs no X
server, it is how the deliverable video gets made, and it is the only way to
judge the renderer without a lossy codec in the way. Hence two sinks -- a
lossless one for looking closely, and a compressed one for watching motion.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np


class VideoWriter:
    """Pipe raw RGB frames to ffmpeg. Uses NVENC when available.

    `crf`/`cq` is quality, lower is better. 0 is lossless for hevc_nvenc but
    large; for judging the *renderer* prefer PNG frames instead of trusting any
    lossy setting.
    """

    def __init__(self, path: str | Path, width: int, height: int, fps: float = 30.0,
                 cq: int = 18, codec: str | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found")
        if width % 2 or height % 2:
            # yuv420p subsamples chroma 2x2; odd dimensions make ffmpeg die
            # mid-stream with a broken pipe rather than a useful error.
            raise ValueError(f"yuv420p needs even dimensions, got {width}x{height}")
        if codec is None:
            enc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                                 capture_output=True, text=True).stdout
            codec = "hevc_nvenc" if "hevc_nvenc" in enc else "libx264"
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{width}x{height}", "-r", f"{fps}", "-i", "-",
               "-an", "-c:v", codec]
        if "nvenc" in codec:
            cmd += ["-preset", "p5", "-rc", "vbr", "-cq", str(cq)]
        else:
            cmd += ["-preset", "slow", "-crf", str(cq)]
        if "hevc" in codec or "h265" in codec:
            # Apple platforms require the 'hvc1' sample-entry tag for HEVC in
            # MP4. ffmpeg defaults to 'hev1', which QuickTime, Preview and
            # Photos all refuse to open -- the file looks corrupt rather than
            # unsupported, which is a confusing way to fail.
            cmd += ["-tag:v", "hvc1"]
        # yuv420p for universal playback; note this is 4:2:0, so fine chroma
        # detail is lost -- another reason to inspect PNGs, not this.
        #
        # Pin the RGB->YUV matrix to BT.709 and say so in the stream metadata.
        # Left alone, swscale converts with BT.601 coefficients and writes no
        # tag, so any player that assumes 709 (typical at HD frame sizes)
        # shows a hue shift relative to the PNGs.
        cmd += ["-vf", "scale=out_color_matrix=bt709",
                "-colorspace", "bt709", "-color_primaries", "bt709",
                "-color_trc", "bt709",
                "-pix_fmt", "yuv420p", str(self.path)]
        self.codec = codec
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
        self.n = 0

    def add(self, rgb: np.ndarray) -> None:
        self.proc.stdin.write(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())
        self.n += 1

    def close(self) -> None:
        self.proc.stdin.close()
        err = self.proc.stderr.read().decode(errors="replace")
        self.proc.stderr.close()
        rc = self.proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg failed ({rc}): {err[-800:]}")

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def write_png(path: str | Path, rgb: np.ndarray) -> None:
    """Lossless single frame, for judging the image rather than the motion."""
    import matplotlib.image as mpimg
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    mpimg.imsave(str(p), rgb)
