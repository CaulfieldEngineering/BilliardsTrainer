"""Session video encoding.

OpenCV's VideoWriter only offers ancient codecs (mp4v = MPEG-4 part 2) at a
low fixed bitrate — recordings looked like security-camera footage. When
ffmpeg is available (it ships with the audio feature), frames are piped to a
hardware H.264 encoder (VideoToolbox on Apple silicon) at a proper bitrate
instead; otherwise we fall back to the old writer so recording never breaks.

Also detects the active image box: the T3i's live-view HDMI feed fills only
~63% of the 1080p frame (the picture floats in black bars). Recording just the
content box makes the clip all image, no letterbox.
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time

import numpy as np

from .audio import find_ffmpeg

log = logging.getLogger("capture.videowriter")


def content_box(frame: np.ndarray, thresh: int = 12
                ) -> tuple[int, int, int, int] | None:
    """(x0, y0, x1, y1) of the non-black active image, or None to keep all.

    Conservative: only crops when the borders are genuinely dark AND the
    content still covers most of the frame (>=40%), so a dim room never gets
    cropped into oblivion. Coordinates are padded and snapped to even numbers
    (yuv420 requires even dimensions).
    """
    import cv2
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray > thresh)
    if len(xs) == 0:
        return None
    h, w = gray.shape
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    area = (x1 - x0 + 1) * (y1 - y0 + 1)
    if area < 0.40 * w * h or area > 0.98 * w * h:
        return None
    pad = 4
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(w - 1, x1 + pad), min(h - 1, y1 + pad)
    # even-align (shrink inward so we never exceed the frame)
    if (x1 - x0 + 1) % 2:
        x1 -= 1
    if (y1 - y0 + 1) % 2:
        y1 -= 1
    return (x0, y0, x1, y1)


class FfmpegWriter:
    """cv2.VideoWriter-compatible surface over an ffmpeg H.264 pipe."""

    def __init__(self, path: str, fps: float, size: tuple[int, int],
                 bitrate: str = "10M"):
        self._size = size  # (w, h)
        w, h = size
        ffmpeg = find_ffmpeg()
        if ffmpeg is None:
            raise RuntimeError("ffmpeg not available")
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
               "-r", f"{fps:.3f}", "-i", "-"]
        if w < 1000:
            # The T3i's live feed is ~760 real columns AND noisy. Recipe:
            # light temporal denoise FIRST (noise caps how hard you can
            # sharpen — it amplifies into grain), Lanczos up to a standard
            # 1080 width, then a stronger unsharp than noise would otherwise
            # allow. No invented detail, but markedly crisper on phones.
            cmd += ["-vf", ("hqdn3d=1.5:1.5:6:6,"
                            "scale=1080:-2:flags=lanczos,"
                            "unsharp=5:5:0.6:5:5:0.0")]
            bitrate = "14M"   # more pixels after the upscale deserve more bits
        cmd += ["-c:v", "h264_videotoolbox", "-b:v", bitrate,
                # tag bt709 explicitly — untagged video renders washed-out on
                # iPhones, which reads as further softness
                "-color_primaries", "bt709", "-color_trc", "bt709",
                "-colorspace", "bt709",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                "-y", path]
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        log.info("H.264 recording via ffmpeg/videotoolbox %sx%s @ %s", w, h, bitrate)
        # PACED writing (the OBS model): capture rate varies with load, but the
        # encoder is fed at EXACTLY the declared fps on a wall-clock schedule —
        # the freshest frame is duplicated when capture lags and extras are
        # dropped when it bursts. Output is true CFR, so the audio track lines
        # up at every instant (the old declared-vs-actual retiming only matched
        # the endpoints; the middle drifted seconds under load swings).
        self._latest: bytes | None = None
        self._emitted = 0
        self._lock = threading.Lock()
        self._paused = False
        self._stop = threading.Event()
        self._interval = 1.0 / max(1.0, fps)
        self._pacer = threading.Thread(target=self._pace, daemon=True,
                                       name="rec-pacer")
        self._pacer.start()

    def _pace(self) -> None:
        next_t = None
        while not self._stop.is_set():
            if next_t is None:
                time.sleep(0.005)
                with self._lock:
                    ready = self._latest is not None and not self._paused
                if ready:
                    next_t = time.monotonic()
                continue
            now = time.monotonic()
            if now < next_t:
                time.sleep(min(self._interval, next_t - now))
                continue
            with self._lock:
                buf, paused = self._latest, self._paused
            if paused:
                next_t = None      # pause stops the clock; resume restarts it
                continue
            if buf is not None and self._proc.poll() is None:
                try:
                    self._proc.stdin.write(buf)
                    self._emitted += 1
                except (BrokenPipeError, OSError):
                    log.warning("recording encoder pipe closed early")
                    return
            next_t += self._interval
            if next_t < time.monotonic() - 1.0:   # fell far behind; resync
                next_t = time.monotonic()

    def isOpened(self) -> bool:  # noqa: N802 - cv2 surface
        return self._proc.poll() is None

    def write(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        if (w, h) != self._size:
            return  # size changed mid-recording; drop rather than corrupt
        with self._lock:
            self._latest = frame.tobytes()

    def pause(self, on: bool) -> None:
        with self._lock:
            self._paused = on
            if on:
                self._latest = None   # never duplicate a stale frame on resume

    def release(self) -> None:
        self._stop.set()
        try:
            self._pacer.join(timeout=2)
        except RuntimeError:
            pass
        # A recording must never be EMPTY: if the pacer never got a wall-clock
        # tick (ultra-short recording), flush the last frame once.
        with self._lock:
            buf = self._latest
        if self._emitted == 0 and buf is not None and self._proc.poll() is None:
            try:
                self._proc.stdin.write(buf)
            except (BrokenPipeError, OSError):
                pass
        try:
            self._proc.stdin.close()
            self._proc.wait(timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            self._proc.kill()


def open_writer(path: str, fps: float, size: tuple[int, int]):
    """Best available session writer: H.264 via ffmpeg, else cv2 mp4v."""
    try:
        return FfmpegWriter(path, fps, size)
    except (RuntimeError, OSError) as exc:
        log.warning("H.264 writer unavailable (%s) — falling back to mp4v", exc)
        import cv2
        return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
