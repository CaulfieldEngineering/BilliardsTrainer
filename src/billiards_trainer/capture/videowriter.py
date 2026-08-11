"""Session video encoding.

OpenCV's VideoWriter only offers ancient codecs (mp4v = MPEG-4 part 2) at a
low fixed bitrate — recordings looked like security-camera footage. When
ffmpeg is available (it ships with the audio feature), frames are piped to a
hardware H.264 encoder at a proper bitrate instead; otherwise we fall back to
the old writer so recording never breaks.

The encoder is PROBED, not assumed (see ``pick_h264_encoder``): AMF on the
Radeon rig PC, VideoToolbox on Apple, NVENC/QSV elsewhere, libx264 as the
always-present tail.

Also detects the active image box: the T3i's live-view HDMI feed fills only
~63% of the 1080p frame (the picture floats in black bars). Recording just the
content box makes the clip all image, no letterbox.
"""

from __future__ import annotations

import collections
import logging
import subprocess
import sys
import threading
import time
from functools import lru_cache

import numpy as np

from .audio import NO_WINDOW, find_ffmpeg

log = logging.getLogger("capture.videowriter")

# H.264 encoder preference, per platform, best first. Hardware encoders come
# first because they cost almost no CPU — and the same box is simultaneously
# running two inference passes per frame, so a software encoder competes
# directly with detection. There is always a software tail (libx264) so the
# list can never come up empty.
_H264_PREFERENCE = {
    "darwin": ("h264_videotoolbox", "libx264"),
    # AMF is AMD's encoder (the Radeon 780M in the rig PC); the others cover
    # NVIDIA / Intel / any-GPU-via-MediaFoundation boxes.
    "win32": ("h264_amf", "h264_nvenc", "h264_qsv", "h264_mf", "libx264"),
}
_H264_FALLBACK_ORDER = ("h264_nvenc", "h264_vaapi", "libx264")

# Extra flags an encoder needs to behave in a REAL-TIME pipe.
_ENCODER_ARGS = {
    # x264's default preset ("medium") cannot keep up with a live feed while
    # the GPU is busy detecting; veryfast holds the rate at our bitrate.
    "libx264": ("-preset", "veryfast"),
}


@lru_cache(maxsize=4)
def _available_encoders(ffmpeg: str) -> frozenset[str]:
    """Encoder names this ffmpeg build actually has (cached — it forks a proc)."""
    try:
        res = subprocess.run([ffmpeg, "-hide_banner", "-encoders"],
                             capture_output=True, text=True, timeout=20,
                             creationflags=NO_WINDOW)
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("could not enumerate ffmpeg encoders: %s", exc)
        return frozenset()
    names = set()
    for line in res.stdout.splitlines():
        # rows look like " V....D h264_amf   AMD AMF H.264 Encoder (codec h264)"
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("V"):
            names.add(parts[1])
    return frozenset(names)


def pick_h264_encoder(ffmpeg: str) -> tuple[str, tuple[str, ...]]:
    """(encoder name, extra flags) — the best H.264 encoder this build offers.

    Probing beats hardcoding: h264_videotoolbox exists only on Apple builds, and
    asking for a missing encoder makes ffmpeg exit immediately, which used to
    produce a silently empty recording.
    """
    available = _available_encoders(ffmpeg)
    for name in _H264_PREFERENCE.get(sys.platform, _H264_FALLBACK_ORDER):
        if name in available:
            return name, _ENCODER_ARGS.get(name, ())
    return "libx264", _ENCODER_ARGS["libx264"]


def quality_args(encoder: str, qp: int) -> list[str]:
    """Rate-control flags targeting a QUALITY level rather than a bitrate.

    A fixed bitrate is badly wrong for this footage: a pool table is motionless
    most of the time, and 14Mbps CBR spent the same bits on a still table as on
    a break. Measured on a real 60s segment (924x1630@30), against the 14Mbps
    original at 7.31 GB/hour:

        h264 QP16   3.09 GB/hr   PSNR 48.2   SSIM 0.990
        h264 QP20   0.97 GB/hr   PSNR 46.5   SSIM 0.986   <- default
        h264 QP24   0.35 GB/hr   PSNR 45.1   SSIM 0.982
        hevc QP20   0.66 GB/hr   PSNR 46.7   SSIM 0.986

    H.264 over HEVC despite HEVC being ~30% smaller at equal quality: these get
    watched back in Dropbox, whose preview is dependable for H.264 and patchy
    for HEVC. AV1 was rejected outright — av1_amf pads to its own alignment
    (924x1630 came back as 960x1632), which would shift the geometry on replay.
    """
    if encoder.endswith("_amf"):
        return ["-rc", "cqp", "-qp_i", str(qp), "-qp_p", str(qp), "-quality", "0"]
    if encoder.endswith("_nvenc"):
        return ["-rc", "constqp", "-qp", str(qp), "-preset", "p5"]
    if encoder.endswith("_qsv"):
        return ["-global_quality", str(qp)]
    if encoder.endswith("videotoolbox"):
        # VideoToolbox has no QP mode; -q:v is 0-100, roughly inverse of QP
        return ["-q:v", str(max(1, min(100, 100 - qp * 2)))]
    return ["-crf", str(qp)]        # libx264 and friends


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


class _CfrRetimer:
    """VFR→CFR retiming driven by capture timestamps, not a wall-clock sampler.

    Each arriving frame is assigned the output slot its timestamp falls in
    (``slot = round((ts - t0) * fps)``) and ``advance`` returns how many copies
    to write: 0 when the slot is already filled (drop), 1 in the steady state,
    >1 to fill a real capture gap with duplicates.

    Why this and not a fixed-rate pacer: a pacer samples "the latest frame" at
    rigid wall-clock instants, so ANY drift between the camera's frame clock
    and the OS timer — or arrival jitter comparable to the frame period —
    periodically lands as a duplicated or dropped frame, which plays back as a
    barely-perceptible stutter. Slot assignment from the frame's own timestamp
    absorbs jitter up to ~half a slot and never beats against a second clock,
    while output remains exactly ``fps`` frames per wall-clock second, so the
    audio track still lines up at every instant.
    """

    # A capture gap longer than this many seconds is not bridged with frozen
    # duplicates; the timeline is resynced instead (matches the old pacer's
    # fell-far-behind behaviour, and bounds a burst into the encoder pipe).
    MAX_GAP_S = 2.0
    # Slow phase servo: each frame nudges the epoch toward centring arrivals
    # mid-slot. Absorbs the first frame's arbitrary phase and tracks slow
    # camera-clock skew, so neither ever parks arrivals on a slot boundary
    # (where jitter would flip frames between slots = dup/drop pairs).
    ALPHA = 0.05
    # Host-clock drift clamp: video duration is kept within this many slots of
    # real elapsed time, so tracking the camera clock can never walk the audio
    # out of sync — a genuinely fast/slow camera costs a rare single-frame
    # slip instead of accumulating A/V drift.
    DRIFT_SLOTS = 3

    def __init__(self, fps: float):
        self.fps = float(fps)
        self.t0: float | None = None
        self._start: float | None = None
        self.emitted = 0

    def advance(self, ts: float) -> int:
        """Number of copies of the frame stamped ``ts`` to emit (0 = drop)."""
        if self.t0 is None:
            self.t0 = self._start = ts
            self.emitted = 1
            return 1
        pos = (ts - self.t0) * self.fps
        slot = int(round(pos))
        self.t0 += (pos - slot) * self.ALPHA / self.fps   # phase-centre arrivals
        need = slot + 1 - self.emitted
        drift = self.emitted / self.fps - (ts - self._start)  # video vs real time
        if need <= 0:
            return 0
        if drift > self.DRIFT_SLOTS / self.fps:
            need -= 1              # video ahead of real time: skip one slot
        elif drift < -self.DRIFT_SLOTS / self.fps:
            need += 1              # video behind real time: emit one extra
        cap = max(1, int(self.MAX_GAP_S * self.fps))
        if need > cap:
            self.t0 += (need - cap) / self.fps  # drop the dead time instead
            need = cap
        if need <= 0:
            return 0
        self.emitted += need
        return need

    def shift(self, dt: float) -> None:
        """Move the epoch forward by ``dt`` seconds (a paused stretch)."""
        if self.t0 is not None:
            self.t0 += dt
        if self._start is not None:
            self._start += dt


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
            vf = ["hqdn3d=1.5:1.5:6:6",
                  "scale=1080:-2:flags=lanczos",
                  "unsharp=5:5:0.6:5:5:0.0"]
            bitrate = "14M"   # more pixels after the upscale deserve more bits
        else:
            # HD feed (the forced-1080i era): no upscale needed, but the source
            # is still noisy (temporal sigma ~1.4-2.6 on static felt) and soft
            # at ball scale — which is what makes the 9's yellow stripe hard to
            # separate from its white body (measured: the COLOURS are already
            # 164 saturation apart, so the limit is spatial detail, not colour).
            # A light denoise + mild unsharp measured +35% frame detail
            # (laplacian 48.5 -> 65.4) with no visible processing artefacts.
            vf = ["hqdn3d=1:1:4:4", "unsharp=5:5:0.3:5:5:0.0"]
            bitrate = "14M"   # noise eats bits; give the encoder headroom
        # Force EVEN dimensions, always. yuv420p subsamples chroma 2x2, so an
        # odd width/height is invalid H.264 — and AMD's AMF encoder rejects it
        # outright ("SubmitInput() failed with error 29") leaving a 0-byte file,
        # where VideoToolbox had quietly tolerated it. The active-picture box
        # is odd often enough to matter (1633x928 is a real measured feed), and
        # this costs at most one row/column.
        vf.append("crop=trunc(iw/2)*2:trunc(ih/2)*2")
        cmd += ["-vf", ",".join(vf)]
        encoder, enc_args = pick_h264_encoder(ffmpeg)
        cmd += ["-c:v", encoder, *enc_args, "-b:v", bitrate,
                # tag bt709 explicitly — untagged video renders washed-out on
                # iPhones, which reads as further softness
                "-color_primaries", "bt709", "-color_trc", "bt709",
                "-colorspace", "bt709",
                "-pix_fmt", "yuv420p",
                # FRAGMENTED mp4 for the in-progress file: this Mac panics
                # (SSD hardware fault) often enough that "the app died mid
                # recording" is a normal event, and a plain mp4 with no moov
                # atom is an unplayable brick. Fragments are watchable as far
                # as they got. The finalizing mux re-containers with faststart.
                "-movflags", "+frag_keyframe+empty_moov+default_base_moof",
                "-y", path]
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            creationflags=NO_WINDOW)
        # Keep the tail of ffmpeg's stderr. This used to be DEVNULL, which meant
        # a refused encoder produced an empty file and NO message anywhere —
        # the failure mode that hid h264_videotoolbox being Apple-only.
        self._errlog: collections.deque[str] = collections.deque(maxlen=20)
        self._errpump = threading.Thread(target=self._drain_stderr, daemon=True,
                                         name="rec-stderr")
        self._errpump.start()
        log.info("H.264 recording via ffmpeg/%s %sx%s @ %s", encoder, w, h, bitrate)
        # ARRIVAL-DRIVEN CFR (see _CfrRetimer): each captured frame is written
        # into the output slot its own timestamp lands in. Output is still
        # exactly `fps` frames per second of wall time (audio stays aligned at
        # every instant), but emission is driven by frame arrivals — a steady
        # camera maps 1:1 with ZERO dup/drop. The previous rigid wall-clock
        # pacer sampled 'latest frame' at fixed instants, so any drift between
        # the camera clock and the OS timer (or arrival jitter under load)
        # showed up as periodic duplicate/dropped frames — the recorded
        # micro-stutter Joe kept seeing.
        self._retimer = _CfrRetimer(max(1.0, fps))
        self._last_buf: bytes | None = None
        self._lock = threading.Lock()
        self._paused = False
        self._pause_t: float | None = None

    def _drain_stderr(self) -> None:
        """Consume ffmpeg's stderr so the pipe can't fill and wedge the encoder,
        keeping the last few lines to explain a failure."""
        stream = self._proc.stderr
        if stream is None:
            return
        try:
            for raw in stream:
                line = raw.decode("utf-8", "replace").strip()
                if line:
                    self._errlog.append(line)
        except (OSError, ValueError):
            pass

    def isOpened(self) -> bool:  # noqa: N802 - cv2 surface
        return self._proc.poll() is None

    def write(self, frame: np.ndarray, ts: float | None = None) -> None:
        """Write one captured frame, retimed into CFR output slots.

        ``ts`` is the frame's capture timestamp (time.monotonic()); omitted, the
        arrival time is used. Runs on the recording worker thread — the stdin
        write may block briefly on an encoder stall, which is why this must
        never be called from the camera grab thread directly.
        """
        h, w = frame.shape[:2]
        if (w, h) != self._size:
            return  # size changed mid-recording; drop rather than corrupt
        if ts is None:
            ts = time.monotonic()
        with self._lock:
            if self._paused:
                return
            n = self._retimer.advance(ts)
        if n <= 0:
            return
        buf = frame.tobytes()
        self._last_buf = buf
        if self._proc.poll() is not None:
            return
        try:
            for _ in range(n):
                self._proc.stdin.write(buf)
        except (BrokenPipeError, OSError):
            log.warning("recording encoder pipe closed early")

    def pause(self, on: bool) -> None:
        with self._lock:
            if on and not self._paused:
                self._pause_t = time.monotonic()
            elif not on and self._paused and self._pause_t is not None:
                # resume: the paused stretch never existed on the output clock
                self._retimer.shift(time.monotonic() - self._pause_t)
                self._pause_t = None
            self._paused = on

    def release(self) -> None:
        # A recording must never be EMPTY: an ultra-short recording whose only
        # frame landed in slot 0 already wrote it, but if nothing was ever
        # emitted (e.g. paused throughout), flush the last frame once.
        if self._retimer.emitted == 0 and self._last_buf is not None \
                and self._proc.poll() is None:
            try:
                self._proc.stdin.write(self._last_buf)
            except (BrokenPipeError, OSError):
                pass
        try:
            self._proc.stdin.close()
            self._proc.wait(timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            self._proc.kill()
        self._errpump.join(timeout=2)
        if self._proc.returncode:
            log.error("recording encoder exited %s: %s", self._proc.returncode,
                      " | ".join(self._errlog) or "(no output)")


def open_writer(path: str, fps: float, size: tuple[int, int]):
    """Best available session writer: H.264 via ffmpeg, else cv2 mp4v."""
    try:
        return FfmpegWriter(path, fps, size)
    except (RuntimeError, OSError) as exc:
        log.warning("H.264 writer unavailable (%s) — falling back to mp4v", exc)
        import cv2
        return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
