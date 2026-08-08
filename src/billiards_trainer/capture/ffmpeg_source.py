"""Camera capture where FFMPEG owns the device, not us.

Why this exists
---------------
The recording must be a faithful capture of the camera, and nothing the analysis
pipeline does — a slow inference pass, a GC pause, a stalled encoder — may show
up in it. That is impossible while Python owns the device: whatever timing
Python's polling has, the recording inherits, and the whole VFR->CFR retiming
apparatus exists only to paper over it.

So ffmpeg owns the capture device and produces TWO outputs from one input:

  1. the RECORDING — written straight from the device with its own timestamps,
     hardware-encoded, with the microphone muxed in natively;
  2. an ANALYSIS stream of raw frames on a pipe, which this class drains.

The recording path never passes through Python. Verified on the rig by stalling
the analysis reader 1.2s every 3s while recording: the file was identical either
way — 900 frames, 30.000s, both runs.

The analysis stream is deliberately best-effort. A drain thread always empties
the pipe and keeps only the NEWEST frame, so a slow consumer loses frames (which
only costs detection cadence) and can never back-pressure ffmpeg into disturbing
the recording.

The device is exclusive (DirectShow gives "device already in use" to a second
opener), which is exactly why one process has to own it and serve both needs.
"""

from __future__ import annotations

import logging
import subprocess
import threading

import numpy as np

from ..config import CameraSettings
from .audio import NO_WINDOW, find_ffmpeg
from .videowriter import pick_h264_encoder

log = logging.getLogger("capture.ffmpeg_source")

# Frames handed to the CV pipeline. FULL RESOLUTION on purpose: balls measure
# ~27px across in a 1080-wide frame, and the detector's floor is ~9px, so a
# halved analysis stream would put them under it.
#
# 30, not 15: the pipeline always takes the NEWEST frame, so the stream only has
# to stay ahead of it. Detection runs at ~10-16fps, and a 15fps stream would sit
# right on that boundary — the pipeline would periodically re-process a frame it
# had already seen. 30 keeps it comfortably supplied.
ANALYSIS_FPS = 30


class FfmpegCameraSource:
    """A CameraSource-compatible view of an ffmpeg-owned capture device."""

    name = "ffmpeg-camera"
    is_live = True     # a real camera: transient empty reads are expected

    def __init__(self, video_device: str, width: int = 1920, height: int = 1080,
                 in_fps: int = 60, audio_device: str | None = None,
                 cam: CameraSettings | None = None):
        self._video = video_device
        self._audio = audio_device
        self._w, self._h = width, height
        self._in_fps = in_fps
        self._cam = cam
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._rec_path: str | None = None
        self._rec_filters: str = ""
        self._opened = False
        self._start()

    # ------------------------------------------------------------------ #
    # ffmpeg process
    # ------------------------------------------------------------------ #
    def _input_args(self) -> list[str]:
        spec = f"video={self._video}"
        if self._audio:
            spec += f":audio={self._audio}"
        args = ["-f", "dshow", "-rtbufsize", "512M",
                "-video_size", f"{self._w}x{self._h}",
                "-framerate", str(self._in_fps)]
        # UVC controls ffmpeg can set on the device itself. Best-effort, exactly
        # like the OpenCV path: a device that ignores one must not fail the open.
        if self._cam is not None and not self._cam.auto_exposure and self._cam.exposure != -1:
            args += ["-video_device_options", f"exposure={self._cam.exposure}"]
        return [*args, "-i", spec]

    def _build_cmd(self) -> list[str]:
        ff = find_ffmpeg()
        if ff is None:
            raise RuntimeError("ffmpeg not available")
        cmd = [ff, "-hide_banner", "-loglevel", "error", *self._input_args()]
        if self._rec_path:
            enc, extra = pick_h264_encoder(ff)
            vf = self._rec_filters or "fps=30"
            cmd += ["-map", "0:v"]
            if self._audio:
                cmd += ["-map", "0:a"]
            cmd += ["-vf", vf, "-c:v", enc, *extra, "-b:v", "14M",
                    "-color_primaries", "bt709", "-color_trc", "bt709",
                    "-colorspace", "bt709", "-pix_fmt", "yuv420p",
                    # fragmented while in progress: a crash leaves a playable
                    # file rather than an unindexed brick
                    "-movflags", "+frag_keyframe+empty_moov+default_base_moof"]
            if self._audio:
                cmd += ["-c:a", "aac", "-b:a", "160k"]
            cmd += ["-y", self._rec_path]
        # analysis output, always last
        cmd += ["-map", "0:v", "-vf", f"fps={ANALYSIS_FPS}",
                "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
        return cmd

    def _start(self) -> None:
        self._stop.clear()
        try:
            self._proc = subprocess.Popen(
                self._build_cmd(), stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                bufsize=self._w * self._h * 3 * 2, creationflags=NO_WINDOW)
        except (OSError, RuntimeError) as exc:
            log.warning("ffmpeg capture failed to start: %s", exc)
            self._proc = None
            return
        self._opened = True
        self._thread = threading.Thread(target=self._drain, daemon=True,
                                        name="ffmpeg-analysis")
        self._thread.start()
        threading.Thread(target=self._drain_stderr, daemon=True,
                         name="ffmpeg-stderr").start()
        log.info("ffmpeg capture started (%s%s)%s", self._video,
                 f" + {self._audio}" if self._audio else "",
                 f" recording -> {self._rec_path}" if self._rec_path else "")

    def _drain(self) -> None:
        """Keep the analysis pipe empty; hold only the newest frame.

        Never blocks ffmpeg: whatever the CV loop's cadence, this thread reads at
        the stream's rate and simply overwrites. Frames the pipeline misses are
        lost to ANALYSIS only — the recording is a separate ffmpeg output.
        """
        n = self._w * self._h * 3
        stream = self._proc.stdout if self._proc else None
        if stream is None:
            return
        while not self._stop.is_set():
            buf = stream.read(n)
            if len(buf) < n:
                break
            frame = np.frombuffer(buf, np.uint8).reshape(self._h, self._w, 3)
            with self._lock:
                self._latest = frame

    def _drain_stderr(self) -> None:
        stream = self._proc.stderr if self._proc else None
        if stream is None:
            return
        try:
            for raw in stream:
                line = raw.decode("utf-8", "replace").strip()
                if line:
                    log.warning("ffmpeg: %s", line)
        except (OSError, ValueError):
            pass

    def _restart(self) -> None:
        """Rebuild the pipeline (adding or removing the recording output).

        ffmpeg cannot gain an output mid-run, so starting/stopping a recording
        restarts the process. Analysis pauses for well under a second; the
        RECORDING is unaffected because it only ever exists in one process.
        """
        self._teardown()
        self._start()

    def _teardown(self) -> None:
        # NOTE the order: the drain thread must keep reading until ffmpeg has
        # exited. Stopping it first leaves ffmpeg blocked writing analysis frames
        # into a pipe nobody is emptying, so it never reaches its own shutdown
        # and never finalises the recording.
        p, self._proc = self._proc, None
        if p is None:
            self._stop.set()
            return
        # "q" is ffmpeg's graceful quit: it drains its buffers and finalises the
        # mp4. It must then be given TIME — terminate() on Windows is a hard
        # kill, and issuing it straight after the "q" truncated the tail of every
        # recording (measured: a 20s capture landed as 16.1s).
        try:
            if p.stdin:
                p.stdin.write(b"q\n")
                p.stdin.flush()
                p.stdin.close()
        except (OSError, ValueError):
            pass
        try:
            p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            log.warning("ffmpeg did not exit on 'q'; terminating")
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        # ffmpeg is gone; the drain thread's read() now returns EOF and it exits
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None

    # ------------------------------------------------------------------ #
    # CameraSource surface
    # ------------------------------------------------------------------ #
    @property
    def opened(self) -> bool:
        return self._opened and self._proc is not None and self._proc.poll() is None

    @property
    def fps(self) -> float:
        return float(ANALYSIS_FPS)

    def read(self) -> np.ndarray | None:
        with self._lock:
            return self._latest

    def release(self) -> None:
        self._teardown()
        self._opened = False

    # ------------------------------------------------------------------ #
    # Recording
    # ------------------------------------------------------------------ #
    def start_recording(self, path: str, filters: str = "") -> None:
        """Begin writing ``path`` straight from the device.

        ``filters`` is the ffmpeg -vf chain for the recording only (rotation,
        letterbox crop, denoise/sharpen) — it never touches the analysis stream.
        """
        self._rec_path = path
        self._rec_filters = filters
        self._restart()

    def stop_recording(self) -> None:
        if self._rec_path is None:
            return
        self._rec_path = None
        self._rec_filters = ""
        self._restart()

    @property
    def recording(self) -> bool:
        return self._rec_path is not None
