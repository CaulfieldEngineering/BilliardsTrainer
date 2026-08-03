"""Frame sources.

A ``FrameSource`` yields BGR frames via ``read()``. Four kinds:

* ``CameraSource`` — OpenCV VideoCapture on a device index (DirectShow on Win).
* ``VideoSource``  — a video file, looping.
* ``ImageSource``  — a single still, repeated (useful for tuning).
* ``DemoSource``   — a fully synthetic, perspective-correct pool table with a
  scripted make-shot. Exercises the entire pipeline (felt → rectify → balls →
  track → shot/make) with no hardware, so the make/miss flow is demoable cold.

``open_source(spec)`` maps a settings string to the right source:
    "0"/"1"     -> camera index
    "demo"      -> DemoSource
    "*.mp4/..." -> VideoSource ; image extensions -> ImageSource
"""

import logging
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from ..config import CameraSettings

log = logging.getLogger("capture.camera")

_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}
_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class FrameSource:
    name = "source"
    is_live = False  # True for real cameras (transient empty reads are tolerated)

    def read(self) -> np.ndarray | None:
        raise NotImplementedError

    def release(self) -> None:
        pass

    @property
    def fps(self) -> float:
        return 30.0


class CameraSource(FrameSource):
    is_live = True

    def __init__(self, index: int, cam: CameraSettings | None = None):
        self.name = f"Camera {index}"
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(index, backend)
        # request a sensible resolution; the camera will clamp to what it supports
        w = cam.width if cam else 1280
        h = cam.height if cam else 720
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        # Prefer MJPEG delivery: HDMI capture dongles renegotiate their raw
        # pixel format when the input signal changes mode, and a stale YUV
        # interpretation shows as green/purple chroma garbage. Compressed MJPEG
        # sidesteps raw-format ambiguity entirely; backends that don't support
        # it ignore the request.
        try:
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except cv2.error:
            pass
        if cam is not None:
            self._apply_controls(cam)

    def _apply_controls(self, cam: CameraSettings) -> None:
        """Best-effort UVC control set (focus / exposure / white balance / gain).

        Called once, before any grab thread starts, so it never races read().
        Not all backends honour every property — failures are silently ignored,
        which is why these are exposed as 'requests' in the UI, not guarantees.
        The tethered DSLR path does NOT use these (it has no UVC controls); it
        goes through :class:`~billiards_trainer.capture.tether.GphotoSource`.
        """
        def _set(prop: int, value: float) -> None:
            try:
                self._cap.set(prop, value)
            except cv2.error:
                pass

        _set(cv2.CAP_PROP_AUTOFOCUS, 1.0 if cam.auto_focus else 0.0)
        if not cam.auto_focus and cam.focus >= 0:
            _set(cv2.CAP_PROP_FOCUS, cam.focus)
        # AUTO_EXPOSURE: 0.75 = auto, 0.25 = manual is the common V4L2/DShow convention.
        _set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if cam.auto_exposure else 0.25)
        if not cam.auto_exposure and cam.exposure != -1:
            _set(cv2.CAP_PROP_EXPOSURE, cam.exposure)
        _set(cv2.CAP_PROP_AUTO_WB, 1.0 if cam.auto_wb else 0.0)
        if not cam.auto_wb and cam.wb_temperature >= 0:
            _set(cv2.CAP_PROP_WB_TEMPERATURE, cam.wb_temperature)
        if cam.gain >= 0:
            _set(cv2.CAP_PROP_GAIN, cam.gain)

    @property
    def opened(self) -> bool:
        return self._cap.isOpened()

    def read(self) -> np.ndarray | None:
        if not self._cap.isOpened():
            return None
        ok, frame = self._cap.read()
        return frame if ok else None

    def release(self) -> None:
        self._cap.release()

    @property
    def fps(self) -> float:
        f = self._cap.get(cv2.CAP_PROP_FPS)
        return f if f and f > 1 else 30.0


class ThreadedCameraSource(CameraSource):
    """CameraSource with a dedicated grab thread that always holds only the
    NEWEST frame.

    Two real-time problems this solves for live tracking:
    * ``VideoCapture.read()`` BLOCKS until the driver delivers a frame (up to a
      full frame period) — on the worker tick that stall serialized capture with
      inference and capped throughput.
    * When processing runs slower than the camera, the driver's internal queue
      backs up and the app tracks second-old frames. Draining continuously and
      keeping only the latest frame pins latency at ~1 frame.

    ``read()`` has take-semantics: it returns each frame once and ``None`` when
    nothing new has arrived — the controller already tolerates transient ``None``
    reads from a live source. Pure ``threading`` (no Qt objects), so it cannot
    reproduce the worker-thread QTimer lifetime crashes.
    """

    def __init__(self, index: int, cam: CameraSettings | None = None):
        super().__init__(index, cam)
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Optional real-time frame sink. When set, the grab thread hands EVERY
        # raw frame to this callback the instant it arrives — the recording path
        # uses it to capture the source at true camera cadence, decoupled from
        # the (slower, variable) CV tick. Assignment is atomic in CPython, so no
        # lock is needed to publish/clear it.
        self._sink = None
        self._sink_sig: bytes | None = None   # last picture handed to the sink
        self._sink_last = 0.0
        # Cache everything queried later BEFORE the grab thread starts —
        # VideoCapture calls are not safe concurrently with read() on another
        # thread (opened/fps are read by the controller after construction).
        self._opened = self._cap.isOpened()
        self._fps = super().fps if self._opened else 30.0
        if self._opened:
            try:
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # best-effort; not all backends honour it
            except cv2.error:
                pass
            self._thread = threading.Thread(target=self._grab_loop,
                                            name="camera-grab", daemon=True)
            self._thread.start()

    @property
    def opened(self) -> bool:
        return self._opened

    @property
    def fps(self) -> float:
        return self._fps

    def set_frame_sink(self, callback) -> None:
        """Register (or clear, with ``None``) a callback the grab thread invokes
        with every raw frame, at true camera cadence. Used by the recorder to
        capture the source in real time without riding the CV tick."""
        self._sink = callback

    # A capture backend re-serves the SAME picture when polled faster than the
    # camera produces: measured on the rig, read() returns 50/s while the feed
    # carries 30 unique pictures/s, so 40% of reads are stale repeats. The
    # recorder's CFR retimer slots frames by TIMESTAMP and cannot see that, so
    # it emitted whichever copy happened to land first in each slot. Replaying
    # the real capture log through it: 17.4% of written frames repeated the
    # previous picture while 17.2% of the camera's fresh pictures were thrown
    # away — one frame in six, which is exactly the residual micro-stutter.
    # Skipping stale reads here takes that to 0.44%.
    _SINK_STRIDE = 32        # subsample compared to decide "same picture"
    _SINK_KEEPALIVE_S = 0.5  # forward anyway if the feed genuinely freezes

    def _is_fresh(self, frame) -> bool:
        """True when this frame differs from the last one handed to the sink.

        Compares a strided subsample, which is decisive on a live sensor (noise
        dithers every pixel) and costs a few KB per frame. A genuinely frozen
        feed still forwards on the keepalive, so the retimer's timeline never
        stalls and a stuck camera is still recorded as stuck rather than as
        nothing at all.
        """
        sig = frame[::self._SINK_STRIDE, ::self._SINK_STRIDE].tobytes()
        now = time.monotonic()
        if sig != self._sink_sig:
            self._sink_sig = sig
            self._sink_last = now
            return True
        if now - self._sink_last >= self._SINK_KEEPALIVE_S:
            self._sink_last = now
            return True
        return False

    def _grab_loop(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    ok, frame = self._cap.read()
                except cv2.error:  # backend hiccup — treat like a missed frame
                    ok, frame = False, None
                if not ok or frame is None:
                    # dead/unplugged camera: don't spin hot; keep checking until
                    # release() or the device comes back
                    if self._stop.wait(0.01):
                        break
                    continue
                with self._lock:
                    self._latest = frame
                # Feed the recording sink OUTSIDE the lock (it copies the frame
                # with tobytes(); never hold the capture lock across it) and
                # never let a sink error kill the grab loop / camera handle.
                sink = self._sink
                if sink is not None and self._is_fresh(frame):
                    try:
                        sink(frame)
                    except Exception:  # noqa: BLE001 - recording must not crash capture
                        log.exception("frame sink raised; dropping this frame")
        finally:
            # The grab thread OWNS the capture handle: releasing it here (only
            # after the loop exits) guarantees the handle is freed exactly once
            # and never while a concurrent read() is in flight — even when
            # release() below gives up waiting on a wedged driver.
            try:
                self._cap.release()
            except cv2.error:
                pass

    def read(self) -> np.ndarray | None:
        with self._lock:
            frame, self._latest = self._latest, None
        return frame

    def release(self) -> None:
        self._stop.set()
        t = self._thread
        if t is None:
            self._cap.release()  # thread never started (camera didn't open)
            return
        t.join(timeout=2.0)
        # If the join timed out the loop is wedged inside the driver's read();
        # its finally block releases the handle as soon as that call returns.


class VideoSource(FrameSource):
    is_video = True  # supports seek/scrub/step (transport controls)

    def __init__(self, path: str):
        self.name = Path(path).name
        self._path = path
        self._cap = cv2.VideoCapture(path)

    @property
    def opened(self) -> bool:
        return self._cap.isOpened()

    def read(self) -> np.ndarray | None:
        ok, frame = self._cap.read()
        if not ok:  # loop
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
        return frame if ok else None

    def release(self) -> None:
        self._cap.release()

    @property
    def fps(self) -> float:
        f = self._cap.get(cv2.CAP_PROP_FPS)
        return f if f and f > 1 else 30.0

    @property
    def frame_count(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def position(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)

    def seek(self, frame_idx: int) -> None:
        n = self.frame_count
        idx = max(0, min(frame_idx, n - 1) if n else frame_idx)
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))


class ImageSource(FrameSource):
    def __init__(self, path: str):
        self.name = Path(path).name
        self._img = cv2.imread(path)

    @property
    def opened(self) -> bool:
        return self._img is not None

    def read(self) -> np.ndarray | None:
        return None if self._img is None else self._img.copy()


class DemoSource(FrameSource):
    """Synthetic perspective pool table with a scripted, looping make-shot."""

    name = "Demo simulation"

    def __init__(self, w: int = 1280, h: int = 720):
        self.w, self.h = w, h
        self._t = 0
        self._cycle = 220
        # felt as a perspective trapezoid (top narrower than bottom)
        self._felt_img = np.array([
            [360, 130], [920, 130], [1060, 610], [220, 610]
        ], np.float32)
        # table space: x in [0,1] (short), y in [0,2] (long)
        table_quad = np.array([[0, 0], [1, 0], [1, 2], [0, 2]], np.float32)
        self._H = cv2.getPerspectiveTransform(table_quad, self._felt_img)

    @property
    def opened(self) -> bool:
        return True

    def _project(self, x: float, y: float) -> tuple[int, int]:
        p = cv2.perspectiveTransform(np.array([[[x, y]]], np.float32), self._H)[0, 0]
        return int(p[0]), int(p[1])

    def read(self) -> np.ndarray | None:
        img = np.full((self.h, self.w, 3), (28, 24, 20), np.uint8)  # dark room
        # felt
        cv2.fillConvexPoly(img, self._felt_img.astype(np.int32), (95, 150, 60))
        cv2.polylines(img, [self._felt_img.astype(np.int32)], True, (60, 45, 30), 14, cv2.LINE_AA)
        # pockets (table-space corners + side midpoints)
        for (px, py) in [(0, 0), (1, 0), (1, 2), (0, 2), (0, 1), (1, 1)]:
            cv2.circle(img, self._project(px, py), 13, (12, 12, 12), -1, cv2.LINE_AA)

        cue_xy, obj_xy, obj_alive = self._script()
        if obj_alive:
            self._ball(img, obj_xy, (40, 200, 235))   # yellow-ish solid
        self._ball(img, cue_xy, (245, 245, 245))       # cue (white)

        self._t = (self._t + 1) % self._cycle
        return img

    def _ball(self, img, xy_table, color) -> None:
        cx, cy = self._project(*xy_table)
        # radius scaled by local projection (project a small offset)
        ox, oy = self._project(xy_table[0] + 0.05, xy_table[1])
        r = max(9, int(np.hypot(ox - cx, oy - cy)))
        cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), r, (20, 20, 20), 1, cv2.LINE_AA)
        # a little shading so it reads as a sphere
        cv2.circle(img, (cx - r // 3, cy - r // 3), max(2, r // 4), (255, 255, 255), -1, cv2.LINE_AA)

    def _script(self):
        """Return (cue_xy, obj_xy, obj_alive) for the current cycle frame.

        Timeline (cycle = 220 frames), shaped to satisfy the hardened shot gates
        (sustained strike, real travel, brief pocket dwell, then the ball drops):
          0-30    settle (cue home, object set)
          30-44   cue strikes FAST (sustained, well above the strike threshold)
          44-62   object rolls FAST a long way to the bottom-right pocket
          62-70   object settles AT the pocket lip (a few frames of dwell)
          70-160  object has dropped in (gone); cue returns home SLOWLY
          160-220 settle; object reappears at loop start (new, stationary track)
        """
        t = self._t
        home = np.array([0.30, 1.30])
        contact = np.array([0.55, 1.55])
        pocket = np.array([1.0, 2.0])      # bottom-right corner pocket (table-space)
        obj_start = np.array([0.55, 1.45])

        if t < 30:                                   # settle
            return tuple(home), tuple(obj_start), True
        if t < 44:                                   # strike (fast, sustained)
            a = (t - 30) / 14.0
            return tuple(home + (contact - home) * a), tuple(obj_start), True
        if t < 58:                                   # object rolls to pocket (fast, far)
            a = (t - 44) / 14.0
            return tuple(contact), tuple(obj_start + (pocket - obj_start) * a), True
        if t < 61:                                   # object at the pocket lip (a few frames)
            return tuple(contact), tuple(pocket), True
        if t < 160:                                  # object dropped in; cue returns slow
            a = (t - 61) / 99.0
            return tuple(contact + (home - contact) * a), (0.0, 0.0), False
        return tuple(home), (0.0, 0.0), False        # settle, object gone until loop


def open_source(spec: str, *, demo_size=(1280, 720),
                cam: CameraSettings | None = None) -> FrameSource:
    spec = (spec or "0").strip()
    if spec.lower() == "demo":
        return DemoSource(*demo_size)
    if spec.lower() in ("tether", "canon", "gphoto"):
        # Lazy import: the tether module is only needed on this path and keeps
        # camera.py free of a hard dependency on it (and avoids an import cycle).
        from ..config import TetherSettings
        from .tether import GphotoSource
        return GphotoSource(cam.tether if cam else TetherSettings())
    if spec.isdigit():
        return ThreadedCameraSource(int(spec), cam)
    ext = Path(spec).suffix.lower()
    if ext in _VIDEO_EXT:
        return VideoSource(spec)
    if ext in _IMAGE_EXT:
        return ImageSource(spec)
    # fall back to treating as camera 0
    return ThreadedCameraSource(0, cam)
