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

import sys
import threading
from pathlib import Path

import cv2
import numpy as np

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

    def __init__(self, index: int):
        self.name = f"Camera {index}"
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(index, backend)
        # request a sensible resolution; the camera will clamp to what it supports
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

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

    def __init__(self, index: int):
        super().__init__(index)
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
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


def open_source(spec: str, *, demo_size=(1280, 720)) -> FrameSource:
    spec = (spec or "0").strip()
    if spec.lower() == "demo":
        return DemoSource(*demo_size)
    if spec.isdigit():
        return ThreadedCameraSource(int(spec))
    ext = Path(spec).suffix.lower()
    if ext in _VIDEO_EXT:
        return VideoSource(spec)
    if ext in _IMAGE_EXT:
        return ImageSource(spec)
    # fall back to treating as camera 0
    return ThreadedCameraSource(0)
