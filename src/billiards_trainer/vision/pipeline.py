"""Per-frame orchestration: calibrate once, then detect → track → events.

This is the single object the capture/worker thread drives. It owns the
calibration, detector, tracker, and shot detector, and returns a
``PipelineResult`` describing the table state for the current frame. It does NOT
touch Qt or the DB — those are wired in the controller, keeping this testable.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config import CALIBRATION_PATH, Settings
from ..events.shot_detector import ShotDetector, ShotEvent
from .background import BackgroundModel, downscale, flow_activity
from .calibration import CalibrationManager
from .geometry import TableModel, expected_ball_radius_px
from .overlay import draw_perspective, draw_rectified, render_schematic
from .tracking import BallTracker
from .types import Detection, Track

log = logging.getLogger("vision.pipeline")


def _median_frames(frames: list[np.ndarray]) -> np.ndarray:
    """Per-pixel median of a small frame list. The common 3-frame case uses the
    exact max/min identity instead of np.median — np.median sorts via partition,
    which at 1080p costs ~110ms/frame and tanked playback to ~7fps. The identity
    is ~5-7x faster and bit-exact for n=3."""
    n = len(frames)
    if n == 1:
        return frames[0]
    if n == 2:  # no true median of 2; average is the sensible smoother
        return ((frames[0].astype(np.uint16) + frames[1]) // 2).astype(np.uint8)
    if n == 3:
        a, b, c = frames
        mn = np.minimum(a, b)
        mx = np.maximum(a, b)
        return np.maximum(mn, np.minimum(mx, c))
    return np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)


@dataclass
class PipelineResult:
    status: str = "init"            # init | calibrating | tracking | deviated
    frame_bgr: np.ndarray | None = None
    rect_bgr: np.ndarray | None = None
    tracks: list[Track] = field(default_factory=list)
    detections: list[Detection] = field(default_factory=list)
    raw_dets: list[Detection] = field(default_factory=list)  # camera-coord dets + guessed numbers (labelling)
    table: TableModel | None = None
    corners: np.ndarray | None = None
    shot_event: ShotEvent | None = None
    shot_state: str = "settled"
    n_balls: int = 0
    deviated: bool = False
    diag: dict = field(default_factory=dict)


class Pipeline:
    def __init__(self, settings: Settings, source: str = ""):
        self.settings = settings
        self.source = source
        self.paused = False
        # Auto-detection master switch. The *engine* defaults to detecting (so the
        # demo/tests exercise the full chain), but the app drives a live camera
        # PREVIEW with this OFF by default — a clean empty table + manual scoring,
        # never a wrong-coloured phantom. Flipped by the Sandbox Detection toggle.
        self.detect_enabled = True
        self._preview_table: TableModel | None = None
        self.calib = CalibrationManager()
        self._strategy = self._load_strategy()  # the single raw-frame detector
        self.tracker = BallTracker()
        self.shots = ShotDetector(settings.detection, settings.balls)
        self._frame_idx = 0
        self._deviation_every = 30  # frames between watchdog checks
        self._tried_load = False
        self._prev_gray = None
        self._bg = BackgroundModel()
        self._prev_small = None
        self._last_flow = 0.0
        self._last_ms = 0.0
        self._frame_ring: deque | None = None  # temporal-median buffer

    def reconfigure(self, settings: Settings) -> None:
        """Apply edited settings (e.g. new felt range / detector) and recalibrate."""
        self.settings = settings
        self._strategy = self._load_strategy()
        self.tracker.reset()
        self.shots = ShotDetector(settings.detection, settings.balls)
        self.calib.clear()
        self._preview_table = None
        self._reset_motion()

    def _load_strategy(self):
        """Resolve the live detector. 'auto' (default) picks the best available —
        a trained YOLO model if one is in the models dir, else the cue-ball
        heuristic. An explicit name forces one. There is ONE detection path now;
        the old classical-Hough-on-rectified 'legacy' detector has been removed."""
        name = getattr(self.settings.balls, "live_strategy", "auto")
        if name == "legacy":
            name = "auto"  # legacy detector removed — resolve to the best available
        try:
            from ..detector_strategies import discover
            strategies = discover()
            if name and name != "auto":
                strat = strategies.get(name)
                if strat is not None:
                    return strat
                log.warning("Live strategy '%s' not found; falling back to auto", name)
            return self._resolve_auto(strategies)
        except Exception as exc:  # noqa: BLE001 - never let strategy loading break the app
            log.warning("Could not load live strategy '%s' (%s); using legacy", name, exc)
            return None

    @staticmethod
    def _resolve_auto(strategies: dict):
        """Pick the best detector: the best trained pool/ball YOLO model first,
        then the cue-ball heuristic, then any blob. Models are SCORED by name so a
        purpose-built pool model wins over a generic 'ball' model and a 'pocket'
        model is never chosen as the ball detector."""
        def score(name: str) -> int:
            if "pocket" in name:
                return -1  # a pocket detector is not a ball detector
            return (3 if "pool" in name else 0) + (2 if "yolo" in name else 0) + \
                   (1 if "ball" in name else 0)
        onnx = [(score(n), n, s) for n, s in strategies.items() if n.startswith("onnx_")]
        onnx = [t for t in onnx if t[0] >= 0]
        if onnx:
            onnx.sort(key=lambda t: t[0], reverse=True)
            _, n, s = onnx[0]
            log.info("auto detector -> %s (trained model)", n)
            return s
        if "cue_ball_white" in strategies:
            log.info("auto detector -> cue_ball_white (no model found)")
            return strategies["cue_ball_white"]
        return next(iter(strategies.values()), None)

    def set_strategy(self, name: str) -> None:
        """Switch the live detector without clearing calibration."""
        self.settings.balls.live_strategy = name
        self._strategy = self._load_strategy()
        self.tracker.reset()
        log.info("Live detector strategy -> %s", name)

    def request_recalibration(self) -> None:
        self.calib.clear()
        self.tracker.reset()
        self.shots.reset()
        self._reset_motion()
        self._tried_load = True  # don't immediately reload the stale saved one

    def _reset_motion(self) -> None:
        # ROI size can change after recalibration; relearn the table baseline.
        self._bg.reset()
        self._prev_gray = None
        self._prev_small = None

    # ------------------------------------------------------------------ #
    def _preview_result(self, frame: np.ndarray, res: PipelineResult) -> PipelineResult:
        """Camera-preview mode (auto-detection OFF): show the raw live feed and a
        clean, empty proportional table overhead — no ball/shot detection at all,
        so there are zero phantom detections to mis-render. Manual +Make/-Miss
        drive scoring in this mode."""
        res.status = "preview"
        res.frame_bgr = frame
        res.shot_state = "preview"
        res.n_balls = 0
        # Use the real locked table if a prior detection pass found one; otherwise
        # a default regulation-proportioned rectangle so the overhead still reads
        # as a pool table.
        if self.calib.is_calibrated:
            table = self.calib.calib.table
        else:
            table = self._default_preview_table()
        res.table = table
        if self.settings.ui.schematic_birdseye:
            res.rect_bgr = render_schematic(table, [], accent=self.settings.ui.accent,
                                            show_traj=False, show_ids=False, debug=False)
        return res

    def _stabilize(self, frame: np.ndarray) -> np.ndarray:
        """Per-pixel median of the last few frames, to suppress sensor noise
        before detection (settings.balls.temporal_median). A still scene becomes
        pixel-identical frame to frame, so blob sizes stop pumping and balls stop
        flickering in/out of the area filter. Returns the frame unchanged when
        disabled or until the buffer fills."""
        if not getattr(self.settings.balls, "temporal_median", True):
            self._frame_ring = None
            return frame
        n = max(2, int(getattr(self.settings.balls, "temporal_median_frames", 3)))
        ring = self._frame_ring
        if ring is None or ring.maxlen != n:
            ring = deque(maxlen=n)
            self._frame_ring = ring
        if ring and ring[-1].shape != frame.shape:  # source/calibration changed
            ring.clear()
        ring.append(frame)
        if len(ring) < n:
            return frame
        return _median_frames(list(ring))

    def _project_raw_to_rect(self, raw_dets, calib):
        """Map RAW-frame detections into the rectified plane (via calib.H) so the
        tracker + bird's-eye schematic (both rectified-space) consume them."""
        if not raw_dets:
            return []
        from .rectify import project_points
        pts = np.array([[d.x, d.y] for d in raw_dets], np.float64)
        rect = project_points(pts, calib.H)
        off = np.array([[d.x + max(d.radius, 1.0), d.y] for d in raw_dets], np.float64)
        rect_off = project_points(off, calib.H)
        out = []
        for d, (rx, ry), (ox, oy) in zip(raw_dets, rect, rect_off, strict=False):
            out.append(Detection(float(rx), float(ry), float(np.hypot(ox - rx, oy - ry)),
                                 d.bgr, d.cls, d.score, number=d.number))
        return out

    def _draw_raw_dets(self, frame, dets):
        """Draw detection circles straight onto the live (raw) frame — used when
        there's no homography to project into the bird's-eye."""
        from .overlay import ball_color
        img = frame.copy()
        for d in dets:
            color, _uncertain = ball_color(d, self.settings.ui.measured_ball_colors)
            c = (int(d.x), int(d.y))
            cv2.circle(img, c, max(4, int(d.radius)), color, 2, cv2.LINE_AA)
        return img

    def _default_preview_table(self) -> TableModel:
        if self._preview_table is None:
            pad = max(8, int(self.settings.rectify.pad_px))
            aspect = max(1.2, float(self.settings.rectify.aspect))  # long:short
            short = 360
            size = (short + 2 * pad, int(short * aspect) + 2 * pad)  # (w, h) portrait
            self._preview_table = TableModel.from_rect(
                size, pad, self.settings.table.pocket_radius_frac)
        return self._preview_table

    # ------------------------------------------------------------------ #
    def process(self, frame: np.ndarray, t: float,
                annotate: bool = True, detect: bool = True) -> PipelineResult:
        """``detect=False`` is a display-only frame: skip the expensive raw-frame
        detection (and the median preprocessing it feeds) and reuse the existing
        tracks, so playback can run faster than detection. The controller drops the
        detection cadence at higher playback speeds (every Nth frame)."""
        t_start = time.perf_counter()
        self._frame_idx += 1
        res = PipelineResult(frame_bgr=frame)

        if not self.detect_enabled:
            self._last_ms = (time.perf_counter() - t_start) * 1000.0
            return self._preview_result(frame, res)

        # Noise-suppressed frame fed to the raw-frame strategy (display/projection
        # still use the original `frame`). The temporal median is a classical-blob
        # noise crutch — a trained model is robust to sensor noise and is only
        # smeared by blending past frames — so skip it for model-based detectors
        # (and on display-only frames, where the median is the costliest skip).
        if not detect or getattr(self._strategy, "model_based", False):
            det_frame = frame
        else:
            det_frame = self._stabilize(frame)

        if not self.calib.is_calibrated:
            if not self._acquire_calibration(frame):
                # No table lock yet — but DON'T refuse to detect. Run the strategy
                # on the raw frame and draw boxes directly on the live view; we just
                # can't project to the bird's-eye without a homography.
                if self._strategy is not None and annotate:
                    try:
                        raw_dets = self._strategy.detect(det_frame, None)
                    except Exception:  # noqa: BLE001
                        raw_dets = []
                    res.detections = raw_dets
                    res.n_balls = len(raw_dets)
                    res.frame_bgr = self._draw_raw_dets(frame, raw_dets)
                    res.status = "detecting_nolock"
                    if self.settings.ui.schematic_birdseye:
                        res.rect_bgr = render_schematic(
                            self._default_preview_table(), [], accent=self.settings.ui.accent,
                            show_traj=False, show_ids=False)
                else:
                    res.status = "calibrating"
                    res.frame_bgr = frame
                self._last_ms = (time.perf_counter() - t_start) * 1000.0
                return res

        calib = self.calib.calib
        res.corners = calib.corners
        res.table = calib.table

        rect = self.calib.rectify(frame)
        if rect is None:
            res.status = "calibrating"
            return res

        if not detect:
            # Display-only frame (cadence): reuse the current tracks, don't re-detect.
            tracks = self.tracker.tracks
            detections = []
        else:
            # Detect on the RAW frame, project results into the rectified plane so
            # tracking + the bird's-eye schematic consume rectified-space points.
            if self._strategy is not None:
                try:
                    raw_dets = self._strategy.detect(det_frame, calib)
                except Exception as exc:  # noqa: BLE001 - a bad frame must not kill the loop
                    log.debug("detector failed on a frame: %s", exc)
                    raw_dets = []
                res.raw_dets = list(raw_dets)   # camera-coord, for the in-app labeller
                detections = self._project_raw_to_rect(raw_dets, calib)
                # Physical-size prior: reject blobs whose radius is far from the
                # known ball radius (pocket-shadow "balls" too big, speckle too
                # small). Skipped for model-based detectors, which already validate
                # ball-ness with high confidence.
                if not getattr(self._strategy, "model_based", False):
                    exp_r = expected_ball_radius_px(calib.table, self.settings.table.size)
                    tol = getattr(self.settings.balls, "size_prior_tol", 0.25)
                    if exp_r > 2.0 and tol > 0:
                        lo, hi = exp_r * (1.0 - tol), exp_r * (1.0 + tol)
                        detections = [d for d in detections if lo <= d.radius <= hi]
            else:
                detections = []
            # Confidence floor. A trained model is well-calibrated and already
            # thresholded in the strategy, so apply only the base confidence_floor.
            # A heuristic (non-model) detector gets the stricter render_floor —
            # "draw nothing rather than a wrong phantom".
            det = self.settings.detection
            if getattr(self._strategy, "model_based", False):
                floor = det.confidence_floor
            else:
                floor = max(det.confidence_floor, getattr(det, "render_floor", 0.0))
            detections = [d for d in detections if d.score >= floor]
            tracks = self.tracker.update(detections, calib.table.short_side)
        res.tracks = tracks
        res.detections = detections
        res.n_balls = len(tracks)

        # motion energy: percentage of playing-area pixels that changed
        # *significantly* between frames. This discriminates a real moving ball
        # (a tight cluster of big changes) from compression/lighting flicker
        # (scattered small changes), unlike a plain mean difference.
        gray = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)
        tbl = calib.table
        roi = gray[int(tbl.y0):int(tbl.y1), int(tbl.x0):int(tbl.x1)]
        if self._prev_gray is not None and self._prev_gray.shape == roi.shape:
            motion = float((cv2.absdiff(roi, self._prev_gray) > 25).mean()) * 100.0
        else:
            motion = 0.0
        self._prev_gray = roi

        # extra modalities for evidence fusion: background-subtraction foreground
        # area + coherent optical-flow activity (both on a downscaled ROI). These
        # (MOG2 + Farneback) are the heaviest per-frame ops and feed ONLY shot
        # detection, so they run only when fusion is enabled — off by default
        # while cue-ball tracking (M2) is the focus, freeing the real-time budget.
        evidence = {"motion": motion}
        if self.settings.detection.use_fusion:
            small = downscale(roi)
            fg = self._bg.update(small)
            # optical flow is the costliest + least-weighted signal — every 2nd frame
            if self._frame_idx % 2 == 0:
                self._last_flow = flow_activity(self._prev_small, small)
            self._prev_small = small
            evidence = {"motion": motion, "flow": self._last_flow * 100.0, "fg": fg * 100.0}

        if self.paused:
            # keep showing video + tracking, but never count shots while paused
            self.shots.reset()
            res.shot_state = "paused"
        else:
            event = self.shots.update(tracks, calib.table, t, motion, evidence)
            res.shot_event = event
            res.shot_state = self.shots.state
            res.diag = dict(self.shots.last_diag)
            res.diag["ms"] = round(self._last_ms, 1)
            res.diag["fps"] = int(1000 / self._last_ms) if self._last_ms > 0.1 else 0

        # periodic deviation watchdog (cheap: only every N frames)
        if self._frame_idx % self._deviation_every == 0:
            self.calib.check_deviation(frame, self.settings)
            if self.calib.deviated and self.settings.table.auto_relock:
                log.info("Auto-relocking table after deviation")
                self.request_recalibration()
        res.deviated = self.calib.deviated
        res.status = "deviated" if self.calib.deviated else "tracking"

        ui = self.settings.ui
        overlays = annotate and ui.show_overlays
        # Draw every ball at its known physical radius (unless the raw-size debug
        # toggle is on) so the overhead shows uniform regulation balls.
        norm_r = (expected_ball_radius_px(calib.table, self.settings.table.size)
                  if (ui.normalize_ball_size and not ui.show_raw_detection_size) else None)
        # Bird's-eye: a clean rendered schematic (proportional) by default, rather
        # than the warped/clipped camera image.
        if annotate and ui.schematic_birdseye:
            res.rect_bgr = render_schematic(
                calib.table, tracks, accent=ui.accent,
                show_traj=ui.show_trajectories, show_ids=ui.show_ball_ids,
                debug=ui.debug_overlay, detections=detections, diag=res.diag,
                measured_colors=ui.measured_ball_colors, fixed_radius=norm_r,
            )
        elif overlays:
            res.rect_bgr = draw_rectified(
                rect, tracks, calib.table, show_traj=ui.show_trajectories,
                show_ids=ui.show_ball_ids, accent=ui.accent,
                measured_colors=ui.measured_ball_colors, fixed_radius=norm_r,
            )
        else:
            res.rect_bgr = rect

        # Live camera view keeps the real feed (with light overlay unless off).
        if overlays:
            res.frame_bgr = draw_perspective(
                frame, calib.corners, tracks, calib.Hinv, accent=ui.accent
            )
        else:
            res.frame_bgr = frame
        self._last_ms = (time.perf_counter() - t_start) * 1000.0
        return res

    # ------------------------------------------------------------------ #
    def _acquire_calibration(self, frame: np.ndarray) -> bool:
        """Restore a saved calibration if available + matching, else detect and
        persist a fresh one."""
        if (not self._tried_load and self.settings.table.persist_calibration
                and self.source):
            self._tried_load = True
            if self.calib.try_load(CALIBRATION_PATH, self.source, frame.shape, self.settings):
                return True
        if not self.calib.calibrate(frame, self.settings):
            return False
        if self.settings.table.persist_calibration and self.source:
            self.calib.save(CALIBRATION_PATH, self.source, frame.shape, self.settings)
        return True
