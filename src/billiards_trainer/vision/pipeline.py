"""Per-frame orchestration: calibrate once, then detect → track → events.

This is the single object the capture/worker thread drives. It owns the
calibration, detector, tracker, and shot detector, and returns a
``PipelineResult`` describing the table state for the current frame. It does NOT
touch Qt or the DB — those are wired in the controller, keeping this testable.
"""

import logging
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config import CALIBRATION_PATH, Settings
from ..events.shot_detector import ShotDetector, ShotEvent
from .background import BackgroundModel, downscale, flow_activity
from .balls import make_detector
from .calibration import CalibrationManager
from .geometry import TableModel
from .overlay import draw_perspective, draw_rectified, render_schematic
from .tracking import BallTracker
from .types import Detection, Track

log = logging.getLogger("vision.pipeline")


@dataclass
class PipelineResult:
    status: str = "init"            # init | calibrating | tracking | deviated
    frame_bgr: np.ndarray | None = None
    rect_bgr: np.ndarray | None = None
    tracks: list[Track] = field(default_factory=list)
    detections: list[Detection] = field(default_factory=list)
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
        self.detector = make_detector(settings.balls, settings.felt)
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

    def reconfigure(self, settings: Settings) -> None:
        """Apply edited settings (e.g. new felt range / backend) and recalibrate."""
        self.settings = settings
        self.detector = make_detector(settings.balls, settings.felt)
        self.tracker.reset()
        self.shots = ShotDetector(settings.detection, settings.balls)
        self.calib.clear()
        self._preview_table = None
        self._reset_motion()

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
                annotate: bool = True) -> PipelineResult:
        t_start = time.perf_counter()
        self._frame_idx += 1
        res = PipelineResult(frame_bgr=frame)

        if not self.detect_enabled:
            self._last_ms = (time.perf_counter() - t_start) * 1000.0
            return self._preview_result(frame, res)

        if not self.calib.is_calibrated:
            if not self._acquire_calibration(frame):
                res.status = "calibrating"
                if annotate:
                    res.frame_bgr = frame
                return res
            self.detector = make_detector(self.settings.balls, self.calib.calib.felt)

        calib = self.calib.calib
        res.corners = calib.corners
        res.table = calib.table

        rect = self.calib.rectify(frame)
        if rect is None:
            res.status = "calibrating"
            return res

        detections = self.detector.detect(rect, calib.rect_mask, calib.table)
        # With auto-detection ON, demand the stricter render floor: better to draw
        # nothing than a low-confidence phantom. Falls back to the looser tracking
        # floor only if render_floor wasn't set (old settings files).
        det = self.settings.detection
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
        # area + coherent optical-flow activity (both on a downscaled ROI)
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
        # Bird's-eye: a clean rendered schematic (proportional) by default, rather
        # than the warped/clipped camera image.
        if annotate and ui.schematic_birdseye:
            res.rect_bgr = render_schematic(
                calib.table, tracks, accent=ui.accent,
                show_traj=ui.show_trajectories, show_ids=ui.show_ball_ids,
                debug=ui.debug_overlay, detections=detections, diag=res.diag,
                measured_colors=ui.measured_ball_colors,
            )
        elif overlays:
            res.rect_bgr = draw_rectified(
                rect, tracks, calib.table, show_traj=ui.show_trajectories,
                show_ids=ui.show_ball_ids, accent=ui.accent,
                measured_colors=ui.measured_ball_colors,
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
            if self.calib.try_load(CALIBRATION_PATH, self.source, frame.shape):
                return True
        if not self.calib.calibrate(frame, self.settings):
            return False
        if self.settings.table.persist_calibration and self.source:
            self.calib.save(CALIBRATION_PATH, self.source, frame.shape, self.settings)
        return True
