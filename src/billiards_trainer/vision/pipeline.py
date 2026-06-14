"""Per-frame orchestration: calibrate once, then detect → track → events.

This is the single object the capture/worker thread drives. It owns the
calibration, detector, tracker, and shot detector, and returns a
``PipelineResult`` describing the table state for the current frame. It does NOT
touch Qt or the DB — those are wired in the controller, keeping this testable.
"""

import logging
from dataclasses import dataclass, field

import numpy as np

from ..config import CALIBRATION_PATH, Settings
from ..events.shot_detector import ShotDetector, ShotEvent
from .balls import make_detector
from .calibration import CalibrationManager
from .geometry import TableModel
from .overlay import draw_perspective, draw_rectified
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


class Pipeline:
    def __init__(self, settings: Settings, source: str = ""):
        self.settings = settings
        self.source = source
        self.calib = CalibrationManager()
        self.detector = make_detector(settings.balls, settings.felt)
        self.tracker = BallTracker()
        self.shots = ShotDetector(settings.balls)
        self._frame_idx = 0
        self._deviation_every = 30  # frames between watchdog checks
        self._tried_load = False

    def reconfigure(self, settings: Settings) -> None:
        """Apply edited settings (e.g. new felt range / backend) and recalibrate."""
        self.settings = settings
        self.detector = make_detector(settings.balls, settings.felt)
        self.tracker.reset()
        self.shots.reset()
        self.calib.clear()

    def request_recalibration(self) -> None:
        self.calib.clear()
        self.tracker.reset()
        self.shots.reset()
        self._tried_load = True  # don't immediately reload the stale saved one

    # ------------------------------------------------------------------ #
    def process(self, frame: np.ndarray, t: float,
                annotate: bool = True) -> PipelineResult:
        self._frame_idx += 1
        res = PipelineResult(frame_bgr=frame)

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
        tracks = self.tracker.update(detections, calib.table.short_side)
        res.tracks = tracks
        res.detections = detections
        res.n_balls = len(tracks)

        event = self.shots.update(tracks, calib.table, t)
        res.shot_event = event
        res.shot_state = self.shots.state

        # periodic deviation watchdog (cheap: only every N frames)
        if self._frame_idx % self._deviation_every == 0:
            self.calib.check_deviation(frame, self.settings)
            if self.calib.deviated and self.settings.table.auto_relock:
                log.info("Auto-relocking table after deviation")
                self.request_recalibration()
        res.deviated = self.calib.deviated
        res.status = "deviated" if self.calib.deviated else "tracking"

        overlays = annotate and self.settings.ui.show_overlays
        if overlays:
            res.rect_bgr = draw_rectified(
                rect, tracks, calib.table,
                show_traj=self.settings.ui.show_trajectories,
                show_ids=self.settings.ui.show_ball_ids,
                accent=self.settings.ui.accent,
            )
            res.frame_bgr = draw_perspective(
                frame, calib.corners, tracks, calib.Hinv, accent=self.settings.ui.accent
            )
        else:
            res.rect_bgr = rect
            res.frame_bgr = frame
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
