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
from .types import BallClass, Detection, Track

log = logging.getLogger("vision.pipeline")

# Impossible-geometry watchdog budget. Each relock destroys tracker and shot
# state, so a watchdog that can't fix its own trigger must stop trying.
_MAX_FRUITLESS_RELOCKS = 2
# Clean watchdog checks required before the relock budget is handed back.
_IMPOSSIBLE_REARM = 20


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
        self._last_detections: list = []
        self._last_raw_dets: list = []
        self._frames_since_ingest = 0
        self._impossible_streak = 0   # geometry-watchdog counter (see process)
        self._impossible_relocks = 0  # relocks spent on the CURRENT impossibility
        self._impossible_clear = 0    # consecutive clean checks (re-arms the budget)
        self._impossible_gave_up = False
        self._paths_settle_t = None   # when the table last came to rest
        self._paths_alpha = 1.0       # play-path opacity (fades after settle)
        # Play-path trails (broadcast look): every ball that moves during a
        # play leaves a persistent colored path — the cue ball's in white —
        # shown until the NEXT play begins. id -> {pts, bgr, cue}
        self._play_paths: dict[int, dict] = {}
        self._prev_shot_state = "settled"
        self._bg = BackgroundModel()
        self._prev_small = None
        self._last_flow = 0.0
        self._last_ms = 0.0
        self._last_stages: dict = {}  # previous frame's per-stage ms (perf HUD)
        self._frame_ring: deque | None = None  # temporal-median buffer
        self._cam_pose: tuple | None = None  # (key, C) parallax-camera cache

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
            strat = None
            if name and name != "auto":
                strat = strategies.get(name)
                if strat is None:
                    log.warning("Live strategy '%s' not found; falling back to auto", name)
            if strat is None:
                strat = self._resolve_auto(strategies)
            if strat is not None and hasattr(strat, "far_rail_rescan"):
                # Honor the tuning knob even on an overhead camera. The old
                # "overhead has no foreshortened far rail -> skip the 2nd pass"
                # shortcut measurably LOSES balls on the real rig: the portrait
                # frame letterboxes to 640px and top-band balls shrink below the
                # model's floor — the band rescan is the only pass that finds
                # them (verified 2026-07-23: two top-rail balls, 0.83 conf in
                # the rescan, absent at full-frame scale).
                strat.far_rail_rescan = bool(
                    getattr(self.settings.balls, "far_rail_rescan", True))
            return strat
        except Exception as exc:  # noqa: BLE001 - never let strategy loading break the app
            log.warning("Could not load live strategy '%s' (%s); using legacy", name, exc)
            return None

    @staticmethod
    def _resolve_auto(strategies: dict):
        """Pick the best detector: the find+identify ENSEMBLE when both of its
        models are present, else the best trained pool/ball YOLO model, then the
        cue-ball heuristic, then any blob. Models are SCORED by name so a
        purpose-built pool model wins over a generic 'ball' model and a 'pocket'
        model is never chosen as the ball detector.

        The ensemble is checked FIRST and by name, because the onnx_ scoring
        below can't see it: it is registered as ``ensemble_findid``, so a
        prefix filter of "onnx_" skipped it entirely and 'auto' silently fell
        back to the position-only finder — balls detected but never NUMBERED.
        That only stayed hidden because the setting was pinned explicitly on
        the old machine; a fresh install got no identities at all.
        """
        ens = next((s for n, s in strategies.items() if n.startswith("ensemble_")), None)
        if ens is not None:
            log.info("auto detector -> %s (find+identify ensemble)", ens.name)
            return ens

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

    def _apply_alignment_grid(self, res) -> None:
        """Rule-of-thirds grid over the camera view for squaring the physical
        mount. Drawn on a COPY — the analysed/recorded frames stay clean."""
        if not getattr(self.settings.ui, "alignment_grid", False):
            return
        img = res.frame_bgr
        if img is None:
            return
        img = img.copy()
        h, w = img.shape[:2]
        for f in (1 / 3, 2 / 3):
            x, y = int(w * f), int(h * f)
            cv2.line(img, (x, 0), (x, h), (0, 0, 0), 3, cv2.LINE_AA)
            cv2.line(img, (x, 0), (x, h), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.line(img, (0, y), (w, y), (0, 0, 0), 3, cv2.LINE_AA)
            cv2.line(img, (0, y), (w, y), (255, 255, 255), 1, cv2.LINE_AA)
        c = (w // 2, h // 2)
        cv2.drawMarker(img, c, (0, 0, 0), cv2.MARKER_CROSS, 44, 3, cv2.LINE_AA)
        cv2.drawMarker(img, c, (255, 255, 255), cv2.MARKER_CROSS, 40, 1, cv2.LINE_AA)
        res.frame_bgr = img

    def _apply_detections(self, raw_dets, calib, frame_shape):
        """Project raw-frame detections to rect space, run the sanity filters,
        and update the tracker. Shared by the synchronous path (video replay,
        seek/step, tests) and ingest_raw_detections (live async worker)."""
        detections = self._project_raw_to_rect(raw_dets, calib, frame_shape)
        # Physical-size prior: reject blobs whose radius is far from the known
        # ball radius. Skipped for model-based detectors, which already
        # validate ball-ness with high confidence.
        if not getattr(self._strategy, "model_based", False):
            exp_r = expected_ball_radius_px(calib.table, self.settings.table.size)
            tol = getattr(self.settings.balls, "size_prior_tol", 0.25)
            if exp_r > 2.0 and tol > 0:
                lo, hi = exp_r * (1.0 - tol), exp_r * (1.0 + tol)
                detections = [d for d in detections if lo <= d.radius <= hi]
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
        # Geometry sanity — the single-class ball-finder fires on things that
        # aren't balls. Two safe rejects: (1) anything off the playing surface
        # (floor/rail/hand beyond the bed); (2) an empty POCKET void, which is
        # dark and gets mistaken for the 8-ball — a detection sitting in a
        # pocket capture zone that reads as EIGHT/UNKNOWN is the pocket itself,
        # not a ball. A genuinely potted 8 is transient, so dropping it here is
        # harmless. Real balls on the bed (incl. near rails) are untouched.
        tbl = calib.table
        edge = tbl.pocket_radius * 0.5
        # Head/foot SPOT phantom: the felt marking detects as a ball and reads
        # UNKNOWN (a grey '?' parked mid-table). An UNKNOWN detection centred
        # on a spot is the spot, not a ball — a real ball there classifies.
        cx = (tbl.x0 + tbl.x1) / 2.0
        spots = [(cx, tbl.y0 + (tbl.y1 - tbl.y0) * f) for f in (0.25, 0.75)]
        spot_r2 = (0.8 * expected_ball_radius_px(tbl, self.settings.table.size)) ** 2
        kept = []
        for d in detections:
            if not tbl.on_table(d.x, d.y, margin=edge):
                continue
            if (d.cls in (BallClass.EIGHT, BallClass.UNKNOWN)
                    and tbl.pocket_at(d.x, d.y, scale=0.9) is not None):
                continue
            if (d.cls == BallClass.UNKNOWN
                    and any((d.x - sx) ** 2 + (d.y - sy) ** 2 <= spot_r2
                            for sx, sy in spots)):
                continue
            kept.append(d)
        detections = kept
        tracks = self.tracker.update(
            detections, calib.table.short_side,
            bounds=(tbl.x0, tbl.y0, tbl.x1, tbl.y1),
            pockets=[(p.x, p.y) for p in tbl.pockets],
            pocket_r=float(tbl.pocket_radius),
        )
        self._last_detections = detections
        self._last_raw_dets = list(raw_dets)
        self._frames_since_ingest = 0
        return tracks, detections

    def ingest_raw_detections(self, raw_dets, frame_shape) -> None:
        """Apply detector output produced OFF the display path (the live async
        worker). Runs on the pipeline's own thread via a queued slot, so all
        tracker/state mutation stays single-threaded."""
        calib = self.calib.calib
        if calib is None:
            return
        self._apply_detections(raw_dets, calib, frame_shape)

    def _update_play_paths(self, tracks, shot_state: str, t: float) -> None:
        """Accumulate each moving ball's path for the CURRENT play. Paths hold
        through the settle for review, then FADE OUT starting 3s after all
        table movement stops (Joe's spec); a new play clears them instantly."""
        if shot_state == "moving" and self._prev_shot_state != "moving":
            self._play_paths.clear()
        # The fade clock starts only when the COMPLETE shot is over: every
        # tracked ball at rest. "Movement" needs a REAL threshold — detection
        # jitter keeps instantaneous velocity nonzero forever (which froze the
        # fade entirely), so a ball counts as moving only if it actually
        # DISPLACED over its recent history (~last half second), or the shot
        # state machine says a strike is in progress.
        move_px = 1.2 * expected_ball_radius_px(  # ~0.6 ball diameters (Joe's spec)
            self.calib.calib.table, self.settings.table.size) if self.calib.calib else 15.0
        def _really_moving(tr) -> bool:
            hist = tr.history[-8:]
            if len(hist) < 2:
                return False
            net = (abs(hist[-1][0] - hist[0][0]) ** 2
                   + abs(hist[-1][1] - hist[0][1]) ** 2) ** 0.5
            return net > move_px
        any_motion = shot_state == "moving" or any(
            tr.misses == 0 and _really_moving(tr) for tr in tracks)
        if any_motion:
            self._paths_settle_t = None
            self._paths_alpha = 1.0
        elif self._play_paths:
            if self._paths_settle_t is None:
                self._paths_settle_t = t
            dt = t - self._paths_settle_t
            self._paths_alpha = 1.0 if dt < 3.0 else max(0.0, 1.0 - (dt - 3.0))
            if self._paths_alpha <= 0.0:
                self._play_paths.clear()
        self._prev_shot_state = shot_state
        from .balls import pool_ball_bgr
        for tr in tracks:
            if tr.misses > 0 or (abs(tr.vx) + abs(tr.vy)) < 1.2:
                continue
            e = self._play_paths.get(tr.id)
            if e is None:
                e = self._play_paths[tr.id] = {"pts": [], "bgr": (200, 200, 200),
                                               "cue": False}
            # identity can firm up mid-roll — keep colour/cue flag current
            if tr.number == 0:
                e["cue"], e["bgr"] = True, (250, 250, 250)
            elif tr.number > 0:
                e["bgr"] = pool_ball_bgr(tr.number)
            elif not e["cue"]:
                e["bgr"] = tuple(int(v) for v in tr.bgr)
            pts = e["pts"]
            q = (float(tr.x), float(tr.y))
            if (not pts or abs(pts[-1][0] - q[0]) + abs(pts[-1][1] - q[1]) >= 1.5) \
                    and len(pts) < 800:
                pts.append(q)

    def view_tracks(self, tracks):
        """Velocity-extrapolated copies for RENDERING between async detection
        updates: detection lands ~10-14x/s while display runs at camera rate,
        so rolling balls glide instead of stepping. Only clearly-moving,
        currently-seen balls are extrapolated; the real tracks are untouched
        (events/state consume those)."""
        k = min(6, self._frames_since_ingest)
        if k <= 0:
            return tracks
        from dataclasses import replace as _dc_replace
        out = []
        for tr in tracks:
            if tr.misses == 0 and (abs(tr.vx) + abs(tr.vy)) > 1.0:
                tr = _dc_replace(tr, x=tr.x + tr.vx * k, y=tr.y + tr.vy * k)
            out.append(tr)
        return out

    def _project_raw_to_rect(self, raw_dets, calib, frame_shape=None):
        """Map RAW-frame detections into the rectified plane (via calib.H) so the
        tracker + bird's-eye schematic (both rectified-space) consume them.

        Includes the ball-height parallax correction: the homography maps the
        CLOTH plane, but a ball's centre sits one radius above it, so an oblique
        camera projects every centre radially outward from itself — rail balls
        landed visibly IN the rail on the overhead (up to ~2 ball diameters at
        the far rail). With the camera position recovered from the homography,
        sliding each point back along the camera ray by radius/height fixes the
        bias everywhere (and tightens pocket-capture geometry for free)."""
        if not raw_dets:
            return []
        from .rectify import project_points
        pts = np.array([[d.x, d.y] for d in raw_dets], np.float64)
        rect = project_points(pts, calib.H)
        off = np.array([[d.x + max(d.radius, 1.0), d.y] for d in raw_dets], np.float64)
        rect_off = project_points(off, calib.H)
        cam = self._camera_position(calib, frame_shape) if frame_shape else None
        if cam is not None:
            r_ball = expected_ball_radius_px(calib.table, self.settings.table.size)
            shrink = max(0.0, 1.0 - r_ball / float(cam[2]))
            rect = cam[:2] + (rect - cam[:2]) * shrink
        out = []
        for d, (rx, ry), (ox, oy) in zip(raw_dets, rect, rect_off, strict=False):
            out.append(Detection(float(rx), float(ry), float(np.hypot(ox - rx, oy - ry)),
                                 d.bgr, d.cls, d.score, number=d.number))
        return out

    def _camera_position(self, calib, frame_shape) -> np.ndarray | None:
        """Camera centre in rect coords for the parallax correction, cached per
        homography (recomputed only when the lock changes). None = no correction
        (degenerate pose, implausible height, or the feature flag is off)."""
        if not getattr(self.settings.balls, "parallax_correction", True):
            return None
        # Overhead camera: homography pose decomposition is DEGENERATE for a
        # fronto-parallel view, so the camera position can't be recovered — but
        # parallax still exists (the lens is a point ~5-6ft up; rail balls are
        # viewed obliquely and project ~0.25" outward, measured live: a
        # cushion-resting ball at 0.77r instead of 1.0r). Use a synthetic pose:
        # directly above the table centre at the configured lens height.
        if getattr(self.settings.camera, "overhead", False):
            calib_t = calib.table
            h_in = float(getattr(self.settings.camera, "height_in", 0.0) or 0.0)
            if h_in < 12.0:
                return None
            from ..config import _BED_SHORT_IN
            bed_in = _BED_SHORT_IN.get(self.settings.table.size, 46.0)
            px_per_in = calib_t.play_w / max(1.0, bed_in) if calib_t.play_w < calib_t.play_h \
                else calib_t.play_h / max(1.0, bed_in)
            cx = (calib_t.x0 + calib_t.x1) / 2.0
            cy = (calib_t.y0 + calib_t.y1) / 2.0
            return np.array([cx, cy, h_in * px_per_in])
        key = (calib.H.tobytes(), frame_shape[1], frame_shape[0])
        if self._cam_pose is not None and self._cam_pose[0] == key:
            return self._cam_pose[1]
        from .rectify import estimate_camera_position
        cam = estimate_camera_position(calib.Hinv, (frame_shape[1], frame_shape[0]))
        if cam is not None:
            # plausibility: the camera should sit somewhere between "just above
            # the table" and "gymnasium ceiling", in rect-px units
            long_side = float(max(calib.table.play_w, calib.table.play_h))
            if not (0.1 * long_side <= cam[2] <= 20.0 * long_side):
                log.info("parallax: implausible camera height %.0f px — disabled", cam[2])
                cam = None
            else:
                log.info("parallax: camera at (%.0f, %.0f) height %.0f rect px",
                         cam[0], cam[1], cam[2])
        self._cam_pose = (key, cam)
        return cam

    def _warp_gray_roi(self, frame: np.ndarray, calib) -> np.ndarray | None:
        """Warp ONLY the playing-area ROI of the gray frame into rectified space
        (for motion energy). Composing a translation into H and warping the
        1-channel ROI is ~3x cheaper than warping the full 3-channel bird's-eye;
        values match gray-of-warp to within interpolation rounding."""
        tbl = calib.table
        x0, y0 = int(tbl.x0), int(tbl.y0)
        w, h = int(tbl.x1) - x0, int(tbl.y1) - y0
        if w <= 0 or h <= 0:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        T = np.array([[1.0, 0.0, -x0], [0.0, 1.0, -y0], [0.0, 0.0, 1.0]])
        return cv2.warpPerspective(gray, T @ calib.H, (w, h), flags=cv2.INTER_LINEAR)

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
        st: dict[str, float] = {}  # per-stage ms — perf HUD + tools/bench_pipeline.py
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
            t0 = time.perf_counter()
            det_frame = self._stabilize(frame)
            st["median"] = (time.perf_counter() - t0) * 1000.0

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
                    self._apply_alignment_grid(res)
                    if self.settings.ui.schematic_birdseye:
                        res.rect_bgr = render_schematic(
                            self._default_preview_table(), [], accent=self.settings.ui.accent,
                            show_traj=False, show_ids=False)
                else:
                    res.status = "calibrating"
                    res.frame_bgr = frame
                self._apply_alignment_grid(res)
                self._last_ms = (time.perf_counter() - t_start) * 1000.0
                return res

        calib = self.calib.calib
        res.corners = calib.corners
        res.table = calib.table

        ui = self.settings.ui
        # The full-frame 3-channel bird's-eye warp is only needed when the warped
        # CAMERA image is displayed (schematic off). The default schematic view
        # renders from state, so skip the warp entirely — motion energy gets its
        # own cheap gray-ROI warp below. Independent of ``annotate``: Training
        # (label mode) keeps the same schematic bird's-eye as the Sandbox.
        need_rect_bgr = not ui.schematic_birdseye
        t0 = time.perf_counter()
        rect = self.calib.rectify(frame) if need_rect_bgr else None
        if need_rect_bgr and rect is None:
            res.status = "calibrating"
            return res
        st["warp"] = (time.perf_counter() - t0) * 1000.0

        if not detect:
            # Display-only frame: reuse the current tracks, don't re-detect.
            # (Live async mode lands here every frame — detections arrive via
            # ingest_raw_detections from the worker thread's results.)
            tracks = self.tracker.tracks
            detections = self._last_detections
            res.raw_dets = self._last_raw_dets   # labeller sees the async dets
            self._frames_since_ingest += 1
        else:
            t0 = time.perf_counter()
            # Detect on the RAW frame, project results into the rectified plane so
            # tracking + the bird's-eye schematic consume rectified-space points.
            if self._strategy is not None:
                try:
                    raw_dets = self._strategy.detect(det_frame, calib)
                except Exception as exc:  # noqa: BLE001 - a bad frame must not kill the loop
                    log.debug("detector failed on a frame: %s", exc)
                    raw_dets = []
                res.raw_dets = list(raw_dets)   # camera-coord, for the in-app labeller
            else:
                raw_dets = []
            st["detect"] = (time.perf_counter() - t0) * 1000.0
            t0 = time.perf_counter()
            tracks, detections = self._apply_detections(raw_dets, calib, frame.shape)
            st["track"] = (time.perf_counter() - t0) * 1000.0
        res.tracks = tracks
        res.detections = detections
        res.n_balls = len(tracks)

        # motion energy: percentage of playing-area pixels that changed
        # *significantly* between frames. This discriminates a real moving ball
        # (a tight cluster of big changes) from compression/lighting flicker
        # (scattered small changes), unlike a plain mean difference.
        t0 = time.perf_counter()
        if rect is not None:
            # full bird's-eye already computed for display — reuse it
            tbl = calib.table
            crop = rect[int(tbl.y0):int(tbl.y1), int(tbl.x0):int(tbl.x1)]
            roi = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.size else None
        else:
            roi = self._warp_gray_roi(frame, calib)
        if (roi is not None and self._prev_gray is not None
                and self._prev_gray.shape == roi.shape):
            motion = float((cv2.absdiff(roi, self._prev_gray) > 25).mean()) * 100.0
        else:
            motion = 0.0
        self._prev_gray = roi
        st["motion"] = (time.perf_counter() - t0) * 1000.0

        # extra modalities for evidence fusion: background-subtraction foreground
        # area + coherent optical-flow activity (both on a downscaled ROI). These
        # (MOG2 + Farneback) are the heaviest per-frame ops and feed ONLY shot
        # detection, so they run only when fusion is enabled — off by default
        # while cue-ball tracking (M2) is the focus, freeing the real-time budget.
        evidence = {"motion": motion}
        if self.settings.detection.use_fusion and roi is not None:
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
            self._update_play_paths(tracks, res.shot_state, t)
            res.diag = dict(self.shots.last_diag)
            res.diag["ms"] = round(self._last_ms, 1)
            res.diag["fps"] = int(1000 / self._last_ms) if self._last_ms > 0.1 else 0
        if res.diag:  # not while paused — an empty diag keeps the debug HUD off
            res.diag["stages"] = self._last_stages  # prev frame's, for the HUD

        # periodic deviation watchdog (cheap: only every N frames)
        if self._frame_idx % self._deviation_every == 0:
            self.calib.check_deviation(frame, self.settings)
            if self.calib.deviated and self.settings.table.auto_relock:
                log.info("Auto-relocking table after deviation")
                self.request_recalibration()
            # Impossible-geometry watchdog: a SETTLED ball resting beyond the
            # cushion-nose bounds cannot physically exist — it means the table
            # lock has drifted (seen live: the animation drew the cue ball
            # inside the cushion while a fresh calibration placed it perfectly).
            # Sustained impossibility -> relock.
            #
            # EXCEPT in a pocket. A ball sitting in a jaw IS legitimately beyond
            # the nose bounds, so counting it as "impossible" made a perfectly
            # good lock look drifted: measured on the rig, one jawed ball fired
            # this every 3-6 minutes, and each relock clears the lock for ~0.75s
            # (request_recalibration -> calib.clear) before the new one lands —
            # which is the "table not detected" flicker. Calibration itself never
            # failed once. Pocket zones are generous (2x radius) because a jawed
            # ball straddles the boundary by definition.
            tbl2 = calib.table
            r_wd = expected_ball_radius_px(tbl2, self.settings.table.size)
            bad = sum(1 for tr in tracks
                      if tr.misses == 0 and abs(tr.vx) + abs(tr.vy) < 1.0
                      and not tbl2.on_table(tr.x, tr.y, margin=-0.4 * r_wd)
                      and tbl2.pocket_at(tr.x, tr.y, scale=2.0) is None)
            if bad:
                self._impossible_clear = 0
                self._impossible_streak += 1
                if self._impossible_streak >= 4 and self.settings.table.auto_relock:
                    self._impossible_streak = 0
                    # BOUNDED. Relocking cannot move a real ball: one resting on
                    # a rail or in a pocket jaw is beyond the nose bounds no
                    # matter how well the table is locked, so an unbounded
                    # watchdog relocks forever (seen on the rig: every ~4.2s,
                    # and each relock wipes tracker + shot state). After a
                    # couple of fruitless attempts, conclude the geometry is
                    # right and the BALL is genuinely off the playing surface.
                    if self._impossible_relocks >= _MAX_FRUITLESS_RELOCKS:
                        if not self._impossible_gave_up:
                            log.warning(
                                "%d settled ball(s) beyond the nose bounds after "
                                "%d relocks — leaving the lock alone (a ball is "
                                "probably resting on a rail or in a jaw)",
                                bad, self._impossible_relocks)
                            self._impossible_gave_up = True
                    else:
                        self._impossible_relocks += 1
                        log.info("Auto-relocking (%d/%d): %d settled ball(s) resting "
                                 "beyond the nose bounds — the lock has drifted",
                                 self._impossible_relocks, _MAX_FRUITLESS_RELOCKS, bad)
                        self.request_recalibration()
            else:
                self._impossible_streak = 0
                # Only a sustained clean stretch re-arms the watchdog, so a ball
                # briefly leaving view can't reset the budget and restart the loop.
                self._impossible_clear += 1
                if self._impossible_clear >= _IMPOSSIBLE_REARM:
                    self._impossible_relocks = 0
                    self._impossible_gave_up = False
        res.deviated = self.calib.deviated
        res.status = "deviated" if self.calib.deviated else "tracking"

        t0 = time.perf_counter()
        overlays = annotate and ui.show_overlays
        # Draw every ball at its known physical radius (unless the raw-size debug
        # toggle is on) so the overhead shows uniform regulation balls.
        norm_r = (expected_ball_radius_px(calib.table, self.settings.table.size)
                  if (ui.normalize_ball_size and not ui.show_raw_detection_size) else None)
        # Bird's-eye: a clean rendered schematic (proportional) by default, rather
        # than the warped/clipped camera image. Also used in Training/label mode
        # so both tabs share the same camera + animated-schematic layout.
        vtracks = self.view_tracks(tracks)
        if ui.schematic_birdseye:
            res.rect_bgr = render_schematic(
                calib.table, vtracks, accent=ui.accent,
                play_paths=self._play_paths if ui.show_trajectories else None,
                paths_alpha=self._paths_alpha,
                show_traj=ui.show_trajectories, show_ids=ui.show_ball_ids,
                debug=ui.debug_overlay, detections=detections, diag=res.diag,
                measured_colors=ui.measured_ball_colors, fixed_radius=norm_r,
            )
        elif overlays:
            res.rect_bgr = draw_rectified(
                rect, vtracks, calib.table, show_traj=ui.show_trajectories,
                show_ids=ui.show_ball_ids, accent=ui.accent,
                measured_colors=ui.measured_ball_colors, fixed_radius=norm_r,
            )
        else:
            res.rect_bgr = rect

        # Live camera view keeps the real feed (with light overlay unless off).
        if overlays:
            res.frame_bgr = draw_perspective(
                frame, calib.corners, vtracks, calib.Hinv, accent=ui.accent,
                table=calib.table,
            )
        else:
            res.frame_bgr = frame
        st["render"] = (time.perf_counter() - t0) * 1000.0
        self._apply_alignment_grid(res)
        self._last_ms = (time.perf_counter() - t_start) * 1000.0
        self._last_stages = {k: round(v, 2) for k, v in st.items()}
        res.diag["stages"] = self._last_stages  # this frame's — for the bench
        return res

    # ------------------------------------------------------------------ #
    def _acquire_calibration(self, frame: np.ndarray) -> bool:
        """Restore a saved calibration if available + matching, else detect and
        persist a fresh one."""
        if (not self._tried_load and self.settings.table.persist_calibration
                and self.source):
            self._tried_load = True
            if self.calib.try_load(CALIBRATION_PATH, self.source, frame.shape, self.settings):
                # Trust, but verify against what's actually in the frame RIGHT
                # NOW — a stale lock (camera nudged since it was saved) would
                # otherwise render a wrong table for many seconds before the
                # watchdog catches up.
                if self.calib.validate_against(frame, self.settings):
                    return True
                self.calib.clear()  # stale — fall through to a fresh lock
        if not self.calib.calibrate(frame, self.settings):
            return False
        if self.settings.table.persist_calibration and self.source:
            self.calib.save(CALIBRATION_PATH, self.source, frame.shape, self.settings)
        return True
