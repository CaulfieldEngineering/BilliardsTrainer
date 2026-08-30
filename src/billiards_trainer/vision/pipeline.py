"""Per-frame orchestration: calibrate once, then detect → track → events.

This is the single object the capture/worker thread drives. It owns the
calibration, detector, tracker, and shot detector, and returns a
``PipelineResult`` describing the table state for the current frame. It does NOT
touch Qt or the DB — those are wired in the controller, keeping this testable.
"""

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config import CALIBRATION_PATH, Settings
from ..core.geometry import TableModel, expected_ball_radius_px
from ..core.types import BallClass, Detection, Track
from ..events.shot_detector import ShotDetector, ShotEvent
from .background import BackgroundModel, downscale, flow_activity
from .calibration import CalibrationManager
from .overlay import draw_perspective, draw_rectified, render_schematic
from ..measure.tracker import MotionTracker

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
    #: Hand-context for the sidecar (v2): which tracks are foreign-adjacent
    #: (carried) this frame, and the bed fraction covered by hands/arms.
    #: The recall audit needs these to tell a stroke from ball-gathering.
    carried_ids: set = field(default_factory=set)
    foreign_frac: float = 0.0


#: vacant DETECT frames before a numbered track lets go of its number.
#: Ten (~1s at the 10Hz detect cadence) is long enough to ride out the
#: detector blinking on a resting ball, and far short of the 60 frames
#: (~6s) it takes to kill the track — a whole shot is only ~7s.
_RELEASE_AFTER = 10


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
        self.tracker = MotionTracker()   # the ONE tracker (round 39)
        # THE Measurement Core seam (docs/ARCHITECTURE.md L1): presence +
        # the hardened shadow tracker + divergence counters. Consumers
        # read core.present / core.tracks — nobody keeps a private copy.
        from ..measure.core import MeasurementCore
        self.core = MeasurementCore()
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
        self._last_schematic = None
        self._bg = BackgroundModel()
        self._prev_small = None
        self._last_flow = 0.0
        self._last_ms = 0.0
        self._last_stages: dict = {}  # previous frame's per-stage ms (perf HUD)
        self._frame_ring: deque | None = None  # temporal-median buffer
        # Cached playback (Joe: "we shouldn't process anything in realtime
        # during playback"): when a SidecarReader is attached, detection,
        # tracking, and shot detection are BYPASSED — tracks come from the
        # cache, interpolated to the frame clock, and this pipeline only
        # calibrates (once) and renders. Smooth by construction.
        self.playback_cache = None
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
            if strat is not None and hasattr(strat, "inference_provider"):
                # "cpu" moves inference off the iGPU so the desktop compositor
                # keeps its 3D engine (input-lag fix — see DetectionSettings).
                strat.inference_provider = str(
                    getattr(self.settings.detection, "inference_provider", "auto"))
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
    def _process_cached(self, frame, t: float, res: PipelineResult, t_start) -> PipelineResult:
        """Playback from the analysis sidecar: no models, no tracker — look
        up, interpolate, render. Schematic refreshes at half rate; the
        perspective overlay is cheap enough for every frame."""
        calib = self.calib.calib
        ui = self.settings.ui
        # ONE CLOCK: t is VIDEO time; pre-origin-fix sidecars run ahead of
        # it — query in the sidecar's own clock or the drawn state trails
        # the picture by the session's offset (2026-08-25)
        tracks = self.playback_cache.tracks_at(
            t + self.playback_cache.video_time_offset())
        res.status = "tracking"
        res.tracks = tracks
        res.n_balls = len(tracks)
        res.table = calib.table
        res.corners = calib.corners
        norm_r = expected_ball_radius_px(calib.table, self.settings.table.size)             if getattr(ui, "normalize_ball_size", True) else None
        self._cached_flip = not getattr(self, "_cached_flip", False)
        if ui.schematic_birdseye:
            if self._cached_flip or self._last_schematic is None:
                res.rect_bgr = render_schematic(
                    calib.table, tracks, accent=ui.accent,
                    show_traj=False, show_ids=ui.show_ball_ids,
                    measured_colors=ui.measured_ball_colors, fixed_radius=norm_r)
                self._last_schematic = res.rect_bgr
            else:
                res.rect_bgr = self._last_schematic
        if getattr(ui, "show_overlays", True):
            res.frame_bgr = draw_perspective(
                frame, calib.corners, tracks, calib.Hinv,
                accent=ui.accent, table=calib.table)
        else:
            res.frame_bgr = frame
        self._last_ms = (time.perf_counter() - t_start) * 1000.0
        res.diag = {"cached": True, "ms": round(self._last_ms, 1)}
        return res

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

    def prepare_detections(self, raw_dets, calib, frame_shape, frame=None,
                           refresh_foreign: bool = False):
        """Project raw-frame detections to rect space and run EVERY sanity
        filter (size prior, foreign veto, rigid-body repair, number
        arbitration, confidence floor, geometry, blur recovery) — the full
        bought stack, WITHOUT touching the tracker. Shared by the live
        path (_apply_detections) and the M1 measurement engine, which
        skipped it once and re-learned why it exists (229px coordinate
        offset AND every filtered phantom class at once)."""
        # THE ENGINE WAS BLIND TO HANDS AND STICKS (bench round 7,
        # vision-verified 2026-08-28): the foreign/glove veto below reads
        # self._foreign_last, which only the LIVE process() loop ever
        # computed - so in offline re-processing it was always None and
        # the veto was inert. Overlay proof: at 170.5s the engine drew
        # NINE phantom balls down the cue stick's shaft and labelled the
        # stick's tip "the cue ball" while the real cue ball sat
        # unnamed. Offline callers pass refresh_foreign=True and get the
        # same protection the live feed has (one box, both feeds).
        if refresh_foreign and frame is not None:
            try:
                self._foreign_state(frame, calib)
            except Exception:  # noqa: BLE001 - a mask failure must not
                pass          # kill the frame; the veto just stays off
        detections = self._project_raw_to_rect(raw_dets, calib, frame_shape)
        # Physical-size prior: reject detections whose radius is far from the
        # known ball radius. This used to be skipped for model-based detectors
        # on the assumption that a model "already validates ball-ness" — which
        # session footage disproved: the model detects the drill position
        # markers stuck to the felt (donut stickers, chalk dots) as balls, some
        # at 0.88 confidence, and being static they confirm, settle, and live
        # nearly forever in the tracker. Markers project to well under 0.7x the
        # geometric ball radius while real balls stay above it, so the size
        # prior — with a band matched to each detector's radius behaviour — is
        # exactly the right knife. (One day the markers become a feature:
        # detect them as a marker class for DrillRoom-style drills.)
        exp_r = expected_ball_radius_px(calib.table, self.settings.table.size)
        if getattr(self._strategy, "model_based", False):
            lo_f = getattr(self.settings.balls, "model_size_lo", 0.72)
            # Ceiling 1.75, not 1.55: rectification is uniform only for the
            # table PLANE — a ball's disc rides above it, so the warp
            # inflates it toward the corners. The 4 sat in the top-right
            # corner projecting at 1.59x expected radius and was discarded
            # by the old 1.55 cap EVERY FRAME of a session (a real ball,
            # half a pixel over the line, nameless for a minute of footage).
            # Merged two-ball blobs project near 2.0x and stay rejected.
            hi_f = getattr(self.settings.balls, "model_size_hi", 1.75)
            if exp_r > 2.0 and lo_f > 0:
                lo, hi = exp_r * lo_f, exp_r * hi_f
                detections = [d for d in detections if lo <= d.radius <= hi]
        else:
            tol = getattr(self.settings.balls, "size_prior_tol", 0.25)
            if exp_r > 2.0 and tol > 0:
                lo, hi = exp_r * (1.0 - tol), exp_r * (1.0 + tol)
                detections = [d for d in detections if lo <= d.radius <= hi]
        # Hands are not balls: a detection whose centre lies inside a KEPT
        # foreign blob (hand/arm-scale by construction — foreign_mask floors
        # out ball-sized blobs, so a lone 8-ball never lands here) is a glove
        # knuckle / wrist / cue butt. Measured on session-20260802-173553: a
        # gloved bridge hand resting on the cushion for rail shots tracked as
        # a resting "#4" flanked by two unknown ghosts — 25 impossible
        # overlaps, the sole G4 per-session blocker. A real ball the hand is
        # touching merges into the blob and is dropped too, which is correct:
        # its track coasts on the occlusion budget and resumes on reappearance.
        foreign = getattr(self, "_foreign_last", None)
        if detections and foreign and foreign[1] is not None:
            _ffrac, fmask, fs, fx0, fy0 = foreign
            mh, mw = fmask.shape[:2]
            kept = []
            for d in detections:
                mx, my = int((d.x - fx0) * fs), int((d.y - fy0) * fs)
                if 0 <= mx < mw and 0 <= my < mh and fmask[my, mx]:
                    continue
                kept.append(d)
            detections = kept
        # Rigid-body repair: two balls cannot interpenetrate, so when two
        # ball-sized detections sit closer than one diameter, either they are
        # two REAL touching balls whose centroids were pulled inward (daylight
        # shadow bridging a rack pair drags both centres 10-35% together), or
        # they are ONE ball detected twice. The two cases demand opposite
        # treatment, and the first corpus run after unconditional push-apart
        # proved it: repairing duplicates LAUNDERS them into legal-looking
        # pairs (phantom tracks, stripe misreads — 22.6/1k on one session).
        # Identity is the discriminator: two DISTINCT confident numbers is
        # evidence of two real balls -> push apart to touching. Same number or
        # unknown -> the ensemble numbers only one box per physical ball, so
        # treat as a duplicate and keep the stronger. Iterated because a
        # repair can re-tighten a neighbouring pair in a chain (a rack).
        if exp_r > 2.0 and len(detections) >= 2:
            target = 2.0 * exp_r
            drop: set[int] = set()
            for _ in range(8):
                moved = False
                for i in range(len(detections)):
                    if i in drop:
                        continue
                    for j in range(i + 1, len(detections)):
                        if j in drop:
                            continue
                        a, b = detections[i], detections[j]
                        dx, dy = b.x - a.x, b.y - a.y
                        d = math.hypot(dx, dy)
                        if d >= target:
                            continue
                        distinct = (a.number >= 0 and b.number >= 0
                                    and a.number != b.number)
                        if distinct and d >= 1.0:
                            push = 0.5 * (target - d) / d
                            a.x -= dx * push
                            a.y -= dy * push
                            b.x += dx * push
                            b.y += dy * push
                            moved = True
                        else:
                            loser = j if a.score >= b.score else i
                            drop.add(loser)
                            if loser == i:
                                break   # a is gone; stop comparing against it
                if not moved:
                    break
            if drop:
                detections = [d for k, d in enumerate(detections) if k not in drop]
        # Uniqueness at the source: one cue ball, one of each number, per frame.
        # When two detections claim the same identity the weaker one is either a
        # phantom or a misread — either way its CLAIM is wrong. Demote the claim
        # (not the detection: the ball may be real, just mislabelled) so a
        # sticker or glare blob can never outvote the real ball downstream.
        # This is the frame-level complement of the tracker's _arbitrate_numbers.
        best_by_num: dict[int, Detection] = {}
        for d in detections:
            if d.number >= 0:
                cur = best_by_num.get(d.number)
                if cur is None or d.score > cur.score:
                    best_by_num[d.number] = d
        cues = [d for d in detections if d.cls == BallClass.CUE]
        best_cue = max(cues, key=lambda d: d.score) if cues else None
        for d in detections:
            if d.number >= 0 and best_by_num.get(d.number) is not d:
                d.number = -1
            if d.cls == BallClass.CUE and d is not best_cue:
                d.cls = BallClass.UNKNOWN
                d.number = -1
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
            # A pocketed ball resting in the basket is visible from overhead —
            # past the cushion line, inside a pocket zone — and classifies
            # confidently (it IS a real ball), so the class-gated void check
            # below never rejected it. Measured on session-20260729: a #5/#6
            # pair sat in the side-pocket basket for 40+ seconds as settled
            # tracks, firing overlapping_balls every frame. Off the bed AND in
            # a pocket zone = pocketed, whatever the class says. Balls hanging
            # at the jaw sit ON the bed side of the nose line and are kept.
            in_bed = tbl.x0 <= d.x <= tbl.x1 and tbl.y0 <= d.y <= tbl.y1
            if not in_bed and tbl.pocket_at(d.x, d.y, scale=1.4) is not None:
                continue
            if (d.cls in (BallClass.EIGHT, BallClass.UNKNOWN)
                    and tbl.pocket_at(d.x, d.y, scale=0.9) is not None):
                continue
            if (d.cls == BallClass.UNKNOWN
                    and any((d.x - sx) ** 2 + (d.y - sy) ** 2 <= spot_r2
                            for sx, sy in spots)):
                continue
            # JAW PHANTOM (bench 220247, vision-verified 2026-08-28: the
            # bottom-left pocket leather detected as a 0.48-score SOLID
            # just inside the bed at the jaw, alive for 60+ seconds -
            # spawning ghost tracks, phantom episodes and off-table
            # trails). Near a pocket, a LOW-CONFIDENCE read is the
            # pocket furniture; real balls - including jaw-hangers -
            # score like balls on this rig (>=0.85 typical).
            if (d.score < 0.60
                    and tbl.pocket_at(d.x, d.y, scale=2.2) is not None):
                continue
            kept.append(d)
        detections = kept
        # balls the detector lost to motion blur, recovered from the pixels
        if frame is not None and getattr(self.settings.detection,
                                         "blur_recovery", True):
            try:
                if getattr(self, "_blur", None) is None:
                    from .blur_recovery import BlurRecovery
                    self._blur = BlurRecovery()
                extra = self._blur.find(frame, calib, self.tracker, detections)
                # OFF by default: measured 2026-08-23 on the ground-truth
                # strike, the sweep FAILED both acceptance gates — 37
                # phantoms on a quiet table, and ball-4 moving coverage
                # DROPPED 8 -> 2 because sweep blobs register as coverage
                # and suppress the targeted recovery that was working.
                # Fix before re-enabling: run sweep AFTER find(), exclude
                # sweep emissions from every coverage test, re-verify.
                if getattr(self.settings.detection, "sweep_channel", False):
                    extra = extra + self._blur.sweep(frame, calib,
                                                     self.tracker, detections)
                if extra:
                    detections = detections + self._project_raw_to_rect(
                        extra, calib, frame_shape)
            except Exception:  # noqa: BLE001 - recovery must never break tracking
                log.exception("blur recovery failed")
        return detections

    def _apply_detections(self, raw_dets, calib, frame_shape, frame=None):
        """Prepared detections -> tracker update + vacancy pruning. The
        preparation stage lives in prepare_detections (shared with the
        M1 engine); this tail is the LIVE tracker's path only."""
        detections = self.prepare_detections(raw_dets, calib, frame_shape,
                                             frame=frame)
        tbl = calib.table
        exp_r = expected_ball_radius_px(tbl, self.settings.table.size)
        # ONE TRACKER (round 39). This was BallTracker; the offline engine
        # ran a different one, and measured head to head on identical
        # detections the live one named 57.6% of ball sightings correctly
        # against the engine's 98.9% - it called the striped 9 a "3" in
        # 221 of 221 samples, so the red 3 was never right either
        # (tools/tracker_bakeoff.py, tools/live_path_check.py). Live and
        # offline now differ only in where frames come from.
        self.tracker.set_geometry([(p.x, p.y) for p in tbl.pockets],
                                  float(tbl.pocket_radius))
        tracks = self.tracker.update(
            detections, float(getattr(self, "_last_t", 0.0)))
        # VACANCY PRUNING (Joe: "false positive cue balls and lingering cue
        # ball assumed positions"): a STILL track whose spot is plainly
        # visible — no detection near it AND no hand/foreign blob covering
        # it — is a ghost, not an occluded ball. The occlusion budget
        # exists for balls hidden by arms; it must not keep a picked-up
        # ball parked for minutes, nor keep a glove-born white blob alive
        # after the glove moves on. Numbered settled balls get patience
        # (flicker happens); unnumbered blobs get very little.
        self._vacant = getattr(self, "_vacant", {})
        foreign = getattr(self, "_foreign_last", None)

        def _covered(x: float, y: float) -> bool:
            # NEIGHBORHOOD test, not a single pixel: at address the stick
            # and bridge hand hide the cue while the exact centre pixel
            # stays outside the foreign blob — that killed a real resting
            # cue mid-address and derived a phantom scratch (Joe's shot 31,
            # first field session after this feature shipped).
            if not foreign or foreign[1] is None:
                return False
            _ff, fmask, fs, fx0, fy0 = foreign
            mh, mw = fmask.shape[:2]
            r = 1.5 * max(exp_r, 6.0)
            for dx, dy in ((0, 0), (r, 0), (-r, 0), (0, r), (0, -r)):
                mx = int((x + dx - fx0) * fs)
                my = int((y + dy - fy0) * fs)
                if 0 <= mx < mw and 0 <= my < mh and fmask[my, mx]:
                    return True
            return False

        nonfelt = getattr(self, "_nonfelt_last", None)

        def _spot_occupied(x: float, y: float) -> bool:
            # SPOT-OCCUPANCY (the 005647 lesson): Joe's address routine
            # holds the stick over resting balls LONGER than any patience
            # counter we dare set, and a stick is too thin for the blob-
            # floored foreign mask — so "no detection + no foreign cover"
            # still killed real resting balls and derived phantom
            # departures (machine 1/7 vs Joe's verdicts). Vacancy is a
            # claim about PIXELS, so ask the pixels: felt-coloured means
            # truly empty; anything else — ball, stick, ball-under-stick —
            # means the spot is NOT vacant, whatever the detector thinks.
            if nonfelt is None:
                return False
            rmask, rs, rx0, ry0 = nonfelt
            mh, mw = rmask.shape[:2]
            mx, my = int((x - rx0) * rs), int((y - ry0) * rs)
            if not (0 <= mx < mw and 0 <= my < mh):
                return False
            win = rmask[max(0, my - 1):my + 2, max(0, mx - 1):mx + 2]
            # a ball spans ~5px in the 160px-wide mask, so a centred 3x3
            # window lies inside its disc; 1/3 non-felt tolerates edge noise
            return win.size > 0 and float(win.mean()) >= 0.34

        near_r2 = (2.2 * max(exp_r, 6.0)) ** 2
        self._released = getattr(self, "_released", set())
        doomed: list[int] = []
        vacated: list[int] = []
        live_ids: set[int] = set()
        for tr in tracks:
            live_ids.add(tr.id)
            still = (abs(tr.vx) + abs(tr.vy)) < 1.0
            has_det = any((d.x - tr.x) ** 2 + (d.y - tr.y) ** 2 <= near_r2
                          for d in detections)
            # Spot-occupancy protects NUMBERED residents only: an unnumbered
            # blob on a shadowed patch must not become immortal — ghosts are
            # overwhelmingly unnumbered, real long-occluded balls numbered.
            protected = _covered(tr.x, tr.y) or (
                tr.number >= 0 and _spot_occupied(tr.x, tr.y))
            if still and not has_det and not protected:
                self._vacant[tr.id] = self._vacant.get(tr.id, 0) + 1
            else:
                self._vacant[tr.id] = 0
            # numbered balls get LONG patience: a cue at address sits
            # under the stick for many seconds with no detection and no
            # (blob-sized) foreign cover — 60 detect frames still kills a
            # true lingerer 25x faster than the occlusion budget did
            # Patience is about what this track IS, not what it currently
            # answers to. Releasing the number below must not quietly drop a
            # resting ball into the unnumbered bucket and kill it at 8 frames
            # instead of 60 — that would make letting go of a name 7x more
            # lethal than keeping it, the opposite of the intent.
            was_named = tr.number >= 0 or tr.id in self._released
            if self._vacant[tr.id] >= (60 if was_named else 8):
                doomed.append(tr.id)
            elif tr.number >= 0 and self._vacant[tr.id] >= _RELEASE_AFTER:
                # Its spot is bare felt and no detection is near it, so this
                # ball is demonstrably somewhere else — but the track keeps
                # its long occlusion patience because killing a resting ball
                # early is what produced phantom departures (spot-occupancy
                # was written for exactly that). Killing is dangerous;
                # letting go of the NUMBER is not. 005048 @233: the 4's
                # track correctly refuses the arriving cue ball, then sits
                # at the address spot holding number 4 for the whole 7s
                # shot, so the real 4 — found near the pocket by a fresh
                # track — can never be named and its path is lost.
                vacated.append(tr.id)
        self._vacant = {k: v for k, v in self._vacant.items()
                        if k in live_ids}
        # a track that got its number back (a detection returned) is no
        # longer "released", and dead ids must not leak
        self._released = {i for i in self._released if i in live_ids} - {
            t.id for t in tracks if t.number >= 0}
        if vacated:
            self.tracker.release_numbers(vacated)
            self._released.update(vacated)
        if doomed:
            self.tracker.remove_ids(doomed)
            tracks = [t for t in tracks if t.id not in doomed]

        # The PUBLISHED state must obey physics too. The detection-level repair
        # moves interpenetrating pairs apart, but a settled track's anti-shimmer
        # lock swallows that few-pixel correction as jitter, so the track pair
        # can stay frozen at impossible positions indefinitely (measured: two
        # balls parked touching in a pocket jaw during drills, flagged every
        # frame). Apply the same identity-gated projection to the track copies
        # we hand out — the tracker's internal state is untouched, so matching
        # behaviour doesn't change.
        if exp_r > 2.0 and len(tracks) >= 2:
            target = 2.0 * exp_r
            for _ in range(4):
                moved = False
                for i in range(len(tracks)):
                    for j in range(i + 1, len(tracks)):
                        a, b = tracks[i], tracks[j]
                        if not (a.number >= 0 and b.number >= 0
                                and a.number != b.number):
                            continue
                        dx, dy = b.x - a.x, b.y - a.y
                        d = math.hypot(dx, dy)
                        if d < 1.0 or d >= target:
                            continue
                        push = 0.5 * (target - d) / d
                        a.x -= dx * push
                        a.y -= dy * push
                        b.x += dx * push
                        b.y += dy * push
                        moved = True
                if not moved:
                    break
        self._last_detections = detections
        self._last_raw_dets = list(raw_dets)
        self._frames_since_ingest = 0
        # Feed the Measurement Core: the SAME prepared detections drive the
        # hardened shadow tracker, and the champion's emitted tracks drive
        # presence + divergence scoring. Async ingests land between frames,
        # so t is the last process() stamp (<=33ms stale — fine for
        # divergence counting; promoted output plumbs capture stamps, see
        # MEASUREMENT_CORE.md 0.1).
        t_now = getattr(self, "_last_t", -1.0)
        self.core.ingest([(d.x, d.y, d.radius,
                           int(getattr(d, "number", -1)))
                          for d in detections], t_now)
        self.core.observe_tracks(tracks, t_now)
        return tracks, detections

    def ingest_raw_detections(self, raw_dets, frame_shape) -> None:
        """Apply detector output produced OFF the display path (the live async
        worker). Runs on the pipeline's own thread via a queued slot, so all
        tracker/state mutation stays single-threaded."""
        calib = self.calib.calib
        if calib is None:
            return
        self._apply_detections(raw_dets, calib, frame_shape)
        # The tracks just changed, so the cached schematic is STALE. Live
        # display frames all carry detect=False (the async split), so without
        # this invalidation the bird's-eye rendered once at startup — an
        # empty table — and froze forever while the tracker published balls
        # (Joe, three times: "the schematic isn't showing any balls").
        # Re-render happens on the next display frame, i.e. the schematic
        # updates at detection cadence, which is exactly the old economy.
        self._last_schematic = None

    def _update_play_paths(self, tracks, shot_state: str, t: float) -> None:
        """Accumulate each moving ball's path for the CURRENT play. Paths hold
        through the settle for review, then FADE OUT starting 3s after all
        table movement stops (Joe's spec); a new play clears them instantly."""
        if shot_state == "moving" and self._prev_shot_state != "moving":
            # A new shot ARMED — but arming lags the strike by ~half a second
            # (banked ball-motion evidence), and the struck ball's first steps
            # are ALREADY in the paths. Clearing everything here amputated the
            # first half of every trail (Joe: "the trail is only the second
            # half of the movement"). Prune only STALE entries — leftovers of
            # the previous play — and keep anything that moved recently: that
            # IS this stroke's opening.
            stale = [tid for tid, e in self._play_paths.items()
                     if t - e.get("t_last", -1e9) > 1.5]
            for tid in stale:
                self._play_paths.pop(tid, None)
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
        from ..core.balls import pool_ball_bgr
        for tr in tracks:
            if tr.misses > 0 or (abs(tr.vx) + abs(tr.vy)) < 1.2:
                continue
            e = self._play_paths.get(tr.id)
            if e is None:
                e = self._play_paths[tr.id] = {"pts": [], "bgr": (200, 200, 200),
                                               "cue": False, "t_last": t}
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
                e["t_last"] = t

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
        from ..core.rectify import project_points
        pts = np.array([[d.x, d.y] for d in raw_dets], np.float64)
        rect = project_points(pts, calib.H)
        off = np.array([[d.x + max(d.radius, 1.0), d.y] for d in raw_dets], np.float64)
        rect_off = project_points(off, calib.H)
        cam = self._camera_position(calib, frame_shape) if frame_shape else None
        if cam is not None:
            r_ball = expected_ball_radius_px(calib.table, self.settings.table.size)
            shrink = max(0.0, 1.0 - r_ball / float(cam[2]))
            rect = cam[:2] + (rect - cam[:2]) * shrink
            # THE OFFSET POINT MUST MOVE WITH THE CENTRE (round 49). The
            # radius below is |rect_off - rect|, and this correction used
            # to slide only `rect`, so the radius was a distance between
            # two DIFFERENT coordinate frames. The error is directional -
            # the offset is +x in raw, so on the side where +x points
            # toward the camera nadir the radius came out SHORT and on
            # the far side LONG. Measured on the bench: the purple 4,
            # sitting still in plain sight near the left rail, was found
            # every frame at score 0.87 and correctly named 4, projected
            # to r=8.72 against a size floor of 8.94, and was DISCARDED
            # BY 0.22 PIXELS for 82 consecutive seconds (19s-101s, 81 of
            # the 88 blind checks on the whole clip). The same bug on the
            # other side is why the ceiling was raised to 1.75 for a ball
            # "in the top-right corner projecting at 1.59x" - that was a
            # bandage on this, not a property of the optics.
            rect_off = cam[:2] + (rect_off - cam[:2]) * shrink
        out = []
        for d, (rx, ry), (ox, oy) in zip(raw_dets, rect, rect_off, strict=False):
            out.append(Detection(float(rx), float(ry), float(np.hypot(ox - rx, oy - ry)),
                                 d.bgr, d.cls, d.score, number=d.number,
                                 # carry the MEASURED colour across the
                                 # projection. Dropping it here quietly
                                 # starved every consumer downstream:
                                 # Detection.measured_bgr is the only feed
                                 # for _Internal.colour_hist, so the
                                 # colour-consensus and colour-adoption
                                 # machinery in the tracker -- tested, and
                                 # hardened by review -- never once ran on
                                 # real footage. Measured on 005048 @233:
                                 # 0 of 6 detections carried a colour.
                                 measured_bgr=getattr(d, "measured_bgr", None)))
            out[-1].recovered_for = getattr(d, "recovered_for", None)
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
        from ..core.rectify import estimate_camera_position
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

    def _foreign_state(self, frame: np.ndarray, calib) -> tuple[float, object, float, float, float]:
        """(fraction, mask, scale, x0, y0) of foreign coverage on the bed.

        Computed on a TINY dedicated colour warp (~160px wide, scale composed
        into H) so it costs well under a millisecond and works on the
        schematic path, which never builds a full colour warp. Cached every
        3rd frame — presence is a slow signal. The mask geometry maps a
        rectified point (x, y) to mask pixel ((x - x0) * scale, (y - y0) * scale).
        """
        self._arm_tick = (getattr(self, "_arm_tick", 0) + 1) % 3
        if self._arm_tick != 1 and hasattr(self, "_foreign_last"):
            return self._foreign_last
        from .foreign import foreign_mask
        tbl = calib.table
        x0, y0 = tbl.x0, tbl.y0
        w, h = tbl.x1 - x0, tbl.y1 - y0
        if w <= 0 or h <= 0:
            self._foreign_last = (0.0, None, 1.0, 0.0, 0.0)
            self._nonfelt_last = None
            return self._foreign_last
        s = 160.0 / w
        S = np.array([[s, 0.0, -x0 * s], [0.0, s, -y0 * s], [0.0, 0.0, 1.0]])
        tiny = cv2.warpPerspective(frame, S @ calib.H, (160, max(1, int(h * s))),
                                   flags=cv2.INTER_LINEAR)
        frac, mask, raw = foreign_mask(tiny, None)
        self._foreign_last = (frac, mask, s, x0, y0)
        # Raw non-felt snapshot (pre-floor): vacancy pruning's spot-occupancy
        # test reads it — "is there ANYTHING at this spot", ball-sized or not.
        self._nonfelt_last = (raw, s, x0, y0) if raw is not None else None
        return self._foreign_last

    def _carried_ids(self, tracks, foreign) -> set[int]:
        """Track ids currently adjacent to a foreign blob (hand/arm/stick).

        A carried ball moves WITH the hand, so it is foreign-adjacent for its
        whole displacement; a struck ball leaves the stick within a frame or
        two. The shot detector counts only non-adjacent motion, which is what
        finally separates drills' ball-gathering from actual shots."""
        frac, mask, s, x0, y0 = foreign
        if mask is None or frac <= 0.0:
            return set()
        H, W = mask.shape[:2]
        out = set()
        for tr in tracks:
            mx = int((tr.x - x0) * s)
            my = int((tr.y - y0) * s)
            # neighbourhood ~1.5 ball radii at mask scale
            r = max(1, int(tr.radius * s * 1.5))
            x_lo, x_hi = max(0, mx - r), min(W, mx + r + 1)
            y_lo, y_hi = max(0, my - r), min(H, my + r + 1)
            if x_lo < x_hi and y_lo < y_hi and mask[y_lo:y_hi, x_lo:x_hi].any():
                out.add(tr.id)
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
        st: dict[str, float] = {}  # per-stage ms — perf HUD + tools/bench_pipeline.py
        self._frame_idx += 1
        self._last_t = t   # async ingest lands between frames; it borrows this
        res = PipelineResult(frame_bgr=frame)

        if not self.detect_enabled:
            self._last_ms = (time.perf_counter() - t_start) * 1000.0
            return self._preview_result(frame, res)

        if self.playback_cache is not None and self.calib.calib is not None:
            return self._process_cached(frame, t, res, t_start)

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
            tracks, detections = self._apply_detections(
                raw_dets, calib, frame.shape, frame=frame)
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
        self.last_motion = motion   # controller's idle throttle reads this
        st["motion"] = (time.perf_counter() - t0) * 1000.0

        # extra modalities for evidence fusion: background-subtraction foreground
        # area + coherent optical-flow activity (both on a downscaled ROI). These
        # (MOG2 + Farneback) are the heaviest per-frame ops and feed ONLY shot
        # detection, so they run only when fusion is enabled — off by default
        # while cue-ball tracking (M2) is the focus, freeing the real-time budget.
        foreign = self._foreign_state(frame, calib)
        evidence = {"motion": motion, "arm": foreign[0],
                    "carried_ids": self._carried_ids(tracks, foreign)}
        res.carried_ids = set(evidence["carried_ids"])
        res.foreign_frac = float(foreign[0])
        if self.settings.detection.use_fusion and roi is not None:
            small = downscale(roi)
            fg = self._bg.update(small)
            # optical flow is the costliest + least-weighted signal — every 2nd frame
            if self._frame_idx % 2 == 0:
                self._last_flow = flow_activity(self._prev_small, small)
            self._prev_small = small
            evidence["flow"] = self._last_flow * 100.0
            evidence["fg"] = fg * 100.0

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
            # Playback coast frames (detect=False) reuse the last schematic:
            # nothing detection-visible changed, and this render costs ~20ms
            # of the 33ms frame budget — re-rendering it every frame is half
            # of why watching a clip back ran at 9fps (Joe: "super super
            # slow"). The bird's-eye updates at detection cadence (~10Hz);
            # the video pane stays full rate.
            if not detect and self._last_schematic is not None:
                res.rect_bgr = self._last_schematic
            else:
                res.rect_bgr = render_schematic(
                    calib.table, vtracks, accent=ui.accent,
                    play_paths=self._play_paths if ui.show_trajectories else None,
                    paths_alpha=self._paths_alpha,
                    show_traj=ui.show_trajectories, show_ids=ui.show_ball_ids,
                    debug=ui.debug_overlay, detections=detections, diag=res.diag,
                    measured_colors=ui.measured_ball_colors, fixed_radius=norm_r,
                )
                self._last_schematic = res.rect_bgr
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

    def _dump_calib_debug(self, frame: np.ndarray, tag: str) -> None:
        """Ground-truth breadcrumbs for calibration failures: the ACTUAL frame
        the pipeline saw, with the live felt corners (if any) and the current/
        restored lock drawn in. Throttled hard — this is evidence, not a log.
        (Written after an evening of guessing at coordinate spaces from three
        different code paths; one image answers what ten traces did not.)"""
        import time as _t
        now = _t.monotonic()
        if now - getattr(self, "_calib_dbg_t", 0.0) < 10.0 \
                or getattr(self, "_calib_dbg_n", 0) >= 6:
            return
        self._calib_dbg_t = now
        self._calib_dbg_n = getattr(self, "_calib_dbg_n", 0) + 1
        try:
            from pathlib import Path

            from .felt import detect_felt
            out = Path("_eval") / "calib_debug"
            out.mkdir(parents=True, exist_ok=True)
            img = frame.copy()
            felt = detect_felt(img, self.settings.felt)
            if getattr(felt, "has_corners", False):
                cv2.polylines(img, [np.int32(felt.corners.reshape(-1, 1, 2))],
                              True, (0, 255, 0), 2)      # live felt = green
            if self.calib.calib is not None:
                cv2.polylines(img, [np.int32(self.calib.calib.corners.reshape(-1, 1, 2))],
                              True, (0, 0, 255), 2)      # current lock = red
            cv2.putText(img, f"{tag} shape={frame.shape[1]}x{frame.shape[0]}",
                        (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.imwrite(str(out / f"{tag}_{self._calib_dbg_n}.png"), img)
            log.info("calib debug frame dumped: %s #%d", tag, self._calib_dbg_n)
        except Exception:  # noqa: BLE001 - diagnostics must never hurt the app
            pass

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
                self._dump_calib_debug(frame, "restore_rejected")
                self.calib.clear()  # stale — fall through to a fresh lock
        if not self.calib.calibrate(frame, self.settings):
            self._dump_calib_debug(frame, "no_lock")
            return False
        if self.settings.table.persist_calibration and self.source:
            self.calib.save(CALIBRATION_PATH, self.source, frame.shape, self.settings)
        return True
