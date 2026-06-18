"""Capture + pipeline controller.

Lives on its own QThread (CV must not run on the UI thread — a core lesson from
the original prototype). A QTimer in that thread ticks the capture→pipeline loop;
control methods are invoked via queued connections so they execute in the worker
thread. The controller owns the DB session and shot clock, records shots, and
emits signals the UI renders.

Reliability behaviours live here: tolerating transient camera read failures,
buffering recent frames for instant replay, and logging every shot event.
"""

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

import cv2
import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot

from ..config import EXPORTS_DIR, SHOTLOG_PATH, Settings
from ..db.repository import Repository
from ..events.shot_detector import ShotEvent
from ..game.shot_clock import ShotClock
from ..version import __version__
from ..vision.felt import felt_from_point
from ..vision.pipeline import Pipeline

log = logging.getLogger("controller")

# How many consecutive empty reads from a *live* camera we tolerate before
# declaring it disconnected (~2 s at 30 fps). Transient empty reads are normal.
_CAMERA_MISS_TOLERANCE = 60


@dataclass
class FramePacket:
    perspective: np.ndarray | None = None
    birdseye: np.ndarray | None = None
    status: str = "idle"
    fps: float = 0.0
    n_balls: int = 0
    shot_state: str = "settled"
    clock_remaining: float = 0.0
    clock_enabled: bool = False
    clock_warning: bool = False
    deviated: bool = False
    tracks: list = field(default_factory=list)
    raw_dets: list = field(default_factory=list)   # camera-coord dets + guessed numbers (labelling)


class PipelineController(QObject):
    frame_ready = Signal(object)        # FramePacket
    stats_updated = Signal(dict)        # session summary
    shot_recorded = Signal(object)      # ShotEvent
    status_changed = Signal(str)
    clock_event = Signal(str)           # 'warn' | 'expired'
    error = Signal(str)
    settings_changed = Signal(object)   # Settings (after an in-view tweak, e.g. felt pick)
    replay_saved = Signal(str)          # path to a saved replay clip
    shot_suggested = Signal(object)     # ShotEvent — manual-confirm mode (not recorded)
    recording_changed = Signal(bool)    # recording on/off
    detection_changed = Signal(bool)    # auto-detection on/off
    capture_progress = Signal(str)      # analysis-capture status text
    capture_saved = Signal(str)         # path to a finished analysis-capture zip
    failure_flagged = Signal(str)       # path to a staged debug bundle
    source_is_video = Signal(bool, int, float)  # (seekable video?, frame_count, fps)
    video_state = Signal(int, int, bool)  # (current frame, total frames, playing)

    def __init__(self, settings: Settings, repository: Repository):
        super().__init__()
        self._settings = settings
        self._repo = repository
        self._pipeline: Pipeline | None = None
        self._source = None
        self._timer: QTimer | None = None
        self._clock = ShotClock(settings.shot_clock)
        self._t0 = 0.0
        self._fps = 0.0
        self._src_fps = 30.0
        self._prev_state = "settled"
        self._turn_start_t = 0.0
        self._session_id: int | None = None
        self._mode = "free_play"
        self._running = False
        self._miss_count = 0
        self._last_frame: np.ndarray | None = None
        self._replay: deque = deque(maxlen=150)
        self._recorder = None
        self._recording_path = ""
        # App default: detection ON. The trained YOLO model is reliable, so the
        # old "preview-only, show nothing rather than something wrong" default
        # (which existed because classical CV was untrustworthy) no longer applies
        # — the user wants to USE the tracker, not opt into it every launch.
        self._detection_enabled = True
        # Training mode: when on, the camera view is sent UN-annotated (so the live
        # page can draw the labelling overlay) and raw detections + guessed numbers
        # ride along in the packet for correcting.
        self._label_mode = False
        self._capture: dict | None = None  # active analysis-capture context
        # video transport state (only meaningful for a video-file source)
        self._video_paused = False
        self._speed = 1.0
        self._play_tick = 0  # playback frame counter for detection cadence
        self._base_interval = 33
        self._video_pos = 0

    # ------------------------------------------------------------------ #
    @Slot()
    def on_started(self) -> None:
        self._timer = QTimer()
        self._timer.setTimerType(Qt.PreciseTimer)
        self._timer.timeout.connect(self._tick)

    # ------------------------------------------------------------------ #
    # Control (invoked queued from the UI thread)
    # ------------------------------------------------------------------ #
    @Slot(str, str, str)
    def start(self, source_spec: str, mode: str = "free_play", drill_key: str = "") -> None:
        from ..capture.camera import open_source

        self.stop()
        # For a camera index, re-resolve by saved friendly name so a reshuffled
        # USB camera still opens the right device (and warn if it moved).
        resolved_name = ""
        if source_spec.isdigit():
            from ..capture.devices import resolve_camera
            idx, resolved_name, warn = resolve_camera(int(source_spec), self._settings.source_name)
            if warn:
                self.error.emit(warn)
            source_spec = str(idx)
        # Sanity log on every Start so "wrong camera" issues are diagnosable.
        log.info('[start] opening source=%s name="%s" mode=%s (settings.source=%s name="%s")',
                 source_spec, resolved_name, mode, self._settings.source, self._settings.source_name)
        try:
            self._source = open_source(source_spec)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(f"Could not open source '{source_spec}': {exc}")
            return
        if hasattr(self._source, "opened") and not self._source.opened:
            if source_spec.isdigit():
                self.error.emit(f"Camera {source_spec} didn't open — it may be in use by "
                                "another app (close Zoom/Teams/OBS) or disconnected. On "
                                "Windows, also check Settings → Privacy → Camera.")
            else:
                self.error.emit(f"Couldn't open '{source_spec}'. Check the file path.")
            return

        self._pipeline = Pipeline(self._settings, source=source_spec)
        # Honour the current auto-detection state — launch defaults to preview.
        self._pipeline.detect_enabled = self._detection_enabled
        self._clock = ShotClock(self._settings.shot_clock)
        self._mode = mode
        from ..game.drills import get_drill
        drill = get_drill(drill_key) if drill_key else None
        self._session_id = self._repo.start_session(
            mode=mode, drill_key=drill_key or None,
            drill_target=drill.target_makes if drill else 0,
            table_size=self._settings.table.size,
        )
        self._t0 = time.perf_counter()
        self._prev_state = "settled"
        self._turn_start_t = 0.0
        self._running = True
        self._miss_count = 0
        self._last_frame = None
        self._src_fps = float(getattr(self._source, "fps", 30.0)) or 30.0
        self._replay = deque(maxlen=int(max(30, min(self._src_fps, 30) * 5)))
        interval = max(8, int(1000.0 / max(1.0, min(self._src_fps, 60.0))))
        self._base_interval = interval
        self._video_paused = False
        self._speed = 1.0
        self._video_pos = 0
        if self._timer is None:
            self.on_started()
        self._timer.start(interval)
        self.status_changed.emit("running")
        self.detection_changed.emit(self._detection_enabled)
        # tell the UI whether to show video transport controls
        is_vid = bool(getattr(self._source, "is_video", False))
        self.source_is_video.emit(is_vid, getattr(self._source, "frame_count", 0) if is_vid else 0,
                                  self._src_fps)
        log.info("Started: source=%s mode=%s detect=%s video=%s", source_spec, mode,
                 self._detection_enabled, is_vid)

    @Slot()
    def stop(self) -> None:
        if self._timer:
            self._timer.stop()
        if self._capture is not None:
            self._finalize_capture()  # save whatever we captured before stopping
        if self._running and self._session_id is not None:
            self._repo.end_session(self._session_id)
            self.stats_updated.emit(self._repo.global_summary())
        if self._recorder is not None and hasattr(self._recorder, "release"):
            self._recorder.release()
        self._recorder = None
        if self._source:
            try:
                self._source.release()
            except Exception:  # noqa: BLE001
                pass
        self._source = None
        self._running = False
        self.status_changed.emit("stopped")

    @Slot()
    def recalibrate(self) -> None:
        if self._pipeline:
            self._pipeline.request_recalibration()

    @Slot(object)
    def apply_settings(self, settings: Settings) -> None:
        self._settings = settings
        self._clock = ShotClock(settings.shot_clock)
        if self._pipeline:
            self._pipeline.reconfigure(settings)

    @Slot(float, float)
    def pick_felt(self, x_frac: float, y_frac: float) -> None:
        """Sample the felt colour at a clicked point (normalised coords) on the
        last camera frame, seed the felt settings, and recalibrate."""
        if self._last_frame is None:
            self.error.emit("No frame yet — start the camera before picking felt.")
            return
        h, w = self._last_frame.shape[:2]
        px, py = int(x_frac * w), int(y_frac * h)
        new_felt = felt_from_point(self._last_frame, px, py, self._settings.felt.sensitivity)
        self._settings.felt = new_felt
        if self._pipeline:
            self._pipeline.reconfigure(self._settings)
        self.settings_changed.emit(self._settings)
        log.info("Felt picked at (%d,%d) hue~%d", px, py, new_felt.picked_hsv[0])

    @Slot()
    def save_replay(self) -> None:
        """Write the buffered recent frames to an mp4 so the user can rewatch
        exactly what the detector saw."""
        frames = list(self._replay)
        if not frames:
            self.error.emit("Nothing to replay yet.")
            return
        EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        dest = EXPORTS_DIR / f"replay-{stamp}.mp4"
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(dest), fourcc, max(10.0, min(self._src_fps, 30)), (w, h))
        try:
            for f in frames:
                writer.write(f)
        finally:
            writer.release()
        log.info("Saved replay: %s (%d frames)", dest, len(frames))
        self.replay_saved.emit(str(dest))

    @Slot(bool)
    def set_paused(self, paused: bool) -> None:
        if self._pipeline:
            self._pipeline.paused = paused
        self.status_changed.emit("paused" if paused else "running")

    @Slot(str)
    def set_detector_strategy(self, name: str) -> None:
        """Switch the live detector (simple_blob / felt_mask_hough / onnx_* /
        legacy) without dropping the table calibration."""
        self._settings.balls.live_strategy = name
        if self._pipeline:
            self._pipeline.set_strategy(name)

    @Slot(bool)
    def set_detection_enabled(self, on: bool) -> None:
        """Turn auto ball/shot detection on or off. OFF = clean camera preview +
        manual scoring (the launch default); ON = full CV pipeline."""
        self._detection_enabled = on
        if self._pipeline:
            self._pipeline.detect_enabled = on
            if not on:
                # drop any in-flight tracks/shot state so nothing lingers on the
                # overhead when we fall back to the empty-table preview
                self._pipeline.tracker.reset()
                self._pipeline.shots.reset()
        self.detection_changed.emit(on)
        log.info("Auto-detection %s", "ON" if on else "OFF")

    @Slot(bool)
    def set_label_mode(self, on: bool) -> None:
        """Training mode: send the raw camera frame (so the UI can draw the
        labelling overlay) + raw detections with guessed numbers in each packet."""
        self._label_mode = bool(on)
        # Re-process the CURRENT frame immediately. Entering Training Mode on a
        # paused video would otherwise wait for a next frame that never comes —
        # leaving the UI with no frame size and no detections (clicks then fall
        # back to the origin). This re-emits the raw frame + guessed numbers now.
        if on and self._last_frame is not None and self._pipeline is not None:
            self._run_frame(self._last_frame, detect=True)

    @Slot(object)
    def save_training_frame(self, balls) -> None:
        """Persist the current RAW frame + corrected boxes (list of
        (number, cx, cy, w, h), normalised) to the ball-ID training store."""
        if self._last_frame is None or not balls:
            return
        try:
            from ..config import APP_DIR
            from ..train.store import LabeledBall, TrainingStore
            store = TrainingStore(APP_DIR / "training" / "ballid")
            labeled = [LabeledBall(int(n), float(cx), float(cy), float(w), float(h))
                       for (n, cx, cy, w, h) in balls]
            # Write a PRIVATE contiguous copy — never hand the live frame buffer
            # (shared with the UI thread's overlay) to cv2.imwrite.
            frame = np.ascontiguousarray(self._last_frame).copy()
            saved = store.add_frame(frame, labeled)
            self.capture_progress.emit(f"Saved {saved} labelled balls "
                                       f"({store.count()} frames collected)")
        except Exception as exc:  # noqa: BLE001 - never crash the worker on a save
            log.exception("save_training_frame failed")
            self.error.emit(f"Couldn't save training frame: {exc}")

    @Slot()
    def reset_counters(self) -> None:
        """Start a fresh session so the make/miss counters reset to zero."""
        if not self._running or self._session_id is None:
            return
        self._repo.end_session(self._session_id)
        self._session_id = self._repo.start_session(
            mode=self._mode, table_size=self._settings.table.size)
        if self._pipeline:
            self._pipeline.shots.reset()
        self.stats_updated.emit(self._repo.session_summary(self._session_id))

    @Slot(str)
    def record_manual_shot(self, outcome: str) -> None:
        """Record a shot the user tapped manually (make/miss/scratch)."""
        if self._session_id is None:
            return
        self._repo.record_shot(self._session_id, outcome=outcome,
                               num_pocketed=1 if outcome == "make" else 0)
        self.stats_updated.emit(self._repo.session_summary(self._session_id))

    @Slot(bool)
    def set_recording(self, on: bool) -> None:
        if on and self._recorder is None:
            EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            self._recording_path = str(EXPORTS_DIR / f"session-{stamp}.mp4")
            self._recorder = "pending"  # opened lazily on first frame (need size)
            self.recording_changed.emit(True)
        elif not on and self._recorder is not None:
            if hasattr(self._recorder, "release"):
                self._recorder.release()
            self._recorder = None
            self.recording_changed.emit(False)
            if self._recording_path:
                self.replay_saved.emit(self._recording_path)

    @Slot()
    def start_analysis_capture(self, seconds: float = 60.0) -> None:
        """Record raw camera frames (+ a metadata sidecar) to a zip for offline
        YOLO training. Captures the UNANNOTATED feed so the frames are usable as
        real training images. Bounded in time and frame count to keep the zip
        sane; sampled every other frame (~15 fps)."""
        if self._capture is not None:
            self.capture_progress.emit("Already capturing…")
            return
        if not self._running or self._source is None:
            self.error.emit("Start the camera before capturing a session.")
            return
        EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        cap_dir = EXPORTS_DIR / f"capture-{stamp}"
        (cap_dir / "frames").mkdir(parents=True, exist_ok=True)
        self._capture = {
            "dir": cap_dir, "stamp": stamp, "saved": 0, "seen": 0,
            "stride": 2, "max_frames": 900,
            "deadline": time.perf_counter() + max(5.0, seconds),
            "started_iso": datetime.now(timezone.utc).isoformat(),
        }
        self.capture_progress.emit("Capturing 0 frames…")
        log.info("Analysis capture started -> %s", cap_dir)

    def _write_capture(self, frame: np.ndarray) -> None:
        cap = self._capture
        if cap is None:
            return
        cap["seen"] += 1
        done = (time.perf_counter() >= cap["deadline"]
                or cap["saved"] >= cap["max_frames"])
        if not done and cap["seen"] % cap["stride"] == 0:
            small = self._small(frame, max_w=720)
            path = cap["dir"] / "frames" / f"f{cap['saved']:05d}.jpg"
            try:
                cv2.imwrite(str(path), small, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                cap["saved"] += 1
                if cap["saved"] % 30 == 0:
                    self.capture_progress.emit(f"Capturing {cap['saved']} frames…")
            except Exception:  # noqa: BLE001
                pass
        if done:
            self._finalize_capture()

    def _finalize_capture(self) -> None:
        import shutil
        import zipfile
        from pathlib import Path
        cap = self._capture
        self._capture = None
        if cap is None:
            return
        calib = None
        if self._pipeline and self._pipeline.calib.is_calibrated:
            c = self._pipeline.calib.calib
            calib = {"corners": np.asarray(c.corners).tolist(),
                     "dst_size": list(c.dst_size)}
        meta = {
            "app_version": __version__, "started_iso": cap["started_iso"],
            "source": self._settings.source, "source_name": self._settings.source_name,
            "src_fps": self._src_fps, "stride": cap["stride"],
            "frame_count": cap["saved"], "calibration": calib,
            "note": "Raw camera frames for YOLO fine-tuning (labelling + training "
                    "tooling ships in v0.2.15).",
        }
        cap_dir: Path = cap["dir"]
        try:
            (cap_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            zip_path = EXPORTS_DIR / f"capture-{cap['stamp']}.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for p in sorted(cap_dir.rglob("*")):
                    if p.is_file():
                        zf.write(p, p.relative_to(cap_dir))
            shutil.rmtree(cap_dir, ignore_errors=True)
        except OSError as exc:
            self.error.emit(f"Capture failed to save: {exc}")
            return
        log.info("Analysis capture saved: %s (%d frames)", zip_path, cap["saved"])
        self.capture_progress.emit(f"Saved {cap['saved']} frames")
        self.capture_saved.emit(str(zip_path))

    @Slot()
    def flag_failure(self) -> None:
        """Save the recent preview buffer + detector state to a zip and stage it
        for the dev machine (Settings -> Debug -> "Save clip + flag as failure").
        Captures what the detector just saw so failures can be reproduced offline."""
        frames = list(self._replay)
        if not frames:
            self.error.emit("Nothing to flag yet — let the camera run for a few seconds.")
            return
        import zipfile

        from .. import debug_upload
        EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        zpath = EXPORTS_DIR / f"failure-{stamp}.zip"
        meta = {
            "kind": "detection-failure", "flagged_iso": datetime.now(timezone.utc).isoformat(),
            "app_version": __version__, "source": self._settings.source,
            "source_name": self._settings.source_name,
            "detection_enabled": self._detection_enabled,
            "backend": self._settings.balls.backend, "frame_count": len(frames),
            "last_diag": getattr(self._pipeline, "_last_ms", None),
            "note": "Recent annotated preview buffer captured for offline debugging.",
        }
        try:
            with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
                for i, fr in enumerate(frames):
                    ok, buf = cv2.imencode(".jpg", fr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if ok:
                        zf.writestr(f"frames/f{i:04d}.jpg", buf.tobytes())
                zf.writestr("meta.json", json.dumps(meta, indent=2))
            dst = debug_upload.stage_bundle(zpath)
        except OSError as exc:
            self.error.emit(f"Couldn't save failure bundle: {exc}")
            return
        log.info("Flagged failure bundle: %s (staged -> %s)", zpath, dst)
        self.failure_flagged.emit(str(dst))

    @property
    def session_id(self) -> int | None:
        return self._session_id

    # ------------------------------------------------------------------ #
    # Loop
    # ------------------------------------------------------------------ #
    @Slot()
    def _tick(self) -> None:
        if not self._running or self._source is None or self._pipeline is None:
            return
        # A paused video freezes on the current frame (detection stays on it); we
        # just stop advancing. Step/seek re-run detection on the chosen frame.
        if self._video_paused and getattr(self._source, "is_video", False):
            return
        frame = self._source.read()
        if frame is None:
            # Tolerate transient empty reads from a live camera; only give up
            # after a sustained outage. Files/demo never miss, so fail fast.
            if getattr(self._source, "is_live", False):
                self._miss_count += 1
                if self._miss_count <= _CAMERA_MISS_TOLERANCE:
                    return
                self.error.emit("Camera stopped delivering frames — disconnected?")
            else:
                self.error.emit("Source ended.")
            self.stop()
            return
        # Detection cadence: a fast video at 2x/4x can't run detection on every
        # frame, so display every frame but only DETECT every Nth (stride == round
        # of playback speed). Step/seek/stop call _run_frame directly with detect
        # forced on, so a paused/landed frame is always fully analysed.
        self._play_tick += 1
        stride = max(1, round(self._speed)) if getattr(self._source, "is_video", False) else 1
        self._run_frame(frame, detect=(self._play_tick % stride == 0))

    def _run_frame(self, frame: np.ndarray, detect: bool = True) -> None:
        """Process one frame through the pipeline and emit results. Shared by the
        live tick and the video transport (step/seek/stop)."""
        t_wall0 = time.perf_counter()
        self._miss_count = 0
        self._last_frame = frame
        if getattr(self._source, "is_video", False):
            self._video_pos = max(0, self._source.position() - 1)

        # analysis capture writes the RAW (unannotated) frame for training data
        if self._capture is not None:
            self._write_capture(frame)

        t = time.perf_counter() - self._t0
        try:
            res = self._pipeline.process(frame, t, annotate=not self._label_mode, detect=detect)
        except Exception as exc:  # noqa: BLE001 - never let one bad frame kill the loop
            log.exception("pipeline error")
            self.error.emit(f"Pipeline error: {exc}")
            return

        self._handle_state(res.shot_state, t)
        if res.shot_event is not None:
            if self._settings.detection.manual_confirm:
                # suggest only — the user commits via the make/miss buttons
                self.shot_suggested.emit(res.shot_event)
            else:
                self._record_shot(res.shot_event, t)

        clock_edge = self._clock.poll(t)
        if clock_edge:
            self.clock_event.emit(clock_edge)

        # buffer a downscaled annotated frame for instant replay
        if res.frame_bgr is not None:
            self._replay.append(self._small(res.frame_bgr))

        # recording mode: write the annotated camera frame to disk for offline analysis
        if self._recorder is not None and res.frame_bgr is not None:
            self._write_recording(res.frame_bgr)

        dt = time.perf_counter() - t_wall0
        inst = 1.0 / dt if dt > 0 else 0.0
        self._fps = 0.9 * self._fps + 0.1 * inst if self._fps else inst

        self.frame_ready.emit(FramePacket(
            perspective=res.frame_bgr, birdseye=res.rect_bgr, status=res.status,
            fps=self._fps, n_balls=res.n_balls, shot_state=res.shot_state,
            clock_remaining=self._clock.remaining(t),
            clock_enabled=self._clock.enabled, clock_warning=self._clock.is_warning(t),
            deviated=res.deviated, tracks=res.tracks, raw_dets=res.raw_dets,
        ))
        if getattr(self._source, "is_video", False):
            self.video_state.emit(self._video_pos, self._source.frame_count,
                                  not self._video_paused)

    # ------------------------------------------------------------------ #
    # Video transport (only act on a seekable video source)
    # ------------------------------------------------------------------ #
    def _is_video(self) -> bool:
        return self._source is not None and getattr(self._source, "is_video", False)

    @Slot(bool)
    def set_video_paused(self, paused: bool) -> None:
        self._video_paused = paused
        if self._is_video():
            self.video_state.emit(self._video_pos, self._source.frame_count, not paused)

    @Slot()
    def video_stop(self) -> None:
        """Pause and return to the first frame."""
        if not self._is_video():
            return
        self._video_paused = True
        self._source.seek(0)
        f = self._source.read()
        if f is not None:
            self._run_frame(f)

    @Slot(int)
    def video_seek(self, frame_idx: int) -> None:
        if not self._is_video():
            return
        self._source.seek(int(frame_idx))
        f = self._source.read()
        if f is not None:
            self._run_frame(f)

    @Slot(int)
    def video_step(self, delta: int) -> None:
        """Step one (or delta) frames while staying paused — for frame-by-frame
        detector debugging. Re-runs detection on the new frame."""
        if not self._is_video():
            return
        self._video_paused = True
        target = max(0, self._video_pos + int(delta))
        self._source.seek(target)
        f = self._source.read()
        if f is not None:
            self._run_frame(f)

    @Slot(float)
    def set_playback_speed(self, mult: float) -> None:
        self._speed = max(0.1, float(mult))
        if self._timer is not None:
            self._timer.start(max(4, int(self._base_interval / self._speed)))

    @staticmethod
    def _small(frame: np.ndarray, max_w: int = 480) -> np.ndarray:
        h, w = frame.shape[:2]
        if w <= max_w:
            return frame.copy()
        scale = max_w / w
        return cv2.resize(frame, (max_w, int(h * scale)))

    def _write_recording(self, frame: np.ndarray) -> None:
        small = self._small(frame, max_w=640)
        if self._recorder == "pending":
            h, w = small.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._recorder = cv2.VideoWriter(
                self._recording_path, fourcc, max(10.0, min(self._src_fps, 30)), (w, h))
        try:
            self._recorder.write(small)
        except Exception:  # noqa: BLE001
            pass

    def _handle_state(self, state: str, t: float) -> None:
        if self._prev_state != "settled" and state == "settled":
            self._clock.start(t)        # no-op when the clock is disabled
            self._turn_start_t = t
        elif self._prev_state == "settled" and state == "moving":
            self._clock.stop()
        self._prev_state = state

    def _record_shot(self, event: ShotEvent, t: float) -> None:
        if self._session_id is None:
            return
        shot_seconds = max(0.0, event.start_t - self._turn_start_t) if self._turn_start_t else 0.0
        self._repo.record_shot(
            self._session_id, outcome=event.outcome.value,
            num_pocketed=event.num_pocketed, target_pocket=event.target_pocket,
            cue_scratch=event.cue_scratch, duration_s=event.duration_s,
            shot_seconds=shot_seconds,
        )
        self._log_shot(event, shot_seconds)
        self.shot_recorded.emit(event)
        self.stats_updated.emit(self._repo.session_summary(self._session_id))

    def _log_shot(self, event: ShotEvent, shot_seconds: float) -> None:
        """Append a structured shot event to a debug log for later iteration."""
        try:
            SHOTLOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            rec = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "session": self._session_id,
                "mode": self._mode,
                "outcome": event.outcome.value,
                "num_pocketed": event.num_pocketed,
                "target_pocket": event.target_pocket,
                "cue_scratch": event.cue_scratch,
                "duration_s": round(event.duration_s, 3),
                "shot_seconds": round(shot_seconds, 3),
                "pocketed": [
                    {"track_id": p.track_id, "cls": p.cls.value, "pocket": p.pocket}
                    for p in event.pocketed
                ],
            }
            with open(SHOTLOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass


def make_controller_thread(controller: PipelineController) -> QThread:
    """Move the controller onto a dedicated thread and start it."""
    thread = QThread()
    thread.setObjectName("pipeline")
    controller.moveToThread(thread)
    thread.started.connect(controller.on_started)
    thread.start()
    return thread
