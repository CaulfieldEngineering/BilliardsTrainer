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
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

import cv2
import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot

from ..capture import audio as audio_mod
from ..capture.preprocess import preprocess_frame
from ..config import EXPORTS_DIR, SHOTLOG_PATH, Settings
from ..db.repository import Repository
from ..events.shot_detector import ShotEvent
from ..game.shot_clock import ShotClock
from ..version import __version__
from ..vision.felt import felt_from_point
from ..vision.pipeline import Pipeline
from ..vision.types import BallClass

log = logging.getLogger("controller")

# How long a *live* camera may go without delivering a frame before we declare
# it disconnected. WALL-TIME, not ticks — tick rate varies with the reported
# fps, so a tick count would mean wildly different real time per camera.
_CAMERA_MISS_SECONDS = 2.5
# Warm-up allowance BEFORE the first-ever frame: the threaded grabber returns
# None instantly while the camera driver is still initialising (auto-exposure
# etc. can take seconds), so startup gets a longer leash.
_CAMERA_STARTUP_SECONDS = 8.0


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
    feed_sd: bool = False   # camera fell back to the 480p HDMI mode (re-arm ML)
    feed_info: str = ""     # corner stats chip: container/active resolution + fps


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
    _detections_ready = Signal(object, object)  # (raw_dets, frame_shape) worker -> this thread

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
        self._miss_t0: float | None = None  # first consecutive empty-read time
        self._got_frame = False  # ever received a frame from the current source
        self._last_frame: np.ndarray | None = None
        self._replay: deque = deque(maxlen=150)
        self._recorder = None
        self._recording_path = ""
        self._recording_paused = False
        self._audio = None            # AudioRecorder (or None when not recording)
        self._audio_dir = None        # temp dir holding audio segments
        self._rec_crop = None         # (x0,y0,x1,y1) HDMI content box for this recording
        self._feed_sd = False         # 480p-fallback watchdog state
        self._feed_check_tick = 0
        self._feed_info = ""          # corner stats chip text (resolution/fps)
        self._recording_tmp = ""      # hidden .part path while a recording is open
        self._rec_frames = 0          # frames actually written this recording
        self._rec_fps = 0.0           # fps declared to the video writer
        self._rec_t0 = None           # start of current unpaused stretch
        self._rec_elapsed = 0.0       # accumulated unpaused wall-clock seconds
        # App default: detection ON. The trained YOLO model is reliable, so the
        # old "preview-only, show nothing rather than something wrong" default
        # (which existed because classical CV was untrustworthy) no longer applies
        # — the user wants to USE the tracker, not opt into it every launch.
        self._detection_enabled = bool(getattr(settings.detection, "enabled", True))
        # Training mode: when on, the camera view is sent UN-annotated (so the live
        # page can draw the labelling overlay) and raw detections + guessed numbers
        # ride along in the packet for correcting.
        self._label_mode = False
        self._capture: dict | None = None  # active analysis-capture context
        # Latest cue-stroke metrics from the Bluetooth IMU worker, joined to the
        # next recorded shot by wall-clock freshness (the strike precedes the
        # balls settling by however long they roll).
        self._last_stroke: dict | None = None
        # Cue-ball shot-clock state (Joe's rule: the countdown starts when the
        # CUE BALL stops; the strike stops it = made it in time).
        self._cue_still = 0            # consecutive at-rest frames
        self._saw_cue_t = -1e9         # last time a cue track existed
        self._clock_armed = True       # a new turn may start the clock
        # Flow rule: a shot clock only makes sense while PLAYING — live camera
        # sources only. Reviewing a recorded video must never run a countdown.
        self._clock_allowed = False
        # Async vision (live camera): one worker thread runs inference off the
        # display path. Frames are offered via a 1-slot queue (busy worker =>
        # the frame is display-only); results come back through a queued signal
        # so ALL tracker/pipeline mutation stays on this controller thread.
        self._det_queue: queue.Queue = queue.Queue(maxsize=1)
        self._det_thread: threading.Thread | None = None
        self._detections_ready.connect(self._on_detections_ready, Qt.QueuedConnection)
        # video transport state (only meaningful for a video-file source)
        self._video_paused = False
        self._speed = 1.0
        self._play_tick = 0  # playback frame counter for detection cadence
        self._base_interval = 33
        self._video_pos = 0

    # ------------------------------------------------------------------ #
    @Slot()
    def on_started(self) -> None:
        # PARENT the timer to the controller. A parentless QTimer on a worker
        # thread is owned only by its Python reference; under PySide6 it can be
        # freed while its OS timer is still registered with the event dispatcher,
        # so the next tick delivers a QTimerEvent into freed memory and crashes
        # natively in QCoreApplication::notifyInternal2 (confirmed from two crash
        # dumps: sendTimerEvent -> notifyInternal2 on a freed receiver, on this
        # worker thread). Parenting ties the timer's lifetime to the controller —
        # it lives for the whole session and is destroyed (and unregistered) with us.
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.PreciseTimer)
        self._timer.timeout.connect(self._tick)
        log.info("playback timer created (parented), thread=%s",
                 __import__("threading").current_thread().name)

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
            self._source = open_source(source_spec, cam=self._settings.camera)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(f"Could not open source '{source_spec}': {exc}")
            return
        if hasattr(self._source, "opened") and not self._source.opened:
            if source_spec.lower() in ("tether", "canon", "gphoto"):
                self.error.emit("Tethered camera didn't open. Check the USB cable, close any "
                                "other tether app (Smart Shooter, EOS Utility), and make sure "
                                "the camera is switched on.")
            elif source_spec.isdigit():
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
        # STATS BELONG TO A RECORDING SESSION. Vision runs continuously, but no
        # DB session opens (and no shots count) until the user hits Record —
        # set_recording(True) opens one, set_recording(False) closes it.
        self._session_id = None
        self.stats_updated.emit({"makes": 0, "misses": 0, "make_pct": 0,
                                 "current_streak": 0})
        self._t0 = time.perf_counter()
        self._prev_state = "settled"
        self._turn_start_t = 0.0
        self._cue_still = 0
        self._saw_cue_t = -1e9
        self._clock_armed = True
        self._clock_allowed = bool(getattr(self._source, "is_live", False))
        self._running = True
        self._miss_t0: float | None = None  # first consecutive empty-read time
        self._got_frame = False
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
        if self._recorder is not None:
            self.set_recording(False)  # closes the stats session with the video
        if self._session_id is not None:  # safety: never leave a session open
            self._repo.end_session(self._session_id)
            self._session_id = None
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

    @Slot()
    def refocus(self) -> None:
        """Touchless camera sync for the ceiling-mounted DSLR: re-drive autofocus
        and re-apply the saved exposure config.

        Two paths: when the tether IS the video source, ask its grab thread; in
        the HDMI-dongle rig (video from the capture dongle, USB purely for
        control) open a transient PTP session instead — apply config + AF, then
        disconnect so the camera's HDMI liveview resumes. The transient path
        blocks this worker for a few seconds; video stalls briefly, which is
        acceptable for a manual button press."""
        fn = getattr(self._source, "refocus", None)
        if callable(fn):
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                self.error.emit(f"Refocus failed: {exc}")
            return
        from ..capture.tether import remote_camera_sync
        msg = remote_camera_sync(self._settings.camera.tether, autofocus=True)
        if msg:
            self.error.emit(f"Camera sync: {msg}")

    @Slot(object)
    def apply_settings(self, settings: Settings) -> None:
        self._settings = settings
        self.set_detection_enabled(bool(getattr(settings.detection, "enabled", True)))
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
                # without tracking there is no strike to detect — never leave a
                # countdown running that nothing can stop
                self._clock.stop()
                self._clock_armed = True
        self.detection_changed.emit(on)
        log.info("Auto-detection %s", "ON" if on else "OFF")

    @Slot(bool)
    def set_label_mode(self, on: bool) -> None:
        """Training mode: send the raw camera frame (so the UI can draw the
        labelling overlay) + raw detections with guessed numbers in each packet."""
        import threading
        self._label_mode = bool(on)
        log.info("set_label_mode=%s (thread=%s) video=%s last_frame=%s", on,
                 threading.current_thread().name,
                 getattr(self._source, "is_video", False),
                 None if self._last_frame is None else self._last_frame.shape)
        # Auto-pause the video on entering Training Mode. Labelling is done on a
        # still frame (scrub to it, then correct) — pausing stops the frame
        # stream from blowing away the user's selection mid-correction AND stops
        # the continuous 1080p repaint racing DirectML inference on the GPU.
        if on and getattr(self._source, "is_video", False):
            self._video_paused = True
        if on:
            self._clock.stop()   # training is not play — no countdown pressure
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
        import threading
        log.info("save_training_frame: %d balls (thread=%s, frame=%s)", len(balls),
                 threading.current_thread().name, self._last_frame.shape)
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
                               num_pocketed=1 if outcome == "make" else 0,
                               stroke_json=self._consume_stroke())
        self.stats_updated.emit(self._repo.session_summary(self._session_id))

    # ------------------------------------------------------------------ #
    # Cue-stroke sensor (Bluetooth IMU on the cue butt)
    # ------------------------------------------------------------------ #
    @Slot(object)
    def on_cue_impact(self, stroke: dict) -> None:
        """A confirmed cue-ball strike from the IMU — fires within ~0.5 s of
        contact, long before the balls settle. This is the precise 'shot
        taken' moment: it stops the shot clock (made it in time) even if the
        camera hasn't registered motion yet."""
        log.info("cue impact felt: %.1f g", stroke.get("peak_g", 0.0))
        if self._clock.enabled and self._clock.running:
            self._clock.stop()
            self._clock_armed = True    # next cue-ball rest starts the next turn
            log.info("shot clock stopped by cue impact (made it in time)")

    @Slot(object)
    def on_stroke_metrics(self, metrics: dict) -> None:
        """Full stroke metrics (~2.6 s after the strike). Kept until the next
        recorded shot consumes them (vision-settled or manual +Make/-Miss)."""
        self._last_stroke = metrics

    _STROKE_JOIN_SECONDS = 25.0  # max stroke→record gap (long rolls + settle time)

    def _consume_stroke(self) -> str:
        """The latest stroke metrics as JSON iff fresh; consumed exactly once
        so one physical stroke can never annotate two shots."""
        m, self._last_stroke = self._last_stroke, None
        if not m or time.time() - m.get("hit_epoch", 0.0) > self._STROKE_JOIN_SECONDS:
            return ""
        try:
            return json.dumps(m)
        except (TypeError, ValueError):
            return ""

    @Slot(bool)
    def set_recording_paused(self, paused: bool) -> None:
        """Pause/resume writing frames to the active recording (session stays open)."""
        self._recording_paused = paused
        if self._recorder is not None and hasattr(self._recorder, "pause"):
            self._recorder.pause(paused)
        if self._recorder is None or self._audio is None:
            return
        # Audio must pause WITH the frames or the tracks drift apart: each
        # unpaused stretch is its own segment, concatenated at stop.
        if paused:
            self._audio.pause()
            if self._rec_t0 is not None:
                self._rec_elapsed += audio_mod.elapsed_monotonic() - self._rec_t0
                self._rec_t0 = None
        elif self._audio_dir is not None:
            self._audio.start(self._audio_dir)

    @Slot(bool)
    def set_recording(self, on: bool) -> None:
        if on and self._recorder is None:
            rec = self._settings.recording
            rec_dir = rec.resolved_dir()
            try:
                rec_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                # A vanished synced folder must not kill the session start.
                log.warning("recordings dir %s unusable (%s) — using exports", rec_dir, exc)
                rec_dir = EXPORTS_DIR
                rec_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            self._recording_path = str(rec_dir / f"session-{stamp}.mp4")
            # Stage the in-progress file OUTSIDE the synced folder entirely
            # (Dropbox synced the growing hidden .part and re-materialized
            # half-synced copies as unplayable ghost mp4s); it moves into the
            # recordings folder only once finalized. Same APFS volume, so the
            # final replace() is an atomic rename.
            EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
            self._recording_tmp = str(EXPORTS_DIR / f".session-{stamp}.part.mp4")
            self._recorder = "pending"  # opened lazily on first frame (need size)
            self._recording_paused = False
            self._rec_frames = 0
            self._rec_t0 = None
            self._rec_elapsed = 0.0
            self._rec_crop = None
            self._audio = audio_mod.make_recorder(rec.audio, rec.audio_device)
            self._audio_dir = EXPORTS_DIR / f".audio-{stamp}"
            self._audio.start(self._audio_dir)
            # Stats live and die with the recording: fresh session, zeroed count.
            self._session_id = self._repo.start_session(
                mode=self._mode, drill_key=None, drill_target=0,
                table_size=self._settings.table.size)
            self.stats_updated.emit(self._repo.session_summary(self._session_id))
            self.recording_changed.emit(True)
        elif not on and self._recorder is not None:
            if hasattr(self._recorder, "release"):
                self._recorder.release()
            self._recorder = None
            if self._session_id is not None:
                self._repo.end_session(self._session_id)
                self._session_id = None
            self.recording_changed.emit(False)
            if self._rec_t0 is not None:
                self._rec_elapsed += audio_mod.elapsed_monotonic() - self._rec_t0
                self._rec_t0 = None
            if self._audio is not None:
                # The paced writer emits true constant-frame-rate video on a
                # wall-clock schedule, so no retiming is needed at mux.
                self._audio.stop_and_mux(self._recording_tmp, ts_scale=1.0)
                self._audio = None
                self._audio_dir = None
            if self._recording_path:
                from pathlib import Path as _P
                try:
                    if _P(self._recording_tmp).is_file():
                        _P(self._recording_tmp).replace(self._recording_path)
                except OSError:
                    log.exception("finalizing recording failed")
                self.replay_saved.emit(self._recording_path)

    @Slot()
    def start_analysis_capture(self, seconds: float = 150.0) -> None:
        """Record raw, FULL-RESOLUTION camera frames to a zip for training.
        Bounded in time/count to keep the zip sane; sampled every other frame
        (~15 fps)."""
        if self._capture is not None:
            self.capture_progress.emit("Already recording…")
            return
        if not self._running or self._source is None:
            self.error.emit("Start the camera before recording a session.")
            return
        EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        cap_dir = EXPORTS_DIR / f"capture-{stamp}"
        (cap_dir / "frames").mkdir(parents=True, exist_ok=True)
        self._capture = {
            "dir": cap_dir, "stamp": stamp, "saved": 0, "seen": 0,
            "stride": 2, "max_frames": 2000,
            "deadline": time.perf_counter() + max(5.0, seconds),
            "started_iso": datetime.now(timezone.utc).isoformat(),
        }
        self.capture_progress.emit("Recording session… 0 frames")
        log.info("Training-session recording started -> %s", cap_dir)

    def _write_capture(self, frame: np.ndarray) -> None:
        cap = self._capture
        if cap is None:
            return
        cap["seen"] += 1
        done = (time.perf_counter() >= cap["deadline"]
                or cap["saved"] >= cap["max_frames"])
        if not done and cap["seen"] % cap["stride"] == 0:
            # Full resolution (only lightly capped) — the auto-labeller needs the
            # detail to read each ball's colour + solid/stripe.
            img = self._small(frame, max_w=1920) if frame.shape[1] > 1920 else frame
            path = cap["dir"] / "frames" / f"f{cap['saved']:05d}.jpg"
            try:
                cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                cap["saved"] += 1
                if cap["saved"] % 15 == 0:
                    self.capture_progress.emit(f"● Recording session… {cap['saved']} frames")
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
                now = time.perf_counter()
                if self._miss_t0 is None:
                    self._miss_t0 = now
                limit = _CAMERA_MISS_SECONDS if self._got_frame else _CAMERA_STARTUP_SECONDS
                if now - self._miss_t0 <= limit:
                    return
                if self._got_frame:
                    self.error.emit("Camera stopped delivering frames — disconnected?")
                else:
                    self.error.emit("Camera opened but never delivered a frame — it may "
                                    "be in use by another app, or blocked by Windows "
                                    "Settings → Privacy → Camera.")
            else:
                self.error.emit("Source ended.")
            self.stop()
            return
        self._play_tick += 1
        if getattr(self._source, "is_video", False):
            # Detection cadence: a fast video at 2x/4x can't run detection on
            # every frame, so display every frame but only DETECT every Nth
            # (stride == round of playback speed). Step/seek/stop call
            # _run_frame directly with detect forced on, so a paused/landed
            # frame is always fully analysed. Video replay stays SYNCHRONOUS —
            # deterministic for the eval harness and tests.
            stride = max(1, round(self._speed))
            self._run_frame(frame, detect=(self._play_tick % stride == 0))
        else:
            # LIVE camera: async vision. The display path never runs inference
            # (stays at camera rate); frames are handed to the detection worker
            # whenever it's idle, and its results are ingested between ticks.
            self._run_frame(frame, detect="async")

    # --- async vision (live camera) ------------------------------------- #
    def _submit_detection(self, frame: np.ndarray) -> None:
        """Offer a frame to the detection worker. Non-blocking: if the worker
        is mid-inference the frame is simply display-only."""
        if (not self._detection_enabled or self._pipeline is None
                or self._pipeline._strategy is None
                or not self._pipeline.calib.is_calibrated):
            return  # calibration/preview paths handle themselves synchronously
        if self._det_thread is None:
            self._det_thread = threading.Thread(
                target=self._detect_worker, daemon=True, name="detect-worker")
            self._det_thread.start()
        try:
            self._det_queue.put_nowait((frame, self._pipeline.calib.calib))
        except queue.Full:
            pass

    def _detect_worker(self) -> None:
        """Inference loop (worker thread). Only the strategy is touched here —
        it has its own lock — and results go back via a queued signal."""
        while True:
            frame, calib = self._det_queue.get()
            try:
                raw = self._pipeline._strategy.detect(frame, calib)
            except Exception:  # noqa: BLE001 - a bad frame must not kill the worker
                raw = []
            self._detections_ready.emit(raw, frame.shape)

    @Slot(object, object)
    def _on_detections_ready(self, raw_dets, frame_shape) -> None:
        """Apply worker results (controller thread — the only mutator)."""
        if (not self._running or self._pipeline is None
                or not self._detection_enabled
                or getattr(self._source, "is_video", False)):
            return
        try:
            self._pipeline.ingest_raw_detections(raw_dets, frame_shape)
        except Exception:  # noqa: BLE001
            log.exception("async detection ingest failed")

    def _run_frame(self, frame: np.ndarray, detect=True) -> None:
        """Process one frame through the pipeline and emit results. Shared by the
        live tick and the video transport (step/seek/stop)."""
        t_wall0 = time.perf_counter()
        self._miss_t0: float | None = None  # first consecutive empty-read time
        self._got_frame = True
        # Rotation/flip + colour correction apply to the LIVE camera only.
        # Recordings are saved post-preprocess (baked exactly as seen), so
        # re-applying on playback would double-rotate — a session always plays
        # back in the orientation it was recorded in.
        if not getattr(self._source, "is_video", False):
            frame = preprocess_frame(frame, self._settings.camera)
        if detect == "async":
            # Live async vision: hand the (preprocessed) frame to the detection
            # worker when it's idle; this display frame renders with the latest
            # ingested tracks and never blocks on inference.
            self._submit_detection(frame)
            detect = False
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
        self._update_cue_clock(res.tracks, t)
        # Shots only COUNT while a recording session is open — vision analyses
        # the table continuously, but stats belong to a session.
        if res.shot_event is not None and self._session_id is not None:
            self._record_shot(res.shot_event, t)

        clock_edge = self._clock.poll(t)
        if clock_edge:
            self.clock_event.emit(clock_edge)

        # buffer a downscaled annotated frame for instant replay
        if res.frame_bgr is not None:
            self._replay.append(self._small(res.frame_bgr))

        # recording mode: write the RAW (unannotated) frame — replayable through
        # the pipeline for testing, and reusable as training data.
        if (self._recorder is not None and not self._recording_paused
                and self._last_frame is not None):
            self._write_recording(self._last_frame)

        # 480p-fallback watchdog: the ML forced-1080i does not survive a camera
        # power cycle, and a silent fallback means degraded recordings. The
        # geometry is a fingerprint — the SD presentation fills only ~70% of
        # the container's short axis, every verified HD mode fills >=84%.
        self._feed_check_tick += 1
        if (self._feed_check_tick % 150 == 0 and frame is not None
                and not getattr(self._source, "is_video", False)):
            g = cv2.cvtColor(cv2.resize(frame, (frame.shape[1] // 8,
                                                frame.shape[0] // 8)),
                             cv2.COLOR_BGR2GRAY)
            ys, xs = np.where(g > 12)
            if len(xs) > 50:
                rh = (ys.max() - ys.min() + 1) / g.shape[0]
                # A healthy 1080i picture fills the container's FULL height
                # (3:2 image pillarboxed inside 16:9 -> 1621x1080). Every
                # degraded state seen so far is letterboxed vertically: the
                # classic 480p fallback (1730x757) and the post-power-cycle
                # intermediate (1710x875). Height-fill is the discriminator.
                aw = int(round((xs.max() - xs.min() + 1) * 8))
                ah = int(round((ys.max() - ys.min() + 1) * 8))
                sd = rh < 0.95
                self._feed_info = (f"{frame.shape[1]}\u00d7{frame.shape[0]} "
                                   f"@{self._src_fps:.0f}  \u2022  active "
                                   f"{aw}\u00d7{ah}")
                if sd != self._feed_sd:
                    self._feed_sd = sd
                    if sd:
                        log.warning("HDMI feed degraded (active %dx%d, not "
                                    "full-height HD) — re-arm the camera "
                                    "output or re-plug the capture device",
                                    aw, ah)
                    else:
                        log.info("HDMI feed back to HD geometry")
        dt = time.perf_counter() - t_wall0
        inst = 1.0 / dt if dt > 0 else 0.0
        self._fps = 0.9 * self._fps + 0.1 * inst if self._fps else inst
        if self._play_tick and self._play_tick % 900 == 0:
            log.info("health: display %.1f fps, pipeline %.0f ms/frame",
                     self._fps, getattr(self._pipeline, "_last_ms", 0.0))

        self.frame_ready.emit(FramePacket(
            feed_sd=self._feed_sd, feed_info=self._feed_info,
            perspective=res.frame_bgr, birdseye=res.rect_bgr, status=res.status,
            fps=self._fps, n_balls=res.n_balls, shot_state=res.shot_state,
            clock_remaining=self._clock.remaining(t),
            clock_enabled=self._clock.enabled and self._clock_allowed,
            clock_warning=self._clock.is_warning(t),
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
        # Near-full resolution so one recording serves BOTH playback/testing and
        # training. Raw (unannotated) so replaying it re-runs the CURRENT analysis
        # over old footage — the app-testing use case.
        img = self._small(frame, max_w=1920) if frame.shape[1] > 1920 else frame
        if self._recorder == "pending":
            # Crop the HDMI letterbox once per recording: the T3i's live feed
            # fills only ~63% of the 1080p frame, so recording the content box
            # makes the clip all image instead of a picture floating in black.
            # (Played-back clips recalibrate on their own frames, so pipeline
            # coordinates stay valid.)
            from ..capture.videowriter import content_box, open_writer
            self._rec_crop = content_box(img)
            if self._rec_crop is not None:
                x0, y0, x1, y1 = self._rec_crop
                img = img[y0:y1 + 1, x0:x1 + 1]
            h, w = img.shape[:2]
            self._rec_fps = max(10.0, min(self._src_fps, 30))
            self._recorder = open_writer(self._recording_tmp, self._rec_fps, (w, h))
        elif self._rec_crop is not None:
            x0, y0, x1, y1 = self._rec_crop
            img = img[y0:y1 + 1, x0:x1 + 1]
        if self._rec_t0 is None:
            self._rec_t0 = audio_mod.elapsed_monotonic()
        try:
            self._recorder.write(img)
            self._rec_frames += 1
        except Exception:  # noqa: BLE001
            pass

    def _handle_state(self, state: str, t: float) -> None:
        if self._prev_state != "settled" and state == "settled":
            # Table settled: start the clock ONLY when cue-ball tracking isn't
            # available — with a tracked cue, _update_cue_clock owns the start
            # (the countdown begins the moment the CUE BALL stops, which is
            # usually earlier than full-table settle). Live sources only: a
            # countdown over a recorded video is meaningless.
            if self._clock_allowed and t - self._saw_cue_t > self._CUE_GAP_S * 2:
                self._clock.start(t)    # no-op when the clock is disabled
            self._turn_start_t = t
        elif self._prev_state == "settled" and state == "moving":
            if self._clock.running:
                self._clock.stop()      # table-motion strike (fallback stop)
                log.info("shot clock stopped: table motion (made it)")
        self._prev_state = state

    # Cue-ball shot-clock rule (Joe's spec): the countdown starts once the cue
    # ball comes to rest, and the strike stops it — made it in time. The IMU
    # impact (on_cue_impact) is the precise stop; this is the vision side.
    _CUE_MOVE_SPEED = 3.0   # rectified px/frame — clearly rolling, not jitter
    _CUE_STOP_FRAMES = 6    # consecutive at-rest frames before "stopped"
    _CUE_GAP_S = 1.0        # cue absent this long = pocketed / ball-in-hand

    def _update_cue_clock(self, tracks, t: float) -> None:
        if not (self._clock.enabled and self._clock_allowed):
            return
        cue = next((tr for tr in tracks if tr.cls == BallClass.CUE), None)
        if cue is None:
            self._cue_still = 0
            return
        if t - self._saw_cue_t > self._CUE_GAP_S:
            # cue reappeared (scratch -> ball-in-hand, or long occlusion): the
            # next time it rests is a fresh turn even if it was placed gently
            self._clock_armed = True
        self._saw_cue_t = t
        stop_v = max(0.4, float(self._settings.balls.stop_speed))
        if cue.speed > max(self._CUE_MOVE_SPEED, 2.0 * stop_v):
            if self._clock.running:
                self._clock.stop()      # the strike — player made it in time
                log.info("shot clock stopped: cue ball moving (made it)")
            self._clock_armed = True    # rolling cue = the next rest is a new turn
            self._cue_still = 0
            return
        if cue.speed < stop_v:
            self._cue_still += 1
            if (self._cue_still >= self._CUE_STOP_FRAMES
                    and self._clock_armed and not self._clock.running):
                self._clock.start(t)    # cue at rest -> you're on the clock
                self._clock_armed = False  # re-arms on motion/absence, so an
                self._turn_start_t = t     # expired clock can't restart itself
                log.info("shot clock started: cue ball at rest (%ds)",
                         self._settings.shot_clock.seconds)
        else:
            self._cue_still = 0

    def _record_shot(self, event: ShotEvent, t: float) -> None:
        if self._session_id is None:
            return
        shot_seconds = max(0.0, event.start_t - self._turn_start_t) if self._turn_start_t else 0.0
        self._repo.record_shot(
            self._session_id, outcome=event.outcome.value,
            num_pocketed=event.num_pocketed, target_pocket=event.target_pocket,
            cue_scratch=event.cue_scratch, duration_s=event.duration_s,
            shot_seconds=shot_seconds, stroke_json=self._consume_stroke(),
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
