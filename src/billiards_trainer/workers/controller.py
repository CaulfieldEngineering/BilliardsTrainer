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
from ..core.types import BallClass
from ..db.repository import Repository
from ..events.shot_detector import ShotEvent
from ..game.shot_clock import ShotClock
from ..version import __version__
from ..vision.felt import felt_from_point
from ..vision.pipeline import Pipeline

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
    media_t: float = -1.0          # video playback seconds (-1 when live)
    pipeline_t: float = -1.0       # the shot detector's clock (live AND video)
    clock_running: bool = False    # countdown active (status label)
    clock_paused: bool = False
    tracks: list = field(default_factory=list)
    raw_dets: list = field(default_factory=list)   # camera-coord dets + guessed numbers (labelling)
    feed_sd: bool = False   # camera fell back to the 480p HDMI mode (re-arm ML)
    feed_info: str = ""     # corner stats chip: container/active resolution + fps


class PipelineController(QObject):
    frame_ready = Signal(object)        # FramePacket
    stats_updated = Signal(dict)        # session summary
    shot_recorded = Signal(object)      # ShotEvent
    cached_shots = Signal(object)       # list[dict] from the analysis sidecar (UI only)
    status_changed = Signal(str)
    clock_event = Signal(str)           # 'warn' | 'expired'
    error = Signal(str)
    settings_changed = Signal(object)   # Settings (after an in-view tweak, e.g. felt pick)
    replay_saved = Signal(str)          # path to a saved replay clip
    shot_suggested = Signal(object)     # ShotEvent — manual-confirm mode (not recorded)
    recording_changed = Signal(bool)    # recording on/off
    narration = Signal(str)             # spoken table events (ball_in_hand...)
    detection_changed = Signal(bool)    # auto-detection on/off
    capture_progress = Signal(str)      # analysis-capture status text
    capture_saved = Signal(str)         # path to a finished analysis-capture zip
    overlays_loaded = Signal(object)    # playback: shots.json doc (aim/trails)
    failure_flagged = Signal(str)       # path to a staged debug bundle
    source_is_video = Signal(bool, int, float)  # (seekable video?, frame_count, fps)
    video_state = Signal(int, int, bool)  # (current frame, total frames, playing)
    _detections_ready = Signal(object, object)  # (raw_dets, frame_shape) worker -> this thread
    stroke_measured = Signal(object)    # live stroke metrics record (rebased start) -> UI
    _stroke_ready = Signal(object)      # stroke worker -> this thread (the sidecar owner)

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
        # Live recording: the camera grab thread only ENQUEUES (frame, capture
        # timestamp) — a dedicated worker thread does the preprocess + encoder
        # write, so capture is never slowed by recording work and an encoder
        # stall can never stall the camera. Recorder state is shared across the
        # rec worker + this controller thread, guarded by this lock.
        # _live_recording is True while the grab-thread sink owns the feed
        # (live camera); non-live sources fall back to the CV-tick-fed path.
        self._rec_lock = threading.Lock()
        self._live_recording = False
        self._rec_queue: queue.Queue | None = None
        self._rec_thread: threading.Thread | None = None
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
        self._strike_stop_t = -1e9     # break-detection window anchor
        self._break_pending = False    # next countdown gets break_seconds
        from ..game.narration import Narrator
        self._narrator = Narrator()
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
        self._stroke_ready.connect(self._on_stroke_ready, Qt.QueuedConnection)
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
        self._strike_stop_t = -1e9
        self._break_pending = False
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
        self._pace_anchor = None
        if self._timer is None:
            self.on_started()
        self._timer.start(interval)
        self.status_changed.emit("running")
        self.detection_changed.emit(self._detection_enabled)
        # Cached playback: with an analysis sidecar, the pipeline bypasses
        # inference entirely — smooth playback by construction, and the shot
        # timeline populates instantly from the cache.
        if self._pipeline is not None:
            self._pipeline.playback_cache = None   # never leak into live mode
        # ORDER MATTERS: the UI clears its timeline/shot list when the mode
        # signal lands, so the mode must be announced BEFORE the cached shots
        # arrive — the old order delivered 11 shots and wiped them a
        # millisecond later (Joe: the bar says "No shots" while the phone
        # shows them).
        is_vid = bool(getattr(self._source, "is_video", False))
        self.source_is_video.emit(is_vid, getattr(self._source, "frame_count", 0) if is_vid else 0,
                                  self._src_fps)
        if is_vid:
            try:
                from ..vision.analysis_cache import SidecarReader
                if SidecarReader.exists(source_spec):
                    reader = SidecarReader(source_spec)
                    if self._pipeline is not None:
                        self._pipeline.playback_cache = reader
                    off = reader.video_time_offset()
                    # rows in VIDEO time (seeks/lane/hover align with the
                    # picture); 'key' keeps the sidecar clock for the
                    # correction channel
                    self.cached_shots.emit([
                        {**s, "key": s.get("start", 0.0),
                         "start": max(0.0, float(s.get("start", 0.0)) - off),
                         "end": max(0.0, float(s.get("end", 0.0)) - off)}
                        for s in reader.shots])
                    # the compute-once overlay geometry (aim lines, ball
                    # paths) rides the SAME summary the phone reads — the
                    # two surfaces can never disagree (Joe's requirement)
                    try:
                        import json as _json
                        from pathlib import Path as _P
                        sj = _P(str(source_spec) + ".shots.json")
                        if sj.is_file():
                            self.overlays_loaded.emit(
                                _json.loads(sj.read_text(encoding="utf-8")))
                    except (OSError, ValueError):
                        pass
                    log.info("playback from analysis cache (%d states)", len(reader))
            except (OSError, ValueError) as exc:
                log.warning("analysis cache unreadable (%s) — re-analyzing live", exc)
        self._recover_orphan_recordings()
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
    @Slot(bool)
    def set_clock_paused(self, paused: bool) -> None:
        """Joe's rail button. Pause freezes number and edges; resume picks
        up where it left off (the countdown length is not forgiven)."""
        import time as _time
        t = _time.perf_counter() - getattr(self, "_t0", _time.perf_counter())
        if paused:
            self._clock.pause(t)
        else:
            self._clock.resume(t)

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
        if self._recorder == "device":
            # ffmpeg writes one continuous file from the device and cannot be
            # paused mid-stream. Say so rather than leaving a button that looks
            # like it worked; stop/start makes separate files instead.
            log.warning("pause is not available while recording from the device")
            return
        self._recording_paused = paused
        if self._recorder is not None and hasattr(self._recorder, "pause"):
            self._recorder.pause(paused)
        if self._recorder is None or self._audio is None:
            return
        # Audio must pause WITH the frames or the tracks drift apart: each
        # unpaused stretch is its own segment, concatenated at stop.
        if paused:
            self._audio.pause()
            # Guard the pause accounting against the grab thread's _rec_t0 write.
            with self._rec_lock:
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
            self._recording_paused = False
            # Analysis sidecar: the pipeline's output is recorded ONCE, here,
            # so playback never has to re-run inference (Joe's architecture).
            try:
                from ..vision.analysis_cache import SidecarWriter
                self._sidecar = SidecarWriter(self._recording_path,
                                              {"fps": self._src_fps})
                self._sidecar_last_t = -1.0
            except OSError as exc:
                log.warning("sidecar unavailable (%s) — playback will re-analyze", exc)
                self._sidecar = None
            self._rec_frames = 0
            self._rec_t0 = None
            self._rec_elapsed = 0.0
            self._rec_crop = None
            # DEVICE-OWNED RECORDING (preferred). When the source is owned by
            # ffmpeg, the file is written inside ffmpeg straight from the device
            # — its own timestamps, its own encoder, the mic muxed natively — so
            # NOTHING the analysis pipeline does can appear in the saved video.
            # Verified by stalling the analysis reader 1.2s every 3s mid-record:
            # the file came out byte-identical. Frame-pumping paths below can
            # never offer that, because the recording inherits Python's timing.
            if hasattr(self._source, "start_recording"):
                self._recorder = "device"
                self._source.start_recording(
                    self._recording_tmp, filters=self._device_rec_filters())
                self._rec_t0 = audio_mod.elapsed_monotonic()
                self._session_id = self._repo.start_session(
                    mode=self._mode, drill_key=None, drill_target=0,
                    table_size=self._settings.table.size)
                self.stats_updated.emit(self._repo.session_summary(self._session_id))
                self.recording_changed.emit(True)
                return
            self._recorder = "pending"  # opened lazily on first frame (need size)
            # REAL-TIME CAPTURE (priority path): if the source exposes a frame
            # sink (the threaded live camera), record at true camera cadence —
            # decoupled from the CV tick so a busy pipeline can never starve
            # the recording into micro-stutter. The grab thread only enqueues;
            # a worker thread does preprocess + the (possibly blocking) encoder
            # write. Sources without a sink (video review, demo) keep the
            # tick-fed path.
            self._live_recording = hasattr(self._source, "set_frame_sink")
            if self._live_recording:
                self._rec_queue = queue.Queue(maxsize=8)
                self._rec_thread = threading.Thread(
                    target=self._rec_worker, args=(self._rec_queue,),
                    daemon=True, name="rec-writer")
                self._rec_thread.start()
                self._source.set_frame_sink(self._record_frame_live)
            self._audio = audio_mod.make_recorder(rec.audio, rec.audio_device)
            self._audio_dir = EXPORTS_DIR / f".audio-{stamp}"
            # The mic's lead-in (device open latency) is corrected at MUX by
            # measurement, not by gating here: ffmpeg buffers its WAV writes so
            # there is no reliable "flowing yet?" signal to wait on.
            self._audio.start(self._audio_dir)
            # Stats live and die with the recording: fresh session, zeroed count.
            self._session_id = self._repo.start_session(
                mode=self._mode, drill_key=None, drill_target=0,
                table_size=self._settings.table.size)
            self.stats_updated.emit(self._repo.session_summary(self._session_id))
            self.recording_changed.emit(True)
        elif not on and self._recorder == "device":
            # Device-owned: ffmpeg finalises the mp4 itself (audio already muxed
            # in-process), so there is no writer to release and no audio track to
            # splice afterwards.
            self._recorder = None
            self._source.stop_recording()
            if self._rec_t0 is not None:
                self._rec_elapsed += audio_mod.elapsed_monotonic() - self._rec_t0
                self._rec_t0 = None
            if self._session_id is not None:
                self._repo.end_session(self._session_id)
                self._session_id = None
            self.recording_changed.emit(False)
            self._finalize_recording_file()
        elif not on and self._recorder is not None:
            # Teardown order matters: detach the grab-thread sink so no new
            # frame enqueues, drain/stop the rec worker (sentinel + join) so no
            # write is in flight, THEN release the writer.
            if self._live_recording and hasattr(self._source, "set_frame_sink"):
                self._source.set_frame_sink(None)
            q, t = self._rec_queue, self._rec_thread
            self._rec_queue, self._rec_thread = None, None
            if q is not None:
                q.put(None)                      # sentinel: finish then exit
            if t is not None:
                t.join(timeout=5.0)
            with self._rec_lock:
                recorder, self._recorder = self._recorder, None
                self._live_recording = False
            if hasattr(recorder, "release"):
                recorder.release()
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
            self._finalize_recording_file()

    def _content_box_now(self):
        """The active-picture box, remembered across recordings.

        A single dark or dropped frame at the instant Record is pressed would
        otherwise silently give a session framed in black bars, so the last
        good box is reused when the current frame can't be measured. Sampling a
        couple of frames costs a few ms and avoids that on a momentary glitch.
        """
        from ..capture.videowriter import content_box
        for _ in range(3):
            frame = self._source.read() if self._source is not None else None
            if frame is not None:
                box = content_box(frame)
                if box is not None:
                    self._last_content_box = box
                    return box
            time.sleep(0.05)
        return getattr(self, "_last_content_box", None)

    def _device_rec_filters(self) -> str:
        """The -vf chain for a DEVICE-OWNED recording.

        Mirrors what the frame-pumping path applied in Python (orientation from
        the camera settings, then the measured denoise + mild sharpen), but runs
        inside ffmpeg so no frame ever crosses into our process to be recorded.
        Output is forced to 30fps CFR from a 60fps container.
        """
        cam = self._settings.camera
        parts = ["fps=30"]
        # LETTERBOX CROP, first and in the SOURCE orientation. The T3i's live
        # HDMI fills only ~63% of the 1080p frame — the picture floats in black
        # bars — so recording the raw frame gives a video framed in black. The
        # analysis stream is the same un-rotated frame ffmpeg is recording, so
        # the box measured here maps straight onto the recording.
        box = self._content_box_now()
        if box is not None:
            x0, y0, x1, y1 = box
            self._rec_crop = box
            parts.append(f"crop={x1 - x0 + 1}:{y1 - y0 + 1}:{x0}:{y0}")
        else:
            log.warning("no letterbox detected — recording the full frame")
        rot = int(getattr(cam, "rotation", 0)) % 360
        if rot == 90:
            parts.append("transpose=1")
        elif rot == 180:
            parts.append("transpose=1,transpose=1")
        elif rot == 270:
            parts.append("transpose=2")
        if getattr(cam, "flip_h", False):
            parts.append("hflip")
        if getattr(cam, "flip_v", False):
            parts.append("vflip")
        # Measured on the HD feed: light denoise + mild unsharp gave +35% frame
        # detail (laplacian 48.5 -> 65.4) with no visible processing artefacts.
        parts.append("hqdn3d=1:1:4:4")
        parts.append("unsharp=5:5:0.3:5:5:0.0")
        # yuv420p subsamples chroma 2x2, so odd dimensions are invalid H.264 and
        # AMD's encoder refuses them outright.
        parts.append("crop=trunc(iw/2)*2:trunc(ih/2)*2")
        # SQUARE PIXELS, forced. The Cam Link declares its 1920x1080 MJPEG as
        # "SAR 1920:1080 DAR 256:81" — a 16:9 pixel aspect on an already-16:9
        # frame, which is nonsense; square pixels are SAR 1:1. transpose then
        # inverts that lie to 9:16 and every player renders the result squashed.
        # OpenCV never showed this because it ignores aspect metadata and hands
        # over raw pixels; ffmpeg honours it.
        parts.append("setsar=1")
        return ",".join(parts)

    @staticmethod
    def _remux_faststart(path) -> None:
        """Re-container a FINISHED fragmented recording as faststart, in place.

        Recordings are written fragmented (crash-safe: playable at any cut),
        but a fragmented mp4 carries an EMPTY front moov — iOS/Safari can't
        even learn the duration from the head, let alone seek, so playing a
        late shot on the phone forced downloading nearly the whole file
        (measured ~2 GB for a median shot in a 36-min session). A copy-only
        remux moves a complete index to the front: ~1.4 s/GB, no re-encode,
        and the phone's cost drops to ~5-10 MB per seek. Runs on the hidden
        .part file BEFORE it lands in the Dropbox-synced folder, so nothing
        (Dropbox, the derive pass, cv2 readers) can race it. Failure is
        non-fatal: the fragmented original stays, still playable everywhere
        but slow on the phone."""
        import subprocess
        from pathlib import Path as _P

        from ..capture.audio import NO_WINDOW, find_ffmpeg
        p = _P(path)
        ff = find_ffmpeg()
        if ff is None or not p.is_file():
            return
        tmp = p.with_suffix(".fs.mp4")
        try:
            r = subprocess.run(
                [ff, "-v", "error", "-i", str(p), "-c", "copy",
                 "-movflags", "+faststart", "-y", str(tmp)],
                capture_output=True, timeout=300, creationflags=NO_WINDOW)
            # remuxed file drops moof overhead but must stay ~the same size —
            # a short output means ffmpeg bailed partway
            if r.returncode == 0 and tmp.stat().st_size > p.stat().st_size * 0.9:
                tmp.replace(p)
                log.info("recording re-containered faststart: %s", p.name)
            else:
                err = (r.stderr or b"")[-300:].decode("utf-8", "replace")
                log.warning("faststart remux failed (%s); keeping fragmented "
                            "file %s", err.strip() or r.returncode, p.name)
                tmp.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001 - remux is best-effort, never fatal
            log.exception("faststart remux errored; keeping fragmented file")
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    def _finalize_recording_file(self) -> None:
        """Move the finished in-progress file into the recordings folder."""
        # Stop the live stroke worker FIRST: an open cv2 handle on the
        # .part fails the replace() below (Windows). The abort() hook
        # kills an in-flight measurement within ~1s; unmeasured shots are
        # exactly what the close-time annotate_session pass computes.
        stop = getattr(self, "_stroke_stop", None)
        if stop is not None:
            stop.set()
            q = getattr(self, "_stroke_queue", None)
            while q is not None and not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
            th = getattr(self, "_stroke_thread", None)
            if (th is not None and th.is_alive()
                    and getattr(self, "_stroke_busy", False)):
                th.join(timeout=8.0)   # abort() lands within ~1s of decode
        sc = getattr(self, "_sidecar", None)
        if sc is not None:
            sc.close()
            self._sidecar = None
        if not self._recording_path:
            return
        from pathlib import Path as _P
        try:
            if _P(self._recording_tmp).is_file():
                self._remux_faststart(self._recording_tmp)
                for attempt in range(4):
                    try:
                        _P(self._recording_tmp).replace(self._recording_path)
                        break
                    except PermissionError:
                        if attempt == 3:
                            raise
                        time.sleep(1.0)  # straggler read handle draining
        except OSError:
            log.exception("finalizing recording failed")
        self.replay_saved.emit(self._recording_path)
        # Post-recording outcome derivation: the live detector's in-the-
        # moment attribution measured 7/11 against frame truth; deriving
        # from the finished identity record measured 11/11 (GOALS, loops
        # 32-33). Runs OFF the recording path (daemon thread, never raises,
        # sidecar already closed) and APPENDS corrections — Joe's own
        # review verdicts, appended later, still win by file order.
        import threading
        video = self._recording_path

        def _derive() -> None:
            # ONE canonical close pass (vision/shot_pass.py) — this body
            # was one of three hand-copied variants that had already
            # drifted once. Stage details/order live there now.
            try:
                from ..vision.shot_pass import run_close_pass
                out = run_close_pass(video)
                log.info("session close pass for %s: %s", video, out)
            except Exception:  # noqa: BLE001 - never disturb the app
                log.exception("session close pass failed")

        threading.Thread(target=_derive, name="derive-outcomes",
                         daemon=True).start()

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
        if getattr(self._source, "is_video", False):
            # WALL-CLOCK pacing: audio plays at true speed, so video must hold
            # it or the sync layer drags audio back (~every 4s, replaying the
            # last second — Joe's "audio keeps looping"). Playback is an
            # analyzer over a real-time clip: when processing falls behind the
            # clock, DROP frames to catch up (decode-only, ~10ms each) instead
            # of stretching time. Anchor resets on start/seek/speed/pause.
            now = time.perf_counter()
            anchor = getattr(self, "_pace_anchor", None)
            if anchor is None:
                self._pace_anchor = (now, self._video_pos)
            else:
                t0, f0 = anchor
                expected = f0 + (now - t0) * self._src_fps * self._speed
                behind = expected - self._video_pos
                skipped = 0
                while behind > 1.5 and skipped < 6:
                    if self._source.read() is None:
                        break
                    self._video_pos = max(0, self._source.position() - 1)
                    behind -= 1.0
                    skipped += 1
                    self._play_tick += 1
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
            # Detection cadence: detection costs ~83ms/frame on this machine
            # (measured, tools/bench_pipeline.py) while a frame's budget at 1x
            # is 33ms — detecting EVERY frame dragged playback to 9fps ("super
            # super slow", Joe, watching a clip back). Display every frame;
            # DETECT at the same ~10Hz the live async path uses (every 3rd at
            # 1x, sparser at higher speeds). The tracker is built to coast
            # between detections. Step/seek/stop call _run_frame directly with
            # detect forced on, so a paused/landed frame is always fully
            # analysed. Offline tools drive the Pipeline directly and keep
            # their own (deterministic) cadence.
            stride = max(4, round(self._speed))
            self._run_frame(frame, detect=(self._play_tick % stride == 0))
            import time as _time
            now = _time.perf_counter()
            if now - getattr(self, "_pb_log_t", 0.0) >= 5.0:
                n = self._play_tick - getattr(self, "_pb_log_n", 0)
                if getattr(self, "_pb_log_t", 0.0) > 0:
                    log.info("playback worker: %.1f fps produced", n / (now - self._pb_log_t))
                self._pb_log_t, self._pb_log_n = now, self._play_tick
        else:
            # LIVE camera: async vision. The display path never runs inference
            # (stays at camera rate); frames are handed to the detection worker
            # whenever it's idle, and its results are ingested between ticks.
            self._run_frame(frame, detect="async")

    def _recover_orphan_recordings(self) -> None:
        """Rescue in-progress recordings abandoned by a crash.

        The rig's SSD panics (hardware), so sessions do get cut mid-write. The
        .part files are fragmented mp4 and therefore playable, so move anything
        substantial into the recordings folder instead of leaving it hidden.
        """
        try:
            rec_dir = self._settings.recording.resolved_dir()
            rec_dir.mkdir(parents=True, exist_ok=True)
            for part in sorted(EXPORTS_DIR.glob(".session-*.part.mp4")):
                if part.stat().st_size < 2_000_000:      # < ~2MB: not worth it
                    part.unlink(missing_ok=True)
                    continue
                stamp = part.name[len(".session-"):-len(".part.mp4")]
                dest = rec_dir / f"session-{stamp}-recovered.mp4"
                # a truncated fragmented file remuxes up to its last complete
                # fragment; on failure the raw (still playable) file moves as-is
                self._remux_faststart(part)
                part.replace(dest)
                log.info("recovered interrupted recording -> %s", dest.name)
                self.replay_saved.emit(str(dest))
            for d in sorted(EXPORTS_DIR.glob(".audio-*")):
                for f in d.glob("*"):
                    f.unlink(missing_ok=True)
                d.rmdir()
        except OSError:
            log.exception("orphan-recording recovery failed")

    # --- async vision (live camera) ------------------------------------- #
    # --- live stroke metrics (Joe: "populate as soon as the shot is
    # complete") ------------------------------------------------------- #
    def _submit_stroke(self, start: float) -> None:
        """Queue a completed shot (REBASED start) for stroke measurement
        from the growing .part. Same lazy-spawn + self-heal shape as the
        detect worker."""
        if getattr(self, "_stroke_queue", None) is None:
            self._stroke_queue = queue.Queue()
            self._stroke_stop = threading.Event()
            self._stroke_thread = None
        if self._stroke_thread is None or not self._stroke_thread.is_alive():
            if self._stroke_thread is not None:
                log.warning("stroke worker was dead — respawning")
            self._stroke_thread = threading.Thread(
                target=self._stroke_worker, daemon=True, name="stroke-vision")
            self._stroke_thread.start()
        self._stroke_stop.clear()
        self._stroke_queue.put((str(self._recording_tmp), float(start)))

    def _stroke_worker(self) -> None:
        """Measure stroke metrics for completed shots from the in-progress
        recording. Feasibility measured 2026-08-24: a fresh VideoCapture
        indexes every flushed ~8s fragment (a persistent handle dies at
        EOF forever); frame-accurate seeks; [start-6, start+9] becomes
        decodable by strike+18s worst case. Readiness is gated on an
        actual seek+grab at start+POST (CAP_PROP_FRAME_COUNT overstates
        the decodable end by up to ~4s)."""
        try:
            import ctypes
            k = ctypes.WinDLL("kernel32", use_last_error=True)
            k.GetCurrentThread.restype = ctypes.c_void_p
            k.SetThreadPriority.argtypes = [ctypes.c_void_p, ctypes.c_int]
            # LOWEST, one notch under the detect worker: metrics must never
            # steal cadence from live detection
            k.SetThreadPriority(k.GetCurrentThread(), -2)
        except Exception:  # noqa: BLE001 - priority is best-effort
            pass
        from ..vision.stroke_vision import (
            POST_S,
            STROKE_VISION_VERSION,
            AbortMeasurement,
            _Session,
            measure_shot,
        )
        sess = None
        sess_path = None
        while True:
            video, start = self._stroke_queue.get()
            stop = self._stroke_stop
            if stop.is_set():
                continue        # session over — close-time pass covers it
            self._stroke_busy = True
            # readiness: poll a fresh open + seek + grab at start+POST
            ready = False
            for _ in range(12):
                if stop.is_set():
                    break
                cap = cv2.VideoCapture(video)
                cap.set(cv2.CAP_PROP_POS_MSEC, (start + POST_S) * 1000.0)
                ok = cap.grab()
                cap.release()
                if ok:
                    ready = True
                    break
                stop.wait(2.0)
            if not ready or stop.is_set():
                self._stroke_busy = False
                continue
            try:
                if sess_path != video:
                    sess = _Session(video)
                    sess_path = video
                rec = measure_shot(sess, start, abort=stop.is_set)
            except AbortMeasurement:
                self._stroke_busy = False
                continue
            except Exception:  # noqa: BLE001 - one bad shot never kills the worker
                self._stroke_busy = False
                log.exception("live stroke measurement failed @%.1fs", start)
                continue
            rec.update({"type": "stroke_vision", "v": STROKE_VISION_VERSION,
                        "start": round(start, 3)})
            self._stroke_busy = False
            self._stroke_ready.emit(rec)

    def _on_stroke_ready(self, rec: dict) -> None:
        """Controller thread: the ONLY safe place to append to the live
        sidecar (its 'w'-mode buffered handle would overwrite any external
        'a'-mode append on its next flush)."""
        sc = getattr(self, "_sidecar", None)
        if sc is not None:
            try:
                sc.add_stroke(rec)
            except OSError:
                log.exception("live stroke sidecar append failed")
        self.stroke_measured.emit(rec)

    def _submit_detection(self, frame: np.ndarray) -> None:
        """Offer a frame to the detection worker. Non-blocking: if the worker
        is mid-inference the frame is simply display-only."""
        if (not self._detection_enabled or self._pipeline is None
                or self._pipeline._strategy is None
                or not self._pipeline.calib.is_calibrated):
            self._diag_blocked = getattr(self, "_diag_blocked", 0) + 1
            return  # calibration/preview paths handle themselves synchronously
        self._diag_submit = getattr(self, "_diag_submit", 0) + 1
        if self._det_thread is None or not self._det_thread.is_alive():
            # SELF-HEAL: a dead worker starved the schematic of balls for an
            # entire evening (Joe found it, not the loop) — the thread had
            # died on a stale-signal race and nothing ever restarted it. Any
            # submitted frame now resurrects a dead worker.
            if self._det_thread is not None:
                log.warning("detect worker was dead — respawning")
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
        # Joe: "my mouse and keyboard have been completely lagging when the
        # app and AI are running hard." Inference is the app's CPU hog; the
        # RECORDING path stays at normal priority (frames must never drop),
        # but this thread yields to the desktop. Thread-level, so it
        # persists across app restarts unlike the process-level demote
        # applied by hand on 2026-08-23.
        try:
            import ctypes
            # Explicit handle types are REQUIRED: bare windll mangles the
            # 64-bit pseudo-handle and the call fails silently (returns 0,
            # measured 2026-08-23 — this demote was a no-op until typed).
            k = ctypes.WinDLL("kernel32", use_last_error=True)
            k.GetCurrentThread.restype = ctypes.c_void_p
            k.SetThreadPriority.argtypes = [ctypes.c_void_p, ctypes.c_int]
            if not k.SetThreadPriority(k.GetCurrentThread(), -1):  # BELOW_NORMAL
                log.warning("detect-worker thread demote failed (err %d)",
                            ctypes.get_last_error())
        except Exception:  # noqa: BLE001 - priority is best-effort
            pass
        while True:
            frame, calib = self._det_queue.get()
            try:
                raw = self._pipeline._strategy.detect(frame, calib)
            except Exception:  # noqa: BLE001 - a bad frame must not kill the worker
                raw = []
            try:
                self._detections_ready.emit(raw, frame.shape)
            except RuntimeError:
                # The controller QObject backing this signal was deleted (a
                # source-switch race). Exit CLEANLY — _submit_detection
                # respawns a worker for whichever controller is alive.
                log.warning("detect worker: signal source gone, exiting")
                return

    @Slot(object, object)
    def _on_detections_ready(self, raw_dets, frame_shape) -> None:
        """Apply worker results (controller thread — the only mutator)."""
        if (not self._running or self._pipeline is None
                or not self._detection_enabled
                or getattr(self._source, "is_video", False)):
            return
        self._diag_ingest = getattr(self, "_diag_ingest", 0) + 1
        self._diag_raw = len(raw_dets)
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
        # Cached playback syncs to the DISPLAYED frame: wall-clock t drifts
        # seconds away from the video position under pacing/frame drops, and
        # the schematic animated ~3s behind the footage (Joe's report). For
        # video sources, media time IS the truth.
        if getattr(self._source, "is_video", False) and self._src_fps:
            t = max(0.0, self._video_pos / float(self._src_fps))
        try:
            res = self._pipeline.process(frame, t, annotate=not self._label_mode, detect=detect)
        except Exception as exc:  # noqa: BLE001 - never let one bad frame kill the loop
            log.exception("pipeline error")
            self.error.emit(f"Pipeline error: {exc}")
            return

        self._diag_pub = len(res.tracks)
        self._handle_state(res.shot_state, t)
        # narration gate (Joe): "Ball in hand" vs "Table change", live
        if self._clock_allowed:
            by_id = {tr.id: tr for tr in res.tracks}
            carried = [by_id[i] for i in (res.carried_ids or ()) if i in by_id]
            has_cue = any(tr.cls == BallClass.CUE for tr in carried)
            has_obj = any(tr.cls != BallClass.CUE for tr in carried)
            kind = self._narrator.update(has_cue, has_obj,
                                         res.shot_state == "moving", t)
            if kind:
                self.narration.emit(kind)
        # sidecar: record tracking states at ~10Hz while a session records
        sc = getattr(self, "_sidecar", None)
        if sc is not None and res.status == "tracking"                 and t - getattr(self, "_sidecar_last_t", -1.0) >= 0.1:
            self._sidecar_last_t = t
            try:
                sc.add_frame(t, res.tracks,
                             carried_ids=res.carried_ids,
                             foreign_frac=res.foreign_frac)
            except OSError:
                pass
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
        # the pipeline for testing, and reusable as training data. Only for the
        # tick-fed fallback (video review / demo); a live camera records off its
        # grab thread via _record_frame_live, so we must NOT also write here.
        if (self._recorder is not None and not self._recording_paused
                and not self._live_recording and self._last_frame is not None):
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
                # HD-vs-degraded discriminator = the ACTIVE BOX ASPECT.
                # Measured on this rig (docs/feedmeter-log.csv):
                #   healthy  1633x928=1.76, 1633x946=1.73, 1603x1072=1.50,
                #            1621x1080=1.50, 1920x1080=1.78 (menu, full frame)
                #   degraded 1730x757=2.29 (480p wire), 1710x875=1.95
                # Nothing healthy exceeds the container's own 16:9 (1.78): a
                # wider active box means the picture is letterboxed beyond the
                # container, which on this camera means an SD/failed mode.
                # (Height-fill was WRONG — movie-mode HD is legitimately 928.)
                aw = int(round((xs.max() - xs.min() + 1) * 8))
                ah = int(round((ys.max() - ys.min() + 1) * 8))
                sd = (aw / max(1, ah)) > 1.85
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
        if self._play_tick and self._play_tick % 150 == 0:
            # Vision-chain heartbeat: every link, countable. Joe watched an
            # empty schematic for an evening while the log said everything
            # was fine — this line makes "which link is dead" a read, not an
            # investigation. (submit=frames offered, blocked=gate refusals,
            # raw=last detection count, ingest=results applied, pub=tracks)
            log.info("health: display %.1f fps, pipeline %.0f ms | vision "
                     "submit=%d blocked=%d worker=%s raw=%d ingest=%d pub=%d",
                     self._fps, getattr(self._pipeline, "_last_ms", 0.0),
                     getattr(self, "_diag_submit", 0),
                     getattr(self, "_diag_blocked", 0),
                     "alive" if (self._det_thread is not None
                                 and self._det_thread.is_alive()) else "DEAD",
                     getattr(self, "_diag_raw", -1),
                     getattr(self, "_diag_ingest", 0),
                     getattr(self, "_diag_pub", -1))

        self.frame_ready.emit(FramePacket(
            feed_sd=self._feed_sd, feed_info=self._feed_info,
            perspective=res.frame_bgr, birdseye=res.rect_bgr, status=res.status,
            fps=self._fps, n_balls=res.n_balls, shot_state=res.shot_state,
            clock_remaining=self._clock.remaining(t),
            clock_enabled=self._clock.enabled and self._clock_allowed,
            clock_warning=self._clock.is_warning(t),
            deviated=res.deviated, tracks=res.tracks, raw_dets=res.raw_dets,
            media_t=(t if getattr(self._source, "is_video", False) else -1.0),
            pipeline_t=t,
            clock_running=self._clock.running,
            clock_paused=getattr(self._clock, "paused", False),
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
        self._pace_anchor = None   # resume re-anchors the wall clock
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
        self._pace_anchor = None   # new position = new wall-clock anchor
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
        self._pace_anchor = None   # new speed = new wall-clock anchor
        if self._timer is not None:
            self._timer.start(max(4, int(self._base_interval / self._speed)))

    @staticmethod
    def _small(frame: np.ndarray, max_w: int = 480) -> np.ndarray:
        h, w = frame.shape[:2]
        if w <= max_w:
            return frame.copy()
        scale = max_w / w
        return cv2.resize(frame, (max_w, int(h * scale)))

    def _record_frame_live(self, frame: np.ndarray) -> None:
        """Camera grab-thread sink: hand ONE source frame to the recording
        worker, stamped with its capture time. Deliberately does NOTHING else —
        no preprocess, no encoding — so the grab thread's cadence (and CV's
        frame supply) is never disturbed by recording work. Never raises (a
        throw here propagates into the capture loop).

        The capture timestamp is the whole point: the writer retimes frames
        into CFR slots from it, so playback cadence mirrors real capture
        cadence instead of beating against a wall-clock pacer.
        """
        q = self._rec_queue
        if q is None or self._recording_paused:
            return
        try:
            q.put_nowait((frame, time.monotonic()))
        except queue.Full:
            pass  # encoder stalled >~250ms; the retimer bridges the gap

    def _rec_worker(self, q: queue.Queue) -> None:
        """Recording worker: preprocess (rotation/flip/colour, matching the live
        view) + feed the encoder. Blocking stdin writes are fine HERE — the
        queue absorbs brief encoder stalls while capture continues untouched.
        Holds its own queue reference so teardown can drain it to the sentinel."""
        while True:
            item = q.get()
            if item is None:
                return
            frame, ts = item
            try:
                self._write_recording(
                    preprocess_frame(frame, self._settings.camera), ts)
            except Exception:  # noqa: BLE001 - one bad frame must not end recording
                log.exception("recording write failed; frame skipped")

    def _write_recording(self, frame: np.ndarray, ts: float | None = None) -> None:
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
            # FfmpegWriter retimes from the capture timestamp; the cv2 fallback
            # writer only takes the frame.
            try:
                self._recorder.write(img, ts)
            except TypeError:
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
        # Break detection: 5+ balls rolling at once within 4s of a strike =
        # that was the break; the NEXT countdown gets break_seconds (Joe's
        # "time after break" - league convention, survey the spread).
        if t - getattr(self, "_strike_stop_t", -1e9) < 4.0:
            movers = sum(1 for tr in tracks if tr.speed > self._CUE_MOVE_SPEED)
            if movers >= 5 and not self._break_pending:
                self._break_pending = True
                log.info("shot clock: break detected (%d movers) - next "
                         "countdown %ds", movers,
                         self._settings.shot_clock.break_seconds)
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
                self._strike_stop_t = t     # break-detection window opens
            self._clock_armed = True    # rolling cue = the next rest is a new turn
            self._cue_still = 0
            return
        if cue.speed < stop_v:
            # Joe's clarification: the countdown starts when ALL balls come
            # to rest, not just the cue - an object ball still rolling
            # resets the stillness run.
            others_rolling = any(
                tr.cls != BallClass.CUE and tr.active
                and tr.speed > max(self._CUE_MOVE_SPEED, 2.0 * stop_v)
                for tr in tracks)
            if others_rolling:
                self._cue_still = 0
                return
            self._cue_still += 1
            if (self._cue_still >= self._CUE_STOP_FRAMES
                    and self._clock_armed and not self._clock.running):
                if self._break_pending:
                    self._break_pending = False
                    self._clock.set_next_seconds(
                        self._settings.shot_clock.break_seconds)
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
        sc = getattr(self, "_sidecar", None)
        if sc is not None:
            try:
                sc.add_shot(event)
            except OSError:
                pass
        # The UI gets RECORDING time, same base the sidecar writes: live
        # pipeline t is SOURCE UPTIME, so an unrebased event 20 minutes
        # into a camera session landed at t=1200 on a lane showing 0-120
        # and no marker ever appeared while recording (Joe's report — the
        # same timebase bug the sidecar writer fixed, one layer up).
        ui_event = event
        if sc is not None and getattr(sc, "_t0", None) is not None:
            import dataclasses
            try:
                ui_event = dataclasses.replace(
                    event,
                    start_t=max(0.0, event.start_t - sc._t0),
                    end_t=max(0.0, event.end_t - sc._t0))
            except TypeError:
                ui_event = event
        self.shot_recorded.emit(ui_event)
        # Live stroke metrics (Joe: "populate as soon as the shot is
        # complete"): enqueue the REBASED start — same rounding as
        # add_shot, so the close-time annotate_session pass recognises
        # live-measured shots and skips them.
        if sc is not None and getattr(sc, "_t0", None) is not None \
                and self._recording_tmp:
            try:
                self._submit_stroke(round(max(0.0, event.start_t - sc._t0), 3))
            except Exception:  # noqa: BLE001 - metrics are enrichment
                log.exception("stroke enqueue failed")
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
