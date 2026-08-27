"""Main window: navigation rail + stacked pages, wired to the pipeline thread.

Owns the Repository and the PipelineController (on its own QThread). All
controller method calls are made through queued signal/slot connections so CV
work never runs on the UI thread.
"""

import logging
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QMainWindow,
    QStackedWidget,
    QWidget,
)

from ..config import Settings
from ..db.repository import Repository
from ..version import APP_NAME, __version__
from ..workers.controller import PipelineController, make_controller_thread
from .pages.live_page import LivePage
from .pages.settings_page import SettingsPage
from .theme import apply_theme

log = logging.getLogger("ui")


class _AutoLabelWorker(QThread):
    """Runs the session -> VLM -> TrainingStore pipeline off the UI thread.
    Torch-free (ONNX detection + an HTTPS call to the vision model)."""
    progress = Signal(str)
    done = Signal(bool, str)   # ok, human-readable report

    def __init__(self, session: Path, settings, store_dir: Path):
        super().__init__()
        self._session, self._settings, self._store_dir = session, settings, store_dir

    def run(self) -> None:
        try:
            from ..train import TrainingStore
            from ..train.autolabel import autolabel_session
            store = TrainingStore(self._store_dir)
            rep = autolabel_session(self._session, self._settings, store,
                                    progress=self.progress.emit)
            msg = (f"AI labelled {rep['balls_saved']} balls across "
                   f"{rep['layouts']} layouts. Real-time detector agreed "
                   f"{rep['heuristic_agreement_pct']}%. Now tap Train.")
            self.done.emit(rep["balls_saved"] > 0, msg)
        except Exception as exc:  # noqa: BLE001 - surface to the UI
            self.done.emit(False, f"Auto-label failed: {exc}")


class _StreamingTrainWorker(QThread):
    """Runs finetune_ballid.py, streaming per-epoch progress ('Epoch 12/80' →
    15%) so the Training rail can show a real bar + log instead of a black box."""
    progress = Signal(int, str)   # percent (or -1 = indeterminate), latest line
    done = Signal(bool, str)

    def __init__(self, py: str, data: str, out: str):
        super().__init__()
        self._py, self._data, self._out = py, data, out

    def run(self) -> None:
        import re
        import subprocess

        from ..capture.audio import NO_WINDOW
        root = Path(__file__).resolve().parents[3]
        tail: list[str] = []
        try:
            proc = subprocess.Popen(
                [self._py, str(root / "tools" / "finetune_ballid.py"),
                 "--data", self._data, "--out", self._out],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
                creationflags=NO_WINDOW)
            epoch_re = re.compile(r"(\d+)/(\d+)")
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                tail.append(line)
                del tail[:-30]
                m = epoch_re.search(line)
                if ("/" in line and m and "Epoch" not in line
                        and line.split()[0].count("/") == 1):
                    cur, tot = int(m.group(1)), int(m.group(2))
                    if 0 < cur <= tot and tot > 1:
                        self.progress.emit(int(100 * cur / tot),
                                           f"Epoch {cur}/{tot}")
                elif "export" in line.lower():
                    self.progress.emit(97, "Exporting model…")
            proc.wait(timeout=60)
            ok = proc.returncode == 0 and any("FINETUNE_OK" in t for t in tail)
            self.done.emit(ok, "\n".join(tail[-3:]))
        except Exception as exc:  # noqa: BLE001
            self.done.emit(False, str(exc))


class MainWindow(QMainWindow):
    # Signals carrying a Python ``object`` marshal cleanly across the thread
    # boundary (unlike QMetaObject.invokeMethod + Q_ARG(object)), so settings
    # pushes to the worker thread go through this.
    apply_settings_requested = Signal(object)
    start_source = Signal(str, str, str)  # source, mode, drill_key -> controller.start
    set_strategy_requested = Signal(str)  # live detector strategy -> controller

    def __init__(self, settings: Settings):
        super().__init__()
        self._settings = settings
        self.setWindowTitle(APP_NAME)

        self._repo = Repository()
        self._controller = PipelineController(settings, self._repo)
        self._thread = make_controller_thread(self._controller)

        self._update_forced = False
        self._pending_feedback_replay: int | None = None
        self._started_source: str | None = None

        from ..sync.supabase import SyncManager, make_sync_thread
        self._sync = SyncManager(self._repo)
        self._sync_thread = make_sync_thread(self._sync)

        # Cue-stroke sensor (Bluetooth IMU). Owns its own daemon BLE thread;
        # a missing sensor/radio/bleak is just a status, never a failure.
        from ..cue.worker import CueSensorWorker
        self._cue = CueSensorWorker(self)

        self._build_ui()
        self._wire()
        self._cue.apply_settings(settings.cue)
        self._autostart_preview()
        self._maybe_autofetch_model()
        self._maybe_check_updates()

    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # The sessions sidebar IS the navigation: LIVE + recordings + Settings.
        # It lives in a SPLITTER so Joe can drag it wider — the fixed width
        # clipped the session table's "Shots" header to "Sho" with no remedy.
        from PySide6.QtWidgets import QSplitter

        from .widgets.sessions_sidebar import SessionsSidebar
        self._sidebar = SessionsSidebar()
        self._sidebar.recordings_dir = self._settings.recording.resolved_dir()
        self._sidebar.refresh()

        self._stack = QStackedWidget()
        self._live = LivePage(self._settings)
        self._settings_page = SettingsPage(self._settings)
        for page in (self._live, self._settings_page):
            self._stack.addWidget(page)

        split = QSplitter(Qt.Horizontal)
        split.setChildrenCollapsible(False)
        split.addWidget(self._sidebar)
        split.addWidget(self._stack)
        split.setStretchFactor(0, 0)   # dragging grows/shrinks the sidebar…
        split.setStretchFactor(1, 1)   # …window resizes go to the content
        split.setSizes([300, 1100])
        root.addWidget(split, 1)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")

    def _toggle_settings(self) -> None:
        self._stack.setCurrentIndex(1 if self._stack.currentIndex() == 0 else 0)

    def _on_live_selected(self) -> None:
        """Sidebar LIVE clicked: back to the camera."""
        self._stack.setCurrentIndex(0)
        self._live.set_training(False)
        source = self._settings.source or "0"
        self._started_source = source
        self.start_source.emit(source, self._settings.mode, "")

    def _on_session_selected(self, path: str) -> None:
        """Sidebar session clicked: play the recording through the analysis."""
        self._stack.setCurrentIndex(0)
        self._started_source = path
        self._live.set_media_path(path)   # synced audio track for the clip
        self.start_source.emit(path, self._settings.mode, "")

    # ------------------------------------------------------------------ #
    def _wire(self) -> None:
        q = Qt.QueuedConnection
        # UI intent -> controller (queued, runs on the worker thread)
        self.start_source.connect(self._controller.start, q)
        self._live.start_requested.connect(self._controller.start, q)
        self._live.stop_requested.connect(self._controller.stop, q)
        self.set_strategy_requested.connect(self._controller.set_detector_strategy, q)
        self._settings_page.detector_changed.connect(self._on_strategy_changed)
        self._settings_page.recalibrate_requested.connect(self._controller.recalibrate, q)
        self._live.retry_requested.connect(self._retry_camera)
        # video transport (UI -> controller, queued onto the worker thread)
        self._live.video_play_pause.connect(self._controller.set_video_paused, q)
        self._live.video_stop_requested.connect(self._controller.video_stop, q)
        self._live.video_step.connect(self._controller.video_step, q)
        self._live.video_seek.connect(self._controller.video_seek, q)
        self._live.video_speed.connect(self._controller.set_playback_speed, q)
        self._controller.source_is_video.connect(self._live.set_video_mode)
        self._controller.video_state.connect(self._live.update_video_state)
        self._live.open_settings_requested.connect(lambda: self._stack.setCurrentIndex(1))
        # Sidebar navigation (the only navigation)
        self._live.mini_view_requested.connect(self._toggle_mini_view)
        self._sidebar.live_selected.connect(self._on_live_selected)
        self._sidebar.session_selected.connect(self._on_session_selected)
        self._sidebar.settings_toggled.connect(self._toggle_settings)
        self._build_menu()   # Joe: a menu bar so narrow mode never strands anything
        self._live.tuning_changed.connect(self._on_tuning_changed)
        self._live.clock_pause_toggled.connect(self._controller.set_clock_paused, q)
        self._live.clock_enabled_toggled.connect(self._sync_clock_menu)
        # Training Mode (label/correct ball numbers on the playback)
        self._live.label_mode_toggled.connect(self._controller.set_label_mode, q)
        self._live.save_training_frame_requested.connect(self._controller.save_training_frame, q)
        self._live.train_balls_requested.connect(self._train_ball_ids)
        self._live.record_toggled.connect(self._controller.set_recording, q)
        self._live.record_pause_toggled.connect(self._controller.set_recording_paused, q)
        self.apply_settings_requested.connect(self._controller.apply_settings, q)

        # controller -> UI.
        # THREADING RULE (crash found by the macOS CI, latent on Windows): a
        # bare lambda has no receiver QObject, so Qt runs it DIRECTLY on the
        # EMITTING (worker) thread — touching widgets there is native-crash
        # territory. Every cross-thread signal must land on a bound method of
        # a UI-thread QObject so the connection auto-queues.
        self._controller.frame_ready.connect(self._live.on_frame)
        self._controller.stats_updated.connect(self._live.on_stats)
        self._controller.shot_recorded.connect(self._live.on_shot)
        self._controller.shot_recorded.connect(self._on_shot_sound)
        self._controller.stroke_measured.connect(self._live.on_stroke_measured)
        self._controller.cached_shots.connect(self._live.on_cached_shots)
        self._controller.overlays_loaded.connect(self._live.on_overlays_loaded)
        self._controller.shot_suggested.connect(self._live.on_suggestion)
        self._controller.recording_changed.connect(self._live.on_recording)
        self._controller.detection_changed.connect(self._live.set_detection)
        self._controller.status_changed.connect(self._live.on_status)
        self._controller.status_changed.connect(self._on_status)
        self._controller.clock_event.connect(self._on_clock_event)
        self._controller.error.connect(self._on_error)
        self._controller.settings_changed.connect(self._on_settings_changed)
        self._controller.replay_saved.connect(self._on_replay_saved)
        self._controller.capture_progress.connect(self._on_capture_progress)
        self._controller.capture_progress.connect(self._live.set_training_count)
        self._controller.capture_saved.connect(self._on_capture_saved)

        # cue-stroke sensor -> UI + controller. The worker emits from its BLE
        # thread, so every connection below auto-resolves to a queued delivery
        # on the receiver's own thread.
        self._cue.status_changed.connect(self._live.on_cue_status)
        self._cue.status_changed.connect(self._settings_page.set_cue_status)
        self._cue.impact.connect(self._live.on_cue_impact)
        self._cue.impact.connect(self._controller.on_cue_impact, q)
        self._cue.metrics.connect(self._live.on_cue_metrics)
        self._cue.metrics.connect(self._controller.on_stroke_metrics, q)
        self._cue.address_resolved.connect(self._on_cue_address)
        self._settings_page.cue_diagnostics_requested.connect(self._open_cue_diagnostics)
        self._settings_page.cue_pair_requested.connect(self._open_cue_pair)
        # Zero-restart camera ownership: when tethered control is enabled, a
        # resident keeper claims the DSLR the instant it appears on USB (app
        # launch / power event), holds its one-per-power-on session, and turns
        # liveview on for the HDMI capture path — no user action, ever.
        if self._settings.camera.tether.enabled:
            from ..capture.tether import start_session_keeper
            start_session_keeper(self._settings.camera.tether)

        # settings
        self._settings_page.applied.connect(self._on_settings_applied)
        self._settings_page.check_updates_requested.connect(self._check_for_updates_forced)
        self._settings_page.feedback_requested.connect(self._open_feedback)
        self._settings_page.train_balls_requested.connect(self._open_ball_trainer)
        self._settings_page.flag_failure_requested.connect(self._controller.flag_failure, q)
        self._controller.failure_flagged.connect(self._on_failure_flagged)
        self._sync.status.connect(self._on_sync_status)

    # Cross-thread signal landing pads (bound methods of this UI-thread QObject
    # -> Qt auto-queues; see the THREADING RULE note in _wire).
    def _on_capture_progress(self, msg: str) -> None:
        self.statusBar().showMessage(f"Capture: {msg}", 4000)

    def _on_sync_status(self, msg: str) -> None:
        self.statusBar().showMessage(f"Sync: {msg}", 4000)

    # ------------------------------------------------------------------ #
    def _autostart_preview(self) -> None:
        """Open the saved camera and start previewing immediately — no Start click.
        Detection stays off (preview); the user opts in via the toggle."""
        source = self._settings.source or "0"
        self._started_source = source
        log.info("Auto-starting camera preview (source=%s)", source)
        self.start_source.emit(source, self._settings.mode, "")

    def _retry_camera(self) -> None:
        source = self._settings.source or "0"
        self._started_source = source
        self.statusBar().showMessage("Retrying camera…", 3000)
        self.start_source.emit(source, self._settings.mode, "")

    def retry_camera(self) -> None:
        """Public re-scan hook (used after camera permission is granted)."""
        self._retry_camera()

    def _maybe_autofetch_model(self) -> None:
        """First-launch convenience: if the trained ball-detection model isn't
        present yet, fetch it in the background so the app tracks with the real
        model out of the box (the cue-ball heuristic runs meanwhile). Silent +
        best-effort — offline just leaves the heuristic running."""
        import os
        # Never auto-download in headless/CI (offscreen) — tests must not hit the
        # network or spawn a 38 MB download mid-run.
        if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            return
        from ..detector_strategies import model_fetch
        try:
            if model_fetch.is_present("pool_yolo11"):
                return
        except Exception:  # noqa: BLE001
            return
        from PySide6.QtCore import QThread

        from .pages.settings_page import _ModelDownloadWorker
        self.statusBar().showMessage("Downloading ball-detection model (one-time)…", 0)
        self._model_dl_thread = QThread(self)
        self._model_dl_worker = _ModelDownloadWorker("pool_yolo11")
        self._model_dl_worker.moveToThread(self._model_dl_thread)
        self._model_dl_thread.started.connect(self._model_dl_worker.run)
        self._model_dl_worker.done.connect(self._on_model_autofetched)
        self._model_dl_worker.failed.connect(self._on_model_dl_failed)
        for sig in (self._model_dl_worker.done, self._model_dl_worker.failed):
            sig.connect(self._model_dl_thread.quit)
        self._model_dl_thread.start()

    def _on_model_dl_failed(self, msg: str) -> None:
        self.statusBar().showMessage(
            f"Model download failed ({msg}); using the basic cue detector.", 6000)

    def _on_model_autofetched(self, _strategy: str) -> None:
        # Re-resolve the live detector ('auto') so the freshly-downloaded model is
        # picked up immediately — no restart, calibration kept.
        self.set_strategy_requested.emit("auto")
        self.statusBar().showMessage("Ball-detection model ready — tracking with the "
                                     "trained model.", 6000)


    def _on_strategy_changed(self, name: str) -> None:
        self._settings.balls.live_strategy = name
        self._settings.save()
        self.set_strategy_requested.emit(name)        # live switch, keeps calibration
        self.statusBar().showMessage(f"Live detector → {name}", 4000)

    def _on_capture_saved(self, path: str) -> None:
        self._last_capture_path = path
        self.statusBar().showMessage(f"Training session saved: {path}", 8000)
        self._live.set_autolabel_status("Session recorded — Train AI when ready.")

    def _autolabel_session(self) -> None:
        """AI auto-label the most recent recorded session into the training store."""
        from ..config import APP_DIR
        from ..train import vlm
        from ..train.autolabel import find_latest_session
        session = (getattr(self, "_last_capture_path", "")
                   or find_latest_session(self._settings.recording.resolved_dir()))
        if not session:
            self._live.set_autolabel_status("Record a training session first.")
            return
        if not vlm.available(self._settings.autolabel):
            self._live.set_autolabel_status(
                "AI labelling needs an OpenRouter API key — add it in Settings → "
                "AI labelling. (Or label manually with the number pad above.)")
            return
        if getattr(self, "_autolabel_worker", None) is not None:
            return
        self._live.set_autolabel_status("AI is labelling the session…", busy=True)
        self._autolabel_worker = _AutoLabelWorker(
            Path(session), self._settings, APP_DIR / "training" / "ballid")
        self._autolabel_worker.progress.connect(
            lambda m: self._live.set_autolabel_status(m, busy=True))
        self._autolabel_worker.done.connect(self._on_autolabel_done)
        self._autolabel_worker.start()

    def _on_autolabel_done(self, ok: bool, report: str) -> None:
        self._autolabel_worker = None
        self._live.set_autolabel_status(report, busy=False)

    def _on_failure_flagged(self, path: str) -> None:
        self.statusBar().showMessage(f"Failure flagged + staged: {path}", 8000)
        self._settings_page.set_debug_status(f"Flagged → {path}")

    def _on_status(self, status: str) -> None:
        self.statusBar().showMessage({"running": "Live — camera preview",
                                      "stopped": "Stopped"}.get(status, status))

    _CAMERA_ERR_HINTS = ("camera", "open source", "couldn't open", "didn't open",
                         "delivering frames", "in use")

    def _on_error(self, msg: str) -> None:
        self.statusBar().showMessage(msg, 8000)
        log.warning("controller error: %s", msg)
        if any(h in msg.lower() for h in self._CAMERA_ERR_HINTS):
            self._live.show_camera_error(msg)

    def _on_shot_sound(self, event) -> None:
        """Joe: a sound when I scratch - live validation of the analysis
        while playing. Fires when the shot FINALIZES (balls settled), the
        moment the system actually concludes 'cue ball pocketed'."""
        try:
            scratched = bool(getattr(event, "cue_scratch", False)) or                 getattr(getattr(event, "outcome", None), "value", "") == "scratch"
            if scratched:
                from .sounds import play
                play("scratch", volume=int(getattr(
                    self._settings.shot_clock, "vol_scratch", 90)))
        except Exception:  # noqa: BLE001 - a chime is never worth a crash
            log.debug("scratch chime failed", exc_info=True)

    def _on_clock_event(self, edge: str) -> None:
        if not self._settings.shot_clock.audio:
            return
        # warn = single beep at 10 s, tick = 3-2-1 cadence, expired = the buzz
        from .sounds import play
        play(edge, volume=int(getattr(self._settings.shot_clock,
                                      f"vol_{edge}", 100)))

    def _push_settings(self) -> None:
        self.apply_settings_requested.emit(self._settings)

    def _on_settings_applied(self) -> None:
        apply_theme(QApplication.instance(), self._settings.ui.accent)
        self._push_settings()
        self._cue.apply_settings(self._settings.cue)
        self._live.set_cue_enabled(self._settings.cue.enabled)
        self._sidebar.recordings_dir = self._settings.recording.resolved_dir()
        self._sidebar.refresh()
        self._live.reconfigure_audio_meter()
        # If the camera source changed, re-open the preview on the new device.
        if (self._settings.source or "0") != self._started_source:
            self._started_source = self._settings.source or "0"
            self.start_source.emit(self._started_source, self._settings.mode, "")
        self.statusBar().showMessage("Settings saved", 4000)

    def _on_tuning_changed(self) -> None:
        """A live-tuning control on the main window changed. The control already
        mutated the SHARED settings object in place, so the pipeline applies it on
        the next frame — no reconfigure / recalibration. We only persist to disk,
        debounced so dragging a slider doesn't thrash the file."""
        timer = getattr(self, "_tuning_save_timer", None)
        if timer is None:
            from PySide6.QtCore import QTimer
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.setInterval(500)
            timer.timeout.connect(self._settings.save)
            self._tuning_save_timer = timer
        timer.start()

    def _on_settings_changed(self, settings) -> None:
        # came from an in-view tweak on the worker thread (e.g. felt pick)
        self._settings.save()
        self._settings_page.reload()
        self.statusBar().showMessage("Felt colour updated — recalibrating", 4000)

    def _on_replay_saved(self, path: str) -> None:
        if self._pending_feedback_replay is not None:
            self._repo.attach_to_feedback(self._pending_feedback_replay, path)
            self._pending_feedback_replay = None
            self.statusBar().showMessage("Replay attached to feedback", 5000)
            return
        self._live.on_replay_saved(path)
        self._sidebar.refresh()   # a stopped recording appears in the list at once
        self.statusBar().showMessage(f"Replay saved: {path}", 6000)

    # ------------------------------------------------------------------ #
    def _toggle_mini_view(self) -> None:
        """Pop out (or refocus) the always-on-top mini view. The mini is fed by
        the SAME controller signals as the live page — painting only, so having
        it open costs one extra pixmap scale per frame."""
        if getattr(self, "_mini", None) is None:
            from .widgets.mini_view import MiniView
            self._mini = MiniView()
            self._controller.frame_ready.connect(self._mini.on_frame)
            self._controller.recording_changed.connect(self._mini.on_recording)
            self._controller.status_changed.connect(self._mini.on_status)
            self._mini.record_toggled.connect(
                self._controller.set_recording, Qt.QueuedConnection)
            self._mini.closed.connect(self._on_mini_closed)
            self._mini.restore_requested.connect(self._on_mini_restore)
            saved = getattr(self._settings.ui, "mini_geometry", "")
            if saved:
                self._mini.apply_geometry_string(saved)
        self._mini.show()
        self._mini.raise_()

    def _on_mini_closed(self) -> None:
        if getattr(self, "_mini", None) is not None:
            self._settings.ui.mini_geometry = self._mini.geometry_string()
            self._settings.save()

    def _on_mini_restore(self) -> None:
        self._on_mini_closed()
        self._mini.hide()
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def _build_menu(self) -> None:
        """Menu bar (Joe's ask): every control reachable even when narrow
        mode hides its usual home. Wired ONLY to existing handlers."""
        import os

        bar = self.menuBar()
        m_file = bar.addMenu("&File")
        act = m_file.addAction("Open Recordings &Folder")
        act.triggered.connect(
            lambda: os.startfile(str(self._settings.recording.resolved_dir())))
        m_file.addAction("&Settings…").triggered.connect(self._toggle_settings)
        m_file.addSeparator()
        m_file.addAction("E&xit").triggered.connect(self.close)

        m_view = bar.addMenu("&View")
        self._act_sidebar = m_view.addAction("Sessions &Sidebar")
        self._act_sidebar.setCheckable(True)
        self._act_sidebar.setChecked(True)
        self._act_sidebar.toggled.connect(self._apply_sidebar_visibility)
        m_view.addAction("&Mini View").triggered.connect(self._toggle_mini_view)

        # One-tap shot clock (Joe: on/off without the Settings trip, and it
        # runs whether or not a recording is on — sandbox testing included)
        m_play = bar.addMenu("&Play")
        self._act_clock = m_play.addAction("Shot &Clock")
        self._act_clock.setCheckable(True)
        self._act_clock.setChecked(self._settings.shot_clock.enabled)
        self._act_clock.toggled.connect(self._toggle_shot_clock)

        m_help = bar.addMenu("&Help")
        m_help.addAction("Check for &Updates").triggered.connect(
            self._check_for_updates_forced)
        m_help.addAction("Send &Feedback…").triggered.connect(self._open_feedback)

    def _sync_clock_menu(self, on: bool) -> None:
        if hasattr(self, "_act_clock"):
            self._act_clock.blockSignals(True)
            self._act_clock.setChecked(bool(on))
            self._act_clock.blockSignals(False)

    def _toggle_shot_clock(self, on: bool) -> None:
        self._settings.shot_clock.enabled = bool(on)
        self._settings.save()
        self._live.set_clock_enabled_ui(bool(on))   # rail button mirrors
        self.statusBar().showMessage(
            f"Shot clock {'ON' if on else 'off'} "
            f"({self._settings.shot_clock.seconds}s, works without recording)",
            4000)

    def _apply_sidebar_visibility(self) -> None:
        # User preference (View menu) AND room for it: below 1000px the
        # sidebar yields regardless, so side-by-side keeps working.
        wanted = (not hasattr(self, "_act_sidebar")
                  or self._act_sidebar.isChecked())
        self._sidebar.setVisible(wanted and self.width() >= 1000)

    def resizeEvent(self, ev):  # noqa: N802 - Qt override
        # Narrow-window mode: the sessions sidebar yields below ~1000px so the
        # app can sit side-by-side with another window; it returns with room.
        if hasattr(self, "_sidebar"):
            self._apply_sidebar_visibility()
        super().resizeEvent(ev)

    def _maybe_check_updates(self) -> None:
        from ..update.updater import can_self_update
        if not can_self_update():
            log.info("Running from source — no update check (git manages this tree)")
            return
        if self._settings.updates.auto_check:
            log.info("Auto update-check on launch (current v%s)", __version__)
            self._run_update_check(forced=False)
        else:
            log.info("Auto update-check disabled in settings")

    def _check_for_updates_forced(self) -> None:
        log.info("Manual update-check requested")
        self._run_update_check(forced=True)

    def _run_update_check(self, forced: bool) -> None:
        from ..update.updater import UpdateCheckWorker, run_in_thread
        self._update_forced = forced
        self._update_worker = UpdateCheckWorker()
        self._update_worker.finished.connect(self._on_update_info)
        self._update_thread = run_in_thread(self._update_worker)

    def _on_update_info(self, info) -> None:
        if hasattr(self, "_update_thread"):
            self._update_thread.quit()
            self._update_thread.wait(2000)
        if info is None:
            if self._update_forced:
                self._settings_page.set_update_status(
                    f"You're on the latest version (v{__version__}).")
            return
        from ..update.updater import can_self_update
        if not can_self_update():
            # Nothing to offer: a source tree updates with git, and the download
            # flow ends in an exe this checkout would never run.
            self._settings_page.set_update_status(
                f"Version {info.version} is published. This is a source checkout "
                "(v{}) — update it with git pull.".format(__version__))
            return
        if self._update_forced:
            self._settings_page.set_update_status(f"Version {info.version} is available.")
        from .dialogs.update_dialog import maybe_prompt_update
        maybe_prompt_update(info, self)

    # ------------------------------------------------------------------ #
    def _open_ball_trainer(self) -> None:
        from .dialogs.ball_trainer_dialog import BallTrainerDialog
        dlg = BallTrainerDialog(self._settings, self)
        dlg.strategy_retrained.connect(self._on_strategy_changed)
        dlg.exec()

    def _find_train_python(self) -> str:
        """A python with torch+ultralytics (the shipped app is torch-free)."""
        from pathlib import Path
        root = Path(__file__).resolve().parents[2].parent
        for c in (root / "_refs" / "pool_coach" / ".venv" / "Scripts" / "python.exe",
                  root / ".trainvenv" / "Scripts" / "python.exe",
                  root / "_refs" / "pool_coach" / ".venv" / "bin" / "python",
                  root / ".trainvenv" / "bin" / "python"):
            if c.exists():
                return str(c)
        return ""

    def _train_ball_ids(self) -> None:
        """'Train with AI': auto-label the latest session first (when the AI
        backend is configured), then fine-tune on everything saved — one flow,
        one progress readout — and switch to the new model when done."""
        from ..train import vlm
        from ..train.autolabel import find_latest_session
        session = find_latest_session(self._settings.recording.resolved_dir())
        if (vlm.available(self._settings.autolabel) and session
                and getattr(self, "_autolabel_worker", None) is None):
            from ..config import APP_DIR
            self._live.set_autolabel_status("AI labelling the session…", busy=True)
            self._autolabel_worker = _AutoLabelWorker(
                Path(session), self._settings, APP_DIR / "training" / "ballid")
            self._autolabel_worker.progress.connect(
                lambda m: self._live.set_autolabel_status(m, busy=True))
            self._autolabel_worker.done.connect(self._after_autolabel_train)
            self._autolabel_worker.start()
            return
        self._start_finetune()

    def _after_autolabel_train(self, ok: bool, report: str) -> None:
        self._autolabel_worker = None
        self._live.set_autolabel_status(report, busy=True)
        self._start_finetune()

    def _start_finetune(self) -> None:
        """Fine-tune on everything saved, streaming progress to the rail."""
        from ..config import APP_DIR, MODELS_DIR
        from ..train import TrainingStore
        store = TrainingStore(APP_DIR / "training" / "ballid")
        if store.count() < 5:
            self._live.on_train_done(False, "Save at least ~5 labelled frames first.")
            return
        py = self._find_train_python()
        data = str(store.write_data_yaml())
        out = str(MODELS_DIR / "pool_ballid.onnx")
        if not py:
            self._live.on_train_done(False, "No training environment found "
                                            "(.trainvenv missing).")
            return
        if getattr(self, "_ball_train", None) is not None and self._ball_train.isRunning():
            return
        self._live._train_ai_btn.setEnabled(False)
        self._live.on_train_progress(-1, "Starting training…")
        self._ball_train = _StreamingTrainWorker(py, data, out)
        self._ball_train.progress.connect(self._live.on_train_progress)
        self._ball_train.done.connect(self._on_balls_trained)
        self._ball_train.start()

    def _on_balls_trained(self, ok: bool, log: str) -> None:
        if ok:
            # Deploy immediately: the new model becomes the live detector now.
            self._settings.balls.live_strategy = "onnx_pool_ballid"
            self._settings.save()
            self.set_strategy_requested.emit("onnx_pool_ballid")
            self._live.on_train_done(True, "Trained ✓ — saved and active. The app "
                                           "is now using your table's model.")
        else:
            self._live.on_train_done(False, f"Training failed: {log[-200:]}")

    def _on_cue_address(self, addr: str) -> None:
        """Remember the sensor so later scans prefer it (no reconfiguration UX)."""
        if addr and addr != self._settings.cue.address:
            self._settings.cue.address = addr
            self._settings.save()

    def _open_cue_diagnostics(self) -> None:
        from .dialogs.cue_diagnostics_dialog import CueDiagnosticsDialog
        dlg = CueDiagnosticsDialog(self._cue, self)
        dlg.exec()

    def _open_cue_pair(self) -> None:
        from .dialogs.cue_pair_dialog import CuePairDialog
        dlg = CuePairDialog(self._cue, self._settings, self)
        dlg.exec()
        # The dialog may have saved a new address / enabled the sensor.
        self._settings_page.reload()
        self._live.set_cue_enabled(self._settings.cue.enabled)

    def _open_feedback(self) -> None:
        from .dialogs.feedback_dialog import FeedbackDialog
        dlg = FeedbackDialog(self._repo, self)
        dlg.replay_requested.connect(self._on_feedback_replay)
        dlg.submitted.connect(self._on_feedback_submitted)
        dlg.exec()

    def _on_feedback_replay(self, feedback_id: int) -> None:
        # ask the controller (worker thread) to save the replay; we link it when
        # replay_saved fires.
        self._pending_feedback_replay = feedback_id
        from PySide6.QtCore import QMetaObject
        QMetaObject.invokeMethod(self._controller, "save_replay", Qt.QueuedConnection)

    def _on_feedback_submitted(self, feedback_id: int) -> None:
        self.statusBar().showMessage("Feedback saved — thank you!", 5000)
        from ..sync.supabase import trigger_sync
        trigger_sync(self._sync)

    # ------------------------------------------------------------------ #
    def closeEvent(self, event) -> None:
        try:
            self._cue.shutdown()
            from PySide6.QtCore import QMetaObject
            QMetaObject.invokeMethod(self._controller, "stop", Qt.QueuedConnection)
            # final backup attempt (no-op if Supabase isn't configured)
            QMetaObject.invokeMethod(self._sync, "sync_now", Qt.QueuedConnection)
            self._thread.quit()
            self._thread.wait(2500)
            self._sync_thread.quit()
            self._sync_thread.wait(2500)
            dl = getattr(self, "_model_dl_thread", None)
            if dl is not None and dl.isRunning():
                dl.quit()
                dl.wait(2000)
        except Exception:  # noqa: BLE001
            pass
        super().closeEvent(event)
