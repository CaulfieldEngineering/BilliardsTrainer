"""Main window: navigation rail + stacked pages, wired to the pipeline thread.

Owns the Repository and the PipelineController (on its own QThread). All
controller method calls are made through queued signal/slot connections so CV
work never runs on the UI thread.
"""

import logging

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..config import Settings
from ..db.repository import Repository
from ..version import APP_NAME, __version__
from ..workers.controller import PipelineController, make_controller_thread
from .pages.drills_page import DrillsPage
from .pages.live_page import LivePage
from .pages.settings_page import SettingsPage
from .pages.stats_page import StatsPage
from .theme import PALETTE, apply_theme
from .widgets.common import nav_button

log = logging.getLogger("ui")


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
        root.addWidget(self._nav_rail())

        self._stack = QStackedWidget()
        self._live = LivePage(self._settings)
        self._drills = DrillsPage()
        self._stats = StatsPage(self._repo)
        self._settings_page = SettingsPage(self._settings)
        for page in (self._live, self._drills, self._stats, self._settings_page):
            self._stack.addWidget(page)
        root.addWidget(self._stack, 1)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")

    def _nav_rail(self) -> QWidget:
        rail = QFrame()
        rail.setObjectName("Sidebar")
        rail.setFixedWidth(208)
        lay = QVBoxLayout(rail)
        lay.setContentsMargins(14, 18, 14, 16)
        lay.setSpacing(6)

        brand = QHBoxLayout()
        dot = QLabel("●")
        dot.setStyleSheet(f"color: {PALETTE.accent}; font-size: 16px;")
        brand.addWidget(dot)
        name = QLabel("Billiards Trainer")
        name.setStyleSheet("font-size: 15px; font-weight: 700;")
        brand.addWidget(name)
        brand.addStretch(1)
        lay.addLayout(brand)
        lay.addSpacing(14)

        self._nav_group = QButtonGroup(self)
        self._nav_group.setExclusive(True)
        self._nav_items = []
        # nav index -> stacked-page index ('Training' reuses the Sandbox page +
        # flips its Training Mode, so labelling happens on the same video/transport)
        self._nav_to_stack = {0: 0, 1: 0, 2: 1, 3: 2, 4: 3}
        for idx, (ic, label) in enumerate([
            ("activity", "Sandbox"), ("crosshair", "Training"), ("target", "Drills"),
            ("stats", "Stats"), ("settings", "Settings"),
        ]):
            btn = nav_button(ic, label)
            btn.clicked.connect(lambda _=False, i=idx: self._go(i))
            self._nav_group.addButton(btn)
            self._nav_items.append(btn)
            lay.addWidget(btn)
        self._nav_items[0].setChecked(True)

        lay.addStretch(1)
        ver = QLabel(f"v{__version__}")
        ver.setObjectName("Faint")
        lay.addWidget(ver)
        return rail

    def _go(self, nav_index: int) -> None:
        stack_i = self._nav_to_stack.get(nav_index, 0)
        self._stack.setCurrentIndex(stack_i)
        self._live.set_training(nav_index == 1)   # 'Training' nav = label mode on
        if self._stack.currentWidget() is self._stats:
            self._stats.refresh()

    # ------------------------------------------------------------------ #
    def _wire(self) -> None:
        q = Qt.QueuedConnection
        # UI intent -> controller (queued, runs on the worker thread)
        self.start_source.connect(self._controller.start, q)
        self._live.start_requested.connect(self._controller.start, q)
        self._live.stop_requested.connect(self._controller.stop, q)
        self._live.detection_toggled.connect(self._controller.set_detection_enabled, q)
        self.set_strategy_requested.connect(self._controller.set_detector_strategy, q)
        self._settings_page.detector_changed.connect(self._on_strategy_changed)
        self._live.retry_requested.connect(self._retry_camera)
        # video transport (UI -> controller, queued onto the worker thread)
        self._live.video_play_pause.connect(self._controller.set_video_paused, q)
        self._live.video_stop_requested.connect(self._controller.video_stop, q)
        self._live.video_step.connect(self._controller.video_step, q)
        self._live.video_seek.connect(self._controller.video_seek, q)
        self._live.video_speed.connect(self._controller.set_playback_speed, q)
        self._controller.source_is_video.connect(self._live.set_video_mode)
        self._controller.video_state.connect(self._live.update_video_state)
        self._live.open_settings_requested.connect(lambda: self._go(3))
        self._live.recalibrate_requested.connect(self._controller.recalibrate, q)
        self._live.pick_felt_requested.connect(self._controller.pick_felt, q)
        self._live.save_replay_requested.connect(self._controller.save_replay, q)
        self._live.overlays_toggled.connect(self._on_overlays_toggled)
        self._live.tuning_changed.connect(self._on_tuning_changed)
        # Training Mode (label/correct ball numbers on the playback)
        self._live.label_mode_toggled.connect(self._controller.set_label_mode, q)
        self._live.save_training_frame_requested.connect(self._controller.save_training_frame, q)
        self._live.train_balls_requested.connect(self._train_ball_ids)
        self._live.pause_toggled.connect(self._controller.set_paused, q)
        self._live.reset_requested.connect(self._controller.reset_counters, q)
        self._live.manual_shot.connect(self._controller.record_manual_shot, q)
        self._live.record_toggled.connect(self._controller.set_recording, q)
        self.apply_settings_requested.connect(self._controller.apply_settings, q)

        # controller -> UI
        self._controller.frame_ready.connect(self._live.on_frame)
        self._controller.stats_updated.connect(self._live.on_stats)
        self._controller.stats_updated.connect(lambda _s: self._stats.refresh())
        self._controller.shot_recorded.connect(self._live.on_shot)
        self._controller.shot_suggested.connect(self._live.on_suggestion)
        self._controller.recording_changed.connect(self._live.on_recording)
        self._controller.detection_changed.connect(self._live.set_detection)
        self._controller.status_changed.connect(self._live.on_status)
        self._controller.status_changed.connect(self._on_status)
        self._controller.clock_event.connect(self._on_clock_event)
        self._controller.error.connect(self._on_error)
        self._controller.settings_changed.connect(self._on_settings_changed)
        self._controller.replay_saved.connect(self._on_replay_saved)
        self._controller.capture_progress.connect(
            lambda m: self.statusBar().showMessage(f"Capture: {m}", 4000))
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

        # settings + drills
        self._settings_page.applied.connect(self._on_settings_applied)
        self._settings_page.check_updates_requested.connect(self._check_for_updates_forced)
        self._settings_page.feedback_requested.connect(self._open_feedback)
        self._settings_page.capture_requested.connect(
            self._controller.start_analysis_capture, q)
        self._settings_page.train_balls_requested.connect(self._open_ball_trainer)
        self._settings_page.flag_failure_requested.connect(self._controller.flag_failure, q)
        self._controller.failure_flagged.connect(self._on_failure_flagged)
        self._drills.drill_chosen.connect(self._on_drill_chosen)
        self._sync.status.connect(lambda msg: self.statusBar().showMessage(f"Sync: {msg}", 4000))

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
        self._model_dl_worker.failed.connect(
            lambda m: self.statusBar().showMessage(
                f"Model download failed ({m}); using the basic cue detector.", 6000))
        for sig in (self._model_dl_worker.done, self._model_dl_worker.failed):
            sig.connect(self._model_dl_thread.quit)
        self._model_dl_thread.start()

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
        self._live.set_detector_label(name)
        self.statusBar().showMessage(f"Live detector → {name}", 4000)

    def _on_capture_saved(self, path: str) -> None:
        self.statusBar().showMessage(f"Analysis capture saved: {path}", 8000)
        self._settings_page.set_capture_status(f"Saved: {path}")

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

    def _on_clock_event(self, edge: str) -> None:
        if not self._settings.shot_clock.audio:
            return
        # warn = single beep at 10 s, tick = 3-2-1 cadence, expired = the buzz
        from .sounds import play
        play(edge)

    def _push_settings(self) -> None:
        self.apply_settings_requested.emit(self._settings)

    def _on_settings_applied(self) -> None:
        apply_theme(QApplication.instance(), self._settings.ui.accent)
        self._push_settings()
        self._live.set_detector_label(self._settings.balls.live_strategy)
        self._cue.apply_settings(self._settings.cue)
        self._live.set_cue_enabled(self._settings.cue.enabled)
        # If the camera source changed, re-open the preview on the new device.
        if (self._settings.source or "0") != self._started_source:
            self._started_source = self._settings.source or "0"
            self.start_source.emit(self._started_source, self._settings.mode, "")
        self.statusBar().showMessage("Settings saved", 4000)

    def _on_overlays_toggled(self, on: bool) -> None:
        # live page already flipped self._settings.ui.show_overlays (shared object)
        self._push_settings()
        self._settings.save()
        self.statusBar().showMessage(
            f"Detection overlays {'on' if on else 'off'}", 3000)

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
        self.statusBar().showMessage(f"Replay saved: {path}", 6000)

    def _on_drill_chosen(self, drill) -> None:
        self._live.set_drill(drill.key, drill.name)
        self._nav_items[0].setChecked(True)
        self._go(0)
        self.statusBar().showMessage(f"Drill ready: {drill.name} — press Start", 6000)

    # ------------------------------------------------------------------ #
    def _maybe_check_updates(self) -> None:
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
        """Fine-tune on the labelled store (Training Mode) in a torch env, then
        switch the app to the table-trained model."""
        from pathlib import Path

        from ..config import APP_DIR, MODELS_DIR
        from ..train import TrainingStore
        store = TrainingStore(APP_DIR / "training" / "ballid")
        if store.count() < 5:
            self.statusBar().showMessage("Label at least ~5 frames in Training Mode "
                                         "before training.", 6000)
            return
        py = self._find_train_python()
        data = str(store.write_data_yaml())
        out = str(MODELS_DIR / "pool_ballid.onnx")
        if not py:
            self.statusBar().showMessage("No torch env found for training. Run once: "
                                         f"python tools/finetune_ballid.py --data {data} "
                                         f"--out {out}", 15000)
            return
        from .dialogs.ball_trainer_dialog import _TrainWorker
        self.statusBar().showMessage("Training on your labelled balls… (runs in the "
                                     "background, a few minutes)", 0)
        self._ball_train = _TrainWorker(py, data, out)
        self._ball_train.done.connect(self._on_balls_trained)
        self._ball_train.start()

    def _on_balls_trained(self, ok: bool, log: str) -> None:
        if ok:
            self._settings.balls.live_strategy = "onnx_pool_ballid"
            self._settings.save()
            self.set_strategy_requested.emit("onnx_pool_ballid")
            self.statusBar().showMessage("Trained ✓ — now using your table's ball-ID "
                                         "model.", 8000)
        else:
            self.statusBar().showMessage(f"Training failed: {log[-200:]}", 12000)

    def _on_cue_address(self, addr: str) -> None:
        """Remember the sensor so later scans prefer it (no reconfiguration UX)."""
        if addr and addr != self._settings.cue.address:
            self._settings.cue.address = addr
            self._settings.save()

    def _open_cue_diagnostics(self) -> None:
        from .dialogs.cue_diagnostics_dialog import CueDiagnosticsDialog
        dlg = CueDiagnosticsDialog(self._cue, self)
        dlg.exec()

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
