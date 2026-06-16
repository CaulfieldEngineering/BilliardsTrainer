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

        self._build_ui()
        self._wire()
        self._refresh_detection_availability()
        self._autostart_preview()
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
        for idx, (ic, label) in enumerate([
            ("activity", "Sandbox"), ("target", "Drills"),
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

    def _go(self, index: int) -> None:
        self._stack.setCurrentIndex(index)
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
        self._live.retry_requested.connect(self._retry_camera)
        self._live.open_settings_requested.connect(lambda: self._go(3))
        self._live.recalibrate_requested.connect(self._controller.recalibrate, q)
        self._live.pick_felt_requested.connect(self._controller.pick_felt, q)
        self._live.save_replay_requested.connect(self._controller.save_replay, q)
        self._live.overlays_toggled.connect(self._on_overlays_toggled)
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
        self._controller.capture_saved.connect(self._on_capture_saved)

        # settings + drills
        self._settings_page.applied.connect(self._on_settings_applied)
        self._settings_page.check_updates_requested.connect(self._check_for_updates_forced)
        self._settings_page.feedback_requested.connect(self._open_feedback)
        self._settings_page.capture_requested.connect(
            self._controller.start_analysis_capture, q)
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

    def _refresh_detection_availability(self) -> None:
        """Auto-detection is only offered when a YOLO model is present — otherwise
        we hold the line on reliability with manual mode + a banner."""
        from ..vision.balls import yolo_weights_available
        available = (yolo_weights_available()
                     or self._settings.balls.backend == "yolo"
                     or self._settings.detection.allow_without_model)
        self._live.set_detection_available(available)

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
        # Asset-free audio cue.
        QApplication.beep()

    def _push_settings(self) -> None:
        self.apply_settings_requested.emit(self._settings)

    def _on_settings_applied(self) -> None:
        apply_theme(QApplication.instance(), self._settings.ui.accent)
        self._push_settings()
        self._refresh_detection_availability()
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
            from PySide6.QtCore import QMetaObject
            QMetaObject.invokeMethod(self._controller, "stop", Qt.QueuedConnection)
            # final backup attempt (no-op if Supabase isn't configured)
            QMetaObject.invokeMethod(self._sync, "sync_now", Qt.QueuedConnection)
            self._thread.quit()
            self._thread.wait(2500)
            self._sync_thread.quit()
            self._sync_thread.wait(2500)
        except Exception:  # noqa: BLE001
            pass
        super().closeEvent(event)
