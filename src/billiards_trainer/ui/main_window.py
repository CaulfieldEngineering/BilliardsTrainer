"""Main window: navigation rail + stacked pages, wired to the pipeline thread.

Owns the Repository and the PipelineController (on its own QThread). All
controller method calls are made through queued signal/slot connections so CV
work never runs on the UI thread.
"""

import logging

from PySide6.QtCore import Qt
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
    def __init__(self, settings: Settings):
        super().__init__()
        self._settings = settings
        self.setWindowTitle(APP_NAME)

        self._repo = Repository()
        self._controller = PipelineController(settings, self._repo)
        self._thread = make_controller_thread(self._controller)

        self._build_ui()
        self._wire()
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
            ("activity", "Live"), ("target", "Drills"),
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
        self._live.start_requested.connect(self._controller.start, q)
        self._live.stop_requested.connect(self._controller.stop, q)
        self._live.recalibrate_requested.connect(self._controller.recalibrate, q)

        # controller -> UI
        self._controller.frame_ready.connect(self._live.on_frame)
        self._controller.stats_updated.connect(self._live.on_stats)
        self._controller.stats_updated.connect(lambda _s: self._stats.refresh())
        self._controller.shot_recorded.connect(self._live.on_shot)
        self._controller.status_changed.connect(self._live.on_status)
        self._controller.status_changed.connect(self._on_status)
        self._controller.clock_event.connect(self._on_clock_event)
        self._controller.error.connect(self._on_error)

        # settings + drills
        self._settings_page.applied.connect(self._on_settings_applied)
        self._drills.drill_chosen.connect(self._on_drill_chosen)

    def _on_status(self, status: str) -> None:
        self.statusBar().showMessage({"running": "Live — analysing",
                                      "stopped": "Stopped"}.get(status, status))

    def _on_error(self, msg: str) -> None:
        self.statusBar().showMessage(msg, 8000)
        log.warning("controller error: %s", msg)

    def _on_clock_event(self, edge: str) -> None:
        if not self._settings.shot_clock.audio:
            return
        # Asset-free audio cue.
        QApplication.beep()

    def _on_settings_applied(self) -> None:
        apply_theme(QApplication.instance(), self._settings.ui.accent)
        # push the edited settings onto the worker thread
        from PySide6.QtCore import Q_ARG, QMetaObject
        QMetaObject.invokeMethod(self._controller, "apply_settings",
                                 Qt.QueuedConnection, Q_ARG(object, self._settings))
        self.statusBar().showMessage("Settings saved", 4000)

    def _on_drill_chosen(self, drill) -> None:
        self._live.set_drill(drill.key, drill.name)
        self._nav_items[0].setChecked(True)
        self._go(0)
        self.statusBar().showMessage(f"Drill ready: {drill.name} — press Start", 6000)

    # ------------------------------------------------------------------ #
    def _maybe_check_updates(self) -> None:
        if not self._settings.updates.auto_check:
            return
        from ..update.updater import UpdateCheckWorker, run_in_thread
        self._update_worker = UpdateCheckWorker()
        self._update_worker.finished.connect(self._on_update_info)
        self._update_thread = run_in_thread(self._update_worker)

    def _on_update_info(self, info) -> None:
        if hasattr(self, "_update_thread"):
            self._update_thread.quit()
            self._update_thread.wait(2000)
        if info is None:
            return
        from .dialogs.update_dialog import maybe_prompt_update
        maybe_prompt_update(info, self)

    # ------------------------------------------------------------------ #
    def closeEvent(self, event) -> None:
        try:
            from PySide6.QtCore import QMetaObject
            QMetaObject.invokeMethod(self._controller, "stop", Qt.QueuedConnection)
            self._thread.quit()
            self._thread.wait(2500)
        except Exception:  # noqa: BLE001
            pass
        super().closeEvent(event)
