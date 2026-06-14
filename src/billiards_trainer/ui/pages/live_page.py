"""Live analysis page — the product's centrepiece.

Layout: a control bar (start/stop, mode, source, live status) over a 3-pane body
— bird's-eye rectified view, live perspective view, and a stats rail with the
shot clock, KPI tiles, and the last-shot outcome. It is a *view*: it emits intent
signals (start/stop/recalibrate) and renders controller signals; the main window
wires the two together across the thread boundary.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ...config import Settings
from ..icons import icon
from ..theme import PALETTE
from ..widgets.common import Badge, Card, SegmentedControl, StatCard
from ..widgets.shot_clock_widget import ShotClockWidget
from ..widgets.video_view import VideoView


class LivePage(QWidget):
    start_requested = Signal(str, str, str)   # source, mode, drill_key
    stop_requested = Signal()
    recalibrate_requested = Signal()

    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self._settings = settings
        self._running = False
        self._drill_key = ""
        self._drill_name = ""
        self._build()

    # ------------------------------------------------------------------ #
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(14)
        root.addWidget(self._control_bar())

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(8)
        splitter.setChildrenCollapsible(False)

        # bird's-eye (portrait)
        bird_card = Card(padding=10, spacing=6)
        bird_card.add(self._caption("BIRD'S-EYE"))
        self._bird = VideoView("Bird's-eye view")
        bird_card.add(self._bird)
        splitter.addWidget(bird_card)

        # perspective (landscape)
        persp_card = Card(padding=10, spacing=6)
        persp_card.add(self._caption("LIVE CAMERA"))
        self._persp = VideoView("Connect a camera or start the demo")
        persp_card.add(self._persp)
        splitter.addWidget(persp_card)

        # stats rail
        splitter.addWidget(self._stats_rail())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 4)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([420, 560, 300])
        root.addWidget(splitter, 1)

    def _caption(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("StatLabel")
        return lbl

    def _control_bar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("Card")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(12)

        self._start_btn = QPushButton("  Start")
        self._start_btn.setObjectName("Accent")
        self._start_btn.setCursor(Qt.PointingHandCursor)
        self._start_btn.setIcon(icon("play", "#0A0E12"))
        self._start_btn.clicked.connect(self._toggle)
        lay.addWidget(self._start_btn)

        self._demo_btn = QPushButton("  Try demo")
        self._demo_btn.setObjectName("Ghost")
        self._demo_btn.setCursor(Qt.PointingHandCursor)
        self._demo_btn.setIcon(icon("zap", PALETTE.text_dim))
        self._demo_btn.clicked.connect(self._start_demo)
        lay.addWidget(self._demo_btn)

        self._mode = SegmentedControl(
            [("free_play", "Free Play"), ("practice", "Practice"), ("drill", "Drill")],
            current=self._settings.mode)
        lay.addWidget(self._mode)

        lay.addStretch(1)

        self._status_badge = Badge("IDLE", PALETTE.text_faint)
        lay.addWidget(self._status_badge)
        self._balls_lbl = QLabel("0 balls")
        self._balls_lbl.setObjectName("Muted")
        lay.addWidget(self._balls_lbl)
        self._fps_lbl = QLabel("0 fps")
        self._fps_lbl.setObjectName("Faint")
        lay.addWidget(self._fps_lbl)

        self._recal_btn = QPushButton()
        self._recal_btn.setObjectName("Ghost")
        self._recal_btn.setIcon(icon("refresh", PALETTE.text_dim))
        self._recal_btn.setToolTip("Recalibrate the table")
        self._recal_btn.setCursor(Qt.PointingHandCursor)
        self._recal_btn.clicked.connect(self.recalibrate_requested.emit)
        lay.addWidget(self._recal_btn)
        return bar

    def _stats_rail(self) -> QWidget:
        rail = Card(padding=16, spacing=14)
        rail.setMinimumWidth(260)
        rail.setMaximumWidth(360)

        clock_row = QHBoxLayout()
        clock_row.addStretch(1)
        self._clock = ShotClockWidget()
        clock_row.addWidget(self._clock)
        clock_row.addStretch(1)
        rail.layout().addLayout(clock_row)

        self._outcome = Badge("READY", PALETTE.text_dim)
        oc_row = QHBoxLayout()
        oc_row.addStretch(1)
        oc_row.addWidget(self._outcome)
        oc_row.addStretch(1)
        rail.layout().addLayout(oc_row)

        grid = QHBoxLayout()
        grid.setSpacing(10)
        self._k_shots = StatCard("Shots", "0")
        self._k_makes = StatCard("Makes", "0", accent=True)
        grid.addWidget(self._k_shots)
        grid.addWidget(self._k_makes)
        rail.layout().addLayout(grid)

        grid2 = QHBoxLayout()
        grid2.setSpacing(10)
        self._k_pct = StatCard("Make %", "0%", accent=True)
        self._k_streak = StatCard("Streak", "0")
        grid2.addWidget(self._k_pct)
        grid2.addWidget(self._k_streak)
        rail.layout().addLayout(grid2)

        self._drill_lbl = QLabel("")
        self._drill_lbl.setObjectName("Faint")
        self._drill_lbl.setWordWrap(True)
        rail.layout().addWidget(self._drill_lbl)

        rail.layout().addStretch(1)
        hint = QLabel("Set your camera in Settings. The table is detected and "
                      "locked automatically on the first clear frame.")
        hint.setObjectName("Faint")
        hint.setWordWrap(True)
        rail.layout().addWidget(hint)
        return rail

    # ------------------------------------------------------------------ #
    # Intent
    # ------------------------------------------------------------------ #
    def set_drill(self, drill_key: str, drill_name: str) -> None:
        self._drill_key = drill_key
        self._drill_name = drill_name
        self._mode.set_current("drill")
        self._drill_lbl.setText(f"Drill: {drill_name}" if drill_name else "")

    def _toggle(self) -> None:
        if self._running:
            self.stop_requested.emit()
        else:
            mode = self._mode.current()
            self.start_requested.emit(self._settings.source, mode,
                                      self._drill_key if mode == "drill" else "")

    def _start_demo(self) -> None:
        self.start_requested.emit("demo", self._mode.current(), "")

    # ------------------------------------------------------------------ #
    # Controller signal handlers
    # ------------------------------------------------------------------ #
    def set_running(self, running: bool) -> None:
        self._running = running
        if running:
            self._start_btn.setText("  Stop")
            self._start_btn.setIcon(icon("stop", "#0A0E12"))
            self._demo_btn.setEnabled(False)
        else:
            self._start_btn.setText("  Start")
            self._start_btn.setIcon(icon("play", "#0A0E12"))
            self._demo_btn.setEnabled(True)
            self._status_badge.set_text_color("IDLE", PALETTE.text_faint)
            self._persp.clear()
            self._bird.clear()

    def on_frame(self, packet) -> None:
        if packet.perspective is not None:
            self._persp.set_frame(packet.perspective)
        if packet.birdseye is not None:
            self._bird.set_frame(packet.birdseye)
        self._fps_lbl.setText(f"{packet.fps:.0f} fps")
        self._balls_lbl.setText(f"{packet.n_balls} balls")
        self._clock.update_clock(packet.clock_remaining, max(1.0, self._settings.shot_clock.seconds),
                                 packet.clock_warning, packet.clock_enabled)
        if packet.deviated:
            self._status_badge.set_text_color("TABLE MOVED", PALETTE.warn)
        elif packet.status == "calibrating":
            self._status_badge.set_text_color("CALIBRATING", PALETTE.info)
        elif packet.shot_state == "moving":
            self._status_badge.set_text_color("SHOT IN PLAY", PALETTE.accent)
        else:
            self._status_badge.set_text_color("LIVE", PALETTE.success)

    def on_stats(self, summary: dict) -> None:
        self._k_shots.set_value(str(summary.get("shots", 0)))
        self._k_makes.set_value(str(summary.get("makes", 0)))
        self._k_pct.set_value(f"{summary.get('make_pct', 0):.0f}%")
        self._k_streak.set_value(str(summary.get("current_streak", 0)))

    def on_shot(self, event) -> None:
        outcome = event.outcome.value
        color = {"make": PALETTE.success, "miss": PALETTE.danger,
                 "scratch": PALETTE.warn}.get(outcome, PALETTE.text_dim)
        self._outcome.set_text_color(outcome.upper(), color)

    def on_status(self, status: str) -> None:
        if status == "running":
            self.set_running(True)
        elif status == "stopped":
            self.set_running(False)
