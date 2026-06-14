"""Sandbox (free-play) page — the product's foundational surface.

Joe's priority: detect the table reliably, let him shoot freely, and count
makes/misses reliably. So this page leads with a big MAKES / MISSES readout and
keeps everything else (modes, shot clock, drills) secondary and out of the way.

Layout: a control bar (start/stop, demo, mode, tuning tools, status) over a
3-pane body — bird's-eye rectified view, live camera view, and a stats rail
dominated by the make/miss count. It is a *view*: it emits intent signals and
renders controller signals; the main window wires the two across the thread
boundary.
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
    pick_felt_requested = Signal(float, float)  # normalised click coords
    overlays_toggled = Signal(bool)
    save_replay_requested = Signal()

    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self._settings = settings
        self._running = False
        self._drill_key = ""
        self._drill_name = ""
        self._pick_mode = False
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

        bird_card = Card(padding=10, spacing=6)
        bird_card.add(self._caption("BIRD'S-EYE"))
        self._bird = VideoView("Bird's-eye view")
        bird_card.add(self._bird)
        splitter.addWidget(bird_card)

        persp_card = Card(padding=10, spacing=6)
        persp_card.add(self._caption("LIVE CAMERA"))
        self._persp = VideoView("Connect a camera or press Try demo")
        self._persp.clicked.connect(self._on_persp_click)
        persp_card.add(self._persp)
        splitter.addWidget(persp_card)

        splitter.addWidget(self._stats_rail())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 4)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([400, 540, 320])
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
        lay.setSpacing(10)

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
            [("free_play", "Sandbox"), ("practice", "Practice"), ("drill", "Drill")],
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

        # tuning tools
        self._pick_btn = self._tool_btn("crosshair", "Pick felt colour (click the table)")
        self._pick_btn.clicked.connect(self._toggle_pick)
        lay.addWidget(self._pick_btn)

        self._overlay_btn = self._tool_btn("layers", "Toggle detection overlays")
        self._overlay_btn.clicked.connect(self._toggle_overlays)
        lay.addWidget(self._overlay_btn)

        self._recal_btn = self._tool_btn("refresh", "Recalibrate the table")
        self._recal_btn.clicked.connect(self.recalibrate_requested.emit)
        lay.addWidget(self._recal_btn)
        return bar

    def _tool_btn(self, ic: str, tip: str) -> QPushButton:
        btn = QPushButton()
        btn.setObjectName("Ghost")
        btn.setIcon(icon(ic, PALETTE.text_dim))
        btn.setToolTip(tip)
        btn.setCursor(Qt.PointingHandCursor)
        return btn

    def _stats_rail(self) -> QWidget:
        rail = Card(padding=16, spacing=14)
        rail.setMinimumWidth(280)
        rail.setMaximumWidth(380)

        # The headline: makes vs misses, big.
        mm = QHBoxLayout()
        mm.setSpacing(12)
        self._makes_box, self._makes_val = self._big_stat("MAKES", PALETTE.success)
        self._misses_box, self._misses_val = self._big_stat("MISSES", PALETTE.danger)
        mm.addWidget(self._makes_box)
        mm.addWidget(self._misses_box)
        rail.layout().addLayout(mm)

        self._outcome = Badge("READY", PALETTE.text_dim)
        oc_row = QHBoxLayout()
        oc_row.addStretch(1)
        oc_row.addWidget(self._outcome)
        oc_row.addStretch(1)
        rail.layout().addLayout(oc_row)

        grid = QHBoxLayout()
        grid.setSpacing(10)
        self._k_pct = StatCard("Make %", "0%", accent=True)
        self._k_streak = StatCard("Streak", "0")
        grid.addWidget(self._k_pct)
        grid.addWidget(self._k_streak)
        rail.layout().addLayout(grid)

        # Shot clock — only visible when enabled, so it never crowds sandbox.
        self._clock_row = QHBoxLayout()
        self._clock_row.addStretch(1)
        self._clock = ShotClockWidget()
        self._clock_row.addWidget(self._clock)
        self._clock_row.addStretch(1)
        self._clock_holder = QWidget()
        self._clock_holder.setLayout(self._clock_row)
        self._clock_holder.setVisible(self._settings.shot_clock.enabled)
        rail.layout().addWidget(self._clock_holder)

        self._drill_lbl = QLabel("")
        self._drill_lbl.setObjectName("Faint")
        self._drill_lbl.setWordWrap(True)
        rail.layout().addWidget(self._drill_lbl)

        rail.layout().addStretch(1)

        self._replay_btn = QPushButton("  Save last 5s")
        self._replay_btn.setCursor(Qt.PointingHandCursor)
        self._replay_btn.setIcon(icon("activity", PALETTE.text_dim))
        self._replay_btn.setToolTip("Save a clip of what the detector just saw")
        self._replay_btn.clicked.connect(self.save_replay_requested.emit)
        rail.layout().addWidget(self._replay_btn)

        hint = QLabel("Set your camera in Settings, then press Start. The table is "
                      "detected and locked automatically. If detection looks off, "
                      "use Pick felt to tap the cloth.")
        hint.setObjectName("Faint")
        hint.setWordWrap(True)
        rail.layout().addWidget(hint)
        return rail

    def _big_stat(self, label: str, color: str):
        box = Card(padding=14, spacing=2)
        lbl = QLabel(label)
        lbl.setObjectName("StatLabel")
        val = QLabel("0")
        val.setStyleSheet(f"font-size: 40px; font-weight: 800; color: {color};")
        box.add(lbl)
        box.add(val)
        return box, val

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

    def _toggle_pick(self) -> None:
        self._pick_mode = not self._pick_mode
        self._persp.set_pickable(self._pick_mode)
        self._pick_btn.setIcon(icon("crosshair",
                                    PALETTE.accent if self._pick_mode else PALETTE.text_dim))
        if self._pick_mode:
            self._status_badge.set_text_color("TAP THE CLOTH", PALETTE.info)

    def _on_persp_click(self, xf: float, yf: float) -> None:
        if self._pick_mode:
            self.pick_felt_requested.emit(xf, yf)
            self._toggle_pick()  # one-shot

    def _toggle_overlays(self) -> None:
        new = not self._settings.ui.show_overlays
        self._settings.ui.show_overlays = new
        self._overlay_btn.setIcon(icon("layers", PALETTE.accent if new else PALETTE.text_dim))
        self.overlays_toggled.emit(new)

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
        self._clock_holder.setVisible(packet.clock_enabled)
        if packet.clock_enabled:
            self._clock.update_clock(packet.clock_remaining,
                                     max(1.0, self._settings.shot_clock.seconds),
                                     packet.clock_warning, True)
        if self._pick_mode:
            return  # keep the "tap the cloth" prompt
        if packet.deviated:
            self._status_badge.set_text_color("RELOCKING TABLE", PALETTE.warn)
        elif packet.status == "calibrating":
            self._status_badge.set_text_color("FINDING TABLE", PALETTE.info)
        elif packet.shot_state == "moving":
            self._status_badge.set_text_color("SHOT IN PLAY", PALETTE.accent)
        else:
            self._status_badge.set_text_color("LIVE", PALETTE.success)

    def on_stats(self, summary: dict) -> None:
        self._makes_val.setText(str(summary.get("makes", 0)))
        self._misses_val.setText(str(summary.get("misses", 0)))
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

    def on_replay_saved(self, path: str) -> None:
        self._status_badge.set_text_color("REPLAY SAVED", PALETTE.success)
