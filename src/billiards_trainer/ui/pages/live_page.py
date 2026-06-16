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
    QStackedWidget,
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
    pause_toggled = Signal(bool)
    reset_requested = Signal()
    manual_shot = Signal(str)                 # 'make' | 'miss'
    record_toggled = Signal(bool)
    detection_toggled = Signal(bool)          # auto-detection on/off
    retry_requested = Signal()                # re-open the camera after an error
    open_settings_requested = Signal()        # jump to Settings → Camera

    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self._settings = settings
        self._running = False
        self._detect_on = False
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
        # The live view and an inline camera-error panel share a stack, so a
        # camera failure surfaces right here (with Retry / Open Settings) instead
        # of sending the user hunting through menus.
        self._persp_stack = QStackedWidget()
        self._persp = VideoView("Connecting to camera…")
        self._persp.clicked.connect(self._on_persp_click)
        self._persp_stack.addWidget(self._persp)
        self._persp_stack.addWidget(self._camera_error_panel())
        persp_card.add(self._persp_stack)
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

        # Auto-detection is a toggle, not a Start gate: the camera previews the
        # moment the app opens. Detection defaults OFF so nothing fake is drawn.
        self._detect_btn = QPushButton("  Detection: OFF")
        self._detect_btn.setObjectName("Ghost")
        self._detect_btn.setCheckable(True)
        self._detect_btn.setCursor(Qt.PointingHandCursor)
        self._detect_btn.setIcon(icon("zap", PALETTE.text_dim))
        self._detect_btn.setToolTip("Turn AI ball/shot detection on or off")
        self._detect_btn.clicked.connect(self._toggle_detection)
        lay.addWidget(self._detect_btn)

        # Which live detector runs when Detection is ON (Phase-1 winner default).
        # Read-only label here; switch detectors in Settings → Ball detection.
        self._detector_lbl = QLabel(f"Detector: {self._live_strategy_name()}")
        self._detector_lbl.setObjectName("Muted")
        lay.addWidget(self._detector_lbl)

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
        self._pause_btn = self._tool_btn("pause", "Pause shot counting (video keeps running)")
        self._pause_btn.setCheckable(True)
        self._pause_btn.clicked.connect(self._toggle_pause)
        lay.addWidget(self._pause_btn)

        self._record_btn = self._tool_btn("activity", "Record this session to a video clip")
        self._record_btn.setCheckable(True)
        self._record_btn.clicked.connect(lambda: self.record_toggled.emit(self._record_btn.isChecked()))
        lay.addWidget(self._record_btn)

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

    def _toggle_pause(self) -> None:
        paused = self._pause_btn.isChecked()
        self._pause_btn.setIcon(icon("play" if paused else "pause",
                                     PALETTE.warn if paused else PALETTE.text_dim))
        self.pause_toggled.emit(paused)

    def _live_strategy_name(self) -> str:
        return getattr(self._settings.balls, "live_strategy", "simple_blob")

    def set_detector_label(self, name: str) -> None:
        self._detector_lbl.setText(f"Detector: {name}")

    def _camera_error_panel(self) -> QWidget:
        panel = QFrame()
        col = QVBoxLayout(panel)
        col.setContentsMargins(24, 24, 24, 24)
        col.setSpacing(12)
        col.addStretch(1)
        title = QLabel("Camera unavailable")
        title.setStyleSheet("font-size: 16px; font-weight: 700;")
        title.setAlignment(Qt.AlignCenter)
        col.addWidget(title)
        self._cam_err_lbl = QLabel("")
        self._cam_err_lbl.setObjectName("Muted")
        self._cam_err_lbl.setWordWrap(True)
        self._cam_err_lbl.setAlignment(Qt.AlignCenter)
        col.addWidget(self._cam_err_lbl)
        row = QHBoxLayout()
        row.addStretch(1)
        retry = QPushButton("  Retry")
        retry.setObjectName("Accent")
        retry.setCursor(Qt.PointingHandCursor)
        retry.setIcon(icon("refresh", "#0A0E12"))
        retry.clicked.connect(self.retry_requested.emit)
        row.addWidget(retry)
        settings_btn = QPushButton("  Open Settings")
        settings_btn.setObjectName("Ghost")
        settings_btn.setCursor(Qt.PointingHandCursor)
        settings_btn.setIcon(icon("settings", PALETTE.text_dim))
        settings_btn.clicked.connect(self.open_settings_requested.emit)
        row.addWidget(settings_btn)
        row.addStretch(1)
        col.addLayout(row)
        col.addStretch(1)
        return panel

    def _toggle_detection(self) -> None:
        on = self._detect_btn.isChecked()
        self._set_detect_ui(on)
        self.detection_toggled.emit(on)

    def _set_detect_ui(self, on: bool) -> None:
        self._detect_on = on
        self._detect_btn.setChecked(on)
        self._detect_btn.setText("  Detection: ON" if on else "  Detection: OFF")
        self._detect_btn.setObjectName("Accent" if on else "Ghost")
        self._detect_btn.setIcon(icon("zap", "#0A0E12" if on else PALETTE.text_dim))
        # restyle after objectName change
        self._detect_btn.style().unpolish(self._detect_btn)
        self._detect_btn.style().polish(self._detect_btn)

    def show_camera_error(self, msg: str) -> None:
        self._cam_err_lbl.setText(msg)
        self._persp_stack.setCurrentIndex(1)
        self._status_badge.set_text_color("NO CAMERA", PALETTE.danger)

    def _clear_camera_error(self) -> None:
        if self._persp_stack.currentIndex() != 0:
            self._persp_stack.setCurrentIndex(0)

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

        # Manual entry — always available, and the reliable fallback when
        # auto-detection isn't trusted (or in confirm-manually mode).
        manual = QHBoxLayout()
        make_btn = QPushButton("＋ Make")
        make_btn.setObjectName("Accent")
        make_btn.setCursor(Qt.PointingHandCursor)
        make_btn.clicked.connect(lambda: self.manual_shot.emit("make"))
        miss_btn = QPushButton("－ Miss")
        miss_btn.setObjectName("Danger")
        miss_btn.setCursor(Qt.PointingHandCursor)
        miss_btn.clicked.connect(lambda: self.manual_shot.emit("miss"))
        manual.addWidget(make_btn)
        manual.addWidget(miss_btn)
        rail.layout().addLayout(manual)

        reset_btn = QPushButton("Reset counters")
        reset_btn.setObjectName("Ghost")
        reset_btn.setCursor(Qt.PointingHandCursor)
        reset_btn.clicked.connect(self.reset_requested.emit)
        rail.layout().addWidget(reset_btn)

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

        hint = QLabel("Your camera previews automatically. Score with +Make / "
                      "−Miss, or flip Detection: ON for automatic ball detection "
                      "(no setup needed); use Pick felt if the table read looks off.")
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

    def _start_demo(self) -> None:
        # the synthetic table is clean, so the demo showcases detection ON
        self._set_detect_ui(True)
        self.detection_toggled.emit(True)
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
        if not running:
            self._status_badge.set_text_color("IDLE", PALETTE.text_faint)
            self._persp.clear()
            self._bird.clear()

    def set_detection(self, on: bool) -> None:
        """Reflect the controller's auto-detection state (e.g. after Try demo)."""
        self._set_detect_ui(on)

    def on_frame(self, packet) -> None:
        if packet.perspective is not None:
            self._clear_camera_error()  # a frame means the camera is alive
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
        if packet.status == "preview":
            self._status_badge.set_text_color("PREVIEW", PALETTE.text_dim)
        elif packet.status == "detecting_nolock":
            self._status_badge.set_text_color("DETECTING (NO TABLE LOCK)", PALETTE.warn)
        elif packet.deviated:
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

    def on_suggestion(self, event) -> None:
        # confirm-manually mode: detector suggests, user taps Make/Miss to commit
        self._outcome.set_text_color(f"{event.outcome.value.upper()}?", PALETTE.info)

    def on_recording(self, on: bool) -> None:
        self._record_btn.setChecked(on)
        self._record_btn.setIcon(icon("activity", PALETTE.danger if on else PALETTE.text_dim))

    def on_status(self, status: str) -> None:
        if status == "running":
            self.set_running(True)
        elif status == "stopped":
            self.set_running(False)

    def on_replay_saved(self, path: str) -> None:
        self._status_badge.set_text_color("REPLAY SAVED", PALETTE.success)
