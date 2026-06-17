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

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSplitter,
    QStackedWidget,
    QStyle,
    QStyleOptionSlider,
    QVBoxLayout,
    QWidget,
)

from ...config import Settings
from ...vision.types import BallClass
from ..icons import icon
from ..theme import PALETTE
from ..widgets.common import Badge, Card, SegmentedControl, StatCard
from ..widgets.shot_clock_widget import ShotClockWidget
from ..widgets.video_view import VideoView


class _SeekSlider(QSlider):
    """Horizontal slider that JUMPS to the clicked position on the groove.

    Qt's default is to page-step toward a track click (so a click near the end
    nudges a few frames instead of seeking there). We map the click straight to a
    value, then defer to the base class so dragging from there still works and the
    normal sliderPressed/sliderReleased signals still fire.
    """

    def mousePressEvent(self, ev):  # noqa: N802 - Qt override
        if ev.button() == Qt.LeftButton and self.maximum() > self.minimum():
            opt = QStyleOptionSlider()
            self.initStyleOption(opt)
            handle = self.style().subControlRect(
                QStyle.CC_Slider, opt, QStyle.SC_SliderHandle, self)
            pos = ev.position().toPoint() if hasattr(ev, "position") else ev.pos()
            if not handle.contains(pos):  # clicked the groove, not the thumb
                val = QStyle.sliderValueFromPosition(
                    self.minimum(), self.maximum(), pos.x(), self.width())
                self.setValue(val)
                self.sliderMoved.emit(val)
        super().mousePressEvent(ev)


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
    video_play_pause = Signal(bool)           # paused?
    video_stop_requested = Signal()
    video_step = Signal(int)                  # ±frames
    video_seek = Signal(int)                  # absolute frame index
    video_speed = Signal(float)               # playback multiplier
    tuning_changed = Signal()                 # a live-tuning control changed (settings mutated in place)
    label_mode_toggled = Signal(bool)         # Training Mode on/off
    save_training_frame_requested = Signal(list)  # [(number, cx, cy, w, h) normalised]
    train_balls_requested = Signal()          # fine-tune on collected data

    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self._settings = settings
        self._running = False
        self._detect_on = False
        self._video_fps = 30.0
        self._drill_key = ""
        self._drill_name = ""
        self._pick_mode = False
        # transport scrub state
        self._user_is_seeking = False
        self._was_playing = False
        self._pending_seek: int | None = None
        # Training Mode state: editable balls for the CURRENT frame, each
        # [number, x, y, r] in camera px; -1 number = unlabelled/not-a-ball.
        self._training = False
        self._label_balls: list[list] = []
        self._label_sel = -1
        self._frame_wh = (1, 1)
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
        persp_card.add(self._transport_bar())   # video play/seek/step strip
        splitter.addWidget(persp_card)

        # Right rail swaps between the normal tuning/score rail and the Training
        # rail (number pad + save/train), so Training Mode reuses this same video +
        # transport — you scrub the playback and label in place.
        self._rail_stack = QStackedWidget()
        self._rail_stack.addWidget(self._stats_rail())      # 0 = normal
        self._rail_stack.addWidget(self._training_rail())   # 1 = training
        splitter.addWidget(self._rail_stack)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 4)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([400, 540, 340])
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

        # 'Try demo' removed — the synthetic demo source didn't reflect real
        # detection and confused the surface. Demo is still reachable via
        # Settings → Camera → "Demo simulation" for engineering use.
        self._mode = SegmentedControl(
            [("free_play", "Sandbox"), ("practice", "Practice"), ("drill", "Drill")],
            current=self._settings.mode)
        # Practice/Drill are deferred until detection milestones M1–M6 land
        # (see docs/ROADMAP.md): disable so the surface only offers what works.
        self._mode.disable_option("practice", "Coming soon — see roadmap")
        self._mode.disable_option("drill", "Coming soon — see roadmap")
        if self._mode.current() == "":  # saved mode was a now-disabled one
            self._mode.set_current("free_play")
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

    def _transport_bar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("Card")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(6)
        self._play_btn = self._tool_btn("play", "Play / pause")
        self._play_btn.setCheckable(True)
        self._play_btn.clicked.connect(self._toggle_play)
        stop_btn = self._tool_btn("stop", "Stop (back to first frame)")
        stop_btn.clicked.connect(self.video_stop_requested.emit)
        back_btn = QPushButton("◀|")
        back_btn.setObjectName("Ghost")
        back_btn.setToolTip("Step back one frame")
        back_btn.setCursor(Qt.PointingHandCursor)
        back_btn.clicked.connect(lambda: self.video_step.emit(-1))
        fwd_btn = QPushButton("|▶")
        fwd_btn.setObjectName("Ghost")
        fwd_btn.setToolTip("Step forward one frame")
        fwd_btn.setCursor(Qt.PointingHandCursor)
        fwd_btn.clicked.connect(lambda: self.video_step.emit(1))
        for w in (self._play_btn, stop_btn, back_btn, fwd_btn):
            lay.addWidget(w)
        self._seek = _SeekSlider(Qt.Horizontal)
        self._seek.setRange(0, 0)
        self._seek.setSingleStep(1)
        self._seek.sliderPressed.connect(self._on_seek_pressed)
        self._seek.sliderMoved.connect(self._on_seek_moved)
        self._seek.sliderReleased.connect(self._on_seek_released)
        # Debounce the live scrub: each seek decodes + (maybe) detects, so firing
        # on every mouse-move (60+/s) backs the worker up and frames arrive out of
        # order — the "rewinds frame-by-frame" feel. Cap it at ~12/s; the exact
        # final frame is seeked on release.
        self._seek_debounce = QTimer(self)
        self._seek_debounce.setSingleShot(True)
        self._seek_debounce.setInterval(80)
        self._seek_debounce.timeout.connect(self._emit_pending_seek)
        lay.addWidget(self._seek, 1)
        self._time_lbl = QLabel("0:00 / 0:00")
        self._time_lbl.setObjectName("Faint")
        lay.addWidget(self._time_lbl)
        self._speed_combo = QComboBox()
        self._speed_combo.addItems(["0.25×", "0.5×", "1×", "2×", "4×"])
        self._speed_combo.setCurrentText("1×")
        self._speed_combo.activated.connect(
            lambda _i: self.video_speed.emit(float(self._speed_combo.currentText().rstrip("×"))))
        lay.addWidget(self._speed_combo)
        self._transport = bar
        bar.setVisible(False)   # only shown for video-file sources
        return bar

    def _toggle_play(self) -> None:
        paused = self._play_btn.isChecked()
        self._play_btn.setIcon(icon("play" if paused else "pause", PALETTE.text_dim))
        self.video_play_pause.emit(paused)

    # --- seek-bar scrubbing -------------------------------------------------- #
    def _on_seek_pressed(self) -> None:
        """Drag started: stop the playback tick from fighting the thumb. If the
        video was playing we pause for the duration of the scrub and resume on
        release; either way update_video_state stops moving the thumb."""
        self._user_is_seeking = True
        self._was_playing = not self._play_btn.isChecked()  # checked == paused
        if self._was_playing:
            self.video_play_pause.emit(True)
            self._play_btn.setChecked(True)
            self._play_btn.setIcon(icon("play", PALETTE.text_dim))

    def _on_seek_moved(self, value: int) -> None:
        """Live preview while dragging — debounced so we don't flood the worker."""
        self._pending_seek = int(value)
        total = self._seek.maximum() + 1
        self._time_lbl.setText(f"{self._fmt_t(value)} / {self._fmt_t(total)}")
        if not self._seek_debounce.isActive():
            self._seek_debounce.start()

    def _emit_pending_seek(self) -> None:
        if self._pending_seek is not None:
            self.video_seek.emit(self._pending_seek)

    def _on_seek_released(self) -> None:
        """Drag ended: seek the exact final frame, then resume playback if we
        paused for the scrub."""
        self._seek_debounce.stop()
        if self._pending_seek is not None:
            self.video_seek.emit(self._pending_seek)
            self._pending_seek = None
        self._user_is_seeking = False
        if self._was_playing:
            self.video_play_pause.emit(False)
            self._was_playing = False
            self._play_btn.setChecked(False)
            self._play_btn.setIcon(icon("pause", PALETTE.text_dim))

    def _fmt_t(self, frames: int) -> str:
        s = frames / max(1.0, self._video_fps)
        return f"{int(s // 60)}:{int(s % 60):02d}"

    def set_video_mode(self, is_video: bool, total: int, fps: float) -> None:
        self._video_fps = fps or 30.0
        self._transport.setVisible(is_video)
        if is_video:
            self._seek.setRange(0, max(0, total - 1))
            self._time_lbl.setText(f"0:00 / {self._fmt_t(total)}")

    def update_video_state(self, pos: int, total: int, playing: bool) -> None:
        # Never move the thumb out from under the user mid-drag (the old "thumb
        # keeps pushing forward" bug). The range can still be (re)synced.
        if self._seek.maximum() != max(0, total - 1):
            self._seek.setRange(0, max(0, total - 1))
        if not self._user_is_seeking:
            self._seek.blockSignals(True)
            self._seek.setValue(pos)
            self._seek.blockSignals(False)
            self._time_lbl.setText(f"{self._fmt_t(pos)} / {self._fmt_t(total)}")
            self._play_btn.setChecked(not playing)
            self._play_btn.setIcon(icon("play" if not playing else "pause", PALETTE.text_dim))

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
        rail.setMinimumWidth(300)
        rail.setMaximumWidth(400)

        # Live tuning panel up top — adjust while watching the clip (instant, no
        # recalibration). The score readout below stays for manual +Make/-Miss.
        rail.layout().addWidget(self._tuning_section())
        rail.layout().addWidget(self._hsep())

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
    # Live tuning panel  (mutates the shared Settings in place → the pipeline
    # picks changes up on the very next frame, no recalibration)
    # ------------------------------------------------------------------ #
    def _hsep(self) -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFixedHeight(1)
        line.setStyleSheet(f"background:{PALETTE.text_faint};border:none;")
        return line

    def _tuning_section(self) -> QWidget:
        col = QVBoxLayout()
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(8)
        cap = QLabel("TUNING — adjust while watching")
        cap.setObjectName("StatLabel")
        col.addWidget(cap)

        # Live cue-ball status (updated every frame from the tracker output).
        self._cue_status = QLabel("○  CUE: —")
        self._cue_status.setStyleSheet("font-size: 17px; font-weight: 800;")
        col.addWidget(self._cue_status)
        self._cue_sub = QLabel("detection off")
        self._cue_sub.setObjectName("Faint")
        self._cue_sub.setWordWrap(True)
        col.addWidget(self._cue_sub)

        # Detector min-confidence: lower → finds more (recovers a faint cue),
        # higher → stricter (drops false blobs). Applies live.
        col.addWidget(self._conf_row())

        # Display toggles — pure render flags, instant.
        # 'Uniform ball size' is always on (Joe never wants per-frame detector
        # radius wobble), so it's not exposed as a toggle.
        self._settings.ui.normalize_ball_size = True
        for label, attr in (
            ("Show ball numbers", "show_ball_ids"),
            ("Show trajectories", "show_trajectories"),
            ("Clean schematic bird's-eye", "schematic_birdseye"),
            ("Show overlays", "show_overlays"),
        ):
            col.addWidget(self._ui_toggle(label, attr))

        w = QWidget()
        w.setLayout(col)
        return w

    def _conf_row(self) -> QWidget:
        row = QVBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2)
        head = QHBoxLayout()
        lbl = QLabel("Detector min-confidence")
        lbl.setObjectName("Muted")
        self._conf_val = QLabel(f"{self._settings.detection.confidence_floor:.2f}")
        self._conf_val.setObjectName("Faint")
        head.addWidget(lbl)
        head.addStretch(1)
        head.addWidget(self._conf_val)
        row.addLayout(head)
        sld = QSlider(Qt.Horizontal)
        sld.setRange(10, 90)
        sld.setValue(int(round(self._settings.detection.confidence_floor * 100)))
        sld.valueChanged.connect(self._on_conf_changed)
        row.addWidget(sld)
        w = QWidget()
        w.setLayout(row)
        return w

    def _ui_toggle(self, label: str, attr: str) -> QCheckBox:
        cb = QCheckBox(label)
        cb.setCursor(Qt.PointingHandCursor)
        cb.setChecked(bool(getattr(self._settings.ui, attr)))
        cb.toggled.connect(lambda on, a=attr: self._on_ui_toggle(a, on))
        return cb

    def _on_conf_changed(self, v: int) -> None:
        self._settings.detection.confidence_floor = v / 100.0
        self._conf_val.setText(f"{v / 100.0:.2f}")
        self.tuning_changed.emit()

    def _on_ui_toggle(self, attr: str, on: bool) -> None:
        setattr(self._settings.ui, attr, on)
        self.tuning_changed.emit()

    def _update_cue_status(self, packet) -> None:
        if not self._detect_on:
            self._cue_status.setText("○  CUE: —")
            self._cue_status.setStyleSheet("font-size:17px;font-weight:800;color:%s;"
                                           % PALETTE.text_faint)
            self._cue_sub.setText("detection off — flip Detection: ON")
            return
        cue = next((t for t in (packet.tracks or []) if t.cls == BallClass.CUE), None)
        if cue is not None:
            self._cue_status.setText("●  CUE: TRACKED")
            self._cue_status.setStyleSheet("font-size:17px;font-weight:800;color:%s;"
                                           % PALETTE.success)
            self._cue_sub.setText(f"id #{cue.id} · ({cue.x:.0f}, {cue.y:.0f}) · "
                                  f"{packet.n_balls} balls · {packet.fps:.0f} fps")
        else:
            self._cue_status.setText("○  CUE: searching…")
            self._cue_status.setStyleSheet("font-size:17px;font-weight:800;color:%s;"
                                           % PALETTE.warn)
            self._cue_sub.setText(f"{packet.n_balls} balls · {packet.fps:.0f} fps")

    # ------------------------------------------------------------------ #
    # Training Mode — label/correct ball numbers on the playback video
    # ------------------------------------------------------------------ #
    def _training_rail(self) -> QWidget:
        rail = Card(padding=16, spacing=10)
        rail.setMinimumWidth(320)
        rail.setMaximumWidth(440)
        cap = QLabel("TRAINING MODE")
        cap.setObjectName("StatLabel")
        rail.add(cap)
        hint = QLabel("Scrub/pause the video to a clear frame. Click a ball, then "
                      "tap its correct number. Click an empty spot to ADD a missed "
                      "ball. Save good frames, then Train.")
        hint.setObjectName("Faint")
        hint.setWordWrap(True)
        rail.add(hint)
        self._label_status = QLabel("Click a ball on the camera view to select it.")
        self._label_status.setObjectName("Muted")
        self._label_status.setWordWrap(True)
        rail.add(self._label_status)

        pad = QGridLayout()
        pad.setSpacing(5)
        self._label_btns = {}
        items = [("Cue", 0)] + [(str(i), i) for i in range(1, 16)] + [("Not a ball", -1)]
        for idx, (txt, num) in enumerate(items):
            b = QPushButton(txt)
            b.setEnabled(False)
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _=False, n=num: self._assign_label(n))
            self._label_btns[num] = b
            pad.addWidget(b, idx // 4, idx % 4)
        pw = QWidget()
        pw.setLayout(pad)
        rail.add(pw)

        save = QPushButton("＋ Save this frame")
        save.setObjectName("Accent")
        save.setCursor(Qt.PointingHandCursor)
        save.clicked.connect(self._save_label_frame)
        rail.add(save)
        self._label_count = QLabel("0 frames collected")
        self._label_count.setObjectName("Faint")
        rail.add(self._label_count)
        rail.add(self._hsep())
        train = QPushButton("Train model on collected data")
        train.setCursor(Qt.PointingHandCursor)
        train.clicked.connect(self.train_balls_requested.emit)
        rail.add(train)
        tnote = QLabel("Training fine-tunes the model on what you've labelled and "
                       "switches the app to it. Re-train if your camera moves.")
        tnote.setObjectName("Faint")
        tnote.setWordWrap(True)
        rail.add(tnote)
        rail.layout().addStretch(1)
        return rail

    def set_training(self, on: bool) -> None:
        """Enter/leave Training Mode: swap the right rail and make the camera view
        clickable for labelling. Reuses the normal video + transport."""
        self._training = on
        self._rail_stack.setCurrentIndex(1 if on else 0)
        self._persp.set_pickable(on or self._pick_mode)
        self.label_mode_toggled.emit(on)
        if on:
            self._status_badge.set_text_color("TRAINING — label the balls", PALETTE.info)

    def _ingest_label_frame(self, packet) -> None:
        self._last_persp = packet.perspective
        h, w = packet.perspective.shape[:2]
        self._frame_wh = (w, h)
        self._label_balls = [[int(getattr(d, "number", -1)), float(d.x), float(d.y),
                              float(d.radius)] for d in (packet.raw_dets or [])]
        self._label_sel = -1
        self._redraw_label()
        self._update_label_buttons()

    def _default_label_r(self) -> float:
        rs = [b[3] for b in self._label_balls if b[3] > 0]
        return float(np.median(rs)) if rs else max(8.0, self._frame_wh[0] * 0.012)

    def _on_label_click(self, xf: float, yf: float) -> None:
        w, h = self._frame_wh
        x, y = xf * w, yf * h
        best, bd = -1, 1e18
        for i, b in enumerate(self._label_balls):
            dd = (b[1] - x) ** 2 + (b[2] - y) ** 2
            if dd < bd:
                bd, best = dd, i
        if best >= 0 and bd < (0.03 * max(w, h)) ** 2:
            self._label_sel = best
            self._label_status.setText("Tap the correct number for the selected ball.")
        else:
            self._label_balls.append([-1, x, y, self._default_label_r()])
            self._label_sel = len(self._label_balls) - 1
            self._label_status.setText("Added a ball — tap its number (or 'Not a ball' to remove).")
        self._update_label_buttons()
        self._redraw_label()

    def _assign_label(self, num: int) -> None:
        if 0 <= self._label_sel < len(self._label_balls):
            if num < 0:
                self._label_balls.pop(self._label_sel)   # 'not a ball' -> remove
            else:
                self._label_balls[self._label_sel][0] = num
            self._label_sel = -1
            self._update_label_buttons()
            self._redraw_label()

    def _update_label_buttons(self) -> None:
        on = 0 <= self._label_sel < len(self._label_balls)
        for b in self._label_btns.values():
            b.setEnabled(on)

    def _redraw_label(self) -> None:
        frame = getattr(self, "_last_persp", None)
        if frame is None:
            return
        img = frame.copy()
        for i, (num, x, y, r) in enumerate(self._label_balls):
            c = (int(x), int(y))
            sel = i == self._label_sel
            col = (0, 255, 255) if sel else (60, 220, 60)
            cv2.circle(img, c, max(4, int(r)), col, 3 if sel else 2, cv2.LINE_AA)
            lbl = "C" if num == 0 else (str(num) if num > 0 else "?")
            cv2.putText(img, lbl, (c[0] - 9, c[1] - int(r) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)
        self._persp.set_frame(img)

    def _save_label_frame(self) -> None:
        w, h = self._frame_wh
        boxes = [(num, x / w, y / h, 2 * r / w, 2 * r / h)
                 for (num, x, y, r) in self._label_balls if num >= 0]
        if not boxes:
            self._label_status.setText("Label at least one ball before saving.")
            return
        self.save_training_frame_requested.emit(boxes)
        self._label_status.setText(f"Saved {len(boxes)} balls. Scrub to another frame and keep going.")

    def set_training_count(self, text: str) -> None:
        if hasattr(self, "_label_count"):
            self._label_count.setText(text)

    # ------------------------------------------------------------------ #
    # Intent
    # ------------------------------------------------------------------ #
    def set_drill(self, drill_key: str, drill_name: str) -> None:
        self._drill_key = drill_key
        self._drill_name = drill_name
        self._mode.set_current("drill")
        self._drill_lbl.setText(f"Drill: {drill_name}" if drill_name else "")

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
        elif self._training:
            self._on_label_click(xf, yf)

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
            if self._training:
                self._ingest_label_frame(packet)   # draw the labelling overlay
            else:
                self._persp.set_frame(packet.perspective)
        if packet.birdseye is not None:
            self._bird.set_frame(packet.birdseye)
        self._fps_lbl.setText(f"{packet.fps:.0f} fps")
        self._balls_lbl.setText(f"{packet.n_balls} balls")
        self._update_cue_status(packet)
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
