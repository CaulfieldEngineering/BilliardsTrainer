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

import logging
import random

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
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
from ..icons import icon
from ..theme import PALETTE
from ..widgets.common import Badge, Card
from ..widgets.shot_clock_widget import ShotClockWidget
from ..widgets.video_view import VideoView

log = logging.getLogger("ui.live")


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


# Every label each pill can show. The pills size themselves to the widest, so
# adding a longer string here is all it takes to keep the row from clipping.
_MODE_TEXTS = ("LIVE", "PREVIEW", "PLAYBACK", "IDLE", "TRAINING", "NO CAMERA")
_ALERT_TEXTS = ("DEGRADED FEED — CHECK CAPTURE", "NO TABLE LOCK",
                "RELOCKING TABLE", "FINDING TABLE", "SHOT IN PLAY",
                "REPLAY SAVED", "LABEL THE BALLS")


class _StatusPill(QLabel):
    """Broadcast-style status bug: a coloured dot + spaced uppercase label on a
    dark pill — the LIVE/PLAYBACK indicator you'd see on a stream deck, not a
    generic badge.

    FIXED WIDTH, always. These sit in the transport row, and a label that
    resizes with its text drags every button beside it to a new position on each
    state change — disorienting when you're reaching for Record without looking.
    """

    def __init__(self, texts, parent=None):
        """``texts`` is every label this pill can ever show. Its width is
        MEASURED from the widest of them: a fixed width that is too small
        silently truncates (QLabel elides rather than growing), which is how the
        record clock once rendered as "RE"."""
        super().__init__(parent)
        self.setTextFormat(Qt.RichText)
        self.setAlignment(Qt.AlignCenter)
        fm = self.fontMetrics()
        widest = max(fm.horizontalAdvance(t) for t in texts)
        self.setFixedWidth(widest + 52)          # dot + gap + pill padding
        self._blank()

    def _blank(self) -> None:
        self.setText("")
        self.setStyleSheet("background: transparent; border-radius: 5px;"
                           "padding: 4px 12px;")

    def set_text_color(self, text: str, color: str) -> None:
        self.setText(f'<span style="color:{color}; font-size:10px;">●</span>'
                     f'&nbsp;&nbsp;{text}')
        self.setStyleSheet(
            "background: rgba(0,0,0,0.35); border-radius: 5px;"
            "padding: 4px 12px; font-size: 11px; font-weight: 800;"
            f"letter-spacing: 1.5px; color: {PALETTE.text};")

    def clear_status(self) -> None:
        """Nothing to report — keeps its footprint so the row never shifts."""
        self._blank()



class _StatRow(QWidget):
    """One line of the scoreboard: small caps label left, value right.
    Joe: "just do a vertical stats list here? I don't need big blocks
    for each number." set_value keeps the old StatCard call sites."""

    def __init__(self, label: str, color: str, parent=None):
        super().__init__(parent)
        self.setObjectName("StatRow")   # transparent: rows are not boxes
        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 0, 2, 0)
        lay.setSpacing(8)
        lbl = QLabel(label)
        lbl.setObjectName("StatLabel")
        self.value_label = QLabel("—")
        self.value_label.setProperty("statColor", color)
        self.value_label.setStyleSheet(
            f"font-size: 18px; font-weight: 700; color: {PALETTE.text_faint};")
        lay.addWidget(lbl)
        lay.addStretch(1)
        lay.addWidget(self.value_label)

    def set_value(self, value: str) -> None:
        self.value_label.setText(str(value))
        color = self.value_label.property("statColor")
        self.value_label.setStyleSheet(
            f"font-size: 18px; font-weight: 700; color: {color};")


class LivePage(QWidget):
    start_requested = Signal(str, str, str)   # source, mode, drill_key
    stop_requested = Signal()
    record_toggled = Signal(bool)
    record_pause_toggled = Signal(bool)       # pause/resume the active recording
    retry_requested = Signal()                # re-open the camera after an error
    open_settings_requested = Signal()        # jump to Settings → Camera
    video_play_pause = Signal(bool)           # paused?
    video_stop_requested = Signal()
    video_step = Signal(int)                  # ±frames
    video_seek = Signal(int)                  # absolute frame index
    video_speed = Signal(float)               # playback multiplier
    mini_view_requested = Signal()            # pop out the always-on-top mini view
    tuning_changed = Signal()                 # a live-tuning control changed (settings mutated in place)
    clock_pause_toggled = Signal(bool)        # shot-clock Pause/Resume (rail button)
    clock_enabled_toggled = Signal(bool)      # rail on/off - main window syncs the Play menu
    label_mode_toggled = Signal(bool)         # Training Mode on/off
    save_training_frame_requested = Signal(object)  # [(number, cx, cy, w, h) normalised] (object marshals cleanly cross-thread)
    train_balls_requested = Signal()          # fine-tune on collected data

    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self._settings = settings
        self._running = False
        self._detect_on = False
        self._video_fps = 30.0
        self._is_video = False   # playing back a recording (badge: PLAYBACK)
        self._recording_on = False
        self._stats_active = False
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
        self._last_packet = None   # newest frame packet, for labelling on entry
        self._settings.ui.normalize_ball_size = True  # never per-frame radius wobble
        self._build()

    # ------------------------------------------------------------------ #
    def _timeline_panel(self) -> QWidget:
        """The shot-timeline lane in its collapsible fold, plus the page's
        Space play/pause shortcut (it rides with the transport wiring)."""
        from ..widgets.shot_timeline import ShotTimeline
        self._timeline = ShotTimeline(
            pre_roll_s=getattr(self._settings.ui, "pre_shot_s", 5.0))
        self._timeline.clicked.connect(self._on_timeline_clicked)
        # Editor scrubbing on the lane (v3.1): pause for the drag so playback
        # stops fighting the playhead, seek continuously, resume on release —
        # and because every drag position is a REAL seek, Space afterwards
        # resumes from where the finger left off, not from the beginning.
        self._timeline.scrub_started.connect(self._on_seek_pressed)
        self._timeline.scrubbed.connect(self._on_timeline_scrubbed)
        self._timeline.scrub_ended.connect(self._on_seek_released)
        # SPACE = play/pause, globally on this page. Transport buttons give
        # up keyboard focus so Space can never double-fire through a focused
        # button (the classic checkable-button + shortcut trap).
        from PySide6.QtGui import QKeySequence, QShortcut
        sc = QShortcut(QKeySequence(Qt.Key_Space), self)
        sc.setContext(Qt.WindowShortcut)
        sc.activated.connect(self._on_space)
        tl_card = Card(padding=6, spacing=2)
        tl_card.add(self._timeline)
        from ..widgets.collapsible import CollapsibleSection
        self._tl_fold = CollapsibleSection(
            "SHOT TIMELINE", tl_card, "shot_timeline_panel", self._settings)
        return self._tl_fold

    def _build(self) -> None:
        root = QVBoxLayout(self)
        # Joe: "tighten up the entire UI... a lot of negative space" —
        # page chrome shrinks so the video views get the pixels.
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(8)
        root.addWidget(self._control_bar())
        root.addWidget(self._timeline_panel())

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(8)
        splitter.setChildrenCollapsible(False)

        bird_card = Card(padding=6, spacing=4)
        bird_card.add(self._caption("BIRD'S-EYE"))
        self._bird = VideoView("Bird's-eye view")
        bird_card.add(self._bird)
        splitter.addWidget(bird_card)

        persp_card = Card(padding=6, spacing=4)
        persp_card.add(self._caption("LIVE CAMERA"))
        # The live view and an inline camera-error panel share a stack, so a
        # camera failure surfaces right here (with Retry / Open Settings) instead
        # of sending the user hunting through menus.
        self._persp_stack = QStackedWidget()
        self._persp = VideoView("Connecting to camera…")
        self._persp.clicked.connect(self._on_persp_click)
        # Corner feed-stats chip (Joe's ask): resolution truth ON the video,
        # not in the transport. Toggle: Settings -> Appearance.
        self._feed_chip = QLabel("", self._persp)
        self._feed_chip.setStyleSheet(
            "background: rgba(0,0,0,0.55); color: #CFD8DC; border-radius: 4px;"
            "padding: 2px 8px; font-size: 10px; font-weight: 600;"
            "letter-spacing: 0.5px;")
        self._feed_chip.hide()
        self._persp_stack.addWidget(self._persp)
        self._persp_stack.addWidget(self._camera_error_panel())
        persp_card.add(self._persp_stack)
        splitter.addWidget(persp_card)

        # Right rail swaps between the normal tuning/score rail and the Training
        # rail (number pad + save/train), so Training Mode reuses this same video +
        # transport — you scrub the playback and label in place.
        self._rail_stack = QStackedWidget()
        # A QStackedWidget's minimum is the MAX over ALL pages — the hidden
        # Training page (320px floor) was re-imposing a window minimum after
        # the rail became always-visible. Ignored horizontal policy lets the
        # splitter squeeze the stack instead of the window refusing to shrink.
        from PySide6.QtWidgets import QSizePolicy
        self._rail_stack.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self._rail_stack.addWidget(self._stats_rail())      # 0 = normal
        self._rail_stack.addWidget(self._training_rail())   # 1 = training
        from ..widgets.shot_list import ShotListPanel
        self._shot_list = ShotListPanel(
            pre_roll_s=getattr(self._settings.ui, "pre_shot_s", 5.0))
        self._shot_list.shot_selected.connect(self._on_timeline_clicked)
        self._shot_list.outcome_corrected.connect(self._on_outcome_corrected)
        self._shot_list.fix_labels_requested.connect(self._on_fix_labels_at)
        self._shot_list.export_requested.connect(self._on_export_shot)
        self._shot_list.dossier_requested.connect(self._on_export_dossier)
        self._rail_stack.addWidget(self._shot_list)          # 2 = playback review
        splitter.addWidget(self._rail_stack)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 4)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([430, 570, 260])
        root.addWidget(splitter, 1)

    def _caption(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("StatLabel")
        return lbl

    def _control_bar(self) -> QWidget:
        """The media header: status bug, record cluster, playback cluster.
        Everything is always visible; enablement follows context (record on the
        live camera, playback while reviewing a session)."""
        bar = QFrame()
        bar.setObjectName("Card")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(10, 5, 10, 5)
        lay.setSpacing(8)

        # Broadcast bug: ● LIVE / PREVIEW / PLAYBACK. MODE ONLY — this answers
        # "what is the app doing", and must never be commandeered by a passing
        # condition (a shot in play, a table relock). Those go to the alert pill
        # at the far right of this bar.
        self._status_badge = _StatusPill(_MODE_TEXTS)
        lay.addWidget(self._status_badge)
        lay.addWidget(self._vsep())

        # Record cluster: a broadcast-deck capsule — ● ❚❚ ■ + elapsed clock.
        # The capsule (and clock) light up red while recording, amber on pause.
        self._rec_capsule = QFrame()
        self._rec_capsule.setObjectName("RecCapsule")
        cap = QHBoxLayout(self._rec_capsule)
        cap.setContentsMargins(8, 2, 10, 2)
        cap.setSpacing(2)
        self._rec_btn = self._tool_btn("rec", "Start recording this session")
        self._rec_btn.setIcon(icon("rec", "#B0524C", size=26))  # muted red, armed
        self._rec_btn.clicked.connect(self._on_rec_clicked)
        self._rec_pause_btn = self._tool_btn("rec-pause", "Pause / resume recording")
        self._rec_pause_btn.setCheckable(True)
        self._rec_pause_btn.setEnabled(False)
        self._rec_pause_btn.clicked.connect(self._on_rec_pause)
        self._rec_stop_btn = self._tool_btn("rec-stop", "Stop recording + save")
        self._rec_stop_btn.setEnabled(False)
        self._rec_stop_btn.clicked.connect(lambda: self.record_toggled.emit(False))
        for w in (self._rec_btn, self._rec_pause_btn, self._rec_stop_btn):
            cap.addWidget(w)
        # The clock, fourth redesign — and the boring one that ends the
        # saga: PLAIN TEXT ("0:00"), ALWAYS VISIBLE (dim when idle), width
        # fixed once from plain font metrics at first show. Recording state
        # lives in the capsule outline and the colour of these digits — the
        # rich-text "● REC" badge is gone, and with it the mis-measuring,
        # the worst-case ratchet that held the capsule twice as wide as its
        # content (Joe: "way off to the side"), and the first-record
        # re-measure that made the layout jump on first load.
        self._rec_time = QLabel("0:00")
        self._rec_time.setObjectName("RecClock")
        self._rec_time.setTextFormat(Qt.PlainText)
        self._rec_time.setAlignment(Qt.AlignCenter)
        self._rec_time.setStyleSheet(f"color: {PALETTE.text_faint};")
        self._rec_time.setFixedHeight(self._rec_btn.sizeHint().height())
        cap.addWidget(self._rec_time)
        lay.addWidget(self._rec_capsule)
        # Mic level meter: proves the audio path end-to-end at a glance (the
        # HDMI feed has no sound, so this reads the selected USB mic).
        from ..widgets.audio_meter import AudioMeter
        self._audio_meter = AudioMeter()
        lay.addWidget(self._audio_meter)
        self._audio_meter.configure(self._settings.recording.audio,
                                    self._settings.recording.audio_device)
        lay.addWidget(self._vsep())

        # Playback cluster — greyed out until a session is open.
        self._play_btn = self._tool_btn("play-solid", "Play / pause")
        self._play_btn.setCheckable(True)
        self._play_btn.clicked.connect(self._toggle_play)
        pb_stop = self._tool_btn("rec-stop", "Stop (back to first frame)")
        pb_stop.clicked.connect(self.video_stop_requested.emit)
        back_btn = self._tool_btn("step-back", "Step back one frame")
        back_btn.clicked.connect(lambda: self.video_step.emit(-1))
        fwd_btn = self._tool_btn("step-fwd", "Step forward one frame")
        fwd_btn.clicked.connect(lambda: self.video_step.emit(1))
        for w in (self._play_btn, pb_stop, back_btn, fwd_btn):
            lay.addWidget(w)
        self._seek = _SeekSlider(Qt.Horizontal)
        self._seek.setRange(0, 0)
        self._seek.setSingleStep(1)
        self._seek.sliderPressed.connect(self._on_seek_pressed)
        self._seek.sliderMoved.connect(self._on_seek_moved)
        self._seek.sliderReleased.connect(self._on_seek_released)
        # Debounce the live scrub: each seek decodes + (maybe) detects; cap at
        # ~12/s and seek the exact final frame on release.
        self._seek_debounce = QTimer(self)
        self._seek_debounce.setSingleShot(True)
        self._seek_debounce.setInterval(80)
        self._seek_debounce.timeout.connect(self._emit_pending_seek)
        lay.addWidget(self._seek, 1)
        self._time_lbl = QLabel("0:00 / 0:00")
        self._time_lbl.setObjectName("Faint")
        # Fixed so scrubbing (0:00 -> 21:48 -> 1:02:33) can't shuffle the speed
        # selector and Train AI button around under the cursor — measured, so a
        # long session can't truncate it.
        self._time_lbl.setFixedWidth(
            self._time_lbl.fontMetrics().horizontalAdvance("0:00:00 / 0:00:00") + 16)
        self._time_lbl.setAlignment(Qt.AlignCenter)
        lay.addWidget(self._time_lbl)
        self._speed_combo = QComboBox()
        self._speed_combo.addItems(["0.25×", "0.5×", "1×", "2×", "4×"])
        self._speed_combo.setCurrentText("1×")
        self._speed_combo.activated.connect(
            lambda _i: self.video_speed.emit(float(self._speed_combo.currentText().rstrip("×"))))
        lay.addWidget(self._speed_combo)
        # Analysis overlays (Joe): SAME stored geometry the phone draws.
        self._aim_btn = QPushButton("Aim")
        self._paths_btn = QPushButton("Paths")
        self._why_btn = QPushButton("Lines")
        for b, attr in ((self._aim_btn, "overlay_aim"),
                        (self._paths_btn, "overlay_paths"),
                        (self._why_btn, "overlay_why")):
            b.setObjectName("Ghost")
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.setChecked(bool(getattr(self._settings.ui, attr, False)))
            b.toggled.connect(lambda on, a=attr: self._set_overlay(a, on))
            lay.addWidget(b)
        self._fix_labels_btn = QPushButton("Train AI")
        self._fix_labels_btn.setObjectName("Ghost")
        self._fix_labels_btn.setCheckable(True)
        self._fix_labels_btn.setCursor(Qt.PointingHandCursor)
        self._fix_labels_btn.setToolTip("Teach the AI this table: correct ball "
                                        "numbers on this frame")
        self._fix_labels_btn.toggled.connect(self.set_training)
        lay.addWidget(self._fix_labels_btn)

        # Transient CONDITIONS live here, apart from the mode bug and last in the
        # row so nothing sits downstream of them. Fixed width, blank when there
        # is nothing to say.
        lay.addWidget(self._vsep())
        self._alert_badge = _StatusPill(_ALERT_TEXTS)
        lay.addWidget(self._alert_badge)

        # Pop-out mini view (Joe's ask): a small always-on-top feed for keeping
        # an eye on the table while other windows have the screen.
        self._mini_btn = self._tool_btn("pip", "Pop out a small always-on-top view")
        self._mini_btn.clicked.connect(self.mini_view_requested.emit)
        lay.addWidget(self._mini_btn)

        self._playback_widgets = [self._play_btn, pb_stop, back_btn, fwd_btn,
                                  self._seek, self._time_lbl, self._speed_combo,
                                  self._fix_labels_btn]
        for w in self._playback_widgets:
            w.setEnabled(False)
        # The bar's children are fixed-width BY DESIGN (nothing may shift when
        # a label changes), which makes their SUM the window's minimum width —
        # measured at 1788px, the wall Joe hit trying to put Spotify
        # side-by-side. Override the propagated minimum; resizeEvent hides the
        # optional items before anything could clip.
        self._bar_optional = [self._alert_badge, self._audio_meter,
                              self._time_lbl, self._speed_combo,
                              self._fix_labels_btn]
        bar.setMinimumWidth(360)
        return bar

    def _vsep(self) -> QFrame:
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet(f"color:{PALETTE.text_faint};")
        return sep

    def _on_rec_clicked(self) -> None:
        self.record_toggled.emit(True)          # start recording

    def _on_rec_pause(self) -> None:
        paused = self._rec_pause_btn.isChecked()
        # Amber pause glyph while paused — reads at a glance across the room.
        self._rec_pause_btn.setIcon(
            icon("rec-pause", PALETTE.warn if paused else PALETTE.text_dim, size=26))
        self.record_pause_toggled.emit(paused)
        self._tick_rec_time()

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

    def _toggle_play(self) -> None:
        paused = self._play_btn.isChecked()
        self._play_btn.setIcon(icon("play-solid" if paused else "rec-pause", PALETTE.text_dim))
        self.video_play_pause.emit(paused)

    def _on_space(self) -> None:
        """Space = play/pause on a loaded session; inert on live camera."""
        if self._is_video and self._play_btn.isEnabled():
            self._play_btn.toggle()
            self._toggle_play()

    def _on_timeline_scrubbed(self, seconds: float) -> None:
        """Lane drag -> debounced frame seek (same machinery as the slider)."""
        frame = int(seconds * (self._video_fps or 30.0))
        self._pending_seek = max(0, min(self._seek.maximum(), frame))
        if not self._seek_debounce.isActive():
            self._seek_debounce.start()

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

    def _apply_compact(self) -> None:
        """CONTINUOUS chrome fit, re-applied on resize, mode and record
        changes. A fixed threshold left a silent-overflow band (the record
        clock appeared and crushed the capsule at ~1100px, Joe's screenshot
        x2): instead, measure the bar's visible demand against its actual
        width and shed the lowest-priority chrome until the row fits. Mode
        decides the base set — playback transport is dead weight while live,
        the record capsule can't record a playback."""
        if not hasattr(self, "_bar_optional"):
            return
        w = self.width()
        # base set by mode
        for x in self._playback_widgets:
            x.setVisible(self._is_video)
        self._rec_capsule.setVisible(not self._is_video)
        for x in self._bar_optional:
            x.setVisible(True)
        # shed order: least-load-bearing first
        shed = [self._alert_badge, self._audio_meter,
                getattr(self, "_fix_labels_btn", None),
                getattr(self, "_speed_combo", None),
                getattr(self, "_time_lbl", None),
                # emergency tier: the mode pill (the mini view carries its own
                # dot) and finally the clock — the red REC button still burns
                self._status_badge,
                self._rec_time if self._recording_on else None]
        bar = self._rec_capsule.parentWidget()
        margin = 60          # bar paddings + separators slack

        def demand() -> int:
            total = 0
            lay = bar.layout()
            for i in range(lay.count()):
                it = lay.itemAt(i).widget()
                if it is not None and it.isVisibleTo(bar):
                    total += max(it.minimumSizeHint().width(), it.minimumWidth())
            return total + margin
        for x in shed:
            if x is None:
                continue
            if demand() <= w:
                break
            x.setVisible(False)
        if hasattr(self, "_bird"):
            # narrowest tier: the camera IS the app; the bird's-eye yields
            self._bird.parentWidget().setVisible(w >= 640)

    def set_video_mode(self, is_video: bool, total: int, fps: float) -> None:
        self._video_fps = fps or 30.0
        self._is_video = is_video   # badge shows PLAYBACK instead of LIVE
        self._apply_compact()       # bar chrome follows the mode when narrow
        self._update_stats_active()
        for w in self._playback_widgets:
            w.setEnabled(is_video)
        if hasattr(self, "_jump_section"):
            self._jump_section.setVisible(is_video)  # jumping needs a seekable clip
        self._timeline.clear()
        self._shot_list.set_shots([])
        if not self._training:
            self._rail_stack.setCurrentIndex(2 if is_video else 0)
        if is_video:
            self._timeline.set_duration(total / (fps or 30.0))
            self._seek.setRange(0, max(0, total - 1))
            self._time_lbl.setText(f"0:00 / {self._fmt_t(total)}")
        else:
            self.set_media_path("")

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
            self._timeline.set_playhead(pos / (self._video_fps or 30.0))
            self._play_btn.setChecked(not playing)
            self._play_btn.setIcon(icon("play-solid" if not playing else "rec-pause", PALETTE.text_dim))
        self._sync_audio(pos, playing)

    def reconfigure_audio_meter(self) -> None:
        """Re-apply mic-meter settings (device / on-off) after a settings change."""
        self._audio_meter.configure(self._settings.recording.audio,
                                    self._settings.recording.audio_device)

    # --- playback audio ------------------------------------------------------ #
    # Frames replay through the analysis pipeline (cv2), which is silent; a
    # QMediaPlayer plays the same mp4's audio track and is kept glued to the
    # frame clock — play/pause/seek/speed all follow the transport.

    def set_media_path(self, path: str) -> None:
        """The session clip under playback ('' = live camera, audio off)."""
        self._media_path = path or ""
        self._timeline.set_media_source(self._media_path)   # v3 filmstrip
        player = getattr(self, "_audio_player", None)
        if not self._media_path:
            if player is not None:
                player.stop()
            return
        self._ensure_audio_player()
        if self._audio_player is not None:
            from PySide6.QtCore import QUrl
            self._audio_player.setSource(QUrl.fromLocalFile(self._media_path))

    def _ensure_audio_player(self) -> None:
        if getattr(self, "_audio_player", None) is not None or \
                getattr(self, "_audio_player_dead", False):
            return
        try:
            from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
        except Exception:  # noqa: BLE001 - multimedia backend absent
            self._audio_player = None
            self._audio_player_dead = True
            return
        self._audio_out = QAudioOutput(self)
        self._audio_out.setVolume(1.0)
        self._audio_player = QMediaPlayer(self)
        self._audio_player.setAudioOutput(self._audio_out)
        self.video_speed.connect(self._audio_player.setPlaybackRate)

    def _sync_audio(self, pos: int, playing: bool) -> None:
        player = getattr(self, "_audio_player", None)
        if player is None or not getattr(self, "_media_path", ""):
            return
        from PySide6.QtMultimedia import QMediaPlayer
        target_ms = int(pos / max(1.0, self._video_fps) * 1000)
        # Re-anchor RARELY. Frequent setPosition() calls replay the same second
        # over and over — that was the "1s echo" during playback. Only correct
        # a big divergence (a real seek), and never more than once every 2s.
        import time as _t
        now = _t.monotonic()
        if (abs(player.position() - target_ms) > 1000
                and now - getattr(self, "_last_audio_anchor", 0.0) > 2.0):
            self._last_audio_anchor = now
            player.setPosition(target_ms)
        if playing and player.playbackState() != QMediaPlayer.PlayingState:
            player.play()
        elif not playing and player.playbackState() == QMediaPlayer.PlayingState:
            player.pause()

    def show_camera_error(self, msg: str) -> None:
        self._cam_err_lbl.setText(msg)
        self._persp_stack.setCurrentIndex(1)
        self._status_badge.set_text_color("NO CAMERA", PALETTE.danger)

    def _clear_camera_error(self) -> None:
        if self._persp_stack.currentIndex() != 0:
            self._persp_stack.setCurrentIndex(0)

    @staticmethod
    def _repolish(w) -> None:
        """Re-apply QSS after a dynamic property change."""
        w.style().unpolish(w)
        w.style().polish(w)

    def _tool_btn(self, ic: str, tip: str) -> QPushButton:
        from PySide6.QtCore import QSize
        btn = QPushButton()
        btn.setObjectName("Transport")
        btn.setIcon(icon(ic, PALETTE.text_dim, size=26))
        btn.setIconSize(QSize(20, 20))
        btn.setToolTip(tip)
        btn.setCursor(Qt.PointingHandCursor)
        # Media-app rule: transport buttons never hold keyboard focus, so
        # Space always means play/pause (the page shortcut) instead of
        # "click whichever button happens to be focused".
        btn.setFocusPolicy(Qt.NoFocus)
        return btn

    def _clock_panel(self) -> QWidget:
        """The SHOT CLOCK fold: readout, on/off + pause (Joe's rail
        controls), and the per-shot / after-break second dials."""
        from ..widgets.collapsible import CollapsibleSection
        clock_lay = QVBoxLayout()
        clock_lay.setContentsMargins(0, 0, 0, 0)
        clock_lay.setSpacing(6)
        # Table status beside the timer (Joe: "'Shot in Play' when no
        # timer running") - fed per-frame from packet clock state
        self._clock_status = QLabel("TABLE SETTLED")
        self._clock_status.setAlignment(Qt.AlignCenter)
        self._clock_status.setStyleSheet(
            f"font-size: 11px; font-weight: 700; letter-spacing: 1px;"
            f"color: {PALETTE.accent};")
        clock_lay.addWidget(self._clock_status)
        row = QHBoxLayout()
        row.addStretch(1)
        self._clock = ShotClockWidget()
        row.addWidget(self._clock)
        row.addStretch(1)
        clock_lay.addLayout(row)
        # On/Off + Pause/Resume (Joe: controls IN the shot clock field)
        btns = QHBoxLayout()
        self._clock_on_btn = QPushButton()
        self._clock_on_btn.setObjectName("Ghost")
        self._clock_on_btn.setCheckable(True)
        self._clock_on_btn.setChecked(self._settings.shot_clock.enabled)
        self._clock_on_btn.toggled.connect(self._on_clock_enabled)
        self._clock_pause_btn = QPushButton("Pause")
        self._clock_pause_btn.setObjectName("Ghost")
        self._clock_pause_btn.setCheckable(True)
        self._clock_pause_btn.toggled.connect(self._on_clock_pause)
        btns.addWidget(self._clock_on_btn)
        btns.addWidget(self._clock_pause_btn)
        clock_lay.addLayout(btns)
        # Per-shot and after-break seconds, applied live (shared settings
        # object; the next countdown picks the new length up)
        from PySide6.QtWidgets import QSpinBox
        for label, attr in (("Per shot", "seconds"),
                            ("After break", "break_seconds")):
            prow = QHBoxLayout()
            lab = QLabel(label)
            lab.setObjectName("StatLabel")
            spin = QSpinBox()
            spin.setRange(10, 300)
            spin.setSuffix(" s")
            spin.setValue(int(getattr(self._settings.shot_clock, attr)))
            spin.valueChanged.connect(
                lambda v, a=attr: (setattr(self._settings.shot_clock, a, int(v)),
                                   self.tuning_changed.emit()))
            prow.addWidget(lab)
            prow.addStretch(1)
            prow.addWidget(spin)
            clock_lay.addLayout(prow)
        # Per-cue volume (Joe: "a volume control for each beep type").
        # 0 mutes that cue alone; applied on the next play, persisted
        # via the debounced tuning save.
        from PySide6.QtWidgets import QSlider
        for label, attr in (("Start", "vol_start"), ("Warn", "vol_warn"),
                            ("3-2-1", "vol_tick"), ("Buzzer", "vol_expired"),
                            ("Scratch", "vol_scratch"),
                            ("Voice", "vol_voice")):
            vrow = QHBoxLayout()
            lab = QLabel(label)
            lab.setObjectName("StatLabel")
            sld = QSlider(Qt.Horizontal)
            sld.setRange(0, 100)
            sld.setFixedWidth(110)
            sld.setValue(int(getattr(self._settings.shot_clock, attr, 100)))
            sld.valueChanged.connect(
                lambda v, a=attr: (setattr(self._settings.shot_clock, a, int(v)),
                                   self.tuning_changed.emit()))
            sld.sliderReleased.connect(
                lambda a=attr: self._preview_cue(a))
            vrow.addWidget(lab)
            vrow.addStretch(1)
            vrow.addWidget(sld)
            clock_lay.addLayout(vrow)
        self._sync_clock_btn_text()
        self._clock_holder = QWidget()
        self._clock_holder.setLayout(clock_lay)
        self._clock_fold = CollapsibleSection(
            "SHOT CLOCK", self._clock_holder, "shot_clock", self._settings)
        return self._clock_fold

    def _stats_rail(self) -> QWidget:
        # Compact scoreboard — tight spacing, narrow column; more stats land
        # here later, so the layout leaves the room in the middle, not the edges.
        rail = Card(padding=8, spacing=6)
        # No hard minimum: the window's minimum width is the SUM of its
        # children's minimums, and this rail's old 240px floor was a third of
        # why the app refused to shrink for side-by-side use. The rail looks
        # best >=240 wide, so narrow-mode HIDES it (resizeEvent) rather than
        # squeezing it.
        rail.setMaximumWidth(300)

        # A VERTICAL LIST, not blocks (Joe: "I don't need big blocks for
        # each number") — label left, value right, one compact row per
        # stat. No "SESSION" fold either: everything here is per-session
        # by definition, so the header said nothing.
        # SHOTS above MAKES (Joe: "so it's easy to deduce where MAKE %
        # is coming from") — the denominator sits over its parts.
        self._shots_row = _StatRow("SHOTS", PALETTE.text)
        self._makes_row = _StatRow("MAKES", PALETTE.success)
        self._misses_row = _StatRow("MISSES", PALETTE.danger)
        self._k_pct = _StatRow("MAKE %", PALETTE.accent)
        self._k_streak = _StatRow("STREAK", PALETTE.text)
        # Joe's timer: counts the stay-down live (~estimate), then locks to
        # the camera-measured value when the stroke record lands.
        self._k_stay = _StatRow("STAY-DOWN", PALETTE.accent)
        from ..widgets.stay_down import StayDownTimer
        self._stay = StayDownTimer()
        stats_holder = QWidget()
        stats_lay = QVBoxLayout(stats_holder)
        stats_lay.setContentsMargins(0, 0, 0, 0)
        stats_lay.setSpacing(6)
        for row in (self._shots_row, self._makes_row, self._misses_row,
                    self._k_pct, self._k_streak, self._k_stay):
            stats_lay.addWidget(row)
        self._makes_val = self._makes_row.value_label
        self._misses_val = self._misses_row.value_label

        from ..widgets.collapsible import CollapsibleSection

        # Stats fold (Joe: "make the Stats section collapsible")
        self._stats_fold = CollapsibleSection(
            "STATS", stats_holder, "stats_panel", self._settings)
        rail.layout().addWidget(self._stats_fold)

        # Cue-stroke stats (Bluetooth IMU) — hidden unless the sensor is enabled.
        self._stroke_holder = self._stroke_section()
        self._stroke_fold = CollapsibleSection(
            "CUE SENSOR", self._stroke_holder, "cue_sensor", self._settings)
        self._stroke_fold.setVisible(self._settings.cue.enabled)
        rail.layout().addWidget(self._stroke_fold)

        # Shot clock — only visible when enabled, so it never crowds sandbox.
        rail.layout().addWidget(self._clock_panel())

        rail.layout().addStretch(1)

        return rail

    def _update_stats_active(self) -> None:
        """Stats belong to a RECORDING: coloured + counting only while the
        session recorder runs; greyed dashes any other time (Joe's rule)."""
        active = self._recording_on
        if active == self._stats_active:
            return
        self._stats_active = active
        for val in (self._makes_val, self._misses_val):
            color = val.property("statColor") if active else PALETTE.text_faint
            val.setStyleSheet(f"font-size: 18px; font-weight: 700; color: {color};")
        if not active:
            self._makes_val.setText("—")
            self._misses_val.setText("—")
            self._shots_row.set_value("—")
            self._k_pct.set_value("—")
            self._k_streak.set_value("—")
            self._stay.reset()
            self._set_stay("—", "idle")

    def _set_stay(self, text: str, kind: str) -> None:
        """Paint the stay-down row; no-op per frame unless it changed."""
        if text != getattr(self, "_stay_text", None):
            self._stay_text = text
            self._k_stay.set_value(text)
        if kind != getattr(self, "_stay_kind", None):
            self._stay_kind = kind
            color = {"climb": PALETTE.accent, "final": PALETTE.success,
                     "popped": PALETTE.danger}.get(kind, PALETTE.text_faint)
            self._k_stay.value_label.setStyleSheet(
                f"font-size: 18px; font-weight: 700; color: {color};")

    def _preview_cue(self, attr: str) -> None:
        """Releasing a volume slider plays that cue once at the new
        volume - tuning by ear, not by number."""
        vol = int(getattr(self._settings.shot_clock, attr, 100))
        if attr == "vol_voice":
            from ..voice import say
            say("Ten", volume=vol)
            return
        from ..sounds import play
        play(attr.replace("vol_", ""), volume=vol)

    def _sync_clock_btn_text(self) -> None:
        on = self._settings.shot_clock.enabled
        self._clock_on_btn.setText("Clock ON" if on else "Clock off")

    def _on_clock_enabled(self, on: bool) -> None:
        self._settings.shot_clock.enabled = bool(on)
        self._sync_clock_btn_text()
        self.tuning_changed.emit()          # debounced persist to disk
        self.clock_enabled_toggled.emit(bool(on))
        if not on and self._clock_pause_btn.isChecked():
            self._clock_pause_btn.setChecked(False)

    def _on_clock_pause(self, paused: bool) -> None:
        self._clock_pause_btn.setText("Resume" if paused else "Pause")
        self.clock_pause_toggled.emit(bool(paused))

    def set_clock_enabled_ui(self, on: bool) -> None:
        """Menu-driven sync: reflect without re-emitting."""
        self._clock_on_btn.blockSignals(True)
        self._clock_on_btn.setChecked(on)
        self._clock_on_btn.blockSignals(False)
        self._sync_clock_btn_text()

    # ------------------------------------------------------------------ #
    # Cue-stroke card (Bluetooth IMU on the cue butt)
    # ------------------------------------------------------------------ #
    # (key, label, formatter). Labels/formats mirror the validated reference
    # analyzer's stats board; 'impact' is peak_g (known at the strike), the
    # rest arrive ~2.6 s later once the follow-through has streamed in.
    _STROKE_TILES = (
        ("impact", "IMPACT", None),
        ("v_impact", "CUE SPEED", lambda v: f"{v:.2f} m/s"),
        ("stroke_len", "DRAW", lambda v: f"{v * 100:.0f} cm"),
        ("yaw_swing", "STEER", None),
        ("pause", "PAUSE", lambda v: f"{v:.2f} s"),
        ("steer_ratio", "CONTACT", lambda v: f"{max(0.0, (1 - min(v, 1.0)) * 100):.0f}%"),
        ("finish", "FINISH", lambda v: f"{min(v, 9.9):.1f} s"),
        ("interval", "SINCE LAST", lambda v: f"{v:.0f} s"),
    )

    def _stroke_section(self) -> QWidget:
        self._stroke_card = QWidget()
        col = QVBoxLayout(self._stroke_card)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(6)
        col.addWidget(self._hsep())
        head = QHBoxLayout()
        cap = QLabel("CUE STROKE")
        cap.setObjectName("StatLabel")
        head.addWidget(cap)
        head.addStretch(1)
        self._cue_badge = Badge("OFF", PALETTE.text_dim)
        head.addWidget(self._cue_badge)
        col.addLayout(head)
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(6)
        self._stroke_vals: dict[str, QLabel] = {}
        for i, (key, label, _fmt) in enumerate(self._STROKE_TILES):
            cell = QVBoxLayout()
            cell.setContentsMargins(0, 0, 0, 0)
            cell.setSpacing(0)
            lab = QLabel(label)
            lab.setStyleSheet(
                f"font-size: 9px; font-weight: 700; color: {PALETTE.text_faint};")
            val = QLabel("—")
            val.setStyleSheet("font-size: 15px; font-weight: 800;")
            cell.addWidget(lab)
            cell.addWidget(val)
            holder = QWidget()
            holder.setLayout(cell)
            grid.addWidget(holder, i // 2, i % 2)
            self._stroke_vals[key] = val
        col.addLayout(grid)
        self._stroke_card.setVisible(self._settings.cue.enabled)
        return self._stroke_card

    def set_cue_enabled(self, on: bool) -> None:
        self._stroke_card.setVisible(on)
        if hasattr(self, "_stroke_fold"):
            self._stroke_fold.setVisible(on)

    def on_cue_status(self, state: str, detail) -> None:
        text, color = {
            "connected": ("LIVE", PALETTE.success),
            "scanning": ("SCANNING…", PALETTE.info),
            "connecting": ("CONNECTING…", PALETTE.info),
            "reconnecting": ("RECONNECTING…", PALETTE.warn),
            "no_sensor": ("NO SENSOR", PALETTE.warn),
            "bluetooth_off": ("BLUETOOTH OFF", PALETTE.danger),
            "unavailable": ("UNAVAILABLE", PALETTE.text_dim),
            "error": ("RETRYING…", PALETTE.warn),
            "disabled": ("OFF", PALETTE.text_dim),
        }.get(state, (state.upper(), PALETTE.text_dim))
        if state == "connected" and isinstance(detail, dict) \
                and detail.get("battery") is not None:
            text = f"LIVE · {detail['battery']}%"
        self._cue_badge.set_text_color(text, color)
        self._stroke_card.setVisible(self._settings.cue.enabled)
        if hasattr(self, "_stroke_fold"):
            self._stroke_fold.setVisible(self._settings.cue.enabled)

    def on_cue_impact(self, stroke: dict) -> None:
        """The strike itself — show peak g now, '…' while kinematics compute."""
        self._stroke_vals["impact"].setText(f"{stroke.get('peak_g', 0.0):.1f} g")
        for key, _label, _fmt in self._STROKE_TILES[1:]:
            self._stroke_vals[key].setText("…")

    def on_cue_metrics(self, m: dict) -> None:
        for key, _label, fmt in self._STROKE_TILES:
            if key == "impact":
                self._stroke_vals[key].setText(f"{m.get('peak_g', 0.0):.1f} g")
                continue
            if key == "yaw_swing":
                v = m.get("yaw_swing")
                if v is None:
                    txt = "—"
                elif abs(v) < 0.05:
                    txt = "0.0°"
                else:
                    # +yaw ≈ tip toward the aim-frame's lat axis ("L") —
                    # provisional sign, same convention as the reference app
                    txt = f"{'L' if v > 0 else 'R'} {abs(v):.1f}°"
                self._stroke_vals[key].setText(txt)
                continue
            v = m.get(key)
            self._stroke_vals[key].setText("—" if v is None else fmt(v))

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

    # ------------------------------------------------------------------ #
    # Training Mode — label/correct ball numbers on the playback video
    # ------------------------------------------------------------------ #
    def _training_rail(self) -> QWidget:
        rail = Card(padding=10, spacing=8)
        rail.setMinimumWidth(320)
        rail.setMaximumWidth(440)
        cap = QLabel("TRAINING MODE")
        cap.setObjectName("StatLabel")
        rail.add(cap)

        hint = QLabel("Scrub/pause to a clear frame — each ball shows the model's "
                      "current guess (C = cue, ? = unsure). Click any that are WRONG "
                      "and tap the correct number; click an empty spot to ADD a missed "
                      "ball. Save good frames, then Train.")
        hint.setObjectName("Faint")
        hint.setWordWrap(True)
        rail.add(hint)
        self._label_status = QLabel("Each ball shows the model's guess — click any "
                                    "that are wrong to fix them.")
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

        # Quick jump to a fresh frame to label (video sources only). Saves
        # scrubbing the nav bar between saved frames.
        self._jump_section = QWidget()
        jcol = QVBoxLayout(self._jump_section)
        jcol.setContentsMargins(0, 4, 0, 0)
        jcol.setSpacing(6)
        jlab = QLabel("Jump to a new frame to label")
        jlab.setObjectName("Faint")
        jcol.addWidget(jlab)
        jrow = QHBoxLayout()
        jrow.setSpacing(6)
        for txt, fn in (("Random", lambda _=False: self._jump_random()),
                        ("+30s", lambda _=False: self._jump_ahead(30)),
                        ("+1m", lambda _=False: self._jump_ahead(60)),
                        ("+5m", lambda _=False: self._jump_ahead(300))):
            jb = QPushButton(txt)
            jb.setObjectName("Ghost")
            jb.setCursor(Qt.PointingHandCursor)
            jb.clicked.connect(fn)
            jrow.addWidget(jb)
        jcol.addLayout(jrow)
        rail.add(self._jump_section)

        rail.layout().addStretch(1)

        # Bottom action cluster (Joe's spec): Save · Auto-Label with AI · Train.
        rail.add(self._hsep())
        row = QHBoxLayout()
        row.setSpacing(6)
        save_b = QPushButton("＋ Save")
        save_b.setObjectName("Accent")
        save_b.setCursor(Qt.PointingHandCursor)
        save_b.setToolTip("Add this frame's labels to the training data")
        save_b.clicked.connect(self._save_label_frame)
        row.addWidget(save_b)
        self._train_ai_btn = QPushButton("✨ Train with AI")
        self._train_ai_btn.setCursor(Qt.PointingHandCursor)
        self._train_ai_btn.setToolTip("Auto-label the recorded session (if AI is "
                                      "configured), fine-tune on everything saved, "
                                      "and switch to the new model automatically")
        self._train_ai_btn.clicked.connect(self.train_balls_requested.emit)
        row.addWidget(self._train_ai_btn)
        rw = QWidget()
        rw.setLayout(row)
        rail.add(rw)
        from PySide6.QtWidgets import QProgressBar
        self._train_progress = QProgressBar()
        self._train_progress.setRange(0, 100)
        self._train_progress.setValue(0)
        self._train_progress.setTextVisible(True)
        self._train_progress.setVisible(False)
        rail.add(self._train_progress)
        self._train_log = QLabel("")
        self._train_log.setObjectName("Faint")
        self._train_log.setWordWrap(True)
        rail.add(self._train_log)
        self._autolabel_status = self._train_log   # shared status line
        return rail

    def on_train_progress(self, pct: int, line: str) -> None:
        """Live training progress: bar + latest log line; -1 pct = indeterminate."""
        self._train_progress.setVisible(True)
        if pct < 0:
            self._train_progress.setRange(0, 0)      # busy animation
        else:
            self._train_progress.setRange(0, 100)
            self._train_progress.setValue(pct)
        if line:
            self._train_log.setText(line)

    def on_train_done(self, ok: bool, msg: str) -> None:
        self._train_progress.setVisible(False)
        self._train_progress.setRange(0, 100)
        self._train_log.setText(msg)
        self._train_ai_btn.setEnabled(True)

    def set_autolabel_status(self, text: str, busy: bool = False) -> None:
        self._train_log.setText(text)
        self._train_ai_btn.setEnabled(not busy)
        if busy:
            self.on_train_progress(-1, text)

    def set_training(self, on: bool) -> None:
        """Enter/leave label mode ('Fix labels'): swap the right rail to the
        number pad and make the camera view clickable for labelling."""
        self._training = on
        # leaving training returns to the mode's home rail: shots in
        # playback, stats while live
        self._rail_stack.setCurrentIndex(
            1 if on else (2 if self._is_video else 0))
        if hasattr(self, "_fix_labels_btn") and self._fix_labels_btn.isChecked() != on:
            self._fix_labels_btn.blockSignals(True)
            self._fix_labels_btn.setChecked(on)
            self._fix_labels_btn.blockSignals(False)
        self._persp.set_pickable(on)
        if not on:
            self._persp.set_overlay([])   # clear labelling markers
        self.label_mode_toggled.emit(on)
        if on:
            # Mode pill is fixed-width: keep it to the MODE, and put the
            # instruction in the alert pill where there is room for it.
            self._status_badge.set_text_color("TRAINING", PALETTE.info)
            self._alert_badge.set_text_color("LABEL THE BALLS", PALETTE.info)
            # Label the frame already on screen (e.g. a paused video) right away,
            # showing the model's current guess on each ball — don't wait for a
            # next frame. The controller also re-emits a raw frame momentarily.
            if self._last_packet is not None and getattr(self._last_packet,
                                                          "perspective", None) is not None:
                self._ingest_label_frame(self._last_packet)

    def _set_overlay(self, attr: str, on: bool) -> None:
        setattr(self._settings.ui, attr, bool(on))
        try:
            self._settings.save()
        except Exception:  # noqa: BLE001 - persistence is best-effort
            pass

    def on_overlays_loaded(self, doc) -> None:
        """Playback overlay geometry (shots.json) for the open video —
        the same summary the phone reads (compute-once, never diverges)."""
        self._overlay_doc = doc

    def _feed_analysis_overlays(self, packet) -> None:
        doc = getattr(self, "_overlay_doc", None)
        t = getattr(packet, "media_t", -1.0)
        ui = self._settings.ui
        want_aim = bool(getattr(ui, "overlay_aim", False))
        want_paths = bool(getattr(ui, "overlay_paths", False))
        want_why = bool(getattr(ui, "overlay_why", False))
        if doc is None or t < 0 or not (want_aim or want_paths or want_why):
            self._persp.set_analysis(None, None, -1.0, None)
            return
        shot = None
        for sh in doc.get("shots", []):
            if sh.get("start", 0) - 5.0 <= t <= sh.get("end", 0) + 1.5:
                shot = sh
                break
        if shot is None:
            self._persp.set_analysis(None, None, -1.0, None)
            return
        aim = (shot.get("aim") or {}).get("p") if want_aim else None
        trails = shot.get("trails") if want_paths else None
        tags = None
        if getattr(ui, "overlay_why", False):
            tags = dict(shot.get("tags") or {})
            if shot.get("lines"):
                tags["lines"] = shot["lines"]
            tags = tags or None
        self._persp.set_analysis(aim, trails, t, tags)

    def _ingest_label_frame(self, packet) -> None:
        try:
            import threading
            h, w = packet.perspective.shape[:2]
            self._frame_wh = (w, h)
            balls = []
            for d in (packet.raw_dets or []):
                x, y, r = float(d.x), float(d.y), float(d.radius)
                if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(r)):
                    continue
                balls.append([int(getattr(d, "number", -1)), x, y, r])
            self._label_balls = balls
            self._label_sel = -1
            log.info("training: ingest frame %dx%d, %d balls (thread=%s)", w, h,
                     len(balls), threading.current_thread().name)
            self._persp.set_frame(packet.perspective)   # raw frame; overlay is Qt-drawn
            self._refresh_overlay()
            self._update_label_buttons()
        except Exception:  # noqa: BLE001 - a bad frame must never crash the app
            log.exception("training: ingest frame failed")

    def _default_label_r(self) -> float:
        rs = [b[3] for b in self._label_balls if b[3] > 0]
        return float(np.median(rs)) if rs else max(8.0, self._frame_wh[0] * 0.012)

    def _on_label_click(self, xf: float, yf: float) -> None:
        w, h = self._frame_wh
        if w <= 1 or h <= 1:   # size not yet known — recover it from the view
            sz = self._persp.image_size()
            if sz is None:
                return         # no frame loaded; don't drop a phantom ball at (0,0)
            w, h = sz
            self._frame_wh = (w, h)
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
        self._refresh_overlay()

    def _assign_label(self, num: int) -> None:
        log.info("training: assign %d to ball %d/%d", num, self._label_sel,
                 len(self._label_balls))
        if 0 <= self._label_sel < len(self._label_balls):
            if num < 0:
                self._label_balls.pop(self._label_sel)   # 'not a ball' -> remove
            else:
                self._label_balls[self._label_sel][0] = num
            self._label_sel = -1
            self._update_label_buttons()
            self._refresh_overlay()

    def _update_label_buttons(self) -> None:
        on = 0 <= self._label_sel < len(self._label_balls)
        for b in self._label_btns.values():
            if b.isEnabled() != on:   # avoid redundant state churn / restyle
                b.setEnabled(on)

    def _refresh_overlay(self) -> None:
        """Push the labelling markers to the view as Qt overlay items — drawn with
        QPainter on the UI thread, NOT OpenCV (concurrent cv2 across the UI + worker
        threads can crash natively)."""
        try:
            items = []
            for i, (num, x, y, r) in enumerate(self._label_balls):
                text = "C" if num == 0 else (str(num) if num > 0 else "?")
                items.append((x, y, r, text, i == self._label_sel))
            self._persp.set_overlay(items)
        except Exception:  # noqa: BLE001 - overlay must never crash the app
            log.exception("training: overlay refresh failed")

    def _save_label_frame(self) -> None:
        w, h = self._frame_wh
        boxes = [(num, x / w, y / h, 2 * r / w, 2 * r / h)
                 for (num, x, y, r) in self._label_balls if num >= 0]
        if not boxes:
            self._label_status.setText("Label at least one ball before saving.")
            return
        log.info("training: emit save_training_frame_requested (%d boxes)", len(boxes))
        self.save_training_frame_requested.emit(boxes)
        self._label_status.setText(f"Saved {len(boxes)} balls. Scrub to another frame and keep going.")

    def set_training_count(self, text: str) -> None:
        if hasattr(self, "_label_count"):
            self._label_count.setText(text)

    # --- Training Mode: jump to a fresh frame to label ---------------------- #
    def _jump_to(self, target: int) -> None:
        total = self._seek.maximum() + 1
        if total <= 1:
            return
        target = max(0, min(int(target), total - 1))
        self._seek.blockSignals(True)        # set the thumb without re-seeking
        self._seek.setValue(target)
        self._seek.blockSignals(False)
        self._time_lbl.setText(f"{self._fmt_t(target)} / {self._fmt_t(total)}")
        self.video_seek.emit(target)          # worker re-detects + re-ingests it
        log.info("training: jump to frame %d/%d", target, total)

    def _jump_ahead(self, seconds: float) -> None:
        self._jump_to(self._seek.value() + int(seconds * self._video_fps))

    def _jump_random(self) -> None:
        total = self._seek.maximum() + 1
        if total > 1:
            self._jump_to(random.randint(0, total - 1))

    # ------------------------------------------------------------------ #
    # Intent
    # ------------------------------------------------------------------ #

    def _on_persp_click(self, xf: float, yf: float) -> None:
        if self._training:
            self._on_label_click(xf, yf)

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
        """Track the controller's auto-detection state (toggle lives in Settings)."""
        self._detect_on = on

    @staticmethod
    def _ball_markers(packet) -> list:
        """Hover-reveal markers for the bird's-eye: (x, y, r, number-label) in
        rectified pixels — the same space the schematic balls are drawn in.
        Numbers live here (on hover) instead of on the balls themselves."""
        out = []
        for tr in getattr(packet, "tracks", None) or []:
            n = getattr(tr, "number", -1)
            label = "Cue" if n == 0 else (str(n) if n and n > 0 else "?")
            out.append((float(tr.x), float(tr.y),
                        float(getattr(tr, "radius", 8.0) or 8.0), label))
        return out

    def on_frame(self, packet) -> None:
        import time as _time
        _t0 = _time.perf_counter()
        self._last_packet = packet
        if packet.perspective is not None:
            self._clear_camera_error()  # a frame means the camera is alive
            h, w = packet.perspective.shape[:2]
            self._frame_wh = (w, h)     # keep fresh even outside Training Mode
            if self._training:
                self._ingest_label_frame(packet)   # draw the labelling overlay
            else:
                self._feed_analysis_overlays(packet)
                self._persp.set_frame(packet.perspective)
        if packet.birdseye is not None:
            self._bird.set_frame(packet.birdseye)
            self._bird.set_balls(self._ball_markers(packet))
        # Throttled perf truth: UI-thread cost per frame + delivered rate.
        # This is what settles "is it the worker or the paint" from a log.
        self._ui_ms_acc = getattr(self, "_ui_ms_acc", 0.0) + (_time.perf_counter() - _t0) * 1000.0
        self._ui_n = getattr(self, "_ui_n", 0) + 1
        now = _time.perf_counter()
        if now - getattr(self, "_ui_log_t", 0.0) >= 5.0 and self._ui_n:
            log.info("ui frame handler: %.1f ms avg over %d frames (%.1f fps delivered)",
                     self._ui_ms_acc / self._ui_n, self._ui_n,
                     self._ui_n / (now - getattr(self, "_ui_log_t", now - 5.0)))
            self._ui_log_t, self._ui_ms_acc, self._ui_n = now, 0.0, 0
        if packet.clock_enabled:
            self._clock.update_clock(packet.clock_remaining,
                                     max(1.0, self._settings.shot_clock.seconds),
                                     packet.clock_warning, True)
        info = getattr(packet, "feed_info", "")
        if getattr(self._settings.ui, "feed_stats", True) and info and not self._is_video:
            self._feed_chip.setText(info)
            if getattr(packet, "feed_sd", False):
                self._feed_chip.setStyleSheet(
                    "background: rgba(120,80,0,0.75); color: #FFD54F;"
                    "border-radius: 4px; padding: 2px 8px; font-size: 10px;"
                    "font-weight: 700; letter-spacing: 0.5px;")
            else:
                self._feed_chip.setStyleSheet(
                    "background: rgba(0,0,0,0.55); color: #CFD8DC;"
                    "border-radius: 4px; padding: 2px 8px; font-size: 10px;"
                    "font-weight: 600; letter-spacing: 0.5px;")
            self._feed_chip.adjustSize()
            self._feed_chip.move(self._persp.width() - self._feed_chip.width() - 10, 8)
            self._feed_chip.show()
        else:
            self._feed_chip.hide()
        # MODE — what the app is doing. Independent of any condition below, so
        # "LIVE" stays "LIVE" while a shot is in play or the table is relocking.
        if packet.status == "preview":
            self._status_badge.set_text_color("PREVIEW", PALETTE.text_dim)
        elif self._is_video:
            self._status_badge.set_text_color("PLAYBACK", PALETTE.info)
        else:
            # Red dot = ON RECORD, nothing else — a red "LIVE" while not
            # recording tells the universal broadcast lie (Joe's call). Green
            # = camera live, not being kept.
            self._status_badge.set_text_color(
                "LIVE", PALETTE.danger if self._recording_on else PALETTE.success)

        # CONDITION — transient, most severe first, blank when all is well.
        if getattr(packet, "feed_sd", False) and not self._is_video:
            # Camera fell back to 480p (ML forced-1080i lost after a power
            # cycle) — shout it BEFORE a degraded session gets recorded.
            self._alert_badge.set_text_color("DEGRADED FEED — CHECK CAPTURE", PALETTE.warn)
        elif packet.status == "detecting_nolock":
            self._alert_badge.set_text_color("NO TABLE LOCK", PALETTE.warn)
        elif packet.deviated:
            self._alert_badge.set_text_color("RELOCKING TABLE", PALETTE.warn)
        elif packet.status == "calibrating":
            self._alert_badge.set_text_color("FINDING TABLE", PALETTE.info)
        elif packet.shot_state == "moving":
            self._alert_badge.set_text_color("SHOT IN PLAY", PALETTE.accent)
        else:
            self._alert_badge.clear_status()

        # Stay-down timer (Joe's ask): climbs from the moving edge on live
        # feeds; new climbs only while recording (stats belong to a session).
        if not self._is_video:
            self._set_stay(*self._stay.tick(
                packet.shot_state, getattr(packet, "pipeline_t", -1.0),
                counting=self._recording_on))
            from ...game.shot_clock import status_text
            st = status_text(packet.shot_state,
                             getattr(packet, "clock_running", False),
                             getattr(packet, "clock_paused", False),
                             self._settings.shot_clock.enabled)
            if st != getattr(self, "_clock_status_last", None):
                self._clock_status_last = st
                self._clock_status.setText(st)

    def on_stats(self, summary: dict) -> None:
        if not self._stats_active:
            return   # idle: keep the greyed dashes
        self._shots_row.set_value(
            str(summary.get("makes", 0) + summary.get("misses", 0)))
        self._makes_val.setText(str(summary.get("makes", 0)))
        self._misses_val.setText(str(summary.get("misses", 0)))
        self._k_pct.set_value(f"{summary.get('make_pct', 0):.0f}%")
        self._k_streak.set_value(str(summary.get("current_streak", 0)))

    def on_shot(self, event) -> None:
        # Scoreboard numbers update via on_stats. The timeline gets a clip:
        # start_t/end_t are pipeline media seconds, which match the seek bar
        # in playback; live clips accumulate on the same clock.
        try:
            self._stay.on_shot(float(event.start_t))   # better strike estimate
            self._timeline.add_shot(event.start_t, event.end_t,
                                    event.outcome.value, event.num_pocketed)
            self._shot_list.add_shot({
                "start": event.start_t, "end": event.end_t,
                "outcome": event.outcome.value,
                "pocketed": event.num_pocketed})
        except AttributeError:
            pass

    def on_stroke_measured(self, rec: dict) -> None:
        """Live stroke metrics landing ~20-40s after the shot's row appeared
        (Joe: "populate as soon as the shot is complete"). rec carries the
        REBASED start — the same clock the rows were added on."""
        if isinstance(rec, dict):
            # the timer hears abstentions too (measured-or-abstained: an
            # unmeasurable shot shows "—", never a forever-climb)
            self._set_stay(*self._stay.on_stroke(rec))
        if not isinstance(rec, dict) or rec.get("confidence") == "none":
            return
        keys = ("stay_down_s", "popped_early", "back_depth_px", "pause_ms",
                "delivery_ms", "practice_strokes", "confidence")
        stroke = {k: rec[k] for k in keys if k in rec}
        if not stroke:
            return
        start = float(rec.get("start", -1.0))
        try:
            self._timeline.set_shot_stroke(start, stroke)
            self._shot_list.set_shot_stroke(start, stroke)
        except AttributeError:
            pass

    def on_cached_shots(self, shots) -> None:
        """A session opened with an analysis sidecar: the whole timeline
        lands at once — no waiting for re-detection."""
        self._timeline.clear()
        for s in shots or []:
            self._timeline.add_shot(float(s.get("start", 0.0)),
                                    float(s.get("end", 0.0)),
                                    str(s.get("outcome", "miss")),
                                    int(s.get("pocketed", 0)))
        self._shot_list.set_shots(list(shots or []))
        self._load_shot_thumbs([float(s.get("start", 0.0)) for s in shots or []])

    def _load_shot_thumbs(self, starts: list) -> None:
        """Extract per-shot thumbnails off-thread and hand the BGR frames to
        the list on the UI thread (pixmap conversion must stay there)."""
        video = getattr(self, "_media_path", "")
        if not video or not starts:
            return
        import threading

        from PySide6.QtCore import QObject, Signal

        class _Bridge(QObject):
            ready = Signal(dict)     # worker thread -> UI thread, queued by Qt

        self._thumb_bridge = _Bridge(self)
        self._thumb_bridge.ready.connect(self._shot_list.set_thumbnails)

        def work(bridge=self._thumb_bridge):
            from ..widgets.shot_thumbs import extract_thumbs
            try:
                thumbs = extract_thumbs(video, starts)
            except Exception:                      # noqa: BLE001 - decorative path
                return
            if thumbs:
                bridge.ready.emit(thumbs)

        threading.Thread(target=work, daemon=True, name="shot-thumbs").start()

    def _on_outcome_corrected(self, start: float, outcome: str) -> None:
        """A review verdict: persist to the session's sidecar (append-only
        log — survives re-opens) and repaint the lane to match."""
        if getattr(self, "_media_path", ""):
            from ...vision.analysis_cache import append_correction
            try:
                append_correction(self._media_path, start, outcome)
            except OSError:
                log.warning("could not persist correction", exc_info=True)
        self._timeline.clear()
        for s in self._shot_list._shots:
            self._timeline.add_shot(float(s.get("start", 0.0)),
                                    float(s.get("end", 0.0)),
                                    str(s.get("outcome", "miss")),
                                    int(s.get("pocketed", 0)))

    def _on_export_shot(self, shot_no: int, start: float, end: float) -> None:
        """G6: 'export a favourite shot without touching a file manager'.
        Stream copy — instant, exact source quality — then reveal it."""
        if not getattr(self, "_media_path", ""):
            return
        import subprocess

        from ...vision.analysis_cache import clip_export_cmd
        cmd, dest = clip_export_cmd(
            self._media_path, start, end,
            pre_roll_s=getattr(self._settings.ui, "pre_shot_s", 5.0),
            shot_no=shot_no)
        try:
            creation = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=60, creationflags=creation)
            if r.returncode == 0 and dest.exists():
                subprocess.Popen(["explorer", "/select,", str(dest)],
                                 creationflags=creation)
            else:
                log.warning("clip export failed: %s", (r.stderr or "")[-200:])
        except (OSError, subprocess.TimeoutExpired) as exc:
            log.warning("clip export failed: %s", exc)

    def _on_export_dossier(self, shot_no: int) -> None:
        """North-star surface: one right-click turns a shot into its dossier
        (facts JSON + trajectory diagram) next to the session's clips, then
        reveals it. Same files the future VLM coach will read."""
        video = getattr(self, "_media_path", "")
        if not video:
            return
        import subprocess
        try:
            import sys as _sys
            from pathlib import Path as _P

            tools = str(_P(__file__).resolve().parents[3].parent / "tools")
            if tools not in _sys.path:
                _sys.path.insert(0, tools)
            from export_shot_dossier import export_shot

            from ...vision.analysis_cache import SidecarReader
            reader = SidecarReader(video)
            if not 1 <= shot_no <= len(reader.shots):
                return
            out_root = _P(video).parent / "dossiers"
            d = export_shot(_P(video), reader, shot_no,
                            reader.shots[shot_no - 1], out_root)
            creation = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            subprocess.Popen(["explorer", str(d)], creationflags=creation)
        except Exception:  # noqa: BLE001 - an export must never hurt playback
            log.exception("dossier export failed")

    def _on_fix_labels_at(self, start: float) -> None:
        """Jump to the shot and open Training Mode there: Joe corrects ball
        labels on the exact frames that were wrong; saves feed the training
        store."""
        if self._is_video and self._video_fps > 0:
            self.video_seek.emit(int(start * self._video_fps))
        self.set_training(True)

    def _on_timeline_clicked(self, seconds: float) -> None:
        if self._is_video and self._video_fps > 0:
            self.video_seek.emit(int(seconds * self._video_fps))

    def on_suggestion(self, event) -> None:
        # confirm-manually mode is retired from the surface; nothing to show.
        pass

    def on_recording(self, on: bool) -> None:
        """Reflect recording state on the transport cluster + run the elapsed clock."""
        self._recording_on = on
        self._update_stats_active()
        self._apply_compact()   # the clock changes the bar's width demand
        self._rec_btn.setIcon(icon("rec", PALETTE.danger if on else "#B0524C", size=26))
        self._rec_btn.setEnabled(not on)
        self._rec_pause_btn.setEnabled(on)
        self._rec_stop_btn.setEnabled(on)
        for w in (self._rec_capsule, self._rec_time):
            w.setProperty("recOn", on)
            w.setProperty("paused", False)
            self._repolish(w)
        timer = getattr(self, "_rec_timer", None)
        if timer is None:
            from PySide6.QtCore import QTimer
            self._rec_timer = timer = QTimer(self)
            timer.setInterval(1000)
            timer.timeout.connect(self._tick_rec_time)
        if on:
            import time
            self._rec_t0 = time.monotonic()
            # Styling is final by now (repolish above) — re-measure so the
            # fixed width matches what THIS style actually renders.
            self._rec_time.setStyleSheet(f"color: {PALETTE.danger};")
            self._log_geometry("record-start")
            # the lane becomes a rolling capture timeline while recording
            self._timeline.clear()
            self._timeline.follow_window_s = 120.0
            self._tick_rec_time()
            timer.start()
        else:
            timer.stop()
            self._rec_pause_btn.setChecked(False)
            self._rec_pause_btn.setIcon(icon("rec-pause", PALETTE.text_dim, size=26))
            self._rec_time.setText("0:00")
            self._rec_time.setStyleSheet(f"color: {PALETTE.text_faint};")
            self._timeline.follow_window_s = 0.0   # back to show-everything

    def _log_geometry(self, tag: str) -> None:
        """One log line of root-child geometry — Joe sees a first-record
        layout jump that offscreen probes cannot reproduce; this captures
        the truth from the real environment when it next happens."""
        try:
            import logging
            lay = self.layout()
            parts = [f"{type(lay.itemAt(k).widget()).__name__}="
                     f"{lay.itemAt(k).widget().height()}"
                     for k in range(lay.count()) if lay.itemAt(k).widget()]
            parts.append(f"tl={self._timeline.height()}")
            parts.append(f"win={self.window().width()}x{self.window().height()}")
            logging.getLogger("ui.geometry").info(
                "%s: %s", tag, " ".join(parts))
        except Exception:  # noqa: BLE001 - diagnostics only
            pass

    def _tick_rec_time(self) -> None:
        import time
        elapsed = time.monotonic() - getattr(self, "_rec_t0", time.monotonic())
        # float seconds: the lane interpolates between these syncs for
        # smooth scrolling (Joe: "smooth continuous scrolling rather than
        # discrete seconds progression")
        self._timeline.set_live_clock(max(0.0, elapsed))
        secs = int(elapsed)
        paused = self._rec_pause_btn.isChecked()
        colour = "#E3B341" if paused else PALETTE.danger
        self._rec_time.setStyleSheet(f"color: {colour};")
        self._rec_time.setText(f"{secs // 60}:{secs % 60:02d}")
        # Ratchet every tick: if THIS string renders wider than the label,
        # the label grows right now. No measurement scheme to trust.
        need = self._rec_time.sizeHint().width()
        if need > self._rec_time.minimumWidth():
            self._rec_time.setMinimumWidth(need)
        if self._rec_time.property("paused") != paused:
            self._rec_time.setProperty("paused", paused)
            self._repolish(self._rec_time)

    def resizeEvent(self, ev):  # noqa: N802 - Qt override
        # Narrow-window mode (Joe's ask: app side-by-side with other windows).
        # Each tier yields something the video needs more than chrome does; it
        # all returns the moment there is room again.
        # Tier thresholds must sit ABOVE the floor the previous tier enforces,
        # or the window jams: it cannot shrink past a minimum whose reduction
        # requires shrinking past it (measured: a 900px tier behind a 924px
        # floor was unreachable by dragging). The stats rail is exempt from
        # hiding — Joe wants makes/misses visible at every width, and with no
        # minimum it compresses instead.
        self._apply_compact()
        super().resizeEvent(ev)

    def on_status(self, status: str) -> None:
        if status == "running":
            self.set_running(True)
        elif status == "stopped":
            self.set_running(False)

    def on_replay_saved(self, path: str) -> None:
        # A passing notification, not a mode — it must not evict "LIVE".
        self._alert_badge.set_text_color("REPLAY SAVED", PALETTE.success)
