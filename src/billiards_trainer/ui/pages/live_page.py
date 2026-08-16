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
from ..widgets.common import Badge, Card, StatCard
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
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(14)
        root.addWidget(self._control_bar())
        from ..widgets.shot_timeline import ShotTimeline
        self._timeline = ShotTimeline(
            pre_roll_s=getattr(self._settings.ui, "pre_shot_s", 5.0))
        self._timeline.clicked.connect(self._on_timeline_clicked)
        root.addWidget(self._timeline)

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
        self._rail_stack.addWidget(self._stats_rail())      # 0 = normal
        self._rail_stack.addWidget(self._training_rail())   # 1 = training
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
        lay.setContentsMargins(14, 8, 14, 8)
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
        self._rec_time = QLabel("")
        self._rec_time.setObjectName("RecClock")
        self._rec_time.setTextFormat(Qt.RichText)
        # FIXED, not minimum: the clock counts up and a minimum width still lets
        # it grow, nudging the whole capsule sideways. The width is MEASURED
        # from the worst-case string rather than guessed — a hardcoded 96px
        # clipped "❚❚ PAUSED 125:30" down to "RE", because QLabel truncates
        # instead of growing once the width is fixed.
        self._rec_time.setFixedWidth(
            self._rec_time.fontMetrics().horizontalAdvance("❚❚ PAUSED   000:00") + 18)
        self._rec_time.setAlignment(Qt.AlignCenter)
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
        """Narrow-window chrome policy, re-applied on resize AND mode change.

        The bar's children are fixed-width, so when narrow something must
        yield — and the honest candidates are the controls that are dead
        weight in the CURRENT mode: playback transport while live, the record
        capsule while in playback. Hiding by context (instead of clipping)
        is what stops the record capsule's clock being cut off mid-recording.
        """
        w = self.width()
        compact = w < 1000
        if hasattr(self, "_bar_optional"):
            for x in self._bar_optional:
                x.setVisible(not compact)
        if hasattr(self, "_playback_widgets"):
            for x in self._playback_widgets:
                x.setVisible(not compact or self._is_video)
        if hasattr(self, "_rec_capsule"):
            self._rec_capsule.setVisible(not compact or not self._is_video)
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
        return btn

    def _stats_rail(self) -> QWidget:
        # Compact scoreboard — tight spacing, narrow column; more stats land
        # here later, so the layout leaves the room in the middle, not the edges.
        rail = Card(padding=12, spacing=8)
        # No hard minimum: the window's minimum width is the SUM of its
        # children's minimums, and this rail's old 240px floor was a third of
        # why the app refused to shrink for side-by-side use. The rail looks
        # best >=240 wide, so narrow-mode HIDES it (resizeEvent) rather than
        # squeezing it.
        rail.setMaximumWidth(300)

        # The headline: makes vs misses, big. Always visible — it IS the rail.
        mm = QHBoxLayout()
        mm.setSpacing(8)
        self._makes_box, self._makes_val = self._big_stat("MAKES", PALETTE.success)
        self._misses_box, self._misses_val = self._big_stat("MISSES", PALETTE.danger)
        mm.addWidget(self._makes_box)
        mm.addWidget(self._misses_box)
        rail.layout().addLayout(mm)

        # Everything below the headline folds (Joe's ask: collapsible sections
        # for narrow view). Collapsed state persists per section.
        from ..widgets.collapsible import CollapsibleSection

        grid_holder = QWidget()
        grid = QHBoxLayout(grid_holder)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(8)
        self._k_pct = StatCard("Make %", "—", accent=True)
        self._k_streak = StatCard("Streak", "—")
        grid.addWidget(self._k_pct)
        grid.addWidget(self._k_streak)
        rail.layout().addWidget(CollapsibleSection(
            "SESSION", grid_holder, "session", self._settings))

        # Cue-stroke stats (Bluetooth IMU) — hidden unless the sensor is enabled.
        self._stroke_holder = self._stroke_section()
        self._stroke_fold = CollapsibleSection(
            "CUE SENSOR", self._stroke_holder, "cue_sensor", self._settings)
        self._stroke_fold.setVisible(self._settings.cue.enabled)
        rail.layout().addWidget(self._stroke_fold)

        # Shot clock — only visible when enabled, so it never crowds sandbox.
        self._clock_row = QHBoxLayout()
        self._clock_row.addStretch(1)
        self._clock = ShotClockWidget()
        self._clock_row.addWidget(self._clock)
        self._clock_row.addStretch(1)
        self._clock_holder = QWidget()
        self._clock_holder.setLayout(self._clock_row)
        self._clock_fold = CollapsibleSection(
            "SHOT CLOCK", self._clock_holder, "shot_clock", self._settings)
        self._clock_fold.setVisible(self._settings.shot_clock.enabled)
        rail.layout().addWidget(self._clock_fold)

        rail.layout().addStretch(1)

        return rail

    def _big_stat(self, label: str, color: str):
        box = Card(padding=14, spacing=2)
        lbl = QLabel(label)
        lbl.setObjectName("StatLabel")
        val = QLabel("—")
        val.setProperty("statColor", color)
        val.setStyleSheet(f"font-size: 40px; font-weight: 800; color: {PALETTE.text_faint};")
        box.add(lbl)
        box.add(val)
        return box, val

    def _update_stats_active(self) -> None:
        """Stats belong to a RECORDING: coloured + counting only while the
        session recorder runs; greyed dashes any other time (Joe's rule)."""
        active = self._recording_on
        if active == self._stats_active:
            return
        self._stats_active = active
        for val in (self._makes_val, self._misses_val):
            color = val.property("statColor") if active else PALETTE.text_faint
            val.setStyleSheet(f"font-size: 40px; font-weight: 800; color: {color};")
        if not active:
            self._makes_val.setText("—")
            self._misses_val.setText("—")
            self._k_pct.set_value("—")
            self._k_streak.set_value("—")

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
        rail = Card(padding=16, spacing=10)
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
        self._rail_stack.setCurrentIndex(1 if on else 0)
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
        self._last_packet = packet
        if packet.perspective is not None:
            self._clear_camera_error()  # a frame means the camera is alive
            h, w = packet.perspective.shape[:2]
            self._frame_wh = (w, h)     # keep fresh even outside Training Mode
            if self._training:
                self._ingest_label_frame(packet)   # draw the labelling overlay
            else:
                self._persp.set_frame(packet.perspective)
        if packet.birdseye is not None:
            self._bird.set_frame(packet.birdseye)
            self._bird.set_balls(self._ball_markers(packet))
        self._clock_fold.setVisible(packet.clock_enabled)
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

    def on_stats(self, summary: dict) -> None:
        if not self._stats_active:
            return   # idle: keep the greyed dashes
        self._makes_val.setText(str(summary.get("makes", 0)))
        self._misses_val.setText(str(summary.get("misses", 0)))
        self._k_pct.set_value(f"{summary.get('make_pct', 0):.0f}%")
        self._k_streak.set_value(str(summary.get("current_streak", 0)))

    def on_shot(self, event) -> None:
        # Scoreboard numbers update via on_stats. The timeline gets a clip:
        # start_t/end_t are pipeline media seconds, which match the seek bar
        # in playback; live clips accumulate on the same clock.
        try:
            self._timeline.add_shot(event.start_t, event.end_t,
                                    event.outcome.value, event.num_pocketed)
        except AttributeError:
            pass

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
            self._tick_rec_time()
            timer.start()
        else:
            timer.stop()
            self._rec_pause_btn.setChecked(False)
            self._rec_pause_btn.setIcon(icon("rec-pause", PALETTE.text_dim, size=26))
            self._rec_time.setText("")

    def _tick_rec_time(self) -> None:
        import time
        secs = int(time.monotonic() - getattr(self, "_rec_t0", time.monotonic()))
        paused = self._rec_pause_btn.isChecked()
        if paused:
            tag = "❚❚ PAUSED"
        else:
            dot = PALETTE.danger if secs % 2 == 0 else "transparent"
            # same glyph every tick, colour alternates -> metrics never change,
            # so the transport row stops shifting on each blink
            tag = f'<span style="color:{dot}">●</span> REC'
        self._rec_time.setText(f"{tag}&nbsp;&nbsp;{secs // 60}:{secs % 60:02d}")
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
