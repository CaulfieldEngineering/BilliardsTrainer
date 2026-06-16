"""Settings page — edits the live ``Settings`` object and persists it.

Grouped into Source, Table & Felt, Ball detection, Shot clock, Appearance, and
Updates. Emits ``applied`` so the main window can re-theme / restart capture.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...config import Settings
from ..widgets.common import Card, section_header


class SettingsPage(QWidget):
    applied = Signal()
    check_updates_requested = Signal()
    feedback_requested = Signal()
    capture_requested = Signal()

    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self._s = settings
        self._loaded = False  # guards auto-save during initial population
        self._build()
        self._load_from_settings()
        self._loaded = True

    # ------------------------------------------------------------------ #
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(16)

        from ..icons import icon
        from ..theme import PALETTE
        header = QHBoxLayout()
        header.addWidget(section_header("Settings"))
        header.addStretch(1)
        # Always-visible "Check for updates" button so it can't be missed,
        # regardless of scroll position or window size.
        self._header_check_btn = QPushButton("  Check for updates")
        self._header_check_btn.setObjectName("Ghost")
        self._header_check_btn.setIcon(icon("download", PALETTE.text))
        self._header_check_btn.setToolTip("Manually check for new versions")
        self._header_check_btn.setCursor(Qt.PointingHandCursor)
        self._header_check_btn.clicked.connect(self._on_check_clicked)
        header.addWidget(self._header_check_btn)
        self._save_btn = QPushButton("Save changes")
        self._save_btn.setObjectName("Accent")
        self._save_btn.setCursor(Qt.PointingHandCursor)
        self._save_btn.clicked.connect(self._save)
        header.addWidget(self._save_btn)
        root.addLayout(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body = QWidget()
        grid = QGridLayout(body)
        grid.setSpacing(16)
        grid.setContentsMargins(0, 0, 8, 0)

        # Updates card sits in the top row so the button is visible without
        # scrolling; the header button covers every other case.
        grid.addWidget(self._source_card(), 0, 0)
        grid.addWidget(self._updates_card(), 0, 1)
        grid.addWidget(self._felt_card(), 1, 0)
        grid.addWidget(self._ball_card(), 1, 1)
        grid.addWidget(self._clock_card(), 2, 0)
        grid.addWidget(self._appearance_card(), 2, 1)
        grid.addWidget(self._detection_card(), 3, 0)
        grid.addWidget(self._feedback_card(), 3, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        scroll.setWidget(body)
        root.addWidget(scroll, 1)

    def _card(self, title: str) -> tuple[Card, QFormLayout]:
        card = Card(padding=18, spacing=12)
        card.add(QLabel(f"<b>{title}</b>"))
        form = QFormLayout()
        form.setSpacing(12)
        form.setLabelAlignment(Qt.AlignLeft)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        holder = QWidget()
        holder.setLayout(form)
        card.add(holder)
        return card, form

    def _source_card(self) -> Card:
        from ..icons import icon
        from ..theme import PALETTE
        card, form = self._card("Camera / source")

        self._source_combo = QComboBox()
        self._source_combo.setMinimumWidth(220)
        refresh = QPushButton()
        refresh.setObjectName("Ghost")
        refresh.setIcon(icon("refresh", PALETTE.text_dim))
        refresh.setToolTip("Re-scan for cameras (e.g. after plugging one in)")
        refresh.setCursor(Qt.PointingHandCursor)
        refresh.clicked.connect(lambda: self._populate_cameras(keep=True))
        crow = QHBoxLayout()
        crow.addWidget(self._source_combo, 1)
        crow.addWidget(refresh)
        cw = QWidget()
        cw.setLayout(crow)
        form.addRow("Camera", cw)

        # The live Sandbox view is the camera preview now (it's always on), so a
        # separate "Test preview" modal is redundant — picking a camera here shows
        # up immediately on the Sandbox tab.
        brow = QHBoxLayout()
        file_btn = QPushButton("Use a file…")
        file_btn.setObjectName("Ghost")
        file_btn.setCursor(Qt.PointingHandCursor)
        file_btn.clicked.connect(self._choose_file)
        brow.addWidget(file_btn)
        brow.addStretch(1)
        bw = QWidget()
        bw.setLayout(brow)
        form.addRow("", bw)

        self._source_hint = QLabel("")
        self._source_hint.setObjectName("Faint")
        self._source_hint.setWordWrap(True)
        form.addRow("", self._source_hint)

        self._mirror = QCheckBox("Mirror preview horizontally")
        form.addRow("", self._mirror)

        # Training-data capture: records ~60 s of the raw camera feed to a zip we
        # can fine-tune YOLO on (the path to real AI detection on Joe's table).
        self._capture_btn = QPushButton("  Capture 60s for analysis")
        self._capture_btn.setObjectName("Ghost")
        self._capture_btn.setCursor(Qt.PointingHandCursor)
        self._capture_btn.setToolTip("Save 60 seconds of raw frames to a zip for "
                                     "AI-detection training")
        self._capture_btn.clicked.connect(self._on_capture_clicked)
        form.addRow("", self._capture_btn)
        self._capture_status = QLabel("")
        self._capture_status.setObjectName("Faint")
        self._capture_status.setWordWrap(True)
        form.addRow("", self._capture_status)

        self._cam_names: dict[str, str] = {}
        self._populate_cameras()
        # Auto-save the camera the moment it's picked — no Save click needed, so
        # the dropdown is actually wired to what the live preview opens.
        self._source_combo.currentIndexChanged.connect(self._on_source_changed)
        return card

    def _on_capture_clicked(self) -> None:
        self._capture_status.setText("Capturing… keep the camera pointed at the table.")
        self.capture_requested.emit()

    def set_capture_status(self, text: str) -> None:
        self._capture_status.setText(text)

    def _on_source_changed(self) -> None:
        if not self._loaded:
            return  # ignore programmatic population during load
        spec = str(self._source_combo.currentData() or "0")
        self._s.source = spec
        self._s.source_name = self._cam_names.get(spec, "") if spec.isdigit() else ""
        self._s.save()
        self.applied.emit()  # persists + pushes the new source to the worker thread
        self._source_hint.setText(f"Using {self._source_combo.currentText()} — saved.")

    def _populate_cameras(self, keep: bool = False) -> None:
        from ...capture.devices import list_cameras
        target = self._source_combo.currentData() if keep else None
        self._source_combo.blockSignals(True)
        self._source_combo.clear()
        self._cam_names = {}
        cams = list_cameras()
        for c in cams:
            self._source_combo.addItem(c.label(), str(c.index))
            self._cam_names[str(c.index)] = c.name
        self._source_combo.addItem("Demo simulation (no camera)", "demo")
        if not cams:
            self._source_hint.setText("No cameras detected. Plug one in and press the "
                                      "refresh icon, or use Demo / a file.")
        else:
            self._source_hint.setText("Pick your camera, then Test preview to confirm.")
        self._source_combo.blockSignals(False)
        if target is not None:
            self._select_source_data(target)

    def _select_source_data(self, spec: str) -> None:
        idx = self._source_combo.findData(spec)
        if idx < 0 and spec not in ("demo",):
            # a file path or an index that's not currently present — add it
            label = spec if not spec.isdigit() else f"Camera {spec} (not connected)"
            self._source_combo.addItem(label, spec)
            idx = self._source_combo.findData(spec)
        if idx >= 0:
            self._source_combo.setCurrentIndex(idx)

    def _choose_file(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose a video or image",
            filter="Media (*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.png);;All files (*)")
        if path:
            self._source_combo.addItem(f"\U0001F4C4 {path.split('/')[-1]}", path)
            self._select_source_data(path)

    def _felt_card(self) -> Card:
        card, form = self._card("Table & felt")
        self._table_size = QComboBox()
        self._table_size.addItems(["9ft", "8ft", "7ft"])
        form.addRow("Table size", self._table_size)

        self._sensitivity = QSlider(Qt.Horizontal)
        self._sensitivity.setRange(0, 100)
        self._sens_label = QLabel("82")
        sens_row = QHBoxLayout()
        sens_row.addWidget(self._sensitivity, 1)
        sens_row.addWidget(self._sens_label)
        self._sensitivity.valueChanged.connect(lambda v: self._sens_label.setText(str(v)))
        sw = QWidget()
        sw.setLayout(sens_row)
        form.addRow("Felt sensitivity", sw)

        self._h_min = self._spin(0, 180)
        self._h_max = self._spin(0, 180)
        hue_row = QHBoxLayout()
        hue_row.addWidget(QLabel("min"))
        hue_row.addWidget(self._h_min)
        hue_row.addWidget(QLabel("max"))
        hue_row.addWidget(self._h_max)
        hw = QWidget()
        hw.setLayout(hue_row)
        form.addRow("Felt hue range", hw)
        self._auto_relock = QCheckBox("Auto re-lock if the table shifts")
        form.addRow("", self._auto_relock)
        self._persist_calib = QCheckBox("Remember calibration between launches")
        form.addRow("", self._persist_calib)
        tip = QLabel("Tip: on the Sandbox tab, click 'Pick felt' then tap the "
                     "cloth to seed these from your real table.")
        tip.setObjectName("Faint")
        tip.setWordWrap(True)
        form.addRow("", tip)
        return card

    def _ball_card(self) -> Card:
        card, form = self._card("Ball detection")
        self._backend = QComboBox()
        self._backend.addItems(["auto", "classical", "yolo"])
        form.addRow("Backend", self._backend)
        note = QLabel("Auto uses YOLO when weights are present, else classical. "
                      "Classical (Hough + colour) runs everywhere with no model. "
                      "YOLO needs the optional [yolo] extra + weights in models/.")
        note.setObjectName("Faint")
        note.setWordWrap(True)
        form.addRow("", note)
        self._param2 = self._spin(5, 60)
        form.addRow("Detector strictness", self._param2)
        self._yolo_url = QLineEdit()
        self._yolo_url.setPlaceholderText("https://…/pool_balls.pt (auto-fetched)")
        form.addRow("YOLO weights URL", self._yolo_url)
        return card

    def _detection_card(self) -> Card:
        card, form = self._card("Detection")
        banner = QLabel("⚠ Shot detection uses on-device CV and is still being "
                        "tuned — it can miscount on a noisy feed. Tune the gates "
                        "below, use Confirm-manually, or drop YOLO weights in the "
                        "models folder + set Backend = yolo for a big accuracy jump.")
        banner.setObjectName("Faint")
        banner.setWordWrap(True)
        form.addRow("", banner)

        self._preset = QComboBox()
        self._preset.addItems(["conservative", "balanced", "aggressive"])
        self._preset.activated.connect(self._on_preset_changed)
        form.addRow("Preset", self._preset)
        self._fusion = QCheckBox("Multi-modal evidence fusion (bg-subtraction + optical flow)")
        form.addRow("", self._fusion)

        self._manual_confirm = QCheckBox("Confirm shots manually (auto-detect only suggests)")
        form.addRow("", self._manual_confirm)
        self._require_cue = QCheckBox("Require a cue ball to count a shot")
        form.addRow("", self._require_cue)

        self._motion_active = QDoubleSpinBox()
        self._motion_active.setRange(0.05, 5.0)
        self._motion_active.setSingleStep(0.05)
        form.addRow("Motion sensitivity", self._motion_active)
        self._min_travel = self._spin(20, 600)
        form.addRow("Min ball travel (px)", self._min_travel)
        self._warmup = self._spin(0, 30)
        form.addRow("Warm-up (s)", self._warmup)
        self._cooldown = self._spin(0, 30)
        form.addRow("Cool-down between shots (s)", self._cooldown)
        self._pocket_frames = self._spin(2, 60)
        form.addRow("Pocket dwell (frames)", self._pocket_frames)

        self._debug_overlay = QCheckBox("Show debug overlay (raw blobs + shot state)")
        form.addRow("", self._debug_overlay)
        self._schematic = QCheckBox("Clean schematic overhead view (vs warped camera)")
        form.addRow("", self._schematic)
        return card

    def _on_preset_changed(self, _index: int) -> None:
        from ...config import apply_detection_preset
        apply_detection_preset(self._s.detection, self._preset.currentText())
        # reflect the preset's gate values in the widgets
        d = self._s.detection
        self._require_cue.setChecked(d.require_cue)
        self._motion_active.setValue(d.motion_active)
        self._min_travel.setValue(int(d.min_travel_px))
        self._warmup.setValue(int(d.warmup_seconds))
        self._cooldown.setValue(int(d.cooldown_seconds))
        self._pocket_frames.setValue(d.pocket_frames)

    def _clock_card(self) -> Card:
        card, form = self._card("Shot clock")
        self._clock_enabled = QCheckBox("Enable shot clock")
        form.addRow("", self._clock_enabled)
        self._clock_seconds = self._spin(5, 120)
        form.addRow("Duration (s)", self._clock_seconds)
        self._clock_warn = self._spin(0, 60)
        form.addRow("Warn at (s left)", self._clock_warn)
        self._clock_audio = QCheckBox("Audio cue")
        form.addRow("", self._clock_audio)
        self._clock_autoreset = QCheckBox("Auto-reset on detected shot")
        form.addRow("", self._clock_autoreset)
        return card

    def _appearance_card(self) -> Card:
        card, form = self._card("Appearance")
        self._accent = QLineEdit()
        self._accent.setPlaceholderText("#3DDC97")
        form.addRow("Accent colour", self._accent)
        self._show_overlays = QCheckBox("Show detection overlays")
        form.addRow("", self._show_overlays)
        self._show_traj = QCheckBox("Show ball trajectories")
        form.addRow("", self._show_traj)
        self._show_ids = QCheckBox("Show ball IDs")
        form.addRow("", self._show_ids)
        note = QLabel("Accent changes apply after Save.")
        note.setObjectName("Faint")
        form.addRow("", note)
        return card

    def _updates_card(self) -> Card:
        card, form = self._card("Updates")
        from ...version import __version__
        ver = QLabel(f"Installed version: <b>{__version__}</b>")
        ver.setObjectName("Muted")
        form.addRow("", ver)
        self._auto_check = QCheckBox("Check for updates on launch")
        form.addRow("", self._auto_check)

        self._check_btn = QPushButton("  Check for updates now")
        self._check_btn.setObjectName("Accent")
        self._check_btn.setToolTip("Manually check for new versions")
        self._check_btn.setCursor(Qt.PointingHandCursor)
        self._check_btn.clicked.connect(self._on_check_clicked)
        form.addRow("", self._check_btn)
        self._update_status = QLabel("")
        self._update_status.setObjectName("Faint")
        self._update_status.setWordWrap(True)
        form.addRow("", self._update_status)
        av = QLabel("Updates are checksum-verified and roll back automatically if "
                    "they don't start. If antivirus interferes, add the install "
                    "folder to its exclusions (see the README).")
        av.setObjectName("Faint")
        av.setWordWrap(True)
        form.addRow("", av)
        return card

    def _feedback_card(self) -> Card:
        card, form = self._card("Feedback & backup")
        msg = QLabel("Found a bug or want a feature? Send it from here — it's saved "
                     "locally and backed up if cloud sync is set up.")
        msg.setObjectName("Faint")
        msg.setWordWrap(True)
        form.addRow("", msg)
        send = QPushButton("  Send feedback")
        send.setObjectName("Accent")
        send.setCursor(Qt.PointingHandCursor)
        send.clicked.connect(self.feedback_requested.emit)
        form.addRow("", send)
        from ...sync import sync_status
        self._sync_status = QLabel(f"Cloud sync: <b>{sync_status()}</b>")
        self._sync_status.setObjectName("Muted")
        form.addRow("", self._sync_status)
        hint = QLabel("To enable cloud backup, see docs/SUPABASE.md and drop "
                      "credentials into supabase.json in the app data folder.")
        hint.setObjectName("Faint")
        hint.setWordWrap(True)
        form.addRow("", hint)
        return card

    def _on_check_clicked(self) -> None:
        self._update_status.setText("Checking…")
        self._check_btn.setEnabled(False)
        self._header_check_btn.setEnabled(False)
        self.check_updates_requested.emit()

    def set_update_status(self, text: str) -> None:
        self._update_status.setText(text)
        self._check_btn.setEnabled(True)
        self._header_check_btn.setEnabled(True)

    @staticmethod
    def _spin(lo: int, hi: int) -> QSpinBox:
        s = QSpinBox()
        s.setRange(lo, hi)
        return s

    # ------------------------------------------------------------------ #
    def reload(self) -> None:
        """Refresh the controls from the (possibly externally-mutated) settings."""
        self._load_from_settings()

    def _select_source(self, spec: str, name: str) -> None:
        # Prefer matching the saved friendly name (survives index reshuffles).
        if spec.isdigit() and name:
            for data, cam_name in self._cam_names.items():
                if cam_name == name:
                    self._select_source_data(data)
                    return
        self._select_source_data(spec)

    def _load_from_settings(self) -> None:
        s = self._s
        self._select_source(s.source, s.source_name)
        self._mirror.setChecked(s.ui.mirror_preview)
        self._table_size.setCurrentText(s.table.size)
        self._sensitivity.setValue(s.felt.sensitivity)
        self._sens_label.setText(str(s.felt.sensitivity))
        self._h_min.setValue(s.felt.h_min)
        self._h_max.setValue(s.felt.h_max)
        self._auto_relock.setChecked(s.table.auto_relock)
        self._persist_calib.setChecked(s.table.persist_calibration)
        self._backend.setCurrentText(s.balls.backend)
        self._param2.setValue(s.balls.detect_param2)
        self._yolo_url.setText(s.balls.yolo_weights_url)
        self._show_overlays.setChecked(s.ui.show_overlays)
        self._preset.setCurrentText(s.detection.preset)
        self._fusion.setChecked(s.detection.use_fusion)
        self._manual_confirm.setChecked(s.detection.manual_confirm)
        self._require_cue.setChecked(s.detection.require_cue)
        self._motion_active.setValue(s.detection.motion_active)
        self._min_travel.setValue(int(s.detection.min_travel_px))
        self._warmup.setValue(int(s.detection.warmup_seconds))
        self._cooldown.setValue(int(s.detection.cooldown_seconds))
        self._pocket_frames.setValue(s.detection.pocket_frames)
        self._debug_overlay.setChecked(s.ui.debug_overlay)
        self._schematic.setChecked(s.ui.schematic_birdseye)
        self._clock_enabled.setChecked(s.shot_clock.enabled)
        self._clock_seconds.setValue(s.shot_clock.seconds)
        self._clock_warn.setValue(s.shot_clock.warn_seconds)
        self._clock_audio.setChecked(s.shot_clock.audio)
        self._clock_autoreset.setChecked(s.shot_clock.auto_reset_on_shot)
        self._accent.setText(s.ui.accent)
        self._show_traj.setChecked(s.ui.show_trajectories)
        self._show_ids.setChecked(s.ui.show_ball_ids)
        self._auto_check.setChecked(s.updates.auto_check)

    def _save(self) -> None:
        s = self._s
        spec = str(self._source_combo.currentData() or "0")
        s.source = spec
        s.source_name = self._cam_names.get(spec, "") if spec.isdigit() else ""
        s.ui.mirror_preview = self._mirror.isChecked()
        s.table.size = self._table_size.currentText()
        s.felt.sensitivity = self._sensitivity.value()
        s.felt.h_min = self._h_min.value()
        s.felt.h_max = self._h_max.value()
        s.table.auto_relock = self._auto_relock.isChecked()
        s.table.persist_calibration = self._persist_calib.isChecked()
        s.balls.backend = self._backend.currentText()
        s.balls.detect_param2 = self._param2.value()
        s.balls.yolo_weights_url = self._yolo_url.text().strip()
        s.detection.preset = self._preset.currentText()
        s.detection.use_fusion = self._fusion.isChecked()
        s.detection.manual_confirm = self._manual_confirm.isChecked()
        s.detection.require_cue = self._require_cue.isChecked()
        s.detection.motion_active = self._motion_active.value()
        s.detection.min_travel_px = float(self._min_travel.value())
        s.detection.warmup_seconds = float(self._warmup.value())
        s.detection.cooldown_seconds = float(self._cooldown.value())
        s.detection.pocket_frames = self._pocket_frames.value()
        s.ui.debug_overlay = self._debug_overlay.isChecked()
        s.ui.schematic_birdseye = self._schematic.isChecked()
        s.shot_clock.enabled = self._clock_enabled.isChecked()
        s.shot_clock.seconds = self._clock_seconds.value()
        s.shot_clock.warn_seconds = self._clock_warn.value()
        s.shot_clock.audio = self._clock_audio.isChecked()
        s.shot_clock.auto_reset_on_shot = self._clock_autoreset.isChecked()
        accent = self._accent.text().strip()
        if accent:
            s.ui.accent = accent
        s.ui.show_overlays = self._show_overlays.isChecked()
        s.ui.show_trajectories = self._show_traj.isChecked()
        s.ui.show_ball_ids = self._show_ids.isChecked()
        s.updates.auto_check = self._auto_check.isChecked()
        s.save()
        self.applied.emit()
