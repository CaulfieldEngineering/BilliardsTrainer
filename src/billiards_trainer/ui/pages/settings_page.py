"""Settings page — edits the live ``Settings`` object and persists it.

Grouped into Source, Table & Felt, Ball detection, Shot clock, Appearance, and
Updates. Emits ``applied`` so the main window can re-theme / restart capture.
"""

from PySide6.QtCore import QObject, Qt, QThread, Signal
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
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...config import Settings
from ..widgets.common import Card, section_header


class _ModelDownloadWorker(QObject):
    """Downloads a fetchable detector model off the UI thread."""
    progress = Signal(float)
    done = Signal(str)       # strategy name now available
    failed = Signal(str)

    def __init__(self, key: str):
        super().__init__()
        self._key = key

    def run(self) -> None:
        try:
            from ...detector_strategies import model_fetch
            model_fetch.fetch_model(self._key, progress=self.progress.emit)
            self.done.emit(model_fetch.FETCHABLE[self._key]["strategy"])
        except Exception as exc:  # noqa: BLE001 - surface any failure to the user
            self.failed.emit(str(exc))


class SettingsPage(QWidget):
    applied = Signal()
    check_updates_requested = Signal()
    feedback_requested = Signal()
    flag_failure_requested = Signal()
    detector_changed = Signal(str)   # live detector strategy switched
    recalibrate_requested = Signal()  # re-detect the table now
    train_balls_requested = Signal()  # open the in-app Ball ID Trainer
    cue_diagnostics_requested = Signal()  # open the cue-sensor waveform dialog

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

        # User-facing settings only. Live tuning (detector sensitivity, overlay
        # toggles) lives on the Sandbox panel; shot-detection gates are deferred
        # until cue-ball tracking is locked; dev/internal config (weights URL,
        # backend, detector strictness, felt HSV) is no longer surfaced.
        # One card per row (Joe's preference: a plain vertical list, no columns);
        # capped width so form rows don't stretch across the whole window.
        sections = [
            ("Camera & audio", self._source_card()),
            ("Camera controls", self._camera_card()),
            ("Table", self._table_card()),
            ("AI detection", self._model_card()),
            ("Recording", self._recording_card()),
            ("Shot clock", self._clock_card()),
            ("Appearance", self._appearance_card()),
            ("Cue sensor", self._cue_card()),
            ("AI labelling", self._ai_card()),
            ("Updates", self._updates_card()),
            ("Feedback", self._feedback_card()),
            ("Debug", self._debug_card()),
        ]
        self._section_cards = []
        for row, (_title, card) in enumerate(sections):
            card.setMaximumWidth(820)
            grid.addWidget(card, row, 0)
            self._section_cards.append(card)
        grid.setColumnStretch(0, 1)

        scroll.setWidget(body)
        self._scroll = scroll

        # Section nav: a slim list on the left that jumps the scroll to a card.
        from PySide6.QtWidgets import QListWidget
        nav = QListWidget()
        nav.setFixedWidth(160)
        nav.setFrameShape(QScrollArea.NoFrame)
        for title, _card in sections:
            nav.addItem(title)
        nav.currentRowChanged.connect(self._scroll_to_section)

        split = QHBoxLayout()
        split.setSpacing(16)
        split.addWidget(nav)
        split.addWidget(scroll, 1)
        root.addLayout(split, 1)

    def _scroll_to_section(self, row: int) -> None:
        if 0 <= row < len(self._section_cards):
            card = self._section_cards[row]
            self._scroll.verticalScrollBar().setValue(max(0, card.y() - 8))

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
        card, form = self._card("Camera & audio")

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

        self._source_hint = QLabel("")
        self._source_hint.setObjectName("Faint")
        self._source_hint.setWordWrap(True)
        form.addRow("", self._source_hint)

        self._mirror = QCheckBox("Mirror preview horizontally")
        form.addRow("", self._mirror)

        # Audio input lives with the camera: one place for capture sources.
        self._rec_device = QComboBox()
        self._rec_device.addItem("System default input", "default")
        try:
            from PySide6.QtMultimedia import QMediaDevices
            for dev in QMediaDevices.audioInputs():
                name = dev.description()
                if name:
                    self._rec_device.addItem(name, name)
        except Exception:  # noqa: BLE001 - multimedia module absent
            pass
        self._rec_device.currentIndexChanged.connect(self._on_recording_changed)
        form.addRow("Microphone", self._rec_device)

        mic_hint = QLabel("The camera's HDMI feed carries no live audio (Canon "
                          "T3i limitation) — plug any USB mic into the Mac and "
                          "pick it here to get sound with your sessions.")
        mic_hint.setObjectName("Faint")
        mic_hint.setWordWrap(True)
        form.addRow("", mic_hint)

        self._cam_names: dict[str, str] = {}
        self._populate_cameras()
        # Auto-save the camera the moment it's picked — no Save click needed, so
        # the dropdown is actually wired to what the live preview opens.
        self._source_combo.currentIndexChanged.connect(self._on_source_changed)
        return card

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
            self._source_combo.addItem(c.name, str(c.index))
            self._cam_names[str(c.index)] = c.name
        if not cams:
            self._source_hint.setText("No cameras detected. Plug one in and press the "
                                      "refresh icon, or use a file.")
        else:
            self._source_hint.setText("")
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

    @staticmethod
    def _gain_spin() -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(0.3, 2.5)
        s.setSingleStep(0.05)
        s.setDecimals(2)
        return s

    def _camera_card(self) -> Card:
        card, form = self._card("Camera controls")

        # --- Orientation ------------------------------------------------- #
        self._orientation = QComboBox()
        self._orientation.addItem("Landscape", 0)
        self._orientation.addItem("Portrait (90° CW)", 90)
        self._orientation.addItem("Portrait (90° CCW)", 270)
        self._orientation.addItem("Upside down (180°)", 180)
        self._orientation.currentIndexChanged.connect(self._on_camera_changed)
        form.addRow("Orientation", self._orientation)

        self._flip_h = QCheckBox("Flip horizontally (mirror left/right)")
        self._flip_h.toggled.connect(self._on_camera_changed)
        form.addRow("", self._flip_h)
        self._flip_v = QCheckBox("Flip vertically (mirror top/bottom)")
        self._flip_v.toggled.connect(self._on_camera_changed)
        form.addRow("", self._flip_v)

        self._overhead = QCheckBox("Camera is directly overhead")
        self._overhead.setToolTip("Turns off the oblique-only parallax + far-rail "
                                  "corrections (≈2× frame rate) now that the view is top-down.")
        self._overhead.toggled.connect(self._on_camera_changed)
        form.addRow("", self._overhead)

        # --- Colour correction ------------------------------------------- #
        self._cc_mode = QComboBox()
        self._cc_mode.addItem("Auto (pool table)", "auto")
        self._cc_mode.addItem("Manual", "manual")
        self._cc_mode.addItem("Off", "off")
        self._cc_mode.currentIndexChanged.connect(self._on_cc_mode_changed)
        form.addRow("Colour correction", self._cc_mode)

        self._cc_r = self._gain_spin()
        self._cc_g = self._gain_spin()
        self._cc_b = self._gain_spin()
        self._cc_sat = self._gain_spin()
        for w in (self._cc_r, self._cc_g, self._cc_b, self._cc_sat):
            w.valueChanged.connect(self._on_camera_changed)
        grow = QHBoxLayout()
        for lbl, w in (("R", self._cc_r), ("G", self._cc_g), ("B", self._cc_b)):
            grow.addWidget(QLabel(lbl))
            grow.addWidget(w)
        gw = QWidget()
        gw.setLayout(grow)
        form.addRow("Manual gain", gw)
        form.addRow("Saturation", self._cc_sat)

        return card

    def _on_cc_mode_changed(self) -> None:
        manual = self._cc_mode.currentData() == "manual"
        for w in (self._cc_r, self._cc_g, self._cc_b, self._cc_sat):
            w.setEnabled(manual)
        self._on_camera_changed()

    def _on_camera_changed(self) -> None:
        if not self._loaded:
            return
        s = self._s
        s.camera.rotation = int(self._orientation.currentData() or 0)
        s.camera.flip_h = self._flip_h.isChecked()
        s.camera.flip_v = self._flip_v.isChecked()
        s.camera.overhead = self._overhead.isChecked()
        s.camera.color.mode = str(self._cc_mode.currentData() or "auto")
        s.camera.color.gain_r = round(self._cc_r.value(), 2)
        s.camera.color.gain_g = round(self._cc_g.value(), 2)
        s.camera.color.gain_b = round(self._cc_b.value(), 2)
        s.camera.color.saturation = round(self._cc_sat.value(), 2)
        s.save()
        self.applied.emit()

    def _ai_card(self) -> Card:
        card, form = self._card("AI labelling (Training)")
        msg = QLabel("Let a vision model label your recorded training sessions "
                     "automatically — the 'Auto-label with AI' button on the "
                     "Training tab. Needs an OpenRouter API key.")
        msg.setObjectName("Faint")
        msg.setWordWrap(True)
        form.addRow("", msg)

        self._ai_backend = QComboBox()
        self._ai_backend.addItem("Off", "off")
        self._ai_backend.addItem("OpenRouter", "openrouter")
        self._ai_backend.currentIndexChanged.connect(self._on_ai_changed)
        form.addRow("Backend", self._ai_backend)

        self._ai_key = QLineEdit()
        self._ai_key.setEchoMode(QLineEdit.Password)
        self._ai_key.setPlaceholderText("sk-or-…")
        self._ai_key.editingFinished.connect(self._on_ai_changed)
        form.addRow("API key", self._ai_key)

        self._ai_model = QLineEdit()
        self._ai_model.editingFinished.connect(self._on_ai_changed)
        form.addRow("Model", self._ai_model)

        hint = QLabel("A vision-capable model id, e.g. anthropic/claude-3.7-sonnet "
                      "or openai/gpt-4o. Your key stays on this machine.")
        hint.setObjectName("Faint")
        hint.setWordWrap(True)
        form.addRow("", hint)
        return card

    def _recording_card(self) -> Card:
        card, form = self._card("Recording")
        msg = QLabel("Where session recordings are saved. Point it at a Dropbox "
                     "or iCloud folder to back sessions up automatically.")
        msg.setObjectName("Faint")
        msg.setWordWrap(True)
        form.addRow("", msg)

        from PySide6.QtWidgets import QHBoxLayout, QWidget
        self._rec_dir = QLineEdit()
        self._rec_dir.setPlaceholderText("Default — app exports folder")
        self._rec_dir.editingFinished.connect(self._on_recording_changed)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_rec_dir)
        row = QWidget()
        rlay = QHBoxLayout(row)
        rlay.setContentsMargins(0, 0, 0, 0)
        rlay.addWidget(self._rec_dir, 1)
        rlay.addWidget(browse)
        form.addRow("Folder", row)

        self._rec_audio = QCheckBox("Record audio into session clips")
        self._rec_audio.toggled.connect(self._on_recording_changed)
        form.addRow("", self._rec_audio)

        return card

    def _browse_rec_dir(self) -> None:
        import os

        from PySide6.QtWidgets import QFileDialog
        start = self._rec_dir.text().strip() or os.path.expanduser("~")
        path = QFileDialog.getExistingDirectory(self, "Choose recordings folder", start)
        if path:
            self._rec_dir.setText(path)
            self._on_recording_changed()

    def _on_recording_changed(self) -> None:
        if not self._loaded:
            return
        r = self._s.recording
        r.directory = self._rec_dir.text().strip()
        r.audio = self._rec_audio.isChecked()
        r.audio_device = str(self._rec_device.currentData() or "default")
        self._s.save()
        # Always apply: the audio METER must re-attach the moment a
        # different mic is picked (it kept listening to the old device
        # until an app restart — "the meter isn\'t live").
        self.applied.emit()

    def _on_ai_changed(self) -> None:
        if not self._loaded:
            return
        a = self._s.autolabel
        a.backend = str(self._ai_backend.currentData() or "off")
        a.api_key = self._ai_key.text().strip()
        a.model = self._ai_model.text().strip() or a.model
        self._s.save()

    def _table_card(self) -> Card:
        card, form = self._card("Table")
        self._table_size = QComboBox()
        self._table_size.addItems(["9ft", "8ft", "7ft"])
        form.addRow("Table size", self._table_size)
        self._cushion_in = QDoubleSpinBox()
        self._cushion_in.setRange(0.0, 4.0)
        self._cushion_in.setSingleStep(0.25)
        self._cushion_in.setSuffix('"')
        self._cushion_in.setToolTip("Cushion width: felt edge to cushion nose. Balls "
                                    "rebound off the nose, not the felt edge.")
        form.addRow("Cushion inset", self._cushion_in)
        self._auto_relock = QCheckBox("Auto re-lock if the table shifts")
        form.addRow("", self._auto_relock)
        self._persist_calib = QCheckBox("Remember calibration between launches")
        form.addRow("", self._persist_calib)
        recal = QPushButton("Recalibrate table now")
        recal.setObjectName("Ghost")
        recal.setCursor(Qt.PointingHandCursor)
        recal.setToolTip("Re-detect the table (use if the read looks off or the "
                         "camera moved)")
        recal.clicked.connect(self.recalibrate_requested.emit)
        form.addRow("", recal)
        return card

    def _model_card(self) -> Card:
        card, form = self._card("AI detection")
        msg = QLabel("Ball tracking uses a trained model that runs locally on your "
                     "GPU. It's selected automatically — there's nothing to configure.")
        msg.setObjectName("Faint")
        msg.setWordWrap(True)
        form.addRow("", msg)

        # One-click "get the detector" — the only model action a user needs. The
        # model is auto-selected once present; if it's missing this fetches it.
        self._dl_yolo_btn = QPushButton("  Download ball-detection model")
        self._dl_yolo_btn.setObjectName("Ghost")
        self._dl_yolo_btn.setCursor(Qt.PointingHandCursor)
        self._dl_yolo_btn.clicked.connect(self._download_yolo11)
        self._dl_status = QLabel("")
        self._dl_status.setObjectName("Faint")
        self._dl_status.setWordWrap(True)
        self._refresh_yolo_button()
        form.addRow("", self._dl_yolo_btn)
        form.addRow("", self._dl_status)

        # Perf/recall trade-off: the detector can re-scan the foreshortened far
        # rail with a second inference pass. Better far-ball recall, ~2x GPU cost.
        self._far_rescan = QCheckBox("Extra far-rail scan (finds distant balls; "
                                     "~2× GPU per frame)")
        form.addRow("", self._far_rescan)

        # Active-learning: teach the model THIS table's ball numbers/colours. The
        # generic model can't tell adjacent colours apart on a specific felt, so a
        # quick correct-the-guesses pass + fine-tune adapts it. Redo if the camera
        # moves.
        train_btn = QPushButton("  Train ball IDs on my table…")
        train_btn.setObjectName("Ghost")
        train_btn.setCursor(Qt.PointingHandCursor)
        train_btn.clicked.connect(self.train_balls_requested.emit)
        form.addRow("", train_btn)
        note = QLabel("Teach the model your exact balls: it guesses each number, you "
                      "correct the wrong ones, then fine-tune. Improves number/colour "
                      "accuracy for your table; redo if your camera angle changes.")
        note.setObjectName("Faint")
        note.setWordWrap(True)
        form.addRow("", note)
        return card

    def _refresh_yolo_button(self) -> None:
        from ...detector_strategies import model_fetch
        if model_fetch.is_present("pool_yolo11"):
            self._dl_yolo_btn.setEnabled(False)
            self._dl_yolo_btn.setText("  Model installed ✓")
            self._dl_status.setText("The ball-detection model is installed and in use.")
        else:
            self._dl_yolo_btn.setEnabled(True)
            self._dl_yolo_btn.setText("  Download ball-detection model")
            self._dl_status.setText(
                "Fetches the ball-detection model (~38 MB, runs locally on your GPU). "
                "Needed once; it's selected automatically afterwards.")

    def _download_yolo11(self) -> None:
        self._dl_yolo_btn.setEnabled(False)
        self._dl_status.setText("Downloading… 0%")
        self._dl_thread = QThread(self)
        self._dl_worker = _ModelDownloadWorker("pool_yolo11")
        self._dl_worker.moveToThread(self._dl_thread)
        self._dl_thread.started.connect(self._dl_worker.run)
        self._dl_worker.progress.connect(
            lambda f: self._dl_status.setText(f"Downloading… {int(f * 100)}%"))
        self._dl_worker.done.connect(self._on_yolo_downloaded)
        self._dl_worker.failed.connect(self._on_yolo_failed)
        for sig in (self._dl_worker.done, self._dl_worker.failed):
            sig.connect(self._dl_thread.quit)
        self._dl_thread.start()

    def _on_yolo_downloaded(self, strategy: str) -> None:
        self._refresh_yolo_button()
        self._dl_status.setText("Installed — the model is now selected automatically.")

    def _on_yolo_failed(self, msg: str) -> None:
        self._dl_yolo_btn.setEnabled(True)
        self._dl_status.setText(f"Download failed: {msg}. Check your connection and retry.")

    def _clock_card(self) -> Card:
        card, form = self._card("Shot clock")
        msg = QLabel("The countdown starts when the cue ball comes to rest and "
                     "stops when you strike it — beat the buzzer. One beep at "
                     "10 s left, tick beeps at 3-2-1, buzz at 0. Runs during "
                     "LIVE camera play only — never over a video you're reviewing.")
        msg.setObjectName("Faint")
        msg.setWordWrap(True)
        form.addRow("", msg)
        self._clock_enabled = QCheckBox("Enable shot clock")
        form.addRow("", self._clock_enabled)
        self._clock_seconds = QComboBox()
        self._clock_seconds.addItem("30 seconds", 30)
        self._clock_seconds.addItem("60 seconds", 60)
        form.addRow("Per shot", self._clock_seconds)
        self._clock_warn = self._spin(0, 60)
        form.addRow("Warn at (s left)", self._clock_warn)
        self._clock_audio = QCheckBox("Audio cues (beeps + buzzer)")
        form.addRow("", self._clock_audio)
        self._clock_autoreset = QCheckBox("Auto-reset on detected shot")
        form.addRow("", self._clock_autoreset)
        return card

    def _appearance_card(self) -> Card:
        card, form = self._card("Detection & display")
        self._detection_enabled = QCheckBox("Ball detection enabled")
        self._detection_enabled.setToolTip("Master switch for AI ball/shot detection")
        form.addRow("", self._detection_enabled)
        self._accent = QLineEdit()
        self._accent.setPlaceholderText("#3DDC97")
        form.addRow("Accent colour", self._accent)
        self._show_overlays = QCheckBox("Show detection overlays")
        form.addRow("", self._show_overlays)
        self._align_grid = QCheckBox("Alignment grid (rule of thirds) on the camera view")
        form.addRow("", self._align_grid)
        self._show_traj = QCheckBox("Show ball trajectories")
        form.addRow("", self._show_traj)
        self._show_ids = QCheckBox("Show ball IDs")
        form.addRow("", self._show_ids)
        self._measured_colors = QCheckBox("Draw balls in their real measured colour "
                                          "(grey ? when unsure)")
        form.addRow("", self._measured_colors)
        self._schematic = QCheckBox("Clean schematic bird's-eye (vs warped camera)")
        form.addRow("", self._schematic)

        # Detector min-confidence: lower finds more (recovers a faint ball),
        # higher is stricter (drops false blobs). Applies live on Save.
        from PySide6.QtWidgets import QSlider
        self._conf_slider = QSlider(Qt.Horizontal)
        self._conf_slider.setRange(10, 90)
        self._conf_val = QLabel("")
        self._conf_slider.valueChanged.connect(
            lambda v: self._conf_val.setText(f"{v / 100.0:.2f}"))
        crow = QHBoxLayout()
        crow.addWidget(self._conf_slider, 1)
        crow.addWidget(self._conf_val)
        cw = QWidget()
        cw.setLayout(crow)
        form.addRow("Detection confidence", cw)

        note = QLabel("Changes apply after Save.")
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

    def _debug_card(self) -> Card:
        card, form = self._card("Debug")
        msg = QLabel("Caught the detector getting it wrong? Save the last few "
                     "seconds + detector output to a zip and flag it as a failure. "
                     "The bundle is staged locally for analysis (synced to the dev "
                     "machine / eval harness).")
        msg.setObjectName("Faint")
        msg.setWordWrap(True)
        form.addRow("", msg)
        flag = QPushButton("  Save clip + flag as failure")
        flag.setObjectName("Danger")
        flag.setCursor(Qt.PointingHandCursor)
        flag.clicked.connect(self.flag_failure_requested.emit)
        form.addRow("", flag)
        self._debug_status = QLabel("")
        self._debug_status.setObjectName("Faint")
        self._debug_status.setWordWrap(True)
        form.addRow("", self._debug_status)
        return card

    def set_debug_status(self, text: str) -> None:
        self._debug_status.setText(text)

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

    def _cue_card(self) -> Card:
        card, form = self._card("Cue stroke sensor")
        msg = QLabel("A Bluetooth motion sensor on the cue butt measures each "
                     "stroke: cue speed, draw length, steer, pause and contact "
                     "quality appear on the Sandbox panel after every shot. "
                     "Everything works normally when no sensor is around.")
        msg.setObjectName("Faint")
        msg.setWordWrap(True)
        form.addRow("", msg)
        self._cue_enabled = QCheckBox("Enable cue sensor (connects automatically)")
        form.addRow("", self._cue_enabled)
        self._cue_floor = QDoubleSpinBox()
        self._cue_floor.setRange(1.0, 4.0)
        self._cue_floor.setSingleStep(0.1)
        self._cue_floor.setDecimals(1)
        self._cue_floor.setSuffix(" g")
        self._cue_floor.setToolTip("Impact sensitivity: the minimum shock that can "
                                   "count as a shot. Lower finds softer pokes; the "
                                   "validated hit signature filters the noise.")
        form.addRow("Impact floor", self._cue_floor)
        diag = QPushButton("  Sensor diagnostics…")
        diag.setObjectName("Ghost")
        diag.setCursor(Qt.PointingHandCursor)
        diag.setToolTip("Live waveforms + connection details — for checking the "
                        "mounting and stream health")
        diag.clicked.connect(self.cue_diagnostics_requested.emit)
        form.addRow("", diag)
        self._cue_status = QLabel("")
        self._cue_status.setObjectName("Faint")
        self._cue_status.setWordWrap(True)
        form.addRow("", self._cue_status)
        note = QLabel("Mounting: sensor on the butt, +Z pointing butt→tip. "
                      "The sensor is never written to by this app.")
        note.setObjectName("Faint")
        note.setWordWrap(True)
        form.addRow("", note)
        return card

    def set_cue_status(self, state: str, detail) -> None:
        pretty = {
            "disabled": "Sensor disabled.",
            "unavailable": "Bluetooth support isn't available in this build.",
            "bluetooth_off": "Bluetooth appears to be off — enable it in Windows.",
            "scanning": "Scanning for the sensor…",
            "no_sensor": "No sensor found — is it powered and in range? "
                         "Rescanning periodically.",
            "connecting": "Sensor found — connecting…",
            "reconnecting": "Connection dropped — reconnecting…",
            "error": "Connection error — retrying.",
        }.get(state, state)
        if state == "connected" and isinstance(detail, dict):
            batt = detail.get("battery")
            pretty = (f"Connected to {detail.get('name') or detail.get('address', '')}"
                      + (f" · battery {batt}%" if batt is not None else ""))
        self._cue_status.setText(pretty)

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
        # Camera controls
        oi = self._orientation.findData(s.camera.rotation)
        self._orientation.setCurrentIndex(oi if oi >= 0 else 0)
        self._flip_h.setChecked(s.camera.flip_h)
        self._flip_v.setChecked(s.camera.flip_v)
        self._overhead.setChecked(s.camera.overhead)
        ci = self._cc_mode.findData(s.camera.color.mode)
        self._cc_mode.setCurrentIndex(ci if ci >= 0 else 0)
        self._cc_r.setValue(s.camera.color.gain_r)
        self._cc_g.setValue(s.camera.color.gain_g)
        self._cc_b.setValue(s.camera.color.gain_b)
        self._cc_sat.setValue(s.camera.color.saturation)
        manual = s.camera.color.mode == "manual"
        for w in (self._cc_r, self._cc_g, self._cc_b, self._cc_sat):
            w.setEnabled(manual)
        self._table_size.setCurrentText(s.table.size)
        self._cushion_in.setValue(float(getattr(s.table, 'cushion_inset_in', 2.0)))
        self._auto_relock.setChecked(s.table.auto_relock)
        self._persist_calib.setChecked(s.table.persist_calibration)
        self._show_overlays.setChecked(s.ui.show_overlays)
        self._detection_enabled.setChecked(bool(getattr(s.detection, 'enabled', True)))
        self._clock_enabled.setChecked(s.shot_clock.enabled)
        idx = self._clock_seconds.findData(s.shot_clock.seconds)
        if idx < 0:  # legacy custom duration (old spinbox) — keep honouring it
            self._clock_seconds.addItem(f"{s.shot_clock.seconds} seconds",
                                        s.shot_clock.seconds)
            idx = self._clock_seconds.count() - 1
        self._clock_seconds.setCurrentIndex(idx)
        self._clock_warn.setValue(s.shot_clock.warn_seconds)
        self._clock_audio.setChecked(s.shot_clock.audio)
        self._clock_autoreset.setChecked(s.shot_clock.auto_reset_on_shot)
        self._accent.setText(s.ui.accent)
        self._show_traj.setChecked(s.ui.show_trajectories)
        self._show_ids.setChecked(s.ui.show_ball_ids)
        self._measured_colors.setChecked(s.ui.measured_ball_colors)
        self._schematic.setChecked(s.ui.schematic_birdseye)
        self._align_grid.setChecked(bool(getattr(s.ui, 'alignment_grid', False)))
        self._conf_slider.setValue(int(round(s.detection.confidence_floor * 100)))
        self._conf_val.setText(f"{s.detection.confidence_floor:.2f}")
        self._far_rescan.setChecked(s.balls.far_rail_rescan)
        self._rec_dir.setText(s.recording.directory)
        self._rec_audio.setChecked(s.recording.audio)
        di = self._rec_device.findData(s.recording.audio_device)
        self._rec_device.setCurrentIndex(di if di >= 0 else 0)
        self._auto_check.setChecked(s.updates.auto_check)
        self._cue_enabled.setChecked(s.cue.enabled)
        self._cue_floor.setValue(s.cue.impact_g)
        bi = self._ai_backend.findData(s.autolabel.backend)
        self._ai_backend.setCurrentIndex(bi if bi >= 0 else 0)
        self._ai_key.setText(s.autolabel.api_key)
        self._ai_model.setText(s.autolabel.model)

    def _save(self) -> None:
        s = self._s
        spec = str(self._source_combo.currentData() or "0")
        s.source = spec
        s.source_name = self._cam_names.get(spec, "") if spec.isdigit() else ""
        s.ui.mirror_preview = self._mirror.isChecked()
        s.table.size = self._table_size.currentText()
        s.table.cushion_inset_in = round(self._cushion_in.value(), 2)
        s.table.auto_relock = self._auto_relock.isChecked()
        s.table.persist_calibration = self._persist_calib.isChecked()
        s.shot_clock.enabled = self._clock_enabled.isChecked()
        s.shot_clock.seconds = int(self._clock_seconds.currentData())
        s.shot_clock.warn_seconds = self._clock_warn.value()
        s.shot_clock.audio = self._clock_audio.isChecked()
        s.shot_clock.auto_reset_on_shot = self._clock_autoreset.isChecked()
        accent = self._accent.text().strip()
        if accent:
            s.ui.accent = accent
        s.ui.show_overlays = self._show_overlays.isChecked()
        s.detection.enabled = self._detection_enabled.isChecked()
        s.ui.show_trajectories = self._show_traj.isChecked()
        s.ui.show_ball_ids = self._show_ids.isChecked()
        s.ui.measured_ball_colors = self._measured_colors.isChecked()
        s.ui.schematic_birdseye = self._schematic.isChecked()
        s.ui.alignment_grid = self._align_grid.isChecked()
        s.detection.confidence_floor = self._conf_slider.value() / 100.0
        s.balls.far_rail_rescan = self._far_rescan.isChecked()
        s.updates.auto_check = self._auto_check.isChecked()
        s.cue.enabled = self._cue_enabled.isChecked()
        s.cue.impact_g = round(self._cue_floor.value(), 2)
        s.save()
        self.applied.emit()
