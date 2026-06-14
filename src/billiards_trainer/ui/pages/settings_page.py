"""Settings page — edits the live ``Settings`` object and persists it.

Grouped into Source, Table & Felt, Ball detection, Shot clock, Appearance, and
Updates. Emits ``applied`` so the main window can re-theme / restart capture.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
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

    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self._s = settings
        self._build()
        self._load_from_settings()

    # ------------------------------------------------------------------ #
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(16)

        header = QHBoxLayout()
        header.addWidget(section_header("Settings"))
        header.addStretch(1)
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

        grid.addWidget(self._source_card(), 0, 0)
        grid.addWidget(self._felt_card(), 0, 1)
        grid.addWidget(self._ball_card(), 1, 0)
        grid.addWidget(self._clock_card(), 1, 1)
        grid.addWidget(self._appearance_card(), 2, 0)
        grid.addWidget(self._updates_card(), 2, 1)
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
        card, form = self._card("Video source")
        self._source_edit = QLineEdit()
        self._source_edit.setPlaceholderText("Camera index (0) or path to video/image")
        form.addRow("Source", self._source_edit)
        hint = QLabel("Use a camera index like 0, or a file path. "
                      "Type 'demo' for the built-in simulation.")
        hint.setObjectName("Faint")
        hint.setWordWrap(True)
        form.addRow("", hint)
        self._mirror = QCheckBox("Mirror preview horizontally")
        form.addRow("", self._mirror)
        return card

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
        self._backend.addItems(["classical", "yolo"])
        form.addRow("Backend", self._backend)
        note = QLabel("Classical (Hough + colour) runs everywhere with no model. "
                      "YOLO needs the optional [yolo] extra + weights.")
        note.setObjectName("Faint")
        note.setWordWrap(True)
        form.addRow("", note)
        self._param2 = self._spin(5, 60)
        form.addRow("Detector strictness", self._param2)
        self._yolo_url = QLineEdit()
        self._yolo_url.setPlaceholderText("https://…/pool_balls.pt (auto-fetched)")
        form.addRow("YOLO weights URL", self._yolo_url)
        return card

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
        self._auto_check = QCheckBox("Check for updates on launch")
        form.addRow("", self._auto_check)
        from ...version import __version__
        ver = QLabel(f"Installed version: <b>{__version__}</b>")
        ver.setObjectName("Muted")
        form.addRow("", ver)
        return card

    @staticmethod
    def _spin(lo: int, hi: int) -> QSpinBox:
        s = QSpinBox()
        s.setRange(lo, hi)
        return s

    # ------------------------------------------------------------------ #
    def reload(self) -> None:
        """Refresh the controls from the (possibly externally-mutated) settings."""
        self._load_from_settings()

    def _load_from_settings(self) -> None:
        s = self._s
        self._source_edit.setText(s.source)
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
        s.source = self._source_edit.text().strip() or "0"
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
