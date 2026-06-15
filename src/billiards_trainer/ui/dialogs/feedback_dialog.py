"""In-app feedback form: bug reports + feature requests.

Local-first: every submission is written to the SQLite ``feedback`` table
immediately (and queued for optional Supabase sync). Optional attachments — a
screenshot of the current window and the last-5s replay clip — help diagnose.
"""

import platform
import sys
from datetime import datetime, timezone

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
)

from ...config import EXPORTS_DIR
from ...version import __version__


class FeedbackDialog(QDialog):
    # Emitted after a successful save so the main window can attach a replay
    # clip (which must be produced on the controller thread).
    replay_requested = Signal(int)   # feedback_id
    submitted = Signal(int)          # feedback_id

    def __init__(self, repository, parent=None):
        super().__init__(parent)
        self._repo = repository
        self._parent_win = parent
        self.setWindowTitle("Send feedback")
        self.setMinimumWidth(480)
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 22, 24, 18)
        root.setSpacing(12)

        title = QLabel("Send feedback")
        title.setObjectName("H2")
        root.addWidget(title)
        sub = QLabel("Reports are saved on your machine and (optionally) backed up. "
                     "Thanks for helping improve the app.")
        sub.setObjectName("Muted")
        sub.setWordWrap(True)
        root.addWidget(sub)

        # type
        type_row = QHBoxLayout()
        self._bug = QRadioButton("Bug report")
        self._feature = QRadioButton("Feature request")
        self._bug.setChecked(True)
        grp = QButtonGroup(self)
        grp.addButton(self._bug)
        grp.addButton(self._feature)
        self._bug.toggled.connect(self._on_type_changed)
        type_row.addWidget(self._bug)
        type_row.addWidget(self._feature)
        type_row.addStretch(1)
        root.addLayout(type_row)

        self._title = QLineEdit()
        self._title.setMaxLength(80)
        self._title.setPlaceholderText("Short summary (required)")
        root.addWidget(self._title)

        self._desc = QPlainTextEdit()
        self._desc.setPlaceholderText("What happened, or what would you like? Steps to "
                                      "reproduce help a lot for bugs.")
        self._desc.setMinimumHeight(120)
        root.addWidget(self._desc)

        sev_row = QHBoxLayout()
        self._sev_label = QLabel("Severity")
        self._severity = QComboBox()
        self._severity.addItems(["low", "medium", "high"])
        self._severity.setCurrentText("medium")
        sev_row.addWidget(self._sev_label)
        sev_row.addWidget(self._severity)
        sev_row.addStretch(1)
        root.addLayout(sev_row)

        self._attach_shot = QCheckBox("Attach a screenshot of the current view")
        self._attach_shot.setChecked(True)
        root.addWidget(self._attach_shot)
        self._attach_replay = QCheckBox("Attach the last 5 seconds (replay clip)")
        root.addWidget(self._attach_replay)

        self._status = QLabel("")
        self._status.setObjectName("Faint")
        root.addWidget(self._status)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.setObjectName("Ghost")
        cancel.clicked.connect(self.reject)
        buttons.addWidget(cancel)
        self._submit = QPushButton("Send")
        self._submit.setObjectName("Accent")
        self._submit.setCursor(Qt.PointingHandCursor)
        self._submit.clicked.connect(self._on_submit)
        buttons.addWidget(self._submit)
        root.addLayout(buttons)

    def _on_type_changed(self) -> None:
        is_bug = self._bug.isChecked()
        self._sev_label.setVisible(is_bug)
        self._severity.setVisible(is_bug)

    def _system_info(self) -> str:
        screen = QGuiApplication.primaryScreen()
        res = ""
        if screen:
            g = screen.geometry()
            res = f"{g.width()}x{g.height()}"
        return (f"{platform.system()} {platform.release()} | {res} | "
                f"Py{sys.version.split()[0]}")

    def _grab_screenshot(self) -> str:
        try:
            target = self._parent_win if self._parent_win is not None else self
            pix = target.grab()
            folder = EXPORTS_DIR / "feedback"
            folder.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            path = folder / f"screenshot-{stamp}.png"
            pix.save(str(path), "PNG")
            return str(path)
        except Exception:  # noqa: BLE001
            return ""

    def _on_submit(self) -> None:
        title = self._title.text().strip()
        if not title:
            self._status.setText("Please add a short title.")
            self._title.setFocus()
            return
        kind = "bug" if self._bug.isChecked() else "feature"
        severity = self._severity.currentText() if kind == "bug" else ""
        attachments = []
        if self._attach_shot.isChecked():
            shot = self._grab_screenshot()
            if shot:
                attachments.append(shot)

        fid = self._repo.add_feedback(
            kind=kind, title=title, description=self._desc.toPlainText().strip(),
            severity=severity, app_version=__version__,
            system_info=self._system_info(), attachments=attachments,
        )
        if self._attach_replay.isChecked():
            self.replay_requested.emit(fid)
        self.submitted.emit(fid)

        self._status.setText("Sent! Thanks 🎱")
        self._submit.setEnabled(False)
        from PySide6.QtCore import QTimer
        QTimer.singleShot(900, self.accept)
