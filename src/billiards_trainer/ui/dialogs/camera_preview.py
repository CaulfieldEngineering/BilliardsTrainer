"""A small modal that previews a camera so the user can confirm the right one
before committing. Opens the device, streams it into a VideoView, and cleans up
on close. Surfaces the common failure modes clearly (in use / not opening)."""

import sys

import cv2
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from ..theme import PALETTE
from ..widgets.video_view import VideoView


class CameraPreviewDialog(QDialog):
    def __init__(self, index: int, name: str = "", parent=None):
        super().__init__(parent)
        self._index = index
        self.setWindowTitle(f"Preview — {name or f'Camera {index}'}")
        self.setMinimumSize(560, 460)
        self._cap = None
        self._miss = 0
        self._build(name)
        self._open()

    def _build(self, name: str) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 14)
        root.setSpacing(10)
        title = QLabel(f"<b>{name or f'Camera {self._index}'}</b>  ·  index {self._index}")
        root.addWidget(title)
        self._view = VideoView("Opening camera…")
        root.addWidget(self._view, 1)
        self._status = QLabel("")
        self._status.setObjectName("Faint")
        self._status.setWordWrap(True)
        root.addWidget(self._status)
        btns = QHBoxLayout()
        btns.addStretch(1)
        close = QPushButton("Close")
        close.setObjectName("Accent")
        close.clicked.connect(self.accept)
        btns.addWidget(close)
        root.addLayout(btns)

    def _open(self) -> None:
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(self._index, backend)
        if not self._cap.isOpened():
            self._fail("Couldn't open this camera. It may be in use by another app "
                       "(close Zoom/Teams/OBS), or disconnected. On Windows, also check "
                       "Settings → Privacy → Camera.")
            return
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)

    def _tick(self) -> None:
        if self._cap is None:
            return
        ok, frame = self._cap.read()
        if not ok or frame is None:
            self._miss += 1
            if self._miss > 30:
                self._fail("Camera opened but isn't delivering frames — another app may "
                           "be using it.")
                self._timer.stop()
            return
        self._miss = 0
        self._status.setText("Live ✓  — looks right? Close and Save.")
        self._status.setStyleSheet(f"color: {PALETTE.success};")
        self._view.set_frame(frame)

    def _fail(self, msg: str) -> None:
        self._status.setText(msg)
        self._status.setStyleSheet(f"color: {PALETTE.danger};")

    def _release(self) -> None:
        try:
            if hasattr(self, "_timer"):
                self._timer.stop()
            if self._cap is not None:
                self._cap.release()
                self._cap = None
        except Exception:  # noqa: BLE001
            pass

    def closeEvent(self, event) -> None:
        self._release()
        super().closeEvent(event)

    def accept(self) -> None:
        self._release()
        super().accept()
