"""Always-on-top mini view: the live feed in a small floating card.

Joe's ask: keep an eye on the camera and recording status while other
windows (Spotify, a browser) have the screen. This is a frameless,
always-on-top window fed by the SAME controller signals as the main live
page — no second capture path, no extra decode work; it just paints the
packets it is handed.

Interactions, kept deliberately few:
- drag anywhere on the top bar to move
- bottom-right size grip to resize
- double-click the video to switch camera <-> bird's-eye
- the dot button returns to the full app; closing does the same
Geometry persists in ``ui.mini_geometry`` so it reopens where it lived.
"""

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizeGrip,
    QVBoxLayout,
    QWidget,
)

from ..theme import PALETTE
from .video_view import VideoView


class MiniView(QWidget):
    closed = Signal()
    restore_requested = Signal()

    def __init__(self, parent=None):
        # A top-level tool-style window: stays on top, no native frame, not a
        # taskbar entry of its own (it belongs to the app).
        super().__init__(parent,
                         Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self.setWindowTitle("Billiards Trainer — Mini")
        self.setMinimumSize(220, 180)
        self.resize(360, 320)
        self._drag: QPoint | None = None
        self._show_birdseye = False
        self._recording = False

        self.setStyleSheet(
            f"MiniView {{ background: {PALETTE.bg}; border: 1px solid #333;"
            "border-radius: 10px; }}")

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 8)
        root.setSpacing(6)

        # --- top bar: status + controls (and the drag handle) -------------- #
        bar = QHBoxLayout()
        bar.setSpacing(8)
        self._status = QLabel("")
        self._status.setTextFormat(Qt.RichText)
        self._status.setStyleSheet(
            "font-size: 11px; font-weight: 800; letter-spacing: 1.2px;"
            f"color: {PALETTE.text};")
        bar.addWidget(self._status)
        bar.addStretch(1)
        restore = QPushButton("⤢")
        restore.setToolTip("Back to the full app")
        close = QPushButton("✕")
        close.setToolTip("Close mini view")
        for b in (restore, close):
            b.setCursor(Qt.PointingHandCursor)
            b.setFixedSize(22, 22)
            b.setStyleSheet(
                "QPushButton { background: rgba(255,255,255,0.06); border: none;"
                f"border-radius: 4px; color: {PALETTE.text}; font-size: 12px; }}"
                "QPushButton:hover { background: rgba(255,255,255,0.16); }")
        restore.clicked.connect(self.restore_requested.emit)
        close.clicked.connect(self.close)
        bar.addWidget(restore)
        bar.addWidget(close)
        root.addLayout(bar)

        # --- the feed ------------------------------------------------------ #
        self._video = VideoView()
        root.addWidget(self._video, 1)

        # --- footer: hint + size grip -------------------------------------- #
        foot = QHBoxLayout()
        self._hint = QLabel("double-click: camera ⇄ table")
        self._hint.setStyleSheet(
            f"color: {PALETTE.text_dim}; font-size: 9px;")
        foot.addWidget(self._hint)
        foot.addStretch(1)
        foot.addWidget(QSizeGrip(self), 0, Qt.AlignBottom | Qt.AlignRight)
        root.addLayout(foot)

        self._set_status_text("LIVE", PALETTE.success)

    # ------------------------------------------------------------------ #
    # Feed — same packets the live page gets; painting only.
    # ------------------------------------------------------------------ #
    def on_frame(self, packet) -> None:
        if not self.isVisible():
            return
        img = packet.birdseye if (self._show_birdseye
                                  and packet.birdseye is not None) \
            else packet.perspective
        if img is not None:
            self._video.set_frame(img)

    def on_recording(self, on: bool) -> None:
        self._recording = on
        self._refresh_status()

    def on_status(self, status: str) -> None:
        self._running = status == "running"
        self._refresh_status()

    def _refresh_status(self) -> None:
        if self._recording:
            self._set_status_text("REC", PALETTE.danger)
        elif getattr(self, "_running", True):
            self._set_status_text("LIVE", PALETTE.success)
        else:
            self._set_status_text("IDLE", PALETTE.text_dim)

    def _set_status_text(self, text: str, color: str) -> None:
        self._status.setText(
            f'<span style="color:{color}; font-size:10px;">●</span>'
            f'&nbsp;&nbsp;{text}')

    # ------------------------------------------------------------------ #
    # Frameless-window ergonomics
    # ------------------------------------------------------------------ #
    def mousePressEvent(self, ev):  # noqa: N802 - Qt override
        if ev.button() == Qt.LeftButton:
            self._drag = ev.globalPosition().toPoint() - self.frameGeometry().topLeft()
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):  # noqa: N802 - Qt override
        if self._drag is not None and ev.buttons() & Qt.LeftButton:
            self.move(ev.globalPosition().toPoint() - self._drag)
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):  # noqa: N802 - Qt override
        self._drag = None
        super().mouseReleaseEvent(ev)

    def mouseDoubleClickEvent(self, ev):  # noqa: N802 - Qt override
        self._show_birdseye = not self._show_birdseye
        super().mouseDoubleClickEvent(ev)

    def closeEvent(self, ev):  # noqa: N802 - Qt override
        self.closed.emit()
        super().closeEvent(ev)

    # ------------------------------------------------------------------ #
    def geometry_string(self) -> str:
        g = self.geometry()
        return f"{g.x()},{g.y()},{g.width()},{g.height()}"

    def apply_geometry_string(self, s: str) -> None:
        try:
            x, y, w, h = (int(v) for v in s.split(","))
            self.setGeometry(x, y, max(self.minimumWidth(), w),
                             max(self.minimumHeight(), h))
        except (ValueError, AttributeError):
            pass
