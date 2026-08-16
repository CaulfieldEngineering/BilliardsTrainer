"""Always-on-top mini view: the live feed in a small floating card.

Joe's asks, in order given: keep an eye on the camera while other windows
have the screen; then — record from here, borderless, window hugging the
video frame. So v2 is a pure video surface: the frame IS the window (the
window snaps to the video's aspect ratio, so there are never letterbox
bars), and all chrome lives on a hover overlay that vanishes when the
mouse leaves. A small status dot stays visible so REC state is glanceable
without hovering.

Fed by the SAME controller signals as the main live page — no second
capture path; painting only. The record button emits the same signal the
main record button drives.

Interactions:
- drag anywhere to move; bottom-right grip to resize (aspect preserved)
- hover: REC toggle, back-to-app, close
- double-click: switch camera <-> bird's-eye
Geometry persists in ``ui.mini_geometry``.
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
    record_toggled = Signal(bool)     # same contract as LivePage.record_toggled

    def __init__(self, parent=None):
        super().__init__(parent,
                         Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self.setWindowTitle("Billiards Trainer — Mini")
        self.setMinimumSize(200, 120)
        self.resize(384, 216)
        self._drag: QPoint | None = None
        self._show_birdseye = False
        self._recording = False
        self._running = True
        self._aspect = 16 / 9        # updated from real frames
        self._in_aspect_snap = False

        # The video IS the window: no margins, no card, no bars.
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self._video = VideoView()
        self._video.setMinimumSize(1, 1)   # the WINDOW minimum governs
        root.addWidget(self._video)

        # --- hover overlay: all chrome lives here ------------------------- #
        self._overlay = QWidget(self)
        self._overlay.setStyleSheet(
            "background: rgba(0,0,0,0.55); border-radius: 0px;")
        bar = QHBoxLayout(self._overlay)
        bar.setContentsMargins(10, 6, 10, 6)
        bar.setSpacing(10)

        self._rec_btn = QPushButton("● REC")
        self._rec_btn.setToolTip("Start/stop recording (same as the main record button)")
        self._rec_btn.setCursor(Qt.PointingHandCursor)
        self._rec_btn.setCheckable(True)
        self._rec_btn.setStyleSheet(
            "QPushButton { background: rgba(255,255,255,0.10); border: none;"
            "border-radius: 4px; padding: 3px 10px; font-size: 11px;"
            f"font-weight: 800; letter-spacing: 1px; color: {PALETTE.text}; }}"
            "QPushButton:hover { background: rgba(255,255,255,0.22); }"
            "QPushButton:checked { background: #B3261E; color: white; }")
        self._rec_btn.clicked.connect(
            lambda checked: self.record_toggled.emit(bool(checked)))
        bar.addWidget(self._rec_btn)
        bar.addStretch(1)

        restore = QPushButton("⤢")
        restore.setToolTip("Back to the full app")
        close = QPushButton("✕")
        close.setToolTip("Close mini view")
        for b in (restore, close):
            b.setCursor(Qt.PointingHandCursor)
            b.setFixedSize(24, 24)
            b.setStyleSheet(
                "QPushButton { background: rgba(255,255,255,0.10); border: none;"
                f"border-radius: 4px; color: {PALETTE.text}; font-size: 12px; }}"
                "QPushButton:hover { background: rgba(255,255,255,0.25); }")
        restore.clicked.connect(self.restore_requested.emit)
        close.clicked.connect(self.close)
        bar.addWidget(restore)
        bar.addWidget(close)
        self._overlay.hide()

        # Always-visible status dot (tiny, top-left): REC red / LIVE green.
        self._dot = QLabel("●", self)
        self._dot.setStyleSheet(
            f"color: {PALETTE.success}; font-size: 11px; background: transparent;")
        self._dot.move(8, 4)

        # Resize grip, overlaid bottom-right; aspect snap happens in resizeEvent.
        self._grip = QSizeGrip(self)
        self._grip.setFixedSize(16, 16)
        self._grip.setStyleSheet("background: transparent;")

        self.setMouseTracking(True)
        self._video.setMouseTracking(True)

    # ------------------------------------------------------------------ #
    # Feed
    # ------------------------------------------------------------------ #
    def on_frame(self, packet) -> None:
        if not self.isVisible():
            return
        # Half the feed rate and bounded pixels: this is a glance monitor, not
        # an editing surface. Full-res frames at 30fps into a ~380px window
        # measurably lagged the mouse — all of it lands on the UI thread.
        self._tick = getattr(self, "_tick", 0) + 1
        if self._tick % 2:
            return
        img = packet.birdseye if (self._show_birdseye
                                  and packet.birdseye is not None) \
            else packet.perspective
        if img is None:
            return
        if img.shape[1] > 2 * self.width():
            import cv2
            scale = 2 * self.width() / img.shape[1]
            img = cv2.resize(img, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        if h > 0 and w > 0:
            aspect = w / h
            if abs(aspect - self._aspect) > 0.01:
                self._aspect = aspect
                self._snap_to_aspect()
        self._video.set_frame(img)

    def on_recording(self, on: bool) -> None:
        self._recording = on
        self._rec_btn.setChecked(on)
        self._rec_btn.setText("■ STOP" if on else "● REC")
        self._refresh_dot()

    def on_status(self, status: str) -> None:
        self._running = status == "running"
        self._refresh_dot()

    def _refresh_dot(self) -> None:
        color = (PALETTE.danger if self._recording
                 else PALETTE.success if self._running else PALETTE.text_dim)
        self._dot.setStyleSheet(
            f"color: {color}; font-size: 11px; background: transparent;")

    # ------------------------------------------------------------------ #
    # The window hugs the video: height follows width by aspect ratio.
    # ------------------------------------------------------------------ #
    def _snap_to_aspect(self) -> None:
        if self._in_aspect_snap:
            return
        self._in_aspect_snap = True
        try:
            w = self.width()
            self.resize(w, max(self.minimumHeight(), round(w / self._aspect)))
        finally:
            self._in_aspect_snap = False

    def resizeEvent(self, ev):  # noqa: N802 - Qt override
        self._snap_to_aspect()
        self._overlay.setGeometry(0, 0, self.width(), 36)
        self._grip.move(self.width() - 16, self.height() - 16)
        super().resizeEvent(ev)

    # ------------------------------------------------------------------ #
    # Hover chrome + frameless ergonomics
    # ------------------------------------------------------------------ #
    def enterEvent(self, ev):  # noqa: N802 - Qt override
        self._overlay.show()
        self._overlay.raise_()
        self._grip.raise_()
        super().enterEvent(ev)

    def leaveEvent(self, ev):  # noqa: N802 - Qt override
        self._overlay.hide()
        super().leaveEvent(ev)

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
