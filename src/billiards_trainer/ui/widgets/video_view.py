"""A widget that displays OpenCV BGR frames, scaled to fit with letterboxing.

Uses ``QImage.Format_BGR888`` so no colour conversion copy is needed. Keeps a
reference to the current ndarray so its buffer stays alive while Qt paints it.
"""


import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..theme import PALETTE


class VideoView(QWidget):
    # Emitted on click with normalised image coords (0..1), accounting for the
    # letterbox. Used by the felt colour picker.
    clicked = Signal(float, float)

    def __init__(self, placeholder: str = "No signal", parent=None):
        super().__init__(parent)
        self._buf: np.ndarray | None = None
        self._pixmap: QPixmap | None = None
        self._placeholder = placeholder
        self._draw_rect: QRectF | None = None  # where the image is painted
        self._overlay: list = []   # [(x, y, r, text, selected)] in image px (Qt-drawn)
        self._balls: list = []     # [(x, y, r, label)] in image px — hover-to-reveal
        self._mouse_pos: QPointF | None = None
        self._pickable = False
        self.setMinimumSize(120, 90)   # small floor so the WINDOW can get small
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setMouseTracking(True)   # hover events without a pressed button

    def set_pickable(self, on: bool) -> None:
        self._pickable = on
        self.setCursor(Qt.CrossCursor if on else Qt.ArrowCursor)

    def mousePressEvent(self, event) -> None:
        if self._pickable and self._draw_rect is not None and self._pixmap is not None:
            pos = event.position()
            if self._draw_rect.contains(pos):
                xf = (pos.x() - self._draw_rect.x()) / self._draw_rect.width()
                yf = (pos.y() - self._draw_rect.y()) / self._draw_rect.height()
                self.clicked.emit(float(xf), float(yf))
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        # Only the schematic bird's-eye sets balls; elsewhere this is a no-op.
        if self._balls:
            self._mouse_pos = event.position()
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:
        if self._mouse_pos is not None:
            self._mouse_pos = None
            self.update()
        super().leaveEvent(event)

    def set_balls(self, balls: list) -> None:
        """Ball markers for hover-to-reveal: [(x, y, r, label)] in image pixels.
        Hovering a ball shows its number in a small chip (numbers are deliberately
        off the balls themselves)."""
        self._balls = balls or []
        if not self._balls and self._mouse_pos is not None:
            self._mouse_pos = None
        # no forced repaint here — set_frame() already drives the per-frame update

    def set_frame(self, frame: np.ndarray) -> None:
        if frame is None or frame.size == 0:
            return
        # A hidden view converting full-res frames 30x/second is pure UI-thread
        # tax (mouse events queue behind paint work — felt as cursor lag). The
        # next frame after becoming visible repaints; nothing is lost.
        if not self.isVisible():
            return
        buf = np.ascontiguousarray(frame)
        self._buf = buf  # keep buffer alive
        h, w = buf.shape[:2]
        if buf.ndim == 2:
            img = QImage(buf.data, w, h, buf.strides[0], QImage.Format_Grayscale8)
        else:
            img = QImage(buf.data, w, h, buf.strides[0], QImage.Format_BGR888)
        self._pixmap = QPixmap.fromImage(img)
        self.update()

    def image_size(self) -> tuple[int, int] | None:
        """(width, height) of the currently displayed frame in image pixels, or
        None if no frame is shown. The labeller maps clicks against this so it
        never relies on a separately-tracked size that can go stale."""
        if self._pixmap is None:
            return None
        return (self._pixmap.width(), self._pixmap.height())

    def set_overlay(self, items: list) -> None:
        """Set labelling markers drawn with Qt (NOT OpenCV) over the frame, so no
        cv2 call happens on the UI thread. items: [(x, y, r, text, selected)] in
        image pixels."""
        self._overlay = items or []
        self.update()

    def clear(self) -> None:
        self._buf = None
        self._pixmap = None
        self._overlay = []
        self._balls = []
        self._mouse_pos = None
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)
        rect = self.rect()
        p.fillRect(rect, QColor("#06080B"))
        if self._pixmap is None:
            p.setPen(QColor(PALETTE.text_faint))
            p.drawText(rect, Qt.AlignCenter, self._placeholder)
            p.end()
            return
        scaled = self._pixmap.size().scaled(rect.size(), Qt.KeepAspectRatio)
        x = (rect.width() - scaled.width()) // 2
        y = (rect.height() - scaled.height()) // 2
        self._draw_rect = QRectF(x, y, scaled.width(), scaled.height())
        p.drawPixmap(self._draw_rect, self._pixmap, QRectF(self._pixmap.rect()))
        if self._overlay:
            pw = max(1, self._pixmap.width())
            sx = self._draw_rect.width() / pw
            for (ox, oy, orr, text, sel) in self._overlay:
                cx = self._draw_rect.x() + ox * sx
                cy = self._draw_rect.y() + oy * sx
                rr = max(3.0, orr * sx)
                col = QColor(0, 255, 255) if sel else QColor(60, 220, 60)
                p.setPen(QPen(col, 3 if sel else 2))
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(QRectF(cx - rr, cy - rr, 2 * rr, 2 * rr))
                if text:
                    p.setFont(QFont("Arial", max(8, int(rr)), QFont.Bold))
                    p.drawText(QRectF(cx - rr, cy - rr - 22, 2 * rr + 60, 20),
                               Qt.AlignLeft | Qt.AlignVCenter, text)
        self._draw_hover_label(p, rect)
        p.end()

    def _draw_hover_label(self, p: QPainter, rect) -> None:
        """If the cursor is over a ball, reveal its number in a small chip above it."""
        if not (self._balls and self._mouse_pos is not None
                and self._draw_rect is not None and self._pixmap is not None):
            return
        pw = max(1, self._pixmap.width())
        ph = max(1, self._pixmap.height())
        sx = self._draw_rect.width() / pw
        sy = self._draw_rect.height() / ph
        mx, my = self._mouse_pos.x(), self._mouse_pos.y()
        best, best_d = None, 1e18
        for (bx, by, br, label) in self._balls:
            wx = self._draw_rect.x() + bx * sx
            wy = self._draw_rect.y() + by * sy
            d = ((wx - mx) ** 2 + (wy - my) ** 2) ** 0.5
            if d <= max(12.0, br * sx * 1.4) and d < best_d:
                best, best_d = (wx, wy, br * sx, str(label)), d
        if best is None:
            return
        wx, wy, wr, label = best
        # marker ring on the hovered ball
        p.setPen(QPen(QColor(255, 255, 255, 200), 2))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(wx, wy), max(6.0, wr), max(6.0, wr))
        # chip above the ball
        p.setFont(QFont("Arial", 11, QFont.Bold))
        fm = p.fontMetrics()
        tw = fm.horizontalAdvance(label)
        ch = fm.height() + 8
        cw = tw + 16
        cx = min(max(2.0, wx - cw / 2), rect.width() - cw - 2)
        cy = max(2.0, wy - max(14.0, wr) - ch - 6)
        chip = QRectF(cx, cy, cw, ch)
        p.setBrush(QColor(18, 22, 28, 235))
        p.setPen(QPen(QColor(120, 150, 175), 1))
        p.drawRoundedRect(chip, 6, 6)
        p.setPen(QColor(240, 244, 250))
        p.drawText(chip, Qt.AlignCenter, label)
