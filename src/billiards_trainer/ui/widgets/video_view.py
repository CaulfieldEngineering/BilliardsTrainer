"""A widget that displays OpenCV BGR frames, scaled to fit with letterboxing.

Uses ``QImage.Format_BGR888`` so no colour conversion copy is needed. Keeps a
reference to the current ndarray so its buffer stays alive while Qt paints it.
"""


import numpy as np
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
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
        self._pickable = False
        self.setMinimumSize(160, 120)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)

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

    def set_frame(self, frame: np.ndarray) -> None:
        if frame is None or frame.size == 0:
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

    def clear(self) -> None:
        self._buf = None
        self._pixmap = None
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
        p.end()
