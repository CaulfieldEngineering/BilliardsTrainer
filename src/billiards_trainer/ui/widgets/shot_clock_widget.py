"""Circular shot-clock countdown ring (QPainter)."""

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..theme import PALETTE


class ShotClockWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._remaining = 0.0
        self._total = 30.0
        self._warning = False
        self._enabled = False
        self.setMinimumSize(96, 96)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.resize(96, 96)

    def update_clock(self, remaining: float, total: float, warning: bool, enabled: bool) -> None:
        self._remaining = max(0.0, remaining)
        self._total = max(1.0, total)
        self._warning = warning
        self._enabled = enabled
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        side = min(self.width(), self.height())
        margin = 8
        rect = QRectF(margin, margin, side - 2 * margin, side - 2 * margin)

        track = QColor(PALETTE.border)
        if not self._enabled:
            accent = QColor(PALETTE.text_faint)
        elif self._remaining <= 0:
            accent = QColor(PALETTE.danger)
        elif self._warning:
            accent = QColor(PALETTE.warn)
        else:
            accent = QColor(PALETTE.accent)

        p.setPen(QPen(track, 6))
        p.drawArc(rect, 0, 360 * 16)

        frac = self._remaining / self._total if self._enabled else 0.0
        span = int(-360 * 16 * frac)
        p.setPen(QPen(accent, 6, Qt.SolidLine, Qt.RoundCap))
        p.drawArc(rect, 90 * 16, span)

        p.setPen(accent if self._enabled else QColor(PALETTE.text_faint))
        f = QFont()
        f.setPointSize(int(side * 0.22))
        f.setBold(True)
        p.setFont(f)
        text = f"{int(self._remaining)}" if self._enabled else "—"
        p.drawText(rect, Qt.AlignCenter, text)
        p.end()
