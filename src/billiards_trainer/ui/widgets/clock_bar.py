"""Matchroom-style shot-clock bar (Joe): a full-width strip under the
video that DEPLETES as the countdown runs — the broadcast look. Hidden
whenever no countdown is active, so free play costs no pixels.
"""

from __future__ import annotations

from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QWidget

from ..theme import PALETTE


class ClockBarWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(8)
        self._frac = 0.0
        self._warning = False
        self._expired = False
        self.hide()

    def set_state(self, remaining: float, total: float,
                  warning: bool, running: bool) -> None:
        if not running:
            if self.isVisible():
                self.hide()
            return
        frac = max(0.0, min(1.0, remaining / max(1.0, total)))
        expired = remaining <= 0.0
        if (abs(frac - self._frac) > 0.003 or warning != self._warning
                or expired != self._expired or not self.isVisible()):
            self._frac, self._warning, self._expired = frac, warning, expired
            self.show()
            self.update()

    def paintEvent(self, _ev) -> None:  # noqa: N802 - Qt override
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(PALETTE.border))
        col = (PALETTE.danger if self._expired or self._frac <= 0.12
               else PALETTE.warn if self._warning else PALETTE.accent)
        w = int(self.width() * self._frac)
        # deplete from both ends toward the center - the Matchroom read
        x = (self.width() - w) // 2
        p.fillRect(x, 0, w, self.height(), QColor(col))
        p.end()
