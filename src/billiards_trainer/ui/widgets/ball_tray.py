"""Matchroom-style ball tray (Joe): 9-ball rack chips 1-9, present on
the table vs potted, live during play.

Presence logic is pure and debounced: a ball is PRESENT until it has
been unseen for ABSENT_S (occlusion by an arm must not flicker the
tray), and a never-yet-seen ball is presumed present (a mis-ID at the
rack must not show a phantom pot). A potted ball that reappears
(respot, or the tracker recovering) turns present again.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

from ..theme import PALETTE

ABSENT_S = 3.0

BALL_COLORS = {
    1: "#FDD835", 2: "#1E88E5", 3: "#E53935", 4: "#8E24AA", 5: "#FB8C00",
    6: "#43A047", 7: "#8D6E63", 8: "#212121", 9: "#FDD835",
}


class BallPresence:
    """Pure presence tracker: update(seen_numbers, t) -> {n: present}."""

    def __init__(self, numbers=tuple(range(1, 10)), absent_s: float = ABSENT_S):
        self._numbers = tuple(numbers)
        self._absent_s = absent_s
        self._last_seen: dict[int, float] = {}
        self._ever_seen: set[int] = set()

    def update(self, seen, t: float) -> dict[int, bool]:
        for n in seen:
            if n in self._numbers:
                self._last_seen[n] = t
                self._ever_seen.add(n)
        out = {}
        for n in self._numbers:
            if n not in self._ever_seen:
                out[n] = True            # presumed racked until proven gone
            else:
                out[n] = t - self._last_seen.get(n, -1e9) < self._absent_s
        return out

    def reset(self) -> None:
        self._last_seen.clear()
        self._ever_seen.clear()


class BallTrayWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(34)
        self._present: dict[int, bool] = {n: True for n in range(1, 10)}

    def set_presence(self, present: dict) -> None:
        if present != self._present:
            self._present = dict(present)
            self.update()

    def paintEvent(self, _ev) -> None:  # noqa: N802 - Qt override
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        n = len(self._present) or 9
        d = min(self.height() - 8, max(14, (self.width() - 8) // n - 8))
        gap = (self.width() - n * d) / (n + 1)
        y = (self.height() - d) / 2
        f = QFont()
        f.setPointSize(max(7, int(d * 0.42)))
        f.setBold(True)
        p.setFont(f)
        x = gap
        for num in sorted(self._present):
            here = self._present[num]
            col = QColor(BALL_COLORS.get(num, "#9AA4B2"))
            if here:
                p.setBrush(col)
                p.setPen(QPen(QColor(PALETTE.border), 1))
            else:
                p.setBrush(Qt.NoBrush)   # potted: hollow ghost
                ghost = QColor(col)
                ghost.setAlpha(70)
                p.setPen(QPen(ghost, 1.5))
            p.drawEllipse(int(x), int(y), int(d), int(d))
            if num == 9 and here:        # the money ball's stripe
                p.setPen(QPen(QColor("#FFFFFF"), max(2, int(d * 0.16))))
                p.drawLine(int(x + d * 0.15), int(y + d / 2),
                           int(x + d * 0.85), int(y + d / 2))
            p.setPen(QColor("#FFFFFF" if num != 1 and num != 9 else "#212121")
                     if here else QColor(PALETTE.text_faint))
            p.drawText(int(x), int(y), int(d), int(d), Qt.AlignCenter, str(num))
            x += d + gap
        p.end()
