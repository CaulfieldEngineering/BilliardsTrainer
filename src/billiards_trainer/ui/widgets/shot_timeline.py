"""DAW-style shot timeline: the session as a strip of clips.

Joe's spec, verbatim intent: distinguish shots from one another on a
timeline "a la a DAW" — a hit is detected (cue-ball movement + mic clatter
+ cue sensor when present), a configurable PRE-SHOT ROUTINE window precedes
it, and the clip runs until every ball stops rolling or is pocketed.

This widget is the visible half: shots render as rounded clips on a time
ruler — green = make, slate = miss, amber = scratch — each with a dimmer
lead-in block for the pre-shot routine. Clicking a clip seeks playback to
the START OF THE ROUTINE (you want to watch the setup, not join mid-shot).

The data half is the audio-validated ShotDetector: its start_t/end_t are
the strike and the all-balls-settled moment. The pre-roll is display-side
(``detection.pre_shot_s``), so re-tuning it never touches detection.
"""

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

from ..theme import PALETTE

_OUTCOME_COLOURS = {
    "make": "#3FB950",
    "miss": "#7D8590",
    "scratch": "#E3B341",
}


class ShotTimeline(QWidget):
    """Clips on a ruler. ``clicked(seconds)`` asks the owner to seek."""

    clicked = Signal(float)

    def __init__(self, pre_roll_s: float = 5.0, parent=None):
        super().__init__(parent)
        self.setFixedHeight(34)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Shot timeline — click a clip to replay that shot "
                        "from the start of your pre-shot routine")
        self._duration = 0.0
        self._playhead = -1.0
        self._shots: list[dict] = []
        self.pre_roll_s = pre_roll_s

    # ------------------------------------------------------------------ #
    def set_duration(self, seconds: float) -> None:
        self._duration = max(0.0, float(seconds))
        self.update()

    def set_playhead(self, seconds: float) -> None:
        self._playhead = float(seconds)
        self.update()

    def add_shot(self, start_t: float, end_t: float, outcome: str,
                 num_pocketed: int = 0) -> None:
        self._shots.append({"start": float(start_t), "end": float(end_t),
                            "outcome": outcome, "pocketed": int(num_pocketed)})
        # A live session's duration grows with its newest shot.
        if end_t > self._duration:
            self._duration = float(end_t)
        self.update()

    def clear(self) -> None:
        self._shots.clear()
        self._playhead = -1.0
        self.update()

    @property
    def shots(self) -> list[dict]:
        return list(self._shots)

    # ------------------------------------------------------------------ #
    def _x(self, t: float) -> float:
        if self._duration <= 0:
            return 0.0
        return max(0.0, min(1.0, t / self._duration)) * self.width()

    def _t(self, x: float) -> float:
        if self.width() <= 0:
            return 0.0
        return max(0.0, x / self.width()) * self._duration

    def shot_at(self, t: float) -> dict | None:
        """The shot whose clip (including pre-roll) covers time ``t``."""
        for s in self._shots:
            if s["start"] - self.pre_roll_s <= t <= s["end"]:
                return s
        return None

    # ------------------------------------------------------------------ #
    def mousePressEvent(self, ev):  # noqa: N802 - Qt override
        if ev.button() != Qt.LeftButton or self._duration <= 0:
            return
        t = self._t(ev.position().x() if hasattr(ev, "position") else ev.pos().x())
        s = self.shot_at(t)
        # Clip click -> start of the pre-shot routine; bare-ruler click -> there.
        self.clicked.emit(max(0.0, s["start"] - self.pre_roll_s) if s else t)

    def paintEvent(self, ev):  # noqa: N802 - Qt override
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        # ruler bed
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(PALETTE.surface))
        p.drawRoundedRect(QRectF(0, h * 0.30, w, h * 0.40), 4, 4)
        if self._duration > 0:
            # minute ticks
            p.setPen(QPen(QColor(PALETTE.border), 1))
            t = 60.0
            while t < self._duration:
                x = self._x(t)
                p.drawLine(int(x), int(h * 0.30), int(x), int(h * 0.70))
                t += 60.0
            # clips
            for s in self._shots:
                colour = QColor(_OUTCOME_COLOURS.get(s["outcome"], PALETTE.text_dim))
                x0 = self._x(max(0.0, s["start"] - self.pre_roll_s))
                x1 = self._x(s["start"])
                x2 = self._x(max(s["end"], s["start"] + 0.5))
                lead = QColor(colour)
                lead.setAlpha(70)
                p.setPen(Qt.NoPen)
                p.setBrush(lead)                                  # pre-shot routine
                p.drawRoundedRect(QRectF(x0, h * 0.22, max(1.0, x1 - x0), h * 0.56), 3, 3)
                p.setBrush(colour)                                # the shot itself
                p.drawRoundedRect(QRectF(x1, h * 0.14, max(2.0, x2 - x1), h * 0.72), 3, 3)
            # playhead
            if self._playhead >= 0:
                x = self._x(self._playhead)
                p.setPen(QPen(QColor(PALETTE.text), 2))
                p.drawLine(int(x), 2, int(x), h - 2)
        p.end()
