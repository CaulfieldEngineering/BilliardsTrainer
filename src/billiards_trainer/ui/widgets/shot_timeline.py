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
        self.setFixedHeight(72)
        #: While recording, the lane FOLLOWS: it shows the trailing window so
        #: clips stay readable in hour-long sessions, scrolling like a video
        #: editor capture lane (Joe's spec). 0 = show everything (playback).
        self.follow_window_s = 0.0
        self._live_now = 0.0
        self.setCursor(Qt.PointingHandCursor)
        self.setMouseTracking(True)   # hover cards need move events
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

    def set_live_clock(self, seconds: float) -> None:
        """Recording mode: the lane rolls with the record clock."""
        self._live_now = float(seconds)
        if seconds > self._duration:
            self._duration = float(seconds)
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
    def _range(self) -> tuple[float, float]:
        """Visible time range: everything, or the rolling live window."""
        if self.follow_window_s > 0 and self._live_now > self.follow_window_s:
            return self._live_now - self.follow_window_s, self._live_now
        return 0.0, max(self._duration, self.follow_window_s or self._duration)

    def _x(self, t: float) -> float:
        lo, hi = self._range()
        if hi <= lo:
            return 0.0
        return max(0.0, min(1.0, (t - lo) / (hi - lo))) * self.width()

    def _t(self, x: float) -> float:
        lo, hi = self._range()
        if self.width() <= 0:
            return lo
        return lo + max(0.0, x / self.width()) * (hi - lo)

    def shot_at(self, t: float) -> dict | None:
        """The shot whose clip (including pre-roll) covers time ``t``."""
        for s in self._shots:
            if s["start"] - self.pre_roll_s <= t <= s["end"]:
                return s
        return None

    # ------------------------------------------------------------------ #
    def hover_text(self, t: float) -> str | None:
        """Hover-card text for time ``t`` (pure — the event handler is a
        thin shell around this so it can be tested without fake QEvents)."""
        if self._duration <= 0:
            return None
        s = self.shot_at(t)
        if s is None:
            return "Click to seek — clips replay from your pre-shot routine"
        no = self._shots.index(s) + 1
        start = s["start"]
        mm, ss = int(start) // 60, int(start) % 60
        dur = max(0.0, s["end"] - start)
        pot = int(s.get("pocketed", 0))
        nl = "\n"
        return (f"Shot {no} — {s['outcome'].upper()}"
                + (" (corrected)" if s.get("corrected") else "")
                + f"{nl}{mm}:{ss:02d} · {dur:.1f}s"
                + (f" · {pot} potted" if pot else "")
                + f"{nl}Click: replay from routine · "
                  "Right-click list row: export/fix")

    def mouseMoveEvent(self, ev):  # noqa: N802 - Qt override
        from PySide6.QtWidgets import QToolTip
        pos = ev.position().toPoint() if hasattr(ev, "position") else ev.pos()
        text = self.hover_text(self._t(float(pos.x())))
        if text:
            QToolTip.showText(self.mapToGlobal(pos), text, self)
        else:
            QToolTip.hideText()
        super().mouseMoveEvent(ev)

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
        if self._duration <= 0 or not self._shots:
            # Empty state: say what this lane IS instead of presenting a
            # mystery sliver (Joe: "the clip window is small. Is that it?").
            p.setPen(QColor(PALETTE.text_faint))
            p.drawText(self.rect(), Qt.AlignCenter,
                       "Shot timeline — clips appear here as you play; "
                       "click one to replay from your pre-shot routine")
        if self._duration > 0:
            lo, hi = self._range()
            span = max(1.0, hi - lo)
            # Editor-style ruler: labelled ticks at a step that keeps ~6-10
            # labels visible regardless of zoom/window.
            step = 10.0
            for cand in (10.0, 30.0, 60.0, 120.0, 300.0, 600.0):
                if span / cand <= 10:
                    step = cand
                    break
            p.setPen(QPen(QColor(PALETTE.border), 1))
            font = p.font()
            font.setPointSizeF(7.5)
            p.setFont(font)
            t = (int(lo // step) + 1) * step
            while t < hi:
                x = self._x(t)
                p.drawLine(int(x), int(h * 0.34), int(x), int(h * 0.66))
                p.setPen(QColor(PALETTE.text_faint))
                p.drawText(int(x) + 3, int(h * 0.98) - 2,
                           f"{int(t) // 60}:{int(t) % 60:02d}")
                p.setPen(QPen(QColor(PALETTE.border), 1))
                t += step
            # clips — editor styling: filled block, brighter top edge, shot
            # number when there is room; the NEWEST clip gets an accent ring
            # while the lane is live (the "shot detected" highlight).
            for i, s in enumerate(self._shots):
                colour = QColor(_OUTCOME_COLOURS.get(s["outcome"], PALETTE.text_dim))
                x0 = self._x(max(0.0, s["start"] - self.pre_roll_s))
                x1 = self._x(s["start"])
                x2 = self._x(max(s["end"], s["start"] + 0.5))
                if x2 <= 0 or x0 >= w:
                    continue                       # outside the live window
                lead = QColor(colour)
                lead.setAlpha(70)
                p.setPen(Qt.NoPen)
                p.setBrush(lead)                                  # pre-shot routine
                p.drawRoundedRect(QRectF(x0, h * 0.24, max(1.0, x1 - x0), h * 0.44), 3, 3)
                body = QRectF(x1, h * 0.14, max(3.0, x2 - x1), h * 0.62)
                p.setBrush(colour)                                # the shot itself
                p.drawRoundedRect(body, 3, 3)
                top = QColor(255, 255, 255, 60)                   # lit top edge
                p.setBrush(top)
                p.drawRoundedRect(QRectF(body.x(), body.y(), body.width(), 2.5), 1, 1)
                newest = i == len(self._shots) - 1
                if newest and self.follow_window_s > 0:
                    ring = QPen(QColor(PALETTE.accent), 2)
                    p.setPen(ring)
                    p.setBrush(Qt.NoBrush)
                    p.drawRoundedRect(body.adjusted(-1.5, -1.5, 1.5, 1.5), 4, 4)
                    p.setPen(Qt.NoPen)
                if body.width() >= 16:
                    p.setPen(QColor(0, 0, 0, 170))
                    p.drawText(body, Qt.AlignCenter, str(i + 1))
                    p.setPen(Qt.NoPen)
            # playhead (or the live record head at the lane's right edge)
            head = self._live_now if self.follow_window_s > 0 else self._playhead
            if head >= 0:
                x = self._x(head)
                p.setPen(QPen(QColor(PALETTE.text), 2))
                p.drawLine(int(x), 2, int(x), h - 2)
                p.setBrush(QColor(PALETTE.text))
                p.drawRoundedRect(QRectF(x - 3.5, 0, 7, 6), 2, 2)   # editor handle
        p.end()
