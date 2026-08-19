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
    #: Editor scrubbing (v3.1): press-drag the lane to scrub. The page pauses
    #: playback for the duration (no more playhead fighting) and every drag
    #: position becomes a real seek — so Space afterwards resumes from THERE.
    scrub_started = Signal()
    scrubbed = Signal(float)          # seconds under the cursor
    scrub_ended = Signal()

    #: Number of filmstrip thumbnails sampled across the visible range.
    STRIP_N = 12

    def __init__(self, pre_roll_s: float = 5.0, parent=None):
        super().__init__(parent)
        # v3 (Joe: "a visual timeline of the video, with thumbnails, like a
        # video editing system, and the entirety of the shot highlighted"):
        # a filmstrip band under translucent full-shot regions. v3.2 (Joe:
        # "a tighter rectangle across the top... a little too much padding
        # and negative space"): compact lane — the filmstrip stays legible
        # at this height and the ruler proportions are relative already.
        self.setFixedHeight(78)
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
        #: Zoomed view window (lo, hi) in seconds, or None = whole session.
        #: Editor controls: wheel zooms around the cursor, middle-drag pans.
        self._view: tuple[float, float] | None = None
        self._pan_anchor_x: float | None = None
        #: Filmstrip: media path + {rounded-second: QImage} cache, filled by
        #: a worker thread; painting only ever READS the cache.
        self._media: str = ""
        self._strip: dict[int, object] = {}
        self._strip_pending: set[int] = set()
        self._press_x: float | None = None
        self._dragging = False

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
        self._view = None
        self.update()

    @property
    def shots(self) -> list[dict]:
        return list(self._shots)

    # ------------------------------------------------------------------ #
    def _range(self) -> tuple[float, float]:
        """Visible time range: live rolling window > zoomed view > everything."""
        if self.follow_window_s > 0 and self._live_now > self.follow_window_s:
            return self._live_now - self.follow_window_s, self._live_now
        if self._view is not None:
            return self._view
        return 0.0, max(self._duration, self.follow_window_s or self._duration)

    # -- editor zoom/pan (pure methods; event handlers are thin shells) ----- #
    MIN_SPAN_S = 10.0

    def zoom_at(self, x_px: float, factor: float) -> None:
        """Scale the visible span by 1/``factor`` keeping the time under the
        cursor at the same screen position (video-editor wheel zoom)."""
        if self._duration <= 0 or self.follow_window_s > 0:
            return                      # the live lane owns its own window
        lo, hi = self._range()
        span = hi - lo
        new_span = span / max(1e-6, factor)
        if new_span >= self._duration:
            self._view = None           # zoomed all the way back out
            self.update()
            return
        new_span = max(self.MIN_SPAN_S, new_span)
        frac = min(1.0, max(0.0, x_px / max(1, self.width())))
        t = lo + frac * span
        new_lo = t - frac * new_span
        new_lo = min(max(0.0, new_lo), self._duration - new_span)
        self._view = (new_lo, new_lo + new_span)
        self.update()

    def pan_by(self, dx_px: float) -> None:
        """Slide a zoomed view sideways (middle-drag). Content follows the
        cursor: dragging right shows earlier time. No-op when not zoomed."""
        if self._view is None or self._duration <= 0:
            return
        lo, hi = self._view
        span = hi - lo
        dt = -dx_px * span / max(1, self.width())
        new_lo = min(max(0.0, lo + dt), self._duration - span)
        self._view = (new_lo, new_lo + span)
        self.update()

    def view_range(self) -> tuple[float, float]:
        return self._range()

    # -- filmstrip (v3) -------------------------------------------------- #
    def strip_times(self, lo: float, hi: float) -> list[int]:
        """Rounded-second sample times for STRIP_N thumbnails across a view
        (pure; stable keys = stable cache across repaints and small pans)."""
        if hi <= lo:
            return []
        step = (hi - lo) / self.STRIP_N
        return sorted({int(lo + (i + 0.5) * step) for i in range(self.STRIP_N)})

    def set_media_source(self, path: str) -> None:
        """Point the filmstrip at a video file ('' = live, no filmstrip)."""
        if path == self._media:
            return
        self._media = path or ""
        self._strip.clear()
        self._strip_pending.clear()
        self.update()

    def _request_strip(self, times: list[int]) -> None:
        missing = [t for t in times
                   if t not in self._strip and t not in self._strip_pending]
        if not missing or not self._media:
            return
        self._strip_pending.update(missing)
        import threading

        from PySide6.QtCore import QObject, Signal as _Sig

        if not hasattr(self, "_strip_bridge"):
            class _Bridge(QObject):
                ready = _Sig(int, object)
            self._strip_bridge = _Bridge(self)
            self._strip_bridge.ready.connect(self._on_strip_ready)

        def work(media=self._media, want=list(missing), bridge=self._strip_bridge):
            import cv2
            from PySide6.QtGui import QImage
            cap = cv2.VideoCapture(media)
            if not cap.isOpened():
                return
            for sec in sorted(want):
                cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000.0)
                ok, fr = cap.read()
                if not ok:
                    continue
                fr = cv2.resize(fr, (96, 54), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB).copy()
                img = QImage(rgb.data, 96, 54, 3 * 96, QImage.Format_RGB888).copy()
                bridge.ready.emit(sec, img)
            cap.release()

        threading.Thread(target=work, daemon=True, name="timeline-strip").start()

    def _on_strip_ready(self, sec: int, img) -> None:
        self._strip_pending.discard(sec)
        if len(self._strip) > 400:          # bounded cache across zooms
            self._strip.clear()
        self._strip[sec] = img
        self.update()

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
        """Hover-card text for time ``t`` — SHOT REGIONS ONLY (the always-on
        instructional tooltip was noise; Joe: "I don't need the tooltip to
        be so prevalent"). Pure, for tests."""
        if self._duration <= 0:
            return None
        s = self.shot_at(t)
        if s is None:
            return None
        no = self._shots.index(s) + 1
        start = s["start"]
        mm, ss = int(start) // 60, int(start) % 60
        dur = max(0.0, s["end"] - start)
        pot = int(s.get("pocketed", 0))
        return (f"Shot {no} — {s['outcome'].upper()}"
                + (" (corrected)" if s.get("corrected") else "")
                + f"  ·  {mm}:{ss:02d} · {dur:.1f}s"
                + (f" · {pot} potted" if pot else ""))

    def wheelEvent(self, ev):  # noqa: N802 - Qt override
        x = ev.position().x() if hasattr(ev, "position") else ev.pos().x()
        notches = ev.angleDelta().y() / 120.0
        if notches:
            self.zoom_at(float(x), 1.25 ** notches)
        ev.accept()

    def mouseMoveEvent(self, ev):  # noqa: N802 - Qt override
        from PySide6.QtWidgets import QToolTip
        pos = ev.position().toPoint() if hasattr(ev, "position") else ev.pos()
        if self._pan_anchor_x is not None and (ev.buttons() & Qt.MiddleButton):
            self.pan_by(float(pos.x()) - self._pan_anchor_x)
            self._pan_anchor_x = float(pos.x())
            return
        if self._press_x is not None and (ev.buttons() & Qt.LeftButton):
            if not self._dragging and abs(float(pos.x()) - self._press_x) > 4:
                self._dragging = True
                self.scrub_started.emit()
            if self._dragging:
                t = self._t(float(pos.x()))
                self.set_playhead(t)          # the lane follows the finger
                self.scrubbed.emit(t)
                return
        text = self.hover_text(self._t(float(pos.x())))
        if text:
            QToolTip.showText(self.mapToGlobal(pos), text, self)
        else:
            QToolTip.hideText()

    def mouseReleaseEvent(self, ev):  # noqa: N802 - Qt override
        if ev.button() == Qt.MiddleButton:
            self._pan_anchor_x = None
        elif ev.button() == Qt.LeftButton and self._press_x is not None:
            x = ev.position().x() if hasattr(ev, "position") else ev.pos().x()
            if self._dragging:
                t = self._t(float(x))
                self.scrubbed.emit(t)
                self.scrub_ended.emit()
            else:
                t = self._t(float(x))
                s = self.shot_at(t)
                # click -> routine start on a shot; raw time on bare lane
                self.clicked.emit(max(0.0, s["start"] - self.pre_roll_s) if s else t)
            self._press_x = None
            self._dragging = False

    def mousePressEvent(self, ev):  # noqa: N802 - Qt override
        x = ev.position().x() if hasattr(ev, "position") else ev.pos().x()
        if ev.button() == Qt.MiddleButton:
            self._pan_anchor_x = float(x)     # start an editor pan
            return
        if ev.button() != Qt.LeftButton or self._duration <= 0:
            return
        # Nothing is decided at press: a CLICK (release nearby) snaps a shot
        # to its routine start; a DRAG scrubs raw time. Deciding on release
        # lets both coexist.
        self._press_x = float(x)
        self._dragging = False

    def paintEvent(self, ev):  # noqa: N802 - Qt override
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        # ruler bed (below the filmstrip band)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(PALETTE.surface))
        p.drawRoundedRect(QRectF(0, h * 0.66, w, h * 0.22), 4, 4)
        if self._duration > 0:
            lo, hi = self._range()
            span = max(1.0, hi - lo)
            # FILMSTRIP band (v3): the video itself as thumbnails across the
            # visible range — the timeline reads like an editor, and shots
            # highlight ON the footage instead of floating as bare bars.
            strip_top, strip_h = 6.0, h * 0.52
            if self._media and self.follow_window_s <= 0:
                times = self.strip_times(lo, hi)
                self._request_strip(times)
                slot_w = w / max(1, len(times))
                for k, sec in enumerate(times):
                    img = self._strip.get(sec)
                    slot = QRectF(k * slot_w, strip_top, slot_w - 1.0, strip_h)
                    if img is not None:
                        p.setOpacity(0.72)
                        p.drawImage(slot, img)
                        p.setOpacity(1.0)
                    else:
                        p.setPen(Qt.NoPen)
                        p.setBrush(QColor(PALETTE.surface))
                        p.drawRect(slot)
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
                p.drawLine(int(x), int(h * 0.68), int(x), int(h * 0.86))
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
                # THE ENTIRE SHOT, highlighted (Joe's ask): a translucent
                # region over the filmstrip spanning strike -> settled, a
                # dimmer lead-in for the routine, a bright strike tick, and
                # a solid outcome bar underlining the region.
                region_top, region_h = 4.0, h * 0.56
                lead = QColor(colour)
                lead.setAlpha(36)
                p.setPen(Qt.NoPen)
                p.setBrush(lead)
                p.drawRect(QRectF(x0, region_top, max(1.0, x1 - x0), region_h))
                body = QColor(colour)
                body.setAlpha(64)
                p.setBrush(body)
                p.drawRect(QRectF(x1, region_top, max(3.0, x2 - x1), region_h))
                p.setBrush(colour)                 # solid outcome underline
                p.drawRoundedRect(QRectF(x1, region_top + region_h,
                                         max(3.0, x2 - x1), 5.0), 2, 2)
                p.setPen(QPen(QColor(colour).lighter(130), 2))
                p.drawLine(int(x1), int(region_top),
                           int(x1), int(region_top + region_h + 5))  # strike
                p.setPen(Qt.NoPen)
                newest = i == len(self._shots) - 1
                if newest and self.follow_window_s > 0:
                    ring = QPen(QColor(PALETTE.accent), 2)
                    p.setPen(ring)
                    p.setBrush(Qt.NoBrush)
                    p.drawRect(QRectF(x1, region_top, max(3.0, x2 - x1), region_h))
                    p.setPen(Qt.NoPen)
                if x2 - x1 >= 16:
                    p.setPen(QColor(255, 255, 255, 220))
                    p.drawText(QRectF(x1, region_top + region_h - 16,
                                      x2 - x1, 16), Qt.AlignCenter, str(i + 1))
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
