"""Cue-sensor diagnostics: live waveforms + stream health.

The main UI shows only per-shot stroke STATS; the raw waveforms live here, for
checking the mounting, stream rate, and impact detection while setting up.
Two lanes over a rolling 12 s window: motion energy (|a| deviation from 1 g)
and rotation rate (|gyro| dps), with detected cue-ball impacts marked.

Subscribes directly to the CueSensorWorker's signals (sample/impact/status);
everything is disconnected again on close so an idle dialog costs nothing.
"""

import math
import time
from collections import deque

from PySide6.QtCore import QPointF, QTimer
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ..theme import PALETTE

_WINDOW_S = 12.0


class _WavePane(QWidget):
    """One scrolling waveform lane."""

    def __init__(self, title: str, color: str, scale: float, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(120)
        self._title = title
        self._color = color
        self._scale = scale          # full-lane value (autoscales up beyond it)
        self.points: deque = deque()  # (t, value)
        self.marks: deque = deque()   # impact times

    def add(self, t: float, v: float) -> None:
        self.points.append((t, v))
        while self.points and self.points[0][0] < t - _WINDOW_S:
            self.points.popleft()
        while self.marks and self.marks[0] < t - _WINDOW_S:
            self.marks.popleft()

    def paintEvent(self, _ev) -> None:  # noqa: N802 - Qt override
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(PALETTE.bg_elevated))
        p.setPen(QPen(QColor(PALETTE.text_faint)))
        p.drawText(8, 16, self._title)
        if not self.points:
            p.setPen(QPen(QColor(PALETTE.text_dim)))
            p.drawText(8, h // 2, "no data")
            return
        now = self.points[-1][0]
        peak = max(v for _t, v in self.points)
        scale = max(self._scale, peak * 1.1)
        pen = QPen(QColor(self._color))
        pen.setWidthF(1.6)
        p.setPen(pen)
        last = None
        for t, v in self.points:
            x = w - (now - t) / _WINDOW_S * w
            y = h - 6 - (min(v, scale) / scale) * (h - 28)
            if last is not None:
                p.drawLine(QPointF(last[0], last[1]), QPointF(x, y))
            last = (x, y)
        mark_pen = QPen(QColor(PALETTE.danger))
        mark_pen.setWidthF(1.4)
        p.setPen(mark_pen)
        for mt in self.marks:
            x = w - (now - mt) / _WINDOW_S * w
            p.drawLine(QPointF(x, 20.0), QPointF(x, float(h - 4)))
        p.setPen(QPen(QColor(PALETTE.text_faint)))
        p.drawText(w - 64, 16, f"scale {scale:.0f}")


class CueDiagnosticsDialog(QDialog):
    def __init__(self, worker, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cue sensor diagnostics")
        self.resize(720, 420)
        self._worker = worker
        self._n_accel = deque()  # sample arrival times, for the rate readout

        lay = QVBoxLayout(self)
        lay.setSpacing(10)
        self._status = QLabel("Waiting for sensor…")
        self._status.setObjectName("Muted")
        lay.addWidget(self._status)

        self._accel_pane = _WavePane("MOTION  |a|−1g  (g)", PALETTE.accent, 2.0)
        self._gyro_pane = _WavePane("ROTATION  |ω|  (dps)", PALETTE.info, 300.0)
        lay.addWidget(self._accel_pane, 1)
        lay.addWidget(self._gyro_pane, 1)

        row = QHBoxLayout()
        self._rate = QLabel("rate: —")
        self._last_hit = QLabel("last impact: —")
        for x in (self._rate, self._last_hit):
            x.setObjectName("Faint")
            row.addWidget(x)
        row.addStretch(1)
        hint = QLabel("red marks = detected cue-ball impacts")
        hint.setObjectName("Faint")
        row.addWidget(hint)
        lay.addLayout(row)

        worker.sample.connect(self._on_sample)
        worker.impact.connect(self._on_impact)
        worker.status_changed.connect(self._on_status)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._repaint)
        self._timer.start(50)

    # ------------------------------------------------------------------ #
    def _on_sample(self, kind: str, d: dict, t: float) -> None:
        mag = math.sqrt(d["x"] ** 2 + d["y"] ** 2 + d["z"] ** 2)
        if kind == "accel":
            self._accel_pane.add(t, abs(mag - 1.0))
            self._n_accel.append(t)
            while self._n_accel and self._n_accel[0] < t - 2.0:
                self._n_accel.popleft()
        else:
            self._gyro_pane.add(t, mag)

    def _on_impact(self, stroke: dict) -> None:
        t = stroke.get("hit_t", time.time())
        self._accel_pane.marks.append(t)
        self._gyro_pane.marks.append(t)
        self._last_hit.setText(f"last impact: {stroke.get('peak_g', 0.0):.1f} g")

    def _on_status(self, state: str, detail) -> None:
        if state == "connected" and isinstance(detail, dict):
            batt = detail.get("battery")
            self._status.setText(
                f"Connected: {detail.get('name') or detail.get('address', '')}"
                + (f" · battery {batt}%" if batt is not None else ""))
        else:
            self._status.setText(f"Status: {state}"
                                 + (f" — {detail}" if detail else ""))

    def _repaint(self) -> None:
        if self._n_accel:
            self._rate.setText(f"rate: {len(self._n_accel) / 2.0:.0f} Hz")
        self._accel_pane.update()
        self._gyro_pane.update()

    # ------------------------------------------------------------------ #
    def closeEvent(self, ev) -> None:  # noqa: N802 - Qt override
        try:
            self._worker.sample.disconnect(self._on_sample)
            self._worker.impact.disconnect(self._on_impact)
            self._worker.status_changed.disconnect(self._on_status)
        except (RuntimeError, TypeError):
            pass  # already disconnected
        self._timer.stop()
        super().closeEvent(ev)
