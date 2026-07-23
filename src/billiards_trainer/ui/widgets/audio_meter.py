"""A small broadcast-style audio level meter for the media header.

Reads the selected input device with QtMultimedia (QAudioSource), computes RMS
level ~20x/s, and paints a compact segmented bar (green -> yellow -> red) with
a peak-hold tick. Silent-but-working input shows an empty bar; no device or no
permission hides the meter entirely (it must never break the video view).
"""

from __future__ import annotations

import math

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QSizePolicy, QWidget

_FLOOR_DB = -60.0


class AudioMeter(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._level = 0.0        # 0..1 smoothed
        self._peak = 0.0         # 0..1 with slow decay
        self._source = None      # QAudioSource
        self._io = None          # its QIODevice
        self.setFixedSize(74, 12)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setToolTip("Microphone level (recorded into session clips)")
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._poll)
        self.hide()

    # ------------------------------------------------------------------ #
    def configure(self, enabled: bool, device_name: str = "default") -> None:
        """(Re)start monitoring the chosen input, or stop and hide."""
        self._teardown()
        if not enabled:
            return
        try:
            from PySide6.QtMultimedia import (
                QAudioFormat,
                QAudioSource,
                QMediaDevices,
            )
        except Exception:  # noqa: BLE001 - multimedia backend absent
            return
        dev = QMediaDevices.defaultAudioInput()
        if device_name and device_name != "default":
            for d in QMediaDevices.audioInputs():
                if d.description() == device_name:
                    dev = d
                    break
        if dev.isNull():
            return
        fmt = QAudioFormat()
        fmt.setSampleRate(8000)
        fmt.setChannelCount(1)
        fmt.setSampleFormat(QAudioFormat.Int16)
        if not dev.isFormatSupported(fmt):
            fmt = dev.preferredFormat()
        try:
            self._source = QAudioSource(dev, fmt, self)
            self._io = self._source.start()
        except Exception:  # noqa: BLE001
            self._teardown()
            return
        if self._io is None:
            self._teardown()
            return
        self._fmt = fmt
        self._timer.start()
        self.show()

    def _teardown(self) -> None:
        self._timer.stop()
        if self._source is not None:
            try:
                self._source.stop()
            except Exception:  # noqa: BLE001
                pass
        self._source = None
        self._io = None
        self._level = self._peak = 0.0
        self.hide()

    # ------------------------------------------------------------------ #
    def _poll(self) -> None:
        if self._io is None:
            return
        data = bytes(self._io.readAll().data())
        if len(data) >= 4:
            import array
            samples = array.array("h")
            samples.frombytes(data[: len(data) - (len(data) % 2)])
            if samples:
                rms = math.sqrt(sum(s * s for s in samples) / len(samples))
                db = 20.0 * math.log10(max(1.0, rms) / 32768.0)
                lin = max(0.0, min(1.0, 1.0 - db / _FLOOR_DB))
                # fast attack, slow release — reads like a real VU
                self._level = max(lin, self._level * 0.82)
                self._peak = max(self._peak * 0.97, self._level)
        self.update()

    def paintEvent(self, ev) -> None:  # noqa: N802 - Qt override
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        w, h = self.width(), self.height()
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 90))
        p.drawRoundedRect(0, 0, w, h, 3, 3)
        segs = 12
        gap = 2
        seg_w = (w - 6 - gap * (segs - 1)) / segs
        lit = int(round(self._level * segs))
        for i in range(segs):
            frac = (i + 1) / segs
            if frac <= 0.6:
                col = QColor(61, 220, 151)          # green
            elif frac <= 0.85:
                col = QColor(255, 193, 7)           # yellow
            else:
                col = QColor(255, 82, 82)           # red
            if i >= lit:
                col.setAlpha(46)                    # unlit ghost segments
            p.setBrush(col)
            x = 3 + i * (seg_w + gap)
            p.drawRect(int(x), 3, max(1, int(seg_w)), h - 6)
        if self._peak > 0.02:                        # peak-hold tick
            px = 3 + self._peak * (w - 6)
            p.setBrush(QColor(255, 255, 255, 200))
            p.drawRect(int(px) - 1, 2, 2, h - 4)
        p.end()
