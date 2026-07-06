"""Asset-free audio cues for the shot clock.

winsound square-wave beeps on Windows (the shipped platform), falling back to
the plain system beep elsewhere. winsound.Beep is SYNCHRONOUS, so every cue
plays on a short daemon thread — the UI/worker thread must never block on
audio. Failures are swallowed: sound is a nicety, never an error.

Cadence (Joe's spec): single beep at 10 s left, tick beeps at 3-2-1, buzz at 0.
"""

import logging
import sys
import threading

log = logging.getLogger("ui.sounds")

# edge -> [(frequency_hz, duration_ms), ...]
_CUES = {
    "warn": [(880, 220)],                 # one heads-up beep (10 s left)
    "tick": [(1320, 130)],                # short, urgent: the 3-2-1 cadence
    "expired": [(220, 260), (185, 520)],  # two falling low tones = the buzz
}


def _play_seq(seq) -> None:
    try:
        if sys.platform == "win32":
            import winsound
            for freq, ms in seq:
                winsound.Beep(freq, ms)
            return
    except Exception:  # noqa: BLE001 - no audio device / server session
        log.debug("winsound beep failed", exc_info=True)
    try:
        from PySide6.QtWidgets import QApplication
        QApplication.beep()
    except Exception:  # noqa: BLE001
        pass


def play(edge: str) -> None:
    """Play the cue for a shot-clock edge ('warn' | 'tick' | 'expired').
    Unknown edges are silently ignored. Returns immediately."""
    seq = _CUES.get(edge)
    if not seq:
        return
    threading.Thread(target=_play_seq, args=(seq,), daemon=True,
                     name="shotclock-beep").start()
