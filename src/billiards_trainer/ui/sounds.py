"""Asset-free audio cues for the shot clock — cross-platform.

Windows: winsound square-wave beeps (synchronous, so every cue plays on a
short daemon thread). macOS: the same tones rendered once to tiny WAV files
(pure stdlib) and played with the built-in ``afplay``. Anywhere else: the
plain system beep. Failures are swallowed — sound is a nicety, never an error.

Cadence (Joe's spec): single beep at 10 s left, tick beeps at 3-2-1, buzz at 0.
"""

import logging
import struct
import sys
import threading

log = logging.getLogger("ui.sounds")

# edge -> [(frequency_hz, duration_ms), ...]
_CUES = {
    "start": [(660, 110), (990, 150)],    # rising pair: you're on the clock
    "warn": [(880, 220)],                 # one heads-up beep (10 s left)
    "tick": [(1320, 130)],                # short, urgent: the 3-2-1 cadence
    "expired": [(220, 260), (185, 520)],  # two falling low tones = the buzz
}

_wav_cache: dict[tuple, str] = {}
_SR = 44100


def _render_wav(seq, volume: int = 100) -> str:
    """Render a tone sequence to a cached mono 16-bit WAV. `volume` 0-100
    scales sample amplitude — winsound.Beep has NO volume control, so
    per-cue volume (Joe's ask) plays rendered files instead of beeps."""
    import tempfile
    import wave
    from pathlib import Path

    amp = 0.35 * max(0, min(100, int(volume))) / 100.0
    key = (tuple(seq), int(volume))
    cached = _wav_cache.get(key)
    if cached and Path(cached).exists():
        return cached
    frames = bytearray()
    for freq, ms in seq:
        n = int(_SR * ms / 1000)
        fade = max(1, int(_SR * 0.005))          # 5 ms ramps kill the clicks
        half_period = _SR / (2.0 * freq)
        for i in range(n):
            v = amp if int(i / half_period) % 2 == 0 else -amp
            env = min(1.0, i / fade, (n - i) / fade)
            frames += struct.pack("<h", int(v * env * 32767))
    path = str(Path(tempfile.gettempdir()) / f"bt-cue-{abs(hash(key)):x}.wav")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(_SR)
        w.writeframes(bytes(frames))
    _wav_cache[key] = path
    return path


def _play_seq(seq, volume: int = 100) -> None:
    if volume <= 0:
        return                            # muted cue
    try:
        if sys.platform == "win32":
            import winsound
            try:
                winsound.PlaySound(_render_wav(seq, volume),
                                   winsound.SND_FILENAME | winsound.SND_ASYNC
                                   | winsound.SND_NODEFAULT)
                return
            except Exception:  # noqa: BLE001 - render/play failed
                for freq, ms in seq:      # full-volume beeps beat silence
                    winsound.Beep(freq, ms)
                return
        if sys.platform == "darwin":
            import subprocess
            subprocess.run(["afplay", _render_wav(seq, volume)], check=False,
                           capture_output=True, timeout=10)
            return
    except Exception:  # noqa: BLE001 - no audio device / server session
        log.debug("tone playback failed", exc_info=True)
    try:
        from PySide6.QtWidgets import QApplication
        QApplication.beep()
    except Exception:  # noqa: BLE001
        pass


def play(edge: str, volume: int = 100) -> None:
    """Play the cue for a shot-clock edge ('start' | 'warn' | 'tick' |
    'expired') at 0-100 volume. Unknown edges are silently ignored.
    Returns immediately."""
    seq = _CUES.get(edge)
    if not seq:
        return
    threading.Thread(target=_play_seq, args=(seq, volume), daemon=True,
                     name="shotclock-beep").start()
