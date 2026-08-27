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

# edge -> [(frequency_hz, duration_ms), ...] rendered as BELL tones
# (harmonic stack + exponential decay - Joe: "better sounds than the
# video game chirps"; the old square waves are gone)
_CUES = {
    "start": [(659, 260), (988, 420)],    # warm rising bell pair
    "warn": [(784, 620)],                 # one mellow bell
    "tick": [(1175, 200)],                # bright woodblock tap: 3-2-1
    "expired": [(196, 500), (147, 900)],  # low gong pair = time
    "scratch": [(392, 250), (311, 250), (247, 550)],   # falling minor phrase
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
    import math
    frames = bytearray()
    # bell voice: fundamental + soft harmonics, fast attack, exponential
    # decay - warm and musical where the old square wave was a chirp
    HARMONICS = ((1.0, 1.0), (2.0, 0.42), (3.0, 0.18), (4.2, 0.07))
    for freq, ms in seq:
        n = int(_SR * ms / 1000)
        attack = max(1, int(_SR * 0.004))
        tau = max(0.06, ms / 1000.0 * 0.55)      # decay scaled to note length
        for i in range(n):
            ts = i / _SR
            v = sum(a * math.sin(2 * math.pi * freq * k * ts)
                    for k, a in HARMONICS)
            env = min(1.0, i / attack) * math.exp(-ts / tau)
            end_fade = min(1.0, (n - i) / attack)
            s = amp * 0.62 * v * env * end_fade
            frames += struct.pack("<h", int(max(-1.0, min(1.0, s)) * 32767))
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
