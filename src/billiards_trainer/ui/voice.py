"""Spoken cues (Joe: "a voice that says 'Ten' at ten seconds" — and the
foothold for full narration: "Scratch", "Ball in hand", ball names).

Phrases are a FINITE set, so speech is pre-rendered once to WAVs in
APP_DIR/voice/ and playback is instant, offline, and identical every
time. Renderer chain: edge-tts neural voice (natural; needs network at
RENDER time only) -> Windows SAPI via PowerShell (offline, dated but
serviceable) -> silence (a missing voice never breaks play).

say() returns immediately; rendering happens on a daemon thread the
first time a phrase is requested, so the very first utterance of a new
phrase may be skipped rather than delayed — every one after is instant.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
from pathlib import Path

from ..config import APP_DIR

log = logging.getLogger("ui.voice")

VOICE_DIR = APP_DIR / "voice"
# a calm, natural US voice; render-time only (playback is local WAV)
_EDGE_VOICE = "en-US-GuyNeural"
_render_lock = threading.Lock()


def _slug(phrase: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in phrase.lower()).strip("-")


def wav_path(phrase: str) -> Path:
    return VOICE_DIR / f"{_slug(phrase)}.wav"


def _render_edge(phrase: str, out: Path) -> bool:
    try:
        import asyncio

        import edge_tts

        async def go():
            tts = edge_tts.Communicate(phrase, _EDGE_VOICE)
            await tts.save(str(out.with_suffix(".mp3")))
        asyncio.run(go())
        # winsound needs WAV: transcode with the bundled ffmpeg
        from ..capture.audio import NO_WINDOW, find_ffmpeg
        ff = find_ffmpeg()
        if ff is None:
            return False
        r = subprocess.run([ff, "-v", "error", "-i",
                            str(out.with_suffix(".mp3")), "-y", str(out)],
                           capture_output=True, timeout=30,
                           creationflags=NO_WINDOW)
        out.with_suffix(".mp3").unlink(missing_ok=True)
        return r.returncode == 0 and out.is_file()
    except Exception:  # noqa: BLE001 - offline / package missing
        log.debug("edge-tts render failed for %r", phrase, exc_info=True)
        return False


def _render_sapi(phrase: str, out: Path) -> bool:
    if sys.platform != "win32":
        return False
    try:
        ps = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.SetOutputToWaveFile('{out}'); "
            f"$s.Speak('{phrase}'); $s.Dispose()"
        )
        r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                           capture_output=True, timeout=30,
                           creationflags=0x08000000)
        return r.returncode == 0 and out.is_file()
    except Exception:  # noqa: BLE001
        log.debug("SAPI render failed for %r", phrase, exc_info=True)
        return False


def ensure(phrase: str) -> Path | None:
    """Blocking: render the phrase if not cached. Returns the WAV or None."""
    out = wav_path(phrase)
    if out.is_file() and out.stat().st_size > 1000:
        return out
    with _render_lock:
        if out.is_file() and out.stat().st_size > 1000:
            return out
        VOICE_DIR.mkdir(parents=True, exist_ok=True)
        if _render_edge(phrase, out) or _render_sapi(phrase, out):
            log.info("voice rendered: %r", phrase)
            return out
    return None


def say(phrase: str, volume: int = 100) -> None:
    """Speak a cached phrase; render-and-cache on first request (that
    first call may stay silent rather than stall). Returns immediately."""
    if volume <= 0:
        return

    def go():
        out = ensure(phrase)
        if out is None:
            return
        try:
            if sys.platform == "win32":
                import winsound
                winsound.PlaySound(str(out), winsound.SND_FILENAME
                                   | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
            elif sys.platform == "darwin":
                subprocess.run(["afplay", str(out)], check=False,
                               capture_output=True, timeout=15)
        except Exception:  # noqa: BLE001
            log.debug("voice playback failed", exc_info=True)
    threading.Thread(target=go, daemon=True, name="voice-say").start()


def prewarm(phrases) -> None:
    """Render a phrase list in the background (app start)."""
    def go():
        for p in phrases:
            ensure(p)
    threading.Thread(target=go, daemon=True, name="voice-prewarm").start()
