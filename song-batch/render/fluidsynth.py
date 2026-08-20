"""Render a MIDI file to audio with fluidsynth, then encode for the phone.

Deliberately crude. This exists to close the feedback loop *today*: transform
MIDI, hear the result on a phone, iterate. It renders through a General MIDI
soundfont, so it tells you about timing, dynamics and arrangement - and nothing
at all about SSD's actual tone. The JUCE host replaces it for that.

fluidsynth's ``-F`` writes a file faster than realtime, so a 4 minute song takes
a couple of seconds.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Searched in order. First hit wins. Override with $SONG_BATCH_SF2.
SOUNDFONT_CANDIDATES = (
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "/usr/share/sounds/sf2/default-GM.sf2",
    "/usr/share/soundfonts/FluidR3_GM.sf2",
    # macOS / Homebrew
    "/opt/homebrew/share/fluid-soundfont/FluidR3_GM.sf2",
    "/opt/homebrew/share/soundfonts/default.sf2",
    "/usr/local/share/fluid-soundfont/FluidR3_GM.sf2",
    "/usr/local/share/soundfonts/default.sf2",
)


class RenderError(Exception):
    pass


@dataclass
class RenderResult:
    midi: Path
    wav: Optional[Path] = None
    mp3: Optional[Path] = None
    soundfont: Optional[Path] = None
    command: List[str] = field(default_factory=list)
    duration_seconds: Optional[float] = None

    @property
    def output(self) -> Optional[Path]:
        """The file to hand to the operator - mp3 if we made one, else wav."""
        return self.mp3 or self.wav


def find_soundfont(explicit: Optional[str | Path] = None) -> Optional[Path]:
    """Locate a General MIDI soundfont, or None."""
    if explicit:
        p = Path(explicit).expanduser()
        return p if p.exists() else None
    env = os.environ.get("SONG_BATCH_SF2")
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p
    for candidate in SOUNDFONT_CANDIDATES:
        p = Path(candidate)
        if p.exists():
            return p
    return None


def preflight(soundfont: Optional[str | Path] = None) -> List[str]:
    """Return human-readable problems that would stop a render. Empty = good.

    Called by the CLI before doing any work, so a phone-operated session gets
    "install fluidsynth" rather than a stack trace.
    """
    problems: List[str] = []
    if not shutil.which("fluidsynth"):
        problems.append(
            "fluidsynth not on PATH. macOS: `brew install fluid-synth`. "
            "Debian/Ubuntu: `apt install fluidsynth`."
        )
    if find_soundfont(soundfont) is None:
        problems.append(
            "No General MIDI soundfont found. macOS: `brew install fluid-synth` ships one, "
            "or set $SONG_BATCH_SF2 to a .sf2 path."
        )
    if not (shutil.which("ffmpeg") or shutil.which("lame")):
        problems.append(
            "Neither ffmpeg nor lame found, so renders stay as .wav (large for phone "
            "download). macOS: `brew install ffmpeg`."
        )
    return problems


def _encode_mp3(wav: Path, mp3: Path, bitrate: str) -> bool:
    """WAV -> MP3 via whichever encoder is around. False if none is."""
    if shutil.which("ffmpeg"):
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-b:a", bitrate, str(mp3)]
    elif shutil.which("lame"):
        cmd = ["lame", "--quiet", "-b", bitrate.rstrip("k"), str(wav), str(mp3)]
    else:
        return False
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RenderError(f"mp3 encode failed:\n{result.stderr.strip()}")
    return True


def render(
    midi_path: Path | str,
    out_path: Path | str,
    soundfont: Optional[str | Path] = None,
    gain: float = 0.6,
    sample_rate: int = 44100,
    bitrate: str = "192k",
    keep_wav: bool = False,
    dry_run: bool = False,
) -> RenderResult:
    """Render ``midi_path`` to ``out_path`` (``.mp3`` or ``.wav``).

    ``gain`` defaults low because FluidR3's drum kit clips readily at 1.0 and a
    clipped preview is a misleading preview.
    """
    midi_path = Path(midi_path)
    out_path = Path(out_path)
    if not midi_path.exists():
        raise RenderError(f"no such MIDI file: {midi_path}")

    sf2 = find_soundfont(soundfont)
    if sf2 is None and not dry_run:
        raise RenderError("no soundfont available; run `./sb render --check` for install hints")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    want_mp3 = out_path.suffix.lower() == ".mp3"
    wav_path = out_path.with_suffix(".wav")

    cmd = [
        "fluidsynth",
        "-ni",                      # no shell, no MIDI input device
        "-F", str(wav_path),        # render to file, faster than realtime
        "-r", str(sample_rate),
        "-g", f"{gain:g}",
        str(sf2 or "<soundfont>"),
        str(midi_path),
    ]
    result = RenderResult(midi=midi_path, soundfont=sf2, command=cmd)
    if dry_run:
        return result

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not wav_path.exists():
        raise RenderError(
            f"fluidsynth failed (exit {proc.returncode}):\n"
            f"{(proc.stderr or proc.stdout).strip()[:2000]}"
        )
    result.wav = wav_path

    try:
        import wave

        with wave.open(str(wav_path)) as handle:
            result.duration_seconds = round(handle.getnframes() / float(handle.getframerate()), 2)
    except Exception:  # pragma: no cover - duration is nice-to-have only
        pass

    if want_mp3:
        if _encode_mp3(wav_path, out_path, bitrate):
            result.mp3 = out_path
            if not keep_wav:
                wav_path.unlink(missing_ok=True)
                result.wav = None

    return result
