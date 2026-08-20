import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from render.fluidsynth import RenderError, find_soundfont, preflight, render
from tools.make_fixture import build as build_fixture

have_fluidsynth = shutil.which("fluidsynth") is not None and find_soundfont() is not None
needs_synth = pytest.mark.skipif(not have_fluidsynth, reason="fluidsynth or soundfont unavailable")


@pytest.fixture
def midi(tmp_path):
    path = tmp_path / "drums.mid"
    build_fixture().save(str(path))
    return path


def test_preflight_returns_strings_not_exceptions():
    assert all(isinstance(p, str) for p in preflight())


def test_dry_run_builds_a_sane_command(midi, tmp_path):
    result = render(midi, tmp_path / "out.wav", dry_run=True)
    assert result.command[0] == "fluidsynth"
    assert "-F" in result.command
    assert str(midi) == result.command[-1]


def test_missing_midi_is_a_clear_error(tmp_path):
    with pytest.raises(RenderError, match="no such MIDI"):
        render(tmp_path / "absent.mid", tmp_path / "out.wav")


@needs_synth
def test_renders_a_wav_with_actual_audio(midi, tmp_path):
    import wave

    result = render(midi, tmp_path / "out.wav")
    assert result.wav.exists()
    with wave.open(str(result.wav)) as handle:
        frames = handle.getnframes()
        assert frames > 0
        # 16 bars at 148bpm is ~26s; anything under 10s means it bailed early.
        assert frames / handle.getframerate() > 10
        peak = max(wave.struct.unpack(f"<{frames * handle.getnchannels()}h", handle.readframes(frames)))
    assert peak > 1000, "rendered audio is silent"


@needs_synth
def test_renders_an_mp3_when_an_encoder_exists(midi, tmp_path):
    if not (shutil.which("ffmpeg") or shutil.which("lame")):
        pytest.skip("no mp3 encoder")
    result = render(midi, tmp_path / "out.mp3")
    assert result.mp3 and result.mp3.exists()
    assert result.mp3.stat().st_size > 10_000
    assert result.output == result.mp3
