import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from render.master import (
    Loudness, MasterError, build_filter_chain, db_to_linear, has_loudness_stage,
    load_chain, master, measure, stage_names,
)

have_ffmpeg = shutil.which("ffmpeg") is not None
needs_ffmpeg = pytest.mark.skipif(not have_ffmpeg, reason="ffmpeg unavailable")


# ------------------------------------------------------------------ chain

def test_db_to_linear():
    assert db_to_linear(0) == pytest.approx(1.0)
    assert db_to_linear(-6) == pytest.approx(0.5012, abs=1e-4)


def test_limiter_ceiling_is_converted_to_linear():
    """ffmpeg's alimiter `limit` is linear 0.0625-1 and does NOT take a dB
    suffix. Passing dB straight through silently clamps to 1.0 (no limiting)."""
    out = build_filter_chain([{"stage": "limiter", "ceiling_db": -6}])
    assert "limit=0.501187" in out
    assert "dB" not in out


def test_compressor_makeup_is_converted_and_clamped():
    """`makeup` is a linear factor 1-64; negative dB is not expressible."""
    assert "makeup=1.25893" in build_filter_chain([{"stage": "compressor", "makeup_db": 2}])
    assert "makeup=1" in build_filter_chain([{"stage": "compressor", "makeup_db": -6}])


def test_compressor_threshold_keeps_its_db_suffix():
    assert "threshold=-18dB" in build_filter_chain([{"stage": "compressor", "threshold_db": -18}])


def test_stages_are_joined_in_order():
    chain = [{"stage": "highpass"}, {"stage": "limiter"}]
    assert build_filter_chain(chain).index("highpass") < build_filter_chain(chain).index("alimiter")


def test_disabled_stage_is_skipped():
    chain = [{"stage": "highpass"}, {"stage": "limiter", "enabled": False}]
    assert "alimiter" not in build_filter_chain(chain)
    assert stage_names(chain) == ["highpass"]


def test_unknown_stage_is_a_clear_error():
    with pytest.raises(MasterError, match="unknown master stage"):
        build_filter_chain([{"stage": "exciter"}])


def test_non_mapping_stage_is_a_clear_error():
    with pytest.raises(MasterError, match="must be a mapping"):
        build_filter_chain(["limiter"])


def test_empty_chain_is_an_empty_string():
    assert build_filter_chain([]) == ""


def test_two_pass_injects_measured_values():
    measured = {"input_i": "-20.0", "input_tp": "-3.0", "input_lra": "5.0",
                "input_thresh": "-30.0", "target_offset": "0.5"}
    out = build_filter_chain([{"stage": "loudness"}], measured=measured)
    assert "measured_I=-20.0" in out and "linear=true" in out


def test_has_loudness_stage():
    assert has_loudness_stage([{"stage": "loudness"}])
    assert not has_loudness_stage([{"stage": "loudness", "enabled": False}])
    assert not has_loudness_stage([{"stage": "limiter"}])


def test_shipped_chain_is_valid():
    chain = load_chain()
    assert chain, "master.yaml should define a chain"
    assert build_filter_chain(chain)


# ------------------------------------------------------------- measurement

@needs_ffmpeg
def test_measure_reads_the_summary_not_the_first_progress_line(tmp_path):
    """ebur128 prints a running measurement per frame as well as a Summary.
    The early frames report -70 LUFS because there is not yet enough audio to
    integrate, so parsing the whole stream reads the startup value."""
    wav = tmp_path / "tone.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", "sine=frequency=1000:duration=6:sample_rate=44100", str(wav)],
        check=True,
    )
    result = measure(wav)
    assert result.integrated_lufs is not None
    assert result.integrated_lufs > -40, f"looks like the -70 startup value: {result}"
    assert result.true_peak_db is not None


def test_measure_of_a_missing_file_returns_empty_not_a_crash(tmp_path):
    assert measure(tmp_path / "absent.wav").integrated_lufs is None


def test_loudness_str_handles_missing_values():
    assert "?" in str(Loudness())


# -------------------------------------------------------------- end to end

@pytest.fixture
def quiet_audio(tmp_path):
    if not have_ffmpeg:
        pytest.skip("ffmpeg unavailable")
    path = tmp_path / "quiet.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", "sine=frequency=220:duration=8:sample_rate=44100",
         "-af", "volume=-24dB", str(path)],
        check=True,
    )
    return path


@needs_ffmpeg
def test_master_raises_loudness_toward_target(quiet_audio, tmp_path):
    chain = load_chain()
    result = master(quiet_audio, tmp_path / "out.wav", chain)
    assert result.after.integrated_lufs > result.before.integrated_lufs
    assert result.after.integrated_lufs == pytest.approx(-14, abs=2.0)


@needs_ffmpeg
def test_master_respects_the_true_peak_ceiling(quiet_audio, tmp_path):
    """The delivered file must stay under -1.0 dBTP. master.yaml carries 1 dB of
    codec headroom precisely so that lossy encoding does not push it over."""
    result = master(quiet_audio, tmp_path / "out.mp3", load_chain())
    assert result.after.true_peak_db is not None
    assert result.after.true_peak_db <= -1.0


@needs_ffmpeg
def test_two_pass_is_used_when_the_chain_normalises(quiet_audio, tmp_path):
    assert master(quiet_audio, tmp_path / "a.wav", load_chain()).two_pass


@needs_ffmpeg
def test_two_pass_can_be_disabled(quiet_audio, tmp_path):
    assert not master(quiet_audio, tmp_path / "b.wav", load_chain(), two_pass=False).two_pass


def test_dry_run_does_not_write(tmp_path, quiet_audio):
    out = tmp_path / "nope.wav"
    result = master(quiet_audio, out, load_chain(), dry_run=True)
    assert not out.exists()
    assert result.filter_chain


def test_missing_source_is_a_clear_error(tmp_path):
    with pytest.raises(MasterError, match="no such audio"):
        master(tmp_path / "absent.wav", tmp_path / "out.wav", load_chain())
