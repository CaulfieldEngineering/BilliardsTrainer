"""Recording folder + audio sidecar behavior (no real capture devices)."""

from pathlib import Path

from billiards_trainer.capture.audio import AudioRecorder, _NullAudio, make_recorder
from billiards_trainer.config import EXPORTS_DIR, RecordingSettings


def test_resolved_dir_defaults_to_exports():
    assert RecordingSettings().resolved_dir() == EXPORTS_DIR
    assert RecordingSettings(directory="  ").resolved_dir() == EXPORTS_DIR


def test_resolved_dir_expands_user(tmp_path):
    r = RecordingSettings(directory=str(tmp_path / "clips"))
    assert r.resolved_dir() == tmp_path / "clips"
    assert RecordingSettings(directory="~/x").resolved_dir() == Path.home() / "x"


def test_make_recorder_disabled_is_noop(tmp_path):
    rec = make_recorder(False)
    assert isinstance(rec, _NullAudio)
    assert rec.start(tmp_path) is False
    assert rec.stop_and_mux(str(tmp_path / "v.mp4")) is False


def test_make_recorder_without_ffmpeg_is_noop(tmp_path, monkeypatch):
    monkeypatch.setattr("billiards_trainer.capture.audio.find_ffmpeg", lambda: None)
    rec = make_recorder(True)
    assert isinstance(rec, _NullAudio)


def test_audio_recorder_mux_with_no_segments_leaves_video_alone(tmp_path):
    rec = AudioRecorder()
    video = tmp_path / "session.mp4"
    video.write_bytes(b"fake video")
    assert rec.stop_and_mux(str(video)) is False
    assert video.read_bytes() == b"fake video"   # untouched on failure
