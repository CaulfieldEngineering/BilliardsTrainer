"""The on-demand re-measure job's guards (one at a time, never beside
a recording) — the cheap refusals that keep the button safe."""

from billiards_trainer.measure import job


def test_refuses_when_another_job_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(job, "M1_DIR", tmp_path)
    monkeypatch.setattr(job, "RUNNING", tmp_path / "RUNNING")
    monkeypatch.setattr(job, "RESULT", tmp_path / "res.json")
    (tmp_path / "RUNNING").write_text("busy")
    out = job.run("C:/nowhere/session-x.mp4")
    assert "another heavy job" in out["refused"]


def test_refuses_while_recording(tmp_path, monkeypatch):
    rec = tmp_path / "rec"
    rec.mkdir()
    (rec / ".session-live.part.mp4").write_bytes(b"x")
    monkeypatch.setattr(job, "M1_DIR", tmp_path)
    monkeypatch.setattr(job, "RUNNING", tmp_path / "RUNNING")
    monkeypatch.setattr(job, "RESULT", tmp_path / "res.json")
    monkeypatch.setattr(job, "EXPORTS_DIR", rec)
    out = job.run("C:/nowhere/session-x.mp4")
    assert out["refused"] == "recording live"
    assert not (tmp_path / "RUNNING").exists()
