"""Smoke tests for the Phase-0 eval harness + the debug-upload skeleton.

Keeps the CI eval path honest: the harness must run end-to-end on the committed
synthetic clip and emit a report + metrics, and the debug-bundle stager must copy
into the inbox.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "eval" / "demo_clip.mp4"


def test_debug_upload_stage(tmp_path, monkeypatch):
    import billiards_trainer.debug_upload as du

    monkeypatch.setattr(du, "inbox_dir", lambda: tmp_path / "inbox")
    src = tmp_path / "failure-x.zip"
    src.write_bytes(b"PK\x03\x04 fake zip")
    dst = du.stage_bundle(src)
    assert dst.exists()
    assert dst.parent.name == "inbox"
    assert dst.read_bytes() == src.read_bytes()


@pytest.mark.skipif(not FIXTURE.exists(), reason="eval fixture clip missing")
def test_eval_harness_runs_on_fixture(tmp_path):
    out = tmp_path / "report"
    r = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "eval_harness.py"),
         "--clips", str(FIXTURE.parent), "--out", str(out),
         "--timestamp", "test", "--stride", "8"],
        capture_output=True, text=True, timeout=300, cwd=str(ROOT))
    assert r.returncode == 0, r.stderr[-2000:]
    metrics = out / "metrics.json"
    assert (out / "report.html").exists()
    assert metrics.exists()
    data = json.loads(metrics.read_text(encoding="utf-8"))
    agg = data["aggregate"]
    # the synthetic demo table should calibrate + the harness should produce
    # well-formed aggregate metrics (we don't assert accuracy, just structure)
    assert agg["frames"] > 0
    assert "calib_rate" in agg and "total_dets" in agg and "fp_shots_per_min" in agg


def test_eval_ci_check_seeds_and_compares(tmp_path, monkeypatch):
    """ci-check seeds a baseline then reports no regression against itself."""
    import importlib

    metrics = {"aggregate": {"calib_rate": 1.0, "total_dets": 50, "shots": 1, "frames": 30}}
    mpath = tmp_path / "metrics.json"
    mpath.write_text(json.dumps(metrics), encoding="utf-8")
    baseline = tmp_path / "ci_baseline.json"

    sys.path.insert(0, str(ROOT / "tools"))
    mod = importlib.import_module("eval_ci_check")
    monkeypatch.setattr(mod, "BASELINE", baseline)
    # seed (no baseline yet) -> exit 0, baseline written
    monkeypatch.setattr(sys, "argv", ["x", "--metrics", str(mpath)])
    assert mod.main() == 0
    assert baseline.exists()
    # compare against itself -> no regression, exit 0
    assert mod.main() == 0
