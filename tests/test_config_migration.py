"""Settings migration + detector-default tests.

The detector default is now ``auto`` (resolves to the trained YOLO model when
present, else the cue-ball heuristic). Earlier defaults — the ``legacy`` rectified
detector and the classical blob strategies (``simple_blob`` etc.) — are migrated
forward to ``auto`` exactly once, so old settings files land on the good path.
"""

import json

from billiards_trainer.config import Settings, _migrate_settings


def test_fresh_settings_default_is_auto():
    assert Settings().balls.live_strategy == "auto"


def test_legacy_is_migrated_to_auto():
    s = Settings()
    s.balls.live_strategy = "legacy"
    s.balls.detector_migration = 0
    changed = _migrate_settings(s)
    assert changed
    # legacy -> simple_blob (migration 1) -> auto (migration 2), in one pass
    assert s.balls.live_strategy == "auto"
    assert s.balls.detector_migration == 2


def test_classical_blob_default_is_migrated_to_auto():
    """A user persisted on an old classical blob detector is moved to auto, which
    resolves to the trained model."""
    for old in ("simple_blob", "felt_mask_hough", "classical"):
        s = Settings()
        s.balls.live_strategy = old
        s.balls.detector_migration = 1  # past migration 1, before 2
        changed = _migrate_settings(s)
        assert changed
        assert s.balls.live_strategy == "auto"
        assert s.balls.detector_migration == 2


def test_explicit_model_choice_is_left_alone():
    """A deliberate onnx_* choice survives migration (we don't override it)."""
    s = Settings()
    s.balls.live_strategy = "onnx_pool_yolo11"
    s.balls.detector_migration = 1
    _migrate_settings(s)
    assert s.balls.live_strategy == "onnx_pool_yolo11"


def test_load_migrates_and_persists(tmp_path):
    """A settings file written with an old detector is migrated on load AND the
    migration is written back so it sticks across restarts."""
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"balls": {"live_strategy": "legacy"}}), encoding="utf-8")

    s = Settings.load(p)
    assert s.balls.live_strategy == "auto"

    on_disk = json.loads(p.read_text(encoding="utf-8"))
    assert on_disk["balls"]["live_strategy"] == "auto"
    assert on_disk["balls"]["detector_migration"] == 2
