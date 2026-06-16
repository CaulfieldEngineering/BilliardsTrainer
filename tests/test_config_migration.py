"""Settings migration + detector-default tests.

The 'Live detector' dropdown shipped showing only 'legacy' because frozen
strategy discovery was broken; users got 'legacy' persisted to their settings.
``Settings.load()`` migrates those users back to the intended default exactly
once, while leaving a *deliberate* later choice of legacy intact.
"""

import json

from billiards_trainer.config import Settings, _migrate_settings


def test_fresh_settings_default_is_simple_blob():
    assert Settings().balls.live_strategy == "simple_blob"


def test_legacy_is_migrated_to_simple_blob():
    s = Settings()
    s.balls.live_strategy = "legacy"
    s.balls.detector_migration = 0
    changed = _migrate_settings(s)
    assert changed
    assert s.balls.live_strategy == "simple_blob"
    assert s.balls.detector_migration == 1


def test_deliberate_legacy_after_migration_sticks():
    """Once the one-time migration has run (marker == 1), a user who deliberately
    picks legacy keeps it — we don't fight their choice forever."""
    s = Settings()
    s.balls.live_strategy = "legacy"
    s.balls.detector_migration = 1  # already migrated
    changed = _migrate_settings(s)
    assert not changed
    assert s.balls.live_strategy == "legacy"


def test_load_migrates_and_persists(tmp_path):
    """A settings file written with legacy + no marker is migrated on load AND the
    migration is written back so it sticks across restarts."""
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"balls": {"live_strategy": "legacy"}}), encoding="utf-8")

    s = Settings.load(p)
    assert s.balls.live_strategy == "simple_blob"

    # persisted: re-reading the file shows the migrated value + marker
    on_disk = json.loads(p.read_text(encoding="utf-8"))
    assert on_disk["balls"]["live_strategy"] == "simple_blob"
    assert on_disk["balls"]["detector_migration"] == 1
