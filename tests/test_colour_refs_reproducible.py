"""A fresh clone must name balls correctly with no manual step.

The engine's naming depends on measured per-ball colour references: they
are what corrects the identifier's misread of the purple 4 as the 7
(round 33 measured that correction as the difference between 2/136 and
136/136 for the 4). Those references live in APP_DIR, which is machine
state, not repo state.

Until round 52 the loader read APP_DIR ONLY and swallowed every failure
into an empty dict. On any machine that had not run
`tools/build_colour_refs.py --install` - a fresh clone, a rebuilt
profile, a new PC - naming silently lost its correction and the
scorecard would simply print a worse number with no cause attached.
docs/colour_refs.json is the committed version of record and is now a
fallback, so a checkout is correct by default.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "docs" / "colour_refs.json"
BENCH_BALLS = {1, 2, 3, 4, 9}      # 0 is the cue: white, never a reference


class TestTheVersionOfRecord:
    def test_it_is_committed(self):
        assert RECORD.is_file(), (
            "docs/colour_refs.json is the only thing that makes ball naming "
            "reproducible on a machine that has never run --install")

    def test_it_parses_and_covers_the_bench_balls(self):
        doc = json.loads(RECORD.read_text(encoding="utf-8"))
        refs = doc.get("refs")
        assert refs, "the record has no `refs` block"
        have = {int(k) for k in refs}
        missing = BENCH_BALLS - have
        assert not missing, f"the record is missing references for {missing}"

    def test_every_reference_is_usable(self):
        doc = json.loads(RECORD.read_text(encoding="utf-8"))
        for k, v in doc["refs"].items():
            lab = v.get("lab")
            assert lab and len(lab) == 3, f"ball {k}: no usable lab colour"
            assert int(v.get("n", 0)) > 0, f"ball {k}: no samples behind it"


class TestTheLoaderFallsBack:
    def test_a_machine_with_no_live_copy_still_gets_references(self, tmp_path,
                                                               monkeypatch):
        """The whole point: APP_DIR empty, naming still works."""
        import billiards_trainer.config as cfg
        from billiards_trainer.core import balls
        monkeypatch.setattr(cfg, "APP_DIR", tmp_path)      # no live copy here
        monkeypatch.setattr(balls, "_MEASURED_REFS", None)  # clear the cache
        refs = balls._load_measured_refs()
        assert refs, (
            "a fresh clone got NO colour references - the purple 4 will be "
            "called a 7 and nothing will say why")
        assert BENCH_BALLS <= set(refs) | {9}, "bench balls not covered"

    def test_the_live_copy_still_wins_when_present(self, tmp_path, monkeypatch):
        import billiards_trainer.config as cfg
        from billiards_trainer.core import balls
        (tmp_path / "colour_refs.json").write_text(
            json.dumps({"3": {"lab": [10.0, 20.0, 30.0], "n": 99}}),
            encoding="utf-8")
        monkeypatch.setattr(cfg, "APP_DIR", tmp_path)
        monkeypatch.setattr(balls, "_MEASURED_REFS", None)
        refs = balls._load_measured_refs()
        assert set(refs) == {3} and abs(float(refs[3][0]) - 10.0) < 1e-3, (
            "the live copy must take precedence over the version of record")
