"""Shot description (hierarchy #3) + correction ranking: human verdicts
are final; derived re-runs stand down."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from billiards_trainer.vision.analysis_cache import (  # noqa: E402
    SidecarReader,
    append_correction,
)
from billiards_trainer.vision.describe import compose_text  # noqa: E402
from billiards_trainer.vision.outcomes import derive_and_correct  # noqa: E402


def _write(tmp_path, states, shots):
    vid = tmp_path / "s.mp4"
    vid.write_bytes(b"0")
    with open(str(vid) + ".analysis.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "meta", "v": 1, "fps": 30}) + "\n")
        for t, tracks in states:
            f.write(json.dumps({"type": "f", "t": t,
                                "tracks": [[i, x, y, 12.0, n, c, True]
                                           for i, x, y, n, c in tracks]}) + "\n")
        for s in shots:
            f.write(json.dumps({"type": "shot", **s}) + "\n")
    return vid


class TestCorrectionRanking:
    def _mismatched(self, tmp_path):
        """Recorded miss; ball 5 demonstrably departs -> derived make."""
        states = []
        t = 0.0
        while t <= 30.0:
            tracks = [(1, 100.0, 100.0, 0, "cue"), (2, 300.0, 700.0, 9, "stripe")]
            if t < 8.0:
                tracks.append((3, 200.0, 500.0, 5, "solid"))
            states.append((round(t, 2), tracks))
            t += 0.25
        return _write(tmp_path, states,
                      [{"start": 8.0, "end": 12.0, "outcome": "miss",
                        "pocketed": 0}])

    def test_derived_never_overrides_review(self, tmp_path):
        vid = self._mismatched(tmp_path)
        append_correction(vid, 8.0, "miss", src="review")   # human says miss
        n = derive_and_correct(vid)                          # derives make
        assert n == 0, "derivation overrode a human verdict"
        assert SidecarReader(vid).shots[0]["outcome"] == "miss"

    def test_review_overrides_derived(self, tmp_path):
        vid = self._mismatched(tmp_path)
        derive_and_correct(vid)                              # derived: make
        append_correction(vid, 8.0, "scratch", src="review")
        s = SidecarReader(vid).shots[0]
        assert s["outcome"] == "scratch" and s["corrected"]

    def test_derived_corrections_carry_no_review_ring(self, tmp_path):
        vid = self._mismatched(tmp_path)
        derive_and_correct(vid)
        s = SidecarReader(vid).shots[0]
        assert s["outcome"] == "make" and not s.get("corrected"), \
            "derived corrections must not wear the human-verdict ring"

    def test_legacy_untagged_corrections_rank_as_review(self, tmp_path):
        vid = self._mismatched(tmp_path)
        sc = Path(str(vid) + ".analysis.jsonl")
        with open(sc, "a", encoding="utf-8") as f:      # pre-src-field line
            f.write(json.dumps({"type": "correction", "start": 8.0,
                                "outcome": "miss"}) + "\n")
        assert derive_and_correct(vid) == 0
        assert SidecarReader(vid).shots[0]["outcome"] == "miss"


class TestComposeText:
    def test_make_line(self):
        d = {"outcome": "make", "action": "stroke", "pace": "firm",
             "n_movers": 2, "cue_travel_tw": 1.4, "first_object": 1,
             "object_travel_tw": 0.8,
             "potted": [{"ball": 1, "pocket": "top-left"}],
             "duration_s": 5.0}
        line = compose_text(d)
        assert "pace" not in line, "pace is retired from rendered lines"
        assert "the 1 into the top-left pocket" in line

    def test_scratch_and_relocation_lines(self):
        d = {"outcome": "scratch", "action": "stroke", "pace": "soft",
             "n_movers": 1, "cue_travel_tw": 0.5, "first_object": None,
             "object_travel_tw": None, "potted": [{"ball": 0}],
             "duration_s": 4.0}
        assert "scratched" in compose_text(d)
        assert compose_text({"action": "ball_in_hand"}).startswith("Ball in hand")


class TestShotsExport:
    def test_summary_reflects_final_sidecar_state(self, tmp_path):
        import json as _json

        from billiards_trainer.vision.shots_export import export_shots_summary, summary_path
        vid = TestCorrectionRanking()._mismatched(tmp_path)
        derive_and_correct(vid)                      # miss -> make (derived)
        append_correction(vid, 8.0, "scratch", src="review")
        out = export_shots_summary(vid)
        assert out == summary_path(vid) and out.is_file()
        doc = _json.loads(out.read_text())
        assert doc["v"] == 1 and doc["session"] == "s.mp4"
        assert doc["shots"][0]["outcome"] == "scratch"   # review verdict wins
        assert doc["shots"][0]["corrected"] is True
        assert doc["shots"][0]["action"] == "stroke"
        assert doc["duration_s"] > 25

    def test_no_sidecar_is_quiet_none(self, tmp_path):
        from billiards_trainer.vision.shots_export import export_shots_summary
        assert export_shots_summary(tmp_path / "nope.mp4") is None


def test_ranked_miss_clears_phantom_pots(tmp_path):
    """Joe's screenshot: he corrected a shot to MISS but the description
    still read 'potted the 5' — describe recomputes departures from the
    OLD tracking data and only the scratch case was guarded. A ranked
    miss means NOTHING fell; a ranked scratch must say so even when the
    old derivation missed it."""
    from billiards_trainer.vision.describe import compose_text, describe_shot

    class R:                                    # minimal reader stand-in
        _times = []
        _frames = []
        def tracks_at(self, t):
            return []
        def hand_context(self, a, b):
            return set(), 0.0

    base = {"start": 8.0, "end": 12.0, "action": "stroke"}
    d = describe_shot(R(), {**base, "outcome": "miss"})
    assert d["potted"] == [], "ranked miss kept a phantom pot"
    assert "no ball fell" in compose_text(d)
    d = describe_shot(R(), {**base, "outcome": "scratch"})
    assert any(p["ball"] == 0 for p in d["potted"])
    assert "scratched" in compose_text(d)


class TestTheEngineOwnsWhichBallFell:
    """One opinion per fact: the engine credits the pot, the text reports it.

    Round 52, found on the first COLD clip (session-20260823-185550
    @138.1). The engine credited the 4 into the bottom-right - correct,
    confirmed by watching it: the purple ball runs down and drops while
    the yellow rolls on and stays on the table. The sentence Joe reads
    said "potted the 1", because describe.py re-derived which ball had
    departed and used the engine's answer only for the POCKET name.
    """

    def _desc(self, shot, gone):
        from billiards_trainer.vision import describe as D
        engine = {int(n) for n in (shot.get("pocketed_balls") or [])}
        if engine:
            gone = engine | (gone & {0})
        return gone

    def test_the_engine_credit_overrides_the_derivation(self):
        shot = {"pocketed_balls": [4], "pocketed_at": ["bottom-right"],
                "outcome": "make"}
        assert self._desc(shot, {1}) == {4}, (
            "the text named a different ball than the engine credited")

    def test_a_measured_scratch_survives_the_override(self):
        shot = {"pocketed_balls": [4], "pocketed_at": ["bottom-right"],
                "outcome": "scratch"}
        assert self._desc(shot, {0, 1}) == {0, 4}

    def test_the_derivation_still_runs_for_old_sidecars(self):
        shot = {"outcome": "make"}          # no engine credit recorded
        assert self._desc(shot, {2}) == {2}
