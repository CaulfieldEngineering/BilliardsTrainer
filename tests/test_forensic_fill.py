"""The forensic-fill stage of the canonical close pass.

Now load-bearing (it recovered 180 verdicts across the library the day
it joined the pass), so its contract gets pinned: only unanswered
misses are attempted, records use the established forensic convention
the reader ranks below review, one bad shot never stops the fill, and
a re-run is a no-op (the appended records answer their own shots).
"""

import json

from billiards_trainer.core.types import BallClass, Track
from billiards_trainer.events.shot_detector import ShotEvent, ShotOutcome
from billiards_trainer.vision import shot_pass
from billiards_trainer.vision.analysis_cache import SidecarReader, SidecarWriter


def _track(tid, x, y, num=3):
    return Track(id=tid, x=x, y=y, radius=15.0, number=num,
                 cls=BallClass.SOLID, active=True)


def _session(tmp_path):
    video = tmp_path / "session-f.mp4"
    w = SidecarWriter(video, {"fps": 30.0})
    w.add_frame(0.0, [_track(1, 100.0, 200.0)])
    for start, outcome in ((10.0, ShotOutcome.MISS), (30.0, ShotOutcome.MISS),
                           (50.0, ShotOutcome.MAKE)):
        w.add_shot(ShotEvent(outcome=outcome, num_pocketed=0,
                             start_t=start, end_t=start + 4.0))
    w.close()
    return video


class TestForensicFill:
    def test_only_unanswered_misses_attempted(self, tmp_path, monkeypatch):
        video = _session(tmp_path)
        # pre-answer the first miss at review rank
        with open(str(video) + ".analysis.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"type": "tag_correction", "start": 10.0,
                                "miss_side": "left", "src": "review"}) + "\n")
        seen = []
        monkeypatch.setattr(
            shot_pass, "forensic_fill", shot_pass.forensic_fill)
        import billiards_trainer.vision.forensic_repass as fr
        monkeypatch.setattr(fr, "repass_shot", lambda v, s, e, ball=None: (
            seen.append(s) or {"ok": True, "side": "right", "cut": "left"}))
        n = shot_pass.forensic_fill(video)
        assert seen == [30.0], "only the unanswered miss may be attempted"
        assert n == 1

    def test_record_shape_and_reader_ranking(self, tmp_path, monkeypatch):
        video = _session(tmp_path)
        import billiards_trainer.vision.forensic_repass as fr
        monkeypatch.setattr(fr, "repass_shot",
                            lambda v, s, e, ball=None:
                            {"ok": True, "side": "right", "cut": "straight"})
        shot_pass.forensic_fill(video)
        r = SidecarReader(video)
        miss = next(s for s in r.shots if abs(s["start"] - 10.0) < 0.3)
        fo = miss.get("_tag_forensic")
        assert fo and fo["miss_side"] == "right" and fo["cut"] == "straight"

    def test_one_bad_shot_never_stops_the_fill(self, tmp_path, monkeypatch):
        video = _session(tmp_path)
        import billiards_trainer.vision.forensic_repass as fr

        def flaky(v, s, e, ball=None):
            if s < 20.0:
                raise RuntimeError("corridor exploded")
            return {"ok": True, "side": "left"}
        monkeypatch.setattr(fr, "repass_shot", flaky)
        assert shot_pass.forensic_fill(video) == 1   # the second miss landed

    def test_rerun_is_a_noop(self, tmp_path, monkeypatch):
        video = _session(tmp_path)
        import billiards_trainer.vision.forensic_repass as fr
        calls = {"n": 0}

        def once(v, s, e, ball=None):
            calls["n"] += 1
            return {"ok": True, "side": "left"}
        monkeypatch.setattr(fr, "repass_shot", once)
        assert shot_pass.forensic_fill(video) == 2
        assert shot_pass.forensic_fill(video) == 0   # answered now
        assert calls["n"] == 2
