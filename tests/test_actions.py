"""Action classification plumbing: append-only records, reader attach,
idempotence, and the unambiguous classifier rules. Threshold fitting is
measured against frame-verified labels (see GOALS); these tests pin the
contract, not the tuning."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from billiards_trainer.vision.actions import (  # noqa: E402
    classify,
    classify_and_mark,
)
from billiards_trainer.vision.analysis_cache import SidecarReader  # noqa: E402


def _write(tmp_path, states, shots):
    vid = tmp_path / "s.mp4"
    vid.write_bytes(b"0")
    with open(str(vid) + ".analysis.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "meta", "v": 1, "fps": 30}) + "\n")
        for t, tracks, carried in states:
            d = {"type": "f", "t": t,
                 "tracks": [[i, x, y, 12.0, n, c, True]
                            for i, x, y, n, c in tracks]}
            if carried:
                d["c"] = carried
            f.write(json.dumps(d) + "\n")
        for s in shots:
            f.write(json.dumps({"type": "shot", **s}) + "\n")
    return vid


def _session(tmp_path, mover):
    """20s session, one 8-12s shot window; `mover(t)` yields the moving
    ball's (x, y, carried) at time t."""
    states = []
    t = 0.0
    while t <= 20.0:
        x, y, carried = mover(t)
        tracks = [(1, 300.0, 700.0, 9, "stripe"), (2, x, y, 0, "cue")]
        states.append((round(t, 2), tracks, [2] if carried else []))
        t += 0.1
    return _write(tmp_path, states,
                  [{"start": 8.0, "end": 12.0, "outcome": "miss",
                    "pocketed": 0}])


class TestClassifierRules:
    def test_still_window_is_nothing(self, tmp_path):
        vid = _session(tmp_path, lambda t: (100.0, 100.0, False))
        counts = classify_and_mark(vid)
        assert counts == {"nothing": 1}
        assert SidecarReader(vid).shots[0]["action"] == "nothing"

    def test_free_fast_ball_is_stroke(self, tmp_path):
        def mover(t):
            if t < 8.5:
                return 100.0, 100.0, False
            x = min(600.0, 100.0 + (t - 8.5) * 400.0)
            return x, 100.0, False
        vid = _session(tmp_path, mover)
        assert classify_and_mark(vid) == {"stroke": 1}
        # default action needs NO record appended (log stays clean)
        raw = Path(str(vid) + ".analysis.jsonl").read_text()
        assert '"type": "action"' not in raw

    def test_hand_carried_mover_is_ball_in_hand(self, tmp_path):
        def mover(t):
            if t < 8.5 or t > 11.0:
                return (100.0, 100.0, False) if t < 8.5 else (400.0, 500.0, False)
            x = 100.0 + (t - 8.5) * 120.0
            return x, 100.0 + (t - 8.5) * 160.0, True   # moves IN the hand
        vid = _session(tmp_path, mover)
        assert classify_and_mark(vid) == {"ball_in_hand": 1}
        assert SidecarReader(vid).shots[0]["action"] == "ball_in_hand"

    def test_break_rule(self):
        f = {"dur": 6.0, "peak": 900.0, "n_movers": 9, "n_fast": 8,
             "hand_frac": 0.1, "carried_frac": 0.0,
             "carried_at_launch": False, "fast_sync": 0.6,
             "set_changed": True, "n_states": 40}
        assert classify(f) == "break"
        # mass-gathering: same ball counts, launches spread over 20s
        assert classify({**f, "fast_sync": 12.0}) != "break"

    def test_idempotent_and_review_wins(self, tmp_path):
        from billiards_trainer.vision.actions import append_action
        vid = _session(tmp_path, lambda t: (100.0, 100.0, False))
        classify_and_mark(vid)
        raw1 = Path(str(vid) + ".analysis.jsonl").read_text()
        classify_and_mark(vid)
        assert Path(str(vid) + ".analysis.jsonl").read_text() == raw1
        append_action(vid, 8.0, "stroke")          # Joe's review verdict
        assert SidecarReader(vid).shots[0]["action"] == "stroke"


class TestActionAwareSurfaces:
    """Shots counts and views consume action labels: relocations stay
    visible (badged) but never count as attempts anywhere."""

    SHOTS = [
        {"start": 10.0, "end": 14.0, "outcome": "make"},
        {"start": 20.0, "end": 24.0, "outcome": "miss",
         "action": "ball_in_hand"},
        {"start": 30.0, "end": 34.0, "outcome": "make", "action": "break"},
        {"start": 40.0, "end": 41.0, "outcome": "miss", "action": "nothing"},
        {"start": 50.0, "end": 54.0, "outcome": "make"},
    ]

    def test_views_exclude_non_attempts(self):
        from billiards_trainer.ui.widgets.shot_list import view_shots
        makes = view_shots(self.SHOTS, "Makes")
        assert [n for n, _ in makes] == [1, 3, 5]
        misses = view_shots(self.SHOTS, "Misses")
        assert misses == []                     # both misses are non-attempts
        assert len(view_shots(self.SHOTS, "All")) == 5   # badged, not hidden
        streaks = view_shots(self.SHOTS, "Streaks")
        assert [n for n, _ in streaks] == []    # relocations break no streaks

    def test_summary_counts_attempts_only(self, tmp_path):
        import json

        from billiards_trainer.vision.session_summaries import _sidecar_shot_count
        vid = tmp_path / "s.mp4"
        vid.write_bytes(b"0")
        with open(str(vid) + ".analysis.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps({"type": "meta", "v": 1}) + "\n")
            for s in self.SHOTS:
                a = s.get("action")
                f.write(json.dumps({"type": "shot", "start": s["start"],
                                    "end": s["end"], "outcome": s["outcome"],
                                    "pocketed": 0}) + "\n")
                if a:
                    f.write(json.dumps({"type": "action", "start": s["start"],
                                        "action": a}) + "\n")
        assert _sidecar_shot_count(vid) == 3    # 2 strokes + 1 break

    def test_companion_attaches_action(self, tmp_path):
        import json

        from billiards_trainer.companion.server import read_shots
        vid = tmp_path / "s.mp4"
        vid.write_bytes(b"0")
        with open(str(vid) + ".analysis.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps({"type": "shot", "start": 10.0, "end": 14.0,
                                "outcome": "make", "pocketed": 1}) + "\n")
            f.write(json.dumps({"type": "action", "start": 10.0,
                                "action": "ball_in_hand"}) + "\n")
        shots = read_shots(vid)
        assert shots[0]["action"] == "ball_in_hand"


class TestClosePassWiring:
    def test_controller_close_pass_classifies_actions(self):
        """The post-recording daemon must run BOTH passes: outcomes and
        action labels — a new recording is born fully labeled."""
        from pathlib import Path

        import billiards_trainer.workers.controller as m
        src = Path(m.__file__).read_text(encoding="utf-8")
        i = src.find("def _derive")
        assert i > 0 and "classify_and_mark" in src[i:i + 1200], \
            "session-close pass lost action classification"

    def test_backfill_runs_both_passes(self):
        from pathlib import Path
        src = Path("tools/build_analysis_cache.py").read_text(encoding="utf-8")
        assert "derive_and_correct" in src and "classify_and_mark" in src


class TestQuietShuffle:
    def test_objects_displaced_cue_still_nothing_fast_is_rearrange(self):
        """Joe's report: slow-shuffled object balls with the hand mask
        barely firing must not read as a stroke — the cue never moved and
        nothing reached stroke speed."""
        f = {"dur": 6.7, "peak": 114.0, "n_movers": 1, "n_obj_movers": 1,
             "displaced_obj": 3, "cue_displaced": False, "n_fast": 0,
             "hand_frac": 0.11, "carried_frac": 0.0,
             "carried_at_launch": False, "fast_sync": 0.0,
             "set_changed": True, "n_states": 45}
        assert classify(f) == "rearrange"

    def test_blur_invisible_scratch_stroke_is_protected(self):
        """A real stroke whose cue vanished into a pocket has
        cue_displaced=True — the quiet-shuffle rule must not eat it."""
        f = {"dur": 4.3, "peak": 1.0, "n_movers": 0, "n_obj_movers": 0,
             "displaced_obj": 2, "cue_displaced": True, "n_fast": 0,
             "hand_frac": 0.0, "carried_frac": 0.0,
             "carried_at_launch": False, "fast_sync": 0.0,
             "set_changed": True, "n_states": 28}
        assert classify(f) == "stroke"


class TestRolledInBirths:
    """Joe's @214/@466 verdicts: returning balls from the pockets births
    tracks that FIRST APPEAR already rolling and decelerate to rest —
    impossible for a struck ball, which accelerates from a tracked rest
    position. Two such births classify the episode as rearrange no matter
    what the hand gate saw (pocket-mouth reaches stay under it)."""

    def _episode(self, tmp_path, rollers):
        """rollers: list of (birth_t, x0, y0). Each decelerates to rest
        over ~3s after its birth; nothing exists for them before it."""
        states = []
        t = 0.0
        while t <= 20.0:
            tracks = [(1, 300.0, 700.0, 9, "stripe")]
            for k, (bt, x0, y0) in enumerate(rollers):
                if t < bt:
                    continue
                age = min(3.0, t - bt)
                # decelerating roll: fast at birth, at rest by +3s
                d = 150.0 * age - 25.0 * age * age
                tracks.append((10 + k, x0 + d, y0, 2 + k, "solid"))
            states.append((round(t, 2), tracks, []))
            t += 0.1
        return _write(tmp_path, states,
                      [{"start": 8.0, "end": 16.0, "outcome": "miss",
                        "pocketed": 0}])

    def test_two_rolled_in_births_classify_rearrange(self, tmp_path):
        vid = self._episode(tmp_path, [(9.0, 100.0, 200.0),
                                       (12.0, 100.0, 900.0)])
        assert classify_and_mark(vid) == {"rearrange": 1}

    def test_single_birth_stays_default(self, tmp_path):
        vid = self._episode(tmp_path, [(9.0, 100.0, 200.0)])
        assert classify_and_mark(vid) == {"stroke": 1}


class TestVerdictContainmentMatching:
    """A verdict anchored to an OLD segmentation start must survive
    re-analysis: attach by containment, then nearest-within-8s; farther
    stays unattached (never guess)."""

    def _vid(self, tmp_path, verdict_t):
        states = []
        t = 0.0
        while t <= 40.0:
            states.append((round(t, 2), [(1, 300.0, 700.0, 9, "stripe")], []))
            t += 0.25
        vid = _write(tmp_path, states,
                     [{"start": 10.0, "end": 18.0, "outcome": "miss",
                       "pocketed": 0}])
        with open(str(vid) + ".analysis.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"type": "action", "start": verdict_t,
                                "action": "rearrange", "src": "review"})
                    + "\n")
        return vid

    def test_contained_verdict_attaches(self, tmp_path):
        r = SidecarReader(self._vid(tmp_path, 14.0))   # inside [10,18]
        assert r.shots[0]["action"] == "rearrange"
        assert r.shots[0]["_action_reviewed"]

    def test_nearby_verdict_attaches(self, tmp_path):
        r = SidecarReader(self._vid(tmp_path, 5.5))    # 4.5s before start
        assert r.shots[0]["action"] == "rearrange"

    def test_distant_verdict_stays_unattached(self, tmp_path):
        r = SidecarReader(self._vid(tmp_path, 35.0))   # 17s past end
        assert r.shots[0].get("action", "stroke") == "stroke"
