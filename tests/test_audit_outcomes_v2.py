"""Identity-aware outcome derivation: the mechanisms frame-verified on the
9-ball session (derived 11/11 vs live-recorded 7/11).

Each test builds a tiny synthetic sidecar exercising one mechanism:
  - a numbered ball departing            -> make
  - departed number back, same spot, no hands -> flicker, miss
  - departed number back near a hand     -> replaced (Joe re-spotting), make
  - digit-down ball (num=-1 for life) potted -> anonymous resident, make
  - unnumbered resident re-ID'd mid-shot -> newborn cancels credit, miss
  - cue departing                        -> scratch
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from audit_outcomes_v2 import audit  # noqa: E402


def _write(tmp_path, states, shots):
    """states: list of (t, tracks, carried_ids); tracks: (id,x,y,num,cls)."""
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


CUE = (1, 100.0, 100.0, 0, "cue")
NINE = (2, 300.0, 700.0, 9, "stripe")


def _timeline(tmp_path, before_extra, after_extra, carried_at=(), t_end=30.0):
    """Quiet table 0-8s, shot 8-12s, quiet to t_end. Extras join the two
    resident balls (cue + 9) in their respective phases."""
    states = []
    t = 0.0
    while t <= t_end:
        tracks = [CUE, NINE]
        tracks += before_extra if t < 8.0 else after_extra
        carried = list(carried_at) if 12.0 < t < 14.0 else []
        states.append((round(t, 2), tracks, carried))
        t += 0.25
    return _write(tmp_path, states, [{"start": 8.0, "end": 12.0,
                                      "outcome": "miss", "pocketed": 0}])


class TestDerivedOutcomes:
    def test_numbered_departure_is_make(self, tmp_path):
        five = (3, 200.0, 500.0, 5, "solid")
        vid = _timeline(tmp_path, [five], [])
        row = audit(vid)[0]
        assert row["derived"] == "make" and row["departed"] == [5]

    def test_flicker_return_same_spot_no_hands_is_miss(self, tmp_path):
        five = (3, 200.0, 500.0, 5, "solid")
        states = []
        t = 0.0
        while t <= 30.0:
            tracks = [CUE, NINE]
            # gone 8-16s, back at the SAME spot hands-free after
            if t < 8.0 or t > 16.0:
                tracks.append(five)
            states.append((round(t, 2), tracks, []))
            t += 0.25
        vid = _write(tmp_path, states,
                     [{"start": 8.0, "end": 12.0, "outcome": "miss",
                       "pocketed": 0}])
        assert audit(vid)[0]["derived"] == "miss"

    def test_replaced_return_near_hand_is_make(self, tmp_path):
        five = (3, 200.0, 500.0, 5, "solid")
        moved = (9, 400.0, 900.0, 5, "solid")   # new id, new spot
        states = []
        t = 0.0
        while t <= 30.0:
            tracks = [CUE, NINE]
            if t < 8.0:
                tracks.append(five)
            elif t > 16.0:
                tracks.append(moved)
            carried = [9] if 15.0 < t < 17.0 else []
            states.append((round(t, 2), tracks, carried))
            t += 0.25
        vid = _write(tmp_path, states,
                     [{"start": 8.0, "end": 12.0, "outcome": "miss",
                       "pocketed": 0}])
        assert audit(vid)[0]["derived"] == "make"

    def test_digit_down_resident_pot_is_make(self, tmp_path):
        anon = (4, 314.0, 937.0, -1, "unknown")   # the 6-ball, digit down
        vid = _timeline(tmp_path, [anon], [])
        row = audit(vid)[0]
        assert row["derived"] == "make" and row["departed"] == []

    def test_reid_newborn_cancels_anon_credit(self, tmp_path):
        anon = (4, 314.0, 937.0, -1, "unknown")
        reborn = (8, 500.0, 400.0, -1, "unknown")   # same ball, new id+spot
        vid = _timeline(tmp_path, [anon], [reborn])
        assert audit(vid)[0]["derived"] == "miss"

    def test_carried_resident_gets_no_credit(self, tmp_path):
        anon = (4, 314.0, 937.0, -1, "unknown")
        vid = _timeline(tmp_path, [anon], [], carried_at=(4,))
        assert audit(vid)[0]["derived"] == "miss"

    def test_cue_departure_is_scratch(self, tmp_path):
        states = []
        t = 0.0
        while t <= 30.0:
            tracks = [NINE] if t >= 8.0 else [NINE, CUE]
            states.append((round(t, 2), tracks, []))
            t += 0.25
        vid = _write(tmp_path, states,
                     [{"start": 8.0, "end": 12.0, "outcome": "miss",
                       "pocketed": 0}])
        assert audit(vid)[0]["derived"] == "scratch"


class TestDeriveAndCorrect:
    """The session-close pass: derivation appends corrections (append-only,
    idempotent), and a human verdict appended later always wins."""

    def _mismatched(self, tmp_path):
        """Recorded says miss, but ball 5 demonstrably departs."""
        five = (3, 200.0, 500.0, 5, "solid")
        states = []
        t = 0.0
        while t <= 30.0:
            tracks = [CUE, NINE] + ([five] if t < 8.0 else [])
            states.append((round(t, 2), tracks, []))
            t += 0.25
        return _write(tmp_path, states,
                      [{"start": 8.0, "end": 12.0, "outcome": "miss",
                        "pocketed": 0}])

    def test_mismatch_is_corrected_and_idempotent(self, tmp_path):
        from billiards_trainer.vision.analysis_cache import SidecarReader
        from billiards_trainer.vision.outcomes import derive_and_correct
        vid = self._mismatched(tmp_path)
        assert derive_and_correct(vid) == 1
        r = SidecarReader(vid)
        assert r.shots[0]["outcome"] == "make"
        assert not r.shots[0].get("corrected"),             "derived corrections must not wear the human-verdict ring"
        assert derive_and_correct(vid) == 0     # second pass changes nothing

    def test_agreeing_outcome_is_left_alone(self, tmp_path):
        from billiards_trainer.vision.outcomes import derive_and_correct
        five = (3, 200.0, 500.0, 5, "solid")
        states = []
        t = 0.0
        while t <= 30.0:
            tracks = [CUE, NINE] + ([five] if t < 8.0 else [])
            states.append((round(t, 2), tracks, []))
            t += 0.25
        vid = _write(tmp_path, states,
                     [{"start": 8.0, "end": 12.0, "outcome": "make",
                       "pocketed": 1}])
        assert derive_and_correct(vid) == 0

    def test_human_verdict_appended_later_wins(self, tmp_path):
        from billiards_trainer.vision.analysis_cache import (
            SidecarReader, append_correction)
        from billiards_trainer.vision.outcomes import derive_and_correct
        vid = self._mismatched(tmp_path)
        derive_and_correct(vid)                  # derivation says make
        append_correction(vid, 8.0, "miss")      # Joe's review says miss
        assert SidecarReader(vid).shots[0]["outcome"] == "miss"

    def test_missing_sidecar_is_a_quiet_zero(self, tmp_path):
        from billiards_trainer.vision.outcomes import derive_and_correct
        assert derive_and_correct(tmp_path / "nope.mp4") == 0

