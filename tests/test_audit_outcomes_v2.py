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



def _write_v2(tmp_path, states, shots):
    """Like _write but tracks carry an explicit active flag:
    (id, x, y, num, cls, active)."""
    vid = tmp_path / "s.mp4"
    vid.write_bytes(b"0")
    with open(str(vid) + ".analysis.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "meta", "v": 1, "fps": 30}) + "\n")
        for t, tracks, carried in states:
            d = {"type": "f", "t": t,
                 "tracks": [[i, x, y, 12.0, n, c, a]
                            for i, x, y, n, c, a in tracks]}
            if carried:
                d["c"] = carried
            f.write(json.dumps(d) + "\n")
        for s in shots:
            f.write(json.dumps({"type": "shot", **s}) + "\n")
    return vid


class TestTidContinuityCancellation:
    """The 005647 forensics: a returning ball that rode the SAME never-died
    track through the whole gap (coasting on spot-occupancy) was never off
    the bed — even when a bridge hand rests on it or a stick-nudge moves it
    before its first re-detection. A truly potted-and-replaced ball returns
    on a FRESH track id, which must still count as a departure."""

    def _states(self, five_row):
        """five_row(t) -> track tuple or None; residents always present."""
        cue = (1, 100.0, 100.0, 0, "cue", True)
        nine = (2, 300.0, 700.0, 9, "stripe", True)
        states = []
        t = 0.0
        while t <= 30.0:
            tracks = [cue, nine]
            row = five_row(t)
            if row:
                tracks.append(row)
            # hand over the table around the return, like Joe's next-shot
            # address / re-spot reach
            carried = [3] if 14.0 < t < 17.0 else []
            states.append((round(t, 2), tracks, carried))
            t += 0.25
        return states

    def test_occluded_coasting_return_cancels_departure(self, tmp_path):
        # 5-ball active at rest until the shot ends, coasts (active=False)
        # through the after-window, re-activates on the SAME tid at a NEW
        # spot with a hand present (stick-nudge + bridge hand): old rule
        # said 'replaced' -> make; tid continuity says occlusion -> miss.
        def five(t):
            if t <= 12.2:
                return (3, 200.0, 500.0, 5, "solid", True)
            if t < 16.0:
                return (3, 200.0, 500.0, 5, "solid", False)   # coasting
            return (3, 260.0, 560.0, 5, "solid", True)        # nudged
        vid = _write_v2(tmp_path, self._states(five),
                        [{"start": 8.0, "end": 12.0, "outcome": "miss",
                          "pocketed": 0}])
        row = audit(vid)[0]
        assert row["derived"] == "miss", \
            "occluded coasting ball booked as a pot"
        assert row["returned"][0]["mode"] == "flicker"

    def test_fresh_tid_return_near_hand_still_departs(self, tmp_path):
        # Control: the 5's track DIES at the shot (real pot); a fresh tid
        # brings the number back near a hand (Joe re-spotting) -> the
        # departure must stand (make), exactly as before this rule.
        def five(t):
            if t < 12.2:
                return (3, 200.0, 500.0, 5, "solid", True)
            if t >= 16.0:
                return (9, 260.0, 560.0, 5, "solid", True)    # fresh tid
            return None                                        # track dead
        vid = _write_v2(tmp_path, self._states(five),
                        [{"start": 8.0, "end": 12.0, "outcome": "miss",
                          "pocketed": 0}])
        row = audit(vid)[0]
        assert row["derived"] == "make", \
            "real pot-and-replace was wrongly cancelled"
        assert row["returned"][0]["mode"] == "replaced"


class TestRespotDepartures:
    """005647 @275s: the cue scratched into a pocket and Joe fished it out
    and re-spotted it INSIDE the still-open shot window — the same-number
    rebirth made the census read 'present at end' and the scratch
    vanished. A pocket-death + hand-in-gap + fresh-tid rebirth CONFIRMS
    the departure; a resting ball picked up by hand does not."""

    def _sidecar(self, tmp_path, flight, rebirth_pos=(300.0, 400.0)):
        # bed extremes defined by a far resident at (600,1200)
        states = []
        t = 0.0
        while t <= 24.0:
            tracks = [(2, 600.0, 1200.0, 9, "stripe", True)]
            row = flight(t)
            if row:
                tracks.append(row)
            carried = [9] if 11.5 < t < 13.0 else []   # hand in the gap
            states.append((round(t, 2), tracks, carried))
            t += 0.25
        return _write_v2(tmp_path, states,
                         [{"start": 8.0, "end": 16.0, "outcome": "make",
                           "pocketed": 1}])

    def test_pocket_death_and_respot_inside_window_is_departure(self, tmp_path):
        def flight(t):
            if t < 8.0:
                return (1, 100.0, 100.0, 0, "cue", True)     # at rest
            if t <= 9.0:                                     # rolls to pocket
                f = (t - 8.0) / 1.0
                return (1, 100 + 495 * f, 100 + 1095 * f, 0, "cue", True)
            if t >= 14.0:                                    # hand re-spot
                return (9, 300.0, 400.0, 0, "cue", True)
            return None                                      # in the pocket
        vid = self._sidecar(tmp_path, flight)
        row = audit(vid)[0]
        assert row["derived"] == "scratch", \
            "fast hand re-spot inside the window hid the scratch"

    def test_resting_ball_hand_moved_is_not_departure(self, tmp_path):
        def flight(t):
            # parked AT the pocket corner the whole time — never arrived
            # by rolling; vanishes mid-shot; hand-returned mid-table
            if t < 10.0:
                return (1, 595.0, 1195.0, 0, "cue", True)
            if t >= 14.0:
                return (9, 300.0, 400.0, 0, "cue", True)
            return None
        vid = self._sidecar(tmp_path, flight)
        row = audit(vid)[0]
        assert row["derived"] == "miss", \
            "a hand-moved resting ball was booked as a pot"
