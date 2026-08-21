"""Trails export: one entry per BALL, never per track segment."""

import json

from billiards_trainer.vision.analysis_cache import SidecarReader
from billiards_trainer.vision.shots_export import _shot_trails


def _write(tmp_path, states, shots):
    vid = tmp_path / "s.mp4"
    vid.write_bytes(b"0")
    with open(str(vid) + ".analysis.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "meta", "v": 1, "fps": 30}) + "\n")
        for t, tracks in states:
            f.write(json.dumps({"type": "f", "t": t,
                                "tracks": tracks}) + "\n")
        for s in shots:
            f.write(json.dumps({"type": "shot", **s}) + "\n")
    return vid


IDENT = {"hinv": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "w": 700, "h": 1300}


def test_track_churn_merges_into_one_ball(tmp_path):
    # the cue (num 0) travels as tid 1, dies to blur, resumes as tid 9;
    # an unnumbered churn fragment (tid 5) also moves. Joe saw TWO cue
    # badges and a "?" — the export must yield exactly one entry, n=0.
    states = []
    t = 7.0
    while t <= 13.0:
        rows = []
        x = 100 + (t - 8.0) * 120 if t >= 8.0 else 100
        if t < 10.0:
            rows.append([1, x, 200.0, 12.0, 0, "cue", True])
        elif t >= 10.5:
            rows.append([9, x, 200.0, 12.0, 0, "cue", True])
        if 8.5 <= t <= 9.6:
            rows.append([5, x - 30, 260.0, 12.0, -1, "solid", True])
        states.append((round(t, 2), rows))
        t += 0.15
    vid = _write(tmp_path, states,
                 [{"start": 8.0, "end": 12.0, "outcome": "miss"}])
    r = SidecarReader(vid)
    trails = _shot_trails(r, r.shots[0], IDENT)
    assert [e["n"] for e in trails] == [0], \
        f"expected one cue entry, got {[e['n'] for e in trails]}"
    ts = [p[0] for p in trails[0]["p"]]
    assert ts == sorted(ts), "merged polyline must be time-ordered"
