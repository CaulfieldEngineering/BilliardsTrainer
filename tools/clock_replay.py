"""Score the SHOT CLOCK against a clip's known strokes.

Joe, 2026-08-31, watching the app run: "its false positively restarting
the shot clock frequently as though the cue ball has been struck." He
was right, and the reason it was never caught is that the clock is the
one live-facing behaviour with NO GATE. The scorecard measures strokes,
outcomes, pots and naming; nothing measured whether the countdown starts
and stops at the right moments.

This is that gate, and it is built on the measurement core's own output:
it replays the controller's clock rules over a sidecar - the same
MotionTracker rows the live path publishes - so what is scored here is
what Joe sees.

A correct clock does ONE start and ONE stop per stroke: the countdown
begins when the cue ball comes to rest, and the strike ends it. So the
scoring is:

    starts        should equal the number of strokes
    spurious      a start whose clock stops with NO stroke near it -
                  the exact symptom Joe reported
    missed        a stroke with no running clock to stop
    churn         start->stop->start inside 2s, which is the clock
                  visibly flickering rather than counting down

Speed is recomputed from consecutive sidecar positions the way the
tracker does it - (dx - x)/dt, in rectified px per SECOND - because the
sidecar stores positions, not velocity, and because reconstructing it
here proves the units rather than trusting a comment about them.

    python tools/clock_replay.py [--truth docs/bench_truth.json]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

M1 = Path(r"C:/Users/Joe/AppData/Local/BilliardsTrainer/m1")

# The controller's live constants, mirrored here so the replay scores the
# SHIPPING rule and a change to either side shows up as a diff.
from billiards_trainer.workers.controller import PipelineController  # noqa: E402

CUE_MOVE_SPEED = PipelineController._CUE_MOVE_SPEED
CUE_REST_SPEED = PipelineController._CUE_REST_SPEED
CUE_STOP_FRAMES = PipelineController._CUE_STOP_FRAMES
CUE_GAP_S = PipelineController._CUE_GAP_S

# What shipped before 2026-08-31, for the before/after column: 3.0 as the
# strike threshold and settings.balls.stop_speed's 0.4 floor as "at rest",
# both written as px/frame against a px/second quantity.
OLD_MOVE, OLD_REST = 3.0, 0.4


def _frames(video_name: str):
    """(t, [(id, x, y, number, cls, active, coasting)]) per sidecar frame."""
    p = M1 / f"{video_name}.analysis.jsonl"
    for line in p.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        if d.get("type") != "f":
            continue
        rows = [(int(r[0]), float(r[1]), float(r[2]), int(r[4]), r[5],
                 bool(r[6]), bool(r[7]) if len(r) > 7 else False)
                for r in d.get("tracks", [])]
        yield float(d["t"]), rows


def replay(video_name: str, move_speed: float = CUE_MOVE_SPEED,
           rest_speed: float = CUE_REST_SPEED) -> dict:
    """Run the controller's clock rules over the sidecar."""

    cue_still = 0
    armed = True
    running = False
    saw_cue_t = -1e9
    events: list[tuple[float, str]] = []

    # Drive the REAL tracker. An earlier version of this file recomputed
    # velocity from sidecar positions, which looked reasonable and was
    # wrong by a factor of hundreds: the tracker PREDICTS each track
    # forward and damps it by FRICTION before the measurement update, so
    # the update's `(dx - tr.x + tr.vx*dt)` is a residual correction, not
    # a raw step. Reproducing the one line without the loop around it
    # turned a smoother into an accumulator and reported a resting ball
    # at ~1000 px/s. Track.speed is the controller's input, so the
    # tracker itself has to produce it.
    from billiards_trainer.measure.tracker import MotionTracker
    tk = MotionTracker()
    for t, rows in _frames(video_name):
        # SIGHTINGS only - a coasted row is the tracker's own estimate,
        # and feeding an estimate back in as evidence is circular.
        dets = [(x, y, 16.0, n) for _tid, x, y, n, _c, active, co in rows
                if active and not co]
        live = {tr.id: tr for tr in tk.update(dets, t)}
        speed = {tid: tr.speed for tid, tr in live.items()}
        cue = next(((tr.id, tr.x, tr.y, tr.number, tr.cls, True, tr.coasting)
                    for tr in live.values() if tr.number == 0), None)
        rows = [(tr.id, tr.x, tr.y, tr.number, tr.cls, True, tr.coasting)
                for tr in live.values()]
        if cue is None:
            cue_still = 0
            continue
        if t - saw_cue_t > CUE_GAP_S:
            armed = True
        saw_cue_t = t

        v = speed.get(cue[0], 0.0)
        if v > move_speed:
            cue_still = 0
            if running:
                running = False
                events.append((t, "stop"))
            armed = True
            continue
        if v < rest_speed:
            others = any(r[3] != 0 and r[5]
                         and speed.get(r[0], 0.0) > move_speed
                         for r in rows)
            if others:
                cue_still = 0
                continue
            cue_still += 1
            if cue_still >= CUE_STOP_FRAMES and armed and not running:
                running = True
                armed = False
                events.append((t, "start"))
        else:
            cue_still = 0
    return {"events": events}


def score(events, strikes, near_s: float = 3.0) -> dict:
    starts = [t for t, e in events if e == "start"]
    stops = [t for t, e in events if e == "stop"]
    spurious = [t for t in stops
                if not any(abs(t - s) <= near_s for s in strikes)]
    missed = [s for s in strikes
              if not any(abs(t - s) <= near_s for t in stops)]
    churn = sum(1 for i in range(1, len(events))
                if events[i][1] == "start" and events[i - 1][1] == "stop"
                and events[i][0] - events[i - 1][0] < 2.0)
    return {"starts": len(starts), "stops": len(stops),
            "spurious_stops": len(spurious), "missed_strikes": len(missed),
            "churn": churn, "spurious_t": [round(x, 1) for x in spurious[:8]]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", default=str(ROOT / "docs" / "bench_truth.json"))
    ap.add_argument("--old", action="store_true",
                    help="score the pre-2026-08-31 thresholds as well")
    a = ap.parse_args()

    truth = json.loads(Path(a.truth).read_text(encoding="utf-8"))
    video = truth["session"]
    if not video.endswith(".mp4"):
        video += ".mp4"
    strikes = [float(s["strike"]) for s in truth["strokes"]]

    runs = [("NOW", replay(video))]
    if a.old:
        runs.insert(0, ("BEFORE", replay(video, OLD_MOVE, OLD_REST)))
    print(f"SHOT CLOCK  {video}   {len(strikes)} real strokes\n")
    print(f"{'':22}" + "".join(f"{tag:>12}" for tag, _ in runs))
    scored = [(tag, score(r["events"], strikes)) for tag, r in runs]
    for label, key in (("clock starts", "starts"), ("clock stops", "stops"),
                       ("SPURIOUS stops", "spurious_stops"),
                       ("missed strikes", "missed_strikes"),
                       ("churn (restart<2s)", "churn")):
        print(f"  {label:20}" + "".join(f"{sc[key]:>12}" for _t, sc in scored))
    print(f"\n  a correct clock does {len(strikes)} starts and {len(strikes)} "
          f"stops, with 0 spurious, 0 missed and 0 churn")
    for tag, sc in scored:
        if sc["spurious_t"]:
            print(f"  {tag} spurious at: {sc['spurious_t']}")


if __name__ == "__main__":
    main()
