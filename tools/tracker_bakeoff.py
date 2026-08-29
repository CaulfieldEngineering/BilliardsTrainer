"""Head-to-head: the LIVE tracker vs the OFFLINE tracker, same input.

Joe, 2026-08-30: "Is there a reason there's a replay engine and a separate
live engine? Ideally they're one and the same with just a live processing and
offline processing mode."

There is no design reason - only history. Two trackers exist:

    vision/tracking.py   BallTracker    the LIVE champion
    measure/tracker.py   MotionTracker  the offline engine, and already a
                                        SHADOW inside MeasurementCore

MeasurementCore was built to Joe's one-engine directive and scores divergence
between them precisely so promotion can be "a measured decision, not a hope"
(its own docstring). This is that measurement, run offline where it can be
repeated: one decode of the bench clip, ONE set of prepared detections, fed to
both trackers, each scored against the same pixel-derived naming truth.

    python tools/tracker_bakeoff.py [--session <name>] [--max-frames N]

Nothing here changes the app. It produces the evidence a promotion needs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
REC = Path(r"C:/Users/Joe/Dropbox/Billiards/BilliardsTrainer")
M1 = Path(r"C:/Users/Joe/AppData/Local/BilliardsTrainer/m1")
IDENT_EVERY = 6


class _Reader:
    """Just enough SidecarReader for tools.scorecard._naming_correctness."""

    def __init__(self, hinv):
        self.meta = {"hinv": hinv}


def _row(tid, x, y, radius, number, active=1, coasting=False):
    # sidecar row layout: (id, x, y, radius, number, cls, active, coasting)
    return (int(tid), float(x), float(y), float(radius), int(number),
            2, 1 if active else 0, bool(coasting))


def run(session: str, max_frames: int) -> dict:
    import cv2
    import numpy as np
    from billiards_trainer.config import Settings
    from billiards_trainer.core.geometry import expected_ball_radius_px
    from billiards_trainer.detector_strategies import discover
    from billiards_trainer.measure.engine import _acquire_calib, _pair_identities
    from billiards_trainer.measure.tracker import MotionTracker
    from billiards_trainer.vision.pipeline import Pipeline
    from billiards_trainer.vision.tracking import BallTracker

    pipe = Pipeline(Settings.load())
    calib = _acquire_calib(REC / session, pipe)
    if calib is None:
        raise SystemExit("no calibration")
    tbl = calib.table
    hinv = np.linalg.inv(np.asarray(calib.H, dtype=float)).tolist()
    exp_r = float(expected_ball_radius_px(tbl, pipe.settings.table.size))

    strat = discover()["ensemble_findid"]
    strat.inference_provider = "dml"
    motion = MotionTracker(pockets=[(p.x, p.y) for p in tbl.pockets],
                           pocket_r=float(tbl.pocket_radius))
    live = BallTracker()

    cap = cv2.VideoCapture(str(REC / session))
    times: list = []
    rows_motion: list = []
    rows_live: list = []
    ident_by_pos: list = []
    fi = 0
    prev_pts = -1.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if fi > 0 and pts <= prev_pts:
            pts = prev_pts + 1 / 30.0
        prev_pts = pts
        found = strat.detect(frame, calib) or []
        if fi % IDENT_EVERY == 0:
            ids = strat._identifier.detect(frame, calib) or []
            ident_by_pos = [(d.x, d.y, d.number) for d in ids
                            if getattr(d, "number", -1) >= 0]
        # the frame matters: _pair_identities also runs the measured-colour
        # correction, which is what repairs the purple-4-read-as-7
        _pair_identities(found, ident_by_pos, frame)
        prepared = pipe.prepare_detections(found, calib, frame.shape,
                                           frame=frame, refresh_foreign=True)
        # ONE input, two trackers.
        m_rows = motion.update([(float(d.x), float(d.y), float(d.radius),
                                 int(getattr(d, "number", -1)))
                                for d in prepared], pts)
        l_tracks = live.update(list(prepared), tbl.short_side,
                               bounds=(tbl.x0, tbl.y0, tbl.x1, tbl.y1),
                               pockets=[(p.x, p.y) for p in tbl.pockets],
                               pocket_r=float(tbl.pocket_radius),
                               ball_r=exp_r)
        times.append(pts)
        rows_motion.append([_row(r.id, r.x, r.y, r.radius, r.number,
                                 True, getattr(r, "coasting", False))
                            for r in m_rows])
        rows_live.append([_row(t.id, t.x, t.y, t.radius, t.number,
                               getattr(t, "active", True), False)
                          for t in l_tracks
                          if getattr(t, "active", True)])
        fi += 1
        if fi % 900 == 0:
            print(f"  {fi} frames ({pts:.0f}s)", flush=True)
        if max_frames and fi >= max_frames:
            break
    cap.release()

    from scorecard import _naming_correctness
    reader = _Reader(hinv)
    return {
        "frames": fi,
        "MotionTracker (offline)": _naming_correctness(reader, times, rows_motion),
        "BallTracker (live)": _naming_correctness(reader, times, rows_live),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="session-20260824-220247.mp4")
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()
    out = run(a.session, a.max_frames)
    print(f"\nBAKEOFF over {out['frames']} frames of {a.session}\n")
    for who in ("BallTracker (live)", "MotionTracker (offline)"):
        c = out[who]
        if not c:
            print(f"{who}: no score"); continue
        print(f"{who}")
        print(f"    named correctly : {c['name_right_pct']}%  "
              f"[wrong {c['name_wrong_frames']}, unnamed {c['name_unnamed_frames']}, "
              f"no track {c['name_missing_frames']}]")
        print(f"    per ball        : " + "  ".join(
            f"{b}:{v['right']}/{v['right'] + v['wrong'] + v['unnamed']}"
            for b, v in c["name_per_ball"].items()))
        if c["name_confusions"]:
            print(f"    confusions      : " + ", ".join(
                f"{k} x{v}" for k, v in list(c["name_confusions"].items())[:6]))
        print()
    (ROOT / "_train" / "bench_fix" / "bakeoff.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
