"""Score the engine against VISION GROUND TRUTH and publish it.

Joe, 2026-08-28: "We can't just say 'see you in two weeks'... How can
we break this into intermediate phases and milestones to prove we're
on the right track?" This is the proof: one number, recomputed from
the current engine output after every round, written where Joe can
read it without asking.

    python tools/scorecard.py [--truth docs/bench_truth.json] [--publish]

Scores (all against docs/bench_truth.json, established by eye):
  detected      real strokes the engine found (windowed within 2.5s)
  outcome       of those, how many pot/no-pot verdicts match
  false_strokes episodes fired inside a hand-setup window
  extra         episodes that match no real stroke and no setup window
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
M1 = Path(r"C:/Users/Joe/AppData/Local/BilliardsTrainer/m1")
REC = Path(r"C:/Users/Joe/Dropbox/Billiards/BilliardsTrainer")
PUB = ROOT / "companion-cloud" / "public" / "scorecard.json"
MATCH_S = 2.5


def _evidence(name: str, shots: list, out_dir: Path) -> None:
    """One overlay frame per scored shot: the app's beliefs drawn on the
    real video at that moment (Joe: "I would like screenshots or some
    kind of citation for all of this data"). Small JPEGs - this
    republishes every hour."""
    import cv2
    from debug_overlay import render
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in shots:
        t_at = float(s["strike"]) + 0.6
        try:
            made = render(REC / name, [t_at], out_dir)
        except Exception:  # noqa: BLE001 - evidence is never fatal
            continue
        if not made:
            continue
        img = cv2.imread(str(made[0]))
        made[0].unlink(missing_ok=True)
        if img is None:
            continue
        h, w = img.shape[:2]
        sc2 = 720.0 / max(w, 1)
        small = cv2.resize(img, (720, int(h * sc2)))
        fn = f"shot_{str(s['strike']).replace('.', '_')}.jpg"
        cv2.imwrite(str(out_dir / fn), small,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 72])
        s["img"] = f"journal/evidence/{fn}"


def score(truth_path: Path) -> dict:
    logging.disable(logging.CRITICAL)
    from billiards_trainer.config import Settings
    from billiards_trainer.measure.engine import _acquire_calib
    from billiards_trainer.measure.shots import analyze
    from billiards_trainer.vision.analysis_cache import SidecarReader
    from billiards_trainer.vision.pipeline import Pipeline

    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    name = truth["session"]
    pipe = Pipeline(Settings.load())
    calib = _acquire_calib(REC / name, pipe)
    r = SidecarReader(M1 / name)
    eps = analyze(r._times, r._frames,
                  pockets=[(p.x, p.y) for p in calib.table.pockets],
                  pocket_r=calib.table.pocket_radius,
                  carried=getattr(r, "_carried", None))
    rows, used = [], set()
    for s in truth["strokes"]:
        best, bi = None, -1
        for i, e in enumerate(eps):
            if i in used:
                continue
            d = abs(e.t_strike - s["strike"])
            if d <= MATCH_S and (best is None or d < best):
                best, bi = d, i
        if bi < 0:
            rows.append({"strike": s["strike"], "what": s["what"],
                         "found": False, "outcome_ok": False})
            continue
        used.add(bi)
        e = eps[bi]
        got = bool(e.pocketed)
        got_balls = [b for b, _x, _y in e.pocketed]
        row = {"strike": s["strike"], "what": s["what"], "found": True,
               "outcome_ok": got == s["pot"],
               "engine": "pot" if got else "no pot",
               "truth": "pot" if s["pot"] else "no pot"}
        # ATTRIBUTION (Joe: with object balls satisfied there is no such
        # thing as a pot by an unnamed ball) - a pot must name the ball
        if s["pot"] and s.get("ball") is not None:
            row["ball_truth"] = s["ball"]
            row["ball_engine"] = got_balls[0] if got_balls else None
            row["ball_ok"] = bool(got_balls) and got_balls[0] == s["ball"]
        rows.append(row)
    setup = truth.get("setup_windows", [])
    false_strokes = extra = setup_labelled = 0
    for i, e in enumerate(eps):
        if i in used:
            continue
        if getattr(e, "setup", False):
            setup_labelled += 1      # correctly called hand-work, not a shot
            continue
        if any(a <= e.t_strike <= b for a, b, *_ in setup):
            false_strokes += 1
        else:
            extra += 1
    # ---- capability metrics (Joe's ladder: cue tracking, object-ball
    # ID, make/miss) - computed from the same dense stream ------------
    known = set(truth.get("balls_on_table", []))
    times, frames = r._times, r._frames
    cue_ok = cue_bad = 0
    moving_named = moving_unnamed = 0
    invented: dict = {}
    prev: dict = {}
    for j, rows_f in enumerate(frames):
        # R1 correctness (demoted 2026-08-28 after a vision check found
        # the cue label on empty felt): exactly ONE cue track AND it must
        # be a real sighting this frame, not a coasted estimate.
        cues = [x for x in rows_f if x[6] and x[4] == 0]
        live = [x for x in cues if not (len(x) > 7 and x[7])]
        if len(cues) == 1 and len(live) == 1:
            cue_ok += 1
        else:
            cue_bad += 1
        for tr in rows_f:
            if not tr[6]:
                continue
            n = tr[4]
            if n >= 0 and known and n not in known:
                invented[n] = invented.get(n, 0) + 1
            key = tr[0]
            p0 = prev.get(key)
            prev[key] = (times[j], tr[1], tr[2])
            if p0 is None:
                continue
            dt = times[j] - p0[0]
            if not (0 < dt < 0.2):
                continue
            v = ((tr[1] - p0[1]) ** 2 + (tr[2] - p0[2]) ** 2) ** 0.5 / dt
            if v > 60.0:
                if n >= 0:
                    moving_named += 1
                else:
                    moving_unnamed += 1
    tot_mov = moving_named + moving_unnamed
    caps = {
        "cue_named_pct": round(100.0 * cue_ok / max(1, cue_ok + cue_bad), 1),
        "named_moving_pct": round(100.0 * moving_named / max(1, tot_mov), 1),
        "invented_numbers": sorted(invented),
        "invented_frames": sum(invented.values()),
    }
    found = sum(1 for x in rows if x["found"])
    ok = sum(1 for x in rows if x["outcome_ok"])
    attr_rows = [x for x in rows if "ball_ok" in x]
    attr_ok = sum(1 for x in attr_rows if x["ball_ok"])
    caps["pot_attribution"] = f"{attr_ok}/{len(attr_rows)}"
    n = len(rows)
    return {"session": name,
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "detected": f"{found}/{n}", "outcome": f"{ok}/{n}",
            "false_strokes": false_strokes, "extra_episodes": extra,
            "episodes": len(eps), "shots": rows, "caps": caps,
            "setup_labelled": setup_labelled,
            "perfect": found == n and ok == n
            and false_strokes == 0 and extra == 0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", default=str(ROOT / "docs" / "bench_truth.json"))
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--evidence", action="store_true",
                    help="render one overlay frame per shot as citation")
    a = ap.parse_args()
    sc = score(Path(a.truth))
    if a.evidence:
        _evidence(sc["session"], sc["shots"],
                  ROOT / "companion-cloud" / "public" / "journal" / "evidence")
    print(f"BENCH SCORECARD  {sc['session']}")
    print(f"  strokes found   : {sc['detected']}")
    print(f"  outcomes right  : {sc['outcome']}")
    print(f"  fake strokes    : {sc['false_strokes']} (fired during hand setup)")
    print(f"  unexplained     : {sc['extra_episodes']}")
    c = sc["caps"]
    print(f"  cue ball named  : {c['cue_named_pct']}% of frames (target 99+)")
    print(f"  moving balls named: {c['named_moving_pct']}% (target 95+)")
    print(f"  invented numbers: {c['invented_numbers']} "
          f"({c['invented_frames']} frames, target none)")
    print(f"  pots attributed : {c['pot_attribution']} (right ball named)")
    for row in sc["shots"]:
        mark = "OK " if row["found"] and row["outcome_ok"] else "XX "
        got = row.get("engine", "MISSED")
        print(f"  {mark}{row['strike']:6.1f}s  engine={got:7} "
              f"truth={row.get('truth',''):7} {row['what'][:44]}")
    if a.publish:
        PUB.write_text(json.dumps(sc, indent=1), encoding="utf-8")
        print(f"published -> {PUB}")


if __name__ == "__main__":
    main()
