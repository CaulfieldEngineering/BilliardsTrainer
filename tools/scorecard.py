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
                  pocket_r=calib.table.pocket_radius)
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
        got = bool(eps[bi].pocketed)
        rows.append({"strike": s["strike"], "what": s["what"], "found": True,
                     "outcome_ok": got == s["pot"],
                     "engine": "pot" if got else "no pot",
                     "truth": "pot" if s["pot"] else "no pot"})
    setup = truth.get("setup_windows", [])
    false_strokes = extra = 0
    for i, e in enumerate(eps):
        if i in used:
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
        cues = [x for x in rows_f if x[6] and x[4] == 0]
        if len(cues) == 1:
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
    n = len(rows)
    return {"session": name,
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "detected": f"{found}/{n}", "outcome": f"{ok}/{n}",
            "false_strokes": false_strokes, "extra_episodes": extra,
            "episodes": len(eps), "shots": rows, "caps": caps,
            "perfect": found == n and ok == n
            and false_strokes == 0 and extra == 0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", default=str(ROOT / "docs" / "bench_truth.json"))
    ap.add_argument("--publish", action="store_true")
    a = ap.parse_args()
    sc = score(Path(a.truth))
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
