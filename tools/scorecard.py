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


NAMING_TRUTH = ROOT / "docs" / "bench_naming_truth.json"
NAME_TOL_PX = 30.0      # video pixels; a ball's radius here is ~13


def _naming_correctness(r, times, frames) -> dict:
    """Is each ball called by its RIGHT name? (not merely called something)

    Round 27 shipped a change that lifted every naming number on this
    scorecard while renaming the red 3 to "1" in 1843 frames. Nothing
    caught it, because "named" and "on the inventory" were the only tests
    and a wrong-but-valid name passes both. This compares the app's name
    for a ball against pixel-derived truth, per ball, and reports the
    confusions by name so a swap can never hide inside an average again.

    Truth comes from docs/bench_naming_truth.json (tools/build_naming_truth.py),
    which is derived from colour and the stripe window, never from the app.
    """
    import bisect

    import numpy as np
    try:
        doc = json.loads(NAMING_TRUTH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    # NAMING TRUTH BELONGS TO ONE CLIP (round 54). This path is fixed,
    # so scoring any OTHER session with --truth silently compared the
    # engine against the BENCH's ball positions: the first cold clip
    # came back "NAMED CORRECTLY 4.8%, no track 881" - a number with no
    # meaning at all, printed with the same confidence as a real one. A
    # metric that scores the wrong clip is worse than a missing metric,
    # because it looks like evidence.
    want = str(doc.get("session") or "")
    got = str(r.meta.get("source") or "")
    if want and got and want != got:
        return {"name_truth_mismatch": f"{want} (scoring {got})"}
    try:
        hinv = np.asarray(r.meta["hinv"], dtype=float)
    except (KeyError, TypeError, ValueError):
        return {}

    def to_video(x, y):
        v = hinv @ np.array([x, y, 1.0])
        return v[0] / v[2], v[1] / v[2]

    per: dict = {}
    confusions: dict = {}
    for s in doc.get("samples", []):
        t = s["t"]
        j = bisect.bisect_left(times, t)
        if j and (j >= len(times) or abs(times[j - 1] - t) < abs(times[j] - t)):
            j -= 1
        if j >= len(times) or abs(times[j] - t) > 0.15:
            continue
        live = [x for x in frames[j] if x[6]]
        placed = [(to_video(x[1], x[2]), x) for x in live]
        for n, tx, ty in s["balls"]:
            best, bd = None, 1e9
            for (vx, vy), row in placed:
                d = ((vx - tx) ** 2 + (vy - ty) ** 2) ** 0.5
                if d < bd:
                    bd, best = d, row
            e = per.setdefault(int(n), {"right": 0, "wrong": 0,
                                        "unnamed": 0, "missing": 0})
            if best is None or bd > NAME_TOL_PX:
                e["missing"] += 1          # truth says a ball is here; no track
            elif int(best[4]) < 0:
                e["unnamed"] += 1
            elif int(best[4]) == int(n):
                e["right"] += 1
            else:
                e["wrong"] += 1
                k = f"{int(n)}->{int(best[4])}"
                confusions[k] = confusions.get(k, 0) + 1
    if not per:
        return {}
    tot = {k: sum(v[k] for v in per.values())
           for k in ("right", "wrong", "unnamed", "missing")}
    seen = tot["right"] + tot["wrong"] + tot["unnamed"]
    return {
        "name_right_pct": round(100.0 * tot["right"] / max(1, seen), 1),
        # THE UNFORGIVING FIGURE (round 49, raised by Joe 2026-08-30:
        # "what does it mean to correctly name 99.6% of balls"). The
        # line above divides by right+wrong+unnamed, so every check
        # where truth says a ball IS here and the app has no live
        # sighting is dropped from the denominator - 88 of 1096 on the
        # bench, 8% of the evidence, invisible in the headline. Worse,
        # that headline RISES as tracking gets WORSE: lose a ball and
        # the denominator shrinks with it, so a detector that gave up
        # entirely would score 100%. This one divides by EVERY truth
        # check, so being blind costs exactly what being wrong costs.
        "name_right_all_pct": round(
            100.0 * tot["right"] / max(1, seen + tot["missing"]), 1),
        "name_checks_total": seen + tot["missing"],
        "name_wrong_frames": tot["wrong"],
        "name_unnamed_frames": tot["unnamed"],
        "name_missing_frames": tot["missing"],
        "name_confusions": dict(sorted(confusions.items(),
                                       key=lambda kv: -kv[1])),
        "name_per_ball": {str(k): per[k] for k in sorted(per)},
    }


def _cue_absent_windows() -> list:
    """Seconds where the CUE BALL IS NOT ON THE TABLE, from the naming truth.

    The cue metric below demands a live cue sighting on EVERY frame, and
    on the cold clip that cost 4.6% - measured in round 72, essentially
    all of it one 7-second window where the cue ball is IN A POCKET. It
    is potted at ~101.4s, sits in the jaws, drops, and Joe reaches in and
    replaces it at ~108.6s. Through all of it the app correctly has no
    cue track, and the metric counted every frame as a failure: it was
    penalising the engine for refusing to hallucinate a ball.

    Absence is taken from the naming truth, which is pixel-derived and
    already eye-checked - no new hand-labelled data, one owner for the
    fact. Only a RUN of consecutive samples missing the cue counts: a
    lone missing sample may be the yardstick ABSTAINING rather than the
    ball being gone, and abstention must never excuse the engine. On the
    cold clip that yields exactly one window (102-108s, matching a
    pixel sweep that puts the ball off the bed 101.5-108.5s); on the
    bench, none - its cue is present in all 221 samples.
    """
    try:
        doc = json.loads(NAMING_TRUTH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    ts = sorted(float(s["t"]) for s in doc.get("samples", []))
    if len(ts) < 3:
        return []
    step = min((b - a) for a, b in zip(ts, ts[1:])) or 1.0
    missing = sorted(float(s["t"]) for s in doc.get("samples", [])
                     if not any(int(b[0]) == 0 for b in s["balls"]))
    runs: list = []
    for t in missing:
        if runs and t - runs[-1][1] <= 1.5 * step:
            runs[-1][1] = t
        else:
            runs.append([t, t])
    # a single sample is not evidence of absence - see the docstring
    return [(a - step / 2.0, b + step / 2.0) for a, b in runs if b > a]


def _publishes_what_it_saw(frames) -> dict:
    """Does the app SHOW the name its own evidence supports?

    Five gates sit between a track's votes and the name it publishes -
    the vote majority, uniqueness arbitration, the age bar, hysteresis,
    and the final uniqueness belt - and any of them can suppress a
    correct read. Round 65 found a track publishing "13" on 330 frames
    while its own reads backed that on EIGHT of 366, and round 67 found
    the identifier reading 9 one pixel from a ball the app called 1.
    Both took a bespoke GPU sweep to see, because the evidence behind a
    published name was never recorded anywhere.

    It is now (Track.read, sidecar element 8), so this compares the two
    on every live sighting and reports the disagreement by direction.
    CONTRADICTED is the serious one: the track saw m, said n, and both
    are real names - the shape of both incidents above. SUPPRESSED is
    the track declining to name what it saw, which duplicate-prevention
    does deliberately and often; it is reported, not judged.
    """
    agree = contradicted = suppressed = invented = 0
    worst: dict = {}
    for rows in frames:
        for r in rows:
            if not r[6] or len(r) < 9:
                continue
            shown, saw = int(r[4]), int(r[8])
            if shown == saw:
                agree += 1
            elif shown < 0:
                suppressed += 1
            elif saw < 0:
                invented += 1
            else:
                contradicted += 1
                k = f"saw {saw} -> said {shown}"
                worst[k] = worst.get(k, 0) + 1
    tot = agree + contradicted + suppressed + invented
    if not tot:
        return {}
    return {
        "publish_agree_pct": round(100.0 * agree / tot, 1),
        "publish_checked": tot,
        "publish_contradicted": contradicted,
        "publish_suppressed": suppressed,
        "publish_unbacked": invented,
        "publish_worst": dict(sorted(worst.items(), key=lambda kv: -kv[1])[:6]),
    }


def score(truth_path: Path) -> dict:
    logging.disable(logging.CRITICAL)
    from billiards_trainer.config import Settings
    from billiards_trainer.measure.engine import _acquire_calib
    from billiards_trainer.measure.shots import analyze, is_stroke
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
        if not is_stroke(e):
            # hand-work, or table motion with no cue ball involved: not a
            # stroke by definition (round 16 - every real stroke on the
            # bench moves the cue; tossed-in balls do not). THE SAME
            # function the engine classifies with, not a copy of it -
            # this test had drifted and was still failing the bench for a
            # fake stroke the engine had already stopped emitting
            # (round 51).
            setup_labelled += 1
            continue
        if any(a <= e.t_strike <= b for a, b, *_ in setup):
            false_strokes += 1
        else:
            extra += 1
    # ---- capability metrics (Joe's ladder: cue tracking, object-ball
    # ID, make/miss) - computed from the same dense stream ------------
    # (naming CORRECTNESS lives in _naming_correctness below - the
    # presence-only metrics here cannot see a ball named as another ball)
    known = set(truth.get("balls_on_table", []))
    times, frames = r._times, r._frames
    cue_ok = cue_bad = cue_skipped = 0
    cue_gone = _cue_absent_windows()
    moving_named = moving_unnamed = 0
    invented: dict = {}
    prev: dict = {}
    for j, rows_f in enumerate(frames):
        # R1 correctness (demoted 2026-08-28 after a vision check found
        # the cue label on empty felt): exactly ONE cue track AND it must
        # be a real sighting this frame, not a coasted estimate.
        cues = [x for x in rows_f if x[6] and x[4] == 0]
        live = [x for x in cues if not (len(x) > 7 and x[7])]
        if any(a <= times[j] <= b for a, b in cue_gone):
            cue_skipped += 1        # the ball is in a pocket; see above
        elif len(cues) == 1 and len(live) == 1:
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
            # A COASTED ROW IS NOT A MOVING BALL. This counted every
            # active row, estimates included, so a coasting ghost's
            # PREDICTION DRIFT registered as a ball in flight - and once
            # round 77 correctly stopped that ghost from holding a real
            # ball's number, its rows became "moving and unnamed" and
            # dragged this figure 99.3 -> 98.7%. Measured: all 11 of the
            # bench's contested cases are one coasting track at
            # 18.81-19.11s, clocked at up to 2,052 units/s while sitting
            # on empty felt. The metric was penalising the engine for
            # refusing to name a ghost.
            # The cue metric above has demanded a real sighting since
            # 2026-08-28 for exactly this reason; this one had not caught
            # up. `coasting` is sidecar element 7.
            if len(tr) > 7 and tr[7]:
                continue
            if v > 60.0:
                if n >= 0:
                    moving_named += 1
                else:
                    moving_unnamed += 1
    tot_mov = moving_named + moving_unnamed
    caps = {
        "cue_named_pct": round(100.0 * cue_ok / max(1, cue_ok + cue_bad), 1),
        # ALWAYS REPORTED, never silent: a metric that drops frames from
        # its own denominator has to say how many, or the number stops
        # meaning anything.
        "cue_frames_skipped": cue_skipped,
        "cue_absent_windows": [[round(a, 1), round(b, 1)] for a, b in cue_gone],
        "named_moving_pct": round(100.0 * moving_named / max(1, tot_mov), 1),
        "invented_numbers": sorted(invented),
        "invented_frames": sum(invented.values()),
    }
    caps.update(_naming_correctness(r, times, frames))
    caps.update(_publishes_what_it_saw(frames))
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
    global NAMING_TRUTH
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", default=str(ROOT / "docs" / "bench_truth.json"))
    ap.add_argument("--naming-truth", default=None,
                    help="per-clip naming truth (docs/<clip>_naming_truth.json). "
                         "Without it the BENCH's file is used, and scoring "
                         "another session against it is refused rather than "
                         "reported as a number (round 54).")
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--evidence", action="store_true",
                    help="render one overlay frame per shot as citation")
    a = ap.parse_args()
    if getattr(a, 'naming_truth', None):
        NAMING_TRUTH = Path(a.naming_truth)
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
    _sk = c.get("cue_frames_skipped", 0)
    print(f"  cue ball named  : {c['cue_named_pct']}% of frames (target 99+)"
          + (f" [{_sk} frames skipped: cue in a pocket, "
             f"{c['cue_absent_windows']}]" if _sk else ""))
    print(f"  moving balls named: {c['named_moving_pct']}% (target 95+)")
    if c.get("name_truth_mismatch"):
        print(f"  NAMED CORRECTLY : not scored - the naming truth file is for "
              f"{c['name_truth_mismatch']}")
    if "name_right_pct" in c:
        print(f"  NAMED CORRECTLY : {c['name_right_pct']}% of ball sightings "
              f"(target 95+)  [wrong {c['name_wrong_frames']}, "
              f"unnamed {c['name_unnamed_frames']}, "
              f"no track {c['name_missing_frames']}]")
        print(f"  ...OF ALL CHECKS: {c['name_right_all_pct']}% of "
              f"{c['name_checks_total']} pixel-truth checks (target 95+) "
              f"- counts BLIND as failure, not just WRONG")
        if c["name_confusions"]:
            worst = ", ".join(f"{k} x{v}" for k, v in
                              list(c["name_confusions"].items())[:5])
            print(f"  confusions      : {worst}")
        per = c["name_per_ball"]
        line = "  per ball        : " + "  ".join(
            f"{b}:{v['right']}/{v['right'] + v['wrong'] + v['unnamed']}"
            for b, v in per.items())
        print(line)
    print(f"  invented numbers: {c['invented_numbers']} "
          f"({c['invented_frames']} frames, target none)")
    print(f"  pots attributed : {c['pot_attribution']} (right ball named)")
    if "publish_agree_pct" in c:
        print(f"  shows what it saw: {c['publish_agree_pct']}% of "
              f"{c['publish_checked']} live sightings "
              f"[contradicted {c['publish_contradicted']}, "
              f"suppressed {c['publish_suppressed']}, "
              f"unbacked {c['publish_unbacked']}]")
        if c.get("publish_worst"):
            print("  ...contradicted : " + ", ".join(
                f"{k} x{v}" for k, v in c["publish_worst"].items()))
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
