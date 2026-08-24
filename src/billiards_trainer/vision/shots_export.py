"""Compact per-session shot summary, exported NEXT TO the video.

The cloud phone app reads this file straight out of Dropbox (the
recordings folder syncs both). It carries everything review needs —
outcomes, action labels, descriptions — in a few KB, so the phone never
parses the multi-megabyte analysis sidecar over cellular.

Written by the same close pass that derives outcomes and labels actions,
so it always reflects the final state of the sidecar log (including any
review verdicts appended after — re-export refreshes it).
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from .analysis_cache import SidecarReader

log = logging.getLogger("vision.shots_export")

SUMMARY_SUFFIX = ".shots.json"

#: participants must travel at least this many ball radii to earn a trail
_TRAIL_TRAVEL_R = 1.5


def _video_transform(video_path) -> dict | None:
    """{"hinv": 3x3, "w": px, "h": px} — maps sidecar rect coords back to
    THIS video's pixels by re-acquiring calibration from the video itself
    (uniform for live- and backfill-recorded sessions; the live calib can
    disagree with the recorded geometry). Costs a pipeline warmup (~3s),
    so callers cache it in the summary file and reuse."""
    try:
        import cv2
        import numpy as np

        from ..config import Settings
        from .pipeline import Pipeline
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        pipe = Pipeline(Settings.load())
        shape = None
        n = 0
        while n < 90:
            ok, fr = cap.read()
            if not ok:
                break
            shape = fr.shape
            n += 1
            pipe.process(fr, n / 30.0, annotate=False, detect=False)
            if pipe.calib.calib is not None and n >= 20:
                break
        cap.release()
        calib = pipe.calib.calib
        if calib is None or shape is None:
            return None
        hinv = np.linalg.inv(np.asarray(calib.H, dtype=float))
        return {"hinv": [[round(float(v), 8) for v in row] for row in hinv],
                "w": int(shape[1]), "h": int(shape[0])}
    except Exception:  # noqa: BLE001 - trails are enrichment
        log.exception("video transform failed for %s", video_path)
        return None


def _shot_trails(reader: SidecarReader, s: dict, tf: dict) -> list:
    """Per-participant polylines in NORMALIZED video coords:
    [{"n": ball, "p": [[t, x, y], ...]}, ...] — a few KB per shot."""
    import numpy as np
    hinv = np.asarray(tf["hinv"], dtype=float)
    w, h = float(tf["w"]), float(tf["h"])
    t0, t1 = float(s["start"]), float(s["end"])
    # Attribute every sample to the number the track held AT THAT MOMENT.
    # Stamping a track's FINAL number over its whole path was drawing
    # 005048 @233's object ball as the cue (Joe: "the line the object ball
    # took is drastically wrong"): when Joe struck the 4, the arriving cue
    # ball stole that track at the contact point, so the track finished the
    # shot riding the cue and the exporter relabelled its entire history --
    # including the stretch where it really was sitting on the 4.
    paths: dict = {}
    t = max(0.0, t0 - 1.0)
    while t <= t1 + 1e-9:
        for tr in reader.tracks_at(t):
            if not tr.active or tr.number < 0:
                continue
            e = paths.setdefault((tr.id, tr.number),
                                 {"n": tr.number, "pts": []})
            e["pts"].append((t, tr.x, tr.y, tr.radius))
        t += 0.15
    out = []
    for e in paths.values():
        pts = e["pts"]
        if len(pts) < 3:
            continue
        travel = sum(((pts[i + 1][1] - pts[i][1]) ** 2
                      + (pts[i + 1][2] - pts[i][2]) ** 2) ** 0.5
                     for i in range(len(pts) - 1))
        if travel < _TRAIL_TRAVEL_R * max(6.0, pts[0][3]):
            continue
        poly = []
        for (t_, x, y, _r) in pts[:120]:
            v = hinv @ np.array([x, y, 1.0])
            px, py = v[0] / v[2] / w, v[1] / v[2] / h
            if -0.2 <= px <= 1.2 and -0.2 <= py <= 1.2:
                poly.append([round(t_, 2),
                             round(min(1.0, max(0.0, px)), 4),
                             round(min(1.0, max(0.0, py)), 4)])
        if len(poly) >= 3:
            out.append({"n": e["n"], "p": poly})
    # ONE entry per BALL, not per track segment (Joe: two cue badges and
    # two 7s on one shot — "always wrong to some extent"): identity churn
    # splits a ball's path across track ids mid-flight; merge same-number
    # segments in time order.
    merged: dict = {}
    for e in out:
        m = merged.setdefault(e["n"], {"n": e["n"], "p": []})
        m["p"].extend(e["p"])
    result = []
    for m in merged.values():
        m["p"].sort(key=lambda q: q[0])
        result.append(m)
    return result


def _shot_aim(cap, reader: SidecarReader, s: dict, tf: dict, H) -> dict | None:
    """Where the cue STICK pointed at address, as a video-normalized
    segment. Computed ONCE here — desktop and phone draw this same
    stored geometry, so the two can never disagree (Joe's requirement).
    Decodes one address frame per shot; cached in the summary across
    re-exports."""
    import cv2
    import numpy as np

    from .cue_aim import AIM_VERSION, aim_ray_end, detect_cue_aim
    t0 = float(s.get("start", 0.0))
    hinv = np.asarray(tf["hinv"], dtype=float)
    w, h = float(tf["w"]), float(tf["h"])
    for dt in (0.4, 1.0, 1.8):          # scan the address backwards
        tp = max(0.0, t0 - dt)
        cue = next((tr for tr in reader.tracks_at(tp) if tr.number == 0),
                   None)
        if cue is None:
            continue
        cap.set(cv2.CAP_PROP_POS_MSEC, tp * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        rect = cv2.warpPerspective(frame, H, (700, 1300))
        got = detect_cue_aim(rect, (cue.x, cue.y), cue.radius)
        if got is None or got[1] < 0.35:
            continue
        ang, q, (ax, ay) = got
        ex, ey = aim_ray_end(ax, ay, ang, (30.0, 30.0, 670.0, 1270.0))
        # STOP AT THE BALL IT WOULD HIT (Joe: "stop the line at the object
        # ball ... I don't care about what happens to the cue ball once it
        # hits its object ball"). Running the ray on to the rail drew a
        # long line straight THROUGH the object ball, which read as a
        # prediction about the cue ball's onward travel -- something this
        # overlay is not claiming and cannot know.
        hit = _first_ball_on_ray(reader, tp, (ax, ay), (ex, ey), cue.radius)
        if hit is not None:
            ex, ey = hit
        seg = []
        for (x, y) in ((ax, ay), (ex, ey)):
            v = hinv @ np.array([x, y, 1.0])
            seg.append([round(min(1.2, max(-0.2, v[0] / v[2] / w)), 4),
                        round(min(1.2, max(-0.2, v[1] / v[2] / h)), 4)])
        return {"p": seg, "q": round(float(q), 2), "t": round(tp, 2),
                "v": AIM_VERSION}
    return None


def _first_ball_on_ray(reader, t, a, b, r_cue):
    """Where a cue ball rolling a->b would first touch another ball:
    the CONTACT point (one ball-diameter short of that ball's centre),
    or None if the path is clear. Uses the cue ball's own width, so a
    ball the cue would just squeeze past does not stop the line."""
    import math
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    seg = math.hypot(dx, dy)
    if seg < 1e-6:
        return None
    ux, uy = dx / seg, dy / seg
    best = None
    for tr in reader.tracks_at(t):
        if not tr.active or tr.number == 0:
            continue
        along = (tr.x - ax) * ux + (tr.y - ay) * uy
        if along <= 0.0 or along > seg:
            continue
        perp = abs(-(tr.y - ay) * ux + (tr.x - ax) * uy)
        reach = r_cue + tr.radius
        if perp > reach:
            continue                       # the cue would miss this ball
        # roll back to the centres-touching position
        back = math.sqrt(max(0.0, reach * reach - perp * perp))
        d = along - back
        if d > 0 and (best is None or d < best):
            best = d
    if best is None:
        return None
    return (ax + ux * best, ay + uy * best)


def _rect_to_video(pt, tf: dict) -> list:
    """Rect-space point -> normalized video coords (the overlay frame)."""
    import numpy as np
    hinv = np.asarray(tf["hinv"], dtype=float)
    v = hinv @ np.array([pt[0], pt[1], 1.0])
    return [round(v[0] / v[2] / float(tf["w"]), 4),
            round(v[1] / v[2] / float(tf["h"]), 4)]


def summary_path(video_path) -> Path:
    return Path(str(video_path) + SUMMARY_SUFFIX)


def export_shots_summary(video_path, with_trails: bool = True) -> Path | None:
    """Write <video>.shots.json from the sidecar. Returns the path, or
    None when there is no sidecar. Never raises (enrichment only).

    Trails (Joe: "I'd like to see the animated tails"): per-shot ball
    polylines pre-mapped to normalized video pixels. The calibration
    transform is computed ONCE per video and cached inside the summary —
    re-exports (verdict syncs etc.) reuse it instead of re-running a
    pipeline warmup."""
    try:
        reader = SidecarReader(video_path)
    except OSError:
        return None
    tf = None
    old_aims: dict = {}
    old_space = None
    if with_trails:
        try:
            old_doc = json.loads(summary_path(video_path).read_text(
                encoding="utf-8"))
            tf = old_doc.get("transform")
            old_space = old_doc.get("space")
            for o in old_doc.get("shots", []):
                if o.get("aim"):
                    old_aims[round(float(o.get("start", -1)), 2)] = o["aim"]
        except (OSError, ValueError):
            tf = None
        if tf is None:
            tf = _video_transform(video_path)
    aim_cap = None
    aim_H = None
    # TRUE-INCH frame for miss tagging (Joe's left/right cut, missed
    # left/right, over/undercut). One pipeline warmup per session,
    # cached in the summary like the transform.
    space = None
    try:
        from .tablespace import TableSpace, space_for_video
        if old_space:
            space = TableSpace(**old_space)
        else:
            space = space_for_video(video_path, reader)
    except Exception:  # noqa: BLE001 - tagging is enrichment
        space = None
    shots = []
    for s in reader.shots:
        entry = {
            "start": round(float(s.get("start", 0.0)), 2),
            "end": round(float(s.get("end", 0.0)), 2),
            "outcome": s.get("outcome", "miss"),
            "action": s.get("action", "stroke"),
            "pocketed": int(s.get("pocketed", 0)),
        }
        if s.get("corrected") or s.get("action_corrected"):
            entry["corrected"] = True
        if s.get("reviewed_ok"):
            entry["reviewed"] = True
        if s.get("note"):
            entry["note"] = s["note"]
        sv = s.get("_stroke")
        if sv and sv.get("confidence") != "none":
            # camera-measured stroke metrics; field names align with the
            # BLE cue sensor (back_depth, pause, finish) for later fusion
            entry["stroke"] = {k: sv[k] for k in (
                "stay_down_s", "popped_early", "back_depth_px", "pause_ms",
                "delivery_ms", "practice_strokes", "confidence") if k in sv}
        desc = None
        try:
            from .describe import compose_text, describe_shot
            desc = describe_shot(reader, s)
            entry["text"] = compose_text(desc)
        except Exception:  # noqa: BLE001 - description is enrichment
            pass
        if tf is not None:
            try:
                entry["trails"] = _shot_trails(reader, s, tf)
                # The badge row must NEVER contradict the description
                # (Joe: "it even knows potted the 6, but marks the 2"):
                # every ball the FACTS name — potted, first contact —
                # appears as a badge even when its trail was lost or
                # mislabeled. Empty polyline = badge only, no drawn path.
                if desc is not None:
                    named = {p.get("ball") for p in desc.get("potted", [])}
                    if desc.get("first_object") is not None:
                        named.add(desc["first_object"])
                    named.discard(0)
                    named.discard(None)
                    have = {e["n"] for e in entry["trails"]}
                    for n in sorted(named - have):
                        entry["trails"].append({"n": int(n), "p": []})
            except Exception:  # noqa: BLE001 - trails are enrichment
                pass
            # AIM LINE (Joe's analysis tool): where the stick pointed at
            # address. Cached across re-exports (verdict syncs must not
            # re-decode the whole session); strokes and breaks only.
            try:
                if entry["action"] in ("stroke", "break"):
                    from .cue_aim import AIM_VERSION
                    cached = old_aims.get(entry["start"])
                    if cached and cached.get("v") == AIM_VERSION:
                        entry["aim"] = cached
                    else:
                        if aim_cap is None:
                            import cv2
                            import numpy as np
                            aim_cap = cv2.VideoCapture(str(video_path))
                            aim_H = np.linalg.inv(
                                np.asarray(tf["hinv"], dtype=float))
                        aim = _shot_aim(aim_cap, reader, s, tf, aim_H)
                        if aim:
                            entry["aim"] = aim
            except Exception:  # noqa: BLE001 - aim is enrichment
                pass
        # SHOT LINES for every stroke (Joe: "the revealed overlay should
        # just be something related to the shot line"). The geometry is
        # the same for a make or a miss — where the ball had to go versus
        # where it went — so it is exported for ALL strokes; only the
        # MISS keeps the over/undercut labels.
        if space is not None and entry["action"] in ("stroke", "break"):
            try:
                from .miss_tags import tag_shot
                tg = tag_shot(reader, s, space)
                fo = s.get("_tag_forensic") or {}
                if fo and (tg is None
                           or tg.get("confidence") not in ("high", "review")):
                    # corridor re-pass verdict: fills where the derivation
                    # abstained or was gated; never displaces high/review
                    base = tg or {"pocket_inferred": True}
                    for k in ("cut", "miss_side"):
                        if fo.get(k):
                            base[k] = fo[k]
                    base["confidence"] = "forensic"
                    tg = base
                if tg is None and s.get("_tag_review"):
                    # the machine abstained but Joe called it: his verdict
                    # IS the tag
                    tg = {"confidence": "review", "pocket_inferred": False,
                          **{k: v for k, v in s["_tag_review"].items()}}
                if tg:
                    # map the explanation geometry into normalized video
                    # coords so phone and desktop draw the same figure
                    g = tg.get("geom")
                    if g and tf is not None:
                        tg["geom"] = {k: _rect_to_video(v, tf)
                                      for k, v in g.items()}
                        entry["lines"] = tg["geom"]
                        # the MEASURED outbound path of the object ball,
                        # in the same normalized frame as the rest of the
                        # figure, so both surfaces draw the measurement
                        if tg.get("path"):
                            entry["lines"]["path"] = [
                                _rect_to_video(q, tf) for q in tg["path"]]
                            tg["path"] = entry["lines"]["path"]
                    rv = s.get("_tag_review") or {}
                    if rv:
                        # Joe looked. His word outranks the derivation and
                        # clears any machine-side confidence gate on the
                        # fields he called.
                        for k in ("cut", "miss_side"):
                            if rv.get(k):
                                tg[k] = rv[k]
                        tg["confidence"] = "review"
                    if entry["outcome"] == "miss":
                        entry["tags"] = tg
            except Exception:  # noqa: BLE001 - tagging is enrichment
                pass
        shots.append(entry)
    if aim_cap is not None:
        aim_cap.release()
    doc = {
        "v": 2 if tf is not None else 1,
        "transform": tf,
        "session": Path(video_path).name,
        "duration_s": round(reader._times[-1], 1) if reader._times else 0.0,
        "exported": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "space": ({k: getattr(space, k) for k in ("x0","y0","x1","y1","px_per_in","ball_r_px","size","n_samples")} if space else None),
        "shots": shots,
    }
    out = summary_path(video_path)
    try:
        out.write_text(json.dumps(doc, separators=(",", ":")),
                       encoding="utf-8")
    except OSError:
        log.exception("shots summary write failed for %s", video_path)
        return None
    return out


def export_lifetime_stats(recordings_dir) -> Path | None:
    """Joe's Lifetime Stats: the cut x miss-side tally he reads patterns
    from. TRUSTED tags only (confidence high, or his own review) — one
    wrong side pollutes a pattern more than ten honest abstentions."""
    import json as _json
    d = Path(recordings_dir)
    cells = {}
    clips = {}
    trusted = gated = 0
    makes = misses = scratches = 0
    # stay-down by outcome (Joe: misses come with popping up early) —
    # high-confidence camera measurements only
    stay: dict[str, list] = {"make": [], "miss": []}
    pops: dict[str, int] = {"make": 0, "miss": 0}
    for sp in sorted(d.glob("session-*.mp4.shots.json")):
        try:
            doc = _json.loads(sp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for sh in doc.get("shots", []):
            if sh.get("action") in ("stroke", "break"):
                oc = sh.get("outcome")
                if oc == "make":
                    makes += 1
                elif oc == "miss":
                    misses += 1
                elif oc == "scratch":
                    scratches += 1
                sv = sh.get("stroke") or {}
                if (oc in stay and sv.get("confidence") == "high"
                        and sv.get("stay_down_s") is not None):
                    stay[oc].append(float(sv["stay_down_s"]))
                    if sv.get("popped_early"):
                        pops[oc] += 1
            t = sh.get("tags")
            if not t or not t.get("miss_side"):
                if sh.get("outcome") == "miss" and sh.get("action") in (
                        "stroke", "break"):
                    clips.setdefault("uncat", []).append(
                        {"session": sp.name[:-len(".shots.json")],
                         "start": sh.get("start")})
                continue
            if t.get("confidence") in ("high", "review", "forensic"):
                trusted += 1
                key = f"{t.get('cut', '?')}|{t['miss_side']}"
                cells[key] = cells.get(key, 0) + 1
                clips.setdefault(key, []).append(
                    {"session": sp.name[:-len(".shots.json")],
                     "start": sh.get("start")})
            else:
                gated += 1
                clips.setdefault("uncat", []).append(
                    {"session": sp.name[:-len(".shots.json")],
                     "start": sh.get("start")})
    out = {"updated": __import__("time").strftime("%Y-%m-%d %H:%M"),
           "trusted": trusted, "gated": gated, "cells": cells,
           "clips": clips,
           "makes": makes, "misses": misses, "scratches": scratches,
           "total": makes + misses + scratches}
    if stay["make"] or stay["miss"]:
        out["stay_down"] = {
            oc: {"n": len(v),
                 "avg_s": round(sum(v) / len(v), 2) if v else None,
                 "popped_early": pops[oc]}
            for oc, v in stay.items()}
    fp = d / "lifetime_stats.json"
    fp.write_text(_json.dumps(out, indent=1), encoding="utf-8")
    return fp


def export_library_index(recordings_dir) -> Path | None:
    """One small library.json for the whole recordings folder: every
    session's duration and ATTEMPT count (strokes + breaks — same rule as
    the desktop list). The phone's landing page reads this in a single
    fetch instead of one summary per session."""
    root = Path(recordings_dir)
    entries = []
    for sj in sorted(root.glob("*.shots.json")):
        try:
            doc = json.loads(sj.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        shots = doc.get("shots", [])
        attempts = sum(1 for s in shots
                       if s.get("action", "stroke") in ("stroke", "break"))
        entries.append({"name": doc.get("session", sj.name[:-11]),
                        "dur_s": doc.get("duration_s", 0.0),
                        "shots": attempts})
    out = root / "library.json"
    try:
        out.write_text(json.dumps({"v": 1, "sessions": entries},
                                  separators=(",", ":")), encoding="utf-8")
    except OSError:
        log.exception("library index write failed")
        return None
    return out
