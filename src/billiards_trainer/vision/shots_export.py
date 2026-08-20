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
    paths: dict = {}
    t = max(0.0, t0 - 1.0)
    while t <= t1 + 1e-9:
        for tr in reader.tracks_at(t):
            if not tr.active:
                continue
            e = paths.setdefault(tr.id, {"n": tr.number, "pts": []})
            if tr.number >= 0:
                e["n"] = tr.number
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
    return out


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
    if with_trails:
        try:
            old_doc = json.loads(summary_path(video_path).read_text(
                encoding="utf-8"))
            tf = old_doc.get("transform")
        except (OSError, ValueError):
            tf = None
        if tf is None:
            tf = _video_transform(video_path)
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
        try:
            from .describe import compose_text, describe_shot
            entry["text"] = compose_text(describe_shot(reader, s))
        except Exception:  # noqa: BLE001 - description is enrichment
            pass
        if tf is not None:
            try:
                entry["trails"] = _shot_trails(reader, s, tf)
            except Exception:  # noqa: BLE001 - trails are enrichment
                pass
        shots.append(entry)
    doc = {
        "v": 2 if tf is not None else 1,
        "transform": tf,
        "session": Path(video_path).name,
        "duration_s": round(reader._times[-1], 1) if reader._times else 0.0,
        "exported": datetime.now(timezone.utc).isoformat(timespec="seconds"),
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
