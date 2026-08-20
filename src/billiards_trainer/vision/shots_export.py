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


def summary_path(video_path) -> Path:
    return Path(str(video_path) + SUMMARY_SUFFIX)


def export_shots_summary(video_path) -> Path | None:
    """Write <video>.shots.json from the sidecar. Returns the path, or
    None when there is no sidecar. Never raises (enrichment only)."""
    try:
        reader = SidecarReader(video_path)
    except OSError:
        return None
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
        if s.get("note"):
            entry["note"] = s["note"]
        try:
            from .describe import compose_text, describe_shot
            entry["text"] = compose_text(describe_shot(reader, s))
        except Exception:  # noqa: BLE001 - description is enrichment
            pass
        shots.append(entry)
    doc = {
        "v": 1,
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
