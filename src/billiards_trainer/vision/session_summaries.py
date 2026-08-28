"""Session row summaries: duration + shot count for the sidebar.

The sidebar said "Aug 17 21:59 · 13 MB" — megabytes answer nothing Joe
asks ("which one was the long drill night?", "did that session have any
shots?"). Duration comes from the video header (milliseconds to read);
the shot count from a line-prefix scan of the analysis sidecar. Both are
cached by (size, mtime) in a JSON next to the app settings, so the cost
is paid once per new recording — the sidebar thread reads the cache and
fills rows in place.

Pure logic, no Qt: testable, and the sidebar owns its own threading.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config import APP_DIR

_CACHE_FILE = Path(APP_DIR) / "session_summaries.json"


def _video_duration_s(path: Path) -> float:
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        return frames / fps if fps > 0 else 0.0
    finally:
        cap.release()


def _sidecar_shot_count(path: Path) -> int | None:
    """None = no sidecar (unanalyzed), else the number of shot records."""
    sc = Path(str(path) + ".analysis.jsonl")
    if not sc.is_file():
        return None
    starts: list[float] = []
    actions: dict[float, str] = {}
    try:
        with open(sc, encoding="utf-8") as f:
            for line in f:
                is_shot = line.startswith('{"type":"shot"')                     or line.startswith('{"type": "shot"')
                is_action = line.startswith('{"type":"action"')                     or line.startswith('{"type": "action"')
                if not (is_shot or is_action):
                    continue
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if is_shot:
                    starts.append(float(d.get("start", 0.0)))
                else:
                    actions[round(float(d.get("start", -1)), 1)] =                         d.get("action", "stroke")
    except OSError:
        return None
    # Count real ATTEMPTS (strokes + breaks): ball-in-hand relocations and
    # empty windows were a third of "shots" in hand-tracked sessions.
    return sum(1 for t in starts
               if actions.get(round(t, 1), "stroke") in ("stroke", "break"))


def load_cache() -> dict:
    try:
        return json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_cache(cache: dict) -> None:
    try:
        _CACHE_FILE.write_text(json.dumps(cache), encoding="utf-8")
    except OSError:
        pass


def prune_cache(cache: dict, existing_names) -> None:
    """Drop entries for videos that no longer exist (Joe deleted 29
    sessions; their summaries lingered — 2011 keys for 17 videos). Keys
    are 'name:size:mtime'; callers pass the names they just listed."""
    keep = set(existing_names)
    for k in [k for k in cache if k.split(":", 1)[0] not in keep]:
        del cache[k]


def summarize(path: Path, cache: dict) -> dict:
    """{"dur_s": float, "shots": int|None} — cached by name+size+mtime."""
    st = path.stat()
    key = f"{path.name}:{st.st_size}:{int(st.st_mtime)}"
    hit = cache.get(key)
    if hit is not None:
        return hit
    out = {"dur_s": round(_video_duration_s(path), 1),
           "shots": _sidecar_shot_count(path)}
    cache[key] = out
    return out


def row_text(when: str, mb: float, summary: dict | None) -> str:
    """The sidebar row label. Duration and shots lead; MB lives in the
    tooltip (see the sidebar) because bytes answer no review question."""
    if not summary or summary.get("dur_s", 0) <= 0:
        return f"{when}   ·  {mb:.0f} MB"
    dur = summary["dur_s"]
    d = f"{int(dur // 3600)}h{int(dur % 3600 // 60):02d}" if dur >= 3600 \
        else f"{int(dur // 60)}m" if dur >= 60 else f"{int(dur)}s"
    shots = summary.get("shots")
    tail = f"  ·  {shots} shots" if shots else ""
    return f"{when}   ·  {d}{tail}"


def is_stub(summary: dict | None, mb: float) -> bool:
    """A recording too short to review — a false start. Dimmed, not hidden
    (it is still Joe's data)."""
    if summary and summary.get("dur_s", 0) > 0:
        return summary["dur_s"] < 30.0
    return mb < 5.0
