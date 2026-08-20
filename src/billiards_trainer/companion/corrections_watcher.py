"""Ingest phone review verdicts from the synced corrections folder.

The cloud app can't reach this machine, but Dropbox can: the phone's
"Correct this clip" button uploads a tiny JSON to
<recordings>/corrections/ via the Vercel proxy, the Dropbox client syncs
it down here, and this watcher applies it to the session's sidecar as a
REVIEW-ranked record (final — derived re-runs stand down), then
re-exports the phone summaries so every surface converges. Processed
files move to corrections/done/ (kept for audit, out of the queue).

Correction file shape (one verdict per file):
    {"session": "session-....mp4", "start": 123.4,
     "outcome": "make"|"miss"|"scratch",          # optional
     "action": "stroke"|"break"|...}              # optional
"""

import json
import logging
import threading
import time
from pathlib import Path

log = logging.getLogger("companion.corrections")

_OUTCOMES = {"make", "miss", "scratch"}
_ACTIONS = {"stroke", "break", "ball_in_hand", "rearrange", "nothing"}


def apply_correction_file(path: Path, recordings: Path) -> bool:
    """Apply one verdict file. True = applied (or hopeless — archive it);
    False = transient failure, retry later."""
    from ..vision.actions import append_action
    from ..vision.analysis_cache import append_correction, sidecar_path
    from ..vision.shots_export import export_library_index, export_shots_summary
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        log.warning("unreadable correction %s — archiving", path.name)
        return True
    name = d.get("session", "")
    video = recordings / name
    if ("/" in name or "\\" in name or not name.endswith(".mp4")
            or not video.is_file() or not sidecar_path(video).is_file()):
        log.warning("correction for unknown session %r — archiving", name)
        return True
    try:
        start = float(d["start"])
    except (KeyError, TypeError, ValueError):
        return True
    if d.get("clear"):
        import json as _json
        from ..vision.actions import classify_and_mark
        from ..vision.analysis_cache import sidecar_path as _sp
        from ..vision.outcomes import derive_and_correct
        with open(_sp(video), "a", encoding="utf-8") as fh:
            fh.write(_json.dumps({"type": "correction_clear",
                                  "start": round(start, 3)}) + "\n")
        # re-derive right away so the cleared shot converges to the
        # machine's best answer instead of sitting on stale originals
        derive_and_correct(video)
        classify_and_mark(video)
        export_shots_summary(video)
        export_library_index(recordings)
        log.info("verdict CLEARED: %s @ %.1fs", name, start)
        return True
    did = False
    if d.get("outcome") in _OUTCOMES:
        did |= append_correction(video, start, d["outcome"], src="review")
    if d.get("action") in _ACTIONS:
        did |= append_action(video, start, d["action"], src="review")
    note = str(d.get("note", "")).strip()[:500]
    if note:
        import json as _json
        from ..vision.analysis_cache import sidecar_path as _sp
        with open(_sp(video), "a", encoding="utf-8") as fh:
            fh.write(_json.dumps({"type": "note", "start": round(start, 3),
                                  "text": note, "src": "review"}) + "\n")
        did = True
    if did:
        export_shots_summary(video)
        export_library_index(recordings)
        log.info("phone verdict applied: %s @ %.1fs %s", name, start,
                 {k: d[k] for k in ("outcome", "action") if d.get(k)})
    return True


def scan_once(recordings: Path) -> int:
    """Process every pending correction file. Returns count applied."""
    box = recordings / "corrections"
    if not box.is_dir():
        return 0
    done = box / "done"
    n = 0
    for f in sorted(box.glob("*.json")):
        try:
            if apply_correction_file(f, recordings):
                done.mkdir(exist_ok=True)
                f.replace(done / f.name)
                n += 1
        except Exception:  # noqa: BLE001 - one bad file must not stop the box
            log.exception("correction %s failed", f.name)
    return n


def start_watcher(recordings: Path, interval_s: float = 10.0) -> threading.Thread:
    """Daemon thread: poll the corrections folder forever."""
    def run() -> None:
        while True:
            try:
                scan_once(recordings)
            except Exception:  # noqa: BLE001
                log.exception("corrections scan failed")
            time.sleep(interval_s)
    t = threading.Thread(target=run, daemon=True, name="corrections-watcher")
    t.start()
    return t
