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
    from ..vision.shots_export import (export_library_index,
                                       export_lifetime_stats,
                                       export_shots_summary)
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
    if isinstance(d.get("split"), (int, float)):
        import json as _json

        from ..vision.actions import classify_and_mark
        from ..vision.analysis_cache import sidecar_path as _sp
        from ..vision.outcomes import derive_and_correct
        with open(_sp(video), "a", encoding="utf-8") as fh:
            fh.write(_json.dumps({"type": "split",
                                  "start": round(start, 3),
                                  "at": round(float(d["split"]), 3)}) + "\n")
        derive_and_correct(video)
        classify_and_mark(video)
        export_shots_summary(video)
        export_library_index(recordings)
        export_lifetime_stats(recordings)
        log.info("shot SPLIT: %s @ %.1fs at t=%.1fs", name, start,
                 float(d["split"]))
        return True
    if d.get("rife"):
        # Smooth slow-mo request (Joe: "Rife looks great... it pushes to
        # a slow mo playlist in a separate folder"): render 4x-interpolated
        # clip into <recordings>/slowmo/ and append it to the Slow-mo
        # playlist. Never while Joe is recording — the render owns the GPU
        # for ~35s. Returning False keeps the request queued for retry.
        import time as _time
        now = _time.time()
        if any(now - p.stat().st_mtime < 600
               for p in recordings.glob("session-*.mp4")):
            log.info("rife request deferred: recording activity")
            return False
        from .rife_render import add_to_slowmo_playlist, render_slowmo
        end = float(d.get("end", start + 8.0))
        out = render_slowmo(video, start, end)
        if out is not None:
            label = f"{name.replace('.mp4', '')} @{int(start)}s"
            add_to_slowmo_playlist(recordings, out.name, label)
            log.info("rife: %s ready and playlisted", out.name)
        return True        # rendered or hopeless — archive the request
    if d.get("confirm"):
        import json as _json

        from ..vision.analysis_cache import sidecar_path as _sp
        with open(_sp(video), "a", encoding="utf-8") as fh:
            fh.write(_json.dumps({"type": "reviewed",
                                  "start": round(start, 3)}) + chr(10))
        export_shots_summary(video)
        export_library_index(recordings)
        export_lifetime_stats(recordings)
        log.info("shot CONFIRMED reviewed: %s @ %.1fs", name, start)
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
        export_lifetime_stats(recordings)
        log.info("verdict CLEARED: %s @ %.1fs", name, start)
        return True
    did = False
    if d.get("outcome") in _OUTCOMES:
        did |= append_correction(video, start, d["outcome"], src="review")
    if d.get("action") in _ACTIONS:
        did |= append_action(video, start, d["action"], src="review")
    # Joe's cut / miss-side verdicts (the ground truth the miss stats and
    # the trajectory-fit validation feed on). Review-ranked like outcomes.
    tc = {k: d[k] for k in ("cut", "miss_side")
          if d.get(k) in ("left", "right", "straight")}
    if tc:
        import json as _json
        from ..vision.analysis_cache import sidecar_path as _sc
        with open(_sc(video), "a", encoding="utf-8") as fh:
            fh.write(_json.dumps({"type": "tag_correction",
                                  "start": round(start, 3),
                                  **tc, "src": "review"}) + "\n")
        did = True
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
        export_lifetime_stats(recordings)
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
