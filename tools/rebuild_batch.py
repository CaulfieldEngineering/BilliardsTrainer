"""Overnight library rebuild: bring every stale sidecar onto the current
pipeline, one session at a time, without ever getting in Joe's way.

Joe's presence outranks loop verification (AUTONOMY incident 2026-08-18):
the batch checks for recording activity before EVERY session and stops
dead if he starts playing — a later run resumes where it left off (the
staleness test is the resume marker; rebuilt sidecars are no longer
stale). Review verdicts carry forward automatically (build --force calls
carry_review_verdicts).

    python tools/rebuild_batch.py --before 2026-08-20T18:00 [--dry-run]

Writes progress to _eval/rebuild_batch.log and holds
_eval/rebuild_batch.lock (pid) while running so autonomy loops can see a
heavy job is active and not start another.
"""

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from billiards_trainer.config import Settings  # noqa: E402

LOCK = ROOT / "_eval" / "rebuild_batch.lock"
LOG = ROOT / "_eval" / "rebuild_batch.log"


def _log(msg: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    line = f"[{stamp}Z] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def recording_active(lib: Path) -> bool:
    now = time.time()
    return any(now - p.stat().st_mtime < 600
               for p in lib.glob("session-*.mp4"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--before", required=True,
                    help="rebuild sidecars whose mtime is before this "
                         "LOCAL time (YYYY-MM-DDTHH:MM)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cutoff = datetime.strptime(args.before, "%Y-%m-%dT%H:%M").timestamp()
    lib = Path(Settings.load().recording.resolved_dir())
    todo = []
    for v in sorted(lib.glob("session-*.mp4")):
        sc = Path(str(v) + ".analysis.jsonl")
        if not sc.is_file() or sc.stat().st_mtime < cutoff:
            todo.append(v)
    _log(f"batch start: {len(todo)} session(s) stale (before {args.before})")
    if args.dry_run:
        for v in todo:
            print(" ", v.name)
        return 0
    if LOCK.is_file():
        _log(f"another batch holds {LOCK} — aborting")
        return 2
    LOCK.write_text(str(os.getpid()))
    done = 0
    try:
        from build_analysis_cache import build
        for v in todo:
            if recording_active(lib):
                _log(f"recording activity — stopping ({done}/{len(todo)} "
                     "done; next run resumes)")
                return 0
            _log(f"rebuilding {v.name}...")
            t0 = time.time()
            try:
                ok = build(v, force=True)
            except Exception as e:  # noqa: BLE001 - one bad file, keep going
                _log(f"  FAILED {v.name}: {e}")
                continue
            done += 1
            _log(f"  {'done' if ok else 'FAILED'} in "
                 f"{(time.time() - t0)/60:.1f} min ({done}/{len(todo)})")
        _log(f"batch complete: {done}/{len(todo)}")
        return 0
    finally:
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
