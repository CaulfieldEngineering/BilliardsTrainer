"""Backfill camera-measured stroke metrics over the session library.

New recordings get a stroke_vision pass at session close; this annotates
old sessions once. Idempotent (per-shot version records — a rerun skips
everything current). Stops between sessions if a recording starts.

    python tools/backfill_stroke_vision.py [--limit N]
"""
import argparse
import glob
import os
import sys
import time
from pathlib import Path

from _lowprio import demote

demote()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SESS_DIR = "C:/Users/Joe/Dropbox/Billiards/BilliardsTrainer"


def recording_active() -> bool:
    from billiards_trainer.config import EXPORTS_DIR
    if list(EXPORTS_DIR.glob(".session-*.part.mp4")):
        return True
    newest = max((os.path.getmtime(f) for f in
                  glob.glob(os.path.join(SESS_DIR, "session-*.mp4"))),
                 default=0)
    return (time.time() - newest) < 600


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    from billiards_trainer.vision.stroke_vision import annotate_session
    files = sorted(glob.glob(os.path.join(SESS_DIR, "session-*.mp4")),
                   key=os.path.getmtime, reverse=True)
    total = sessions = 0
    for p in files:
        if recording_active():
            print("recording active - stopping (rerun later)")
            break
        if not os.path.isfile(p + ".analysis.jsonl"):
            continue
        t0 = time.time()
        n = annotate_session(p)
        if n:
            sessions += 1
            total += n
            print(f"{os.path.basename(p)}: {n} shot(s) in "
                  f"{time.time() - t0:.0f}s")
            from billiards_trainer.vision.shots_export import export_shots_summary
            export_shots_summary(p)
        if args.limit and sessions >= args.limit:
            break
    if total:
        from billiards_trainer.vision.shots_export import (
            export_library_index,
            export_lifetime_stats,
        )
        export_lifetime_stats(SESS_DIR)
        export_library_index(SESS_DIR)
    print(f"backfill: {total} shots across {sessions} sessions")


if __name__ == "__main__":
    main()
