"""Backfill phone playback proxies for the whole library, one at a time.

Every guard render_proxy already has applies (defer when a recording is
live, kill mid-encode if one starts). On top: the standing heavy-job
rule — never start the next render if any session file was modified in
the last 10 minutes.

Usage: python tools/backfill_proxies.py [--limit N]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lowprio import demote  # noqa: E402

demote()

from billiards_trainer.config import Settings  # noqa: E402
from billiards_trainer.vision.proxy_render import has_proxy, render_proxy  # noqa: E402


def table_quiet(rec_dir: Path, quiet_s: float = 600.0) -> bool:
    now = time.time()
    for f in rec_dir.glob("session-*.mp4"):
        if now - f.stat().st_mtime < quiet_s:
            return False
    if list(rec_dir.glob(".session-*.part.mp4")):
        return False
    # Joe at the MACHINE counts too (2026-08-26: a backfill starved his
    # session lists while he was at the desktop)
    from _presence import joe_present
    return not joe_present()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="stop after N renders")
    args = ap.parse_args()
    rec_dir = Settings.load().recording.resolved_dir()
    videos = sorted(rec_dir.glob("session-*.mp4"))
    todo = [v for v in videos if not has_proxy(v)]
    print(f"{len(videos)} sessions, {len(todo)} need proxies", flush=True)
    done = failed = 0
    for v in todo:
        if args.limit and done >= args.limit:
            break
        if not table_quiet(rec_dir):
            print("table active - waiting", flush=True)
            while not table_quiet(rec_dir):
                time.sleep(60)
        t0 = time.time()
        ok = render_proxy(v)
        if ok:
            done += 1
            print(f"[{done}/{len(todo)}] {v.name} in {time.time()-t0:.0f}s",
                  flush=True)
        else:
            failed += 1
            print(f"FAILED/deferred: {v.name}", flush=True)
            time.sleep(30)   # deferred for a live recording? don't spin
    print(f"done: {done} rendered, {failed} failed/deferred", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
