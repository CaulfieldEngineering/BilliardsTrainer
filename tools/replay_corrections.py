"""Replay archived phone verdicts whose effects are missing.

Why this exists: an early watcher build archived note-only correction
files WITHOUT writing note records (the outcome half of the pipeline
existed; notes didn't yet), so Joe's review comments silently vanished
from the phone — the files in corrections/done/ are the only copy.
This tool re-applies any archived verdict whose record is absent from
the session sidecar, using the CURRENT apply logic, then re-exports.
Idempotent: verdicts already present are skipped.

    python tools/replay_corrections.py [--dry-run]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from billiards_trainer.config import Settings  # noqa: E402
from billiards_trainer.vision.analysis_cache import sidecar_path  # noqa: E402


def sidecar_has(video: Path, kind: str, start: float, text: str = "") -> bool:
    p = sidecar_path(video)
    if not p.is_file():
        return True         # nothing to replay into — skip
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("type") != kind:
            continue
        if abs(float(d.get("start", -1)) - start) > 0.3:
            continue
        if kind == "note" and text and d.get("text") != text:
            continue
        return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    recordings = Path(Settings.load().recording.resolved_dir())
    done = recordings / "corrections" / "done"
    if not done.is_dir():
        print("no archive to replay")
        return 0
    from billiards_trainer.companion.corrections_watcher import (
        apply_correction_file,
    )
    touched: set = set()
    replayed = 0
    for f in sorted(done.glob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        video = recordings / d.get("session", "")
        if not video.is_file():
            continue
        try:
            start = float(d["start"])
        except (KeyError, TypeError, ValueError):
            continue
        missing = []
        note = str(d.get("note", "")).strip()[:500]
        if note and not sidecar_has(video, "note", start, note):
            missing.append(f"note {note[:40]!r}")
        if d.get("confirm") and not sidecar_has(video, "reviewed", start):
            missing.append("confirm")
        # outcome/action corrections were never affected by the bug, and
        # replaying them is harmless-but-noisy (append-only log): only
        # replay files that carry something missing.
        if not missing:
            continue
        print(f"{f.name}: {d.get('session')} @{start:.1f}s missing "
              + ", ".join(missing))
        replayed += 1
        touched.add(d.get("session"))
        if not args.dry_run:
            apply_correction_file(f, recordings)
    print(f"{'would replay' if args.dry_run else 'replayed'}: {replayed} "
          f"file(s) across {len(touched)} session(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
