"""G1 render audit: prove no two balls ever RENDER with the same identity.

The definition of done for GOALS G1. Audits analysis sidecars (what the
app actually shows) for frames where two tracks carry the same number —
cheap (no video decode), so it runs over the whole library in seconds.
Sidecars written before the exclusive-assignment layer measure the OLD
code; re-backfill with --force via build_analysis_cache to audit current
behaviour.

    python tools/audit_render.py            # all sidecars in the library
    python tools/audit_render.py --video <session.mp4>
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from billiards_trainer.config import Settings  # noqa: E402
from billiards_trainer.vision.analysis_cache import sidecar_path  # noqa: E402


def audit(video: Path) -> dict | None:
    p = sidecar_path(video)
    if not p.is_file():
        return None
    frames = dupes = 0
    dupe_numbers: Counter = Counter()
    with open(p, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("type") != "f":
                continue
            frames += 1
            nums = [r[4] for r in d["tracks"] if r[4] > 0]
            for n, c in Counter(nums).items():
                if c > 1:
                    dupes += 1
                    dupe_numbers[n] += 1
                    break
    return {"video": video.name, "frames": frames, "dupe_frames": dupes,
            "rate": (100.0 * dupes / frames) if frames else 0.0,
            "numbers": dict(dupe_numbers)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", default="")
    args = ap.parse_args()
    if args.video:
        vids = [Path(args.video)]
    else:
        d = Path(Settings.load().recording.resolved_dir())
        vids = sorted(d.glob("session-*.mp4"))
    total_f = total_d = 0
    for v in vids:
        r = audit(v)
        if r is None:
            continue
        total_f += r["frames"]
        total_d += r["dupe_frames"]
        flag = "  CLEAN" if r["dupe_frames"] == 0 else f"  DUPES {r['numbers']}"
        print(f"{r['video']:46s} {r['frames']:6d} states  {r['rate']:5.2f}%{flag}")
    if total_f:
        print(f"\nTOTAL: {total_d}/{total_f} states with duplicate identities "
              f"({100.0 * total_d / total_f:.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
