"""Would padded per-shot clips COVER the library? Measure before building.

Joe (2026-08-26, on splicing sessions into per-shot clips): "Problem is
we don't perfectly identify shots yet." The design answer is coverage,
not perfect identification: pad each detected shot's window generously
and then MEASURE what escapes — an audio onset with real ball motion
that no padded window contains is footage a clip library would lose
(the full session stays as fallback regardless).

Reuses the VALIDATED recall-audit classifier (quiet-before, hand-context
veto, launch profile, gathering convergence) rather than a naive
sound+motion test — the first cut of this tool counted racking and
gathering as "events" and read 62% coverage; the real question is
whether MISSED-STROKE CANDIDATES escape the padded windows:

  windows  = [shot.start - PRE_S, shot.end + POST_S] per detected shot
  keepers  = detected shots (covered by construction)
             + the recall audit's missed_shot_candidates
  escaped  = candidates inside no window -> would-be "activity" clips

    python tools/audit_clip_coverage.py [--limit N]
"""

import argparse
import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from _lowprio import demote

demote()

from audit_shot_recall import audit_session  # noqa: E402

from billiards_trainer.vision.analysis_cache import SidecarReader  # noqa: E402

SESS_DIR = "C:/Users/Joe/Dropbox/Billiards/BilliardsTrainer"
PRE_S = 10.0      # clip pre-roll: routine + practice strokes live here
POST_S = 5.0      # clip post-roll: settle + stay-down
GROUP_S = 10.0    # escaped events this close merge into one activity segment


def audit(video: Path) -> dict | None:
    base = audit_session(video, quiet=True)
    if base is None:
        return None
    reader = SidecarReader(video)
    windows = [(float(s["start"]) - PRE_S,
                float(s.get("end", float(s["start"]) + 8.0)) + POST_S)
               for s in reader.shots]
    cands = list(base.get("missed_shot_candidates") or [])
    escaped = [c for c in cands
               if not any(w0 <= c <= w1 for (w0, w1) in windows)]
    segs = []
    for c in escaped:                     # merge into would-be activity clips
        if segs and c - segs[-1][1] <= GROUP_S:
            segs[-1][1] = c
        else:
            segs.append([c, c])
    return {"video": video.name, "shots": base["shots"],
            "candidates": len(cands), "escaped": len(escaped),
            "segments": len(segs),
            "escaped_at": [f"{a}-{b}" for a, b in segs]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(SESS_DIR, "session-*.mp4")),
                   key=os.path.getmtime, reverse=True)
    tot_shots = tot_c = tot_esc = tot_seg = n = 0
    for p in files:
        if args.limit and n >= args.limit:
            break
        r = audit(Path(p))
        if r is None:
            continue
        n += 1
        tot_shots += r["shots"]
        tot_c += r["candidates"]
        tot_esc += r["escaped"]
        tot_seg += r["segments"]
        flag = "  <-- " + ", ".join(r["escaped_at"][:4]) if r["escaped"] else ""
        print(f"{r['video']}: {r['shots']} shots, {r['candidates']} missed-"
              f"stroke candidates, {r['escaped']} escape ({r['segments']} "
              f"segs){flag}", flush=True)
    total_keep = tot_shots + tot_c
    if total_keep:
        pct = 100.0 * (total_keep - tot_esc) / total_keep
        print(f"\nLIBRARY: {tot_shots} detected shots + {tot_c} missed-stroke "
              f"candidates over {n} sessions; {tot_esc} escape padded windows "
              f"-> keeper coverage {pct:.2f}%; {tot_seg} activity clip(s) "
              f"would catch the rest", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
