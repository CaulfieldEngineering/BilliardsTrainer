"""Outcome audit: do recorded MAKE/MISS verdicts match what the table says?

Joe's progression: "ensure we're detecting shots correctly, THEN move on
to whether they were makes/misses." Shots are verified; this is the next
step. Ground truth from the sidecar itself: a MAKE must REMOVE at least
one ball from the table across the shot window; a MISS must remove none;
a SCRATCH removes the cue ball. Ball counts come from confirmed active
tracks just before the strike vs just after the settle.

    python tools/audit_outcomes.py --video <session.mp4>
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from billiards_trainer.vision.analysis_cache import SidecarReader  # noqa: E402


def balls_at(reader: SidecarReader, t: float) -> tuple[int, bool]:
    """(count of active balls, cue present) at time t."""
    tracks = [tr for tr in reader.tracks_at(t) if tr.active]
    cue = any(tr.number == 0 for tr in tracks)
    return len(tracks), cue


def audit(video: Path) -> list[dict]:
    reader = SidecarReader(video)
    out = []
    for i, s in enumerate(reader.shots, 1):
        t0, t1 = float(s["start"]), float(s["end"])
        before, cue_before = balls_at(reader, max(0.0, t0 - 0.5))
        # Sample AFTER as close to the settle as possible, and only while no
        # hand is on the table: Joe's ball-in-hand routine (pick the cue up
        # right after the stroke) otherwise reads as a pocketed ball. The v2
        # sidecar knows when hands touch balls — walk forward from the settle
        # and take the last hands-free moment within 1.5s.
        t_after = t1 + 0.3
        for cand in (t1 + 0.3, t1 + 0.6, t1 + 1.0, t1 + 1.5):
            carried, _peak = reader.hand_context(t1, cand)
            if carried:
                break
            t_after = cand
        after, cue_after = balls_at(reader, t_after)
        delta = before - after
        rec = s.get("outcome", "?")
        if rec == "make":
            ok = delta >= 1
        elif rec == "scratch":
            ok = cue_before and not cue_after
        else:                       # miss
            ok = delta <= 0
        out.append({"shot": i, "outcome": rec,
                    "pocketed_rec": s.get("pocketed", 0),
                    "balls_before": before, "balls_after": after,
                    "delta": delta, "agrees": ok})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    args = ap.parse_args()
    rows = audit(Path(args.video))
    bad = 0
    for r in rows:
        mark = "OK " if r["agrees"] else "?? "
        bad += not r["agrees"]
        print(f"{mark} shot {r['shot']:2d}: {r['outcome']:8s} "
              f"(rec pocketed {r['pocketed_rec']})  balls {r['balls_before']} -> "
              f"{r['balls_after']}  (delta {r['delta']:+d})")
    print(f"\n{len(rows) - bad}/{len(rows)} outcomes agree with table state")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
