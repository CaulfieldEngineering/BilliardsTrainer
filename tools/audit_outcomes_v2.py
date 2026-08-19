"""Identity-aware outcome audit: which NUMBERED balls left the table?

The count-based audit drowned in ghost churn (deltas of +-2 on hand-heavy
shots). This one asks the robust question: the identity layer guarantees
one-of-each numbered ball (G1) with rest-frozen identities (G4), so a
shot's ground truth is the set difference of NUMBERS present before vs
after — ignoring unnumbered tracks entirely, excluding balls that were
hand-adjacent, and treating a departed number that REAPPEARS hands-free
within a grace window as never having left (detection flicker).

    departed == {}            -> miss
    departed == {0}           -> scratch
    departed ⊇ {objects}      -> make (count = len)
    0 in departed + objects   -> scratch (cue trumps)

    python tools/audit_outcomes_v2.py --video <session.mp4>
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from billiards_trainer.vision.analysis_cache import SidecarReader  # noqa: E402
from billiards_trainer.vision.outcomes import (  # noqa: E402
    anon_departures,
    departed_for_shot,
    derive_outcome,
    numbers_at,
    stable_numbers,
)

__all__ = ["numbers_at", "stable_numbers", "anon_departures",
           "departed_for_shot", "audit"]


def audit(video: Path) -> list[dict]:
    reader = SidecarReader(video)
    rows = []
    for i, s in enumerate(reader.shots, 1):
        derived, detail = derive_outcome(reader, s)
        rows.append({"shot": i, "recorded": s.get("outcome", "?"),
                     "derived": derived,
                     "departed": detail.pop("departed"),
                     "agrees": derived == s.get("outcome", "?"), **detail})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    args = ap.parse_args()
    rows = audit(Path(args.video))
    ok = 0
    for r in rows:
        mark = "OK " if r["agrees"] else "?? "
        ok += r["agrees"]
        extra = f"  (returns: {r['returned']})" if r.get("returned") else ""
        print(f"{mark} shot {r['shot']:2d}: recorded {r['recorded']:8s} "
              f"derived {r['derived']:8s} departed {r['departed']}{extra}")
    print(f"\nrecorded-vs-derived agreement: {ok}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
