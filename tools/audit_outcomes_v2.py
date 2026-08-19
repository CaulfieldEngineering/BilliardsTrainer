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


def numbers_at(reader: SidecarReader, t: float) -> set[int]:
    """Numbered balls visible at t (active tracks only; ghosts have no
    number and therefore no vote)."""
    return {tr.number for tr in reader.tracks_at(t) if tr.active and tr.number >= 0}


def stable_numbers(reader: SidecarReader, t0: float, t1: float, step: float = 0.3) -> set[int]:
    """Numbers seen in a MAJORITY of samples across [t0, t1] — one flickered
    frame neither adds nor removes a ball."""
    from collections import Counter
    c: Counter = Counter()
    n = 0
    t = t0
    while t <= t1 + 1e-9:
        for num in numbers_at(reader, t):
            c[num] += 1
        n += 1
        t += step
    return {num for num, k in c.items() if k >= max(2, n // 2)}


def anon_departures(reader: SidecarReader, t0: float, t1: float) -> tuple[int, list]:
    """Departures the NUMBER layer cannot see: a ball whose digit never
    faced the camera carries num=-1 its whole life (this session's 6-ball),
    so number set-difference is blind to its pot. But its TRACK ID is a
    stable resident of the before-window; if that id dies during the shot,
    was never hand-adjacent (not picked up), and no freshly-settled
    unnumbered track appears after (which would mean mid-flight re-ID, a
    moved ball not a potted one), an anonymous object ball left the bed."""
    from collections import Counter
    seen: Counter = Counter()
    n = 0
    t = max(0.0, t0 - 5.0)
    while t <= t0 - 0.2 + 1e-9:
        for tr in reader.tracks_at(t):
            if tr.active and tr.number < 0:
                seen[tr.id] += 1
        n += 1
        t += 0.5
    residents = {i for i, k in seen.items() if k >= max(2, int(0.7 * n))}
    after_ids: set[int] = set()
    newborn = 0
    t = t1 + 0.2
    while t <= t1 + 1.6 + 1e-9:
        for tr in reader.tracks_at(t):
            if tr.active and tr.number < 0:
                after_ids.add(tr.id)
                if tr.id not in seen:
                    newborn = 1
        t += 0.3
    carried, _ = reader.hand_context(t0, t1 + 1.6)
    dead = [i for i in residents if i not in after_ids and i not in carried]
    return max(0, len(dead) - newborn), dead


def departed_for_shot(reader: SidecarReader, s: dict) -> tuple[set[int], dict]:
    t0, t1 = float(s["start"]), float(s["end"])
    before = stable_numbers(reader, max(0.0, t0 - 1.5), t0 - 0.2)
    after = stable_numbers(reader, t1 + 0.2, t1 + 1.6)
    gone = before - after
    detail = {"before": sorted(before), "after": sorted(after)}
    confirmed: set[int] = set()
    for num in gone:
        # Where was it last seen before vanishing?
        last_pos = None
        t = t0 - 0.2
        while t <= t1 + 1e-9:
            for tr in reader.tracks_at(t):
                if tr.active and tr.number == num:
                    last_pos = (tr.x, tr.y, tr.radius)
            t += 0.3
        # Does it come back — and HOW? A flickered ball returns hands-free
        # at the SAME spot; a potted-then-REPLACED ball (Joe's re-spread
        # drills) returns near a hand or at a different spot. Only the
        # flicker case cancels the departure.
        flicker = False
        t = t1 + 1.6
        horizon = min(t1 + 12.0, reader._times[-1] if reader._times else t1)
        while t <= horizon:
            hit = next((tr for tr in reader.tracks_at(t)
                        if tr.active and tr.number == num), None)
            if hit is not None:
                same_spot = (last_pos is not None
                             and ((hit.x - last_pos[0]) ** 2
                                  + (hit.y - last_pos[1]) ** 2) ** 0.5
                             < 2.0 * max(6.0, last_pos[2]))
                hands, _ = reader.hand_context(max(t1, t - 2.5), t + 0.3)
                flicker = same_spot and not hands
                detail.setdefault(
                    "returned", []).append(
                    {"num": num, "t": round(t, 1),
                     "mode": "flicker" if flicker else "replaced"})
                break
            t += 0.5
        if not flicker:
            confirmed.add(num)
    n_anon, dead_ids = anon_departures(reader, t0, t1)
    if n_anon:
        detail["anon"] = {"n": n_anon, "track_ids": sorted(dead_ids)}
    return confirmed, detail


def audit(video: Path) -> list[dict]:
    reader = SidecarReader(video)
    rows = []
    for i, s in enumerate(reader.shots, 1):
        gone, detail = departed_for_shot(reader, s)
        rec = s.get("outcome", "?")
        anon = detail.get("anon", {}).get("n", 0)
        if 0 in gone:
            derived = "scratch"
        elif gone or anon:
            derived = "make"
        else:
            derived = "miss"
        rows.append({"shot": i, "recorded": rec, "derived": derived,
                     "departed": sorted(gone), "agrees": derived == rec,
                     **detail})
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
