"""Identity-derived shot outcomes — the recomputable source of truth.

The live detector attributes outcomes in the moment, with the moment's
blind spots; frame-verification on a real session measured it 7/11 while
deriving outcomes from the sidecar's identity record measured 11/11
(docs/GOALS.md, loops 32-33). So outcomes are POST-COMPUTED: which
numbered balls left the bed across a shot (majority-sampled presence
sets), with two mechanisms frame truth forced:

  * return-mode discrimination — a departed ball that returns hands-free
    at the same spot merely flickered; one returning near a hand or at a
    new spot was potted and REPLACED (re-spread drills);
  * anonymous residents — a ball whose digit never faced the camera
    carries no number, but its settled track id dying during a shot
    (not hand-carried, no newborn re-ID) is a pot the number layer
    cannot see.

`derive_and_correct` runs after a recording closes and APPENDS
corrections (the sidecar is a log; the reader takes last-wins), so a
later human verdict in review always outranks the derivation.
"""

import logging
from collections import Counter

from .analysis_cache import SidecarReader, append_correction

log = logging.getLogger("vision.outcomes")


def numbers_at(reader: SidecarReader, t: float) -> set:
    """Numbered balls visible at t (active tracks only; ghosts have no
    number and therefore no vote)."""
    return {tr.number for tr in reader.tracks_at(t) if tr.active and tr.number >= 0}


def stable_numbers(reader: SidecarReader, t0: float, t1: float,
                   step: float = 0.3) -> set:
    """Numbers seen in a MAJORITY of samples across [t0, t1] — one
    flickered frame neither adds nor removes a ball."""
    c: Counter = Counter()
    n = 0
    t = t0
    while t <= t1 + 1e-9:
        for num in numbers_at(reader, t):
            c[num] += 1
        n += 1
        t += step
    return {num for num, k in c.items() if k >= max(2, n // 2)}


def anon_departures(reader: SidecarReader, t0: float, t1: float) -> tuple:
    """Departures the NUMBER layer cannot see: a ball whose digit never
    faced the camera carries num=-1 its whole life, so number
    set-difference is blind to its pot. But its TRACK ID is a stable
    resident of the before-window; if that id dies during the shot, was
    never hand-adjacent (not picked up), and no freshly-settled
    unnumbered track appears after (which would mean mid-flight re-ID, a
    moved ball not a potted one), an anonymous object ball left the bed."""
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
    after_ids: set = set()
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


def departed_for_shot(reader: SidecarReader, s: dict) -> tuple:
    """(confirmed departed numbers, detail dict) for one shot record."""
    t0, t1 = float(s["start"]), float(s["end"])
    before = stable_numbers(reader, max(0.0, t0 - 1.5), t0 - 0.2)
    after = stable_numbers(reader, t1 + 0.2, t1 + 1.6)
    gone = before - after
    detail = {"before": sorted(before), "after": sorted(after)}
    confirmed: set = set()
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
        # at the SAME spot; a potted-then-REPLACED ball (re-spread drills)
        # returns near a hand or at a different spot. Only the flicker
        # case cancels the departure.
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
                detail.setdefault("returned", []).append(
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


def derive_outcome(reader: SidecarReader, s: dict) -> tuple:
    """(derived outcome string, detail) for one shot record."""
    gone, detail = departed_for_shot(reader, s)
    anon = detail.get("anon", {}).get("n", 0)
    if 0 in gone:
        out = "scratch"
    elif gone or anon:
        out = "make"
    else:
        out = "miss"
    detail["departed"] = sorted(gone)
    return out, detail


def derive_and_correct(video_path) -> int:
    """Post-recording pass: derive every shot's outcome from the identity
    record and append a correction where it disagrees with what the live
    detector wrote. Returns the number of corrections appended.

    Append-only and idempotent: the reader applies existing corrections
    before this compares, so a re-run changes nothing, and a human verdict
    appended later in review always wins (last-wins by file order)."""
    try:
        reader = SidecarReader(video_path)
    except OSError:
        return 0
    fixed = 0
    for s in reader.shots:
        try:
            derived, _ = derive_outcome(reader, s)
        except Exception:  # noqa: BLE001 - derivation must never break close
            log.exception("outcome derivation failed for shot at %.1fs",
                          float(s.get("start", -1)))
            continue
        if s.get("_reviewed"):
            continue        # a human looked; the derivation stands down
        if derived != s.get("outcome"):
            if append_correction(video_path, float(s["start"]), derived,
                                 src="derived"):
                fixed += 1
                log.info("outcome corrected at %.1fs: %s -> %s (derived)",
                         float(s["start"]), s.get("outcome"), derived)
    return fixed
