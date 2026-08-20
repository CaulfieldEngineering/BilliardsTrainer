"""Action classification for detected table events — Joe's hierarchy #1.

The shot detector segments MOTION EPISODES; not every episode is a shot.
Joe's review found "a lot of false positive shots" — mostly ball-in-hand
relocations (picking up / re-spotting balls) recorded as strokes, plus
some zero-motion windows. Each sidecar shot is post-classified as:

    stroke        a real offensive attempt (cue stick strikes the ball)
    break         the opening smash into a racked cluster
    ball_in_hand  hand placing THE CUE BALL (position play, post-scratch)
    rearrange     hand shuffling OBJECT balls (re-spreading, gathering,
                  racking between drills — Joe: "different than ball in
                  hand")
    nothing       no meaningful ball activity in the window

Like outcomes, actions are RECOMPUTABLE and append-only: the classifier
writes {"type":"action"} records the reader attaches to shots
(last-wins), so review verdicts appended later always override, and the
whole library re-labels in seconds when the logic improves.

Feature notes (measured on frame-verified samples, Aug 2026):
  * hand fraction over the WINDOW is not enough — a bridge hand on the
    felt during a real stroke reads 0.5+;
  * the discriminating combination is hand presence PLUS how the
    principal mover travels (carried-adjacent, gradual hand-speed) PLUS
    whether anything moved at all;
  * fast balls can vanish from states during flight (motion blur), so
    a real stroke can show deceptively low sampled peak speed — never
    classify "nothing" unless the window is long-sampled AND still.
"""

import json
import logging

from .analysis_cache import SidecarReader, sidecar_path

log = logging.getLogger("vision.actions")


def shot_features(reader: SidecarReader, s: dict) -> dict:
    """Motion/hand features for one sidecar shot window."""
    t0, t1 = float(s["start"]), float(s["end"])
    prev = {}
    speed_hist = {}
    hand_states = 0
    n_states = 0
    t = t0
    while t <= t1 + 1e-9:
        n_states += 1
        ids_h, ff = reader.hand_context(t, t)
        if ids_h or ff > 0.02:
            hand_states += 1
        for tr in reader.tracks_at(t):
            if not tr.active:
                continue
            if tr.id in prev:
                pt, px, py = prev[tr.id]
                v = (((tr.x - px) ** 2 + (tr.y - py) ** 2) ** 0.5
                     / max(1e-3, t - pt))
                speed_hist.setdefault(tr.id, []).append((t, v))
            prev[tr.id] = (t, tr.x, tr.y)
        t += 0.15
    peaks = {i: max(v for _, v in h) for i, h in speed_hist.items()}
    movers = [i for i, v in peaks.items() if v > 60.0]
    fast = [i for i, v in peaks.items() if v > 250.0]
    carried_frac = 0.0
    carried_at_launch = False
    if peaks:
        main = max(peaks, key=peaks.get)
        moving = [(t, v) for t, v in speed_hist[main] if v > 60.0]
        if moving:
            c = 0
            for t, _ in moving:
                ids, _ = reader.hand_context(t - 0.08, t + 0.08)
                if main in ids:
                    c += 1
            carried_frac = c / len(moving)
            # THE launch test: a hand-thrown ball flies free (carried~0 in
            # flight, fast) exactly like a struck one — but it launches
            # FROM the hand, while a struck ball launches from rest with
            # the stick, bridge hand behind it, not on it (frame-verified:
            # every hand-rolled false stroke had the hand on the ball at
            # motion onset; real strokes measured 0.0-0.1 there).
            tl = moving[0][0]
            ids, _ = reader.hand_context(tl - 0.4, tl + 0.1)
            carried_at_launch = main in ids
    # Break vs mass-gathering: a break scatters everything in ONE burst;
    # gathering moves many balls fast too, but spread across the window
    # (hand-tossed one after another, 20s+). Launch synchrony separates.
    fast_sync = 0.0
    if len(fast) >= 2:
        launches = []
        for i in fast:
            for t, v in speed_hist[i]:
                if v > 250.0:
                    launches.append(t)
                    break
        fast_sync = max(launches) - min(launches)
    # A fast ball can be motion-blur invisible for its whole flight (the
    # 9-ball session's scratch stroke sampled peak=1), so "nothing moved"
    # is only believable when the BALL SET is unchanged across the window.
    from .outcomes import numbers_at
    set_changed = (numbers_at(reader, max(0.0, t0 - 0.5))
                   != numbers_at(reader, t1 + 1.0))
    # WHO did the hand move? Cue-only handling is ball-in-hand; object
    # balls moving under hands is rearranging (a different practice act).
    # Racking/gathering occludes everything, so tracked "movers" can be
    # empty — DISPLACEMENT across the window (who ended somewhere new)
    # is the robust signal, with mover identities as the fast path.
    mover_nums = set()
    for i in movers:
        num = next((tr.number for t_ in (t0,)
                    for tr in reader.tracks_at((speed_hist[i][0][0]
                                                if speed_hist.get(i) else t0))
                    if tr.id == i), -2)
        mover_nums.add(num)
    n_obj_movers = sum(1 for n in mover_nums if n != 0)
    a = {tr.number: (tr.x, tr.y) for tr in reader.tracks_at(max(0.0, t0 - 0.5))
         if tr.active and tr.number >= 0}
    b = {tr.number: (tr.x, tr.y) for tr in reader.tracks_at(t1 + 1.0)
         if tr.active and tr.number >= 0}
    moved_nums = {n for n in a if n in b
                  and ((a[n][0] - b[n][0]) ** 2
                       + (a[n][1] - b[n][1]) ** 2) ** 0.5 > 30.0}
    changed = moved_nums | (set(a) ^ set(b))
    displaced_obj = sum(1 for n in changed if n != 0)
    cue_displaced = 0 in changed
    return {
        "dur": t1 - t0,
        "peak": max(peaks.values(), default=0.0),
        "n_movers": len(movers),
        "n_obj_movers": n_obj_movers,
        "displaced_obj": displaced_obj,
        "cue_displaced": cue_displaced,
        "n_fast": len(fast),
        "hand_frac": hand_states / max(1, n_states),
        "carried_frac": carried_frac,
        "carried_at_launch": carried_at_launch,
        "fast_sync": fast_sync,
        "set_changed": set_changed,
        "n_states": n_states,
    }


def classify(f: dict) -> str:
    """Action from features, fitted to 27 frame-verified labels plus the
    11 verified real strokes of the 9-ball session (Aug 2026)."""
    # Opening smash: many balls at speed in ONE synchronized burst —
    # mass-gathering also moves many balls fast, spread over the window.
    if f["n_fast"] >= 5 and f["n_movers"] >= 7 and f["fast_sync"] <= 2.0:
        return "break"
    # Nothing moved across a well-sampled window — believable only when
    # the ball set is intact (a blur-invisible fast ball leaves a set
    # change behind: a real stroke, not an empty window).
    if f["peak"] < 60.0 and f["n_states"] >= 8 and not f["set_changed"]:
        return "nothing"
    # Hand relocations: the mover rode in a hand, LAUNCHED from a hand
    # (thrown/rolled balls fly free but start in the palm), or hands
    # owned a window where little else happened. WHICH kind depends on
    # what the hand moved: cue only = ball-in-hand (position play);
    # object balls = rearranging (re-spreads, gathering, racking).
    handled = (f["carried_frac"] >= 0.5 or f["carried_at_launch"]
               or (f["hand_frac"] >= 0.85 and f["n_movers"] <= 2))
    if handled:
        objs = max(f.get("n_obj_movers", 0), f.get("displaced_obj", 0))
        return "rearrange" if objs >= 1 else "ball_in_hand"
    # Mass re-spread: many OBJECT balls moving under sustained hands with
    # unsynchronized launches — gathering, not a stroke (a break's burst
    # is synchronized and was taken above; real strokes keep hands off
    # the felt for most of the window).
    if (f.get("n_obj_movers", 0) >= 4 and f["hand_frac"] >= 0.6
            and f["fast_sync"] > 2.0):
        return "rearrange"
    return "stroke"


def append_action(video_path, start: float, action: str) -> bool:
    p = sidecar_path(video_path)
    if not p.is_file():
        return False
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"type": "action",
                             "start": round(float(start), 3),
                             "action": action}) + "\n")
    return True


def classify_and_mark(video_path) -> dict:
    """Post-pass: classify every shot and append action records for
    non-default events. Idempotent (existing action attached before
    compare); later review verdicts win by append order."""
    try:
        reader = SidecarReader(video_path)
    except OSError:
        return {}
    counts: dict = {}
    for s in reader.shots:
        try:
            action = classify(shot_features(reader, s))
        except Exception:  # noqa: BLE001 - classification must never break
            log.exception("action classification failed at %.1fs",
                          float(s.get("start", -1)))
            continue
        counts[action] = counts.get(action, 0) + 1
        if action != s.get("action", "stroke"):
            append_action(video_path, float(s["start"]), action)
    return counts
