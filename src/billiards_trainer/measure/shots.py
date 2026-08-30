"""Shot detection + outcome judgment FROM dense tracks — stages of the
one Input→Output box (Joe, 2026-08-28: nothing happens to replay clips
that the core doesn't do to the live feed; this module is pure over
track streams and doesn't know which feed produced them).

Rules, uniform for every shot:
- An EPISODE opens when any ball sustains motion and closes when all
  balls are quiet for SETTLE_S.
- A ball's outcome is judged AT SETTLE: resting on the table = not
  pocketed, no matter what happens to it later (Joe's 3-ball was
  picked up seconds after settling mid-table and the old path scored
  the vanish as a make — case law, session-20260826-002906 @ 80s).
- POCKETED means the track ended WHILE MOVING and never returned
  during the episode. Pocket NAMING (which pocket) waits on visual
  pocket localization; until then the last position rides along.
- The cue ball pocketing is a SCRATCH.
"""

from __future__ import annotations

from dataclasses import dataclass, field

MOVE_PX_S = 60.0     # above this a ball is moving (rest jitter ~10)
SUSTAIN_S = 0.20     # motion must persist this long to open an episode
BACKDATE_S = 0.30    # strike precedes the first confirmed-motion frame
SETTLE_S = 0.70      # all-quiet this long closes the episode
LINGER_S = 2.0       # a vanished ball may reappear this long after
                     # settle and still cancel its pocket credit
PRESENT_S = 0.5      # seen within this window of settle = on the table
POT_PRESENT_S = 0.5  # a ball must have been seen this soon after the
                     # episode opened to be eligible for POT credit. It
                     # was either already on the table or it was not; a
                     # ball that materialises mid-shot beside a pocket is
                     # a misname, not a make.


@dataclass
class Episode:
    t_strike: float
    t_settle: float
    movers: set = field(default_factory=set)
    setup: bool = False          # hand-driven (placing/gathering), not a stroke
    cue_travel: float = 0.0      # px the cue ball covered in this episode
    cue_peak: float = 0.0        # px/s, the cue ball's fastest step here -
                                 # a struck ball reaches 691+ on the bench,
                                 # a hand-rolled one 213 (round 35)
    cue_moved: bool = False      # the cue ball moved: the definition of a
                                 # stroke. Measured on the bench (round 16):
                                 # all 10 real strokes have it, and the
                                 # invented ones - balls TOSSED onto the
                                 # table, which roll free and look exactly
                                 # like a shot - do not. Hand-adjacency
                                 # cannot catch a tossed ball; this can.
    pocketed: list = field(default_factory=list)   # (number, pocket_x, pocket_y)
    resting: set = field(default_factory=set)
    lost: list = field(default_factory=list)       # (number, last_x, last_y)
    scratch: bool = False


UNNAMED_MIN_SPAN = 250.0     # px of travel for an unnamed track to count
UNNAMED_MIN_SPEED = 250.0    # px/s peak - a rolling ball, not a glove


def _series(times, frames):
    """Position/time series from dense frame rows (id,x,y,radius,number,
    cls,active). Named balls key by NUMBER; unnamed tracks key by
    -track_id (bench round 5: the potted 5 crossed the whole table as an
    unnamed track - one fifth of real motion has no name yet, and the
    box must not be blind to it)."""
    by_n: dict[int, list] = {}
    for j, rows in enumerate(frames):
        for tr in rows:
            if not tr[6]:
                continue
            n = tr[4]
            key = n if n >= 0 else -int(tr[0])
            by_n.setdefault(key, []).append((times[j], tr[1], tr[2]))
    # unnamed tracks must EARN ball status: substantial travel at ball
    # speed. Gloves/stick blobs dwell and creep; rolling balls cross felt.
    for key in [k for k in by_n if k < 0]:
        pts = by_n[key]
        xs = [p[1] for p in pts]
        ys = [p[2] for p in pts]
        span = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5
        peak = 0.0
        for k2 in range(1, len(pts)):
            dt = pts[k2][0] - pts[k2 - 1][0]
            if 0 < dt < 0.5:
                v = ((pts[k2][1] - pts[k2 - 1][1]) ** 2
                     + (pts[k2][2] - pts[k2 - 1][2]) ** 2) ** 0.5 / dt
                peak = max(peak, v)
        if span < UNNAMED_MIN_SPAN or peak < UNNAMED_MIN_SPEED:
            del by_n[key]
    return by_n


MIN_CUE_TRAVEL = 150.0   # px: below this the cue was addressed or nudged
                         # with the stick, not struck (bench: real strokes
                         # >=263px, nudges 8-29px)
MIN_CUE_PEAK = 400.0     # px/s: below this the cue was MOVED, not struck.
                         # Distance alone cannot tell Joe placing balls
                         # from a shot - the bench's one surviving fake
                         # covered 210px, over the 150px bar - but speed
                         # can: that fake peaks at 213 px/s while every
                         # one of the ten real strokes peaks at 691+.
                         # The bar sits in the empty middle of that gap.
CARRIED_SETUP = 0.5      # this share of moving samples hand-adjacent =
                         # the hand moved the balls, not a cue


def analyze(times, frames, pockets=None, pocket_r: float = 40.0,
            unnamed_pots: bool = False, carried=None) -> list[Episode]:
    """Batch shot finding over a dense stream. Pure; no I/O.

    pockets: [(x, y), ...] in the stream's coordinate space. A pocket
    CREDIT requires the track to die within 2.2 pocket radii of one —
    a track that dies mid-table is LOST (occlusion, pickup, tracker
    failure), never scored. Real data taught this within the hour:
    Joe's 3-ball track died at (126, 442) mid-table during his pickup
    and the position-blind rule scored it a make all over again.
    Without pockets, nothing is ever credited as pocketed."""
    if not times:
        return []
    by_n = _series(times, frames)
    # per-number motion timestamps
    moving_ts: dict[int, list] = {}
    for n, pts in by_n.items():
        out = moving_ts.setdefault(n, [])
        for k in range(1, len(pts)):
            (t0, x0, y0), (t1, x1, y1) = pts[k - 1], pts[k]
            dt = t1 - t0
            if 0 < dt < 0.5:
                v = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 / dt
                if v > MOVE_PX_S:
                    out.append(t1)
    all_moving = sorted(t for ts in moving_ts.values() for t in ts)
    episodes: list[Episode] = []
    i = 0
    while i < len(all_moving):
        # sustained motion: SUSTAIN_S of consecutive moving samples
        j = i
        while (j + 1 < len(all_moving)
               and all_moving[j + 1] - all_moving[j] < 0.2):
            j += 1
            if all_moving[j] - all_moving[i] >= SUSTAIN_S:
                break
        if all_moving[j] - all_moving[i] < SUSTAIN_S:
            i = j + 1
            continue
        t_open = all_moving[i]
        # close: all-quiet gap of SETTLE_S in the moving stream
        k = j
        while (k + 1 < len(all_moving)
               and all_moving[k + 1] - all_moving[k] < SETTLE_S):
            k += 1
        t_close = all_moving[k]
        ep = Episode(t_strike=round(t_open - BACKDATE_S, 3),
                     t_settle=round(t_close, 3))
        if carried:
            # count moving frames whose movers were hand-adjacent
            hot = tot = 0
            for j, ts in enumerate(times):
                if not (t_open <= ts <= t_close):
                    continue
                ids = {tr[0] for tr in frames[j] if tr[6]}
                if not ids:
                    continue
                tot += 1
                if set(carried[j] if j < len(carried) else ()) & ids:
                    hot += 1
            ep.setup = tot > 0 and hot / tot >= CARRIED_SETUP
        _judge(ep, by_n, moving_ts, t_open, t_close, pockets, pocket_r,
               unnamed_pots)
        ep.cue_moved = 0 in ep.movers
        # HOW FAR the cue actually went. Measured on the bench (round 16):
        # real strokes move it 263px at minimum, while addressing the ball
        # or pushing it with the stick registers 8-29px. Joe nudges the cue
        # constantly while setting up; that is not a shot.
        cue_pts = [p for p in by_n.get(0, [])
                   if t_open - 0.3 <= p[0] <= t_close]
        ep.cue_travel = sum(
            ((cue_pts[k][1] - cue_pts[k - 1][1]) ** 2
             + (cue_pts[k][2] - cue_pts[k - 1][2]) ** 2) ** 0.5
            for k in range(1, len(cue_pts)))
        # HOW FAST, not just how far. A hand cannot roll a ball at stroke
        # speed, and that is what separates a real shot from Joe placing
        # balls: measured over the whole bench, the one hand-setup that
        # beat the distance test peaks at 213 px/s while every real
        # stroke peaks between 691 and 2063. Distance alone could not
        # split them - the fake covered 210px against a 150px bar.
        # NB: `q`, not `k` - the enclosing episode scan uses `k` for the
        # close index and then advances with `i = k + 1`. A statement-level
        # loop rebinds it (the cue_travel generator above has its own
        # scope and does not), which sent the scan back over episodes it
        # had already emitted and turned analyze() into a ~30-minute
        # crawl. Cheap to write, expensive to find.
        peak = 0.0
        for q in range(1, len(cue_pts)):
            dt = cue_pts[q][0] - cue_pts[q - 1][0]
            if 0 < dt < 0.2:
                v = (((cue_pts[q][1] - cue_pts[q - 1][1]) ** 2
                      + (cue_pts[q][2] - cue_pts[q - 1][2]) ** 2) ** 0.5) / dt
                peak = max(peak, v)
        ep.cue_peak = peak
        episodes.append(ep)
        i = k + 1
    return episodes


def _judge(ep: Episode, by_n, moving_ts, t_open, t_close,
           pockets=None, pocket_r: float = 40.0,
           unnamed_pots: bool = False) -> None:
    for n, pts in by_n.items():
        # participated? seen in the second before the strike or moved
        pre = [p for p in pts if t_open - 1.0 <= p[0] <= t_close]
        if not pre:
            continue
        moved = [t for t in moving_ts.get(n, []) if t_open <= t <= t_close]
        if moved:
            ep.movers.add(n)
        win = [p for p in pts if p[0] <= t_close + LINGER_S]
        if not win:
            continue
        if not moved:
            last = win[-1]
        else:
            # CONTIGUOUS chain only (bench round 3: the potted 2's track
            # died in the pocket, then a jaw-furniture flicker 2s later
            # was relabeled "2" and read as the ball RESTING on-table -
            # a make scored as a miss). Rest evidence must share an
            # unbroken chain with the motion; a >0.5s hole then rebirth
            # is a new life, not a resting ball.
            i = next((k for k, p in enumerate(win)
                      if p[0] >= moved[-1] - 1e-9), len(win) - 1)
            j = i
            while j + 1 < len(win) and win[j + 1][0] - win[j][0] <= 0.5:
                j += 1
            last = win[j]
        if last is None:
            continue
        if not moved:
            # never moved this episode: on-table bystander
            if last[0] >= t_close - PRESENT_S:
                ep.resting.add(n)
            continue
        # THE discriminator: was the ball ever seen AT REST after its
        # final motion? A pocketed ball's track dies mid-flight (last
        # sighting == last motion); a resting ball keeps being tracked
        # after it stops — and stays "resting" even if picked up later
        # (the 3-ball case).
        if last[0] - moved[-1] > 0.3:
            # "resting" needs more than slow samples (bench round 4: both
            # uncredited pots HOVERED at the lip below the motion
            # threshold, then their chains DIED at the mouth). A resting
            # ball stays tracked; rest evidence counts only if the chain
            # survives the linger window or the rest is away from pockets.
            alive_through = last[0] >= t_close + LINGER_S - 0.2
            at_mouth = any(((last[1] - qx) ** 2
                            + (last[2] - qy) ** 2) ** 0.5 < 2.6 * pocket_r
                           for (qx, qy) in (pockets or []))
            if alive_through or not at_mouth:
                ep.resting.add(n)
                continue
            # fell asleep at the mouth, then vanished: fall through to
            # the pocket-credit path
        # YOU CANNOT POT A BALL THAT WAS NOT ON THE TABLE. A ball that
        # first appears in the MIDDLE of a shot, already beside a pocket,
        # and then dies there was never rolling into it - it is a brief
        # misname of the ball that actually fell, or of the traffic
        # around the mouth. Measured on the bench: the 31.7 stroke potted
        # the 1 (its samples run from before the strike, across the felt,
        # into the bottom-right pocket) but ALSO credited the 4, whose
        # entire series was 8 samples starting 0.9s into the shot, 40px
        # from that same pocket. Two balls cannot fall into one pocket at
        # one instant, and the 4 was resting mid-table throughout.
        # `pre`, not `pts`: the question is whether THIS episode had the
        # ball at its start, not whether the number appears somewhere in
        # the session. The real 4 was on the felt all clip - it simply
        # was not tracked under that number during this shot, which is
        # precisely why the phantom could borrow it.
        if not any(p[0] <= t_open + POT_PRESENT_S for p in pre):
            ep.lost.append((n, round(last[1], 1), round(last[2], 1)))
            continue
        # track died with motion: pocket credit iff the FINAL PATH passes
        # through a pocket zone. The death point alone is not enough -
        # a dropping ball's track coasts THROUGH the mouth and dies
        # beyond the bed (bench: the potted 2 died at (-54,1380), 200px
        # past the pocket it fell into), and the jaw filter thins the
        # blurred entry reads. Any of the last samples inside 2.6 radii
        # of a pocket = a drop.
        tail = [p for p in pts
                if last[0] - 1.0 <= p[0] <= last[0]]
        near = None
        for (qx, qy) in (pockets or []):
            for (_t, tx, ty) in tail:
                if ((tx - qx) ** 2 + (ty - qy) ** 2) ** 0.5 < 2.6 * pocket_r:
                    near = (qx, qy)
                    break
            if near:
                break
        if near is None and pockets:
            # BED-EXIT rule (bench: the potted 2's straight-line coast
            # passed 100px wide of the pocket CENTER, but a ball can only
            # leave the bed through a pocket). Bed bounds = the pocket
            # extremes; if the track's last on-bed position (just before
            # samples go off-bed) sits at a pocket mouth, that's the drop.
            xs = [q for q, _ in pockets]
            ys = [q for _, q in pockets]
            bx0, bx1, by0, by1 = min(xs), max(xs), min(ys), max(ys)
            exit_p = None
            seen_off = False
            for (_t, tx, ty) in reversed(tail):
                if bx0 <= tx <= bx1 and by0 <= ty <= by1:
                    if seen_off:
                        exit_p = (tx, ty)
                    break
                seen_off = True
            if exit_p is not None:
                for (qx, qy) in pockets:
                    d = ((exit_p[0] - qx) ** 2
                         + (exit_p[1] - qy) ** 2) ** 0.5
                    if d < 3.0 * pocket_r:
                        near = (qx, qy)
                        break
        if near is None:
            ep.lost.append((n, round(last[1], 1), round(last[2], 1)))
        elif n == 0:
            ep.scratch = True
        elif n >= 1 or unnamed_pots:
            # n < 0 = an unnamed track's drop. GATED until hand-context
            # lands: on the bench, the glove carrying balls out of a
            # pocket passes the span/speed gates and fakes a pot
            # (round 5's own false make at 25.3s). Real unnamed pots
            # wait rather than fabricate.
            ep.pocketed.append((n, round(near[0], 1), round(near[1], 1)))
        else:
            ep.lost.append((n, round(last[1], 1), round(last[2], 1)))
