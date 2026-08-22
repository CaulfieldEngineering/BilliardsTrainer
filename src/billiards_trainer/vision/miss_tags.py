"""Tag every miss in Joe's own words: left/right cut, missed left/right,
overcut/undercut.

Joe: "I'm interested in patternizing what I meant versus what happened
but I'm concerned about overcomplicating things. Let's start by simply
tagging each miss." So this produces LABELS first — the numbers ride
along underneath for when a pattern shows up and he wants magnitudes.

Conventions (pinned by tests; every verdict inverts if these drift):
  * Perspective is the SHOT's, not the room's: "left" means left when
    facing along the cue ball's direction of travel, so it is correct
    wherever Joe stands.
  * Signed cut angle: negative = LEFT cut (object ball goes left),
    positive = RIGHT cut. Straight-in is |cut| < STRAIGHT_DEG.
  * Overcut = hit too THIN (more cut than the shot needed);
    undercut = too FULL. Independent of side, so the word always means
    the same thing.

Everything here comes from the tracking record alone — no cue-stick
detection — so it works on every stroke shot in the archive, including
the ones where aim detection found nothing.
"""

import logging
import math

log = logging.getLogger("vision.miss_tags")

#: below this the shot plays as straight-in and over/undercut is meaningless
STRAIGHT_DEG = 5.0
#: rect px/sample step that counts as "moving" (sampling is ~7-10Hz)
_MOVE_PX = 6.0
_STEP = 0.1

#: Why the last tag_shot() call abstained. Diagnostics only — an abstention
#: is silent by design (a guessed label is worse than none), but when
#: coverage moves you need to know WHICH gate moved it. Same reasoning as
#: RECOV_DEBUG: instrument the rejection paths, do not theorise about them.
LAST_ABSTAIN = ""


def _no(reason: str):
    global LAST_ABSTAIN
    LAST_ABSTAIN = reason
    return None


def _cross(ux, uy, vx, vy) -> float:
    """z of u x v in image coords (y grows DOWN). Negative = v lies to
    the LEFT of u from the shooter's view (verified in tests)."""
    return ux * vy - uy * vx


def _unit(dx, dy):
    n = math.hypot(dx, dy)
    return (dx / n, dy / n) if n > 1e-9 else (0.0, 0.0)


def _signed_angle(ux, uy, vx, vy) -> float:
    """Degrees from u to v; negative = v to the shooter's left."""
    return math.degrees(math.atan2(_cross(ux, uy, vx, vy), ux * vx + uy * vy))


def _track_path(reader, num: int, t0: float, t1: float) -> list:
    """[(t, x, y)] for one ball across a window (active samples only)."""
    out = []
    t = t0
    while t <= t1 + 1e-9:
        for tr in reader.tracks_at(t):
            if tr.number == num and tr.active:
                out.append((t, tr.x, tr.y))
                break
        t += _STEP
    return out


def _first_motion(path: list) -> int | None:
    """Index where the ball first displaces meaningfully from its start."""
    if not path:
        return None
    x0, y0 = path[0][1], path[0][2]
    for i, (_t, x, y) in enumerate(path):
        if math.hypot(x - x0, y - y0) > _MOVE_PX:
            return i
    return None


def tag_shot(reader, shot: dict, space) -> dict | None:
    """Label one shot. Returns None when the geometry can't be read — an
    honest abstention beats a guessed label (Joe's standing rule), and
    ``LAST_ABSTAIN`` then says which gate stopped it.

    ``space`` is a TableSpace (true-inch frame); its pockets define the
    targets and its scale turns the miss into inches.
    """
    t0, t1 = float(shot.get("start", 0.0)), float(shot.get("end", 0.0))
    if t1 <= t0 or space is None:
        return _no("no shot window or no table space")
    lo, hi = max(0.0, t0 - 1.2), t1 + 0.5
    # --- the TARGET is the first object ball that moves in the window
    target, tpath, tidx = None, None, None
    for num in range(1, 16):
        p = _track_path(reader, num, lo, hi)
        i = _first_motion(p)
        if i is None or i + 3 >= len(p):
            continue
        if target is None or p[i][0] < tpath[tidx][0]:
            target, tpath, tidx = num, p, i
    if target is None:
        return _no("nothing was struck")
    contact_t = tpath[tidx][0]
    # CONTACT IS WHERE THE BALL WAS SITTING, not where it was first seen
    # moving. _first_motion returns the first sample DISPLACED from the
    # start, and on a hard shot that sample is already far downrange: on
    # 005048@233 the 4 had travelled 280px by the time it was first caught
    # moving, which put "contact" almost at the pocket and inverted
    # everything measured from it. The sample before is the ball at rest.
    cidx = max(0, tidx - 1)
    ox, oy = tpath[cidx][1], tpath[cidx][2]
    # --- object departure: the line of centres at impact (physics), and
    # the one direction this footage measures reliably.
    # Measured to where the ball GOT TO before it came back, not over a
    # fixed sample window. A fixed window was fine while balls were only
    # ever tracked slowly; now that fast ones are recovered it runs
    # straight into the rebound off the rail and reports the ball leaving
    # UP the table (measured on @233: v = (-0.118,-0.993), pointing back
    # the way it came). The outbound extremum is the honest departure.
    # Measured to where the ball GOT TO before it turned back toward
    # contact. Compared against the two alternatives across all 701
    # strokes in the library:
    #     rule       tagged  nonsense-angle  @233 (truth LEFT)
    #     window        290        199         RIGHT   <- wrong
    #     extremum      289        203         LEFT
    #     heading       253        238         LEFT
    # The original fixed 6-sample window ties this within noise on both
    # aggregate counts and is wrong on the only shot with ground truth, so
    # this is chosen for correctness at no measurable cost — not, as I
    # first assumed, because it improves coverage. Stopping instead at a
    # HEADING swing is clearly worse: the reference heading gets locked
    # from a baseline of half a ball radius, which is far too short to be
    # stable, so noise sets it and real samples then break the test.
    far, fx, fy = -1.0, ox, oy
    seg = []
    for q in tpath[tidx:]:
        d = math.hypot(q[1] - ox, q[2] - oy)
        if d < far - 0.5 * space.ball_r_px:
            break                          # it turned back: rail or pocket
        seg.append(q)
        if d > far:
            far, fx, fy = d, q[1], q[2]
    vx, vy = _unit(fx - ox, fy - oy)
    if (vx, vy) == (0.0, 0.0):
        return _no("object ball never displaced")
    # STRAIGHT-ROLL VALIDATION (Joe, on the shot still reading "can't
    # tell": the data knew the side; only the post-strike hole gated it).
    # A rolling ball travels in a straight line, so if EVERY observed
    # outbound point lies on one line from the resting spot, nothing
    # happened inside the gap — a ball cannot leave the line and return to
    # it. The gate below should fire only when the observations BEND,
    # which is what a rail contact hidden inside the hole would produce.
    # Measured on 005048@233: residuals 4.7px and 0.0px on a 14.3px ball.
    # This is the trajectory-fit idea in miniature: judged on ALL
    # observations with a physical threshold, not two adjacent samples.
    obs = [q for q in seg
           if math.hypot(q[1] - ox, q[2] - oy) > 0.5 * space.ball_r_px]
    line_ok = len(obs) >= 2 and all(
        abs(vx * (q[2] - oy) - vy * (q[1] - ox)) <= 0.75 * space.ball_r_px
        for q in obs)
    # PATH CONTINUITY (005048 @233, Joe: "this one should be a pretty
    # clear miss left" against a tag that said right). That label was
    # measured across a 0.9s HOLE in the object ball's track: the strike
    # blurs the ball, the tracker drops it, and a fresh track picks it up
    # near the pocket. Departure direction is then a chord across the
    # hole, not the ball's real line, and left/right can invert. Measured
    # library-wide, 62% of tags carried a hole or a mid-flight track
    # switch. Those keep their numbers but are excluded from the pattern
    # counts -- a guessed side is worse than no side.
    # the biggest hole in the OUTBOUND leg — the stretch the direction is
    # actually measured from. A gap here means the line is a chord across
    # missing frames rather than the ball's real path.
    departure_gap = max(
        [seg[i + 1][0] - seg[i][0] for i in range(len(seg) - 1)]
        + [tpath[tidx][0] - tpath[cidx][0]], default=0.0)
    # --- cue direction WITHOUT tracking the cue in flight. Measured on
    # real footage: the tracker loses a struck cue ball for seconds (at
    # 005048@233 it reported "cue moving" 2.4s AFTER contact), so a
    # direct velocity read is unusable. But the geometry is exact: an
    # object ball leaves along the LINE OF CENTRES, so the cue ball's
    # centre at impact is one ball-diameter back along that line, and
    # the cue's path is address -> that point (its flight is straight —
    # verified to 0.15px over 232px in the frame forensics).
    cue = _track_path(reader, 0, lo, contact_t - 0.05)
    if len(cue) < 3:
        return _no("cue ball not tracked before contact")
    ax = sum(q[1] for q in cue[:5]) / min(5, len(cue))
    ay = sum(q[2] for q in cue[:5]) / min(5, len(cue))
    d = 2.0 * float(space.ball_r_px)
    cx_hit, cy_hit = ox - d * vx, oy - d * vy
    ux, uy = _unit(cx_hit - ax, cy_hit - ay)
    if (ux, uy) == (0.0, 0.0):
        return _no("degenerate cue direction")
    if math.hypot(cx_hit - ax, cy_hit - ay) < 3.0 * d:
        return _no("cue too close to the object ball to resolve a direction")
    cut = _signed_angle(ux, uy, vx, vy)
    if abs(cut) > 88.0:
        return _no("cut angle beyond 88 deg")
    # --- which pocket was he playing? the one whose line from the object
    # ball best explains where it actually went (ahead of it, smallest
    # angular disagreement). Inference, flagged as such.
    best, best_off = None, 1e9
    for name, px, py in space.pockets():
        wx, wy = _unit(px - ox, py - oy)
        if wx * vx + wy * vy <= 0:
            continue                      # behind the object ball
        off = abs(_signed_angle(vx, vy, wx, wy))
        if off < best_off:
            best, best_off = (name, px, py), off
    if best is None or best_off > 45.0:
        return _no("object path points at no pocket (>45 deg off)")
    pname, px, py = best
    # --- required cut for that pocket, and the error
    rx, ry = _unit(px - ox, py - oy)
    required = _signed_angle(ux, uy, rx, ry)
    over_deg = abs(cut) - abs(required)   # + = too thin = OVERCUT
    # --- which side of the pocket the ball actually passed
    side_sign = _cross(vx, vy, px - ox, py - oy)
    miss_side = "right" if side_sign < 0 else "left"
    # perpendicular miss distance at the pocket
    miss_px = abs(side_sign) / max(1e-9, math.hypot(vx, vy))
    straight = abs(required) < STRAIGHT_DEG
    # CONFIDENCE — the pocket is INFERRED, and a wrong inference inverts
    # every label downstream. Two tells, both measured on the archive:
    # a "miss" of a foot or more usually means the ball was played at a
    # DIFFERENT pocket than the one geometry picked; and a sub-inch
    # "miss" at the mouth is a ball that went in (or rattled), not a
    # miss at all. Those tags stay in the record but are excluded from
    # the pattern counts rather than quietly skewing them.
    miss_in_val = (abs(side_sign) / max(1e-9, math.hypot(vx, vy))
                   / space.px_per_in)
    if departure_gap > 3.0 * _STEP and not line_ok:
        conf = (f"low: object ball untracked for {departure_gap:.1f}s right "
                f"after contact -- its departure line is a guess")
    elif miss_in_val > 8.0:
        conf = f"low: pocket inference doubtful (missed by {miss_in_val:.0f}in)"
    elif miss_in_val < 0.8:
        conf = "low: ball was at the pocket mouth (rattle or mislabelled)"
    elif best_off > 25.0:
        conf = f"low: object path {best_off:.0f} deg off any pocket line"
    else:
        conf = "high"
    tags = {
        "target": int(target),
        "pocket": pname,
        "cut": "straight" if straight else ("left" if cut < 0 else "right"),
        "cut_deg": round(cut, 1),
        "required_deg": round(required, 1),
        "miss_side": miss_side,
        "miss_in": round(miss_px / space.px_per_in, 2),
        "pocket_inferred": True,
        "confidence": conf,
    }
    # GEOMETRY THE LABEL IS MADE OF (Joe: "an optional visual overlay to
    # understand where your 'shot left, misses left' definition is coming
    # from"). Rect-space points; the exporter maps them to video coords so
    # both surfaces draw the identical figure.
    # THE OBJECT BALL'S ACTUAL OUTBOUND PATH, cleaned. Joe: "the 4 ball is
    # still being projected to the wrong path" and "I don't care what
    # happens to the cue ball once it hits its object ball". The raw track
    # for the target carries two things that are not the object ball
    # travelling to the pocket: a leading stub from before/at contact that
    # actually belongs to the CUE ball (the arriving cue steals the resting
    # track -- see the strike-time swap), and the return leg after the ball
    # rattles out. Both read as "the line went somewhere the ball didn't".
    # Keep only samples that make forward progress toward the pocket.
    outbound = []
    denom = math.hypot(px - ox, py - oy) or 1.0
    prog = -1e9
    for (_t, qx, qy) in tpath[cidx:]:
        # from CONTACT, not from the first sample seen moving — otherwise
        # the drawn line starts wherever the ball was first caught, which
        # on a hard shot is most of the way to the rail (@233: it began at
        # 0.23,0.81 instead of at the ball's address, a 2-point stub)
        d = ((qx - ox) * (px - ox) + (qy - oy) * (py - oy)) / denom
        if d < -0.5 * space.ball_r_px:
            continue                      # behind contact: not this ball
        if d < prog - 0.5 * space.ball_r_px:
            break                         # stopped advancing: rattle/return
        prog = max(prog, d)
        outbound.append([round(qx, 1), round(qy, 1)])
    if len(outbound) >= 2:
        tags["path"] = outbound
    tags["geom"] = {
        "cue": [round(ax, 1), round(ay, 1)],          # cue ball at address
        "obj": [round(ox, 1), round(oy, 1)],          # object ball at contact
        "pocket": [round(px, 1), round(py, 1)],       # the pocket it was on
        # where the object ball ACTUALLY went (extended to the pocket's range)
        "went": [round(ox + vx * math.hypot(px - ox, py - oy), 1),
                 round(oy + vy * math.hypot(px - ox, py - oy), 1)],
    }
    if not straight:
        tags["fullness"] = "overcut" if over_deg > 0 else "undercut"
        tags["error_deg"] = round(over_deg, 1)
        # ball fractions are how players think about thickness
        tags["error_balls"] = round(
            abs(math.sin(math.radians(over_deg))) * 2.0, 2)
    return tags


def label(tags: dict) -> str:
    """The one-line human form: 'Left cut, missed left — overcut'."""
    if not tags:
        return ""
    cut = tags.get("cut", "")
    head = "Straight-in" if cut == "straight" else f"{cut.title()} cut"
    line = f"{head}, missed {tags.get('miss_side', '?')}"
    if tags.get("fullness"):
        line += f" — {tags['fullness']}"
    return line
