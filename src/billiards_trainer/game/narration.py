"""Live narration gate (Joe: distinguish "Ball in hand" from "Table
change" — the latter being object balls moved in a non-shot fashion).

Distills the tracker's hand-context (which balls are hand-adjacent this
frame) into at most one spoken announcement per EPISODE:

  carried includes any OBJECT ball  -> "table_change" (racking, drills,
                                       rearranging - even if the cue is
                                       also in hand)
  carried is the CUE ball alone     -> "ball_in_hand" (the scratch pickup)

An episode starts when a hand first touches balls and ends after
QUIET_S with no carried balls; re-announcing within an episode is
suppressed, and nothing is announced while a shot is in flight (balls
rolling under a reaching hand are not a table change).
"""

from __future__ import annotations

QUIET_S = 2.5       # carried-empty this long = the episode ended
COOLDOWN_S = 8.0    # same announcement never repeats faster than this
MOVE_R = 1.2        # a "carried" ball must DISPLACE this many radii to count
ANCHOR_TTL_S = 2.0  # carry anchors expire this long after the hand leaves


class CarryMotion:
    """Movement gate for hand-adjacency (Joe: "a number of false table
    changes"): reaching over resting balls to address a shot flags them
    hand-adjacent, but a table change requires balls that actually MOVED
    while in hand. Anchors each ball's position when first flagged and
    reports which classes have displaced past MOVE_R radii."""

    def __init__(self):
        self._anchors: dict = {}    # tid -> (x, y, last_carried_t)

    def update(self, carried_tracks, t: float) -> tuple[bool, bool]:
        """carried_tracks: iterable of (tid, x, y, radius, is_cue).
        Returns (cue_moved_in_hand, object_moved_in_hand)."""
        cue_moved = obj_moved = False
        seen = set()
        for (tid, x, y, r, is_cue) in carried_tracks:
            seen.add(tid)
            a = self._anchors.get(tid)
            if a is None:
                self._anchors[tid] = (x, y, t)
                continue
            ax, ay, _ = a
            self._anchors[tid] = (ax, ay, t)
            if ((x - ax) ** 2 + (y - ay) ** 2) ** 0.5 > MOVE_R * max(r, 8.0):
                if is_cue:
                    cue_moved = True
                else:
                    obj_moved = True
        for tid in list(self._anchors):
            if tid not in seen and t - self._anchors[tid][2] > ANCHOR_TTL_S:
                del self._anchors[tid]
        return cue_moved, obj_moved


class Narrator:
    def __init__(self):
        self._episode = False
        self._last_carried_t = -1e9
        self._last_said: dict = {}
        self._episode_kind: str | None = None

    def update(self, carried_has_cue: bool, carried_has_object: bool,
               shot_in_flight: bool, t: float) -> str | None:
        carried = carried_has_cue or carried_has_object
        if carried:
            self._last_carried_t = t
        elif self._episode and t - self._last_carried_t > QUIET_S:
            self._episode = False
            self._episode_kind = None
        if not carried or shot_in_flight:
            return None
        kind = "table_change" if carried_has_object else "ball_in_hand"
        if self._episode:
            # an episode can ESCALATE (cue pickup becomes a full rerack)
            # but never repeats or de-escalates
            if kind == self._episode_kind or kind == "ball_in_hand":
                return None
        self._episode = True
        self._episode_kind = kind
        if t - self._last_said.get(kind, -1e9) < COOLDOWN_S:
            return None
        self._last_said[kind] = t
        return kind

    def note_cue_reappeared(self, t: float) -> str | None:
        """Second ball-in-hand trigger (Joe: "missed some of the ball in
        hands"): the cue ball vanishing >1s then reappearing at rest is a
        pickup even when the hand never registered over the felt (a quick
        clean grab). Same episode/cooldown machinery as the carried path,
        so the two triggers never double-announce."""
        if self._episode and self._episode_kind == "table_change":
            return None                  # a rerack subsumes the pickup
        self._last_carried_t = t
        if self._episode and self._episode_kind == "ball_in_hand":
            return None
        self._episode = True
        self._episode_kind = "ball_in_hand"
        if t - self._last_said.get("ball_in_hand", -1e9) < COOLDOWN_S:
            return None
        self._last_said["ball_in_hand"] = t
        return "ball_in_hand"
