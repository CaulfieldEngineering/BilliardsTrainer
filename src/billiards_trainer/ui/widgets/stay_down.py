"""Live stay-down timer: the counter Joe asked for.

"I'd like to have available a timer that counts the time stayed down on
a shot." The readout climbs in real time from the strike and then locks
to the camera-measured value when the stroke record lands (~20-40s
later), so the live number is an estimate (prefixed ~) and the final
number is the measurement.

Three anchors, best-available wins, all on the controller's pipeline
clock:
  1. the settled->moving edge (real time, the moment balls move)
  2. the shot event's backdated start (better strike estimate, ~1.5s
     median detection lag removed)
  3. the measured stroke record (truth; replaces the estimate)

Pure logic, no Qt — the page owns the label and colours by `kind`:
  idle   nothing to show ("—")
  climb  counting live ("~1.2s")
  wait   climb capped, measurement pending ("…")
  final  measured ("1.5s")
  popped measured, popped up early ("0.6s POP")
"""

from __future__ import annotations

# A stay-down beyond this is no longer a reflex being trained — stop
# climbing and wait for the measurement (also what a missed stroke
# record degrades to, instead of counting forever).
CLIMB_CAP_S = 12.0
# Anchor/record matching tolerance: backdated starts and rebased stroke
# starts land within ~2s of the moving edge; 6s keeps distinct shots
# (>=8s apart by the detector's floors) unambiguous.
MATCH_S = 6.0


class StayDownTimer:
    def __init__(self):
        self._anchor: float | None = None    # strike estimate, pipeline s
        self._prev_state = "settled"
        self._text, self._kind = "—", "idle"

    def reset(self) -> None:
        """Recording stopped: back to the idle dash (keeps the motion
        state, so a mid-motion stop can't fake a fresh edge later)."""
        self._anchor, self._text, self._kind = None, "—", "idle"

    # ------------------------------------------------------------------ #
    def tick(self, shot_state: str, t: float, counting: bool) -> tuple[str, str]:
        """Per-frame: arm on the settled->moving edge, climb while armed.
        `counting` gates NEW arms (Joe's rule: stats belong to a
        recording) — a shot already climbing finishes its measurement."""
        moved = self._prev_state != "moving" and shot_state == "moving"
        self._prev_state = shot_state
        if moved and counting and t >= 0.0:
            self._anchor = t
            self._kind = "climb"
        if self._anchor is not None and self._kind in ("climb", "wait"):
            up = max(0.0, t - self._anchor)
            if up > CLIMB_CAP_S:
                self._text, self._kind = "…", "wait"
            else:
                self._text, self._kind = f"~{up:.1f}s", "climb"
        return self._text, self._kind

    def on_shot(self, start_t: float) -> None:
        """The detector's backdated start: a better strike estimate on the
        same clock — re-anchor the climb it belongs to."""
        if (self._anchor is not None and self._kind in ("climb", "wait")
                and abs(start_t - self._anchor) <= MATCH_S):
            self._anchor = start_t

    def on_stroke(self, rec: dict) -> tuple[str, str]:
        """The measured record lands: lock the readout to truth. Records
        for other shots (stale worker drain) are ignored."""
        if self._anchor is None:
            return self._text, self._kind
        try:
            start = float(rec.get("start", -1.0))
        except (TypeError, ValueError):
            return self._text, self._kind
        if abs(start - self._anchor) > MATCH_S:
            return self._text, self._kind
        sd = rec.get("stay_down_s")
        if sd is None or rec.get("confidence") == "none":
            self._text, self._kind = "—", "idle"   # measured-or-abstained
        elif rec.get("popped_early"):
            self._text, self._kind = f"{float(sd):.1f}s POP", "popped"
        else:
            self._text, self._kind = f"{float(sd):.1f}s", "final"
        self._anchor = None                        # done; next edge re-arms
        return self._text, self._kind
