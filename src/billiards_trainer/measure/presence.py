"""THE table-presence authority (Joe: "they should all be referencing
the same measurement core").

One instance, owned by the controller, answers "which numbered balls
are on the table right now" from the SAME per-frame tracks the
schematic draws. The ball tray renders this dict verbatim; the shot
detector cross-checks pot credits against the same tracks. Nobody
keeps a private copy of this opinion.

Semantics are DETECTION TRUTH, not comfort smoothing (Joe: "It should
update instantaneously with a list of balls it detects as on the
table"): a ball is present iff detected within ABSENT_S — just enough
grace to bridge the identifier's vote cadence and single-frame blinks.
A never-yet-seen ball shows absent (an unreadable number is an honest
gap, and flicker under an occluding arm is diagnostic signal). A
potted ball that reappears (respot, tracker recovery) turns present
again.
"""

from __future__ import annotations

ABSENT_S = 0.6


class TablePresence:
    """Pure presence tracker: update(seen_numbers, t) -> {n: present}."""

    def __init__(self, numbers=tuple(range(1, 10)), absent_s: float = ABSENT_S):
        self._numbers = tuple(numbers)
        self._absent_s = absent_s
        self._last_seen: dict[int, float] = {}

    def update(self, seen, t: float) -> dict[int, bool]:
        for n in seen:
            if n in self._numbers:
                self._last_seen[n] = t
        return {n: t - self._last_seen.get(n, -1e9) < self._absent_s
                for n in self._numbers}

    def reset(self) -> None:
        self._last_seen.clear()
