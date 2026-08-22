"""Label-free tracking quality: score the pipeline against the laws of pool.

Why this exists
---------------
Hand-labelling is the bottleneck that kept the vision stack unmeasurable, and
an unmeasurable stack cannot be improved except by luck. But a pool table is a
rigid-body system with hard constraints, and *violating one is proof of a bug
without anyone labelling anything*:

  - two balls cannot occupy the same place at once
  - there is exactly one cue ball, and exactly one of each number
  - a ball at rest cannot change which ball it is
  - a ball cannot cross a cushion or leave the bed
  - a ball cannot move faster than a break, nor teleport
  - balls do not appear from nowhere, nor vanish away from a pocket

Every one of these is checkable on raw pipeline output over hours of footage
with zero human input. That is the engine that makes iteration autonomous.

The degenerate-detector trap
----------------------------
A detector that outputs NOTHING satisfies every law above perfectly. So a
violation count alone is worse than useless — it actively rewards blindness.
Two defences, both mandatory:

  1. Violations are always reported *per 1000 ball-frames*, so finding nothing
     yields no credit rather than a perfect score.
  2. :class:`InvariantReport` carries ``ball_frames`` and ``coverage`` and
     refuses to call a run good when it tracked implausibly few balls
     (``degenerate``). A scorer you can win by giving up is not a scorer.

Scale independence
------------------
Thresholds are expressed in BALL DIAMETERS, derived from the tracks' own median
radius — never in pixels. The same limits then hold at any resolution, any
rectified size, and on any table, with no per-camera retuning.
"""

import math
from collections import Counter
from dataclasses import dataclass, field

from ..core.types import BallClass, Track

# --------------------------------------------------------------------------- #
# Physical constants, in ball diameters (see "Scale independence" above).
# --------------------------------------------------------------------------- #

#: Centres closer than this are interpenetrating. Exactly 1.0 is touching; the
#: slack absorbs localisation noise so only real overlap is flagged.
_MIN_CENTRE_SEP = 0.80

#: A break is ~12 m/s. A 57mm ball covers ~210 diameters/s, i.e. ~7/frame at
#: 30fps. 9 is comfortably past the hardest possible shot, so anything beyond it
#: is a tracker identity swap rather than a real ball.
_MAX_SPEED_PER_FRAME = 9.0

#: Below this per-frame displacement a ball counts as stationary. Chosen above
#: typical centre jitter so "at rest" does not flicker.
_REST_SPEED = 0.12

#: How far outside the bed a centre may sit before it counts as off-table. A
#: ball resting against a cushion has its centre ~0.5 diameters from the rail
#: line, and pocketed balls sit further out still, hence the allowance.
_OFF_TABLE_MARGIN = 1.5

#: Radius around a pocket mouth within which disappearing is legitimate.
_POCKET_RADIUS = 2.5

#: Ball numbers that exist. 0 is the cue ball.
_VALID_NUMBERS = frozenset(range(0, 16))


@dataclass(frozen=True)
class Violation:
    """One broken physical law, with enough context to go and look at it."""

    kind: str
    frame: int
    detail: str
    #: ``impossible`` = provably a bug. ``implausible`` = suspicious, and worth
    #: mining as a hard example, but conceivably legitimate. Never conflate the
    #: two: only the first kind can be used as a hard regression gate.
    severity: str = "impossible"
    track_ids: tuple[int, ...] = ()

    def __str__(self) -> str:
        return f"[{self.severity}] f{self.frame} {self.kind}: {self.detail}"


@dataclass
class InvariantReport:
    """Aggregate score for a run. Read ``summary()`` for the honest version."""

    frames: int = 0
    ball_frames: int = 0
    violations: list[Violation] = field(default_factory=list)
    #: Per-frame tracked-ball counts, for coverage/stability.
    ball_counts: list[int] = field(default_factory=list)
    #: Frames where the pipeline was actually tracking (not calibrating).
    tracking_frames: int = 0

    # -- aggregates --------------------------------------------------------- #
    @property
    def counts(self) -> dict[str, int]:
        return dict(Counter(v.kind for v in self.violations))

    @property
    def impossible(self) -> int:
        return sum(1 for v in self.violations if v.severity == "impossible")

    @property
    def implausible(self) -> int:
        return sum(1 for v in self.violations if v.severity == "implausible")

    @property
    def coverage(self) -> float:
        """Mean balls tracked per tracking frame."""
        if not self.ball_counts:
            return 0.0
        return sum(self.ball_counts) / len(self.ball_counts)

    @property
    def count_stability(self) -> float:
        """Fraction of frames holding the modal ball count.

        Balls only leave the table when potted, so on any short window the count
        should be near-constant. Churn here is dropout//flicker, and it is
        visible without knowing what the true count actually is.
        """
        if not self.ball_counts:
            return 0.0
        modal = Counter(self.ball_counts).most_common(1)[0][1]
        return modal / len(self.ball_counts)

    @property
    def degenerate(self) -> bool:
        """True when the run tracked so little that its clean score is worthless.

        This is the anti-gaming guard: an empty detector breaks no laws, so
        without this a blind run would top the leaderboard.
        """
        return self.ball_frames == 0 or self.coverage < 1.0

    def rate(self, kind: str | None = None, severity: str | None = None) -> float:
        """Violations per 1000 ball-frames — the comparable headline number.

        Normalising by ball-frames (not frames) is what stops "detect less" from
        being a winning strategy.
        """
        if self.ball_frames == 0:
            return float("inf")
        n = sum(1 for v in self.violations
                if (kind is None or v.kind == kind)
                and (severity is None or v.severity == severity))
        return 1000.0 * n / self.ball_frames

    def worst(self, n: int = 10) -> list[Violation]:
        """Most frequent violation kinds first — where to look, in order."""
        order = {k: i for i, (k, _) in enumerate(Counter(
            v.kind for v in self.violations).most_common())}
        return sorted(self.violations, key=lambda v: order.get(v.kind, 99))[:n]

    def summary(self) -> str:
        lines = [
            f"frames={self.frames} tracking={self.tracking_frames} "
            f"ball-frames={self.ball_frames}",
            f"coverage={self.coverage:.2f} balls/frame  "
            f"count-stability={self.count_stability:.1%}",
            f"IMPOSSIBLE {self.impossible:5d}  ({self.rate(severity='impossible'):7.2f} /1k ball-frames)",
            f"implausible {self.implausible:4d}  ({self.rate(severity='implausible'):7.2f} /1k ball-frames)",
        ]
        if self.degenerate:
            lines.append("  !! DEGENERATE: too few balls tracked — a clean score "
                         "here means blindness, not quality.")
        for kind, n in Counter(v.kind for v in self.violations).most_common():
            lines.append(f"    {kind:22s} {n:6d}  ({1000.0 * n / max(1, self.ball_frames):7.2f} /1k)")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Per-frame checks
# --------------------------------------------------------------------------- #
def _median_diameter(tracks: list[Track]) -> float:
    """Ball diameter in track units, from the tracks themselves.

    Using the observed median rather than a configured constant keeps the whole
    scorer free of per-camera calibration.
    """
    radii = sorted(t.radius for t in tracks if t.radius > 0)
    if not radii:
        return 0.0
    mid = len(radii) // 2
    r = radii[mid] if len(radii) % 2 else 0.5 * (radii[mid - 1] + radii[mid])
    return 2.0 * r


def check_frame(tracks: list[Track], frame: int,
                bed: tuple[float, float, float, float] | None = None,
                max_number: int | None = None,
                diameter: float | None = None) -> list[Violation]:
    """Spatial laws that hold within a single frame.

    ``bed`` is ``(x0, y0, x1, y1)`` of the playing surface in track coordinates
    (rectified space), or None to skip containment checks. ``max_number``
    rejects balls that cannot be in play — in 9-ball, anything above 9.
    ``diameter`` is the known ball diameter (see SequenceScorer); when absent
    it is estimated from the tracks' own radii.
    """
    out: list[Violation] = []
    diam = diameter if diameter and diameter > 0 else _median_diameter(tracks)

    # -- identity uniqueness: there is one 3-ball, and one cue ball ---------- #
    by_number: dict[int, list[Track]] = {}
    for t in tracks:
        if t.number >= 0:
            by_number.setdefault(t.number, []).append(t)
    for num, group in by_number.items():
        if len(group) > 1:
            out.append(Violation(
                "duplicate_number", frame,
                f"{len(group)}x ball #{num} at " +
                ", ".join(f"({t.x:.0f},{t.y:.0f})" for t in group),
                track_ids=tuple(t.id for t in group)))
        if num not in _VALID_NUMBERS:
            out.append(Violation("invalid_number", frame, f"ball #{num} does not exist",
                                 track_ids=(group[0].id,)))
        if max_number is not None and 0 < num > max_number:
            # Not "impossible" in the physics sense — it is impossible *for this
            # game*, which is a config assumption, so it stays advisory.
            out.append(Violation(
                "out_of_game_number", frame,
                f"ball #{num} cannot be in play (max {max_number})",
                severity="implausible", track_ids=(group[0].id,)))

    cues = [t for t in tracks if t.cls == BallClass.CUE or t.number == 0]
    if len(cues) > 1:
        out.append(Violation("multiple_cue_balls", frame, f"{len(cues)} cue balls",
                             track_ids=tuple(t.id for t in cues)))

    # -- rigid bodies do not interpenetrate --------------------------------- #
    if diam > 0:
        floor = _MIN_CENTRE_SEP * diam
        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                a, b = tracks[i], tracks[j]
                d = math.hypot(a.x - b.x, a.y - b.y)
                if d < floor:
                    out.append(Violation(
                        "overlapping_balls", frame,
                        f"tracks {a.id}(#{a.number},m{a.misses})/"
                        f"{b.id}(#{b.number},m{b.misses}) centres "
                        f"{d / diam:.2f} diameters apart at "
                        f"({a.x:.0f},{a.y:.0f})/({b.x:.0f},{b.y:.0f})",
                        track_ids=(a.id, b.id)))

    # -- containment: balls stay on the bed --------------------------------- #
    if bed is not None and diam > 0:
        x0, y0, x1, y1 = bed
        m = _OFF_TABLE_MARGIN * diam
        for t in tracks:
            if not (x0 - m <= t.x <= x1 + m and y0 - m <= t.y <= y1 + m):
                out.append(Violation(
                    "off_table", frame,
                    f"track {t.id} at ({t.x:.0f},{t.y:.0f}) outside bed",
                    track_ids=(t.id,)))
    return out


# --------------------------------------------------------------------------- #
# Temporal checks — the ones that catch identity bugs
# --------------------------------------------------------------------------- #
class SequenceScorer:
    """Streaming scorer: feed frames in order, read the report at the end.

    Streaming (rather than collecting every frame first) so this can run over
    hour-long sessions in bounded memory.
    """

    def __init__(self, bed: tuple[float, float, float, float] | None = None,
                 max_number: int | None = None,
                 pockets: list[tuple[float, float]] | None = None,
                 diameter: float | None = None):
        """``diameter`` is the geometric ball diameter in track units (from
        table calibration). Neither ruler is trustworthy alone: the detector's
        box sizes run up to ~25% large on some sessions (touching balls then
        read 0.79 diameters and flip to violations on box drift), while the
        geometric estimate inherits any table-lock error (one session's lock
        includes rail margin, inflating it ~25% the other way). The effective
        ruler is therefore the per-frame box median CLAMPED to +-20% of the
        geometric diameter — box bias is capped by geometry, and bounded
        miscalibration is absorbed by the boxes."""
        self.report = InvariantReport()
        self._bed = bed
        self._max_number = max_number
        self._pockets = pockets or []
        self._diameter = diameter if diameter and diameter > 0 else None
        self._prev: dict[int, Track] = {}
        self._prev_diam = 0.0

    def _effective_diameter(self, tracks: list[Track]) -> float:
        measured = _median_diameter(tracks)
        geo = self._diameter
        if geo is None:
            return measured or self._prev_diam
        if measured <= 0:
            return self._prev_diam or geo
        return max(0.8 * geo, min(1.2 * geo, measured))

    # -- ingest ------------------------------------------------------------- #
    def add(self, tracks: list[Track], frame: int, tracking: bool = True) -> None:
        rep = self.report
        rep.frames += 1
        if not tracking:
            # Calibration frames carry no information about tracking quality;
            # counting them would dilute every rate.
            self._prev = {}
            return
        rep.tracking_frames += 1
        rep.ball_frames += len(tracks)
        rep.ball_counts.append(len(tracks))

        diam = self._effective_diameter(tracks)
        rep.violations.extend(check_frame(tracks, frame, self._bed, self._max_number,
                                          diameter=diam or None))
        if diam > 0:
            rep.violations.extend(self._temporal(tracks, frame, diam))
            self._prev_diam = diam
        self._prev = {t.id: t for t in tracks}

    def _temporal(self, tracks: list[Track], frame: int, diam: float) -> list[Violation]:
        out: list[Violation] = []
        seen = set()
        for t in tracks:
            seen.add(t.id)
            prev = self._prev.get(t.id)
            if prev is None:
                continue
            step = math.hypot(t.x - prev.x, t.y - prev.y) / diam

            # A ball cannot outrun a break. Beyond this the tracker has swapped
            # identity between two balls, which is the failure that silently
            # corrupts every downstream statistic.
            if step > _MAX_SPEED_PER_FRAME:
                out.append(Violation(
                    "teleport", frame,
                    f"track {t.id} moved {step:.1f} diameters in one frame",
                    track_ids=(t.id,)))

            # THE identity check: a ball that did not move cannot have become a
            # different ball. No labels needed, and it is exactly the bug that
            # makes ball IDs untrustworthy today.
            if (prev.number >= 0 and t.number >= 0 and prev.number != t.number
                    and step < _REST_SPEED):
                out.append(Violation(
                    "id_flicker", frame,
                    f"track {t.id} changed #{prev.number} -> #{t.number} while at rest",
                    track_ids=(t.id,)))
            if (prev.cls != BallClass.UNKNOWN and t.cls != BallClass.UNKNOWN
                    and prev.cls != t.cls and step < _REST_SPEED):
                out.append(Violation(
                    "class_flicker", frame,
                    f"track {t.id} changed {prev.cls.value} -> {t.cls.value} at rest",
                    track_ids=(t.id,)))

        # A ball that vanishes away from a pocket did not go anywhere — the
        # detector simply lost it. Near a pocket it is a legitimate pot.
        for tid, prev in self._prev.items():
            if tid in seen:
                continue
            if self._near_pocket(prev, diam):
                continue
            out.append(Violation(
                "vanished_mid_table", frame,
                f"track {tid} (#{prev.number}) disappeared at "
                f"({prev.x:.0f},{prev.y:.0f}), no pocket nearby",
                severity="implausible", track_ids=(tid,)))
        return out

    def _near_pocket(self, t: Track, diam: float) -> bool:
        if not self._pockets:
            return True   # unknown geometry: never accuse, stay conservative
        lim = _POCKET_RADIUS * diam
        return any(math.hypot(t.x - px, t.y - py) <= lim for px, py in self._pockets)


def check_sequence(frames: list[list[Track]],
                   bed: tuple[float, float, float, float] | None = None,
                   max_number: int | None = None,
                   pockets: list[tuple[float, float]] | None = None) -> InvariantReport:
    """Convenience wrapper over :class:`SequenceScorer` for in-memory sequences."""
    s = SequenceScorer(bed=bed, max_number=max_number, pockets=pockets)
    for i, tracks in enumerate(frames):
        s.add(tracks, i)
    return s.report


def bed_and_pockets(width: float, height: float
                    ) -> tuple[tuple[float, float, float, float],
                               list[tuple[float, float]]]:
    """Bed rect and the six pocket mouths for a rectified view of that size.

    In rectified space the bed *is* the image, so the pockets are the four
    corners plus the two side midpoints.
    """
    bed = (0.0, 0.0, width, height)
    pockets = [(0.0, 0.0), (width, 0.0), (0.0, height), (width, height),
               (0.0, height / 2.0), (width, height / 2.0)]
    return bed, pockets
