"""THE Measurement Core (Joe: "We just need the one Measurement Core
engine handling all of the data. Any animations, analysis or
interpretations should simply be pulling data from this Measurement
Core engine.").

One instance per live pipeline. Everything downstream — schematic,
ball tray, announcements, shot analysis, exports — READS from here and
holds no private opinion of the table. This module is also the C++
rebuild's seam: the API below is the spec of what the engine owes its
consumers.

Two feeds, one truth:

  ingest(dets, t)          — prepared detections (post filter-stack).
                             Runs the HARDENED MotionTracker (the same
                             rules the offline M1 engine proved:
                             gated association, coasting, rest-frozen
                             identity, track merge). Shadow until the
                             corpus gates promote it (M3 -> M4 in
                             docs/MEASUREMENT_CORE.md).
  observe_tracks(tracks,t) — the live champion tracker's emitted
                             tracks (what the schematic draws today).
                             Updates presence and scores DIVERGENCE
                             between champion and shadow, so promotion
                             is a measured decision, not a hope.

Reads:
  .present            {ball#: on-table} — detection truth, 0.6s grace
  .tracks             the authoritative track list (champion until
                      promotion; consumers never pick a tracker)
  .shadow_rows        the hardened tracker's current emissions
  .divergence_summary() champion-vs-shadow disagreement counters
"""

from __future__ import annotations

from .presence import TablePresence
from .tracker import MotionTracker

#: same-number positions farther apart than this many radii disagree
_POS_TOL_R = 2.0


class MeasurementCore:
    def __init__(self):
        self._shadow = MotionTracker()
        self._presence = TablePresence()
        self._present: dict[int, bool] = dict.fromkeys(range(1, 10), False)
        self._tracks: list = []
        self._shadow_rows: list = []
        self._t: float = -1.0
        self._div = {"frames": 0, "pos_mismatch": 0,
                     "shadow_missing": 0, "shadow_extra": 0}

    # ------------------------------------------------------------ feeds
    def ingest(self, dets, t: float) -> None:
        """Prepared detections [(x, y, radius, number), ...] at time t."""
        self._shadow_rows = self._shadow.update(
            [(float(x), float(y), float(r), int(n)) for x, y, r, n in dets], t)
        self._t = t

    def observe_tracks(self, tracks, t: float) -> None:
        """The champion tracker's emitted tracks for this instant."""
        self._tracks = list(tracks or [])
        self._present = self._presence.update(
            {getattr(tr, "number", -1) for tr in self._tracks
             if getattr(tr, "active", True)}, t)
        self._score_divergence()

    # ------------------------------------------------------------ reads
    @property
    def present(self) -> dict[int, bool]:
        return dict(self._present)

    @property
    def tracks(self) -> list:
        return list(self._tracks)

    @property
    def shadow_rows(self) -> list:
        return list(self._shadow_rows)

    def divergence_summary(self, reset: bool = False) -> dict:
        out = dict(self._div)
        if reset:
            for k in self._div:
                self._div[k] = 0
        return out

    # ---------------------------------------------------------- scoring
    def _score_divergence(self) -> None:
        """Champion vs shadow, numbered balls only — the promotion metric.
        Counted per observe (display cadence), comparing each side's
        latest opinion; a persistent disagreement accumulates weight."""
        if not self._shadow_rows and not self._tracks:
            return
        live = {getattr(tr, "number", -1): tr for tr in self._tracks
                if getattr(tr, "active", True)
                and getattr(tr, "number", -1) >= 1}
        shadow = {getattr(r, "number", -1): r for r in self._shadow_rows
                  if getattr(r, "number", -1) >= 1}
        self._div["frames"] += 1
        for n, tr in live.items():
            row = shadow.get(n)
            if row is None:
                self._div["shadow_missing"] += 1
                continue
            tol = _POS_TOL_R * max(getattr(tr, "radius", 0.0) or 0.0, 8.0)
            d = ((tr.x - row.x) ** 2 + (tr.y - row.y) ** 2) ** 0.5
            if d > tol:
                self._div["pos_mismatch"] += 1
        for n in shadow:
            if n not in live:
                self._div["shadow_extra"] += 1
