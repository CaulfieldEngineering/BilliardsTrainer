"""Find+identify ensemble: two models, each doing what it's best at.

The single-class finder (pool_yolo11) has the recall — 9.9 balls/frame on the
benchmark session — but no identity. The 16-class round-2 model knows WHICH
ball it's looking at (96% of track-frames numbered, mAP50 0.94) but inherits
detection blind spots from its bootstrapped training data (8.6 balls/frame).
So: positions come from the finder, and each found ball takes the number of
the nearest round-2 detection. Unmatched finds keep their colour-heuristic
guess — never fewer balls than the finder alone.
"""

from __future__ import annotations

import logging

from ..vision.balls import pool_ball_bgr, stripe_reading
from ..vision.types import BallClass
from . import DetectorStrategy, onnx_model

log = logging.getLogger("detector.ensemble")


class FindIdEnsemble(DetectorStrategy):
    model_based = True

    def __init__(self, finder, identifier):
        self._finder = finder
        self._identifier = identifier
        self.name = "ensemble_findid"
        self.description = (f"positions from {finder.name}, identities from "
                            f"{identifier.name}")
        self._tick = 0
        self._last_ids = []

    # the pipeline tunes this knob on whatever strategy is live
    @property
    def far_rail_rescan(self):
        return self._finder.far_rail_rescan

    @far_rail_rescan.setter
    def far_rail_rescan(self, v) -> None:
        self._finder.far_rail_rescan = v
        self._identifier.far_rail_rescan = v

    def detect(self, frame_bgr, calib, rescan: bool | None = None):
        found = self._finder.detect(frame_bgr, calib, rescan)
        if not found:
            return found
        # Identity pass every 2nd cycle (it costs a full tiled inference and
        # settled balls don't change number between cycles); stale identity
        # detections still match — the pairing radius absorbs the drift.
        self._tick += 1
        if self._tick % 2 == 1 or rescan is not None:
            try:
                self._last_ids = [d for d in
                                  self._identifier.detect(frame_bgr, calib, rescan)
                                  if d.number >= 0]
            except Exception:  # noqa: BLE001 - identity is enrichment, never fatal
                log.debug("identity pass failed", exc_info=True)
        if not self._last_ids:
            return found
        # Greedy nearest pairing, each identity used at most once, so two
        # adjacent finds can't both claim the same number.
        pairs = []
        for fi, f in enumerate(found):
            lim = (0.9 * max(f.radius, 6.0)) ** 2
            for di, d in enumerate(self._last_ids):
                d2 = (f.x - d.x) ** 2 + (f.y - d.y) ** 2
                if d2 <= lim:
                    pairs.append((d2, fi, di))
        pairs.sort(key=lambda p: p[0])
        used_f: set[int] = set()
        used_d: set[int] = set()
        for _d2, fi, di in pairs:
            if fi in used_f or di in used_d:
                continue
            used_f.add(fi)
            used_d.add(di)
            src = self._last_ids[di]
            f = found[fi]
            f.number, f.cls, f.bgr = src.number, src.cls, src.bgr
            self._fix_stripe_bit(frame_bgr, f)
            self._fix_colour(frame_bgr, f)
        return found

    @staticmethod
    def _fix_colour(frame_bgr, f) -> None:
        """Correct a model misread using THIS table's measured colours.

        The model was trained on canonical ball colours, but under Joe's warm
        light the purple 4 measures NAVY — so it reads as the 7 ("two sevens
        on the table"). The measured references (medians over ~460 labelled
        crops from this table, per-number, Lab space) separate the confusable
        pairs by 60+ units. When the crop's glare-trimmed colour is CLOSE to a
        different number's reference and FAR from the model's claim, trust the
        table over the model. Same base-colour family with a stripe/solid
        disagreement is left to _fix_stripe_bit, which reads actual pixels.
        """
        n = f.number
        # SOLID FAMILY ONLY (1..8): a stripe's whole-crop median is
        # white-dominant, so its measured colour cannot arbitrate identity
        # (first attempt corrected stripes too and misreads doubled). The 8
        # is included — the dark trio 4/7/8 is THE confusion cluster under
        # warm light (Joe: "an 8 ball is also being mistook for a 7").
        if n is None or n <= 0 or n > 8:
            return
        from ..vision.balls import lab_distance_to_ref, measured_identity
        rr = max(2, int(round(f.radius * 0.7)))
        y0, x0 = max(0, int(f.y) - rr), max(0, int(f.x) - rr)
        crop = frame_bgr[y0:int(f.y) + rr + 1, x0:int(f.x) + rr + 1]
        if crop.size < 30:
            return
        import numpy as np
        px = crop.reshape(-1, 3).astype(np.float32)
        keep = px[px.mean(1) <= np.percentile(px.mean(1), 75)]   # trim glare
        if len(keep) < 10:
            return
        med = tuple(int(v) for v in np.median(keep, axis=0))
        claimed_d = lab_distance_to_ref(med, n)
        if claimed_d is None:
            return   # no trustworthy reference for the claim — hands off
        # RELATIVE margin, not absolute gates: 7-vs-8 references sit only 41
        # Lab units apart, so 'claim within 40 = fine' let every 8-as-7 slip
        # through. Correct only when another solid-family reference beats the
        # claimed one decisively.
        m = measured_identity(med, max_dist=45.0)
        if m <= 0 or m == n or m > 8:
            return   # correct within the solid family (incl. the 8) only
        alt_d = lab_distance_to_ref(med, m)
        if alt_d is None or claimed_d - alt_d < 12.0:
            return   # not a decisive win — trust the model
        f.number = m
        f.cls = BallClass.EIGHT if m == 8 else BallClass.SOLID
        f.bgr = pool_ball_bgr(m)

    @staticmethod
    def _fix_stripe_bit(frame_bgr, f) -> None:
        """Repair the one bit the identity model reliably gets wrong.

        Measured on session-20260729: the model reads the purple 4 as the 12 and
        the yellow 9 as the 1 — HUE correct, stripe/solid inverted, once in each
        direction. Since stripe == solid + 8, a confident pixel reading of the
        crop fixes the number outright. stripe_reading() abstains when the crop
        is ambiguous, so this only ever overrides a clear disagreement.

        The cue (0) and the 8 are excluded: neither has a +/-8 partner, and the
        cue is all-white so it would always read as a stripe.
        """
        n = f.number
        if n is None or n <= 0 or n > 15 or n == 8:
            return
        rr = max(2, int(round(f.radius)))
        y0, x0 = max(0, int(f.y) - rr), max(0, int(f.x) - rr)
        crop = frame_bgr[y0:int(f.y) + rr + 1, x0:int(f.x) + rr + 1]
        if crop.size == 0:
            return
        reads_stripe = stripe_reading(crop)
        if reads_stripe is None:
            return
        if reads_stripe and n <= 7:
            f.number, f.cls = n + 8, BallClass.STRIPE
        elif not reads_stripe and n >= 9:
            f.number, f.cls = n - 8, BallClass.SOLID
        else:
            return
        f.bgr = pool_ball_bgr(f.number)


def _build():
    strategies = {s.name: s for s in getattr(onnx_model, "STRATEGIES", [])}
    finder = next((s for n, s in strategies.items() if "yolo11" in n), None)
    ident = next((s for n, s in strategies.items()
                  if "ballid" in n and "yolo11" not in n), None)
    if finder is None or ident is None:
        return []
    log.info("ensemble available: %s + %s", finder.name, ident.name)
    return [FindIdEnsemble(finder, ident)]


STRATEGIES = _build()
