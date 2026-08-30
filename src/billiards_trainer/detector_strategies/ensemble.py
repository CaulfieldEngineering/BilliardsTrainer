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

from ..core.balls import number_to_class, pool_ball_bgr, stripe_reading
from ..core.types import BallClass
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

    # same forwarding for the execution-provider knob (input-lag fix): without
    # this the pipeline's hasattr check skips the ensemble and the inner
    # sessions silently stay on DirectML.
    @property
    def inference_provider(self):
        return getattr(self._finder, "inference_provider", "auto")

    @inference_provider.setter
    def inference_provider(self, v) -> None:
        self._finder.inference_provider = v
        self._identifier.inference_provider = v

    def detect(self, frame_bgr, calib, rescan: bool | None = None):
        found = self._finder.detect(frame_bgr, calib, rescan)
        if not found:
            return found
        # Measure every ball's colour up front. Everything downstream that
        # reasons about appearance depends on it, and it was previously set
        # only on a rare correction path (see sample_colour).
        for f in found:
            try:
                self.sample_colour(frame_bgr, f)
            except Exception:  # noqa: BLE001 - a colour read is never fatal
                pass
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
        # Greedy nearest pairing, each identity used at most once, so two
        # adjacent finds can't both claim the same number. (An empty
        # identity pass skips pairing but NOT the naming/correction stage
        # below — heuristic guesses need checking most exactly when the
        # identifier went blind; the early return here once skipped both.)
        used_f: set[int] = set()
        pairs = []
        for fi, f in enumerate(found):
            lim = (0.9 * max(f.radius, 6.0)) ** 2
            for di, d in enumerate(self._last_ids):
                d2 = (f.x - d.x) ** 2 + (f.y - d.y) ** 2
                if d2 <= lim:
                    pairs.append((d2, fi, di))
        pairs.sort(key=lambda p: p[0])
        used_d: set[int] = set()
        for _d2, fi, di in pairs:
            if fi in used_f or di in used_d:
                continue
            used_f.add(fi)
            used_d.add(di)
            src = self._last_ids[di]
            f = found[fi]
            f.number, f.cls, f.bgr = src.number, src.cls, src.bgr
            self.repair_identity(frame_bgr, f)
        # Frame-level uniqueness for colour naming: numbers already worn by
        # any detection this frame are off the menu, and each name we hand
        # out joins the exclusion — two green blobs cannot both become the 6
        # (review finding: parallel vote streams would commit two tracks to
        # one number and the at-rest contest then demotes the REAL ball).
        present = {f.number for f in found
                   if f.number is not None and f.number > 0}
        for fi, f in enumerate(found):
            if fi in used_f:
                continue
            self.repair_unread(frame_bgr, f, present)
        return found

    @staticmethod
    def repair_unread(frame_bgr, f, present: set) -> None:
        """Check a find the IDENTIFIER never read — one copy (round 66).

        An unread find keeps the FINDER's colour-heuristic guess, which
        nothing checks. Under Joe's warm light the purple 4 guesses BLUE
        (its crop measures 7.8 Lab from the 4's reference and 69 from the
        2's), so the 4 voted '2' every frame of a session, arbitration
        stripped the duplicate, and the ball stayed nameless. Same
        decisive-margin machinery, same trust order: measured table
        colour over a canonical-palette guess.

        THE ENGINE NEVER DID THIS (round 66). _pair_identities gates the
        whole repair behind `if num >= 0`, and the identifier reads only
        a fraction of the balls on the table - on the cold clip's
        176-196s it reads the cue and the green 6 and NOTHING ELSE, so
        the stripe, the 8 and the 7 were named by the unchecked
        heuristic. That is where the invented numbers come from: the
        heuristic emits stripe numbers directly, and 330 frames of "13"
        were its guess with no measured colour ever consulted.
        """
        if f.number is None or f.number < 0:
            FindIdEnsemble._name_unknown(frame_bgr, f, present)
            if f.number is not None and f.number > 0:
                present.add(f.number)
        else:
            FindIdEnsemble._fix_colour(frame_bgr, f)

    @staticmethod
    def _name_unknown(frame_bgr, f, taken=frozenset()) -> None:
        """Name a ball NEITHER model could — the green 6 on turquoise felt.

        The identifier misses digit-down felt-coloured balls, and the colour
        heuristic ERASES their pixels along with the felt (UNKNOWN by
        design), so the 6 went unnamed for an entire session and its pot
        derived as a miss. This table's measured references know its real
        colour: the 6's tight crop median measures 9 Lab units from its
        reference while bare felt measures 64+ and a half-felt crop 25+
        (probed on real footage) — so a TIGHT absolute bar plus a decisive
        margin admits the ball and nothing else. Solids only (a stripe's
        band is model-readable from any orientation; whites never name), and
        the result is an ordinary per-frame READ: the tracker still demands
        a 3-vote majority and global uniqueness before commitment."""
        from ..core.balls import lab_distance_to_ref, measured_identity
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
        m = measured_identity(med, taken=set(taken), max_dist=18.0)
        if not 1 <= m <= 7:
            return
        best = lab_distance_to_ref(med, m)
        # Margin against EVERY loaded reference, stripes included — the 14
        # measures ~29 Lab from the 6 on this table, and a runner scan of
        # solids only would never see it standing two units behind a "6"
        # (review finding). And FAIL CLOSED: no runner-up reference to
        # measure a margin against means no decisive naming, not a free
        # pass — a sparse regenerated refs file must not widen the gate.
        from ..core.balls import _load_measured_refs
        runner = min((d for k in _load_measured_refs() if k != m
                      for d in [lab_distance_to_ref(med, k)] if d is not None),
                     default=None)
        if best is None or runner is None or runner - best < 12.0:
            return
        f.number = m
        f.cls = BallClass.SOLID
        f.bgr = med
        f.measured_bgr = med

    # TRIED AND REVERTED (round 58) - class_contradicts(): refuse an
    # identifier number whose class disagrees with the finder's
    # solid/stripe judgement. Measured over 900 cold-clip frames the two
    # agree in 6357 pairings and contradict in 1727, nearly all of it a
    # gold SOLID called "9", so it looked like the fix for that clip's
    # invented numbers. The scorecard rejected it immediately: the
    # BENCH's 9 is a yellow STRIPE whose body reads SOLID to the finder,
    # so the rule vetoed a correct name and took naming 99.6% -> 76.8%,
    # outcomes 10/10 -> 8/10 and pots 4/4 -> 2/4. The finder's class is
    # a GUESS about stripes, not a measurement of them - which is
    # exactly why the stripe-band reader (_fix_stripe_bit) exists.

    @staticmethod
    def sample_colour(frame_bgr, f) -> None:
        """Record what this ball ACTUALLY looks like, on every detection.

        The tracker has a whole colour-evidence subsystem — colour_hist,
        _colour_consensus, colour adoption, and the settled-track colour
        veto — and all of it feeds on Detection.measured_bgr. That field
        was only ever set on the rare path where the ensemble corrects a
        number from an unambiguous colour, so it was None on 6 of 6
        detections through the 005048 @233 strike: the subsystem was
        starved, and the veto that should have stopped the purple 4's
        track adopting the white cue ball could not fire.

        Same tight, glare-trimmed crop the naming path already trusts,
        just recorded unconditionally. This is a MEASUREMENT, not a
        verdict: nothing here names a ball.
        """
        import numpy as np
        rr = max(2, int(round(f.radius * 0.7)))
        y0, x0 = max(0, int(f.y) - rr), max(0, int(f.x) - rr)
        crop = frame_bgr[y0:int(f.y) + rr + 1, x0:int(f.x) + rr + 1]
        if crop.size < 30:
            return
        px = crop.reshape(-1, 3).astype(np.float32)
        keep = px[px.mean(1) <= np.percentile(px.mean(1), 75)]   # trim glare
        if len(keep) < 10:
            return
        f.measured_bgr = tuple(int(v) for v in np.median(keep, axis=0))

    @staticmethod
    def repair_identity(frame_bgr, f) -> None:
        """THE repair applied to a model identity — one copy (round 65).

        There were two. The live path repaired the STRIPE BIT and then the
        COLOUR; the offline engine (_pair_identities) called only the
        colour half, so _fix_stripe_bit had never run on a recorded clip
        in its life — every scorecard number the campaign has published
        was measured with half the repair missing, and round 64's fix to
        the stripe reader changed literally nothing on either clip
        because the code it fixed was unreachable. Same class of bug as
        round 48 (sample_colour live-only) and the same cause: a
        SEQUENCE is a fact, and it had two owners.
        """
        FindIdEnsemble._fix_stripe_bit(frame_bgr, f)
        FindIdEnsemble._fix_colour(frame_bgr, f)

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
        if n is None or n <= 0:
            return
        if n > 8:
            return FindIdEnsemble._fix_stripe_colour(frame_bgr, f)
        # SOLID FAMILY (1..8): whole-crop median arbitrates identity — the
        # dark trio 4/7/8 is THE confusion cluster under warm light.
        from ..core.balls import _load_measured_refs as _loaded_refs
        from ..core.balls import lab_distance_to_ref, measured_identity
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
            # NO REFERENCE FOR THE CLAIM. Handing off here means an
            # unknown number is unchallengeable, which made the repair of
            # the purple-4-read-as-7 depend on an UNVALIDATED 2026-08-15
            # reference for a 7 that is not even on this table: install
            # only the measured six and the 4 collapses again, 136/136 ->
            # 7/136 (round 34). Absence of evidence for the claim must
            # not protect it. If the crop instead sits decisively on a
            # reference we DID measure, that measurement wins.
            m0 = measured_identity(med, max_dist=20.0)
            if not 1 <= m0 <= 8 or m0 == n:
                return
            best = lab_distance_to_ref(med, m0)
            runner = min((d for k in _loaded_refs() if k != m0
                          for d in [lab_distance_to_ref(med, k)]
                          if d is not None), default=None)
            if best is None or runner is None or runner - best < 25.0:
                return   # not decisive — leave the model alone
            f.number = m0
            f.cls = BallClass.EIGHT if m0 == 8 else BallClass.SOLID
            f.bgr = pool_ball_bgr(m0)
            f.measured_bgr = med
            return
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
        f.measured_bgr = med

    @staticmethod
    def _fix_stripe_colour(frame_bgr, f) -> None:
        """Stripe hue via the BAND, not the whole crop (9-as-13 x11 on
        ground truth: the whole-crop median of a stripe is white and says
        nothing). Sample only the saturated band pixels and compare to the
        SOLID references (a band is its base colour). No band at all means
        the "stripe" is the CUE (0-as-15 x5) — hand it back."""
        from ..core.balls import _load_measured_refs as _loaded_refs
        from ..core.balls import (band_colour, lab_distance_to_ref,
                                  measured_identity)
        n = f.number
        rr = max(2, int(round(f.radius * 0.85)))
        y0, x0 = max(0, int(f.y) - rr), max(0, int(f.x) - rr)
        crop = frame_bgr[y0:int(f.y) + rr + 1, x0:int(f.x) + rr + 1]
        band = band_colour(crop)
        if band is None:
            # No saturated band: the CUE — but only when the crop is
            # overwhelmingly white. A thin/edge-on band (the 11 viewed
            # pole-on) also fails the band minimum, and declaring it cue
            # regressed the 11 to 0% on ground truth. Below the white bar,
            # abstain: the model keeps its read.
            import cv2
            import numpy as np
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            s = hsv[:, :, 1].astype(np.float32)
            v = hsv[:, :, 2].astype(np.float32)
            if float(np.mean((s < 110) & (v > 170))) > 0.75:
                f.number, f.cls = 0, BallClass.CUE
                f.bgr = pool_ball_bgr(0)
            return
        base_claim = n - 8
        claimed_d = lab_distance_to_ref(band, base_claim)
        if claimed_d is None:
            # NO REFERENCE FOR THE CLAIM — and absence of evidence must
            # not protect it. _fix_colour bought this exact lesson in
            # round 34; this second copy never learned it, so a "13" on a
            # table with no 5 reference (round 60 excluded 3 and 5 there
            # as inseparable) was UNCHALLENGEABLE, and 330 frames of the
            # cold clip's 9 answered to it.
            # Thresholds are the BAND's own, measured over all 66
            # naming-truth sightings of that ball in round 65 — a band is
            # a narrower, noisier sample than a whole crop, so
            # _fix_colour's 20/25 would reject the truth here:
            #     nearest reference   the gold 1 in 66 of 66
            #     best distance       min 5.0  p50 13.8  max 24.1
            #     margin over runner  min 0.9  p10 11.9  p50 15.5
            # 30 admits every real sighting; a 10 margin abstains on the
            # near-ties rather than guessing, which leaves the model's
            # answer standing — the fail-closed direction.
            m0 = measured_identity(band, max_dist=30.0)
            if not 1 <= m0 <= 8 or m0 == base_claim:
                return
            best = lab_distance_to_ref(band, m0)
            runner = min((d for k in _loaded_refs() if k != m0
                          for d in [lab_distance_to_ref(band, k)]
                          if d is not None), default=None)
            if best is None or runner is None or runner - best < 10.0:
                return
            f.number = m0 + 8
            f.cls = BallClass.STRIPE
            f.bgr = pool_ball_bgr(f.number)
            f.measured_bgr = band
            return
        m = measured_identity(band, max_dist=45.0)
        if m <= 0 or m > 8 or m == base_claim:
            return
        alt_d = lab_distance_to_ref(band, m)
        if alt_d is None or claimed_d - alt_d < 12.0:
            return
        f.number = m + 8
        f.cls = BallClass.STRIPE
        f.bgr = pool_ball_bgr(f.number)
        f.measured_bgr = band

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
        # PROMOTION ONLY (2026-08-26). The stripe->solid DEMOTION branch
        # was calibrated on old glare-era footage where a washed-out 9
        # measured stripe-white; on the corrected 1/320 exposure a band-up
        # 9 measures white_frac ~0.19 — under the solid threshold — and
        # the demotion flipped TRUE 9s into 1s. Attribution chain, fully
        # measured: raw c7 answers 9 on 15/15 gameplay 9s; the ensemble
        # WITH demotion returned 1 for nine of them. Promotion-only,
        # gated on three labelled sets:
        #   gameplay (s11/s12): 9-ball 40%->100%, overall 87%->100%
        #   racks (s7-s10):     9-ball 52%->67%, overall 83%->84%
        #   old-era archive:    stripes 81%->89%, overall 95%->94% (the
        #     1 gives back model-said-9 repairs there — archived footage
        #     the live path never re-analyzes; accepted)
        if n is None or n <= 0 or n > 7:
            return                     # only a solid can be promoted
        rr = max(2, int(round(f.radius)))
        y0, x0 = max(0, int(f.y) - rr), max(0, int(f.x) - rr)
        crop = frame_bgr[y0:int(f.y) + rr + 1, x0:int(f.x) + rr + 1]
        if crop.size == 0:
            return
        if stripe_reading(crop) is True:
            f.number, f.cls = n + 8, BallClass.STRIPE
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
