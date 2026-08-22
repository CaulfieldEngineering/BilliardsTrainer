"""Ball identity: who each track IS.

Split out of tracking.py (it had grown past 1000 lines) along its natural
seam: BallTracker's motion half decides WHERE things are — association,
coasting, budgets — and this half decides WHO they are. Number arbitration,
identity migration after a strike, and the measured-colour reasoning move
together because they share one vocabulary (votes, colour consensus,
reachability) and none of it touches motion state except through track
fields.

Everything here is a VERBATIM move — behaviour is pinned by
tests/test_tracking.py and must not change in a refactor.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import TYPE_CHECKING

from ..core.types import BallClass, Detection

if TYPE_CHECKING:                          # annotation-only; no runtime import,
    from .tracking import _Internal  # which would be circular

#: squared BGR distance at which a SETTLED track refuses a detection as
#: "not my ball". White vs purple is ~300 (90k squared); glare and motion
#: blur move a ball's own measured colour by well under half that, so the
#: bar sits high enough to veto only the unambiguous cases.
_COLOUR_VETO_SQ = 150.0 ** 2


class IdentityArbitration:
    """Mixin for BallTracker: the identity half of the tracker."""

    def _migrate_departed_numbers(self) -> None:
        """The struck ball keeps its NAME, not the ghost it left behind.

        Measured at 005647@386 (and 9 other shots in that session): when
        a ball is struck, its numbered track frequently does not die at
        all — it coasts frozen at the address spot on the occlusion
        budget while a FRESH anonymous track carries the real ball across
        the table. The number sits on the motionless ghost for the whole
        flight, so analytics saw no cue ball exactly when it mattered
        (0% coverage during flight on 10 of 36 shots).

        Rest- and flight-linking cannot help, because nothing died. The
        rule that applies: a number held by a COASTING, motionless track
        migrates to the single confirmed anonymous track that was BORN
        AT THAT SPOT and is demonstrably moving — i.e. the ball itself,
        departing. Every clause is load-bearing: one candidate only (a
        break scatters several), born at the spot (not any passing
        stranger), and actually in motion (a resting neighbour keeps its
        own identity)."""
        near = 6.0 * max(6.0, self._ball_r)
        for t in self._tracks:
            if t.committed_number < 0 or t.misses < 3:
                continue
            cands = []
            for o in self._tracks:
                if o is t or not o.confirmed or o.committed_number >= 0:
                    continue
                if o.misses != 0 or not o.moving() or not o.pos_hist:
                    continue
                if self._frame_n - getattr(o, "born_frame", 0) > 25:
                    continue
                bx, by = o.pos_hist[0]
                if math.hypot(bx - t.x, by - t.y) <= near:
                    cands.append(o)
            if len(cands) == 1:
                o = cands[0]
                o.committed_number = t.committed_number
                t.committed_number = -1
                ledger = getattr(self, "_num_last", {})
                ledger[o.committed_number] = (self._frame_n, o.x, o.y)
                self._num_last = ledger

    def _arbitrate_numbers(self) -> None:
        """GLOBAL EXCLUSIVE ASSIGNMENT of ball identities — one of each,
        decided jointly, every frame.

        Prior art (Pool-Aid's mutual exclusion, PoolLiveAid's bipartite
        matching) and our own corpus agree: per-track identity decided
        independently is why duplicates existed at all. Each confirmed track
        scores every candidate number from three signals — vote evidence
        (num_hist), measured-colour proximity (this table's references), and
        a stickiness bonus for its currently committed number (identity must
        not churn). A greedy best-score-first pass assigns each number at
        most once; any track NOT in demonstrable motion keeps its committed
        number as a hard constraint (physics: a resting ball cannot become a
        different ball — and motion is judged from the PUBLISHED step streak,
        the same quantity the physics scorer measures, so neither a one-frame
        glare blip nor a sub-threshold smoothing glide can open the
        reassignment window a settled-bit reset used to). Tracks left
        without a plausible free identity show unknown rather than a guess.

        Ghost cleanup rides along as before: a stale coasting claimant of a
        LIVE number is deleted outright (its frozen graphic haunted the
        table otherwise)."""

        tracks = [t for t in self._tracks if t.confirmed]
        if not tracks:
            return

        # --- stale-ghost cleanup (pre-pass, unchanged behaviour) ---------- #
        by_num: dict[int, list[_Internal]] = {}
        for t in tracks:
            if t.committed_number >= 0:
                by_num.setdefault(t.committed_number, []).append(t)
        doomed: set[int] = set()
        for _num, ts in by_num.items():
            if len(ts) >= 2 and any(t.misses == 0 for t in ts):
                for t in ts:
                    if t.misses > self.max_misses:
                        doomed.add(t.id)
        if doomed:
            self._tracks = [t for t in self._tracks if t.id not in doomed]
            tracks = [t for t in tracks if t.id not in doomed]

        # --- candidate scoring ------------------------------------------- #
        # score units are commensurable "evidence points": one recent vote
        # ~= 1.0; colour proximity contributes up to ~6; stickiness +4;
        # settled commitment is hard (inf).
        STICKY = 4.0
        COLOUR_MAX = 6.0
        # A ball not in demonstrable motion cannot become a different ball,
        # so its commitment is a hard constraint below; and a track with NO
        # number yet needs a formed vote majority (3 reads) before greedy
        # may name it — one wobbly read plus a colour match was exactly the
        # rack-time churn the corpus flagged (commit #5 at rack, flip "at
        # rest" to #9 five votes later).
        rest = [not t.moving() for t in tracks]
        cands: list[tuple[float, int, int]] = []   # (score, track_idx, number)
        # CUE-SIZE FLOOR (005647 @209s): a persistent white speck on the
        # felt (drill sticker, r10) kept winning the freed cue number
        # through arbitration while the real cue lay in a pocket — masking
        # the scratch. Frame check on every chronically-small NUMBERED
        # track shows real object balls do live at r9-11.5 in some table
        # regions (a purple 4, a yellow 1 — digits visible), so a general
        # radius floor would strip real identities. But the CUE is white
        # and maximally reflective: it never reads small (observed minimum
        # 12.2 vs the speck's lifetime 10.0). Number 0 therefore demands
        # lifetime full-size evidence, measured against this session's own
        # committed-ball population.
        pop = sorted(t.radius for t in tracks if t.committed_number >= 0)
        cue_floor = 0.9 * pop[len(pop) // 2] if len(pop) >= 3 else (
            0.85 * self._ball_r if self._ball_r > 2.0 else 0.0)
        for i, t in enumerate(tracks):
            if t.vacated:
                continue      # its spot was seen empty; it is not that ball now
            votes = Counter(n for n in t.num_hist if n is not None and n >= 0)
            colour_num, colour_frac = self._colour_consensus(t)
            nums = set(votes) | ({colour_num} if colour_num > 0 else set())
            if t.committed_number >= 0:
                nums.add(t.committed_number)
            for n in nums:
                if n == 0 and cue_floor and t.r_max < cue_floor:
                    continue   # never full-size in its life: not the cue
                if t.committed_number < 0 and votes.get(n, 0) < 3:
                    # COLOUR ADOPTION — the digit-down path. No vote majority
                    # will ever form for a ball whose number faces the felt,
                    # so a mature track may be named by colour alone, under
                    # rack-churn guards: at rest, long-lived, a strong and
                    # stable colour consensus, solids only (a stripe's band
                    # is CNN-readable from any orientation; solids 1-7 are
                    # the balls that go dark when the digit hides), and the
                    # number free (uniqueness via the same greedy pool).
                    if not (n == colour_num and colour_frac >= 0.8
                            and len(t.colour_hist) >= 25
                            and 1 <= n <= 7
                            and not t.moving() and t.hits >= 60):
                        continue
                s = float(votes.get(n, 0))
                if n == colour_num:
                    s += COLOUR_MAX
                if n == t.committed_number:
                    s += STICKY
                cands.append((s, i, n))

        # --- assignment: at-rest commitments first, then best-score greedy #
        taken: set[int] = set()
        assigned: dict[int, int] = {}
        no_rename: set[int] = set()
        ledger = getattr(self, "_num_last", {})

        def _reachable(t, n) -> bool:
            seen = ledger.get(n)
            if seen is None:
                return True
            df = self._frame_n - seen[0]
            if not 0 < df <= 12:
                return True
            reach = df * 0.35 * self._short_side + 3.0 * max(
                6.0, self._ball_r if hasattr(self, "_ball_r") else 12.0)
            return ((t.x - seen[1]) ** 2
                    + (t.y - seen[2]) ** 2) ** 0.5 <= reach

        # When two RESTING tracks contest one number, the winner used to be
        # whichever came first in the track list — rank, not evidence — so
        # three stray misreads on a neighbour could strip a published
        # colour-adopted identity at rest, permanently (review finding).
        # Order claimants by evidence for their own claim instead: recent
        # votes plus the measured-colour consensus agreeing.
        def _claim_strength(idx: int) -> float:
            t = tracks[idx]
            n = t.committed_number
            s = float(sum(1 for v in t.num_hist if v == n))
            cn, cf = self._colour_consensus(t)
            if cn == n and cf >= 0.8:
                s += COLOUR_MAX
            return s
        order = sorted((i for i, t in enumerate(tracks)
                        if rest[i] and t.committed_number >= 0 and not t.vacated),
                       key=_claim_strength, reverse=True)
        for i in order:
            t = tracks[i]
            if True:
                if (t.committed_number == 0 and cue_floor
                        and t.r_max < cue_floor):
                    # vote-path commit (_commit_number) landed number 0 on
                    # a track that has never once read full-size: the cue
                    # is white and never reads small — this is the felt
                    # speck, not the cue. Same floor as candidacy.
                    t.committed_number = -1
                    continue
                if (t.shown_number != t.committed_number
                        and not _reachable(t, t.committed_number)):
                    # never published this number and it just teleported
                    # here (first-commit on a garbage track): refuse, and
                    # drop the commitment so votes must re-earn it in place
                    t.committed_number = -1
                    continue
                if t.committed_number not in taken:
                    assigned[i] = t.committed_number
                    taken.add(t.committed_number)
                elif t.shown_number >= 0:
                    # Two resting tracks contest one number (a respawn next
                    # to its own lingering ghost, or a dark-trio double
                    # read). A loser that already PUBLISHED the number must
                    # show UNKNOWN, not be greedily renamed — renaming a
                    # resting published ball is exactly the impossible
                    # "#5 -> #9 while at rest" the corpus flags. A loser
                    # that never published stays fair game: naming it by
                    # measured colour is the "two sevens" fix, and a
                    # -1 -> n transition is never impossible.
                    no_rename.add(i)
        # REACHABILITY GATE (Joe's shot-36 family, measured at 2.41
        # hops/1k states library-wide): a number may only MOVE BETWEEN
        # tracks as fast as a ball travels. A candidate track farther from
        # the number's last published position than ~0.35 short-sides per
        # frame (a hard-struck ball) is an assignment jump, not motion —
        # rejected. Re-emergence after occlusion has a large frame gap, so
        # its reach is table-wide and unaffected.
        for s, i, n in sorted(cands, key=lambda c: -c[0]):
            if i in assigned or i in no_rename or n in taken or s < 2.0:
                continue
            t = tracks[i]
            if t.committed_number != n and not _reachable(t, n):
                continue
            assigned[i] = n
            taken.add(n)
        for i, t in enumerate(tracks):
            t.committed_number = assigned.get(i, -1)
        # refresh the ledger with what is now published
        for t in tracks:
            if t.committed_number >= 0:
                ledger[t.committed_number] = (self._frame_n, t.x, t.y)
        self._num_last = ledger

    @staticmethod
    def _colour_contradicts(t: _Internal, d: Detection) -> bool:
        """A settled ball's own colour cannot change. 005048 @233: Joe
        struck the purple 4; the arriving white CUE BALL came to rest
        against it, and the 4's settled track took that white detection
        28px away while the cue's own track sat 287px back at address,
        far outside its gate. The track then rode the cue for the rest of
        the shot and the 4's real path was never recorded — which is what
        put Joe's miss on the wrong side of the pocket.

        Distance cannot separate two touching balls, and the CLASS veto in
        update() deliberately exempts healthy tracks so one-frame class
        flicker never starves live tracking. Measured colour is the stable
        signal the class vote is not: white against purple is a BGR
        distance of ~300, far outside anything glare or blur does to a
        ball's own reading (measured: the 4's own detections score 1-14,
        the cue scores 108,820). Only a LARGE mismatch vetoes, and only
        for a settled track that has actually established a colour.
        """
        mb = getattr(d, "measured_bgr", None)
        if mb is None or not t.settled or len(t.mbgr_hist) < 8:
            return False
        # compare MEASURED to MEASURED. t.bgr is the classifier's palette
        # constant, so comparing against it vetoed a ball's own detections.
        ref = [sorted(c[i] for c in t.mbgr_hist)[len(t.mbgr_hist) // 2]
               for i in range(3)]
        return sum((float(a) - float(b)) ** 2
                   for a, b in zip(mb, ref, strict=False)) > _COLOUR_VETO_SQ

    @staticmethod
    def _colour_consensus(t: _Internal) -> tuple[int, float]:
        """(majority colour identity, fraction of samples agreeing) over the
        track's recent confident colour samples — the stable form of the old
        single-frame measured_identity(t.bgr) (one glare frame no longer
        swings the +COLOUR_MAX evidence). Falls back to the single current
        sample (frac 0.0, so it can never satisfy the adoption gate) until
        enough history exists."""
        from ..core.balls import measured_identity
        hist = [n for n in t.colour_hist if n > 0]
        if len(hist) >= 8:
            num, k = Counter(hist).most_common(1)[0]
            return num, k / len(t.colour_hist)
        m = measured_identity(tuple(int(v) for v in t.bgr))
        return (m if m > 0 else -1), 0.0

    @staticmethod
    def _colour_identity(t: _Internal, taken: set[int]) -> int:
        """Best FREE ball number for a track from its sampled colour, else -1.

        Hue names the base colour; the stripe bit comes from the track's class
        history (stripe = base + 8). Both variants are tried, class-preferred
        order, so a free identity is found even when the stripe read is stale.
        """
        import cv2
        import numpy as np

        from ..core.balls import _hue_to_base, measured_identity

        # This table's measured references first — they separate 4/7 by 71 Lab
        # units where canonical hues collapse them.
        m = measured_identity(tuple(int(v) for v in t.bgr), taken=taken)
        if m > 0:
            return m

        b, g, r = (int(v) for v in t.bgr)
        h, s, v = cv2.cvtColor(
            np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
        if s < 45 and v > 170:
            return -1          # white-ish: cue/9-pole; never guess from this
        base = _hue_to_base(float(h), float(v))
        if base <= 0:
            return -1
        # ONE candidate, matched to the track's class evidence — offering the
        # stripe variant as a fallback for a taken solid INVENTED balls 10-15
        # (out-of-game misreads doubled on real footage before this narrowed).
        is_stripe = t.committed_cls == BallClass.STRIPE or any(
            c == BallClass.STRIPE for c in list(t.cls_hist)[-5:])
        cand = base + 8 if is_stripe else base
        return cand if 1 <= cand <= 15 and cand not in taken else -1
