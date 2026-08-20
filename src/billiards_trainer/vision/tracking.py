"""Multi-object tracker for balls.

A compact ByteTrack-style tracker: greedy distance association with gating, a
constant-velocity motion model so tracks coast through brief occlusions, and a
hit/miss lifecycle for stable IDs. Class labels are majority-voted over a short
history so cue/solid/stripe doesn't flicker frame to frame.

No scipy/lap dependency — greedy nearest-neighbour with a distance gate is plenty
for <=16 near-rigid objects and keeps the installer lean.
"""

import math
from collections import Counter, deque
from dataclasses import dataclass, field

from .types import BallClass, Detection, Track


@dataclass
class _Internal:
    id: int
    x: float
    y: float
    radius: float
    vx: float = 0.0
    vy: float = 0.0
    age: int = 0
    hits: int = 0
    misses: int = 0
    confirmed: bool = False
    bgr: tuple = (200, 200, 200)
    still_count: int = 0       # consecutive ~stationary frames
    settled: bool = False      # confirmed AND has been still a while (a resting ball)
    move_streak: int = 0       # consecutive matched frames with real PUBLISHED motion
    shown_number: int = -1     # the number last actually PUBLISHED for this track
    committed_number: int = -1  # hysteresis: the held identity (resists flicker)
    committed_cls: BallClass = BallClass.UNKNOWN  # sticky class for unnumbered balls
    cls_hist: deque = field(default_factory=lambda: deque(maxlen=15))
    num_hist: deque = field(default_factory=lambda: deque(maxlen=45))
    # Measured-colour identity samples (nearest Lab reference per confident
    # frame). A ball whose digit faces the FELT never lands a number vote its
    # whole life — the 6 went unnamed for a full session — but its colour
    # names it from any orientation. Arbitration adopts a mature consensus.
    colour_hist: deque = field(default_factory=lambda: deque(maxlen=40))
    pos_hist: deque = field(default_factory=lambda: deque(maxlen=64))

    def predict(self) -> None:
        self.x += self.vx
        self.y += self.vy
        self.age += 1

    def moving(self) -> bool:
        """Demonstrably in motion RIGHT NOW — the only state in which the
        physics scorer does not treat an identity change as impossible.

        Judged from the PUBLISHED per-frame step (move_streak, kept by
        _apply_match), because that is exactly what the scorer measures. A
        settled-bit reset, a one-frame glare excursion, a post-revival
        smoothing glide, a creeping sub-threshold drift: all of those stay
        below the streak bar, so identity stays frozen through every window
        that used to leak "changed #5 -> #9 while at rest" onto the corpus.
        Three consecutive real steps mean the ball was actually struck."""
        return self.move_streak >= 3

    @property
    def cls(self) -> BallClass:
        # Identity follows the (stickier) number vote so a few bad frames during a
        # shot — motion blur misreading the cue — don't flip its class. Falls back
        # to the class-history vote until a number is established. The number-
        # derived class is REMEMBERED (committed_cls) so that losing the number
        # to arbitration for a few contested frames does not flip the published
        # class back to a stale colour vote ("eight -> solid at rest").
        n = self.number
        if n >= 0:
            if n == 0:
                self.committed_cls = BallClass.CUE
            elif n == 8:
                self.committed_cls = BallClass.EIGHT
            elif n <= 7:
                self.committed_cls = BallClass.SOLID
            else:
                self.committed_cls = BallClass.STRIPE
            return self.committed_cls
        if not self.cls_hist:
            return self.committed_cls
        vote = Counter(self.cls_hist).most_common(1)[0][0]
        # Class is sticky too: a near-50/50 solid<->unknown vote must not strobe
        # the rendered ball. Adopt a non-UNKNOWN majority; keep it until another
        # non-UNKNOWN majority replaces it — and NEVER while the ball is at
        # rest (a resting ball cannot change class any more than number; the
        # corpus flagged "solid -> cue at rest" from exactly this vote drift).
        if vote != BallClass.UNKNOWN and (
                self.committed_cls == BallClass.UNKNOWN or self.moving()):
            self.committed_cls = vote
        return self.committed_cls

    @property
    def number(self) -> int:
        # The HELD identity, with hysteresis (updated in _apply_match). Resists the
        # frame-to-frame flicker (1->3->5) that pure per-frame voting shows when a
        # ball's measured colour wanders across hue boundaries.
        return self.committed_number

    def _commit_number(self) -> None:
        """Update the held identity: adopt the windowed-vote winner only when it
        clearly dominates the current one, so identity is sticky, not twitchy."""
        votes = [n for n in self.num_hist if n is not None and n >= 0]
        if not votes:
            return
        cnt = Counter(votes)
        top, topn = cnt.most_common(1)[0]
        if self.committed_number < 0:
            # FIRST commit: wait for a formed majority (3 agreeing reads).
            # The old spawn-read instant commit was the churn the corpus
            # flagged at rack time: commit the first wobbly read, then "flip
            # at rest" to the real one five votes later. Invisible in the UI
            # — a track publishes at min_hits=5, by which point a consistent
            # read has 3 votes. (-1 -> n transitions are never impossible.)
            if topn >= 3:
                self.committed_number = top
        elif top == self.committed_number:
            pass
        elif topn >= cnt.get(self.committed_number, 0) + 5:
            # A ball at rest cannot become a different ball — physics, not
            # preference. Unless the ball is in demonstrable motion, identity
            # is FROZEN: challenger votes (glare, a neighbour's edge in the
            # crop) accumulate but cannot flip it. The re-vote rides the
            # motion itself — stale votes are dropped when a motion episode
            # starts, so the post-strike reads decide it.
            if self.moving():
                self.committed_number = top   # challenger clearly won — switch


class BallTracker:
    # Defaults promoted from the 2026-07-02 autotune sweep (45 trials scored by
    # tools/eval_tracking.py physics metrics, winners re-verified on full
    # segments WITH the recall-aware score + human-verified ball counts):
    # settled jitter 0.074 -> 0.049 px, phantom churn down, identity flips
    # 2.4 -> ~2.1/track-min, recall identical to the previous defaults.
    def __init__(self, max_dist_frac: float = 0.16, max_misses: int = 24,
                 min_hits: int = 5, vel_alpha: float = 0.42,
                 pos_alpha_slow: float = 0.20, pos_alpha_fast: float = 0.72,
                 speed_lo: float = 3.0, speed_hi: float = 6.0,
                 still_speed_frac: float = 0.014, still_frames: int = 8,
                 lock_dist_frac: float = 0.0115, occluded_budget: int = 1800):
        self.max_dist_frac = max_dist_frac
        self.max_misses = max_misses
        self.min_hits = min_hits
        self.vel_alpha = vel_alpha
        # Adaptive position smoothing: heavy (slow alpha) when a ball is ~still so
        # the bird's-eye doesn't jitter while Joe tunes, light (fast alpha) when it
        # moves so a struck cue ball is FOLLOWED, not trailed by a foot. Alpha is
        # interpolated between slow/fast over [speed_lo, speed_hi] px/frame.
        self.pos_alpha_slow = pos_alpha_slow
        self.pos_alpha_fast = pos_alpha_fast
        self.speed_lo = speed_lo
        self.speed_hi = speed_hi
        # Stationary handling. A ball moving slower than still_speed (as a fraction
        # of the table short side) for still_frames becomes "settled". A settled
        # ball's position is LOCKED — detections within lock_dist of it don't move
        # it — which kills the few-pixel shimmer on resting balls. And a settled
        # ball that then vanishes from detection is treated as OCCLUDED (a hand/cue
        # over it), not gone: it survives occluded_budget frames instead of the
        # short max_misses, so it stops flickering in and out. A ball that vanishes
        # while MOVING is treated as pocketed/removed and ages out fast.
        self.still_speed_frac = still_speed_frac
        self.still_frames = still_frames
        self.lock_dist_frac = lock_dist_frac
        self.occluded_budget = occluded_budget
        self._short_side = 400.0
        self._ball_r = 0.0
        self._tracks: list[_Internal] = []
        self._next_id = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    # ------------------------------------------------------------------ #
    def remove_ids(self, ids) -> None:
        """Hard-drop tracks by id — the pipeline's vacancy pruning calls
        this when a still track's spot is plainly VISIBLE and EMPTY (no
        detection, no hand covering it): the ball is gone, and coasting on
        the occlusion budget would park a ghost at its old position for
        minutes (Joe: "lingering cue ball assumed positions")."""
        ids = set(ids)
        self._tracks = [t for t in self._tracks if t.id not in ids]

    def update(self, detections: list[Detection], short_side: float,
               bounds: tuple[float, float, float, float] | None = None,
               pockets: list[tuple[float, float]] | None = None,
               pocket_r: float = 0.0,
               ball_r: float = 0.0) -> list[Track]:
        self._short_side = max(1.0, short_side)
        self._frame_n = getattr(self, "_frame_n", 0) + 1
        # Table-wide expected ball radius: the FLOOR for the motion-streak
        # bar. A shrunken ghost's own radius would put its bar below the
        # scorer's rest threshold, silently re-opening the at-rest identity
        # window for exactly the degenerate tracks most prone to misreads.
        self._ball_r = float(ball_r)
        # Velocity-aware gate: a struck ball can cross more than the static gate
        # in one frame; letting the gate grow with track speed keeps the SAME
        # track attached through the fast phase instead of coasting a stale one.
        gate = max(8.0, self.max_dist_frac * short_side)

        for t in self._tracks:
            t.predict()
        # Cushion-aware coasting: reflect predicted positions off the nose lines
        # (bounds = playing-area rect). Motion blur drops detection exactly when
        # a ball is fastest — right at a rail impact — so without this the
        # coasted track either sails through the cushion or dies short of it and
        # the animation shows the ball "rebounding in thin air". Reflecting the
        # prediction makes the coasted ball bounce WHERE the real ball bounces,
        # which also puts it next to the reappearing detection for re-matching.
        if bounds is not None:
            bx0, by0, bx1, by1 = bounds
            capture_sq = (1.6 * pocket_r) ** 2 if pockets and pocket_r > 0 else 0.0
            for t in self._tracks:
                # NO cushion at a pocket mouth: a ball coasting there must sail
                # INTO the pocket (and age out), never bounce back onto the
                # felt — that phantom rebound was visible in the animation.
                if capture_sq and any((t.x - px) ** 2 + (t.y - py) ** 2 <= capture_sq
                                      for px, py in pockets):
                    continue
                r = max(2.0, t.radius)
                lo, hi = bx0 + r, bx1 - r
                if lo < hi:
                    if t.x < lo:
                        t.x, t.vx = min(hi, 2 * lo - t.x), abs(t.vx)
                    elif t.x > hi:
                        t.x, t.vx = max(lo, 2 * hi - t.x), -abs(t.vx)
                lo, hi = by0 + r, by1 - r
                if lo < hi:
                    if t.y < lo:
                        t.y, t.vy = min(hi, 2 * lo - t.y), abs(t.vy)
                    elif t.y > hi:
                        t.y, t.vy = max(lo, 2 * hi - t.y), -abs(t.vy)

        unmatched_dets = set(range(len(detections)))
        # Build all (track, det) pairs within the gate, then assign greedily. The
        # gate WIDENS with a track's speed so a hard-struck ball — which jumps far
        # in one frame — can still match its detection instead of being left behind.
        #
        # APPEARANCE GATING (005647 forensics): identity must survive contact.
        # On a stop shot the cue parks in the object ball's vacated spot, and
        # closest-first handed the WHITE detection to the settled num-3 track
        # by a 2px margin over the cue's own track — inverting a make into a
        # scratch. So category contradiction (cue vs object) COSTS: agreeing
        # pairs always sort first, and a contradicting pair only wins when
        # nothing that agrees is in reach. Harder still: a COASTING numbered
        # ball cannot change category while unseen — the maroon 7, coasting at
        # a pocket, adopted the arriving white cue and followed it in, booking
        # the cue's pocket entry as the 7's departure. Healthy tracks
        # (misses == 0) are exempt from the veto so one-frame class flicker
        # never starves live tracking.
        def _contradicts(t, d) -> bool:
            tc = t.cls
            return (d.cls != BallClass.UNKNOWN and tc != BallClass.UNKNOWN
                    and (tc == BallClass.CUE) != (d.cls == BallClass.CUE))

        cls_pen = 0.30 * self._short_side
        pairs = []
        for ti, t in enumerate(self._tracks):
            t_speed = (t.vx * t.vx + t.vy * t.vy) ** 0.5
            t_gate = max(gate, 3.0 * t_speed)
            # A resting (settled) ball can only move by being STRUCK, and a hard hit
            # jumps it far in one frame — so give settled tracks an expanded strike
            # gate. This keeps a struck object ball ON its track (then it coasts on
            # its velocity into the pocket) instead of the track being lost and the
            # ball vanishing off the overhead. (greedy closest-first limits mismatch)
            if t.settled:
                t_gate = max(t_gate, 0.18 * self._short_side)
            for di in unmatched_dets:
                d = detections[di]
                dist = math.hypot(t.x - d.x, t.y - d.y)
                if dist <= t_gate:
                    contra = t.confirmed and _contradicts(t, d)
                    if contra and t.committed_number >= 0 and t.misses >= 1:
                        continue
                    pairs.append((dist + (cls_pen if contra else 0.0), ti, di))
        pairs.sort(key=lambda p: p[0])

        matched_tracks = set()
        matched_dets = set()
        for _dist, ti, di in pairs:
            if ti in matched_tracks or di in matched_dets:
                continue
            matched_tracks.add(ti)
            matched_dets.add(di)
            self._apply_match(self._tracks[ti], detections[di])

        # Single-cue rebind: there is exactly ONE cue ball, so if its track went
        # unmatched (a hard strike jumped it past the gate, or motion blur dropped a
        # couple of frames) but a cue detection exists, bind them regardless of
        # distance — the cue FOLLOWS the strike instead of lagging or spawning a new
        # id a few feet away. (Joe's #1: the cue must track fast hits.)
        cue_ti = [ti for ti, t in enumerate(self._tracks)
                  if ti not in matched_tracks and t.confirmed and t.cls == BallClass.CUE]
        cue_di = [di for di in range(len(detections))
                  if di not in matched_dets and detections[di].cls == BallClass.CUE]
        if len(cue_ti) == 1 and len(cue_di) == 1:
            ti, di = cue_ti[0], cue_di[0]
            t, d = self._tracks[ti], detections[di]
            # Radius sanity (005647 @209s): the lost cue snapped 560px onto a
            # sub-ball white SPECK on the felt (r~10 vs the cue's 15-17) and
            # sat there while the real cue lay in a pocket — masking the
            # scratch. Blur INFLATES a fast ball's blob, never shrinks it
            # 30%: bind at any distance, but only to a cue-sized detection.
            if 0.7 * t.radius <= d.radius <= 1.5 * t.radius:
                matched_tracks.add(ti)
                matched_dets.add(di)
                self._apply_match(t, d)

        # Second-chance revival: a fast ball motion-blurs out of detection and
        # reappears FEET away a few frames later — far beyond the primary gate.
        # Without this it spawns a brand-new track while the old one holds its
        # last position: the "several copies of the same ball" ghost. Re-bind
        # unmatched detections to lost confirmed tracks with a gate that grows
        # the longer the track has been missing, and SNAP across the gap (the
        # intermediate path was never observed — smoothing it in would whip the
        # velocity estimate).
        lost_tis = [ti for ti, t in enumerate(self._tracks)
                    if ti not in matched_tracks and t.confirmed and t.misses >= 1]
        if lost_tis and len(matched_dets) < len(detections):
            revive = []
            for ti in lost_tis:
                t = self._tracks[ti]
                r_gate = min(0.45 * self._short_side,
                             gate + 0.06 * self._short_side * t.misses)
                for di in range(len(detections)):
                    if di in matched_dets:
                        continue
                    d = detections[di]
                    # A ball cannot change category (cue vs object) while
                    # unseen — reviving across one is how identities leak
                    # between colliding balls (005647 forensics).
                    if _contradicts(t, d):
                        continue
                    dist = math.hypot(t.x - d.x, t.y - d.y)
                    if dist <= r_gate:
                        revive.append((dist, ti, di))
            revive.sort(key=lambda p: p[0])
            for dist, ti, di in revive:
                if ti in matched_tracks or di in matched_dets:
                    continue
                matched_tracks.add(ti)
                matched_dets.add(di)
                t = self._tracks[ti]
                if dist > gate:
                    d = detections[di]
                    t.x, t.y = d.x, d.y
                    t.vx = t.vy = 0.0
                    t.settled = False
                    t.still_count = 0
                    # The identity rides the snap (revival exists for "the
                    # same ball reappearing"), but trim the history so votes
                    # read at the OLD spot cannot dominate a later re-vote.
                    for hist in (t.num_hist, t.cls_hist, t.colour_hist):
                        recent = list(hist)[-3:]
                        hist.clear()
                        hist.extend(recent)
                self._apply_match(t, detections[di])

        # Unmatched tracks -> coast / age out
        for ti, t in enumerate(self._tracks):
            if ti in matched_tracks:
                continue
            t.misses += 1
            # Mild rolling-friction damping while coasting: a blurred-out ball is
            # still rolling at nearly full speed, so the coast must carry it (into
            # the cushion reflection above), not kill it in a few frames. Drift is
            # bounded — a moving track ages out after max_misses anyway.
            t.vx *= 0.92
            t.vy *= 0.92

        # Unmatched detections -> new tentative tracks
        for di, d in enumerate(detections):
            if di in matched_dets:
                continue
            self._spawn(d)

        # Merge duplicates: two tracks closer than a ball-diameter are physically
        # the SAME ball — a fast direction change (rail bounce / recoil) briefly
        # overshot the old track and spawned a ghost beside it. Keep the one matched
        # this frame (it sits on the real detection); drop the stale coaster. This
        # kills the "two cue balls / duplicate ball" artefact on bounces.
        if len(self._tracks) > 1:
            # Align the duplicate threshold with physics: two tracks closer
            # than 0.8 ball diameters are impossible, so merge them. The old
            # 0.035*short fraction worked out to 0.776 diameters — leaving a
            # 2.4% sliver where a starved duplicate could ride the settled
            # occlusion budget for a minute, parked inside a real ball (the
            # dedupe already prefers the detection-backed track of a pair).
            merge_dist = max(0.035 * self._short_side, 1.6 * ball_r)
            order = sorted(range(len(self._tracks)),
                           key=lambda i: (self._tracks[i].misses == 0, self._tracks[i].hits),
                           reverse=True)
            keep_idx: list[int] = []
            for i in order:
                ti = self._tracks[i]
                if any(math.hypot(self._tracks[j].x - ti.x, self._tracks[j].y - ti.y) < merge_dist
                       for j in keep_idx):
                    continue
                keep_idx.append(i)
            self._tracks = [self._tracks[i] for i in keep_idx]

        # Velocity-aware keep-alive: a ball that vanished while SETTLED (resting) is
        # almost certainly occluded (a hand/cue over it) — keep it for a long budget
        # so it doesn't flicker. A ball that vanished while MOVING was pocketed or
        # picked up — let it age out fast. EXCEPT near a pocket: a slow-rolled pot
        # settles at the jaw for a beat before dropping, and the occlusion budget
        # then parks a ghost "on the table" for a minute (Joe watched it live).
        # Nothing occludes a ball inside a pocket mouth — settled + vanished
        # there = potted, short budget.
        if pockets and pocket_r > 0:
            cap_sq2 = (1.8 * pocket_r) ** 2
            def _at_pocket(t):
                return any((t.x - px) ** 2 + (t.y - py) ** 2 <= cap_sq2
                           for px, py in pockets)
        else:
            def _at_pocket(t):
                return False
        # The long occlusion budget exists so an IDENTIFIED resting ball
        # survives a hand/cue over it. A vanished track that never earned an
        # identity is a glare/shadow blob, not a ball — since deferred
        # commits stopped handing those instant numbers (which the stale-
        # ghost cleanup used to reap them by), deny them the long budget or
        # they linger for a minute as unknown phantoms.
        # DEATH REGISTRY for rest-linking: a numbered ball whose track dies
        # while (recently) in motion is a FLYER the camera lost to blur —
        # its identity should survive to wherever it lands (Joe's shot-36:
        # the struck 7 lost mid-roll). Record the loss; a lone new track
        # settling soon after inherits it under strict uniqueness.
        deaths = getattr(self, "_flyer_deaths", [])
        keep = []
        for t in self._tracks:
            limit = (self.occluded_budget
                     if t.settled and t.committed_number >= 0
                     and not _at_pocket(t) else self.max_misses)
            if t.misses <= limit:
                keep.append(t)
            elif (t.committed_number >= 0 and not _at_pocket(t)
                  and (t.move_streak > 0 or not t.settled)):
                deaths.append({"n": t.committed_number,
                               "frame": self._frame_n,
                               "x": t.x, "y": t.y})
        self._tracks = keep
        self._flyer_deaths = [d for d in deaths
                              if self._frame_n - d["frame"] <= 30]
        # A track that came to rest PAST the cushion line inside a pocket zone
        # is a ball lying in the basket — pocketed, done. Without this the
        # occlusion budget keeps it as a ghost for another minute after the
        # ingestion filter stops feeding it (and before that filter existed, a
        # basket pair sat as settled tracks for 40+ s firing impossible-overlap
        # violations every frame).
        if bounds is not None and pockets and pocket_r > 0:
            bx0, by0, bx1, by1 = bounds
            cap_sq = (1.4 * pocket_r) ** 2
            self._tracks = [
                t for t in self._tracks
                if (bx0 <= t.x <= bx1 and by0 <= t.y <= by1)
                or not any((t.x - px) ** 2 + (t.y - py) ** 2 <= cap_sq
                           for px, py in pockets)]
        # REST-LINKING: exactly one fresh unnumbered settled track + exactly
        # one recent flyer death whose number nobody holds -> inherit. Any
        # ambiguity (two orphans, two deaths, number re-acquired) -> skip;
        # wrong links are worse than anonymous balls.
        deaths = getattr(self, "_flyer_deaths", [])
        if deaths:
            held = {t.committed_number for t in self._tracks
                    if t.committed_number >= 0}
            free_deaths = [d for d in deaths if d["n"] not in held]
            orphans = [t for t in self._tracks
                       if t.confirmed and t.settled
                       and t.committed_number < 0
                       and self._frame_n - getattr(t, "born_frame", 0) <= 45]
            if len(free_deaths) == 1 and len(orphans) == 1:
                d = free_deaths[0]
                t = orphans[0]
                t.committed_number = d["n"]
                ledger = getattr(self, "_num_last", {})
                ledger[d["n"]] = (self._frame_n, t.x, t.y)
                self._num_last = ledger
                self._flyer_deaths = [x for x in deaths if x is not d]
        self._arbitrate_numbers()
        return self._public()

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
        from .balls import measured_identity

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
        for i, t in enumerate(tracks):
            votes = Counter(n for n in t.num_hist if n is not None and n >= 0)
            colour_num, colour_frac = self._colour_consensus(t)
            nums = set(votes) | ({colour_num} if colour_num > 0 else set())
            if t.committed_number >= 0:
                nums.add(t.committed_number)
            for n in nums:
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
                        if rest[i] and t.committed_number >= 0),
                       key=_claim_strength, reverse=True)
        for i in order:
            t = tracks[i]
            if True:
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
    def _colour_consensus(t: _Internal) -> tuple[int, float]:
        """(majority colour identity, fraction of samples agreeing) over the
        track's recent confident colour samples — the stable form of the old
        single-frame measured_identity(t.bgr) (one glare frame no longer
        swings the +COLOUR_MAX evidence). Falls back to the single current
        sample (frac 0.0, so it can never satisfy the adoption gate) until
        enough history exists."""
        from .balls import measured_identity
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

        from .balls import _hue_to_base, measured_identity

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

    def _apply_match(self, t: _Internal, d: Detection) -> None:
        # Anchor for the published per-frame step: the last position handed
        # out, NOT the post-predict() one — prediction already carries the
        # velocity, so anchoring after it would hide steady rolling motion
        # (a perfectly-tracked ball would look stationary to the streak).
        prev_pub = t.pos_hist[-1] if t.pos_hist else (t.x, t.y)
        meas_vx = d.x - t.x
        meas_vy = d.y - t.y
        spd = (meas_vx * meas_vx + meas_vy * meas_vy) ** 0.5
        t.vx = self.vel_alpha * meas_vx + (1 - self.vel_alpha) * t.vx
        t.vy = self.vel_alpha * meas_vy + (1 - self.vel_alpha) * t.vy
        if spd < self.speed_lo:
            # a ~still ball shouldn't carry velocity, or predict() injects jitter
            t.vx *= 0.25
            t.vy *= 0.25
        # Stillness bookkeeping: count consecutive ~stationary frames; once a
        # confirmed ball has held still long enough it is "settled".
        still_speed = self.still_speed_frac * self._short_side
        if spd < still_speed:
            t.still_count += 1
        else:
            t.still_count = 0
        t.settled = t.confirmed and t.still_count >= self.still_frames
        # Position update. A SETTLED ball is LOCKED: a detection within lock_dist
        # of it does NOT move it, so resting balls stop shimmering by a few pixels.
        # Once a detection lands beyond lock_dist the ball has really moved — unlock
        # and follow it.
        lock_dist = max(2.0, self.lock_dist_frac * self._short_side)
        if t.settled and spd <= lock_dist:
            pass  # frozen — kills sub-pixel-to-few-pixel jitter on a resting ball
        else:
            if spd > lock_dist:
                t.settled = False
                t.still_count = 0
            frac = (spd - self.speed_lo) / max(1e-6, self.speed_hi - self.speed_lo)
            frac = max(0.0, min(1.0, frac))
            pos_a = self.pos_alpha_slow + (self.pos_alpha_fast - self.pos_alpha_slow) * frac
            t.x = pos_a * d.x + (1 - pos_a) * t.x
            t.y = pos_a * d.y + (1 - pos_a) * t.y
        # Radius: once a track is established, reject wild size jumps (sensor-noise
        # outliers) and smooth slowly, so a held ball stops "pumping" in size.
        if t.confirmed and t.radius > 0 and abs(d.radius - t.radius) > 0.35 * t.radius:
            pass  # keep the stable estimate; this frame's size is an outlier
        else:
            a = 0.2 if t.confirmed else 0.5
            t.radius = a * d.radius + (1 - a) * t.radius
        # Colour is STICKY: an UNKNOWN crop (occlusion, glare, a bad sample)
        # must not paint an identified ball grey for a frame — that read as
        # "balls flickering between colour and grey" live. Keep the last
        # confidently-sampled colour until a new confident one arrives.
        if d.cls != BallClass.UNKNOWN:
            t.bgr = d.bgr
        # Colour evidence comes ONLY from measured pixels. Detection.bgr is
        # usually a canonical palette constant, so sampling it built a
        # consensus that merely ECHOED the classifier's guess — a ball
        # misread 7 once every 23 frames adopted 7 by "colour" eventually
        # (review finding). measured_bgr is set only by code that actually
        # sampled the frame.
        mb = getattr(d, "measured_bgr", None)
        if mb is not None:
            from .balls import measured_identity
            t.colour_hist.append(
                measured_identity(tuple(int(v) for v in mb)))
        t.cls_hist.append(d.cls)
        t.num_hist.append(d.number)
        t.pos_hist.append((t.x, t.y))
        # Motion-episode bookkeeping from the PUBLISHED step — the quantity
        # the physics scorer measures. The bar sits ABOVE the scorer's rest
        # threshold (0.12 diam), so identity can only ever change on frames
        # the scorer itself must call motion. One glare blip yields at most
        # two big published steps (the excursion and the ease-back) — never
        # three — so it cannot open the freeze. At the third real step the
        # ball was genuinely struck: drop the stale rest-time vote pile so
        # the RE-VOTE the freeze defers is decided by what the model reads
        # from here on, not outvoted by a minute of history at the old spot.
        pub_step = math.hypot(t.x - prev_pub[0], t.y - prev_pub[1])
        if pub_step > 0.30 * max(6.0, t.radius, self._ball_r):
            t.move_streak += 1
            if t.move_streak == 3:
                for hist in (t.num_hist, t.cls_hist, t.colour_hist):
                    recent = list(hist)[-3:]
                    hist.clear()
                    hist.extend(recent)
        else:
            t.move_streak = 0
        t._commit_number()
        t.hits += 1
        t.misses = 0
        if t.hits >= self.min_hits:
            t.confirmed = True

    def _spawn(self, d: Detection) -> None:
        t = _Internal(id=self._next_id, x=d.x, y=d.y, radius=d.radius, bgr=d.bgr)
        t.born_frame = getattr(self, "_frame_n", 0)
        t.cls_hist.append(d.cls)
        t.num_hist.append(d.number)
        # NO instant identity commit: the first read is the least reliable one
        # (rack-time glare, a half-occluded crop). _commit_number names the
        # ball once 3 reads agree — before the track ever publishes.
        t.pos_hist.append((d.x, d.y))
        t.hits = 1
        self._next_id += 1
        self._tracks.append(t)

    def _public(self) -> list[Track]:
        # One cue ball per table. Number arbitration alone doesn't guarantee it:
        # a track stripped of number 0 falls back to its class-vote history,
        # which is still full of CUE votes, so two tracks can both RENDER as
        # cue. Rank cue claimants by evidence; everyone else shows as unknown.
        claimants = [t for t in self._tracks if t.confirmed and t.cls == BallClass.CUE]
        cue_id = None
        if claimants:
            cue_id = max(claimants, key=lambda t: (
                t.committed_number == 0,
                sum(1 for c in t.cls_hist if c == BallClass.CUE),
                t.hits, -t.misses)).id
        out = []
        for t in self._tracks:
            if not t.confirmed:
                continue
            cls, num = t.cls, t.number
            if cls == BallClass.CUE and t.id != cue_id:
                cls, num = BallClass.UNKNOWN, -1
            t.shown_number = num
            tr = Track(
                id=t.id, x=t.x, y=t.y, radius=t.radius, vx=t.vx, vy=t.vy,
                cls=cls, number=num, bgr=t.bgr, age=t.age, hits=t.hits,
                misses=t.misses, active=(t.misses == 0), history=list(t.pos_hist),
            )
            out.append(tr)
        return out

    @property
    def tracks(self) -> list[Track]:
        return self._public()
