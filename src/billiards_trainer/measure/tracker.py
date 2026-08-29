"""M1 motion-model tracker: coast through blur instead of freezing.

The live tracker's rest-frozen identity was bought by real incidents
(duplicate identities, physics-impossible teleports) — this tracker
re-implements those RULES with a motion model:

  - exclusive assignment: one detection feeds at most one track, one
    track eats at most one detection (greedy nearest by PREDICTED
    position — no duplicate identities by construction)
  - gated association: a match beyond the prediction gate is a new
    track, never a teleport (protects the physics-impossible metric)
  - coasting: an unmatched track predicts forward under rolling
    friction for up to COAST_S — a motion-blurred takeoff keeps its
    identity and its (estimated) positions instead of freezing at rest
  - identity: ball number = recent-majority vote over detections, so a
    single misread never flips a track's number

Pure logic, no I/O — the engine feeds detections, this returns state
rows for the dense sidecar.
"""

from __future__ import annotations

from dataclasses import dataclass, field

FRICTION = 0.88          # per-0.1s velocity retention while coasting
COAST_S = 0.6            # coast this long without a detection, then inactive
GATE_R = 3.2             # association gate, in ball radii, around prediction
ACQUIRE_R = 8.0          # wider gate for a young/still track: a struck
                         # ball's first frames outrun the tight gate and
                         # used to fragment into one track per frame
VOTE_N = 9               # number votes considered for identity
HYST_K = 5               # frames a new number must lead before shown
REST_HYST_K = 45         # ...and this many while the ball is at REST,
                         # where flicker is likeliest and a correction
                         # rarest: ~1.5s of unanimous contrary reads.
                         # Not infinity, which is what it used to be.
FRESH_S = 2.5            # a number-read older than this carries no CLAIM
                         # weight in arbitration (identity majority still
                         # uses all votes). Bought 2026-08-28: Joe's
                         # standard 2-ball shot went trail-less because a
                         # saturated stale claim was mathematically
                         # unbeatable (9 capped votes vs "lead by 2") and
                         # the real ball could never take its number back.


@dataclass
class _Track:
    id: int
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    radius: float = 16.0
    t: float = 0.0
    t0: float = -1.0             # birth time (furniture age test)
    bx: float = 0.0              # birth position
    by: float = 0.0
    ever_moved: bool = False
    votes: list = field(default_factory=list)
    misses: float = 0.0          # seconds since last real detection
    active: bool = True
    emitted: int = -1            # the number this track SHOWS (hysteresis)
    pend: int = -1               # candidate awaiting confirmation
    pend_k: int = 0              # consecutive frames the candidate has led
    age_frames: int = 0          # matched frames (acquisition-gate window)
    last_v: float = 0.0          # RAW speed of the last match: the
                                 # smoothed vx/vy lag a fresh strike by
                                 # ~6x, so gating uses this instead
    ax: float = 0.0              # rest anchor (identity freezes at rest)
    ay: float = 0.0

    @property
    def number(self) -> int:
        """Majority over retained votes (votes are (number, t) pairs)."""
        if not self.votes:
            return -1
        counts: dict[int, int] = {}
        for v, _vt in self.votes:
            counts[v] = counts.get(v, 0) + 1
        return max(counts, key=lambda k: (counts[k],))

    def fresh_claim(self, n: int, now: float) -> int:
        """How many RECENT reads back this track's claim to number n."""
        return sum(1 for v, vt in self.votes
                   if v == n and now - vt <= FRESH_S)


MIN_ID_FRAMES = 3        # real sightings a track needs before it may SHOW
                         # a number at all. One detection is a guess, not
                         # an identity (round 36); a genuine ball passes
                         # this in a tenth of a second.
RETIRE_S = 3.0           # inactive this long AND last seen leaving the
                         # table = the track is DELETED, so it can neither
                         # match a new ball nor lend one a dead ball's
                         # name (round 30/31)
FURNITURE_S = 8.0        # a pocket-zone track this old that has NEVER
                         # moved is furniture (leather/shadow), not a ball


class MotionTracker:
    def __init__(self, pockets=None, pocket_r: float = 0.0):
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1
        self._holder: dict[int, int] = {}   # number -> incumbent track id
        # POCKET FURNITURE (bench round 6, vision-verified: a detection
        # on the bottom-left pocket's leather scored ABOVE the
        # confidence floor and lived the whole session, faking rests
        # and stealing episodes). Confidence can't separate it; TIME
        # can - furniture never moves, a jaw-hanging ball arrives and
        # leaves. Needs geometry, so the engine passes it in.
        self._pockets = list(pockets or [])
        self._pocket_r = float(pocket_r)

    def _left_the_table(self, tr) -> bool:
        """Did this track end where a ball leaves play - a pocket, or
        past the bed edge? Only then may its identity be retired; a ball
        that went out of sight mid-felt is occluded, not gone (the bridge
        hand closing over one at 154.2s costs the whole stroke if that
        distinction is not made)."""
        if not self._pockets:
            return True
        if self._pocket_r > 0 and any(
                ((tr.x - px) ** 2 + (tr.y - py) ** 2) ** 0.5 < 2.2 * self._pocket_r
                for px, py in self._pockets):
            return True
        xs = [p[0] for p in self._pockets]
        ys = [p[1] for p in self._pockets]
        pad = 2.0 * max(tr.radius, 8.0)
        return not (min(xs) - pad <= tr.x <= max(xs) + pad
                    and min(ys) - pad <= tr.y <= max(ys) + pad)

    def update(self, dets, t: float) -> list:
        """dets: iterable of (x, y, radius, number) for one frame at time
        t. Returns the frame's track rows (Track-like objects)."""
        # 1. predict every live track forward
        for tr in self._tracks.values():
            dt = max(0.0, t - tr.t)
            if dt > 0:
                damp = FRICTION ** (dt / 0.1)
                tr.x += tr.vx * dt
                tr.y += tr.vy * dt
                tr.vx *= damp
                tr.vy *= damp
        # 2. exclusive greedy association by predicted distance
        pairs = []
        for di, (dx, dy, dr, _dn) in enumerate(dets):
            for tr in self._tracks.values():
                # VELOCITY-AWARE GATE (bench round 8, vision-verified
                # 2026-08-28 - the single biggest defect found so far):
                # a struck ball covers ~60px per frame, but a track born
                # this frame has NO velocity, so a fixed 3.2-radius
                # (~35px) window missed its own next sighting and spawned
                # a fresh track EVERY frame. The overlay showed the cue
                # ball's path as nine simultaneous phantom balls with the
                # "cue" label stranded on the first one - fragmenting
                # identity, trails, counts and episodes all at once.
                # A young/still track therefore gets an ACQUISITION gate
                # wide enough for real ball speed; established tracks keep
                # the tight predicted gate plus their own travel.
                speed = max((tr.vx ** 2 + tr.vy ** 2) ** 0.5, tr.last_v)
                base = GATE_R * max(tr.radius, dr, 8.0)
                dt_g = max(0.0, t - tr.t)
                # no usable velocity (at rest, or just born) = nothing to
                # predict with, so the gate must cover a real first step
                if speed < 30.0:
                    gate = max(base, ACQUIRE_R * max(tr.radius, dr, 8.0))
                else:
                    gate = base + speed * dt_g
                dd = ((tr.x - dx) ** 2 + (tr.y - dy) ** 2) ** 0.5
                if dd <= gate:
                    pairs.append((dd, di, tr.id))
        pairs.sort()
        used_d: set = set()
        used_t: set = set()
        for _dd, di, tid in pairs:
            if di in used_d or tid in used_t:
                continue
            used_d.add(di)
            used_t.add(tid)
            tr = self._tracks[tid]
            dx, dy, dr, dn = dets[di]
            dt = max(1e-3, t - tr.t)
            if tr.misses == 0.0:            # velocity from real consecutive fixes
                a = min(1.0, dt / 0.2)      # smoothed, not raw single-step
                tr.vx = (1 - a) * tr.vx + a * (dx - tr.x + tr.vx * dt) / dt
                tr.vy = (1 - a) * tr.vy + a * (dy - tr.y + tr.vy * dt) / dt
            else:                           # re-acquired after coasting
                tr.vx = (dx - tr.x) / dt
                tr.vy = (dy - tr.y) / dt
            tr.last_v = (((dx - tr.x) ** 2 + (dy - tr.y) ** 2) ** 0.5) / dt
            tr.x, tr.y, tr.radius, tr.t = dx, dy, dr, t
            tr.misses = 0.0
            tr.active = True
            tr.age_frames += 1
            if dn >= 0:
                tr.votes.append((dn, t))
                del tr.votes[:-VOTE_N]
        # 3. unmatched detections found nothing in gate: new tracks
        for di, (dx, dy, dr, dn) in enumerate(dets):
            if di in used_d:
                continue
            tr = _Track(self._next_id, dx, dy, radius=dr, t=t,
                        t0=t, bx=dx, by=dy)
            if dn >= 0:
                tr.votes.append((dn, t))
            self._tracks[tr.id] = tr
            self._next_id += 1
        # 4. unmatched tracks coast; too long unseen -> inactive
        for tr in self._tracks.values():
            if tr.id in used_t or tr.t == t:
                continue
            tr.misses = t - tr.t if tr.misses == 0.0 else tr.misses + 0.0
            # COASTING INTO NOWHERE (round 15, pixels looked at first per
            # RULE 0): the "lost detections" on open felt were largely
            # not missed balls at all - they were predictions that drifted
            # off the ball, and in one case clean off the table onto the
            # floor. A prediction that leaves the bed is not a ball; stop
            # inventing motion for it.
            off_bed = False
            if self._pockets:
                xs = [p[0] for p in self._pockets]
                ys = [p[1] for p in self._pockets]
                pad = 2.0 * max(tr.radius, 8.0)
                off_bed = not (min(xs) - pad <= tr.x <= max(xs) + pad
                               and min(ys) - pad <= tr.y <= max(ys) + pad)
            if t - tr.t > COAST_S or off_bed:
                tr.active = False
                tr.vx = tr.vy = 0.0
            else:
                tr.misses = t - tr.t
        # 4b. TRACK MERGE (the live tracker's rigid-body rule at TRACK
        # level, learned from the gate: 7,494 overlapping-ball events =
        # one physical ball carried by two tracks - a coasted ghost
        # beside its re-acquired self). Two ACTIVE tracks overlapping
        # under 0.8 diameters: the weaker (fewer votes, then younger)
        # dies immediately.
        # 4c. POCKET FURNITURE: a track born in a pocket zone that has
        # never left its birth spot and is older than FURNITURE_S is
        # leather, not a ball. A real jaw-hanger either moves in (so it
        # was born elsewhere) or is knocked out (so it moves).
        if self._pockets and self._pocket_r > 0:
            for tr in self._tracks.values():
                if not tr.active or tr.t0 < 0:
                    continue
                # ONLY A REAL SIGHTING PROVES MOVEMENT (tr.t == t means
                # this track was matched to a detection this frame; a
                # coasting track keeps its last sighting time). Coasted
                # positions are predictions, and the pocket leather's
                # own coast drifted it ~20px off its birth spot, set
                # ever_moved, and thereby exempted itself from the
                # furniture rule that exists to kill it.
                if not tr.ever_moved and tr.t == t:
                    d0 = ((tr.x - tr.bx) ** 2 + (tr.y - tr.by) ** 2) ** 0.5
                    if d0 > max(tr.radius, 8.0):
                        tr.ever_moved = True
                if (not tr.ever_moved and t - tr.t0 > FURNITURE_S
                        and any(((tr.x - px) ** 2 + (tr.y - py) ** 2) ** 0.5
                                < 2.2 * self._pocket_r
                                for px, py in self._pockets)):
                    tr.active = False
                    tr.vx = tr.vy = 0.0
        # 4d. RETIREMENT - A NAME MUST NOT OUTLIVE ITS BALL. Going
        # inactive was never the end: step 2 associates against every
        # track in the dict with no liveness test, and the acquisition
        # gate widens with time unseen, so a long-dead track could claim
        # a brand-new ball anywhere on the felt and hand it the dead
        # ball's identity. Traced on the bench: id3 was genuinely the
        # yellow 1, followed it into the bottom-right pocket at 32.2s,
        # re-appeared across the table at 45.4s, then latched onto the
        # RED 3 at 109.2s - which answered to "1" for the rest of the
        # clip while every detection under it read 3. The same stale "1"
        # was what the static 9 wore whenever the real 1 was away.
        # ...and only for something that was ever actually a BALL. A
        # track that has never moved is furniture, and 4c above exists to
        # condemn it after FURNITURE_S. Retiring it first (at RETIRE_S)
        # deleted it before that verdict could be reached, so the pocket
        # leather was re-born as a fresh young track over and over; each
        # cycle of sighting -> coast -> re-snap reads as a ball moving at
        # 500-1700 px/s, and that fake motion opened the 154.2 episode
        # 6 seconds early. Never-moved tracks are furniture's business,
        # not retirement's.
        for tid in [k for k, tr in self._tracks.items()
                    if not tr.active and t - tr.t > RETIRE_S
                    and tr.ever_moved and self._left_the_table(tr)]:
            del self._tracks[tid]
            for num, holder in list(self._holder.items()):
                if holder == tid:
                    del self._holder[num]      # its claim dies with it
        live = [tr for tr in self._tracks.values() if tr.active]
        for i, a in enumerate(live):
            for b in live[i + 1:]:
                if not (a.active and b.active):
                    continue
                d = ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5
                if d < 0.8 * (a.radius + b.radius):
                    weaker = min(a, b, key=lambda tr: (len(tr.votes), tr.id))
                    weaker.active = False
                    weaker.vx = weaker.vy = 0.0
        # 5. NUMBER ARBITRATION (the live system's zero-duplicate rule,
        # re-learned the hard way: the first marathon run emitted the same
        # number on two tracks in 85% of frames). Each number lives on at
        # most ONE active track - the one with the strongest claim (vote
        # count, then recency); every other claimant emits unnumbered.
        claims: dict[int, _Track] = {}
        for tr in self._tracks.values():
            if not tr.active:
                continue
            n = tr.number
            if n < 0:
                continue
            cur = claims.get(n)
            if cur is None:
                claims[n] = tr
                continue
            # STICKY arbitration (gate: 2,032 id-flickers from per-frame
            # winner oscillation): the incumbent holds the number unless
            # the challenger leads by MARGIN clear votes.
            inc = cur if self._holder.get(n) == cur.id else (
                tr if self._holder.get(n) == tr.id else None)
            if inc is None:
                if ((tr.fresh_claim(n, t), tr.t)
                        > (cur.fresh_claim(n, t), cur.t)):
                    claims[n] = tr
            else:
                # FRESH claims only (the deadlock fix): a saturated but
                # STALE incumbent no longer outvotes a live challenger -
                # the ball actually being read as n takes the number
                # within ~FRESH_S. Two tracks BOTH freshly read as n
                # (true ambiguity) still favor the incumbent (sticky).
                chal = tr if inc is cur else cur
                if chal.fresh_claim(n, t) >= inc.fresh_claim(n, t) + 2:
                    claims[n] = chal
                else:
                    claims[n] = inc
        for n, tr in claims.items():
            self._holder[n] = tr.id
        # 6. EMIT HYSTERESIS (gate round 2: 163 residual id_flickers were
        # vote majorities oscillating between two real numbers at rest).
        # A track's SHOWN number changes n->m only after the new number
        # has led for HYST_K consecutive frames; dropping to unnumbered
        # (arbitration loss) stays immediate - duplicate prevention
        # outranks flicker prevention.
        emitted_claims: dict[int, int] = {}
        out = []
        for tr in self._tracks.values():
            if not tr.active:
                continue
            cand = tr.number
            if cand >= 0 and claims.get(cand) is not tr:
                cand = -1                   # arbitration loser
            # ONE SIGHTING IS NOT AN IDENTITY. Below, a track's FIRST
            # number is shown with no delay - right for a real ball,
            # which should not wait five frames to be named, but it also
            # let a single detection name itself and then coast. Both of
            # the bench's remaining invented numbers are exactly that
            # shape: 18 rows, ONE real sighting, 17 coasted, never
            # moving, asserting a number read from one frame (id11 an
            # "10" at 119.7s, id14 an "8" at 236.1s - both while the cue
            # stick lay on the table). A real ball clears this bar in a
            # tenth of a second and is unaffected; a one-frame blob never
            # does.
            if tr.age_frames < MIN_ID_FRAMES:
                cand = -1
            # REST-FROZEN IDENTITY (the live tracker's actual bought
            # rule, gate round 3: 79 residual flickers were sustained
            # misreads outlasting 5-frame hysteresis). A ball that has
            # not moved cannot become a different ball: while within
            # half a radius of its rest anchor, a SHOWN number never
            # changes to another number.
            moved = ((tr.x - tr.ax) ** 2 + (tr.y - tr.ay) ** 2) ** 0.5                 > 0.5 * max(tr.radius, 8.0)
            if moved:
                tr.ax, tr.ay = tr.x, tr.y
            at_rest = not moved
            if cand == -1:
                tr.emitted, tr.pend, tr.pend_k = -1, -1, 0
            elif tr.emitted == -1 or cand == tr.emitted:
                if tr.emitted == -1:
                    tr.emitted = cand       # first identity: no delay
                    tr.ax, tr.ay = tr.x, tr.y
                tr.pend, tr.pend_k = -1, 0
            elif at_rest and tr.emitted >= 0:
                # A resting ball must not FLICKER - but it must not be
                # permanently uncorrectable either. This branch used to
                # reset the pending counter, so evidence against a
                # resting track's name could never accumulate at all: the
                # name it happened to hold when it settled was final. The
                # bench's static 9 took "1" during one 6-frame lapse at
                # 156.3 and wore it for the remaining 80 seconds while
                # the identifier read 9 underneath in 114 of 114 frames,
                # and three separate attempts to fix its INPUTS only
                # changed which wrong name it froze on. A misread still
                # bounces off - nothing survives REST_HYST_K consecutive
                # frames by accident - but a sustained, unanimous
                # correction now lands.
                if cand == tr.pend:
                    tr.pend_k += 1
                else:
                    tr.pend, tr.pend_k = cand, 1
                if tr.pend_k >= REST_HYST_K:
                    tr.emitted, tr.pend, tr.pend_k = cand, -1, 0
            else:
                if cand == tr.pend:
                    tr.pend_k += 1
                else:
                    tr.pend, tr.pend_k = cand, 1
                if tr.pend_k >= HYST_K:
                    tr.emitted, tr.pend, tr.pend_k = cand, -1, 0
            n = tr.emitted
            # final uniqueness belt: hysteresis lag must never show one
            # number on two tracks in the same frame
            if n >= 0:
                if n in emitted_claims:
                    n = -1
                else:
                    emitted_claims[n] = tr.id
            out.append(_Row(tr.id, tr.x, tr.y, tr.radius, n,
                            tr.misses > 0.0))
        return out


@dataclass
class _Row:
    """SidecarWriter-compatible track row."""
    id: int
    x: float
    y: float
    radius: float
    number: int
    coasting: bool

    @property
    def cls(self):
        from ..core.types import BallClass
        if self.number == 0:
            return BallClass.CUE
        if 1 <= self.number <= 7:
            return BallClass.SOLID
        if self.number == 8:
            return BallClass.EIGHT
        if self.number >= 9:
            return BallClass.STRIPE
        return BallClass.UNKNOWN

    @property
    def active(self) -> bool:
        return True
