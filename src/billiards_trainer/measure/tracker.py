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
VOTE_N = 9               # number votes considered for identity
HYST_K = 5               # frames a new number must lead before shown


@dataclass
class _Track:
    id: int
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    radius: float = 16.0
    t: float = 0.0
    votes: list = field(default_factory=list)
    misses: float = 0.0          # seconds since last real detection
    active: bool = True
    emitted: int = -1            # the number this track SHOWS (hysteresis)
    pend: int = -1               # candidate awaiting confirmation
    pend_k: int = 0              # consecutive frames the candidate has led
    ax: float = 0.0              # rest anchor (identity freezes at rest)
    ay: float = 0.0

    @property
    def number(self) -> int:
        if not self.votes:
            return -1
        best, n = -1, 0
        for v in set(self.votes):
            c = self.votes.count(v)
            if c > n:
                best, n = v, c
        return best


class MotionTracker:
    def __init__(self):
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1
        self._holder: dict[int, int] = {}   # number -> incumbent track id

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
                gate = GATE_R * max(tr.radius, dr, 8.0)
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
            tr.x, tr.y, tr.radius, tr.t = dx, dy, dr, t
            tr.misses = 0.0
            tr.active = True
            if dn >= 0:
                tr.votes.append(dn)
                del tr.votes[:-VOTE_N]
        # 3. unmatched detections found nothing in gate: new tracks
        for di, (dx, dy, dr, dn) in enumerate(dets):
            if di in used_d:
                continue
            tr = _Track(self._next_id, dx, dy, radius=dr, t=t)
            if dn >= 0:
                tr.votes.append(dn)
            self._tracks[tr.id] = tr
            self._next_id += 1
        # 4. unmatched tracks coast; too long unseen -> inactive
        for tr in self._tracks.values():
            if tr.id in used_t or tr.t == t:
                continue
            tr.misses = t - tr.t if tr.misses == 0.0 else tr.misses + 0.0
            if t - tr.t > COAST_S:
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
                if (tr.votes.count(n), tr.t) > (cur.votes.count(n), cur.t):
                    claims[n] = tr
            else:
                chal = tr if inc is cur else cur
                if chal.votes.count(n) >= inc.votes.count(n) + 2:
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
                tr.pend, tr.pend_k = -1, 0  # frozen: misreads bounce off
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
