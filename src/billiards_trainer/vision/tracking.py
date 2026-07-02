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
    committed_number: int = -1  # hysteresis: the held identity (resists flicker)
    cls_hist: deque = field(default_factory=lambda: deque(maxlen=15))
    num_hist: deque = field(default_factory=lambda: deque(maxlen=45))
    pos_hist: deque = field(default_factory=lambda: deque(maxlen=64))

    def predict(self) -> None:
        self.x += self.vx
        self.y += self.vy
        self.age += 1

    @property
    def cls(self) -> BallClass:
        # Identity follows the (stickier) number vote so a few bad frames during a
        # shot — motion blur misreading the cue — don't flip its class. Falls back
        # to the class-history vote until a number is established.
        n = self.number
        if n == 0:
            return BallClass.CUE
        if n == 8:
            return BallClass.EIGHT
        if 1 <= n <= 7:
            return BallClass.SOLID
        if 9 <= n <= 15:
            return BallClass.STRIPE
        if not self.cls_hist:
            return BallClass.UNKNOWN
        return Counter(self.cls_hist).most_common(1)[0][0]

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
        if self.committed_number < 0 or top == self.committed_number:
            self.committed_number = top
        elif topn >= cnt.get(self.committed_number, 0) + 5:
            self.committed_number = top   # challenger clearly won — switch


class BallTracker:
    def __init__(self, max_dist_frac: float = 0.08, max_misses: int = 30,
                 min_hits: int = 2, vel_alpha: float = 0.6,
                 pos_alpha_slow: float = 0.15, pos_alpha_fast: float = 0.92,
                 speed_lo: float = 3.0, speed_hi: float = 6.0,
                 still_speed_frac: float = 0.009, still_frames: int = 6,
                 lock_dist_frac: float = 0.012, occluded_budget: int = 1800):
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
        self._tracks: list[_Internal] = []
        self._next_id = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    # ------------------------------------------------------------------ #
    def update(self, detections: list[Detection], short_side: float) -> list[Track]:
        self._short_side = max(1.0, short_side)
        gate = max(8.0, self.max_dist_frac * short_side)

        for t in self._tracks:
            t.predict()

        unmatched_dets = set(range(len(detections)))
        # Build all (track, det) pairs within the gate, then assign greedily. The
        # gate WIDENS with a track's speed so a hard-struck ball — which jumps far
        # in one frame — can still match its detection instead of being left behind.
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
                    pairs.append((dist, ti, di))
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
            matched_tracks.add(ti)
            matched_dets.add(di)
            self._apply_match(self._tracks[ti], detections[di])

        # Unmatched tracks -> coast / age out
        for ti, t in enumerate(self._tracks):
            if ti in matched_tracks:
                continue
            t.misses += 1
            # damp velocity while coasting so it doesn't drift forever
            t.vx *= 0.6
            t.vy *= 0.6

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
            merge_dist = 0.035 * self._short_side
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
        # picked up — let it age out fast.
        self._tracks = [t for t in self._tracks
                        if t.misses <= (self.occluded_budget if t.settled else self.max_misses)]
        self._arbitrate_numbers()
        return self._public()

    def _arbitrate_numbers(self) -> None:
        """There is exactly ONE of each ball on a table, but per-track hysteresis
        can leave the same number committed on two tracks (measured: duplicate
        numbers alive in ~80% of frames on real footage). Arbitrate globally:
        the track with the strongest recent evidence keeps the number, weaker
        claimants render as unknown ('?') until the evidence sorts itself out."""
        by_num: dict[int, list[_Internal]] = {}
        for t in self._tracks:
            if t.confirmed and t.committed_number >= 0:
                by_num.setdefault(t.committed_number, []).append(t)
        for num, ts in by_num.items():
            if len(ts) < 2:
                continue
            ts.sort(key=lambda t: (sum(1 for n in t.num_hist if n == num),
                                   t.hits, -t.misses), reverse=True)
            for t in ts[1:]:
                t.committed_number = -1

    def _apply_match(self, t: _Internal, d: Detection) -> None:
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
        t.bgr = d.bgr
        t.cls_hist.append(d.cls)
        t.num_hist.append(d.number)
        t._commit_number()
        t.pos_hist.append((t.x, t.y))
        t.hits += 1
        t.misses = 0
        if t.hits >= self.min_hits:
            t.confirmed = True

    def _spawn(self, d: Detection) -> None:
        t = _Internal(id=self._next_id, x=d.x, y=d.y, radius=d.radius, bgr=d.bgr)
        t.cls_hist.append(d.cls)
        t.num_hist.append(d.number)
        t.committed_number = d.number
        t.pos_hist.append((d.x, d.y))
        t.hits = 1
        self._next_id += 1
        self._tracks.append(t)

    def _public(self) -> list[Track]:
        out = []
        for t in self._tracks:
            if not t.confirmed:
                continue
            tr = Track(
                id=t.id, x=t.x, y=t.y, radius=t.radius, vx=t.vx, vy=t.vy,
                cls=t.cls, number=t.number, bgr=t.bgr, age=t.age, hits=t.hits,
                misses=t.misses, active=(t.misses == 0), history=list(t.pos_hist),
            )
            out.append(tr)
        return out

    @property
    def tracks(self) -> list[Track]:
        return self._public()
