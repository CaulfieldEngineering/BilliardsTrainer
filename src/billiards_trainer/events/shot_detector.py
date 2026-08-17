"""Shot / make / miss detection — hard-evidence state machine.

Tuned to NOT fire on noise (lighting flicker, compression artifacts, a blob
blinking in and out). The decisive signal is **motion energy** — the mean
frame-to-frame pixel change in the playing area — which is robust even when fast
balls outrun the tracker (per-ball velocity is unreliable exactly when it matters
most). A shot is only counted when:

  * a **cue ball** is identified (configurable; no cue -> no detection),
  * the table is **actively in motion** for N consecutive frames (energy gate),
  * a ball **travels a meaningful distance** from its rest position, and
  * a pot only counts when a ball **approaches a pocket and then drops in**
    (stops being detected near the pocket) — not a single-frame blink,

with a **warm-up** after Start and a **cool-down** between shots. Every gate is
configurable from Settings -> Detection. ``last_diag`` exposes the intermediate
signals for the debug overlay.
"""

import logging
import math
from dataclasses import dataclass, field
from enum import Enum

from ..config import BallSettings, DetectionSettings
from ..vision.geometry import TableModel
from ..vision.types import BallClass, Track

log = logging.getLogger("events.shot")


class ShotOutcome(str, Enum):
    MAKE = "make"
    MISS = "miss"
    SCRATCH = "scratch"


class _State(str, Enum):
    SETTLED = "settled"
    MOVING = "moving"


@dataclass
class PocketedBall:
    track_id: int
    cls: BallClass
    pocket: str
    t: float


@dataclass
class ShotEvent:
    outcome: ShotOutcome
    pocketed: list[PocketedBall] = field(default_factory=list)
    target_pocket: str | None = None
    duration_s: float = 0.0
    cue_scratch: bool = False
    num_pocketed: int = 0
    start_t: float = 0.0
    end_t: float = 0.0
    max_travel: float = 0.0
    suggested: bool = False  # set by the controller in manual-confirm mode


@dataclass
class _Seen:
    x: float
    y: float
    cls: BallClass
    speed: float


class ShotDetector:
    def __init__(self, detection: DetectionSettings, balls: BallSettings,
                 settle_frames: int = 7, min_shot_frames: int = 3,
                 min_shot_s: float = 1.2, settle_s: float = 0.5):
        self.det = detection
        self.balls = balls
        self.settle_frames = settle_frames
        self.min_shot_frames = min_shot_frames
        # TIME floors alongside the frame counts: frame counts are cadence-
        # dependent (7 quiet updates at 30 fps is 0.23s — a struck ball whose
        # flight is motion-blurred out of tracking hasn't even REAPPEARED
        # yet, so the shot resolved with ~35px of travel and was discarded;
        # measured on 011510 @ 72.2s). Physics: a stroke's balls roll for
        # over a second; holding the shot open that long lets the flyer's
        # revival land inside it, where its snap displacement counts.
        self.min_shot_s = min_shot_s
        self.settle_s = settle_s
        self.reset()

    def reset(self) -> None:
        self._state = _State.SETTLED
        self._cue_id: int | None = None
        self._first_t: float | None = None
        self._last_shot_t: float = -1e9
        self._rest: dict[int, tuple] = {}
        self._min_pdist: dict[int, tuple] = {}      # id -> (closest dist, pocket name)
        self._shot_ids: set = set()                 # ids seen during this shot
        self._shot_cls: dict[int, BallClass] = {}   # id -> last class seen
        self._prev: dict[int, _Seen] = {}
        self._prev_seen_at: dict[int, int] = {}
        self._pocketed: list[PocketedBall] = []
        self._active_run = 0
        self._quiet_run = 0
        self._shot_frames = 0
        self._arm_frames = 0
        self._still_run = 0
        self._calm_since: float | None = None   # start of the current quiet+still spell
        self._carried_moving = 0                # moving updates with a hand on a mover
        self._any_moving = 0                    # moving updates with any tracked step
        self._free_frames: dict[int, int] = {}
        self._free_travel: dict[int, float] = {}
        self._pending_free: dict[int, tuple] = {}
        self._pending_free_t0: dict[int, float] = {}
        self._pending_quiet: dict[int, int] = {}
        self._last_carried: dict[int, int] = {}
        self._frame_idx = 0
        self._start_t = 0.0
        self._max_travel = 0.0
        self._motion = 0.0
        self._fused = 0.0
        self._evidence: dict = {}
        self.last_event: ShotEvent | None = None
        self.last_diag: dict = {}

    # ------------------------------------------------------------------ #
    @property
    def state(self) -> str:
        return self._state.value

    @property
    def is_shooting(self) -> bool:
        return self._state == _State.MOVING

    @property
    def has_cue(self) -> bool:
        return self._cue_id is not None

    # ------------------------------------------------------------------ #
    def update(self, tracks: list[Track], table: TableModel, t: float,
               motion: float = 0.0, evidence: dict | None = None) -> ShotEvent | None:
        if self._first_t is None:
            self._first_t = t
        det = self.det
        self._motion = motion
        self._evidence = evidence or {"motion": motion, "flow": 0.0, "fg": 0.0}
        by_id = {tr.id: tr for tr in tracks}

        cue_present = self._update_cue(tracks)
        self._frame_idx += 1
        for tid in (self._evidence.get("carried_ids") or set()):
            self._last_carried[tid] = self._frame_idx
        # Fused multi-modal activity (or plain motion energy if fusion is off).
        if det.use_fusion:
            self._fused = self._fuse(motion, self._evidence)
            active = self._fused >= det.fusion_active
            quiet = self._fused < (det.fusion_active * 0.5)
        else:
            self._fused = motion
            active = motion > det.motion_active
            quiet = motion < det.motion_quiet
        warming = (t - self._first_t) < det.warmup_seconds
        cooling = (t - self._last_shot_t) < det.cooldown_seconds

        self._active_run = self._active_run + 1 if active else 0
        self._quiet_run = self._quiet_run + 1 if quiet else 0

        # rest positions: snapshot while the table is calm and settled
        if self._state == _State.SETTLED and quiet:
            for tr in tracks:
                self._rest[tr.id] = (tr.x, tr.y)
        # Pre-shot motion bank: a struck ball is already moving while the arm
        # gate deliberates — bank its free steps so travel/pocket credit and
        # the shot START reach back to the strike itself. Banking is PER-BALL
        # and independent of the whole-table motion signal: frame-difference
        # energy of one fast ball on a big table hovers around the activity
        # threshold, and a single sub-threshold frame used to wipe the bank
        # (measured on session-011928: a clean stroke died at run 5/6 and the
        # detector armed 1.2s late on a secondary surge — the G5 recall audit
        # flagged 721 candidate misses across 10 sessions on exactly this).
        if self._state == _State.SETTLED:
            carried_pre = self._evidence.get("carried_ids") or set()
            for tr in tracks:
                # A ball carried NOW or recently doesn't bank: a just-placed
                # ball wobbles free for a few frames after release, and
                # banking that wobble resurrected a false shot the audio
                # harness had already killed once. A struck object ball was
                # never carried; the cue ball's stick contact happens at the
                # strike itself, inside the shot, not before it.
                if (tr.id in carried_pre or self._frame_idx
                        - self._last_carried.get(tr.id, -10**9) < 15):
                    self._pending_free.pop(tr.id, None)
                    self._pending_free_t0.pop(tr.id, None)
                    self._pending_quiet.pop(tr.id, None)
                    continue
                prev = self._prev.get(tr.id)
                step = math.hypot(tr.x - prev.x, tr.y - prev.y) if prev else 0.0
                if step > 1.0:
                    f, d = self._pending_free.get(tr.id, (0, 0.0))
                    if f == 0:
                        self._pending_free_t0[tr.id] = t
                    self._pending_free[tr.id] = (f + 1, d + step)
                    self._pending_quiet[tr.id] = 0
                elif tr.id in self._pending_free:
                    q = self._pending_quiet.get(tr.id, 0) + 1
                    self._pending_quiet[tr.id] = q
                    if q >= 4:      # flurry ended without arming: not a shot
                        self._pending_free.pop(tr.id, None)
                        self._pending_free_t0.pop(tr.id, None)
                        self._pending_quiet.pop(tr.id, None)

        event: ShotEvent | None = None
        if self._state == _State.SETTLED:
            # A ball demonstrably STRUCK is the primary arm signal: >=3 banked
            # free frames and >=2.5 ball radii of integrated free travel — a
            # placed ball's release wobble reaches neither. The whole-table
            # motion run stays as a fallback for breaks and multi-ball chaos
            # where the tracker briefly loses the fast balls.
            ball_r = 0.0225 * table.short_side
            struck = any(f >= 3 and d >= 2.5 * ball_r
                         for f, d in self._pending_free.values())
            if (not warming and not cooling and (not det.require_cue or cue_present)
                    and (struck or self._active_run >= det.strike_frames)):
                self._begin_shot(t)
        else:  # MOVING
            self._shot_frames += 1
            if self._evidence.get("arm", 0.0) >= 0.010:
                self._arm_frames += 1          # diagnostics only
            # Carried-ball discrimination: a hand-moved ball is adjacent to a
            # foreign blob (hand/arm/stick) for its WHOLE displacement; a
            # struck ball leaves the stick within a frame or two. Only free
            # (non-adjacent) motion counts toward travel and pocket credit —
            # this is what separates drill ball-gathering from actual shots.
            # (A first attempt used whole-table arm DWELL and failed both
            # ways: the cue stick hovers over the felt through every real
            # shot, and a placing hand is too small to move the fraction.)
            carried = self._evidence.get("carried_ids") or set()
            stepping = {tr.id for tr in tracks
                        if self._prev.get(tr.id) is not None
                        and math.hypot(tr.x - self._prev[tr.id].x,
                                       tr.y - self._prev[tr.id].y) > 1.0}
            if stepping:
                self._any_moving += 1
                if stepping & carried:
                    self._carried_moving += 1
            max_step = 0.0
            for tr in tracks:
                if tr.id in carried:
                    continue
                # Observed per-frame displacement, not the tracker's velocity
                # estimate — the EMA lags a fresh strike and is zeroed by the
                # settled lock, while position deltas are direct truth.
                prev = self._prev.get(tr.id)
                step = math.hypot(tr.x - prev.x, tr.y - prev.y) if prev else 0.0
                max_step = max(max_step, step)
                if step > 1.0:
                    self._free_frames[tr.id] = self._free_frames.get(tr.id, 0) + 1
                    # Travel is displacement INTEGRATED over free frames — not
                    # displacement from rest. A hand-carried ball ends up far
                    # from its rest spot (displacement large) but never moves
                    # freely (integral ~0); a placed ball wobbles freely for a
                    # few slow frames (integral ~20px); a struck ball rolls
                    # fast and free (integral hundreds of px). One number
                    # separates all three.
                    ft = self._free_travel.get(tr.id, 0.0) + step
                    self._free_travel[tr.id] = ft
                    if ft > self._max_travel:
                        self._max_travel = ft
            self._accumulate(by_id, table)
            # Joe's spec, verbatim: the shot continues until all balls stop
            # rolling or are pocketed. "Quiet" frame-diff energy alone is not
            # that — a lone slow-rolling ball barely registers as motion — so
            # settling additionally requires the TRACKED balls to be still
            # for the same run of frames.
            self._still_run = self._still_run + 1 if max_step <= 1.0 else 0
            calm = (quiet and max_step <= 1.0)
            if calm and self._calm_since is None:
                self._calm_since = t
            elif not calm:
                self._calm_since = None
            # Resolve on TIME, not update counts: the shot must have lived
            # min_shot_s (long enough for a blurred-out flyer to reappear
            # and be counted) and the table must have been continuously
            # calm for settle_s. Frame-count floors remain as guards.
            if (self._quiet_run >= self.settle_frames
                    and self._still_run >= self.settle_frames
                    and self._shot_frames >= self.min_shot_frames
                    and t - self._start_t >= self.min_shot_s
                    and self._calm_since is not None
                    and t - self._calm_since >= self.settle_s):
                self._finalize_pockets(by_id, table, t)
                event = self._resolve(t)

        # Positional memory RETAINS vanished balls for ~3s: a struck ball
        # motion-blurs out of tracking and its track revives far away — the
        # revival step (last seen -> reappearance) is real displacement and
        # must bank/count. Rebuilding _prev from current tracks forgot the
        # flyer, so its 500px snap read as step 0 (011510 @ 72.2s).
        for tr in tracks:
            self._prev[tr.id] = _Seen(tr.x, tr.y, tr.cls, tr.speed)
            self._prev_seen_at[tr.id] = self._frame_idx
        stale = [tid for tid, at in self._prev_seen_at.items()
                 if self._frame_idx - at > 90]
        for tid in stale:
            self._prev.pop(tid, None)
            self._prev_seen_at.pop(tid, None)
        self._set_diag(t, motion, cue_present, warming, cooling)
        return event

    # ------------------------------------------------------------------ #
    def _fuse(self, motion: float, ev: dict) -> float:
        """Weighted evidence score in 0..1 from the three modalities.
        Normalisation references were set from measured demo-shot vs flicker
        values: ~1% motion, ~5% coherent flow, ~3% foreground == "full"."""
        nm = min(1.0, motion / 1.0)
        nf = min(1.0, ev.get("flow", 0.0) / 6.0)
        ng = min(1.0, ev.get("fg", 0.0) / 0.8)
        d = self.det
        total = (d.w_motion + d.w_flow + d.w_fg) or 1.0
        return (d.w_motion * nm + d.w_flow * nf + d.w_fg * ng) / total

    def _update_cue(self, tracks: list[Track]) -> bool:
        cues = [tr for tr in tracks if tr.cls == BallClass.CUE]
        if cues and (self._cue_id is None or self._cue_id not in {tr.id for tr in tracks}):
            self._cue_id = cues[0].id
        return bool(cues)

    def _begin_shot(self, t: float) -> None:
        self._state = _State.MOVING
        self._pocketed = []
        self._min_pdist = {}
        self._shot_ids = set()
        self._shot_cls = {}
        self._quiet_run = 0
        self._shot_frames = 0
        self._arm_frames = 0
        self._still_run = 0
        self._calm_since = None
        self._carried_moving = 0
        self._any_moving = 0
        self._free_frames = {tid: f for tid, (f, _d) in self._pending_free.items()}
        self._free_travel = {tid: d for tid, (_f, d) in self._pending_free.items()}
        self._max_travel = max(self._free_travel.values(), default=0.0)
        # The shot STARTS at the strike, not at the arm decision: backdate to
        # the first banked free step (capped — a stale bank must not drag the
        # start into the previous rack). This is what puts the strike inside
        # the clip and the audio onset inside the claim window.
        t0 = min(self._pending_free_t0.values(), default=t)
        self._start_t = max(t0, t - 2.0)
        self._pending_free = {}
        self._pending_free_t0 = {}
        self._pending_quiet = {}
        log.debug("Shot started @ %.2f (backdated from %.2f, motion %.1f)",
                  self._start_t, t, self._motion)

    def _accumulate(self, by_id: dict, table: TableModel) -> None:
        """Track, per ball seen during the shot, its class and closest-ever
        approach to a pocket."""
        for tr in by_id.values():
            self._shot_ids.add(tr.id)
            self._shot_cls[tr.id] = tr.cls
            pk, d = table.nearest_pocket(tr.x, tr.y)
            best, _ = self._min_pdist.get(tr.id, (1e9, ""))
            if d < best:
                self._min_pdist[tr.id] = (d, pk.name)

    def _finalize_pockets(self, by_id: dict, table: TableModel, t: float) -> None:
        """At resolution, count as potted any ball that approached a pocket and is
        no longer being *actively* detected (it dropped in). Treating coasting
        (inactive) tracks as gone is what makes this robust — a potted ball stops
        being detected at the pocket lip while the tracker briefly coasts it. A
        ball still actively tracked on the table is not a pot."""
        active = {tid for tid, tr in by_id.items() if tr.active}
        gate = table.pocket_radius * 2.2
        for tid in self._shot_ids:
            if tid in active or self._already(tid):
                continue
            # A ball that never moved freely was picked up, not potted. A
            # fast pot can reach the pocket in 3-4 tracked frames; a lifted
            # ball wobbles for 0-2. Three is the floor between them.
            if self._free_frames.get(tid, 0) < 3:
                continue
            dist, name = self._min_pdist.get(tid, (1e9, ""))
            if dist < gate and name:
                cls = self._shot_cls.get(tid, BallClass.UNKNOWN)
                self._pocketed.append(PocketedBall(tid, cls, name, t))
                log.debug("Ball %d dropped into %s (closest %.0fpx)", tid, name, dist)

    def _already(self, tid: int) -> bool:
        return any(p.track_id == tid for p in self._pocketed)

    def _resolve(self, t: float) -> ShotEvent | None:
        self._state = _State.SETTLED
        # the decisive gate: a real shot moved a ball a meaningful distance
        if self._max_travel < self.det.min_travel_px:
            log.debug("Discarded non-shot (travel %.0f < %.0f)", self._max_travel,
                      self.det.min_travel_px)
            self._pocketed = []
            return None
        # Carried-dwell veto: in a real stroke the stick leaves the ball
        # within a frame or two, so moving updates with a hand ON a mover
        # are rare; in ball-gathering they are the majority. This is what
        # the audio audit exposed as "silent shots with hands involved" —
        # gatherings whose intermittent free frames beat the travel gate.
        if self._any_moving >= 4 and self._carried_moving > 0.6 * self._any_moving:
            log.debug("Discarded gathering (hand on movers %d/%d updates)",
                      self._carried_moving, self._any_moving)
            self._pocketed = []
            return None

        cue_scratch = any(p.track_id == self._cue_id or p.cls == BallClass.CUE
                          for p in self._pocketed)
        object_pockets = [p for p in self._pocketed if p.cls != BallClass.CUE]
        if cue_scratch:
            outcome = ShotOutcome.SCRATCH
        elif object_pockets:
            outcome = ShotOutcome.MAKE
        else:
            outcome = ShotOutcome.MISS

        self._last_shot_t = t
        event = ShotEvent(
            outcome=outcome, pocketed=list(self._pocketed),
            target_pocket=object_pockets[0].pocket if object_pockets else None,
            duration_s=max(0.0, t - self._start_t), cue_scratch=cue_scratch,
            num_pocketed=len(object_pockets), start_t=self._start_t, end_t=t,
            max_travel=self._max_travel,
        )
        self.last_event = event
        self._pocketed = []
        log.info("Shot resolved: %s (%d balls, travel %.0fpx, %.1fs)",
                 outcome.value, len(object_pockets), self._max_travel, event.duration_s)
        return event

    def _set_diag(self, t: float, motion: float, cue: bool,
                  warming: bool, cooling: bool) -> None:
        self.last_diag = {
            "state": self._state.value,
            "cue": cue,
            "motion": round(motion, 1),
            "flow": round(self._evidence.get("flow", 0.0), 1),
            "fg": round(self._evidence.get("fg", 0.0), 1),
            "fused": round(self._fused, 2),
            "active_run": self._active_run,
            "arm": round(self._evidence.get("arm", 0.0), 3),
            "travel": round(self._max_travel, 0),
            "warming": warming,
            "cooling": cooling,
            "potted": len(self._pocketed),
        }
