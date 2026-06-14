"""Shot / make / miss detection state machine.

Consumes tracked balls (rectified coords) frame by frame and emits a ``ShotEvent``
when a shot resolves. The model:

    SETTLED ──(a ball starts moving)──► MOVING
    MOVING  ──(all balls stopped for N frames)──► resolve → SETTLED

While MOVING we watch for balls that vanish next to a pocket — those are pockets.
On resolution:
    * cue ball pocketed              -> SCRATCH
    * >=1 object ball pocketed       -> MAKE
    * nothing pocketed               -> MISS

Quality is capped by tracking quality (a dropped track far from any pocket is
not counted as a pocket). Thresholds come from ``BallSettings``.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum

from ..config import BallSettings
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


@dataclass
class _LastSeen:
    x: float
    y: float
    cls: BallClass
    speed: float


class ShotDetector:
    def __init__(self, balls: BallSettings, settle_frames: int = 7,
                 min_shot_frames: int = 3):
        self.balls = balls
        self.settle_frames = settle_frames
        self.min_shot_frames = min_shot_frames
        self.reset()

    def reset(self) -> None:
        self._state = _State.SETTLED
        self._last: dict[int, _LastSeen] = {}
        self._cue_id: int | None = None
        self._pocketed: list[PocketedBall] = []
        self._stopped_run = 0
        self._shot_frames = 0
        self._start_t = 0.0
        self._frames_since_start = 0
        self.last_event: ShotEvent | None = None

    # ------------------------------------------------------------------ #
    @property
    def state(self) -> str:
        return self._state.value

    @property
    def is_shooting(self) -> bool:
        return self._state == _State.MOVING

    def pocketed_this_shot(self) -> list[PocketedBall]:
        return list(self._pocketed)

    # ------------------------------------------------------------------ #
    def update(self, tracks: list[Track], table: TableModel, t: float) -> ShotEvent | None:
        moving_thresh = max(self.balls.stop_speed * 2.5, 3.0)
        stop_thresh = self.balls.stop_speed

        by_id = {tr.id: tr for tr in tracks}
        self._update_cue(tracks)

        # Detect pockets: ids present last frame but gone now, last seen near a pocket.
        if self._state == _State.MOVING:
            self._detect_vanished_pockets(by_id, table, t)
            # also detect balls currently sitting inside a pocket capture zone
            self._detect_inside_pockets(tracks, table, t)

        max_speed = max((tr.speed for tr in tracks), default=0.0)
        any_moving = max_speed > moving_thresh
        all_stopped = max_speed < stop_thresh

        event: ShotEvent | None = None
        if self._state == _State.SETTLED:
            if any_moving:
                self._begin_shot(t)
        else:  # MOVING
            self._shot_frames += 1
            self._frames_since_start += 1
            if all_stopped:
                self._stopped_run += 1
            else:
                self._stopped_run = 0
            if self._stopped_run >= self.settle_frames and \
                    self._shot_frames >= self.min_shot_frames:
                event = self._resolve(table, t)

        # remember last-seen positions for vanish detection
        self._last = {
            tr.id: _LastSeen(tr.x, tr.y, tr.cls, tr.speed) for tr in tracks
        }
        return event

    # ------------------------------------------------------------------ #
    def _update_cue(self, tracks: list[Track]) -> None:
        cues = [tr for tr in tracks if tr.cls == BallClass.CUE]
        if self._cue_id in {tr.id for tr in tracks}:
            return  # keep stable identity if still present
        if cues:
            self._cue_id = cues[0].id

    def _begin_shot(self, t: float) -> None:
        self._state = _State.MOVING
        self._pocketed = []
        self._stopped_run = 0
        self._shot_frames = 0
        self._frames_since_start = 0
        self._start_t = t
        log.debug("Shot started @ %.2f", t)

    def _detect_vanished_pockets(self, by_id: dict, table: TableModel, t: float) -> None:
        present = set(by_id)
        for tid, last in self._last.items():
            if tid in present:
                continue
            pocket = table.pocket_at(last.x, last.y, scale=1.8)
            if pocket is not None and not self._already_pocketed(tid):
                self._pocketed.append(PocketedBall(tid, last.cls, pocket.name, t))
                log.debug("Ball %d vanished into %s", tid, pocket.name)

    def _detect_inside_pockets(self, tracks: list[Track], table: TableModel, t: float) -> None:
        for tr in tracks:
            pocket = table.pocket_at(tr.x, tr.y, scale=1.0)
            if pocket is not None and not self._already_pocketed(tr.id):
                self._pocketed.append(PocketedBall(tr.id, tr.cls, pocket.name, t))
                log.debug("Ball %d inside %s", tr.id, pocket.name)

    def _already_pocketed(self, tid: int) -> bool:
        return any(p.track_id == tid for p in self._pocketed)

    def _resolve(self, table: TableModel, t: float) -> ShotEvent:
        cue_scratch = any(
            p.track_id == self._cue_id or p.cls == BallClass.CUE for p in self._pocketed
        )
        object_pockets = [p for p in self._pocketed if p.cls != BallClass.CUE]
        if cue_scratch:
            outcome = ShotOutcome.SCRATCH
        elif object_pockets:
            outcome = ShotOutcome.MAKE
        else:
            outcome = ShotOutcome.MISS

        target = object_pockets[0].pocket if object_pockets else None
        event = ShotEvent(
            outcome=outcome,
            pocketed=list(self._pocketed),
            target_pocket=target,
            duration_s=max(0.0, t - self._start_t),
            cue_scratch=cue_scratch,
            num_pocketed=len(object_pockets),
            start_t=self._start_t,
            end_t=t,
        )
        self.last_event = event
        self._state = _State.SETTLED
        self._pocketed = []
        log.info("Shot resolved: %s (%d balls, %.1fs)",
                 outcome.value, len(object_pockets), event.duration_s)
        return event
