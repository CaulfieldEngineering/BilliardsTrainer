"""Capture + pipeline controller.

Lives on its own QThread (CV must not run on the UI thread — a core lesson from
the original prototype). A QTimer in that thread ticks the capture→pipeline loop;
control methods are invoked via queued connections so they execute in the worker
thread. The controller owns the DB session and shot clock, records shots, and
emits signals the UI renders.
"""

import logging
import time
from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot

from ..config import Settings
from ..db.repository import Repository
from ..events.shot_detector import ShotEvent
from ..game.shot_clock import ShotClock
from ..vision.pipeline import Pipeline

log = logging.getLogger("controller")


@dataclass
class FramePacket:
    perspective: np.ndarray | None = None
    birdseye: np.ndarray | None = None
    status: str = "idle"
    fps: float = 0.0
    n_balls: int = 0
    shot_state: str = "settled"
    clock_remaining: float = 0.0
    clock_enabled: bool = False
    clock_warning: bool = False
    deviated: bool = False
    tracks: list = field(default_factory=list)


class PipelineController(QObject):
    frame_ready = Signal(object)        # FramePacket
    stats_updated = Signal(dict)        # session summary
    shot_recorded = Signal(object)      # ShotEvent
    status_changed = Signal(str)
    clock_event = Signal(str)           # 'warn' | 'expired'
    error = Signal(str)

    def __init__(self, settings: Settings, repository: Repository):
        super().__init__()
        self._settings = settings
        self._repo = repository
        self._pipeline: Pipeline | None = None
        self._source = None
        self._timer: QTimer | None = None
        self._clock = ShotClock(settings.shot_clock)
        self._t0 = 0.0
        self._fps = 0.0
        self._prev_state = "settled"
        self._turn_start_t = 0.0
        self._session_id: int | None = None
        self._mode = "free_play"
        self._running = False

    # ------------------------------------------------------------------ #
    # Thread setup
    # ------------------------------------------------------------------ #
    @Slot()
    def on_started(self) -> None:
        self._timer = QTimer()
        self._timer.setTimerType(Qt.PreciseTimer)
        self._timer.timeout.connect(self._tick)

    # ------------------------------------------------------------------ #
    # Control (invoked queued from the UI thread)
    # ------------------------------------------------------------------ #
    @Slot(str, str, str)
    def start(self, source_spec: str, mode: str = "free_play", drill_key: str = "") -> None:
        from ..capture.camera import open_source

        self.stop()  # clean any prior run
        try:
            self._source = open_source(source_spec)
        except Exception as exc:  # noqa: BLE001 - surface any capture failure
            self.error.emit(f"Could not open source '{source_spec}': {exc}")
            return
        if hasattr(self._source, "opened") and not self._source.opened:
            self.error.emit(f"Source '{source_spec}' did not open. Check the camera/path.")
            return

        self._pipeline = Pipeline(self._settings)
        self._clock = ShotClock(self._settings.shot_clock)
        self._mode = mode
        from ..game.drills import get_drill
        drill = get_drill(drill_key) if drill_key else None
        self._session_id = self._repo.start_session(
            mode=mode, drill_key=drill_key or None,
            drill_target=drill.target_makes if drill else 0,
            table_size=self._settings.table.size,
        )
        self._t0 = time.perf_counter()
        self._prev_state = "settled"
        self._turn_start_t = 0.0
        self._running = True
        fps = getattr(self._source, "fps", 30.0)
        interval = max(8, int(1000.0 / max(1.0, min(fps, 60.0))))
        if self._timer is None:
            self.on_started()
        self._timer.start(interval)
        self.status_changed.emit("running")
        log.info("Started: source=%s mode=%s", source_spec, mode)

    @Slot()
    def stop(self) -> None:
        if self._timer:
            self._timer.stop()
        if self._running and self._session_id is not None:
            self._repo.end_session(self._session_id)
            self.stats_updated.emit(self._repo.global_summary())
        if self._source:
            try:
                self._source.release()
            except Exception:  # noqa: BLE001
                pass
        self._source = None
        self._running = False
        self.status_changed.emit("stopped")

    @Slot()
    def recalibrate(self) -> None:
        if self._pipeline:
            self._pipeline.request_recalibration()

    @Slot(object)
    def apply_settings(self, settings: Settings) -> None:
        self._settings = settings
        self._clock = ShotClock(settings.shot_clock)
        if self._pipeline:
            self._pipeline.reconfigure(settings)

    @property
    def session_id(self) -> int | None:
        return self._session_id

    # ------------------------------------------------------------------ #
    # Loop
    # ------------------------------------------------------------------ #
    @Slot()
    def _tick(self) -> None:
        if not self._running or self._source is None or self._pipeline is None:
            return
        t_wall0 = time.perf_counter()
        frame = self._source.read()
        if frame is None:
            self.error.emit("No frame from source (camera disconnected or video ended).")
            self.stop()
            return

        t = time.perf_counter() - self._t0
        try:
            res = self._pipeline.process(frame, t)
        except Exception as exc:  # noqa: BLE001 - never let one bad frame kill the loop
            log.exception("pipeline error")
            self.error.emit(f"Pipeline error: {exc}")
            return

        self._handle_state(res.shot_state, t)
        if res.shot_event is not None:
            self._record_shot(res.shot_event, t)

        clock_edge = self._clock.poll(t)
        if clock_edge:
            self.clock_event.emit(clock_edge)

        dt = time.perf_counter() - t_wall0
        inst = 1.0 / dt if dt > 0 else 0.0
        self._fps = 0.9 * self._fps + 0.1 * inst if self._fps else inst

        self.frame_ready.emit(FramePacket(
            perspective=res.frame_bgr, birdseye=res.rect_bgr, status=res.status,
            fps=self._fps, n_balls=res.n_balls, shot_state=res.shot_state,
            clock_remaining=self._clock.remaining(t),
            clock_enabled=self._clock.enabled, clock_warning=self._clock.is_warning(t),
            deviated=res.deviated, tracks=res.tracks,
        ))

    def _handle_state(self, state: str, t: float) -> None:
        if self._prev_state != "settled" and state == "settled":
            # balls came to rest -> the player's turn begins; start the clock
            self._clock.start(t)
            self._turn_start_t = t
        elif self._prev_state == "settled" and state == "moving":
            # shot taken in time -> stop the countdown
            self._clock.stop()
        self._prev_state = state

    def _record_shot(self, event: ShotEvent, t: float) -> None:
        if self._session_id is None:
            return
        shot_seconds = max(0.0, event.start_t - self._turn_start_t) if self._turn_start_t else 0.0
        self._repo.record_shot(
            self._session_id, outcome=event.outcome.value,
            num_pocketed=event.num_pocketed, target_pocket=event.target_pocket,
            cue_scratch=event.cue_scratch, duration_s=event.duration_s,
            shot_seconds=shot_seconds,
        )
        self.shot_recorded.emit(event)
        self.stats_updated.emit(self._repo.session_summary(self._session_id))


def make_controller_thread(controller: PipelineController) -> QThread:
    """Move the controller onto a dedicated thread and start it."""
    thread = QThread()
    thread.setObjectName("pipeline")
    controller.moveToThread(thread)
    thread.started.connect(controller.on_started)
    thread.start()
    return thread
