"""Shot clock — pure timing logic, driven by the controller's frame timestamps.

Kept UI- and audio-free so it's unit-testable; the controller emits signals for
the audio edges and the UI plays the cue + draws the countdown.

Joe's cadence: one warning beep at ``warn_seconds`` (10 s), a tick beep as the
clock crosses 3, 2 and 1 seconds, and the buzz at 0. Starting/stopping is the
controller's job (cue ball stops -> start; the strike -> stop = made it).
"""

import math
from dataclasses import dataclass

from ..config import ShotClockSettings


@dataclass
class ShotClock:
    settings: ShotClockSettings
    _start_t: float = 0.0
    _running: bool = False
    _warned: bool = False
    _expired: bool = False
    _last_tick: int = 0     # last 3/2/1 second already beeped (0 = none yet)

    @property
    def enabled(self) -> bool:
        return self.settings.enabled

    @property
    def running(self) -> bool:
        return self._running

    def start(self, t: float) -> None:
        # When disabled the clock never runs — it must not interfere with
        # sandbox/free-play (no countdown, no warn/expire edges, no audio).
        if not self.settings.enabled:
            self._running = False
            return
        self._start_t = t
        self._running = True
        self._warned = False
        self._expired = False
        self._last_tick = 0

    def stop(self) -> None:
        self._running = False

    def reset(self, t: float) -> None:
        self.start(t)

    def remaining(self, t: float) -> float:
        if not self._running:
            return float(self.settings.seconds)
        return max(0.0, self.settings.seconds - (t - self._start_t))

    def is_warning(self, t: float) -> bool:
        return self._running and 0 < self.remaining(t) <= self.settings.warn_seconds

    def is_expired(self, t: float) -> bool:
        return self._running and self.remaining(t) <= 0.0

    def poll(self, t: float) -> str:
        """Advance state and return a one-shot edge event: '', 'warn' (single
        beep at warn_seconds), 'tick' (3-2-1 cadence), or 'expired' (the buzz)."""
        if not self._running:
            return ""
        rem = self.remaining(t)
        if not self._expired and rem <= 0.0:
            self._expired = True
            self._running = False
            return "expired"
        sec = math.ceil(rem)            # rem in (2, 3] -> 3, so the tick fires
        if not self._warned and rem <= self.settings.warn_seconds:
            self._warned = True
            if rem <= 3.0:              # warn inside cadence range: no double beep
                self._last_tick = sec
            return "warn"
        if rem <= 3.0 and 1 <= sec <= 3 and sec != self._last_tick:
            self._last_tick = sec
            return "tick"
        return ""
