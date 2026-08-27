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
    _start_edge: bool = False   # one-shot "countdown began" announcement
    _run_seconds: float = 0.0   # THIS countdown's length (break shots differ)
    _next_seconds: float = 0.0  # one-shot override for the next start (0 = none)
    _paused_at: float = -1.0    # pipeline t when paused (-1 = not paused)
    _said_ten: bool = False     # one-shot spoken "Ten" at 10s remaining

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
        self._start_edge = True   # poll announces the countdown (Joe's ask)
        self._paused_at = -1.0
        self._said_ten = False
        # a one-shot length override (the shot after a break gets longer)
        self._run_seconds = self._next_seconds or float(self.settings.seconds)
        self._next_seconds = 0.0

    def set_next_seconds(self, seconds: float) -> None:
        """Length for the NEXT countdown only (e.g. time after the break)."""
        self._next_seconds = max(0.0, float(seconds))

    # -- pause/resume (Joe's rail button): edges freeze with the number --- #
    @property
    def paused(self) -> bool:
        return self._running and self._paused_at >= 0.0

    def pause(self, t: float) -> None:
        if self._running and self._paused_at < 0.0:
            self._paused_at = t

    def resume(self, t: float) -> None:
        if self._running and self._paused_at >= 0.0:
            self._start_t += t - self._paused_at
            self._paused_at = -1.0

    def stop(self) -> None:
        self._running = False

    def reset(self, t: float) -> None:
        self.start(t)

    def remaining(self, t: float) -> float:
        if not self._running:
            return float(self.settings.seconds)
        if self._paused_at >= 0.0:
            t = self._paused_at           # the number freezes while paused
        return max(0.0, self._run_seconds - (t - self._start_t))

    def is_warning(self, t: float) -> bool:
        return self._running and 0 < self.remaining(t) <= self.settings.warn_seconds

    def is_expired(self, t: float) -> bool:
        return self._running and self.remaining(t) <= 0.0

    def poll(self, t: float) -> str:
        """Advance state and return a one-shot edge event: '', 'start' (the
        countdown began), 'warn' (single beep at warn_seconds), 'tick'
        (3-2-1 cadence), or 'expired' (the buzz)."""
        if not self._running:
            return ""
        if self._start_edge:
            self._start_edge = False
            return "start"
        if self._paused_at >= 0.0:
            return ""                     # no edges advance while paused
        rem = self.remaining(t)
        if not self._expired and rem <= 0.0:
            self._expired = True
            self._running = False
            return "expired"
        # Spoken "Ten" at 10s remaining (Joe) - only when the warn bell
        # sits elsewhere (warn_seconds != 10) and the countdown began
        # above 10s (a 10s clock would announce at start, which is noise)
        if (not self._said_ten and rem <= 10.0
                and self._run_seconds > 10.5
                and int(self.settings.warn_seconds) != 10):
            self._said_ten = True
            return "ten"
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


def status_text(shot_state: str, running: bool, paused: bool,
                enabled: bool) -> str:
    """Table status beside the timer (Joe: "'Shot in Play' when no timer
    running. etc etc"). One source of truth for the rail label."""
    if not enabled:
        return "CLOCK OFF"
    if paused:
        return "PAUSED"
    if running:
        return "ON THE CLOCK"
    if shot_state == "moving":
        return "SHOT IN PLAY"
    return "TABLE SETTLED"
