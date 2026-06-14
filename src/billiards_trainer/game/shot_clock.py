"""Shot clock — pure timing logic, driven by the controller's frame timestamps.

Kept UI- and audio-free so it's unit-testable; the controller emits signals for
warning/expiry and the UI plays the cue + draws the countdown.
"""

from dataclasses import dataclass

from ..config import ShotClockSettings


@dataclass
class ShotClock:
    settings: ShotClockSettings
    _start_t: float = 0.0
    _running: bool = False
    _warned: bool = False
    _expired: bool = False

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
        """Advance state and return a one-shot edge event: '', 'warn', or 'expired'."""
        if not self._running:
            return ""
        if not self._expired and self.is_expired(t):
            self._expired = True
            self._running = False
            return "expired"
        if not self._warned and self.is_warning(t):
            self._warned = True
            return "warn"
        return ""
