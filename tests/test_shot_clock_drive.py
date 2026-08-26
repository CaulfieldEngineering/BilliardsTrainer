"""The cue-ball shot-clock drive (Joe: "base it purely off of cue ball
motion and sound for now" — this is the motion half; the sound stop
waits on a safe tap of the recorder's mic).

Pinned before enabling for real: countdown starts when the cue ball
rests, the strike stops it, an expired clock cannot restart itself,
and ball-in-hand (cue vanishes then reappears) arms a fresh turn.
Also the pure ShotClock edge cadence: warn, 3-2-1 ticks, the buzz.
"""

from types import SimpleNamespace

from billiards_trainer.config import ShotClockSettings
from billiards_trainer.core.types import BallClass
from billiards_trainer.game.shot_clock import ShotClock
from billiards_trainer.workers.controller import PipelineController


def _drive():
    """A bare drive harness: the controller's cue-clock methods bound to a
    plain object (no QObject init — the methods touch only Python attrs)."""
    d = SimpleNamespace()
    d._clock = ShotClock(ShotClockSettings(enabled=True, seconds=30,
                                           warn_seconds=10))
    d._settings = SimpleNamespace(
        shot_clock=d._clock.settings,
        balls=SimpleNamespace(stop_speed=0.5))
    d._clock_allowed = True
    d._saw_cue_t = -1e9
    d._clock_armed = True
    d._cue_still = 0
    d._turn_start_t = 0.0
    d._prev_state = "settled"
    # the drive's tuning constants live on the controller class
    for k in ("_CUE_MOVE_SPEED", "_CUE_STOP_FRAMES", "_CUE_GAP_S"):
        setattr(d, k, getattr(PipelineController, k))
    d.update = PipelineController._update_cue_clock.__get__(d)
    return d


def _cue(speed):
    return SimpleNamespace(cls=BallClass.CUE, speed=speed)


def _rest(d, t0, frames=6, dt=1 / 30):
    for k in range(frames):
        d.update([_cue(0.1)], t0 + k * dt)
    return t0 + frames * dt


class TestCueClockDrive:
    def test_rest_starts_the_countdown(self):
        d = _drive()
        _rest(d, 100.0)
        assert d._clock.running
        assert d._clock.remaining(101.0) < 30.0

    def test_strike_stops_it(self):
        d = _drive()
        _rest(d, 100.0)
        d.update([_cue(8.0)], 101.0)      # the strike: cue clearly rolling
        assert not d._clock.running

    def test_expired_clock_cannot_restart_itself(self):
        # Continuous 30fps frames (a >1s update gap reads as ball-in-hand
        # and legitimately re-arms — the first cut of this test did that).
        d = _drive()
        t = _rest(d, 100.0)
        expired = False
        while t < 140.0:                  # countdown runs out mid-loop
            d.update([_cue(0.1)], t)
            if d._clock.poll(t) == "expired":
                expired = True
            t += 1 / 30
        assert expired
        assert not d._clock.running, "8s of continued rest must not restart"
        # a real strike re-arms; the next rest starts a fresh turn
        d.update([_cue(8.0)], t)
        _rest(d, t + 0.1)
        assert d._clock.running

    def test_ball_in_hand_arms_a_fresh_turn(self):
        d = _drive()
        t = _rest(d, 100.0)
        d.update([_cue(8.0)], t + 1.0)    # strike (clock stops)
        # cue pocketed: absent > 1s, then placed gently (never "rolling")
        d.update([_cue(0.1)], t + 10.0)   # reappears at rest
        _rest(d, t + 10.1)
        assert d._clock.running, "ball-in-hand placement must start the turn"

    def test_disabled_clock_never_runs(self):
        d = _drive()
        d._clock.settings.enabled = False
        _rest(d, 100.0)
        assert not d._clock.running


class TestClockEdges:
    def test_warn_tick_buzz_cadence(self):
        c = ShotClock(ShotClockSettings(enabled=True, seconds=30,
                                        warn_seconds=10))
        c.start(0.0)
        edges = []
        t = 0.0
        while t < 31.0:
            e = c.poll(t)
            if e:
                edges.append((round(t, 1), e))
            t += 0.1
        kinds = [e for _, e in edges]
        assert kinds == ["warn", "tick", "tick", "tick", "expired"]
        assert edges[0][0] == 20.0        # warn at 10s remaining
        assert [round(30 - x) for x, _ in edges[1:4]] == [3, 2, 1]
