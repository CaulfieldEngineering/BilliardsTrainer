"""The live stay-down timer's state machine (Joe's counter).

The contract: arms only on a settled->moving edge while recording,
re-anchors to the detector's backdated start, climbs as an estimate
(~), caps instead of counting forever, and locks to the measured
record — including honest abstention ("—") and the popped-early flag.
"""

from billiards_trainer.ui.widgets.stay_down import (
    CLIMB_CAP_S,
    StayDownTimer,
)


def _climb_to(t, timer, until):
    out = None
    x = 10.0
    while x <= until:
        out = timer.tick("moving", x, counting=True)
        x += 0.5
    return out


class TestStayDownTimer:
    def test_idle_until_moving_edge(self):
        t = StayDownTimer()
        assert t.tick("settled", 5.0, counting=True) == ("—", "idle")
        text, kind = t.tick("moving", 10.0, counting=True)
        assert kind == "climb" and text == "~0.0s"

    def test_climb_counts_from_the_edge(self):
        t = StayDownTimer()
        t.tick("moving", 10.0, counting=True)
        text, kind = t.tick("moving", 11.2, counting=True)
        assert (text, kind) == ("~1.2s", "climb")

    def test_no_arm_while_not_recording(self):
        t = StayDownTimer()
        assert t.tick("moving", 10.0, counting=False) == ("—", "idle")

    def test_no_rearm_without_a_fresh_edge(self):
        t = StayDownTimer()
        t.tick("moving", 10.0, counting=True)
        t.tick("settled", 13.0, counting=True)   # balls settle, still armed
        text, _ = t.tick("settled", 14.0, counting=True)
        assert text == "~4.0s", "settling must not restart or stop the count"

    def test_backdated_start_reanchors(self):
        t = StayDownTimer()
        t.tick("moving", 10.0, counting=True)
        t.on_shot(8.5)                    # detector backdates the strike
        text, _ = t.tick("moving", 11.0, counting=True)
        assert text == "~2.5s"

    def test_unrelated_start_ignored(self):
        t = StayDownTimer()
        t.tick("moving", 100.0, counting=True)
        t.on_shot(50.0)                   # stale event from long ago
        text, _ = t.tick("moving", 101.0, counting=True)
        assert text == "~1.0s"

    def test_climb_caps_to_waiting(self):
        t = StayDownTimer()
        t.tick("moving", 10.0, counting=True)
        out = t.tick("moving", 10.0 + CLIMB_CAP_S + 1.0, counting=True)
        assert out == ("…", "wait")

    def test_measurement_locks_the_readout(self):
        t = StayDownTimer()
        t.tick("moving", 10.0, counting=True)
        out = t.on_stroke({"start": 9.4, "stay_down_s": 1.47,
                           "confidence": "high"})
        assert out == ("1.5s", "final")
        # later frames keep the locked value
        assert t.tick("settled", 40.0, counting=True) == ("1.5s", "final")

    def test_popped_early_is_named(self):
        t = StayDownTimer()
        t.tick("moving", 10.0, counting=True)
        out = t.on_stroke({"start": 10.2, "stay_down_s": 0.62,
                           "popped_early": True, "confidence": "high"})
        assert out == ("0.6s POP", "popped")

    def test_abstention_shows_dash_not_forever_climb(self):
        t = StayDownTimer()
        t.tick("moving", 10.0, counting=True)
        assert t.on_stroke({"start": 10.0, "confidence": "none"}) == ("—", "idle")

    def test_stroke_for_another_shot_ignored(self):
        t = StayDownTimer()
        t.tick("moving", 100.0, counting=True)
        out = t.on_stroke({"start": 20.0, "stay_down_s": 2.0})
        assert out[1] == "climb", "an old shot's record must not lock the timer"

    def test_next_edge_rearms_after_final(self):
        t = StayDownTimer()
        t.tick("moving", 10.0, counting=True)
        t.on_stroke({"start": 10.0, "stay_down_s": 1.5})
        t.tick("settled", 30.0, counting=True)     # balls at rest between shots
        text, kind = t.tick("moving", 60.0, counting=True)
        assert kind == "climb" and text == "~0.0s"

    def test_reset_returns_to_idle(self):
        t = StayDownTimer()
        t.tick("moving", 10.0, counting=True)
        t.reset()
        assert (t._text, t._kind) == ("—", "idle")
        # no fake edge from the kept motion state
        assert t.tick("moving", 12.0, counting=True) == ("—", "idle")
