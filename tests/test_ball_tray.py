"""The presence authority: detection truth, instant, respot-aware.

Joe: "It should update instantaneously with a list of balls it detects
as on the table." One opinion (measure.presence.TablePresence) feeds
the tray, and pot credits are cross-checked against the same tracks —
these pin the truth semantics that replaced the old 3s comfort debounce.
"""

from billiards_trainer.measure.presence import TablePresence
from billiards_trainer.ui.widgets.ball_tray import BallPresence


def test_tray_reexports_the_one_authority():
    # the widget module must not grow a private presence opinion again
    assert BallPresence is TablePresence


class TestTablePresence:
    def test_never_seen_shows_absent(self):
        # an unread number is an honest gap, not a presumed ball
        p = TablePresence()
        out = p.update(set(), 10.0)
        assert not any(out.values())

    def test_detection_shows_present_instantly(self):
        p = TablePresence()
        out = p.update({1, 5, 9}, 10.0)
        assert out[1] and out[5] and out[9] and not out[2]

    def test_pot_shows_within_the_grace_window(self):
        p = TablePresence()
        p.update({1, 2, 9}, 10.0)
        out = p.update({1, 2}, 10.7)          # 9 unseen for 0.7s > 0.6s
        assert out[9] is False and out[1] is True

    def test_single_frame_blink_does_not_flicker(self):
        p = TablePresence()
        p.update({5}, 10.0)
        assert p.update(set(), 10.3)[5] is True     # inside the grace
        assert p.update(set(), 10.7)[5] is False    # a real absence shows

    def test_respot_returns_to_present(self):
        p = TablePresence()
        p.update({9}, 10.0)
        assert p.update(set(), 20.0)[9] is False
        assert p.update({9}, 21.0)[9] is True
