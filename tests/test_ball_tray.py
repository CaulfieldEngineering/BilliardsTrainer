"""The tray's presence logic: debounced, occlusion-proof, respot-aware."""

from billiards_trainer.ui.widgets.ball_tray import BallPresence


class TestBallPresence:
    def test_all_presumed_present_before_first_sighting(self):
        p = BallPresence()
        out = p.update(set(), 10.0)
        assert all(out.values())

    def test_potted_ball_goes_absent_after_debounce(self):
        p = BallPresence()
        p.update({1, 2, 9}, 10.0)
        out = p.update({1, 2}, 14.0)          # 9 unseen for 4s > 3s
        assert out[9] is False and out[1] is True

    def test_brief_occlusion_never_flickers(self):
        p = BallPresence()
        p.update({5}, 10.0)
        out = p.update(set(), 12.0)           # arm over the 5 for 2s
        assert out[5] is True

    def test_respot_returns_to_present(self):
        p = BallPresence()
        p.update({9}, 10.0)
        assert p.update(set(), 20.0)[9] is False
        assert p.update({9}, 21.0)[9] is True
