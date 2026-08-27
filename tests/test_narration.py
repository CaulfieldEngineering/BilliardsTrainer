"""The narration gate: one announcement per episode, correct kind."""

from billiards_trainer.game.narration import Narrator


class TestNarrator:
    def test_cue_pickup_says_ball_in_hand_once(self):
        n = Narrator()
        assert n.update(True, False, False, 10.0) == "ball_in_hand"
        for k in range(20):                       # keeps holding the cue
            assert n.update(True, False, False, 10.1 + k * 0.1) is None

    def test_object_shuffle_says_table_change(self):
        n = Narrator()
        assert n.update(False, True, False, 10.0) == "table_change"
        assert n.update(True, True, False, 10.5) is None   # same episode

    def test_cue_plus_objects_is_table_change(self):
        n = Narrator()
        assert n.update(True, True, False, 10.0) == "table_change"

    def test_episode_escalates_but_never_deescalates(self):
        n = Narrator()
        assert n.update(True, False, False, 10.0) == "ball_in_hand"
        assert n.update(True, True, False, 11.0) == "table_change"
        assert n.update(True, False, False, 12.0) is None

    def test_new_episode_after_quiet_reannounces(self):
        n = Narrator()
        assert n.update(True, False, False, 10.0) == "ball_in_hand"
        for k in range(40):                       # hands off for 4s
            assert n.update(False, False, False, 11.0 + k * 0.1) is None
        assert n.update(True, False, False, 30.0) == "ball_in_hand"

    def test_cooldown_blocks_rapid_reepisodes(self):
        n = Narrator()
        assert n.update(True, False, False, 10.0) == "ball_in_hand"
        n.update(False, False, False, 11.0)
        n.update(False, False, False, 14.0)       # episode ended (quiet)
        assert n.update(True, False, False, 14.5) is None   # < 8s cooldown

    def test_silent_during_shot_flight(self):
        n = Narrator()
        assert n.update(False, True, True, 10.0) is None
