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


class TestFoulTaxonomy:
    def test_no_contact_is_a_foul(self):
        from billiards_trainer.events.shot_detector import (
            ShotOutcome,
            foul_for,
        )
        assert foul_for(False, ShotOutcome.MISS, False) == "no_contact"

    def test_contact_miss_is_clean(self):
        from billiards_trainer.events.shot_detector import (
            ShotOutcome,
            foul_for,
        )
        assert foul_for(False, ShotOutcome.MISS, True) is None

    def test_scratch_is_called_scratch_not_foul(self):
        from billiards_trainer.events.shot_detector import (
            ShotOutcome,
            foul_for,
        )
        assert foul_for(True, ShotOutcome.SCRATCH, False) is None

    def test_make_never_fouls_on_this_axis(self):
        from billiards_trainer.events.shot_detector import (
            ShotOutcome,
            foul_for,
        )
        assert foul_for(False, ShotOutcome.MAKE, True) is None


class TestClockMetadata:
    def test_clock_events_round_trip_the_sidecar(self, tmp_path):
        from billiards_trainer.vision.analysis_cache import (
            SidecarReader,
            SidecarWriter,
        )
        video = tmp_path / "session-c.mp4"
        w = SidecarWriter(video, {"fps": 30.0})
        w.add_clock({"type": "clock", "t": 12.5, "ev": "start", "seconds": 60.0})
        w.add_clock({"type": "clock", "t": 30.1, "ev": "stop", "seconds": 60.0})
        w.close()
        r = SidecarReader(video)
        assert r.clock_events == [
            {"t": 12.5, "ev": "start", "seconds": 60.0},
            {"t": 30.1, "ev": "stop", "seconds": 60.0},
        ]
