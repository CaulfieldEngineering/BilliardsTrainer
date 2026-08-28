"""The one Measurement Core: shadow tracker + presence + divergence.

Joe's architecture directive pinned: consumers read .present/.tracks
from ONE core; the hardened tracker shadows the champion and their
disagreement is COUNTED so promotion is a measured decision.
"""

from types import SimpleNamespace

from billiards_trainer.measure.core import MeasurementCore


def _trk(n, x, y, active=True, radius=10.0):
    return SimpleNamespace(number=n, x=x, y=y, active=active, radius=radius)


def _feed_rest(core, n, x, y, t0=0.0, frames=12):
    """Feed the shadow enough resting frames to emit a track."""
    for k in range(frames):
        core.ingest([(x, y, 10.0, n)], t0 + k / 30.0)
    return t0 + frames / 30.0


class TestReads:
    def test_present_is_detection_truth_from_observed_tracks(self):
        c = MeasurementCore()
        c.observe_tracks([_trk(5, 100, 100), _trk(9, 200, 200)], 10.0)
        assert c.present[5] and c.present[9] and not c.present[1]

    def test_present_returns_a_copy(self):
        c = MeasurementCore()
        c.observe_tracks([_trk(5, 100, 100)], 10.0)
        c.present[5] = False
        assert c.present[5] is True

    def test_tracks_are_the_champion_view(self):
        c = MeasurementCore()
        tr = _trk(3, 50, 60)
        c.observe_tracks([tr], 1.0)
        assert c.tracks == [tr]


class TestDivergence:
    def test_agreement_scores_clean(self):
        c = MeasurementCore()
        t = _feed_rest(c, 5, 100.0, 100.0)
        c.observe_tracks([_trk(5, 100.0, 100.0)], t)
        d = c.divergence_summary()
        assert d["frames"] == 1
        assert d["pos_mismatch"] == 0 and d["shadow_missing"] == 0

    def test_position_disagreement_is_counted(self):
        c = MeasurementCore()
        t = _feed_rest(c, 5, 100.0, 100.0)
        # champion claims the 5 is 300px away from where the shadow has it
        c.observe_tracks([_trk(5, 400.0, 100.0)], t)
        assert c.divergence_summary()["pos_mismatch"] == 1

    def test_champion_only_number_is_shadow_missing(self):
        c = MeasurementCore()
        t = _feed_rest(c, 5, 100.0, 100.0)
        c.observe_tracks([_trk(5, 100.0, 100.0), _trk(7, 300.0, 300.0)], t)
        assert c.divergence_summary()["shadow_missing"] == 1

    def test_shadow_only_number_is_shadow_extra(self):
        c = MeasurementCore()
        t = _feed_rest(c, 5, 100.0, 100.0)
        c.observe_tracks([], t)
        assert c.divergence_summary()["shadow_extra"] == 1

    def test_reset_clears_counters(self):
        c = MeasurementCore()
        t = _feed_rest(c, 5, 100.0, 100.0)
        c.observe_tracks([_trk(5, 400.0, 100.0)], t)
        c.divergence_summary(reset=True)
        assert c.divergence_summary()["pos_mismatch"] == 0
