"""The dense-trail merge gates (never regress a replay silently)."""

from billiards_trainer.measure.trails_merge import _agrees


class TestMergeGates:
    def _dense(self, n=60):
        return [[100 + i * 0.033, 0.3 + i * 0.004, 0.5] for i in range(n)]

    def test_matching_dense_accepted(self):
        dense = self._dense()
        sparse = [[t + 7.0, x, y] for (t, x, y) in dense[::12]] + \
                 [[112.0, dense[-1][1], dense[-1][2]]]
        assert _agrees(dense, sparse)

    def test_wrong_path_rejected(self):
        dense = self._dense()
        sparse = [[t, x, 0.62] for (t, x, _y) in dense[::12]]
        assert not _agrees(dense, sparse)

    def test_wrong_endpoint_rejected(self):
        dense = self._dense()
        sparse = [pt[:] for pt in dense[::12]]
        sparse[-1] = [sparse[-1][0], 0.9, 0.9]
        assert not _agrees(dense, sparse)

    def test_stationary_sparse_abstains(self):
        dense = self._dense()
        sparse = [[100.0, 0.3, 0.5], [101.0, 0.3, 0.5], [102.0, 0.3, 0.5]]
        assert not _agrees(dense, sparse)


class TestEndpointVerdict:
    def test_dense_wins_at_the_resting_ball(self):
        from billiards_trainer.measure.arbitrate import endpoint_verdict
        assert endpoint_verdict((0.5, 0.5), (0.4, 0.4),
                                [(0.505, 0.498)], [], False) == "dense"

    def test_sparse_wins_when_reality_agrees_with_it(self):
        from billiards_trainer.measure.arbitrate import endpoint_verdict
        assert endpoint_verdict((0.5, 0.5), (0.4, 0.4),
                                [(0.398, 0.402)], [], False) == "sparse"

    def test_pocketed_ball_judged_by_pocket_proximity(self):
        from billiards_trainer.measure.arbitrate import endpoint_verdict
        assert endpoint_verdict((0.9, 0.5), (0.6, 0.5), [],
                                [(0.92, 0.5)], True) == "dense"

    def test_ambiguity_keeps_sparse(self):
        from billiards_trainer.measure.arbitrate import endpoint_verdict
        assert endpoint_verdict((0.5, 0.5), (0.4, 0.4), [], [],
                                False) == "unknown"
        assert endpoint_verdict((0.5, 0.5), (0.51, 0.5),
                                [(0.505, 0.5)], [], False) == "unknown"


class TestEarnedTieBreak:
    """prefer_dense (Joe: replays looked unchanged because the timid
    default kept sparse on every ambiguity): unknown goes to dense on
    gate-green sessions; an ACTIVE sparse verdict still wins."""

    def _setup(self):
        from types import SimpleNamespace

        import numpy as np
        times = [98.0 + i * 0.033 for i in range(160)]
        frames = [[(1, 0.3 + i * 0.002, 0.5, 0.01, 3, 2, 1)]
                  for i in range(160)]
        reader = SimpleNamespace(
            meta={"hinv": np.eye(3).tolist(), "w": 1.0, "h": 1.0},
            _times=times, _frames=frames)
        # sparse trail on a DIFFERENT path -> bootstrap gate refuses
        sparse = [[107.0 + i * 0.15, 0.3 + i * 0.01, 0.7]
                  for i in range(8)]
        doc = {"shots": [{"start": 107.0, "end": 109.0,
                          "stroke": {"strike": 100.0},
                          "trails": [{"n": 3, "p": sparse}]}]}
        return reader, doc

    def _arb(self, verdict):
        from types import SimpleNamespace
        return SimpleNamespace(verdict=lambda *a, **k: verdict)

    def test_unknown_keeps_sparse_by_default(self):
        from billiards_trainer.measure.trails_merge import merge_trails
        reader, doc = self._setup()
        stats = merge_trails("x.mp4", reader, doc, arbiter=self._arb("unknown"))
        assert stats["shots_upgraded"] == 0
        assert not doc["shots"][0]["trails"][0].get("dense")

    def test_unknown_takes_dense_when_earned(self):
        from billiards_trainer.measure.trails_merge import merge_trails
        reader, doc = self._setup()
        stats = merge_trails("x.mp4", reader, doc,
                             arbiter=self._arb("unknown"), prefer_dense=True)
        assert stats["shots_upgraded"] == 1
        assert doc["shots"][0]["trails"][0]["dense"] is True

    def test_active_sparse_verdict_still_wins(self):
        from billiards_trainer.measure.trails_merge import merge_trails
        reader, doc = self._setup()
        stats = merge_trails("x.mp4", reader, doc,
                             arbiter=self._arb("sparse"), prefer_dense=True)
        assert stats["shots_upgraded"] == 0
        assert not doc["shots"][0]["trails"][0].get("dense")

    def test_no_arbiter_prefers_dense_when_earned(self):
        from billiards_trainer.measure.trails_merge import merge_trails
        reader, doc = self._setup()
        stats = merge_trails("x.mp4", reader, doc, prefer_dense=True)
        assert stats["shots_upgraded"] == 1
