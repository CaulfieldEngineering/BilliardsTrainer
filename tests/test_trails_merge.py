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
