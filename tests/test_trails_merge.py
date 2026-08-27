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
