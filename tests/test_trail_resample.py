"""The parked dense-resampler's guardrails (pure logic only).

The module is inert behind these gates pending the measurement-core
rebuild — which makes the gates the load-bearing part: they are what
keeps an experimental path from ever shipping a wrong trail.
"""

import numpy as np

from billiards_trainer.vision.trail_resample import _ball_blobs, agrees


def _line(n=30, y=0.5):
    return [[i * 0.033, 0.1 + i * 0.02, y] for i in range(n)]


class TestAgreementGate:
    def test_matching_paths_agree(self):
        dense = _line()
        sub = dense[::6] + [dense[-1]]   # sparse ends where the ball rests
        sparse = [[t * 4.5 + 99.0, x, y] for (t, x, y) in sub]
        assert agrees(dense, sparse)     # time bases differ; geometry rules

    def test_divergent_path_rejected(self):
        dense = _line(y=0.5)
        sparse = [[t, x, 0.62] for (t, x, _y) in dense[::6]]   # parallel, far
        assert not agrees(dense, sparse)

    def test_endpoint_mismatch_rejected(self):
        dense = _line()
        sparse = [pt[:] for pt in dense[::6]]
        sparse[-1] = [sparse[-1][0], 0.95, 0.95]   # ends somewhere else
        assert not agrees(dense, sparse)

    def test_too_little_sparse_motion_abstains(self):
        dense = _line()
        sparse = [[t, 0.1, 0.5] for t in (0.0, 0.3, 0.6)]      # all at rest
        assert not agrees(dense, sparse)


class TestBallBlobs:
    def _mask(self):
        return np.zeros((200, 200), np.uint8)

    def test_ball_sized_blob_accepted(self):
        m = self._mask()
        m[100:108, 50:58] = 255                     # ~64 px ~ ball area
        blobs = _ball_blobs(m, area=64.0, r_scaled=4.0)
        assert len(blobs) == 1
        assert abs(blobs[0][0] - 53.5) < 1 and abs(blobs[0][1] - 103.5) < 1

    def test_arm_sized_blob_rejected(self):
        m = self._mask()
        m[20:120, 20:120] = 255                     # huge mover: an arm
        assert _ball_blobs(m, area=64.0, r_scaled=4.0) == []

    def test_fragment_glued_to_arm_rejected(self):
        m = self._mask()
        m[20:120, 20:120] = 255                     # the arm
        m[125:133, 60:68] = 255                     # ball-sized, 5px away
        assert _ball_blobs(m, area=64.0, r_scaled=4.0) == []

    def test_ball_far_from_arm_accepted(self):
        m = self._mask()
        m[20:120, 20:120] = 255
        m[180:188, 150:158] = 255                   # well clear of the arm
        blobs = _ball_blobs(m, area=64.0, r_scaled=4.0)
        assert len(blobs) == 1
