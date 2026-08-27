"""The M1 motion tracker's bought rules (the live system's incident
classes, re-implemented with a motion model — each rule pinned)."""

from billiards_trainer.measure.tracker import COAST_S, MotionTracker


def _step(tk, dets, t):
    return {r.id: r for r in tk.update(dets, t)}


class TestMotionTracker:
    def test_track_follows_a_moving_ball(self):
        tk = MotionTracker()
        for i in range(10):
            rows = _step(tk, [(100 + i * 20, 200, 16, 5)], i / 30)
        assert len(rows) == 1
        r = next(iter(rows.values()))
        assert abs(r.x - 280) < 10 and r.number == 5

    def test_coast_through_blur_keeps_identity_and_moves(self):
        tk = MotionTracker()
        for i in range(10):                       # establish velocity
            rows = _step(tk, [(100 + i * 20, 200, 16, 5)], i / 30)
        tid = next(iter(rows))
        x_last = rows[tid].x
        for i in range(10, 16):                   # 6 frames unseen (0.2s)
            rows = _step(tk, [], i / 30)
        assert tid in rows, "blur must not kill the track"
        assert rows[tid].x > x_last + 20, "coasting must PREDICT, not freeze"
        rows = _step(tk, [(rows[tid].x + 5, 200, 16, 5)], 16 / 30)
        assert tid in rows, "re-acquisition keeps the same identity"

    def test_long_absence_deactivates(self):
        tk = MotionTracker()
        _step(tk, [(100, 200, 16, 5)], 0.0)
        rows = _step(tk, [], COAST_S + 0.2)
        assert rows == {}

    def test_no_teleport_association(self):
        tk = MotionTracker()
        _step(tk, [(100, 200, 16, 5)], 0.0)
        rows = _step(tk, [(900, 900, 16, -1)], 1 / 30)
        assert len(rows) == 2, "far detection is a NEW track, never a jump"

    def test_number_never_duplicates_across_tracks(self):
        # the first marathon run's failure: 85% duplicate frames
        tk = MotionTracker()
        for i in range(6):                        # two tracks, both voting 5
            rows = _step(tk, [(100, 200, 16, 5), (400, 500, 16, 5)], i / 30)
        nums = [r.number for r in rows.values() if r.number >= 0]
        assert len(nums) == len(set(nums)), "one number, one track - always"
        assert nums == [5], "the stronger claim keeps it; the loser is -1"

    def test_single_misread_never_flips_identity(self):
        tk = MotionTracker()
        for i in range(8):
            _step(tk, [(100, 200, 16, 5)], i / 30)
        rows = _step(tk, [(100, 200, 16, 3)], 8 / 30)   # one bad read
        assert next(iter(rows.values())).number == 5

    def test_overlapping_ghost_track_dies(self):
        # gate finding: a coasted ghost beside its re-acquired self
        tk = MotionTracker()
        for i in range(8):
            _step(tk, [(100 + i * 15, 200, 16, 0)], i / 30)
        for i in range(8, 12):                    # blur: track coasts on
            _step(tk, [], i / 30)
        # re-acquisition lands NEAR the coasted prediction -> new track risk
        rows = _step(tk, [(100 + 11 * 15 + 5, 200, 16, 0)], 12 / 30)
        active_near = [r for r in rows.values()]
        # whatever the track topology, no two active rows may overlap
        for i, a in enumerate(active_near):
            for b in active_near[i + 1:]:
                d = ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5
                assert d >= 0.8 * (a.radius + b.radius), "ghost must die"

    def test_identity_is_sticky_under_contention(self):
        tk = MotionTracker()
        # two tracks, incumbent earns 5 first
        for i in range(4):
            rows = _step(tk, [(100, 200, 16, 5), (400, 500, 16, -1)], i / 30)
        # challenger gets sporadic 5 votes - never 2 clear ahead
        flips = 0
        last = None
        for i in range(4, 16):
            n2 = 5 if i % 3 == 0 else -1
            rows = _step(tk, [(100, 200, 16, 5 if i % 2 == 0 else -1),
                              (400, 500, 16, n2)], i / 30)
            who = next((rid for rid, r in rows.items()
                        if r.number == 5 and abs(r.x - 100) < 50), None)
            if last is not None and who != last:
                flips += 1
            last = who
        assert flips == 0, "the incumbent must hold the number steadily"
