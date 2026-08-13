"""Fixes for the overlapping_balls violations the invariant scorer surfaced.

Three mechanisms, all found by looking at actual violation frames:
1. Pocketed balls resting in the basket (visible past the cushion line) were
   tracked as on-table balls — the void filter was class-gated and a real
   pocketed #5 classifies confidently.
2. Tile-seam truncated boxes: a ball straddling the tile seam yields a
   clipped box (centre shifted, radius deflated) that slipped every
   distance-based dedupe — while the other tile saw the same ball whole.
3. The cross-tile merge measured distance against the CANDIDATE's own radius,
   so a deflated box shrank its own merge zone and survived.
"""

from billiards_trainer.config import Settings
from billiards_trainer.detector_strategies.onnx_model import OnnxModelStrategy
from billiards_trainer.vision.geometry import expected_ball_radius_px
from billiards_trainer.vision.tracking import BallTracker
from billiards_trainer.vision.types import BallClass, Detection


# --------------------------------------------------------------------------- #
# _merge_boxes: (x, y, r, conf, cls)
# --------------------------------------------------------------------------- #
class TestMergeBoxes:
    def test_deflated_truncated_box_merges_into_real_one(self):
        """A clipped duplicate (small radius) 1.2 real-radii away must merge —
        under the old candidate-radius measure it survived."""
        real = (100.0, 100.0, 15.0, 0.9, 0)
        clipped = (100.0, 118.0, 7.0, 0.6, 0)     # 18px = 1.2x real radius
        kept = OnnxModelStrategy._merge_boxes([real, clipped])
        assert kept == [real]

    def test_touching_neighbours_never_merge(self):
        """Two real touching balls: 2r true separation, >=1.3r measured under
        the worst shadow-bias seen in daylight footage. Must both survive."""
        a = (100.0, 100.0, 15.0, 0.9, 0)
        b = (100.0, 100.0 + 1.3 * 15.0, 15.0, 0.85, 0)
        kept = OnnxModelStrategy._merge_boxes([a, b])
        assert len(kept) == 2

    def test_higher_confidence_wins(self):
        weak = (100.0, 100.0, 15.0, 0.55, 0)
        strong = (102.0, 101.0, 15.0, 0.95, 0)
        kept = OnnxModelStrategy._merge_boxes([weak, strong])
        assert kept == [strong]


class TestSeamDrop:
    def test_truncated_at_interior_edge_dropped(self):
        # top tile: interior (bottom) edge at y=600; ball centre 592, r=15
        boxes = [(100.0, 592.0, 15.0, 0.8, 0)]
        assert OnnxModelStrategy._drop_seam_truncated(boxes, 600.0, "above") == []

    def test_fully_inside_kept(self):
        boxes = [(100.0, 500.0, 15.0, 0.8, 0)]
        assert OnnxModelStrategy._drop_seam_truncated(boxes, 600.0, "above") == boxes

    def test_bottom_tile_top_edge(self):
        boxes = [(100.0, 408.0, 15.0, 0.8, 0),    # clipped by seam at 400
                 (100.0, 460.0, 15.0, 0.8, 0)]    # clear of it
        kept = OnnxModelStrategy._drop_seam_truncated(boxes, 400.0, "below")
        assert kept == [boxes[1]]

    def test_exterior_edges_untouched(self):
        """A ball at the frame's true top edge is NOT seam-truncated — no other
        tile sees it. 'above' only tests the bottom of the tile."""
        boxes = [(100.0, 8.0, 15.0, 0.8, 0)]      # clipped by the frame top
        assert OnnxModelStrategy._drop_seam_truncated(boxes, 600.0, "above") == boxes


# --------------------------------------------------------------------------- #
# Pocket basket: ingestion + tracker retirement
# --------------------------------------------------------------------------- #
def _fake_calib():
    import numpy as np

    from billiards_trainer.vision.geometry import TableModel

    class C:
        H = np.eye(3)
        table = TableModel.from_rect((675, 1271), pad=0)
    return C()


def _apply(p, calib, raw):
    seen = {}
    orig = p.tracker.update

    def spy(dets, *a, **k):
        seen["dets"] = dets
        return orig(dets, *a, **k)
    p.tracker.update = spy
    p.settings.balls.parallax_correction = False
    p._apply_detections(raw, calib, (1271, 675, 3))
    return seen["dets"]


class TestPocketBasket:
    def test_confident_ball_in_basket_rejected(self):
        """The exact session-20260729 failure: a classified #5 past the
        cushion line inside the side-pocket zone."""
        from billiards_trainer.vision.pipeline import Pipeline
        s = Settings()
        p = Pipeline(s)

        class FS:
            model_based = True
        p._strategy = FS()
        calib = _fake_calib()
        tbl = calib.table
        exp = expected_ball_radius_px(tbl, s.table.size)
        side = next(pk for pk in tbl.pockets if pk.is_side and pk.x > 300)
        raw = [Detection(side.x + 0.6 * tbl.pocket_radius, side.y + 8.0, exp,
                         cls=BallClass.SOLID, number=5, score=0.9)]
        assert _apply(p, calib, raw) == []

    def test_jaw_ball_on_bed_kept(self):
        """A ball hanging at the jaw sits ON the bed side of the nose line —
        must not be eaten by the basket rule."""
        from billiards_trainer.vision.pipeline import Pipeline
        s = Settings()
        p = Pipeline(s)

        class FS:
            model_based = True
        p._strategy = FS()
        calib = _fake_calib()
        tbl = calib.table
        exp = expected_ball_radius_px(tbl, s.table.size)
        raw = [Detection(tbl.x0 + exp, tbl.y0 + exp, exp,
                         cls=BallClass.SOLID, number=3, score=0.9)]
        assert len(_apply(p, calib, raw)) == 1

    def test_tracker_retires_track_in_basket(self):
        """A struck ball that comes to rest inside a pocket zone past the bed
        edge must be retired, not kept as a settled ghost."""
        tr = BallTracker(min_hits=2, still_frames=3)
        bounds = (0.0, 0.0, 675.0, 1271.0)
        pockets = [(675.0, 635.0)]
        # confirm + settle ON the bed first
        for _ in range(8):
            tr.update([Detection(600.0, 630.0, 15.0, cls=BallClass.SOLID,
                                 number=5, score=0.9)],
                      short_side=675.0, bounds=bounds, pockets=pockets,
                      pocket_r=30.0)
        assert len(tr.tracks) == 1
        # now it sits past the bed edge in the pocket zone
        for _ in range(4):
            tr.update([Detection(690.0, 640.0, 15.0, cls=BallClass.SOLID,
                                 number=5, score=0.9)],
                      short_side=675.0, bounds=bounds, pockets=pockets,
                      pocket_r=30.0)
        assert tr.tracks == [], "in-basket track must retire"


class TestRigidBodyRepair:
    def _pipeline(self):
        from billiards_trainer.vision.pipeline import Pipeline
        s = Settings()
        p = Pipeline(s)

        class FS:
            model_based = True
        p._strategy = FS()
        return p, _fake_calib(), s

    def test_interpenetrating_pair_pushed_to_touching(self):
        p, calib, s = self._pipeline()
        tbl = calib.table
        exp = expected_ball_radius_px(tbl, s.table.size)
        cx, cy = (tbl.x0 + tbl.x1) / 2, (tbl.y0 + tbl.y1) / 2
        # centres 1.4 radii apart (the daylight shadow-bias case)
        raw = [Detection(cx, cy, exp, cls=BallClass.SOLID, number=2, score=0.9),
               Detection(cx + 1.4 * exp, cy, exp, cls=BallClass.SOLID,
                         number=4, score=0.85)]
        dets = _apply(p, calib, raw)
        assert len(dets) == 2
        import math as m
        d = m.hypot(dets[0].x - dets[1].x, dets[0].y - dets[1].y)
        assert abs(d - 2 * exp) < 0.5, f"pair must touch, got {d / (2 * exp):.2f} diam"
        # symmetric push: midpoint preserved
        mid = (dets[0].x + dets[1].x) / 2
        assert abs(mid - (cx + 0.7 * exp)) < 0.5

    def test_separated_pair_untouched(self):
        p, calib, s = self._pipeline()
        tbl = calib.table
        exp = expected_ball_radius_px(tbl, s.table.size)
        cx, cy = (tbl.x0 + tbl.x1) / 2, (tbl.y0 + tbl.y1) / 2
        raw = [Detection(cx, cy, exp, cls=BallClass.SOLID, number=2, score=0.9),
               Detection(cx + 3 * exp, cy, exp, cls=BallClass.SOLID,
                         number=4, score=0.85)]
        dets = _apply(p, calib, raw)
        assert abs(dets[0].x - cx) < 1e-6 and abs(dets[1].x - (cx + 3 * exp)) < 1e-6

    def test_rack_cluster_converges(self):
        """Three balls in a squeezed row: after repair every pair >= ~1 diam."""
        import math as m
        p, calib, s = self._pipeline()
        tbl = calib.table
        exp = expected_ball_radius_px(tbl, s.table.size)
        cx, cy = (tbl.x0 + tbl.x1) / 2, (tbl.y0 + tbl.y1) / 2
        raw = [Detection(cx + i * 1.5 * exp, cy, exp, cls=BallClass.SOLID,
                         number=i + 1, score=0.9) for i in range(3)]
        dets = _apply(p, calib, raw)
        assert len(dets) == 3
        for i in range(3):
            for j in range(i + 1, 3):
                d = m.hypot(dets[i].x - dets[j].x, dets[i].y - dets[j].y)
                assert d >= 2 * exp - 1.0, f"pair {i},{j} at {d / (2 * exp):.2f} diam"


class TestIdentityGatedRepair:
    def test_duplicate_without_distinct_ids_merged_not_pushed(self):
        """One ball detected twice (only one box gets a number): the pair must
        MERGE, not be pushed apart into a fake touching pair — unconditional
        push-apart laundered duplicates into phantom tracks (caught by the
        corpus: 7.68 -> 22.60 impossible/1k on session-20260802-173553)."""
        from billiards_trainer.vision.pipeline import Pipeline
        s = Settings()
        p = Pipeline(s)

        class FS:
            model_based = True
        p._strategy = FS()
        calib = _fake_calib()
        tbl = calib.table
        exp = expected_ball_radius_px(tbl, s.table.size)
        cx, cy = (tbl.x0 + tbl.x1) / 2, (tbl.y0 + tbl.y1) / 2
        raw = [Detection(cx, cy, exp, cls=BallClass.SOLID, number=3, score=0.9),
               Detection(cx + 1.4 * exp, cy, exp, cls=BallClass.UNKNOWN,
                         number=-1, score=0.55)]
        dets = _apply(p, calib, raw)
        assert len(dets) == 1, "duplicate must merge"
        assert dets[0].number == 3, "the identified det must win"

    def test_same_number_pair_merges_keeping_stronger(self):
        from billiards_trainer.vision.pipeline import Pipeline
        s = Settings()
        p = Pipeline(s)

        class FS:
            model_based = True
        p._strategy = FS()
        calib = _fake_calib()
        tbl = calib.table
        exp = expected_ball_radius_px(tbl, s.table.size)
        cx, cy = (tbl.x0 + tbl.x1) / 2, (tbl.y0 + tbl.y1) / 2
        raw = [Detection(cx, cy, exp, cls=BallClass.SOLID, number=2, score=0.6),
               Detection(cx + 1.3 * exp, cy, exp, cls=BallClass.SOLID,
                         number=2, score=0.9)]
        dets = _apply(p, calib, raw)
        assert len(dets) == 1
        assert dets[0].score == 0.9


class TestPublishedTracksObeyPhysics:
    def test_settled_interpenetrating_pair_projected_apart(self):
        """The anti-shimmer lock swallows the det-level repair (a few px is
        'jitter'), freezing track pairs at impossible positions — the
        published copies must be projected apart when identities are
        distinct."""
        import math as m

        from billiards_trainer.vision.pipeline import Pipeline
        s = Settings()
        p = Pipeline(s)

        class FS:
            model_based = True
        p._strategy = FS()
        calib = _fake_calib()
        tbl = calib.table
        exp = expected_ball_radius_px(tbl, s.table.size)
        cx, cy = (tbl.x0 + tbl.x1) / 2, (tbl.y0 + tbl.y1) / 2
        # Let two distinct balls settle at an interpenetrating separation by
        # feeding identical frames; dets get repaired, but settled tracks lock.
        raw = [Detection(cx, cy, exp, cls=BallClass.SOLID, number=1, score=0.9),
               Detection(cx + 1.6 * exp, cy, exp, cls=BallClass.SOLID,
                         number=5, score=0.9)]
        tracks = None
        for _ in range(20):
            tracks = p._apply_detections(
                [Detection(d.x, d.y, d.radius, cls=d.cls, number=d.number,
                           score=d.score) for d in raw],
                calib, (1271, 675, 3))[0]
        pair = [t for t in tracks if t.number in (1, 5)]
        assert len(pair) == 2
        d = m.hypot(pair[0].x - pair[1].x, pair[0].y - pair[1].y)
        assert d >= 2 * exp - 1.0, \
            f"published pair at {d / (2 * exp):.2f} diam — must be >= touching"


class TestGhostRetirement:
    def test_duplicate_track_in_physics_gap_merged(self):
        """Tracks 0.79 diameters apart sat in the sliver between the old
        merge threshold (0.776 diam) and the physics floor (0.8 diam) and
        rode the occlusion budget for a minute. With ball_r supplied, the
        merge threshold is aligned to physics and the detection-backed
        track of the pair wins."""
        tr = BallTracker(min_hits=2, still_frames=3)
        real = Detection(300, 300, 15, cls=BallClass.SOLID, number=3, score=0.9)
        ghost = Detection(324, 300, 15, cls=BallClass.UNKNOWN, score=0.6)
        # 24px apart = 0.79 diam for ball_r=15.2: survived the old threshold
        for _ in range(8):
            tr.update([real, ghost], short_side=675.0, ball_r=15.2)
        assert len(tr.tracks) == 1, "physics-gap duplicate must merge"

    def test_without_ball_r_old_threshold_preserved(self):
        tr = BallTracker(min_hits=2, still_frames=3)
        a = Detection(300, 300, 15, cls=BallClass.SOLID, number=3, score=0.9)
        b = Detection(324, 300, 15, cls=BallClass.SOLID, number=5, score=0.9)
        for _ in range(8):
            tr.update([a, b], short_side=675.0)      # no ball_r
        assert len(tr.tracks) == 2

    def test_hand_occluded_ball_still_survives(self):
        """The occlusion budget must keep working when nothing else stands on
        the covered ball's spot."""
        tr = BallTracker(min_hits=2, still_frames=3)
        a = Detection(300, 300, 15, cls=BallClass.SOLID, number=3, score=0.9)
        b = Detection(600, 600, 15, cls=BallClass.SOLID, number=5, score=0.9)
        for _ in range(8):
            tr.update([a, b], short_side=675.0, ball_r=15.2)
        for _ in range(30):                    # a hand covers ball 3
            tr.update([b], short_side=675.0, ball_r=15.2)
        nums = sorted(t.number for t in tr.tracks)
        assert nums == [3, 5], f"occluded ball must survive, got {nums}"
