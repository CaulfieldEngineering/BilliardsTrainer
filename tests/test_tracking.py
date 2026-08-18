"""Tracker tests using synthetic detections (deterministic, detector-independent)."""

from billiards_trainer.vision.tracking import BallTracker
from billiards_trainer.vision.types import BallClass, Detection


def det(x, y, cls=BallClass.SOLID, r=10):
    return Detection(x=x, y=y, radius=r, cls=cls)


def test_track_confirms_after_min_hits():
    tr = BallTracker(min_hits=3)
    short = 400
    for _ in range(2):
        out = tr.update([det(100, 100)], short)
        assert out == []          # not yet confirmed
    out = tr.update([det(100, 100)], short)
    assert len(out) == 1          # confirmed on the 3rd hit


def test_radius_outlier_is_rejected_on_confirmed_track():
    """Once a track is established, a single wildly-different detection radius
    (sensor-noise outlier) must NOT pump the track's size — the cause of the
    'ball keeps changing size on a still scene' complaint."""
    tr = BallTracker(min_hits=3)
    short = 400
    for _ in range(4):                       # establish a stable radius-10 track
        tr.update([det(100, 100, r=10)], short)
    out = tr.update([det(100, 100, r=22)], short)   # +120% radius spike
    assert out[0].radius < 13                # held near 10, not blown up to ~22
    # a moderate change is still allowed to drift in slowly
    for _ in range(8):
        out = tr.update([det(100, 100, r=12)], short)
    assert 10 < out[0].radius <= 12.5


def test_adaptive_smoothing_beats_fixed_alpha_on_both_axes():
    """The adaptive position alpha must keep stillness jitter as low as a heavily-
    smoothed tracker AND follow fast motion with much less lag — the fix for the
    'cue ball lags a foot' report without sacrificing the still-scene tuning."""
    import statistics as st

    import numpy as np

    def lag(tr, vx=8.0):
        short = 400.0
        tx, ty = 100.0, 200.0
        out = []
        for _ in range(12):
            tr.update([det(tx, ty)], short)
        for _ in range(25):
            tx += vx
            o = tr.update([det(tx, ty)], short)
            if o:
                out.append(((o[0].x - tx) ** 2 + (o[0].y - ty) ** 2) ** 0.5)
        return st.mean(out[-8:])

    def jitter(tr):
        rng = np.random.default_rng(0)
        short = 400.0
        pos = []
        for i in range(40):
            o = tr.update([det(100 + rng.normal(0, 1.2), 200 + rng.normal(0, 1.2))], short)
            if o and i > 10:
                pos.append((o[0].x, o[0].y))
        xs = [p[0] for p in pos]
        ys = [p[1] for p in pos]
        return (st.pstdev(xs) ** 2 + st.pstdev(ys) ** 2) ** 0.5

    def mk(slow, fast):
        return BallTracker(min_hits=2, pos_alpha_slow=slow, pos_alpha_fast=fast)

    smooth_lag, smooth_jit = lag(mk(0.15, 0.15)), jitter(mk(0.15, 0.15))
    adapt_lag, adapt_jit = lag(mk(0.15, 0.85)), jitter(mk(0.15, 0.85))
    # adaptive keeps the smooth tracker's low jitter ...
    assert adapt_jit <= smooth_jit + 0.05
    # ... but follows motion with much less lag
    assert adapt_lag < smooth_lag * 0.7


def test_id_persists_across_motion():
    tr = BallTracker(min_hits=2)
    short = 400
    tr.update([det(100, 100)], short)
    out = tr.update([det(112, 100)], short)
    assert len(out) == 1
    tid = out[0].id
    out = tr.update([det(124, 100)], short)
    assert out[0].id == tid       # same identity
    assert out[0].vx > 5          # velocity picked up rightward motion


def test_two_balls_keep_separate_ids():
    tr = BallTracker(min_hits=2)
    short = 400
    tr.update([det(100, 100), det(300, 300)], short)
    out = tr.update([det(105, 100), det(305, 300)], short)
    ids = sorted(t.id for t in out)
    assert len(ids) == 2 and ids[0] != ids[1]


def test_track_survives_brief_occlusion():
    tr = BallTracker(min_hits=2, max_misses=10)
    short = 400
    tr.update([det(100, 100)], short)
    out = tr.update([det(110, 100)], short)
    tid = out[0].id
    # 3 frames with no detection (occluded) — track should coast, not die
    for _ in range(3):
        tr.update([], short)
    out = tr.update([det(150, 100)], short)
    assert any(t.id == tid for t in out)


def test_track_dies_after_max_misses():
    tr = BallTracker(min_hits=2, max_misses=3)
    short = 400
    tr.update([det(100, 100)], short)
    tr.update([det(100, 100)], short)
    for _ in range(5):
        tr.update([], short)
    assert tr.tracks == []


def test_number_unique_across_tracks():
    """One ball number must never be committed on two live tracks — the
    weaker claimant renders as unknown (measured on real footage: duplicate
    numbers were alive in ~80% of frames before arbitration)."""
    tr = BallTracker(min_hits=2)
    short = 400
    # two well-separated balls both repeatedly detected as number 5; the one
    # at (100,100) accumulates more evidence first
    for _ in range(4):
        tr.update([Detection(100, 100, 10, cls=BallClass.SOLID, number=5)], short)
    for _ in range(3):
        out = tr.update([
            Detection(100, 100, 10, cls=BallClass.SOLID, number=5),
            Detection(300, 300, 10, cls=BallClass.SOLID, number=5),
        ], short)
    fives = [t for t in out if t.number == 5]
    assert len(fives) == 1, f"number 5 alive on {len(fives)} tracks"
    assert len(out) == 2, "the losing track must survive (as unknown), not die"
    assert fives[0].x < 200, "the stronger-evidence track must keep the number"


def test_fast_ball_revival_same_id_no_ghost():
    """A ball that motion-blurs out of detection and reappears far away must
    re-bind to its OWN track (same id, snapped to the new spot) instead of
    freezing a ghost in place and spawning a duplicate — the 'several copies
    of the same ball on the table' report."""
    tr = BallTracker(min_hits=3)
    short = 400
    for _ in range(12):                    # settled, confirmed track
        out = tr.update([det(100, 100)], short)
    tid = out[0].id
    for _ in range(5):                     # blur dropout while thrown
        tr.update([], short)
    out = tr.update([det(240, 160)], short)   # reappears ~1.5 gates away
    assert len(out) == 1                   # ONE ball, no ghost + duplicate
    assert out[0].id == tid
    assert abs(out[0].x - 240) < 8 and abs(out[0].y - 160) < 8


def test_stale_ghost_with_duplicate_number_is_deleted():
    """When ball N is tracked LIVE somewhere on the table, a long-missing track
    still claiming N is a ghost of the same physical ball — it must be removed
    entirely, not merely have its number blanked (the graphic would stay)."""
    tr = BallTracker(min_hits=3, max_misses=6)
    short = 400

    def num_det(x, y):
        return Detection(x=x, y=y, radius=10, cls=BallClass.SOLID, number=5)

    for _ in range(12):                    # settled track for ball 5 at (60,60)
        tr.update([num_det(60, 60)], short)
    # ball reappears far beyond any revival gate (picked up and dropped);
    # the old settled track would normally survive on the occlusion budget
    for _ in range(10):
        out = tr.update([num_det(350, 350)], short)
    fives = [t for t in out if t.number == 5]
    assert len(fives) == 1
    assert abs(fives[0].x - 350) < 8
    assert all(abs(t.x - 60) > 20 or abs(t.y - 60) > 20 for t in out)


def test_coasting_track_bounces_off_cushion_bounds():
    """A fast ball lost to blur right at the rail must not die in open felt:
    the coasted prediction reflects off the cushion-nose bounds, so the
    animated ball bounces AT the rail — not 'in thin air'."""
    tr = BallTracker(min_hits=2)
    short = 400
    bounds = (20.0, 20.0, 780.0, 380.0)
    for x in range(600, 760, 20):          # sweeping right at ~20 px/frame
        out = tr.update([det(float(x), 200.0)], short, bounds=bounds)
    for _ in range(6):                     # blur dropout through the impact
        out = tr.update([], short, bounds=bounds)
    assert len(out) == 1
    t = out[0]
    assert t.x <= 780.0 - t.radius + 1e-6  # never rendered inside the cushion
    assert t.vx < 0                        # rebounded off the right cushion


# --------------------------------------------------------------------------- #
# Rest-frozen identity: "a ball at rest cannot become a different ball" made
# airtight. The corpus flagged resting balls flipping number (#5 -> #9) in the
# one-frame windows where a glare-shifted detection reset the settled bit, and
# rack-time churn where the wobbly FIRST read was committed then corrected.
# --------------------------------------------------------------------------- #

def ndet(x, y, num, cls=BallClass.SOLID, r=10):
    return Detection(x=x, y=y, radius=r, cls=cls, number=num)


def test_resting_ball_identity_survives_glare_streak_and_blip():
    """A long challenger streak (model misreads 5 as 9 under glare) plus a
    single far-shifted detection — the exact combo that used to flip a
    RESTING ball's identity — must change nothing."""
    tr = BallTracker(min_hits=3)
    short = 400
    out = None
    for _ in range(20):
        out = tr.update([ndet(100, 100, 5)], short)
    assert out[0].number == 5
    for i in range(30):
        x = 114 if i == 10 else 100      # one glare-shifted detection
        out = tr.update([ndet(x, 100, 9, cls=BallClass.STRIPE)], short)
        assert out[0].number == 5, "resting ball flipped identity"
        assert out[0].cls == BallClass.SOLID


def test_struck_ball_revotes_at_its_new_spot():
    """The freeze defers the re-vote to real motion — so once the ball is
    genuinely struck and re-read at its new location, the new majority must
    actually win (stale rest-time votes are dropped at the rest->motion
    edge, or a minute of old history would outvote the new spot forever)."""
    tr = BallTracker(min_hits=3)
    short = 400
    for _ in range(20):
        tr.update([ndet(100, 100, 5)], short)
    x, out = 100.0, None
    for _ in range(6):                    # struck: streams away
        x += 20
        out = tr.update([ndet(x, 100, 9, cls=BallClass.STRIPE)], short)
    for _ in range(15):                   # settles at the new spot, reads 9
        out = tr.update([ndet(x, 100, 9, cls=BallClass.STRIPE)], short)
    assert out[0].number == 9, "post-move majority must win the re-vote"


def test_first_identity_waits_for_three_agreeing_reads():
    """Rack-time churn fix: the first wobbly read must NOT be committed and
    then 'corrected' at rest (that correction is an impossible id_flicker to
    the physics scorer). The ball publishes unknown until 3 reads agree."""
    tr = BallTracker(min_hits=3)
    short = 400
    tr.update([ndet(100, 100, 5)], short)          # one wobbly 5-read
    published = []
    out = None
    for _ in range(10):
        out = tr.update([ndet(100, 100, 9, cls=BallClass.STRIPE)], short)
        if out:
            published.append(out[0].number)
    assert out[0].number == 9
    assert 5 not in published, f"wobbly first read was published: {published}"


class TestGhostMergeAndSnapCooldown:
    def test_overlapping_pair_merges_at_published_radii_bar(self):
        """The 0.75-0.79-diameter sliver: published radii run larger than the
        geometric expectation, so a ghost parked inside a real ball cleared
        the old exp_r-based bar. The bar now uses the pair's own radii."""
        tr = BallTracker(min_hits=2)
        short = 400
        for _ in range(6):     # two "balls" 20px apart with fat radii (14):
            out = tr.update([Detection(100, 100, 14, cls=BallClass.SOLID),
                             Detection(120, 100, 14, cls=BallClass.SOLID)],
                            short, ball_r=10.0)
        assert len(out) == 1, "interpenetrating pair must merge to one track"

    def test_real_touching_pair_survives(self):
        """Two REAL touching balls sit at 1.0x the pair diameter — the merge
        bar (0.8x) must leave them alone."""
        tr = BallTracker(min_hits=2)
        short = 400
        for _ in range(6):
            out = tr.update([Detection(100, 100, 14, cls=BallClass.SOLID),
                             Detection(129, 100, 14, cls=BallClass.SOLID)],
                            short, ball_r=10.0)
        assert len(out) == 2, "touching balls are not duplicates"

    def test_no_serial_revival_snapping(self):
        """015737 track 9: a starved track ping-ponged between two far balls
        through the revival gate (five 9.5-diameter teleports in 20s). After
        one beyond-gate snap, a second vanish must NOT snap again — the far
        ball gets a fresh id."""
        tr = BallTracker(min_hits=3, max_misses=30)
        short = 400
        for _ in range(12):                        # settled at A
            out = tr.update([det(100, 100)], short)
        tid = out[0].id
        for _ in range(5):                         # vanish
            tr.update([], short)
        out = tr.update([det(240, 180)], short)    # far B (inside the 0.45s
        assert out and out[0].id == tid            # revival cap): first snap
        for _ in range(3):
            tr.update([det(240, 180)], short)
        for _ in range(5):                         # vanish again immediately
            tr.update([], short)
        out = tr.update([det(120, 300)], short)    # far C, inside cooldown
        ids = {t.id for t in out}
        assert tid not in ids or all(
            abs(t.x - 120) > 30 for t in out if t.id == tid), \
            "serial snap: the old id must not teleport to C"
