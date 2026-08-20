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


class TestColourAdoption:
    """Digit-down identity: a ball whose number never faces the camera must
    still be named — by mature colour consensus — because an unnamed ball is
    invisible to outcome derivation (the 6 went unnamed for a whole session
    and its pot read as a miss). Guards: at rest, long-lived, stable colour,
    solids only, uniqueness."""

    GREEN6 = (128, 161, 18)     # this table's measured 6-ball colour (fixture)

    def _refs(self, monkeypatch, tmp_path):
        import shutil
        from pathlib import Path

        import billiards_trainer.config as cfg
        from billiards_trainer.vision import balls
        fixture = Path(__file__).parent / "fixtures" / "colour_refs.json"
        shutil.copy2(fixture, tmp_path / "colour_refs.json")
        monkeypatch.setattr(cfg, "APP_DIR", tmp_path)
        monkeypatch.setattr(balls, "_MEASURED_REFS", None)
        return balls

    def _settle(self, tr, bgr, n_frames, number=-1, x=100, y=100):
        out = []
        for _ in range(n_frames):
            d = det(x, y)
            d.bgr = bgr
            d.measured_bgr = bgr    # the ensemble sets this from real pixels
            d.number = number
            out = tr.update([d], 400)
        return out

    def test_digit_down_solid_adopts_colour_identity(self, monkeypatch, tmp_path):
        self._refs(monkeypatch, tmp_path)
        tr = BallTracker(min_hits=2, still_frames=3)
        out = self._settle(tr, self.GREEN6, 70)   # never a number vote
        assert out[0].number == 6

    def test_young_track_does_not_adopt(self, monkeypatch, tmp_path):
        self._refs(monkeypatch, tmp_path)
        tr = BallTracker(min_hits=2, still_frames=3)
        out = self._settle(tr, self.GREEN6, 30)   # colour yes, maturity no
        assert out[0].number == -1

    def test_taken_number_is_never_duplicated(self, monkeypatch, tmp_path):
        self._refs(monkeypatch, tmp_path)
        tr = BallTracker(min_hits=2, still_frames=3)
        for _ in range(70):
            real6 = det(300, 300)
            real6.bgr = real6.measured_bgr = self.GREEN6
            real6.number = 6                      # digit-read real 6
            anon = det(100, 100)
            anon.bgr = anon.measured_bgr = self.GREEN6   # same colour, digit down
            anon.number = -1
            out = tr.update([real6, anon], 400)
        nums = sorted(t.number for t in out)
        assert nums.count(6) == 1

    def test_unstable_colour_never_adopts(self, monkeypatch, tmp_path):
        self._refs(monkeypatch, tmp_path)
        tr = BallTracker(min_hits=2, still_frames=3)
        maroon7 = (30, 14, 80)
        out = []
        for i in range(70):                       # alternating colour reads
            d = det(100, 100)
            d.bgr = d.measured_bgr = self.GREEN6 if i % 2 else maroon7
            d.number = -1
            out = tr.update([d], 400)
        assert out[0].number == -1

    def test_moving_track_never_adopts(self, monkeypatch, tmp_path):
        """Pin for 'not t.moving()': identity must never be baptised
        mid-flight (mutation deleting the guard survived the old suite)."""
        self._refs(monkeypatch, tmp_path)
        tr = BallTracker(min_hits=2, still_frames=3)
        out, x = [], 100.0
        for _ in range(70):
            d = det(x, 100)
            d.bgr = d.measured_bgr = self.GREEN6
            d.number = -1
            out = tr.update([d], 400)
            x += 8.0                              # > 0.30*radius every frame
        assert out[0].number == -1

    def test_sparse_colour_samples_never_adopt(self, monkeypatch, tmp_path):
        """Pin for len(colour_hist) >= 25: a mature track whose colour was
        confidently measured only a handful of times must not adopt
        (mutation deleting the floor survived the old suite)."""
        from billiards_trainer.vision.types import BallClass
        self._refs(monkeypatch, tmp_path)
        tr = BallTracker(min_hits=2, still_frames=3)
        out = []
        for i in range(80):
            d = det(100, 100)
            d.number = -1
            if i % 7 == 0:                        # ~11 measured samples
                d.bgr = d.measured_bgr = self.GREEN6
            else:
                d.cls = BallClass.UNKNOWN         # no measurement this frame
            out = tr.update([d], 400)
        assert out[0].number == -1

    def test_non_solid_colours_never_adopt(self, monkeypatch, tmp_path):
        """Pin for '1 <= n <= 7': the 8 (any dark shadow matches its ref)
        and stripe references must not be adoptable by colour (mutation
        deleting the range guard survived the old suite)."""
        self._refs(monkeypatch, tmp_path)
        for bgr in ((39, 21, 16), (153, 146, 73)):   # 8-ref, 14-ref colours
            tr = BallTracker(min_hits=2, still_frames=3)
            out = self._settle(tr, bgr, 70)
            assert out[0].number == -1, f"adopted from {bgr}"

    def test_adopter_beats_stray_misreads_in_at_rest_contest(
            self, monkeypatch, tmp_path):
        """Two RESTING tracks contest one number: the ball whose measured
        colour consensus backs the claim must win over a neighbour that
        collected a few stray misread votes — evidence, not track-list rank
        (review finding: rank resolution stripped the real 6 at rest)."""
        self._refs(monkeypatch, tmp_path)
        tr = BallTracker(min_hits=2, still_frames=3)
        out = []
        for i in range(110):
            neigh = det(300, 300)                 # spawns first: earlier rank
            neigh.bgr = neigh.measured_bgr = (30, 14, 80)   # maroon (a 7)
            # 3 stray '6' misreads arriving AFTER the real 6 adopted (~i=60)
            neigh.number = 6 if i in (75, 80, 85) else -1
            green = det(100, 100)
            green.bgr = green.measured_bgr = self.GREEN6
            green.number = -1
            out = tr.update([neigh, green], 400)
        by_pos = {(round(t.x), round(t.y)): t.number for t in out}
        assert by_pos.get((100, 100)) == 6, f"real 6 lost the contest: {by_pos}"
        assert by_pos.get((300, 300)) != 6


class TestReachabilityGate:
    """A number may only move between tracks at ball speed (Joe's shot-36
    family: the lost 7's number landed on garbage tracks across the table
    within a frame — 401 hops measured library-wide)."""

    def test_number_cannot_teleport_across_the_table(self):
        tr = BallTracker(min_hits=2, still_frames=3)
        d7 = det(100, 100, cls=BallClass.SOLID)
        d7.number = 7
        for _ in range(8):                       # the real 7, settled
            tr.update([d7], 675)
        # the 7's track vanishes; a far blob starts collecting 7-votes
        far = det(600, 1100, cls=BallClass.SOLID)
        far.number = 7
        out = []
        for _ in range(4):                       # within the recent window
            out = tr.update([far], 675)
        thief = next((t for t in out if t.x > 400), None)
        assert thief is not None and thief.number != 7, \
            "the 7 teleported 1100px in 4 frames"

    def test_reemergence_after_a_gap_is_allowed(self):
        tr = BallTracker(min_hits=2, still_frames=3)
        d7 = det(100, 100, cls=BallClass.SOLID)
        d7.number = 7
        for _ in range(8):
            tr.update([d7], 675)
        for _ in range(15):                      # long occlusion, no 7 seen
            tr.update([], 675)
        far = det(600, 1100, cls=BallClass.SOLID)
        far.number = 7
        out = []
        for _ in range(6):
            out = tr.update([far], 675)
        assert any(t.number == 7 for t in out), \
            "re-emergence after a real gap must be allowed"


class TestRestLinking:
    """A numbered ball lost to motion blur keeps its identity where it
    lands — under STRICT uniqueness (one recent flyer death, one fresh
    orphan, number unheld). Joe's shot-36 family, constructive half."""

    def _fly_and_die(self, tr):
        d7 = det(100, 100, cls=BallClass.SOLID)
        d7.number = 7
        for _ in range(8):                       # settled as the 7
            tr.update([d7], 675)
        x = 100.0
        for _ in range(4):                       # struck: real motion
            x += 60.0
            m = det(x, 100, cls=BallClass.SOLID)
            m.number = 7
            tr.update([m], 675)
        for _ in range(26):                      # blur: gone past max_misses
            tr.update([], 675)

    def test_lone_orphan_inherits_the_flyer_number(self):
        tr = BallTracker(min_hits=2, still_frames=3)
        self._fly_and_die(tr)
        landed = det(520, 900, cls=BallClass.SOLID)   # lands far away
        out = []
        for _ in range(6):
            out = tr.update([landed], 675)
        assert any(t.number == 7 for t in out), \
            "the flyer's identity died with its track"

    def test_two_orphans_is_ambiguous_no_link(self):
        tr = BallTracker(min_hits=2, still_frames=3)
        self._fly_and_die(tr)
        a = det(520, 900, cls=BallClass.SOLID)
        b = det(200, 600, cls=BallClass.SOLID)
        out = []
        for _ in range(6):
            out = tr.update([a, b], 675)
        assert not any(t.number == 7 for t in out), \
            "ambiguous landing must stay anonymous"

    def test_reacquired_number_blocks_the_link(self):
        tr = BallTracker(min_hits=2, still_frames=3)
        self._fly_and_die(tr)
        real7 = det(300, 300, cls=BallClass.SOLID)
        real7.number = 7                          # digit read elsewhere
        orphan = det(520, 900, cls=BallClass.SOLID)
        out = []
        for _ in range(8):
            out = tr.update([real7, orphan], 675)
        sevens = [t for t in out if t.number == 7]
        assert len(sevens) == 1 and abs(sevens[0].x - 300) < 20, \
            "the orphan stole a held number"


class TestAppearanceGatedAssociation:
    """The 005647 forensics: identity must survive contact. A numbered track
    must not adopt a detection whose category (cue vs object) contradicts
    it, and the single-cue rebind must not glue the cue onto felt debris."""

    def _settle(self, tr, dets, short, n=12):
        for _ in range(n):
            out = tr.update(dets, short)
        return out

    def test_stop_shot_cue_wins_the_white_detection(self):
        # A settled solid sits at (500,500); the cue rests at (500,470).
        # After contact the ONE white detection lands at (500,498) — 2px
        # closer to the solid's track than to the cue's. Closest-first gave
        # it to the solid (@526s: make inverted into scratch); the class
        # penalty must give it to the cue.
        tr = BallTracker(min_hits=3)
        short = 400
        self._settle(tr, [det(500, 500, cls=BallClass.SOLID, r=15),
                          det(500, 470, cls=BallClass.CUE, r=15)], short)
        out = tr.update([det(500, 498, cls=BallClass.CUE, r=16)], short)
        cue = next(t for t in out if t.cls == BallClass.CUE)
        assert cue.misses == 0, "cue track lost the white detection"
        assert abs(cue.y - 490) < 6, "cue track not moved onto the white det"
        # the solid must NOT have taken it: it either coasts unmatched or
        # (this tight layout) is deduped away as an overlapping duplicate
        solids = [t for t in out if t.cls == BallClass.SOLID]
        assert all(t.misses > 0 for t in solids), \
            "solid track stole the white detection"

    def test_coasting_numbered_track_rejects_cross_category(self):
        # A numbered solid coasts at a pocket; the white cue arrives at its
        # spot (@209s: the 7 adopted the cue and followed it into the
        # pocket). The coasting veto must leave the white detection to
        # spawn/match elsewhere, never re-attach the 7 to it.
        tr = BallTracker(min_hits=3)
        short = 400
        seven = Detection(x=300, y=300, radius=15, cls=BallClass.SOLID,
                          number=7)
        self._settle(tr, [seven], short, n=16)
        tr.update([], short)                     # goes coasting (misses=1)
        out = tr.update([det(302, 302, cls=BallClass.CUE, r=15)], short)
        # The 7 must NOT ride the white ball: the veto leaves the detection
        # to spawn a fresh track; the stale coasting 7 may then be deduped
        # away (number freed for honest re-arbitration) — either way, no
        # ACTIVE track may carry number 7 on the cue's detection.
        assert not any(t.number == 7 and t.misses == 0 for t in out), \
            "coasting numbered ball re-attached across categories"

    def test_single_cue_rebind_rejects_speck(self):
        # The lost cue must NOT bind to a sub-ball white speck (r~10 vs the
        # cue's 15-17) at any distance (@209s: masked the real scratch)...
        tr = BallTracker(min_hits=3)
        short = 400
        self._settle(tr, [det(100, 100, cls=BallClass.CUE, r=16)], short)
        tr.update([], short)
        out = tr.update([det(350, 350, cls=BallClass.CUE, r=10)], short)
        cue = next(t for t in out if t.cls == BallClass.CUE)
        assert cue.misses > 0, "cue bound to a sub-ball speck"

    def test_single_cue_rebind_still_takes_blurred_cue(self):
        # ...but a genuinely fast cue whose blob blur INFLATES the radius
        # must still bind across the table (the rebind's whole purpose).
        tr = BallTracker(min_hits=3)
        short = 400
        self._settle(tr, [det(100, 100, cls=BallClass.CUE, r=16)], short)
        tr.update([], short)
        out = tr.update([det(350, 350, cls=BallClass.CUE, r=20)], short)
        cue = next(t for t in out if t.cls == BallClass.CUE)
        assert cue.misses == 0, "blurred far cue failed to rebind"
        assert cue.x > 150, "rebound cue not moving toward its detection"


class TestCueSizeFloor:
    """005647 @209s: a persistent white felt speck (r10) kept winning the
    freed cue number while the real cue lay in a pocket. The cue is white
    and never reads small, so number 0 demands lifetime full-size evidence.
    Real object balls DO live at r9-11 in places (frame-verified purple 4,
    yellow 1) — the floor applies to number 0 only."""

    def _lose_cue(self, tr, short):
        cue = Detection(x=100, y=100, radius=16, cls=BallClass.CUE, number=0)
        for _ in range(16):
            out = tr.update([cue], short, ball_r=15.0)
        cue_id = next(t.id for t in out if t.number == 0)
        tr.remove_ids([cue_id])          # vacancy killed it (potted)
        return tr

    def test_speck_cannot_win_cue_number(self):
        tr = self._lose_cue(BallTracker(min_hits=3), 400)
        speck = Detection(x=300, y=300, radius=10, cls=BallClass.CUE,
                          number=0)
        out = []
        for _ in range(30):
            out = tr.update([speck], 400, ball_r=15.0)
        assert not any(t.number == 0 for t in out), \
            "chronic sub-ball speck won the cue number"

    def test_real_returning_cue_still_wins_number(self):
        tr = self._lose_cue(BallTracker(min_hits=3), 400)
        ball = Detection(x=300, y=300, radius=15, cls=BallClass.CUE,
                         number=0)
        out = []
        for _ in range(30):
            out = tr.update([ball], 400, ball_r=15.0)
        assert any(t.number == 0 for t in out), \
            "full-size returning cue denied its number"
