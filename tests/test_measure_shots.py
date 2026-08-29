"""The box's shot/outcome stages (pure, feed-agnostic).

Case law pinned: a ball that RESTS after the stroke is a miss even if
it later vanishes (Joe's 3-ball was picked up post-settle and the old
path scored the pickup as a make)."""

from billiards_trainer.measure.shots import analyze

FPS = 30.0


def _stream(spec, t0=10.0, n_frames=600):
    """spec: {number: fn(k)->(x,y) or None}. Builds (times, frames)."""
    times, frames = [], []
    for k in range(n_frames):
        t = t0 + k / FPS
        rows = []
        for n, fn in spec.items():
            p = fn(k)
            if p is not None:
                rows.append((100 + n, p[0], p[1], 14.0, n, 2 if n else 0, 1))
        times.append(t)
        frames.append(rows)
    return times, frames


def _rest(x, y):
    return lambda k: (x, y)


def _roll_then_rest(k0, k1, x0, y0, x1, y1):
    """Rolls between frames k0..k1, rests at both ends."""
    def fn(k):
        if k < k0:
            return (x0, y0)
        if k >= k1:
            return (x1, y1)
        f = (k - k0) / (k1 - k0)
        return (x0 + f * (x1 - x0), y0 + f * (y1 - y0))
    return fn


def _roll_then_vanish(k0, k1, x0, y0, x1, y1):
    fn = _roll_then_rest(k0, k1, x0, y0, x1, y1)
    return lambda k: None if k >= k1 else fn(k)


class TestEpisodes:
    def test_quiet_table_has_no_episodes(self):
        times, frames = _stream({0: _rest(100, 100), 5: _rest(300, 300)})
        assert analyze(times, frames) == []

    def test_one_stroke_one_episode_with_backdated_strike(self):
        times, frames = _stream({0: _roll_then_rest(60, 150, 100, 100, 400, 400),
                                 5: _rest(300, 600)})
        eps = analyze(times, frames)
        assert len(eps) == 1
        assert abs(eps[0].t_strike - (10.0 + 60 / FPS - 0.3)) < 0.15
        assert 0 in eps[0].movers and 5 not in eps[0].movers


class TestOutcomes:
    def test_ball_resting_at_settle_is_not_pocketed(self):
        times, frames = _stream({0: _roll_then_rest(60, 100, 100, 100, 250, 250),
                                 3: _roll_then_rest(70, 200, 300, 600, 120, 460)})
        ep = analyze(times, frames)[0]
        assert 3 in ep.resting and not ep.pocketed

    def test_pickup_after_settle_is_still_not_pocketed(self):
        # THE 3-BALL CASE: rests at k=200, vanishes at k=380 (pickup)
        roll = _roll_then_rest(70, 200, 300, 600, 120, 460)
        times, frames = _stream({0: _roll_then_rest(60, 100, 100, 100, 250, 250),
                                 3: lambda k: None if k >= 380 else roll(k)},
                                n_frames=420)
        ep = analyze(times, frames)[0]
        assert 3 in ep.resting
        assert not ep.pocketed, "a post-settle pickup must never score"

    def test_vanish_moving_at_a_pocket_is_pocketed(self):
        times, frames = _stream({0: _roll_then_rest(60, 100, 100, 100, 250, 250),
                                 9: _roll_then_vanish(70, 160, 300, 600, 20, 20)})
        ep = analyze(times, frames, pockets=[(15, 15)])[0]
        assert [p[0] for p in ep.pocketed] == [9] and not ep.lost

    def test_vanish_moving_mid_table_is_lost_never_scored(self):
        # the tracker-loss/pickup-occlusion case: dies far from pockets
        times, frames = _stream({0: _roll_then_rest(60, 100, 100, 100, 250, 250),
                                 9: _roll_then_vanish(70, 160, 300, 600, 320, 400)})
        ep = analyze(times, frames, pockets=[(15, 15), (650, 15)])[0]
        assert not ep.pocketed and [p[0] for p in ep.lost] == [9]

    def test_no_pockets_means_no_credit_ever(self):
        times, frames = _stream({0: _roll_then_rest(60, 100, 100, 100, 250, 250),
                                 9: _roll_then_vanish(70, 160, 300, 600, 20, 20)})
        ep = analyze(times, frames)[0]
        assert not ep.pocketed and [p[0] for p in ep.lost] == [9]

    def test_cue_vanish_at_pocket_is_scratch(self):
        times, frames = _stream({0: _roll_then_vanish(60, 150, 100, 100, 600, 700),
                                 5: _rest(300, 600)})
        ep = analyze(times, frames, pockets=[(610, 710)])[0]
        assert ep.scratch and not ep.pocketed


class TestPathThroughPocket:
    def test_coast_through_pocket_credited(self):
        # bench: the potted 2 coasted THROUGH the mouth and died beyond
        # the bed - the death point misses the zone but the path does not
        def fn(k):
            if k < 60:
                return (300, 600)
            if k >= 160:
                return None
            f = (k - 60) / 100
            return (300 - f * 340, 600 + f * 660)   # dies at (-40, 1260)
        times, frames = _stream({0: _roll_then_rest(60, 100, 100, 100, 250, 250),
                                 2: fn})
        ep = analyze(times, frames, pockets=[(20, 1200)], pocket_r=25)[0]
        assert [p[0] for p in ep.pocketed] == [2]

    def test_coast_past_far_from_pocket_still_lost(self):
        def fn(k):
            if k < 60:
                return (300, 600)
            if k >= 160:
                return None
            f = (k - 60) / 100
            return (300 + f * 200, 600)             # dies mid-felt path
        times, frames = _stream({0: _roll_then_rest(60, 100, 100, 100, 250, 250),
                                 2: fn})
        ep = analyze(times, frames, pockets=[(20, 1200)], pocket_r=25)[0]
        assert not ep.pocketed and [p[0] for p in ep.lost] == [2]


class TestChainContinuity:
    def test_furniture_rebirth_does_not_unmake_a_pot(self):
        # mover dies in the pocket; 2s later a static read near the
        # pocket gets the same number - must NOT count as resting
        def fn(k):
            if k < 60:
                return (300, 600)
            if 60 <= k < 160:
                f = (k - 60) / 100
                return (300 - f * 290, 600 + f * 610)   # into (10,1210)
            if k >= 220:
                return (95, 1150)                        # furniture rebirth
            return None                                  # 2s hole
        times, frames = _stream({0: _roll_then_rest(60, 100, 100, 100, 250, 250),
                                 2: fn}, n_frames=400)
        ep = analyze(times, frames, pockets=[(20, 1200)], pocket_r=25)[0]
        assert [p[0] for p in ep.pocketed] == [2], "rebirth faked a rest"

    def test_continuous_jaw_rattle_rest_still_a_miss(self):
        # ball rattles the jaws and STAYS, continuously tracked into
        # rest near the mouth - genuinely on-table, not a pot
        def fn(k):
            if k < 60:
                return (300, 600)
            if k < 160:
                f = (k - 60) / 100
                return (300 - f * 230, 600 + f * 560)   # to (70,1160)
            return (70, 1160)                            # rests at the jaw
        times, frames = _stream({0: _roll_then_rest(60, 100, 100, 100, 250, 250),
                                 2: fn}, n_frames=400)
        ep = analyze(times, frames, pockets=[(20, 1200)], pocket_r=25)[0]
        assert not ep.pocketed and 2 in ep.resting


class TestLipHover:
    def test_lip_hover_then_chain_death_is_a_pot(self):
        # bench round 4: the potted ball creeps below the motion
        # threshold AT the mouth, then its chain dies - that's a drop
        def fn(k):
            if k < 60:
                return (300, 600)
            if k < 150:
                f = (k - 60) / 90
                return (300 - f * 270, 600 + f * 570)   # to (30, 1170)
            if k < 175:
                return (30 - (k - 150) * 0.3, 1170 + (k - 150) * 0.8)
            return None                                  # chain dies at mouth
        times, frames = _stream({0: _roll_then_rest(60, 100, 100, 100, 250, 250),
                                 9: fn}, n_frames=400)
        ep = analyze(times, frames, pockets=[(20, 1200)], pocket_r=25)[0]
        assert [p[0] for p in ep.pocketed] == [9], "lip hover + death = drop"

    def test_lip_hover_surviving_chain_still_resting(self):
        # the true rattle-and-stay: hovers at the mouth and KEEPS being
        # tracked - resting, not potted
        def fn(k):
            if k < 60:
                return (300, 600)
            if k < 150:
                f = (k - 60) / 90
                return (300 - f * 270, 600 + f * 570)
            return (30, 1170)                            # tracked forever
        times, frames = _stream({0: _roll_then_rest(60, 100, 100, 100, 250, 250),
                                 9: fn}, n_frames=400)
        ep = analyze(times, frames, pockets=[(20, 1200)], pocket_r=25)[0]
        assert not ep.pocketed and 9 in ep.resting


class TestUnnamedMovers:
    def test_unnamed_fast_roll_into_pocket_is_a_make(self):
        # bench: the potted 5 crossed the table as an unnamed track
        def unnamed(k):
            if k < 60:
                return (500, 300)
            if k >= 170:
                return None
            f = (k - 60) / 110
            return (500 - f * 470, 300 + f * 880)   # long roll to (30,1180)
        times, frames = [], []
        for k in range(400):
            t = 10.0 + k / 30.0
            rows = [(100, 250.0, 250.0, 14.0, 0, 0, 1)]
            p = unnamed(k)
            if p is not None:
                rows.append((77, p[0], p[1], 14.0, -1, 5, 1))
            times.append(t)
            frames.append(rows)
        ep = analyze(times, frames, pockets=[(20, 1200)], pocket_r=25,
                     unnamed_pots=True)[0]
        assert ep.pocketed and ep.pocketed[0][0] < 0, "unnamed pot must score"

    def test_slow_unnamed_blob_never_counts(self):
        # glove-like: dwells near the pocket, small span, slow
        times, frames = [], []
        for k in range(400):
            t = 10.0 + k / 30.0
            rows = [(100, 250.0, 250.0, 14.0, 0, 0, 1)]
            if 60 <= k < 200:
                rows.append((88, 40 + (k - 60) * 0.5, 1180, 14.0, -1, 5, 1))
            times.append(t)
            frames.append(rows)
        eps = analyze(times, frames, pockets=[(20, 1200)], pocket_r=25)
        assert all(not e.pocketed for e in eps)


class TestCueDefinesAStroke:
    """Round 16: a stroke means the cue ball moved. Balls TOSSED onto
    the table roll free and are indistinguishable from a shot by any
    hand-adjacency test - but the cue never moves for them."""

    def test_cue_motion_marks_a_stroke(self):
        times, frames = _stream({0: _roll_then_rest(60, 150, 100, 100, 400, 400),
                                 3: _rest(300, 600)})
        ep = analyze(times, frames)[0]
        assert ep.cue_moved is True

    def test_tossed_ball_alone_is_not_a_stroke(self):
        # object ball rolls the table; the cue never moves
        times, frames = _stream({0: _rest(100, 100),
                                 3: _roll_then_rest(60, 160, 300, 200, 300, 900)})
        eps = analyze(times, frames)
        assert eps and eps[0].cue_moved is False


class TestCueTravelGate:
    """Round 16: Joe nudges the cue while addressing; real strokes send
    it 263px+ on the bench, a nudge 8-29px."""

    def test_stroke_records_real_travel(self):
        times, frames = _stream({0: _roll_then_rest(60, 150, 100, 100, 500, 500)})
        ep = analyze(times, frames)[0]
        assert ep.cue_travel > 150

    def test_nudge_records_tiny_travel(self):
        times, frames = _stream({0: _roll_then_rest(60, 90, 100, 100, 130, 100),
                                 3: _roll_then_rest(60, 160, 300, 200, 300, 800)})
        ep = analyze(times, frames)[0]
        assert ep.cue_travel < 150
