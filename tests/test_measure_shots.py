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
