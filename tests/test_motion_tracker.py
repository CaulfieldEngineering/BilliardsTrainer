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

    def test_emitted_number_survives_vote_oscillation(self):
        # gate round 2: majorities flip 5<->3 at rest; the SHOWN number
        # must hold unless the new read leads 5 straight frames
        tk = MotionTracker()
        for i in range(6):
            rows = _step(tk, [(100, 200, 16, 5)], i / 30)
        seen = set()
        for i in range(6, 30):
            n = 3 if i % 2 == 0 else 5        # alternating misreads
            rows = _step(tk, [(100, 200, 16, n)], i / 30)
            seen.add(next(iter(rows.values())).number)
        assert seen == {5}, f"shown number flickered: {seen}"

    def test_sustained_new_read_flips_only_in_motion(self):
        # superseded expectation: rest-frozen identity outranks
        # hysteresis - a RESTING ball never flips (see the frozen test);
        # a MOVING ball with a sustained new read does re-earn identity
        tk = MotionTracker()
        for i in range(6):
            _step(tk, [(100 + i * 12, 200, 16, 5)], i / 30)
        rows = {}
        for i in range(6, 26):                 # rolling on, consistent 3
            rows = _step(tk, [(100 + i * 12, 200, 16, 3)], i / 30)
        assert next(iter(rows.values())).number == 3

    def test_resting_ball_identity_is_frozen(self):
        # the live tracker's bought rule: sustained misreads at rest
        # (outlasting any hysteresis window) must bounce off
        tk = MotionTracker()
        for i in range(6):
            _step(tk, [(100, 200, 16, 5)], i / 30)
        rows = {}
        for i in range(6, 40):                 # 34 frames of consistent 3
            rows = _step(tk, [(100, 200, 16, 3)], i / 30)
        assert next(iter(rows.values())).number == 5, "at rest = frozen"

    def test_identity_can_change_after_movement(self):
        tk = MotionTracker()
        for i in range(6):
            _step(tk, [(100, 200, 16, 5)], i / 30)
        # the ball moves away (occlusion swap scenario), then reads 3
        rows = {}
        for i in range(6, 30):
            x = 100 + (i - 5) * 12             # rolling
            rows = _step(tk, [(x, 200, 16, 3)], i / 30)
        assert next(iter(rows.values())).number == 3, "moving balls re-earn identity"


class TestStaleClaimRelease:
    """The arbitration deadlock (Joe's trail-less 2-ball, 2026-08-28):
    a saturated-but-stale claim was unbeatable (9 capped votes vs
    'lead by 2'), so a departed ball's number stayed parked on its old
    spot forever. Fresh reads now decide; stale ones don't vote."""

    def test_live_ball_reclaims_number_from_stale_holder(self):
        from billiards_trainer.measure.tracker import MotionTracker
        tk = MotionTracker()
        t = 0.0
        # track A saturates its claim to "2" at (100,100)
        for _ in range(12):
            tk.update([(100.0, 100.0, 14.0, 2)], t)
            t += 1 / 30
        # A's spot keeps a detection but NO number read (ball gone,
        # something else there); the real 2 sits at (400,400) being
        # read as 2 every identifier pass
        for _ in range(120):                     # 4s > FRESH_S
            rows = tk.update([(100.0, 100.0, 14.0, -1),
                              (400.0, 400.0, 14.0, 2)], t)
            t += 1 / 30
        by_pos = {round(r.x): r.number for r in rows}
        assert by_pos[400] == 2, "the freshly-read ball must hold its number"
        assert by_pos[100] == -1, "the stale holder must release it"

    def test_fresh_incumbent_still_beats_fresh_challenger(self):
        # two tracks both being read as the same number (true ambiguity,
        # e.g. neighbor misreads): stickiness must still prevent flicker
        from billiards_trainer.measure.tracker import MotionTracker
        tk = MotionTracker()
        t = 0.0
        for _ in range(12):
            tk.update([(100.0, 100.0, 14.0, 5)], t)
            t += 1 / 30
        for _ in range(30):
            rows = tk.update([(100.0, 100.0, 14.0, 5),
                              (400.0, 400.0, 14.0, 5)], t)
            t += 1 / 30
        by_pos = {round(r.x): r.number for r in rows}
        assert by_pos[100] == 5, "actively-read incumbent keeps the number"
        assert by_pos[400] == -1


class TestPocketFurniture:
    """Bench round 6, vision-verified: a detection on the pocket
    leather scored ABOVE the confidence floor and lived the whole
    session, faking rests and stealing episodes. Confidence can't
    separate it; time can."""

    def _run(self, pos, frames=400, pockets=((20.0, 1200.0),)):
        from billiards_trainer.measure.tracker import MotionTracker
        tk = MotionTracker(pockets=list(pockets), pocket_r=25.0)
        t, rows = 0.0, []
        for k in range(frames):
            rows = tk.update([(*pos(k), 14.0, -1)], t)
            t += 1 / 30
        return rows

    def test_immortal_pocket_blob_dies(self):
        rows = self._run(lambda k: (30.0, 1195.0))      # never moves
        assert rows == [] or all(r.number == -1 for r in rows)
        assert not rows, "pocket furniture must not survive"

    def test_ball_arriving_at_the_jaw_survives(self):
        # born mid-table, rolls INTO the jaw and stays: a real hanger
        def pos(k):
            if k < 100:
                f = k / 100
                return (300 - f * 270, 600 + f * 580)
            return (30.0, 1180.0)
        rows = self._run(pos)
        assert rows, "a real ball that rolled into the jaw must live"

    def test_mid_table_immortal_is_untouched(self):
        rows = self._run(lambda k: (300.0, 600.0))      # a resting ball
        assert rows, "resting balls away from pockets are not furniture"


class TestRestingBallStillGoesInactiveQuickly:
    """Round 70 case law: widening the coast budget for a resting ball
    costs POTS.

    A ball the player stands over has its detections vetoed by the
    foreign mask, and 0.6s of coast is far too short for that - measured
    at 6.3s and 6.7s of occlusion on the two clips. Widening the budget
    to 8s for a CONFIRMED, AT-REST track fixed the blindness exactly as
    designed (bench "no track" 7 -> 1) and the scorecard threw it out:
    bench outcomes 10/10 -> 8/10 and pots 4/4 -> 2/4, cold 9/9 -> 7/9
    and 5/5 -> 3/5.

    The reason is physical: A POTTED BALL IS ALSO A CONFIRMED BALL AT
    REST THAT STOPS BEING DETECTED. It decelerates into the pocket, so
    its last frames read settled, and the ghost then sat on the table
    for 8s with the pot never seen. Any future widening must be
    conditioned on the ball actually being under FOREIGN COVER, which
    needs the mask plumbed into update() - not on rest alone.
    """

    def test_a_resting_ball_that_vanishes_is_dropped_promptly(self):
        from billiards_trainer.measure.tracker import COAST_S, MotionTracker
        tk = MotionTracker()
        t = 0.0
        for _ in range(60):                     # 2s of a settled ball
            rows = tk.update([(300.0, 600.0, 14.0, 3)], t)
            t += 1 / 30
        assert rows and rows[0].number == 3, "the ball should be tracked"
        # now it stops being detected - a pot looks exactly like this
        gone_by = None
        for k in range(120):                    # up to 4s of nothing
            rows = tk.update([], t)
            t += 1 / 30
            if not rows:
                gone_by = (k + 1) / 30.0
                break
        assert gone_by is not None, (
            "a resting ball that stops being detected must go inactive - "
            "if it lingers, a potted ball stays on the table and the pot "
            "is never seen (round 70: 4 pots lost across the two clips)")
        assert gone_by <= COAST_S + 0.5, (
            f"took {gone_by:.2f}s to drop a vanished resting ball; the "
            f"budget is COAST_S={COAST_S}s. Widening this without a "
            f"foreign-cover test costs pots - see the class docstring.")


class TestFastBallStaysOneTrack:
    """Bench round 8: a struck ball covers ~60px/frame; a track born
    this frame has no velocity, so the tight gate missed its own next
    sighting and spawned a new track EVERY frame - the cue ball's path
    rendered as a dozen simultaneous phantom balls."""

    def test_fast_ball_keeps_one_identity(self):
        from billiards_trainer.measure.tracker import MotionTracker
        tk = MotionTracker()
        t, rows = 0.0, []
        # rest, then a hard stroke: 60px per frame across the table
        for _ in range(10):
            tk.update([(150.0, 300.0, 11.0, 0)], t)
            t += 1 / 30
        for k in range(12):
            rows = tk.update([(150.0, 300.0 + 60.0 * (k + 1), 11.0, -1)], t)
            t += 1 / 30
        assert len(rows) == 1, f"fast ball fragmented into {len(rows)} tracks"
        assert rows[0].number == 0, "identity must survive the stroke"

    def test_two_balls_do_not_swap_under_the_wide_gate(self):
        from billiards_trainer.measure.tracker import MotionTracker
        tk = MotionTracker()
        t = 0.0
        for _ in range(10):
            tk.update([(150.0, 300.0, 11.0, 1), (150.0, 380.0, 11.0, 2)], t)
            t += 1 / 30
        rows = tk.update([(150.0, 300.0, 11.0, -1), (150.0, 380.0, 11.0, -1)], t)
        by_y = {round(r.y): r.number for r in rows}
        assert by_y[300] == 1 and by_y[380] == 2, "neighbours swapped"

    def test_accelerating_ball_keeps_its_track(self):
        """THE @85 CASE (round 47), replayed with the measured positions.

        Joe: "20260824-220247@85 is a bank that misses the initial
        transient tail of the 3 ball." The 3 rested at (258.8, 376.1),
        was struck, and its detections ran away down-left. The gate gave
        the wide acquisition window only while speed < 30, so the frame
        AFTER the ball first moved its gate collapsed from 95px to 47.7
        while its own detection - already named 3 - sat 50.1px away. Two
        pixels short: a new track was born on the real ball and the 3's
        track coasted on a dead prediction, so the trail drew a straight
        line across the gap. A ball ACCELERATES off the cue, so last
        frame's speed always under-predicts; the floor must hold through
        the strike, not be withdrawn at the first sign of motion."""
        tk = MotionTracker()
        t = 0.0
        for _ in range(12):                       # at rest, confirmed
            tk.update([(258.8, 376.1, 11.9, 3)], t)
            t += 1 / 30
        # the real measured opening: creep, then hard acceleration
        for x, y in ((250.0, 379.8), (203.5, 402.5), (158.5, 424.2),
                     (113.7, 445.7), (100.3, 452.2)):
            rows = tk.update([(x, y, 11.0, 3)], t)
            t += 1 / 30
        assert len(rows) == 1, (
            f"the struck 3 fragmented into {len(rows)} tracks")
        assert rows[0].number == 3, "the 3 lost its name off the cue"
        assert abs(rows[0].x - 100.3) < 12 and abs(rows[0].y - 452.2) < 12, (
            "the track must be ON the ball, not on a stale prediction")

    def test_a_long_dead_track_cannot_reach_across_the_table(self):
        """The gate's travel term is bounded by the coast window.

        Bought on the bench at 32.9s (round 47): dt_g is time since the
        last real SIGHTING, so a track nothing had matched for fifteen
        seconds grew a gate of hundreds of pixels. A dead nameless blob
        parked at the bottom-LEFT pocket since 17.1s reached 496px across
        the table, adopted a detection in the bottom-RIGHT pocket, and
        published it as a "5" - a ball this table does not have."""
        tk = MotionTracker()
        t = 0.0
        for _ in range(8):                        # a blob, then abandoned
            tk.update([(103.0, 1196.0, 10.7, -1)], t)
            t += 1 / 30
        for _ in range(450):                      # fifteen seconds unseen
            tk.update([], t)
            t += 1 / 30
        rows = tk.update([(599.0, 1181.0, 17.8, 5)], t)
        far = [r for r in rows if r.x > 500]
        assert far, "the far detection must produce a track"
        assert len(rows) >= 2 or not any(r.x < 200 for r in rows), (
            "the parked blob reached across the table and adopted it")
        near = [r for r in rows if r.x < 200]
        assert not near or near[0].id != far[0].id, (
            "one track cannot be in both pockets at once")

    def test_a_named_track_outbids_a_closer_nameless_one(self):
        """THE 170.6s LONG POT (round 47).

        The identifier had already labelled the moving ball `1`. Its own
        track was 32.8px away - but a nameless blob parked in the
        bottom-left pocket sat 28.6px away and won on distance alone by
        4.2px. The 1's track was left coasting, died, and the ball
        entered the pocket unnamed, so a real pot could be attributed to
        no ball at all. The identifier's read is evidence; distance is
        only a guess about which ball this is."""
        tk = MotionTracker()
        t = 0.0
        for _ in range(12):                       # the 1, named and settled
            tk.update([(130.0, 1061.0, 12.6, 1), (103.0, 1196.0, 10.7, -1)], t)
            t += 1 / 30
        for x, y in ((118.0, 1088.0), (101.0, 1131.0)):
            tk.update([(x, y, 13.1, 1), (103.0, 1196.0, 10.7, -1)], t)
            t += 1 / 30
        # the blob goes unseen; the 1 keeps travelling toward the pocket
        rows = tk.update([(83.6, 1171.9, 13.3, 1)], t)
        named = [r for r in rows if r.number == 1]
        assert named, "the named ball lost its identity to a nameless blob"
        assert abs(named[0].x - 83.6) < 12 and abs(named[0].y - 1171.9) < 12, (
            "the 1's own track must hold the ball into the pocket")


class TestACushionBounce:
    """A ball that bounces off a rail must keep its track (round 57).

    A bounce REVERSES the velocity component across the rail, so a
    constant-velocity prediction is at its most wrong exactly there: it
    sails on through the cushion. For a ball running down a long rail
    toward a corner, that means the prediction goes straight into the
    pocket - and a track that dies inside a pocket zone IS the pot
    credit. Measured on the cold clip at 174.5s: the gold ball ran the
    rail, its track predicted on to 2.22 pocket radii from the
    bottom-right and died there, and a shot in which nothing was potted
    was scored a make on a ball the table does not have. The ball had
    simply bounced and come back up.
    """

    def _tk(self):
        from billiards_trainer.measure.tracker import MotionTracker
        # bed corners at (60,60)-(600,1200), like the real table
        return MotionTracker(pockets=[(60.0, 60.0), (600.0, 60.0),
                                      (600.0, 1200.0), (60.0, 1200.0)],
                             pocket_r=25.0)

    def test_a_ball_bouncing_off_the_end_rail_keeps_its_track(self):
        tk = self._tk()
        t, y = 0.0, 1000.0
        for _ in range(10):                    # running down toward the rail
            tk.update([(300.0, y, 13.0, 4)], t)
            y += 40.0
            t += 1 / 30
        rows = tk.update([(300.0, 1195.0, 13.0, 4)], t)   # at the cushion
        t += 1 / 30
        assert rows, "the ball at the cushion must be tracked"
        tid = min(rows, key=lambda r: abs(r.y - 1195.0)).id
        y = 1160.0
        for _ in range(6):                     # and back UP the table
            rows = tk.update([(300.0, y, 13.0, 4)], t)
            y -= 40.0
            t += 1 / 30
        same = [r for r in rows if r.id == tid]
        assert same, "the bouncing ball lost its track at the cushion"
        assert same[0].y < 1100.0, "the track did not follow the ball back out"

    def test_an_on_bed_prediction_is_not_mirrored(self):
        """The reflection must only apply when the prediction left the bed."""
        tk = self._tk()
        mid = type("T", (), {"x": 300.0, "y": 600.0, "radius": 13.0})()
        assert tk._bounced(mid) is None
        # the cushion a BALL CENTRE meets is one radius short of the bed
        # bound: 1200 - 13 = 1187, so 1260 reflects to 1187 - 73 = 1114
        past = type("T", (), {"x": 300.0, "y": 1260.0, "radius": 13.0})()
        got = tk._bounced(past)
        assert got is not None and abs(got[1] - 1114.0) < 1e-6, got
