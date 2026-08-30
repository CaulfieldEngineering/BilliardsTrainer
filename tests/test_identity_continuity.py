"""Identity must not outlive its ball, and a guess must not outrank a read.

Round 30/31 case law, all three bought with measurements on the bench clip:

  * the finder's colour heuristic called the yellow STRIPED 9 a "1" in 43 of
    43 samples while the trained identifier read it 9 in 71 of 72 - and the
    engine applied the identifier ONLY where the heuristic had not already
    guessed, so the correct read was discarded every frame;
  * a track was never deleted, so the yellow 1's track followed it into a
    pocket at 32.2s and then latched onto the RED 3 at 109.2s, which answered
    to "1" for the rest of the clip;
  * a coasting track's own predicted drift set `ever_moved`, exempting the
    pocket leather from the furniture rule that exists to kill it - and the
    resulting sighting/coast/re-snap cycle read as a ball moving at 500-1700
    px/s, opening a shot window 6 seconds early.
"""

from billiards_trainer.measure.engine import _pair_identities
from billiards_trainer.measure.tracker import RETIRE_S, MotionTracker


class _Det:
    def __init__(self, x, y, radius=13.0, number=-1):
        self.x, self.y, self.radius, self.number = x, y, radius, number


class TestIdentifierOutranksTheHeuristic:
    def test_identifier_read_replaces_a_heuristic_guess(self):
        """THE 9-as-1 case: the heuristic guessed, the identifier knows."""
        found = [_Det(100.0, 100.0, number=1)]      # heuristic says "yellow"
        _pair_identities(found, [(100.0, 100.0, 9)])
        assert found[0].number == 9

    def test_identifier_still_names_an_unguessed_find(self):
        found = [_Det(100.0, 100.0, number=-1)]
        _pair_identities(found, [(100.0, 100.0, 3)])
        assert found[0].number == 3

    def test_heuristic_guess_survives_where_the_identifier_is_silent(self):
        """The heuristic exists for balls the identifier cannot see."""
        found = [_Det(100.0, 100.0, number=4)]
        _pair_identities(found, [])
        assert found[0].number == 4

    def test_one_read_feeds_one_find(self):
        found = [_Det(100.0, 100.0, number=-1), _Det(112.0, 100.0, number=-1)]
        _pair_identities(found, [(100.0, 100.0, 2)])
        assert [d.number for d in found] == [2, -1]


POCKET = (60.0, 60.0)


def _tracker():
    return MotionTracker(pockets=[POCKET, (600.0, 60.0), (60.0, 1200.0),
                                  (600.0, 1200.0)], pocket_r=25.0)


def _rows(tk, dets, t0=0.0, n=6, step=1 / 30.0):
    """Feed the same detections n times and return the last frame's rows."""
    t, rows = t0, []
    for _ in range(n):
        rows = tk.update(dets, t)
        t += step
    return rows


def _feed(tk, dets, t0=0.0, n=12, step=1 / 30.0):
    t = t0
    for _ in range(n):
        tk.update(dets, t)
        t += step
    return t


class TestANameDoesNotOutliveItsBall:
    def test_a_potted_ball_releases_its_number(self):
        """Roll a named ball into the pocket, let it die, then put a NEW
        ball on the far side of the table. It must not inherit the name."""
        tk = _tracker()
        t = 0.0
        # a ball rolling toward the pocket, named 1 all the way
        x, y = 300.0, 300.0
        for _ in range(30):
            tk.update([(x, y, 13.0, 1)], t)
            x -= 8.0
            y -= 8.0
            t += 1 / 30.0
        # it vanishes at the pocket; nothing is seen for well past RETIRE_S
        for _ in range(int((RETIRE_S + 2.0) * 30)):
            tk.update([], t)
            t += 1 / 30.0
        # a different ball appears far away, with no identity of its own
        rows = None
        for _ in range(12):
            rows = tk.update([(400.0, 900.0, 13.0, -1)], t)
            t += 1 / 30.0
        assert rows, "the new ball should be tracked"
        assert all(r.number != 1 for r in rows), \
            "a potted ball's name latched onto a different ball"

    def test_an_occluded_ball_mid_felt_keeps_its_identity(self):
        """The bridge hand closing over a ball is not a pot. Retiring it
        splits the episode and the stroke is never seen."""
        tk = _tracker()
        t = _feed(tk, [(300.0, 600.0, 13.0, 3)], n=30)
        for _ in range(int((RETIRE_S + 2.0) * 30)):      # hidden by a hand
            tk.update([], t)
            t += 1 / 30.0
        rows = None
        for _ in range(12):
            rows = tk.update([(300.0, 600.0, 13.0, -1)], t)
            t += 1 / 30.0
        assert any(r.number == 3 for r in rows), \
            "a ball that was merely hidden lost its identity"


class TestOnlyASightingProvesMovement:
    def test_a_coasting_track_is_not_credited_with_moving(self):
        """The pocket leather drifts while coasting; that drift must not
        count as movement, or it escapes the furniture rule."""
        tk = _tracker()
        t = 0.0
        # Furniture: seen repeatedly at ONE spot in the pocket zone, but
        # given a velocity by detection jitter so its coast drifts it.
        px, py = POCKET[0] + 20.0, POCKET[1] + 20.0
        for k in range(8):
            tk.update([(px + (k % 2) * 3.0, py, 11.0, -1)], t)
            t += 1 / 30.0
        born = dict(tk._tracks)
        assert born, "expected a track on the furniture"
        for _ in range(18):               # coast: no detections at all
            tk.update([], t)
            t += 1 / 30.0
        drifted = max(
            ((tr.x - tr.bx) ** 2 + (tr.y - tr.by) ** 2) ** 0.5
            for tr in tk._tracks.values()) if tk._tracks else 0.0
        for tr in tk._tracks.values():
            assert tr.ever_moved is False, (
                f"a coast of {drifted:.0f}px was counted as the ball moving; "
                "that exempts pocket leather from the furniture rule")


class TestTheLiveTrackerContract:
    """Round 38: one track type, one tracker API.

    MotionTracker now publishes the shared core.types.Track (it used to
    emit a private six-field row while the live path had its own richer
    type - two shapes for one idea), and carries the three methods the
    live pipeline drives its tracker with. This is the groundwork for
    deleting the second tracker entirely."""

    def test_it_publishes_the_shared_track_type(self):
        from billiards_trainer.core.types import Track
        tk = _tracker()
        rows = _rows(tk, [(300.0, 600.0, 13.0, 3)], n=6)
        assert rows and all(isinstance(r, Track) for r in rows)

    def test_a_published_track_carries_what_both_consumers_read(self):
        tk = _tracker()
        rows = _rows(tk, [(300.0, 600.0, 13.0, 3)], n=6)
        r = rows[0]
        for attr in ("id", "x", "y", "radius", "vx", "vy", "cls", "number",
                     "bgr", "age", "hits", "misses", "active", "history",
                     "coasting"):
            assert hasattr(r, attr), f"published track is missing {attr}"
        assert r.history, "trails need position history"

    def test_a_coasted_row_is_flagged_as_an_estimate(self):
        tk = _tracker()
        t = 0.0
        for _ in range(8):
            tk.update([(300.0, 600.0, 13.0, 3)], t)
            t += 1 / 30.0
        seen = tk.update([(300.0, 600.0, 13.0, 3)], t)
        assert all(not r.coasting for r in seen)
        t += 1 / 30.0
        coasted = tk.update([], t)          # nothing detected: predict
        assert coasted and all(r.coasting for r in coasted)

    def test_remove_ids_kills_a_track(self):
        tk = _tracker()
        rows = _rows(tk, [(300.0, 600.0, 13.0, 3)], n=6)
        tk.remove_ids([rows[0].id])
        assert not tk.update([], 1.0)

    def test_release_numbers_frees_the_name_without_killing_the_ball(self):
        """005048 @233: a resting track held number 4 for a whole shot, so
        the real 4 could never be named. Killing it is dangerous; letting
        go of the name is not."""
        tk = _tracker()
        rows = _rows(tk, [(300.0, 600.0, 13.0, 4)], n=8)
        assert any(r.number == 4 for r in rows)
        tk.release_numbers([rows[0].id])
        after = tk.update([(300.0, 600.0, 13.0, -1)], 1.0)
        assert after, "the ball must survive losing its name"
        assert all(r.number != 4 for r in after)

    def test_reset_forgets_everything(self):
        tk = _tracker()
        _rows(tk, [(300.0, 600.0, 13.0, 3)], n=6)
        tk.reset()
        assert not tk.update([], 5.0)


class TestOneSightingIsNotAnIdentity:
    """Round 36. A track's FIRST number is shown with no delay, which is
    right for a real ball - but it also let a single detection name
    itself and then coast. Both of the bench's remaining invented
    numbers were exactly that: 18 rows, ONE real sighting, 17 coasted,
    never moving, asserting a number read from one frame."""

    def test_a_single_detection_never_asserts_a_number(self):
        from billiards_trainer.measure.tracker import MIN_ID_FRAMES
        assert MIN_ID_FRAMES >= 2, "one frame must not be enough"
        tk = _tracker()
        t = 0.0
        rows = tk.update([(300.0, 600.0, 13.0, 8)], t)      # seen once...
        assert all(r.number < 0 for r in rows)
        for _ in range(10):                                  # ...then coasts
            t += 1 / 30.0
            rows = tk.update([], t)
            assert all(r.number < 0 for r in rows), \
                "a one-frame blob named itself and coasted on the name"

    def test_a_real_ball_is_named_almost_immediately(self):
        from billiards_trainer.measure.tracker import MIN_ID_FRAMES
        tk = _tracker()
        t = 0.0
        rows = []
        for _ in range(MIN_ID_FRAMES + 1):
            rows = tk.update([(300.0, 600.0, 13.0, 3)], t)
            t += 1 / 30.0
        assert any(r.number == 3 for r in rows), \
            "a genuine ball must clear this bar in a fraction of a second"
        assert MIN_ID_FRAMES <= 5, "the delay must stay imperceptible"


class TestARestingBallCanStillBeCorrected:
    """Round 33. A resting ball must not FLICKER, but it must not be
    permanently uncorrectable either.

    The rest-freeze used to reset the pending counter, so evidence
    against a resting track's name could never accumulate: the name it
    held when it settled was final. The bench's static 9 took "1" during
    a 6-frame lapse at 156.3 and wore it for the next 80 seconds while
    the identifier read 9 underneath in 114 of 114 frames."""

    def _settled(self, tk, number, n=40, t0=0.0):
        t = t0
        for _ in range(n):
            tk.update([(300.0, 600.0, 13.0, number)], t)
            t += 1 / 30.0
        return t

    def test_a_brief_misread_still_bounces_off(self):
        tk = _tracker()
        t = self._settled(tk, 9)
        for _ in range(8):                     # a short burst of nonsense
            tk.update([(300.0, 600.0, 13.0, 1)], t)
            t += 1 / 30.0
        rows = tk.update([(300.0, 600.0, 13.0, 9)], t)
        assert any(r.number == 9 for r in rows), \
            "a resting ball flipped on a brief misread"

    def test_a_sustained_correction_lands(self):
        from billiards_trainer.measure.tracker import REST_HYST_K
        tk = _tracker()
        t = self._settled(tk, 1)               # settled on the WRONG name
        rows = []
        for _ in range(REST_HYST_K + 15):      # unanimous, sustained truth
            rows = tk.update([(300.0, 600.0, 13.0, 9)], t)
            t += 1 / 30.0
        assert any(r.number == 9 for r in rows), \
            "a resting ball could never be corrected - the freeze was permanent"


class TestATrackCannotComeBackFromTheDead:
    """A dead track object must not sit in the table forever.

    Round 50, measured on the bench: track id3 WAS the real yellow 1,
    tracked 10.9s-33.2s and potted at 31.7. It died mid-felt, 7.4 pocket
    radii from anything, so RETIRE_S - which only removes a track that
    ended in a pocket or off the bed - never touched it. It then came
    back from the dead twice: once 20 seconds later, and once 154
    SECONDS later, when it seized 11 frames of the moving 2 diving into
    the bottom-right and the shot was scored a MAKE that "potted the 1".
    An occlusion lasts a second. Nothing on this table is occluded for
    two and a half minutes.
    """

    def test_a_track_lost_mid_felt_is_forgotten(self):
        from billiards_trainer.measure.tracker import FORGET_S, MotionTracker
        tk = MotionTracker(pockets=[(0.0, 0.0), (600.0, 1200.0)],
                           pocket_r=25.0)
        t = 0.0
        for i in range(12):                       # a real, MOVING ball
            tk.update([(300.0 + i * 20, 600.0, 12.0, 1)], t)
            t += 1 / 30
        assert any(r.number == 1 for r in tk.tracks), "the 1 must exist"
        for _ in range(int((FORGET_S + 2.0) * 30)):   # it vanishes mid-felt
            tk.update([], t)
            t += 1 / 30
        assert not tk.tracks, "a track gone far longer than any occlusion survived"

    def test_the_forgotten_name_cannot_seize_another_ball(self):
        """The bench failure itself: the dead 1 grabbing the moving 2."""
        from billiards_trainer.measure.tracker import FORGET_S, MotionTracker
        tk = MotionTracker(pockets=[(0.0, 0.0), (600.0, 1200.0)],
                           pocket_r=25.0)
        t = 0.0
        for i in range(12):
            tk.update([(300.0 + i * 20, 600.0, 12.0, 1)], t)
            t += 1 / 30
        for _ in range(int((FORGET_S + 2.0) * 30)):
            tk.update([], t)
            t += 1 / 30
        rows = tk.update([(545.0, 605.0, 12.0, 2)], t)   # a DIFFERENT ball
        assert all(r.number != 1 for r in rows), (
            "a ball that left the table minutes ago claimed a live ball")


class TestARestingBallIsNotStolenByAPasserBy:
    """TRIED AND REVERTED (round 55) - kept as case law, not as a rule.

    A name-mismatch veto (a settled, confidently named track refuses any
    detection the identifier calls something else) changed nothing
    measurable on either clip, and on the case that motivated it it is
    actively harmful: during the 159.4s collision on
    session-20260823-185550 the identifier mislabels BOTH balls - the
    white cue reads "5" and the orange 5 reads "3" - so the veto would
    stop the 5's track from re-acquiring its own ball exactly when it
    matters. Names are what BREAKS in a collision; they cannot guard it.
    The colour veto below survives because colour does not swap.
    """


class TestABallDoesNotChangeColour:
    """The second half of the theft veto (round 55).

    The name veto could not save the orange 5: the cue ball that stole
    its track was moving too fast for the identifier to name, so the
    detection arrived unnamed and there was no mismatch to see. Colour
    cannot go missing that way - measured on that frame, orange
    (11,86,238) against white (183,234,238) is 227 apart, while a
    misread of the same ball sits under 40.
    """

    class _Det:
        def __init__(self, x, y, bgr, number=-1, radius=13.0):
            self.x, self.y, self.radius, self.number = x, y, radius, number
            self.measured_bgr = bgr

    def test_a_white_ball_cannot_take_an_orange_balls_track(self):
        from billiards_trainer.measure.tracker import MotionTracker
        tk = MotionTracker()
        t = 0.0
        for _ in range(20):
            tk.update([self._Det(100.0, 900.0, (11, 86, 238), 5)], t)
            t += 1 / 30
        five = [r for r in tk.tracks if r.number == 5]
        assert five, "the orange 5 must be established"
        tid = five[0].id
        for x in (140.0, 152.0, 164.0):       # the cue sweeps past, UNNAMED
            rows = tk.update([self._Det(x, 860.0, (183, 234, 238), -1)], t)
            t += 1 / 30
        stolen = [r for r in rows if r.id == tid and abs(r.x - 100.0) > 25]
        assert not stolen, "a white ball took the orange ball's track"

    def test_the_same_ball_is_still_matched_when_the_name_flickers(self):
        """Colour must not strand a ball over an identifier hiccup."""
        from billiards_trainer.measure.tracker import MotionTracker
        tk = MotionTracker()
        t = 0.0
        for _ in range(20):
            tk.update([self._Det(100.0, 900.0, (11, 86, 238), 5)], t)
            t += 1 / 30
        rows = tk.update([self._Det(118.0, 900.0, (14, 92, 233), -1)], t)
        assert any(abs(r.x - 118.0) < 8 for r in rows), (
            "the ball's own slightly-shifted detection was refused")
