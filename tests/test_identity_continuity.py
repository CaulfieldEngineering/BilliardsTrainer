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
