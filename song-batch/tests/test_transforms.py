import mido
import pytest

from lib.drummap import DrumMap
from lib.midi_io import is_note_on
from lib.transforms import add_ghost_notes, humanize, resolve_chokes, scale_velocity, shape_velocity, swing

TPB = 480


def dm():
    return DrumMap({
        "unmapped_policy": "keep",
        "entries": [
            {"gm": 38, "to": 38, "articulation": "snare_center"},
            {"gm": 42, "to": 42, "articulation": "hihat_closed"},
            {"gm": 46, "to": 46, "articulation": "hihat_open"},
        ],
        "articulations": {"snare_ghost": {"to": 71}},
        "choke_groups": {"hihat": ["hihat_closed", "hihat_open"]},
    })


def on(note, tick, vel=100):
    return (tick, mido.Message("note_on", note=note, velocity=vel, channel=9))


def off(note, tick):
    return (tick, mido.Message("note_off", note=note, velocity=0, channel=9))


def backbeat(bars=2):
    """Snare on 2 and 4 across whole bars - ghost notes only fill within the
    span of existing material, so the fixture has to be bar-length."""
    events = []
    for bar in range(bars):
        for beat in (1, 3):
            tick = bar * TPB * 4 + beat * TPB
            events += [on(38, tick), off(38, tick + 60)]
    return events


def four_on_the_floor(bars=2):
    events = []
    for bar in range(bars):
        for beat in range(4):
            tick = bar * TPB * 4 + beat * TPB
            events += [on(36, tick), off(36, tick + 120)]
    return events


# ------------------------------------------------------------------ humanize

def test_humanize_is_deterministic():
    events = four_on_the_floor()
    a = humanize(events, seed="song-a", timing_ticks=10, velocity_spread=20)
    b = humanize(events, seed="song-a", timing_ticks=10, velocity_spread=20)
    assert [(t, m.note, m.velocity) for t, m in a] == [(t, m.note, m.velocity) for t, m in b]


def test_humanize_differs_between_seeds():
    events = four_on_the_floor()
    a = humanize(events, seed="song-a", timing_ticks=10, velocity_spread=20)
    b = humanize(events, seed="song-b", timing_ticks=10, velocity_spread=20)
    assert [t for t, _ in a] != [t for t, _ in b]


def test_humanize_is_stable_when_an_unrelated_note_is_inserted():
    """The whole point of hashing note identity instead of running an RNG:
    adding one note must not reshuffle every other note's jitter, or a one-note
    edit produces a whole-file diff."""
    base = four_on_the_floor()
    a = humanize(base, seed="s", timing_ticks=10, velocity_spread=20)
    inserted = base + [on(38, 5 * TPB), off(38, 5 * TPB + 120)]
    b = humanize(inserted, seed="s", timing_ticks=10, velocity_spread=20)

    def fingerprint(events):
        return sorted((t, m.note, m.velocity) for t, m in events if is_note_on(m) and m.note == 36)

    assert fingerprint(a) == fingerprint(b)


def test_humanize_moves_note_off_with_its_note_on():
    events = [on(36, 1000), off(36, 1120)]
    out = humanize(events, seed="s", timing_ticks=20)
    ticks = [t for t, _ in out]
    assert ticks[1] - ticks[0] == 120  # duration intact


def test_humanize_never_produces_illegal_velocity():
    events = [on(36, 0, vel=126), on(36, 480, vel=2)]
    out = humanize(events, seed="s", velocity_spread=60)
    assert all(1 <= m.velocity <= 127 for _, m in out if is_note_on(m))


def test_humanize_never_produces_negative_ticks():
    out = humanize([on(36, 2), off(36, 10)], seed="s", timing_ticks=50)
    assert all(t >= 0 for t, _ in out)


def test_humanize_noop_returns_input_unchanged():
    events = four_on_the_floor()
    assert humanize(events, seed="s") == list(events)


# ------------------------------------------------------------ shape_velocity

def test_shape_velocity_accents_downbeat_and_ducks_offbeats():
    events = [on(42, 0, 100), on(42, TPB // 4, 100), on(42, TPB, 100)]
    out = shape_velocity(events, TPB, {"downbeat": 1.2, "offbeat": 0.5}, subdivision=4)
    by_tick = {t: m.velocity for t, m in out}
    assert by_tick[0] == 120           # bar downbeat
    assert by_tick[TPB // 4] == 50     # 16th offbeat
    assert by_tick[TPB] == 100         # on a beat, no rule -> untouched


def test_shape_velocity_exact_slot_beats_wildcard():
    events = [on(42, TPB // 4, 100)]
    out = shape_velocity(events, TPB, {"1.1": 2.0, "offbeat": 0.1}, subdivision=4)
    assert out[0][1].velocity == 127  # 200 clamped, not 10


def test_shape_velocity_clamps():
    out = shape_velocity([on(42, 0, 100)], TPB, {"downbeat": 10.0}, subdivision=4)
    assert out[0][1].velocity == 127


# --------------------------------------------------------------- ghost notes

def test_ghost_notes_uses_dedicated_articulation_when_available():
    events = backbeat()
    out = add_ghost_notes(events, TPB, dm(), seed="s", density=1.0)
    assert any(m.note == 71 for _, m in out if is_note_on(m))


def test_ghost_notes_never_doubles_an_existing_hit():
    tick = TPB // 4  # an offbeat 16th that is already occupied
    events = [on(38, 0), off(38, 60), on(38, tick), off(38, tick + 60)]
    out = add_ghost_notes(events, TPB, dm(), seed="s", density=1.0)
    at_tick = [m for t, m in out if t == tick and is_note_on(m)]
    assert len(at_tick) == 1


def test_ghost_notes_only_land_off_the_beat():
    events = backbeat()
    out = add_ghost_notes(events, TPB, dm(), seed="s", density=1.0)
    ghosts = [t for t, m in out if is_note_on(m) and m.note == 71]
    assert ghosts and all(t % TPB != 0 for t in ghosts)


def test_ghost_notes_are_deterministic():
    events = backbeat()
    a = add_ghost_notes(events, TPB, dm(), seed="s", density=0.5)
    b = add_ghost_notes(events, TPB, dm(), seed="s", density=0.5)
    assert [(t, m.note, m.velocity) for t, m in a] == [(t, m.note, m.velocity) for t, m in b]


def test_ghost_notes_noop_when_no_snare_present():
    events = four_on_the_floor()
    assert add_ghost_notes(events, TPB, dm(), seed="s", density=1.0) == list(events)


# -------------------------------------------------------------------- chokes

def test_open_hat_is_cut_when_the_hat_closes():
    events = [on(46, 0), on(42, 240)]
    out = resolve_chokes(events, dm())
    cut = [(t, m) for t, m in out if m.type == "note_off" and m.note == 46]
    assert cut and cut[0][0] == 240


def test_choke_does_not_fire_on_a_repeat_of_the_same_note():
    events = [on(42, 0), on(42, 240)]
    out = resolve_chokes(events, dm())
    assert not [m for _, m in out if m.type == "note_off"]


# --------------------------------------------------------------------- swing

def test_swing_delays_offbeats_only():
    events = [on(42, 0), on(42, TPB // 2), on(42, TPB)]
    out = swing(events, TPB, amount=1.0, subdivision=8)
    ticks = sorted(t for t, _ in out)
    assert ticks[0] == 0
    assert ticks[1] > TPB // 2
    assert ticks[2] == TPB


def test_swing_zero_is_a_noop():
    events = [on(42, 0), on(42, TPB // 2)]
    assert swing(events, TPB, amount=0.0) == list(events)


def test_scale_velocity_targets_specific_notes():
    events = [on(36, 0, 100), on(38, 0, 100)]
    out = scale_velocity(events, 0.5, notes=[36])
    by_note = {m.note: m.velocity for _, m in out}
    assert by_note == {36: 50, 38: 100}
