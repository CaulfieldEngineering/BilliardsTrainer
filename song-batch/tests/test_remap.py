import mido
import pytest

from lib.drummap import DrumMap
from lib.midi_io import is_note_off, is_note_on, to_events
from lib.remap import remap_events, preview_reverse_table


def make_map(**overrides):
    data = {
        "map_version": 1,
        "unmapped_policy": "keep",
        "entries": [
            {"gm": 38, "to": 70, "articulation": "snare_center"},
            {"gm": 40, "to": 70, "articulation": "snare_center"},
            {"gm": 36, "to": 60, "articulation": "kick"},
        ],
        "articulations": {"snare_ghost": {"to": 71}},
        "conditional": {"rules": []},
    }
    data.update(overrides)
    return DrumMap(data)


def on(note, tick, vel=100):
    return (tick, mido.Message("note_on", note=note, velocity=vel, channel=9))


def off(note, tick):
    return (tick, mido.Message("note_off", note=note, velocity=0, channel=9))


def test_basic_remap():
    out = remap_events([on(36, 0), off(36, 100)], make_map())
    assert [m.note for _, m in out] == [60, 60]


def test_other_channels_untouched():
    events = [(0, mido.Message("note_on", note=36, velocity=100, channel=0))]
    out = remap_events(events, make_map())
    assert out[0][1].note == 36


def test_folding_collision_does_not_cut_the_surviving_voice():
    """GM 38 and GM 40 both land on 70. Overlapped, the first note_off must not
    silence the note that is still ringing."""
    events = [on(38, 0), on(40, 10), off(38, 100), off(40, 200)]
    out = sorted(remap_events(events, make_map()), key=lambda e: e[0])
    offs = [(t, m) for t, m in out if is_note_off(m)]
    assert len(offs) == 1, "only the last voice to release should emit note_off"
    assert offs[0][0] == 200


def test_conditional_rule_binds_target_at_note_on():
    """A velocity-conditional route is chosen from the note_on velocity. The
    note_off carries velocity 0, so re-running the lookup would send it
    somewhere else and strand the note."""
    dm = make_map(conditional={"rules": [
        {"enabled": True, "gm": 38, "when": {"velocity": [1, 34]}, "to_articulation": "snare_ghost"}
    ]})
    out = remap_events([on(38, 0, vel=20), off(38, 100)], dm)
    assert [m.note for _, m in out] == [71, 71]


def test_disabled_rule_is_ignored():
    dm = make_map(conditional={"rules": [
        {"enabled": False, "gm": 38, "when": {"velocity": [1, 34]}, "to_articulation": "snare_ghost"}
    ]})
    out = remap_events([on(38, 0, vel=20)], dm)
    assert out[0][1].note == 70


def test_rule_pointing_at_unfilled_articulation_falls_through():
    """This is what lets the map ship half-verified without breaking builds."""
    dm = make_map(
        articulations={"snare_ghost": {"to": None}},
        conditional={"rules": [
            {"enabled": True, "gm": 38, "when": {"velocity": [1, 34]}, "to_articulation": "snare_ghost"}
        ]},
    )
    out = remap_events([on(38, 0, vel=20)], dm)
    assert out[0][1].note == 70


def test_drop_policy_removes_note_and_its_note_off():
    dm = make_map(unmapped_policy="drop")
    out = remap_events([on(99, 0), off(99, 100), on(36, 200), off(36, 300)], dm)
    assert [m.note for _, m in out] == [60, 60]


def test_keep_policy_passes_unmapped_through():
    out = remap_events([on(99, 0), off(99, 100)], make_map())
    assert [m.note for _, m in out] == [99, 99]


def test_hanging_note_gets_a_synthesised_note_off():
    report_events = remap_events([on(36, 0)], make_map())
    assert sum(1 for _, m in report_events if is_note_off(m)) == 1


def test_orphan_note_off_is_counted_not_crashed_on():
    from lib.remap import RemapReport

    report = RemapReport()
    remap_events([off(38, 0)], make_map(), report=report)
    assert report.orphan_offs == 1


def test_preview_reverse_picks_lowest_gm_source():
    table = preview_reverse_table(make_map())
    assert table[70] == 38  # not 40
    assert table[60] == 36


def test_duplicate_gm_entry_is_rejected():
    from lib.drummap import DrumMapError

    with pytest.raises(DrumMapError, match="mapped twice"):
        DrumMap({"entries": [{"gm": 38, "to": 1}, {"gm": 38, "to": 2}]})
