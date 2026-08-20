import mido
import pytest

from lib.midi_io import describe, is_note_off, is_note_on, to_events, to_track


def test_roundtrip_preserves_timing():
    track = mido.MidiTrack([
        mido.Message("note_on", note=36, velocity=100, time=0),
        mido.Message("note_off", note=36, velocity=0, time=120),
        mido.Message("note_on", note=38, velocity=90, time=360),
    ])
    assert [m.time for m in to_track(to_events(track))] == [0, 120, 360]


def test_note_off_sorts_before_note_on_at_same_tick():
    """A re-struck note must see its predecessor's note_off first, or the
    second hit gets swallowed by the synth."""
    events = [
        (0, mido.Message("note_on", note=42, velocity=100)),
        (480, mido.Message("note_on", note=42, velocity=100)),
        (480, mido.Message("note_off", note=42, velocity=0)),
    ]
    types = [(m.type, m.velocity) for m in to_track(events)]
    assert types == [("note_on", 100), ("note_off", 0), ("note_on", 100)]


def test_note_on_velocity_zero_counts_as_note_off():
    assert is_note_off(mido.Message("note_on", note=36, velocity=0))
    assert not is_note_on(mido.Message("note_on", note=36, velocity=0))


def test_describe_counts_only_sounding_notes():
    mid = mido.MidiFile(ticks_per_beat=480)
    mid.tracks.append(mido.MidiTrack([
        mido.Message("note_on", note=36, velocity=100, channel=9, time=0),
        mido.Message("note_on", note=36, velocity=0, channel=9, time=120),
    ]))
    assert describe(mid)["notes"] == 1
