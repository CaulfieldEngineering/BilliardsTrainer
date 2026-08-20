import sys
from pathlib import Path

import mido
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.midi_io import to_events
from lib.sections import bar_fingerprints, detect_sections, summarise
from tools.make_fixture import build as build_fixture

TPB = 480


def on(note, tick, vel=100):
    return (tick, mido.Message("note_on", note=note, velocity=vel, channel=9))


def test_identical_bars_share_a_fingerprint():
    events = [on(36, 0), on(36, TPB * 4)]
    bars = bar_fingerprints(events, TPB, 4)
    assert bars[0].fingerprint == bars[1].fingerprint


def test_velocity_is_ignored_by_default():
    """Same pattern played softer is structurally the same bar."""
    a = bar_fingerprints([on(36, 0, 100)], TPB, 4)
    b = bar_fingerprints([on(36, 0, 40)], TPB, 4)
    assert a[0].fingerprint == b[0].fingerprint


def test_detects_the_fixture_arrangement():
    """The fixture is deliberately A A B A. If detection says otherwise the
    phrase-length search has regressed."""
    mid = build_fixture()
    events = []
    for track in mid.tracks:
        events.extend(to_events(track))
    sections = detect_sections(events, mid.ticks_per_beat, 4, channel=9)
    assert [s.label for s in sections] == ["A", "A", "B", "A"]
    assert all(s.length_bars == 4 for s in sections)


def test_detection_survives_small_timing_jitter():
    """Fingerprints quantise to 16ths, so humanisation must not fracture the
    structure."""
    from lib.transforms import humanize

    mid = build_fixture()
    events = []
    for track in mid.tracks:
        events.extend(to_events(track))
    jittered = humanize(events, seed="s", timing_ticks=8)
    sections = detect_sections(jittered, mid.ticks_per_beat, 4, channel=9)
    assert [s.label for s in sections] == ["A", "A", "B", "A"]


def test_empty_input_is_not_a_crash():
    assert detect_sections([], TPB, 4) == []
    assert summarise([]) == "(no sections detected)"


def test_section_dict_is_one_indexed_for_humans():
    mid = build_fixture()
    events = []
    for track in mid.tracks:
        events.extend(to_events(track))
    sections = detect_sections(events, mid.ticks_per_beat, 4, channel=9)
    assert sections[0].as_dict()["start_bar"] == 1
