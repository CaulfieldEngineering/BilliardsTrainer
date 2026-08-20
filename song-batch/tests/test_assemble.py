import sys
from pathlib import Path

import mido
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.assemble import NamedTrack, assemble, conductor_track, marker_events
from lib.sections import Section
from lib.spec import Spec
from lib.template import Template

TPB = 480


def sec(label, start_bar, length=4):
    return Section(label=label, start_bar=start_bar, end_bar=start_bar + length, fingerprint="")


def note(tick, n=36):
    return (tick, mido.Message("note_on", note=n, velocity=100, channel=9))


def test_markers_land_on_section_downbeats():
    events = marker_events([sec("Intro", 0), sec("Chorus", 8)], TPB, 4)
    assert [t for t, _ in events] == [0, 8 * TPB * 4]
    assert [m.text for _, m in events] == ["Intro", "Chorus"]


def test_conductor_carries_tempo_meter_and_markers():
    track = conductor_track([sec("Verse 1", 0)], TPB, 4, bpm=148, time_signature=(7, 8))
    types = {m.type for m in track}
    assert {"track_name", "set_tempo", "time_signature", "marker"} <= types
    tempo = next(m for m in track if m.type == "set_tempo")
    assert round(mido.tempo2bpm(tempo.tempo)) == 148
    meter = next(m for m in track if m.type == "time_signature")
    assert (meter.numerator, meter.denominator) == (7, 8)


def test_conductor_is_track_zero_and_instruments_follow():
    mid = assemble(
        [NamedTrack("Drums SSD", [note(0)]), NamedTrack("Bass", [note(0, 40)])],
        [sec("Intro", 0)], TPB, bpm=120,
    )
    assert [t.name for t in mid.tracks] == ["Conductor", "Drums SSD", "Bass"]
    assert mid.type == 1


def test_instrument_tracks_do_not_carry_competing_tempo_meta():
    """Tempo lives on the conductor track only - a stray copy on an instrument
    track makes the DAW's import ambiguous about which one wins."""
    noisy = [
        (0, mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(90))),
        (0, mido.MetaMessage("time_signature", numerator=3, denominator=4)),
        (0, mido.MetaMessage("track_name", name="Wrong Name")),
        note(0),
    ]
    mid = assemble([NamedTrack("Drums SSD", noisy)], [sec("Intro", 0)], TPB, bpm=148)
    drums = mid.tracks[1]
    assert drums.name == "Drums SSD"
    assert not [m for m in drums if m.type in ("set_tempo", "time_signature")]
    tempo = next(m for m in mid.tracks[0] if m.type == "set_tempo")
    assert round(mido.tempo2bpm(tempo.tempo)) == 148


def test_notes_survive_assembly():
    mid = assemble([NamedTrack("Drums SSD", [note(0), note(TPB)])], [], TPB, bpm=120)
    assert sum(1 for m in mid.tracks[1] if m.type == "note_on" and m.velocity > 0) == 2


def test_empty_section_list_still_produces_a_valid_file():
    mid = assemble([NamedTrack("Drums SSD", [note(0)])], [], TPB, bpm=120)
    assert len(mid.tracks) == 2
    assert not [m for m in mid.tracks[0] if m.type == "marker"]


# ------------------------------------------------------------------ template

def test_template_orders_tracks_by_mixconsole_position():
    template = Template.load()
    assert template.order_key("drums") < template.order_key("bass")
    assert template.order_key("unknown-role") > template.order_key("bass")


def test_template_names_a_target_from_its_role():
    assert Template.load().track_name_for("drums") == "Drums SSD"


def test_unknown_role_falls_back_to_the_target_name():
    assert Template.load().track_name_for("theremin") == "theremin"


# ---------------------------------------------------------- declared sections

SPEC_BODY = """slug: s
title: S
status:
  stage: idea
musical:
  bpm: 120
  time_signature: [4, 4]
template:
  revision: 1
sections:
  - label: Intro
    start_bar: 1
    length_bars: 4
  - label: Chorus
    start_bar: 9
    length_bars: 8
"""


def test_declared_sections_convert_from_one_indexed_bars(tmp_path):
    d = tmp_path / "s"
    d.mkdir()
    (d / "spec.yaml").write_text(SPEC_BODY)
    sections = Spec.load(d / "spec.yaml").declared_sections()
    assert [s.label for s in sections] == ["Intro", "Chorus"]
    assert sections[0].start_bar == 0     # spec.yaml bar 1 -> 0-based 0
    assert sections[1].start_bar == 8     # spec.yaml bar 9 -> 0-based 8
    assert sections[1].length_bars == 8


def test_sections_without_a_label_are_ignored(tmp_path):
    d = tmp_path / "s"
    d.mkdir()
    (d / "spec.yaml").write_text(SPEC_BODY + "  - start_bar: 20\n")
    assert len(Spec.load(d / "spec.yaml").declared_sections()) == 2
