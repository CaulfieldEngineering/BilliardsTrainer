"""Assemble per-target MIDI into one Cubase-ready session file.

The output is a Type 1 MIDI file that carries everything Cubase needs to build
the session on import:

* **Track 0 is the conductor track** - tempo, time signature, and one marker per
  arrangement section ("Verse 1", "Chorus"). Cubase reads these into the Tempo
  and Marker tracks.
* **Every other track is one named instrument track**, named to match the Cubase
  template so the imported tracks line up with what is already there.

This is the last artifact that needs no Cubase running and no format guesswork -
a Type 1 MIDI file with named tracks, tempo and markers is bog-standard and
every DAW made in the last thirty years reads it. The `.dawproject` writer that
supersedes this can carry routing, colours and device chains too, but it is an
interchange format with a lossy round trip, so it has to be validated on a
throwaway project first. This does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import mido

from .midi_io import Event, to_events, to_track
from .sections import Section


@dataclass
class NamedTrack:
    """One instrument track destined for the session file."""

    name: str
    events: List[Event]
    channel: Optional[int] = None


def marker_events(
    sections: Sequence[Section],
    ticks_per_beat: int,
    beats_per_bar: int,
) -> List[Event]:
    """One marker meta event at the downbeat of each section."""
    bar_ticks = ticks_per_beat * beats_per_bar
    events: List[Event] = []
    for section in sections:
        tick = section.start_bar * bar_ticks
        events.append((tick, mido.MetaMessage("marker", text=section.label)))
    return events


def conductor_track(
    sections: Sequence[Section],
    ticks_per_beat: int,
    beats_per_bar: int,
    bpm: Optional[float],
    time_signature: Sequence[int] = (4, 4),
    name: str = "Conductor",
) -> mido.MidiTrack:
    """Tempo + time signature + section markers, as track 0."""
    events: List[Event] = [(0, mido.MetaMessage("track_name", name=name))]
    if bpm:
        events.append((0, mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm))))
    events.append(
        (0, mido.MetaMessage(
            "time_signature",
            numerator=int(time_signature[0]),
            denominator=int(time_signature[1]),
        ))
    )
    events.extend(marker_events(sections, ticks_per_beat, beats_per_bar))
    return to_track(events)


def instrument_track(track: NamedTrack) -> mido.MidiTrack:
    """One named instrument track, with meta events from the source stripped.

    Tempo and time signature belong on the conductor track only; leaving copies
    on an instrument track makes Cubase's import ambiguous about which wins.
    Track names are set here, not inherited.
    """
    events = [
        (tick, msg)
        for tick, msg in track.events
        if not (msg.is_meta and msg.type in ("set_tempo", "time_signature", "track_name", "marker"))
    ]
    out = to_track([(0, mido.MetaMessage("track_name", name=track.name))] + events)
    return out


def assemble(
    tracks: Sequence[NamedTrack],
    sections: Sequence[Section],
    ticks_per_beat: int,
    beats_per_bar: int = 4,
    bpm: Optional[float] = None,
    time_signature: Sequence[int] = (4, 4),
    conductor_name: str = "Conductor",
) -> mido.MidiFile:
    """Build the importable session file."""
    mid = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)
    mid.tracks.append(
        conductor_track(sections, ticks_per_beat, beats_per_bar, bpm, time_signature, conductor_name)
    )
    for track in tracks:
        mid.tracks.append(instrument_track(track))
    return mid


def tracks_from_file(mid: mido.MidiFile, name: str) -> List[NamedTrack]:
    """Flatten a built target file into a single named track.

    A target's transforms operate per-track, but a target is conceptually one
    Cubase track, so its tracks are merged before naming.
    """
    merged: List[Event] = []
    for track in mid.tracks:
        merged.extend(to_events(track))
    return [NamedTrack(name=name, events=merged)]
