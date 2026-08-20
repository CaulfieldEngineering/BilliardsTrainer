"""Tempo map and time-signature helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import mido

from .midi_io import Event, to_events, to_track, clone_empty


@dataclass(frozen=True)
class TempoChange:
    tick: int
    bpm: float


def tempo_map(mid: mido.MidiFile) -> List[TempoChange]:
    """Every tempo change in the file, in time order. Empty means 120 implied."""
    changes: List[TempoChange] = []
    for track in mid.tracks:
        for tick, msg in to_events(track):
            if msg.is_meta and msg.type == "set_tempo":
                changes.append(TempoChange(tick, round(mido.tempo2bpm(msg.tempo), 4)))
    return sorted(changes, key=lambda c: c.tick)


def initial_bpm(mid: mido.MidiFile, default: float = 120.0) -> float:
    changes = tempo_map(mid)
    return changes[0].bpm if changes else default


def time_signature(mid: mido.MidiFile) -> Tuple[int, int]:
    """First time signature in the file, defaulting to 4/4."""
    for track in mid.tracks:
        for msg in track:
            if msg.is_meta and msg.type == "time_signature":
                return msg.numerator, msg.denominator
    return 4, 4


def beats_per_bar(mid: mido.MidiFile) -> int:
    """Quarter-note beats per bar, derived from the time signature.

    7/8 gives 3.5 quarter notes; we round up rather than return a float because
    every grid helper downstream works in whole beats. Odd meters are a known
    rough edge - see CLAUDE.md.
    """
    num, den = time_signature(mid)
    return max(1, int(round(num * 4 / den)))


def set_single_tempo(mid: mido.MidiFile, bpm: float) -> mido.MidiFile:
    """Strip every tempo change and stamp one tempo at tick 0.

    This is the "normalise" step: Suno exports sometimes carry a drifting tempo
    map, and a song that will be tracked to a click wants exactly one tempo.
    """
    tempo = mido.bpm2tempo(bpm)
    out = clone_empty(mid)
    for index, track in enumerate(mid.tracks):
        events: List[Event] = [
            (tick, msg)
            for tick, msg in to_events(track)
            if not (msg.is_meta and msg.type == "set_tempo")
        ]
        if index == 0:
            events.insert(0, (0, mido.MetaMessage("set_tempo", tempo=tempo, time=0)))
        new_track = to_track(events)
        new_track.name = track.name
        out.tracks.append(new_track)
    return out


def set_time_signature(mid: mido.MidiFile, numerator: int, denominator: int) -> mido.MidiFile:
    out = clone_empty(mid)
    for index, track in enumerate(mid.tracks):
        events = [
            (tick, msg)
            for tick, msg in to_events(track)
            if not (msg.is_meta and msg.type == "time_signature")
        ]
        if index == 0:
            events.insert(
                0,
                (0, mido.MetaMessage("time_signature", numerator=numerator, denominator=denominator, time=0)),
            )
        new_track = to_track(events)
        new_track.name = track.name
        out.tracks.append(new_track)
    return out
