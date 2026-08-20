"""Absolute-time MIDI event handling.

mido gives you delta times, which are miserable to transform: insert one note
and every subsequent delta is wrong. Every transform in this library instead
works on a list of ``(abs_tick, Message)`` pairs and converts back at the end.

The only subtle part is re-serialising. At a single tick the order of messages
matters: a note_off for a note must precede a note_on that re-strikes the same
note, or the synth swallows the second hit. :func:`to_track` enforces that
while otherwise preserving the original relative order of events (stable sort),
which keeps diffs clean.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Tuple

import mido

Event = Tuple[int, mido.Message]

# Sort priority within a single tick. Lower goes first.
#   meta/setup first, then note_off, then note_on.
# This ordering is what stops a re-struck note from being cut by the previous
# note's note_off landing after it.
_PRIORITY_META = 0
_PRIORITY_NOTE_OFF = 10
_PRIORITY_OTHER = 20
_PRIORITY_NOTE_ON = 30


def is_note_off(msg: mido.Message) -> bool:
    """True for a real note_off *or* the note_on-with-velocity-0 idiom."""
    if msg.type == "note_off":
        return True
    return msg.type == "note_on" and msg.velocity == 0


def is_note_on(msg: mido.Message) -> bool:
    """True only for a note_on that actually sounds (velocity > 0)."""
    return msg.type == "note_on" and msg.velocity > 0


def _priority(msg: mido.Message) -> int:
    if msg.is_meta:
        return _PRIORITY_META
    if is_note_off(msg):
        return _PRIORITY_NOTE_OFF
    if is_note_on(msg):
        return _PRIORITY_NOTE_ON
    return _PRIORITY_OTHER


def to_events(track: Iterable[mido.Message]) -> List[Event]:
    """Delta-time track -> absolute-tick event list."""
    events: List[Event] = []
    now = 0
    for msg in track:
        now += msg.time
        events.append((now, msg))
    return events


def to_track(events: Sequence[Event]) -> mido.MidiTrack:
    """Absolute-tick event list -> delta-time track.

    Stable-sorts by ``(tick, priority, original_index)`` so that equal events
    keep the order they came in with. Messages are copied, so the input list is
    left untouched.
    """
    indexed = sorted(
        ((tick, _priority(msg), i, msg) for i, (tick, msg) in enumerate(events)),
        key=lambda item: (item[0], item[1], item[2]),
    )
    track = mido.MidiTrack()
    prev = 0
    for tick, _prio, _i, msg in indexed:
        delta = tick - prev
        if delta < 0:  # pragma: no cover - impossible after the sort
            raise ValueError(f"negative delta at tick {tick}")
        track.append(msg.copy(time=delta))
        prev = tick
    return track


def iter_notes(events: Sequence[Event]) -> Iterator[Tuple[int, mido.Message]]:
    """Yield only the sounding note_on events, in time order."""
    for tick, msg in events:
        if is_note_on(msg):
            yield tick, msg


def load(path: Path | str) -> mido.MidiFile:
    return mido.MidiFile(str(path))


def save(mid: mido.MidiFile, path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(out))
    return out


def clone_empty(mid: mido.MidiFile) -> mido.MidiFile:
    """A new MidiFile with the same type/division but no tracks."""
    return mido.MidiFile(type=mid.type, ticks_per_beat=mid.ticks_per_beat)


def map_tracks(mid: mido.MidiFile, fn) -> mido.MidiFile:
    """Apply ``fn(events) -> events`` to every track, returning a new file."""
    out = clone_empty(mid)
    for track in mid.tracks:
        new_events = fn(to_events(track))
        new_track = to_track(new_events)
        new_track.name = track.name
        out.tracks.append(new_track)
    return out


def ticks_per(mid: mido.MidiFile, beats: float) -> int:
    """Tick count for a number of quarter notes. Rounds to nearest tick."""
    return int(round(mid.ticks_per_beat * beats))


def describe(mid: mido.MidiFile) -> dict:
    """Cheap summary used by the CLI and by tests as a regression fingerprint."""
    notes = 0
    channels = set()
    pitches = set()
    for track in mid.tracks:
        for msg in track:
            if is_note_on(msg):
                notes += 1
                channels.add(msg.channel)
                pitches.add(msg.note)
    return {
        "type": mid.type,
        "ticks_per_beat": mid.ticks_per_beat,
        "tracks": len(mid.tracks),
        "notes": notes,
        "channels": sorted(channels),
        "pitches": sorted(pitches),
        "length_ticks": max(
            (sum(m.time for m in t) for t in mid.tracks), default=0
        ),
    }
