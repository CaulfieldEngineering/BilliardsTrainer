#!/usr/bin/env python3
"""Generate a stand-in for a Suno MIDI drum export.

Suno's MIDI is grid-locked and dynamically flat, which is exactly the thing the
transform library exists to fix. This fixture reproduces those properties on
purpose: dead-on-grid, near-uniform velocity, GM note numbers, one tempo, and a
simple A A B A' structure with fills so section detection has something to find.

Used by the tests and by songs/example-riff so the pipeline runs end to end
before any real export exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mido

TPB = 480
CH = 9

KICK, SNARE, HAT_CLOSED, HAT_OPEN, CRASH = 36, 38, 42, 46, 49
TOM_HIGH, TOM_MID, TOM_FLOOR = 50, 47, 43

# 16 slots per bar (16th notes). 1 = hit.
PATTERN_A = {
    KICK:       "1000001000100000",
    SNARE:      "0000100000001000",
    HAT_CLOSED: "1010101010101010",
}
PATTERN_B = {
    KICK:       "1001001000100100",
    SNARE:      "0000100000001000",
    HAT_CLOSED: "1000100010001000",
    HAT_OPEN:   "0010001000100010",
}
FILL = {
    SNARE:      "1010000000000000",
    TOM_HIGH:   "0000101000000000",
    TOM_MID:    "0000000010100000",
    TOM_FLOOR:  "0000000000001010",
}


def _bar(events, bar_index, pattern, velocity=100):
    slot = TPB // 4
    base = bar_index * TPB * 4
    for note, mask in pattern.items():
        for i, char in enumerate(mask):
            if char != "1":
                continue
            tick = base + i * slot
            # Suno-flat: only the crudest accent, no human variation.
            vel = velocity if i % 4 == 0 else velocity - 8
            events.append((tick, mido.Message("note_on", note=note, velocity=vel, channel=CH)))
            events.append((tick + slot // 2, mido.Message("note_off", note=note, velocity=0, channel=CH)))


def build(bpm: float = 148.0, bars: int = 16) -> mido.MidiFile:
    mid = mido.MidiFile(type=1, ticks_per_beat=TPB)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    meta.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    meta.append(mido.MetaMessage("track_name", name="Tempo", time=0))
    mid.tracks.append(meta)

    events = []
    # A A B A', 4 bars each, fill on the last bar of each phrase.
    plan = ["A", "A", "B", "A"]
    for phrase, kind in enumerate(plan):
        pattern = PATTERN_A if kind == "A" else PATTERN_B
        for offset in range(4):
            bar = phrase * 4 + offset
            if bar >= bars:
                break
            if offset == 3:
                _bar(events, bar, FILL, velocity=108)
            else:
                _bar(events, bar, pattern)
        # Crash on the downbeat of each new phrase.
        events.append((phrase * 4 * TPB * 4, mido.Message("note_on", note=CRASH, velocity=112, channel=CH)))
        events.append((phrase * 4 * TPB * 4 + TPB, mido.Message("note_off", note=CRASH, velocity=0, channel=CH)))

    from lib.midi_io import to_track

    track = to_track(events)
    track.name = "Drums"
    mid.tracks.append(track)
    return mid


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "songs/example-riff/suno/drums.mid")
    out.parent.mkdir(parents=True, exist_ok=True)
    build().save(str(out))
    print(f"wrote {out}")
