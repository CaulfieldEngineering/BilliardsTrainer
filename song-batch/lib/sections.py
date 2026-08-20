"""Infer arrangement structure from repetition in the MIDI itself.

Suno exports have no markers - just a wall of bars. But an arrangement leaves
fingerprints: the same 8 bars recurring is a chorus, a bar with three times the
usual note count at the end of a phrase is a fill.

This module reads that structure back out so ``spec.yaml`` can be pre-filled and
markers can be written into the ``.mid`` / ``.dawproject``. It is a *proposal*,
not a truth - the operator edits the result in spec.yaml and that wins.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .midi_io import Event, is_note_on

PHRASE_CANDIDATES = (16, 8, 4, 2)


@dataclass
class Bar:
    index: int          # 0-based
    fingerprint: str
    note_count: int


@dataclass
class Section:
    label: str          # "A", "B", ...
    start_bar: int      # 0-based, inclusive
    end_bar: int        # 0-based, exclusive
    fingerprint: str
    is_fill: bool = False

    @property
    def length_bars(self) -> int:
        return self.end_bar - self.start_bar

    def marker_name(self, occurrence: Optional[int] = None) -> str:
        base = f"{self.label}"
        if occurrence is not None:
            base = f"{base}{occurrence}"
        return base

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "start_bar": self.start_bar + 1,  # spec.yaml is 1-indexed for humans
            "length_bars": self.length_bars,
            "fill": self.is_fill,
        }


def bar_fingerprints(
    events: Sequence[Event],
    ticks_per_beat: int,
    beats_per_bar: int = 4,
    velocity_buckets: int = 0,
    channel: Optional[int] = None,
) -> List[Bar]:
    """One fingerprint per bar, derived from note positions within the bar.

    Velocity is ignored by default: two bars playing the same pattern with
    different dynamics are structurally the same bar. Set ``velocity_buckets``
    to fold coarse dynamics into the fingerprint if you want them distinguished.
    """
    bar_ticks = ticks_per_beat * beats_per_bar
    if bar_ticks <= 0:
        return []

    grid = ticks_per_beat / 4  # quantise to 16ths for fingerprinting only
    buckets: Dict[int, List[Tuple[int, int, int]]] = {}
    last_bar = -1

    for tick, msg in events:
        if not is_note_on(msg):
            continue
        if channel is not None and getattr(msg, "channel", None) != channel:
            continue
        # Quantise BEFORE deciding which bar the note belongs to. A humanised
        # note sitting a few ticks behind a bar line belongs to the bar it was
        # played *for*; bucketing it into the previous bar would fracture both
        # bars' fingerprints and destroy the repeat structure.
        quantised = int(round(tick / grid)) * grid
        bar = int(quantised // bar_ticks)
        slot = int(round((quantised % bar_ticks) / grid))
        vel = (msg.velocity * velocity_buckets // 128) if velocity_buckets else 0
        buckets.setdefault(bar, []).append((slot, msg.note, vel))
        last_bar = max(last_bar, bar)

    total_bars = last_bar + 1
    bars: List[Bar] = []
    for index in range(total_bars):
        items = sorted(buckets.get(index, []))
        payload = ";".join(f"{s}:{n}:{v}" for s, n, v in items)
        digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=6).hexdigest()
        bars.append(Bar(index=index, fingerprint=digest, note_count=len(items)))
    return bars


def _best_phrase_length(bars: Sequence[Bar]) -> int:
    """Longest candidate phrase length that both fits and actually repeats."""
    prints = [b.fingerprint for b in bars]
    for length in PHRASE_CANDIDATES:
        if len(prints) < length * 2:
            continue
        chunks = [tuple(prints[i : i + length]) for i in range(0, len(prints), length)]
        full = [c for c in chunks if len(c) == length]
        if len(full) - len(set(full)) > 0:  # at least one repeat
            return length
    return min(4, max(1, len(prints)))


def detect_sections(
    events: Sequence[Event],
    ticks_per_beat: int,
    beats_per_bar: int = 4,
    channel: Optional[int] = None,
    fill_threshold: float = 1.6,
) -> List[Section]:
    """Segment the performance into labelled, repeating sections.

    Strategy: pick the phrase length that repeats most convincingly, chunk the
    bar fingerprints at that length, then hand each distinct chunk a letter in
    order of first appearance. Chunks that differ only in their final bar are
    still given distinct labels, but that final bar is flagged as a fill - which
    is usually exactly what it is.
    """
    bars = bar_fingerprints(events, ticks_per_beat, beats_per_bar, channel=channel)
    if not bars:
        return []

    length = _best_phrase_length(bars)
    mean_notes = sum(b.note_count for b in bars) / len(bars) if bars else 0.0

    sections: List[Section] = []
    labels: Dict[Tuple[str, ...], str] = {}
    next_label = ord("A")

    for start in range(0, len(bars), length):
        chunk = bars[start : start + length]
        key = tuple(b.fingerprint for b in chunk)
        if key not in labels:
            labels[key] = chr(next_label)
            next_label += 1
            if next_label > ord("Z"):
                next_label = ord("A")  # wrap; pathological input only
        last = chunk[-1]
        is_fill = bool(mean_notes) and last.note_count > mean_notes * fill_threshold
        sections.append(
            Section(
                label=labels[key],
                start_bar=chunk[0].index,
                end_bar=chunk[-1].index + 1,
                fingerprint=hashlib.blake2b("".join(key).encode(), digest_size=4).hexdigest(),
                is_fill=is_fill,
            )
        )

    return sections


def summarise(sections: Sequence[Section]) -> str:
    """A one-line arrangement map, e.g. ``A A B A* | 4 bars each, 16 total``."""
    if not sections:
        return "(no sections detected)"
    parts = [s.label + ("*" if s.is_fill else "") for s in sections]
    lengths = {s.length_bars for s in sections}
    length_note = f"{lengths.pop()} bars each" if len(lengths) == 1 else "varying length"
    total = sum(s.length_bars for s in sections)
    return f"{' '.join(parts)}  |  {length_note}, {total} bars total  (* = fill)"
