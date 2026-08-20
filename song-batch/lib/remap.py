"""GM -> SSD note remapping.

Looks trivial ("rewrite msg.note") and isn't. Three things bite:

1. **note_off must follow its note_on.** Velocity-conditional routing means the
   target depends on the *note_on* velocity, but a note_off carries a different
   (usually zero) velocity. Re-running the lookup on the note_off can route it
   to a different note than the one that is actually sounding, leaving a stuck
   note. We remember the target chosen at note_on and reuse it.

2. **Folding creates collisions.** GM 38 and GM 40 both land on SSD 38. If they
   overlap, a naive pass emits on/on/off/off and the first off silences a note
   that should still be ringing. We reference-count each sounding target note
   and only emit note_off when the last voice releases.

3. **Orphans exist.** Suno exports (and plenty of human ones) contain note_offs
   with no matching note_on. Those are passed through on a best-effort lookup
   rather than crashed on.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import mido

from .drummap import DrumMap, Target
from .midi_io import Event, is_note_off, is_note_on, map_tracks

DRUM_CHANNEL = 9  # MIDI channel 10, zero-indexed


@dataclass
class RemapReport:
    """What the remap actually did. Printed by the CLI, asserted on in tests."""

    moved: Counter = field(default_factory=Counter)      # (gm, ssd) -> count
    passthrough: Counter = field(default_factory=Counter)  # gm -> count
    dropped: Counter = field(default_factory=Counter)      # gm -> count
    orphan_offs: int = 0
    stuck_notes: int = 0

    @property
    def total_in(self) -> int:
        return sum(self.moved.values()) + sum(self.passthrough.values()) + sum(self.dropped.values())

    def render(self, gm_names: Optional[Dict[int, str]] = None) -> str:
        gm_names = gm_names or {}
        lines = [f"remapped {self.total_in} drum notes"]
        for (gm, ssd), count in sorted(self.moved.items()):
            arrow = "==" if gm == ssd else "->"
            name = gm_names.get(gm, "?")
            lines.append(f"  GM {gm:>3} {arrow} SSD {ssd:>3}  x{count:<5} {name}")
        for gm, count in sorted(self.passthrough.items()):
            lines.append(f"  GM {gm:>3} ..  (unmapped, kept)  x{count:<5} {gm_names.get(gm, '?')}")
        for gm, count in sorted(self.dropped.items()):
            lines.append(f"  GM {gm:>3} XX  DROPPED  x{count:<5} {gm_names.get(gm, '?')}")
        if self.orphan_offs:
            lines.append(f"  {self.orphan_offs} orphan note_off(s) passed through")
        if self.stuck_notes:
            lines.append(f"  {self.stuck_notes} note(s) left hanging at end of track (note_off synthesised)")
        return "\n".join(lines)


def remap_events(
    events: List[Event],
    drum_map: DrumMap,
    channel: int = DRUM_CHANNEL,
    report: Optional[RemapReport] = None,
) -> List[Event]:
    """Remap the drum channel in one track's event list."""
    report = report if report is not None else RemapReport()
    out: List[Event] = []

    # (channel, source_note) -> stack of targets chosen at note_on time.
    # A stack, not a single value, because the same note can be re-struck
    # before its first note_off arrives.
    pending: Dict[Tuple[int, int], List[Optional[Target]]] = defaultdict(list)
    # (channel, target_note) -> how many voices are currently sounding there.
    sounding: Counter = Counter()

    for tick, msg in events:
        if msg.is_meta or getattr(msg, "channel", None) != channel:
            out.append((tick, msg))
            continue

        if is_note_on(msg):
            target = drum_map.lookup(msg.note, msg.velocity)
            pending[(msg.channel, msg.note)].append(target)
            if target is None:
                report.dropped[msg.note] += 1
                continue
            if target.via == "passthrough":
                report.passthrough[msg.note] += 1
            else:
                report.moved[(msg.note, target.note)] += 1
            out.append((tick, msg.copy(note=target.note)))
            sounding[(msg.channel, target.note)] += 1

        elif is_note_off(msg):
            stack = pending.get((msg.channel, msg.note))
            if stack:
                target = stack.pop(0)  # FIFO: oldest voice releases first
            else:
                report.orphan_offs += 1
                target = drum_map.lookup(msg.note, 64)
            if target is None:
                continue
            key = (msg.channel, target.note)
            if sounding[key] > 0:
                sounding[key] -= 1
                if sounding[key] > 0:
                    # Another voice is still ringing on this target note.
                    # Swallow the off so we don't cut it short.
                    continue
            out.append((tick, msg.copy(note=target.note)))

        else:
            out.append((tick, msg))

    # Anything still sounding got no note_off. Synthesise one at the end so the
    # file is well-formed - fluidsynth in particular will ring forever otherwise.
    leftovers = [(key, count) for key, count in sounding.items() if count > 0]
    if leftovers:
        end = max((tick for tick, _ in events), default=0)
        for (chan, note), count in sorted(leftovers):
            report.stuck_notes += count
            for _ in range(count):
                out.append((end, mido.Message("note_off", note=note, velocity=0, channel=chan)))

    return out


def remap_file(
    mid: mido.MidiFile,
    drum_map: DrumMap,
    channel: int = DRUM_CHANNEL,
) -> Tuple[mido.MidiFile, RemapReport]:
    """Remap every track of a MidiFile. Returns the new file and a report."""
    report = RemapReport()
    out = map_tracks(mid, lambda evs: remap_events(evs, drum_map, channel, report))
    return out, report


def preview_reverse_table(drum_map: DrumMap) -> Dict[int, int]:
    """SSD note -> a representative GM note, for previewing through a GM synth.

    The forward map is not injective (GM 38 and GM 40 both land on SSD 38), so
    this picks the lowest GM note that reaches each target. Good enough to judge
    a groove on a phone; never used for anything that ships.
    """
    table: Dict[int, int] = {}
    for gm, entry in sorted(drum_map._entries.items()):  # noqa: SLF001 - same package
        to = entry.get("to")
        if isinstance(to, int) and to not in table:
            table[to] = gm
    return table


def to_gm_preview(mid: mido.MidiFile, drum_map: DrumMap, channel: int = DRUM_CHANNEL) -> mido.MidiFile:
    """Undo the SSD remap so a General MIDI soundfont plays the right drums.

    Without this the crude fluidsynth preview is misleading: an SSD-mapped file
    fed to a GM synth plays whatever GM happens to have at those note numbers,
    which is not what Cubase will play.
    """
    table = preview_reverse_table(drum_map)

    def convert(events: List[Event]) -> List[Event]:
        out: List[Event] = []
        for tick, msg in events:
            if not msg.is_meta and getattr(msg, "channel", None) == channel and msg.type in ("note_on", "note_off"):
                out.append((tick, msg.copy(note=table.get(msg.note, msg.note))))
            else:
                out.append((tick, msg))
        return out

    return map_tracks(mid, convert)
