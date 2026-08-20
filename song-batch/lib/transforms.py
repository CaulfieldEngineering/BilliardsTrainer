"""Deterministic MIDI transforms.

Every transform here is a pure function ``events -> events`` over the
absolute-tick representation from :mod:`lib.midi_io`.

**Determinism rule.** Randomness is never drawn from a running RNG, because a
running RNG makes the Nth note's jitter depend on how many notes came before
it - insert one note at bar 1 and every later note changes, turning a one-note
edit into a whole-file git diff. Instead each note derives its own jitter by
hashing its own identity (seed, transform, tick, note, channel). Same note,
same seed, same jitter, forever, regardless of its neighbours.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Sequence, Tuple

import mido

from .drummap import DrumMap
from .midi_io import Event, is_note_off, is_note_on

DRUM_CHANNEL = 9


# --------------------------------------------------------------------- random

def _unit(seed: str, kind: str, *parts: int) -> float:
    """A stable pseudo-random float in [0, 1) for this exact note identity."""
    key = f"{seed}|{kind}|" + "|".join(str(p) for p in parts)
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def _signed(seed: str, kind: str, *parts: int) -> float:
    """Stable pseudo-random float in [-1, 1)."""
    return _unit(seed, kind, *parts) * 2.0 - 1.0


def _clamp_vel(value: float) -> int:
    return max(1, min(127, int(round(value))))


# ------------------------------------------------------------------ humanize

def humanize(
    events: Sequence[Event],
    seed: str,
    timing_ticks: int = 0,
    velocity_spread: int = 0,
    channel: Optional[int] = DRUM_CHANNEL,
) -> List[Event]:
    """Push notes off the grid and vary their velocity, reproducibly.

    ``timing_ticks`` is the maximum shift in either direction; ``velocity_spread``
    likewise for velocity. A note_off is shifted by the same amount as its
    note_on so durations survive intact.

    This is the antidote to Suno's grid-locked output.
    """
    if not timing_ticks and not velocity_spread:
        return list(events)

    shifts: Dict[Tuple[int, int], List[int]] = {}
    out: List[Event] = []

    for tick, msg in events:
        if msg.is_meta or (channel is not None and getattr(msg, "channel", None) != channel):
            out.append((tick, msg))
            continue

        if is_note_on(msg):
            shift = 0
            if timing_ticks:
                shift = int(round(_signed(seed, "time", tick, msg.note, msg.channel) * timing_ticks))
            new_msg = msg
            if velocity_spread:
                delta = _signed(seed, "vel", tick, msg.note, msg.channel) * velocity_spread
                new_msg = msg.copy(velocity=_clamp_vel(msg.velocity + delta))
            shifts.setdefault((msg.channel, msg.note), []).append(shift)
            out.append((max(0, tick + shift), new_msg))

        elif is_note_off(msg):
            stack = shifts.get((msg.channel, msg.note))
            shift = stack.pop(0) if stack else 0
            out.append((max(0, tick + shift), msg))

        else:
            out.append((tick, msg))

    return out


# ------------------------------------------------------------ velocity shape

def shape_velocity(
    events: Sequence[Event],
    ticks_per_beat: int,
    accents: Dict[str, float],
    beats_per_bar: int = 4,
    subdivision: int = 4,
    channel: Optional[int] = DRUM_CHANNEL,
) -> List[Event]:
    """Scale velocity by where a note sits in the bar.

    ``accents`` maps a grid slot to a multiplier. Slots are addressed as
    ``"<beat>.<sub>"`` (1-indexed beat, 0-indexed subdivision) with two
    wildcards: ``"downbeat"`` for beat 1 slot 0, and ``"offbeat"`` for any slot
    that is not on a beat. A missing slot means "leave it alone".

    Example - hit the downbeat, duck the in-between 16ths::

        {"downbeat": 1.15, "offbeat": 0.8}
    """
    if not accents:
        return list(events)

    slot_ticks = ticks_per_beat / subdivision
    bar_ticks = ticks_per_beat * beats_per_bar
    out: List[Event] = []

    for tick, msg in events:
        if msg.is_meta or not is_note_on(msg):
            out.append((tick, msg))
            continue
        if channel is not None and msg.channel != channel:
            out.append((tick, msg))
            continue

        in_bar = tick % bar_ticks
        slot = int(round(in_bar / slot_ticks))
        beat, sub = divmod(slot, subdivision)

        factor = None
        exact = f"{beat + 1}.{sub}"
        if exact in accents:
            factor = accents[exact]
        elif sub == 0 and beat == 0 and "downbeat" in accents:
            factor = accents["downbeat"]
        elif sub != 0 and "offbeat" in accents:
            factor = accents["offbeat"]
        elif sub == 0 and "onbeat" in accents:
            factor = accents["onbeat"]

        if factor is None:
            out.append((tick, msg))
        else:
            out.append((tick, msg.copy(velocity=_clamp_vel(msg.velocity * factor))))

    return out


def scale_velocity(
    events: Sequence[Event],
    factor: float,
    channel: Optional[int] = DRUM_CHANNEL,
    notes: Optional[Sequence[int]] = None,
) -> List[Event]:
    """Multiply velocity, optionally only for specific notes."""
    note_set = set(notes) if notes else None
    out: List[Event] = []
    for tick, msg in events:
        if (
            not msg.is_meta
            and is_note_on(msg)
            and (channel is None or msg.channel == channel)
            and (note_set is None or msg.note in note_set)
        ):
            out.append((tick, msg.copy(velocity=_clamp_vel(msg.velocity * factor))))
        else:
            out.append((tick, msg))
    return out


# --------------------------------------------------------------- ghost notes

def add_ghost_notes(
    events: Sequence[Event],
    ticks_per_beat: int,
    drum_map: DrumMap,
    seed: str,
    density: float = 0.35,
    velocity: int = 22,
    velocity_spread: int = 6,
    articulation: str = "snare_ghost",
    fallback_articulation: str = "snare_center",
    beats_per_bar: int = 4,
    channel: int = DRUM_CHANNEL,
) -> List[Event]:
    """Fill empty 16th slots around the snare with quiet ghost strokes.

    Only slots that are (a) off the beat and (b) currently silent on the snare
    are candidates, so this never doubles an existing hit. Whether a candidate
    fires is a deterministic function of its position, so ``density`` behaves
    like a stable threshold rather than a dice roll.

    Uses SSD's dedicated ghost articulation when ``maps/ssd.json`` has a note
    for it, otherwise falls back to a quiet hit on the main snare.
    """
    ghost_note = drum_map.articulation_note(articulation)
    used_fallback = False
    if ghost_note is None:
        ghost_note = drum_map.articulation_note(fallback_articulation)
        used_fallback = True
    if ghost_note is None:
        return list(events)  # nothing sane to place

    snare_note = drum_map.articulation_note(fallback_articulation)
    slot_ticks = ticks_per_beat / 4  # 16ths
    bar_ticks = ticks_per_beat * beats_per_bar

    occupied = {
        tick
        for tick, msg in events
        if is_note_on(msg) and getattr(msg, "channel", None) == channel and msg.note in (snare_note, ghost_note)
    }
    if not occupied:
        return list(events)

    last_tick = max(tick for tick, _ in events)
    out = list(events)
    added = 0

    slot = 0
    while slot * slot_ticks <= last_tick:
        tick = int(round(slot * slot_ticks))
        in_bar = tick % bar_ticks
        sub = int(round(in_bar / slot_ticks)) % 4
        if sub == 0 or tick in occupied:
            slot += 1
            continue
        if _unit(seed, "ghost", tick) >= density:
            slot += 1
            continue
        vel = velocity
        if velocity_spread:
            vel = _clamp_vel(velocity + _signed(seed, "ghostvel", tick) * velocity_spread)
        out.append((tick, mido.Message("note_on", note=ghost_note, velocity=vel, channel=channel)))
        out.append(
            (tick + max(1, int(slot_ticks // 2)), mido.Message("note_off", note=ghost_note, velocity=0, channel=channel))
        )
        added += 1
        slot += 1

    if added and used_fallback:
        # Not an error - just worth knowing the ghosts are velocity-only.
        pass
    return out


# -------------------------------------------------------------------- groove

def swing(
    events: Sequence[Event],
    ticks_per_beat: int,
    amount: float,
    subdivision: int = 8,
    channel: Optional[int] = DRUM_CHANNEL,
) -> List[Event]:
    """Delay every second subdivision. ``amount`` 0.0 = straight, 1.0 = triplet.

    At ``amount=1.0`` and ``subdivision=8`` the offbeat 8th lands two thirds of
    the way through the beat, i.e. a shuffle.
    """
    if not amount:
        return list(events)

    # `subdivision` names a note value: 8 = eighth notes. A quarter note is
    # ticks_per_beat, so an Nth note is ticks_per_beat * 4 / N.
    sub_ticks = ticks_per_beat * 4 / subdivision
    if sub_ticks <= 0:
        return list(events)
    pair_ticks = sub_ticks
    # Full swing puts the offbeat two thirds of the way through the pair, which
    # is a shift of one third of a subdivision.
    max_shift = sub_ticks / 3.0
    shifts: Dict[Tuple[int, int], List[int]] = {}
    out: List[Event] = []

    for tick, msg in events:
        if msg.is_meta or (channel is not None and getattr(msg, "channel", None) != channel):
            out.append((tick, msg))
            continue

        if is_note_on(msg):
            index = int(round(tick / pair_ticks))
            shift = int(round(max_shift * amount)) if index % 2 == 1 else 0
            shifts.setdefault((msg.channel, msg.note), []).append(shift)
            out.append((tick + shift, msg))
        elif is_note_off(msg):
            stack = shifts.get((msg.channel, msg.note))
            shift = stack.pop(0) if stack else 0
            out.append((tick + shift, msg))
        else:
            out.append((tick, msg))

    return out


# -------------------------------------------------------------------- chokes

def resolve_chokes(
    events: Sequence[Event],
    drum_map: DrumMap,
    channel: int = DRUM_CHANNEL,
) -> List[Event]:
    """Cut a ringing note when a member of the same choke group is struck.

    An open hi-hat that is still sounding when the hat closes should stop. Suno
    MIDI routinely leaves both ringing; a sampler then plays them on top of each
    other and the groove turns to mud.
    """
    groups = {
        name: [n for n in (drum_map.articulation_note(a) for a in members) if n is not None]
        for name, members in drum_map.data.get("choke_groups", {}).items()
        if not name.startswith("_") and isinstance(members, list)
    }
    group_of: Dict[int, str] = {}
    for name, notes in groups.items():
        for note in notes:
            group_of[note] = name
    if not group_of:
        return list(events)

    out: List[Event] = []
    # group -> (note, tick_index_in_out) of the currently sounding member
    active: Dict[str, int] = {}

    for tick, msg in events:
        if msg.is_meta or getattr(msg, "channel", None) != channel:
            out.append((tick, msg))
            continue

        if is_note_on(msg) and msg.note in group_of:
            group = group_of[msg.note]
            previous = active.get(group)
            if previous is not None and previous != msg.note:
                out.append((tick, mido.Message("note_off", note=previous, velocity=0, channel=channel)))
            active[group] = msg.note
            out.append((tick, msg))
        elif is_note_off(msg) and msg.note in group_of:
            group = group_of[msg.note]
            if active.get(group) == msg.note:
                active.pop(group, None)
            out.append((tick, msg))
        else:
            out.append((tick, msg))

    return out
