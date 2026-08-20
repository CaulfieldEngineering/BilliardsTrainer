"""Run the transform chain described by a song's ``spec.yaml``.

``spec.yaml`` names transforms and their arguments; this module owns the
registry that turns those names into functions, injects the context each one
needs (ticks per beat, drum map, seed, meter) and reports what happened.

Adding a transform is: write the function in :mod:`lib.transforms`, add one
entry to ``REGISTRY``, document it in CLAUDE.md. Nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import mido

from . import transforms as T
from .drummap import DrumMap
from .midi_io import Event, describe, map_tracks, save, to_events, to_track
from .remap import DRUM_CHANNEL, RemapReport, remap_events, to_gm_preview
from .assemble import NamedTrack, assemble, tracks_from_file
from .sections import Section, detect_sections, summarise
from .template import Template
from .spec import Spec
from . import tempo as tempo_mod


class PipelineError(Exception):
    pass


@dataclass
class Context:
    """Everything a transform might need that isn't in its own arguments."""

    ticks_per_beat: int
    drum_map: DrumMap
    seed: str
    beats_per_bar: int = 4
    channel: int = DRUM_CHANNEL
    report: RemapReport = field(default_factory=RemapReport)


# Each entry takes (events, ctx, **step_args) and returns events.
def _step_remap(events: List[Event], ctx: Context, **kw) -> List[Event]:
    return remap_events(events, ctx.drum_map, kw.get("channel", ctx.channel), ctx.report)


def _step_humanize(events: List[Event], ctx: Context, **kw) -> List[Event]:
    return T.humanize(
        events,
        seed=ctx.seed,
        timing_ticks=int(kw.get("timing_ticks", 0)),
        velocity_spread=int(kw.get("velocity_spread", 0)),
        channel=kw.get("channel", ctx.channel),
    )


def _step_shape_velocity(events: List[Event], ctx: Context, **kw) -> List[Event]:
    return T.shape_velocity(
        events,
        ticks_per_beat=ctx.ticks_per_beat,
        accents={str(k): float(v) for k, v in (kw.get("accents") or {}).items()},
        beats_per_bar=int(kw.get("beats_per_bar", ctx.beats_per_bar)),
        subdivision=int(kw.get("subdivision", 4)),
        channel=kw.get("channel", ctx.channel),
    )


def _step_ghost_notes(events: List[Event], ctx: Context, **kw) -> List[Event]:
    return T.add_ghost_notes(
        events,
        ticks_per_beat=ctx.ticks_per_beat,
        drum_map=ctx.drum_map,
        seed=ctx.seed,
        density=float(kw.get("density", 0.35)),
        velocity=int(kw.get("velocity", 22)),
        velocity_spread=int(kw.get("velocity_spread", 6)),
        beats_per_bar=int(kw.get("beats_per_bar", ctx.beats_per_bar)),
        channel=int(kw.get("channel", ctx.channel)),
    )


def _step_swing(events: List[Event], ctx: Context, **kw) -> List[Event]:
    return T.swing(
        events,
        ticks_per_beat=ctx.ticks_per_beat,
        amount=float(kw.get("amount", 0.0)),
        subdivision=int(kw.get("subdivision", 8)),
        channel=kw.get("channel", ctx.channel),
    )


def _step_resolve_chokes(events: List[Event], ctx: Context, **kw) -> List[Event]:
    return T.resolve_chokes(events, ctx.drum_map, int(kw.get("channel", ctx.channel)))


def _step_scale_velocity(events: List[Event], ctx: Context, **kw) -> List[Event]:
    return T.scale_velocity(
        events,
        factor=float(kw.get("factor", 1.0)),
        channel=kw.get("channel", ctx.channel),
        notes=kw.get("notes"),
    )


REGISTRY: Dict[str, Callable[..., List[Event]]] = {
    "remap": _step_remap,
    "humanize": _step_humanize,
    "shape_velocity": _step_shape_velocity,
    "ghost_notes": _step_ghost_notes,
    "swing": _step_swing,
    "resolve_chokes": _step_resolve_chokes,
    "scale_velocity": _step_scale_velocity,
}


@dataclass
class BuildResult:
    target: str
    source: Path
    midi_out: Optional[Path] = None
    preview_midi: Optional[Path] = None
    audio_out: Optional[Path] = None
    steps: List[str] = field(default_factory=list)
    report: Optional[RemapReport] = None
    arrangement: Optional[str] = None
    sections: List[Any] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def apply_chain(
    mid: mido.MidiFile,
    steps: List[Dict[str, Any]],
    ctx: Context,
) -> tuple[mido.MidiFile, List[str]]:
    """Apply a list of ``{name: ..., **args}`` steps to every track."""
    applied: List[str] = []
    current = mid
    for step in steps:
        if not isinstance(step, dict):
            raise PipelineError(f"transform step must be a mapping, got {step!r}")
        name = step.get("name")
        if name not in REGISTRY:
            raise PipelineError(
                f"unknown transform {name!r}. Known: {', '.join(sorted(REGISTRY))}"
            )
        args = {k: v for k, v in step.items() if k != "name"}
        fn = REGISTRY[name]
        current = map_tracks(current, lambda evs, _fn=fn, _a=args: _fn(evs, ctx, **_a))
        applied.append(name)
    return current, applied


def build_target(
    spec: Spec,
    name: str,
    config: Dict[str, Any],
    drum_map: DrumMap,
    render_audio: bool = True,
) -> BuildResult:
    """Build one target (e.g. "drums") from a song spec."""
    sources = spec.sources()
    source_key = config.get("source", name)
    if source_key not in sources:
        raise PipelineError(
            f"{spec.slug}: target {name!r} wants source {source_key!r}, "
            f"which is not in sources ({', '.join(sources) or 'none'})"
        )
    source = sources[source_key]
    if not source.exists():
        raise PipelineError(f"{spec.slug}: source file missing: {source}")

    result = BuildResult(target=name, source=source)
    mid = mido.MidiFile(str(source))

    # Tempo / meter normalisation happens before anything grid-dependent, so
    # the transforms see the meter the operator declared, not Suno's guess.
    if spec.bpm:
        mid = tempo_mod.set_single_tempo(mid, spec.bpm)
    num, den = spec.time_signature
    mid = tempo_mod.set_time_signature(mid, num, den)

    ctx = Context(
        ticks_per_beat=mid.ticks_per_beat,
        drum_map=drum_map,
        seed=spec.seed,
        beats_per_bar=spec.beats_per_bar,
        channel=int(config.get("channel", DRUM_CHANNEL)),
    )

    # Read the arrangement BEFORE transforming. Ghost notes and humanisation are
    # deliberately position-dependent, so after they run no two bars are
    # byte-identical and every repeat looks like a new section. Structure is a
    # property of the source performance, so that is what we read it from.
    merged: List[Event] = []
    for track in mid.tracks:
        merged.extend(to_events(track))
    sections = detect_sections(merged, mid.ticks_per_beat, spec.beats_per_bar, channel=ctx.channel)
    result.arrangement = summarise(sections)
    result.sections = sections

    steps = list(config.get("transforms") or [])
    mid, applied = apply_chain(mid, steps, ctx)
    result.steps = applied
    result.report = ctx.report

    build_dir = spec.build_dir
    result.midi_out = save(mid, build_dir / f"{name}.mid")

    if config.get("preview", True):
        preview = to_gm_preview(mid, drum_map, ctx.channel)
        result.preview_midi = save(preview, build_dir / f"{name}.preview.mid")

    if render_audio and config.get("render", True):
        from render.fluidsynth import RenderError, preflight, render as do_render

        problems = preflight()
        blocking = [p for p in problems if "fluidsynth" in p or "soundfont" in p]
        if blocking:
            result.warnings.extend(blocking)
        else:
            result.warnings.extend(p for p in problems if p not in blocking)
            try:
                rendered = do_render(
                    result.preview_midi or result.midi_out,
                    build_dir / f"{name}.mp3",
                    gain=float(config.get("render_gain", 0.6)),
                )
                result.audio_out = rendered.output
            except RenderError as exc:
                result.warnings.append(f"render failed: {exc}")

    return result


def clean_build_dir(spec: Spec) -> int:
    """Delete the song's derived artifacts. Returns how many files went.

    build/ is entirely derived, so a build starts from empty. Without this,
    turning a target off leaves its last output sitting there looking current -
    and since build/ is committed, a stale render can be listened to, or
    reviewed, weeks after the thing that made it stopped existing.
    """
    build_dir = spec.build_dir
    if not build_dir.exists():
        return 0
    removed = 0
    for path in build_dir.iterdir():
        if path.is_file():
            path.unlink()
            removed += 1
    return removed


def build_song(
    spec: Spec,
    drum_map: DrumMap,
    render_audio: bool = True,
    only: Optional[List[str]] = None,
) -> List[BuildResult]:
    # Only safe to wipe when building everything; a --target run would
    # otherwise delete the outputs of targets it is not rebuilding.
    if only is None:
        clean_build_dir(spec)
    results: List[BuildResult] = []
    for name, config in spec.build_targets().items():
        if only and name not in only:
            continue
        if config.get("enabled") is False:
            continue
        results.append(build_target(spec, name, config, drum_map, render_audio))
    return results


@dataclass
class SessionResult:
    """The song-level artifacts: the file you import, and the one you listen to."""

    midi: Path
    preview_midi: Optional[Path] = None
    audio: Optional[Path] = None
    mastered: Optional[Path] = None
    tracks: List[str] = field(default_factory=list)
    markers: List[str] = field(default_factory=list)
    master_stages: List[str] = field(default_factory=list)
    loudness_before: Optional[Any] = None
    loudness_after: Optional[Any] = None
    warnings: List[str] = field(default_factory=list)


def _assemble_from(results, attr, spec, template, sections, reference_tpb):
    tracks: List[NamedTrack] = []
    for result in sorted(results, key=lambda r: template.order_key(r.target)):
        path = getattr(result, attr)
        if not path or not path.exists():
            continue
        tracks.extend(tracks_from_file(mido.MidiFile(str(path)), template.track_name_for(result.target)))
    if not tracks:
        return None
    return assemble(
        tracks=tracks,
        sections=sections,
        ticks_per_beat=reference_tpb,
        beats_per_bar=spec.beats_per_bar,
        bpm=spec.bpm,
        time_signature=spec.time_signature,
    )


def write_session(
    spec: Spec,
    results: List[BuildResult],
    template: Optional[Template] = None,
    render_audio: bool = True,
) -> Optional[SessionResult]:
    """Write the song-level artifacts.

    Three files come out of here:

    * ``<slug>.mid``          - SSD note numbers. **This is what goes into Cubase.**
    * ``<slug>.preview.mid``  - the same session reverse-mapped to General MIDI.
    * ``<slug>.master.mp3``   - that preview rendered and run through the master
      chain, so what lands on the phone is judged at realistic loudness.

    Track 0 of both MIDI files is a conductor track carrying tempo, meter and
    one marker per arrangement section. Section names come from spec.yaml when
    the operator has written them; otherwise the detector's structural labels
    are used, which is honest about nothing having named them yet.
    """
    usable = [r for r in results if r.midi_out and r.midi_out.exists()]
    if not usable:
        return None

    template = template or Template.load()

    sections = spec.declared_sections()
    if not sections:
        for result in usable:
            if result.sections:
                sections = list(result.sections)
                break

    reference_tpb = mido.MidiFile(str(usable[0].midi_out)).ticks_per_beat
    session = _assemble_from(usable, "midi_out", spec, template, sections, reference_tpb)
    if session is None:
        return None

    out = SessionResult(
        midi=save(session, spec.build_dir / f"{spec.slug}.mid"),
        tracks=[template.track_name_for(r.target) for r in usable],
        markers=[s.label for s in sections],
    )

    preview = _assemble_from(usable, "preview_midi", spec, template, sections, reference_tpb)
    if preview is not None:
        out.preview_midi = save(preview, spec.build_dir / f"{spec.slug}.preview.mid")

    if render_audio and out.preview_midi:
        _render_and_master(spec, out)

    return out


def _render_and_master(spec: Spec, session: SessionResult) -> None:
    """Render the preview session, then run the master chain over it."""
    from render.fluidsynth import RenderError, preflight, render as do_render
    from render.master import MasterError, load_chain, master, stage_names

    problems = preflight()
    blocking = [p for p in problems if "fluidsynth" in p or "soundfont" in p]
    if blocking:
        session.warnings.extend(blocking)
        return

    try:
        rendered = do_render(session.preview_midi, spec.build_dir / f"{spec.slug}.mp3")
        session.audio = rendered.output
    except RenderError as exc:
        session.warnings.append(f"render failed: {exc}")
        return

    chain = spec.raw.get("build", {}).get("master")
    if chain is None:
        chain = load_chain()
    if chain is False or not chain:
        return

    try:
        result = master(session.audio, spec.build_dir / f"{spec.slug}.master.mp3", chain)
    except MasterError as exc:
        session.warnings.append(f"master failed: {exc}")
        return

    session.mastered = result.output
    session.master_stages = result.stages
    session.loudness_before = result.before
    session.loudness_after = result.after
