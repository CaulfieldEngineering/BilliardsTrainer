"""``sb`` - the command line for song-batch.

Output here is written to be read on a phone: short lines, no wide tables, the
important thing first. Every command is safe to run repeatedly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.drummap import DrumMap, DrumMapError  # noqa: E402
from lib.pipeline import PipelineError, build_song, write_session  # noqa: E402
from lib.template import Template  # noqa: E402
from lib.spec import STAGES, Spec, discover  # noqa: E402

SONGS_DIR = REPO_ROOT / "songs"
TEMPLATE_DIR = SONGS_DIR / "_template"


def _load_map(path: Optional[str]) -> DrumMap:
    try:
        return DrumMap.load(path)
    except DrumMapError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)


def _select(slugs: List[str]) -> List[Spec]:
    specs = discover(SONGS_DIR)
    if not slugs:
        return specs
    by_slug = {s.slug: s for s in specs}
    chosen = []
    for slug in slugs:
        if slug not in by_slug:
            print(f"error: no song {slug!r} (have: {', '.join(sorted(by_slug)) or 'none'})", file=sys.stderr)
            raise SystemExit(2)
        chosen.append(by_slug[slug])
    return chosen


# ------------------------------------------------------------------ sections

def cmd_sections(args) -> int:
    """Propose an arrangement from the source MIDI, optionally into spec.yaml."""
    import mido

    from lib.midi_io import to_events
    from lib.sections import detect_sections, summarise

    for spec in _select(args.slugs):
        sources = spec.sources()
        if not sources:
            print(f"{spec.slug}: no sources")
            continue
        source = list(sources.values())[0]
        mid = mido.MidiFile(str(source))
        events = []
        for track in mid.tracks:
            events.extend(to_events(track))
        detected = detect_sections(events, mid.ticks_per_beat, spec.beats_per_bar, channel=args.channel)

        print(f"{spec.slug}: {summarise(detected)}")
        for section in detected:
            fill = "  (ends with a fill)" if section.is_fill else ""
            print(f"  bar {section.start_bar + 1:>3}  {section.label}  x{section.length_bars} bars{fill}")

        if args.write:
            if spec.declared_sections() and not args.force:
                print("  ! spec.yaml already has sections; pass --force to overwrite")
                continue
            spec.raw["sections"] = [s.as_dict() for s in detected]
            spec.save()
            print(f"  written to {spec.path.relative_to(REPO_ROOT)}")
            print("  now rename the labels to Intro / Verse 1 / Chorus - those become the markers")
    return 0


# --------------------------------------------------------------------- build

def cmd_build(args) -> int:
    drum_map = _load_map(args.map)
    specs = _select(args.slugs)
    if not specs:
        print("no songs found under songs/ - `./sb new <slug>` to make one")
        return 0

    failures = 0
    for spec in specs:
        errors = spec.errors()
        if errors:
            print(f"\n{spec.slug}: SPEC INVALID")
            for issue in errors:
                print(f"  {issue}")
            failures += 1
            continue

        print(f"\n{spec.slug} - {spec.title}  [{spec.stage}]")
        try:
            results = build_song(
                spec,
                drum_map,
                render_audio=not args.no_render,
                only=[args.target] if args.target else None,
            )
        except PipelineError as exc:
            print(f"  build failed: {exc}")
            failures += 1
            continue

        if not results:
            print("  (nothing to build - no enabled targets in build.targets)")
            continue

        for result in results:
            chain = " -> ".join(result.steps) or "(no transforms)"
            print(f"  {result.target}: {chain}")
            if result.report and result.report.total_in:
                for line in result.report.render(DrumMap.gm_names()).splitlines()[1:]:
                    print(f"    {line.strip()}")
            if result.arrangement:
                print(f"    arrangement: {result.arrangement}")
            for out in (result.midi_out, result.preview_midi, result.audio_out):
                if out:
                    print(f"    -> {out.relative_to(REPO_ROOT)}")
            for warning in result.warnings:
                print(f"    ! {warning}")

        template = Template.load()
        session = write_session(spec, results, template, render_audio=not args.no_render)
        if session:
            print(f"  session -> {session.midi.relative_to(REPO_ROOT)}")
            print(f"    tracks:  {', '.join(session.tracks)}")
            print(f"    markers: {' '.join(session.markers) or '(none)'}")
            if session.mastered:
                print(f"    master -> {session.mastered.relative_to(REPO_ROOT)}"
                      f"  [{' -> '.join(session.master_stages)}]")
                print(f"      before: {session.loudness_before}")
                print(f"      after:  {session.loudness_after}")
            elif session.audio:
                print(f"    audio  -> {session.audio.relative_to(REPO_ROOT)}")
            for warning in session.warnings:
                print(f"    ! {warning}")
            if not spec.declared_sections():
                print("    ! markers are structural labels - run `./sb sections "
                      f"{spec.slug} --write` and rename them to Verse/Chorus in spec.yaml")
            if not template.verified:
                print("    ! template.yaml is a placeholder - track names are guesses")

    return 1 if failures else 0


# -------------------------------------------------------------------- status

def cmd_status(args) -> int:
    specs = discover(SONGS_DIR)
    if args.blocked:
        specs = [s for s in specs if s.is_blocked]
    if args.stage:
        specs = [s for s in specs if s.stage == args.stage]
    if args.ready_for:
        target = args.ready_for
        if target not in STAGES:
            print(f"error: unknown stage {target!r}", file=sys.stderr)
            return 2
        index = STAGES.index(target)
        specs = [s for s in specs if s.stage_index == index - 1 and not s.is_blocked]

    if args.json:
        print(json.dumps(
            [
                {
                    "slug": s.slug,
                    "title": s.title,
                    "stage": s.stage,
                    "blocked_by": s.blocked_by,
                    "next_action": s.next_action,
                    "bpm": s.bpm,
                    "key": s.key,
                    "template_revision": s.template_revision,
                    "pending_migrations": s.pending_migrations,
                }
                for s in specs
            ],
            indent=2,
        ))
        return 0

    if not specs:
        print("no songs match")
        return 0

    print(f"{len(specs)} song(s)\n")
    for spec in sorted(specs, key=lambda s: (-s.stage_index, s.slug)):
        flag = "BLOCKED" if spec.is_blocked else spec.stage
        print(f"  {spec.slug:<24} {flag}")
        if spec.is_blocked:
            print(f"      blocked by: {spec.blocked_by}")
        if spec.next_action:
            print(f"      next: {spec.next_action}")
        if spec.pending_migrations:
            print(f"      pending migrations: {', '.join(spec.pending_migrations)}")

    # A tally by stage, so "what's the state of all 20" is one glance.
    print("\nby stage:")
    for stage in STAGES:
        count = sum(1 for s in specs if s.stage == stage)
        if count:
            print(f"  {stage:<20} {'#' * count} {count}")
    return 0


# ------------------------------------------------------------------ validate

def cmd_validate(args) -> int:
    drum_map = _load_map(args.map)
    problems = 0

    print(drum_map.summary())
    for issue in drum_map.doctor():
        if issue.level in ("error", "warn"):
            print(f"  {issue}")
        if issue.level == "error":
            problems += 1

    specs = discover(SONGS_DIR)
    print(f"\n{len(specs)} song spec(s)")
    for spec in specs:
        issues = spec.validate()
        if not issues:
            print(f"  {spec.slug}: ok")
            continue
        print(f"  {spec.slug}:")
        for issue in issues:
            print(f"    {issue}")
            if issue.level == "error":
                problems += 1

    print(f"\n{problems} error(s)")
    return 1 if problems else 0


# ----------------------------------------------------------------------- map

def cmd_map(args) -> int:
    drum_map = _load_map(args.map)
    if args.action == "show":
        gm_names = DrumMap.gm_names()
        print(drum_map.summary())
        print()
        for entry in sorted(drum_map.data.get("entries", []), key=lambda e: e["gm"]):
            gm = entry["gm"]
            to = entry.get("to")
            mark = " " if to == gm else "*"
            print(f"  GM {gm:>3} -> {str(to):>4} {mark} {entry.get('articulation','?'):<20} {gm_names.get(gm,'')}")
        print("\n  (* = note number changes)")
        return 0

    issues = drum_map.doctor()
    print(drum_map.summary())
    print()
    for level in ("error", "warn", "info"):
        matching = [i for i in issues if i.level == level]
        for issue in matching:
            print(f"  {issue}")
    errors = sum(1 for i in issues if i.level == "error")
    print(f"\n{errors} error(s), {sum(1 for i in issues if i.level=='warn')} warning(s)")
    return 1 if errors else 0


# ------------------------------------------------------------------- inspect

def cmd_inspect(args) -> int:
    import mido

    from lib.midi_io import describe, to_events
    from lib.sections import detect_sections, summarise
    from lib.tempo import beats_per_bar, initial_bpm, tempo_map, time_signature

    mid = mido.MidiFile(args.midi)
    info = describe(mid)
    num, den = time_signature(mid)
    print(f"{args.midi}")
    print(f"  format {info['type']}, {info['ticks_per_beat']} ticks/beat, {info['tracks']} track(s)")
    print(f"  {info['notes']} notes, channels {info['channels']}")
    print(f"  {initial_bpm(mid):g} bpm, {num}/{den}")
    changes = tempo_map(mid)
    if len(changes) > 1:
        print(f"  ! {len(changes)} tempo changes - set musical.bpm in spec.yaml to flatten")

    gm_names = DrumMap.gm_names()
    counts = {}
    for track in mid.tracks:
        for msg in track:
            if msg.type == "note_on" and msg.velocity > 0 and getattr(msg, "channel", None) == args.channel:
                counts[msg.note] = counts.get(msg.note, 0) + 1
    if counts:
        print(f"\n  drum notes on channel {args.channel}:")
        for note, count in sorted(counts.items()):
            print(f"    {note:>3}  x{count:<5} {gm_names.get(note, '(not a GM percussion note)')}")

    merged = []
    for track in mid.tracks:
        merged.extend(to_events(track))
    sections = detect_sections(merged, mid.ticks_per_beat, beats_per_bar(mid), channel=args.channel)
    print(f"\n  arrangement: {summarise(sections)}")
    return 0


# -------------------------------------------------------------------- render

def cmd_render(args) -> int:
    from render.fluidsynth import RenderError, find_soundfont, preflight, render

    problems = preflight(args.soundfont)
    if args.check:
        if problems:
            print("render environment NOT ready:")
            for problem in problems:
                print(f"  - {problem}")
            return 1
        print(f"render environment ok (soundfont: {find_soundfont(args.soundfont)})")
        return 0

    if not args.midi or not args.out:
        print("error: need MIDI and OUT (or --check)", file=sys.stderr)
        return 2
    blocking = [p for p in problems if "fluidsynth" in p or "soundfont" in p]
    if blocking:
        for problem in blocking:
            print(f"error: {problem}", file=sys.stderr)
        return 1

    try:
        result = render(args.midi, args.out, soundfont=args.soundfont, gain=args.gain)
    except RenderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    length = f" ({result.duration_seconds:g}s)" if result.duration_seconds else ""
    print(f"wrote {result.output}{length}")
    return 0


# -------------------------------------------------------------------- master

def cmd_master(args) -> int:
    """Show or apply the master chain."""
    from render.master import (
        MasterError, build_filter_chain, load_chain, master as do_master, measure, stage_names,
    )

    chain = load_chain(args.chain)
    if not chain:
        print("no master chain defined (expected master.yaml)", file=sys.stderr)
        return 1

    if args.check or not args.audio:
        print(f"master chain: {' -> '.join(stage_names(chain))}\n")
        print(build_filter_chain(chain))
        return 0

    if not args.out:
        print("error: need AUDIO and OUT (or --check)", file=sys.stderr)
        return 2
    try:
        result = do_master(args.audio, args.out, chain, two_pass=not args.no_two_pass)
    except MasterError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {result.output}"
          f"{'  (two-pass)' if result.two_pass else ''}")
    print(f"  before: {result.before}")
    print(f"  after:  {result.after}")
    return 0


# ----------------------------------------------------------------------- new

def cmd_new(args) -> int:
    import shutil

    dest = SONGS_DIR / args.slug
    if dest.exists():
        print(f"error: {dest} already exists", file=sys.stderr)
        return 2
    if not TEMPLATE_DIR.exists():
        print(f"error: no template at {TEMPLATE_DIR}", file=sys.stderr)
        return 2

    shutil.copytree(TEMPLATE_DIR, dest)
    (dest / "suno").mkdir(exist_ok=True)
    (dest / "build").mkdir(exist_ok=True)

    spec = Spec.load(dest / "spec.yaml")
    spec.raw["slug"] = args.slug
    spec.raw["title"] = args.title or args.slug.replace("-", " ").title()
    spec.save()
    print(f"created {dest.relative_to(REPO_ROOT)}")
    print(f"  next: drop the Suno MIDI export in {dest.relative_to(REPO_ROOT)}/suno/ and edit spec.yaml")
    return 0


# ---------------------------------------------------------------------- main

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sb", description="song-batch pipeline")
    parser.add_argument("--map", help="path to a drum map (default maps/ssd.json)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="build songs from their spec.yaml")
    p_build.add_argument("slugs", nargs="*", help="song slugs (default: all)")
    p_build.add_argument("--no-render", action="store_true", help="skip audio rendering")
    p_build.add_argument("--target", help="build only this target")
    p_build.set_defaults(func=cmd_build)

    p_status = sub.add_parser("status", help="where every song stands")
    p_status.add_argument("--blocked", action="store_true", help="only blocked songs")
    p_status.add_argument("--stage", help="only songs at this stage")
    p_status.add_argument("--ready-for", help="songs whose next stage is this")
    p_status.add_argument("--json", action="store_true")
    p_status.set_defaults(func=cmd_status)

    p_sections = sub.add_parser("sections", help="propose an arrangement from the source MIDI")
    p_sections.add_argument("slugs", nargs="*")
    p_sections.add_argument("--write", action="store_true", help="write into spec.yaml")
    p_sections.add_argument("--force", action="store_true", help="overwrite existing sections")
    p_sections.add_argument("--channel", type=int, default=9)
    p_sections.set_defaults(func=cmd_sections)

    p_validate = sub.add_parser("validate", help="check the drum map and every spec")
    p_validate.set_defaults(func=cmd_validate)

    p_map = sub.add_parser("map", help="inspect the GM->SSD map")
    p_map.add_argument("action", choices=["doctor", "show"], nargs="?", default="doctor")
    p_map.set_defaults(func=cmd_map)

    p_inspect = sub.add_parser("inspect", help="describe a MIDI file")
    p_inspect.add_argument("midi")
    p_inspect.add_argument("--channel", type=int, default=9)
    p_inspect.set_defaults(func=cmd_inspect)

    p_render = sub.add_parser("render", help="render a MIDI file to audio")
    p_render.add_argument("midi", nargs="?")
    p_render.add_argument("out", nargs="?")
    p_render.add_argument("--check", action="store_true", help="report whether rendering is possible")
    p_render.add_argument("--soundfont")
    p_render.add_argument("--gain", type=float, default=0.6)
    p_render.set_defaults(func=cmd_render)

    p_master = sub.add_parser("master", help="show or apply the master chain")
    p_master.add_argument("audio", nargs="?")
    p_master.add_argument("out", nargs="?")
    p_master.add_argument("--check", action="store_true", help="print the chain, do nothing")
    p_master.add_argument("--chain", help="path to a chain file (default master.yaml)")
    p_master.add_argument("--no-two-pass", action="store_true")
    p_master.set_defaults(func=cmd_master)

    p_new = sub.add_parser("new", help="scaffold a new song from the template")
    p_new.add_argument("slug")
    p_new.add_argument("--title")
    p_new.set_defaults(func=cmd_new)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
