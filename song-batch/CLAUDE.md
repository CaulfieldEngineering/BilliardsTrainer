# song-batch

A pipeline for developing ~20 songs in parallel, driven by Claude Code from a
phone while the operator is away from the computer.

**If you are an agent starting cold: read this file, then run `./sb status`.**
That is the whole onboarding.

---

## Operator context

- **Cubase 15 on macOS is the DAW.** Not negotiable, not changing. REAPER is
  acceptable for throwaway proof-of-concept work only.
- ~20 songs, currently riff ideas rather than arrangements.
- **Real guitar is tracked by the operator. No AI-generated audio ships.**
  Suno is an ideation tool here, not a source of final audio.
- Drums are **Steven Slate Drums (SSD)**. All drum MIDI must be remapped GM -> SSD.
- One Cubase template file; all 20 projects derive from it.

### About Suno MIDI

Suno Studio exports MIDI per stem (extract stems -> select stem -> "Get MIDI").
Two things to know, both of which shape this codebase:

1. **It is grid-locked and dynamically flat.** Fine as a drum skeleton. A
   liability for melodic parts. Undoing that flatness is most of what
   `lib/transforms.py` is for.
2. **Stem separation applies a second generation pass, not a clean split.** A
   stem can contain artifacts that are not in the original mix. Validate before
   building on it. Do not assume a stem is a faithful extraction.

---

## Golden rules

1. **`maps/ssd.json` is the only place a drum note number may live.** If you are
   typing a bare integer drum note anywhere else, you are writing a bug.
2. **`songs/<slug>/suno/` is immutable.** Never edit a source export in place. A
   better export is a *new file* and a `sources:` change.
3. **`songs/<slug>/build/` is entirely derived.** Safe to delete at any moment;
   `./sb build` regenerates it. Never hand-edit anything in there.
4. **`spec.yaml` is the source of truth for status.** Not Notion, not ClickUp.
   The PM tool is a view. Status lives in git next to the song.
5. **Every transform is deterministic.** Same input + same seed = byte-identical
   output. This is load-bearing: it is what makes a build reviewable as a git
   diff from a phone. There is a test for it
   (`test_build_is_byte_reproducible`). Do not break it.
6. **Never reorder or delete a Cubase template track.** See below - this one is
   silent and expensive when broken.

---

## Cubase template invariants

Cubase has no project inheritance. Once a project is derived from the template
it diverges, permanently. The only batch-update mechanism is **Save/Load
Selected Channels** (`.vmx`, MixConsole Functions menu), and it has two
properties that make it dangerous:

- Settings apply **positionally** - left to right in MixConsole order. **Not by
  name.**
- Input/output routing is **not** saved.

Therefore:

> **Channel order in all 20 projects stays identical to the template.**
> Never delete a template track - disable or hide it instead.
> Never reorder. Song-specific tracks go at the far right, past the shared block.

Break this once, in one project, and that project silently receives the wrong
settings on every future push. There is no error message. You find out by ear,
months later.

### What propagates for free

These live in user preferences, not the project, so they update everywhere at
once. **Push everything you possibly can down to this layer:**

- Key commands and macros (`Key Commands.xml`)
- Project Logical Editor / Logical Editor presets
- Track presets, FX chain presets
- **Drum maps (`.drm`)** - this is the GM->SSD translation layer as Cubase sees
  it, and it is a global asset. Generating it from `maps/ssd.json` means the map
  propagates to all 20 projects for free.

### Template revisions

Each song's `spec.yaml` records `template.revision` and
`template.pending_migrations`. Migrations are applied **lazily** - the next time
that song is opened in Cubase - rather than in one painful batch. Bump
`CURRENT_TEMPLATE_REVISION` in `lib/spec.py` when the template changes, and add
the migration string to each affected song.

---

## Architecture: three tiers

Split by whether Cubase has to be running.

### Tier 1 - headless, no DAW. ~80% of the value. **Built.**

Pure MIDI transforms in Python with `mido`. Deterministic, batchable across all
20 songs, reviewable as a git diff, needs no Cubase and no human present.

### Tier 2 - headless, generating Cubase-consumable artifacts. **Not built.**

Files Cubase already imports, generated offline:

- **Drum Maps (`.drm`)** - I-note/O-note pairs. Global asset. *Verify first:
  whether `.drm` is cleanly writable or must be produced via the GUI once and
  then templated.*
- PLE/LE preset XML -> `~/Documents/Steinberg/Cubase/User Presets/Project Logical Editor`
- Macro entries in `Key Commands.xml`
- Track Archive XML
- **`.dawproject`** (zip + XML, open spec, Cubase supports import/export since
  14.0.20). **The highest-leverage target**: emit complete projects with tempo
  map, section markers, named/coloured/routed tracks, drum MIDI already placed,
  device chains in the XML. No running Cubase, no plugin-instantiation limits.
  *Caveat: it is an interchange format and the round trip is lossy - automation
  especially. Validate on a throwaway project before trusting it.*

### Tier 3 - requires Cubase running. Build last, keep thin. **Not built.**

MIDI Remote API (JavaScript, ES5, runs inside the host) + virtual MIDI port +
external daemon. See `bridge/README.md` for the full design notes - **read that
before writing any of it.**

API 1.3 (Cubase 15.0.20) confirmed capabilities: Channel EQ and Pre-filter as
host values; `get/setDisplayValue` for setting parameters by displayed value
("8000 Hz") rather than a normalized float; third-party plugin instantiation via
`trySetSlotPlugin`.

**Hard blockers in 1.3:** no track creation, no audio region rendering, no
external process communication. MIDI is the only transport in and out of the
script sandbox.

---

## The feedback loop

Without audio, the operator can read a diff from a phone but cannot judge
whether the groove is any good. So:

1. **Crude, working today:** `render/fluidsynth.py`. MIDI -> mp3 through a GM
   soundfont. Judges groove, dynamics and arrangement. Says nothing about SSD's
   actual tone.
2. **Real, not built:** a minimal headless JUCE VST3 host that loads SSD,
   consumes a MIDI file and writes a WAV, offline and faster than realtime. The
   operator builds JUCE plugins professionally; this is a few hundred lines.

**Note on why the loop does not run through Cubase:** rendering from Cubase
requires Export Audio Mixdown, which is a modal dialog - i.e. the brittle tier.
MIDI-in / audio-out through our own headless host is the reliable path. This is
the reason Tier 3 is last and thin.

### Preview mapping - important

Once MIDI is remapped to SSD note numbers, playing it through a **General MIDI**
soundfont produces the wrong sounds. So `./sb build` writes two files:

- `build/drums.mid` - SSD note numbers. **This is the artifact for Cubase.**
- `build/drums.preview.mid` - reverse-mapped back to GM, and the file that
  actually gets rendered to `build/drums.mp3`.

Never import the `.preview.mid` into Cubase.

### Where renders land

`build/*.mp3` is **committed to git**. That is deliberate: it is what makes a
render reachable from a phone (open the file on GitHub, download, listen). The
intermediate `.wav` is gitignored.

---

## Working in this repo

```bash
./sb status                      # where all 20 songs stand
./sb status --blocked            # what is stuck, and on what
./sb status --ready-for guitars-tracked
./sb status --json               # for feeding anything else
./sb validate                    # check the drum map and every spec
./sb build                       # build every song
./sb build example-riff          # build one
./sb build --no-render           # skip audio (fast)
./sb inspect path/to.mid         # what is actually in a MIDI file
./sb map show                    # the GM->SSD map as a table
./sb map doctor                  # what in the map is unverified or unfilled
./sb render in.mid out.mp3
./sb render --check              # is rendering possible on this machine?
./sb new my-song --title "My Song"
python3 -m pytest                # 69 tests
```

Answering "what changed since yesterday" needs no tooling:
`git log --since=yesterday -p -- 'songs/*/spec.yaml'`.

---

## `spec.yaml` reference

One per song. `songs/_template/spec.yaml` is the annotated blank;
`songs/example-riff/spec.yaml` is a filled-in working example.

| Field | Meaning |
|---|---|
| `status.stage` | `idea` -> `arranged` -> `drums-programmed` -> `scratch-tracked` -> `guitars-tracked` -> `vocals` -> `mixed` -> `mastered` -> `done` |
| `status.blocked_by` | Free text, or `null`. Non-null means blocked. |
| `status.next_action` | The single next thing. Write it for someone with no context - that someone is you, on a phone, in three weeks. |
| `template.revision` | Which template revision this song derives from. |
| `template.pending_migrations` | Migrations owed but not yet applied. |
| `musical.bpm` | Flattens the source's tempo map to this single tempo. |
| `sources` | Immutable inputs, relative to the song directory. |
| `build.seed` | Seed for deterministic transforms. Defaults to the slug. Change it to reroll humanisation. |
| `build.targets.<name>.transforms` | The ordered transform chain. |

### Transform reference

Order matters. **`remap` goes first** so that every later step is thinking in
SSD note numbers.

| Transform | Arguments | Does |
|---|---|---|
| `remap` | `channel` | GM -> SSD via `maps/ssd.json`. |
| `resolve_chokes` | `channel` | Cuts a ringing note when a choke-group sibling is struck (open hat when the hat closes). |
| `shape_velocity` | `accents`, `subdivision`, `beats_per_bar` | Scales velocity by position in the bar. Slots are `"<beat>.<sub>"`, plus `downbeat` / `onbeat` / `offbeat` wildcards. |
| `ghost_notes` | `density`, `velocity`, `velocity_spread` | Fills empty offbeat 16ths around the snare with quiet strokes. Uses SSD's ghost articulation if `maps/ssd.json` has a note for it. |
| `swing` | `amount` (0-1), `subdivision` | Delays every second subdivision. `1.0` = true triplet feel. |
| `humanize` | `timing_ticks`, `velocity_spread` | Pushes notes off the grid, reproducibly. The main antidote to Suno. |
| `scale_velocity` | `factor`, `notes` | Blunt velocity scaling, optionally for specific notes. |

Adding one: write the function in `lib/transforms.py`, add one entry to
`REGISTRY` in `lib/pipeline.py`, add a row to this table. Nothing else.

---

## The SSD map

`maps/ssd.json` is the single source of truth. `maps/gm.json` is a factual GM
percussion reference and should not be edited to taste.

> **STATUS: PROVISIONAL.** The current note numbers assume SSD's GM-compatible
> layout. That is the correct starting guess, but it has **not** been checked
> against the operator's actual kit. `./sb map doctor` lists exactly what is
> unverified.

To verify: play each pad in SSD, read the note number, set `to`, flip
`confidence` to `"verified"`, and set the top-level `verified: true`. Nine extra
SSD articulations (ghost snare, rimshot, hi-hat degrees, ride edge, chokes) have
no note numbers yet and are inert until filled in - a transform that wants one
falls back gracefully rather than failing.

---

## Design notes worth not re-deriving

Three non-obvious things this code already gets right. If you rewrite any of it,
keep these properties:

- **Note-off pairing in `lib/remap.py`.** A velocity-conditional route is chosen
  from the *note_on* velocity; re-running the lookup on the note_off (velocity 0)
  would route it elsewhere and strand the note. And because folding is
  many-to-one (GM 38 and GM 40 both land on SSD 38), overlapping voices are
  reference-counted so the first note_off does not silence a note that is still
  ringing.
- **Jitter is hashed, not drawn.** `lib/transforms.py` derives each note's
  humanisation from a hash of its own identity rather than from a running RNG.
  With an RNG, inserting one note at bar 1 changes every later note and a
  one-note edit becomes a whole-file diff.
- **Structure is read before transforming, not after.** `ghost_notes` and
  `humanize` are position-dependent, so after they run no two bars are identical
  and every repeat looks like a new section. `lib/pipeline.py` detects sections
  from the source.

---

## Verify before building

Recorded here so it does not get lost. All of this is secondhand or recent and
should be checked against a primary source before code depends on it:

- Exact `trySetSlotPlugin` signature and calling convention (docs lag; check the
  Steinberg forum thread and the API reference).
- `.dawproject` device-chain XML schema, and how faithfully Cubase 15 imports it.
- Whether `.drm` is cleanly writable, or must be produced via the GUI and templated.
- **The actual note layout of the operator's SSD kit** - the operator supplies this.
- `.vmx` binary format. Assume opaque and GUI-only until proven otherwise.

---

## Build order

1. ~~`mido` transform library + `maps/ssd.json`~~ **done**
2. ~~`spec.yaml` schema + `build.py`~~ **done**
3. ~~fluidsynth render~~ **done**
4. PM layer + status tracking - see `pm/README.md`. Needed early; 20 songs gets
   unmanageable fast. (`./sb status` already covers the phone-first views.)
5. `.dawproject` writer - validate on a throwaway before trusting
6. MIDI bridge SysEx protocol + daemon - see `bridge/README.md`
7. JUCE headless SSD host - replaces #3, biggest quality jump
8. MIDI Remote script - live-session ops, `trySetSlotPlugin`, EQ/filter control

1-5 need no Cubase running and are the bulk of the value.
