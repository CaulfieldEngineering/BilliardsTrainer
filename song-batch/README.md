# song-batch

A remote-operable music production pipeline. Develop ~20 songs in parallel in
Cubase, with as much of the work as possible driven by Claude Code **from a
phone, while away from the computer**.

The bet: most of what stalls a song between "riff idea" and "ready to track" is
mechanical — remapping drum MIDI, undoing Suno's quantisation, shaping
velocities, keeping track of what is blocked on what. None of that needs a DAW
open, or hands on a guitar. It needs a deterministic batch pipeline and a way to
hear the result on a phone.

## Quick start

```bash
pip install -r requirements-dev.txt
./sb validate          # check the drum map and every song spec
./sb build             # transform + render every song
./sb status            # where all 20 stand
python3 -m pytest      # 69 tests
```

`./sb build` on the bundled example produces:

```
example-riff - Example Riff  [drums-programmed]
  drums: remap -> resolve_chokes -> shape_velocity -> ghost_notes -> humanize
    GM  36 == SSD  36  x42    Bass Drum 1
    GM  38 == SSD  38  x32    Acoustic Snare
    ...
    arrangement: A A B A  |  4 bars each, 16 bars total  (* = fill)
    -> songs/example-riff/build/drums.mid          <- for Cubase
    -> songs/example-riff/build/drums.preview.mid  <- reverse-mapped to GM
    -> songs/example-riff/build/drums.mp3          <- listen on your phone
```

## What is here

| Path | |
|---|---|
| `CLAUDE.md` | **Start here.** Conventions, the SSD map, Cubase template invariants, how to operate this from a phone. |
| `lib/` | The transform library. Deterministic MIDI transforms, GM→SSD remapping, section detection, the `spec.yaml` schema. |
| `maps/ssd.json` | Single source of truth for GM→SSD note translation. |
| `songs/<slug>/` | Per song: `spec.yaml` (status + build config), `suno/` (immutable inputs), `build/` (derived), `notes.md`. |
| `render/` | fluidsynth wrapper today; a headless JUCE VST3 host later. |
| `pm/`, `bridge/` | Design notes for the parts not built yet. Read before building them. |
| `tools/make_fixture.py` | Generates a synthetic Suno-like export so the pipeline runs before real material exists. |

## Status

**Tier 1 (headless MIDI transforms) is built and tested.** That is roughly 80%
of the value and needs no Cubase, no plugins and nobody present.

Not built yet, in intended order: the PM sync layer, the `.dawproject` writer,
the MIDI Remote SysEx bridge, and the JUCE headless SSD host. `CLAUDE.md` has
the full build order and `pm/README.md` / `bridge/README.md` carry the design
decisions already made for each.

**The drum map is provisional** — it currently assumes SSD's GM-compatible
layout, which has not been checked against the real kit. Run `./sb map doctor`
to see exactly what is unverified.

## Design properties worth preserving

- **Builds are byte-reproducible.** Same spec, same seed, identical output file.
  This is what makes a build reviewable as a git diff from a phone.
- **Inputs are immutable, outputs are derived.** `build/` can be deleted at any
  time and regenerated.
- **The repo answers the operational questions on its own.** `./sb status` works
  with every external service unreachable.
