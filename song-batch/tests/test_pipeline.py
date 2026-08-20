import shutil
import sys
from pathlib import Path

import mido
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.drummap import DrumMap
from lib.midi_io import describe
from lib.pipeline import PipelineError, apply_chain, build_song, Context
from lib.spec import Spec
from tools.make_fixture import build as build_fixture

SPEC = """slug: t
title: T
status:
  stage: drums-programmed
musical:
  bpm: 148
  time_signature: [4, 4]
template:
  revision: 1
sources:
  drums: suno/drums.mid
build:
  targets:
    drums:
      source: drums
      render: false
      transforms:
        - name: remap
        - name: resolve_chokes
        - name: shape_velocity
          accents: {downbeat: 1.1, offbeat: 0.85}
        - name: ghost_notes
          density: 0.3
        - name: humanize
          timing_ticks: 6
          velocity_spread: 12
"""


@pytest.fixture
def song(tmp_path):
    d = tmp_path / "t"
    (d / "suno").mkdir(parents=True)
    build_fixture().save(str(d / "suno" / "drums.mid"))
    (d / "spec.yaml").write_text(SPEC)
    return Spec.load(d / "spec.yaml")


def test_end_to_end_build_produces_midi(song):
    results = build_song(song, DrumMap.load(), render_audio=False)
    assert len(results) == 1
    result = results[0]
    assert result.midi_out.exists()
    assert result.steps == ["remap", "resolve_chokes", "shape_velocity", "ghost_notes", "humanize"]
    assert result.report.total_in > 0


def test_build_is_byte_reproducible(song):
    """Two builds of an unchanged spec must produce an identical file, or the
    'review the build as a git diff' workflow is worthless."""
    first = build_song(song, DrumMap.load(), render_audio=False)[0].midi_out.read_bytes()
    second = build_song(song, DrumMap.load(), render_audio=False)[0].midi_out.read_bytes()
    assert first == second


def test_changing_the_seed_changes_the_output(song):
    a = build_song(song, DrumMap.load(), render_audio=False)[0].midi_out.read_bytes()
    song.raw["build"]["seed"] = "different"
    b = build_song(song, DrumMap.load(), render_audio=False)[0].midi_out.read_bytes()
    assert a != b


def test_arrangement_is_read_from_the_source_not_the_output(song):
    """Ghost notes make every bar unique, so structure has to be read before
    the transforms run."""
    result = build_song(song, DrumMap.load(), render_audio=False)[0]
    assert [s.label for s in result.sections] == ["A", "A", "B", "A"]


def test_tempo_is_flattened_to_the_spec_value(song):
    from lib.tempo import tempo_map

    result = build_song(song, DrumMap.load(), render_audio=False)[0]
    changes = tempo_map(mido.MidiFile(str(result.midi_out)))
    assert len(changes) == 1
    assert round(changes[0].bpm) == 148


def test_preview_midi_is_written_and_maps_back_to_gm(song):
    result = build_song(song, DrumMap.load(), render_audio=False)[0]
    assert result.preview_midi.exists()
    pitches = describe(mido.MidiFile(str(result.preview_midi)))["pitches"]
    assert all(0 <= p <= 127 for p in pitches)


def test_ghost_notes_actually_added_notes(song):
    result = build_song(song, DrumMap.load(), render_audio=False)[0]
    before = describe(build_fixture())["notes"]
    after = describe(mido.MidiFile(str(result.midi_out)))["notes"]
    assert after > before


def test_unknown_transform_is_a_clear_error(song):
    song.raw["build"]["targets"]["drums"]["transforms"] = [{"name": "reticulate_splines"}]
    with pytest.raises(PipelineError, match="unknown transform"):
        build_song(song, DrumMap.load(), render_audio=False)


def test_missing_source_is_a_clear_error(song):
    song.raw["sources"]["drums"] = "suno/absent.mid"
    with pytest.raises(PipelineError, match="missing"):
        build_song(song, DrumMap.load(), render_audio=False)


def test_target_pointing_at_undefined_source_is_a_clear_error(song):
    song.raw["build"]["targets"]["drums"]["source"] = "nope"
    with pytest.raises(PipelineError, match="not in sources"):
        build_song(song, DrumMap.load(), render_audio=False)


def test_disabled_target_is_skipped(song):
    song.raw["build"]["targets"]["drums"]["enabled"] = False
    assert build_song(song, DrumMap.load(), render_audio=False) == []


def test_build_prunes_stale_artifacts(song):
    """build/ is entirely derived. A target turned off must not leave its last
    output sitting there looking current - especially as build/ is committed."""
    build_song(song, DrumMap.load(), render_audio=False)
    stale = song.build_dir / "drums.mp3"
    stale.write_bytes(b"an old render nobody regenerates any more")
    build_song(song, DrumMap.load(), render_audio=False)
    assert not stale.exists()


def test_targeted_build_does_not_prune_other_targets(song):
    """--target rebuilds one thing; it must not delete the others' outputs."""
    build_song(song, DrumMap.load(), render_audio=False)
    other = song.build_dir / "bass.mid"
    other.write_bytes(b"another target's output")
    build_song(song, DrumMap.load(), render_audio=False, only=["drums"])
    assert other.exists()
