import pytest
import yaml

from lib.spec import CURRENT_TEMPLATE_REVISION, STAGES, Spec, discover


def write_spec(tmp_path, body, name="song-a"):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "spec.yaml").write_text(body)
    return d / "spec.yaml"


# Kept flush-left so tests can substitute and append without indentation games.
BASE = """slug: song-a
title: Song A
status:
  stage: arranged
musical:
  bpm: 120
  time_signature: [4, 4]
template:
  revision: 1
sources: {}
build:
  targets: {}
"""


def test_loads_and_reads_status(tmp_path):
    spec = Spec.load(write_spec(tmp_path, BASE))
    assert spec.slug == "song-a"
    assert spec.stage == "arranged"
    assert spec.stage_index == STAGES.index("arranged")
    assert not spec.is_blocked
    assert spec.validate() == []


def test_seed_defaults_to_slug(tmp_path):
    assert Spec.load(write_spec(tmp_path, BASE)).seed == "song-a"


def test_unknown_keys_survive_a_round_trip(tmp_path):
    """The operator will put things in here that the code doesn't know about.
    Losing them on save would be unforgivable."""
    path = write_spec(tmp_path, BASE + "\nmy_own_field: keep me\n")
    spec = Spec.load(path)
    spec.set_stage("mixed")
    spec.save()
    assert yaml.safe_load(path.read_text())["my_own_field"] == "keep me"


def test_bad_stage_is_an_error(tmp_path):
    spec = Spec.load(write_spec(tmp_path, BASE.replace("stage: arranged", "stage: noodling")))
    assert any(i.field == "status.stage" and i.level == "error" for i in spec.validate())


def test_implausible_bpm_is_an_error(tmp_path):
    spec = Spec.load(write_spec(tmp_path, BASE.replace("bpm: 120", "bpm: 9000")))
    assert any(i.field == "musical.bpm" and i.level == "error" for i in spec.validate())


def test_missing_source_file_is_an_error(tmp_path):
    spec = Spec.load(write_spec(tmp_path, BASE.replace("sources: {}", "sources:\n  drums: suno/nope.mid")))
    assert any(i.field == "sources.drums" and i.level == "error" for i in spec.validate())


def test_target_referencing_undefined_source_is_an_error(tmp_path):
    body = BASE.replace("targets: {}", "targets:\n    drums:\n      source: ghost\n")
    spec = Spec.load(write_spec(tmp_path, body))
    assert any("source" in i.field and i.level == "error" for i in spec.validate())


def test_stale_template_revision_warns(tmp_path):
    body = BASE.replace("revision: 1", "revision: 0")
    spec = Spec.load(write_spec(tmp_path, body))
    assert any(i.field == "template.revision" for i in spec.validate())


def test_set_stage_rejects_nonsense(tmp_path):
    spec = Spec.load(write_spec(tmp_path, BASE))
    with pytest.raises(ValueError, match="unknown stage"):
        spec.set_stage("done-ish")


def test_beats_per_bar_from_time_signature(tmp_path):
    spec = Spec.load(write_spec(tmp_path, BASE.replace("[4, 4]", "[6, 8]")))
    assert spec.beats_per_bar == 3


def test_discover_skips_underscore_dirs(tmp_path):
    write_spec(tmp_path, BASE, name="song-a")
    write_spec(tmp_path, BASE, name="_template")
    assert [s.slug for s in discover(tmp_path)] == ["song-a"]
