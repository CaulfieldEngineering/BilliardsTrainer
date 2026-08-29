"""The committed colour references must stay able to do their job.

Round 33's naming result (99.5%) rests on these: the identifier reads the
purple 4 as the 7 under Joe's warm light, and the correction that repairs it
compares the crop against per-ball colours measured from this table. Those
values used to live only in APP_DIR, rebuilt from a corpus that is not in git -
so a fresh clone silently lost the fix. docs/colour_refs.json is now the
version of record, restored by `tools/build_colour_refs.py --install`.

These tests pin the properties the naming path actually depends on, not the
exact numbers - re-measuring on new footage should be free to move them.
"""

import json
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "colour_refs.json"
MEASURED = {"0", "1", "2", "3", "4", "9"}      # the bench table's six balls
MIN_REF_SAMPLES = 5                            # core.balls._MIN_REF_SAMPLES


@pytest.fixture(scope="module")
def refs():
    return json.loads(DOC.read_text(encoding="utf-8"))["refs"]


def _lab(bgr):
    a = np.array([[list(bgr)]], np.uint8)
    return cv2.cvtColor(a, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)


def test_every_ball_on_the_bench_table_has_a_reference(refs):
    assert MEASURED <= set(refs), "the bench table's own balls must be covered"


def test_measured_entries_carry_enough_samples_to_be_loaded(refs):
    """core.balls ignores a reference with fewer than _MIN_REF_SAMPLES."""
    for k in sorted(MEASURED):
        n = int(refs[k].get("n", 0))
        assert n >= MIN_REF_SAMPLES, f"ball {k} has only {n} samples"


def test_the_purple_4_is_not_recorded_as_navy(refs):
    """THE round-33 case. The orphaned 2026-08-15 file had the 4 at
    BGR (142.5, 25.5, 36.0) - navy, sitting beside the real blue 2 - which
    is why the 4 could not be told apart from a 2 or repaired from a 7."""
    four = refs["4"]["bgr"]
    stale_navy = (142.5, 25.5, 36.0)
    drift = float(np.linalg.norm(_lab(four) - _lab(stale_navy)))
    assert drift > 5.0, (
        f"ball 4 has drifted back to the navy value that caused the misread "
        f"({four})")


def test_the_confusable_pair_4_and_2_stay_far_apart(refs):
    """Colour is what separates them; if this collapses, naming does."""
    d = float(np.linalg.norm(_lab(refs["4"]["bgr"]) - _lab(refs["2"]["bgr"])))
    assert d > 60.0, f"the purple 4 and the blue 2 are only {d:.1f} Lab apart"


def test_inherited_entries_are_labelled_as_unvalidated(refs):
    """Balls not on this table carry 2026-08-15 values that phantoms match
    (round 27). They may stay, but they must not look measured."""
    for k, v in refs.items():
        if k in MEASURED:
            assert v["source"].startswith("measured"), k
        else:
            assert "UNVALIDATED" in v["source"], k


def test_the_record_matches_what_install_would_write(refs):
    """--install strips the provenance key and writes the rest verbatim;
    anything else would mean the live file and the record disagree."""
    for k, v in refs.items():
        assert set(v) - {"source"} == {"bgr", "lab", "n"}, f"ball {k}: {set(v)}"
