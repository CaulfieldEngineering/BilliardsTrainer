"""THE QUEUE MUST NOT ROT SILENTLY.

docs/BACKLOG.md Tier 0 is what every autonomous round, watchdog recovery
and fresh session reads to choose its next target. Rounds 34-47 updated
it with anchor-text `str.replace()` calls whose anchors no longer
existed. A missing needle is not an error - `replace` returns the string
unchanged - so fourteen rounds printed "backlog updated" while writing
nothing. The queue still advertised object-ball naming at 74.5% with an
invented "11" when the engine measured 99.4% with none, which would send
the next session off to re-fix solved problems.

These tests fail the suite if the machine-written state block is gone,
empty, or hand-edited into an unparseable shape, and if the writer is
ever allowed to no-op again.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BACKLOG = ROOT / "docs" / "BACKLOG.md"


def _load():
    spec = importlib.util.spec_from_file_location(
        "campaign_state", ROOT / "tools" / "campaign_state.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["campaign_state"] = mod
    spec.loader.exec_module(mod)
    return mod


class TestCampaignStateBlock:
    def test_backlog_has_the_state_markers(self):
        src = BACKLOG.read_text(encoding="utf-8")
        assert "<!-- CAMPAIGN-STATE:BEGIN" in src, (
            "Tier 0's machine-written state block is gone; the queue can "
            "go stale again without anything failing")
        assert "<!-- CAMPAIGN-STATE:END -->" in src

    def test_the_block_is_not_empty(self):
        src = BACKLOG.read_text(encoding="utf-8")
        i = src.find("<!-- CAMPAIGN-STATE:BEGIN")
        block = src[src.find("-->", i) + 3:src.find("<!-- CAMPAIGN-STATE:END")]
        assert "written" in block, "state block is empty or unparseable"
        assert "rules_v" in block, "state block does not name the engine rules"

    def test_writer_refuses_to_no_op_when_markers_are_missing(self, tmp_path):
        """The exact failure that rotted the queue: a silent no-op."""
        mod = _load()
        fake = tmp_path / "BACKLOG.md"
        fake.write_text("# no markers here\n", encoding="utf-8")
        mod.BACKLOG = fake
        with pytest.raises(mod.MarkerMissing):
            mod.write("anything")
        assert fake.read_text(encoding="utf-8") == "# no markers here\n", (
            "the writer must not touch a file it cannot place the block in")

    def test_writer_replaces_the_block_in_place(self, tmp_path):
        mod = _load()
        fake = tmp_path / "BACKLOG.md"
        fake.write_text(
            "top\n<!-- CAMPAIGN-STATE:BEGIN note -->\nOLD\n"
            "<!-- CAMPAIGN-STATE:END -->\nbottom\n", encoding="utf-8")
        mod.BACKLOG = fake
        mod.write("NEW BODY")
        out = fake.read_text(encoding="utf-8")
        assert "NEW BODY" in out and "OLD" not in out
        assert out.startswith("top\n") and out.rstrip().endswith("bottom")
