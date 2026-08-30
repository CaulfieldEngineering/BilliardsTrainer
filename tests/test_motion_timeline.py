"""The instrument that can see a MISSED shot.

Round 53. Every metric in this campaign scores shots a truth file
already lists, so a stroke the engine failed to report is invisible to
all of them - absent from the shot list, absent from the tracked stream,
absent from the scorecard. tools/motion_timeline.py measures the one
thing the engine cannot suppress: pixels moving on the cloth. These pin
the burst grouping, which is what decides where a human has to look.
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "motion_timeline", ROOT / "tools" / "motion_timeline.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["motion_timeline"] = mod
    spec.loader.exec_module(mod)
    return mod


class TestBursts:
    def test_quiet_footage_yields_nothing(self):
        m = _load()
        times = [i / 30 for i in range(90)]
        assert m.bursts(times, [0.0001] * 90, floor=0.004) == []

    def test_one_run_of_motion_is_one_burst(self):
        m = _load()
        times = [i / 30 for i in range(90)]
        energy = [0.0001] * 30 + [0.02] * 30 + [0.0001] * 30
        out = m.bursts(times, energy, floor=0.004)
        assert len(out) == 1
        t0, t1, peak = out[0]
        assert abs(t0 - 1.0) < 0.1 and abs(t1 - 1.97) < 0.1
        assert abs(peak - 0.02) < 1e-9

    def test_a_brief_dip_does_not_split_a_burst(self):
        """A ball passing behind the cue stick must not read as two shots."""
        m = _load()
        times = [i / 30 for i in range(120)]
        energy = ([0.0001] * 20 + [0.02] * 20 + [0.0001] * 10
                  + [0.02] * 20 + [0.0001] * 50)
        assert len(m.bursts(times, energy, floor=0.004, gap_s=1.0)) == 1

    def test_separate_events_stay_separate(self):
        m = _load()
        times = [i / 30 for i in range(300)]
        energy = ([0.0001] * 20 + [0.02] * 20 + [0.0001] * 200
                  + [0.02] * 20 + [0.0001] * 40)
        assert len(m.bursts(times, energy, floor=0.004, gap_s=1.0)) == 2
