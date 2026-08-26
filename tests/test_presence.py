"""The Joe-presence guard for heavy background jobs.

Pinned after the 2026-08-26 incident (stroke backfill starved the
desktop's session lists while Joe sat at it). The ctypes signatures
must stay TYPED — the bare-windll SetPriorityClass silent no-op is the
standing lesson — and failure must fail toward "present" (jobs defer),
never toward "absent" (jobs run over Joe).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import _presence


class TestPresence:
    def test_idle_seconds_returns_sane_value(self):
        v = _presence.idle_seconds()
        assert isinstance(v, float)
        assert 0.0 <= v < 60 * 60 * 24 * 50   # under the tick rollover span

    def test_ctypes_signatures_are_typed(self):
        # untyped windll calls no-op silently; these must stay declared
        assert _presence._user32.GetLastInputInfo.restype is not None
        assert _presence._user32.GetLastInputInfo.argtypes is not None
        assert _presence._kernel32.GetTickCount.restype is not None

    def test_api_failure_reads_as_present(self, monkeypatch):
        monkeypatch.setattr(_presence._user32, "GetLastInputInfo",
                            lambda ref: 0)
        assert _presence.idle_seconds() == 0.0
        assert _presence.joe_present() is True   # fail safe: defer the job

    def test_threshold_direction(self, monkeypatch):
        monkeypatch.setattr(_presence, "idle_seconds", lambda: 30.0)
        assert _presence.joe_present(idle_min=15) is True
        monkeypatch.setattr(_presence, "idle_seconds", lambda: 16 * 60.0)
        assert _presence.joe_present(idle_min=15) is False
