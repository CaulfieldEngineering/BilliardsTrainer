"""Write the campaign's CURRENT state into docs/BACKLOG.md, from the
measurements — never by hand.

WHY THIS EXISTS (round 48). BACKLOG Tier 0 is THE queue: every
autonomous round, watchdog recovery and fresh session picks its target
by reading it. Rounds 34-47 updated it with anchor-text `.replace()`
calls whose anchor strings no longer existed. `str.replace` with a
missing needle is not an error - it returns the string unchanged - so
fourteen rounds of "backlog updated" were printed while nothing was
written. The queue rotted for two weeks of campaign time and still
advertised naming at 74.5% with an invented "11" when the engine was at
99.4% with none. A session trusting it would go and re-fix solved
problems.

So: the state block is MACHINE-WRITTEN from the bench sidecar and shot
summary, this tool RAISES if its markers are missing (never a silent
no-op), and tests/test_campaign_state.py fails the suite if the block
is absent, unparseable or older than the newest bench measurement.

    python tools/campaign_state.py            # rewrite the block
    python tools/campaign_state.py --check    # exit 1 if stale
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

BACKLOG = ROOT / "docs" / "BACKLOG.md"
BEGIN = "<!-- CAMPAIGN-STATE:BEGIN"
END = "<!-- CAMPAIGN-STATE:END -->"
BENCH = "session-20260824-220247.mp4"


class MarkerMissing(RuntimeError):
    """The block markers are gone - refuse to write, never no-op."""


def _bench_paths() -> tuple[Path, Path]:
    """Sidecar and shot summary for the bench clip.

    The summary is written BESIDE THE RECORDING, which is Joe's synced
    Dropbox folder when RecordingSettings.directory is set - not
    EXPORTS_DIR. Looking only in EXPORTS_DIR found nothing and the block
    rendered "? entries", which is the same silent half-failure this
    tool exists to stop.
    """
    from billiards_trainer.config import APP_DIR, EXPORTS_DIR, Settings
    sidecar = Path(APP_DIR) / "m1" / f"{BENCH}.analysis.jsonl"
    roots = []
    try:
        d = (Settings.load().recording.directory or "").strip()
        if d:
            roots.append(Path(d))
    except Exception:                                # noqa: BLE001
        pass
    roots.append(Path(EXPORTS_DIR))
    for root in roots:
        p = root / f"{BENCH}.shots.json"
        if p.is_file():
            return sidecar, p
    return sidecar, roots[0] / f"{BENCH}.shots.json"


def measure() -> dict:
    """Read the CURRENT bench measurement. No opinions, no history."""
    sidecar, summary = _bench_paths()
    out: dict = {"bench": BENCH, "sidecar_exists": sidecar.is_file()}
    if sidecar.is_file():
        meta = json.loads(sidecar.read_text(encoding="utf-8").splitlines()[0])
        out["rules_v"] = meta.get("rules_v")
        out["measured_utc"] = dt.datetime.fromtimestamp(
            sidecar.stat().st_mtime, dt.timezone.utc).strftime(
                "%Y-%m-%dT%H:%MZ")
    if summary.is_file():
        doc = json.loads(summary.read_text(encoding="utf-8"))
        shots = doc.get("shots", doc) if isinstance(doc, dict) else doc
        out["shots"] = len(shots)
        out["strokes"] = sum(1 for s in shots
                             if (s.get("action") or "") == "stroke")
        out["makes"] = sum(1 for s in shots
                           if (s.get("outcome") or "") == "make")
    try:
        from scorecard import bench_report          # type: ignore
        out.update(bench_report() or {})
    except Exception:                                # noqa: BLE001
        pass                                        # scorecard is optional here
    return out


def render(m: dict) -> str:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    lines = [
        "**CURRENT STATE — machine-written, do not hand-edit.**",
        "",
        f"    written        {stamp}",
        f"    bench          {m.get('bench')}",
        f"    engine rules_v {m.get('rules_v')}",
        f"    measured       {m.get('measured_utc', 'never')}",
        f"    shot list      {m.get('shots', '?')} entries "
        f"({m.get('strokes', '?')} strokes, {m.get('makes', '?')} makes)"
        + ("" if m.get("shots") is not None
           else "   <-- SUMMARY NOT FOUND, this block is incomplete"),
        "",
        "Run `python tools/scorecard.py` for the full card; that is the",
        "gate of record. This block exists so a session picking up the",
        "queue cannot be told something the measurements disagree with.",
    ]
    return "\n".join(lines)


def write(text: str | None = None) -> str:
    src = BACKLOG.read_text(encoding="utf-8")
    i, j = src.find(BEGIN), src.find(END)
    if i < 0 or j < 0 or j < i:
        raise MarkerMissing(
            f"{BACKLOG} has no CAMPAIGN-STATE markers - refusing to write. "
            "This is the exact silent no-op that rotted the queue for "
            "fourteen rounds; restore the markers rather than removing "
            "this check.")
    head_end = src.find("-->", i)
    if head_end < 0:
        raise MarkerMissing("CAMPAIGN-STATE:BEGIN comment is unterminated")
    body = text if text is not None else render(measure())
    out = src[:head_end + 3] + "\n\n" + body + "\n\n" + src[j:]
    BACKLOG.write_text(out, encoding="utf-8")
    return body


def check() -> int:
    src = BACKLOG.read_text(encoding="utf-8")
    if BEGIN not in src or END not in src:
        print("FAIL: CAMPAIGN-STATE markers missing from docs/BACKLOG.md")
        return 1
    i = src.find(END)
    block = src[src.find("-->", src.find(BEGIN)) + 3:i]
    if "written" not in block:
        print("FAIL: CAMPAIGN-STATE block is empty or unparseable")
        return 1
    sidecar, _ = _bench_paths()
    if sidecar.is_file():
        m = re.search(r"written\s+(\S+)", block)
        meas = dt.datetime.fromtimestamp(sidecar.stat().st_mtime,
                                         dt.timezone.utc)
        if m:
            try:
                written = dt.datetime.strptime(
                    m.group(1), "%Y-%m-%dT%H:%MZ").replace(
                        tzinfo=dt.timezone.utc)
            except ValueError:
                print("FAIL: CAMPAIGN-STATE timestamp is unparseable")
                return 1
            if written < meas - dt.timedelta(minutes=5):
                print(f"FAIL: CAMPAIGN-STATE written {m.group(1)} is older "
                      f"than the bench measurement "
                      f"{meas:%Y-%m-%dT%H:%MZ} - run "
                      "`python tools/campaign_state.py`")
                return 1
    print("CAMPAIGN-STATE ok")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the block is missing or stale")
    a = ap.parse_args()
    if a.check:
        return check()
    print(write())
    return 0


if __name__ == "__main__":
    sys.exit(main())
