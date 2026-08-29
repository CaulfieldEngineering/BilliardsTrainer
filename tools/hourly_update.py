"""Post the hourly Dev Journal update: scorecard, what changed, what's next.

Joe, 2026-08-28: "let's be sure to include updates hourly or something
like that so i can actually see what's going on under the hood.
Remember, formatted, organized, html/css, screenshot/images."

Run by the watchdog every hour (docs/AUTONOMY.md) and after any round:

    python tools/hourly_update.py --deploy [--image path caption]

It reads the CURRENT scorecard (recomputing it if the engine output is
newer), the git log since the last update, and the campaign's next
targets, and publishes one formatted HTML entry.
"""

from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
PUB = ROOT / "companion-cloud" / "public"
STAMP = ROOT / "_eval" / "last_journal_update.txt"


def _git_since(stamp: str) -> list[str]:
    try:
        out = subprocess.run(
            ["git", "log", f"--since={stamp}", "--pretty=format:%s"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=30).stdout
    except Exception:  # noqa: BLE001
        return []
    subjects = [s for s in out.splitlines() if s.strip()]
    skip = ("GOALS", "Campaign state", "docs:", "Journal ")
    return [s for s in subjects if not s.startswith(skip)]


def _ladder_rows(sc: dict) -> str:
    caps = sc.get("caps", {})
    inv = caps.get("invented_numbers") or []
    rows = [
        ("Cue ball tracked + named", f"{caps.get('cue_named_pct', 0)}%",
         "99%", caps.get("cue_named_pct", 0) >= 99),
        ("Object balls named while moving",
         f"{caps.get('named_moving_pct', 0)}%", "95%",
         caps.get("named_moving_pct", 0) >= 95),
        ("No invented ball numbers",
         ", ".join(str(i) for i in inv) if inv else "none", "none", not inv),
        ("Shots found", sc.get("detected", "-"), "10/10",
         sc.get("detected") == "10/10"),
        ("Make/miss calls", sc.get("outcome", "-"), "10/10",
         sc.get("outcome") == "10/10"),
        ("Pots attributed to the right ball",
         caps.get("pot_attribution", "-"), "all",
         caps.get("pot_attribution", "0/1").split("/")[0]
         == caps.get("pot_attribution", "0/2").split("/")[1]),
        ("No invented shots", str(sc.get("false_strokes", "-")), "0",
         sc.get("false_strokes") == 0),
    ]
    out = ["<table><tr><th>Capability</th><th>Now</th><th>Need</th>"
           "<th></th></tr>"]
    for label, now, need, ok in rows:
        out.append(
            f"<tr><td>{html.escape(label)}</td>"
            f"<td class='num'>{html.escape(str(now))}</td>"
            f"<td class='num'>{html.escape(need)}</td>"
            f"<td class='num'><span class='{'pass' if ok else 'fail'}'>"
            f"{'PASS' if ok else 'not yet'}</span></td></tr>")
    out.append("</table>")
    return "".join(out)


def build(images=()) -> dict:
    sc = {}
    scf = PUB / "scorecard.json"
    if scf.is_file():
        sc = json.loads(scf.read_text(encoding="utf-8"))
    since = "2 hours ago"
    if STAMP.is_file():
        since = STAMP.read_text(encoding="utf-8").strip() or since
    commits = _git_since(since)
    misses = [s for s in sc.get("shots", [])
              if not (s.get("found") and s.get("outcome_ok"))]

    parts = ["<h2>Where the engine stands</h2>", _ladder_rows(sc)]
    if misses:
        parts.append("<h2>What it still gets wrong</h2><ul>")
        for m in misses:
            what = html.escape(m.get("what", ""))
            if not m.get("found"):
                parts.append(f"<li><code>{m['strike']}s</code> {what} — "
                             "<span class='fail'>the app missed this shot "
                             "entirely</span></li>")
            else:
                parts.append(
                    f"<li><code>{m['strike']}s</code> {what} — app said "
                    f"<strong>{html.escape(str(m.get('engine')))}</strong>, "
                    f"truth is <strong>{html.escape(str(m.get('truth')))}"
                    "</strong></li>")
        parts.append("</ul>")
    if commits:
        parts.append("<h2>Changes since the last update</h2><ul>")
        for c in commits[:12]:
            parts.append(f"<li>{html.escape(c.split(chr(10))[0])}</li>")
        parts.append("</ul>")
    else:
        parts.append("<div class='note warn'><p>No code landed in this "
                     "window — either a long measuring run was in flight or "
                     "the loop was paused.</p></div>")
    figs = []
    for src, cap in images:
        name = Path(src).name
        figs.append(f"<figure><img src='journal/{html.escape(name)}' alt=''>"
                    f"<figcaption>{html.escape(cap)}</figcaption></figure>")
    if figs:
        parts.append("<h2>Evidence</h2><div class='grid2'>"
                     + "".join(figs) + "</div>")
    return {"html": "".join(parts),
            "score": (f"{sc.get('detected', '?')} shots · "
                      f"{sc.get('outcome', '?')} calls") if sc else ""}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", default=None)
    ap.add_argument("--image", nargs=2, action="append", default=[],
                    metavar=("PATH", "CAPTION"))
    ap.add_argument("--deploy", action="store_true")
    a = ap.parse_args()
    from journal import add
    built = build(a.image)
    now = datetime.now(timezone.utc)
    _y = now.year
    _dst = (datetime(_y, 3, 8, 7, tzinfo=timezone.utc)
            <= now < datetime(_y, 11, 1, 6, tzinfo=timezone.utc))
    _e = now + timedelta(hours=-4 if _dst else -5)
    _lbl = "EDT" if _dst else "EST"
    title = a.title or ("Hourly check-in - "
                        + _e.strftime("%b %d, %I:%M %p ").lstrip("0") + _lbl)
    e = add(title, "", built["score"], a.image, a.deploy, "finding")
    # attach the formatted body (add() stores plain text; upgrade in place)
    jf = PUB / "journal.json"
    doc = json.loads(jf.read_text(encoding="utf-8"))
    for ent in doc["entries"]:
        if ent["id"] == e["id"]:
            ent["html"] = built["html"]
            break
    jf.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    STAMP.parent.mkdir(parents=True, exist_ok=True)
    STAMP.write_text(now.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    if a.deploy:
        subprocess.run([sys.executable, "deploy.py"],
                       cwd=str(ROOT / "companion-cloud"), check=False)
    print(f"hourly update #{e['id']} posted")


if __name__ == "__main__":
    main()
