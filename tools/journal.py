"""Post a Dev Journal entry to the phone app (Joe: "I'd like to pop in
and follow along with your findings... comparison images as well. Just
layman explanations").

The autonomous loop calls this at the end of every grind round, so the
journal stays current without anyone remembering to write it.

    python tools/journal.py --title "..." --score "9/10 shots" \
        --body "Plain language, one idea per line." \
        --image path.png "caption" [--image ...] [--deploy]

Rules for the body: layman terms, no jargon, no metric names Joe hasn't
seen. Say what was wrong, what it looked like, what changed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "companion-cloud" / "public"
JSON_PATH = PUB / "journal.json"
IMG_DIR = PUB / "journal"


def add(title: str, body: str, score: str = "", images=(),
        deploy: bool = False, kind: str = "finding") -> dict:
    """kind: "finding" (engine-campaign round) or "update" (a change Joe
    can see in the app). The Dev Journal is the ONE channel - What's New
    was merged into it 2026-08-28."""
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    doc = {"entries": []}
    if JSON_PATH.is_file():
        try:
            doc = json.loads(JSON_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    entries = doc.get("entries", [])
    nid = (max((e.get("id", 0) for e in entries), default=0)) + 1
    stamp = datetime.now(timezone.utc)
    # Joe reads these on his own clock: US Eastern. DST rule is the
    # simple one (Mar 2nd Sun - Nov 1st Sun); the ISO ts stays UTC so
    # sorting never depends on the display zone.
    _y = stamp.year
    _dst = (datetime(_y, 3, 8, 7, tzinfo=timezone.utc)
            <= stamp < datetime(_y, 11, 1, 6, tzinfo=timezone.utc))
    east = stamp + timedelta(hours=-4 if _dst else -5)
    label = "EDT" if _dst else "EST"
    imgs = []
    for src, caption in images:
        src = Path(src)
        if not src.is_file():
            continue
        name = f"{nid:03d}-{src.stem}{src.suffix}"
        shutil.copy2(src, IMG_DIR / name)
        imgs.append({"src": f"journal/{name}", "caption": caption})
    entry = {"id": nid,
             "ts": stamp.strftime("%Y-%m-%dT%H:%M:%S"),
             "date": east.strftime("%b %d, %I:%M %p ") + label,
             "kind": kind,
             "title": title,
             "score": score,
             "body": body,
             "images": imgs}
    entries.insert(0, entry)
    doc["entries"] = entries[:80]          # keep the file phone-sized
    JSON_PATH.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    if deploy:
        subprocess.run([sys.executable, "deploy.py"],
                       cwd=str(ROOT / "companion-cloud"), check=False)
    return entry


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", required=True)
    ap.add_argument("--body", required=True)
    ap.add_argument("--score", default="")
    ap.add_argument("--image", nargs=2, action="append", default=[],
                    metavar=("PATH", "CAPTION"))
    ap.add_argument("--kind", default="finding",
                    choices=("finding", "update"))
    ap.add_argument("--deploy", action="store_true")
    a = ap.parse_args()
    e = add(a.title, a.body, a.score, a.image, a.deploy, a.kind)
    print(f"journal entry #{e['id']} added ({len(e['images'])} images)")


if __name__ == "__main__":
    main()
