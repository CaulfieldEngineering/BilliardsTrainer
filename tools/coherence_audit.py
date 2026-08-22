"""System coherence audit — do the features still form ONE system?

Joe: "I'm more concerned about the system as a whole. We're building
vertically or horizontally so focused that by the time we add 20 new
features there's not necessarily contextual relationship or coherence
between the features of system."

That is a different failure from messy code. A codebase can be lint-
clean and still be twenty private worlds: four modules each deciding
where the pockets are, a field written by the exporter that no surface
reads, a capability that exists on the phone but silently not on the
desktop. Those are measurable, so this measures them.

  concepts   one idea, one owner — flags modules re-implementing a
             concept another module already owns (the canonical
             example: pocket geometry, independently computed in
             describe/outcomes/tablespace/miss_tags)
  contract   shots.json is the spine between analysis and both
             surfaces: report fields WRITTEN but never read, and READ
             but never written (fragile consumers)
  parity     Joe's hard rule is that phone and desktop never disagree;
             list capabilities present on one surface only

    python tools/coherence_audit.py [--json]
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "billiards_trainer"
PHONE = ROOT / "companion-cloud" / "public" / "index.html"

#: The architectural intent: each concept has ONE owning module. Anything
#: else that implements it is drift, not necessarily a bug — the audit
#: reports, a human decides.
CONCEPTS = {
    "pocket geometry": {
        "owner": "vision/tablespace.py",
        "signals": [r"top-left", r"bottom-right", r"left-middle",
                    r"pocket_name", r"pockets\("],
    },
    "true-inch scale": {
        "owner": "vision/tablespace.py",
        "signals": [r"2\.25", r"px_per_in", r"BALL_DIAM"],
    },
    "shot geometry (cut/aim angles)": {
        "owner": "vision/miss_tags.py",
        "signals": [r"signed_angle", r"atan2\([^)]*cross", r"cut_deg"],
    },
    "verdict ranking (review vs derived)": {
        "owner": "vision/analysis_cache.py",
        "signals": [r'src.*==.*"review"', r'src="review"', r"_reviewed"],
    },
    "shot-to-record matching": {
        "owner": "vision/analysis_cache.py",
        "signals": [r"_shot_for", r"abs\(float\(s\.get\(.start.*\) - "],
    },
    "cue-stick detection": {
        "owner": "vision/cue_aim.py",
        "signals": [r"detect_cue_aim", r"HoughLinesP"],
    },
}


def _py_modules() -> list:
    return [p for p in SRC.rglob("*.py")
            if "__pycache__" not in p.parts and p.name != "__init__.py"]


def _rel(p: Path) -> str:
    return str(p.relative_to(SRC)).replace("\\", "/")


def check_concepts() -> dict:
    out = {}
    for name, spec in CONCEPTS.items():
        hits = []
        for p in _py_modules():
            rel = _rel(p)
            if rel == spec["owner"]:
                continue
            try:
                src = p.read_text(encoding="utf-8")
            except OSError:
                continue
            n = sum(len(re.findall(sig, src)) for sig in spec["signals"])
            if n >= 2:                    # one mention is a reference; two is logic
                hits.append((rel, n))
        if hits:
            out[name] = {"owner": spec["owner"],
                         "also_implemented_in": sorted(hits, key=lambda kv: -kv[1])}
    return out


def check_contract() -> dict:
    """shots.json fields: produced by the exporter, consumed by surfaces."""
    exp = (SRC / "vision" / "shots_export.py").read_text(encoding="utf-8")
    written = set(re.findall(r'entry\["(\w+)"\]\s*=', exp))
    # keys set in the entry DICT LITERAL count too (they were being missed,
    # which wrongly flagged "action" as read-but-never-written)
    lit = exp.split("entry = {")
    if len(lit) > 1:
        written |= set(re.findall(r'"(\w+)":', lit[1].split("}")[0]))
    written |= set(re.findall(r'"(\w+)":\s', exp.split("doc = {")[-1]))
    written.discard("shots")
    readers = ""
    if PHONE.is_file():
        readers += PHONE.read_text(encoding="utf-8")
    for p in (SRC / "ui").rglob("*.py"):
        readers += p.read_text(encoding="utf-8")
    for p in (SRC / "workers").rglob("*.py"):
        readers += p.read_text(encoding="utf-8")
    unread = sorted(f for f in written
                    if not re.search(rf"[\.\[]\"?{f}\"?[\]\s\.\),]", readers))
    # fields surfaces read out of a shot object that the exporter never writes
    phone_reads = set(re.findall(r"\bs(?:h|hot)?\.(\w+)\b",
                                 PHONE.read_text(encoding="utf-8")
                                 if PHONE.is_file() else ""))
    ghost = sorted(f for f in phone_reads
                   if f not in written and f not in
                   {"length", "name", "start", "end", "map", "filter",
                    "forEach", "push", "slice", "tags", "trails", "aim",
                    "textContent", "style", "id", "className", "clips",
                    "session", "mod", "slowmo", "label", "then", "catch"})
    return {"written": sorted(written), "written_never_read": unread,
            "read_never_written": ghost}


def check_parity() -> dict:
    """Capabilities on one surface only (Joe's rule: they must agree)."""
    phone = PHONE.read_text(encoding="utf-8") if PHONE.is_file() else ""
    desktop = ""
    for sub in ("ui", "workers"):
        for p in (SRC / sub).rglob("*.py"):
            desktop += p.read_text(encoding="utf-8")
    caps = {
        "aim-line overlay": (r"ov-aim|drawAim", r"overlay_aim|_aim\b"),
        "ball-paths overlay": (r"ov-paths|drawTrails", r"overlay_paths|trails"),
        "miss tags": (r"\.tags\b", r"tags"),
        "playlists": (r"playlists", r"playlist"),
        "smooth slo-mo": (r"rife", r"rife"),
        "shot verdicts": (r"api/correct", r"correction|verdict"),
        "drawing tools": (r"drawcv|Draw\.", r"draw_tools|annotation"),
        "frame stepping": (r"fprev|fnext", r"frame_step|step_frame"),
    }
    out = {}
    for name, (pre, dre) in caps.items():
        on_phone = bool(re.search(pre, phone))
        on_desktop = bool(re.search(dre, desktop))
        if on_phone != on_desktop:
            out[name] = "phone only" if on_phone else "desktop only"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    res = {"concepts": check_concepts(), "contract": check_contract(),
           "parity": check_parity()}
    (ROOT / "_eval" / "coherence_audit.json").write_text(json.dumps(res, indent=1))
    if args.json:
        print(json.dumps(res, indent=1))
        return 0
    print("SYSTEM COHERENCE\n")
    print(f"scattered concepts ({len(res['concepts'])}):")
    for name, d in res["concepts"].items():
        others = ", ".join(f"{m} x{n}" for m, n in d["also_implemented_in"])
        print(f"  {name}")
        print(f"     owner: {d['owner']}")
        print(f"     also in: {others}")
    c = res["contract"]
    print(f"\nshots.json contract: {len(c['written'])} fields written")
    print(f"  written but no surface reads: {c['written_never_read'] or 'none'}")
    print(f"  read but never written:       {c['read_never_written'] or 'none'}")
    print(f"\nsurface parity gaps ({len(res['parity'])}):")
    for k, v in res["parity"].items():
        print(f"  {k}: {v}")
    if not res["parity"]:
        print("  none — every capability exists on both surfaces")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
