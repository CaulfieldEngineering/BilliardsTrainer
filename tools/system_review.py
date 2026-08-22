"""Top-down system review — block diagram first, syntax last.

Joe: "the review should start at the FULL system level and work down to
granularity. Viewing everything from block diagram level and working
code to code/syntax. The idea is to perpetually maintain this entire
software as a whole cohesive project and ensure all new features
maintain a solid foundation."

So the review runs in four levels, reported in that order, because a
finding at L1 makes L4 findings irrelevant — no amount of tidy syntax
saves a subsystem that reaches upward into the UI.

  L1 ARCHITECTURE  the block diagram, derived from real imports:
                   subsystems, their edges, layering violations, cycles
  L2 DATA SPINE    the artifacts features share (sidecar, shots.json,
                   playlists) — who writes, who reads, what is orphaned
  L3 FEATURES      capability parity across surfaces, concept ownership
  L4 CODE          lint, size, duplication, debt (tools/hygiene_audit)

    python tools/system_review.py [--json] [--level 1|2|3|4]
"""

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "billiards_trainer"
PHONE = ROOT / "companion-cloud" / "public" / "index.html"
API = ROOT / "companion-cloud" / "api"

#: INTENDED ARCHITECTURE — the layering new features must respect.
#: Lower numbers are foundations; a subsystem may import its own layer
#: and anything BELOW it, never above. This is the contract the block
#: diagram is checked against.
LAYERS = {
    "config": 0, "db": 0, "version": 0,
    "capture": 1, "vision": 1, "events": 1, "detector_strategies": 1,
    "pose": 1,
    "cue": 2, "game": 2, "train": 2, "eval": 2, "sync": 2, "update": 2,
    "companion": 3, "workers": 3,
    "ui": 4, "debug_upload": 4,
    "app": 5, "__main__": 5, "__init__": 5,
}

#: Artifacts that carry meaning BETWEEN features — the system's spine.
SPINE = {
    "analysis sidecar (.analysis.jsonl)": [r"analysis\.jsonl", r"SidecarWriter",
                                           r"SidecarReader"],
    "shot summary (.shots.json)": [r"shots\.json", r"shots_export",
                                   r"export_shots_summary"],
    "library index (library.json)": [r"library\.json", r"export_library_index"],
    "playlists (playlists.json)": [r"playlists\.json", r"/api/playlists"],
    "corrections queue": [r"corrections/", r"corrections_watcher",
                          r"/api/correct"],
}


def _subsystem(p: Path) -> str:
    rel = p.relative_to(SRC).parts
    return rel[0] if len(rel) > 1 else p.stem


def level1() -> dict:
    """The block diagram, from imports."""
    edges = defaultdict(set)
    files = defaultdict(int)
    for p in SRC.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        sub = _subsystem(p)
        files[sub] += 1
        try:
            src = p.read_text(encoding="utf-8")
        except OSError:
            continue
        for m in re.finditer(r"^\s*from\s+(\.+)(\w+)?", src, re.M):
            dots, mod = m.group(1), m.group(2) or ""
            if len(dots) >= 2 and mod in LAYERS:
                if mod != sub:
                    edges[sub].add(mod)
        for m in re.finditer(r"billiards_trainer\.(\w+)", src):
            if m.group(1) in LAYERS and m.group(1) != sub:
                edges[sub].add(m.group(1))
    violations, cycles = [], []
    for a, outs in edges.items():
        for b in outs:
            if LAYERS.get(a, 9) < LAYERS.get(b, 9):
                violations.append(f"{a}(L{LAYERS.get(a)}) imports "
                                  f"{b}(L{LAYERS.get(b)}) — reaches upward")
            if a in edges.get(b, set()) and a < b:
                cycles.append(f"{a} <-> {b}")
    return {"subsystems": {k: files[k] for k in sorted(files)},
            "edges": {k: sorted(v) for k, v in sorted(edges.items())},
            "violations": sorted(violations), "cycles": sorted(cycles)}


def level2() -> dict:
    """The data spine: who writes and reads each shared artifact."""
    scopes = {}
    for p in SRC.rglob("*.py"):
        if "__pycache__" not in p.parts:
            scopes[f"src/{_subsystem(p)}"] = scopes.get(f"src/{_subsystem(p)}", "") \
                + p.read_text(encoding="utf-8", errors="ignore")
    if PHONE.is_file():
        scopes["phone"] = PHONE.read_text(encoding="utf-8", errors="ignore")
    if API.is_dir():
        scopes["cloud api"] = "".join(
            f.read_text(encoding="utf-8", errors="ignore")
            for f in API.glob("*.js"))
    out = {}
    for name, sigs in SPINE.items():
        touch = sorted(k for k, blob in scopes.items()
                       if any(re.search(s, blob) for s in sigs))
        out[name] = touch
    orphans = [k for k, v in out.items() if len(v) < 2]
    return {"artifacts": out, "single_touch": orphans}


def level3() -> dict:
    try:
        r = subprocess.run([sys.executable, "tools/coherence_audit.py", "--json"],
                           cwd=ROOT, capture_output=True, text=True, timeout=300)
        return json.loads(r.stdout or "{}")
    except (subprocess.SubprocessError, ValueError, OSError) as e:
        return {"error": str(e)}


def level4() -> dict:
    try:
        r = subprocess.run([sys.executable, "tools/hygiene_audit.py", "--json"],
                           cwd=ROOT, capture_output=True, text=True, timeout=600)
        return json.loads(r.stdout or "{}")
    except (subprocess.SubprocessError, ValueError, OSError) as e:
        return {"error": str(e)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--level", type=int, default=0)
    args = ap.parse_args()
    res = {}
    want = (lambda n: args.level in (0, n))
    if want(1):
        res["L1_architecture"] = level1()
    if want(2):
        res["L2_data_spine"] = level2()
    if want(3):
        res["L3_features"] = level3()
    if want(4):
        res["L4_code"] = level4()
    (ROOT / "_eval" / "system_review.json").write_text(json.dumps(res, indent=1))
    if args.json:
        print(json.dumps(res, indent=1))
        return 0

    if "L1_architecture" in res:
        a = res["L1_architecture"]
        print("L1 ARCHITECTURE — the block diagram\n")
        for sub in sorted(a["subsystems"], key=lambda k: LAYERS.get(k, 9)):
            deps = a["edges"].get(sub, [])
            print(f"  L{LAYERS.get(sub, '?')} {sub:12s} "
                  f"({a['subsystems'][sub]:2d} files)"
                  + (f"  ->  {', '.join(deps)}" if deps else "  ->  (foundation)"))
        print(f"\n  layering violations: {len(a['violations'])}")
        for v in a["violations"]:
            print(f"     {v}")
        print(f"  cycles: {a['cycles'] or 'none'}")
    if "L2_data_spine" in res:
        b = res["L2_data_spine"]
        print("\nL2 DATA SPINE — what features share\n")
        for name, who in b["artifacts"].items():
            print(f"  {name}")
            print(f"     touched by: {', '.join(who) if who else 'NOBODY'}")
        if b["single_touch"]:
            print(f"  single-touch artifacts (not shared): {b['single_touch']}")
    if "L3_features" in res:
        c = res["L3_features"]
        print("\nL3 FEATURES — parity and concept ownership\n")
        par = c.get("parity", {})
        print(f"  surface-parity gaps: {len(par)}")
        for k, v in par.items():
            print(f"     {k}: {v}")
        print(f"  scattered concepts: {len(c.get('concepts', {}))}")
        for k, d in c.get("concepts", {}).items():
            print(f"     {k} (owner {d['owner']}) also in "
                  f"{len(d['also_implemented_in'])} modules")
    if "L4_code" in res:
        d = res["L4_code"]
        print("\nL4 CODE — syntax level\n")
        print(f"  lint {d.get('lint', {}).get('n')} findings; "
              f"{len(d.get('sizes', {}).get('files', {}))} oversized files; "
              f"{len(d.get('sizes', {}).get('funcs', {}))} oversized functions; "
              f"{d.get('dupes', {}).get('n')} duplicate blocks")
        worse = d.get("worse") or []
        print(f"  drift since baseline: {worse or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
