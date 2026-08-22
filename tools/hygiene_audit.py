"""Code hygiene audit — does the codebase stay clean as features land?

Joe: "We're adding so many features that we need to make sure, once the
code is working, that it stays clean system-wide and everything's not
turning into spaghetti."

The design principle is DRIFT, not absolutes. A 1200-line pipeline is
not a bug; a pipeline that gained 300 lines this week without a test is
a smell. So the audit records a baseline in _eval/hygiene_baseline.json
and reports what got WORSE — which is the signal a watchdog can act on
without crying wolf about pre-existing size.

Checks:
  lint        ruff over src/tools/tests (the repo's own config)
  bigfiles    files past a size band, and their growth since baseline
  bigfuncs    functions past a length band (spaghetti's usual shape)
  dupes       repeated 6+ line blocks across the tree
  debt        TODO/FIXME/HACK/XXX markers
  untested    src modules no test imports, weighted by size
  deadtools   tools/ scripts nothing references and nobody ran

    python tools/hygiene_audit.py [--json] [--baseline]
"""

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "_eval" / "hygiene_baseline.json"

BIG_FILE = 900          # lines: beyond this a module is doing too much
BIG_FUNC = 90           # lines: beyond this a function hides its own logic
DUPE_LINES = 6          # identical normalized runs this long are duplication


def _py_files() -> list:
    out = []
    for sub in ("src", "tools", "tests"):
        out += [p for p in (ROOT / sub).rglob("*.py")
                if "__pycache__" not in p.parts]
    return sorted(out)


def _rel(p: Path) -> str:
    return str(p.relative_to(ROOT)).replace("\\", "/")


def check_lint() -> dict:
    try:
        r = subprocess.run([sys.executable, "-m", "ruff", "check",
                            "src", "tools", "tests", "--output-format=json"],
                           cwd=ROOT, capture_output=True, text=True,
                           timeout=300)
        items = json.loads(r.stdout or "[]")
    except (subprocess.SubprocessError, ValueError, OSError):
        return {"n": -1, "by_rule": {}}
    return {"n": len(items),
            "by_rule": dict(Counter(i.get("code") or "?"
                                    for i in items).most_common(8))}


def check_sizes() -> dict:
    big, funcs = {}, {}
    for p in _py_files():
        try:
            src = p.read_text(encoding="utf-8")
        except OSError:
            continue
        n = src.count("\n") + 1
        if n >= BIG_FILE:
            big[_rel(p)] = n
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                ln = (getattr(node, "end_lineno", node.lineno) - node.lineno)
                if ln >= BIG_FUNC:
                    funcs[f"{_rel(p)}::{node.name}"] = ln
    # the phone app is one file by necessity (a PWA) — track it too
    html = ROOT / "companion-cloud" / "public" / "index.html"
    if html.is_file():
        big[_rel(html)] = html.read_text(encoding="utf-8").count("\n") + 1
    return {"files": big, "funcs": funcs}


def check_dupes() -> dict:
    seen = defaultdict(list)
    for p in _py_files():
        try:
            lines = [ln.strip() for ln in
                     p.read_text(encoding="utf-8").splitlines()]
        except OSError:
            continue
        body = [ln for ln in lines
                if ln and not ln.startswith("#") and len(ln) > 12]
        for i in range(len(body) - DUPE_LINES):
            blk = "\n".join(body[i:i + DUPE_LINES])
            seen[hashlib.md5(blk.encode()).hexdigest()].append(_rel(p))
    hits = {h: sorted(set(v)) for h, v in seen.items()
            if len(v) > 1 and len(set(v)) > 1}
    return {"n": len(hits),
            "pairs": sorted({" + ".join(v) for v in hits.values()})[:8]}


def check_debt() -> dict:
    pat = re.compile(r"\b(TODO|FIXME|HACK|XXX)\b")
    by = Counter()
    for p in _py_files():
        try:
            for m in pat.finditer(p.read_text(encoding="utf-8")):
                by[m.group(1)] += 1
        except OSError:
            continue
    return {"n": sum(by.values()), "by_kind": dict(by)}


def check_untested() -> dict:
    tested = set()
    for p in (ROOT / "tests").rglob("*.py"):
        try:
            src = p.read_text(encoding="utf-8")
        except OSError:
            continue
        for m in re.finditer(r"billiards_trainer\.([\w.]+)", src):
            tested.add(m.group(1).split(".")[-1])
        for m in re.finditer(r"^from (\w+) import|^import (\w+)", src, re.M):
            tested.add((m.group(1) or m.group(2) or "").strip())
    out = {}
    for p in (ROOT / "src").rglob("*.py"):
        if "__pycache__" in p.parts or p.name.startswith("__"):
            continue
        if p.stem in tested:
            continue
        n = p.read_text(encoding="utf-8").count("\n") + 1
        if n >= 120:                     # ignore trivial modules
            out[_rel(p)] = n
    return {"modules": dict(sorted(out.items(), key=lambda kv: -kv[1])[:12]),
            "n": len(out)}


def check_deadtools() -> dict:
    refs = ""
    for sub in ("src", "tools", "tests", "docs"):
        for p in (ROOT / sub).rglob("*"):
            if p.is_file() and p.suffix in (".py", ".md", ".cmd", ".txt"):
                try:
                    refs += p.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    pass
    dead = []
    for p in sorted((ROOT / "tools").glob("*.py")):
        if refs.count(p.stem) <= 1:      # only its own definition
            dead.append(_rel(p))
    return {"n": len(dead), "tools": dead[:12]}


def run() -> dict:
    return {"lint": check_lint(), "sizes": check_sizes(),
            "dupes": check_dupes(), "debt": check_debt(),
            "untested": check_untested(), "deadtools": check_deadtools()}


def compare(cur: dict, base: dict) -> list:
    """What got WORSE since the baseline — the watchdog's signal."""
    out = []
    if base is None:
        return out
    c, b = cur["lint"]["n"], base.get("lint", {}).get("n", 0)
    if c > b >= 0:
        out.append(f"lint findings {b} -> {c}")
    for key, label in (("files", "file"), ("funcs", "function")):
        cb, bb = cur["sizes"][key], base.get("sizes", {}).get(key, {})
        for name, n in cb.items():
            old = bb.get(name)
            if old is None:
                out.append(f"new oversized {label}: {name} ({n} lines)")
            elif n > old + 40:
                out.append(f"{label} grew: {name} {old} -> {n} lines")
    if cur["debt"]["n"] > base.get("debt", {}).get("n", 0):
        out.append(f"debt markers {base.get('debt', {}).get('n', 0)} -> "
                   f"{cur['debt']['n']}")
    if cur["dupes"]["n"] > base.get("dupes", {}).get("n", 0) + 2:
        out.append(f"duplicated blocks {base.get('dupes', {}).get('n', 0)}"
                   f" -> {cur['dupes']['n']}")
    if cur["untested"]["n"] > base.get("untested", {}).get("n", 0):
        out.append(f"untested modules {base.get('untested', {}).get('n', 0)}"
                   f" -> {cur['untested']['n']}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--baseline", action="store_true",
                    help="record the current state as the baseline")
    args = ap.parse_args()
    cur = run()
    base = None
    if BASELINE.is_file():
        try:
            base = json.loads(BASELINE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            base = None
    worse = compare(cur, base)
    if args.baseline:
        BASELINE.write_text(json.dumps(cur, indent=1))
        print(f"baseline recorded: {BASELINE}")
        return 0
    if args.json:
        print(json.dumps({"worse": worse, **cur}, indent=1))
        return 0
    print("CODE HYGIENE")
    print(f"  lint       {cur['lint']['n']} findings {cur['lint']['by_rule']}")
    print(f"  big files  {len(cur['sizes']['files'])} over {BIG_FILE} lines")
    for k, v in sorted(cur["sizes"]["files"].items(), key=lambda kv: -kv[1])[:5]:
        print(f"               {v:5d}  {k}")
    print(f"  big funcs  {len(cur['sizes']['funcs'])} over {BIG_FUNC} lines")
    for k, v in sorted(cur["sizes"]["funcs"].items(), key=lambda kv: -kv[1])[:5]:
        print(f"               {v:5d}  {k}")
    print(f"  dupes      {cur['dupes']['n']} repeated {DUPE_LINES}-line blocks")
    print(f"  debt       {cur['debt']['n']} {cur['debt']['by_kind']}")
    print(f"  untested   {cur['untested']['n']} modules >=120 lines")
    for k, v in list(cur["untested"]["modules"].items())[:5]:
        print(f"               {v:5d}  {k}")
    print(f"  dead tools {cur['deadtools']['n']} {cur['deadtools']['tools'][:4]}")
    print("\nDRIFT since baseline:" if base else "\n(no baseline yet)")
    for w in worse:
        print(f"  WORSE  {w}")
    if base and not worse:
        print("  nothing got worse")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
