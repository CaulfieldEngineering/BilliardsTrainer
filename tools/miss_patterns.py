"""Miss patterns: tag every miss in the archive and count them.

Joe: "I'd like to start seeing some patterns in my misses." The tags
are labels, not degrees — left/right cut, missed left/right, over/
undercut — and the pattern IS the count table. Numbers ride along for
when a pattern shows up and he wants magnitudes.

    python tools/miss_patterns.py [--session NAME] [--json]
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from billiards_trainer.config import Settings  # noqa: E402
from billiards_trainer.vision.miss_tags import label  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", default="")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    st = Settings.load()
    d = Path(st.recording.resolved_dir())
    vids = ([d / args.session] if args.session
            else sorted(d.glob("session-*.mp4")))
    rows, skipped = [], Counter()
    for v in vids:
        # Read the EXPORTED tags — the same ones the phone shows — so the
        # pattern table can never disagree with what Joe sees on a clip.
        sj = Path(str(v) + ".shots.json")
        if not sj.is_file():
            continue
        try:
            doc = json.loads(sj.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for s in doc.get("shots", []):
            if s.get("outcome") != "miss":
                continue
            if s.get("action", "stroke") not in ("stroke", "break"):
                continue
            tags = s.get("tags")
            if not tags:
                skipped["untagged"] += 1
                continue
            tags = dict(tags)
            if not str(tags.get("confidence", "high")).startswith("high"):
                skipped["low_confidence"] += 1
                continue
            tags["session"] = v.name
            tags["start"] = round(float(s.get("start", 0)), 2)
            rows.append(tags)
        if args.limit and len(rows) >= args.limit:
            break
    cuts = Counter(r["cut"] for r in rows)
    pairs = Counter((r["cut"], r.get("fullness", r["miss_side"]))
                    for r in rows)
    out = {"tagged": len(rows), "skipped": dict(skipped),
           "by_cut": dict(cuts),
           "by_cut_fullness": {f"{k[0]}/{k[1]}": n for k, n in pairs.items()},
           "rows": rows}
    (ROOT / "_eval" / "miss_patterns.json").write_text(json.dumps(out, indent=1))
    if args.json:
        print(json.dumps({k: out[k] for k in
                          ("tagged", "skipped", "by_cut", "by_cut_fullness")},
                         indent=1))
        return 0
    print(f"MISSES TAGGED: {len(rows)}   (skipped: {dict(skipped)})")
    print("\nby shot type:")
    for k, n in cuts.most_common():
        print(f"  {k:9s} {n}")
    print("\nthe pattern table:")
    for cut in ("left", "right", "straight"):
        sub = [r for r in rows if r["cut"] == cut]
        if not sub:
            continue
        if cut == "straight":
            c = Counter(r["miss_side"] for r in sub)
            print(f"  straight-in ({len(sub)}): "
                  f"missed left {c['left']}, missed right {c['right']}")
        else:
            c = Counter(r["fullness"] for r in sub)
            tot = max(1, len(sub))
            print(f"  {cut} cuts ({len(sub)}): overcut {c['overcut']} "
                  f"({100 * c['overcut'] / tot:.0f}%), "
                  f"undercut {c['undercut']} "
                  f"({100 * c['undercut'] / tot:.0f}%)")
    print("\nrecent misses:")
    for r in rows[-12:]:
        sid = r["session"].replace("session-", "").replace(".mp4", "")
        print(f"  {sid}@{r['start']:.0f}  {label(r)}"
              f"  (ball {r['target']} -> {r['pocket']}, "
              f"{r['miss_in']:.1f}in)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
