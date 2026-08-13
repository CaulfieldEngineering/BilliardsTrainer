"""Champion vs challenger: score a candidate identifier against the frozen corpus.

The promotion contract (docs/VISION_ROADMAP.md): a challenger model ships ONLY
if it beats the champion on the corpus — impossible-rate must not rise,
coverage must not fall, and the identifier's target metric (out-of-game number
claims) should drop. Champions are files; rollback is a file copy.

    python tools/score_challenger.py --model _train/challenger/pool_ballid_c2.onnx
    python tools/score_challenger.py --model ... --champion-agg _eval/corpus/post_overlap_fix/aggregate.json

Mechanics: the ONNX strategies load whatever sits in the user models dir, and
score_corpus runs each session in a fresh subprocess, so the challenger is
evaluated by swapping the file in (with a .bak) and restoring it in a finally
block. If this script dies mid-run, restore by renaming the .bak back.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from billiards_trainer.config import MODELS_DIR  # noqa: E402


def find_champion() -> Path | None:
    """The deployed identifier: the 'ballid' onnx that is not the finder."""
    for p in sorted(MODELS_DIR.glob("*.onnx")):
        if "ballid" in p.stem and "yolo11" not in p.stem:
            return p
    return None


def rate_of(agg: dict, key: str) -> float:
    bf = agg.get("total_ball_frames") or 0
    if not bf:
        return float("inf")
    total = sum(d.get(key) or 0 for d in agg.get("sessions_detail", []))
    return 1000.0 * total / bf


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="challenger .onnx")
    ap.add_argument("--champion-agg", default="",
                    help="champion aggregate.json (default: newest _eval/corpus/*/aggregate.json)")
    ap.add_argument("--out", default="", help="output corpus dir for the challenger run")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--max-seconds", type=float, default=300.0)
    args = ap.parse_args()

    challenger = Path(args.model)
    if not challenger.is_file():
        print(f"no challenger at {challenger}", file=sys.stderr)
        return 2
    champion = find_champion()
    if champion is None:
        print(f"no champion identifier in {MODELS_DIR}", file=sys.stderr)
        return 2

    if args.champion_agg:
        champ_agg_path = Path(args.champion_agg)
    else:
        cands = sorted(ROOT.glob("_eval/corpus/*/aggregate.json"),
                       key=lambda p: p.stat().st_mtime)
        if not cands:
            print("no champion aggregate — run tools/score_corpus.py first", file=sys.stderr)
            return 2
        champ_agg_path = cands[-1]
    champ_agg = json.loads(champ_agg_path.read_text(encoding="utf-8"))

    out_dir = Path(args.out) if args.out else ROOT / "_eval" / "corpus" / f"challenger_{challenger.stem}"
    backup = champion.with_suffix(".onnx.bak")

    print(f"champion:   {champion.name}  (baseline: {champ_agg_path})")
    print(f"challenger: {challenger}")
    print(f"swap in -> score corpus -> restore. Output: {out_dir}")

    shutil.copy2(champion, backup)
    try:
        shutil.copy2(challenger, champion)
        cmd = [sys.executable, str(ROOT / "tools" / "score_corpus.py"),
               "--out", str(out_dir), "--stride", str(args.stride),
               "--max-seconds", str(args.max_seconds)]
        subprocess.run(cmd, check=False, timeout=4 * 3600)
    finally:
        shutil.copy2(backup, champion)
        backup.unlink(missing_ok=True)
        print(f"champion restored: {champion.name}")

    chall_agg_path = out_dir / "aggregate.json"
    if not chall_agg_path.exists():
        print("challenger run produced no aggregate — treat as FAILED", file=sys.stderr)
        return 1
    chall_agg = json.loads(chall_agg_path.read_text(encoding="utf-8"))

    # ---- the gates ------------------------------------------------------- #
    champ_imp = champ_agg.get("rate_impossible") or float("inf")
    chall_imp = chall_agg.get("rate_impossible") or float("inf")
    champ_cov = _mean_coverage(champ_agg)
    chall_cov = _mean_coverage(chall_agg)
    champ_oog = rate_of(champ_agg, "out_of_game")     # absent in older runs -> 0
    chall_oog = rate_of(chall_agg, "out_of_game")

    print("\n=== verdict ===")
    print(f"impossible/1k : champion {champ_imp:7.2f}  challenger {chall_imp:7.2f}")
    print(f"mean coverage : champion {champ_cov:7.2f}  challenger {chall_cov:7.2f}")
    if champ_oog or chall_oog:
        print(f"out-of-game/1k: champion {champ_oog:7.2f}  challenger {chall_oog:7.2f}")

    gates = {
        "impossible must not rise >10%": chall_imp <= champ_imp * 1.10 + 0.05,
        "coverage must not fall >5%": chall_cov >= champ_cov * 0.95,
        "no degenerate sessions": not chall_agg.get("degenerate"),
        "no failed sessions beyond champion": (chall_agg.get("failed") or 0)
                                              <= (champ_agg.get("failed") or 0),
    }
    for name, ok in gates.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if all(gates.values()):
        print("\nPROMOTABLE. To promote:")
        print(f"  copy \"{challenger}\" \"{champion}\"")
        print("(keep the old champion — rollback is copying it back)")
        return 0
    print("\nREJECTED — champion stays.")
    return 1


def _mean_coverage(agg: dict) -> float:
    rows = [d for d in agg.get("sessions_detail", [])
            if not d.get("failed") and d.get("coverage") is not None]
    return sum(d["coverage"] for d in rows) / len(rows) if rows else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
