"""Compare an eval run's metrics.json against the committed CI baseline and emit
a markdown summary (to stdout + $GITHUB_STEP_SUMMARY). Flags detector-behaviour
regressions on the fixed synthetic CI clip.

This guards against *accidental* detector changes — it runs on the committed
synthetic `demo_clip.mp4`, not Joe's real footage (which is gitignored). On a PR
that intentionally changes the detector, update the baseline with --update.

    python tools/eval_ci_check.py --metrics <run>/metrics.json
    python tools/eval_ci_check.py --metrics <run>/metrics.json --update   # reseed baseline
    python tools/eval_ci_check.py --metrics <run>/metrics.json --strict   # fail on regression
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:  # Windows consoles default to cp1252 — make our markdown output UTF-8 safe
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "tests" / "fixtures" / "eval" / "ci_baseline.json"

# fractional tolerance on behaviour metrics before we call it a regression
CALIB_DROP_TOL = 0.10     # calibration rate may drop at most 10 percentage-equiv
DET_FRAC_TOL = 0.40       # total detections may move +/-40% (detector is noisy/changing)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--update", action="store_true", help="overwrite the committed baseline")
    ap.add_argument("--strict", action="store_true", help="exit non-zero on regression")
    args = ap.parse_args()

    run = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    agg = run.get("aggregate", {})
    cur = {"calib_rate": agg.get("calib_rate", 0.0),
           "total_dets": agg.get("total_dets", 0),
           "shots": agg.get("shots", 0),
           "frames": agg.get("frames", 0)}

    if args.update or not BASELINE.exists():
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps(cur, indent=2), encoding="utf-8")
        _emit(f"### Eval CI\n\nBaseline {'updated' if args.update else 'seeded'} → "
              f"`{_rel(BASELINE)}`\n\n```json\n{json.dumps(cur, indent=2)}\n```")
        return 0

    base = json.loads(BASELINE.read_text(encoding="utf-8"))
    issues = []
    if cur["calib_rate"] < base["calib_rate"] - CALIB_DROP_TOL:
        issues.append(f"calibration rate dropped: {base['calib_rate']:.2f} → {cur['calib_rate']:.2f}")
    b_det = max(1, base["total_dets"])
    if abs(cur["total_dets"] - base["total_dets"]) / b_det > DET_FRAC_TOL:
        issues.append(f"detection count moved >{int(DET_FRAC_TOL*100)}%: "
                      f"{base['total_dets']} → {cur['total_dets']}")

    rows = "\n".join(
        f"| {k} | {base.get(k)} | {cur.get(k)} |" for k in ("calib_rate", "total_dets", "shots", "frames"))
    status = "⚠️ REGRESSION" if issues else "✅ no regression"
    body = (f"### Eval CI — {status}\n\n"
            f"Synthetic clip `demo_clip.mp4` vs committed baseline.\n\n"
            f"| metric | baseline | this run |\n|---|---|---|\n{rows}\n")
    if issues:
        body += "\n**Flagged:**\n" + "\n".join(f"- {i}" for i in issues)
        body += ("\n\n_If this change to the detector is intentional, reseed with "
                 "`python tools/eval_ci_check.py --metrics <run>/metrics.json --update`._")
    _emit(body)
    if issues and args.strict:
        return 1
    return 0


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return p.name


def _emit(md: str):
    print(md)
    sp = os.environ.get("GITHUB_STEP_SUMMARY")
    if sp:
        with open(sp, "a", encoding="utf-8") as f:
            f.write(md + "\n")


if __name__ == "__main__":
    sys.exit(main())
