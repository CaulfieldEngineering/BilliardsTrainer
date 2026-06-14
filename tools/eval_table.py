"""Table-detection eval harness.

Measures how closely the Python felt/rectify port reproduces a labelled
reference capture: corner RMSE (pixels) and homography agreement. Run it after
any change to the table-detection code to catch regressions.

    python tools/eval_table.py [fixture.json] [frame.png]

The fixture JSON is the telemetry sidecar the original C++ app exported
(felt.corners_px + rectification.H). If no frame is given it looks for a
sibling PNG next to the JSON.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from billiards_trainer.config import FeltSettings  # noqa: E402
from billiards_trainer.vision.felt import detect_felt  # noqa: E402
from billiards_trainer.vision.rectify import rectify_tabletop  # noqa: E402

CORNER_ORDER = ("TL", "TR", "BR", "BL")


def load_truth(fixture_json: Path):
    data = json.loads(fixture_json.read_text())
    cp = data["felt"]["corners_px"]
    corners = np.array([[cp[k]["x"], cp[k]["y"]] for k in CORNER_ORDER], dtype=np.float32)
    # NB: the fixture's saved hsv_params were stale (excluded the felt's true hue),
    # so we evaluate the *shipping* defaults — that is what users actually run.
    felt = FeltSettings()
    H = np.array(data["rectification"]["H"], dtype=np.float64)
    return corners, felt, H


def corner_rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def main() -> int:
    fixture = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "tests/fixtures/table_fixture.json"
    if len(sys.argv) > 2:
        frame_path = Path(sys.argv[2])
    else:
        frame_path = ROOT / "tests/fixtures/table_live.png"

    truth_corners, felt, truth_H = load_truth(fixture)
    frame = cv2.imread(str(frame_path))
    if frame is None:
        print(f"!! could not read frame: {frame_path}")
        return 2

    res = detect_felt(frame, felt)
    print(f"Frame: {frame_path.name}  {frame.shape[1]}x{frame.shape[0]}")
    print(f"Felt ok={res.ok}  has_corners={res.has_corners}  area_ratio={res.area_ratio:.3f}")
    if not res.has_corners:
        print("!! no corners detected")
        return 1

    rmse = corner_rmse(res.corners, truth_corners)
    print("\nCorners (detected vs truth):")
    for i, lbl in enumerate(CORNER_ORDER):
        d, t = res.corners[i], truth_corners[i]
        print(f"  {lbl}: ({d[0]:7.1f},{d[1]:7.1f})  truth ({t[0]:7.1f},{t[1]:7.1f})  "
              f"d={np.linalg.norm(d - t):5.1f}px")
    print(f"\nCorner RMSE: {rmse:.2f} px  (diag {np.hypot(*frame.shape[:2]):.0f} px)")

    rect = rectify_tabletop(frame, res.mask, res.corners,
                            pad_px=40, aspect=2.0, refine=True)
    print(f"\nRectify ok={rect.ok}  dst_size={rect.dst_size}")
    if rect.ok:
        # Compare where each truth corner lands under our H vs the reference H.
        pts = truth_corners.reshape(-1, 1, 2).astype(np.float64)
        ours = cv2.perspectiveTransform(pts, rect.H).reshape(-1, 2)
        theirs = cv2.perspectiveTransform(pts, truth_H).reshape(-1, 2)
        hdiff = corner_rmse(ours, theirs)
        print(f"Homography corner-mapping RMSE vs reference H: {hdiff:.2f} px")

    status = "PASS" if rmse < 25 else "WARN"
    print(f"\n[{status}] corner RMSE {rmse:.1f}px (threshold 25px)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
