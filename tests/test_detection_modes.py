"""v0.2.14 graceful-degradation tests: camera-preview mode (auto-detection OFF),
pocket-region masking, the strict ball-size band, and YOLO weight discovery.

These guard the "hold the line on reliability — no fake detections" contract:
with detection off the pipeline must render a clean empty table and emit zero
detections; with detection on, blobs in pockets / of the wrong size are dropped.
"""

import numpy as np

from billiards_trainer.capture.camera import DemoSource
from billiards_trainer.config import Settings
from billiards_trainer.vision.balls import (
    ClassicalBallDetector,
    find_yolo_weights,
    yolo_weights_available,
)
from billiards_trainer.vision.geometry import TableModel
from billiards_trainer.vision.pipeline import Pipeline


# ---- preview mode (auto-detection OFF) ----------------------------------- #
def test_preview_mode_emits_no_detections():
    """With detection off, a moving scene must still produce a clean empty
    overhead and zero balls/shots — never a phantom."""
    settings = Settings()
    src = DemoSource()
    pipe = Pipeline(settings)
    pipe.detect_enabled = False

    for i in range(120):  # well past the scripted strike — real motion happens
        res = pipe.process(src.read(), t=i * 0.033)
        assert res.status == "preview"
        assert res.n_balls == 0
        assert res.shot_event is None
        assert not res.tracks
        assert res.rect_bgr is not None  # clean schematic still rendered
        # live view is the untouched camera frame
        assert res.frame_bgr is not None


def test_toggling_detection_on_resumes_detection():
    settings = Settings()
    settings.detection.warmup_seconds = 0.0
    src = DemoSource()
    pipe = Pipeline(settings)

    pipe.detect_enabled = False
    res = pipe.process(src.read(), t=0.0)
    assert res.n_balls == 0

    pipe.detect_enabled = True
    seen = 0
    for i in range(30):
        res = pipe.process(src.read(), t=1.0 + i * 0.033)
        seen = max(seen, res.n_balls)
    assert seen >= 1, "detection should resume once re-enabled"


# ---- pocket masking + size band ------------------------------------------ #
def _felt_rect(w: int, h: int) -> np.ndarray:
    img = np.empty((h, w, 3), np.uint8)
    img[:] = (70, 120, 60)  # BGR green felt
    return img


def test_pocket_blobs_rejected_but_real_ball_kept():
    table = TableModel.from_rect((400, 800), 40)  # short_side 320
    rect = _felt_rect(400, 800)
    # a real, regulation-sized ball mid-table -> must be detected
    r = int(0.024 * table.short_side)  # ~regulation radius, inside the band
    cx, cy = 200, 400
    import cv2
    cv2.circle(rect, (cx, cy), r, (245, 245, 245), -1)
    # a ball-shaped blob sitting in the bottom-right pocket -> must be dropped
    pocket = next(p for p in table.pockets if p.name == "bottom-right")
    cv2.circle(rect, (int(pocket.x), int(pocket.y)), r, (245, 245, 245), -1)

    det = ClassicalBallDetector(Settings().balls, Settings().felt)
    dets = det.detect(rect, None, table)

    assert any(abs(d.x - cx) < 12 and abs(d.y - cy) < 12 for d in dets), \
        "the real mid-table ball should be detected"
    for d in dets:
        _, dist = table.nearest_pocket(d.x, d.y)
        assert dist >= table.pocket_radius, "no detection may sit inside a pocket"


def test_oversized_blob_rejected_by_size_band():
    table = TableModel.from_rect((400, 800), 40)
    rect = _felt_rect(400, 800)
    import cv2
    # a blob far larger than any pool ball (shadow / artefact)
    cv2.circle(rect, (200, 400), int(0.10 * table.short_side), (245, 245, 245), -1)
    det = ClassicalBallDetector(Settings().balls, Settings().felt)
    dets = det.detect(rect, None, table)
    # nothing the size of a beach ball should survive the regulation size band
    assert all(d.radius <= 0.034 * table.short_side * 1.1 for d in dets)


# ---- YOLO weight discovery (gates the auto-detection toggle) -------------- #
def test_yolo_weights_available(tmp_path):
    assert not yolo_weights_available(tmp_path)
    assert find_yolo_weights(tmp_path) is None
    (tmp_path / "pool_balls.pt").write_bytes(b"not-a-real-model")
    assert yolo_weights_available(tmp_path)
    assert find_yolo_weights(tmp_path).name == "pool_balls.pt"
