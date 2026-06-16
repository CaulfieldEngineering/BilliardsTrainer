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
from billiards_trainer.vision.types import BallClass, Track


# ---- ball colour rendering (the "why were balls yellow/black" fix) -------- #
def test_ball_color_uses_measured_not_class_palette():
    from billiards_trainer.vision.overlay import ball_color

    blue = Track(id=1, x=0, y=0, radius=8, cls=BallClass.SOLID, bgr=(200, 40, 30))
    color, uncertain = ball_color(blue, measured=True)
    assert color == (200, 40, 30) and not uncertain  # its REAL colour, not yellow

    cue = Track(id=2, x=0, y=0, radius=8, cls=BallClass.CUE, bgr=(0, 0, 0))
    assert ball_color(cue, measured=True)[0] == (255, 255, 255)  # cue always white

    unknown = Track(id=3, x=0, y=0, radius=8, cls=BallClass.UNKNOWN, bgr=(10, 200, 250))
    color, uncertain = ball_color(unknown, measured=True)
    assert uncertain and color == (150, 150, 150)  # grey "?", never a fake colour

    # legacy palette still available behind the flag
    legacy, _ = ball_color(blue, measured=False)
    assert legacy == (60, 200, 255)  # the old fixed SOLID yellow


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


def test_onnx_weights_preferred_over_pt(tmp_path):
    (tmp_path / "pool_balls.pt").write_bytes(b"x")
    (tmp_path / "pool_balls.onnx").write_bytes(b"x")
    assert find_yolo_weights(tmp_path).suffix == ".onnx"


def test_make_detector_degrades_to_classical(tmp_path):
    """No model -> classical; a corrupt model must degrade, never crash."""
    from billiards_trainer.vision.balls import ClassicalBallDetector, make_detector
    s = Settings()  # backend 'auto'
    assert isinstance(make_detector(s.balls, s.felt, models_dir=tmp_path),
                      ClassicalBallDetector)
    (tmp_path / "pool_balls.onnx").write_bytes(b"not-a-real-onnx")
    assert isinstance(make_detector(s.balls, s.felt, models_dir=tmp_path),
                      ClassicalBallDetector)


class _FakeOnnxSession:
    def __init__(self, output):
        self._out = output

    def run(self, _names, _feeds):
        return [self._out]


def test_onnx_decode_maps_boxes_to_rect_coords():
    """Validate the ONNX YOLO decode math (letterbox undo, xywh->centre, class-32
    COCO filter, NMS) with a mocked session — no onnxruntime/model file needed."""
    from billiards_trainer.vision.balls import OnnxYoloDetector
    from billiards_trainer.vision.geometry import TableModel

    out = np.zeros((1, 84, 8400), np.float32)   # COCO layout: 4 box + 80 classes
    out[0, 0, 100] = 320.0   # cx (letterbox space)
    out[0, 1, 100] = 320.0   # cy
    out[0, 2, 100] = 40.0    # w
    out[0, 3, 100] = 40.0    # h
    out[0, 4 + 32, 100] = 0.9  # "sports ball" score

    det = OnnxYoloDetector.__new__(OnnxYoloDetector)
    det._sess = _FakeOnnxSession(out)
    det._input_name = "images"
    det._size = 640
    det._conf = 0.25
    det._iou = 0.45
    s = Settings()
    det.balls, det.felt = s.balls, s.felt

    rect = np.full((640, 640, 3), (70, 120, 60), np.uint8)  # square => ratio 1.0
    table = TableModel.from_rect((640, 640), 40)
    dets = det.detect(rect, None, table)
    assert len(dets) == 1
    d = dets[0]
    assert abs(d.x - 320) < 3 and abs(d.y - 320) < 3
    assert abs(d.radius - 20) < 4
    assert d.score > 0.8
