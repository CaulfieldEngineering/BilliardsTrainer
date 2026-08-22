"""Detection-mode tests: camera-preview (auto-detection OFF), detection resume,
ball-colour rendering, and the ONNX YOLO decode math.

Guards the "no fake detections" contract: with detection off the pipeline renders
a clean empty table and emits zero detections; with detection on it tracks the
cue. Synthetic-frame tests pin the cue-ball heuristic (the trained model can't
detect the synthetic demo scene — that's validated on real footage by the eval
harness, tools/eval_cue_ball.py).
"""

import numpy as np

from billiards_trainer.capture.camera import DemoSource
from billiards_trainer.config import Settings
from billiards_trainer.core.types import BallClass, Track
from billiards_trainer.vision.pipeline import Pipeline


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
        assert res.frame_bgr is not None


def test_toggling_detection_on_resumes_detection():
    settings = Settings()
    settings.balls.live_strategy = "cue_ball_white"  # heuristic works on synthetic
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
    assert seen >= 1, "detection should resume once re-enabled (the white cue)"


# ---- ONNX YOLO decode (the model path) ----------------------------------- #
class _FakeOnnxSession:
    def __init__(self, output):
        self._out = output

    def run(self, _names, _feeds):
        return [self._out]

    def get_providers(self):
        return ["CPUExecutionProvider"]


def test_onnx_decode_maps_boxes_to_rect_coords():
    """Validate the ONNX YOLO decode (letterbox undo, xywh->centre, single-class
    keep, NMS) with a mocked session — no onnxruntime/model file needed."""
    from billiards_trainer.detector_strategies.onnx_model import OnnxModelStrategy

    # single-class pool model layout: [1, 4+1, 8400] = 4 box coords + 1 ball score
    out = np.zeros((1, 5, 8400), np.float32)
    out[0, 0, 100] = 320.0   # cx (letterbox space)
    out[0, 1, 100] = 320.0   # cy
    out[0, 2, 100] = 40.0    # w
    out[0, 3, 100] = 40.0    # h
    out[0, 4, 100] = 0.9     # ball score

    det = OnnxModelStrategy.__new__(OnnxModelStrategy)
    det._sess = _FakeOnnxSession(out)

    class _Inp:
        name = "images"
        shape = [1, 3, 640, 640]
    det._inp = _Inp()
    det._size = 640
    det._conf = 0.25
    det._iou = 0.45

    frame = np.full((640, 640, 3), (70, 120, 60), np.uint8)  # square => ratio 1.0
    # rescan=False forces the single full-frame pass — this test validates the
    # letterbox-undo decode math, which is identical for the tiled passes.
    dets = det.detect(frame, None, rescan=False)
    assert dets
    d = min(dets, key=lambda b: (b.x - 320) ** 2 + (b.y - 320) ** 2)
    assert abs(d.x - 320) < 3 and abs(d.y - 320) < 3
    assert abs(d.radius - 20) < 4
    assert d.score > 0.8
