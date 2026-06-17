"""ONNX YOLO detector strategies — RAW frame, no torch (onnxruntime).

Auto-registers one strategy per ``.onnx`` file found in the eval models dir
(``_eval/models/``), so dropping a new model there adds a variant with zero code.
Decodes YOLOv8-style output ([1, 4+nc, 8400]) with numpy + OpenCV NMS. If the model
has 80 classes it's treated as COCO (keep class 32 "sports ball"); otherwise every
class is treated as a ball and coloured via the classical classifier.

onnxruntime is imported lazily — if it's missing the strategy raises at detect()
time and the harness self-heals (marks the variant FAILED), never blocking others.
"""

import logging
from pathlib import Path

import cv2
import numpy as np

from ..config import MODELS_DIR as USER_MODELS_DIR
from ..vision.types import BallClass, Detection
from . import DetectorStrategy, ball_radius_raw, classify_crop, table_polygon_mask

log = logging.getLogger("detector.onnx")

ROOT = Path(__file__).resolve().parents[3]
# Dev models live in _eval/models (gitignored); the SHIPPED app reads the per-user
# models dir (%LOCALAPPDATA%\BilliardsTrainer\models) so a downloaded/imported model
# (e.g. an exported YOLO11 ball detector) is found in the frozen build too.
MODEL_DIRS = [ROOT / "_eval" / "models", USER_MODELS_DIR]


class OnnxModelStrategy(DetectorStrategy):
    model_based = True  # trained detector — skip the classical size prior

    def __init__(self, model_path: Path, conf: float = 0.25, iou: float = 0.45):
        self._path = Path(model_path)
        self.name = f"onnx_{self._path.stem}"
        self.description = f"ONNX YOLO on raw frame: {self._path.name} (conf {conf})"
        self._conf, self._iou = conf, iou
        self._sess = None
        self._inp = None
        self._size = 640

    def _session(self):
        if self._sess is None:
            import onnxruntime as ort
            # GPU first, CPU last. DirectML (Dml) is the pragmatic Windows GPU path
            # (any DX12 GPU, no CUDA toolkit); CUDA is tried too for non-Windows /
            # onnxruntime-gpu installs. Filter to what's actually available so an
            # unavailable provider never raises — and LOG the one chosen, so a
            # silent CPU fallback (the real-time killer) is visible.
            avail = set(ort.get_available_providers())
            preferred = ["DmlExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
            providers = [p for p in preferred if p in avail] or ["CPUExecutionProvider"]
            self._sess = ort.InferenceSession(str(self._path), providers=providers)
            active = self._sess.get_providers()
            log.info("ONNX detector %s on %s", self._path.name, active[0] if active else "?")
            if active and active[0] == "CPUExecutionProvider":
                log.warning("ONNX running on CPU — real-time will be slow; install "
                            "onnxruntime-directml (Windows) or onnxruntime-gpu for GPU.")
            self._inp = self._sess.get_inputs()[0]
            s = self._inp.shape[2]
            self._size = int(s) if isinstance(s, int) and s > 0 else 640
        return self._sess

    def detect(self, frame_bgr, calib):
        sess = self._session()
        n = self._size
        h, w = frame_bgr.shape[:2]
        # CENTERED letterbox (matches Ultralytics) — padding the image top-left
        # instead shifts small/clustered objects and tanks recall on a rack.
        ratio = min(n / h, n / w)
        nh, nw = int(round(h * ratio)), int(round(w * ratio))
        dw, dh = (n - nw) // 2, (n - nh) // 2
        canvas = np.full((n, n, 3), 114, np.uint8)
        canvas[dh:dh + nh, dw:dw + nw] = cv2.resize(frame_bgr, (nw, nh))
        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[None]
        out = np.squeeze(sess.run(None, {self._inp.name: blob})[0], 0)
        if out.shape[0] < out.shape[1]:
            out = out.T                       # [N, 4+nc]
        boxes, scores = out[:, :4], out[:, 4:]
        nc = scores.shape[1]
        cid, conf = scores.argmax(1), scores.max(1)
        keep = conf >= self._conf
        if nc >= 80:
            keep &= cid == 32
        idx = np.where(keep)[0]
        if idx.size == 0:
            return []
        rects, confs, cls_ids = [], [], []
        for i in idx:
            cx, cy, bw, bh = boxes[i]
            rects.append([float((cx - bw / 2 - dw) / ratio), float((cy - bh / 2 - dh) / ratio),
                          float(bw / ratio), float(bh / ratio)])
            confs.append(float(conf[i]))
            cls_ids.append(int(cid[i]))
        keep_nms = cv2.dnn.NMSBoxes(rects, confs, self._conf, self._iou)
        if len(keep_nms) == 0:
            return []
        table = table_polygon_mask(frame_bgr.shape, calib)
        rmax = ball_radius_raw(calib, frame_bgr.shape) * 4.0
        out_dets = []
        for k in np.array(keep_nms).flatten():
            x, y, bw, bh = rects[int(k)]
            ccx, ccy, rr = x + bw / 2, y + bh / 2, (bw + bh) / 4.0
            if rr > rmax * 1.5 or rr < 2:
                continue
            ix, iy = int(np.clip(ccx, 0, w - 1)), int(np.clip(ccy, 0, h - 1))
            if table[iy, ix] == 0:            # drop clearly off-table detections
                continue
            cls, bgr = classify_crop(frame_bgr, ccx, ccy, rr)
            out_dets.append(Detection(ccx, ccy, rr, bgr, cls, float(confs[int(k)])))
        self._tag_single_cue(frame_bgr, out_dets)
        return out_dets

    def _tag_single_cue(self, frame_bgr, dets):
        """Force exactly ONE cue ball = the whitest detected ball. The trained
        model finds every ball reliably; identifying WHICH is the cue is an
        appearance call, and the cue is the uniquely pure-white ball. This is far
        more robust than the per-crop classical classifier, which mislabels white
        balls under tinted lighting. (M2: cue-ball-first.)"""
        if not dets:
            return
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        h, w = frame_bgr.shape[:2]
        best_i, best_score = -1, 1e9
        whiteness = []
        for i, d in enumerate(dets):
            x0, y0 = max(0, int(d.x - d.radius)), max(0, int(d.y - d.radius))
            x1, y1 = min(w, int(d.x + d.radius) + 1), min(h, int(d.y + d.radius) + 1)
            patch = hsv[y0:y1, x0:x1]
            if patch.size == 0:
                whiteness.append((255.0, 0.0))
                continue
            mean_s = float(patch[:, :, 1].mean())
            mean_v = float(patch[:, :, 2].mean())
            whiteness.append((mean_s, mean_v))
            # whitest = lowest saturation, and bright enough to actually be white
            if mean_v > 140 and mean_s < best_score:
                best_score, best_i = mean_s, i
        for i, d in enumerate(dets):
            if i == best_i:
                d.cls = BallClass.CUE
                d.bgr = (245, 245, 245)
            elif d.cls == BallClass.CUE:
                # a second "cue" from the per-crop classifier — demote it
                d.cls = BallClass.SOLID


def _build():
    seen, out = set(), []
    for d in MODEL_DIRS:
        if not d.exists():
            continue
        for p in sorted(d.glob("*.onnx")):
            if p.stem in seen:
                continue
            seen.add(p.stem)
            out.append(OnnxModelStrategy(p))
    return out


STRATEGIES = _build()
