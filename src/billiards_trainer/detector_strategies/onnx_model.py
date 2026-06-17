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
from ..vision.balls import classify_pool_ball
from ..vision.types import BallClass, Detection
from . import DetectorStrategy, ball_radius_raw, table_polygon_mask

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

    def _infer(self, image, ox: int = 0, oy: int = 0):
        """Run the model on one image and return [(cx, cy, r, conf)] in FULL-frame
        coords (add the image's (ox, oy) offset)."""
        sess = self._session()
        n = self._size
        h, w = image.shape[:2]
        if h < 8 or w < 8:
            return []
        # CENTERED letterbox (matches Ultralytics) — padding top-left shifts
        # small/clustered objects and tanks recall on a rack.
        ratio = min(n / h, n / w)
        nh, nw = int(round(h * ratio)), int(round(w * ratio))
        dw, dh = (n - nw) // 2, (n - nh) // 2
        canvas = np.full((n, n, 3), 114, np.uint8)
        canvas[dh:dh + nh, dw:dw + nw] = cv2.resize(image, (nw, nh))
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
        rects, confs = [], []
        for i in idx:
            cx, cy, bw, bh = boxes[i]
            rects.append([float((cx - bw / 2 - dw) / ratio), float((cy - bh / 2 - dh) / ratio),
                          float(bw / ratio), float(bh / ratio)])
            confs.append(float(conf[i]))
        keep_nms = cv2.dnn.NMSBoxes(rects, confs, self._conf, self._iou)
        res = []
        for k in np.array(keep_nms).flatten():
            x, y, bw, bh = rects[int(k)]
            res.append((x + bw / 2 + ox, y + bh / 2 + oy, (bw + bh) / 4.0, confs[int(k)]))
        return res

    @staticmethod
    def _merge_boxes(boxes):
        """Dedupe near-coincident boxes (the same ball seen by the full + far-rail
        scans), keeping the higher-confidence one."""
        kept = []
        for b in sorted(boxes, key=lambda t: t[3], reverse=True):
            mind = 0.7 * max(b[2], 8.0)
            if all((b[0] - k[0]) ** 2 + (b[1] - k[1]) ** 2 > mind * mind for k in kept):
                kept.append(b)
        return kept

    def detect(self, frame_bgr, calib):
        h, w = frame_bgr.shape[:2]
        boxes = self._infer(frame_bgr)
        # Far-rail recall: the far cushion is foreshortened, so balls there are tiny
        # after the 640 downscale and often missed. Re-scan the top ~60% of the
        # frame (where the far rail sits) upscaled, then merge — recovers small far
        # balls without changing the model.
        th = int(h * 0.60)
        if th > 64:
            boxes = self._merge_boxes(boxes + self._infer(frame_bgr[0:th, 0:w]))
        if not boxes:
            return []
        table = table_polygon_mask(frame_bgr.shape, calib)
        rmax = ball_radius_raw(calib, frame_bgr.shape) * 4.0
        out_dets = []
        for ccx, ccy, rr, cf in boxes:
            if rr > rmax * 1.5 or rr < 2:
                continue
            ix, iy = int(np.clip(ccx, 0, w - 1)), int(np.clip(ccy, 0, h - 1))
            if table[iy, ix] == 0:            # drop clearly off-table detections
                continue
            crop = frame_bgr[max(0, int(ccy - rr)):int(ccy + rr) + 1,
                             max(0, int(ccx - rr)):int(ccx + rr) + 1]
            cls, number, bgr = classify_pool_ball(crop)
            out_dets.append(Detection(ccx, ccy, rr, bgr, cls, float(cf), number=number))
        self._enforce_single_cue(frame_bgr, out_dets)
        return out_dets

    def _enforce_single_cue(self, frame_bgr, dets):
        """Exactly ONE cue ball, robustly.

        classify_pool_ball separates the cue from the 9-ball (white + yellow stripe)
        by coloured-fraction. Here we make the count exactly one:
          - >1 cue (rare, glare) -> keep the least-saturated (truest white);
          - 0 cues but a clearly white-ish ball exists -> PROMOTE the whitest, so a
            marginal-appearance frame (blur/shadow) doesn't lose the cue. The 9-ball
            (yellow band => higher saturation) loses to the true cue, so this keeps
            the cue/9 distinction while restoring high cue recall.
        If no ball is white-ish, leave no cue (it's genuinely occluded — the tracker
        keep-alive holds the real cue track)."""
        if not dets:
            return
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        h, w = frame_bgr.shape[:2]

        def stats(d):
            x0, y0 = max(0, int(d.x - d.radius)), max(0, int(d.y - d.radius))
            x1, y1 = min(w, int(d.x + d.radius) + 1), min(h, int(d.y + d.radius) + 1)
            p = hsv[y0:y1, x0:x1]
            if p.size == 0:
                return 255.0, 0.0
            return float(p[:, :, 1].mean()), float(p[:, :, 2].mean())

        cues = [d for d in dets if d.cls == BallClass.CUE]
        if len(cues) == 1:
            return
        if len(cues) > 1:
            cues.sort(key=lambda d: stats(d)[0])      # whitest (lowest S) first
            for d in cues[1:]:
                d.cls, d.number, d.bgr = BallClass.SOLID, -1, (190, 190, 190)
            return
        # zero cues -> the cue is the whitest (least-saturated) ball on the table,
        # so promote it. Loose gate (just "bright and not strongly coloured") so a
        # dim/motion-blurred cue during a shot is still caught — matching the old
        # always-find-the-cue robustness. The 9-ball's yellow band keeps its mean
        # saturation above a true cue, so the cue still wins when both are present.
        best, best_s = None, 1e9
        for d in dets:
            ms, mv = stats(d)
            if mv > 105 and ms < 130 and ms < best_s:
                best_s, best = ms, d
        if best is not None:
            best.cls, best.number, best.bgr = BallClass.CUE, 0, (245, 245, 245)


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
