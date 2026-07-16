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
import threading
from pathlib import Path

import cv2
import numpy as np

from ..config import MODELS_DIR as USER_MODELS_DIR
from ..vision.balls import classify_pool_ball, number_to_class, pool_ball_bgr
from ..vision.types import BallClass, Detection
from . import DetectorStrategy, ball_radius_raw, table_polygon_mask

log = logging.getLogger("detector.onnx")

ROOT = Path(__file__).resolve().parents[3]
# Dev models live in _eval/models (gitignored); the SHIPPED app reads the per-user
# models dir (%LOCALAPPDATA%\BilliardsTrainer\models) so a downloaded/imported model
# (e.g. an exported YOLO11 ball detector) is found in the frozen build too.
MODEL_DIRS = [ROOT / "_eval" / "models", USER_MODELS_DIR]


def pick_providers(available) -> list[str]:
    """Execution providers in accelerator-first order, filtered to what this
    install actually has (an unavailable provider must never raise):
    DirectML (Windows, any DX12 GPU), CUDA (onnxruntime-gpu), CoreML (Apple
    silicon — GPU/ANE, ships in the standard macOS wheel), then CPU. The
    caller LOGS the one chosen, so a silent CPU fallback (the real-time
    killer) is always visible."""
    avail = set(available)
    preferred = ["DmlExecutionProvider", "CUDAExecutionProvider",
                 "CoreMLExecutionProvider", "CPUExecutionProvider"]
    return [p for p in preferred if p in avail] or ["CPUExecutionProvider"]


class OnnxModelStrategy(DetectorStrategy):
    model_based = True  # trained detector — skip the classical size prior
    # Class-level defaults (not set in __init__) so tests can build instances via
    # __new__ with hand-set attributes:
    _canvas = None       # reusable letterbox canvas (avoids a 1.2MB alloc/frame)
    # detect() is called from the worker pipeline AND (rarely) the UI-thread
    # Ball ID Trainer on the SAME strategy singleton — the reused canvas and
    # self._nc must never interleave between threads.
    _detect_lock = threading.Lock()
    # Second inference pass over the (foreshortened) far-rail region. Recovers
    # tiny far balls but doubles GPU cost; the pipeline wires this from
    # settings.balls.far_rail_rescan so it can be tuned live.
    far_rail_rescan = True

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
            providers = pick_providers(ort.get_available_providers())
            self._sess = ort.InferenceSession(str(self._path), providers=providers)
            active = self._sess.get_providers()
            log.info("ONNX detector %s on %s", self._path.name, active[0] if active else "?")
            if active and active[0] == "CPUExecutionProvider":
                log.warning("ONNX running on CPU — real-time may be slow; install "
                            "onnxruntime-directml (Windows) or onnxruntime-gpu "
                            "(CUDA) for GPU. On Apple silicon the standard "
                            "onnxruntime wheel ships CoreML.")
            self._inp = self._sess.get_inputs()[0]
            s = self._inp.shape[2]
            self._size = int(s) if isinstance(s, int) and s > 0 else 640
        return self._sess

    def _infer(self, image, ox: int = 0, oy: int = 0):
        """Run the model on one image and return [(cx, cy, r, conf, cls_id)] in
        FULL-frame coords (add the image's (ox, oy) offset)."""
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
        canvas = self._canvas
        if canvas is None or canvas.shape[0] != n:
            canvas = self._canvas = np.empty((n, n, 3), np.uint8)
        # fill ONLY the padding bands (the interior is fully overwritten by the
        # resize below); the >= guards cover odd-rounding leftover rows/cols
        if dh or dh + nh < n:
            canvas[:dh].fill(114)
            canvas[dh + nh:].fill(114)
        if dw or dw + nw < n:
            canvas[:, :dw].fill(114)
            canvas[:, dw + nw:].fill(114)
        canvas[dh:dh + nh, dw:dw + nw] = cv2.resize(image, (nw, nh))
        # one fused C++ pass: uint8 -> float32 NCHW, /255, BGR->RGB
        blob = cv2.dnn.blobFromImage(canvas, scalefactor=1.0 / 255.0, swapRB=True)
        out = np.squeeze(sess.run(None, {self._inp.name: blob})[0], 0)
        if out.shape[0] < out.shape[1]:
            out = out.T                       # [N, 4+nc]
        boxes, scores = out[:, :4], out[:, 4:]
        nc = scores.shape[1]
        self._nc = nc
        cid, conf = scores.argmax(1), scores.max(1)
        keep = conf >= self._conf
        if nc >= 80:
            keep &= cid == 32
        idx = np.where(keep)[0]
        if idx.size == 0:
            return []
        # un-letterbox, vectorised: [x, y, w, h] in original-image coords
        b = boxes[idx].astype(np.float32)
        xy = (b[:, :2] - b[:, 2:] * 0.5 - np.array([dw, dh], np.float32)) / ratio
        wh = b[:, 2:] / ratio
        rects = np.hstack([xy, wh])
        confs = conf[idx].astype(np.float32)
        cls_ids = cid[idx]
        keep_nms = cv2.dnn.NMSBoxes(rects, confs, self._conf, self._iou)
        res = []
        for k in np.asarray(keep_nms).flatten():
            x, y, bw, bh = rects[int(k)]
            res.append((x + bw / 2 + ox, y + bh / 2 + oy, (bw + bh) / 4.0,
                        float(confs[int(k)]), int(cls_ids[int(k)])))
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

    def detect(self, frame_bgr, calib, rescan: bool | None = None):
        """``rescan`` overrides the instance's far_rail_rescan for THIS call —
        the strategy is a process-wide singleton, so per-call overrides (the
        offline labeller wants full recall regardless of the live pipeline's
        perf setting) must not mutate shared state."""
        with self._detect_lock:
            return self._detect(frame_bgr, calib, rescan)

    def _detect(self, frame_bgr, calib, rescan: bool | None = None):
        h, w = frame_bgr.shape[:2]
        boxes = self._infer(frame_bgr)
        # Far-rail recall: the far cushion is foreshortened, so balls there are tiny
        # after the 640 downscale and often missed. Re-scan the top ~60% of the
        # frame (where the far rail sits) upscaled, then merge — recovers small far
        # balls without changing the model. Costs a second inference per frame;
        # settings.balls.far_rail_rescan turns it off when the budget is tight.
        th = int(h * 0.60)
        do_rescan = self.far_rail_rescan if rescan is None else rescan
        if th > 64 and do_rescan:
            boxes = self._merge_boxes(boxes + self._infer(frame_bgr[0:th, 0:w]))
        if not boxes:
            return []
        # No calibration => no polygon to test against; skip the mask entirely
        # (an all-255 full-frame mask filters nothing and costs a 2MP alloc).
        table = table_polygon_mask(frame_bgr.shape, calib) if calib is not None else None
        rmax = ball_radius_raw(calib, frame_bgr.shape) * 4.0
        # A 16-class model is the trained cue+1..15 detector — its class IS the ball
        # number, so trust it directly (no colour heuristic). Anything else (the
        # single-class ball model, or COCO) gets the appearance classifier.
        numbered = getattr(self, "_nc", 0) == 16
        out_dets = []
        for ccx, ccy, rr, cf, cid in boxes:
            if rr > rmax * 1.5 or rr < 2:
                continue
            ix, iy = int(np.clip(ccx, 0, w - 1)), int(np.clip(ccy, 0, h - 1))
            if table is not None and table[iy, ix] == 0:  # drop clearly off-table detections
                continue
            if numbered:
                number = int(cid)             # class index == ball number (0=cue)
                cls, bgr = number_to_class(number), pool_ball_bgr(number)
            else:
                crop = frame_bgr[max(0, int(ccy - rr)):int(ccy + rr) + 1,
                                 max(0, int(ccx - rr)):int(ccx + rr) + 1]
                cls, number, bgr = classify_pool_ball(crop)
            out_dets.append(Detection(ccx, ccy, rr, bgr, cls, float(cf), number=number))
        if numbered:
            self._enforce_unique_numbers(out_dets)
        else:
            self._enforce_single_cue(frame_bgr, out_dets)
        return out_dets

    @staticmethod
    def _enforce_unique_numbers(dets):
        """There is exactly one of each ball, so keep only the highest-confidence
        detection per number; demote duplicates to unknown (kept on the table as a
        grey '?', not dropped, so a real ball isn't lost to a misread)."""
        best: dict[int, object] = {}
        for d in dets:
            if d.number < 0:
                continue
            cur = best.get(d.number)
            if cur is None or d.score > cur.score:
                best[d.number] = d
        for d in dets:
            if d.number >= 0 and best.get(d.number) is not d:
                d.number, d.cls, d.bgr = -1, BallClass.UNKNOWN, (190, 190, 190)

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
        cues = [d for d in dets if d.cls == BallClass.CUE]
        if len(cues) == 1:
            return  # common case — no pixel work at all
        h, w = frame_bgr.shape[:2]

        def stats(d):
            # HSV of just the ball crop — converting the whole frame for a few
            # ~20px crops was the single costliest step of this path.
            x0, y0 = max(0, int(d.x - d.radius)), max(0, int(d.y - d.radius))
            x1, y1 = min(w, int(d.x + d.radius) + 1), min(h, int(d.y + d.radius) + 1)
            if x1 <= x0 or y1 <= y0:
                return 255.0, 0.0
            p = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
            return float(p[:, :, 1].mean()), float(p[:, :, 2].mean())
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
