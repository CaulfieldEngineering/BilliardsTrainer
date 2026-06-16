"""Ball detection on the rectified bird's-eye view.

Two backends behind a common interface:

* ``ClassicalBallDetector`` (default) — Hough circles on the playing surface,
  each candidate validated as non-felt and colour-classified (cue / solid /
  stripe / eight). No model, no torch; runs anywhere. This is deliberately the
  default because pre-trained YOLO has no pool-ball classes and a fine-tuned
  model + labelled data were out of scope for the overnight build.

* ``YoloBallDetector`` — optional. Lazy-imports Ultralytics and loads weights
  from the models dir. Used only when ``balls.backend == 'yolo'`` AND the deps +
  weights are present; otherwise the factory falls back to classical.

Detections are returned in rectified pixel coordinates.
"""

import logging

import cv2
import numpy as np

from ..config import BallSettings, FeltSettings
from .geometry import TableModel
from .types import BallClass, Detection

log = logging.getLogger("vision.balls")


def classify_ball(patch_bgr: np.ndarray, mask: np.ndarray | None = None) -> tuple[BallClass, tuple[int, int, int]]:
    """Classify a ball patch into cue/solid/stripe/eight and return its mean BGR.

    Heuristics on the circular interior:
      * mostly bright + desaturated         -> cue (white)
      * mostly very dark                    -> eight (black)
      * meaningful white fraction + colour  -> stripe
      * otherwise                           -> solid
    """
    if patch_bgr.size == 0:
        return BallClass.UNKNOWN, (200, 200, 200)
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    if mask is not None:
        sel = mask > 0
        if not np.any(sel):
            sel = np.ones(patch_bgr.shape[:2], bool)
    else:
        sel = np.ones(patch_bgr.shape[:2], bool)

    px = patch_bgr[sel].reshape(-1, 3)
    s = hsv[sel][:, 1].astype(np.float32)
    v = hsv[sel][:, 2].astype(np.float32)
    mean_bgr = tuple(int(c) for c in px.mean(axis=0))

    white_frac = float(np.mean((s < 55) & (v > 165)))
    dark_frac = float(np.mean(v < 65))
    med_v = float(np.median(v))
    med_s = float(np.median(s))

    if dark_frac > 0.55:
        return BallClass.EIGHT, mean_bgr
    if white_frac > 0.75:
        return BallClass.CUE, mean_bgr
    # Stripes show a white band alongside a saturated colour.
    if white_frac > 0.25 and med_s > 70:
        return BallClass.STRIPE, mean_bgr
    if med_s > 60 or med_v < 200:
        return BallClass.SOLID, mean_bgr
    return BallClass.CUE if med_v > 180 else BallClass.SOLID, mean_bgr


class BallDetector:
    """Interface."""

    def detect(self, rect_bgr, rect_mask, table: TableModel) -> list[Detection]:
        raise NotImplementedError


class ClassicalBallDetector(BallDetector):
    def __init__(self, balls: BallSettings, felt: FeltSettings):
        self.balls = balls
        self.felt = felt

    def _felt_mask(self, rect_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2HSV)
        f = self.felt
        if f.h_min <= f.h_max:
            m = cv2.inRange(hsv, np.array([f.h_min, f.s_min, f.v_min], np.uint8),
                            np.array([f.h_max, f.s_max, f.v_max], np.uint8))
        else:
            a = cv2.inRange(hsv, np.array([0, f.s_min, f.v_min], np.uint8),
                            np.array([f.h_max, f.s_max, f.v_max], np.uint8))
            b = cv2.inRange(hsv, np.array([f.h_min, f.s_min, f.v_min], np.uint8),
                            np.array([180, f.s_max, f.v_max], np.uint8))
            m = cv2.bitwise_or(a, b)
        return m

    def detect(self, rect_bgr, rect_mask, table: TableModel) -> list[Detection]:
        if rect_bgr is None or rect_bgr.size == 0:
            return []
        short = max(table.short_side, 1.0)
        min_r = max(3, int(self.balls.min_radius_frac * short))
        max_r = max(min_r + 2, int(self.balls.max_radius_frac * short))

        gray = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)

        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=int(min_r * 1.6),
            param1=110, param2=max(8, self.balls.detect_param2),
            minRadius=min_r, maxRadius=max_r,
        )
        if circles is None:
            return []
        circles = np.round(circles[0]).astype(int)

        felt_mask = self._felt_mask(rect_bgr)
        detections: list[Detection] = []
        h, w = rect_bgr.shape[:2]
        for cx, cy, r in circles:
            # must sit on the playing surface, away from the rail band
            if not table.on_table(cx, cy, margin=-r * 0.2):
                continue
            # Mask out the whole pocket region: the dark hole + its surround read
            # as a ball-sized dark blob, which is exactly the "balls floating in
            # the pockets" artefact. A ball that has reached a pocket has been
            # potted, so there is nothing real to detect there.
            pocket, pdist = table.nearest_pocket(cx, cy)
            if pdist < table.pocket_radius:
                continue
            x0, y0 = max(0, cx - r), max(0, cy - r)
            x1, y1 = min(w, cx + r + 1), min(h, cy + r + 1)
            patch = rect_bgr[y0:y1, x0:x1]
            if patch.size == 0:
                continue
            # circular mask for the patch
            cmask = np.zeros(patch.shape[:2], np.uint8)
            cv2.circle(cmask, (cx - x0, cy - y0), r, 255, -1)
            felt_patch = felt_mask[y0:y1, x0:x1]
            felt_frac = float(np.mean(felt_patch[cmask > 0] > 0)) if np.any(cmask) else 1.0
            # a ball is mostly non-felt; reject felt-coloured false circles
            if felt_frac > 0.6:
                continue
            cls, bgr = classify_ball(patch, cmask)
            detections.append(Detection(x=float(cx), y=float(cy), radius=float(r),
                                        bgr=bgr, cls=cls, score=1.0 - felt_frac))

        # The cue ball is uniquely white and easy to segment by colour — a
        # dedicated pass makes its detection far more reliable than Hough alone.
        detections.extend(self._detect_cue(rect_bgr, table, min_r, max_r))
        return _dedupe(detections, int(min_r * 1.4))

    def _detect_cue(self, rect_bgr, table: TableModel, min_r: int, max_r: int) -> list[Detection]:
        hsv = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv, np.array([0, 0, 160], np.uint8),
                            np.array([180, 85, 255], np.uint8))
        white = cv2.morphologyEx(white, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out: list[Detection] = []
        min_area = np.pi * (min_r * 0.7) ** 2
        max_area = np.pi * (max_r * 1.6) ** 2
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (min_area <= area <= max_area):
                continue
            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            if r <= 0:
                continue
            circularity = area / (np.pi * r * r)
            if circularity < 0.55:           # roughly round
                continue
            if not table.on_table(cx, cy, margin=-r * 0.2):
                continue
            _, pdist = table.nearest_pocket(cx, cy)
            if pdist < table.pocket_radius:
                continue
            out.append(Detection(x=float(cx), y=float(cy), radius=float(r),
                                  bgr=(245, 245, 245), cls=BallClass.CUE, score=0.95))
        return out


class YoloBallDetector(BallDetector):
    """Optional Ultralytics backend. Raises at construction if unavailable so the
    factory can fall back."""

    def __init__(self, weights_path, conf: float = 0.25):
        try:
            from ultralytics import YOLO  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional dep
            raise RuntimeError("ultralytics not installed; install the [yolo] extra") from exc
        from ultralytics import YOLO
        self._model = YOLO(str(weights_path))
        self._conf = conf

    def detect(self, rect_bgr, rect_mask, table: TableModel) -> list[Detection]:  # pragma: no cover
        results = self._model.predict(rect_bgr, conf=self._conf, verbose=False)
        out: list[Detection] = []
        for res in results:
            for box in res.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                r = (x2 - x1 + y2 - y1) / 4
                if not table.on_table(cx, cy, margin=-r * 0.2):
                    continue
                patch = rect_bgr[int(max(0, cy - r)):int(cy + r), int(max(0, cx - r)):int(cx + r)]
                cls, bgr = classify_ball(patch) if patch.size else (BallClass.UNKNOWN, (200, 200, 200))
                out.append(Detection(cx, cy, r, bgr, cls, float(box.conf[0])))
        return _dedupe(out, 4)


def _dedupe(dets: list[Detection], min_dist: float) -> list[Detection]:
    """Greedy non-max suppression by centre distance (keep higher score)."""
    dets = sorted(dets, key=lambda d: d.score, reverse=True)
    kept: list[Detection] = []
    for d in dets:
        if all(np.hypot(d.x - k.x, d.y - k.y) > min_dist for k in kept):
            kept.append(d)
    return kept


def _fetch_weights(url: str, dest) -> bool:
    """Download YOLO weights to ``dest`` if not already present. Best-effort."""
    if dest.exists():
        return True
    try:
        import requests
        log.info("Fetching YOLO weights: %s", url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            tmp = dest.with_suffix(".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    if chunk:
                        f.write(chunk)
            tmp.replace(dest)
        return True
    except Exception as exc:  # noqa: BLE001 - network/io best effort
        log.warning("Could not fetch YOLO weights (%s)", exc)
        return False


_WEIGHT_NAMES = ("pool_balls.pt", "billiards.pt", "best.pt", "joe_table.pt", "yolov8n.pt")


def find_yolo_weights(models_dir=None):
    """Return the path to bundled/downloaded YOLO weights, or None.

    This is the single source of truth for "can we do AI detection?" — the UI
    uses it to decide whether to offer the auto-detection toggle or fall back to
    manual mode with a banner.
    """
    from ..config import MODELS_DIR
    mdir = models_dir or MODELS_DIR
    for name in _WEIGHT_NAMES:
        cand = mdir / name
        if cand.exists():
            return cand
    return None


def yolo_weights_available(models_dir=None) -> bool:
    return find_yolo_weights(models_dir) is not None


def make_detector(balls: BallSettings, felt: FeltSettings, models_dir=None) -> BallDetector:
    """Build the best available detector.

    YOLO is preferred whenever weights are on disk (or fetchable) AND Ultralytics
    is importable — a fine-tuned net beats classical CV on a real overhead camera.
    Classical (Hough + colour) is the fallback when there's no model or the deps
    are missing. ``backend == 'classical'`` forces classical regardless.
    """
    if balls.backend != "classical":
        from ..config import MODELS_DIR
        mdir = models_dir or MODELS_DIR
        weights = find_yolo_weights(mdir)
        # auto-fetch from a configured URL if nothing is on disk
        if weights is None and getattr(balls, "yolo_weights_url", ""):
            target = mdir / "pool_balls.pt"
            if _fetch_weights(balls.yolo_weights_url, target):
                weights = target
        if weights is not None:
            try:
                log.info("Using YOLO ball detector: %s", weights)
                return YoloBallDetector(weights, conf=getattr(balls, "yolo_conf", 0.25))
            except RuntimeError as exc:
                log.warning("YOLO unavailable (%s); falling back to classical", exc)
        else:
            log.warning("No YOLO weights in %s and no fetch URL; using classical", mdir)
    return ClassicalBallDetector(balls, felt)
