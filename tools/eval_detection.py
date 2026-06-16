"""Detection eval harness — quantify false positives and true positives.

Two scenarios:
  * NOISY IDLE: a settled table with realistic camera noise + occasional
    compression-block artifacts (the failure Joe hit — counter creeping while
    nothing happens). Metric: false shots per minute. Lower is better.
  * DEMO: the scripted make-shot. Metric: makes detected (true positives must be
    preserved). Higher (== expected) is better.

    python tools/eval_detection.py [seconds]

Run it before and after a detector change to prove the improvement with numbers.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from billiards_trainer.capture.camera import DemoSource  # noqa: E402
from billiards_trainer.config import Settings  # noqa: E402
from billiards_trainer.vision.pipeline import Pipeline  # noqa: E402

FPS = 30.0


def settled_frame() -> np.ndarray:
    """A demo frame with balls at rest (the 'idle table' baseline)."""
    src = DemoSource()
    f = None
    for _ in range(10):          # frames 0-9 are the settle phase
        f = src.read()
    return f


# Noisy/low-light camera: heavy incoherent sensor noise (spikes a pixel-diff
# motion metric every frame, but has NO coherent optical flow) + MJPEG blocks.
NOISE_SIGMA = 11


def add_camera_noise(frame: np.ndarray, rng: np.random.Generator, idx: int) -> np.ndarray:
    out = frame.astype(np.int16)
    out += rng.normal(0, NOISE_SIGMA, frame.shape).astype(np.int16)   # sensor noise
    # A flickering specular highlight on the felt (a real false-positive source):
    # a fixed region whose brightness jitters every frame. High pixel-change energy
    # but NO coherent optical flow — exactly what multi-signal fusion should reject.
    h, w = frame.shape[:2]
    fy, fx = int(h * 0.45), int(w * 0.40)
    out[fy:fy + 70, fx:fx + 70] += int(rng.integers(-45, 45))
    if idx % 11 == 0:                                                 # periodic blocks
        for _ in range(int(rng.integers(3, 7))):
            y, x = int(rng.integers(0, h - 24)), int(rng.integers(0, w - 24))
            out[y:y + 24, x:x + 24] += int(rng.integers(-50, 50))
    return np.clip(out, 0, 255).astype(np.uint8)


def eval_idle(seconds: float, settings: Settings) -> dict:
    n = int(seconds * FPS)
    rng = np.random.default_rng(1234)
    base = settled_frame()
    pipe = Pipeline(settings)
    pipe.process(base, t=0.0)  # calibrate
    false_counts = false_starts = 0
    prev_state = "settled"
    sig = {"motion": [], "flow": [], "fg": [], "fused": []}
    for i in range(n):
        noisy = add_camera_noise(base, rng, i)
        res = pipe.process(noisy, t=1.0 + i / FPS)
        if res.shot_event is not None:
            false_counts += 1
        if prev_state != "moving" and res.shot_state == "moving":
            false_starts += 1
        prev_state = res.shot_state
        for k in sig:
            sig[k].append(res.diag.get(k, 0.0))
    minutes = n / FPS / 60.0
    return {"false_counts_per_min": false_counts / minutes,
            "false_starts_per_min": false_starts / minutes,
            "sig": {k: (float(np.mean(v)), float(np.max(v))) for k, v in sig.items()}}


def eval_demo(settings: Settings) -> dict:
    s = Settings.from_dict(settings.to_dict())
    s.detection.warmup_seconds = 2.5   # > MOG2 warmup so fg is ready
    s.detection.cooldown_seconds = 0.5
    src = DemoSource()
    pipe = Pipeline(s)
    makes = 0
    moving = {"motion": [], "flow": [], "fg": [], "fused": []}
    for i in range(480):
        res = pipe.process(src.read(), t=i / FPS)
        if res.shot_event and res.shot_event.outcome.value == "make":
            makes += 1
        if res.diag.get("state") == "moving":
            for k in moving:
                moving[k].append(res.diag.get(k, 0.0))
    sig = {k: (round(np.mean(v), 2) if v else 0, round(np.max(v), 2) if v else 0)
           for k, v in moving.items()}
    return {"demo_makes": makes, "moving_sig": sig}


def eval_video(path: str, settings: Settings) -> dict:
    """Run the full detection pipeline over a RECORDED clip (no camera needed).

    This is the fast inner loop for detection work: record once (or use a
    Capture-for-analysis zip's frames / any mp4), then iterate the detector
    against the fixture instead of standing at the table."""
    from billiards_trainer.capture.camera import open_source
    s = Settings.from_dict(settings.to_dict())
    src = open_source(path)
    if hasattr(src, "opened") and not src.opened:
        print(f"  could not open {path}")
        return {}
    pipe = Pipeline(s, source=path)
    pipe.detect_enabled = True
    n = int(getattr(src, "frame_count", 0)) or 600
    balls, makes, frames = [], 0, 0
    for i in range(min(n, 1800)):
        frame = src.read()
        if frame is None:
            break
        res = pipe.process(frame, t=i / FPS)
        frames += 1
        balls.append(res.n_balls)
        if res.shot_event and res.shot_event.outcome.value == "make":
            makes += 1
    src.release()
    return {"frames": frames, "calibrated": pipe.calib.is_calibrated,
            "mean_balls": float(np.mean(balls)) if balls else 0.0,
            "max_balls": int(np.max(balls)) if balls else 0, "makes": makes}


def main() -> int:
    if "--video" in sys.argv:
        path = sys.argv[sys.argv.index("--video") + 1]
        print(f"=== Detection eval on recorded clip: {path} ===")
        r = eval_video(path, Settings())
        print(f"  frames={r.get('frames')} calibrated={r.get('calibrated')} "
              f"mean_balls={r.get('mean_balls', 0):.1f} max_balls={r.get('max_balls')} "
              f"makes={r.get('makes')}")
        return 0
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    print(f"=== Detection eval ({seconds:.0f}s noisy-idle flicker + demo) ===")
    for label, fusion in [("BASELINE motion-only", False), ("FUSED (bgsub+flow)", True)]:
        s = Settings()
        s.detection.use_fusion = fusion
        idle = eval_idle(seconds, s)
        demo = eval_demo(s)
        sg = idle["sig"]
        print(f"\n[{label}]")
        print(f"  idle false counts/min: {idle['false_counts_per_min']:.2f}   "
              f"false starts/min: {idle['false_starts_per_min']:.2f}")
        print(f"  idle signals (mean,max): motion={sg['motion']}  flow={sg['flow']}  "
              f"fg={sg['fg']}  fused={sg['fused']}")
        print(f"  demo makes: {demo['demo_makes']}  moving signals={demo['moving_sig']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
