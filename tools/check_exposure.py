"""Exposure / motion-smear check for a session clip " the fixed version.

The 2026-08 blur saga ended with a measurement bug, not a camera bug: a
smear metric that ranked ALL elongated moving blobs put cue-stick
fragments at the top (a stick is elongated at rest " no motion needed)
and read like "shutter not applying" for days, against the operator's
correct direct observation. Two fixes are structural here:

  1. Ball gating: a smeared BALL keeps its minor axis ~= ball diameter
     (motion stretches the major axis only). Blobs whose minor axis is
     far from ball size are reported separately and never headline.
  2. Evidence with the number: crops of the top blobs are saved to a
     contact sheet next to the report, so "worst smear 60px" can be
     eyeballed as ball-vs-stick in one glance instead of trusted.

Usage: python tools/check_exposure.py [clip.mp4] [--ball-diam 28]
Defaults to the newest session recording. ASCII output (cp1252 console).
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

SESS_DIR = "C:/Users/Joe/Dropbox/Billiards/BilliardsTrainer"


def newest_session() -> str:
    files = glob.glob(os.path.join(SESS_DIR, "session-*.mp4"))
    if not files:
        sys.exit("no session recordings found")
    return max(files, key=os.path.getmtime)


def main() -> None:
    from _lowprio import demote
    demote()
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", nargs="?", default=None)
    ap.add_argument("--ball-diam", type=float, default=28.0,
                    help="expected ball diameter in raw px (rig default 28)")
    ap.add_argument("--samples", type=int, default=240)
    args = ap.parse_args()
    clip = args.clip or newest_session()

    cap = cv2.VideoCapture(clip)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"clip: {os.path.basename(clip)}  ({n / fps / 60:.1f} min @ {fps:.0f}fps)")

    # median background from sparse frames
    bg_frames = []
    for i in np.linspace(0, n - 1, 25).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            bg_frames.append(f)
    bg = np.median(np.stack(bg_frames), axis=0).astype(np.uint8)
    # felt mask from the background's dominant hue: balls in flight are always
    # over felt, while shoes/arm/stick reach in from outside it -- the main
    # contamination the 2026-08 saga was built on. Erode so rail shadows and
    # the felt edge don't leak through.
    hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [180], [0, 180]).ravel()
    hue = int(hist.argmax())
    felt = cv2.inRange(hsv, (max(0, hue - 12), 60, 40), (min(180, hue + 12), 255, 255))
    felt = cv2.erode(felt, np.ones((25, 25), np.uint8))
    g = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    fm = g[felt > 0]
    fmv = fm.mean() if fm.size else -1
    print(f"exposure: felt_mean {fmv:.0f} (target 133-201)   "
          f"clipped(>=250) {100.0 * (g >= 250).mean():.2f}%")

    d = args.ball_diam
    balls, others = [], []   # (smear, t, crop, w, h)
    step = max(1, n // args.samples)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step:
            idx += 1
            continue
        ok, frame = cap.retrieve()
        idx += 1
        if not ok:
            break
        diff = cv2.absdiff(frame, bg).max(axis=2)
        mask = (diff > 40).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if len(c) < 5 or cv2.contourArea(c) < 0.25 * d * d:
                continue
            (cx, cy), (ax1, ax2), _ang = cv2.fitEllipse(c)
            major, minor = max(ax1, ax2), min(ax1, ax2)
            if major > 12 * d:            # walls of person/shadow " skip
                continue
            smear = major - minor
            m = int(major / 2 + 8)
            x0, y0 = max(0, int(cx - m)), max(0, int(cy - m))
            crop = frame[y0:y0 + 2 * m, x0:x0 + 2 * m].copy()
            rec = (smear, idx / fps, crop, major, minor)
            # ball gate: motion stretches major only, so a real (even badly
            # smeared) ball keeps minor ~= ball diameter
            on_felt = felt[min(felt.shape[0] - 1, int(cy)),
                           min(felt.shape[1] - 1, int(cx))] > 0
            (balls if on_felt and 0.6 * d <= minor <= 1.6 * d and major <= 5 * d
             else others).append(rec)
    cap.release()

    def report(name, rows):
        rows.sort(key=lambda r: -r[0])
        print(f"\n{name} (top 5 of {len(rows)}):")
        for smear, t, _c, major, minor in rows[:5]:
            print(f"  t={t:7.1f}s  {major:3.0f} x {minor:3.0f}   smear {smear:3.0f}px")
        return rows[:5]

    top_b = report("BALL-LIKE moving blobs (the smear verdict)", balls)
    report("non-ball elongated blobs (stick/arm/shadow - IGNORED for verdict)", others)

    if top_b:
        tiles = []
        size = 96
        for smear, _t, crop, *_ in top_b:
            c = cv2.resize(crop, (size, size))
            cv2.putText(c, f"{smear:.0f}px", (4, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            tiles.append(c)
        sheet = cv2.hconcat(tiles)
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "_eval", "exposure_check.png")
        out = os.path.abspath(out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        cv2.imwrite(out, sheet)
        print(f"\ncrops of top ball-like blobs: {out}")
        worst = top_b[0][0]
        print("VERDICT: worst ball smear {:.0f}px -> {}".format(
            worst, "CRISP (fast shutter confirmed)" if worst < 12
            else "check the contact sheet before blaming the camera"))


if __name__ == "__main__":
    main()
