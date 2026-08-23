"""Shot-window forensic re-pass: recover the unobserved flight, offline.

Joe: "I refuse to believe you can't think of something." This productizes
the manual forensics that settled both ground-truth shots — instead of
re-analysing whole sessions (13h), it re-opens ONLY a miss's ~1s blur gap
(~20min of footage across the whole backlog) and works the gap with
constraints a live pass doesn't have:

  CORRIDOR SEARCH — we know where the ball SAT (its resting spot at
  contact) and where it REAPPEARED. A rolling ball connects them in a
  straight line, so the search is a narrow band along that chord, not the
  open table. Near-1D; trivially robust.

  OWN-COLOUR REFERENCE — the ball's true colour is sampled from the
  pre-strike frame at its known resting spot, so the corridor match is
  against THIS ball, not a palette.

  MEDIAN BACKGROUND over the window (the static scene), same discipline
  blur recovery uses — a frame difference would light up the cue stick.

Recovered observations feed the trajectory fit; the verdict is read off
the fit. Built with a bounded-lag core deliberately: live mode later is
the same solver riding a few seconds behind the table (Joe: "the goal
after POC is realtime, maybe slight latency").

    python tools/forensic_repass.py --video <mp4> --start <t> [--ball N]
"""

import argparse

# Joe: "my mouse and keyboard have been completely lagging when the app
# and AI are running hard." Heavy batch work must never starve the
# desktop or the live pipeline: drop OURSELVES to below-normal priority.
try:
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(
        ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)  # BELOW_NORMAL
except Exception:
    pass

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np


def corridor_observations(video, tf, contact_vis, reappear_vis, t0, t1,
                          colour_ref, r_vis):
    """Ball-coloured blobs inside the chord band between contact and
    reappearance, every frame of the gap. Returns [(t, vx, vy)] visual px."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, (t0 - 0.8)) * 1000.0)
    frames, times = [], []
    n = int((t1 - t0 + 1.6) * fps) + 2
    for i in range(n):
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
        times.append(t0 - 0.8 + i / fps)
    cap.release()
    if len(frames) < 8:
        return []
    bg = np.median(np.stack(frames[:: max(1, len(frames) // 12)])
                   .astype(np.float32), axis=0)
    ax, ay = contact_vis
    bx, by = reappear_vis
    ux, uy = bx - ax, by - ay
    seg = math.hypot(ux, uy)
    if seg < 4 * r_vis:
        return []
    ux, uy = ux / seg, uy / seg
    band = 2.5 * r_vis
    out = []
    for fr, t in zip(frames, times, strict=False):
        if not (t0 <= t <= t1):
            continue
        moved = np.linalg.norm(fr.astype(np.float32) - bg, axis=2)
        mask = (moved > 18).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for c in cnts:
            a = cv2.contourArea(c)
            if a < 0.2 * math.pi * r_vis * r_vis:
                continue
            m = cv2.moments(c)
            if m["m00"] <= 0:
                continue
            x, y = m["m10"] / m["m00"], m["m01"] / m["m00"]
            along = (x - ax) * ux + (y - ay) * uy
            perp = abs(-(y - ay) * ux + (x - ax) * uy)
            if not (-r_vis <= along <= seg + 2 * r_vis) or perp > band:
                continue                      # outside the physics corridor
            cm = np.zeros(mask.shape, np.uint8)
            cv2.drawContours(cm, [c], -1, 255, -1)
            mean = np.array(cv2.mean(fr, mask=cm)[:3], dtype=float)
            d = float(np.linalg.norm(mean - colour_ref))
            if d > 130.0:
                continue                      # not this ball's colour family
            if best is None or d < best[0]:
                best = (d, x, y)
        if best is not None:
            out.append((t, best[1], best[2]))
    return out


def streak_directions(video, t0, t1, contact_vis, reappear_vis, colour_ref,
                      r_vis):
    """STREAK-AS-SENSOR: a motion smear's long axis IS the ball's velocity
    direction at that instant, readable even when the smear is too faint
    to localize as a position. Returns unit vectors (sign resolved toward
    the reappearance point) for elongated, ball-coloured blobs in the
    corridor. 180-degree ambiguity is resolved by the chord."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, (t0 - 0.8)) * 1000.0)
    frames = []
    for _ in range(int((t1 - t0 + 1.6) * fps) + 2):
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    if len(frames) < 8:
        return []
    bg = np.median(np.stack(frames[:: max(1, len(frames) // 12)])
                   .astype(np.float32), axis=0)
    ax, ay = contact_vis
    bx, by = reappear_vis
    cux, cuy = bx - ax, by - ay
    seg = math.hypot(cux, cuy)
    if seg < 1e-6:
        return []
    cux, cuy = cux / seg, cuy / seg
    dirs = []
    for i, fr in enumerate(frames):
        t = t0 - 0.8 + i / fps
        if not (t0 <= t <= t1):
            continue
        moved = np.linalg.norm(fr.astype(np.float32) - bg, axis=2)
        mask = (moved > 14).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if len(c) < 5 or cv2.contourArea(c) < 0.3 * math.pi * r_vis ** 2:
                continue
            (x, y), (MA, ma), ang = cv2.fitEllipse(c)
            major, minor = max(MA, ma), min(MA, ma)
            if major < 2.2 * r_vis or major / max(minor, 1) < 1.6:
                continue                      # round blob: not a streak
            along = (x - ax) * cux + (y - ay) * cuy
            perp = abs(-(y - ay) * cux + (x - ax) * cuy)
            if not (-r_vis <= along <= seg + 2 * r_vis) or perp > 3 * r_vis:
                continue
            cm = np.zeros(mask.shape, np.uint8)
            cv2.drawContours(cm, [c], -1, 255, -1)
            mean = np.array(cv2.mean(fr, mask=cm)[:3], dtype=float)
            if float(np.linalg.norm(mean - colour_ref)) > 150.0:
                continue
            # ellipse angle: OpenCV's is the MINOR-axis bearing from
            # vertical; the major axis direction in image coords:
            th = math.radians(ang + 90.0)
            dx, dy = math.cos(th), math.sin(th)
            if dx * cux + dy * cuy < 0:       # resolve the 180-deg ambiguity
                dx, dy = -dx, -dy
            dirs.append((dx, dy))
    return dirs


def repass_shot(video, start, end, ball=None):
    """Recover the gap for one shot and return the fitted verdict."""
    from billiards_trainer.vision import miss_tags as MT
    from billiards_trainer.vision.analysis_cache import SidecarReader
    from billiards_trainer.vision.tablespace import space_for_video
    from billiards_trainer.vision.trajectory import fit_shot

    video = Path(video)
    reader = SidecarReader(video)
    space = space_for_video(video, reader)
    doc = json.loads((Path(str(video) + ".shots.json")).read_text())
    hinv = np.asarray(doc["transform"]["hinv"], dtype=float)
    H = np.linalg.inv(hinv)

    def to_vis(x, y):
        v = hinv @ np.array([x, y, 1.0])
        return (v[0] / v[2], v[1] / v[2])

    def to_rect(x, y):
        v = H @ np.array([x, y, 1.0])
        return (v[0] / v[2], v[1] / v[2])

    lo, hi = max(0.0, start - 1.2), end + 0.5
    # target = first mover (same rule as the tagger) unless forced
    tgt, tpath, tidx = None, None, None
    nums = [ball] if ball else range(1, 16)
    for num in nums:
        pth = MT._track_path(reader, num, lo, hi)
        i = MT._first_motion(pth)
        if i is None or i + 3 >= len(pth):
            continue
        if tgt is None or pth[i][0] < tpath[tidx][0]:
            tgt, tpath, tidx = num, pth, i
    if tgt is None:
        return {"ok": False, "why": "no moving object ball in the record"}
    cidx = max(0, tidx - 1)
    contact_t, ox, oy = tpath[cidx][0], tpath[cidx][1], tpath[cidx][2]
    gap_t0, gap_t1 = contact_t, tpath[tidx][0]
    contact_vis = to_vis(ox, oy)
    reappear_vis = to_vis(tpath[tidx][1], tpath[tidx][2])
    r_vis = float(space.ball_r_px) * 1.15   # visual radius approx

    # the ball's OWN colour, sampled where it rested, one frame pre-strike
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, contact_t - 0.3) * 1000.0)
    ok, fr = cap.read()
    cap.release()
    if not ok:
        return {"ok": False, "why": "video unreadable"}
    cx, cy = int(contact_vis[0]), int(contact_vis[1])
    rr = int(0.7 * r_vis)
    crop = fr[max(0, cy - rr):cy + rr, max(0, cx - rr):cx + rr]
    if crop.size < 30:
        return {"ok": False, "why": "reference crop empty"}
    px = crop.reshape(-1, 3).astype(np.float32)
    keep = px[px.mean(1) <= np.percentile(px.mean(1), 75)]
    colour_ref = np.median(keep, axis=0)

    rec = corridor_observations(video, None, contact_vis, reappear_vis,
                                gap_t0, gap_t1, colour_ref, r_vis)
    merged = ([(contact_t, ox, oy)]
              + [(t,) + to_rect(x, y) for (t, x, y) in rec]
              + list(tpath[tidx:]))
    merged.sort(key=lambda q: q[0])
    fit = fit_shot(merged, space, space.ball_r_px)
    basis = "fit"
    if fit is None or fit.residual > 1.5 * space.ball_r_px:
        # PREFIX FALLBACK (the backlog's dominant case): the path is
        # densely observed but bent by rattle/aftermath — the first leg
        # alone is still clean. Tighter residual bar than the global fit.
        from billiards_trainer.vision.trajectory import fit_first_leg
        fl = fit_first_leg(merged, space.ball_r_px)
        if fl is not None and fl[2] >= 5:
            class _LegFit:
                departure = (fl[0], fl[1])
                residual = fl[3]
                rail = None
            fit = _LegFit()
            basis = f"first-leg x{fl[2]}"
    if basis == "fit" and (fit is None
                           or fit.residual > 1.5 * space.ball_r_px):
        # STREAK FALLBACK: when positions are too sparse or noisy for a
        # trusted fit, the smears themselves can corroborate the chord.
        # Two independent streaks each within 15 degrees of the chord =
        # the ball demonstrably travelled the chord line.
        dirs = streak_directions(video, gap_t0, gap_t1, contact_vis,
                                 reappear_vis, colour_ref, r_vis)
        cvx, cvy = reappear_vis[0] - contact_vis[0], reappear_vis[1] - contact_vis[1]
        n = math.hypot(cvx, cvy) or 1.0
        cvx, cvy = cvx / n, cvy / n
        agree = [d for d in dirs
                 if d[0] * cvx + d[1] * cvy >= math.cos(math.radians(15.0))]
        if len(agree) < 2:
            return {"ok": False, "why": "fit not trustworthy",
                    "recovered": len(rec), "streaks": len(agree),
                    "residual": None if fit is None else round(fit.residual, 1)}
        # departure = the chord, in RECT space, validated by the streaks
        r0 = to_rect(*contact_vis)
        r1 = to_rect(*reappear_vis)
        class _ChordFit:
            departure = MT._unit(r1[0] - r0[0], r1[1] - r0[1])
            residual = -1.0
            rail = None
        fit = _ChordFit()
        basis = f"streaks x{len(agree)}"
    ux, uy = fit.departure
    best = None
    for name, px_, py_ in space.pockets():
        wx, wy = MT._unit(px_ - ox, py_ - oy)
        if wx * ux + wy * uy <= 0:
            continue
        off = abs(MT._signed_angle(ux, uy, wx, wy))
        if best is None or off < best[0]:
            best = (off, name, px_, py_)
    if best is None or best[0] > 45.0:
        return {"ok": False, "why": "no pocket ahead", "recovered": len(rec)}
    off, pname, px_, py_ = best
    side = "right" if MT._cross(ux, uy, px_ - ox, py_ - oy) < 0 else "left"
    # CUT, same construction as the tagger: cue address from the sidecar,
    # contact point one diameter back along the fitted departure
    cut = None
    cue = MT._track_path(reader, 0, lo, contact_t - 0.05)
    if len(cue) >= 3:
        ax = sum(q[1] for q in cue[:5]) / min(5, len(cue))
        ay = sum(q[2] for q in cue[:5]) / min(5, len(cue))
        d2 = 2.0 * float(space.ball_r_px)
        cxh, cyh = ox - d2 * ux, oy - d2 * uy
        cux, cuy = MT._unit(cxh - ax, cyh - ay)
        if (cux, cuy) != (0.0, 0.0) and math.hypot(cxh - ax, cyh - ay) >= 3 * d2:
            ang = MT._signed_angle(cux, cuy, ux, uy)
            if abs(ang) <= 88.0:
                cut = ("straight" if abs(MT._signed_angle(
                    cux, cuy, *MT._unit(px_ - ox, py_ - oy))) < MT.STRAIGHT_DEG
                    else ("left" if ang < 0 else "right"))
    return {"ok": True, "ball": tgt, "pocket": pname, "side": side,
            "cut": cut, "rail": fit.rail, "recovered": len(rec),
            "basis": basis, "residual": round(fit.residual, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--ball", type=int, default=None)
    a = ap.parse_args()
    end = a.end if a.end is not None else a.start + 8.0
    print(json.dumps(repass_shot(a.video, a.start, end, a.ball), indent=1))


if __name__ == "__main__":
    main()
