"""Mine the replay log for the failure modes Joe described:
jitter, missed analysis, spotty detection, latchy tracking."""
import json

OUT = "/private/tmp/claude-501/-Users-joe-Documents-GitHub--CaulfieldEngineering-BilliardsTrainer/d52d3d24-6c8e-4160-8774-20b9bffe9ca7/scratchpad/review"
rows = [json.loads(l) for l in open(f"{OUT}/replay.jsonl")]
n = len(rows)
print(f"frames: {n}")

# ---- motion bursts (independent signal) ----
MOTION_ON = 1.6      # mean abs diff threshold; tuned by eye on the histogram
bursts = []
in_b = False
for r in rows:
    if r["motion"] > MOTION_ON and not in_b:
        in_b = True
        bursts.append([r["i"], r["i"]])
    elif r["motion"] <= MOTION_ON * 0.6 and in_b:
        in_b = False
    if in_b:
        bursts[-1][1] = r["i"]
bursts = [b for b in bursts if b[1] - b[0] >= 3]
print(f"motion bursts (>=3 frames): {len(bursts)}")

# ---- detection stability ----
import statistics

ndets = [r["ndet"] for r in rows]
quiet = [r["ndet"] for r in rows if r["motion"] < 0.8]
print(f"ndet: mean {statistics.mean(ndets):.1f}  quiet-frame mean {statistics.mean(quiet):.1f} "
      f"stdev {statistics.pstdev(quiet):.2f}")
# dropouts: quiet frames where ndet falls >=3 below the rolling median
drops = []
window = [r["ndet"] for r in rows[:60]]
for k, r in enumerate(rows):
    if k >= 60:
        window.pop(0); window.append(r["ndet"])
    med = sorted(window)[len(window) // 2]
    if r["motion"] < 0.8 and med - r["ndet"] >= 3:
        drops.append((r["i"], r["ndet"], med))
print(f"quiet-frame detection dropouts (>=3 under rolling median): {len(drops)}")
for d in drops[:8]:
    print("   frame", d)

# ---- tracking: lag, teleports, id churn ----
# build per-id position history
hist = {}
for r in rows:
    for tid, x, y, num, spd, misses in r["tracks"]:
        hist.setdefault(tid, []).append((r["i"], x, y, num, spd, misses))
print(f"distinct track ids: {len(hist)} (physical balls ~16)")
short = sum(1 for v in hist.values() if len(v) < 90)
print(f"  short-lived ids (<5s): {short}")

teleports = []
lag_events = []
numchanges = []
for tid, seq in hist.items():
    prev = None
    nums = set()
    for (i, x, y, num, spd, misses) in seq:
        if num >= 0:
            nums.add(num)
        if prev is not None and i - prev[0] == 1:
            d = ((x - prev[1]) ** 2 + (y - prev[2]) ** 2) ** 0.5
            if d > 70:
                teleports.append((i, tid, round(d)))
        prev = (i, x, y)
    if len(nums) > 1:
        numchanges.append((tid, sorted(nums)))
print(f"single-frame teleports (>70px): {len(teleports)}")
for t in teleports[:10]:
    print("   frame", t)
print(f"tracks that changed number: {len(numchanges)}")
for t in numchanges[:10]:
    print("   ", t)

# ---- motion-start -> track-response lag per burst ----
lags = []
for b0, b1 in bursts:
    resp = None
    for r in rows[b0:min(b1 + 40, n)]:
        if any(t[4] > 6.0 for t in r["tracks"]):
            resp = r["i"]
            break
    lags.append((b0, b1, (resp - b0) if resp is not None else -1))
no_resp = [l for l in lags if l[2] < 0]
resp_l = [l[2] for l in lags if l[2] >= 0]
if resp_l:
    print(f"burst->track-response lag frames: median {sorted(resp_l)[len(resp_l)//2]} "
          f"max {max(resp_l)}   bursts with NO track response: {len(no_resp)}/{len(bursts)}")
print("slowest responses:", sorted(lags, key=lambda l: -l[2])[:6])

# ---- shot events ----
shots = [(r["i"], r.get("shot")) for r in rows if "shot" in r]
print(f"shot events: {len(shots)}")
for s in shots:
    print("   ", s)

# flagged windows for visual review: worst lags + biggest teleports + dropouts
flags = set()
for l in sorted(lags, key=lambda l: -l[2])[:4]:
    flags.add(("lag", l[0]))
for t in sorted(teleports, key=lambda t: -t[2])[:4]:
    flags.add(("teleport", t[0]))
for d in drops[:3]:
    flags.add(("dropout", d[0]))
for b, _s in shots[:4]:
    flags.add(("shot", b))
print("FLAGGED:", sorted(flags, key=lambda f: f[1]))
json.dump(sorted(flags, key=lambda f: f[1]), open(f"{OUT}/flags.json", "w"))
