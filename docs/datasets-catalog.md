# Dataset catalog — Roboflow Universe scour

Search of Roboflow Universe for pool / billiards / snooker / table datasets, then
**downloaded + verified** the top candidates. Every dataset was inspected with
actual box statistics (boxes/image, box-size distribution) + a visual GT overlay
— because dataset *descriptions* mislead (the original "Pool V2" looked like a
ball model but is close-up ball photos with garbage class names). All downloads
went to `_eval/datasets/<name>/` (gitignored, 747 MB total) via
`tools/fetch_datasets.py`. **All verified datasets are CC BY 4.0** (free to use
with attribution; Joe doesn't distribute, so this is fine).

How to read "verdict": a *usable wide-view ball dataset* has many small,
ball-sized boxes per image (a real table has 6–16 balls). A dataset with ~1 huge
box/image is close-up crops or whole-table boxes — useless for training a
ball detector for Joe's wide camera.

## Downloaded + verified

| dataset (slug) | imgs | boxes/img | median box | ball-sized | classes | verdict |
|---|---|---|---|---|---|---|
| **nxera/snooker-pocket-and-ball-detection** | **17,103** | 11.0 | 0.02×0.04 | 100% | 9 (snooker colours + **pocket**) | ✅✅ **best volume** — real wide snooker tables, balls + pockets. Domain gap: snooker ≠ pool (balls/felt differ). |
| **billiard-acbcc/pool-billiard-nwmsh** | 488 | 3.0 | 0.03×0.05 | 100% | 12 (Cue_Ball, Eight, One–Nine, Object_Ball, Break) | ✅✅ **closest to Joe's setup** — angled pool table, scattered balls, per-ball classes. Small but on-domain. |
| **ipool/billiard-pool-wpb3z** | 746 | 7.3 | 0.03×0.05 | 87% | 12 (numbered balls + white + rack) | ✅ real multi-ball pool, ball-sized boxes, numbered. |
| **ben-gann-lscqy/pool-ball-detection** | 3,289 | 3.6 | 0.18×0.19 | 8% | 16 (`0`–`15`) | ⚠️ largest pool set + numbered, but **zoomed racks** (median box 18% of frame), not wide-table. Useful augmentation, not the primary. |
| **skylep/8-ball-pool-fmk6g** | 26 | 14.7 | 0.02×0.03 | 100% | 17 (solid/stripe per colour) | ✅ genuine full-table (14.7 balls/img!) but **tiny** (26 imgs). Good eval sample, too small to train on. |
| **snooker-ball-detection/snooker-ball-detector-gzeiw** | 802 | — | — | — | 8 (snooker colours) | ⚠️ sampled test labels were **empty** — verify label coverage before use. |
| billiards-tracking-ai/billiards-ai-1svbr | 169 | 1.0 | 0.57×0.50 | 0% | 6 (ball, table, cue_ball, cue_stick, pocket, 8Ball) | ❌ close-ups / huge boxes (1 box/img) — not wide-table ball GT. (Class list is nice in theory.) |
| pool-table/pool-v2 (from prior turn) | 751 | 1.2 | 0.47×0.55 | 3% | 4 (corrupted names) | ❌ close-up single-ball photos, garbage classes (documented in baseline.md). |

Failed to download: `home-tagdk/pool-table-detection-zhjd9` ("version 1 not found"
— wrong default version; retry with an explicit `--version`).

## Found in search, not downloaded (long tail)

Pool: `project-d6t5z/8-ball-pool`, `mark-dj0yk/pool-balls-detection-srlqi` (131),
`siv-poolbot/pool_bot_v2` (tagged balls — robotics, niche),
`new-workspace-va9vn/balls-detection`, `billiard-ball/billiard-ball`,
`nidacorian-protonmail-com/pool-billiard`, `pooltafel/pooltable-balls`,
`billardtabledetector/pool-table-cyrrm` (600, table-only).
Snooker: `jan-nhhmt/snooker-l8nrg` (50), `morganlewis/snooker-vision-tjn0z`,
`snooker-znuzc/snooker-zmtks` (922, 8 colours), `objectdetection-3dexx/snooker-4acib`,
`snooker-yuabl/snooker-balls-hfcp3`.
(Roboflow also reports a `billiards` workspace hub and `class:ball` search with
many more.)

## Recommendations (ranked for our build)

1. **`pool-billiard-nwmsh`** — train/fine-tune target #1: real angled *pool*
   tables with per-ball classes, the closest match to Joe's camera. Small (488),
   so combine with augmentation or the others.
2. **`snooker-pocket-and-ball-detection`** — by far the most data (17k) and it
   labels **pockets** too (we have no pocket geometry). Great for pretraining a
   ball+pocket detector; expect a snooker→pool domain shift to fine-tune away.
3. **`billiard-pool-wpb3z`** + **`pool-ball-detection`** — extra pool volume
   (numbered balls) to mix in; the latter is zoomed-rack framing.
4. **`8-ball-pool-fmk6g`** (26 imgs) — too small to train, but a clean **wide
   full-table eval sample** to add to the harness later.
5. **Hard negatives / empty tables / other angles** — not well covered by these;
   the `*-table-*` sets (table-only boxes) could seed empty-table FP tests. Joe's
   own idle clips already serve the empty/FP role.

**Caveat (the honest part):** these are mostly broadcast / standard-angle pool &
snooker, plus some zoomed racks — **none is Joe's specific low side-angle, blue-felt
room setup.** They're a real step up from "nothing" (and from Pool V2), but a
detector trained purely on them may still need fine-tuning on Joe's own footage
(Capture-for-analysis flow) to hit his conditions. Snooker is a related-but-distinct
domain (different ball set, green felt, larger table).

## License + attribution

All verified datasets are **CC BY 4.0** (recorded from each `data.yaml`'s
`roboflow.license`). Required attribution if any derived model/output is shared:
credit each project (e.g. "Snooker pocket and ball detection by Nxera, Roboflow
Universe"; "Pool Billiard by billiard, Roboflow Universe"). URLs:
`https://universe.roboflow.com/<slug>` for each slug above.

## Provenance / security

Downloaded with `tools/fetch_datasets.py` using Joe's Roboflow API key passed via
`--api-key` (downloads only — hosted-inference trial credit untouched). The key
is **never** logged, printed, or written to any tracked file; datasets +
`_eval/` are gitignored. This catalog (committed) contains no secrets.
