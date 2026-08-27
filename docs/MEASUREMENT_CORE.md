# MEASUREMENT CORE — Plan of Record

Status: PLAN OF RECORD for the measurement-engine respin. Companion to `docs/VISION_ROADMAP.md` and `docs/GOALS.md`. (Written to `C:\Users\Joe\Documents\GitHub\BilliardsTrainer\docs\MEASUREMENT_CORE.md`.)

---

## 0. END-STATE AND THE RELIABILITY LADDER (Joe, 2026-08-27)

The long-term destination is a ground-up rebuild in **C++**: a rock-solid
sandbox of real-time data collection with no UI, no layout, no
interpretation - testable from the mobile app - so no orphaned code
lingers in the foundation. The Python measurement core built here is the
**proving ground**: every rule validated in Python becomes the C++ spec;
nothing unproven gets ported.

Joe's reliability ladder - expected solid in Python BEFORE the rewrite:

1. **Table detection** - achieved.
2. **Pocket detection** - HALF-ADDRESSED: pockets are placed geometrically
   from the calibrated rectangle (canonical marks + pocket_radius_frac,
   `vision/calibration.py:139-157`); drop events are
   detection-by-disappearance near those regions. No visual localization
   of the actual pocket mouths; the calibration comment itself notes
   "pockets off their marks". A visual pocket locator is ladder work.
3. **Cue ball detection + tracking, real-time trails** - the core of this
   respin (M1's dense tracker; the takeoff-blur coast).
4. **Shot detection** - ~95 percent recall today; rides on 3.
5. **Shot clock** - live, cue-ball driven; proves out 4.
6. **Object ball detection + tracking, real-time trails** - same engine
   as 3, multi-target.

Milestones M1-M4 below serve the ladder; the C++ rewrite starts only
when the ladder holds in Python.

## 1. WHY

The owner's directive is explicit: "rock solid measurements and tracking before we even think about interpreting misses vs makes." The headroom is already measured: GPU inference runs 94 fps on DirectML vs 20.7 fps CPU, yet live throughput only moved 4.1 → 14.6 results/s when the provider flipped — the remaining 6x gap lives entirely in our own plumbing (the 33 ms Qt tick, the 1-slot drop queue, the serial ensemble, the queued-signal ingest), not in the model. Measuring ball positions on **every** frame at 30 fps during play eliminates the 1.5–2 s takeoff blur blindness, gives real per-frame velocities instead of interpolated fiction between 10 Hz samples, holds identity through flight instead of patching it back afterwards, and makes the sidecar a true record of the table rather than a 100 ms-quantized sketch.

---

## 2. KEEP — the IP inventory

Everything below was bought with a real incident. The new engine re-implements the **rule**, not necessarily the mechanism, and each rule stays protected by the metric or proof named with it. Deduplicated across the five path surveys.

### 2.1 Capture and recording (the chain that must never be disturbed)

| Rule / contract | Where | Protecting metric / proof |
|---|---|---|
| Device-owned capture: ffmpeg owns the exclusive DirectShow device; the recording path NEVER passes through Python. Any per-frame consumer is a pure pipe-drain that cannot back-pressure ffmpeg. | `capture/ffmpeg_source.py:1-28` | Rig proof: analysis reader stalled 1.2 s every 3 s → recording byte-identical (900 frames / 30.000 s). Re-run this exact test for the new consumer. |
| Drain thread always empties the analysis pipe; teardown keeps reading until ffmpeg exits (else ffmpeg blocks mid-write and never finalises the recording). | `ffmpeg_source.py:149-166, 199-232` | Recording finalisation (no truncated tails). |
| Never attach a fresh clock to a frame of unknown age — timestamp at capture, not at processing; drop the old process's last frame across a restart. | `ffmpeg_source.py:188-197` (ONE-CLOCK incident, 2026-08-25) | Legacy skew corrected by `video_time_offset()` (`vision/analysis_cache.py:312-330`); new sessions must measure offset = 0. |
| Graceful ffmpeg shutdown: write `q`, close stdin, wait 15 s, only then terminate (Windows terminate is a hard kill). | `ffmpeg_source.py:208-227` | Measured: hard kill truncated a 20 s capture to 16.1 s. |
| ThreadedCameraSource: dedicated grab thread + take-semantics `read()` (each frame once, `None` when nothing new); caps cached before the grab thread starts. | `capture/camera.py:120-146, 150-160, 263-266` | Latency pinned at ~1 frame (driver-queue backlog incident). |
| Stale-read dedupe on the recording sink (`_is_fresh`, stride-32 signature, 0.5 s keepalive): the new engine's every-frame target counts UNIQUE captures, not deliveries. | `camera.py:192-221`, commit 58bfd24 | Measured: 40% stale reads at 50 polls/s; 17.4% duplicate written frames → 0.44% after dedupe. |
| Synchronous sink detach under `_sink_lock`; sink only enqueues; sink exceptions never kill the grab loop; signature reset so a new recording never skips its first frame. | `camera.py:176-190, 227-241`, commit 084710b | Detach race class. |
| CFR retiming from the frame's OWN capture stamp, never a wall-clock pacer (slot = round((ts−t0)·fps); ALPHA=0.05, drift clamp 3 slots, gap resync >2.0 s). | `capture/videowriter.py:158-224`, `workers/controller.py:1507-1524`, commit 8e274d3 | Recording jitter (dup/drop beat between two clocks). |
| Grab thread owns and releases the capture handle in its `finally` (join 2.0 s on a wedged driver). | `camera.py:243-252, 270-276` | Exactly-once release. |
| Empty-read tolerance is WALL-TIME, not ticks: 2.5 s live, 8.0 s pre-first-frame. | `controller.py:38-46, 1051-1069` | Covers driver auto-exposure init and (marginally) the device reopen window. |
| MJPEG FOURCC request on the OpenCV path (dongles renegotiate raw formats; stale YUV = chroma garbage). | `camera.py:62-71` | — |
| FULL-resolution analysis stream, never downscaled (balls ~27 px in 1080-wide, detector floor ~9 px). | `ffmpeg_source.py:44-47` | Detector recall floor. |
| `hqdn3d` BEFORE fps decimation in the analysis filter chain; recording untouched. | `ffmpeg_source.py:120-123` | A/B in `tools/exp_hqdn3d_ab.py`; +35% frame detail (laplacian 48.5 → 65.4). |
| Preprocess (rotate/flip/colour) exactly once, at ingest, LIVE only; recordings saved post-preprocess; playback never re-applies. | `controller.py:1304-1311`, `capture/preprocess.py:1-9` | Double-rotate class. |
| `setsar=1` on device recordings; even-dimension enforcement everywhere yuv420p flows (AMF error 29 → 0-byte file); encoders PROBED never assumed, encoder stderr tail kept. | `controller.py:768-780`, `videowriter.py:57-88, 257-283` | Silent-empty-recording class. |
| Fragmented mp4 while in progress + orphan `.part` recovery ≥2 MB; `.part` staged hidden OUTSIDE the synced folder, atomic `replace()` on finalize. | `ffmpeg_source.py:113-115`, `videowriter.py:294-301`, `controller.py:594-601, 1097-1125` | SSD-panic and Dropbox ghost-mp4 incidents. |
| Camera resolved by persisted friendly NAME before index; passive enumeration on macOS. | `capture/devices.py:96-121`, `controller.py:200-207` | USB index reshuffle. |
| Recording teardown order: detach sink → sentinel+join rec worker → release writer, under `_rec_lock`; a recording is never EMPTY (`release()` flushes the last frame). | `controller.py:681-699`, `videowriter.py:352-360` | — |
| 480p-fallback watchdog discriminated by ACTIVE-BOX ASPECT >1.85 (healthy 1.50–1.78; degraded 2.29, 1.95) — height-fill was the wrong discriminator. | `controller.py:1371-1405`, data in `docs/feedmeter-log.csv` | Degraded-recording detection. |

### 2.2 Inference and worker discipline

| Rule / contract | Where | Protecting metric / proof |
|---|---|---|
| Display path NEVER runs inference; a full queue means the frame is display-only, dropped silently, never blocked. | `controller.py:1096-1102, 1225-1247` | 9 fps playback incident; display fps in the heartbeat. |
| ALL tracker/pipeline mutation on ONE thread (queued signal → `ingest_raw_detections`, "the only mutator"); stale-result guards drop results when stopped / detection off / source is video. | `controller.py:163-167, 1287-1299`, `pipeline.py:683-697` | Cross-thread mutation class. |
| Self-healing workers: any submit resurrects a dead worker with a WARN; clean exit on RuntimeError from a deleted signal source; a bad frame / detector failure / bad shot never kills its loop. | `controller.py:1234-1243, 1276-1284, 1136-1143`, `pipeline.py:1043-1046` | The empty-schematic evening (Joe found it, not the loop). |
| Per-link countable heartbeat every 150 ticks: `submit/blocked/worker-alive/raw/ingest/pub`. The new engine must keep every chain link countable. | `controller.py:1408-1428` | "Which link is dead" is a read, not an investigation. |
| Schematic cache invalidation on ingest. | `pipeline.py:697-706` | Frozen-empty bird's-eye (reported three times). |
| `strategy.detect()` serialized by `_detect_lock` — worker and UI-thread Ball ID Trainer share the singleton. | `onnx_model.py:57-60, 212-213` | — |
| Table-crop before letterbox (playing area 58%×73% of frame; 57% of inference was carpet; 16.0 → ~23 px/ball vs ~9 px floor). | `detector_strategies/__init__.py:69-95` | Class-level removal of hand/cue false positives. |
| Two-pass tiled inference (top/bottom 60%): +20% more balls; never re-add the "overhead camera skips rescan" shortcut. | `onnx_model.py:61-64, 215-250`, `pipeline.py:159-168` | 2026-07-23: two top-rail balls at 0.83 conf existed ONLY in the rescan. |
| Seam-truncation drop + 1.25×larger-radius tile-dedupe merge. | `onnx_model.py:169-205` | `overlapping_balls` violations at the tile seam; phantom-track ping-pong. |
| CENTERED letterbox matching Ultralytics; padding-band-only fill. | `onnx_model.py:122-137` | Rack recall. |
| Provider order DML > CUDA > CoreML > CPU, never raise; LOUD CPU-fallback warning ("the real-time killer"); rebuild session only on recorded pref change. | `onnx_model.py:39-49, 85-89, 102-108` | Silent-CPU class. |
| Typed-ctypes Win32 priority calls with checked returns (bare `windll` mangled the 64-bit pseudo-handle; demote was a silent no-op). | `controller.py:1249-1270` | Measured 2026-08-23. |
| `sample_colour` on EVERY found ball, unconditionally — `measured_bgr` is the sole feed of the colour-evidence subsystem; carried across raw→rect projection; never build consensus on the palette constant (`Detection.bgr`), that echoes the classifier. | `ensemble.py:61-68, 177-210`, `pipeline.py:806-822`, `core/types.py:70-76` | 005048 @233: 0/6 detections carried colour, colour veto could not fire. |
| Empty identity pass must NOT skip the naming/correction stage. | `ensemble.py:82-84` | — |
| `FramePacket.raw_dets` stays CAMERA-coordinate for the in-app labeller. | `controller.py:62`, `pipeline.py:1035-1047` | Training mode at any cadence. |

### 2.3 Tracker identity invariants (the crown jewels — re-derive, do not delete)

Behaviour is pinned by `tests/test_tracking.py` (37 tests, each an incident-as-spec). The new tracker must pass the **intent** of every one, rewritten for its own internals.

| Rule (mechanism-independent) | Where today | Protecting metric |
|---|---|---|
| REST-FROZEN IDENTITY: identity may only change on frames the engine itself judges to be motion; motion judgment must tolerate ≤2 glare-blip steps and be derived from the engine's own state, not consecutive detector hits. | `tracking.py:62-73, 655-668` | `id_flicker`/`class_flicker` (`eval/invariants.py:66, 419-431`, rest = 0.12 diam/frame); `id_flips`/track-min. Corpus incident: "changed #5 → #9 while at rest." |
| DEFERRED FIRST COMMIT + HYSTERESIS: first identity needs 3 agreeing reads; a challenger needs committed+5 votes AND motion; invisible in UI because min_hits=5. | `tracking.py:114-142` | `id_flips` 2.4 → ~2.1/track-min (2026-07-02 autotune). |
| GLOBAL EXCLUSIVE NUMBER ASSIGNMENT each frame; at-rest commitments are hard constraints ranked by claim strength; resting losers that published show unknown, never renamed. | `identity.py:78-269`, `pipeline.py:440-457` | `duplicate_number`, `multiple_cue_balls`, `dup_numbers`. |
| REACHABILITY GATE: a number moves ≤0.35 short-sides/frame within a 12-frame window of its last published position; longer gaps are table-wide. | `identity.py:188-198, 247-263` | `id_hops` 2.41 → ~1/1k states; tripwire GREEN <2, RED >4 /1k (`tools/health_check.py:274-298`, sidecar-based — transfers to the new engine unchanged). |
| CUE-SIZE FLOOR: number 0 demands lifetime r_max ≥ 0.9× median committed radius (cue min observed 12.2 vs felt-speck 10.0; a general radius floor is WRONG — real balls live at r9–11.5). Rebind radius sanity 0.7–1.5×; revival never accepts 20% shrink. | `identity.py:134-151, 219-227`, `tracking.py:334-352, 377-384` | The 560 px speck-snap that masked a scratch (005647 @209 s). |
| APPEARANCE GATING in association: cue-vs-object contradiction penalty (0.30×short_side); a COASTING numbered track never takes a cross-category detection; healthy tracks exempt. | `tracking.py:276-341, 373-376` | 005647 forensics: 2 px stop-shot inversion; the 7 booking the cue's pocket entry. |
| MEASURED-COLOUR VETO for touching balls: settled track with ≥8 samples refuses detections >150² BGR-dist² from its own MEASURED median (measured-to-measured only). | `identity.py:271-297` | 005048 @233: own detections 1–14 vs arriving cue 108,820. |
| Colour ADOPTION strictly gated: solids 1–7 only, consensus ≥0.8 over ≥25 samples, at rest, hits ≥60, number free. | `identity.py:160-174` | Names the digit-down 6 without rack churn. |
| NUMBER RELEASE separated from TRACK KILL (release at 10 vacant detect frames ~1 s; kill at 60 was-named / 8 never-named); spot-occupancy asks the raw pixels; `_covered` is a neighbourhood test, not centre-pixel. | `tracking.py:199-218`, `pipeline.py:539-645, 551-592` | Phantom departures vs lost paths (005048 @233: 7 s shot vs 6 s kill patience; shot-31 phantom scratch). |
| NUMBER MIGRATION off the coasting ghost: single confirmed anonymous mover, born at that spot, ≤25 frames old, within 6 radii, moving — every clause load-bearing. | `identity.py:36-76` | 005647 @386 + 9 shots; 0% cue flight coverage on 10/36 shots. |
| REST-LINKING + FLIGHT-LINKING death registry: exactly ONE free death + exactly ONE orphan → inherit; ANY ambiguity → skip ("wrong links are worse than anonymous balls"). | `tracking.py:478-499, 518-571` | Shot-36; median 73% cue numbered-coverage. |
| BLUR RECOVERY NEVER NAMES: recovered blob is unnumbered, carries `recovered_for`, still competes on distance; NOT-THE-HOLE-IT-LEFT, not-bare-felt, relative colour contest, size cap 3.2 expected radii, median background not frame-diff. | `blur_recovery.py:30-34, 149-158, 200-259`, `tracking.py:311-317` | The track pinned to its own absence class. |
| FOREIGN-HAND INGESTION FILTER: detections inside a kept (blob-floored, `_MIN_BLOB_FRAC=0.004`) foreign blob dropped before the tracker; a ball merged into the hand blob coasts on the occlusion budget — correct. | `pipeline.py:373-395`, `foreign.py:26-30` | Gloved-hand incident: 25 impossible overlaps, sole per-session G4 blocker. |
| Dup merge at 0.8 ball diameters preferring the detection-backed track; occlusion-budget ASYMMETRY: settled+identified vanished = occluded (1800), moving vanished = pocketed (24), unnumbered denied the long budget; jaw-ghost = short budget at pocket mouths; in-pocket resters deleted outright. | `tracking.py:431-476, 503-517` | 2.4% sliver duplicate; 40 s impossible-overlap basket pair. |
| CUSHION-AWARE COASTING with pocket-mouth exception (no reflection within 1.6×pocket_r — sail in and age out); revival SNAPS, zeroes velocity, trims history to last 3 votes. | `tracking.py:240-268, 356-410` | Thin-air rebound / velocity-whip classes. |
| ONE CUE PER PUBLISH (ranked claimants, losers render UNKNOWN); class is number-derived, `committed_cls` remembered; class never changes at rest. | `tracking.py:76-106, 686-703` | "Two rendered cues", "solid → cue at rest". |

### 2.4 Sidecar, close pass, and truth handling

| Rule / contract | Where | Protecting metric / proof |
|---|---|---|
| Writer t0 rebase: first written state defines t=0; the reader's legacy guard re-normalizes any file whose first state is >30.0 s — the new engine MUST keep its first state within ~2 s of zero. | `analysis_cache.py:70-82, 273-283` | Evening-session t=1160 incident. |
| Append-only log, ranked verdicts: `review` is FINAL > `forensic` > `derived`; last-wins within rank; legacy no-src correction = review, no-src action = derived. | `analysis_cache.py:43-58, 181-215, 259-271` | Shot-5 clobbered human call. |
| `_shot_for` attachment cascade: exact <0.2 s → containment ±1.0 s → nearest ≤8 s. Re-segmentation by a better tracker MUST NOT orphan review verdicts. | `analysis_cache.py:128-151` | @214/@466 verdicts orphaned by a 3.1 s shift; scoreboard blind to 12/13 verdicts. |
| Live writer is single-owner ('w'-mode buffered); mid-session appends marshalled to the controller thread; external appends only post-close. | `analysis_cache.py:108-117`, `controller.py:1213-1222` | Silent overwrite class. |
| `carry_review_verdicts` on `--force` rebuild: machine data is recomputable, Joe's verdicts are not; sidecar copied to `.prev`. | `analysis_cache.py:381-413`, `build_analysis_cache.py:43-52` | Loop-30 overwrite incident. |
| `tracks_at` snap-don't-tween across gaps >1.0 s; `_to_track` synthesizes bgr from number. | `analysis_cache.py:341-343, 359-375` | Invented motion / all-white schematic. |
| `video_time_offset()` + per-shot strike anchor machinery REMAINS, returning 0 for one-clock sessions — old sessions still need it. | `analysis_cache.py:312-328`, `shots_export.py:187-198, 246-260`, `corrections_watcher.py:78-86` | 1.8–3.3 s legacy skew; ≥3 high-confidence strokes else 0. |
| Outcomes derived post-close from the identity record, idempotent, human verdict always outranks. | `outcomes.py:1-33, 289-320` | 7/11 live vs 11/11 derived on frame truth (GOALS loops 32-33). |
| Return-mode discrimination: same-spot hands-free = flicker; near-hand or new-spot = potted-and-replaced; tid-continuity = never off the bed; same-number rebirth after pocket-death with a hand in the ≥0.8 s gap = hidden scratch; anonymous departures count as pots. | `outcomes.py:59-264` | 005647 forensics (phantom makes/scratches). |
| Exporter attributes each sample to the number held AT THAT MOMENT, never the final number; one trail per BALL (same-number segments merged); trail window opens at the TRUE strike; `_bridge_unnumbered` only when both flanking runs agree. | `shots_export.py:67-99, 108-139, 160-172` | "Drastically wrong line" (005048 @233); double badges; "1 ft of tail." |
| Measured-or-abstained: every absent field carries a stated reason; dense resample ships only when sparse samples geometrically agree (time-free check); badge row never contradicts the description; aim ray stops at the first ball it would hit. | `shots_export.py:222-247, 320-336, 455-470`, `trail_resample.py:149-171`, `miss_tags.py` | "I need this reliable." |
| `run_close_pass` is THE single canonical close sequence for every caller. | `shot_pass.py:1-23` | Three-hand-copied-variants drift. |
| Action classifier: launch-from-hand test; rolled-in-birth (40–400 px/s decelerating = returned from pocket); never classify "nothing" unless long-sampled AND still AND ball set unchanged. | `actions.py:74-213` | 27 frame-verified labels + 11 verified strokes. |
| Shot detector holds TIME floors alongside frame counts (min_shot_s=1.2, settle_s=0.5) — the new 30 fps engine keeps time-based floors. | `events/shot_detector.py:75-89` | 011510 @72.2 s discarded-shot incident. |
| Sidecar written only while `res.status=='tracking'` and a recording is open; shots counted only in-session. | `controller.py:1342-1358` | — |
| v2 hand-context (`c`, `ff`) omitted when absent; v1/absent = UNKNOWN, never "no hands". | `analysis_cache.py:87-93, 291-302` | Gathering-vs-stroke discrimination. |
| Key-order dependency: `type` MUST remain the first key of shot/action/correction records (line-prefix matchers). | `companion/server.py:48-75`, `session_summaries.py:36-63` | Silent zero-shots. |
| Phone correction-file protocol; every verdict triggers `run_close_pass`; rife renders defer while any session mp4 modified <600 s ago. | `corrections_watcher.py:1-150` | Recording owns the GPU. |

### 2.5 Evaluation and promotion (the gate the new engine is judged through)

| Rule / contract | Where | Protecting metric / proof |
|---|---|---|
| THE FOUR MECHANICAL GATES: impossible ≤ champion×1.10 + 0.05; coverage ≥ champion×0.95; zero degenerate sessions; failed ≤ champion's. Exit 0/1. | `tools/score_challenger.py:143-147` | Survived five challenger gates (c2–c7). |
| HELD-OUT verdict via `--exclude` — training sessions dropped from BOTH aggregates. | `score_challenger.py:54-59, 167-179` | c2 memorisation-assisted "tie". |
| SAME-BATCH RULER: champion scored fresh in the same code+environment, never a saved aggregate. | `score_challenger.py:71-99` | Stale-ruler incident (0.27 vs 1.40 on identical code). |
| Mechanical pass is NECESSARY, NOT SUFFICIENT: the mission metric (held-out out-of-game rate) must DROP; a rise rejects. | GOALS log 2026-08-18 | c6 passed all four, DECLINED (oog 26.17 → 28.60). |
| Degenerate-detector trap: rates per 1000 ball-frames; coverage <1.0 = degenerate auto-fail. "A scorer you can win by giving up is not a scorer." | `eval/invariants.py:20-36, 137-152` | Do not weaken. |
| Score-the-failure: a session that never locks is DATA (`{failed: true, ...}`), not a gap. Never gate on a piped exit code — verdicts come from written JSON artifacts. | `score_corpus.py:76-84`, `score_session.py:141-144`, `score_challenger.py:127-130` | Swallowed-failure incident. |
| Judge with the pipeline's OWN calibration (expected_ball_radius_px), never frame-derived; all thresholds in BALL DIAMETERS (overlap <0.80d, speed >9.0d/step, rest <0.12d, off-table 1.5d, pocket 2.5d). | `score_session.py:104-121`, `invariants.py:31-68` | Scale independence, no per-camera retune. |
| `out_of_game_number` is implausible, not impossible — identifier misreads stay out of the physics gate. | `invariants.py:226-234` | — |
| Champion fingerprint + rollback-is-a-file-copy (`_eval/champion.json`; `.bak` swap in a `finally`; health_check RED on size mismatch, AMBER while `.bak` exists). | `score_challenger.py:110-125`, `health_check.py:151-163` | — |
| Identity-wander tripwire (hop = number reappearing >8.0 diam away <1.2 s on a different tid) and G1 render audit (zero shared numbers, 0/149,094 states at DONE) run from SIDECARS — they transfer to the new engine's output unchanged. | `audit_identity_wander.py:28-29`, `audit_render.py`, `health_check.py:132-149, 277-300` | GREEN ≤2.0/1k, RED >4.0/1k; any dupe = RED. |
| Audio as INDEPENDENT witness with asymmetric claim windows (−2.5 s/+1.5 s; pre-shot 6.0 s); shot-cadence audits run at STRIDE 1 only. | `score_shots.py`, `audit_shot_recall.py:40-52` | 210621 break/address measurements. |
| Three-set gate for identifier changes: gameplay ground truth + racks + old-era archive. | GOALS log 2026-08-26 | gameplay 100%, racks 83→84, old-era 95→94 accepted. |
| RECORDING OUTRANKS THE GATE: corpus/gate/training run BelowNormal and are killed the moment a `.part` appears; gate wrappers restore the champion deterministically. | `tools/_lowprio.py`, `score_corpus.py:19-23` | Joe's live play is never contended with. |

---

## 3. DEBT — what the respin replaces, with measured cost

1. **No capture timestamps or sequence numbers on analysis frames.** The rawvideo pipe discards PTS (`ffmpeg_source.py:124-125`), the drain stores a bare ndarray (`:160-166`), and the consumer stamps time at PROCESSING (`controller.py:1330`). Cost: frame-to-frame dt is unknown (velocity structurally noisy), and the one-clock bug class is always one oversight away.
2. **Non-destructive `read()` with no freshness signal** (`ffmpeg_source.py:245-247`). A 30 Hz poller against a 30 fps latest-slot silently re-processes duplicates and skips on phase jitter — the recorder measured exactly this: 17.4% duplicate frames until the sink-side dedupe.
3. **The 1-slot drop-when-busy queue + queued-signal ingest** (`controller.py:165, 1244-1246, 1279-1298`). Caps measurement at worker latency, not camera rate: 14.6 of 30 frames/s measured; the other half are lost to measurement forever. This chain is the entire gap between the 94 fps bench and 14.6 live results/s.
4. **Qt QTimer tick as the measurement clock** (33 ms interval, `controller.py:257`), with one `_tick` doing capture poll + preprocess + submit + sidecar + watchdogs + UI packet (`controller.py:1019-1096, 1300-1440`). Cadence rides event-loop jitter and everything sharing the tick.
5. **Detections ingested with unmodeled latency and no timestamp of their own** (`controller.py:1287-1298`): tracks lag reality by inference+queue delay; the sidecar records them against a later t; `view_tracks` extrapolation (`pipeline.py:766-781`) exists only to hide this.
6. **The 10 Hz sidecar write cap** — BOTH gates: `controller.py:1343-1352` and `build_analysis_cache.py:68-74`. Everything downstream measures interpolated fiction between 100 ms samples; the docstring contract "~7-10 Hz" (`analysis_cache.py:23`) is baked into consumer comments.
7. **Serial ensemble, ~68 ms/cycle**: 2–4 sequential `sess.run` (finder every cycle, identifier every 2nd, 2 tiles each at 10.6 ms) + 10–20 ms interleaved CPU colour work + per-inference pre/post — GPU at ~1/3 duty (`ensemble.py:58-77`, `onnx_model.py:242-250`). No tile batching, no overlap, one process-wide `_detect_lock` and one reused canvas preclude two in-flight frames by construction.
8. **BELOW_NORMAL worker demote** (`controller.py:1249-1270`): measurement cadence degrades under desktop load BY DESIGN; the premise (inference starves the compositor) was undercut by the RF-interference diagnosis of input lag.
9. **Rest-freeze instead of motion-model coasting** (`tracking.py:604-610`, lock_dist=0.0115×short): published position stops being a measurement; forces the published-copy re-projection dance (`pipeline.py:648-676`); is what the 1.5–2 s takeoff blindness holds on screen.
10. **Naive coasting physics**: constant-velocity predict, flat 0.92/frame damping, elastic axis-aligned reflection, revival zeroes velocity, gate grows linearly with misses instead of with uncertainty (`tracking.py:59-61, 248-268, 365-367, 412-419`). No friction model, no covariance — why coasting is only trusted a few frames and everything else needs budgets.
11. **Greedy nearest-neighbour association** (`tracking.py:7-9`): the 2 px stop-shot inversion was a greedy artifact; the appearance-gating patch chain exists because of it. Global assignment (motion+appearance jointly) removes the patch chain's reason to exist.
12. **Frame-rate-denominated constants throughout**, tuned for ~10 Hz: `_RELEASE_AFTER=10` ("~1 s") releases numbers 3× too fast at 30 fps; kill 60/8 frames; `vel_alpha 0.42`, `pos_alpha` over px/FRAME speeds, `still_frames 8`, `move_streak` bar, `occluded_budget 1800`, `blur _BUF=15`, classifier speed bands fitted to 10 Hz-sampled peaks (`actions.py:150-166`) — all silently rescale. Re-derive in seconds and diameters/second.
13. **Takeoff blur blindness ~1.5–2 s** (`trail_resample.py:4-8`) and its compensator stack: BlurRecovery (sweep channel OFF after 37 phantoms + coverage 8→2 regression), dense trail resample (PARKED "pending the measurement-core rebuild"), forensic_repass corridor search (~20 min of re-decoded footage, a fresh SidecarReader per miss).
14. **Identity-during-flight special-case chain** (migration, rest/flight-linking, single-cue rebind, second-chance revival): measured residue is the cue anonymous for its ENTIRE flight on 10/36 shots, median numbered coverage 73%. Retire by making them unreachable via every-frame association — never by deleting the invariants they encode.
15. **Blur recovery DEAD on the live path** (`pipeline.py:683-688` ingests without a frame vs the gate at `:509-527`): live play — the case with real motion blur — never benefits.
16. **ANALYSIS_FPS=30 decimation from a 60 fps input inside ffmpeg** (`ffmpeg_source.py:52, 89-91`): half the temporal resolution discarded before Python sees a frame; the fps filter's dup/drop choices invisible.
17. **Record toggle restarts the capture process**: analysis blank for the measured 1.8–3.3 s reopen window, tracker loses all continuity exactly when measurement matters most; `_CAMERA_MISS_SECONDS=2.5` does not cover the 3.3 s top.
18. **Per-frame Python preprocess at 1080p on the tick thread** (`preprocess.py:100-117`): rotate+flip+auto-gain quantile — a meaningful slice of the 33 ms budget before any measurement happens.
19. **Video detect stride floor 4** (`controller.py:1083`): bakes the 83 ms CPU-era cost into playback, capping analysis at ~7.5 Hz regardless of the 94 fps GPU (comment says every-3rd, code says 4 — abandoned tuning).
20. **Foreign/occupancy signals throttled and coarse** (every 3rd frame at 160 px, `pipeline.py:583-592, 889-891`): sub-ball evidence invisible by construction, vacancy verdicts lag up to 3 frames.
21. **`move_streak` counts MATCHED frames only** (`tracking.py:655-668`): the motion judgment arrives late or never during blur — exactly the episode it exists for.
22. **Sidecar shot start lags the true strike 1.8–3.0 s** (`stroke_vision.py:12-14`): spawned the strike re-detection subsystem, `video_time_offset()`, the per-shot `o = start − strike` bridge, the backwards address scan. A one-clock engine collapses three compensation layers (which must then report 0, not disappear).
23. **SidecarReader parses the whole JSONL into RAM per construction, constructed profligately** (one per close-pass stage + one PER MISS in forensic fill): today 12.3 MB/54k states max; at 30 fps a 90-min session is ~162k states/~37 MB and a 50-shot close pass parses ~2 GB of JSON.
24. **The ruler itself is stride-5, 5-minute-window, subprocess-per-session with 12 h gate passes** (`score_corpus.py:41-46, 70-75`; `score_challenger.py:88-121`): temporal thresholds are per PROCESSED step, so cross-stride comparison is invalid; rare late-session failures never meet the gate; at 94 fps GPU the whole-session stride-1 corpus becomes affordable.
25. **The gate assumes the engine is one ONNX file** (`score_challenger.py:31-36, 110-125`): there is no champion/challenger mechanism for CODE — no fingerprint, no `.bak`, no health_check verification.
26. **No live-throughput bar anywhere**: the health board's only fps check is CACHED playback ≥30 fps (`health_check.py:95-131`); nothing measures live loop rate, latency, or dropped-frame ratio — the new engine's whole point.
27. **Measurement latency invisible to instrumentation**: the health log's "pipeline ms" only sees the display path (`pipeline.py:1229`); the worker's 68 ms never appears.
28. **Stale decoys**: six generations of frozen aggregates in `_eval/`; GOALS G4 text still points at the incomparable handfilter_c5 ruler (0.50/1k vs the current-era same-batch 1.64/1k); CI gate is synthetic ±40%; the audio-onset scorecard is unreproducible post-mic-reseat.

---

## 4. TARGET ARCHITECTURE

The new engine is a **measurement core**: a capture-clocked, every-frame position pipeline that lives beside the Qt app instead of inside its tick, and that speaks the existing contracts at its edges (FrameSource, Track/Detection dataclasses, sidecar JSONL row shape, FramePacket, `run_close_pass`).

**Capture.** ffmpeg keeps exclusive device ownership and the recording path stays untouched — the new consumer is still a pure pipe-drain that can never back-pressure ffmpeg (re-verify with the stall test: 900 frames / 30.000 s, byte-identical). What changes is that every analysis frame is stamped at the drain with a monotonic capture timestamp and a sequence number the moment `stream.read(n)` completes, and handed over through a small take-semantics ring (latest-N, unique-capture counting per commit 58bfd24's lesson): duplicates and skips become countable, never silent. The 60 fps input container stops being pre-decimated to 30 inside ffmpeg only if and when the engine earns it; 30 fps every-frame is the target of record.

**Inference.** A dedicated measurement thread (normal priority — the BELOW_NORMAL demote dies with its premise) pulls frames and runs the ensemble as **batched GPU work**: the two finder tiles submitted as one `[2,3,640,640]` run, identifier every frame (at 10.6 ms/inference the every-2nd-cycle economy is pure latency), and pre/post + colour sampling double-buffered against the GPU so the CPU works on frame N's colours while frame N+1 is in `sess.run`. All the detection IP survives verbatim: table-crop before centered letterbox, tiled rescan, seam-drop + 1.25r merge, provider probing with the loud CPU warning, `sample_colour` on every ball, `measured_bgr` carried end-to-end.

**Tracker.** A motion-model tracker replaces rest-freeze and budget-coasting: per-track filter with process/measurement covariance and a rolling-friction deceleration model, so it **coasts through blur** on physics (uncertainty grows, gate grows with covariance, not linearly with misses) and **predicts during occlusion** instead of freezing. Association is a global assignment over a joint cost (predicted position, size, class agreement, measured colour) — which is what makes the greedy-era patch chain unreachable rather than deleted. Every identity invariant in §2.3 is re-derived in **seconds and diameters/second**: rest threshold 0.12 diam/frame-equivalent, number release ~1 s, kill ~6 s, reachability 0.35 short-sides per frame-interval, motion state judged from the filter's own state (fixing the `move_streak`-under-blur inversion). The pinned suite (`tests/test_tracking.py`, 37 incidents) is ported test-by-test to the new internals.

**Output.** One clock: state t = capture timestamp rebased to the first written state (within 2 s of zero — the reader's >30 s legacy guard must never fire on new files). The sidecar keeps the exact JSONL row shape (`type:'f'`, `tracks:[[id,x,y,r,num,cls,active]]`, v2 `c`/`ff`, `type` first key) but writes **densely during motion**: every frame while shot_state = moving, throttled (~10 Hz or on-change) at rest — the burst rate is what the trails, outcomes, and forensic layers were starving for, and the rest-throttle keeps a 90-min session near today's file economics instead of a flat 3×. `video_time_offset()` returns 0 by construction and the compensation machinery stays in place for legacy sessions. A streaming/indexed reader path replaces parse-everything-per-construction before close passes meet dense files.

**Instrumentation.** Per-link counters end-to-end (captured/unique/enqueued/inferred/associated/published/written + measurement latency p50/p95) in the 150-tick heartbeat, and a live-throughput bar on the health board — the number the whole respin exists to move.

```
                       ffmpeg (owns device)                        [UNTOUCHED]
                      /                    \
        recording mp4 (never via Python)    analysis pipe, full-res 30fps
                                                 |
                              drain thread: stamp (t_capture, seq) at read()
                                                 |
                              take-semantics ring (latest-N, unique-count)
                                                 |
              measurement thread (normal prio) --+-- per-link counters
                |                                          |
                |  batched GPU inference                   |
                |  finder tiles [2,3,640,640] one run      |
                |  identifier EVERY frame                  |
                |  pre/post + colour double-buffered       |
                |                                          |
                v                                          v
        motion-model tracker                        heartbeat / health bar
        filter + friction + covariance              (live results/s, latency)
        global assignment (pos+size+cls+colour)
        coast through blur, predict occlusion
        identity invariants in s and diam/s
                |
                +---------------------------+
                |                           |
      dense sidecar writer          FramePacket -> Qt UI  [existing contract]
      same JSONL contract           (display never runs inference)
      30Hz while moving,
      throttled at rest, t0-rebased
                |
      close pass / exports / phone  [UNTOUCHED — run_close_pass, shots.json]
```

---

### M1 measurement log

- 2026-08-27: budget split measured (offline, live app concurrently on the
  same GPU): finder 87-138 ms/frame = **2 serial model runs x ~37 ms**
  (two-pass tiling) + only 12.4 ms Python glue; identifier 67 ms (every
  2nd tick). Bare 640x640 run benches 10.6 ms - the 37 ms per-run gap is
  GPU contention with the live pipeline + input-size differences (audit
  pending). DESIGN CONSEQUENCES: (1) batch the two tiles into one run;
  (2) batch identifier crops; (3) contention is a first-class constraint -
  the engine's end-state REPLACES the old worker (M3 shadow is a
  transition, not a steady state); (4) M1 exit is measured under real
  concurrent load, not idle-GPU numbers.

## 5. MILESTONES

- 2026-08-27 (overnight): FIRST FULL DENSE RUNS. Small session + the
  90-min marathon re-processed end-to-end: every frame written, 30.0
  states/s through the canonical reader. THE HEADLINE: dense cue-ball
  takeoff response median **-34ms** from the verified strike (n=100
  shots; sparse era 1600-2200ms) - the blur blindness is solved in
  data. THE LESSON: 85% duplicate-identity frames - positional
  exclusivity was ported, NUMBER exclusivity was not (the KEEP table
  said so). Fixed twice over (one-to-one finder<->identifier pairing;
  emit-time number arbitration), 2000-frame verify = 0 duplicates,
  6 tracker contract tests. Marathon re-running on the corrected
  engine (~5.4h). Next: dense-trail merge exporter -> the marathon's
  shots.json -> Joe's replay goes pixel-locked.


Ordered; each has a measurable exit. Recording always outranks any milestone work (BelowNormal, killed on `.part` appearance).

### M1 — Offline engine: dense re-processing of recorded sessions
Build the engine as an offline harness first (no controller, no Qt): decode a recorded session, run capture-stamped batched inference + motion-model tracker, write a dense sidecar to a scratch path.
**Exit:** re-processes `session-20260826-002906` (90 min, 196 shots, 54k legacy states) end-to-end at ≥30 fps sustained throughput on the rig's GPU; output sidecar parses through `SidecarReader` (row shape, t0 within 2 s of zero, `type`-first keys, v2 fields); state density ≥28 states/s during shot windows; `eval/invariants.py` SequenceScorer runs on the output with zero degenerate flags; the ported `tests/test_tracking.py` intent-suite passes.

### M2 — Corpus gates matched or beaten
Because the ruler is stride-sensitive (debt #24: no cross-stride comparison is valid), first re-baseline the champion **same-batch at the new engine's stride** (stride 1, whole sessions — affordable at GPU speed), then score the new engine in the identical batch, with `--exclude` for any session used during development tuning. Extend `score_challenger.py`'s harness to gate a code engine, not just an ONNX file (see M4 fingerprinting). Verdicts from written `aggregate.json` artifacts only — never a piped exit code.
**Exit — the exact scorecard, all rows required:**
1. Physics-impossible rate ≤ same-batch champion ×1.10 + 0.05 absolute (current same-batch champion ruler: 1.638/1k ball-frames, 497/303,424 over 33 sessions — to be re-measured at stride 1 in the same batch).
2. Mean coverage ≥ champion ×0.95 (champion ~6.59 balls/frame).
3. Out-of-game number rate (mission metric) DROPS on held-out sessions (c7's held-out mark: 23.77/1k on 24 sessions; a rise rejects even if rows 1–2 pass — c6 precedent).
4. Duplicate rendered identities: 0 tolerated (library at DONE: 0/149,094 states; any dupe = RED).
5. Identity hops ≤ current production 0.13/1k states on the 3 newest sessions; hard tripwires AMBER >2.0, RED >4.0 /1k.
6. Zero degenerate sessions; failed sessions ≤ champion's failed count.
7. Shot detection at stride 1 vs the audio witness: precision ≥ the G5 mark (279/283 audio-confirmed, zero confirmed false shots) and recall ~95% of onsets — after re-baselining the onset profile against post-mic-reseat recordings (the G5 numbers are otherwise unreproducible).
8. Per-class identity vs hand labels (`measure_class_accuracy.py`): dark cluster 3/4/7/8 ≥95%, scored on all three sets (gameplay / racks / old-era archive) per the three-set rule.
9. New rows this respin adds to the board: measurement latency (capture→published track) p95 ≤ 100 ms; unique-capture measurement rate ≥28/30 frames on live-rate replay.

### M3 — Live shadow mode
The engine runs beside the old pipeline on the live rig: same drained frames (via the stamped ring; the old path keeps its current `read()`), writes `<video>.analysis.shadow.jsonl` next to the real sidecar, publishes nothing to UI, tracker, or phone.
**Exit:** across ≥3 real recorded sessions — (a) recordings byte-equivalent risk retired by re-running the stall test with the shadow consumer attached; (b) old-path behavior unchanged: display fps, heartbeat `submit/ingest/pub` counters, and the real sidecar statistically indistinguishable from pre-shadow sessions; (c) shadow engine sustains ≥28 results/s live with p95 latency ≤100 ms through whole sessions including record toggles (surviving the 1.8–3.3 s reopen window without tripping camera-miss); (d) shadow sidecars pass the sidecar-based audits (`audit_render.py` 0 dupes, `audit_identity_wander.py` within tripwires) and beat the real sidecar on shot-window state density; (e) `run_close_pass` executed against a COPY of a shadow sidecar produces outcomes/trails/stroke fields at ≥ the real sidecar's accuracy on Joe's reviewed shots (`_eval/review_scoreboard.json` attachment rules — no orphaned verdicts).

### M4 — Promotion via the standard gate, with a rollback plan
Promote through the existing conventions, extended to code: an engine fingerprint (git commit + model file sizes) recorded in `_eval/champion.json`-style metadata; `health_check.py` verifies it and gains the live-throughput bar (results/s and latency RED thresholds) alongside the existing 14 checks. Promotion flips the controller to consume the new engine's tracks and dense sidecar writer; both cadence gates from debt #6 (`controller.py` and `build_analysis_cache.py`) change together.
**Exit:** M2 scorecard reconfirmed same-batch at promotion time; first full production session post-promotion reviewed against the 2026-08-26 reference session (196 shots / 147 attempts benchmarks: stroke coverage with stated abstentions, id_hops ≤0.15/1k, t_offset ≈0); phone/companion surfaces verified on the new sidecar (line-prefix scan still finds shots).
**Rollback plan (a file copy + a flag, rehearsed before promotion):** the old engine remains in-tree behind a config flag for at least one full session; rollback = flip the flag + restore fingerprint metadata; sidecars written by the new engine remain readable by all consumers either way (same contract), so no data migration is ever part of rollback; `carry_review_verdicts` semantics guarantee Joe's verdicts survive any re-backfill in either direction.

---

## 6. NON-GOALS

Explicitly OUT of this respin — the engine is promoted **behind** these contracts, not through them:

- **App shell / Qt UI rewrite.** The controller slims to a consumer of the engine; the clips-player redesign is a separate track (`ui-redesign-target-is-clips-player`). FramePacket stays.
- **Phone app and its protocol.** `shots.json` schema, the correction-file protocol, key attachment rules, Dropbox transport — unchanged. The phone never reads the sidecar today and still won't.
- **Close pass rewrite.** `run_close_pass` and its stages (outcomes, actions, stroke_vision, shots_export, miss_tags) are consumers; they get re-tuned constants (debt #12, #29) and a faster reader, not a redesign. Retiring the blur-compensator stack (trail_resample, forensic_repass) happens only after the corpus gates prove dense tracks made it dead code.
- **Recording chain changes.** ffmpeg device ownership, encoder ladder, fragmented-mp4/.part staging, CFR retimer — untouched and re-verified, never modified, by this work.
- **Camera hardware.** The T3i/90D upgrade path (`capture-chain-t3i-limits`) is orthogonal; the engine targets the current 30 fps full-res analysis stream.
- **Interpreting misses vs makes.** No new coaching, diagnosis, or miss-tag intelligence until the measurement core is promoted — that is the directive this document exists to serve (`no-oversimplified-diagnosis` still governs whatever comes after).