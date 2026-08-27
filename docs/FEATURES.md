# BilliardsTrainer — Features & Requirements

One line per feature. Status legend:

| Mark | Meaning |
|------|---------|
| ✅ | Live and verified |
| 🟡 | Partially there / needs polish |
| 🔜 | Planned, buildable now |
| 🔒 | Defined, gated on dense measurement (the core respin) |

**Operating principle (Joe):** fast and accurate data points first — animate
and interpret however we want *after*. Measurement before interpretation,
always. End-state: a ground-up C++ real-time engine once the reliability
ladder holds in Python (`docs/MEASUREMENT_CORE.md`).

---

## 1. Measurement & tracking — the reliability ladder

- ✅ Table detection: felt-edge calibration, consensus lock, auto-relock, persisted across launches
- ✅ Camera chain: Canon → Cam Link capture, recording-first discipline (recording can never be disturbed by analysis)
- ✅ GPU inference (DirectML): live detection throughput 3.5× the CPU era
- 🟡 Pocket detection: geometric placement from calibration only — visual localization of real pocket mouths is ladder work
- 🟡 Cue ball tracking: solid at rest and mid-roll; blind ~1.5–2 s at takeoff (motion blur) — the core respin's first target
- 🟡 Object ball tracking: same engine, same takeoff gap
- 🔒 Ball positions measured EVERY frame (30 fps) live during play — milestone M1–M4 of the measurement core
- 🔒 Motion-model tracker that coasts through blur and occlusion instead of freezing
- ✅ Ball identity: zero duplicate identities corpus-wide; id-hops 0.13/1k (best on record)
- ✅ Identity champion gating: challengers promote only by beating the corpus scorecard

## 2. Shot detection & game events

- ✅ Shot detection ~95% recall, ~100% precision (audio-audited); rides on tracking quality
- ✅ Make / miss / scratch outcomes, ball-vanish pot detection (incl. motion-blurred "unseen pots")
- ✅ Scratch vs foul distinguished; foul `no_contact` (cue never touched a ball) measured live
- 🔒 Foul `wrong_ball_first` (9-ball: lowest ball first) — needs first-contact resolution inside the takeoff window
- 🔒 Foul `no_rail` (contact but nothing touches a rail after) — needs dense post-contact tracking
- 🔜 Missed-shot recovery: recall audit on recent sessions feeding the correction loop
- ✅ Break detection (5+ balls scattering) with league-style longer countdown after

## 3. Shot clock

- ✅ Cue-ball–driven: countdown starts when ALL balls rest, stops the instant the cue ball is struck
- ✅ Cadence: start bell · bell at 20 s · spoken "Ten" at 10 s · 3-2-1 taps · buzzer at zero
- ✅ Pause / Resume, per-shot seconds, time-after-break — all in the rail panel
- ✅ On/off from the panel and the Play menu, works with or without recording
- ✅ Table status readout: ON THE CLOCK / SHOT IN PLAY / TABLE SETTLED / PAUSED
- ✅ Clock transitions recorded to the session sidecar (start/stop/pause/resume + countdown length)
- 🔜 Shot-clock replay as a video overlay on iOS (data already exported in the dossier)
- 🔜 Frame-exact strike stop via the strike SOUND (recorder-tee design, tested off-hours)

## 4. Audio & narration — live verification you can hear

- ✅ Bell-tone cue set (harmonic synthesis; the square-wave chirps are gone)
- ✅ Per-cue volume sliders with release-to-preview; zero mutes a cue individually
- ✅ Neural voice (local WAV cache, offline playback): "Ten", "Scratch", "Foul", "Ball in hand", "Table change"
- ✅ "Ball in hand" (cue picked up alone) vs "Table change" (object balls moved, non-shot) — episode-gated, silent during flight
- 🔜 Made-ball calls by name ("Nine ball") — needs ball number carried on the live pocket event
- 🔜 Narration set expansion as analysis widens (runs, streaks, position quality)
- 🔜 Haptics companion for narration (Joe's sketch; after voice matures)

## 5. Stroke & body metrics

- ✅ Stay-down time per shot (camera-measured), popped-early flag on misses
- ✅ Live stay-down timer on the scoreboard: climbs from the strike, locks to the measured value
- ✅ Backstroke depth, pause, practice strokes; delivery time v2 (abstains instead of guessing)
- 🟡 Delivery on live-measured legacy shots: library re-measurement ~70% complete (resumes in quiet windows)
- 🔜 Cue-sensor (Bluetooth IMU) fusion when Joe installs it — camera remains the fallback

## 6. Desktop review

- ✅ Session timeline: hover cards, thumbnails, prev/next, sortable shot list, zoom/pan, one-click clip export
- ✅ Live scoreboard (recording-gated), collapsible Stats / Shot Timeline / Shot Clock / Cue Sensor panels
- ✅ Window squishes to ~450 px for side-by-side with VS Code; menu bar keeps every control reachable
- ✅ Shot details populate as soon as each shot completes (live stroke rows)
- 🔜 Per-shot detail parity with the phone's Details sheet

## 7. Phone review (iOS web app)

- ✅ Sessions list: instant cached render, timeout + auto-retry, self-reporting failures, full build id in header
- ✅ Player: shot navigation, aim lines, full-length ball trails (strike to rest, cue + object balls)
- 🟡 Trail smoothness: frozen-prefix rendering + physics-shaped takeoff interpolation — true pixel-lock arrives with dense measurement
- ✅ 720p playback proxies (~1/5 the data), library-wide; phone picks them automatically
- ✅ Details bottom-sheet: measured-or-abstained fields with stated reasons; correction channel (Joe's verdicts are final)
- ✅ Slow-mo requests, playlists, Lifetime Stats (miss patterns, trusted shots only)
- ✅ What's New changelog; diagnostics HUD (triple-tap top bar); runtime smoke gate blocks broken deploys
- 🔜 Shot-clock overlay during playback
- 🔜 Data-saver 480p tier for weak-signal sessions
- 🔒 Per-shot clips ("streaming shots") — coverage audit done; cutting waits on dense trails

## 8. Data & analysis pipeline

- ✅ One clock: all shot times anchored to video via per-shot verified strikes
- ✅ One pass: every session finishes through the same canonical close pass (drift-tested)
- ✅ Measured-or-abstained contract: absent fields state their reason, everywhere
- ✅ Session sidecar: tracking states, shots, strokes, corrections, clock transitions — append-only, replayable
- ✅ Forensic corridor re-pass recovers miss verdicts the derivation abstained on
- ✅ Library index + lifetime stats refreshed at close
- 🔒 Dense sidecar (≥28 states/s during shots) — measurement core M1

## 9. Models & training

- ✅ Champion/challenger with held-out corpus gates; promotion-only stripe repair (gameplay identification 100%)
- ✅ Auto-labelling loop (montage + VLM) feeding the training store
- 🔜 c8 challenger when dense-layout labels accumulate (rack recall is the weak spot)
- 🔜 8-ball rack minute from Joe → stripe training data

## 10. Ops & autonomy

- ✅ Autonomous work loop: health-first, one measured chunk, honest progress log, watchdog recovery
- ✅ Guards: recording outranks everything; Joe-presence defers heavy jobs; two-heavy-jobs ban
- ✅ Runtime smoke gate on phone deploys; pre-push hook runs the suite
- ✅ App self-updates from git on launch; What's New tells Joe what changed in his terms

## 11. Horizon

- 🔜 C++ ground-up measurement engine (after the ladder holds in Python) — no UI, no interpretation, mobile-testable
- 🔜 VLM shot coaching on the dossier ("you missed left, popped up, cue swerved") — only after measurement is rock solid
- 🔜 90D camera upgrade path (60 fps HDMI) when hardware lands
- 🔜 Real-time mobile live view of the table state
