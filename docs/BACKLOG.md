# BACKLOG — the one work queue

**This is THE queue.** Autonomous work sessions, M1 push sessions, and
watchdog recoveries take the TOPMOST unblocked item and do one bounded
chunk. `docs/GOALS.md` is the append-only LOG of what happened; this
file is what's NEXT. Reorder here, never fork a second list (L1).
An agent that "finds nothing to work on" hasn't read Tier 4.

Ground rules for every item: ARCHITECTURE.md checklist first; suite
green before commit; demolition chunks DELETE the superseded code in
the same commit; one GOALS line per chunk landed; heavy GPU jobs are
presence/recording-guarded, one at a time.

---

## TIER 0 — THE ENGINE RELIABILITY CAMPAIGN (Joe, 2026-08-28)

**Everything else waits.** Joe: "just get this engine reliability
going... none of my other features matter if we can't get this right...
keep churning, ideally 24/7... you have autonomy to keep working at
your own direction so long as you perpetually use your LLM Vision to
corroborate the results of the app's analysis and animations."

Loop (repeat forever until the bar is met): pick the top disagreement
→ instrument it → fix → re-measure the bench (~6 min) → **WATCH the
frames and compare** → log the agreement score → next. Heavy GPU work
(marathon, library, retrains) runs overnight. New Joe requests during
the campaign are APPENDED to Tier 3, never built.

BAR: every real shot detected, windowed at its strike, named,
classified (stroke vs setup), and scored correctly — verified by
Claude's vision, not by metrics.

<!-- CAMPAIGN-STATE:BEGIN -- machine-written by tools/campaign_state.py;
     do not hand-edit. Rounds 34-47 hand-edited this file with anchor
     `.replace()` calls whose anchors no longer existed, so the edits
     silently did nothing and THE QUEUE went stale for fourteen rounds
     while every script reported success. A session reading Tier 0 on
     2026-08-30 would have been told naming was 74.5% with an invented
     "11" (actual: 99.4% and none) and gone off to re-fix solved
     problems.
     Regenerated from the bench sidecar + summary, and pinned by
     tests/test_campaign_state.py, which FAILS if this block is missing,
     unparseable, or older than the newest bench measurement. -->

**CURRENT STATE — machine-written, do not hand-edit.**

    written        2026-08-30T23:26Z
    bench          session-20260824-220247.mp4
    engine rules_v 20
    measured       2026-08-30T22:14Z
    shot list      12 entries (9 strokes, 3 makes)

Run `python tools/scorecard.py` for the full card; that is the
gate of record. This block exists so a session picking up the
queue cannot be told something the measurements disagree with.

<!-- CAMPAIGN-STATE:END -->

### NEXT TARGETS (top first) — round 85

*** TRIED TO READ THE PRINTED NUMBERS OFF THE BALLS. THE CONTROL FAILED,
    SO THE METHOD IS REJECTED. ***
    The cold 6, 7 and 8 are never potted, so round 84's pot-order check
    cannot reach them - and they are the most load-bearing references on
    that table (round 82: the 7 and 8 are named ENTIRELY by colour).
    ATTEMPT 1, and it was a good idea: those balls are MOTIONLESS all
    clip, so 400 frames each were averaged to cancel sensor noise and
    compression. The averages are strikingly clean and the colours became
    unmistakable - but the DIGITS did not resolve. What is visible is the
    white number CIRCLE, foreshortened onto the ball's upper surface by
    the overhead camera.
    THE CONTROL IS WHAT SETTLED IT: ball 2, independently confirmed by
    its own pot in round 84, does not read as a "2" either. A method that
    cannot read a ball whose answer is known cannot be trusted on one
    whose answer is not. Rejected, and recorded so it is not retried as
    though it were new.
    ATTEMPT 2, which is what the round actually delivers: the averaged
    colours are unambiguous and force the assignment ON A STANDARD SET.
        claimed 6:  L=93  a*=100  strongly GREEN, 85+ Lab from either dark
        claimed 7:  L=21  a*=145  dark RED
        claimed 8:  L=10  a*=126  NEUTRAL and darkest
    The only plausible swap was 7 against 8, and they differ on exactly
    the right axis - 19 units of red chroma against neutral, and the 8
    darker. So 6/7/8 move from "by eye, assumption unstated" to
    "canonical-set, assumption STATED, colours measured, swap risk
    assessed".
    THIS IS WEAKER THAN THE POT ORDER AND SAYS SO. It is independent of
    the APP - which is what round 62's circularity warning was about -
    but NOT independent of the convention. If Joe ever plays with a
    non-standard set, these three entries break first. A test guards
    against anyone re-labelling them pot-order derived.
    NO ENGINE CODE CHANGED; both clips verified identical.

0. EVERY TRUTH-SIDE ENTRY ON BOTH CLIPS NOW CARRIES ITS PROVENANCE AND
    ITS LIMIT. Bench: pot-order derived and cross-checked 1082/1082
    (83). Cold: five balls pot-order confirmed (84), the 1 via the stripe
    test, three by canonical set with the assumption stated (85). There
    is no further truth-side work that does not need NEW EVIDENCE -
    either a non-standard-set clip to test the assumption, or footage
    where 6/7/8 are potted.

1. THE HAND VETO IS BLOCKED ON A CORPUS (round 80): needs genuine hand
    detections; session-20260802-173553 is not in the library. BLOCKED.

2. THE PHONE PAYLOAD ON WEAK CELLULAR: the biggest session's shots.json
    is 1,962 KB of dense 30fps trails, and the cost to pull it on a bad
    connection is unmeasured. UNBLOCKED and product-visible, though it is
    delivery rather than engine.

3. THE ENGINE'S REMAINING ERRORS ARE INDIVIDUALLY NAMED: bench one
    unnamed and one blind sighting with ZERO wrong; cold 4 unnamed 5s and
    one 9->1. Both clips pass every gate. The honest position is that the
    measurement engine is at the end of what these two clips can teach -
    NEW FOOTAGE is now worth more than another pass over these.

4. tools/phone_view.py (round 74) screenshots the real player; --local
    serves the working tree so a UI fix is checked BEFORE it ships.

5. METHOD WARNINGS, all bought: the naming truth samples ~1/sec on
    settled moments - a fine YARDSTICK and a biased SURVEY (65); a
    hypothesis written into this backlog is not a finding but inherits
    the authority of one (67, 72); a truth-side sample can be
    contaminated rather than imprecise (69, 80); aggregating over a
    window hides a gap inside it (70); measure the discriminator itself
    before building on it (71, 73, 77); a metric can be the defect (72,
    78) and a posted regression deserves the same scrutiny as a posted
    win (78); an angle test needs a magnitude guard (75); a regeneration
    needs a gate or it silently deletes (76); a favourable number
    deserves auditing too (79); check what your control group actually
    contains (80); a statistic about detections is only as good as its
    definition of one (81); a rebuilt object loses every field the
    constructor is not told about (82); a yardstick nobody has checked is
    not evidence (83); when a method cannot reach something, say which
    something (84); AND PUT A KNOWN ANSWER THROUGH EVERY NEW METHOD - the
    control is what rejected this round's headline idea (85).

6. The palette is hand-labelled and does not scale; the identifier
    mislabels balls mid-collision (55); colour cannot separate gold from
    white at speed (56); both naming figures in the phone STATUS view;
    recovered detections lose their name; _locate is ~37% of engine wall
    time; rebuild_batch.py still drives an OLD build() path; events/
    shot.py is a third shot detector in the live path; delete vision/
    tracking.py and the MeasurementCore shadow scaffolding.

CAPABILITY LADDER (Joe, 2026-08-28: "break it up by clip yes but also
by feature/requirement"). Rungs are ordered so each depends only on
the ones below it; every rung is measured by tools/scorecard.py and
shown in the phone's STATUS view. A rung is DONE on a clip only when
its gate holds AND a vision watch agrees.
  R0 GEOMETRY - table + POCKET DEAD ZONES (Joe's design): pockets are
     explicit zones; anything resting in a zone is never "on the
     table"; a pot = entered a zone and did not come back out.
     Gate: zero furniture tracks; zone overlay matches the video.
     STATUS: partially done via ad-hoc rules - dead zones replace them.
  R1 CUE BALL - found, named, and continuously tracked; trail covers
     the real path from the strike. Gate: cue correctly named >=99% of
     frames, one cue track per shot, no phantom cue.
     NOW: 99.7% on the HONEST metric (round 11 - live sighting
     required). Residual failures are windows where the label rides a
     coast while the real ball is seen-but-unnamed: an R3 symptom.
     PREVIOUSLY: the metric read 100% but it was WEAK - it only checks that
     exactly one track claims "cue", not that the label sits on the
     real ball. Evidence frame 154.8s shows the cue label on EMPTY
     FELT beside an unnamed white ball. R1 is NOT passing until the
     metric checks correctness (label within a ball radius of a live
     detection) - fix the metric first, then the failure it exposes.
  R2 SHOT EVENTS - every stroke found and windowed AT the strike; no
     stroke invented during hand setup. Gate: 10/10 found, 0 fake.
     *** PASSED *** (round 44+): 10/10 found, 0 fake, 0 unexplained.
     Held every round since. NOTE the prose below this line is HISTORY,
     kept for the case law; the CURRENT STATE block above and
     tools/scorecard.py are the only live numbers. Rounds 34-47 could
     not update this file at all (see the block's note), so anything
     here describing a "NOW" older than round 33 is stale by
     construction.
  R3 ROUND 14 (tried + reverted): "persistence" jaw filter - refuse a
     dim near-pocket read only if a detection sat at the same spot last
     frame. FAILED twice: (a) it read self._last_detections, which ONLY
     the live path writes, so offline it was inert (scores identical to
     deleting the filter); (b) with real history it also killed balls
     RESTING near a pocket - they too are at the same spot every frame -
     and shots found went 10 -> 0. Reverted. LESSON: "has not moved"
     cannot separate leather from ball; both sit still. NEXT: gate on
     ARRIVAL - accept a dim near-pocket read only when a track was
     travelling toward that pocket in the preceding ~0.5s (needs the
     engine's own tracker passed into prepare_detections; the pipeline's
     tracker is empty offline).
  R3 ROUND 13 (tried + reverted): deleting the jaw confidence filter
     recovered ~35% more moving-ball samples (real pots score 0.49-0.74
     at 16-28px from the pocket and were being discarded) BUT the pocket
     leather returned as a phantom "8" in 962 frames (was 18), dropping
     shots 10->8 and calls 8->6. Net negative; reverted. NEXT ATTEMPT:
     make the jaw filter CONDITIONAL - accept a dim read near a pocket
     when a track is moving toward that pocket (a ball arriving), refuse
     it when nothing is approaching (static leather). Also fix the
     furniture-by-time rule so a re-born leather track cannot dodge it.
  R3/R4 ROUND 25 (THE YARDSTICK WAS BROKEN - fixed, and it scored):
     RULE 0 applied to docs/bench_truth.json itself for the first time.
     Colour census of 1534 detections over the whole clip + a 2s
     occupancy timeline found THREE truth errors:
       (a) NO orange 5 exists on this table - zero orange detections.
           ONE yellow solid was named "9" (31.7), "1" (154.2) and "5"
           (170.6) by the original watch. It is the 1: its body yellow
           matches the 9's stripe yellow to 7.3 Lab.
       (b) The real 9 is a yellow STRIPE static at (528,1079) in ALL
           111 samples, 18s->236s. Truth had it potted at 31.7, so the
           scorecard was PENALISING the engine for being correct.
       (c) 130.2 was recorded as a pot; at 0.5s resolution the 1 rolls
           to the bottom-right rail, returns, and settles at (617,877)
           through 144.5s. Nothing fell. Real pots = 4, not 5.
     Truth corrected with inline provenance; my own 2 mislabelled
     corpus boxes (class 5 -> 1) fixed; corpus now 115 boxes / 6
     classes, all >=16 samples.
     RESULT with NO engine change: outcomes 8/10 -> 9/10.
     Remaining real outcome failure: the 170.6 long pot is missed.
  R3 *** PASSED *** / R4 *** PASSED *** - ROUND 33 SHIPPED
     (ENGINE_RULES_V 13 -> 14).
     ROOT CAUSE, after three rounds of attacking the INPUTS: the emit
     stage's REST-FROZEN IDENTITY was absolute. `elif at_rest and
     tr.emitted >= 0: tr.pend, tr.pend_k = -1, 0` RESET the pending
     counter, so evidence against a resting track's name could never
     accumulate - the name a ball held when it settled was final. The
     static 9 took "1" in a 6-frame lapse at 156.3 and wore it for 80s
     while the identifier read 9 in 114/114 frames underneath. That is
     also why rounds 32/33's three input fixes each merely changed WHICH
     wrong name froze (the identical 9->3 x216 signature twice).
     FIX 1: at rest a candidate still has to lead, but for REST_HYST_K
     (45 frames ~ 1.5s) instead of forever. Misreads still bounce off.
     FIX 2 (uncovered by fix 1): with the freeze lifted the purple 4
     went 136/136 -> 2/136, called "7" x129. Not a new bug - the freeze
     had been HIDING it. The identifier genuinely misreads the dark
     4/7/8 cluster under warm light; the ensemble's _fix_colour exists
     for exactly this and the engine never called it. Now called on the
     model's read, against measured colour refs for this table.
     MEASURED, champion -> new:
       named correctly   89.2%  -> 99.5%   (target 95 - PASSED)
       outcomes right    9/10   -> 10/10
       pot attribution   3/4    -> 4/4
       per ball 0:221/221 1:84/85 2:156/156 3:188/189 4:136/136
                9:218/221   (three wrong sightings in the whole clip)
       strokes 10/10, unexplained 0, cue 100% - all HELD
     Vision-checked at t=200s: five balls, five correct names including
     id4 = 9, no phantom circles. Pinned by 9 tests in
     tests/test_identity_continuity.py (a brief misread must still
     bounce off a resting ball; a sustained one must land).
     NOTE / REPRODUCIBILITY GAP: the fix depends on measured colour refs
     in APP_DIR/colour_refs.json, regenerated by
     `tools/build_colour_refs.py --store _train/bench_fix/store --write`
     - but _train is gitignored, so a fresh clone cannot rebuild them.
     Fix that before P2 (cold clip).
  R2 REMAINING ON THE BENCH: 1 fake stroke (hand setup at ~13.1s, needs
     the cue stick recognised) and invented numbers [8, 10] over 36
     frames (a non-ball object taking a name). Both small; next.
  R3 ROUND 32 - EXACT CAUSE OF THE RESIDUAL 9->1; TWO FIXES TRIED,
     BOTH MEASURED WORSE, BOTH REVERTED. Champion holds at 89.2%.
     THE MECHANISM (traced frame by frame): the 9 is wrong in ONE
     unbroken span, t=157.0 -> 236.0, track id4, and the flip happens in
     place at 156.31 while the ball is motionless. The identifier reads
     9 at that position in 114/114 sampled frames across the period - it
     never stops knowing. But _pair_identities is EXCLUSIVE and greedy
     by distance: at 155.8-156.4 the yellow 1 rolls past and its find is
     CLOSER to the 9's own read than the 9's find is, so the read is
     consumed by the passing ball. The 9 is left unpaired and keeps the
     finder's colour guess, which is "1" (43/43). Six such frames
     outvoted 9 permanently (VOTE_N=9).
     ATTEMPT A - unpaired finds carry NO name:
       9: 141/221 -> 219/221 (essentially fixed)
       4: 136/136 -> 0/136   (destroyed - the heuristic is the ONLY
          thing naming the purple 4; the identifier misreads it)
       overall 89.2% -> 83.7%, unnamed frames 28 -> 141. REVERTED.
     ATTEMPT B - repair the guess with the round-27 stripe window
       (_inner_disc frac 0.95, v>150, thresholds 0.28/0.34) applied to
       unpaired finds via _fix_stripe_bit:
       overall 89.2% -> 59.4%, confusions 9->3 x216, 3: 4/189.
       The widened window mis-promotes on the engine path. REVERTED.
     LESSON: both attempts treated the wrong GUESS. The defect is that
     the 9's correct READ is given away.
  R3 NEXT - GIVE THE READ BACK, don't patch the guess:
     idea: a short per-POSITION memory of recent identifier reads. A
     find that is stationary and was read as N within the last ~0.5s
     keeps N when the exclusive pairing hands its read to a closer
     neighbour, instead of falling back to the colour guess. Narrow: it
     cannot touch the purple 4 (which has no competing read to lose).
     Alternatives if that fails: (a) mutual-nearest pairing so a read is
     only consumed by the find it is also nearest to, leaving the
     borrowed case unpaired; (b) let a find keep a name only if the
     colour guess AGREES with the last read at that position.
     Gate: name_right_pct > 89.2 AND strokes 10/10 AND outcomes >= 9/10
     AND per-ball 4 >= 134/136.
  R3 ROUND 31 - *** SHIPPED *** (ENGINE_RULES_V 12 -> 13). First
     champion change of the campaign that improved a gate without
     costing another one.
     THE BLOCKER FROM ROUND 30, SOLVED: the early episode was not caused
     by retirement spawning tracks per se. The bottom-left pocket
     leather flickers (sighting -> coast -> re-snap), and its own
     COASTED drift (~20px off birth) was setting ever_moved, which
     exempted it from the FURNITURE_S rule that exists to kill it; my
     retirement then deleted and re-created it every 3s so it never
     aged into furniture at all. Each cycle read as 500-1700 px/s of
     motion and opened the 154.2 window at 147.98.
     FIX: only a REAL SIGHTING proves movement - guard the ever_moved
     update with `tr.t == t`. A coast is a prediction, not evidence.
     SHIPPED TOGETHER (all three, gated on the full scorecard):
       1. engine._pair_identities: the identifier outranks the finder's
          colour heuristic (dropped the `and d.number < 0` guard).
       2. tracker: retire (delete) a track unseen > RETIRE_S=3.0 that
          EVER MOVED and whose last position says it left the table
          (pocket zone or off-bed); purge its _holder claim.
       3. tracker: ever_moved only from real sightings.
     MEASURED, champion -> new:
       named correctly     76.0% -> 89.2%
       moving balls named  75.7% -> 94.7%
       pot attribution     2/4   -> 3/4
       invented frames     60    -> 36
       strokes 10/10, outcomes 9/10, unexplained 0, cue 100% - ALL HELD
       per ball 0:221/221 1:59/85 2:156/156 3:188/189 4:134/136
                9:141/221  (was 9:0/221)
     Pinned by tests/test_identity_continuity.py (7 tests: identifier
     beats the guess, a potted ball's name cannot latch onto another
     ball, an OCCLUDED ball mid-felt keeps its identity, a coast is not
     movement). Vision-checked at t=200s: 3/cue/4/2 correct, no phantom
     circles, the 9 still labelled 1 - exactly what the metric says.
  R3 NEXT - THE LAST BIG NAMING ERROR: 9->1 x80, and ONLY while the real
     1 is off the table. Same root as fix 1: when the identifier skips a
     frame, the unpaired find keeps the colour heuristic's "yellow = 1"
     guess and that still votes; with the real 1 gone nothing outvotes
     it. Candidate: an unpaired find carries NO number into the tracker
     (the heuristic is measurably bad - 43/43 wrong on the 9, and it is
     the source of the invented 8s and 10s). Worth ~6 points, which
     would put naming near the 95% gate. Measure it - naming may drop if
     the heuristic is carrying balls the identifier cannot see.
  R3 ROUND 30 - BOTH ROOT CAUSES FOUND AND FIXED (not shipped; one
     boundary artifact left). Measured, not theorised:
     CAUSE 1 - PRECEDENCE INVERTED (measure/engine.py _pair_identities):
       `if num >= 0 and getattr(d, "number", -1) < 0:` means the
       identifier's read is applied ONLY where the finder's crude colour
       heuristic has not already guessed. The heuristic calls the yellow
       STRIPED 9 a "1" in 43/43 samples; the identifier reads it 9 in
       71/72 (median score 0.66). The correct read was discarded every
       frame. It is also where the invented numbers come from (the
       heuristic emits 8s and an 11 on this clip).
       FIX: drop the `and d.number < 0` guard - the identifier wins.
       MEASURED ALONE: 9 goes 0/221 -> 141/221, the 9->1 confusion
       (x154) vanishes; but naming overall 76.0 -> 73.8 because the
       freed name "1" then migrates to the red 3 (see cause 2).
     CAUSE 2 - TRACKS ARE NEVER DELETED (measure/tracker.py): going
       inactive is not death. Step 2 associates against every track in
       the dict with no liveness test, and the acquisition gate widens
       with time unseen, so a long-dead track can claim a new ball
       anywhere. Full trace of id3: genuinely the yellow 1 at 10.9s,
       follows it into the bottom-right pocket at 32.2s, re-appears
       across the table at 45.4s, then latches onto the RED 3 at 109.2s
       which answers to "1" for the rest of the clip.
       FIX: retire (delete) a track unseen > RETIRE_S=3.0 AND whose last
       position says it LEFT THE TABLE (pocket zone or off-bed); purge
       its _holder claim too. The left-the-table condition matters:
       blanket retirement at 3s or 8s also retires a ball merely hidden
       by the bridge hand.
     BOTH TOGETHER: naming 76.0% -> 89.2%, invented frames 60 -> 36,
       per ball 0:221/221 1:59/85 2:156/156 3:188/189 4:134/136
       9:141/221 (champion had 9:0/221).
     WHY NOT SHIPPED: strokes 10/10 -> 9/10, outcomes 9/10 -> 8/10.
       The 154.2 stroke is NOT lost - its episode opens at 147.98 (6.2s
       early, outside the scorecard's MATCH_S=2.5) because retirement
       spawns fresh unnamed tracks (-16, -17) whose motion opens the
       window early. movers=[-17,-16,0,1], cue_travel=1331, setup=False.
  R3 NEXT ROUND - fix the episode boundary, then land both fixes as one
     gated bundle. The early opening is hand-driven cue movement at ~148
     that `setup` did not flag; look at the carried/hand context there
     first, and at whether a newly born unnamed track should be allowed
     to OPEN an episode at all (it may only extend one).
     Gate: name_right_pct >= 89 AND strokes 10/10 AND outcomes >= 9/10.
  R3 ROUND 29 - BISECTED ROUND 27 ON THE NEW METRIC; ROOT CAUSE IS
     IDENTITY CONTINUITY, NOT RECOGNITION:
       champion                              76.0%   3: 187/189
       + P5 engine->strat.detect             60.6%   3:  67/189  <-- breaks here
       + measured refs (six balls)           60.0%   3:  67/189
       + withhold names below 0.70 score     60.4%   3:  67/189
     Side effects worth keeping when this is re-landed: measured refs
     restore outcomes to 9/10 and cut invented frames 302 -> 56; the
     score bar removes the invented 8 entirely (56 -> 38, only 5 left).
     Neither touches the 3.
     THE DETECTOR IS INNOCENT: asked directly, the ensemble answers 3
     for the red on every frame - colour distance 14.5 to the 3's
     reference vs 147 to the 1's - and answers 9 for the striped ball.
     THE TRACKER IS INNOCENT: replaying the same frames through a FRESH
     MotionTracker from t=104 feeds it 3, votes [3]*9, emits 3 correctly
     all the way to 122s.
     THE BUG IS HISTORY. The yellow 1 is potted at 33s but its track's
     NAME survives; when the red 3 is hand-placed at 109.2s the stale
     "1" latches onto it (sidecar: id3 born 109.2 already named 1, while
     id4 - the static 9 - has worn "1" since 86s). Two tracks hold "1"
     at once. From 110s to the end the red answers to 1 while the naming
     code underneath keeps saying 3 and is ignored. Same mechanism
     explains 9->1 x154 (the 9 is 0/221).
  R3 NEXT ROUND - IDENTITY CONTINUITY (the real R3 blocker):
     a name must not outlive its ball. Two rules to design and gate:
       (a) when a track dies at a pocket (a pot), RELEASE its number so
           no successor can inherit it;
       (b) a track born far from where a name was last seen must EARN
           that name from fresh reads, not inherit it - the tracker's
           fresh-claim machinery (FRESH_S) already exists for this and
           should be checked at BIRTH, not only in arbitration.
     Gate on name_right_pct per ball; the champion is 76.0% with
     0:221/221 1:66/85 2:156/156 3:187/189 4:136/136 9:0/221.
     Only after that re-land P5 + refs + the score bar as one gated
     bundle - they are correct, they are just downstream of this.
  R3 ROUND 28 - THE INSTRUMENT IS FIXED, AND IT LOCALISED THE DEFECT:
     Added per-ball naming CORRECTNESS to tools/scorecard.py, fed by a
     new pixel-derived truth stream (tools/build_naming_truth.py ->
     docs/bench_naming_truth.json, 221 samples). Truth is colour family
     + the 0.95-window stripe reading, self-contained on purpose: a
     yardstick that imports the app's appearance code moves whenever the
     app does, which is how round 27's regression stayed invisible.
     Verified BY EYE on 4 frames / 23 labels before use
     (_train/bench_fix/naming_truth_check.png).
     BASELINE (champion, unchanged engine): 76.0% named correctly.
       per ball  0:221/221  1:66/85  2:156/156  3:187/189  4:136/136
                 9:0/221
       confusions 9->1 x154 (plus 9->3 x1, 1->2 x1)
     THE DEFECT IS ONE BALL: the static striped 9 is NEVER named
     correctly - 154 sightings called "1", the rest blank. Cue, 2 and 4
     are perfect; the 3 is 187/189. Naming is not diffusely bad, it is
     one total failure. Also 88 truth sightings had no track within
     30px (worth a look later).
     Pinned by tests/test_naming_correctness.py (6 tests incl. the
     literal round-27 case: a ball answering to another ball's name must
     score WRONG, and an average must not launder it).
  R3 NEXT ROUND - RE-LAND ROUND 27 ONE PIECE AT A TIME, gated on
     name_right_pct (not named_moving_pct, which is presence-only and
     was the blind spot). The five pieces, to bisect:
       1. stripe_reading window 0.62 -> 0.95 and v>170 -> v>150
       2. _name_unknown: exclude the +8 partner from the colour margin
       3. _name_unknown: call _fix_stripe_bit after naming
       4. _name_unknown: 0.70 finder-score bar (phantoms score <=0.49)
       5. engine -> strat.detect (the ONE naming owner; law 1)
     Round 27 measured the bundle at naming 80.0% / attribution 3/4 with
     the 9 fixed (6684 frames) but the red 3 renamed "1" x1843. Find
     which piece costs the 3 - suspicion is (2), since dropping the
     partner from the margin also drops 11 from the 3's runner-up scan.
  R3 ROUND 27 - FOUND THE REAL BLOCKER, REVERTED ON A REGRESSION THE
     SCORECARD COULD NOT SEE:
     THE ENGINE NEVER CALLED THE ENSEMBLE. measure/engine.py ran
     strat._finder and strat._identifier itself and paired them with a
     private _pair_identities that carried the model's RAW read. So
     FindIdEnsemble.detect - measured-colour arbitration (_fix_colour),
     the stripe bit (_fix_stripe_bit) and naming balls neither model
     read (_name_unknown) - was DEAD CODE on the engine path. Two
     opinions about one fact; law 1 violated in the load-bearing place.
     That is why rounds 25/26 naming work moved the score by 0.00.
     Asked directly about the same pixels the ensemble answers 9 for the
     static striped ball; the engine's sidecar called it "1" x4613.
     MEASURED with the engine calling strat.detect (+ measured refs,
     + widened stripe window, + a 0.70 score bar on colour naming):
       naming      75.7% -> 80.0%
       attribution 2/4   -> 3/4
       static 9    named 9 in 6684 frames (was "1" in 4613)
       outcomes    9/10 (unchanged), invented frames 60 -> 56
     REVERTED ANYWAY: the red 3 was renamed "1" in 1843 frames.
     *** THE SCORECARD CANNOT SEE THAT. *** "moving balls named" counts
     whether a ball HAS a name; "invented numbers" only flags numbers
     outside the inventory. A wrong-but-valid name scores as success, so
     the run reported a clean sweep while two balls were confused.
  R3 NEXT ROUND IS THE INSTRUMENT, NOT THE ENGINE: add per-ball naming
     CORRECTNESS to tools/scorecard.py - anchor each of the 6 balls to
     its known resting position from bench_truth (static_balls plus the
     occupancy timeline in GOALS round 25) and score name-vs-truth per
     track-frame. Without it every naming number reported so far,
     including today's 80.0%, is unverified. THEN re-land the engine ->
     strat.detect change, which is correct and required.
  R3 SUPPORTING MEASUREMENTS (keep, both verified this round):
     - stripe_reading() is BLIND on this table: all 20 labelled 9s read
       SOLID (0.048-0.141) under the inner-62% window, overlapping the
       1s (0.000-0.068) - a stripe's white is at the POLES, which that
       window discards. Window 0.95 + v>150 separates cleanly:
       solids (n=73) max 0.211 | stripe 9 (n=20) min 0.418 | cue min
       0.903. CAVEAT: all 20 stripe crops are the same static ball in
       one pose - not 20 independent samples.
     - colour_refs.json still carries UNVALIDATED 2026-08-15 entries for
       balls NOT on this table (5,6,7,8,10-15). The phantoms match them:
       dropping them cut invented frames 302 -> 56. An unvalidated
       reference is worse than none.
  R4 ROUND 26 (NEGATIVE, reverted; the 170.6 pot is an R3 problem):
     Traced the last missed outcome end to end. The engine DOES track
     the potted ball into the pocket: id7 dies 14.2px from the centre
     (pocket_r 24.8) at peak 3007 px/s. It scores nothing because the
     track is UNNAMED and shots.py gates unnamed drops
     (elif n >= 1 or unnamed_pots) - deliberately, since a glove
     lifting balls out of a pocket faked a make in round 5.
     => the last missed pot is BLOCKED ON R3 NAMING, not on pot logic.
     Tried on the way: deleting UNNAMED_MIN_SPAN (250px measured over a
     track's WHOLE LIFE, which deleted id7 at 230px). MEASURED WORSE -
     strokes 10/10 -> 9/10, outcomes 9/10 -> 8/10, unexplained 0 -> 1,
     and 170.6 still missed. REVERTED. (The span gate is still wrong in
     principle - whole-life bbox is the wrong evidence - but it is not
     what blocks the pot, and removing it alone regresses.)
  R0 PHANTOM CLAIM CORRECTED - 30% was measured in the WRONG PLACE:
     that was RAW finder output. After prepare_detections only 0.1% of
     track-frames sit in a pocket zone (25/34821), and every in-zone
     track also moved >200px/s (real balls passing through). A pocket
     dead zone would be a near-no-op AND would risk the jaw traffic
     that killed rounds 13/14/17. Phantoms are real but rare: one
     lingering "est" ghost on the bottom-left pocket leather, visible
     in the 140s overlay. NOT the top target; deferred.
     Discriminator if ever needed: detector score separates cleanly -
     phantoms 0.30/0.36/0.49 vs real balls 0.86/0.87/0.86, and the
     STATIC 9 scores 0.86 with the real balls (so "never moves" must
     never be the test - it would delete a genuine resting ball).
  R3 IS NOW THE ONLY THING THAT MATTERS - it blocks the 170.6 pot, pot
     attribution (2/4), and leaves the static 9 unnamed (overlay at
     171.8 shows the 9 wrongly named "1" once the real 1 is potted).
     NEXT ROUND: solid-vs-stripe by WHITE FRACTION (measured on
     labelled crops: solids max 0.098, stripes min 0.189, split 0.143)
     landed TOGETHER with the measured colour refs, gated on the
     scorecard, pinned with a test. Refs alone are NOT safe: measured
     values put the 1 and 9 only 7.3 Lab apart (old refs 22.9), which
     is exactly the confusion seen at 171.8.
  R0 OLD CLAIM (superseded above) - PHANTOMS ARE 30% OF ALL DETECTIONS: three fixed
     non-balls are reported in hundreds of frames - bottom-left pocket
     leather ~(289,1370) x214, a pale felt mark ~(566,1107) x180, a
     dark object ~(223,800) x68 = 462/1534. Recorded in bench_truth
     phantoms_note. Static non-ball rejection is the next round.
  R3 COLOUR REFS HAVE AN OWNER NOW (tools/build_colour_refs.py):
     APP_DIR/colour_refs.json had NO writer in the tree - an orphan
     dated 2026-08-15 whose ball-4 entry recorded the purple 4 as NAVY
     (142,26,36), next to the real blue 2 - the "purple 4 guesses BLUE"
     misread the ensemble patches downstream. Measured drift on this
     table: 2 -> 41.2 Lab, 9 -> 22.9, 3 -> 22.5, 4 -> 11.6.
     NOT WRITTEN YET (law 4 - it changes live naming, so it gates):
     measured refs put the 1 and 9 only 7.3 Lab apart (old refs: 22.9),
     so whole-crop colour must NOT arbitrate 1-vs-9. White fraction
     must: measured solid max 0.098 vs stripe min 0.189 on labelled
     crops - a clean split at 0.143. Next round: land refs + the
     stripe-bit guard together, gate on the scorecard, pin with a test.
  R3 ROUND 24 (trap was a FALSE ALARM; corpus cleared for training):
     drew every box onto the full frames and looked. Frames ARE
     complete - each ball on the felt has exactly one box (166s: six
     balls, six boxes, incl. the orange 5 at (316,1233)); the
     "missing" balls were simply already potted. Also tried filling
     the supposed gaps from tracker state: it produced DUPLICATE boxes
     on balls the finder had already found (two circles on the same
     red, two on the same yellow) - reverted.
     CONFIRMS round 21 from the other side: the FINDER is healthy, the
     IDENTIFIER is the failure - which is exactly what this corpus
     trains. The 115 boxes stand.
     NEXT: train c8 on the corpus (tools/finetune_ballid.py), then gate
     same-batch vs c5 - promote only if bench naming rises AND held-out
     sessions do not regress (MEASUREMENT_CORE M2 rules).
  R3 ROUND 23 (corpus widened; COMPLETENESS TRAP FOUND): mining now
     spreads across the whole session (--limit/--min-gap) and accepts
     explicit times (--at), and tiles carry POSITIONS - which is what
     finally separated the 1 from the 5 (colour could not; at 166s the
     amber ball at (316,1232) is the 5, bottom-left, potted 5s later).
     Corpus: 115 boxes (cue 22, 1:14, 2:19, 3:20, 4:18, 9:20, 5:2).
     TRAP: a YOLO frame teaches about everything in it INCLUDING what
     is unmarked. Mined frames often show 6 balls when 7 are on the
     table - the missing one being exactly the ball the model already
     fails to see - so training on them would REINFORCE the miss. Only
     one verified-complete frame was written this round.
     NEXT (before ANY training): add a manual-add mode to
     tools/mine_ballid.py (Claude views the full frame, supplies the
     missing balls' positions), re-verify the earlier 109 boxes for
     completeness, THEN train c8 and gate same-batch vs c5.
  R3 ROUND 22 (corpus started): tools/mine_ballid.py mines the
     ENGINE'S OWN failure frames (moving-but-unnamed, or a number not in
     the session inventory), crops every detection, and writes labels
     into the existing TrainingStore YOLO layout - no new format.
     First batch: 87 crops from 12 failure moments, 57 labelled BY
     VISION (RULE 0) -> cue 11, 1:9, 2:9, 3:10, 4:8, 9:10. One frame
     DROPPED whole rather than labelled partially (two balls ambiguous
     even to Claude; a half-labelled frame teaches "real ball =
     background"). GAP: zero examples of the orange 5 - it is absent
     from these frames and is one of the never-read balls.
     NEXT: (a) mine frames where the 5 is on the table + a second
     session for variety; (b) target >=40 examples per ball; (c) train
     c8 on the combined corpus (tools/finetune_ballid.py); (d) gate
     same-batch vs c5 per MEASUREMENT_CORE M2 - promote only if the
     bench naming rises AND held-out sessions do not regress.
  R3 ROUND 21 (CADENCE RULED OUT; ROOT CAUSE IS THE MODEL): ran the
     bench with IDENT_EVERY=1 (6x the identifier passes). Naming was
     BIT-IDENTICAL: 75.7% named, 150 seen-unnamed, 636 estimate-only.
     Direct probe of the 170.6 shot: the finder finds 7 balls, the
     identifier reads 4 numbers, and one is a "7" - not on this table.
     Cropped and LOOKED (RULE 0): blue->2 correct, red->3 correct,
     yellow->9, and the "7" is the PURPLE 4. Balls 1, 4, 5 get no read
     at all. Reverted to IDENT_EVERY=6 (identical output, 20% slower).
     => R3 is a MODEL problem. Confusion pairs seen so far: 4->7 (dark
     colours), cue->9 (round 8), 3<->5 (Joe's original report), and
     non-reads of 1/4/5. NEXT: build the c8 retrain corpus by mining
     these exact failures from the dense sidecars (crop at the frames
     where a ball is seen-but-unnamed or read as a number not in the
     session inventory), label, train, gate same-batch vs c5.
  R3 ROUND 20 (built + reverted; DIAGNOSIS CORRECTED): implemented
     identity stitching (a fresh detection continuing a dead track's
     motion inherits its name; corridor test from the last REAL
     sighting, since coasting drifts x/y and death zeroes v). Bench:
     ZERO change (75.7% named, 8/10, 2/4) - reverted as unproven.
     Looked at what reaches the pocket on the failing pots:
       51.5s  id2 num=2 reaches 5px WITH REAL SIGHTINGS - no fragment,
              so round 19's "anonymous fragment" explanation was WRONG
       170.6s the ball reaching the pocket is num=-1 for the whole shot
       31.7s  attributed to ball 4; the truth is ball 9 (misread)
     => The gap is RECOGNITION, not track continuity: the app follows
     balls but cannot read WHICH ball while they are in play.
     NEXT (measurement, not a guess): the identifier runs every
     IDENT_EVERY=6 frames - only a handful of chances during a fast
     shot. Re-measure the bench with IDENT_EVERY=1 and compare naming;
     if cadence is the limit, buy it back (fp16 identifier / crop
     batching). If not, it is the model and the c8 retrain is the work.
  R4 ROUND 19 (built the spec, reverted - DEPENDENCY FOUND): the
     three-clause rule (in-zone AND vanished AND no hand) got ALL SIX
     misses right - incl. the 0.47R hand-catch and both jaw rattles -
     but credited only 1 of 5 pots (6/10 vs baseline 8). Traced per
     ball: on 4 of 5 pots the NAMED track stops short (the 2-ball's
     freezes at 1.8R, all ESTIMATE rows) and the final approach belongs
     to a NEW anonymous fragment, which _series also drops for being
     short (<250px span). So "did ball N reach the pocket" is false
     even though a ball did. The baseline scores better only by being
     identity-blind (anything vanishing near a pocket = pot), which is
     also why it mis-attributes half its pots.
     ORDER CORRECTED: R4 CANNOT close before R3. Naming is the FLOOR
     under the outcome rules, not polish above them. NEXT: identity
     continuity into the pocket - stitch a dying named track to the
     fragment that continues its motion (same direction/speed, <0.4s
     gap, <2 ball diameters), so the ball keeps its name to the drop.
     The three-clause rule is DONE and waiting; re-apply it after.
  R0/R4 ROUND 18 (built + reverted; produced the SPEC): implemented
     Joe's dead zones (enter a zone, do not come back out = potted;
     replacing the death-point/path-tail/bed-exit/lip-hover heuristics).
     Alone: calls 8->6 (the jaw filter still hides the ball entering).
     With the arrival gate: 5/10, attribution 0/4 - the entering track
     is DIM and unnamed, and its coast freezes ~1.8 radii short of the
     mouth (measured on the 2-ball pot: id7 sits at 44px = 1.8R, all
     ESTIMATE rows). MEASURED SEPARATION (closest approach in pocket
     radii, pots vs misses): 0.15 POT, 0.29 POT, 0.47 MISS (hand catch),
     0.54 POT, 0.82 POT, 1.35 miss, 1.48 POT, 1.63 miss, 1.74 miss,
     1.75 miss. NO distance threshold separates them - which is why
     every attempt traded one shot for another.
     SPEC for the next attempt (all three parts, none optional):
       potted = entered zone (<=1.5R) AND stopped being seen AND NOT
       hand-adjacent at the vanish (carried-ids are already in the
       stream since round 10; the 0.47R false case is Joe's hand catch).
     Keep ZONE_R ~1.5R; the arrival gate stays OFF until this lands.
  R0/R4 ROUND 17 (built + reverted, but it SETTLES the design): the
     arrival-gated jaw filter works as designed - engine tracker piped
     into prepare_detections via MotionTracker.live(), pockets marked
     "ball incoming" for 0.6s - and coverage improved (naming 75.7->76.0,
     estimates 19.6->18.6). But CALLS DROPPED 8->7: fixed 154.2, broke
     101.0 (false pot) and 130.2 (lost pot). CAUSE: seeing the ball as it
     drops keeps its track ALIVE INSIDE the pocket, so the outcome stage
     reads "resting" instead of "vanished into the pocket". Pot credit is
     currently built on LOSING sight of the ball - so better sight makes
     judgement worse. CONCLUSION: implement Joe's POCKET DEAD ZONES (R0)
     FIRST - a ball that enters a zone and does not come back out is
     potted regardless of whether it is still detected in the basket.
     Then the arrival gate is safe to re-enable (the branch is small:
     tracker.live(), hot_pockets param, HOT_S).
  R2 ROUND 16 (kept): fake strokes 5->1, unexplained 10->0, shots hold
     10/10. Two measured rules: (a) a stroke means the CUE MOVED - all
     10 real strokes move it, tossed-in balls never do (and hand-
     adjacency cannot catch a tossed ball, since no hand touches it
     once released); (b) a stroke means the cue WENT somewhere - real
     strokes >=263px, addressing/nudging 8-29px, gate at 150px.
     REMAINING fake: the 13.1s setup where Joe pushes the cue 210px
     with the stick - needs the STICK recognised (it is tracked as an
     unnamed mover); next R2 attempt.
  R3 ROUND 15 (kept): looked at the open-felt losses per RULE 0 - they
     were NOT missed balls. Crops show a ball in the glove, two empty
     pocket corners, and one prediction off the table on the FLOOR. The
     tracker was coasting into nowhere and counting it as ball motion.
     Coasts now stop at the bed edge: naming 72.1->75.7%, guessed
     motion 24.2->19.6%, total moving samples 3598->3241. COST: fake
     strokes 3->5 - with fewer ghost tracks diluting it, CARRIED_SETUP
     (0.5) no longer catches Joe's hand-setup episodes. NEXT: retune
     that threshold against the three known setup windows.
  R3 RE-AIMED AGAIN (2026-08-28, Joe caught a false premise): the
     "motion blur" story was WRONG. Cropped the pixels at last: the
     ball the engine loses is SHARP, sitting in the pocket jaw, and the
     lost samples measure a median 202 px/s (<0.5px smear at 1/500s).
     Joe fixed blur at the camera before these clips were shot. Real
     split of the coasted samples: 55% within 3 pocket radii (leather
     occlusion drops confidence, then the round-1 jaw filter discards
     them - the arrival-gated filter is the fix, now for the RIGHT
     reason) and 45% SLOW balls in open play (cause unknown - measure
     it, do not guess). Blur recovery is OFF the plan.
  R3 REDIRECTED (round 12, tools/naming_audit.py): the loss is NOT
     recognition. Of all moving-ball samples: 72.1% named, only 3.8%
     SEEN-but-unnamed, and 24.0% never seen at all - coasted estimates
     where the detector lost the ball to motion blur. You cannot name
     what you cannot see, so the FIRST R3 job is detection coverage on
     fast balls (targeted blur recovery - currently OFF after a 2026-08-23
     sweep failed its gates; re-attempt it aimed at moving tracks only,
     measured by the naming audit), and the c8 retrain is demoted to the
     3.8% tail. Worst offenders are track ids, not ball numbers - they
     are balls that never got a name at all during their flight.
  R3 NOTE (round 9, FAILED + reverted 2026-08-28): a session-inventory
     prior (reject numbers read far less often than the rest) made every
     line WORSE - it cannot separate a phantom from a real ball the
     detector rarely sees (the orange 5 has ~47 tracked states all
     session). The invented "11" is a RECOGNITION failure, not a
     counting one: it belongs to the c8 retrain, not a filter. Next R3
     attack: measure WHERE naming is lost (never-read vs misread vs
     outvoted) per ball, then mine crops for those exact cases.
  R3 OBJECT BALLS - every moving ball tracked AND correctly numbered;
     no invented numbers. Gate: >=95% of moving time named, zero
     invented. NOW: 74.5% named, "11" invented in 318 frames.
  R4 MAKE/MISS - outcomes correct incl. jaw rattles and hand catches,
     with EVERY pot attributed to the right ball number. Gate: 10/10
     calls AND pot attribution 4/4. NOW: 8/10 calls, 2/4 attributed.
     (Joe, 2026-08-28: "There shouldn't be 'pots by unnamed balls' by
     the satisfaction of object balls right?" - correct: an unnamed
     pot is a symptom of R3 failing, never an R4 requirement. The
     unnamed-pot code path in measure/shots.py is SCAFFOLDING and is
     DELETED when R3 passes - see ARCHITECTURE.md demolition ledger.)
  R5 LIVE PARITY - the same box on the camera; a live session scores
     the same as its own re-process; old tracker DELETED.
Run the ladder clip by clip: bench first (all rungs), then a cold
clip, then the library.

PHASES + GATES (Joe, 2026-08-28: "We can't just say 'see you in two
weeks'... break this into intermediate phases and milestones to prove
we're on the right track"). Gate numbers come from
`python tools/scorecard.py --publish`, scored against
docs/bench_truth.json (established by eye, NEVER edited to match the
app) and published to the phone's pinned STATUS view.
  P1 BENCH PERFECT (target 2026-08-30): 10/10 found, 10/10 outcomes,
     0 fake strokes in setup windows, 0 unexplained episodes.
     Baseline at plan time: 9/10, 8/10, 4 fake, 8 unexplained.
  P2 COLD CLIP (P1 + 1 day): watch an UNTOUCHED session by eye first,
     write its truth file, then run the engine cold: >=9/10 both lines,
     0 fake. Guards against tuning to one clip.
  P3 NAMES (P2 + 2-3 days): every moving ball correctly named >=95% of
     its moving time on both clips; ZERO invented ball numbers.
     (Today ~20% of real motion is unnamed; a nonexistent "11" appears.)
  P4 LIBRARY REBUILD (P3 + 1-2 days): every session regenerated from
     the box (not merged); 3 random sessions score >=9/10 on fresh
     vision watches; shot counts match.
  P5 LIVE (P4 + 3-5 days): a live-played session scores the same as
     its own re-process; the old tracker is DELETED.
REPORTING RULES: scorecard republished after EVERY round (wins and
regressions alike); a journal entry per round; a missed phase date is
posted ON the date with the reason - never silent slippage.

CAMPAIGN STATE (update every round; newest first):
- Bench = **session-20260824-220247** (pinned). Ground truth: 10 real
  strokes. Dossier: docs/design/bench-220247-watch-2026-08-28.md.
- **Round 8 (biggest win so far)**: velocity-aware association gate.
  The overlay proved the tracker manufactured 8 phantom balls per
  frame down a struck ball's path (model detected 6, sidecar carried
  14) and stranded the "cue" label on the first phantom. Fast balls
  now stay ONE track. Bench: tracks 14->7 at t=170.5, cue correctly
  named at its real position, physics gate 0.49 -> 0.14/1k.
- Rounds 1-7: jaw-phantom filter; path-tail + bed-exit pocket credit;
  chain-continuity (rebirth != resting); lip-hover drops;
  unnamed movers seen but pot-credit gated; pocket furniture dies by
  time; engine made hand/stick-aware offline (refresh_foreign).
- OPEN, NEXT (round 9+), all bench-verified by vision each round:
  (a0) DONE round 11: dense rows carry an 8th field - sighting vs
      coasted ESTIMATE (additive; every reader indexing 0..6 is
      unaffected, trails keep their blur coverage). R1's metric now
      demands a live sighting: honest cue tracking is 99.7%, and the
      debug overlay tags estimates 'est' so evidence frames never imply
      the app saw what it inferred.
  (a) episode layer now over-fires (21 episodes) and mints FALSE POTS
      for named balls that lose/regain names mid-flight - e.g. pots
      credited to "4" at 13.1s/29.5s and to a nonexistent "11".
      Outcome agreement is 6/7 on matched strokes but only 7 of 10
      strokes now match cleanly. Fix the naming churn before trusting
      any outcome number again.
  (b) a nonexistent ball number ("11") appears - identifier misread
      class; feeds the c8 retrain corpus.
  (c) Joe's POCKET DEAD ZONES design (his suggestion, 2026-08-28):
      make pockets explicit zones - a ball is potted when it ENTERS a
      zone and does not return quickly, and anything sitting in a
      zone is NEVER "on the table". This should REPLACE the ad-hoc
      lip-hover/bed-exit/furniture heuristics with one clean rule.
  (d) hand-context in engine output -> unlocks unnamed-ball pots,
      kills the glove-carry false pot, labels setup vs stroke.
  (e) window keying at the strike; ball-4 detection dropout at rest;
      the 9 never identified; aim-line artifacts.
  (f) THEN: other clips, marathon + library re-run, regeneration
      (shots.json written wholesale from the box), merge-machinery
      deletion, M3/M4 live promotion.
- Tools for every round: `tools/debug_overlay.py` (app beliefs drawn
  on the real video - the proof tool) and `tools/journal.py` (plain-
  language entry + images to the phone's Dev Journal).

## Tier 0b — Standing automatics (do these the moment they're armed)

- [ ] **Marathon 4 landing** (armed ~2026-08-28 00:30): read
  `scratchpad/m1_marathon4.log` FULL-GATE + REMERGE lines. Green ⇒
  update the measurement log in MEASUREMENT_CORE.md, bump the session
  stamp, write the What's New entry (accuracy chain: notify Joe).
  Red ⇒ diagnose against the slice run (0.25/1k), do NOT roll out.
- [ ] **Library dense rollout** (blocked on gate green): 16 remaining
  sessions through `measure/engine.reprocess` + `trails_merge
  .merge_into_session`, oldest-first, one at a time, guarded. Each
  session: gate slice → merge → stamp → GOALS line. Overnight work.
- [ ] **c8 identifier gate** (blocked on flip-mining, Tier 2): train →
  same-batch scorecard vs c5 → promote only through the gate.

## Tier 1 — Demolition (default work; order = payoff/risk; ledger: ARCHITECTURE.md §5)

1. [ ] **Shot identity** — stamp Joe's `stem@start-second` id into
   every record; kill the four time-proximity joins.
   a. `analysis_cache.py` add_shot/add_stroke/append_correction/
      append_action write `id`; SidecarReader._shot_for joins by id,
      time-tolerance demoted to legacy fallback (`:136`).
   b. controller stroke join (`controller.py:580` 25s wall-clock) and
      cached_shots dual-clock "key" (`:308`) → id.
   c. UI joins: live_page on_stroke_measured ±6s (`:1542`),
      stay_down (`:155`) → id.
   d. DELETE the tolerance constants + dual-clock key contract.
   Accept: a correction lands on the right shot with clocks skewed 10s.
2. [ ] **One timebase** — core-owned recording clock; sidecar meta
   carries t0; DELETE the four reconciliations (SidecarWriter._t0
   `analysis_cache.py:76`, controller rebases `:1766` `:1784`,
   shots_export.session_time_offset `:247`).
   Accept: session_time_offset returns ~0 unused; exports unchanged
   on a reference session (byte-diff shots.json trails windows).
3. [ ] **Clock snapshot** — core.clock {remaining,total,fraction,
   phase,enabled}; packet stops synthesizing (`controller.py:1516`
   reaches into `_run_seconds`); ring (`live_page.py:1434`) and bar
   read the same dict. DELETE per-widget denominators/status logic.
   Accept: break countdown renders identically in ring + bar.
4. [ ] **One reader** — `companion/server.py:47` read_shots calls the
   SidecarReader merge; DELETE its private ±0.2s rule.
   Accept: desktop and phone show identical outcomes on a session
   with corrections (test with a review-ranked correction).
5. [ ] **Geometry in meta** — controller writes table/H/corners at
   sidecar open (`controller.py:643` writes fps only); DELETE both 3s
   re-warmup derivations (`shots_export.py:28`, engine warmup stays
   for legacy files only).
   Accept: export of a NEW session does zero pipeline warmups.
6. [ ] **One shot log** — core publishes the reconciled shot list;
   timeline/list/DB-scoreboard/sidebar-count bind it. DELETE the four
   private copies incl. cross-widget `_shots` mutation
   (`live_page.py:1603`) and `session_summaries._sidecar_shot_count`
   private filter (`:35`).
   Accept: a correction updates every surface from one publish.
7. [ ] **Tracker demolition** (blocked on M4 gate): swap core's
   MotionTracker to authoritative; DELETE BallTracker
   (`vision/tracking.py:144`), vacancy/ghost bookkeeping
   (`pipeline.py:561`), trail-fade triplication (`pipeline.py:114`).
   The divergence counters in the health line are the evidence file.

## Tier 2 — Measurement quality (interleave with Tier 1 when GPU free)

- [ ] **The one Input→Output box** (Joe, 2026-08-28 — supersedes the
  "offline outcome judge" framing; NO more replay-only processing):
  shot detection + outcome judgment become stages INSIDE the
  measurement engine, one code path consumed by BOTH feeds. A clip's
  shots.json is simply the box's output for that video — no merge, no
  graft (that machinery gets DELETED on landing, per §5). Steps:
  (a) port the shot detector's gates onto dense tracks inside
  measure/ (the veto logic already reads the same tracks);
  (b) outcome stage: pocketed-into-which-pocket from 33ms tracks +
  pocket regions (pocket localization feeds this);
  (c) validate against Joe's review verdicts
  (`_eval/review_scoreboard.json` attachment rules) — the 3-ball
  false make is test case #1;
  (d) full-regeneration export: shots.json written wholesale from the
  box's output; delete trails_merge/arbitrate scaffolding;
  (e) M3/M4 then promote the SAME box onto the live feed.
  Prereqs: Tier 1 items 1-2 (shot ids + one timebase) so history
  rewrites can't mis-attach.
  PROGRESS 2026-08-28: stages 1-2 built + validated (measure/shots.py;
  the 3-ball case judges MISS); on-demand UI re-measure shipped
  (sidebar right-click → measure/job.py: engine→gate→merge, progress
  % in the status bar, deterministic, rules_v stamped, own process,
  presence-pause skipped on click). NEXT: hand-context + table/pocket
  geometry in engine output; box-vs-review-verdicts across the 13
  dense sessions.
- [ ] **3/5 flip mining**: scan dense sidecars for 3↔5 number changes
  on resting tracks; harvest crops at those timestamps; label; add to
  c8 training set. CPU-light scan; crop decode when GPU/disk idle.
- [ ] **M3 shadow sidecar**: plumb capture timestamps through
  `_on_detections_ready → ingest` (today borrows last process t);
  write `.analysis.shadow.jsonl`; exit criteria in MEASUREMENT_CORE §M3.
- [ ] **Pocket visual localization** (ladder rung 2): locate real
  pocket mouths vs geometric marks; feeds pot-credit gates.
- [~] **Engine-version stamp**: writing DONE (meta rules_v=2,
  2026-08-28); REMAINING: gate/scorecard refuses rules_v mismatches. Also: Aug-23 sessions (185550/191319/194542) gate red on
  overlaps — investigate EOS AF-rectangle contamination; they stay
  honest-sparse until then.
- [ ] **Shot recall audit**: false-negative hunt on dense output vs
  audio witness; extends the G5 precision work to recall.

## Tier 3 — Product (only when Tiers 0-2 idle, or Joe asks)

- [ ] iOS shot-clock + ball-tray replay overlay polish (data ships).
- [ ] Per-shot clips (design approved; gated on dense rollout).
- [ ] Overlay-baked video exports; data-saver 480p tier; desktop
  Details parity. Full list: FEATURES.md (do not duplicate here).
- SHELVED (do not resume autonomously): clips-player IG redesign.

## Tier 4 — Never idle (when everything above is blocked)

- Run the weekly full state-opinion audit (map style,
  `docs/design/state-opinion-map-2026-08-27.md`); update the §5
  scoreboard. Growth = incident.
- Write the missing pinning test for the NEXT demolition item (each
  merge needs its acceptance test before the merge).
- Tighten one debt item from MEASUREMENT_CORE.md §3 with a measured
  before/after.
- Re-verify guards: presence-pause, idle throttle, recording aborts
  (tools/health_check.py --quick).

---
*Maintenance: completed items move to a GOALS log line and are deleted
from here. New work enters at the tier its risk deserves, never at
the top by recency. If this file and GOALS disagree about status,
GOALS (the log) wins and this file gets corrected.*
