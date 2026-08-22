# Root-cause miss analytics — research + forensics (2026-08-22)

Derived from frame-forensics on three of Joe's real misses
(20260820-005048-recovered@233, @74, @96) plus instructional theory.
Joe's standing rule applies: never oversimplify the causal claim.

## Per-shot forensics

### shot233 — 20260820-005048-recovered@233, t=232.73–240.13s. The purple 4-ball cut very slightly to the bottom-left corner pocket; cue ball 26.9" from the object ball, object ball 34.9" from the pocket. Outcome: overcut, object ball caught the short-rail jaw and rattled out.

**Intended pocket:** Bottom-left corner pocket (mouth between jaw tips (178,1347) and (207,1373), centre (192.5,1360); measured mouth 38.9 px = 3.42"). Evidence: (1) the cue ball is up-table of the 4, so the 4 can only be driven downward — that leaves only the two bottom corners; (2) the bottom-right corner would need a 66° cut with the cue ball passing 23 px to the other side of the 4, whereas the natural cue-to-object centre line (17.33° left of down) is within 3.1° of the line to the bottom-left corner (14.20°) — this is nearly a straight-in shot; (3) the measured stick points down the left side of the table, within 0.7° of the pot line; (4) the ball actually travelled to the bottom-left corner and hit its jaw.

**Visual account:** Work dir: C:/Users/Joe/AppData/Local/Temp/claude/c--Users-Joe-Documents-GitHub-BilliardsTrainer/3b1cd652-ea12-49ea-a5ac-97248d95851f/scratchpad/miss/shot233 (330 frames raw_06903–raw_07232, plus ANNOTATED.png, sheet_launch.png, contact_seq.png, corner_seq.png, pocket_zoom.png, stick_zoom.png).

CALIBRATION FIRST — the supplied transform is broken for this file. hinv applied to the stated rect box x∈[135,465], y∈[145,1128] lands on a 346x983 px region of the video that is NOT the table (aspect 2.98). Measuring the cushion noses off the frames directly gives bed = x∈[180,747], y∈[238,1375] in video px: 567 x 1137, aspect 2.006, i.e. a real 2:1 playing surface. Long-rail diamond spacing 143 px, ball diameter 25.5 px; 8*2.25/(25.5/143) = 100.9 in, so this is a 9-FOOT table (50x100" bed), not 7ft. All work below uses 11.40 px/in (anchored on the 2.25" ball). Angles are scale-free; if the table really is a 7-footer, divide every inch figure by 1.28.

ADDRESS (f6960–6979, t=232.10–232.73). Cue ball white at (381.3, 682.3). Seven object balls on the bed; the only one on the cue ball's side of the table and down-table of it is the purple 4 at (290.0, 974.8). Shooter is at the top of the frame, stroking down-and-left. Stick static for 0.27 s, then a 6.7" backswing (tip y 653 -> 577 over f6967–6976), then a 0.10 s forward stroke.

LAUNCH (f6980–6984). Cue ball leaves at f6980; motion streaks at f6982 (344.0, 794.9), f6983 (324.4, 855.1), f6984 (305.2, 913.8). Those three plus the address centre are collinear to 0.15 px residual over 232 px of travel — a dead-straight cue-ball path, no swerve. Speed 62.5 px/frame = 1875 px/s = 164 in/s = 9.3 mph (firm).

CONTACT (between f6984 and f6985, t≈232.92). At f6985 the cue ball is at (290.6, 950.3), one ball diameter up-and-right of where the 4 sat; the 4 is already gone with a blue motion streak trailing down-left. contact_seq.png shows the cue ball striking the 4 on its image-left shoulder — a near-full but clearly off-centre hit.

OBJECT-BALL PATH (f6987–6992). (264.1,1104.9), (252.9,1160.1), (241.1,1212.2), (233.7,1267.0), (226.3,1317.3) — a straight line back-projecting to x=286.0 at the 4's start y (actual 290.0).

MISS (f6992–6996, corner_seq.png). The 4 arrives at the bottom cushion at x≈218, strikes the short-rail jaw ~1.2" outside the pocket, rattles at (192–209, 1343–1361) for ~4 frames, then rebounds and rolls back UP the left long rail (f6998 (203,1343) ... f7010 (199,1261)). No ball fell.

CUE BALL AFTER CONTACT. From (290.6, 950.3) at f6985 to (203.7, 816.8) at f7010 — it reverses up-table. Cue/object separation angle 136.8° (a stun ball would give ~90°), so this was struck well below centre: heavy draw.

SIDECAR CHECK. tracks_at() returns 7 balls of which 5 (numbers 5,6,7,8,9) are frozen at identical coordinates for the whole shot — stale ghosts. Only tracks 0 and 4 move, which does correctly identify the 4 as the object ball. Their coordinates are usable only up to an undocumented affine (video ≈ 1.03*rect + (110.5, 157.3)); the documented rect extent/hinv is simply wrong for this recovered file.

**Geometry:** Angles are "degrees left of straight-down-the-image" (the shooter faces down-image, so image-left is his right).

DISTANCES: cue->object 306.5 px = 26.9"; object->pocket 397.3 px = 34.9".

REQUIRED: object ball must depart at 14.20°. Ghost-ball centre (296.3, 949.9). REQUIRED cue-ball line 17.61°. Required impact parameter b = +1.50 px = 0.059 ball-widths, i.e. 94% full, cut +3.36° (a hair of right-hand cut).

DETECTED AIM LINE: 16.90°, passes 2.75 px right of the cue-ball centre. Would give b = -2.29 px, cut -5.13°, object ball departing 22.03° -> into the LEFT LONG RAIL at y≈1215, about 11" short of the pocket.

MEASURED CUE STICK: 16.93°, from a 14-point sub-pixel track of the cue tip through backswing and forward stroke (rms 0.21 px). The tip is at ball height so this is directly comparable to the ball-centre plane. (A shaft-axis fit gives 16.44°, but the shaft is elevated and perspective biases it ~0.5°; the tip track is the valid measurement.) Backswing and forward-stroke tip lines differ by only 0.32° — the stroke itself is straight.

ACTUAL CUE-BALL LINE: 18.19° (address centre + 3 streak centroids, residual 0.15 px).

ACTUAL CONTACT: b = +4.61 px = +0.180 ball-widths, 82% full, cut +10.34° — struck the 4 on its image-left side. Required was 0.059 ball-widths / 3.36°, so he took roughly 3x the intended amount of ball off the wrong... off the correct side but far too much: overcut by 6.98°.

OBJECT-BALL DEPARTURE: 10.11° measured vs 14.20° required = 4.09° too far right. (Pure ghost-ball geometry off the actual cue-ball line predicts 7.85°; the 2.26° difference is cut-induced throw pulling the 4 back toward the cue-ball path — the expected direction and a normal magnitude for a 10° cut.)

MISS MAGNITUDE: the 4's centre crosses the pocket-mouth segment at s=1.356 (0..1 is inside the jaws) — 13.9 px = 1.22" beyond the short-rail jaw. Versus the pocket centre that is 33.4 px = 2.93" along the mouth, or 2.49" (1.11 ball-widths) measured perpendicular to the ball's path. Missed on the short-rail (bottom) side, i.e. the ball went wide/right of the pocket.

TOLERANCE BUDGET (why sub-degree errors are fatal here): effective aperture perpendicular to the approach is 34.5 px, leaving the ball centre only ±4.42 px = ±0.39" of play at the pocket = ±0.65° of object-ball direction. Aim error is amplified by d(cue->object)/ball-diameter = 306.5/25.65 = 11.9x, so the cue-ball line had to be within ±0.054°. Even assuming a generous 4.5" mouth the tolerance is only ±0.12°. Measured errors: stick -0.68° (12x tolerance), delivered +0.58° (10x), squirt 1.26° (23x).

**Aim-line verdict:** The detected aim line is CORRECT — it is not a detection error, and Joe's read of it is right.

Detected aim line 16.90° vs the cue stick measured independently off the frames (sub-pixel cue-tip track, 14 points, rms 0.21 px) at 16.93°. Agreement 0.03°. It also passes within 2.75 px (0.24") of the cue-ball centre. There is nothing wrong with the detector on this shot.

And the aim line's prediction is exactly what Joe said: delivered along 16.90°, the cue ball would contact the 4 on its right side (b = -2.29 px), the 4 would depart at 22.03°, and it would strike the LEFT LONG RAIL at y≈1215 — roughly 10-11" up-table of the corner pocket. "The aim line indicates I should be hitting the long rail" is confirmed by an independent measurement of his actual cue.

So the answer to the question posed is the second option: he did not deliver the cue ball along the line the cue was pointing. The cue ball launched at 18.19°, which is 1.26° to the shooter's right of the stick line — and that flipped the contact from the right side of the object ball to the left side, turning a predicted long-rail undercut into a real overcut.

**Root cause:** Link (b) DELIVERY is the primary failure, with a smaller opposing (a) AIM error that did not cancel it. Link (c) CONTACT is clean — the cue ball struck exactly the fullness its launch line dictated; there is no separate contact error.

(a) AIM — 0.68° of error, 12x tolerance. The stick was pointed 16.93° when 17.61° was needed, i.e. aimed to hit the 4 slightly on its right side. On its own this sends the 4 into the long rail ~11" short of the pocket. This is real but it is the SMALLER error and it points the opposite way from the actual miss.

(b) DELIVERY — 1.26° of squirt, 23x tolerance, the dominant term. The cue ball left the tip 1.26° to the shooter's right of the cue's line. The mechanism is measurable in the frames: extrapolating the cue-tip track to the cue ball's y gives a tip line 3.26 px = 0.25 ball-radii to the shooter's LEFT of the cue-ball centre — about half a tip of left-hand english. Left english deflects (squirts) the cue ball to the shooter's right, which is precisely the direction and roughly the expected magnitude (~1-2° for half a tip on a normal shaft). It is squirt, not swerve: the cue-ball path is straight to 0.15 px over 232 px, so the deviation is at the tip, not on the cloth. The stroke itself was straight — backswing and forward tip lines differ by only 0.32° — so this is a tip-placement/spin problem, not a steering problem.

NET: aim -0.68° plus squirt +1.26° = +0.58° left of the pot line. Because a 26.9" cue-to-object distance amplifies aim error 11.9x, that 0.58° became 6.98° of excess cut (10.34° actual vs 3.36° required), and the 4 left 4.1° too wide and finished 2.49" (1.1 ball-widths) right of the pocket centre — into the short-rail jaw. That is the overcut Joe describes.

(d) SPEED/other — contributory, not causal for direction. 9.3 mph with heavy draw (cue/object separation 136.8° proves a low hit). Two consequences: side spin at that speed maximises squirt, and the firm pace meant a ball that arrived 1.2" outside the jaw rattled out rather than being dragged in. The real coaching point is that on a near-straight 27"/35" shot into a 3.4" pocket, adding side spin spends ~1.3° against a ±0.05-0.12° budget — the english bought nothing and cost the shot.

**Signals that would diagnose it:**
- SQUIRT ANGLE = detected aim line minus cue-ball launch direction. Here +1.26 deg (launch 18.19 vs aim 16.90). This single number diagnoses the shot: the aim line was right, the ball did not follow it. It needs only the existing aim detection plus a line fit through the cue ball's first 3-4 post-launch positions (streak centroids work; residual was 0.15 px).
- PER-SHOT POTTING TOLERANCE = ((effective_pocket_aperture - ball_diameter)/2) / dist(object,pocket) / (dist(cue,object)/ball_diameter). Here +/-0.054 deg. Without this, a 0.6 deg error looks negligible; with it, every measured error is 10-23x tolerance and the shot is legible as 'no line on this table makes it'. Should be attached to every shot as the yardstick.
- AIM AMPLIFICATION FACTOR = dist(cue,object)/ball_diameter. Here 11.9x. Flags shots where tiny delivery errors are fatal, and is the reason this near-straight-in shot was harder than it looked.
- TIP LATERAL OFFSET AT IMPACT = perpendicular distance from the cue-ball centre to the extrapolated cue-tip track. Here 3.26 px = 0.25 ball-radii on the shooter's left = about half a tip of left english. Predicts both the direction and rough magnitude of the squirt, and is measurable at 30 fps by tracking the ferrule end (the deepest bright low-saturation run) through the forward stroke.
- CUE/OBJECT SEPARATION ANGLE after contact. Here 136.8 deg vs ~90 deg for stun, which detects vertical tip offset (heavy draw) from overhead video without ever seeing the tip on the ball.
- ACTUAL vs REQUIRED IMPACT PARAMETER and CUT ANGLE: b = +4.61 px (0.180 ball-widths, 82% full, cut +10.34 deg) against required +1.50 px (0.059 ball-widths, 94% full, cut +3.36 deg). Directly names 'overcut' and by how much.
- THROW ESTIMATE = measured object-ball departure minus ghost-ball prediction off the actual cue-ball line. Here 10.11 - 7.85 = +2.26 deg toward the cue-ball path. Confirms the contact model is closing and quantifies a term that is otherwise invisible.
- POCKET-MOUTH CROSSING PARAMETER s: where the object-ball centre line crosses the segment between the two jaw tips, normalised 0..1 inside. Here s = 1.356, i.e. 1.22 in outside the short-rail jaw. Converts 'miss' into 'which jaw and by how much', and distinguishes a rattle from a wide miss.
- BACKSWING vs FORWARD-STROKE TIP LINE divergence. Here only 0.32 deg, which exonerates the stroke path and isolates the fault to tip placement rather than steering.
- DATA-QUALITY GUARDS this session needed: (1) validate shots.json 'transform' by projecting the rect bed corners into the frame and checking the implied aspect ratio (should be 2.00; this file gives 2.98) and cushion alignment; (2) flag tracks whose position variance is exactly zero across a whole shot (5 of 7 balls here were frozen ghosts); (3) derive px/inch from the detected ball diameter and cross-check against diamond spacing (25.5 px ball and 143 px spacing say 9 ft, not the assumed 7 ft).

(confidence: high)

### shot74 — 20260820-005048-recovered@74, t=73.83–84.47s. Nearly straight-in 1-ball (yellow) into the bottom-right corner on a 9-ball layout (all nine object balls still up, so the 1 is the legal ball). Cue ball 18.2 in behind the 1; 1-ball 47.6 in from the pocket; 65.8 in of total shot length. Firm draw stroke. Result: the 1-ball hit the right long rail 5.4 in short of the corner, rattled in the jaws and came back out. No ball pocketed.

**Intended pocket:** Bottom-right corner pocket (the corner at rectified table coords (51.11, 102.22) in). Evidence, five independent lines: (1) the 1-ball is the legal ball — all nine object balls are still up in the end-state frame; (2) the cue-ball-to-1-ball line and the 1-ball-to-BR-corner line differ by only 1.08°, i.e. this is the natural near-straight-in shot; (3) the measured cue-stick axis at address (59.48°) points within 0.30° of the ghost-ball line for that pocket; (4) the 1-ball actually travelled straight at that corner and struck its jaw; (5) no other pocket is available — both side pockets and the top-right corner are up-table of the 1-ball and would need a >60° cut back toward the shooter, and the bottom-left corner sits behind the cue ball's line. The path from the 1-ball to the BR corner is also free of obstructing balls (nearest ball clears the line by 9 in).

**Visual account:** Address (t=73.23–73.83, 19 frames): cue ball dead still at table coords (18.78, 44.86) in; yellow 1-ball dead still at (27.92, 60.63) in (both rest positions repeatable to ±0.004 in). Shooter is at the top-left, gloved bridge hand on the cloth, cue pointing down-right. The red 3-ball sits 4.4 in down-left of the cue ball; the line from the 1-ball to the bottom-right corner is completely clear.

Backswing and stroke: the ferrule blob tracks back from (163.4, 413.9) px to a pause at (145.7, 383.9) px at t=73.63, then forward. Backswing bearing 59.48°, forward-stroke bearing 59.54° — the tip runs on one straight line, no loop.

Launch (t=73.865): cue ball first moves at f2215. Four clean pre-contact samples at 73.898/73.932/73.965/73.998 give bearings from rest of 60.53/60.62/60.41/60.35°. A total-least-squares fit through the rest point plus those four gives 60.35° with a maximum residual of 0.23 px (0.02 in) over 13 in of travel — the cue ball path is dead straight, so there is no measurable swerve.

Contact (t≈74.02, frames f2219–f2221 in S_contact_strip.png): the blurred cue ball arrives on the 1-ball's upper-left, the white ball goes sharp (it stops almost dead), and the 1-ball streaks away down-right. Cue-ball centre at contact (288.27, 634.46) px.

1-ball flight: 15 clean detections from t=74.165 to 74.632, bearing from rest 58.45° ± 0.31° (TLS 58.66°, max residual 0.16 in). It runs parallel to but right of the pocket line the whole way.

Impact and rattle (S_pocket_strip.png, f2238–f2244): at t=74.665 the ball centre is at (50.0, 96.9) in — touching the right long rail 5.4 in short of the corner, arriving at 62 in/s (5.2 ft/s). The x-component of velocity flips sign at t=74.68 (bounced off the right rail/jaw, normal along x), then the y-component flips at t=74.78 (bounced off the bottom jaw), then it exits up-left. It rolls all the way back up-table and stops at (23.8, 53.2) in.

Cue ball afterwards: draws straight back along bearing 232° for 15.1 in and finishes frozen against the red 3-ball at (16.9, 47.3) in. End-state frame at t=85.37 confirms nine object balls still on the table.

**Geometry:** SCALE NOTE: the supplied transform is a near-identity fallback and its declared bed extent (x∈[135,465], y∈[145,1128], 2.98:1) does not correspond to this video's table. I rebuilt the rectification from the cushion-nose shadow lines and forced them to a true 2:1 rectangle. Measured table: 51.1 × 102.2 in between cushion noses — a 9-footer, not the 7ft/46in in the brief. Scale 10.800 px/in, ball diameter 24.30 px = 2.25 in (verified against the cue ball's edge profile). All angles below are scale-invariant.

DISTANCES: cue→1-ball 18.23 in; cue→ghost-ball 15.98 in; 1-ball→pocket 47.62 in.

REQUIRED: ghost-ball centre at (26.83, 58.66) in. Required cue-ball bearing 59.78°. Required cut angle −1.08° (99.98% full — effectively dead straight).

DELIVERED: cue-stick axis 59.48° (median of 20 address frames, sd 0.06°). Cue-ball launch 60.35°. Tip contacted the cue ball 0.167 in left of the vertical centre line (15% of ball radius, roughly a quarter tip) — measured as the perpendicular offset of the shaft axis from the ball centre, 1.80 px, stable across all 20 frames.

CONTACT: line of centres 56.85° vs 60.86° required → −4.01°. Actual cut angle +3.50° vs −1.08° required, i.e. over-cut by 4.58°. Fullness 99.81% (vs 99.98% required). Contact point 0.137 in off the centre line on the side away from the right rail, versus 0.042 in the other way that was needed → 0.18 in of contact-point error; the cue ball's centre at contact sat 0.16 in from the ghost position.

THROW: measured 1-ball departure 58.66° vs geometric line of centres 56.85° → +1.81° of throw (spin-induced throw from the left english plus cut-induced throw, both acting the same way, both toward the pocket). Throw recovered about 45% of the geometric error.

MISS: net 1-ball bearing error −2.20° → 1.83 in lateral miss at 47.62 in, on the long-rail side of the pocket. Ball centre reached the right rail at (50.0, 96.9) in, 5.36 in short of the corner, at 62 in/s.

SENSITIVITY: error amplification from cue-ball direction to object-ball direction = Lghost/D = 15.98/2.25 = 7.10×. One degree of cue-ball direction error = 5.90 in of miss at this pocket. Pot window ≈ ±0.19° of cue-ball direction, equivalently ±0.052 in of ghost-ball placement. Delivered net error +0.56° = 3.0× the window (aim alone 1.6×, squirt alone 4.7×).

SPEED: cue ball 93 in/s (7.75 ft/s, 5.3 mph) at contact; 1-ball 67 in/s off the contact, still 62 in/s at the rail. Cue ball drew back 15.1 in and froze on the 3-ball. Speed was appropriate for the position play but firm arrival made the jaw unforgiving — a ball clipping a jaw at 5 ft/s rattles out where a softer roll might have dropped.

**Aim-line verdict:** The sidecar reports aim: null for this shot, so there is no detected aim line to score. I measured the true stick axis directly instead: 59.48°, from a colour-and-elongation-constrained line fit over 20 address frames (spread 59.42–59.62°, sd 0.06°, verified visually in E_shaft4_2211.png). That is 0.30° off the required 59.78° — the aim itself was very nearly right.

Why the detector produced null is visible in the frames and I reproduced the failure. The usable shaft is short and broken into two pieces by the black bridge glove (segments spanning y 221–356 px and 370–401 px, a ~14 px gap), the butt runs off the top-left frame edge behind the player's cap, and the shooter's bare forearm lies immediately alongside the shaft at a shallower angle. My own first two naive bright-pixel line fits over exactly that region returned 54.2° and 25.6° — 5° and 34° wrong — because the forearm got swallowed into the mask. Only after requiring neutral grey/white colour (excluding skin tones), an elongation test, and iterative outlier rejection did the fit lock onto the true 59.48°. A Hough/brightness stick detector on this frame will either return a line ~5° off or fail its own consistency check and emit null, which is what happened.

**Root cause:** DELIVERY. The chain measures out as: aim −0.30°, delivery +0.87°, net cue-ball direction +0.56°.

The stick was pointed within 0.30° of the correct ghost-ball line — better than most players manage — but the tip landed 0.167 in left of the cue ball's vertical centre (15% of ball radius, about a quarter tip) on a shot that needed dead-centre draw. That unintended left english squirted the cue ball 0.87° to the right of the stick line. Three independent observations confirm the same left-of-centre contact: (a) the direct geometric measurement of the shaft axis versus the ball centre, stable over 20 frames; (b) the cue ball leaving 0.87° right of the stick line, which is the direction left english deflects a ball, and a magnitude consistent with a normal (non-low-deflection) cue at 0.30R offset; (c) the +1.81° of throw on the object ball, also to the right, which is what left english does on a near-full hit.

The cue ball's path was straight to within 0.02 in over 13 in, so there was no swerve to bring it back — the cue was level and the squirt went uncorrected.

Then the geometry did the damage. With the cue ball only 16 in from the ghost position and the pocket 47.6 in beyond the object ball, aim error is amplified 7.10× into object-ball error. The pot window on this shot is a brutal ±0.19° of cue-ball direction (±0.052 in of ghost placement). The 0.87° of squirt alone is 4.7× that window; the 0.30° aim error is 1.6×. They pointed opposite ways and partially cancelled, but the squirt dominated and set the sign: net +0.56°, 3.0× the window, producing 4.01° of line-of-centres error. Throw gave 1.81° back, leaving 2.20° = 1.83 in of miss — just outside the ~1.1 in half-window of the pocket, so the ball caught the near jaw instead of missing cleanly.

Counterfactual: same stick line, dead-centre hit, and the line of centres comes out at 63.01° (1.79 in geometric miss the other way), which cut-induced throw pulls back to roughly 0.5–0.9 in — inside the window. The unintended left english is the difference between this dropping and this rattling out.

Contributing but not primary: the firm draw speed. He needed pace for the 15 in of draw, but arriving at the jaw at 5.2 ft/s guaranteed a rattle-out rather than a hang.

DATA CAVEAT: the sidecar is not merely noisy here, it is unusable for this shot, and I ignored it in favour of the frames. The cue ball is carried as an unnumbered track (n=−1) that goes inactive at t=73.9 and freezes at its address position for the whole shot; a numbered n=0 track only appears at t=74.3, which is 0.44 s after the cue ball started moving and 0.27 s after contact — so there is no cue-ball launch segment at all, and the single most diagnostic measurement on this shot is simply absent. Separately, the n=1 trail samples lead the real object ball by 0.19 s of travel: unshifted they sit up to 16 in (mean 13.5 in) away from the ball's true position during the fast phase, and applying a +0.19 s shift collapses that to a 1.05 in mean. The shot's start/end boundaries do line up with the video, so it is the trail samples specifically that lead — either a timestamp lead or a forward-predicting tracker. Any velocity, departure angle, or moment-of-contact derived from those trails is wrong.

**Signals that would diagnose it:**
- tip_offset_at_address: perpendicular distance from the fitted cue-shaft axis to the cue-ball centre. Measured 0.167 in (1.80 px), 15% of ball radius, on the left. Stable to ±0.15 px across 20 frames. This single number is the root cause and is directly measurable at address, before the stroke even lands.
- squirt_angle = cue_ball_launch_bearing − stick_axis_bearing = 60.35 − 59.48 = +0.87 deg. Requires a shaft fit plus a launch fit; both are cheap. Sign matches the tip offset side, so the two cross-validate each other.
- aim_error = stick_axis_bearing − required_ghost_bearing = 59.48 − 59.78 = −0.30 deg. Separating this from squirt is what distinguishes an aiming fault from a stroke fault; reporting only the net cue-ball error would have mislabelled this shot.
- cue_path_straightness: max perpendicular residual of a TLS fit through the cue ball's rest position and its pre-contact samples = 0.23 px (0.02 in) over 13 in. Near-zero means no swerve, which proves the squirt went uncompensated and rules out a curve/elevation explanation.
- shot_amplification = cue_to_ghost_distance / ball_diameter = 7.10x, and the derived pot_window = ±0.19 deg of cue-ball direction / ±0.052 in of ghost placement. This is a per-shot difficulty number computable from the layout alone, before the stroke. It is what makes a 0.87 deg stroke error fatal here and harmless on a short shot.
- object_ball_bearing_error = −2.20 deg and lateral_miss_at_pocket = 1.83 in, with the side flagged (long-rail side). Computed from a 15-sample TLS fit on the object ball's flight, max residual 0.16 in.
- throw = measured_OB_bearing − geometric_line_of_centres = 58.66 − 56.85 = +1.81 deg. Sign independently corroborates the left english, and the magnitude explains why the miss was 1.8 in rather than the 3.3 in pure geometry predicts.
- first_rail_contact_point and rattle signature: ball centre reached (50.0, 96.9) in — the right long rail, 5.36 in short of the corner. The rattle is detectable as sign flips in single velocity components: x flips at t=74.68 (right jaw), y flips at t=74.78 (bottom jaw). This classifies 'rattled the jaws' versus 'missed clean' automatically.
- object_ball_arrival_speed = 62 in/s (5.2 ft/s) at the jaw. Pocket forgiveness shrinks with arrival speed; pairing miss magnitude with arrival speed separates 'shot was outside the window' from 'shot was inside the window but hit too hard to drop'.
- post_contact_cue_reversal: draw distance 15.1 in and rebound bearing 232 deg versus the reversed line of centres. Gives a second, independent estimate of contact fullness that does not depend on the object ball's track at all — useful exactly when the sidecar loses the object ball to motion blur.
- sidecar_vs_frames health checks that would have flagged this file as untrustworthy before any analysis ran: (a) cue ball carried as an unnumbered track that goes inactive at shot start and never reacquires until 0.27 s after contact — i.e. zero cue-ball launch samples; (b) trail-sample time lead of +0.19 s versus video PTS, detectable by cross-correlating trail positions against frame-differenced ball centroids (mean error 13.5 in unshifted vs 1.05 in shifted); (c) the supplied transform is a near-identity fallback whose declared bed rect is 2.98:1 while the real table images at 2.07:1 — a simple aspect-ratio assertion on the rectified bed would have caught it.

(confidence: high)

### 20260820-005048-recovered@96 (t=95.73–103.60 s, outcome "miss", 0 pocketed). Long, near-straight cut on the 1-ball (yellow) from the head end into the far/foot right-hand corner. Cue ball 16.3 in from the object ball; object ball 46.7 in from the pocket; total shot length 63 in. Aim line detected at q=1.0 (t=95.33).

**Intended pocket:** Bottom-right corner of the frame — the far/foot-end right-hand corner, diagonally opposite the shooter. Evidence: (1) the object ball departed at 64.74° and arrived within an inch of that pocket's jaw; (2) the required cut to that pocket is only 2.65°, i.e. the cue ball, the 1-ball and that pocket are almost in line — the natural read; (3) the stick at address (63.35°) sits between the straight-through line (63.93°) and the required ghost line (64.30°), i.e. squarely on this shot; (4) no other pocket is playable — the right-middle needs a ~70° cut, the bottom-left a ~50° cut the wrong way, and the object ball's line is nowhere near either; (5) the path to that corner is clear (nearest ball 4.9 in off the line).

**Visual account:** ADDRESS (t=93.3–95.6): Joe is at the head end, upper-left of the portrait frame, gloved bridge hand on the cloth, cue butt off-table to the upper-left, tip a few inches behind the cue ball. Cue ball sits alone in the upper-left quadrant at rect (231.0, 470.4). The 1-ball (yellow) sits near the centre of the table at rect (320.2, 659.9), down and to the right of the cue ball. Nothing intervenes: the 7-ball (dark maroon, rect 406,691) lies 4.9 in off the intended object line and the 9-ball 8.0 in off it. The stick lies almost exactly along the cue-ball/1-ball line, pointing down-right toward the far corner.

BACKSWING/DELIVERY (t=95.17–95.81): Per-frame RANSAC fits of the shaft centreline (rms 0.014 in over a ~12 in span — a genuinely straight rod) show the stick angle pinned at 63.27°–63.55° for the entire address, pause and forward stroke. The cue ball is at rest through t=95.775 and first moves at t=95.808, so tip contact is at t≈95.79.

LAUNCH (t=95.81–95.94): The cue ball streaks down-right at 94.8 in/s (7.9 ft/s). At t=95.940 the motion-blurred white streak is arriving on the upper-left shoulder of the yellow ball. Frames 95.87–96.14 show the shaft continuing to rotate to 64.6°–65.3° — but that is entirely AFTER the ball has gone (pure follow-through swinging across the line, ~1.8° in 100 ms; a mechanics flag, not causal here).

CONTACT (t≈95.97): At t=95.970 the cue ball is superimposed on the 1-ball's position and the yellow ball is already streaking away down-right. Overlaying the computed ghost-ball circle shows the cue ball arriving just up-and-right of it.

OBJECT BALL PATH: The 1-ball leaves at 64.74° (14-point line fit, rms 0.056 in; the fitted line passes within 0.035 in of the ball's resting centre — a clean consistency check) and runs 4 ft to the foot cushion. At t=96.64 the zoomed corner frame shows it hard against the foot cushion immediately to the left of the pocket leather — it does NOT enter. It rebounds in a clean mirror (65.6° in / 65.6° out), runs up to the right cushion at rect x≈597–599 (t≈96.85, exactly one ball radius off the measured nose line — an independent calibration check), rebounds again and rolls diagonally back up-table for ~2.5 s. At t=99.44–99.58 Joe's gloved hand reaches in and catches the still-rolling ball near the left-centre and repositions it. (The sidecar/track data imply a "collision with the 3-ball" here; the frames show it is his hand. The 3-ball never moves.)

CUE BALL AFTER CONTACT: It reverses and draws back up the aim line ~14 in to rect (276.6, 526.7) — a firm draw stroke on a near-full hit. Joe picks it up at t≈101.8 and re-spots it near its original position.

**Geometry:** CALIBRATION FIRST (the brief's numbers are wrong and would corrupt everything): the stated bed extent x∈[135,465], y∈[145,1128] is 2.98:1 — physically impossible. The true nose-to-nose bed in the SAME rect space is x∈[63.4, 610.6], y∈[66.7, 1204.8], found two independent ways: cushion-shadow brightness troughs (sub-unit parabolic fits) and the object ball's own rail-rebound turning points (agree to 0.2 in). The rect space is anisotropic by 3.9%: 12.436 units/in in x, 12.933 units/in in y — confirmed because ball diameter measures 28.53 x-units and 29.68 y-units (ratio 0.961) exactly matching the ratio needed to make the bed a true 2:1 rectangle. Result: bed = 44 x 88 in — an 8-ft table, not a 7-ft/46-in one. All angles below are computed in true-inch space; computing them in raw rect units introduces a ~0.9° systematic error.

POSITIONS (inches): cue ball (18.58, 36.37); 1-ball (25.75, 51.02); BR pocket centre (47.91, 92.09) — midpoint of measured jaw tips (46.72, 93.17) and (49.10, 91.01), mouth 3.2–3.5 in.

DIRECTIONS (atan2(dy,dx), +x right, +y down-screen):
  Required cue→ghost .......... 64.30°
  Straight-through cue→object . 63.93°
  Detected aim line (t=95.33) .. 63.35°   (−0.95° vs required)
  Measured stick @ address ..... 63.34°   (−0.96°)
  Measured stick @ CONTACT ..... 63.55°   (−0.75°)
  Actual cue-ball launch ....... 63.80°   (−0.50°)
  Required object direction .... 61.64°
  Actual object direction ...... 64.74°   (+3.10°)

CUT / FULLNESS: required cut 2.65°, contact point 0.104 in off the object ball's centre on the thin side. Actual cut 0.94° to the OPPOSITE side, contact 0.037 in off centre. Fullness 99.99% actual vs 99.89% required — both "full ball"; the entire shot lives in a 0.14 in difference of contact point. Impact-parameter error: 0.141 in too full. The cue ball's centre arrived 0.122 in from the required ghost-ball centre.

MISS MAGNITUDE: object direction error +3.10° over 46.67 in = 2.52 in lateral at the pocket. Concretely, the object ball's line meets the foot-cushion nose at X = 45.62 in while the near jaw tip is at X = 46.72 in — the ball's centre passed 1.10 in short of the jaw. It struck the foot cushion just inside the jaw and rebounded out. Miss is to the shooter's left of the pocket, i.e. an UNDERCUT.

LEVERAGE: aim-error amplification = (cue→object distance)/(ball diameter × cos cut) = 16.31/2.25 = 7.26×. Pot window for this shot: ±0.61° of object direction → ±0.085° of cue-ball direction → ±0.024 in of cue-path lateral placement at the object ball. He was off by 0.50° / 0.141 in — about 6× the window.

SPEED: 94.8 in/s (7.9 ft/s) at launch with draw. Appropriate for a 63 in shot; not a factor.

**Aim-line verdict:** CONSISTENT — the detector was right, and it actually caught the fault. Converted properly the detected aim line reads 63.35°, matching my independent per-frame pixel fit of the shaft at the same instant (63.34°) to 0.01°. That is 0.95° on the undercut side of the required 64.30° line, which is exactly the error that produced the miss.

Two caveats worth fixing in the tool, both of which matter more than the fault being diagnosed:
(1) ANISOTROPY. Computing the aim-line angle directly in raw rect coordinates gives 64.24° instead of the true 63.35°, because the rect space has different x and y scales (12.436 vs 12.933 units/in). That 0.9° systematic is larger than the entire 0.75° aim error here, and it happens to point the wrong way — it would have made a bad aim look correct. Angles must be normalised to true inches before use.
(2) TIMING. The aim line was sampled at t=95.33, 0.45 s before contact and during the backswing. The stick at contact was 63.55°, i.e. 0.20° further on. Small here because this stroke was unusually steady (63.27°–63.55° across the whole address and forward stroke), but on a stroke that steers, sampling that early would misreport the delivery.

**Root cause:** LINK (a) AIM. The stick was pointed 0.75° to the undercut side of the required line at the moment of contact (0.95° at address). Everything downstream follows from that.

Chain, quantified:
  AIM      stick @ contact 63.55° vs required 64.30° ....... −0.75°  ← FAILED LINK
  DELIVERY cue ball left at 63.80° vs stick 63.55° ......... +0.25°  (faithful; slightly HELPED)
  CONTACT  b = +0.037 in vs required −0.104 in ............. 0.141 in too full (consequence of the above, not an independent error)
  SPEED    7.9 ft/s with draw ............................. not a factor

Delivery is exonerated and, unusually, it partially rescued the shot. The shaft's centreline passed ~0.06 in (≈0.075 in after correcting for a modelled ~0.1° overhead-parallax bias) to the left of the cue ball's centre — a whisker of left english, which predicts a small squirt to the shooter's right. Observed delivery delta: +0.25° to the right. The prediction and the measurement agree, and that squirt moved the cue ball a third of the way back toward the required line. Had he delivered perfectly along the stick, the object ball would have left at ~66.3° and missed by 3.8 in instead of 2.5 in.

Nor was this a stroke that wandered: the shaft held 63.27°–63.55° (rms 0.014 in per fit) through the entire address, pause and forward stroke. The 1.8° swing across the line at 95.87–95.94 is pure follow-through, after the ball had gone. Worth flagging as a mechanics habit; it did not cause this miss.

The real story in plain terms: this shot needed 2.65° of cut — a contact point just 0.104 in (about a tenth of an inch, 4.6% of a ball) off the object ball's centre, toward the right rail. He played it as dead straight, and in fact aimed a fraction on the WRONG side of centre-to-centre (63.55° vs the straight-through 63.93°). Because the cue ball sits 16.3 in back, that tenth-of-an-inch misread is amplified 7.26× into 3.10° of object-ball error, and because the object ball then has to travel 46.7 in, that becomes a 2.5 in miss — the ball hitting the foot cushion 1.1 in short of the jaw. The failure is a near-straight-shot misread, not a stroke fault: he defaulted to centre-to-centre on a shot that was a hair off straight.

**Signals that would diagnose it:**
- Ghost-ball arrival error: distance from the cue ball's centre at contact to the required ghost centre = 0.122 in. For this shot the pot window is ~0.024 in, so a 5x exceedance. This single scalar diagnoses the miss and needs no angle convention.
- Impact-parameter (fullness) error: b_actual − b_required = +0.141 in, signed toward the pocket side = UNDERCUT. Directly names both the magnitude and the side of the error.
- Cue-ball launch direction vs required ghost line: −0.50°, against a computed pot window of ±0.085°. Expressing the error as a multiple of the shot's own tolerance (5.9x) is far more actionable than raw degrees.
- Stick direction at CONTACT vs required: −0.75°. Must be sampled in the last frame before the cue ball moves (here t=95.775–95.808), not at the address/backswing instant the current detector used (t=95.33).
- Delivery delta = (cue-ball launch direction) − (stick direction at contact) = +0.25°. Near zero exonerates delivery and isolates the fault to aim. This is the single most valuable derived quantity for splitting link (a) from link (b).
- Tip-contact offset = perpendicular distance from the cue ball's centre to the fitted shaft line at contact = +0.06 in (≈0.075 in parallax-corrected) = a touch of left english. Predicts the squirt sign and magnitude, and cross-validates the delivery delta.
- Aim-error amplification factor = (cue-to-object distance)/(ball diameter x cos cut) = 7.26x, and the derived pot window (±0.085° of cue direction, ±0.024 in of cue-path placement). A per-shot difficulty metric that flags long near-straight shots as low-tolerance before the stroke.
- Required cut angle = 2.65° with a contact point only 0.104 in off centre. A 'near-straight but not straight' classifier (say, cut angle under 5° with object-to-pocket distance over 30 in) would have pre-flagged this as the specific trap Joe fell into.
- Object departure direction vs required = +3.10°, and lateral error at the pocket = 2.52 in against a ±0.5 in mouth window. Best single outcome metric.
- Rail-arrival position relative to the jaw tip: the object line meets the foot-cushion nose 1.10 in short of the jaw. A pocket-relative miss coordinate that is robust to pocket-centre definition and reads naturally ('missed by an inch on the near jaw').
- Post-impact stick rotation: +1.8° over 100 ms of follow-through. Non-causal for this shot but a persistent mechanics signal worth trending across a session.
- Cue-ball launch speed 94.8 in/s and post-contact reversal of ~14 in, confirming draw and adequate pace — needed to rule link (d) out rather than leave it unaddressed.

(confidence: high)

## Instructional theory

# How expert pool instruction diagnoses a miss — the standard decomposition

Research summary, primary sources = Dr. Dave Alciatore (billiards.colostate.edu / drdavepoolinfo.com, PBIA Advanced Instructor, Dean of Billiard University), Bob Jewett (Billiards Digest), plus the standard instructor canon. Numbers quoted verbatim where they exist; anything I derived is marked **[derived]**.

Note on `billiards.colostate.edu`: it 403s to fetchers. Everything mirrors at `drdavepoolinfo.com` with identical paths. Article PDFs are the richest source — the FAQ pages are mostly navigation hubs with the numbers living in the BD article PDFs and technical proofs (TP).

---

## 1. The ghost-ball / cut-angle model, and what over/undercut means

**Ghost ball.** The reference model: to send the OB along a chosen line, place an imaginary CB ("ghost ball") in contact with the OB, centered on the line through the OB center back from the target. Drive the CB center into the ghost-ball center. At contact the OB departs along the **line of centers** (CB center → OB center). Everything else in the diagnosis is a *deviation from this model*.

**The one equation that matters.** With ball radius R (2R = 2.25 in = 57.15 mm), if `e` is the perpendicular offset of the CB center from the OB→target line at contact:

```
sin(θ_cut) = e / 2R          and       ball-hit fraction f = 1 − sin θ
```

Exact fractional-ball table **[derived from f = 1 − sinθ; matches Dr. Dave's published values]**:

| hit | sin θ | cut angle | Dr. Dave's stated value |
|---|---|---|---|
| full | 0 | 0° | 0° |
| 3/4 ball | 0.25 | **14.48°** | "3/4-ball hit (14 degree cut)"; quick-ref "~15°" |
| 1/2 ball | 0.50 | **30.00°** | "½-ball hit … 30° cut" (center-to-edge alignment) |
| 1/4 ball | 0.75 | **48.59°** | "1/4-ball hit (49 degree cut)"; quick-ref "45°" |
| 1/8 ball | 0.875 | **61.04°** | "10 and 2 are at 60° (about an 1/8-ball hit)" |

**Sensitivity — the number your system should be built around [derived]:**
`dθ/de = 1 / (2R·cos θ)` → at small cut angles **1 mm of contact-point error ≈ 1.00° of OB direction error**. It amplifies with cut angle: 1.16°/mm at 30°, 1.52°/mm at 48.6°, 2.07°/mm at 61°. This is *the* reason thin cuts are hard — the geometry multiplies your delivery error.

**Overcut vs undercut — the sign convention.**
- **Undercut** = hit too **full/thick** (e too small, |θ_actual| < |θ_required|). The OB is under-deflected and misses on the *near* side — the side the CB came from.
- **Overcut** = hit too **thin** (e too large). The OB over-deflects and misses on the *far* side.

Define a signed shot error `ε = θ_actual − θ_required`, positive = overcut/thinner. Every error source below has a **predictable sign in ε**. That's what makes automated attribution possible.

**Companion rules the diagnosis rests on:**
- **90° rule** (verbatim): *"The CB and OB separate at 90°, regardless of the cut angle (except for a straight-in shot, in which case the CB stops in place)."* Applies to **stun** (no top/bottom spin at impact); the CB *"heads and persists along the tangent line direction, which is perpendicular to the line of centers."*
- **30° rule** (rolling CB): the CB deflects ~30° from its original direction, valid *"between a ¼-ball hit (49 degree cut) and ¾-ball hit (14 degree cut)"*; the true value is *"approximately 34° closer to ½-ball hits"* and *"approximately 27° closer to ¼-ball or ¾-ball hits."*
- These two are **measurement instruments, not just position aids**: the CB's post-contact angle relative to the tangent line tells you what vertical spin was on the ball at impact (90° ⇒ stun, ~30° from original line ⇒ natural roll, >90° off tangent ⇒ draw). You get the player's *tip height* for free from the CB's exit angle.
- **Cut-angle estimation aids** (useful for report language): clock face — *"noon (12) is straight (0°), 11 and 1 are at 30° (½-ball hit), 10 and 2 are at 60° … each minute is 6°"*; slopes — *"1-to-1 slope = 45-degree cut … 1-to-2 slope = ~30° cut … 1-to-4 slope = ~15-degree cut"*; Jewett's cue trick — *"place your tip at the center of the ghost ball and pivot around it from the line to the pocket to the line of the shot … the number of inches the bumper on the cue travels is the number of degrees of the cut angle."*

---

## 2. The standard error sources, with sign and magnitude

Dr. Dave's own ranked list of **why people miss** (verbatim, in order): (1) *"not enough care and focus given to actually visualizing and aiming the shot"*; (2) *"not enough care and focus given to accurately aligning the cue and tip"*; (3) *"inaccurate or inconsistent visual alignment"*; (4) *"lack of understanding or intuition for how to adjust aim for squirt, swerve, and throw"*; (5) *"poor, inconsistent, or inaccurate stroke."* Secondary: eye/head/body movement during the stroke, improperly chalked tip, cling/skid/kick, underestimating easy shots, poor position from the previous shot.

### 2a. The error budget you're spending against
Corner pocket 4.5 in, ball 2.25 in ⇒ straight-on lateral allowance **±1.125 in (±28.6 mm)** at the pocket **[derived]**. Converting to allowable angular error at the OB **[derived]**:

| OB→pocket distance | allowable ε | equivalent contact-point error (small cuts) |
|---|---|---|
| 3 ft | ±1.79° | ±1.8 mm |
| 4 ft | ±1.34° | ±1.3 mm |
| 6 ft | ±0.90° | ±0.9 mm |

And this budget *shrinks* with approach angle and speed. Jewett & Alciatore's measured pocket widths: *"for shots that are nearly along the rail (angle is near 0) the hard and soft shots have close to the same margin of error … As the angle increases, the hard shots do start to see a smaller opening, and for an incoming angle of around six degrees the pocket is twice as wide for a soft shot as a hard one."* Pocket *center* also moves: *"For a slow shot, the point varies by as much as about 0.6 inches for a typical corner pocket and about 0.3 inches for a typical side pocket."* Dr. Dave's compact framing: *"To make a spot shot from anywhere on the table into a 4.5″ pocket takes approximately 25 discrete cut angles per quarter ball (per cut direction), each approximately 3.6° wide."*

Also: *"for a straight shot, shot difficulty is directly related to the product of the distances between the CB & OB and the OB & pocket. A straight shot is most difficult when the OB is exactly halfway between the CB and the pocket."*

### 2b. Squirt (cue-ball deflection) — the largest error when english is used
**Definition:** *"the angular change in the initial cue ball direction due to an off-center hit."* Cause: shaft **endmass** — *"less end-mass = less sideways force = less squirt."*

**Sign: the CB squirts AWAY from the side the english was applied on.** Left english → CB departs right of the cue's aim line.

**Effect on the hit [derived from that sign]:** inside english (english on the same side as the cut direction) → CB arrives thinner → **overcut** (ε > 0). Outside english → CB arrives fuller → **undercut** (ε < 0).

**Magnitudes (measured):**
- Ron Shepard: *"a squirt angle range of about .5 to 2.3° for low- to high-squirt cues"*, *"a pivot point range of about 50″ to 10″."*
- Platinum Billiards robot data: *"1.3 to 2.3° of squirt angle and 7.6″ to 14.1″ for pivot points."* Robot at ~15 mph, 6 mm and 12 mm side offsets, **CB lateral deviation measured at 50 inches**: lowest **1.17 in** (Predator Z-2), highest **2.02 in** (Scorpion break), average **~1.64 in**.
- Dr. Dave: *"A common low-squirt shaft has a squirt angle of about 1.8° at close to maximum tip offset, as compared to 2.5° for a typical regular-squirt cue."*
- **Squirt does NOT depend on shot speed** — CSU cue-testing-machine data: *"squirt is very nearly constant over a wide range of speeds, for a given tip offset."* The "squirt grows with speed" belief is explicitly labeled a myth curve. It *is* nearly linear in tip offset: *"if you pivot the cue twice as much, you create roughly twice as much tip offset (and English) and you create roughly twice as much squirt."*

**Why this dominates [derived]:** 1.64 in over 50 in = **1.88°**. Over a 5-ft CB→OB distance that's ~2.0 in ≈ 50 mm of lateral CB arrival error — i.e. Δsinθ ≈ 0.87, effectively a completely different shot. Even 25% english (~0.5°) over 5 ft gives ~13 mm ⇒ ~13° of cut change on a near-straight shot. **Squirt error scales with CB→OB distance; throw error does not.** That is your cleanest instrumented separator.

**Tip-offset units** (needed to talk to the player): ball radius 1.125 in, *"the maximum recommended cue tip offset is 9/16″ (R/2 or 0.5R)"* — the miscue limit, *"which is the width of the stripe on a striped ball."* Prefer % english over "tips": *"At 100% English, the center of the cue is offset a little less than 1 1/2 tip widths, which explains the '1.4 tips of English.' 50% English corresponds to the tip position that results in a contact point halfway to the miscue limit"* (≈ ¾ tip). "Tips" is tip-shape dependent — *"for '1.5 tips of English,' the tip offset percentage for a thin tip with nickel radius is about 84% of maximum, as compared to 112% … for a fat tip with dime radius."* **Report in % of max offset, not tips.**

### 2c. Swerve — squirt's partial antidote
Curving of the CB *back toward* the english side because the cue is elevated (masse). Verbatim rules:
- *"Swerve increases with cue elevation and the amount of sidespin."*
- *"Swerve occurs with practically all sidespin shots because the cue must be elevated to clear the rails."*
- *"Swerve is delayed with faster shot speed."*
- *"Swerve occurs only while the CB is sliding; once rolling begins, the CB heads in a straight line."*
- *"Swerve occurs earlier with sticky cloth and later on slick cloth."* / *"earlier with a follow shot than with a draw shot."* / *"Swerve angle is larger with a draw shot than with a follow shot."*

**Squerve** = net of squirt + swerve. *"Squerve … can be zero with certain speeds and cue elevations for a given shot distance, amount of sidespin, and cue."* *"Squerve is less for follow vs. draw shots."* **Squirt is speed-independent, squerve is strongly speed-dependent** — so a *speed-correlated* directional bias in your data is swerve, not squirt.

**Instrumented signature:** squirt = a *kink* at the CB's launch (cue line ≠ CB initial velocity). Swerve = *curvature* of the CB path between launch and contact, ending when the ball reaches natural roll. These are separable from a top-down track of the CB path alone.

### 2d. Throw — always present, sign-predictable, distance-independent
*"Throw refers to offline motion of the object ball (OB) caused by friction between the CB and OB during a hit."* Split into **CIT** (cut-induced) and **SIT** (spin-induced).

**CIT sign: always toward the cue ball's tangential slide direction = opposite the cut direction = the OB arrives LESS cut than geometry predicts = an apparent UNDERCUT (ε < 0), always, with no english.** Hence Dr. Dave's fix: *"Aim Thinner to Account for CIT … aim to over-cut the shot slightly. This will shift the tangent-line forward a little, so you also need a touch of backspin to keep the CB on-line."*

**SIT sign:** *"with a cut to the left and left spin, the OB gets thrown to the right. And with lots of right spin, the OB gets thrown to the left."* I.e. the OB is thrown **opposite** to the english side. So **inside english adds to CIT (more undercut); outside english opposes it (toward overcut)**, and at the **gearing** amount throw is exactly zero: *"there is an amount of spin that will result in no throw whatsoever … it is called gearing outside spin since the CB rolls along the OB while in contact, like a meshing gear."* *"The amount of sidespin required for 'gearing' increases with cut angle. At a half-ball hit, the amount of sidespin required is about 50%."*

**Magnitude ceiling:** *"Maximum throw, under typical conditions, is about 1 inch per foot of OB travel, or ½ a ball per diamond on a 9′ table, which is about 5°."* **[derived: that is 3.5× the entire ±1.34° budget on a 4-ft shot — throw alone can miss a long slow stun shot outright.]**

**Full throw rule set (verbatim, Sept 2018 "Got English?" items 16–35):**
- *"For small cut angle shots … the amount of CIT does not vary with shot speed, but increases with cut angle."* / *"CIT is actually independent of speed … at cut angles below about 20°."*
- *"For larger cut angle shots … the amount of CIT is significantly larger for slower speed shots as compared to faster speed shots."*
- *"Maximum CIT occurs at close to a half-ball hit (30-degree cut angle)."*
- *"In general, throw is larger at slower speeds, and for stun shots."*
- *"Both follow and draw reduce throw, and they do so by the same amount"* — *"With full topspin or bottom spin, the amount of throw is about one fourth of the throw expected with a stun shot at the same speed."*
- *"The largest discrepancy between throw values for stun and follow/draw shots occurs close to a half-ball hit."*
- *"SIT is largest for a slow stun shot with about 50% of maximum sidespin"* — more spin beyond that gives *less* throw (*"friction is less at faster sliding speeds between the ball surfaces"*). *"Below about 25% spin, speed has no effect on the amount of SIT."*
- *"IE increases throw at small cut angles, but actually reduces the amount of throw at larger cut angles."* / *"For large cut angles, a small amount of OE can result in more throw than shots with no sidespin."*
- **Cling/skid/kick:** *"if there happens to be a chalk mark on the CB or OB at the point of contact, you get much more throw than expected."* Saliva at the contact point ⇒ *"almost no friction or throw."* This is the one source that is genuinely stochastic — a report must be able to say "outlier, likely cling" rather than blame the player.
- Frozen/small-gap combos: throw dominates; *"when the gap size is exactly 3/8″, the throw cancels the cut over a wide range of angles"* (TP B.21).

### 2e. Aiming/perception error
Systematic, usually **directionally asymmetric** (left cuts vs right cuts differ) because it comes from eye dominance, head position, and the perceptual difficulty of the ghost-ball construction. Dr. Dave ranks it as the #1 cause of misses. Signature in data: **bias that is a smooth function of cut angle and cut direction, independent of speed, english, and distance.**

### 2f. Stroke/delivery error
Verbatim causes: *"twisting your wrist on the final stroke to change direction"*; *"decelerating during the final stroke when using sidespin, causing increased swerve"*; *"Not hitting the cue ball on the vertical centerline creates unintentional sidespin, resulting in squirt, swerve, and spin-induced throw"*; *"movement of eyes, head, or body during the stroke."* On elbow drop Dr. Dave is deliberately non-dogmatic: *"nobody is saying that dropping your elbow is bad … if it works reliably for you, keep doing it. However, if you find that you are not reliable, try the pendulum stroke, as it is a much easier way to get repeatability."*

**Signature in data: variance, not bias.** Delivery error is zero-mean and shows up as spread. The killer compound case is *unintentional* english — a delivery error that then masquerades as squirt+swerve+SIT, i.e. it produces a **bias** the player can't see the cause of. That's the highest-value thing an instrumented system can catch, because it is invisible to the player by construction.

### 2g. Speed
Speed changes: swerve timing (later at speed), CIT at cut angles >20° (less at speed), stun→roll transition distance, and the effective pocket size (roughly halved at ~6° approach for firm vs soft). Squirt itself is speed-independent. So **speed is a covariate you must record on every shot or half the attributions are unidentifiable.**

---

## 3. What an instrumented system can measure, and what each measurement separates

Two-level decomposition. This is the architecturally important point:

**Level A — "where did the cue ball actually arrive?" (the hit error)**
Measure the CB and OB centers at the extrapolated moment of contact (back-extrapolate both trajectories to their intersection — contact lasts ~200 µs, you will never see it in a frame). Compute actual `e`, hence `θ_geometric`. Compare to `θ_required` from OB→pocket geometry.
→ Yields **signed hit error in mm and in ball-fraction**, and thus overcut/undercut with magnitude. Requires no cue data at all. This is the single highest-value measurement.

**Level B — "what happened after contact?" (the throw)**
The **line of centers is directly measurable** from the two ball centers at contact. Fit the OB's departure direction from its trajectory over the first ~500 mm. The residual `θ_OB_departure − θ_line_of_centers` **is the throw angle**, isolated, with no modeling assumptions.
→ Sign vs cut direction distinguishes CIT (always anti-cut) from SIT (follows the english). Magnitude vs the ~5° ceiling flags cling.

Then Level A splits further, and *this* is where you need more sensors:

| Quantity | How measured | Separates |
|---|---|---|
| Cue axis at address & at impact | overhead cue detection (you already track a cue) and/or BLE cue IMU orientation | aim error vs delivery error |
| CB initial velocity vector | first ~10 frames of CB motion after impact | **squirt** = angle(cue axis, CB launch). Speed-independent ⇒ a stable per-cue calibration |
| CB path curvature pre-contact | polynomial fit to CB track between launch and contact | **swerve**, and the roll-transition point |
| CB speed at contact | CB track | covariate for throw, swerve, effective pocket size |
| CB post-contact angle vs tangent line | CB track + line of centers | **tip height**: 90° ⇒ stun, ~30° from original ⇒ natural roll, >90° off tangent ⇒ draw. Free measurement of vertical spin via the 90°/30° rules |
| CB spin (side) | hard optically at 30 px; inferable from squirt angle + swerve curvature + throw sign; the striped-ball wobble is Dr. Dave's own visual test (*"Anytime there is throw, sidespin is transferred to the OB, which causes the vertical stripe to wobble"*) | **intentional vs unintentional english** |
| Stroke kinematics: backswing, pause, acceleration profile, lateral jerk, cue roll | BLE cue sensor | steering, deceleration-on-english, wrist twist, elbow-drop signature |
| CB→OB distance, OB→pocket distance, cut angle, approach angle to pocket | geometry | the error budget for *this* shot; and the distance-scaling that separates squirt from throw |

**Three separators worth stating explicitly:**
1. **Distance scaling.** Squirt/squerve error scales with **CB→OB** distance. Throw is a fixed angle at contact, so its miss scales with **OB→pocket** distance. Regressing signed miss on these two distances separates them across a session even without cue data.
2. **Speed correlation.** Squirt is flat in speed; squerve and throw both grow at slow speed. A speed-correlated bias is squerve or throw, never raw squirt.
3. **Bias vs variance.** Systematic signed bias ⇒ aiming/perception or an uncalibrated compensation model. Zero-mean scatter ⇒ delivery. Left/right asymmetry ⇒ perception/alignment (eye dominance, head position), not stroke.

**Resolution reality check [derived].** At ~30 px ball diameter, 1 mm ≈ 0.53 px, and the whole error budget on a 4-ft shot is ±1.3 mm ≈ ±0.7 px. **Do not measure the contact geometry from instantaneous positions.** Measure *directions* from multi-frame trajectory fits — with ±1 mm centroid noise, a direction fit over 500 mm of OB travel is good to ~0.1°. Then back out the contact geometry from the fitted directions. That converts a sub-pixel position problem into a well-conditioned line-fit problem.

**Calibration the system should acquire per player/per cue (one-time, then drift-tracked):**
- **Natural pivot length** of the cue — *"the bridge position where a cue pivot … exactly cancels CB deflection (squirt) with a perfectly level cue or for a very fast and short shot (where swerve is negligible)."* Published examples: *"The Players cue has a typical amount of CB deflection with a natural pivot length of 9.5″, the Cuetec has lower CB deflection with the pivot at 13″, and the Revo and Z-2 have even lower CB deflection, with the pivot at 19″."* Caveats: *"the natural pivot length of a cue is about 5% longer"* with a rounder tip; *"a 10% heavier CB increases the pivot length by about 9%."*
- **The player's squirt curve** (squirt angle vs % offset) — nearly linear, and measurable directly from cue-axis-vs-CB-launch over a handful of shots.
- **Cloth/ball friction** (throw scale factor), from measured CIT at half-ball stun vs the 5° reference.

**The two compensation techniques the report should be able to prescribe** (these are what "aim adjustment" actually means at the table):
- **BHE** (back-hand english): set up center-ball, move the grip hand sideways; cancels squirt when bridging at the natural pivot length. *"BHE works best for short and fast shots."*
- **FHE** (front-hand english): move the bridge hand, grip stationary — *"a much longer effective pivot length with correspondingly less aim compensation."* *"FHE works best for long and slow shots. For all shots in between, you can use a combination of BHE and FHE."*
- Plus **gearing outside english** and **"aim thinner"** for throw.

**On "BEF":** that acronym isn't a recognized term in the Dr. Dave corpus or the mainstream instructional literature — I suspect it's a garble of **BHE** (back-hand english) or of *BU* (Billiard University, Dr. Dave's standardized exam system). The documented aiming-system families are: **ghost ball** (Dr. Dave's recommendation: *"a better teaching and learning approach that helps players develop faster"*, though he calls all fixed systems *"training wheels"* / *"crutches"*), **contact-point-to-contact-point**, **double-the-distance**, **fractional-ball** (¼/½/¾ with center-to-edge alignment for ½), **CTE/Pro One**, **SAM**, **DAM**, and **parallel/equal-and-opposite**. Dr. Dave's standing caveat on all limited-lines systems: *"fixed-point aiming systems with a limited number of aiming lines are not perfect and will cause you to miss shots if you don't compensate."*

---

## 4. What a per-shot report must say to drive practice

The failure mode to avoid is a physics readout. "Squirt: 1.8°, CIT: −0.9°, delivery jitter: 4.2 mm/s²" is trivia. The **actionable unit** is: *one named error, its sign, its size in units the player can execute, and whether it's this shot or a pattern.*

**Per-shot card — five lines, in this order:**

1. **Verdict in the player's own language, with sign and size.**
   "You **undercut** this by **1.2°** — the 4-ball caught the **right jaw**. That's about **1/16 of a ball** thicker than the line."
   Always give overcut/undercut plus a table-executable unit (ball fraction, or inches at the pocket), never radians.

2. **The budget you had.** "On a 4-foot shot into a 4.5″ pocket your whole margin was **±1.3°**. You spent 1.2° on the hit and 0.6° on throw — you were 0.5° over."
   This reframes a miss from "I'm bad" to "here's the tolerance and here's where it went." It also tells the player when a miss was *not their fault* (a 6-ft thin cut with a ±0.6° budget is a low-percentage shot, and the report should say so — that's shot-selection coaching, which is worth more than aim coaching).

3. **The dominant cause, with its evidence — and a refusal when unidentifiable.**
   "Dominant: **unintentional left english** (~20% offset). Evidence: CB launched 0.4° right of the cue line and curved back 0.2° — that's a squirt/swerve signature, not an aiming error. It also threw the 4 right by 0.5°."
   And when the data doesn't support attribution: "**Can't attribute** — cue not visible this shot. Recorded as hit error only." A coach that guesses loses trust permanently; one that abstains and says why keeps it.

4. **The correction, executable at the table.** "Aim **1/8 ball thinner** on left cuts of this angle" / "**one tip less** left english, or move your bridge to **13 inches**" / "this shot wants **gearing outside** — about 50% right at a half-ball hit." Not "reduce squirt."

5. **Is this a pattern?** A single shot cannot separate bias from variance. The card should carry the running context: "This is your **9th of 12** left-cuts undercut this session; mean **−0.8°**. Right cuts: mean **+0.1°**. That's a **perception/alignment** asymmetry, not a stroke problem — check head position and dominant eye."

**Session/trend layer — the four diagnoses worth naming.** These are the only conclusions a player can actually act on, so the analytics should be built to produce exactly these:

| Pattern in the data | Diagnosis | Prescription |
|---|---|---|
| Signed bias, symmetric L/R, grows with cut angle, no speed or english dependence | **Aiming/perception** (ghost-ball estimation) | Ghost-ball / contact-point drills at the specific cut angles that miss; the wagon-wheel drill |
| Signed bias, **asymmetric L/R** | **Alignment / eye dominance / head position** | Video from behind; the mirror alignment check; stance work |
| Zero-mean scatter, widening with speed and with CB→OB distance | **Delivery** (stroke straightness) | Long straight-in stop shots — if the CB doesn't stop dead, you had unintended spin; pendulum-stroke / SPF work |
| Bias that correlates with **english used** and with **CB→OB distance** | **Squirt compensation miscalibrated** | Pivot-length calibration test, then BHE/FHE per shot length |
| Bias that correlates with **slow speed + stun + near-half-ball** | **Throw not compensated** | "Aim thinner"; faster speed + follow/draw (¼ the throw of stun); gearing outside english |
| Isolated large outlier, throw ≫ 5° ceiling | **Cling/skid/kick** | Not a fault. *"Wipe chalk marks off the CB every chance you get … and keep the OBs as clean as possible."* |

**Two design principles worth adopting explicitly:**
- **Report the residual, not the model.** The player doesn't need your squirt coefficient; they need "your compensation is 0.6° short and consistent." Everything the model *explains* is background; what it *cannot* explain is the coaching.
- **Rank by expected points recovered, not by size of effect.** A 2° squirt error on a shot with a ±3° budget cost nothing. Sort a session's findings by `P(make | error removed) − P(make | actual)` summed over shots. That makes the report say "fixing your left-cut undercut is worth ~1.8 makes per hour of the way you play" — which is the only sentence in the whole thing that changes what someone practices tomorrow.

---

**Sources:**
- [Squirt (Cue Ball Deflection) — Dr. Dave FAQ](https://drdavepoolinfo.com/faq/squirt/)
- [Published Data for Shaft CB Deflections](https://drdavepoolinfo.com/faq/squirt/published-data/)
- [Causes for Squirt and Cue Ball Deflection](https://drdavepoolinfo.com/faq/squirt/cause/)
- [Squirt – Part I: introduction (BD, Aug 2007) PDF](https://drdavepoolinfo.com/bd_articles/2007/aug07.pdf)
- [Squirt – Part VII: cue test machine results (BD, Feb 2008) PDF](https://drdavepoolinfo.com/bd_articles/2008/feb08.pdf)
- [Squirt/pivot-length & tip shape (BD, Jan 2008) PDF](https://drdavepoolinfo.com/bd_articles/2008/jan08.pdf)
- [Natural Pivot Length](https://drdavepoolinfo.com/faq/cue/natural-pivot-length/)
- [Low Squirt (LD) Pool Cue Shafts](https://drdavepoolinfo.com/faq/cue/low-squirt/)
- [Sidespin Squirt, Swerve, and Throw Effects](https://drdavepoolinfo.com/faq/sidespin/aim/effects/)
- [“Got English?” (BD, Sept 2018) PDF — the 35 numbered squirt/swerve/throw rules](https://drdavepoolinfo.com/bd_articles/2018/sept18.pdf)
- [“Everything You Need to Know About Throw” (BD, Nov 2020) PDF](https://drdavepoolinfo.com/bd_articles/2020/nov20.pdf)
- [Throw — FAQ hub](https://drdavepoolinfo.com/faq/throw/)
- [Throw Speed and Spin Effects](https://drdavepoolinfo.com/faq/throw/speed-effects/)
- [Maximum Throw](https://drdavepoolinfo.com/faq/throw/maximum/)
- [Back-Hand English (BHE) and Front-Hand English (FHE)](https://drdavepoolinfo.com/faq/sidespin/bhe-fhe/)
- [HAPS Part II: BHE and FHE (BD, Dec 2014) PDF](https://drdavepoolinfo.com/bd_articles/2014/dec14.pdf)
- [Sidespin Percentage and “Tips” of Spin](https://drdavepoolinfo.com/faq/sidespin/tips-and-percentage/)
- [System for Aiming With Sidespin (SAWS)](https://drdavepoolinfo.com/faq/sidespin/aim/saws/)
- [The 30° Rule](https://drdavepoolinfo.com/faq/30-degree-rule/)
- [The 90° Rule](https://drdavepoolinfo.com/faq/90-degree-rule/)
- [Fractional-Ball Aiming](https://drdavepoolinfo.com/faq/aiming/fractional/)
- [HAPS Part I: Fractional-Ball Aiming (BD, Nov 2014) PDF](https://drdavepoolinfo.com/bd_articles/2014/nov14.pdf)
- [How to Estimate a Cut Angle](https://drdavepoolinfo.com/faq/cut/estimating-angle/)
- [Aiming Systems](https://drdavepoolinfo.com/faq/aiming/systems/) · [Limited Lines of Aim](https://drdavepoolinfo.com/faq/aiming/lines-of-aim/)
- [Why People Miss Shots](https://drdavepoolinfo.com/faq/aiming/missing/)
- [Stroke Technique Advice](https://drdavepoolinfo.com/faq/stroke/technique/) · [Stroke Video Analysis](https://drdavepoolinfo.com/faq/stroke/video-analysis/)
- [Cut Shot Margin For Error](https://drdavepoolinfo.com/faq/cut/margin-for-error/) · [Pocket Effective Size and Center](https://drdavepoolinfo.com/faq/pocket/size-and-center/)
- [Bob Jewett, “Where’s the Pocket?” (BD, May 2013) PDF](https://drdavepoolinfo.com/bd_articles/jewett_may_2013_pocket_size_part1.pdf)
- [Straight-In Pool Shots](https://drdavepoolinfo.com/faq/cut/straight-in/) · [Table Difficulty Factor (TDF)](https://drdavepoolinfo.com/faq/table/tdf/)

## Analytics spec

# ROOT-CAUSE ANALYTICS SPEC — "which link broke, and prove it"

**Grounding:** three hand-measured shots (`s233`, `s74`, `s96`) + Dr. Dave / Jewett canon. Codebase anchors: `src/billiards_trainer/vision/shots_export.py`, `cue_aim.py`, `analysis_cache.py`, `describe.py`, `geometry.py`, `companion-cloud/public/index.html`.

---

## 0. THE PRECONDITION NOBODY BUDGETED FOR

All three forensics **discarded the sidecar and re-measured off the frames**, and each one rebuilt calibration by hand before doing anything else. That is not incidental — it is the finding that determines the architecture.

| File | What was broken | Consequence |
|---|---|---|
| s233 | supplied transform maps the stated rect box to a 2.98:1 region; real bed 2.006:1. 5 of 7 tracks frozen at identical coords all shot | every inch figure wrong; "contact on ball X" unreliable |
| s74 | transform a near-identity fallback; cue ball carried as `n=−1`, inactive at shot start, **zero launch samples**; trail samples lead video PTS by **+0.19 s** (mean 13.5 in of position error during the fast phase, → 1.05 in when shifted) | the single most diagnostic measurement was simply absent |
| s96 | rect space **anisotropic by 3.9%** (12.436 u/in x vs 12.933 u/in y). Angle taken in raw rect = 64.24° vs true 63.35° | **0.9° systematic — larger than the 0.75° fault being diagnosed, and pointing the wrong way. It would have made a bad aim look correct.** |

Also live in the repo right now: `geometry._TABLE_WIDTH_IN["8ft"] = 44.0` vs `config._BED_SHORT_IN["8ft"] = 46.0`. A 4.5 % scale disagreement between two modules — 4.5 % on every inch figure this spec produces.

### 0.1 `vision/tablespace.py` — the TRUE frame, and the right to refuse

Define one coordinate frame used by every number below: **inches, isotropic, origin at the bottom-left cushion nose, y down-screen.** Angles are `atan2(dy,dx)` in that frame, nowhere else.

Construction + audit gates (any failure ⇒ shot is `CALIB_FAIL`, analysis refused with a named reason, never silently degraded):

1. **Aspect assertion.** Bed nose-to-nose must be `2.000 ± 0.02`. Catches all three files. Bed found from cushion-nose shadow troughs (sub-pixel parabolic fit) — s96 cross-validated this against the object ball's own rail-rebound turning points and got 0.2 in agreement.
2. **Two independent rulers, ≤5 % disagreement.** (a) detected ball diameter ≡ 2.25 in; (b) long-rail diamond spacing ≡ bed_long/8. s233: 25.5 px ball, 143 px diamonds ⇒ `8 × 2.25 / (25.5/143) = 100.9 in` ⇒ **9-foot table, not the configured 7ft**. **Table size is DETECTED per session, never configured.**
3. **Anisotropy.** Emit `px_per_in_x` and `px_per_in_y` separately; forbid any angle computation before conversion.
4. **Rail-rebound check.** Any OB rail bounce must have its turning point exactly one ball radius off the nose line (s96 used this; it is free).
5. **Frozen-ghost filter.** Drop any track with exactly-zero position variance across the shot window (s233: 5 of 7).
6. **Trail-lead cross-correlation.** Correlate sidecar track positions against frame-differenced centroids; best lag must be 0. Non-zero ⇒ record, shift, and **fix at source** — those leading trails are what the phone draws as animated tails today.
7. **Launch-sample presence.** `< 3` cue-ball samples between launch and contact ⇒ `NO_LAUNCH`.

**Ship `tools/audit_calibration.py` first.** Run it over the whole archive before building anything else; you find out immediately what fraction of the corpus is analysable at all.

### 0.2 The forensic pass — dense re-decode, not the tracker

The sidecar is a **live-overlay** product at 7–10 Hz. It cannot deliver a launch direction and it should not be asked to (roadmap: do not derail recording). Instead, per stroke shot, re-decode a dense window at native fps — exactly what `shots_export._shot_aim` already does for one frame, scaled up:

| window | span | measured |
|---|---|---|
| `W_address` | `[t_launch − 0.60, t_launch]` | per-frame cue-tip/ferrule position; shaft mask |
| `W_launch` | `[t_launch, t_contact]` | CB centroid (streak centroid when blurred) |
| `W_depart` | `[t_contact, +0.50 s]` | OB centroid |
| `W_arrival` | OB last 0.4 s before first rail/pocket | arrival point, speed, rattle |

~40–60 frames per shot, two small ROIs. ~2 minutes for a 60-shot session, offline, resumable — same shape as `tools/score_corpus.py`.

**Never take a direction from two positions.** At 30 px ball, 1 mm ≈ 0.53 px and the entire budget on a 4-ft shot is ±0.7 px. Every direction is a **total-least-squares fit over many frames**, then contact geometry is *back-solved from the fitted lines*. The forensics prove this works through 30 fps blur: residuals of **0.15 px over 232 px** (s233, three streak centroids), **0.23 px / 0.02 in over 13 in** (s74), **0.056 in rms over 14 OB samples** (s96). Blur is not the problem people assume it is — the problem is asking a single frame to carry the measurement.

---

## 1. THE MEASUREMENT CHAIN

Notation: `D = 2.25 in`, `R = 1.125 in`. All bearings in the TRUE frame.

### M0–M3 · Substrate
- **M0** `frame` — from §0.1, plus `px_per_in`, `table_size`, `audit_flags`.
- **M1** `t_launch` — first frame where CB centroid displaces > 0.02 in from its at-rest median (rest positions repeat to ±0.004 in, s74 — the noise floor is genuinely that low).
- **M2** `t_contact` — **never observed** (contact ≈ 200 µs). Solve: the time at which the CB's fitted launch line places its centre at distance `D` from `P_obj`. Cross-check against the first OB motion frame; disagreement > 1 frame ⇒ flag.
- **M3** `P_cue`, `P_obj` — median rest centres over ≥ 10 address frames.

### M4 · Intended target inference → `T`
Score each pocket `P`:

| term | test | forensic precedent |
|---|---|---|
| `legal` | is this OB the legal ball? | s74: all nine up ⇒ the 1 is legal |
| `reachable` | required cut < 70°, not back toward the shooter | s74 rejected 3 pockets on this alone |
| `clear` | no ball within `D + 0.25 in` of the OB→P segment | s74 nearest 9 in off; s96 4.9 in / 8.0 in |
| `natural` | `|φ_straight_through − φ_req(P)|` small | s233: natural line within 3.1° of the BL corner vs 66° for BR |
| `stick` | `|φ_aim − φ_req(P)|` | s233 within 0.7°; s74 within 0.30° |
| `outcome` | where the OB actually went | **tiebreaker only — see below** |

Winner must beat runner-up by a margin, else `TARGET_UNCERTAIN`.

> ⚠️ **`stick` and `outcome` are circular.** If the target is chosen because the stick pointed at it, the system can never report a large aim error. If it is chosen because the ball went there, it can never report a large miss. Weight both low, and treat the declared-pocket tap (§4.8) as the mechanism that *audits* this, not as an optional nicety.

### M5–M7 · The required shot
- **M5** `G = P_obj − D · unit(T − P_obj)` — ghost-ball centre.
- **M6** `φ_req = bearing(G − P_cue)`.
- **M7** `θ_req`: `sin θ_req = e_req / D`, where `e_req` = signed perpendicular offset of `P_cue` from the OB→T line. Fullness `f = 1 − |sin θ|`.
  Sign in a per-shot local frame: `u = unit(T − P_obj)`, `n = perp(u)` oriented toward the side the cue ball is on. All `e` values signed along `n`. Positive `e` = the required cut direction.
  Record `cut_direction ∈ {L, R}` separately — that is the axis of the asymmetry hunt, and it must survive the sign algebra.

### M8 · Detected aim direction `φ_aim`
Three mandatory changes to the current detector:

1. **Convert to TRUE inches before taking the angle.** (s96: 0.9°, wrong direction.)
2. **Sample at CONTACT, not at address.** `_shot_aim` currently scans backwards `0.4 / 1.0 / 1.8 s` from shot start and takes the first hit. s96's aim was captured **0.45 s before contact, mid-backswing**, and was 0.20° off. Harmless on that unusually steady stroke; fatal on a stroke that steers. Emit **three** values: `φ_aim@address` (median over the address), `φ_aim@contact` (last frame before the CB moves), and `spread` (the steering signal).
3. **Robustness the frames demand.** s74 returned `aim: null`, and the forensic reproduced the failure: a black bridge glove splits the shaft into two segments with a ~14 px gap, the butt runs off-frame behind the cap, and a **bare forearm lies alongside the shaft at a shallower angle**. Naive bright-pixel fits returned 54.2° and 25.6° — 5° and 34° wrong. What worked: neutral-grey/white colour constraint **excluding skin tones**, an elongation test, iterative outlier rejection, and gap-tolerant merging of collinear segments. `cue_aim.py` already has the colour-distance mask and densest-angle-cluster; it needs the skin exclusion, the two-segment bridge, and a per-frame consistency requirement across the address.

**Keep the anchor on the stick's axis.** `cue_aim.py`'s docstring is right and this must never be "fixed": re-anchoring the ray at the ball centre would hide exactly the error being hunted.

### M9 · Tip lateral offset `o_tip` — the highest-value single scalar
Signed perpendicular distance from the CB centre to the extrapolated **cue-tip track**, positive = shooter's left. Report in `R` units and as **% of max offset** (`0.5R` = 9/16 in = miscue limit = "100 % english"); theory is explicit that "tips" is tip-shape dependent and % is the correct unit.

> **Track the TIP, not the shaft.** s233: shaft-axis fit gives 16.44°, cue-tip track gives 16.93° — the shaft is elevated and perspective biases it ~0.5°; the tip is at ball height so it is directly comparable to the ball-centre plane. s96 had to model ~0.1° of overhead parallax on its shaft fit. **A "full-shaft sub-pixel accuracy" upgrade that does not correct elevation will be precisely wrong** — it will deliver 0.05° precision on a 0.5° bias. Redirect that work to the ferrule (the deepest bright low-saturation run), tracked through backswing and forward stroke; s233 got 14 points at 0.21 px rms.

Measured: s233 **0.25 R left (50 %)**, s74 **0.148 R left (30 %)**, s96 **0.053 R left (11 %)**.

### M10–M12 · Delivery
- **M10** `φ_cb` — TLS through CB rest centre + all pre-contact samples.
- **M11** `curvature` — max perpendicular residual, and the quadratic term. Near-zero ⇒ **no swerve** ⇒ the squirt went uncompensated. All three shots: dead straight (0.02 in over 13 in). This is what separates squirt (a kink at launch) from swerve (curvature on the cloth).
- **M12** `v_cb` at contact. s233 164 in/s; s74 93; s96 94.8.

### M13–M14 · Contact
- **M13** `e_act` = signed perpendicular offset of the fitted CB line from `P_obj`; `sin θ_act = e_act/D`; fullness. Plus **`ghost_arrival_error` = |CB centre at contact − G|** in inches — s96 nominates this as the best single scalar (0.122 in against a 0.024 in window; needs no angle convention at all).
- **M14** `φ_loc` = bearing(`P_obj` − CB centre at contact) — the line of centres, directly measurable, zero modelling.

### M15–M16 · Object ball and throw
- **M15** `φ_ob` — TLS over the first 12–15 OB samples. **Free gate:** the fitted line must pass within ~0.05 in of the OB's rest centre (s96: 0.035 in). If it doesn't, the fit or the contact time is wrong — abstain.
- **M16** `throw = φ_ob − φ_loc`. s233 **+2.26°**, s74 **+1.81°**. Sign vs cut direction separates CIT (always anti-cut) from SIT (opposite the english side). > 5° ⇒ cling.

### M17 · Error at the pocket, in inches
- `lateral_miss` = perpendicular distance from the OB's fitted line to the pocket centre at the pocket plane. s233 2.49 in (1.11 balls); s74 1.83 in; s96 2.52 in.
- **`s_mouth`** = normalised crossing parameter of the OB centre line across the **jaw-tip segment**, 0..1 inside. s233 `s = 1.356` ⇒ 1.22 in outside the short-rail jaw. **Prefer this** — it names *which jaw and by how much*, and separates a rattle from a wide miss.
- `rail_arrival_offset` — where the OB line meets the cushion nose relative to the jaw tip. s96: 1.10 in short of the jaw. Robust to any definition of "pocket centre" and reads naturally.

### M18 · The yardstick — per-shot pot window
```
half_window_in = (W_eff − D) / 2          # W_eff = aperture ⟂ to the OB's approach
tol_OB_deg     = atan(half_window_in / d_obj_pocket)
A              = d_cue_ghost / (D · cos θ)          # aim amplification
tol_CB_deg     = tol_OB_deg / A
```
| | s233 | s74 | s96 |
|---|---|---|---|
| A | 11.9× | 7.10× | 7.26× |
| `tol_CB` | **±0.054°** | **±0.19°** | **±0.085°** |
| net CB error | +0.58° (**11×**) | +0.56° (**3.0×**) | −0.50° (**5.9×**) |

`W_eff` must additionally shrink with approach angle and speed (Jewett: at ~6° approach the pocket is **twice as wide** for a soft shot as a hard one; effective centre moves up to 0.6 in). Carry `ob_arrival_speed` alongside — s233 (62 in/s… 164 in/s launch) and s74 (62 in/s at the jaw) both *rattled out* where a softer arrival may have dropped.

**Every error in the report is printed as a multiple of this number.** All three forensics converged on this independently. 0.7° is a rounding error on a 3-ft shot and a catastrophe on s233.

### M19 · The per-link errors (the whole point)
```
aim_err      = φ_aim@contact − φ_req
delivery_err = φ_cb          − φ_aim@contact          (squirt + steer)
net_cb_err   = φ_cb          − φ_req  = aim_err + delivery_err
contact_err  = Δe = e_act − e_req     (+ = thinner/overcut, − = fuller/undercut)
throw        = M16
ob_err       = φ_ob − φ_ob_req
```
| | aim | delivery | net | `o_tip` |
|---|---|---|---|---|
| s233 | −0.68° | **+1.26°** | +0.58° | 0.25 R L |
| s74 | −0.30° | **+0.87°** | +0.56° | 0.148 R L |
| s96 | **−0.75°** | +0.25° | −0.50° | 0.053 R L |

> **On 2 of 3 shots the two links pointed opposite ways and partially cancelled.** s74 states it flatly: *"reporting only the net cue-ball error would have mislabelled this shot."* The split is the product.

**Splitting delivery further.** Fit `squirt_pred = k · (o_tip / 0.5R)` with `k ≈ 2.5°` (Dr. Dave, typical regular-squirt cue at ~max offset). Against the three shots: predicted **1.25 / 0.74 / 0.27°**, measured **1.26 / 0.87 / 0.25°**. A through-origin fit of the three gives **5.24 °/R ⇒ 2.62° at the miscue limit** — Joe's cue is a normal-deflection shaft, ~10–12 in natural pivot length. n=3 is indicative, not established, but it means **the contact model closes and the aggregate hunt will work.**
Then `steer_residual = delivery_err − squirt_pred`. Theory's design principle: *report the residual, not the model.*

### M20–M22 · Free extras
- **M20 `tip_height`** — CB/OB separation angle vs the tangent line. 90° ⇒ stun, ~30° off the original line ⇒ roll, > 90° off tangent ⇒ draw. s233: **136.8° ⇒ heavy draw**, measured from overhead without ever seeing the tip on the ball. Also gives an **independent second estimate of contact fullness from the CB's post-contact reversal alone** (s74: 15.1 in of draw on bearing 232°) — exactly what you need when blur eats the OB track.
- **M21 `followthrough_rotation`** — s96: +1.8° over 100 ms, entirely *after* the ball had gone. Non-causal; a mechanics trend line.
- **M22 `stroke_divergence`** — backswing tip line vs forward-stroke tip line. s233 **0.32°**, s74 **0.06°**. Near-zero **exonerates the stroke path and isolates the fault to tip placement**. This is the separator between "steering" and "where you put the tip" — same link, completely different prescription.

---

## 2. WHICH LINK FAILED — the decision rule

### 2.1 The budget must close
```
φ_ob_actual − φ_ob_required
    = A_ob · (aim_err + delivery_err + swerve_err)   +   throw   +   residual
miss_in = that × d_obj_pocket
```
Convert every link to **inches of miss at the pocket** and rank there — not in degrees. Degrees are not comparable across links (`throw` is unamplified; aim and delivery are amplified 7–12×).

```
C_aim      = A_ob · aim_err      · d_obj_pocket
C_delivery = A_ob · delivery_err · d_obj_pocket
C_swerve   = A_ob · swerve_err   · d_obj_pocket
C_throw    =        throw        · d_obj_pocket
C_residual = miss_observed − Σ
```

### 2.2 The rule
```python
if not audit_ok:                          return REFUSE(reason)
if abstain_reason:                        return CANT_TELL(reason)   # §2.4
if outcome == "make" and clean_drop:      label = CLEAN              # then run position check §2.5

if abs(C_residual) > max(0.5, 0.4 * abs(miss_observed)):
    label = UNEXPLAINED
elif abs(throw) > 5.0:                                    # theory's hard ceiling
    label = CLING                                         # not a fault
elif argmax == C_delivery and abs(delivery_err) > 2*tol_CB:
    if abs(o_tip) > 0.08*R and sign(delivery_err) == -sign(o_tip):
        label = DELIVERY / UNINTENTIONAL_ENGLISH          # s233, s74
    elif stroke_divergence > 0.5:
        label = DELIVERY / STEERING
    elif curvature significant:
        label = DELIVERY / SWERVE
    else:
        label = DELIVERY / UNATTRIBUTED
elif argmax == C_aim and abs(aim_err) > 2*tol_CB:
    label = AIM                                           # s96
elif argmax == C_throw:
    label = THROW_UNCOMPENSATED
elif all links < 2*tol_CB:
    label = SPEED if (rattle_signature and arrival_speed high) else UNEXPLAINED
```

**Threshold justifications**
- `2 × tol_CB` rather than a fixed degree figure: tolerance varies **3.5×** across these three shots (±0.054 to ±0.19°). A fixed threshold is meaningless. s233's 0.68° aim error is 12× tolerance; s96's 0.75° is 8.8× — *same degrees, different verdicts on a shorter shot.*
- `|o_tip| > 0.08R` (≈16 % english): below that, `squirt_pred < 0.3°` for a normal shaft — below the measurement's own useful resolution.
- `throw > 5°`: verbatim theory ceiling ("~1 inch per foot of OB travel").
- `C_residual > 0.4 × miss`: the model has to close or the app abstains. Note that **ignoring throw makes it not close** — 1.8–2.3° in two of three shots, recovering ~45 % of s74's geometric error. Throw is not a nuisance term.

**Contact is not an independent link.** s233: *"the cue ball struck exactly the fullness its launch line dictated; there is no separate contact error."* s96: *"consequence of the above, not an independent error."* `contact_err` is reported as the *consequence* and only becomes its own label when the fitted launch line fails to predict the observed contact — i.e. swerve, an intervening collision, or a bad fit. Treating it as independent double-counts every miss.

### 2.3 Speed and position
- **SPEED is causal** only when the directional error was *inside* the window and the ball still didn't drop (rattle signature + high arrival speed). In s233 and s74 speed was **contributory, not causal** — they were 11× and 3× outside the window; firm pace merely guaranteed a rattle-out rather than a hang. Say exactly that; don't let speed become the lazy default.
- **POSITION-PLAY is never a miss label.** It attaches to (a) *made* shots whose resulting CB rest leaves the next shot's `tol_CB` in the worst decile or the CB frozen/snookered (s74: CB drew back and froze on the 3-ball), and (b) *retroactively to the previous shot* when this shot's pre-shot tolerance is bottom-decile. This is the only way "poor position from the previous shot" ever becomes measurable.

### 2.4 "The data can't tell" — a first-class output
Theory: *a coach that guesses loses trust permanently; one that abstains and says why keeps it.*

| code | trigger | what still ships |
|---|---|---|
| `CALIB_FAIL` | §0.1 gates | nothing — refuse the shot |
| `NO_AIM` | cue undetected (s74) | hit error, throw, miss, tolerance — **everything except the aim/delivery split** |
| `NO_LAUNCH` | < 3 CB samples pre-contact (s74's sidecar had **zero**) | OB-side only |
| `IDENTITY_CHURN` | number changed mid-flight, or a frozen ghost in the participant set (s233) | geometry survives; identity claims do not |
| `TARGET_UNCERTAIN` | top-2 pockets within margin | facts without a "required" line; prompt for the tap |
| `INTERFERENCE` | hand in frame (s96 t=99.44 — Joe's glove caught the rolling ball and the sidecar called it a collision with the 3-ball) | truncate the OB path at the hand |
| `UNEXPLAINED` | residual gate | full numbers, no label |

---

## 3. WHAT JOE SEES

### 3.1 The card — five lines, in this order

**s233, as it would ship:**
> **Overcut.** The 4 finished **2.5 in (1.1 balls)** past the pocket, into the short-rail jaw.
> **Your margin was ±0.05°.** 27 in to the object ball, 35 in to a 3.4 in pocket — aim error is amplified **11.9×** here. This shot only looks easy.
> **Cause: delivery.** The cue ball left **1.26° right of where your stick pointed**. Your tip was ~½ tip (50 %) left of centre; left english squirts the ball right. Your aim was 0.68° the *other* way, so it half-cancelled and the squirt still won. Your stroke was straight — backswing and forward stroke differ by 0.3° — so this is **tip placement, not steering**.
> **Fix: centre ball on this shot.** Half a tip of side spends 1.3° against a 0.05° budget. The english bought nothing and cost the shot.
> **Pattern: 3 of your last 3 measured strokes hit left of centre** (0.25 R, 0.15 R, 0.05 R), all squirting right, mean **+0.79°**. Systematic, and invisible to you by construction.

**s96, as it would ship:**
> **Undercut.** You played a **2.65° cut as dead straight** — the 1 hit the cushion **1.1 in short of the jaw**.
> **Your margin was ±0.09°.** The required contact point was **0.10 in off the 1's centre** — a tenth of an inch — amplified 7.3× and then run out over 47 in.
> **Cause: aim.** Your stick was 0.75° on the undercut side at contact, in fact a hair on the *wrong side* of centre-to-centre. Delivery was faithful (+0.25°) and actually rescued a third of the error.
> **Fix: near-straight is not straight.** On any cut under 5° with over 30 in to the pocket, build the ghost ball explicitly instead of defaulting to centre-to-centre.
> **Pattern: NEAR-STRAIGHT TRAP** — flagged 4× this session.

Rules: always overcut/undercut + a table-executable unit (ball fractions, inches at the pocket); never radians; never a physics readout.

### 3.2 The overlays that PROVE it

Design rule: **every drawn line is a measured quantity, and the card names exactly the lines that back its claim.** Never draw all of them at once — five lines at once prove nothing.

**A — GEOMETRY (always on).** Ghost-ball circle at `G` (dashed white) · required CB line `P_cue→G` (green) · required OB line `P_obj→T` (green dashed) · actual OB path (ball colour, solid) extended to the pocket plane with the gap annotated in inches · **the pocket drawn as its jaw-tip segment** with a marker at `s_mouth`. Answers "over or under, and by how much" with zero cue data.

**B — AIM PROOF.** Required CB line (green) vs detected stick line (amber), both drawn **from the stick's own anchor** so lateral offset from the ball is visible · plus **the ghost consequence path**: extend the aim line, compute the contact it produces, and draw in grey the OB path it *would* have produced.
> This is the single highest-value new overlay. s233's verdict is literally this rendered as prose — *"delivered along 16.90°, the 4 departs at 22.03° into the LEFT LONG RAIL ~11 in short of the pocket"* — and **Joe had already read it off the screen himself**. It converts the aim line from a bare ray into a consequence. Build it.

**C — DELIVERY PROOF.** Stick line (amber) vs actual CB launch line (white), wedge between them shaded and labelled `+1.3° squirt` · **plus a zoomed address inset** showing the shaft/tip axis, the ball centre, and `o_tip` as a labelled perpendicular tick ("½ tip left"). This is theory's "invisible by construction" fault made visible — the highest-value thing an instrumented system can catch.

**D — CONTACT PROOF.** Contact-instant composite: OB at rest, ghost circle dashed, CB at the back-extrapolated contact position solid, `Δe` drawn between the centres in ball fractions · tangent line + line of centres + actual OB ray, with the **throw wedge** shaded.

**Rendering contract (preserves the codebase's existing hard requirement).** All primitives are computed **server-side** and stored in the same normalized-video-coord form the current `aim` field uses. Add per shot in `.shots.json`:
```json
"diag": {
  "v": 1, "label": "DELIVERY/UNINTENTIONAL_ENGLISH", "confidence": "high",
  "card": {"verdict": "...", "budget": "...", "cause": "...", "fix": "...", "pattern": "..."},
  "num":   {"aim_err": -0.68, "delivery_err": 1.26, "tol_cb": 0.054, "A": 11.9,
            "o_tip_R": -0.25, "throw": 2.26, "s_mouth": 1.356, "miss_in": 2.49, ...},
  "proof": {"lines": [...], "circles": [...], "arcs": [...], "labels": [...]}
}
```
Client renders primitives and computes nothing — phone and desktop cannot disagree, which is exactly what `shots_export._shot_aim` and `cue_aim.py` already promise in their docstrings. ~1 KB per shot. Extend the existing `ov-aim` / `ov-paths` toggle pattern with `ov-proof`, defaulting to *the proof set for this shot's label only*.

### 3.3 The aggregate — the systematic bias hunt

Per diagnosable shot, one row:
```
shot_id, session, t, cut_dir(L/R), |θ_req|, d_cue_obj, d_obj_pocket, A, tol_CB,
aim_err, delivery_err, squirt_pred, steer_residual, swerve, throw, net_cb_err,
Δe_ballfrac, o_tip_R, tip_height, v_cb, ob_arrival_speed, miss_in, s_mouth,
outcome, rattled, label, confidence, target_source(inferred|declared)
```

**Hunt 1 — L/R asymmetry.** Mean `net_cb_err / tol_CB` by cut direction. Target sentence: *"you overcut left-cut shots by 1.8° on average; your right cuts are clean."* Theory: signed bias, **asymmetric** L/R ⇒ alignment / eye dominance / head position, **not** stroke. Prescription: video from behind, mirror alignment check, stance. Min n = 12/side (at per-shot σ ≈ 0.5°, SE at n=12 is 0.14°, so a 0.3° asymmetry is detectable and smaller is not — **state the detection floor in the UI**).

**Hunt 2 — unintentional english (already 3/3 in the forensics).** Regress `delivery_err` on `o_tip`. The slope **is Joe's personal squirt curve** — theory's calibration item, obtained from real play with no test protocol. The intercept is a stroke bias. The headline is `mean(o_tip)`: three for three **left**. Prescription: long straight-in stop shots — and **the app can score that drill automatically** via M20 (separation ≈ 90° and the CB stops dead).

**Hunt 3 — squirt vs throw, with no cue data at all.** Squirt error scales with `d_cue_obj`; throw is a fixed angle at contact so its miss scales with `d_obj_pocket`. Regress signed miss on both. Two coefficients, two diagnoses — and an **independent check on the cue-based attribution**.

**Hunt 4 — speed correlation.** Squirt is speed-independent (CSU cue-testing-machine; the opposite is an explicitly labelled myth). Squerve and throw both grow at slow speed. A speed-correlated directional bias is **swerve or throw, never raw squirt**.

**Hunt 5 — the near-straight trap.** Bin by `|θ_req|`: 0–2 / 2–5 / 5–15 / 15–30 / 30–50 / 50+. All three forensic shots had required cuts of **3.36°, 1.08°, 2.65°**. Expect a make-% notch at 2–5° that no "easy shots are easy" model predicts. Also bin by `d_cue_obj × d_obj_pocket` and by the halfway ratio — theory: a straight shot is hardest when the OB is exactly halfway.

**Hunt 6 — tolerance-adjusted skill.** Make% vs `|net_cb_err| / tol_CB`. Separates "shoots badly" from "picks low-percentage shots." Fit `σ` of net error; that feeds the ranking.

**Hunt 7 — THE RANKING (the answer to Joe's actual question).** For each candidate fix, recompute every shot's net error with that bias removed and sum `ΔP(make)`:
> **"Fixing your left-of-centre tip contact is worth ~2.4 makes per hour."**

That one sentence is the top line of the whole product; everything else is supporting evidence. Theory's principle: rank by expected points recovered, not by effect size. Until ~150–300 diagnosable shots exist for a fitted logistic, use the analytic window model `P(make) ≈ Φ(half_window; bias, σ)` — that needs only `σ`, estimable from ~30 shots.

**Hunt 8 — shots you should not have taken.** Flag `tol_CB < σ_stroke`. s233 was ±0.054° against a ~0.6° stroke. Aggregate: *"18 % of your attempts had a margin tighter than your own stroke consistency. On those you made 22 %. Your position play is choosing your make rate."* Theory calls shot-selection coaching worth more than aim coaching, and this costs nothing extra to compute.

**Hunt 9 — drift.** Aim bias and `o_tip` vs time-within-session (fatigue) and across sessions (did the correction stick?).

**Screen layout.** Top: ranked fix list, max 3, with makes/hour. Middle: two plots — signed error vs cut angle split by direction (the asymmetry), and `delivery_err` vs `o_tip` with the fitted squirt line and Joe's coefficient. Bottom: the tolerance-adjusted make curve with his shot selection marked. Everything click-through to the shots that made it.

---

## 4. WHAT'S MISSING — solve vs design around

| # | Gap | Verdict | Evidence / cost |
|---|---|---|---|
| 1 | **Calibration broken, 3 different ways, on 3/3 files.** Anisotropic rect space costs 0.9°. Table size configured, not detected. `geometry` 8ft=44 in vs `config` 8ft=46 in. | **SOLVE — blocking** | Nothing downstream means anything first. ~1 day. |
| 2 | **7–10 Hz tracker can't deliver launch direction** (s74: zero CB samples) | **DESIGN AROUND** — dense re-decode. Do *not* re-engineer the live tracker; it's a live-overlay product and the roadmap forbids derailing recording. | 40–60 frames/shot |
| 3 | **Trail timestamp lead +0.19 s** — 13.5 in mean position error during the fast phase | **SOLVE — cheap, and a correctness bug beyond analytics** (this is the data the phone draws as animated tails today) | cross-correlate + fix at source |
| 4 | **Frozen ghost tracks** (5 of 7 in s233) | **SOLVE — trivial**, ~5 lines. `describe.py` is already patching around this downstream (`_near_cue_at`, `stable_numbers`) — fix belongs upstream. | |
| 5 | **Aim detector nulls / mis-locks on real address frames** — glove splits the shaft, forearm alongside, naive fits 5° and 34° wrong; and it samples 0.45 s early | **SOLVE — high value** | skin-tone exclusion + elongation + outlier rejection + gap-tolerant merge + sample-at-contact |
| 6 | **Shaft-axis fits perspective-biased ~0.5°** | **SOLVE — and redirect the planned sub-pixel work to the TIP** | Otherwise you ship 0.05° precision on a 0.5° bias |
| 7 | **No tip-contact measurement** — lateral, and vertical | **Lateral: SOLVE** (tip-track extrapolation; 3/3, sign always cross-validated by squirt *and* throw). **Vertical: ACCEPT the indirect route** — the 90°/30° rules give tip height free from the CB's separation angle (s233: 136.8° ⇒ heavy draw) | |
| 8 | **Intended pocket** | **SOLVE — hybrid: auto-infer + a cheap tap. See below.** | |
| 9 | **30 fps blur** | **DESIGN AROUND** — 3/3 measured through it via streak centroids and multi-frame direction fits (0.15 px over 232 px). Revisit hardware only if `n_launch_samples < 4` on > 20 % of shots. | |
| 10 | **BLE cue sensor** (roadmap's stated differentiator) | **DEFER — worth less here than it looks.** All three misses were **tip placement**, and the stroke path measured clean from video in all three (0.32°, 0.06°, 0.28°). A butt IMU cannot see where the tip landed on the ball. Its real value is the exact impact timestamp (which would have caught the +0.19 s lead) and cue roll. | Safety: never write to the device |
| 11 | **Cling / skid** | **ACCEPT and abstain** — genuinely stochastic; the >5° gate plus "outlier, likely cling — not a fault" | |
| 12 | **Pockets are points, not mouths.** `TableModel` has centres + `pocket_radius_frac=0.045` | **SOLVE — cheap, high leverage.** Every tolerance figure needs **jaw-tip coordinates**: s233 measured a 3.42 in mouth, s96 3.2–3.5 in. A 3.4 in mouth vs a nominal 4.5 in changes tolerance by **45 %.** | 6 pockets × 2 points, clicked once per table |
| 13 | **"Rattled out" is not an outcome** | **SOLVE — ~15 lines.** Detect as sign flips in single velocity components (s74: x flips at the right jaw, then y at the bottom jaw). *Outside the window* vs *inside the window, hit too hard* are different prescriptions. | 2 of 3 shots rattled |
| 14 | **Identity errors** (roadmap Phase 2; c2–c5 all declined) | **ACCEPT with gating — do not wait for it.** The chain needs only two balls, both identifiable geometrically. Number identity is needed only for report language and legal-ball reasoning; degrade to "the yellow ball." | |

### 4.8 The tap — yes, and it unlocks more than it looks

**Yes, ask Joe to declare his intended pocket — but not for the reason it first appears.** Inference succeeded on 3/3 with 4–5 converging evidence lines, so the tap is not needed for most shots. Its real jobs:

1. **It breaks the circularity.** Two of the strongest inference terms are `stick` and `outcome`. A system that picks the target because the stick pointed at it **can never report a large aim error**. That is a structural blind spot; the tap is the only thing that closes it.
2. **Ask selectively:** (a) top-2 candidates within margin; (b) miss > 2 balls — precisely where inference is most likely to have latched onto the wrong pocket; (c) a random 1-in-10 audit sample regardless, purely to publish the inference's agreement rate. **Until that rate is measured, every number downstream carries the inference's uncertainty and the app should say so.**
3. **Two more fields on the same tap, near-zero extra cost, and one of them is the most valuable label in the system:**
   - **Intended english** (three buttons: centre / L / R). This converts "unintentional english" from an inference into a **fact**. Theory: an unintentional-english bias is *invisible to the player by construction* — and the forensics found it three times out of three. Without this label the app can measure `o_tip = 0.25 R left` but cannot say whether that was a mistake or a plan, which is the difference between a coaching insight and a physics readout.
   - **Intended cue-ball landing** (one tap on the schematic) — the only way position play ever becomes scoreable.
4. **Cost:** ~10–15 % of shots, one tap each, during review. Under 30 s per session. **The cheapest capability upgrade in this entire spec.**
5. **Do not prompt on every shot.** Joe's standing instruction is that he doesn't want to be technical director — he wants to use it. A per-shot prompt gets abandoned, and then the ground-truth channel dies with it.

---

## 5. BUILD ORDER

| Step | Ship | ~Cost | Why here |
|---|---|---|---|
| **0** | `vision/tablespace.py` — TRUE-inch frame, the 7 audit gates, detected table size, the 44-vs-46 fix, `tools/audit_calibration.py` over the whole archive | 1 d | **Blocking.** All three forensics rebuilt calibration by hand and said so first. You also learn on day one how much of the archive is analysable. |
| **1** | `analysis/forensic.py` — dense re-decode, sub-pixel centroids, TLS fits, back-extrapolated contact, jaw-tip pocket model, `s_mouth`, `tol_CB`, `A`, throw. Plus ghost-track filter, trail-lead fix, rattle detector. **Card + overlay set A.** | 2 d | **With no cue data at all this delivers theory's Level A + Level B** — signed overcut/undercut in ball fractions, the miss in inches, which jaw, the throw, and the shot's own tolerance. Theory calls Level A "the single highest-value measurement." It works today even when the aim detector nulls (s74). |
| **2** | Target inference + `TARGET_UNCERTAIN` + the review-screen tap (pocket · intended english · intended CB landing). Start logging agreement rate immediately. | 1 d | Cheapest item in the plan, and it is what makes Step 1's numbers trustworthy rather than circular. |
| **3** | **Cue-tip track** (ferrule detection through backswing + forward stroke) replacing the shaft fit; skin exclusion + gap-tolerant merge; `φ_aim@contact`; `o_tip`; `stroke_divergence`. **Overlay sets B and C.** | 3 d | This is the split — AIM vs DELIVERY — that made 2 of 3 forensics diagnosable. **Redirect the planned "full-shaft sub-pixel" work here**, for the parallax reason (§M9). |
| **4** | The decision rule, the abstention taxonomy, the five-line card copy | 1 d | ~200 lines of thresholds and prose on top of finished inputs. It's the thing Joe actually reads. |
| **5** | Per-shot row schema, the nine hunts, the ranked makes/hour list. **Batch-backfill the forensic pass over the entire archive first** (offline, resumable, same shape as `tools/score_corpus.py`). | 2 d | **Probably the highest-leverage sequencing decision in the plan** — backfilling turns "the answer in six weeks" into "the answer this afternoon," and Hunt 2 needs volume to state Joe's squirt coefficient with confidence. |
| **6+** | BLE stroke kinematics · higher fps · identity model · auto-scored stop-shot calibration drill | — | All deferred behind evidence. |

**Validation gate, in the roadmap's own idiom.** *No analytics change ships without reproducing the three forensic shots.* Freeze their hand-measured values as `tests/fixtures/forensics/{s233,s74,s96}.json` — `φ_cb`, `φ_aim`, `o_tip`, `Δe`, `throw`, `s_mouth`, `tol_CB`, and the root-cause label — and require the pipeline to match within **±0.15° on angles and ±0.1 in on distances**. These three shots are the only ground truth that exists, and they cost a great deal to produce.

---

## 6. FLAGGED — what the three real shots proved is harder than it looks

1. **Calibration failed 3/3, in three different ways.** Assume it is broken until audited.
2. **The sidecar was unusable 3/3**, for three different reasons. Nothing in this spec can be built on it.
3. **Taking an angle in rect space costs ~0.9°** — larger than every fault diagnosed, and on s96 it pointed the wrong way, i.e. it would have exonerated a real aim error.
4. **Shaft-axis fits carry ~0.5° of elevation/parallax bias.** Only the tip track is valid. A precision upgrade on the wrong primitive is worse than no upgrade — it makes a bias look authoritative.
5. **Aim sampled during the backswing is 0.2° stale** on an unusually steady stroke, and arbitrarily wrong on one that steers. The current `_shot_aim` samples 0.4–1.8 s early.
6. **The aim detector is actually good when it works** — 0.03° and 0.01° agreement with independent hand fits — and it **nulled on one shot because of a black glove**. Both facts matter: don't rewrite it, harden it.
7. **Two links pointed opposite ways and partially cancelled on 2 of 3 shots.** Net error alone mislabels both.
8. **Contact is not an independent link** in the common case. Scoring it separately double-counts.
9. **Throw is 1.8–2.3°, not noise**, and recovered ~45 % of s74's geometric error. Omit it and the budget won't close, and the app will abstain on shots it could have diagnosed.
10. **Target inference leans on the stick and the outcome** — both circular. This is the structural argument for the tap.
11. **The budgets are ±0.05–0.19° and the errors are 0.25–1.26°.** There is no slack anywhere in the chain; every bias listed above is signal-sized.
12. **All three misses were near-straight shots** (required cuts 3.36°, 1.08°, 2.65°) — the shots hardest to diagnose *and* the ones he is missing. Build the 2–5° bin early.
13. **A hand entering the frame was read as a ball collision** (s96). The OB path must be truncated at interference, not explained.
14. **Three for three, the tip landed left of centre, and three for three the ball squirted right** — 50 %, 30 %, 11 % of max offset producing 1.26°, 0.87°, 0.25°, against Dr. Dave's published 2.5°-at-max curve which predicts 1.25°, 0.74°, 0.27°. n=3, so indicative rather than established — but if that survives the backfill, **it is the answer to "what should I fix about my game," it was invisible to Joe by construction, and it is the reason to build §M9 before anything else in the cue chain.**