# State-opinion map — 2026-08-27 (evidence for the demolition ledger)

Raw output of the 5-agent audit (map + adversarial gap hunt) that inventoried
every private opinion-holder of table state. Each entry: file:line, what opinion
it holds, what it reads, and its migration onto MeasurementCore.
Findings drive docs/ARCHITECTURE.md §5; line numbers valid at commit 1b1bdac.

```json
{
  "summary": "Map every live-path consumer/deriver of table state + async timing hazards",
  "agentCount": 5,
  "logs": [
    "117 state-opinion sites mapped"
  ],
  "result": {
    "maps": {
      "ui-consumers": {
        "notes": "Architecture as-built: vision/pipeline.py Pipeline owns the OLD live tracker (vision/tracking.py BallTracker, pipeline.py:96) plus its own ShotDetector (pipeline.py:97); measure/tracker.py MotionTracker is used only by the offline M1 engine (measure/engine.py:147); playback uses a THIRD source, SidecarReader tracks (pipeline.py:246). workers/controller.py then re-derives more opinions on top (cue-clock rest/strike from track speeds, settled/moving edges, narration carry-motion, presence) and flattens everything into an ad-hoc FramePacket (controller.py:50-71, assembled at 1504-1519) — that packet is today's de-facto facade, and every UI consumer reads it or keeps a private copy of some slice of it. Known disagreement vectors found: (1) three track sources (live BallTracker vs M1 MotionTracker vs sidecar) with different active/number semantics; (2) shot-clock ring denominator: live_page.py:1435-1437 draws the ring against settings.shot_clock.seconds while clock_bar gets packet.clock_total — a break countdown (break_seconds) renders inconsistently between the ring and the bar; (3) recording/stats gating duplicated: controller gates shot COUNTING on _session_id, live_page separately gates DISPLAY on its own _recording_on/_stats_active copies; (4) StayDownTimer and on_stroke_measured match records to shots by ±6s time tolerance instead of shot identity; (5) shot history exists in four private copies (controller repo/DB, ShotTimeline._shots, ShotListPanel._shots, sidecar), corrections mutate shared dicts to stay in sync; (6) presence authority (TablePresence) lives in the controller, not in measure/, and is fed from OLD-tracker tracks. Migration shape implied by the sites: MeasurementCore facade = {frame(t): {tracks, raw_dets, presence, shot_state, motion, carried}, clock: {remaining,total,warning,running,paused,expired,phase}, session: {recording, elapsed, stats}, shot_log: [ShotRecord keyed by shot id, corrections + stroke merged in-core], overlays_at(t), narration/announcement events}. TablePresence and MotionTracker are the seeds already in measure/; the controller's cue-clock and narration policy should consume core motion primitives (cue_at_rest, all_settled, movers, carried displacement) rather than raw tracks. Renderers (overlay.py render_schematic/draw_perspective, VideoView) are already nearly pure — they just need to be fed core snapshots instead of pipeline-internal state (_play_paths, view_tracks extrapolation, _last_schematic cache). Files verified in full; line numbers are 1-indexed against current main (clean tree).",
        "sites": [
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/ui/pages/live_page.py",
            "line": 127,
            "holds": "_last_packet: cached newest FramePacket so Training Mode can label the frame already on screen",
            "reads": "entire FramePacket (perspective, raw_dets)",
            "migrate": "replace with core.latest() snapshot query on the facade; no UI-side caching of measurement state"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/ui/pages/live_page.py",
            "line": 1396,
            "holds": "_ball_markers: derives hover-reveal marker list (label 'Cue'/'N'/'?') for the bird's-eye schematic",
            "reads": "packet.tracks (number, x, y, radius) in rectified px",
            "migrate": "core.balls() display snapshot with resolved labels; labeling convention lives in the core, view only formats"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/ui/pages/live_page.py",
            "line": 1408,
            "holds": "on_frame: per-frame dispatch; keeps _frame_wh; feeds both video panes + feed chip",
            "reads": "packet.perspective, packet.birdseye, packet.feed_info, packet.feed_sd",
            "migrate": "subscribe to core.frame_snapshot(t); frame geometry and feed-health become core fields, not re-read from the image"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/ui/pages/live_page.py",
            "line": 1434,
            "holds": "shot-clock ring update; DERIVES the ring total from settings.shot_clock.seconds (disagrees with packet.clock_total on break countdowns)",
            "reads": "packet.clock_enabled, packet.clock_remaining, packet.clock_warning, settings.shot_clock.seconds",
            "migrate": "read core.clock {remaining,total,warning,enabled} verbatim; drop the settings-based denominator"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/ui/pages/live_page.py",
            "line": 1458,
            "holds": "_update_indicators: derives MODE pill (LIVE/PREVIEW/PLAYBACK) from packet.status crossed with its own _is_video/_recording_on copies",
            "reads": "packet.status, self._is_video, self._recording_on",
            "migrate": "core.mode enum (live/preview/playback/idle) + core.session.recording; page renders one field"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/ui/pages/live_page.py",
            "line": 1475,
            "holds": "alert-pill severity ladder: private priority ordering over degraded-feed / no-lock / relock / calibrating / shot-in-play",
            "reads": "packet.feed_sd, packet.status, packet.deviated, packet.shot_state",
            "migrate": "core.condition() returns the single most-severe transient condition; the ladder moves into the core"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/ui/pages/live_page.py",
            "line": 1492,
            "holds": "stay-down feed: passes shot_state+pipeline_t into StayDownTimer, gated on its own _recording_on copy",
            "reads": "packet.shot_state, packet.pipeline_t",
            "migrate": "core publishes per-shot stay_down {estimate|pending|final|popped}; page just paints text+kind"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/ui/pages/live_page.py",
            "line": 1496,
            "holds": "clock status label via game.shot_clock.status_text; caches _clock_status_last",
            "reads": "packet.shot_state, packet.clock_running, packet.clock_paused, settings.shot_clock.enabled",
            "migrate": "core.clock.phase (off/paused/on_the_clock/shot_in_play/settled); status_text becomes a core-side derivation"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/ui/pages/live_page.py",
            "line": 1505,
            "holds": "clock-bar feed (depleting Matchroom strip)",
            "reads": "packet.clock_remaining, clock_total, clock_warning, clock_running AND settings.shot_clock.enabled",
            "migrate": "core.clock snapshot including enabled; remove the settings read"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/ui/pages/live_page.py",
            "line": 1513,
            "holds": "tray presence relay (already reformed to render-verbatim)",
            "reads": "packet.present dict from the controller-owned TablePresence",
            "migrate": "unchanged contract, new source: core.presence — TablePresence instance moves inside MeasurementCore"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/ui/pages/live_page.py",
            "line": 1517,
            "holds": "on_stats: scoreboard rows; gated by private _stats_active copy of recording state; derives SHOTS = makes+misses",
            "reads": "controller stats_updated dict (makes, misses, make_pct, current_streak) from repo.session_summary",
            "migrate": "core.session.stats (incl. total shots) + core.session.active; drop the local gate copy"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/ui/pages/live_page.py",
            "line": 1527,
            "holds": "on_shot: appends to timeline + shot list from the raw ShotEvent; re-anchors StayDownTimer",
            "reads": "ShotEvent.start_t/end_t/outcome.value/num_pocketed (controller-rebased times)",
            "migrate": "core.shot_log append events carrying a stable shot id + rebased times; widgets bind to the log"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/ui/pages/live_page.py",
            "line": 1542,
            "holds": "on_stroke_measured: joins stroke records to shots BY START-TIME proximity; forwards to timeline/list/stay-down",
            "reads": "stroke rec dict (start, stay_down_s, popped_early, confidence, ...)",
            "migrate": "core merges stroke into the ShotRecord by shot id before publishing; UI receives one updated record"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/ui/pages/live_page.py",
            "line": 1229,
            "holds": "_feed_analysis_overlays: holds _overlay_doc (shots.json) and re-implements shot containment (start-5s..end+1.5s) per frame",
            "reads": "packet.media_t + sidecar shots doc (aim, trails, tags, lines)",
            "migrate": "core.overlays_at(media_t) — containment + selection logic lives once in the core (shared with the phone)"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/ui/pages/live_page.py",
            "line": 1257,
            "holds": "_ingest_label_frame: builds _label_balls editing model from raw detections; keeps _frame_wh",
            "reads": "packet.raw_dets (camera-px x, y, radius, guessed number), packet.perspective",
            "migrate": "core.raw_detections(t) snapshot; the editable label list stays UI-local (it is user input, not measurement)"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/ui/pages/live_page.py",
            "line": 878,
            "holds": "_update_stats_active + _stats_active/_recording_on: private mirror of recording state that gates the whole scoreboard",
            "reads": "recording_changed signal (bool)",
            "migrate": "core.session.active read directly; single source for 'stats belong to a recording'"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/ui/pages/live_page.py",
            "line": 1688,
            "holds": "on_recording: mirrors recording on/off, runs its own wall-clock rec timer feeding _timeline.set_live_clock (a SECOND clock next to pipeline_t)",
            "reads": "recording_changed bool; time.monotonic()",
            "migrate": "core.session.elapsed (recording timebase) drives both the rec clock and the timeline live window"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/ui/widgets/stay_down.py",
            "line": 155,
            "holds": "StayDownTimer: own strike anchor, prev shot_state edge detector, climb/wait/final/popped state; matches shot events and stroke records by +/-6s tolerance",
            "reads": "shot_state string + pipeline t per frame; ShotEvent.start_t; stroke rec dict",
            "migrate": "delete the derivation: core owns the stay-down measurement per shot id (estimate then final); widget becomes a formatter"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/ui/widgets/ball_tray.py",
            "line": 31,
            "holds": "_present dict copy (render-only; the old private debounced copy was removed after tray/schematic disagreed)",
            "reads": "presence dict handed down from packet.present",
            "migrate": "unchanged rendering; source becomes core.presence via the facade so tray, schematic and announcements share one dict"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/ui/widgets/clock_bar.py",
            "line": 96,
            "holds": "_frac/_warning/_expired; RE-DERIVES expired (remaining<=0) and frac locally",
            "reads": "remaining, total, warning, running scalars relayed by live_page",
            "migrate": "core.clock snapshot carries expired + fraction; widget paints without arithmetic"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/ui/widgets/shot_clock_widget.py",
            "line": 231,
            "holds": "update_clock copies remaining/total/warning/enabled for the countdown ring",
            "reads": "values from live_page.on_frame (total = settings.seconds, NOT packet.clock_total — break-length countdowns render a wrong ring)",
            "migrate": "feed from core.clock {remaining, total} so ring and bar always share the same denominator"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/ui/widgets/shot_timeline.py",
            "line": 64,
            "holds": "_shots private copy of the shot history + own live-clock smoothing state (set_live_clock fed by the UI rec timer)",
            "reads": "add_shot/set_shot_stroke calls (start, end, outcome, pocketed, stroke); matches stroke by start time",
            "migrate": "bind to core.shot_log (id-keyed records incl. corrections + stroke); live playhead from core.session.elapsed"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/ui/widgets/shot_list.py",
            "line": 138,
            "holds": "_shots/_all_shots: second private copy of shot history; outcome corrections mutate the shared dicts to propagate",
            "reads": "set_shots/add_shot dicts; stroke matched by start time",
            "migrate": "same core.shot_log binding as the timeline; corrections go THROUGH the core (append_correction) and republish"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/ui/widgets/video_view.py",
            "line": 27,
            "holds": "_overlay (label markers) + _balls (hover markers): retained per-frame marker geometry in image px",
            "reads": "lists pushed by live_page (_ball_markers from packet.tracks; label overlay from _label_balls)",
            "migrate": "markers derive from core.balls() in the page; view stays a dumb painter"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/ui/widgets/video_view.py",
            "line": 62,
            "holds": "set_analysis: retains _aim/_trails/_tags/_media_t and draws trails with its own 4s fade window (phone-parity constant duplicated here)",
            "reads": "shots.json geometry selected by live_page._feed_analysis_overlays",
            "migrate": "consume core.overlays_at(t) output; fade window becomes a core/render constant shared with the phone"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/ui/widgets/mini_view.py",
            "line": 120,
            "holds": "on_frame: frame-rate halving tick + aspect cache; picks perspective vs birdseye",
            "reads": "packet.perspective, packet.birdseye",
            "migrate": "same core.frame_snapshot subscription as the live page (throttling stays local)"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/ui/widgets/mini_view.py",
            "line": 147,
            "holds": "_recording/_running copies for the status dot",
            "reads": "recording_changed + status_changed signals",
            "migrate": "core.session.recording + core.mode; still event-driven, one authority"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/ui/main_window.py",
            "line": 457,
            "holds": "_on_shot_sound (voice announcements): re-derives scratch vs foul vs make-by-ball-name vs miss, filters cue from pocketed list, picks first named ball, maps pocket keys to spoken names",
            "reads": "ShotEvent.outcome.value, cue_scratch, foul, pocketed[].number/.pocket/.cls (from both shot_recorded and shot_observed)",
            "migrate": "core publishes a resolved AnnouncementEvent {verdict, ball_number, pocket_key} per finalized shot; window only maps to speech"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/ui/main_window.py",
            "line": 427,
            "holds": "_on_narration: maps controller Narrator kinds (ball_in_hand/table_change) to phrases",
            "reads": "narration signal string",
            "migrate": "unchanged mapping; kinds emitted by core table-event stream instead of controller-internal Narrator"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/ui/main_window.py",
            "line": 493,
            "holds": "_on_clock_event: plays warn/tick/expired audio + spoken Ten/Time Foul from clock edge strings",
            "reads": "clock_event signal edges from controller ShotClock.poll",
            "migrate": "core.clock edge events; identical consumer, single clock authority"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/workers/controller.py",
            "line": 50,
            "holds": "FramePacket dataclass: today's ad-hoc facade — flattened copy of tracks/present/shot_state/clock/raw_dets/feed per frame",
            "reads": "PipelineResult + controller-local clock/presence/feed state",
            "migrate": "replace with the MeasurementCore snapshot type; the core (not the controller) assembles it"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/workers/controller.py",
            "line": 112,
            "holds": "self._presence = TablePresence(): THE presence authority currently lives in the controller, updated at line 1501 from OLD-tracker track numbers filtered by tr.active",
            "reads": "res.tracks numbers each frame + pipeline t",
            "migrate": "TablePresence moves inside MeasurementCore, fed by the hardened MotionTracker's rows; packet.present becomes core.presence"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/workers/controller.py",
            "line": 1656,
            "holds": "_handle_state + _prev_state: controller's OWN settled/moving edge detector (duplicates the ShotDetector state machine) driving clock start/stop and _turn_start_t",
            "reads": "res.shot_state per frame",
            "migrate": "core emits settle/strike edge events; clock policy subscribes instead of re-differencing state strings"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/workers/controller.py",
            "line": 1680,
            "holds": "_update_cue_clock: derives cue-at-rest (6-frame stillness run), cue-rolling strike, all-balls-still veto, cue-absence gap => ball-in-hand re-arm, break detection (5+ movers) — counters _cue_still/_saw_cue_t/_clock_armed/_break_pending/_strike_stop_t",
            "reads": "res.tracks (cls, speed, active) + settings.balls.stop_speed",
            "migrate": "core publishes motion primitives {cue_at_rest, cue_moving, all_settled, movers, cue_absent_since}; clock policy consumes primitives, keeps only policy state"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/workers/controller.py",
            "line": 1393,
            "holds": "narration gate: CarryMotion anchors + Narrator episode state derive ball_in_hand/table_change from carried-ball displacement",
            "reads": "res.tracks joined with res.carried_ids (id, x, y, radius, is-cue) + t",
            "migrate": "core owns carried-ball displacement tracking and emits table events; controller (or core) publishes narration kinds"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/workers/controller.py",
            "line": 1419,
            "holds": "shot counting gate: session-open check decides record vs shot_observed; _record_shot rebases event times to sidecar t0 for the UI",
            "reads": "res.shot_event, self._session_id, sidecar._t0",
            "migrate": "core.shot_log does the rebase once and tags records session/sandbox; both signals collapse to one log stream"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/workers/controller.py",
            "line": 1788,
            "holds": "stats emission from repo.session_summary after each recorded shot (separate channel from frame state)",
            "reads": "DB session summary (makes/misses/pct/streak)",
            "migrate": "core.session.stats derived from core.shot_log; DB becomes a sink, not the UI's stats source"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/vision/pipeline.py",
            "line": 96,
            "holds": "Pipeline owns the OLD live tracker (BallTracker) + its own ShotDetector — the tracker MeasurementCore is meant to replace for live",
            "reads": "prepared detections per frame",
            "migrate": "pipeline delegates to core's MotionTracker (measure/tracker.py) behind the facade; ShotDetector consumes core tracks"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/vision/pipeline.py",
            "line": 714,
            "holds": "_update_play_paths: per-ball trail dict, fade alpha, _prev_shot_state edge, own 'really moving' displacement test over track history",
            "reads": "tracks (id, x, y, vx, vy, number, bgr, history, misses) + shot_state + t",
            "migrate": "core maintains trails + settle-fade as derived measurement state; renderer reads core.trails"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/vision/pipeline.py",
            "line": 780,
            "holds": "view_tracks: velocity-extrapolated display copies between async ingests (_frames_since_ingest counter)",
            "reads": "tracker tracks (vx, vy, misses)",
            "migrate": "core.render_tracks(t) — interpolation/extrapolation inside the facade so all surfaces glide identically"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/vision/pipeline.py",
            "line": 1044,
            "holds": "display-only frames reuse self.tracker.tracks + _last_detections/_last_raw_dets caches",
            "reads": "tracker internals directly",
            "migrate": "core snapshot is the only read path; caches become core-internal"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/vision/pipeline.py",
            "line": 1120,
            "holds": "shot_state + shot_event from self.shots.update; also _update_play_paths side effect",
            "reads": "tracks, table, motion energy, foreign/carried evidence",
            "migrate": "core.shot_state/core.shot_events; evidence fusion inputs become core measurements"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/vision/pipeline.py",
            "line": 1203,
            "holds": "schematic render + _last_schematic staleness cache (invalidated by ingest_raw_detections at line 712)",
            "reads": "view_tracks(tracks), _play_paths, _paths_alpha, detections, diag",
            "migrate": "render from core snapshot keyed by a core revision counter; staleness = revision compare, not manual None-ing"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/vision/pipeline.py",
            "line": 246,
            "holds": "_process_cached: playback tracks from SidecarReader.tracks_at (third parallel track source, own clock-offset handling)",
            "reads": "sidecar dense frames + video_time_offset",
            "migrate": "core facade gains a sidecar-backed mode: same tracks_at interface live and playback, offset handled once in-core"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/vision/overlay.py",
            "line": 473,
            "holds": "render_schematic: the bird's-eye schematic renderer (pure function; reads identity from colour/stripe)",
            "reads": "tracks (x, y, number, cls, bgr, history), play_paths, detections",
            "migrate": "signature takes a core RenderSnapshot; no behavioural change needed — it is already state-in, pixels-out"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/vision/overlay.py",
            "line": 558,
            "holds": "draw_perspective: projects rectified tracks back to camera px via Hinv for the live-view overlay",
            "reads": "tracks + calib.corners/Hinv/table",
            "migrate": "consume core snapshot + core.rectify transforms (same inversion the phone overlays use)"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/events/shot_detector.py",
            "line": 113,
            "holds": "ShotDetector private world model: _cue_id cue-presence opinion (line 345), _rest positions, _pocketed credits, _shot_ids/_shot_cls/_shot_num, carried counters — a parallel presence/pocket bookkeeping next to TablePresence",
            "reads": "live tracks + table geometry + fused motion evidence",
            "migrate": "consume core tracks/presence; pot credits cross-checked against core.presence transitions instead of only its own _pocketed list"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/game/shot_clock.py",
            "line": 130,
            "holds": "status_text: pure derivation of the rail label from shot_state + clock flags (already 'one source of truth' for the string)",
            "reads": "shot_state, running, paused, enabled scalars",
            "migrate": "becomes core.clock.phase -> label; called by the core when building the snapshot so every surface gets the same string"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/game/narration.py",
            "line": 26,
            "holds": "CarryMotion (per-track displacement anchors) + Narrator (episode/cooldown state) deriving ball_in_hand/table_change",
            "reads": "carried track positions + cue-reappearance pings from the controller",
            "migrate": "displacement measurement moves into the core; Narrator keeps only utterance policy (cooldowns), consuming core table events"
          }
        ]
      },
      "detector-opinions": {
        "notes": "FILE: c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/events/shot_detector.py (ShotDetector, 550 lines). Caller: vision/pipeline.py:1120 `self.shots.update(tracks, calib.table, t, motion, evidence)` — evidence dict built at pipeline.py:1102, carried_ids computed by pipeline._carried_ids (pipeline.py:925).\n\nUPDATE() ARGS CONSUMED: (1) tracks: list[Track]; (2) table: TableModel — table.short_side (line 242, ball-radius estimate), table.nearest_pocket(x,y) (391, 434), table.pocket_radius (403); (3) t: float — warmup (185), cooldown (186), backdated start (375-376), calm timing (300-313), event timestamps; (4) motion: float — whole-table frame-diff energy, used raw (183-184) or fused (178); (5) evidence: dict — keys read: \"carried_ids\" set[int] (174, 205, 260), \"arm\" float (250, 544), \"flow\" (339, 540), \"fg\" (340, 541), \"motion\" only as its own default (169).\n\nTRACK FIELDS CONSUMED: tr.id (170, 194, 213-233, 261-264, 271-289, 322-323, 347, 387), tr.x/tr.y (194, 220, 263-264, 277, 291 via loop, 323, 391, 461, 465), tr.cls (323, 346, 388), tr.number via getattr (389, 460), tr.active (402, 458), tr.speed (323 only — stored into _Seen.speed and NEVER read afterward: dead). NOT consumed: radius, vx/vy directly, history, age, hits, misses, bgr. LATENT BUG at line 464: `getattr(prev, \"radius\", 0.0)` — prev is a _Seen (x,y,cls,speed only), so radius is ALWAYS 0.0 and the re-acquisition veto radius is always the 12.0px floor; a MeasurementCore last-seen record carrying real radius fixes this for free.\n\nCLASSIFICATION SUMMARY — duplicated table-state opinions that MeasurementCore should own: _state settled/moving machine, _active_run/_quiet_run/_still_run/_calm_since (table activity + all-balls-settled), _cue_id (identity), _shot_cls/_shot_num (last-known identity of dead tracks), _prev/_prev_seen_at (last-seen memory for vanished tracks — tracker.py's coast/rest-anchor already models this), _free_frames/_free_travel/_pending_free/_pending_free_t0/_pending_quiet (per-track free-motion kinematics and rest-departure events), _last_carried (carried history), _rest (DEAD: written at 194, never read — tracker's rest anchor ax/ay superseded it), _frame_idx (private clock), _fuse() (central activity score other subsystems will want), and the pocketed/vanished determination in _finalize_pockets/_still_on_table (ball-absence is exactly what measure/presence.py TablePresence measures; the identity-churn veto at 445-469 exists only because per-track-id bookkeeping diverges from physical-ball truth — presence-by-number kills the whole class of bug). GENUINELY SHOT-SCOPED (fine to keep local): _first_t/_last_shot_t (warmup/cooldown), _shot_ids (participant roster), _min_pdist (closest-pocket-this-shot), _pocketed (attribution of absences to this shot+pocket), _shot_frames/_arm_frames, _carried_moving/_any_moving ratio (gathering veto — though its inputs should come from core), _start_t/_max_travel, last_event/last_diag. Migration shape: ShotDetector keeps only the shot window + attribution; every per-track kinematic/identity/presence question becomes a query against MeasurementCore (tracks incl. recently-dead with last-seen pos/time/radius/class/number/carried flags, free-step and cumulative-free-travel per track, rest-departure events with t0, table settled/active durations, ball-number presence).",
        "sites": [
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/events/shot_detector.py",
            "line": 114,
            "holds": "_state: private SETTLED/MOVING table state machine — its opinion of whether the table is in motion",
            "reads": "motion float + evidence fusion (178-184), per-track steps from _prev, t",
            "migrate": "Core owns table activity state; detector reads core.table_state (settled/moving + since-when) and keeps only 'shot open/closed' as the shot-scoped wrapper"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/events/shot_detector.py",
            "line": 115,
            "holds": "_cue_id: private opinion of which track id is the cue ball, sticky across frames (_update_cue 345-349)",
            "reads": "tr.cls == BallClass.CUE, tr.id",
            "migrate": "Core identity/arbitration owns cue assignment; read core.cue_track() (number 0) — removes the stale-id case _update_cue papers over"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/events/shot_detector.py",
            "line": 116,
            "holds": "_first_t: session start epoch for warmup gate (185)",
            "reads": "t",
            "migrate": "Shot/session-scoped, fine local; optionally core.session_t0"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/events/shot_detector.py",
            "line": 117,
            "holds": "_last_shot_t: cooldown clock between shots (186, 519)",
            "reads": "t",
            "migrate": "Genuinely shot-scoped, keep local"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/events/shot_detector.py",
            "line": 118,
            "holds": "_rest: per-track rest-position snapshot — DEAD STATE, written at 194 while settled+quiet, never read anywhere",
            "reads": "tr.id, tr.x, tr.y",
            "migrate": "Delete; tracker.py _Track.ax/ay rest anchor is the central version of this concept"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/events/shot_detector.py",
            "line": 119,
            "holds": "_min_pdist: per-track closest-ever pocket approach during this shot (id -> (dist, pocket name)), fed at 391-394, consumed at 416",
            "reads": "tr.x, tr.y, table.nearest_pocket",
            "migrate": "Shot-scoped aggregation, fine local — but should be computable from core's dense per-track history over the shot window instead of live-frame sampling (blurred flyers never sample near the pocket)"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/events/shot_detector.py",
            "line": 120,
            "holds": "_shot_ids: roster of track ids seen during the shot (seeded from bank at 358, grown at 387)",
            "reads": "tr.id",
            "migrate": "Shot-scoped, keep — or replace with core.tracks_alive_during(start_t, t)"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/events/shot_detector.py",
            "line": 121,
            "holds": "_shot_cls: id -> last class seen — private identity memory for tracks that die mid-shot (388, read 418/435/506-518)",
            "reads": "tr.cls",
            "migrate": "Duplicates tracker identity memory; core should serve last-known class of any track id incl. dead ones — exists only because dead Tracks vanish from the update list"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/events/shot_detector.py",
            "line": 122,
            "holds": "_shot_num: id -> last confident ball number — same private identity memory for numbers (389-390, read 420/437/460)",
            "reads": "getattr(tr, 'number', -1)",
            "migrate": "Core serves number-of-track incl. after death; with presence.py's number-keyed store this becomes a lookup, not a cache"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/events/shot_detector.py",
            "line": 123,
            "holds": "_prev: private last-seen positional memory per track (x,y,cls,speed as _Seen), retained ~90 frames past death (322-329); basis for ALL step/displacement math (220, 263, 277) and unseen-pot pocket attribution (432-434) and re-acquisition veto (463-468)",
            "reads": "tr.x, tr.y, tr.cls, tr.speed (speed stored but never read)",
            "migrate": "The single biggest duplication: MotionTracker already coasts and knows last positions of vanished tracks; core should serve last_seen(track_id) -> (x, y, t, radius, cls, number) and per-frame step — deleting the 011510 class of bug at one site instead of every consumer"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/events/shot_detector.py",
            "line": 124,
            "holds": "_prev_seen_at: frame index each track was last seen, drives the 90-frame retention GC (325-329)",
            "reads": "tr.id, private _frame_idx",
            "migrate": "Folds into core's last-seen record (timestamped); retention policy becomes core's, uniform for all consumers"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/events/shot_detector.py",
            "line": 125,
            "holds": "_pocketed: accumulated PocketedBall list for this shot — the detector's private verdict that a ball left the table (built in _finalize_pockets 396-440)",
            "reads": "track absence from by_id/tr.active, _min_pdist, _prev, _free_frames, _last_carried, table.pocket_radius/nearest_pocket",
            "migrate": "Ball-absence is a table-state fact TablePresence (measure/presence.py) measures by NUMBER; core should assert 'ball N absent since t' and the detector only attributes that absence to a pocket + this shot. The pocket/time attribution stays shot-scoped"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/events/shot_detector.py",
            "line": 126,
            "holds": "_active_run: consecutive active-energy frames — private table-motion run used as fallback arm signal (188, 246)",
            "reads": "motion arg / fused evidence vs det.motion_active or det.fusion_active",
            "migrate": "Core exposes active_for (duration table has been active); detector compares against its arm threshold"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/events/shot_detector.py",
            "line": 127,
            "holds": "_quiet_run: consecutive quiet-energy frames — private settling opinion (189, 309)",
            "reads": "motion arg / fused evidence vs det.motion_quiet",
            "migrate": "Core exposes quiet_for; settle decision reads it"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/events/shot_detector.py",
            "line": 128,
            "holds": "_shot_frames: updates elapsed inside the open shot (249, gate 310)",
            "reads": "update cadence only",
            "migrate": "Genuinely shot-scoped, keep local"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/events/shot_detector.py",
            "line": 129,
            "holds": "_arm_frames: frames with arm-evidence >= 0.010, diagnostics only (250-251)",
            "reads": "evidence['arm']",
            "migrate": "Shot-scoped diagnostics, keep local"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/events/shot_detector.py",
            "line": 130,
            "holds": "_still_run: consecutive frames in which no tracked ball stepped >1px — private 'all balls at rest' opinion (298, gate 310)",
            "reads": "per-track steps derived from _prev + tr.x/tr.y",
            "migrate": "Duplicates tracker's per-track at_rest (tracker.py:203); core exposes all_settled + settled_for and this counter disappears"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/events/shot_detector.py",
            "line": 131,
            "holds": "_calm_since: wall-clock start of the current quiet-AND-still spell (299-303, gate 312-313)",
            "reads": "quiet flag + max_step, t",
            "migrate": "Same as _still_run: core's settled_since serves it"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/events/shot_detector.py",
            "line": 132,
            "holds": "_carried_moving: moving updates where a hand was on a mover — numerator of the gathering veto (266-268, 500)",
            "reads": "evidence['carried_ids'] intersected with privately-computed stepping set",
            "migrate": "The RATIO is shot-scoped and fine; the inputs (per-track carried flag, per-track step) should come from core Track state instead of a private _prev diff"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/events/shot_detector.py",
            "line": 133,
            "holds": "_any_moving: moving updates with any tracked step — denominator of the gathering veto (265-266, 500)",
            "reads": "same stepping set from _prev",
            "migrate": "Same: keep ratio local, source steps from core"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/events/shot_detector.py",
            "line": 134,
            "holds": "_free_frames: per-track count of free (non-carried) moving frames — kinematic re-derivation (280, read at 410/430/486/517)",
            "reads": "step from _prev, carried_ids exclusion",
            "migrate": "Core maintains free_frames per track (carried-excluded motion is a measurement, not a shot concept); detector reads window deltas"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/events/shot_detector.py",
            "line": 135,
            "holds": "_free_travel: per-track integrated free displacement — the decisive travel number (288-291)",
            "reads": "step from _prev, carried_ids exclusion",
            "migrate": "Core maintains cumulative free travel per track; detector takes travel-within-shot-window as a difference of core readings"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/events/shot_detector.py",
            "line": 136,
            "holds": "_pending_free: pre-shot per-ball motion bank (frames, travel) while SETTLED — private 'this ball departed rest' detector (204-233, arm signal 243-244, seeds shot 358/368-370)",
            "reads": "step from _prev, carried_ids, _last_carried recency, t",
            "migrate": "This IS the tracker's rest-anchor departure event re-implemented; core emits departed_rest(track, t0, free_frames, free_travel) and the bank becomes a query — the 011928 wipe bug class lives here"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/events/shot_detector.py",
            "line": 137,
            "holds": "_pending_free_t0: timestamp of first banked free step per ball — backdates shot start to the strike (224, 375-376)",
            "reads": "t",
            "migrate": "Comes free with core's departed_rest event t0"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/events/shot_detector.py",
            "line": 138,
            "holds": "_pending_quiet: per-ball quiet counter that expires a bank after 4 still frames (227-233)",
            "reads": "step from _prev",
            "migrate": "Core-side: a departure event that re-settles is retracted by the core, uniformly"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/events/shot_detector.py",
            "line": 139,
            "holds": "_last_carried: frame index a hand last touched each track — private carried-history for wobble/lift vetoes (174-175, read 213-214/431/487)",
            "reads": "evidence['carried_ids'], _frame_idx",
            "migrate": "Core keeps carried_until per track (a hand near a ball is a table observation); detector asks carried_recently(id, window)"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/events/shot_detector.py",
            "line": 140,
            "holds": "_frame_idx: private frame counter used as the clock for carried-recency and _prev staleness (173, 214, 326, 431, 487)",
            "reads": "update cadence",
            "migrate": "Use core frame index / timestamps; private clocks drift from the core's when updates are skipped"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/events/shot_detector.py",
            "line": 141,
            "holds": "_start_t: backdated shot start time (375-376)",
            "reads": "_pending_free_t0, t",
            "migrate": "Shot-scoped result, keep; its input t0 should come from core departure events"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/events/shot_detector.py",
            "line": 142,
            "holds": "_max_travel: max integrated free travel across balls this shot — the travel gate value (290-291, 481)",
            "reads": "_free_travel",
            "migrate": "Shot-scoped aggregate, keep; feeds from core free-travel"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/events/shot_detector.py",
            "line": 143,
            "holds": "_motion / _fused / _evidence (143-145): cached copies of the upstream activity signals plus the private _fuse() weighting (334-343) — a second opinion of 'how active is the table' alongside pipeline's",
            "reads": "motion arg, evidence['flow'/'fg'/'arm'/'carried_ids']",
            "migrate": "Fusion into one activity score is a core responsibility (one opinion, all consumers); detector reads core.activity and keeps only its thresholds"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/events/shot_detector.py",
            "line": 445,
            "holds": "_still_on_table veto (445-469): re-queries live tracks to detect identity churn (dead track A reborn as B) before crediting a pot — compensating logic for per-track-id state diverging from physical balls; also carries the latent bug at 464 where getattr(prev,'radius',0.0) on _Seen is always 0.0 so the match radius is always the 12.0 floor",
            "reads": "tr.active, getattr(tr,'number',-1), tr.x/tr.y vs _prev, _shot_num",
            "migrate": "With core presence-by-number (measure/presence.py) as the authority on 'is ball N on the table', this whole veto collapses into one core query and the radius bug dies with it"
          },
          {
            "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/events/shot_detector.py",
            "line": 146,
            "holds": "last_event / last_diag (146-147, _set_diag 534-549): published outputs for controller and debug overlay",
            "reads": "all of the above",
            "migrate": "Legitimate outputs, keep — these are what the detector CONTRIBUTES to the core, not opinions it should be sourcing"
          }
        ]
      },
      "async-timing": {
        "notes": "LIVE ASYNC DETECTION PATH — end-to-end trace (all paths under src/billiards_trainer/).\n\nTHREAD MAP: (A) Controller QThread — PipelineController lives on its own QThread (make_controller_thread, workers/controller.py:1815-1819 moveToThread); QTimer parented there (on_started, :199-206) drives _tick (:1051). ALL pipeline/tracker/presence mutation happens on this thread. (B) detect-worker — a plain daemon threading.Thread (:1291-1294), demoted to BELOW_NORMAL (:1305-1319); the ONLY pipeline object it touches is _strategy.detect (:1324), which is guarded by the strategy's own class-level lock (detector_strategies/onnx_model.py:60 _detect_lock). Results return to thread (A) via _detections_ready with Qt.QueuedConnection (:183), so ingest is serialized with the tick — this is the single-threading invariant, there is NO lock on pipeline state at all (grep confirms: only controller._rec_lock at :135, recording-only, and onnx_model._detect_lock).\n\nSTEP-BY-STEP: _tick (A, :1051) reads frame (:1081) -> live branch calls _run_frame(frame, detect=\"async\") (:1127). _run_frame (A, :1350) preprocesses (:1362) then _submit_detection(frame) at :1366 — NOTE this happens BEFORE t is computed. t = time.perf_counter() - self._t0 at :1376 (video sources override with media time :1380-1381). process(frame, t, detect=False) at :1384. _submit_detection (A, :1256) gates on enabled/strategy/calibrated (:1259-1263), idle-throttles via pipeline.last_motion + _prev_state (:1273-1282), enqueues (frame, self._pipeline.calib.calib) into _det_queue (maxsize=1, :181) at :1295 — **t IS LOST HERE: the tuple carries frame+calib only; t was never even computed at this point** (queue.Full silently drops, :1296-1297). _detect_worker (B, :1299) blocking-get (:1322), strategy.detect(frame, calib) (:1324), emits (raw, frame.shape) at :1328 — t absent AND the calib snapshot is dropped. _on_detections_ready (A via queued signal, :1337) guards not-running/no-pipeline/detection-off/is_video (:1340-1343) — so this ingest path is LIVE-ONLY — then ingest_raw_detections(raw_dets, frame_shape) (:1346), no t. ingest_raw_detections (A, vision/pipeline.py:697) re-fetches a FRESH calib at :701 (can be newer than the one the worker detected against if a relock landed in the inference window) and calls _apply_detections(raw_dets, calib, frame_shape) at :704 with frame=None (so blur recovery, pipeline.py:611-631, silently never runs on the live async path) and NO t; invalidates schematic (:712). _apply_detections (A, :538) -> prepare_detections (:338, the full bought filter stack) returns PREPARED dets at :542-543 with calib in scope but no t, then OLD BallTracker.update (:546-552, vision/tracking.py) plus vacancy pruning / number release / overlap projection (private opinions _vacant/_released, :550-680), caches _last_detections/_last_raw_dets (:693-694).\n\nWHERE t EXISTS vs LOST: t exists only on thread (A) inside _run_frame from :1376 onward (and inside process/shots). The ingest chain (:1295 enqueue -> :1328 emit -> :1337 slot -> pipeline.py:697 -> :704) carries NO t anywhere today. The only place prepared dets + t + calib coexist today is the SYNCHRONOUS video/step path: process(frame, t) detect=True branch calls _apply_detections at pipeline.py:1067 with t in scope in the enclosing process() (calib at :1026) — but t still isn't passed INTO _apply_detections. On the live path the shot detector runs on DISPLAY frames' t against ingest-cadence tracks (pipeline.py:1120), and presence runs on display t at display cadence (controller.py:1501) — the timestamp/cadence skew behind the visible disagreements.\n\nSAFEST SINGLE FEED POINT (no new locks needed): thread t through the existing chain and feed the shadow measure/tracker.MotionTracker + measure/presence.TablePresence at the tail of _apply_detections (or immediately after it returns inside ingest_raw_detections/process): (1) at controller.py:1295 compute t = time.perf_counter() - self._t0 (same clock base as :1376) and enqueue (frame, calib, t); (2) widen Signal at :96 to three objects and emit (raw, frame.shape, t) at :1328 — t rides through the worker untouched, capture-time semantics preserved despite ~83ms inference latency (matches the repo's retime-from-capture lesson); (3) forward t through :1337/:1346 into ingest_raw_detections(raw_dets, frame_shape, t) and _apply_detections(..., t); (4) at pipeline.py:693 (after prepare + tracker + pruning, prepared dets + calib + t all in scope, controller thread, sole mutator) call shadow MotionTracker.update([(d.x, d.y, d.radius, number) ...], t) (measure/tracker.py:69 signature) and TablePresence.update(numbers_seen, t) (measure/presence.py:33). Because _apply_detections is also the video path's tail (via pipeline.py:1067 where t is already in scope), passing t into _apply_detections from BOTH callers gives live and playback one identical feed site. Caveats: prefer passing the worker's calib snapshot through the emit instead of re-fetching at pipeline.py:701 (relock-window mismatch); presence at controller.py:1501 should then become a read of the core's opinion, not a second update from display tracks; note t here is perf-counter-based session time for live (ingest path is live-only per the :1340-1343 guard), media time for video via process().",
        "sites": [
          {
            "file": "src/billiards_trainer/workers/controller.py",
            "line": 1051,
            "holds": "_tick: frame-read loop, camera-miss watchdog state (_miss_t0/_got_frame), playback cadence (_play_tick, stride); runs on controller QThread via QTimer (:205)",
            "reads": "raw frames from self._source.read(); live branch dispatches _run_frame(frame, detect=\"async\") at :1127",
            "migrate": "unchanged; remains the single driver of the controller thread"
          },
          {
            "file": "src/billiards_trainer/workers/controller.py",
            "line": 1350,
            "holds": "_run_frame: display-path state (_last_frame, _fps, _video_pos); computes the frame timestamp t at :1376 AFTER _submit_detection at :1366 — the async submit never sees t",
            "reads": "preprocessed frame; pipeline.process(frame, t, detect=False) result (tracks/shot_state)",
            "migrate": "compute t before :1366 (or inside _submit_detection) and pass it with the frame; everything downstream of process() reads MeasurementCore outputs instead of PipelineResult-derived private state"
          },
          {
            "file": "src/billiards_trainer/workers/controller.py",
            "line": 1256,
            "holds": "_submit_detection: idle-throttle opinion (_quiet_frames, _last_quiet_submit, _diag_* counters) derived from _prev_state + pipeline.last_motion; enqueues (frame, calib) at :1295 — t IS LOST HERE (never enqueued, never computed)",
            "reads": "self._pipeline.calib.calib snapshot, pipeline.last_motion, _prev_state",
            "migrate": "enqueue (frame, calib, t) with t = time.perf_counter() - self._t0; quiet/motion gating should read MeasurementCore.motion instead of pipeline.last_motion"
          },
          {
            "file": "src/billiards_trainer/workers/controller.py",
            "line": 1299,
            "holds": "_detect_worker: runs on dedicated daemon Python thread 'detect-worker' (BELOW_NORMAL); stateless except the blocking queue; emits (raw, frame.shape) at :1328 dropping both t and the calib snapshot",
            "reads": "(frame, calib) tuples from _det_queue (maxsize=1, :181); calls _strategy.detect under onnx_model.py:60 _detect_lock — the ONLY cross-thread touch of pipeline-owned objects",
            "migrate": "pass t (and the calib snapshot) straight through the emit: _detections_ready.emit(raw, frame.shape, t[, calib]); widen Signal declared at :96"
          },
          {
            "file": "src/billiards_trainer/workers/controller.py",
            "line": 1337,
            "holds": "_on_detections_ready: queued-connection slot (wired :183, Qt.QueuedConnection) — re-enters the controller QThread, THE serialization point; guards live-only at :1340-1343; _diag_ingest/_diag_raw counters",
            "reads": "raw_dets + frame_shape from the worker signal; forwards to pipeline.ingest_raw_detections at :1346 with no t",
            "migrate": "forward t: ingest_raw_detections(raw_dets, frame_shape, t); this slot is where the shadow feed becomes race-free for free — same thread as all other pipeline mutation"
          },
          {
            "file": "src/billiards_trainer/vision/pipeline.py",
            "line": 697,
            "holds": "ingest_raw_detections: schematic-staleness invalidation (:712); re-fetches a FRESH calib at :701 that can differ from the calib the worker detected against (relock during inference)",
            "reads": "raw_dets + frame_shape; self.calib.calib; calls _apply_detections(raw_dets, calib, frame_shape) at :704 with frame=None (blur recovery never runs live) and no t",
            "migrate": "accept t (and preferably the worker's calib snapshot) and pass both into _apply_detections; after it returns, this is a valid alternate site to update MeasurementCore (controller thread, single mutator)"
          },
          {
            "file": "src/billiards_trainer/vision/pipeline.py",
            "line": 538,
            "holds": "_apply_detections: OLD BallTracker.update (:546-552) plus private opinions — _vacant vacancy counters, _released number-release set, spot-occupancy/covered heuristics, published-track overlap projection, _last_detections/_last_raw_dets caches (:693-694)",
            "reads": "prepared (post-filter) detections from prepare_detections at :542-543 — HERE prepared dets + calib coexist but t is missing on the live path; on the video path the caller (process :1067) has t in scope but doesn't pass it in",
            "migrate": "add a t parameter from both callers (:704 live, :1067 video); at :693, after pruning, feed shadow measure.MotionTracker.update([(x,y,radius,number)...], t) and MeasurementCore presence — the single place live AND playback converge with dets+calib+t on the sole mutating thread, zero new locks"
          },
          {
            "file": "src/billiards_trainer/vision/pipeline.py",
            "line": 338,
            "holds": "prepare_detections: the full shared filter stack (size prior, foreign veto, rigid repair, number arbitration, pocket/spot vetoes, blur recovery) — stateless w.r.t. tracks but reads _foreign_last/_nonfelt_last/_blur",
            "reads": "raw camera-space dets + calib + frame_shape (+ optional frame for blur recovery)",
            "migrate": "keep as the shared preparation stage for both the live tracker and MeasurementCore (already designed for it — the M1 engine consumes it)"
          },
          {
            "file": "src/billiards_trainer/vision/pipeline.py",
            "line": 1048,
            "holds": "process() detect=False branch: display frames republish self.tracker.tracks + _last_detections/_last_raw_dets and bump _frames_since_ingest staleness counter (:1051, read at :786) — a private 'how stale am I' opinion",
            "reads": "tracker.tracks mutated at ingest cadence; every live display frame lands here",
            "migrate": "read the frame's published state from MeasurementCore.snapshot(t) instead of raw tracker internals"
          },
          {
            "file": "src/billiards_trainer/vision/pipeline.py",
            "line": 1092,
            "holds": "last_motion: motion-energy opinion written by the DISPLAY path (absdiff of consecutive display frames), read cross-purpose by the controller's idle throttle (controller.py:1274)",
            "reads": "rectified gray ROI of consecutive display frames",
            "migrate": "motion becomes a MeasurementCore-owned measurement stamped with t; throttle and shot detector read the same value"
          },
          {
            "file": "src/billiards_trainer/vision/pipeline.py",
            "line": 1120,
            "holds": "shots.update(tracks, table, t, motion, evidence): shot state machine driven at DISPLAY cadence with display t against tracks that changed at INGEST time — the timestamp-skew site behind state disagreements",
            "reads": "tracker.tracks (ingest-cadence), display-frame t, display-frame motion",
            "migrate": "drive shot state from MeasurementCore's ingest-time (dets, t) updates so state transitions carry detection timestamps, not display timestamps"
          },
          {
            "file": "src/billiards_trainer/workers/controller.py",
            "line": 1501,
            "holds": "presence opinion published to the UI: _presence.update({tr.number...}, t) at display cadence with display t — TablePresence instance owned by controller (:112)",
            "reads": "res.tracks (republished ingest-cadence tracks) + display t",
            "migrate": "update presence inside the MeasurementCore feed at ingest time with detection t; this site becomes a pure read of core.present for the FramePacket"
          },
          {
            "file": "src/billiards_trainer/workers/controller.py",
            "line": 96,
            "holds": "_detections_ready = Signal(object, object) — the worker->controller channel shape; today (raw_dets, frame_shape) only",
            "reads": "n/a (wiring)",
            "migrate": "widen to Signal(object, object, object[, object]) to carry t (and the calib snapshot) across the thread boundary"
          },
          {
            "file": "src/billiards_trainer/measure/tracker.py",
            "line": 69,
            "holds": "MotionTracker.update(dets, t): the hardened offline tracker — needs (x, y, radius, number) tuples plus a real per-frame t (predicts with dt = t - tr.t, friction-damped)",
            "reads": "prepared detections + capture-time t (exactly what the ingest path must start carrying)",
            "migrate": "instantiate one shadow instance inside MeasurementCore; fed from _apply_detections tail (pipeline.py:693) on the controller thread"
          },
          {
            "file": "src/billiards_trainer/measure/presence.py",
            "line": 33,
            "holds": "TablePresence.update(seen, t): pure presence authority, 0.6s ABSENT_S grace keyed on t",
            "reads": "set of seen ball numbers + t",
            "migrate": "move the update call from controller.py:1501 (display t) to the MeasurementCore feed (detection t); controller reads the returned dict for FramePacket.present"
          },
          {
            "file": "src/billiards_trainer/workers/controller.py",
            "line": 135,
            "holds": "_rec_lock: the ONLY threading.Lock in the controller — guards recorder state (rec worker + controller thread). NO lock exists around pipeline/tracker state; safety is entirely the queued-connection single-thread discipline",
            "reads": "recording state only — irrelevant to pipeline; listed to answer the locking question exhaustively",
            "migrate": "no change; MeasurementCore needs no lock if fed exclusively from the controller QThread (ingest slot + _run_frame), matching today's invariant"
          }
        ]
      },
      "recording-exports": {
        "sites": [
          {
            "file": "src/billiards_trainer/workers/controller.py",
            "line": 1411,
            "holds": "Persists the per-frame track snapshot (10Hz throttle via _sidecar_last_t at 1408-1409) into the live analysis sidecar; the frame record is the substrate every later derivation (outcomes, actions, forensic, phone dossier) trusts",
            "reads": "res.tracks, res.carried_ids, res.foreign_frac from vision/pipeline.py PipelineResult (the OLD live tracker + hand-context), gated on res.status=='tracking'; t is source-uptime seconds",
            "migrate": "Yes, cleanly: core.frame_snapshot(t) returning tracks+hand-context already rebased to recording time; controller stops choosing cadence and fields — the core (MotionTracker from measure/tracker.py + measure/presence.py) publishes, the sidecar becomes a dumb serializer of core snapshots"
          },
          {
            "file": "src/billiards_trainer/workers/controller.py",
            "line": 388,
            "holds": "Persists shot-clock transitions (start/stop/pause/resume) so replay and the iOS overlay reconstruct the displayed countdown; the transitions themselves are derived from the controller's PRIVATE cue-motion opinion (_handle_state, _cue_still, _clock_armed, _saw_cue_t at ~1700-1743) — a shot-state opinion parallel to the pipeline's shot_state",
            "reads": "Scrapes self._clock._run_seconds (a private ShotClock attr via getattr) plus the controller's own t; sc.add_clock in vision/analysis_cache.py:108",
            "migrate": "Yes: core owns motion/settled state, so clock edges become core-authored events (core.shot_clock.transitions) recorded with core timestamps; kills the getattr on a private field and the duplicate 'is the table moving' opinion"
          },
          {
            "file": "src/billiards_trainer/workers/controller.py",
            "line": 1251,
            "holds": "Appends live-measured stroke_vision records to the sidecar (controller thread only — 'w'-mode handle rule); holds a rounding contract with add_shot (round(start,3) rebased) that annotate_session's idempotence key depends on",
            "reads": "rec from stroke_vision.measure_shot re-DECODING the in-progress .part file (controller.py:1231) — a second, offline measurement of a shot the live tracker already saw; enqueued at controller.py:1176 with a start privately rebased via sc._t0 (controller.py:1784-1786)",
            "migrate": "Yes: core owns shot identity (stable shot id + recording-time start); the stroke worker asks core for the shot window and returns metrics keyed by shot id — the fragile float-rounding join and the per-consumer _t0 rebase disappear"
          },
          {
            "file": "src/billiards_trainer/workers/controller.py",
            "line": 1749,
            "holds": "Writes the SQLite Shot row (db/repository.py:131 record_shot: outcome, num_pocketed, target_pocket, cue_scratch, duration_s, shot_seconds, streak_index, stroke_json) — the LIVE detector's in-the-moment attribution, which the code itself documents at 899-901 as 7/11 accurate vs 11/11 for the close-pass derivation; the DB row is never corrected afterward",
            "reads": "res.shot_event (ShotEvent from the old pipeline's shot detector), self._turn_start_t (controller's private clock opinion for shot_seconds), _consume_stroke() (IMU metrics privately buffered in self._last_stroke, controller.py:580)",
            "migrate": "Partially today, fully with core finalize: core emits the canonical ShotEvent carrying turn timing and joined stroke metrics; DB writes become provisional-with-shot-id so the close pass (or core) can UPDATE the row when derived outcomes land — right now DB truth and sidecar truth permanently diverge"
          },
          {
            "file": "src/billiards_trainer/workers/controller.py",
            "line": 1759,
            "holds": "Persists the shot record into the sidecar (add_shot); this is the line the phone, the review UI, and derive_and_correct all key off (start time = protocol key), yet its outcome field is the live opinion that corrections later override",
            "reads": "Same ShotEvent as the DB write; SidecarWriter.add_shot (analysis_cache.py:99) rebases with its private _t0",
            "migrate": "Yes: core writes its own shot records with core-owned shot ids and timebase; consumers stop matching shots by float start-time proximity"
          },
          {
            "file": "src/billiards_trainer/workers/controller.py",
            "line": 1755,
            "holds": "Appends a structured shot event to SHOTLOG_PATH debug JSONL (body at 1790-1812: ts, session, mode, outcome, pocketed track_id/cls/pocket) — a third persisted copy of the same live opinion",
            "reads": "ShotEvent + shot_seconds + self._session_id/_mode",
            "migrate": "Trivially: log core's canonical event serialization (event.to_record()) verbatim, one schema everywhere"
          },
          {
            "file": "src/billiards_trainer/workers/controller.py",
            "line": 1766,
            "holds": "A SECOND copy of the timebase-rebase logic: dataclasses.replace(event, start_t/end_t - sc._t0) before shot_recorded.emit at 1777, reaching into SidecarWriter's private _t0 so UI markers match sidecar times — the comment admits it is 'the same timebase bug the sidecar writer fixed, one layer up'",
            "reads": "sc._t0 (private), ShotEvent",
            "migrate": "Yes, and it is the strongest argument for the core: core stamps events in recording time at the source; every downstream rebase (here, add_shot, _submit_stroke, shots_export.session_time_offset) deletes"
          },
          {
            "file": "src/billiards_trainer/workers/controller.py",
            "line": 1788,
            "holds": "Emits stats_updated with repo.session_summary — session make/miss/streak derived from provisional live DB rows; identical emits at 545, 555, 668, 697 (session start/reset/manual). This is the on-screen scoreboard that disagrees with the post-close derived outcomes",
            "reads": "SQLite rows just written from the live detector opinion",
            "migrate": "Yes: a core session-stats view fed by core-final shot records; UI subscribes to core, not to a DB snapshot of a superseded opinion"
          },
          {
            "file": "src/billiards_trainer/workers/controller.py",
            "line": 552,
            "holds": "record_manual_shot writes a DB shot row (with IMU stroke_json) but writes NOTHING to the sidecar — manual shots exist in the DB/Supabase universe and are invisible to the sidecar/phone/derivation universe",
            "reads": "User-tapped outcome + _consume_stroke() from self._last_stroke",
            "migrate": "Yes: route through core.record_manual_shot which persists both the DB row and a review-ranked sidecar record so all surfaces converge"
          },
          {
            "file": "src/billiards_trainer/workers/controller.py",
            "line": 528,
            "holds": "Persists a labeled training frame (TrainingStore.add_frame, train/store.py) for ball-ID training",
            "reads": "self._last_frame — the controller's private copy of the latest raw camera frame — plus user-corrected boxes",
            "migrate": "Yes, trivially: core.latest_frame() facade; the only requirement is the same copy-before-write discipline already in place"
          },
          {
            "file": "src/billiards_trainer/workers/controller.py",
            "line": 914,
            "holds": "Launches run_close_pass on a daemon thread at recording finalize — the entire second-opinion engine (derived outcomes, action labels, stroke re-annotation, forensic fill, phone summary, proxy, library index) that supersedes everything the live path just persisted",
            "reads": "The finished video path + its sidecar (closed at 878-881)",
            "migrate": "This IS the proto-core finalize: fold into core.finalize(session) and have it also reconcile the DB rows and Supabase queue that today never learn the derived outcomes"
          },
          {
            "file": "src/billiards_trainer/workers/controller.py",
            "line": 643,
            "holds": "Constructs the live SidecarWriter with meta = {fps} ONLY — no table/H/corners despite the format doc (analysis_cache.py:14) advertising them — which is exactly why every downstream consumer re-derives geometry",
            "reads": "self._src_fps",
            "migrate": "Yes: core supplies calibration (table, H, corners) into sidecar meta at open; shots_export._video_transform's 3s pipeline warmup and measure/engine._acquire_calib both become reads of recorded truth"
          },
          {
            "file": "src/billiards_trainer/vision/analysis_cache.py",
            "line": 64,
            "holds": "SidecarWriter itself holds the timebase opinion: private _t0 set by the FIRST frame written (line 76/82), used to rebase shots (line 100) — an opinion two other sites reach into (controller 1766, 1784); add_frame:77, add_shot:99, add_clock:108, add_stroke:116 define the persisted schema",
            "reads": "Whatever the controller hands it (tracks lists, ShotEvent, clock/stroke dicts)",
            "migrate": "Becomes the core's persistence adapter: core owns t0/rebase and shot identity; writer serializes core records without private state anyone else needs to read"
          },
          {
            "file": "src/billiards_trainer/vision/shots_export.py",
            "line": 384,
            "holds": "export_shots_summary writes <video>.shots.json — the phone/desktop dossier (outcomes, actions, trails, aim, clock transitions) the iOS app reads via Dropbox; runs in the close pass for live sessions",
            "reads": "SidecarReader (persisted frames/shots/corrections/clock) plus _video_transform (line 28) which spins up a FRESH Pipeline for ~3s to re-acquire calibration because 'the live calib can disagree with the recorded geometry' — a second geometry opinion by design",
            "migrate": "Yes once core writes geometry into sidecar meta (see controller.py:643): export reads core-recorded H/corners; also session_time_offset (line 247) — a third timebase reconciliation, measuring how far sidecar clock runs ahead of video clock — deletes when core stamps recording time"
          },
          {
            "file": "src/billiards_trainer/vision/shots_export.py",
            "line": 583,
            "holds": "export_lifetime_stats (583) and export_library_index (661) write cross-session aggregate files next to the recordings — lifetime make% and the library the phone browses",
            "reads": "Every session's summary/sidecar in the recordings dir",
            "migrate": "Yes: read core-final session records; today they aggregate whatever mix of live and derived outcomes each summary happened to capture at its last export"
          },
          {
            "file": "src/billiards_trainer/vision/shot_pass.py",
            "line": 72,
            "holds": "run_close_pass — the canonical finalize sequence appending corrections/action/stroke/tag_correction records and re-exporting; forensic_fill (line 33) appends forensic-ranked miss-side verdicts. Its outputs RANK ABOVE the live opinions but flow only into sidecar+summary, never back into the DB or Supabase",
            "reads": "Video + sidecar via SidecarReader; re-decodes video for strokes/forensic/proxy",
            "migrate": "Absorb into core.finalize(); the ranking ladder (review > forensic > derived > live) belongs inside the core, applied once, with every persistence surface (DB, sidecar, summary, cloud) reading the post-ladder state"
          },
          {
            "file": "src/billiards_trainer/sync/supabase.py",
            "line": 94,
            "holds": "SyncManager.sync_now pushes unsynced sessions (94) and shots (95) rows to Supabase — a permanent cloud copy of the LIVE detector's 7/11 outcomes; close-pass corrections and phone review verdicts never reach it because they live only in the sidecar",
            "reads": "repo.unsynced_sessions()/unsynced_shots() (db/repository.py:293/305) — DB rows written at controller.py:1749/552",
            "migrate": "Yes: sync core-final records — either delay marking shots synced until core.finalize has reconciled them, or push an update when a correction lands; requires the shot-id link from the controller.py:1749 migration"
          },
          {
            "file": "src/billiards_trainer/companion/server.py",
            "line": 47,
            "holds": "read_shots — the phone API's OWN line-scan of the sidecar with its OWN correction-merge rule (abs(start diff) < 0.2, last-wins, no src ranking) — a reimplementation of SidecarReader's resolution that will diverge from it (it already ignores the review-vs-derived ranking analysis_cache._shot_for enforces); list_sessions (93) serves the session_summaries cache",
            "reads": "Raw .analysis.jsonl shot/correction/action lines; vision/session_summaries.py cache",
            "migrate": "Yes, directly: call the core's reader (SidecarReader / core.shots_for(video)) instead of private line parsing; the 'light scan' optimization can live inside the core reader once"
          },
          {
            "file": "src/billiards_trainer/vision/session_summaries.py",
            "line": 79,
            "holds": "summarize + save_cache (72) maintain a duration+shot-count cache consumed by both the sessions sidebar and the phone's /api/sessions — cached counts computed by its own sidecar scan (_sidecar_shot_count, line 35), stale-able against later corrections/splits",
            "reads": "Video files + sidecar shot lines",
            "migrate": "Yes: a core-published session index, invalidated by core when it appends to a sidecar"
          },
          {
            "file": "src/billiards_trainer/companion/corrections_watcher.py",
            "line": 29,
            "holds": "apply_correction_file ingests phone verdicts: appends review-ranked correction/action/split/reviewed/clear records to the sidecar (118-120 and inline appends at 51-56, 96-98, 106-110) and re-runs run_close_pass; the rife branch (line ~63) does yet another timebase reconciliation via session_time_offset before cutting clips",
            "reads": "Dropbox-synced JSON verdict files; sidecar via append_correction/append_action; SidecarReader for the time offset",
            "migrate": "Yes: core.apply_verdict(session, start, verdict) as the single write API (ranking enforced in one place), and core-owned recording-time stamps eliminate the offset correction for clip cuts"
          }
        ],
        "notes": "Live pipeline state originates in vision/pipeline.py PipelineResult (status, tracks, shot_event, shot_state, carried_ids, foreign_frac — lines 55-73), produced by the OLD live tracker; measure/tracker.py MotionTracker + measure/presence.py TablePresence are only wired into the offline measure/engine.py reprocessor today. Four systemic disagreement channels the audit exposes: (1) TIMEBASE — source-uptime vs recording time is reconciled independently in at least four places (SidecarWriter._t0 analysis_cache.py:76, controller's ui_event rebase controller.py:1766-1775, the stroke-enqueue rebase controller.py:1784-1786, and shots_export.session_time_offset:247 which measures the residual error of the first one); core-stamped recording time deletes all four. (2) OUTCOME TRUTH exists in three tiers — live DB row (7/11 accurate per the code's own note at controller.py:899-901), sidecar shot line, and appended correction/tag_correction lines with a src ranking — but the DB, its stats_updated scoreboard, and the Supabase cloud copy are frozen at tier 1 forever, and manual shots (controller.py:552) exist ONLY in tier 1. (3) GEOMETRY — the live sidecar meta stores only fps (controller.py:643), so shots_export._video_transform:28 and measure/engine._acquire_calib:41 each re-derive calibration with a ~3s pipeline warmup that can disagree with the live calib. (4) READER FORKS — companion/server.read_shots:47 reimplements correction merging without the src ranking that analysis_cache._shot_for:136 enforces, so the phone can show a different outcome than the desktop for the same sidecar. Non-persistence emits deliberately excluded: shot_observed (controller.py:1425), narration, and live_page/on_shot UI markers — they consume the same objects but write nothing. The natural facade seam: every site above consumes exactly five things — frame snapshot (tracks+hand context), shot events with stable identity, clock transitions, stroke metrics keyed to a shot, and calibration/timebase — which is a small, closed MeasurementCore surface."
      }
    },
    "missed": [
      {
        "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/vision/pipeline.py",
        "line": 96,
        "holds": "self.tracker = BallTracker() — the OLD live champion tracker instance; its emitted tracks are the app-wide table opinion (packet.tracks, presence feed, schematic) while measure/tracker.py MotionTracker runs nowhere live (MeasurementCore in measure/core.py has it only as shadow and is instantiated by NO ONE — grep shows zero imports)",
        "reads": "prepared Detections from the strategy/filter stack, per frame",
        "migrate": "pipeline feeds dets to MeasurementCore.ingest + observe_tracks; consumers read core.tracks; champion->shadow promotion happens inside the core per docs/MEASUREMENT_CORE.md gates"
      },
      {
        "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/vision/tracking.py",
        "line": 144,
        "holds": "class BallTracker (+ per-track _Internal at line 21): settled/still_count, vacated, committed_number/committed_cls identity hysteresis, move_streak, colour/num vote histories — a complete rival identity+motion+rest opinion vs measure/tracker.py MotionTracker",
        "reads": "Detections per frame; IdentityArbitration (vision/identity.py:33) base class",
        "migrate": "retire after promotion: MotionTracker inside MeasurementCore becomes the one tracker; identity/rest rules live once in the core"
      },
      {
        "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/events/shot_detector.py",
        "line": 94,
        "holds": "class ShotDetector: live pocketed bookkeeping (_pocketed list :125, _finalize_pockets :396, _still_on_table veto :445), per-ball last-seen set (_prev_seen_at :124), cue-presence gate (_update_cue :345), warming/cooling motion state machine, and the live outcome taxonomy (scratch/foul/make at :507-529) — a second outcome authority beside the offline identity-derived outcomes",
        "reads": "tracks + frame-diff motion energy + TableModel pockets, per frame",
        "migrate": "shot state machine and pocket bookkeeping move into MeasurementCore; it publishes shot ids + state transitions; live outcome marked provisional until the derived pass confirms"
      },
      {
        "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/vision/pipeline.py",
        "line": 561,
        "holds": "_vacant dict (+ _released set at :607) with _covered/_spot_occupied tests: per-track spot-vacancy / ghost / number-release bookkeeping — a second presence-like opinion (per-track vacancy) beside TablePresence's per-number 0.6s-grace dict; they can disagree about whether a ball is on the table",
        "reads": "tracks, detections, _foreign_last mask, _nonfelt_last felt mask",
        "migrate": "vacancy/ghost lifecycle becomes core track-lifecycle state so presence, vacancy and number-release share one authority"
      },
      {
        "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/vision/pipeline.py",
        "line": 114,
        "holds": "_play_paths trail dict + _paths_settle_t/_paths_alpha (:109-110) + _prev_shot_state (:115): its own shot-state edge detector and a settle+3s-then-1s fade — a THIRD trail/fade convention beside video_view.py's 4s fade and the shots.json trail windows",
        "reads": "tracks + shot_state per frame",
        "migrate": "trails become core.overlays_at output; one fade/window constant shared by pipeline render, video_view and the phone"
      },
      {
        "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/vision/pipeline.py",
        "line": 919,
        "holds": "_foreign_last / _nonfelt_last (:911-922): retained hand/foreign-object coverage masks — the table-coverage + hand-context opinion that gates vacancy, feeds carried_ids and the covered-table refusal",
        "reads": "frame pixels vs felt model, every 3rd tick (_arm_tick :903)",
        "migrate": "core.coverage / hand-context snapshot fields; carried_ids and vacancy read the same core state"
      },
      {
        "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/workers/controller.py",
        "line": 1656,
        "holds": "_handle_state: a SECOND shot-state edge detector (_prev_state :116) deriving settled->moving/moving->settled transitions to start/stop the clock and stamp _turn_start_t (shot_seconds at :1746) — parallel to ShotDetector's own state machine and the UI StayDownTimer's third edge detector",
        "reads": "res.shot_state string + pipeline t per frame",
        "migrate": "core emits typed state-transition events (settled/moving edges, turn start); clock and stats subscribe instead of re-deriving edges"
      },
      {
        "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/workers/controller.py",
        "line": 1680,
        "holds": "_update_cue_clock: private cue-ball rest/strike/break state machine (_cue_still, _saw_cue_t, _clock_armed, _strike_stop_t, _break_pending at :166-170) — re-derives 'cue at rest', 'all balls settled', 'that was the break (5+ movers)' and 'cue absent = pocketed/ball-in-hand' directly from tracks, beside ShotDetector's motion logic",
        "reads": "tracks (speed, cls, active) + settings.balls.stop_speed + t per frame",
        "migrate": "core publishes cue_at_rest/all_settled/cue_absent/break events; the clock becomes a pure subscriber with no track access"
      },
      {
        "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/workers/controller.py",
        "line": 561,
        "holds": "on_cue_impact: the IMU strike as a THIRD clock-stop authority (beside the table-motion edge at :1666-1670 and the cue-speed strike at :1712) — also mutates _clock_armed; three independent 'strike happened' opinions can fire in any order",
        "reads": "confirmed-impact dict from cue/detector.py LiveImpactDetector (wall-clock epoch t)",
        "migrate": "core arbitrates IMU + vision into ONE strike event per shot id; clock and stay-down consume that single event"
      },
      {
        "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/workers/controller.py",
        "line": 580,
        "holds": "_consume_stroke (+ _last_stroke :163, set at :573): joins IMU stroke metrics to the next recorded shot by WALL-CLOCK freshness (25s window, _STROKE_JOIN_SECONDS :578) — a SECOND stroke-to-shot join convention beside the UI's ±6s start-time-proximity join the map caught at live_page:1542/stay_down:155",
        "reads": "stroke metrics dict (hit_epoch) + time.time() at record_shot",
        "migrate": "core merges stroke into the ShotRecord by shot id before publishing; one join, one convention"
      },
      {
        "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/game/narration.py",
        "line": 26,
        "holds": "CarryMotion._anchors: per-ball position anchors while hand-adjacent, deriving cue_moved/object_moved-in-hand (MOVE_R displacement opinion) — a private ball-displacement judgment separate from the tracker's motion state",
        "reads": "carried track tuples (id,x,y,r,is_cue) from res.carried_ids at controller.py:1399",
        "migrate": "displacement-while-carried becomes a core-owned per-track fact; narration receives typed table_change/ball_in_hand events"
      },
      {
        "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/game/narration.py",
        "line": 60,
        "holds": "Narrator: episode state (_episode/_episode_kind/_last_said) deriving ball_in_hand vs table_change, plus note_cue_reappeared keyed off the controller's private _saw_cue_t — table-event interpretation living outside the core",
        "reads": "CarryMotion output + shot-in-flight flag + cue-absence gap from controller.py:1697-1706",
        "migrate": "core emits the table-change/ball-in-hand events with its own episode suppression; UI/voice just speak them"
      },
      {
        "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/workers/controller.py",
        "line": 1516,
        "holds": "FramePacket assembly derives clock_total = _clock._run_seconds if running else settings.shot_clock.seconds (reaching into the clock's private attr) — the producer-side origin of the ring/bar denominator disagreement the map caught only on the UI side (live_page:1434, shot_clock_widget:231)",
        "reads": "ShotClock private _run_seconds + settings.shot_clock.seconds",
        "migrate": "core.clock snapshot always carries {remaining,total,fraction,expired}; the packet stops synthesizing a denominator"
      },
      {
        "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/workers/controller.py",
        "line": 149,
        "holds": "_rec_elapsed/_rec_t0: pause-aware recording-elapsed timebase (accumulated at :612/:706/:735) — a THIRD clock beside the UI's own wall-clock rec timer (map's live_page:1688) and pipeline_t; the two elapsed clocks can drift across pauses",
        "reads": "audio_mod.elapsed_monotonic() on pause/resume/stop edges",
        "migrate": "core.session.elapsed is the single recording timebase; UI rec label and timeline live-window both read it"
      },
      {
        "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/workers/controller.py",
        "line": 308,
        "holds": "cached_shots emission rebases sidecar shot rows into VIDEO time while stashing the sidecar clock as 'key' — a dual-clock keying convention consumers must silently mirror (shot_list.py:346 emits key-or-start for corrections)",
        "reads": "SidecarReader.shots + reader.video_time_offset()",
        "migrate": "core.shot_log records carry a stable shot id plus both timebases explicitly; no positional 'key' contract between controller and widgets"
      },
      {
        "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/vision/analysis_cache.py",
        "line": 136,
        "holds": "_shot_for: three-tier start-time matching (±0.2s exact, then containment start-1..end+1, then nearest-within-8s) attaching corrections/actions/stroke records to shots — the persistence-layer version of the id-less time-proximity join; SidecarReader applies last-correction-wins on top",
        "reads": "sidecar shot dicts + record start times",
        "migrate": "records written with the shot id (session-stem@start-second convention); the reader joins by id, time matching kept only as legacy fallback"
      },
      {
        "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/companion/server.py",
        "line": 47,
        "holds": "read_shots: the PHONE re-implements shot-record reconciliation — corrections override outcomes by ±0.2s start match (:74) and actions attach the same way (:85); a second copy of the app's correction rule ('same rule as the app' admitted in the docstring) that will diverge the moment the desktop rule changes",
        "reads": "raw .analysis.jsonl line scan (shot/correction/action records)",
        "migrate": "serve the core's reconciled shot_log (or a shots.json the core exports) so phone and desktop read one already-merged record"
      },
      {
        "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/vision/session_summaries.py",
        "line": 35,
        "holds": "_sidecar_shot_count: a private 'real attempts' shot-count opinion (own shot+action scan, action matched by start rounded to 0.1s, filters ball-in-hand relocations) cached by (size,mtime) — a THIRD shot-count authority beside repo.session_summary (DB) and the cached_shots list; sidebar and phone show this count, the scoreboard shows the DB's",
        "reads": "sidecar line prefixes + video header duration",
        "migrate": "counts come from the core's exported shot_log summary; the cache keys the core's export version, not its own filter rule"
      },
      {
        "file": "c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/ui/pages/live_page.py",
        "line": 1603,
        "holds": "_on_outcome_corrected: persists the verdict then rebuilds the timeline by reaching into the sibling widget's private copy (self._shot_list._shots) — correction propagation via cross-widget shared-dict mutation, the live half of the mechanism the map only noted from shot_list's side",
        "reads": "shot_list._shots dicts + append_correction(media_path, start, outcome)",
        "migrate": "corrections go through core.append_correction; the core republishes the shot_log and both widgets rebind — no widget reads another widget's state"
      }
    ],
    "gapNotes": "Scope and method: grepped/read every src/billiards_trainer module for trackers, presence/seen/on-table sets, pocketed bookkeeping, ball/shot-count caches, schematic/overlay state, and cue-clock ball logic. The supplied map is UI-complete but omits the entire PRODUCER side: vision/pipeline.py + vision/tracking.py + events/shot_detector.py (the old live measurement chain), and workers/controller.py's derived opinions (cue-clock state machine, second shot-state edge detector, three strike authorities, stroke joins, second recording clock, clock_total synthesis). It also missed the cross-surface shot-record reconciliation family (analysis_cache._shot_for, companion read_shots, session_summaries count, the controller's dual-clock 'key' convention, live_page's cross-widget correction rebuild) and the narration derivations (CarryMotion/Narrator). Caveats: (1) the map text I received truncates mid-entry at main_window.py:457, so any entries the orchestrator had AFTER that one were invisible to me — I treated main_window:457 itself as covered; if the original map continued past it, dedupe my list against those. (2) measure/core.py MeasurementCore already exists but has ZERO consumers — nothing imports it; controller.py:112 instantiates TablePresence directly, so today the facade is aspirational. Deliberately NOT reported (judged in-engine or stateless, not private downstream opinions): vision/outcomes.py + shot_pass/forensic_repass (the sanctioned recomputable offline authority — the champion/shadow duality is the tracked migration, not a missed site); vision/overlay.py (stateless painters; its 3s+1s trail fade constant is pipeline state, reported at pipeline.py:114); pipeline.py:490-506 pocket-basket/spot-phantom detection filter (stateless per-frame rule, belongs to the ingest stack); game/shot_clock.py ShotClock (single pure-timer instance, no table reads); blur_recovery/calibration/felt (engine components); cue/worker.py BLE state and cue/detector.py internals (sensor-side, surfaced via the controller entries); ui label-editing state (user input per the map's own convention); drills (static templates); shot_thumbs/stroke_text/sessions_sidebar (stateless or non-table); controller._replay frame deque (media buffer, not table state); live stroke-vision worker records keyed by start (covered by the _shot_for/id-migration entries)."
  },
  "workflowProgress": [
    {
      "type": "workflow_phase",
      "index": 1,
      "title": "Map"
    },
    {
      "type": "workflow_phase",
      "index": 2,
      "title": "Verify"
    },
    {
      "type": "workflow_agent",
      "index": 1,
      "label": "map:ui-consumers",
      "phaseIndex": 1,
      "phaseTitle": "Map",
      "agentId": "a8d2ed511f2291e82",
      "model": "claude-fable-5",
      "state": "done",
      "startedAt": 1787881054854,
      "queuedAt": 1787881052605,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "lastToolSummary": "Architecture as-built: vision/pipeline.py Pipeline owns the…",
      "promptPreview": "You are mapping a PySide6 pool-training app at c:/Users/Joe/Documents/GitHub/BilliardsTrainer (src/billiards_trainer/). Context: the owner mandated ONE \"Measurement Core\" engine as the single source of truth for table state (ball tracks, presence, motion, shot state); today multiple subsystems keep private derived opinions and they visibly disagree. measure/tracker.py holds the hardened offline Mo…",
      "lastProgressAt": 1787881375611,
      "tokens": 146775,
      "toolCalls": 27,
      "durationMs": 320756,
      "resultPreview": "{\"notes\":\"Architecture as-built: vision/pipeline.py Pipeline owns the OLD live tracker (vision/tracking.py BallTracker, pipeline.py:96) plus its own ShotDetector (pipeline.py:97); measure/tracker.py MotionTracker is used only by the offline M1 engine (measure/engine.py:147); playback uses a THIRD source, SidecarReader tracks (pipeline.py:246). workers/controller.py then re-derives more opinions on…"
    },
    {
      "type": "workflow_agent",
      "index": 2,
      "label": "map:detector-opinions",
      "phaseIndex": 1,
      "phaseTitle": "Map",
      "agentId": "a266fbe71f3669b88",
      "model": "claude-fable-5",
      "state": "done",
      "startedAt": 1787881054507,
      "queuedAt": 1787881052605,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "lastToolSummary": "FILE: c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/bi…",
      "promptPreview": "You are mapping a PySide6 pool-training app at c:/Users/Joe/Documents/GitHub/BilliardsTrainer (src/billiards_trainer/). Context: the owner mandated ONE \"Measurement Core\" engine as the single source of truth for table state (ball tracks, presence, motion, shot state); today multiple subsystems keep private derived opinions and they visibly disagree. measure/tracker.py holds the hardened offline Mo…",
      "lastProgressAt": 1787881212525,
      "tokens": 38870,
      "toolCalls": 8,
      "durationMs": 158018,
      "resultPreview": "{\"notes\":\"FILE: c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/events/shot_detector.py (ShotDetector, 550 lines). Caller: vision/pipeline.py:1120 `self.shots.update(tracks, calib.table, t, motion, evidence)` — evidence dict built at pipeline.py:1102, carried_ids computed by pipeline._carried_ids (pipeline.py:925).\\n\\nUPDATE() ARGS CONSUMED: (1) tracks: list[Track]; (2) table:…"
    },
    {
      "type": "workflow_agent",
      "index": 3,
      "label": "map:async-timing",
      "phaseIndex": 1,
      "phaseTitle": "Map",
      "agentId": "a71f0f06b8ed896e8",
      "model": "claude-fable-5",
      "state": "done",
      "startedAt": 1787881054550,
      "queuedAt": 1787881052605,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "lastToolSummary": "LIVE ASYNC DETECTION PATH — end-to-end trace (all paths und…",
      "promptPreview": "You are mapping a PySide6 pool-training app at c:/Users/Joe/Documents/GitHub/BilliardsTrainer (src/billiards_trainer/). Context: the owner mandated ONE \"Measurement Core\" engine as the single source of truth for table state (ball tracks, presence, motion, shot state); today multiple subsystems keep private derived opinions and they visibly disagree. measure/tracker.py holds the hardened offline Mo…",
      "lastProgressAt": 1787881256663,
      "tokens": 59665,
      "toolCalls": 16,
      "durationMs": 202112,
      "resultPreview": "{\"notes\":\"LIVE ASYNC DETECTION PATH — end-to-end trace (all paths under src/billiards_trainer/).\\n\\nTHREAD MAP: (A) Controller QThread — PipelineController lives on its own QThread (make_controller_thread, workers/controller.py:1815-1819 moveToThread); QTimer parented there (on_started, :199-206) drives _tick (:1051). ALL pipeline/tracker/presence mutation happens on this thread. (B) detect-worker…"
    },
    {
      "type": "workflow_agent",
      "index": 4,
      "label": "map:recording-exports",
      "phaseIndex": 1,
      "phaseTitle": "Map",
      "agentId": "a4fd2e0db3bcf48ad",
      "model": "claude-fable-5",
      "state": "done",
      "startedAt": 1787881054567,
      "queuedAt": 1787881052605,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "lastToolSummary": "Live pipeline state originates in vision/pipeline.py Pipeli…",
      "promptPreview": "You are mapping a PySide6 pool-training app at c:/Users/Joe/Documents/GitHub/BilliardsTrainer (src/billiards_trainer/). Context: the owner mandated ONE \"Measurement Core\" engine as the single source of truth for table state (ball tracks, presence, motion, shot state); today multiple subsystems keep private derived opinions and they visibly disagree. measure/tracker.py holds the hardened offline Mo…",
      "lastProgressAt": 1787881300607,
      "tokens": 69155,
      "toolCalls": 23,
      "durationMs": 246040,
      "resultPreview": "{\"sites\":[{\"file\":\"src/billiards_trainer/workers/controller.py\",\"line\":1411,\"holds\":\"Persists the per-frame track snapshot (10Hz throttle via _sidecar_last_t at 1408-1409) into the live analysis sidecar; the frame record is the substrate every later derivation (outcomes, actions, forensic, phone dossier) trusts\",\"reads\":\"res.tracks, res.carried_ids, res.foreign_frac from vision/pipeline.py Pipelin…"
    },
    {
      "type": "workflow_agent",
      "index": 5,
      "label": "verify:gaps",
      "phaseIndex": 2,
      "phaseTitle": "Verify",
      "agentId": "ad9d58d21bb91cf0a",
      "model": "claude-fable-5",
      "state": "done",
      "startedAt": 1787881384340,
      "queuedAt": 1787881377080,
      "attempt": 1,
      "lastToolName": "StructuredOutput",
      "lastToolSummary": "Scope and method: grepped/read every src/billiards_trainer …",
      "promptPreview": "You are mapping a PySide6 pool-training app at c:/Users/Joe/Documents/GitHub/BilliardsTrainer (src/billiards_trainer/). Context: the owner mandated ONE \"Measurement Core\" engine as the single source of truth for table state (ball tracks, presence, motion, shot state); today multiple subsystems keep private derived opinions and they visibly disagree. measure/tracker.py holds the hardened offline Mo…",
      "lastProgressAt": 1787881821210,
      "tokens": 106394,
      "toolCalls": 38,
      "durationMs": 436870,
      "resultPreview": "{\"sites\":[{\"file\":\"c:/Users/Joe/Documents/GitHub/BilliardsTrainer/src/billiards_trainer/vision/pipeline.py\",\"line\":96,\"holds\":\"self.tracker = BallTracker() — the OLD live champion tracker instance; its emitted tracks are the app-wide table opinion (packet.tracks, presence feed, schematic) while measure/tracker.py MotionTracker runs nowhere live (MeasurementCore in measure/core.py has it only as sh…"
    }
  ],
  "totalTokens": 420859,
  "totalToolCalls": 112
}```
