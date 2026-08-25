"""Live stroke metrics: the sidecar append contract + UI text helper.

The live worker measures a shot from the growing .part and its record
must (1) survive the 'w'-mode writer's buffered flushes when appended
through the writer itself, (2) attach to the right shot on read, and
(3) carry the exact idempotence key annotate_session skips at close.
"""

from billiards_trainer.core.types import BallClass, Track
from billiards_trainer.events.shot_detector import ShotEvent, ShotOutcome
from billiards_trainer.ui.widgets.stroke_text import stroke_text
from billiards_trainer.vision.analysis_cache import SidecarReader, SidecarWriter
from billiards_trainer.vision.stroke_vision import STROKE_VISION_VERSION


def _track(tid, x, y, num=3):
    return Track(id=tid, x=x, y=y, radius=15.0, number=num,
                 cls=BallClass.SOLID, active=True)


REC = {"type": "stroke_vision", "v": STROKE_VISION_VERSION, "start": 0.2,
       "stay_down_s": 1.4, "popped_early": False, "pause_ms": 167,
       "back_depth_px": 116.2, "practice_strokes": 3, "confidence": "high"}


class TestLiveStrokeSidecar:
    def test_add_stroke_survives_writer_flushes(self, tmp_path):
        """A record appended mid-session through the WRITER must still be
        there after later frames force the writer's own buffered flush —
        the failure mode that forbids external 'a'-mode appends."""
        video = tmp_path / "session-live.mp4"
        w = SidecarWriter(video, {"fps": 30.0})
        w.add_frame(0.0, [_track(1, 100.0, 200.0)])
        w.add_shot(ShotEvent(outcome=ShotOutcome.MAKE, num_pocketed=1,
                             start_t=0.2, end_t=0.9))
        w.add_stroke(dict(REC))
        for i in range(120):          # >2 of the writer's 50-record flushes
            w.add_frame(1.0 + i * 0.1, [_track(1, 100.0 + i, 200.0)])
        w.close()
        r = SidecarReader(video)
        sv = r.shots[0].get("_stroke")
        assert sv is not None, "stroke record lost to a buffered flush"
        assert sv["stay_down_s"] == 1.4

    def test_reader_attaches_to_nearest_shot(self, tmp_path):
        video = tmp_path / "session-two.mp4"
        w = SidecarWriter(video, {"fps": 30.0})
        w.add_frame(0.0, [_track(1, 100.0, 200.0)])
        w.add_shot(ShotEvent(outcome=ShotOutcome.MAKE, num_pocketed=1,
                             start_t=0.2, end_t=0.9))
        w.add_shot(ShotEvent(outcome=ShotOutcome.MISS, num_pocketed=0,
                             start_t=30.0, end_t=31.0))
        w.add_stroke({**REC, "start": 30.0, "stay_down_s": 0.6})
        w.close()
        r = SidecarReader(video)
        assert r.shots[0].get("_stroke") is None
        assert r.shots[1]["_stroke"]["stay_down_s"] == 0.6

    def test_close_pass_skips_live_measured(self, tmp_path):
        """annotate_session's idempotence: a live record with the rebased
        start + current version must be recognised and skipped."""
        import json
        video = tmp_path / "session-skip.mp4"
        w = SidecarWriter(video, {"fps": 30.0})
        w.add_frame(0.0, [_track(1, 100.0, 200.0)])
        w.add_shot(ShotEvent(outcome=ShotOutcome.MAKE, num_pocketed=1,
                             start_t=0.2, end_t=0.9))
        w.add_stroke(dict(REC))
        w.close()
        # replicate annotate_session's have-map logic against this file
        have = {}
        for line in (tmp_path / "session-skip.mp4.analysis.jsonl") \
                .read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            if d.get("type") == "stroke_vision":
                have[round(float(d["start"]), 2)] = int(d.get("v", 0))
        r = SidecarReader(video)
        start = round(float(r.shots[0]["start"]), 2)
        assert have.get(start, 0) >= STROKE_VISION_VERSION


class TestStrokeText:
    def test_full_line(self):
        line = stroke_text({"stroke": dict(REC)})
        assert "stay-down 1.4s" in line
        assert "pause 167ms" in line
        assert "back 116px" in line
        assert "3 practice" in line
        assert "POPPED" not in line

    def test_popped_early_flagged(self):
        line = stroke_text({"_stroke": {**REC, "popped_early": True}})
        assert "POPPED UP EARLY" in line

    def test_none_when_absent_or_unmeasurable(self):
        assert stroke_text({}) is None
        assert stroke_text({"stroke": {"confidence": "none"}}) is None


class TestOneClockOrigin:
    def test_restart_drops_stale_frame(self):
        """The one-clock skew's origin: _restart must clear the old
        process's last frame so the sidecar can't anchor t0 on a stale
        picture during the device-reopen window."""
        import threading

        import numpy as np

        from billiards_trainer.capture.ffmpeg_source import FfmpegCameraSource
        s = FfmpegCameraSource.__new__(FfmpegCameraSource)
        s._lock = threading.Lock()
        s._latest = np.zeros((4, 4, 3), np.uint8)
        s._teardown = lambda: None
        s._start = lambda: None
        s._restart()
        assert s.read() is None, "stale frame served through a restart"


class TestVideoTimeOffset:
    def test_median_of_high_confidence_strikes(self, tmp_path):
        video = tmp_path / "session-off.mp4"
        w = SidecarWriter(video, {"fps": 30.0})
        w.add_frame(0.0, [_track(1, 100.0, 200.0)])
        for start, strike, conf in [
                (10.0, 7.6, "high"), (20.0, 17.5, "high"),
                (30.0, 27.4, "high"), (40.0, 20.0, "low")]:  # low ignored
            w.add_shot(ShotEvent(outcome=ShotOutcome.MAKE, num_pocketed=0,
                                 start_t=start, end_t=start + 1.0))
            w.add_stroke({"type": "stroke_vision", "v": 1, "start": start,
                          "strike": strike, "confidence": conf})
        w.close()
        r = SidecarReader(video)
        assert abs(r.video_time_offset() - 2.5) < 0.11   # median of 2.4/2.5/2.6

    def test_zero_without_enough_records(self, tmp_path):
        video = tmp_path / "session-none.mp4"
        w = SidecarWriter(video, {"fps": 30.0})
        w.add_frame(0.0, [_track(1, 100.0, 200.0)])
        w.add_shot(ShotEvent(outcome=ShotOutcome.MAKE, num_pocketed=0,
                             start_t=5.0, end_t=6.0))
        w.close()
        assert SidecarReader(video).video_time_offset() == 0.0


class TestStripeRepairPromotionOnly:
    """The stripe->solid demotion flipped true 9s to 1s on corrected-
    exposure footage (raw c7: 15/15 gameplay 9s right; ensemble with
    demotion: 9 of them returned as 1). The repair may only PROMOTE."""

    def test_no_demotion_path_remains(self):
        import inspect

        from billiards_trainer.detector_strategies.ensemble import FindIdEnsemble
        src = inspect.getsource(FindIdEnsemble._fix_stripe_bit)
        assert "n - 8" not in src, "stripe->solid demotion re-grew"

    def test_promotes_clear_stripe(self):
        import numpy as np

        from billiards_trainer.core.types import BallClass
        from billiards_trainer.detector_strategies.ensemble import FindIdEnsemble

        class F:
            x, y, radius = 30.0, 30.0, 14.0
            number, cls, bgr = 1, BallClass.SOLID, (0, 0, 0)
        # synthetic 9: white band across a yellow ball
        img = np.zeros((60, 60, 3), np.uint8)
        img[:] = (200, 120, 30)                     # felt-ish
        import cv2
        cv2.circle(img, (30, 30), 14, (40, 200, 235), -1)   # yellow ball
        img[22:38, 16:45] = (250, 250, 250)                 # white band
        FindIdEnsemble._fix_stripe_bit(img, F)
        assert F.number == 9 and F.cls == BallClass.STRIPE

    def test_never_demotes_a_stripe_claim(self):
        import numpy as np

        from billiards_trainer.core.types import BallClass
        from billiards_trainer.detector_strategies.ensemble import FindIdEnsemble

        class F:
            x, y, radius = 30.0, 30.0, 14.0
            number, cls, bgr = 9, BallClass.STRIPE, (0, 0, 0)
        img = np.zeros((60, 60, 3), np.uint8)       # all-dark crop: zero white
        FindIdEnsemble._fix_stripe_bit(img, F)
        assert F.number == 9, "a model stripe answer must never be demoted"
