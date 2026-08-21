"""Cue-aim detection: the stick's angle from synthetic and real geometry.
The overlay is computed ONCE server-side (Joe: desktop and iOS must never
disagree), so this module IS the single source of truth being pinned."""

import math

import cv2
import numpy as np

from billiards_trainer.vision.cue_aim import aim_ray_end, detect_cue_aim

FELT = (200, 160, 40)          # a turquoise-ish BGR


def _table(w=700, h=1300):
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = FELT
    return img


def _stick(img, cx, cy, ang_deg, length=420, width=9,
           color=(230, 235, 240)):
    """Draw a stick whose tip stops one ball-radius short of (cx, cy),
    pointing AT the ball from behind."""
    a = math.radians(ang_deg)
    tip = (int(cx - 18 * math.cos(a)), int(cy - 18 * math.sin(a)))
    butt = (int(cx - (18 + length) * math.cos(a)),
            int(cy - (18 + length) * math.sin(a)))
    cv2.line(img, butt, tip, color, width)
    return img


class TestDetectCueAim:
    def _angle_err(self, got, want_deg):
        want = math.radians(want_deg) % (2 * math.pi)
        d = abs(got - want) % (2 * math.pi)
        return math.degrees(min(d, 2 * math.pi - d))

    def test_recovers_angle_and_sign(self):
        for want in (0, 37, 90, 141, 200, 313):
            img = _stick(_table(), 350, 650, want)
            cv2.circle(img, (350, 650), 15, (255, 255, 255), -1)
            got = detect_cue_aim(img, (350, 650), 15.0)
            assert got is not None, f"no aim at {want} deg"
            ang, q, anchor = got
            # anchor must sit on the stick, behind the ball, not on it
            import math as _m
            d_axis = abs(_m.sin(ang) * (anchor[0] - 350)
                         - _m.cos(ang) * (anchor[1] - 650))
            assert d_axis < 6.0, "anchor off the stick axis" 
            assert self._angle_err(ang, want) < 2.0, \
                f"aim off by {self._angle_err(ang, want):.1f} deg at {want}"
            assert q > 0.2

    def test_arm_blob_does_not_tilt_the_cluster(self):
        # a thick forearm at 12 degrees off the stick, passing near the
        # cue: the naive length-weighted mean tilted visibly (the @100s
        # frame); the cluster must ignore it
        img = _stick(_table(), 350, 650, 30)
        a = math.radians(42)
        cv2.line(img, (int(350 - 60 * math.cos(a)), int(650 - 60 * math.sin(a))),
                 (int(350 - 460 * math.cos(a)), int(650 - 460 * math.sin(a))),
                 (120, 140, 190), 60)
        cv2.circle(img, (350, 650), 15, (255, 255, 255), -1)
        got = detect_cue_aim(img, (350, 650), 15.0)
        assert got is not None
        err = self._angle_err(got[0], 30)
        assert err < 3.0, f"arm tilted the aim by {err:.1f} deg"

    def test_bare_table_yields_nothing(self):
        img = _table()
        cv2.circle(img, (350, 650), 15, (255, 255, 255), -1)
        assert detect_cue_aim(img, (350, 650), 15.0) is None


class TestAimRay:
    def test_ray_stops_at_bed_edge(self):
        ex, ey = aim_ray_end(350, 650, 0.0, (0, 0, 700, 1300))
        assert (round(ex), round(ey)) == (700, 650)
        ex, ey = aim_ray_end(350, 650, math.pi / 2, (0, 0, 700, 1300))
        assert (round(ex), round(ey)) == (350, 1300)
        ex, ey = aim_ray_end(350, 650, math.pi, (0, 0, 700, 1300))
        assert (round(ex), round(ey)) == (0, 650)


class TestInferTarget:
    def test_straight_hit_ghost_two_radii_back(self):
        from billiards_trainer.vision.cue_aim import infer_target
        got = infer_target((100, 100), 0.0, [(5, 400, 100)], 15.0)
        assert got is not None
        num, (gx, gy), t = got
        assert num == 5
        assert abs(gx - 370) < 1e-6 and abs(gy - 100) < 1e-6

    def test_first_ball_in_path_wins(self):
        from billiards_trainer.vision.cue_aim import infer_target
        got = infer_target((100, 100), 0.0,
                           [(9, 600, 100), (5, 400, 110)], 15.0)
        assert got[0] == 5

    def test_offset_hit_and_clean_miss(self):
        import math
        from billiards_trainer.vision.cue_aim import infer_target
        # centre passes 29px from the ball: still a (thin) hit at 2r=30
        got = infer_target((100, 100), 0.0, [(3, 400, 129)], 15.0)
        assert got is not None and got[0] == 3
        # 31px: clean miss
        assert infer_target((100, 100), 0.0, [(3, 400, 131)], 15.0) is None
        # ball BEHIND the cue is never a target
        assert infer_target((100, 100), 0.0, [(3, 50, 100)], 15.0) is None
