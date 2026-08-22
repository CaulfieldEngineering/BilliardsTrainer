"""Miss tagging: the four labels Joe asked for, with the sign
conventions pinned. Every verdict inverts if these drift, so the
geometry is exercised with synthetic shots whose truth is obvious."""

import json

from billiards_trainer.vision.analysis_cache import SidecarReader
from billiards_trainer.vision.miss_tags import label, tag_shot
from billiards_trainer.vision.tablespace import from_calibration


class _T:
    def __init__(self, x0, y0, x1, y1):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1


#: 10.667 px/in, bed 50x100in => a 9ft table in rect pixels
SPACE = from_calibration(_T(0, 0, 533, 1067), 12.0)


def _session(tmp_path, cue_pts, obj_num, obj_pts, name="s.mp4"):
    """Synthetic sidecar: cue ball follows cue_pts, one object ball
    follows obj_pts (both [(t,x,y)]), everything sampled at 10Hz."""
    vid = tmp_path / name
    vid.write_bytes(b"0")
    rows = [{"type": "meta", "v": 1, "fps": 30}]
    times = sorted({round(t, 2) for t, _x, _y in cue_pts + obj_pts})
    def at(pts, t):
        prev = pts[0]
        for p in pts:
            if p[0] <= t + 1e-9:
                prev = p
        return prev[1], prev[2]
    for t in times:
        cx, cy = at(cue_pts, t)
        ox, oy = at(obj_pts, t)
        rows.append({"type": "f", "t": t, "tracks": [
            [1, cx, cy, 12.0, 0, "cue", True],
            [2, ox, oy, 12.0, obj_num, "solid", True]]})
    rows.append({"type": "shot", "start": 1.0, "end": 3.0,
                 "outcome": "miss", "pocketed": 0})
    (tmp_path / (name + ".analysis.jsonl")).write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return vid


def _still(x, y, t0, t1, step=0.1):
    out, t = [], t0
    while t <= t1 + 1e-9:
        out.append((round(t, 2), x, y))
        t += step
    return out


def _run(x0, y0, x1, y1, t0, t1, step=0.1):
    out, t, n = [], t0, max(1, int(round((t1 - t0) / step)))
    for i in range(n + 1):
        f = i / n
        out.append((round(t0 + i * step, 2), x0 + (x1 - x0) * f,
                    y0 + (y1 - y0) * f))
    return out


class TestConventions:
    def _shoot(self, tmp_path, obj_depart, name):
        """Cue travels straight UP the table into a ball at (266,500);
        the object ball then departs toward obj_depart."""
        cue = _still(266, 800, 0.0, 0.9) + _run(266, 800, 266, 530, 1.0, 1.4)
        obj = _still(266, 500, 0.0, 1.4) + _run(266, 500, *obj_depart, 1.5, 2.2)
        return _session(tmp_path, cue, 3, obj, name)

    def test_left_cut_is_negative_and_named_left(self, tmp_path):
        # shooting up-screen, the object ball sent to screen-left is the
        # shooter's left (facing along travel) => LEFT cut, negative
        vid = self._shoot(tmp_path, (120, 380), "l.mp4")
        r = SidecarReader(vid)
        tags = tag_shot(r, r.shots[0], SPACE)
        assert tags is not None
        assert tags["cut"] == "left" and tags["cut_deg"] < 0

    def test_right_cut_is_positive_and_named_right(self, tmp_path):
        vid = self._shoot(tmp_path, (412, 380), "r.mp4")
        r = SidecarReader(vid)
        tags = tag_shot(r, r.shots[0], SPACE)
        assert tags is not None
        assert tags["cut"] == "right" and tags["cut_deg"] > 0

    def test_left_cut_missing_left_is_an_overcut(self, tmp_path):
        # target is the top-left pocket (0,0); sending the ball WIDER
        # (further left) than that line = too thin = overcut, missed left
        vid = self._shoot(tmp_path, (60, 330), "ol.mp4")
        r = SidecarReader(vid)
        tags = tag_shot(r, r.shots[0], SPACE)
        assert tags["cut"] == "left"
        assert tags["pocket"] == "top-left"
        assert tags["miss_side"] == "left"
        assert tags["fullness"] == "overcut"
        assert "Left cut, missed left — overcut" == label(tags)

    def test_left_cut_missing_right_is_an_undercut(self, tmp_path):
        # not enough cut: the ball stays right of the pocket line
        vid = self._shoot(tmp_path, (215, 330), "ul.mp4")
        r = SidecarReader(vid)
        tags = tag_shot(r, r.shots[0], SPACE)
        assert tags["cut"] == "left"
        assert tags["miss_side"] == "right"
        assert tags["fullness"] == "undercut"

    def test_right_cut_missing_right_is_an_overcut(self, tmp_path):
        vid = self._shoot(tmp_path, (473, 330), "or.mp4")
        r = SidecarReader(vid)
        tags = tag_shot(r, r.shots[0], SPACE)
        assert tags["cut"] == "right"
        assert tags["miss_side"] == "right"
        assert tags["fullness"] == "overcut"

    def test_near_straight_shot_has_no_fullness_label(self, tmp_path):
        # a REAL straight-in: cue, object and the left-middle pocket all
        # on one line (y=533). "Straight" is a property of the shot the
        # table offered, so it is judged on the REQUIRED cut, not on what
        # he did — that is why this needs a pocket in line.
        cue = _still(400, 533, 0.0, 0.9) + _run(400, 533, 232, 533, 1.0, 1.4)
        obj = _still(200, 533, 0.0, 1.4) + _run(200, 533, 60, 540, 1.5, 2.2)
        vid = _session(tmp_path, cue, 3, obj, "st.mp4")
        r = SidecarReader(vid)
        tags = tag_shot(r, r.shots[0], SPACE)
        assert tags["pocket"] == "left-middle"
        assert tags["cut"] == "straight"
        assert "fullness" not in tags
        assert label(tags).startswith("Straight-in, missed ")

    def test_abstains_when_nothing_was_struck(self, tmp_path):
        cue = _still(266, 800, 0.0, 0.9) + _run(266, 800, 266, 600, 1.0, 1.4)
        obj = _still(100, 200, 0.0, 2.2)          # never moves
        vid = _session(tmp_path, cue, 3, obj, "none.mp4")
        r = SidecarReader(vid)
        assert tag_shot(r, r.shots[0], SPACE) is None
