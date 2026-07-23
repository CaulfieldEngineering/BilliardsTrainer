"""Session recording quality: H.264 writer + HDMI letterbox crop."""

import numpy as np
import pytest

from billiards_trainer.capture.audio import find_ffmpeg
from billiards_trainer.capture.videowriter import FfmpegWriter, content_box


def _letterboxed(w=640, h=360, box=(60, 40, 579, 319)):
    img = np.zeros((h, w, 3), np.uint8)
    x0, y0, x1, y1 = box
    img[y0:y1 + 1, x0:x1 + 1] = (60, 120, 180)
    return img


def test_content_box_finds_active_image():
    box = content_box(_letterboxed())
    assert box is not None
    x0, y0, x1, y1 = box
    # padded outward a little, even-sized, and inside the frame
    assert x0 <= 60 and y0 <= 40 and x1 >= 579 and y1 >= 319
    assert (x1 - x0 + 1) % 2 == 0 and (y1 - y0 + 1) % 2 == 0


def test_content_box_keeps_full_frame_when_no_borders():
    img = np.full((360, 640, 3), 90, np.uint8)
    assert content_box(img) is None


def test_content_box_refuses_overcrop_on_dark_scene():
    img = np.zeros((360, 640, 3), np.uint8)
    img[170:190, 300:340] = 200     # one small bright blob (lights off)
    assert content_box(img) is None


@pytest.mark.skipif(find_ffmpeg() is None, reason="ffmpeg not installed")
def test_ffmpeg_writer_produces_h264(tmp_path):
    import subprocess
    path = str(tmp_path / "clip.mp4")
    wtr = FfmpegWriter(path, 20.0, (320, 240), bitrate="1M")
    frame = np.zeros((240, 320, 3), np.uint8)
    for i in range(30):
        frame[:] = (i * 5) % 255
        wtr.write(frame)
    wtr.release()
    out = subprocess.run(
        [find_ffmpeg().replace("ffmpeg", "ffprobe"), "-v", "error",
         "-show_entries", "stream=codec_name", "-of", "csv=p=0", path],
        capture_output=True, text=True, timeout=30)
    assert "h264" in out.stdout
