"""Camera enumeration + index-reshuffle-surviving resolution."""

from billiards_trainer.capture import devices
from billiards_trainer.capture.devices import CameraInfo, resolve_camera


def _patch(monkeypatch, cams):
    monkeypatch.setattr(devices, "list_cameras", lambda: cams)


def test_label_includes_index():
    assert CameraInfo(1, "Logitech BRIO").label() == "Logitech BRIO (1)"


def test_resolve_by_name_survives_reshuffle(monkeypatch):
    # the BRIO used to be index 0, now it's index 1 (built-in took 0)
    _patch(monkeypatch, [CameraInfo(0, "Integrated"), CameraInfo(1, "Logitech BRIO")])
    idx, name, warn = resolve_camera(0, "Logitech BRIO")
    assert idx == 1 and name == "Logitech BRIO"
    assert "now camera 1" in warn


def test_resolve_by_index_when_no_name(monkeypatch):
    _patch(monkeypatch, [CameraInfo(0, "Integrated"), CameraInfo(1, "BRIO")])
    idx, name, warn = resolve_camera(1, "")
    assert idx == 1 and name == "BRIO" and warn == ""


def test_resolve_name_gone_uses_index_with_warning(monkeypatch):
    # saved name no longer present, but the saved index is still a valid camera
    _patch(monkeypatch, [CameraInfo(0, "Integrated"), CameraInfo(1, "NewCam")])
    idx, name, warn = resolve_camera(1, "OldCamThatLeft")
    assert idx == 1 and name == "NewCam"
    assert "not found" in warn and "camera 1" in warn


def test_resolve_invalid_index_falls_back_to_first(monkeypatch):
    _patch(monkeypatch, [CameraInfo(0, "Integrated")])
    idx, name, warn = resolve_camera(5, "Gone Cam")
    assert idx == 0 and name == "Integrated"
    assert "not found" in warn


def test_resolve_no_cameras(monkeypatch):
    _patch(monkeypatch, [])
    idx, name, warn = resolve_camera(0, "")
    assert idx == 0
    assert "No cameras" in warn
