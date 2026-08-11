"""The Bluetooth scan/connect picker.

Covers the path the auto-connector cannot: a sensor that has never been paired,
advertises an unrecognised name, and must be chosen by hand.
"""

import pytest

pytest.importorskip("PySide6")

from billiards_trainer.config import Settings  # noqa: E402
from billiards_trainer.cue.worker import CueSensorWorker  # noqa: E402


class _Adv:
    def __init__(self, name, uuids, rssi):
        self.local_name = name
        self.service_uuids = uuids
        self.rssi = rssi


class _Dev:
    def __init__(self, address, name):
        self.address = address
        self.name = name


def test_describe_lists_everything_sensor_first():
    """An unrecognised device must still be OFFERED — that is the whole point."""
    from billiards_trainer.cue import protocol

    found = {
        "a": (_Dev("AA:AA", "ihoment_H7151"), _Adv("ihoment_H7151", [], -56)),
        "b": (_Dev("BB:BB", "A8017180B8CF"), _Adv("A8017180B8CF", [], -40)),
        "c": (_Dev("CC:CC", "JO-BEC12"), _Adv("JO-BEC12", [protocol.SERVICE_UUID], -70)),
    }
    rows = CueSensorWorker._describe(found)
    assert len(rows) == 3, "every device must be listed, not just known ones"
    assert rows[0]["address"] == "CC:CC" and rows[0]["is_sensor"], "sensor ranks first"
    # the rest are strongest-signal first
    assert [r["address"] for r in rows[1:]] == ["BB:BB", "AA:AA"]


def test_describe_survives_missing_fields():
    """Adverts routinely lack a name, uuids or rssi; none may raise."""
    found = {"x": (_Dev("DD:DD", None), _Adv(None, None, None))}
    rows = CueSensorWorker._describe(found)
    assert rows[0]["name"] == "" and rows[0]["rssi"] is None
    assert rows[0]["is_sensor"] is False


def test_select_device_pins_the_address():
    w = CueSensorWorker()
    try:
        w.select_device("11:22:33:44:55:66")
        assert w._saved_address == "11:22:33:44:55:66"
        w.select_device("")          # "forget"
        assert w._saved_address == ""
    finally:
        w.shutdown()


def test_scan_request_is_safe_before_any_settings_applied():
    """Scan must work while the feature is still DISABLED — that is how a sensor
    gets chosen in the first place."""
    w = CueSensorWorker()
    try:
        w.request_scan()             # must not raise, must not need enabling
        assert w._thread is not None
    finally:
        w.shutdown()


def test_pair_dialog_builds_and_lists(qtbot=None):
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    w = CueSensorWorker()
    s = Settings()
    try:
        from billiards_trainer.ui.dialogs.cue_pair_dialog import CuePairDialog
        dlg = CuePairDialog(w, s)
        dlg._on_devices([
            {"address": "CC:CC", "name": "JO-BEC12", "rssi": -70,
             "is_sensor": True, "uuids": []},
            {"address": "BB:BB", "name": "", "rssi": -40,
             "is_sensor": False, "uuids": []},
        ])
        assert dlg._table.rowCount() == 2
        assert dlg._table.item(1, 1).text() == "(unnamed)"
        dlg._table.selectRow(0)
        assert dlg._connect_btn.isEnabled()
        dlg.close()
    finally:
        w.shutdown()
        app.processEvents()
