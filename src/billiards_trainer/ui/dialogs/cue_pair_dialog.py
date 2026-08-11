"""Pick the cue sensor from a Bluetooth scan.

The auto-picker only recognises a device by its service UUID or a name hint, so
a sensor that has never been paired — and advertises an unhelpful hex name —
is invisible to it. This lists EVERYTHING the radio can see and lets one be
chosen by hand; the choice is saved as ``cue.address``, which the worker then
prefers over any guess.

Scanning runs on the worker's own BLE thread (see CueSensorWorker.request_scan)
rather than opening a second radio session from the UI.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ..theme import PALETTE

_STATUS_TEXT = {
    "connected": "Connected",
    "scanning": "Scanning…",
    "connecting": "Connecting…",
    "reconnecting": "Reconnecting…",
    "no_sensor": "No sensor found",
    "bluetooth_off": "Bluetooth unavailable",
    "unavailable": "Bluetooth support missing",
    "error": "Connection failed — retrying",
    "disabled": "Cue sensor is off",
}


class CuePairDialog(QDialog):
    def __init__(self, worker, settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Connect cue sensor")
        self.resize(560, 420)
        self._worker = worker
        self._settings = settings
        self._rows: list = []

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        blurb = QLabel(
            "Power the sensor on and press Scan. Likely matches are listed first; "
            "if the sensor has never been paired it may show only a hex name — "
            "pick the one that appears when you switch it on.")
        blurb.setWordWrap(True)
        blurb.setObjectName("Muted")
        root.addWidget(blurb)

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["", "Name", "Address", "Signal"])
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setSelectionMode(QTableWidget.SingleSelection)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        hh.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self._table.itemSelectionChanged.connect(self._sync_buttons)
        self._table.itemDoubleClicked.connect(lambda _i: self._connect())
        root.addWidget(self._table, 1)

        self._status = QLabel("")
        self._status.setObjectName("Muted")
        root.addWidget(self._status)

        row = QHBoxLayout()
        self._scan_btn = QPushButton("Scan")
        self._scan_btn.clicked.connect(self._scan)
        row.addWidget(self._scan_btn)
        self._forget_btn = QPushButton("Forget saved")
        self._forget_btn.setObjectName("Ghost")
        self._forget_btn.clicked.connect(self._forget)
        row.addWidget(self._forget_btn)
        row.addStretch(1)
        self._connect_btn = QPushButton("Connect")
        self._connect_btn.setObjectName("Accent")
        self._connect_btn.setEnabled(False)
        self._connect_btn.clicked.connect(self._connect)
        row.addWidget(self._connect_btn)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        row.addWidget(close)
        root.addLayout(row)

        worker.devices_found.connect(self._on_devices)
        worker.status_changed.connect(self._on_status)
        self._on_status(worker.state, None)
        self._scan()

    # ------------------------------------------------------------------ #
    def _scan(self) -> None:
        self._scan_btn.setEnabled(False)
        self._scan_btn.setText("Scanning…")
        self._status.setText("Looking for Bluetooth devices…")
        self._worker.request_scan()

    def _on_devices(self, rows) -> None:
        self._scan_btn.setEnabled(True)
        self._scan_btn.setText("Scan")
        self._rows = list(rows or [])
        saved = (getattr(self._settings.cue, "address", "") or "").lower()
        self._table.setRowCount(len(self._rows))
        for i, r in enumerate(self._rows):
            mark = "★" if r["is_sensor"] else ("•" if r["address"].lower() == saved else "")
            tick = QTableWidgetItem(mark)
            tick.setToolTip("Looks like the cue sensor" if r["is_sensor"]
                            else ("Previously chosen" if mark else ""))
            self._table.setItem(i, 0, tick)
            self._table.setItem(i, 1, QTableWidgetItem(r["name"] or "(unnamed)"))
            self._table.setItem(i, 2, QTableWidgetItem(r["address"]))
            rssi = r["rssi"]
            self._table.setItem(i, 3, QTableWidgetItem(
                f"{rssi} dBm" if rssi is not None else "—"))
            if r["is_sensor"]:
                for c in range(4):
                    it = self._table.item(i, c)
                    if it is not None:
                        it.setForeground(Qt.GlobalColor.green)
        if not self._rows:
            self._status.setText(
                "Nothing found. Check the sensor is powered on and in range, "
                "and that Bluetooth is enabled on this PC.")
        else:
            likely = sum(1 for r in self._rows if r["is_sensor"])
            self._status.setText(
                f"{len(self._rows)} device(s) found"
                + (f", {likely} look like the sensor." if likely
                   else " — none recognised, pick by hand."))
        self._sync_buttons()

    def _sync_buttons(self) -> None:
        self._connect_btn.setEnabled(bool(self._table.selectedItems()))
        self._forget_btn.setEnabled(bool(getattr(self._settings.cue, "address", "")))

    def _selected(self):
        rows = {i.row() for i in self._table.selectedItems()}
        return self._rows[min(rows)] if rows else None

    def _connect(self) -> None:
        r = self._selected()
        if not r:
            return
        # Persist BEFORE connecting: if the connection then drops, the app still
        # knows which device to come back to instead of guessing again.
        self._settings.cue.address = r["address"]
        self._settings.cue.enabled = True
        self._settings.save()
        self._worker.apply_settings(self._settings.cue)
        self._worker.select_device(r["address"])
        self._status.setText(f"Connecting to {r['name'] or r['address']}…")

    def _forget(self) -> None:
        self._settings.cue.address = ""
        self._settings.save()
        self._worker.select_device("")
        self._status.setText("Saved sensor cleared — scan and pick one.")
        self._sync_buttons()

    def _on_status(self, state: str, detail) -> None:
        text = _STATUS_TEXT.get(state, state)
        if state == "connected" and isinstance(detail, dict):
            batt = detail.get("battery")
            text = f"Connected to {detail.get('name') or detail.get('address')}"
            if batt is not None:
                text += f" · battery {batt}%"
            self._status.setStyleSheet(f"color: {PALETTE.success};")
        elif state in ("bluetooth_off", "unavailable"):
            self._status.setStyleSheet(f"color: {PALETTE.danger};")
            if detail:
                text = f"{text} ({detail})"
        else:
            self._status.setStyleSheet("")
        self._status.setText(text)

    def closeEvent(self, event):   # noqa: N802 - Qt override
        for sig, slot in ((self._worker.devices_found, self._on_devices),
                          (self._worker.status_changed, self._on_status)):
            try:
                sig.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
        super().closeEvent(event)
