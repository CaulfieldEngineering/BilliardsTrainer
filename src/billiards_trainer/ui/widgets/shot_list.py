"""Per-shot review list: the session's shots as navigable rows.

Dossier slice 2 (Joe: "parse shots, per session, so I can review my
gameplay and review the quality of the tracking"). The playback rail shows
every shot as a row — number, outcome, clock position, duration, balls
potted — click one (or use Prev/Next) and playback seeks to the start of
the pre-shot routine. The same data feeds the timeline lane; this is the
list view of it.
"""

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..theme import PALETTE

_OUTCOME = {
    "make": ("#3FB950", "MAKE"),
    "miss": ("#7D8590", "MISS"),
    "scratch": ("#E3B341", "SCRATCH"),
}

_DOT_CACHE: dict[tuple[str, bool], QIcon] = {}


def _thumb_icon(bgr, colour: str, corrected: bool) -> QIcon:
    """Thumbnail icon with the outcome painted as a left-edge bar (the row
    has ONE icon slot, so the picture and the colour share it). Corrected
    verdicts keep their ring, drawn around the whole thumb."""
    import numpy as np
    from PySide6.QtGui import QImage
    h, w = bgr.shape[:2]
    rgb = np.ascontiguousarray(bgr[..., ::-1])
    img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    pm = QPixmap.fromImage(img)
    p = QPainter(pm)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(colour))
    p.drawRect(0, 0, 4, h)                       # outcome edge bar
    if corrected:
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(PALETTE.text), 2))
        p.drawRect(1, 1, w - 2, h - 2)
    p.end()
    return QIcon(pm)


def _outcome_dot(colour: str, corrected: bool) -> QIcon:
    """Small filled circle in the timeline's outcome colour; a corrected
    verdict gets a ring so Joe's own calls are distinguishable at a glance."""
    key = (colour, corrected)
    if key not in _DOT_CACHE:
        pm = QPixmap(14, 14)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(colour))
        p.drawEllipse(QRectF(3, 3, 8, 8))
        if corrected:
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor(PALETTE.text), 1.2))
            p.drawEllipse(QRectF(1, 1, 12, 12))
        p.end()
        _DOT_CACHE[key] = QIcon(pm)
    return _DOT_CACHE[key]


#: The list's views (G6 spec item 5: "sortable: misses only, longest shots,
#: streaks"). Each maps the full chronological shot list to a displayed
#: sequence of (original_shot_no, shot) — the shot keeps its real number in
#: every view, so "Shot 7" in Misses is the same Shot 7 on the timeline.
VIEWS = ("All", "Misses", "Makes", "Longest", "Streaks")


def is_attempt(s: dict) -> bool:
    """A real offensive attempt: a stroke or a break. Ball-in-hand
    relocations and empty windows stay visible in All (badged) but never
    count as shots anywhere else (Joe's false-positive push)."""
    return s.get("action", "stroke") in ("stroke", "break")


def view_shots(shots: list[dict], view: str) -> list[tuple[int, dict]]:
    seq = [(i + 1, s) for i, s in enumerate(shots or [])]
    if view == "Misses":
        return [(n, s) for n, s in seq
                if is_attempt(s) and s.get("outcome") == "miss"]
    if view == "Makes":
        return [(n, s) for n, s in seq
                if is_attempt(s) and s.get("outcome") == "make"]
    if view == "Longest":
        return sorted([(n, s) for n, s in seq if is_attempt(s)],
                      key=lambda ns: (float(ns[1].get("end", 0))
                                      - float(ns[1].get("start", 0))),
                      reverse=True)
    if view == "Streaks":
        # Shots that belong to a run of >=2 consecutive MAKES, chronological.
        # Original dicts are returned untouched so corrections made from this
        # view flow back to the one shot list everything else shares.
        out: list[tuple[int, dict]] = []
        run: list[tuple[int, dict]] = []
        for n, s in seq + [(0, {"outcome": "_end"})]:
            if is_attempt(s) and s.get("outcome") == "make":
                run.append((n, s))
                continue
            if len(run) >= 2:
                out.extend(run)
            run = []
        return out
    return seq


class ShotListPanel(QWidget):
    """``shot_selected(seconds)`` asks the owner to seek (routine start).

    The CORRECTION channel lives here too (dossier slice 3): right-click a
    shot to fix its outcome — the verdict is appended to the session's
    sidecar and survives re-opens — or to jump into Training Mode at that
    shot and fix ball labels (which feeds the training store)."""

    shot_selected = Signal(float)
    outcome_corrected = Signal(float, str)   # (shot start s, new outcome)
    fix_labels_requested = Signal(float)     # seek here + enter Training Mode
    export_requested = Signal(int, float, float)  # (shot no, start s, end s)
    dossier_requested = Signal(int)               # (shot no) -> JSON + diagram

    def __init__(self, pre_roll_s: float = 5.0, parent=None):
        super().__init__(parent)
        self.pre_roll_s = pre_roll_s
        self._shots: list[dict] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        head = QHBoxLayout()
        cap = QLabel("SHOTS")
        cap.setObjectName("StatLabel")
        head.addWidget(cap)
        self._count = QLabel("")
        self._count.setObjectName("Faint")
        head.addWidget(self._count)
        head.addStretch(1)
        from PySide6.QtWidgets import QComboBox
        self._view = "All"
        self._view_box = QComboBox()
        self._view_box.addItems(list(VIEWS))
        self._view_box.setFixedHeight(24)
        self._view_box.setToolTip("View: all shots, misses only, makes only, "
                                  "longest first, or make-streaks")
        self._view_box.currentTextChanged.connect(self.set_view)
        head.addWidget(self._view_box)
        self._prev = QPushButton("‹")
        self._next = QPushButton("›")
        for b, tip in ((self._prev, "Previous shot"), (self._next, "Next shot")):
            b.setFixedSize(24, 24)
            b.setToolTip(tip)
            b.setCursor(Qt.PointingHandCursor)
        self._prev.clicked.connect(lambda: self._step(-1))
        self._next.clicked.connect(lambda: self._step(1))
        head.addWidget(self._prev)
        head.addWidget(self._next)
        root.addLayout(head)

        self._list = QListWidget()
        self._list.setSelectionMode(QListWidget.SingleSelection)
        self._list.itemActivated.connect(self._on_item)
        self._list.itemClicked.connect(self._on_item)
        self._list.setContextMenuPolicy(Qt.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._menu)
        root.addWidget(self._list, 1)

        self._empty = QLabel("No shots detected in this session yet.")
        self._empty.setObjectName("Faint")
        self._empty.setWordWrap(True)
        root.addWidget(self._empty)
        self._sync_empty()

    # ------------------------------------------------------------------ #
    def set_shots(self, shots: list[dict]) -> None:
        self._all_shots = list(shots or [])
        self._render()

    def set_view(self, view: str) -> None:
        if view in VIEWS:
            self._view = view
            self._render()

    def _render(self) -> None:
        """Rebuild rows from the current view. ``self._shots`` holds the
        VIEWED sequence (row index -> shot dict, the same shared dicts as
        the full list, so corrections made in any view flow everywhere);
        ``self._nos`` holds each row's ORIGINAL shot number."""
        viewed = view_shots(getattr(self, "_all_shots", []),
                            getattr(self, "_view", "All"))
        self._shots = [s for _n, s in viewed]
        self._nos = [n for n, _s in viewed]
        self._list.clear()
        for no, s in viewed:
            colour, name = _OUTCOME.get(s.get("outcome", "miss"),
                                        (PALETTE.text_dim, "?"))
            action = s.get("action", "stroke")
            if action == "ball_in_hand":
                colour, name = PALETTE.text_faint, "IN HAND"
            elif action == "rearrange":
                colour, name = PALETTE.text_faint, "SHUFFLE"
            elif action == "nothing":
                colour, name = PALETTE.text_faint, "—"
            elif action == "break":
                name = "BREAK"
            corrected = bool(s.get("corrected"))
            start = float(s.get("start", 0.0))
            dur = max(0.0, float(s.get("end", start)) - start)
            pot = int(s.get("pocketed", 0))
            mm, ss = int(start) // 60, int(start) % 60
            text = f"{no:>3}   {name:<7} {mm}:{ss:02d}   {dur:.1f}s"
            if pot > 1:
                text += f"   ×{pot}"
            if s.get("reviewed_ok"):
                text += "   ✓"          # Joe confirmed this one from review
            item = QListWidgetItem(text)
            item.setIcon(_outcome_dot(colour, corrected))
            item.setForeground(QColor(PALETTE.text_faint)
                               if not is_attempt(s)
                               else Qt.GlobalColor.white)
            item.setData(Qt.UserRole, start)
            tip = (f"Shot {no}: {name.lower()}, {dur:.1f}s"
                   + (f", {pot} balls potted" if pot else "")
                   + (" — outcome corrected by you" if corrected else "")
                   + (" — confirmed correct by you" if s.get("reviewed_ok")
                      else ""))
            from .stroke_text import stroke_text
            sline = stroke_text(s)
            if sline:
                tip += "\n" + sline
            if s.get("note"):
                tip += "\nYour note: " + str(s["note"])
            item.setToolTip(tip)
            self._list.addItem(item)
        total = len(getattr(self, "_all_shots", []))
        shown = len(self._shots)
        self._count.setText(f"({shown}/{total})" if shown != total else f"({total})")
        self._sync_empty()
        if getattr(self, "_thumbs", None):
            self.set_thumbnails(self._thumbs)   # rebuilds icons for the view

    def add_shot(self, shot: dict) -> None:
        if not hasattr(self, "_all_shots"):
            self._all_shots = []
        self._all_shots.append(shot)
        self._render()

    def set_shot_stroke(self, start: float, stroke: dict) -> None:
        """Late-arriving live stroke metrics: attach to the row nearest
        ``start`` and rebuild (tooltips are built at render time)."""
        best = None
        for s in getattr(self, "_all_shots", []):
            d = abs(float(s.get("start", 0.0)) - start)
            if d <= 2.0 and (best is None
                             or d < abs(float(best.get("start", 0.0)) - start)):
                best = s
        if best is not None:
            best["stroke"] = dict(stroke)
            self._render()

    def set_thumbnails(self, thumbs: dict) -> None:
        """{shot start s: small BGR array} — swap dot icons for thumbnails.
        Missing entries keep their dot. Safe to call repeatedly (icons are
        rebuilt from the same shot list order)."""
        self._thumbs = dict(thumbs or {})
        for row, s in enumerate(self._shots):
            start = float(s.get("start", 0.0))
            bgr = self._match_thumb(start)
            if bgr is None:
                continue
            colour, _name = _OUTCOME.get(s.get("outcome", "miss"),
                                         (PALETTE.text_dim, "?"))
            item = self._list.item(row)
            if item is not None:
                item.setIcon(_thumb_icon(bgr, colour, bool(s.get("corrected"))))
        from PySide6.QtCore import QSize
        self._list.setIconSize(QSize(96, 54))

    def _match_thumb(self, start: float):
        for t, bgr in getattr(self, "_thumbs", {}).items():
            if abs(float(t) - start) < 0.25:
                return bgr
        return None

    def _sync_empty(self) -> None:
        has = bool(getattr(self, "_all_shots", None) or self._shots)
        self._list.setVisible(has)
        self._empty.setVisible(not has)
        self._prev.setEnabled(has)
        self._next.setEnabled(has)

    # ------------------------------------------------------------------ #
    def _on_item(self, item: QListWidgetItem) -> None:
        start = float(item.data(Qt.UserRole) or 0.0)
        self.shot_selected.emit(max(0.0, start - self.pre_roll_s))

    def _menu(self, pos) -> None:
        item = self._list.itemAt(pos)
        if item is None:
            return
        row = self._list.row(item)
        shot = self._shots[row]
        start = float(shot.get("start", 0.0))
        from PySide6.QtWidgets import QMenu
        m = QMenu(self)
        cur = shot.get("outcome", "miss")
        for oc in ("make", "miss", "scratch"):
            if oc == cur:
                continue
            act = m.addAction(f"Mark as {oc.upper()}")
            act.triggered.connect(
                lambda _c=False, o=oc: self._correct(row, o))
        m.addSeparator()
        exp = m.addAction("Export this shot as a clip…")
        exp.triggered.connect(
            lambda _c=False: self.export_requested.emit(
                self._nos[row] if row < len(getattr(self, "_nos", []))
                else row + 1, start, float(shot.get("end", start))))
        dos = m.addAction("Export shot dossier (data + diagram)…")
        dos.triggered.connect(
            lambda _c=False: self.dossier_requested.emit(
                self._nos[row] if row < len(getattr(self, "_nos", []))
                else row + 1))
        fix = m.addAction("Fix ball labels at this shot…")
        fix.triggered.connect(
            lambda _c=False: self.fix_labels_requested.emit(start))
        m.exec(self._list.mapToGlobal(pos))

    def _correct(self, row: int, outcome: str) -> None:
        shot = self._shots[row]          # shared dict: flows to every view
        shot["outcome"] = outcome
        shot["corrected"] = True
        self.outcome_corrected.emit(float(shot.get("key", shot.get("start", 0.0))), outcome)
        self._render()                   # NOT set_shots: keep the full list

    def _step(self, delta: int) -> None:
        if not self._shots:
            return
        row = self._list.currentRow()
        row = 0 if row < 0 else max(0, min(len(self._shots) - 1, row + delta))
        self._list.setCurrentRow(row)
        self._on_item(self._list.item(row))
