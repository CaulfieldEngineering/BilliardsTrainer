"""Inline line icons (Feather-style, MIT) rendered to recolourable QIcons.

Keeping icons as inline SVG paths means zero binary assets to ship and trivial
recolouring to match the theme accent. ``icon(name, color)`` returns a QIcon.
"""

from functools import lru_cache

from PySide6.QtCore import QByteArray, QSize, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

from .theme import PALETTE

# Each entry is the inner SVG markup for a 24x24 viewBox, stroke-based.
_PATHS: dict[str, str] = {
    "play": '<polygon points="5 3 19 12 5 21 5 3"/>',
    "pause": '<rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>',
    "stop": '<rect x="5" y="5" width="14" height="14" rx="2"/>',
    "camera": '<path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/>',
    "settings": '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "stats": '<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "layers": '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>',
    "grid": '<rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/>',
    "refresh": '<polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>',
    "check": '<polyline points="20 6 9 17 4 12"/>',
    "x": '<line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>',
    "download": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>',
    "activity": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    "home": '<path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/>',
    "calendar": '<rect x="3" y="4" width="18" height="18" rx="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/>',
    "crosshair": '<circle cx="12" cy="12" r="10"/><line x1="22" y1="12" x2="18" y2="12"/><line x1="6" y1="12" x2="2" y2="12"/><line x1="12" y1="6" x2="12" y2="2"/><line x1="12" y1="22" x2="12" y2="18"/>',
    "zap": '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
    "trending-up": '<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/>',
    "award": '<circle cx="12" cy="8" r="7"/><polyline points="8.21 13.89 7 23 12 20 17 23 15.79 13.88"/>',
    "sliders": '<line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/>',
    "user": '<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
    "rotate": '<polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/>',
    "flip-h": '<line x1="12" y1="3" x2="12" y2="21"/><path d="M9 6l-5 6 5 6z"/><path d="M15 6l5 6-5 6z"/>',
    "flip-v": '<line x1="3" y1="12" x2="21" y2="12"/><path d="M6 9l6-5 6 5z"/><path d="M6 15l6 5 6-5z"/>',
    # Recording transport (broadcast-style, SOLID media glyphs)
    "rec": '<circle cx="12" cy="12" r="8" fill="{c}" stroke="none"/>',
    "rec-stop": '<rect x="5.5" y="5.5" width="13" height="13" rx="2.5" fill="{c}" stroke="none"/>',
    "rec-pause": ('<rect x="6" y="5" width="4.4" height="14" rx="1.4" fill="{c}" stroke="none"/>'
                  '<rect x="13.6" y="5" width="4.4" height="14" rx="1.4" fill="{c}" stroke="none"/>'),
    "play-solid": '<polygon points="7,4.5 20,12 7,19.5" fill="{c}" stroke="none"/>',
    "step-back": ('<rect x="5" y="5" width="3" height="14" rx="1" fill="{c}" stroke="none"/>'
                  '<polygon points="19,5 9.5,12 19,19" fill="{c}" stroke="none"/>'),
    "step-fwd": ('<polygon points="5,5 14.5,12 5,19" fill="{c}" stroke="none"/>'
                 '<rect x="16" y="5" width="3" height="14" rx="1" fill="{c}" stroke="none"/>'),
    "film": '<rect x="2" y="3" width="20" height="18" rx="2"/><line x1="7" y1="3" x2="7" y2="21"/><line x1="17" y1="3" x2="17" y2="21"/><line x1="2" y1="9" x2="22" y2="9"/><line x1="2" y1="15" x2="22" y2="15"/>',
}


def _svg(name: str, color: str, stroke: float = 2.0) -> str:
    inner = _PATHS.get(name, _PATHS["target"])
    # {c} placeholders let a glyph opt into SOLID fills (media-transport icons);
    # everything else stays stroke-outlined via the svg-level defaults.
    inner = inner.replace("{c}", color)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" '
        f'stroke-linecap="round" stroke-linejoin="round">{inner}</svg>'
    )


@lru_cache(maxsize=256)
def icon(name: str, color: str = PALETTE.text_dim, size: int = 22, stroke: float = 2.0) -> QIcon:
    """Return a QIcon for ``name`` rendered in ``color`` at ``size`` px."""
    renderer = QSvgRenderer(QByteArray(_svg(name, color, stroke).encode("utf-8")))
    pix = QPixmap(QSize(size, size))
    pix.fill(Qt.transparent)
    painter = QPainter(pix)
    renderer.render(painter)
    painter.end()
    return QIcon(pix)


def pixmap(name: str, color: str = PALETTE.text, size: int = 22, stroke: float = 2.0) -> QPixmap:
    renderer = QSvgRenderer(QByteArray(_svg(name, color, stroke).encode("utf-8")))
    pix = QPixmap(QSize(size, size))
    pix.fill(Qt.transparent)
    painter = QPainter(pix)
    renderer.render(painter)
    painter.end()
    return pix
