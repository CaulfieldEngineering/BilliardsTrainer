"""Dark theme: colour palette, Qt stylesheet, and font setup.

Designed in the spirit of OBS Studio / DaVinci Resolve / Anki — a dense, calm,
dark surface with a single confident accent. The accent is user-configurable
(``UiSettings.accent``) and injected into the stylesheet at apply time.
"""

from dataclasses import dataclass

from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication


@dataclass(frozen=True)
class Palette:
    # Near-black layered surfaces with a faint cool tint.
    bg: str = "#0E1116"          # window base
    bg_elevated: str = "#151A21"  # cards / panels
    surface: str = "#1C2530"      # inputs, hover targets
    surface_hi: str = "#243040"   # pressed / selected
    border: str = "#2A3340"
    border_soft: str = "#212A35"
    text: str = "#E6EDF3"
    text_dim: str = "#9AA7B4"
    text_faint: str = "#5E6B79"
    accent: str = "#3DDC97"       # mint (overridden by settings)
    accent_press: str = "#2FBF82"
    info: str = "#2D7FF9"
    success: str = "#3FB950"
    warn: str = "#E3B341"
    danger: str = "#F85149"
    felt: str = "#1F7A52"         # used by the table viz


PALETTE = Palette()

# Font stack. "Segoe UI" is present on every Win10/11 install and renders
# cleanly; we keep modern fallbacks for other platforms. (Qt's QSS font-family
# is happiest with a known-present first entry, so Segoe UI leads.) Inter/Geist
# can be embedded later as bundled .ttf via QFontDatabase.addApplicationFont().
_FONT_STACK = ['"Segoe UI"', '"Inter"', '"Roboto"', "system-ui", "sans-serif"]
_MONO_STACK = ['"Consolas"', '"Cascadia Code"', '"JetBrains Mono"', "monospace"]
_PRIMARY_FAMILY = "Segoe UI"


def primary_font(size: int = 10, weight: int = QFont.Normal) -> QFont:
    f = QFont(_PRIMARY_FAMILY, size, weight)
    f.setStyleStrategy(QFont.PreferAntialias)
    return f


def mono_font(size: int = 10) -> QFont:
    f = QFont("Consolas", size)
    f.setStyleStrategy(QFont.PreferAntialias)
    return f


def _adjust(hex_color: str, factor: float) -> str:
    c = QColor(hex_color)
    return c.lighter(int(factor * 100)).name() if factor >= 1 else c.darker(int(100 / factor)).name()


def build_stylesheet(accent: str = PALETTE.accent) -> str:
    p = PALETTE
    accent_press = QColor(accent).darker(115).name()
    accent_soft = QColor(accent)
    accent_soft.setAlpha(38)
    font_family = ", ".join(_FONT_STACK)
    return f"""
* {{
    font-family: {font_family};
    font-size: 13px;
    color: {p.text};
    outline: none;
}}
QWidget {{ background-color: {p.bg}; }}
QMainWindow, QDialog {{ background-color: {p.bg}; }}

QToolTip {{
    background-color: {p.surface_hi};
    color: {p.text};
    border: 1px solid {p.border};
    padding: 5px 8px;
    border-radius: 6px;
}}

/* ---- Cards / panels ---- */
QFrame#Card, QFrame#Panel {{
    background-color: {p.bg_elevated};
    border: 1px solid {p.border_soft};
    border-radius: 12px;
}}
QFrame#Sidebar {{
    background-color: {p.bg_elevated};
    border: none;
    border-right: 1px solid {p.border_soft};
}}
QFrame#TopBar {{
    background-color: {p.bg_elevated};
    border: none;
    border-bottom: 1px solid {p.border_soft};
}}
QLabel#H1 {{ font-size: 22px; font-weight: 700; color: {p.text}; }}
QLabel#H2 {{ font-size: 16px; font-weight: 600; color: {p.text}; }}
QLabel#Muted {{ color: {p.text_dim}; }}
QLabel#Faint {{ color: {p.text_faint}; font-size: 12px; }}
QLabel#StatValue {{ font-size: 30px; font-weight: 700; color: {p.text}; }}
QLabel#StatValueAccent {{ font-size: 30px; font-weight: 700; color: {accent}; }}
QLabel#StatLabel {{ color: {p.text_dim}; font-size: 12px; }}

/* ---- Buttons ---- */
QPushButton {{
    background-color: {p.surface};
    color: {p.text};
    border: 1px solid {p.border};
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: 600;
}}
QPushButton:hover {{ background-color: {p.surface_hi}; border-color: {accent}; }}
QPushButton:pressed {{ background-color: {p.surface}; }}
QPushButton:disabled {{ color: {p.text_faint}; border-color: {p.border_soft}; }}

QPushButton#Accent {{
    background-color: {accent};
    color: #0A0E12;
    border: none;
}}
QPushButton#Accent:hover {{ background-color: {_adjust(accent, 1.08)}; }}
QPushButton#Accent:pressed {{ background-color: {accent_press}; }}
QPushButton#Accent:disabled {{ background-color: {p.surface}; color: {p.text_faint}; }}

QPushButton#Danger {{ border-color: {p.danger}; color: {p.danger}; }}
QPushButton#Danger:hover {{ background-color: {p.danger}; color: #0A0E12; }}

QPushButton#Ghost {{ background-color: transparent; border: none; color: {p.text_dim}; }}
QPushButton#Ghost:hover {{ color: {p.text}; background-color: {p.surface}; }}

/* ---- Nav buttons (left rail) ---- */
QPushButton#NavItem {{
    background-color: transparent;
    border: none;
    border-radius: 8px;
    padding: 10px 14px;
    text-align: left;
    color: {p.text_dim};
    font-weight: 600;
}}
QPushButton#NavItem:hover {{ background-color: {p.surface}; color: {p.text}; }}
QPushButton#NavItem:checked {{
    background-color: {accent_soft.name(QColor.HexArgb)};
    color: {accent};
}}

/* ---- Inputs ---- */
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit {{
    background-color: {p.surface};
    border: 1px solid {p.border};
    border-radius: 8px;
    padding: 7px 10px;
    selection-background-color: {accent};
    selection-color: #0A0E12;
}}
QComboBox:hover, QLineEdit:hover, QSpinBox:hover {{ border-color: {accent}; }}
QComboBox:focus, QLineEdit:focus, QSpinBox:focus {{ border-color: {accent}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox::down-arrow {{ image: none; border-left: 5px solid transparent;
    border-right: 5px solid transparent; border-top: 6px solid {p.text_dim};
    margin-right: 8px; }}
QComboBox QAbstractItemView {{
    background-color: {p.surface};
    border: 1px solid {p.border};
    border-radius: 8px;
    selection-background-color: {accent};
    selection-color: #0A0E12;
    padding: 4px;
}}

/* ---- Checkboxes / radios ---- */
QCheckBox, QRadioButton {{ spacing: 8px; color: {p.text}; }}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 18px; height: 18px;
    border: 1px solid {p.border}; border-radius: 5px; background: {p.surface};
}}
QRadioButton::indicator {{ border-radius: 9px; }}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background: {accent}; border-color: {accent};
}}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{ border-color: {accent}; }}

/* ---- Sliders ---- */
QSlider::groove:horizontal {{ height: 4px; background: {p.border}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {accent}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: {p.text}; width: 16px; height: 16px;
    margin: -7px 0; border-radius: 8px;
}}
QSlider::handle:horizontal:hover {{ background: {accent}; }}

/* ---- Group boxes ---- */
QGroupBox {{
    border: 1px solid {p.border_soft};
    border-radius: 10px;
    margin-top: 14px;
    padding-top: 10px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 12px; padding: 0 6px;
    color: {p.text_dim};
}}

/* ---- Tabs ---- */
QTabWidget::pane {{ border: 1px solid {p.border_soft}; border-radius: 10px; top: -1px; }}
QTabBar::tab {{
    background: transparent; color: {p.text_dim};
    padding: 8px 16px; margin-right: 4px;
    border-bottom: 2px solid transparent; font-weight: 600;
}}
QTabBar::tab:selected {{ color: {p.text}; border-bottom: 2px solid {accent}; }}
QTabBar::tab:hover {{ color: {p.text}; }}

/* ---- Tables ---- */
QTableView, QTableWidget, QTreeView, QListView {{
    background-color: {p.bg_elevated};
    border: 1px solid {p.border_soft};
    border-radius: 10px;
    gridline-color: {p.border_soft};
    selection-background-color: {accent_soft.name(QColor.HexArgb)};
    selection-color: {p.text};
    alternate-background-color: {p.surface};
}}
QHeaderView::section {{
    background-color: {p.bg_elevated};
    color: {p.text_dim};
    border: none; border-bottom: 1px solid {p.border};
    padding: 8px 10px; font-weight: 600;
}}
QTableView::item {{ padding: 4px 8px; }}

/* ---- Scrollbars ---- */
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {p.border}; border-radius: 5px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: {p.text_faint}; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {p.border}; border-radius: 5px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

/* ---- Misc ---- */
QStatusBar {{ background: {p.bg_elevated}; color: {p.text_dim};
    border-top: 1px solid {p.border_soft}; }}
QMenu {{ background: {p.surface}; border: 1px solid {p.border}; border-radius: 8px; padding: 6px; }}
QMenu::item {{ padding: 7px 24px; border-radius: 6px; }}
QMenu::item:selected {{ background: {accent}; color: #0A0E12; }}
QProgressBar {{ background: {p.surface}; border: none; border-radius: 6px;
    text-align: center; height: 8px; }}
QProgressBar::chunk {{ background: {accent}; border-radius: 6px; }}
QSplitter::handle {{ background: {p.border_soft}; }}
QSplitter::handle:hover {{ background: {accent}; }}
"""


def apply_theme(app: QApplication, accent: str = PALETTE.accent) -> None:
    """Apply the dark palette + stylesheet to the whole application."""
    app.setStyle("Fusion")
    pal = QPalette()
    p = PALETTE
    pal.setColor(QPalette.Window, QColor(p.bg))
    pal.setColor(QPalette.WindowText, QColor(p.text))
    pal.setColor(QPalette.Base, QColor(p.surface))
    pal.setColor(QPalette.AlternateBase, QColor(p.bg_elevated))
    pal.setColor(QPalette.Text, QColor(p.text))
    pal.setColor(QPalette.Button, QColor(p.surface))
    pal.setColor(QPalette.ButtonText, QColor(p.text))
    pal.setColor(QPalette.Highlight, QColor(accent))
    pal.setColor(QPalette.HighlightedText, QColor("#0A0E12"))
    pal.setColor(QPalette.ToolTipBase, QColor(p.surface_hi))
    pal.setColor(QPalette.ToolTipText, QColor(p.text))
    pal.setColor(QPalette.PlaceholderText, QColor(p.text_faint))
    pal.setColor(QPalette.Disabled, QPalette.Text, QColor(p.text_faint))
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(p.text_faint))
    app.setPalette(pal)
    app.setFont(primary_font(10))
    app.setStyleSheet(build_stylesheet(accent))


def load_bundled_fonts() -> None:
    """Hook to register bundled .ttf fonts (Inter/Geist) if present. No-op if absent."""
    from ..config import resource_path

    fonts_dir = resource_path("assets", "fonts")
    if not fonts_dir.exists():
        return
    for ttf in fonts_dir.glob("*.ttf"):
        QFontDatabase.addApplicationFont(str(ttf))
