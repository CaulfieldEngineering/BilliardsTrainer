"""Generate the app icon (an 8-ball) as packaging/app.ico for the exe + window.

Run once; the .ico is committed. Regenerate with:
    python packaging/make_icon.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QRectF, Qt  # noqa: E402
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

OUT = Path(__file__).resolve().parent / "app.ico"


def render(size: int) -> QImage:
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing, True)
    m = size * 0.06
    rect = QRectF(m, m, size - 2 * m, size - 2 * m)
    # black 8-ball body with a subtle lighter rim
    p.setPen(Qt.NoPen)
    p.setBrush(QBrush(QColor("#222A33")))
    p.drawEllipse(rect)
    p.setBrush(QBrush(QColor("#0C0F13")))
    p.drawEllipse(rect.adjusted(size * 0.04, size * 0.04, -size * 0.04, -size * 0.04))
    # top-left specular highlight
    p.setBrush(QColor(255, 255, 255, 28))
    p.drawEllipse(QRectF(size * 0.26, size * 0.20, size * 0.22, size * 0.14))
    # white spot
    r2 = size * 0.34
    cx, cy = size * 0.5, size * 0.5
    p.setBrush(QBrush(QColor("#F2F4F7")))
    p.drawEllipse(QRectF(cx - r2 / 2, cy - r2 / 2, r2, r2))
    # the "8" drawn font-free as two stacked rings
    p.setBrush(Qt.NoBrush)
    p.setPen(QPen(QColor("#0C0F13"), max(1.5, size * 0.022)))
    ring = size * 0.085
    p.drawEllipse(QRectF(cx - ring / 2, cy - ring * 0.92, ring, ring))
    p.drawEllipse(QRectF(cx - ring / 2, cy + ring * 0.92 - ring, ring, ring))
    p.end()
    return pix.toImage()


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)  # noqa: F841
    sizes = [16, 24, 32, 48, 64, 128, 256]
    base = render(256)
    # QImage can write multi-image .ico when given the largest; Qt downscales.
    ok = base.save(str(OUT), "ICO")
    if not ok:
        # fallback: PNG (window icon still works; exe icon optional)
        png = OUT.with_suffix(".png")
        base.save(str(png), "PNG")
        print(f"ICO write unsupported; wrote {png} instead")
        return 1
    print(f"wrote {OUT} ({', '.join(map(str, sizes))} px)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
