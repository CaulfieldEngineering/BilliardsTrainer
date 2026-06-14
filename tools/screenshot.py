"""Render the UI offscreen and save PNG screenshots of each page.

Useful for docs and for eyeballing the design without a display:

    python tools/screenshot.py [out_dir]
"""

import os
import sys
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from billiards_trainer.config import Settings  # noqa: E402
from billiards_trainer.ui.main_window import MainWindow  # noqa: E402
from billiards_trainer.ui.theme import apply_theme  # noqa: E402


def pump(app, ms):
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def grab(win, path):
    win.grab().save(str(path))
    print("wrote", path)


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs" / "screenshots"
    out.mkdir(parents=True, exist_ok=True)

    app = QApplication.instance() or QApplication(sys.argv)
    settings = Settings()
    settings.updates.auto_check = False
    apply_theme(app, settings.ui.accent)
    win = MainWindow(settings)
    win.resize(1440, 900)
    win.show()
    pump(app, 300)

    # Live page with the demo running so the views have content
    win._live.start_requested.emit("demo", "free_play", "")
    pump(app, 4000)
    grab(win, out / "live.png")

    # Drills / Stats / Settings
    for idx, name in [(1, "drills"), (2, "stats"), (3, "settings")]:
        win._go(idx)
        win._nav_items[idx].setChecked(True)
        pump(app, 500)
        grab(win, out / f"{name}.png")

    win._go(0)
    win.close()
    app.quit()


if __name__ == "__main__":
    main()
