"""PyInstaller entry script. Kept tiny; all logic lives in the package."""

import sys
from pathlib import Path

# When run from source (not frozen), make the package importable.
if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from billiards_trainer.app import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
