#!/usr/bin/env python3
"""Build every song from its ``spec.yaml``.

Thin shim over ``./sb build`` so that ``python3 build.py`` does the obvious
thing. All the logic lives in :mod:`lib.pipeline`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["build", *sys.argv[1:]]))
