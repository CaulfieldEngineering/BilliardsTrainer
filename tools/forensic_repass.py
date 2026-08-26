"""CLI shim — the forensic corridor re-pass now lives in
src/billiards_trainer/vision/forensic_repass.py so the canonical close
pass can run it on every new session (it was a one-time backlog tool and
new sessions never got it: the marathon session had 51 misses, 1 tagged).

    python tools/forensic_repass.py --video X --start T [--end T] [--ball N]
"""
import sys
from pathlib import Path

from _lowprio import demote

demote()
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from billiards_trainer.vision.forensic_repass import main  # noqa: E402

if __name__ == "__main__":
    main()
