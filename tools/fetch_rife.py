"""Restore the RIFE interpolation binary to ~/.billiards-tools/rife.

The exe+models are ~35MB and stay out of git; the companion's smooth
slow-mo renderer (companion/rife_render.py) expects them here.

    python tools/fetch_rife.py
"""

import io
import urllib.request
import zipfile
from pathlib import Path

URL = ("https://github.com/nihui/rife-ncnn-vulkan/releases/download/"
       "20221029/rife-ncnn-vulkan-20221029-windows.zip")
DEST = Path.home() / ".billiards-tools" / "rife"


def main() -> int:
    if (DEST / "rife-ncnn-vulkan.exe").is_file():
        print(f"already present: {DEST}")
        return 0
    DEST.parent.mkdir(exist_ok=True)
    print(f"downloading {URL} ...")
    data = urllib.request.urlopen(URL, timeout=120).read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(DEST.parent)
    (DEST.parent / "rife-ncnn-vulkan-20221029-windows").rename(DEST)
    print(f"installed: {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
