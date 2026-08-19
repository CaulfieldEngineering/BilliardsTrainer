"""Deploy the cloud review app. Stamps a BUILD_ID into the page and
version.json (the self-update mechanism compares the two), then ships
via the Vercel CLI using the token in Joe's secrets folder.

    python companion-cloud/deploy.py
"""

import json
import re
import secrets as pysecrets
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
SECRETS = Path("C:/Users/Joe/.billiards-secrets")


def main() -> int:
    token = (SECRETS / "vercel_token.txt").read_text().strip()
    build = time.strftime("%Y%m%d-%H%M%S")

    # stamp the page + version manifest (deploy-time copies; the template
    # keeps the placeholder)
    tpl = (HERE / "public" / "index.html").read_text(encoding="utf-8")
    if "__BUILD_ID__" not in tpl:
        tpl = re.sub(r'const BUILD = "[^"]*"',
                     'const BUILD = "__BUILD_ID__"', tpl)
    (HERE / "public" / "index.html").write_text(
        tpl.replace("__BUILD_ID__", build), encoding="utf-8")
    (HERE / "public" / "version.json").write_text(
        json.dumps({"build": build}), encoding="utf-8")

    cmd = ["npx", "-y", "vercel", "deploy", "--prod", "--yes",
           "--token", token, "--cwd", str(HERE)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                       shell=(sys.platform == "win32"))
    # restore the template placeholder so git never sees a stamped page
    (HERE / "public" / "index.html").write_text(tpl, encoding="utf-8")
    out = (r.stdout or "") + (r.stderr or "")
    url = next((w for w in out.split() if w.startswith("https://")
                and ".vercel.app" in w), None)
    print(out[-800:])
    if r.returncode != 0:
        return 1
    print(f"\nDEPLOYED build {build}: {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
