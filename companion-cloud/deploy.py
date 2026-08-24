"""Deploy the cloud review app. Stamps a BUILD_ID into the page and
version.json (the self-update mechanism compares the two), then ships
via the Vercel CLI using the token in Joe's secrets folder.

    python companion-cloud/deploy.py
"""

import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
SECRETS = Path("C:/Users/Joe/.billiards-secrets")


def main() -> int:
    token = (SECRETS / "vercel_token.txt").read_text().strip()
    build = time.strftime("%Y%m%d-%H%M%S")

    # stamp page + script + version manifest (deploy-time copies; the
    # templates keep the placeholder). index.html carries ?v=BUILD on the
    # app.js reference (cache bust), app.js carries the BUILD const the
    # self-update pill compares against version.json.
    tpl_html = (HERE / "public" / "index.html").read_text(encoding="utf-8")
    if "__BUILD_ID__" not in tpl_html:
        tpl_html = re.sub(r'app\.js\?v=[^"]*', "app.js?v=__BUILD_ID__",
                          tpl_html)
    tpl_js = (HERE / "public" / "app.js").read_text(encoding="utf-8")
    if "__BUILD_ID__" not in tpl_js:
        tpl_js = re.sub(r'const BUILD = "[^"]*"',
                        'const BUILD = "__BUILD_ID__"', tpl_js)
    (HERE / "public" / "index.html").write_text(
        tpl_html.replace("__BUILD_ID__", build), encoding="utf-8")
    (HERE / "public" / "app.js").write_text(
        tpl_js.replace("__BUILD_ID__", build), encoding="utf-8")
    (HERE / "public" / "version.json").write_text(
        json.dumps({"build": build}), encoding="utf-8")

    cmd = ["npx", "-y", "vercel", "deploy", "--prod", "--yes",
           "--token", token, "--cwd", str(HERE)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                       shell=(sys.platform == "win32"))
    # restore the template placeholders so git never sees a stamped page
    (HERE / "public" / "index.html").write_text(tpl_html, encoding="utf-8")
    (HERE / "public" / "app.js").write_text(tpl_js, encoding="utf-8")
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
