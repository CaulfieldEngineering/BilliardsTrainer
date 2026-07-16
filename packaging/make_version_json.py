"""Generate the ``version.json`` update manifest consumed by the in-app updater.

    python packaging/make_version_json.py --version 0.1.42 \
        --url https://.../BilliardsTrainer-0.1.42.exe \
        --notes "..." --out dist/version.json
"""

import argparse
import json
from datetime import datetime, timezone


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--url", required=True, help="Windows .exe download URL")
    ap.add_argument("--notes", default="")
    ap.add_argument("--sha256", default="", help="SHA256 of the installer for integrity checking")
    ap.add_argument("--mac-url", default="", help="macOS .app zip download URL")
    ap.add_argument("--mac-sha256", default="")
    ap.add_argument("--out", default="dist/version.json")
    args = ap.parse_args()

    manifest = {
        "version": args.version,
        "url": args.url,
        "sha256": args.sha256.lower(),
        "mac_url": args.mac_url,
        "mac_sha256": args.mac_sha256.lower(),
        "notes": args.notes,
        "pub_date": datetime.now(timezone.utc).isoformat(),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
