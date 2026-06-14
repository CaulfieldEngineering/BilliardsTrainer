"""Single source of truth for the application version.

CI overwrites ``__version__`` at build time (see ``.github/workflows/build.yml``)
with ``0.1.<github.run_number>`` so every push to main produces a monotonically
increasing version that the in-app updater can compare against ``version.json``.
"""

__version__ = "0.1.0"

APP_NAME = "Billiards Trainer"
APP_ID = "com.caulfieldengineering.billiardstrainer"
ORG_NAME = "Caulfield Engineering"

# Where the in-app updater looks for the latest release manifest.
# GitHub Releases' ``latest/download`` alias always resolves to the newest
# published release's asset of that name, so this URL never changes.
UPDATE_MANIFEST_URL = (
    "https://github.com/CaulfieldEngineering/BilliardsTrainer"
    "/releases/latest/download/version.json"
)
RELEASES_PAGE_URL = (
    "https://github.com/CaulfieldEngineering/BilliardsTrainer/releases/latest"
)
