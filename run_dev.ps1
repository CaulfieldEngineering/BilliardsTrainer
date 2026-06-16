# Fast dev loop: launch the app from source (no PyInstaller build).
# First run: py -3.12 -m venv .venv ; .venv\Scripts\pip install -r requirements-dev.txt -e .
# Then iterate with this — edit code, Ctrl-C, re-run. Seconds, not minutes.
if (-not (Test-Path ".venv")) {
    Write-Host "No .venv — creating one and installing deps..." -ForegroundColor Yellow
    py -3.12 -m venv .venv
    .venv\Scripts\python.exe -m pip install -q -r requirements-dev.txt
    .venv\Scripts\python.exe -m pip install -q -e .
}
.venv\Scripts\python.exe -m billiards_trainer @args
