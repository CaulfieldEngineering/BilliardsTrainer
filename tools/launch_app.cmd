@echo off
rem Billiards Trainer launcher: always the latest build.
rem Pulls the newest pushed code (fast-forward only; never disturbs
rem uncommitted work), then starts the app detached.
cd /d C:\Users\Joe\Documents\GitHub\BilliardsTrainer
git pull --ff-only --quiet
start "" "C:\Users\Joe\Documents\GitHub\BilliardsTrainer\.venv\Scripts\pythonw.exe" -m billiards_trainer
