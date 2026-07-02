@echo off
rem Nightly autonomous tuning run (registered in Windows Task Scheduler as
rem "BilliardsTrainer Autotune"). Writes ranked reports to docs\autotune\.
cd /d "%~dp0.."
if not exist docs\autotune mkdir docs\autotune
.venv\Scripts\python.exe tools\autotune.py --trials 40 >> docs\autotune\nightly.log 2>&1
