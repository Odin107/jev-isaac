@echo off
setlocal
title Jev - adventure trial
cd /d "%~dp0"
set "JEV_PYTHON=%~dp0.venv\Scripts\python.exe"
if exist "%JEV_PYTHON%" (
  "%JEV_PYTHON%" -u launch.py run --policy jev-goal --floor --adventure --jev-player --continue-floors --wait-for-arm --restart-on-death --stay-ready --duration 900 --max-calls 1200 --key-window --report reports\live-floor.json
) else (
  py -3 -u launch.py run --policy jev-goal --floor --adventure --jev-player --continue-floors --wait-for-arm --restart-on-death --stay-ready --duration 900 --max-calls 1200 --key-window --report reports\live-floor.json
)
pause
