@echo off
setlocal
title Jev - responsive room trial
cd /d "%~dp0"
set "JEV_PYTHON=%~dp0.venv\Scripts\python.exe"
if exist "%JEV_PYTHON%" (
  "%JEV_PYTHON%" -u launch.py run --policy jev-goal --wait-for-arm --duration 180 --max-calls 120 --one-room --key-window --report reports\live-jev.json
) else (
  py -3 -u launch.py run --policy jev-goal --wait-for-arm --duration 180 --max-calls 120 --one-room --key-window --report reports\live-jev.json
)
pause
