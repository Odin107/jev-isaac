@echo off
setlocal
title Jev - TypeSafe connection check
cd /d "%~dp0"
set "JEV_PYTHON=%~dp0.venv\Scripts\python.exe"
if exist "%JEV_PYTHON%" (
  "%JEV_PYTHON%" launch.py check-access --provider typesafe --key-window --report reports\typesafe-access.json
) else (
  py -3 launch.py check-access --provider typesafe --key-window --report reports\typesafe-access.json
)
pause
