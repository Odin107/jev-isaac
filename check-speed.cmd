@echo off
setlocal
title Jev - four-request speed check
cd /d "%~dp0"
set "JEV_PYTHON=%~dp0.venv\Scripts\python.exe"
if exist "%JEV_PYTHON%" (
  "%JEV_PYTHON%" -u launch.py measure-latency --provider typesafe --samples 4 --key-window --report reports\typesafe-speed.json
) else (
  py -3 -u launch.py measure-latency --provider typesafe --samples 4 --key-window --report reports\typesafe-speed.json
)
pause
