@echo off
setlocal
cd /d "%~dp0"
set "JEV_PYTHON=%~dp0.venv\Scripts\python.exe"
if exist "%JEV_PYTHON%" (
  "%JEV_PYTHON%" launch.py run --policy baseline --report reports\live-baseline.json
) else (
  py -3 launch.py run --policy baseline --report reports\live-baseline.json
)
pause
