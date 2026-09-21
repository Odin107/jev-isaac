@echo off
setlocal
cd /d "%~dp0"
set "JEV_PYTHON=%~dp0.venv\Scripts\python.exe"
if exist "%JEV_PYTHON%" (
  "%JEV_PYTHON%" launch.py demo --report reports\demo.json
) else (
  py -3 launch.py demo --report reports\demo.json
)
pause
