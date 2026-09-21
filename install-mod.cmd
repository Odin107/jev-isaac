@echo off
setlocal
cd /d "%~dp0"
set "JEV_PYTHON=%~dp0.venv\Scripts\python.exe"
if exist "%JEV_PYTHON%" (
  "%JEV_PYTHON%" launch.py install-mod
) else (
  py -3 launch.py install-mod
)
pause
