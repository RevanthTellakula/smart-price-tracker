@echo off
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PY=%~dp0.venv\Scripts\python.exe"
) else (
    set "PY=python"
)

echo Starting Smart Price Tracker...
echo Keep this window open while you use the app.
echo Then open: http://127.0.0.1:5000
echo.

start "" http://127.0.0.1:5000/
"%PY%" "%~dp0price_tracker_app.py"
pause
