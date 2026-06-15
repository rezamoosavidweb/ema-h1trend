@echo off
REM Launch the git auto-commit/push bot alongside the live MT5 bots.
REM Double-click on the Windows VPS, or add to startup. Ctrl+C to stop.
setlocal
cd /d "%~dp0"

REM Prefer the project virtualenv python; fall back to system python.
if exist "%~dp0.venv\Scripts\python.exe" (
    set "PY=%~dp0.venv\Scripts\python.exe"
) else (
    set "PY=python"
)

"%PY%" "%~dp0auto_commit_bot.py" %*

REM Keep the window open if it exits/crashes so the error is visible.
echo.
echo auto_commit_bot exited with code %ERRORLEVEL%.
pause
