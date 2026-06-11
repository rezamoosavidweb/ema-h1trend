@echo off
REM ===========================================================================
REM  statarb_live — Phase 5 Live PAPER-Trading Loop  (verbose / self-diagnosing)
REM
REM  SAFETY: PAPER run. NO orders are ever sent to the broker. MT5 is used only
REM  for live market data + account equity; all fills are simulated.
REM
REM  This launcher writes a step-by-step diagnostic to BOTH the console and
REM  logs\statarb_live\launcher.log, tests the Python env BEFORE the loop, and
REM  ALWAYS pauses on exit so the window never closes silently.
REM
REM  Usage:  statarb_live\run_statarb_live_loop.bat
REM  Stop:   Ctrl+C
REM ===========================================================================

setlocal EnableExtensions EnableDelayedExpansion
title statarb_live launcher
set "PYTHONIOENCODING=utf-8"

REM --- Resolve repo root from THIS script's folder (statarb_live\ -> parent) ---
set "SCRIPT_DIR=%~dp0"
for %%I in ("%~dp0..") do set "REPO_ROOT=%%~fI"
set "VENV_PYTHON=%REPO_ROOT%\.venv\Scripts\python.exe"

REM --- Diagnostic log (written immediately, even if Python never starts) ------
set "LOG_BASE=%REPO_ROOT%\logs\statarb_live"
if not exist "%LOG_BASE%" mkdir "%LOG_BASE%" 2>nul
if not exist "%LOG_BASE%" set "LOG_BASE=%TEMP%"
set "DIAG_LOG=%LOG_BASE%\launcher.log"
set "LOOP_DIR=%LOG_BASE%\loop"
if not exist "%LOOP_DIR%" mkdir "%LOOP_DIR%" 2>nul

REM --- Operational defaults (.env still overrides where unset) ----------------
if not defined SAL_BROKER set "SAL_BROKER=mt5"
if not defined SAL_MODE set "SAL_MODE=paper"

call :log "============================================================"
call :log "launcher start: %date% %time%"
call :log "SCRIPT_DIR  = %SCRIPT_DIR%"
call :log "REPO_ROOT   = %REPO_ROOT%"
call :log "VENV_PYTHON = %VENV_PYTHON%"
call :log "DIAG_LOG    = %DIAG_LOG%"
call :log "SAL_BROKER  = %SAL_BROKER%   SAL_MODE = %SAL_MODE%"
call :log "============================================================"

REM --- 1) pick a Python interpreter ------------------------------------------
set "PYEXE=%VENV_PYTHON%"
if not exist "%VENV_PYTHON%" (
    call :log "[WARN] venv python NOT found at %VENV_PYTHON%"
    call :log "[WARN] falling back to 'python' on PATH"
    set "PYEXE=python"
)
call :log "using interpreter: !PYEXE!"
"!PYEXE!" --version >> "%DIAG_LOG%" 2>&1
if errorlevel 1 (
    call :log "[ERROR] cannot run Python at !PYEXE!"
    call :log "        Create the venv:  python -m venv .venv"
    call :log "        Then:             .venv\Scripts\pip install -r requirements.txt"
    goto :fail
)

REM --- 2) move to repo root so .env + relative paths resolve ------------------
cd /d "%REPO_ROOT%"
if errorlevel 1 (
    call :log "[ERROR] cannot cd to REPO_ROOT: %REPO_ROOT%"
    goto :fail
)
call :log "cwd = %CD%"

REM --- 3) import smoke-test (surfaces missing deps / engine import errors) ----
call :log "testing imports (statarb_live + engine)..."
"!PYEXE!" -c "import statarb_live; from statarb_live.engine_bridge import eng_pairs; from statarb_live.orchestrator import Orchestrator; print('import OK')" >> "%DIAG_LOG%" 2>&1
if errorlevel 1 (
    call :log "[ERROR] import FAILED - see traceback above in %DIAG_LOG%"
    call :log "        Most common cause: missing deps on this machine."
    call :log "        Fix: !PYEXE! -m pip install -r requirements.txt"
    goto :fail
)
call :log "[OK] imports fine"

REM --- 4) config + universe sanity (also confirms data/universe present) ------
call :log "checking config + frozen universe..."
"!PYEXE!" -m statarb_live info >> "%DIAG_LOG%" 2>&1
if errorlevel 1 (
    call :log "[ERROR] 'statarb_live info' failed - see %DIAG_LOG%"
    call :log "        If it mentions universe/data: ensure statarb_live\_data\universe.json exists."
    goto :fail
)
call :log "[OK] config + universe loaded"

echo ========================================================================
echo  statarb_live — LIVE PAPER LOOP (no broker orders are ever sent)
echo  Repo:   %REPO_ROOT%
echo  Python: !PYEXE!
echo  Broker: %SAL_BROKER%   Mode: %SAL_MODE%
echo  Logs:   %LOG_BASE%
echo  Stop:   Ctrl+C
echo ========================================================================

REM --- 5) auto-restart loop --------------------------------------------------
:loop
for /f "delims=" %%I in ('powershell -NoLogo -NoProfile -Command "Get-Date -Format yyyyMMdd-HHmmss"') do set "RUN_TS=%%I"
if not defined RUN_TS set "RUN_TS=run"
set "LOOP_LOG=%LOOP_DIR%\loop-%RUN_TS%.log"

call :log "launching runner -> %LOOP_LOG%"
"!PYEXE!" -m statarb_live run >> "%LOOP_LOG%" 2>&1
set "EXIT_CODE=!ERRORLEVEL!"
call :log "runner exited with code !EXIT_CODE! (see %LOOP_LOG%)"

REM If it crashed instantly and repeatedly, the loop log holds the traceback.
call :log "restarting in 60s (Ctrl+C to abort)..."
timeout /t 60 /nobreak >nul
goto :loop

:fail
call :log "launcher ABORTED. Read the messages above / %DIAG_LOG%."
echo.
echo  *** Startup failed. The window will stay open so you can read the error. ***
pause
goto :eof

REM --- log subroutine: echo to console AND append to the diagnostic file ------
:log
echo %~1
>> "%DIAG_LOG%" echo [%date% %time%] %~1
goto :eof
