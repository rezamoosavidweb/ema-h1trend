@echo off
REM ===========================================================================
REM  statarb_live — Phase 5 Live PAPER-Trading Loop (cointegration reversion +
REM  carry overlay + continuous regime sizing; frozen NB38 parameters)
REM
REM  SAFETY (read this):
REM      This is a PAPER run. The system NEVER sends orders to the broker.
REM      MetaTrader 5 is used ONLY for live market data + account equity; all
REM      fills are simulated by the execution simulator. There is no way for
REM      this loop to open a real (even demo) position. It is a research-
REM      validation harness, not a trader.
REM
REM  Prerequisites (MT5 / live-data mode):
REM      1. MT5 terminal open + logged into your DEMO account.
REM      2. Universe symbols visible in Market Watch (EURUSD, the 6 pairs' legs).
REM      3. statarb_live\deploy\.env filled in (SAL_BROKER=mt5, MT5_LOGIN, ...).
REM         The bat sets safe defaults below; .env values still take precedence
REM         for anything you don't override here.
REM      4. statarb_live\_data\universe.json present (committed with the repo).
REM
REM  Usage:
REM      statarb_live\run_statarb_live_loop.bat
REM
REM  Behaviour:
REM      `python -m statarb_live run` sleeps internally until the next H1 bar
REM      close (+grace), evaluates all three sleeves, logs every signal, and
REM      simulates fills. Keep this window open — closing it stops the bot.
REM
REM  To stop:
REM      Ctrl+C (clean shutdown — bot_stop event logged). Open paper positions
REM      are recovered from storage on the next launch.
REM
REM  Logs / outputs:
REM      Events JSON    -> logs\statarb_live\events-YYYY-MM-DD.json
REM      Storage (DB)   -> statarb_live\_data\statarb_live.db  (or SAL_DB_URL)
REM      Reports        -> statarb_live\_reports\  (run `report` separately)
REM      Wrapper stdout -> logs\statarb_live\loop\loop-YYYYMMDD-HHMMSS.log
REM
REM  Auto-restart on crash:
REM      The :loop block relaunches Python if it exits (60s backoff).
REM ===========================================================================

REM --- Resolve repo root from this script's location (statarb_live\ -> parent) ---
set SCRIPT_DIR=%~dp0
pushd "%SCRIPT_DIR%.." & set REPO_ROOT=%CD% & popd
set VENV_PYTHON=%REPO_ROOT%\.venv\Scripts\python.exe

REM --- Operational config (env). .env still overrides where you leave these unset) ---
set PYTHONIOENCODING=utf-8
if not defined SAL_BROKER set SAL_BROKER=mt5
if not defined SAL_MODE set SAL_MODE=paper

REM --- Per-launch wrapper log ---
set LOOP_LOG_DIR=%REPO_ROOT%\logs\statarb_live\loop
if not exist "%LOOP_LOG_DIR%" mkdir "%LOOP_LOG_DIR%"
for /f "delims=" %%I in ('powershell -NoLogo -NoProfile -Command "Get-Date -Format yyyyMMdd-HHmmss"') do set RUN_TS=%%I
set LOOP_LOG=%LOOP_LOG_DIR%\loop-%RUN_TS%.log

echo ========================================================================
echo  statarb_live — LIVE PAPER LOOP (no broker orders are ever sent)
echo  Started: %date% %time%
echo  Repo:    %REPO_ROOT%
echo  Broker:  %SAL_BROKER%   Mode: %SAL_MODE%
echo  Logs:    %LOOP_LOG%
echo  Stop:    Ctrl+C  (clean shutdown)
echo ========================================================================

cd /d "%REPO_ROOT%"

if not exist "%VENV_PYTHON%" (
    echo [ERROR] venv python not found at %VENV_PYTHON%
    echo         Create it:  python -m venv .venv ^&^& .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

REM ───────────────────────────────────────────────────────────────────────────
REM  Auto-restart loop: the Python runner is itself a forever-loop and should
REM  not exit under normal operation. If it DOES (crash, MT5 unreachable), wait
REM  60s and relaunch.
REM ───────────────────────────────────────────────────────────────────────────

:loop
echo [%date% %time%] launching statarb_live runner...
echo [%date% %time%] launching statarb_live runner >> "%LOOP_LOG%"

"%VENV_PYTHON%" -m statarb_live run >> "%LOOP_LOG%" 2>&1
set EXIT_CODE=%ERRORLEVEL%

echo [%date% %time%] python exited with code %EXIT_CODE%
echo [%date% %time%] python exited with code %EXIT_CODE% >> "%LOOP_LOG%"

echo [%date% %time%] sleeping 60s before restart (Ctrl+C to abort)...
echo [%date% %time%] sleeping 60s before restart >> "%LOOP_LOG%"
timeout /t 60 /nobreak >nul

goto loop
