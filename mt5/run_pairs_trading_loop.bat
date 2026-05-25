@echo off
REM ===========================================================================
REM  Pairs-trading runner in LOOP mode (forever).
REM
REM  Usage:
REM      Just double-click, or run from cmd:
REM          mt5\run_pairs_trading_loop.bat
REM
REM  Behaviour:
REM      The Python runner sleeps internally until the next H4 bar close + 30s
REM      grace, then evaluates all 4 pairs, logs everything, and sleeps again.
REM      Keep this cmd window open — closing it stops the bot.
REM
REM  To stop:
REM      Ctrl+C in this window (clean shutdown — bot_stop event logged).
REM
REM  Logs:
REM      Structured JSON  -> logs\pairs_trading\pairs-YYYY-MM-DD.json
REM      Wrapper stdout    -> logs\pairs_trading\loop\loop-YYYYMMDD-HHMMSS.log
REM
REM  Auto-restart on crash:
REM      The :loop block re-launches Python if it exits unexpectedly
REM      (with a 60s backoff so we don't tight-loop on a config error).
REM ===========================================================================

set REPO_ROOT=D:\bot\ema-1d trend\ema-h1trend
set VENV_PYTHON=%REPO_ROOT%\.venv\Scripts\python.exe

REM Per-launch log file so each restart has its own trail
set LOOP_LOG_DIR=%REPO_ROOT%\logs\pairs_trading\loop
if not exist "%LOOP_LOG_DIR%" mkdir "%LOOP_LOG_DIR%"

for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set DT=%%I
set RUN_TS=%DT:~0,8%-%DT:~8,6%
set LOOP_LOG=%LOOP_LOG_DIR%\loop-%RUN_TS%.log

echo ========================================================================
echo  Pairs-Trading Dry-Run Loop
echo  Started: %date% %time%
echo  Repo:    %REPO_ROOT%
echo  Logs:    %LOOP_LOG%
echo  Stop:    Ctrl+C  (clean shutdown)
echo ========================================================================

cd /d "%REPO_ROOT%"
set PYTHONIOENCODING=utf-8

REM ───────────────────────────────────────────────────────────────────────────
REM  Auto-restart loop:
REM    Python runner is itself a forever-loop (no --once). It should not exit
REM    under normal operation. If it DOES exit (crash, MT5 unreachable, etc.)
REM    we wait 60s and relaunch — Errante terminal sometimes drops the IPC
REM    and a restart re-establishes it cleanly.
REM ───────────────────────────────────────────────────────────────────────────

:loop
echo [%date% %time%] launching python runner...
echo [%date% %time%] launching python runner >> "%LOOP_LOG%"

"%VENV_PYTHON%" mt5\run_pairs_trading.py --dry-run >> "%LOOP_LOG%" 2>&1
set EXIT_CODE=%ERRORLEVEL%

echo [%date% %time%] python exited with code %EXIT_CODE%
echo [%date% %time%] python exited with code %EXIT_CODE% >> "%LOOP_LOG%"

REM Clean Ctrl+C exit (Python receives KeyboardInterrupt, returns code 0 or 1)
REM If user pressed Ctrl+C, the next Sleep below also gets it and we exit fast.
echo [%date% %time%] sleeping 60s before restart (Ctrl+C to abort)...
echo [%date% %time%] sleeping 60s before restart >> "%LOOP_LOG%"
timeout /t 60 /nobreak >nul

goto loop
