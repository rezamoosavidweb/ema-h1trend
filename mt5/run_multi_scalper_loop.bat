@echo off
REM ===========================================================================
REM  Multi-Symbol Reaction Scalper — LIVE LOOP MODE
REM
REM  ⚠️ WARNING:
REM      This runs the strategy LIVE — orders ARE sent to the broker.
REM      The Python runner has NO --dry-run flag in this wrapper.
REM      Make sure MT5 terminal is logged into a DEMO account before launch.
REM
REM  Usage:
REM      Just run from cmd:
REM          mt5\run_multi_scalper_loop.bat
REM
REM  Behaviour:
REM      The Python runner sleeps internally until the next M5 bar close,
REM      then evaluates each symbol in the golden basket and sends orders
REM      via ExecutionEngine. Keep this cmd window open — closing it stops
REM      the bot (open positions stay open on the broker).
REM
REM  To stop:
REM      Ctrl+C in this window (clean shutdown — bot_stop event logged
REM      per symbol). Open positions are NOT auto-closed; they continue
REM      under their own SL/TP until the bot is relaunched.
REM
REM  Logs:
REM      Per-symbol JSON   -> logs\<SYMBOL>-YYYY-MM-DD.json
REM      Portfolio JSON    -> logs\multi_symbol_scalper.json
REM      Wrapper stdout    -> logs\multi_scalper_loop\loop-YYYYMMDD-HHMMSS.log
REM
REM  Auto-restart on crash:
REM      The :loop block re-launches Python if it exits unexpectedly
REM      (with a 60s backoff so we don't tight-loop on a config error).
REM ===========================================================================

set REPO_ROOT=C:\Users\Administrator\Desktop\tob\ema-h1trend
set VENV_PYTHON=%REPO_ROOT%\.venv\Scripts\python.exe

REM Per-launch log file so each restart has its own trail
set LOOP_LOG_DIR=%REPO_ROOT%\logs\multi_scalper_loop
if not exist "%LOOP_LOG_DIR%" mkdir "%LOOP_LOG_DIR%"

REM Timestamp via PowerShell — wmic is deprecated/removed on Win11/Server 2022+
for /f "delims=" %%I in ('powershell -NoLogo -NoProfile -Command "Get-Date -Format yyyyMMdd-HHmmss"') do set RUN_TS=%%I
set LOOP_LOG=%LOOP_LOG_DIR%\loop-%RUN_TS%.log

echo ========================================================================
echo  Multi-Symbol Scalper — LIVE LOOP
echo  Started: %date% %time%
echo  Repo:    %REPO_ROOT%
echo  Logs:    %LOOP_LOG%
echo  MODE:    LIVE (no --dry-run) — orders WILL be sent to broker
echo  Stop:    Ctrl+C  (clean shutdown; open positions stay open)
echo ========================================================================

cd /d "%REPO_ROOT%"
set PYTHONIOENCODING=utf-8

REM ───────────────────────────────────────────────────────────────────────────
REM  Auto-restart loop:
REM    Python runner is itself a forever-loop. It should not exit under normal
REM    operation. If it DOES exit (crash, MT5 unreachable, etc.) we wait 60s
REM    and relaunch.
REM ───────────────────────────────────────────────────────────────────────────

:loop
echo [%date% %time%] launching python runner...
echo [%date% %time%] launching python runner >> "%LOOP_LOG%"

"%VENV_PYTHON%" mt5\run_multi_scalper.py >> "%LOOP_LOG%" 2>&1
set EXIT_CODE=%ERRORLEVEL%

echo [%date% %time%] python exited with code %EXIT_CODE%
echo [%date% %time%] python exited with code %EXIT_CODE% >> "%LOOP_LOG%"

echo [%date% %time%] sleeping 60s before restart (Ctrl+C to abort)...
echo [%date% %time%] sleeping 60s before restart >> "%LOOP_LOG%"
timeout /t 60 /nobreak >nul

goto loop
