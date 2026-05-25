@echo off
REM ===========================================================================
REM  Windows Task Scheduler wrapper for pairs-trading runner.
REM
REM  Why a .bat?
REM    Task Scheduler can't natively activate a Python venv. This wrapper sets
REM    the cwd, picks the venv python, and forwards exit codes for monitoring.
REM
REM  Edit these two lines ONLY if your install layout differs:
REM ===========================================================================

set REPO_ROOT=D:\bot\ema-1d trend\ema-h1trend
set VENV_PYTHON=%REPO_ROOT%\.venv\Scripts\python.exe

REM ===========================================================================
REM  Logging — capture stdout/stderr so Scheduler "last run result" is useful.
REM  The structured JSON logs go to logs/pairs_trading/ regardless; this is
REM  just a safety net for crashes that happen BEFORE the logger initialises.
REM ===========================================================================

set SCHED_LOG_DIR=%REPO_ROOT%\logs\pairs_trading\scheduler
if not exist "%SCHED_LOG_DIR%" mkdir "%SCHED_LOG_DIR%"

REM Timestamped log per run (Scheduler keeps history of "last run" anyway,
REM but a per-invocation file is invaluable when something goes wrong at 3am)
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set DT=%%I
set RUN_TS=%DT:~0,8%-%DT:~8,6%
set RUN_LOG=%SCHED_LOG_DIR%\run-%RUN_TS%.log

echo [%date% %time%] starting pairs_trading --dry-run --once > "%RUN_LOG%"
echo cwd=%REPO_ROOT% >> "%RUN_LOG%"
echo python=%VENV_PYTHON% >> "%RUN_LOG%"

cd /d "%REPO_ROOT%"
set PYTHONIOENCODING=utf-8

"%VENV_PYTHON%" mt5\run_pairs_trading.py --dry-run --once >> "%RUN_LOG%" 2>&1
set EXIT_CODE=%ERRORLEVEL%

echo [%date% %time%] exit_code=%EXIT_CODE% >> "%RUN_LOG%"

REM Propagate exit code so Task Scheduler can detect failures
exit /b %EXIT_CODE%
