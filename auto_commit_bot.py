"""
auto_commit_bot — periodic git auto-commit/push of bot runtime changes.

Runs on the same (Windows) VPS as the live MT5 bots and, every few hours,
stages **all** working-tree changes (mostly the per-symbol JSON logs under
``logs/``), commits them with a timestamped message, and pushes to ``origin``.

Design notes
------------
* Self-contained: only the Python stdlib + a working ``git`` on PATH. It does
  NOT import the project packages so it can run even if the trading code is
  mid-deploy / broken.
* Internal loop: launch once (e.g. via ``auto_commit_bot.bat``) and it sleeps
  ``--interval-hours`` between cycles. Failures are logged and retried next
  cycle — a transient push error never kills the loop.
* No-op safe: if the working tree is clean, the cycle is skipped (no empty
  commits).

Usage
-----
    python auto_commit_bot.py                      # every 4h (default)
    python auto_commit_bot.py --interval-hours 2   # every 2h
    python auto_commit_bot.py --once               # single cycle, then exit
    python auto_commit_bot.py --no-push            # commit locally only
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
LOG_DIR = REPO_ROOT / "logs" / "auto_commit"

log = logging.getLogger("auto_commit_bot")


def _setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log.setLevel(logging.INFO)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)

    fh = RotatingFileHandler(
        LOG_DIR / "auto_commit_bot.log",
        maxBytes=2_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    log.addHandler(fh)


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    """Run a git command inside the repo, capturing output."""
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def _has_changes() -> bool:
    """True if the working tree (incl. untracked files) has anything to commit."""
    res = _git("status", "--porcelain")
    if res.returncode != 0:
        log.error("git status failed: %s", res.stderr.strip())
        return False
    return bool(res.stdout.strip())


def commit_and_push(push: bool = True) -> bool:
    """
    Run one auto-commit cycle. Returns True if a commit was created.

    Never raises — all git failures are logged and swallowed so the caller's
    loop survives transient errors (network blips, locked index, etc.).
    """
    if not _has_changes():
        log.info("working tree clean — nothing to commit")
        return False

    add = _git("add", "-A")
    if add.returncode != 0:
        log.error("git add failed: %s", add.stderr.strip())
        return False

    # Count staged files for a more useful commit message.
    diff = _git("diff", "--cached", "--name-only")
    n_files = len([l for l in diff.stdout.splitlines() if l.strip()])
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"chore(logs): auto-commit {n_files} change(s) @ {stamp}"

    # --no-gpg-sign: these unattended log commits are unsigned (matching the
    # existing "update logs" history) so the bot never blocks on an SSH-key
    # passphrase prompt.
    commit = _git("commit", "--no-gpg-sign", "-m", msg)
    if commit.returncode != 0:
        log.error("git commit failed: %s", (commit.stderr or commit.stdout).strip())
        return False
    log.info("committed: %s", msg)

    if not push:
        log.info("--no-push set — skipping push")
        return True

    pushed = _git("push", "origin", "HEAD")
    if pushed.returncode != 0:
        log.error(
            "git push failed (commit kept locally, will retry next cycle): %s",
            (pushed.stderr or pushed.stdout).strip(),
        )
        return True

    log.info("pushed to origin")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interval-hours",
        type=float,
        default=4.0,
        help="hours between auto-commit cycles (default: 4)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="run a single cycle and exit (e.g. for an external scheduler)",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="commit locally but do not push to origin",
    )
    args = parser.parse_args(argv)

    _setup_logging()

    # Fail fast if this isn't a git repo.
    if _git("rev-parse", "--git-dir").returncode != 0:
        log.error("not a git repository: %s", REPO_ROOT)
        return 1

    push = not args.no_push

    if args.once:
        commit_and_push(push=push)
        return 0

    interval_s = args.interval_hours * 3600
    log.info(
        "auto_commit_bot started — repo=%s, interval=%.2fh, push=%s",
        REPO_ROOT,
        args.interval_hours,
        push,
    )
    while True:
        try:
            commit_and_push(push=push)
        except Exception:  # noqa: BLE001 — never let the loop die
            log.exception("unexpected error in commit cycle")
        log.info("sleeping %.2fh until next cycle", args.interval_hours)
        try:
            time.sleep(interval_s)
        except KeyboardInterrupt:
            log.info("interrupted — shutting down")
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
