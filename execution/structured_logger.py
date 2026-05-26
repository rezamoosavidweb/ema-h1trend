"""
Append-only JSON-lines event log for the live bot.

Why a custom logger?
    The bot used to mix free-form `print` calls and ad-hoc `json.dumps` lines.
    That made post-mortem analysis with pandas painful (missing fields, mixed
    types, free-form `comment` strings). This logger enforces a stable schema:

        {
            "ts":         "<ISO-8601 UTC>",
            "event":      "<snake_case_event_name>",
            "symbol":     "<broker symbol>",
            ...event-specific fields...
        }

Logging is ALWAYS append-only. The previous incident where logs/XAUUSD.json
went from 461 lines to 101 lines was caused by an external deploy step
truncating the file -- not this logger. With `rotate_daily=True` (the
default) every day gets its own file, so even an accidental truncation only
loses today's history.

File layout with `rotate_daily=True` (default):
    logs/XAUUSD.json                 <- symlink/copy of today's file (best-effort)
    logs/XAUUSD-2026-05-21.json      <- one file per UTC day
    logs/XAUUSD-2026-05-22.json
    ...

File layout with `rotate_daily=False`:
    logs/XAUUSD.json                 <- single ever-growing append-only file

This module does NOT delete or compress old logs -- do that out-of-band
(cron, logrotate) so we never accidentally drop forensic data.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


# Broker wall-clock offset (Errante = EET/EEST → +2 winter, +3 summer EEST).
# Used to derive `timeBroker` so the bar timestamps printed in logs match
# what's shown on the MT5 terminal clock and what the strategy uses for its
# session window. NY offset is broker-7h (DEFAULT_BROKER_TO_NY_H).
#
# These are intentionally constants (not config) — log readability matters
# more than handling broker tz changes, which happen twice a year. If DST
# changes, update here.
BROKER_WALLCLOCK_OFFSET_HOURS = 3
BROKER_TO_NY_HOURS            = 7


class StructuredLogger:
    """
    One instance per symbol; writes JSON-lines to one file per UTC day.

    The path passed in is treated as the *base* path (e.g. logs/XAUUSD.json).
    With rotation enabled, the daily file is derived as
    `<dir>/<stem>-YYYY-MM-DD<ext>`. A "latest" pointer at the base path is
    maintained best-effort so existing tooling (grep/tail on the base name)
    still works on most setups; if symlinks are unsupported (typical Windows
    user account) we silently skip the pointer.
    """

    def __init__(
        self,
        symbol: str,
        log_path: Path,
        echo_to_stdout: bool = True,
        rotate_daily: bool = True,
    ) -> None:
        self.symbol = symbol
        self._base_path = Path(log_path)
        self._base_path.parent.mkdir(parents=True, exist_ok=True)
        self._echo = echo_to_stdout
        self._rotate_daily = rotate_daily
        self._cached_date: str | None = None
        self._cached_path: Path = self._base_path  # filled by _current_path()

    # ── path resolution ──────────────────────────────────────────────────────-

    def _current_path(self) -> Path:
        """Resolve the active log file for the current UTC date (if rotating)."""
        if not self._rotate_daily:
            return self._base_path

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today == self._cached_date:
            return self._cached_path

        stem = self._base_path.stem            # "XAUUSD"
        ext  = self._base_path.suffix or ".json"
        dated = self._base_path.with_name(f"{stem}-{today}{ext}")
        dated.parent.mkdir(parents=True, exist_ok=True)

        # Maintain a "latest" pointer at the base path -- best effort.
        self._update_latest_pointer(dated)

        self._cached_date = today
        self._cached_path = dated
        return dated

    def _update_latest_pointer(self, dated: Path) -> None:
        """
        Make the base path point at the current daily file so existing
        tooling that watches `logs/XAUUSD.json` keeps working.

        Strategy (in order of preference):
            1. Symlink (POSIX / Windows-Dev mode).
            2. If symlinks fail, leave the base file alone -- callers can
               find the dated file via the deterministic naming.
        We deliberately do NOT copy the file every day -- that would defeat
        append-only semantics and double disk usage.
        """
        base = self._base_path
        try:
            if base.is_symlink() or base.exists():
                # Replace existing symlink only -- don't clobber a real file.
                if base.is_symlink():
                    base.unlink()
                else:
                    return  # base is a real file (likely from old single-file mode); leave it
            os.symlink(dated.name, base)
        except OSError:
            # Windows without dev-mode forbids symlinks for normal users.
            # Silent -- this is best-effort observability, not correctness.
            pass

    # ── core ──────────────────────────────────────────────────────────────────

    def event(self, event: str, **fields: Any) -> None:
        """
        Append one JSON line. `fields` are merged into the envelope.

        Each entry carries three TIME views for easy log triage:
            ts            — ISO-8601 UTC with microsecond precision
                            (legacy field, kept for parser compatibility)
            timeUTC       — UTC formatted "YYYY-MM-DD HH:MM:SS UTC"
            timeBroker    — broker wall-clock (Errante = UTC+3 EEST):
                            this matches what MT5 terminal clock shows AND
                            the bar_time in cycle events.
            timeNY        — New York local hour reference (broker −7h)
                            so session-window checks read at a glance.

        And two contextual fields:
            event         — snake_case event name
            symbol        — symbol the logger is bound to

        Use snake_case event names; keep field names consistent across calls
        (e.g. always `retcode`, not `rc` or `code`).
        """
        now_utc      = datetime.now(timezone.utc)
        now_broker   = now_utc + timedelta(hours=BROKER_WALLCLOCK_OFFSET_HOURS)
        now_ny       = now_broker - timedelta(hours=BROKER_TO_NY_HOURS)

        entry = {
            "ts":         now_utc.isoformat(),
            "timeUTC":    now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "timeBroker": now_broker.strftime("%Y-%m-%d %H:%M:%S BROKER"),
            "timeNY":     now_ny.strftime("%Y-%m-%d %H:%M:%S NY"),
            "event":      event,
            "symbol":     self.symbol,
        }
        entry.update(fields)

        line = json.dumps(entry, default=str)
        path = self._current_path()
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

        if self._echo:
            # Console summary -- skip noisy internal fields
            short = {k: v for k, v in fields.items() if k not in ("ob_bar_idx", "stack")}
            print(f"[{entry['timeBroker']}] {event} | {short}", file=sys.stdout, flush=True)

    # ── convenience wrappers ─────────────────────────────────────────────────-

    def error(self, event: str, exc: BaseException | None = None, **fields: Any) -> None:
        """Log an error event with optional stack trace. Never raises."""
        if exc is not None:
            fields.setdefault("error_type",   type(exc).__name__)
            fields.setdefault("error_msg",    str(exc))
            fields.setdefault("stack",        traceback.format_exc())
        self.event(event, **fields)

    def latency(self, event: str, started_at: float, **fields: Any) -> None:
        """
        Log an `execution_latency` companion event. `started_at` is a
        time.monotonic() reading captured before the operation.
        """
        import time
        fields.setdefault("latency_ms", round((time.monotonic() - started_at) * 1000, 1))
        self.event(event, **fields)
