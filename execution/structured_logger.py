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

Standard event names (see execution/README.md for full catalogue):
    bot_start, bot_stop, cycle, signal, skip,
    pending_order_created, pending_order_cancelled, stale_pending_removed,
    broker_validation_failed, fallback_market_execution, market_order_placed,
    position_closed_detected, retry_attempt, spread_too_high,
    execution_latency, telegram_error, telegram_sent, error

The logger is intentionally tiny -- no buffering, no rotation -- because it
is called from a single-threaded cycle that runs every 5 minutes and the file
stays small. Add rotation here if you bump the cycle frequency.
"""

from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class StructuredLogger:
    """One instance per symbol; writes JSON-lines to `<log_dir>/<symbol>.json`."""

    def __init__(self, symbol: str, log_path: Path, echo_to_stdout: bool = True) -> None:
        self.symbol = symbol
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._echo = echo_to_stdout

    # ── core ──────────────────────────────────────────────────────────────────

    def event(self, event: str, **fields: Any) -> None:
        """
        Append one JSON line. `fields` are merged into the envelope.

        Use snake_case event names; keep field names consistent across calls
        (e.g. always `retcode`, not `rc` or `code`).
        """
        entry = {
            "ts":     datetime.now(timezone.utc).isoformat(),
            "event":  event,
            "symbol": self.symbol,
        }
        entry.update(fields)

        line = json.dumps(entry, default=str)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

        if self._echo:
            # Console summary -- skip noisy internal fields
            short = {k: v for k, v in fields.items() if k not in ("ob_bar_idx", "stack")}
            print(f"[{entry['ts']}] {event} | {short}", file=sys.stdout, flush=True)

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
