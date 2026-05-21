"""
MT5 terminal health watchdog.

Why?
    The previous bot called `mt5.initialize()` once at startup and `mt5.shutdown()`
    at the end. If the terminal lost its broker connection mid-session (network
    blip, terminal auto-restart, broker server maintenance), the bot kept calling
    `copy_rates_from_pos` which silently returned None / stale data -- no
    reconnect, no alert.

Responsibilities (infrastructure ONLY -- no strategy or order logic):
    * detect whether MT5 IPC is alive (`mt5.terminal_info()`)
    * detect whether broker-side connection is alive (`terminal.connected`)
    * detect whether AutoTrading is on
    * if any of the above is false, attempt `mt5.initialize(...)` retries
    * log every state transition as a structured event
    * expose `is_healthy()` so the cycle can skip when needed

Crucially: the watchdog does NOT cancel/replace orders on reconnect. The
PendingOrderManager already reconciles in-memory state via `sync_from_broker()`
inside ExecutionEngine.begin_cycle() -- that path is the single source of
truth for "what does the broker actually have".
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

import MetaTrader5 as mt5

from .structured_logger import StructuredLogger


# Connect function signature: takes no args, raises on failure, returns None on success.
ConnectFn = Callable[[], None]


@dataclass
class WatchdogConfig:
    """Tunables for the watchdog. All defaults are conservative."""
    # How long to wait between reconnect attempts.
    reconnect_backoff_seconds: float = 10.0
    # Maximum reconnect attempts in a row before the watchdog gives up and just
    # keeps the bot ticking (cycles will skip on `ensure_healthy() == False`).
    max_reconnect_attempts: int = 6
    # Minimum gap between health-check log entries when status is unchanged.
    # Prevents log spam when MT5 is happy.
    healthy_log_interval_seconds: float = 600.0


class Mt5Watchdog:
    """
    Stateless-ish watchdog. One per bot. Call `ensure_healthy()` at the start
    of every cycle; it returns True when it is safe to proceed.

    Usage:
        watchdog = Mt5Watchdog(logger, connect_fn=lambda: mt5_connect())
        if not watchdog.ensure_healthy():
            continue  # skip this cycle; the watchdog already logged the cause
    """

    def __init__(
        self,
        logger: StructuredLogger,
        connect_fn: ConnectFn,
        config: Optional[WatchdogConfig] = None,
    ) -> None:
        self.logger = logger
        self.connect_fn = connect_fn
        self.config = config or WatchdogConfig()

        self._last_state: str | None = None        # last logged state
        self._last_healthy_log: float = 0.0        # monotonic time
        self._consecutive_failures: int = 0
        self._total_reconnects: int = 0

    # ── status ───────────────────────────────────────────────────────────────-

    def _probe(self) -> tuple[bool, str, dict]:
        """
        Return `(healthy, state_label, fields)`.

        State labels (stable, useful for grep/alerts):
            "healthy"
            "terminal_unreachable"     -- terminal_info() is None
            "broker_disconnected"      -- terminal.connected is False
            "autotrading_disabled"     -- terminal.trade_allowed is False
        """
        ti = mt5.terminal_info()
        if ti is None:
            return False, "terminal_unreachable", {"last_error": str(mt5.last_error())}

        fields = {
            "connected":     bool(ti.connected),
            "trade_allowed": bool(ti.trade_allowed),
            "ping_ms":       getattr(ti, "ping_last", None),
        }

        if not ti.connected:
            return False, "broker_disconnected", fields
        if not ti.trade_allowed:
            return False, "autotrading_disabled", fields

        return True, "healthy", fields

    # ── reconnect ────────────────────────────────────────────────────────────-

    def _attempt_reconnect(self) -> bool:
        """
        One reconnect attempt. Returns True on success. Each attempt is logged
        as `mt5_reconnect_attempt` and the outcome as `mt5_reconnect_success`
        or `mt5_reconnect_failed`.
        """
        attempt_n = self._consecutive_failures + 1
        self.logger.event("mt5_reconnect_attempt",
                          attempt=attempt_n,
                          total_reconnects_so_far=self._total_reconnects)

        # Best effort: shut down any half-broken session before re-init.
        try:
            mt5.shutdown()
        except Exception:
            # `mt5.shutdown` returns silently in most builds; guard anyway.
            pass

        try:
            self.connect_fn()
        except Exception as exc:
            self.logger.error("mt5_reconnect_failed",
                              exc=exc, attempt=attempt_n)
            return False

        # Verify the reconnect actually produced a healthy state.
        healthy, state, fields = self._probe()
        if healthy:
            self._total_reconnects += 1
            self.logger.event("mt5_reconnect_success",
                              attempt=attempt_n,
                              total_reconnects=self._total_reconnects,
                              **fields)
            return True

        self.logger.event("mt5_reconnect_failed",
                          attempt=attempt_n,
                          state=state, **fields)
        return False

    # ── public entry point ───────────────────────────────────────────────────-

    def ensure_healthy(self) -> bool:
        """
        Probe MT5; reconnect if needed; return whether it is safe to trade.

        Important properties:
            * Idempotent and cheap on the happy path.
            * Never raises -- always returns a bool.
            * Logs `mt5_connected` only on healthy <-> unhealthy transitions
              (or once per `healthy_log_interval_seconds` for a heartbeat-like
              "still alive" trace).
            * On consecutive failures, backs off `reconnect_backoff_seconds`
              between attempts (the caller is in a 5-minute cycle anyway, so
              this is mostly defensive against tight retries).
        """
        healthy, state, fields = self._probe()
        now = time.monotonic()

        if healthy:
            if self._last_state != "healthy":
                # Transition into healthy -- always log
                self.logger.event("mt5_connected", state=state, **fields,
                                  recovered_after_failures=self._consecutive_failures)
                self._consecutive_failures = 0
                self._last_state = "healthy"
                self._last_healthy_log = now
            elif now - self._last_healthy_log >= self.config.healthy_log_interval_seconds:
                # Periodic "still healthy" trace (not a heartbeat -- watchdog-specific)
                self.logger.event("mt5_connected", state=state, periodic=True, **fields)
                self._last_healthy_log = now
            return True

        # ── unhealthy path ───────────────────────────────────────────────────
        if self._last_state != state:
            self.logger.event("mt5_disconnected", state=state, **fields,
                              consecutive_failures=self._consecutive_failures)
            self._last_state = state

        # Throttle reconnect attempts -- one per cycle is usually plenty, but
        # we also enforce a hard backoff so a stuck terminal does not generate
        # one attempt per second if `ensure_healthy` is called in a tight loop.
        if self._consecutive_failures >= self.config.max_reconnect_attempts:
            self.logger.event("mt5_reconnect_giveup",
                              attempts=self._consecutive_failures,
                              advice="Operator intervention required.")
            return False

        if self._attempt_reconnect():
            return True

        self._consecutive_failures += 1
        # Backoff before the caller's next attempt -- short, so we don't block
        # the M5 cadence noticeably. The 5-minute outer sleep usually absorbs it.
        time.sleep(self.config.reconnect_backoff_seconds)
        return False
