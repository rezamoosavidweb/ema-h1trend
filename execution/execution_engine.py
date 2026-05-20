"""
Top-level execution facade.

This is the only object the strategy script needs to construct. It owns:
    * SymbolConfig          (one per bot)
    * BrokerValidator
    * OrderFactory
    * PendingOrderManager
    * RiskAdapter
    * FallbackEngine
    * StructuredLogger

and exposes a single high-level call:

    engine.place_signal(signal_dict)

Responsibilities beyond just plumbing:
    * duplicate-order protection      (one ob_key, one outstanding pending)
    * retry throttling                (per-signal exponential cooldown)
    * stale-pending sweep              (every cycle)
    * orphan cancel                    (signal disappeared from current bar)
    * cooldown after repeated broker failures
    * close-detection                  (notify on TP/SL hit)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

import MetaTrader5 as mt5

from .broker_validator import BrokerValidator
from .fallback_engine import FallbackEngine, FallbackResult
from .order_factory import OrderFactory
from .pending_manager import PendingOrderManager
from .risk_adapter import RiskAdapter
from .structured_logger import StructuredLogger
from .symbol_config import SymbolConfig, resolve_symbol


# ── safety limits ────────────────────────────────────────────────────────────-

# After this many broker failures in COOLDOWN_WINDOW_SECONDS, the engine
# enters a cooldown and refuses to send new orders for COOLDOWN_DURATION.
DEFAULT_MAX_FAILURES_BEFORE_COOLDOWN = 5
DEFAULT_COOLDOWN_WINDOW_SECONDS = 5 * 60
DEFAULT_COOLDOWN_DURATION_SECONDS = 15 * 60

# Hard cap on retries per ob_key. After this many tries we give up on the OB.
DEFAULT_MAX_RETRIES_PER_OB = 3


@dataclass
class ExecutionOutcome:
    """Returned by `place_signal`. The strategy script logs/notifies on this."""
    placed: bool
    stage: str             # "limit" / "market" / "rejected" / "skipped"
    ticket: Optional[int]
    reason: str = ""
    fields: dict = field(default_factory=dict)


class ExecutionEngine:
    """Facade -- the only object the bot script needs."""

    def __init__(
        self,
        symbol: str,
        magic: int,
        log_path: Path,
        risk_per_trade: float = 0.01,
        risk_reward: float = 2.0,
        slippage_limit_threshold: float = 4.0,
        slippage_max_points: float = 6.0,
        max_spread_points: int = 200,
        stale_after_seconds: int = 5 * 60,
        comment_prefix: str = "ob_reaction",
        max_failures_before_cooldown: int = DEFAULT_MAX_FAILURES_BEFORE_COOLDOWN,
        cooldown_window_seconds: int = DEFAULT_COOLDOWN_WINDOW_SECONDS,
        cooldown_duration_seconds: int = DEFAULT_COOLDOWN_DURATION_SECONDS,
        max_retries_per_ob: int = DEFAULT_MAX_RETRIES_PER_OB,
    ) -> None:
        # Symbol -- this is the one and only place we resolve the broker name
        self.cfg = resolve_symbol(symbol)

        # Observability
        self.logger = StructuredLogger(symbol=self.cfg.name, log_path=log_path)

        # Components
        self.validator = BrokerValidator(max_spread_points=max_spread_points)
        self.factory = OrderFactory(magic=magic, comment_prefix=comment_prefix)
        self.pendings = PendingOrderManager(
            cfg=self.cfg, factory=self.factory, logger=self.logger,
            stale_after_seconds=stale_after_seconds,
        )
        self.risk = RiskAdapter(risk_per_trade=risk_per_trade)
        self.fallback = FallbackEngine(
            cfg=self.cfg, validator=self.validator, factory=self.factory,
            pendings=self.pendings, risk=self.risk, logger=self.logger,
            slippage_limit_threshold=slippage_limit_threshold,
            slippage_max_points=slippage_max_points,
            risk_reward=risk_reward,
        )

        # Safety state
        self._failures: list[float] = []         # monotonic timestamps of failures
        self._retry_counts: dict[tuple, int] = {}  # ob_key -> retries
        self._cooldown_until: float = 0.0
        self.max_failures_before_cooldown = max_failures_before_cooldown
        self.cooldown_window_seconds = cooldown_window_seconds
        self.cooldown_duration_seconds = cooldown_duration_seconds
        self.max_retries_per_ob = max_retries_per_ob

        # Position-close detection
        self._tracked_position_tickets: set[int] = set()

        self.logger.event(
            "bot_start",
            symbol=self.cfg.name, requested_symbol=self.cfg.requested,
            magic=magic,
            risk_per_trade=risk_per_trade, risk_reward=risk_reward,
            slippage_limit_threshold=slippage_limit_threshold,
            slippage_max_points=slippage_max_points,
            max_spread_points=max_spread_points,
            stale_after_seconds=stale_after_seconds,
        )

    # ── cycle housekeeping ───────────────────────────────────────────────────-

    def begin_cycle(self, active_ob_keys: Iterable[tuple]) -> None:
        """
        Called at the start of every strategy cycle BEFORE place_signal().
        Reconciles broker state, cancels stale/orphan pendings, sweeps closed
        positions.
        """
        # 1) Reconcile in-memory state with broker (handles restarts / external mods)
        try:
            self.pendings.sync_from_broker()
        except Exception as exc:
            self.logger.error("pending_sync_error", exc=exc)

        # 2) Drop pendings older than threshold
        self.pendings.cancel_stale()

        # 3) Drop pendings whose signal disappeared from the new bar
        if active_ob_keys is not None:
            self.pendings.cancel_orphans(active_ob_keys)

        # 4) Detect closures of positions we previously tracked (TP/SL hit)
        self._sweep_closed_positions()

    def _sweep_closed_positions(self) -> list[dict]:
        """
        Compare tracked tickets vs live positions; emit `position_closed_detected`
        for any that disappeared. Returns a list of close events for the caller
        to forward to Telegram.
        """
        live_tickets = {p.ticket for p in (mt5.positions_get(symbol=self.cfg.name) or [])
                        if p.magic == self.factory.magic}
        closed = self._tracked_position_tickets - live_tickets
        events: list[dict] = []

        if not closed:
            self._tracked_position_tickets = live_tickets
            return events

        now_utc = datetime.now(timezone.utc)
        lookback = now_utc - timedelta(hours=24)
        deals = mt5.history_deals_get(lookback, now_utc) or []
        profit_by_position: dict[int, float] = {}
        for d in deals:
            if d.entry == mt5.DEAL_ENTRY_OUT:
                profit_by_position[d.position_id] = (
                    profit_by_position.get(d.position_id, 0.0) + d.profit
                )

        ai = mt5.account_info()
        balance = float(ai.balance) if ai else 0.0
        equity  = float(ai.equity)  if ai else 0.0

        for ticket in closed:
            profit = profit_by_position.get(ticket, 0.0)
            ev = {"ticket": ticket, "profit": round(profit, 2),
                  "balance": round(balance, 2), "equity": round(equity, 2)}
            self.logger.event("position_closed_detected", **ev)
            events.append(ev)

        self._tracked_position_tickets = live_tickets
        return events

    # ── safety gates ─────────────────────────────────────────────────────────-

    def _in_cooldown(self, now: float) -> bool:
        return now < self._cooldown_until

    def _record_failure(self, now: float) -> None:
        """Track failure timestamps and trip cooldown if threshold is reached."""
        self._failures = [t for t in self._failures
                          if now - t < self.cooldown_window_seconds]
        self._failures.append(now)
        if len(self._failures) >= self.max_failures_before_cooldown:
            self._cooldown_until = now + self.cooldown_duration_seconds
            self.logger.event(
                "cooldown_engaged",
                failures=len(self._failures),
                window_s=self.cooldown_window_seconds,
                duration_s=self.cooldown_duration_seconds,
            )
            self._failures.clear()

    # ── main entry point ─────────────────────────────────────────────────────-

    def place_signal(self, signal: dict) -> ExecutionOutcome:
        """
        Validate + execute one signal. The signal dict must contain:
            direction:  "BUY" or "SELL"
            entry:      float
            sl:         float
            tp:         float
            ob_time:    str (used as part of dedup key)
            ... other fields are ignored

        Returns ExecutionOutcome describing what happened.
        """
        now = time.monotonic()
        side = "buy" if signal["direction"] == "BUY" else "sell"
        ob_key = (signal["ob_time"], signal["direction"])

        # ── Cooldown gate ────────────────────────────────────────────────────-
        if self._in_cooldown(now):
            self.logger.event("skip", reason="cooldown_active",
                              cooldown_remaining_s=round(self._cooldown_until - now, 1),
                              ob_key=list(ob_key))
            return ExecutionOutcome(False, "skipped", None, "cooldown_active")

        # ── Per-OB retry cap ─────────────────────────────────────────────────-
        attempts = self._retry_counts.get(ob_key, 0)
        if attempts >= self.max_retries_per_ob:
            self.logger.event("skip", reason="max_retries_exceeded",
                              ob_key=list(ob_key), attempts=attempts,
                              limit=self.max_retries_per_ob)
            return ExecutionOutcome(False, "skipped", None, "max_retries_exceeded")

        # ── Duplicate guard (pending or open position for this OB) ───────────-
        if self.pendings.has_active_for(ob_key):
            self.logger.event("skip", reason="duplicate_pending", ob_key=list(ob_key))
            return ExecutionOutcome(False, "skipped", None, "duplicate_pending")

        if self._has_open_position():
            self.logger.event(
                "skip",
                reason="position_open",
                ob_key=list(ob_key),
                missed_signal={"direction": signal["direction"],
                               "entry": signal["entry"],
                               "sl": signal["sl"], "tp": signal["tp"]},
            )
            return ExecutionOutcome(False, "skipped", None, "position_open")

        # ── Pre-flight (terminal, symbol, spread) ────────────────────────────-
        preflight = self.validator.preflight_all(self.cfg)
        if preflight is not None:
            self.logger.event("broker_validation_failed",
                              stage="preflight",
                              code=preflight.code,
                              detail=preflight.detail,
                              **(preflight.fields or {}))
            return ExecutionOutcome(False, "skipped", None, preflight.code,
                                    preflight.fields or {})

        # ── Account balance for sizing ───────────────────────────────────────-
        ai = mt5.account_info()
        balance = float(ai.balance) if ai else 1000.0

        # ── Cascade ──────────────────────────────────────────────────────────-
        try:
            result: FallbackResult = self.fallback.execute(
                side=side,
                ob_entry=float(signal["entry"]),
                sl=float(signal["sl"]),
                tp=float(signal["tp"]),
                ob_key=ob_key,
                balance=balance,
            )
        except Exception as exc:
            self.logger.error("execution_engine_exception", exc=exc,
                              ob_key=list(ob_key))
            self._record_failure(now)
            return ExecutionOutcome(False, "rejected", None, "exception",
                                    {"error_type": type(exc).__name__})

        # ── Bookkeeping ──────────────────────────────────────────────────────-
        self._retry_counts[ob_key] = attempts + 1

        if result.placed and result.stage == "market" and result.ticket:
            # Track for close-detection (only positions, not pendings)
            self._tracked_position_tickets.add(result.ticket)

        if not result.placed:
            self._record_failure(now)

        return ExecutionOutcome(
            placed=result.placed,
            stage=result.stage,
            ticket=result.ticket,
            reason=result.reason,
            fields=result.fields or {},
        )

    # ── helpers ──────────────────────────────────────────────────────────────-

    def _has_open_position(self) -> bool:
        positions = mt5.positions_get(symbol=self.cfg.name) or []
        return any(p.magic == self.factory.magic for p in positions)

    # ── shutdown ─────────────────────────────────────────────────────────────-

    def shutdown(self) -> None:
        """Idempotent. Cancels open pendings managed by this bot if asked."""
        self.logger.event("bot_stop",
                          tracked_pendings=self.pendings.active_tickets())

    def log_cycle(self, **fields) -> None:
        """Pass-through so the strategy script can write `cycle` events too."""
        self.logger.event("cycle", **fields)
