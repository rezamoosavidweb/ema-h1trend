"""
Pending-order lifecycle manager.

Why client-side lifecycle?
    Broker-side expiration (ORDER_TIME_SPECIFIED + Unix epoch) is fragile -- it
    is interpreted in broker-local time, so a UTC timestamp from Python lands
    in the past on UTC+3 brokers and gets rejected as "Invalid expiration"
    (retcode 10022). See execution/order_factory.py for the broader context.

    Instead, every LIMIT order goes in with `ORDER_TIME_GTC` (never auto-
    expires) and PendingOrderManager:
        * tracks (ticket, created_at_utc) in memory
        * cancels orders older than `stale_after_seconds`
        * cancels orders whose signal key disappeared from the new bar
        * blocks duplicate orders for the same `ob_key`

Responsibilities:
    * place_limit()                  send a limit + start tracking
    * cancel_stale(now)              cancel orders older than the threshold
    * cancel_orphans(active_keys)    cancel orders whose signal has gone
    * has_active_for(ob_key)         dedupe guard
    * sync_from_broker(symbol)       reconcile after restart (read live pendings)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable, Optional

import MetaTrader5 as mt5

from .order_factory import OrderFactory
from .structured_logger import StructuredLogger
from .symbol_config import SymbolConfig


# Default lifetime for a LIMIT order before the client cancels it. One M5 bar.
DEFAULT_STALE_AFTER_SECONDS = 5 * 60


@dataclass
class PendingRecord:
    """In-memory record of a live pending order managed by this bot."""
    ticket:     int
    side:       str             # "buy" or "sell"
    entry:      float
    sl:         float
    tp:         float
    ob_key:     tuple           # (ob_time, direction) -- dedup key
    created_at: float = field(default_factory=time.monotonic)


class PendingOrderManager:
    """
    Tracks the bot's outstanding pending orders for a single symbol.

    Thread-safety: the bot is single-threaded; if you ever introduce concurrent
    cycles, guard `self._records` with a lock.
    """

    def __init__(
        self,
        cfg: SymbolConfig,
        factory: OrderFactory,
        logger: StructuredLogger,
        stale_after_seconds: int = DEFAULT_STALE_AFTER_SECONDS,
    ) -> None:
        self.cfg = cfg
        self.factory = factory
        self.logger = logger
        self.stale_after_seconds = stale_after_seconds
        self._records: dict[int, PendingRecord] = {}

    # ── lookup ────────────────────────────────────────────────────────────────

    def has_active_for(self, ob_key: tuple) -> bool:
        """True if a pending order already exists for this (ob_time, direction)."""
        return any(r.ob_key == ob_key for r in self._records.values())

    def active_tickets(self) -> list[int]:
        return list(self._records.keys())

    # ── place ────────────────────────────────────────────────────────────────-

    def place_limit(
        self,
        side: str,
        volume: float,
        entry: float,
        sl: float,
        tp: float,
        ob_key: tuple,
    ) -> Optional[int]:
        """
        Send a LIMIT order via order_factory and track it. Returns the broker
        ticket on success, None on failure (caller decides whether to fall back).

        The factory builds the request with `ORDER_TIME_GTC` (no broker-side
        expiry); we cancel the order ourselves after `stale_after_seconds`.
        """
        if self.has_active_for(ob_key):
            self.logger.event("skip", reason="duplicate_pending", ob_key=list(ob_key))
            return None

        req = self.factory.build_limit(self.cfg, side, volume, entry, sl, tp)
        result = mt5.order_send(req.request)

        if result is None:
            self.logger.error("pending_order_error",
                              reason="order_send_returned_none",
                              last_error=str(mt5.last_error()),
                              request=req.request)
            return None

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.event(
                "pending_order_error",
                retcode=result.retcode,
                comment=result.comment,
                side=side,
                entry=entry, sl=sl, tp=tp,
                volume=volume,
            )
            return None

        record = PendingRecord(
            ticket=result.order,
            side=side, entry=entry, sl=sl, tp=tp, ob_key=ob_key,
        )
        self._records[result.order] = record

        self.logger.event(
            "pending_order_created",
            ticket=result.order,
            side=side,
            entry=entry, sl=sl, tp=tp, volume=volume,
            ob_key=list(ob_key),
            stale_after_s=self.stale_after_seconds,
        )
        return result.order

    # ── cancel ───────────────────────────────────────────────────────────────-

    def cancel(self, ticket: int, reason: str) -> bool:
        """
        Send a TRADE_ACTION_REMOVE for `ticket`. Always drops the record from
        our tracking dict, even if the broker call fails -- otherwise a stuck
        record would block new orders forever.
        """
        req = self.factory.build_cancel(ticket)
        result = mt5.order_send(req.request)
        ok = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE

        rec = self._records.pop(ticket, None)
        self.logger.event(
            "pending_order_cancelled",
            ticket=ticket,
            reason=reason,
            ok=ok,
            retcode=getattr(result, "retcode", None),
            comment=getattr(result, "comment", None),
            ob_key=list(rec.ob_key) if rec else None,
        )
        return ok

    def cancel_stale(self) -> int:
        """Cancel everything older than `stale_after_seconds`. Returns count."""
        now = time.monotonic()
        cutoff = now - self.stale_after_seconds
        to_kill = [t for t, r in self._records.items() if r.created_at < cutoff]
        for t in to_kill:
            self.cancel(t, reason="stale_timeout")
            self.logger.event("stale_pending_removed", ticket=t,
                              age_s=round(now - self._records.get(t, PendingRecord(
                                  ticket=t, side="", entry=0, sl=0, tp=0, ob_key=()
                              )).created_at, 1))
        return len(to_kill)

    def cancel_orphans(self, active_keys: Iterable[tuple]) -> int:
        """
        Cancel any pending order whose `ob_key` is not in `active_keys`.

        Use case: each cycle the strategy emits a set of currently-valid OB
        keys. If our tracked order is for an OB that no longer appears, the
        thesis is gone and we should free up the slot.
        """
        active = set(tuple(k) for k in active_keys)
        to_kill = [t for t, r in self._records.items() if r.ob_key not in active]
        for t in to_kill:
            self.cancel(t, reason="signal_invalidated")
        return len(to_kill)

    # ── reconcile ────────────────────────────────────────────────────────────-

    def sync_from_broker(self) -> None:
        """
        Reconcile our in-memory state with live pendings reported by MT5.

        Two purposes:
            1. Drop tickets that were filled or cancelled outside our process
               (e.g. by manual intervention or a TP/SL fill mid-cycle).
            2. Adopt orphan pending orders that match our magic/comment after
               a bot restart, so we can manage their lifecycle.
        """
        live = mt5.orders_get(symbol=self.cfg.name) or []
        live_by_ticket = {o.ticket: o for o in live if o.magic == self.factory.magic}

        # 1) Drop our records that are no longer live
        gone = [t for t in self._records if t not in live_by_ticket]
        for t in gone:
            rec = self._records.pop(t)
            self.logger.event("pending_order_disappeared",
                              ticket=t, ob_key=list(rec.ob_key))

        # 2) Adopt live pendings that we don't yet track
        for t, o in live_by_ticket.items():
            if t in self._records:
                continue
            # Synthetic ob_key from comment+price -- best effort after restart;
            # cancel_orphans will clean it up if the signal doesn't match.
            ob_key = ("orphan", float(o.price_open))
            self._records[t] = PendingRecord(
                ticket=t,
                side="buy" if o.type == mt5.ORDER_TYPE_BUY_LIMIT else "sell",
                entry=float(o.price_open),
                sl=float(o.sl),
                tp=float(o.tp),
                ob_key=ob_key,
            )
            self.logger.event("pending_order_adopted", ticket=t,
                              entry=o.price_open, sl=o.sl, tp=o.tp)
