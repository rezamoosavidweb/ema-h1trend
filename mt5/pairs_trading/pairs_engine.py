"""
Two-leg execution engine for pairs trading.

This is the only module that calls `mt5.order_send`. Everything else stays
strictly above the broker: signals decide, this module enacts.

Critical invariant
------------------
**Never leave one leg open without its hedge.**

If the y-leg fills and the x-leg is then rejected, we IMMEDIATELY close y at
market and log `partial_fill_emergency`. The pair is left FLAT in state — the
runner can retry next cycle. We do NOT persist a half-open pair.

Public API
----------
    PairsExecutionEngine(
        cfg, logger, state_store,
        magic_for_pair, dry_run=False,
        notifier=None,        # optional Mt5Notifier
    )

    .open_pair(spread, sizing, beta, alpha, spread_now, z_now, side, bar_close)
    .close_pair(state, reason, spread_now, z_now)
    .resolve_symbols(spread)   -> (SymbolConfig_y, SymbolConfig_x)
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

import MetaTrader5 as mt5

from execution.structured_logger import StructuredLogger
from execution.symbol_config import SymbolConfig, resolve_symbol

from .config import COMMENT_PREFIX, PairsConfig
from .signals import Side
from .sizing import SizingResult
from .state import PairsStateStore, PairState


# ─────────────────────────────────────────────────────────────────────────────
# Order construction
# ─────────────────────────────────────────────────────────────────────────────


def _market_order_request(
    *, symbol: str, side: Side, volume: float, price: float,
    magic: int, comment: str, deviation_points: int = 20,
) -> dict:
    """
    Build an MT5 order_send request dict.

    `side` is the PAIR side; we map to per-leg BUY/SELL inside callers since
    each leg trades in opposite direction.
    """
    return {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       symbol,
        "volume":       float(volume),
        "type":         mt5.ORDER_TYPE_BUY if side == Side.LONG else mt5.ORDER_TYPE_SELL,
        "price":        float(price),
        "deviation":    int(deviation_points),
        "magic":        int(magic),
        "comment":      comment[:31],   # MT5 truncates at 31 chars
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }


def _close_position_request(
    *, position, price: float, deviation_points: int = 20,
) -> dict:
    """Build a closing DEAL for an existing `mt5.positions_get` entry."""
    # To close: send opposite-direction DEAL with the position ticket attached.
    closing_type = (
        mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY
        else mt5.ORDER_TYPE_BUY
    )
    return {
        "action":       mt5.TRADE_ACTION_DEAL,
        "position":     int(position.ticket),
        "symbol":       position.symbol,
        "volume":       float(position.volume),
        "type":         closing_type,
        "price":        float(price),
        "deviation":    int(deviation_points),
        "magic":        int(position.magic),
        "comment":      f"{COMMENT_PREFIX}:close",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }


# ─────────────────────────────────────────────────────────────────────────────
# The engine
# ─────────────────────────────────────────────────────────────────────────────


class PairsExecutionEngine:
    """
    Two-leg execution for ONE pair at a time. Stateless across calls — all
    persistence goes through `state_store`.
    """

    def __init__(
        self,
        cfg:          PairsConfig,
        logger:       StructuredLogger,
        state_store:  PairsStateStore,
        magic_for_pair: dict[str, int],   # pair_key -> magic number
        dry_run:      bool = False,
        notifier=None,                   # optional Mt5Notifier (duck-typed)
    ) -> None:
        self.cfg = cfg
        self.logger = logger
        self.state_store = state_store
        self.magic_for_pair = magic_for_pair
        self.dry_run = dry_run
        self.notifier = notifier

        # Cache resolved SymbolConfigs — broker names rarely change mid-session
        self._symbol_cache: dict[str, SymbolConfig] = {}

    # ── symbol resolution ───────────────────────────────────────────────────

    def resolve_symbol(self, symbol: str) -> SymbolConfig:
        if symbol not in self._symbol_cache:
            self._symbol_cache[symbol] = resolve_symbol(symbol)
        return self._symbol_cache[symbol]

    # ── pricing helpers ─────────────────────────────────────────────────────

    def _current_tick(self, broker_symbol: str):
        tick = mt5.symbol_info_tick(broker_symbol)
        if tick is None:
            raise RuntimeError(
                f"symbol_info_tick({broker_symbol}) returned None: {mt5.last_error()}"
            )
        return tick

    def _price_for_buy(self, broker_symbol: str) -> float:
        return float(self._current_tick(broker_symbol).ask)

    def _price_for_sell(self, broker_symbol: str) -> float:
        return float(self._current_tick(broker_symbol).bid)

    # ── notifier wrapper (silent if no notifier) ────────────────────────────

    def _notify(self, kind: str, text: str) -> None:
        if self.notifier is None:
            return
        try:
            send = getattr(self.notifier, "send", None)
            if callable(send):
                send(f"[pairs:{kind}] {text}")
        except Exception as exc:
            self.logger.error("notifier_send_failed", exc=exc, kind=kind)

    # ── OPEN pair (two-leg) ─────────────────────────────────────────────────

    def open_pair(
        self,
        *,
        pair_key:      str,
        y_symbol:      str,
        x_symbol:      str,
        side:          Side,
        sizing:        SizingResult,
        beta:          float,
        alpha:         float,
        spread_now:    float,
        z_now:         float,
        bar_close_iso: str,
    ) -> Optional[PairState]:
        """
        Open both legs of a pair. Returns the persisted PairState on full
        success; returns None on any failure (including partial fill that we
        rolled back).
        """
        if side not in (Side.LONG, Side.SHORT):
            raise ValueError(f"open_pair: side must be LONG/SHORT, got {side}")

        magic = self.magic_for_pair[pair_key]
        # In a LONG-spread position:  long y,  short x   →  y_side = LONG, x_side = SHORT
        # In a SHORT-spread position: short y, long x   →  y_side = SHORT, x_side = LONG
        y_side = Side.LONG  if side == Side.LONG else Side.SHORT
        x_side = Side.SHORT if side == Side.LONG else Side.LONG

        y_cfg = self.resolve_symbol(y_symbol)
        x_cfg = self.resolve_symbol(x_symbol)

        # Pre-snap prices
        y_price = self._price_for_buy(y_cfg.name)  if y_side == Side.LONG else self._price_for_sell(y_cfg.name)
        x_price = self._price_for_buy(x_cfg.name)  if x_side == Side.LONG else self._price_for_sell(x_cfg.name)

        self.logger.event(
            "pair_open_attempt",
            pair_key      = pair_key,
            side          = side.value,
            y_symbol      = y_cfg.name,
            x_symbol      = x_cfg.name,
            y_lots        = sizing.lots_y,
            x_lots        = sizing.lots_x,
            y_side        = y_side.value,
            x_side        = x_side.value,
            y_price       = y_price,
            x_price       = x_price,
            beta          = beta,
            alpha         = alpha,
            spread_now    = spread_now,
            z_now         = z_now,
            sizing_warning = sizing.warning,
            magic         = magic,
            dry_run       = self.dry_run,
        )

        if self.dry_run:
            self.logger.event(
                "pair_open_dryrun",
                pair_key = pair_key,
                detail   = f"would BUY/SELL {y_cfg.name} {sizing.lots_y} / {x_cfg.name} {sizing.lots_x}",
            )
            return None

        # 1) Submit y-leg ────────────────────────────────────────────────────
        y_req = _market_order_request(
            symbol  = y_cfg.name,
            side    = y_side,
            volume  = sizing.lots_y,
            price   = y_price,
            magic   = magic,
            comment = f"{COMMENT_PREFIX}:{pair_key}:y",
        )
        t0 = time.monotonic()
        y_result = mt5.order_send(y_req)
        self.logger.latency(
            "pair_y_send_done", started_at=t0,
            pair_key=pair_key,
            retcode=getattr(y_result, "retcode", None),
            comment=getattr(y_result, "comment", None),
            deal=getattr(y_result, "deal", None),
            order=getattr(y_result, "order", None),
        )

        if y_result is None or y_result.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.event(
                "pair_open_failed",
                pair_key = pair_key,
                stage    = "y_leg",
                retcode  = getattr(y_result, "retcode", None),
                comment  = getattr(y_result, "comment", None),
                last_error = str(mt5.last_error()),
            )
            self._notify("open_failed",
                         f"{pair_key} y-leg rejected: {getattr(y_result, 'comment', 'unknown')}")
            return None

        y_ticket = int(y_result.order or 0)
        y_fill_price = float(getattr(y_result, "price", y_price))

        # 2) Submit x-leg ────────────────────────────────────────────────────
        x_req = _market_order_request(
            symbol  = x_cfg.name,
            side    = x_side,
            volume  = sizing.lots_x,
            price   = x_price,
            magic   = magic,
            comment = f"{COMMENT_PREFIX}:{pair_key}:x",
        )
        t0 = time.monotonic()
        x_result = mt5.order_send(x_req)
        self.logger.latency(
            "pair_x_send_done", started_at=t0,
            pair_key=pair_key,
            retcode=getattr(x_result, "retcode", None),
            comment=getattr(x_result, "comment", None),
            deal=getattr(x_result, "deal", None),
            order=getattr(x_result, "order", None),
        )

        if x_result is None or x_result.retcode != mt5.TRADE_RETCODE_DONE:
            # 🚨 PARTIAL FILL: y is open but x failed. Close y immediately.
            self.logger.event(
                "partial_fill_emergency",
                pair_key  = pair_key,
                stage     = "x_leg_rejected",
                y_ticket  = y_ticket,
                x_retcode = getattr(x_result, "retcode", None),
                x_comment = getattr(x_result, "comment", None),
                action    = "closing_y_immediately",
            )
            self._notify("EMERGENCY",
                         f"{pair_key} x-leg rejected; closing y-leg #{y_ticket} immediately.")
            self._emergency_close_by_ticket(y_ticket, y_cfg.name, magic, pair_key, leg="y")
            return None

        x_ticket = int(x_result.order or 0)
        x_fill_price = float(getattr(x_result, "price", x_price))

        # 3) Persist ─────────────────────────────────────────────────────────
        state = PairState(
            pair_key         = pair_key,
            side             = side,
            y_symbol         = y_cfg.name,
            x_symbol         = x_cfg.name,
            y_ticket         = y_ticket,
            x_ticket         = x_ticket,
            y_volume         = sizing.lots_y,
            x_volume         = sizing.lots_x,
            y_entry_price    = y_fill_price,
            x_entry_price    = x_fill_price,
            beta_at_open     = beta,
            alpha_at_open    = alpha,
            spread_at_open   = spread_now,
            z_at_open        = z_now,
            opened_at        = datetime.now(timezone.utc).isoformat(timespec="seconds"),
            opened_bar       = bar_close_iso,
            bars_in_position = 0,
        )
        self.state_store.add(state)

        self.logger.event(
            "pair_open_success",
            pair_key  = pair_key,
            side      = side.value,
            y_ticket  = y_ticket,
            x_ticket  = x_ticket,
            y_fill    = y_fill_price,
            x_fill    = x_fill_price,
            y_lots    = sizing.lots_y,
            x_lots    = sizing.lots_x,
            beta      = beta,
            spread_now = spread_now,
            z_now     = z_now,
        )
        self._notify("OPEN",
                     f"{pair_key} {side.value.upper()}  "
                     f"y={y_cfg.name} {sizing.lots_y}@{y_fill_price:.5f}  "
                     f"x={x_cfg.name} {sizing.lots_x}@{x_fill_price:.5f}  "
                     f"(z={z_now:+.2f})")
        return state

    # ── CLOSE pair (two-leg) ────────────────────────────────────────────────

    def close_pair(
        self,
        state:      PairState,
        reason:     str,
        spread_now: float,
        z_now:      float,
    ) -> bool:
        """
        Close both legs of an existing pair position. Returns True on full
        success. On partial failure, the state is updated to reflect what's
        still open and the operator is notified.
        """
        self.logger.event(
            "pair_close_attempt",
            pair_key   = state.pair_key,
            reason     = reason,
            side       = state.side.value,
            y_ticket   = state.y_ticket,
            x_ticket   = state.x_ticket,
            spread_now = spread_now,
            z_now      = z_now,
            dry_run    = self.dry_run,
        )

        if self.dry_run:
            self.logger.event(
                "pair_close_dryrun",
                pair_key = state.pair_key,
                detail   = f"would close {state.y_symbol}#{state.y_ticket} + {state.x_symbol}#{state.x_ticket}",
            )
            return False

        ok_y = self._close_one_leg(state.y_ticket, state.y_symbol, state.pair_key, leg="y")
        ok_x = self._close_one_leg(state.x_ticket, state.x_symbol, state.pair_key, leg="x")

        if ok_y and ok_x:
            self.state_store.remove(state.pair_key)
            self.logger.event(
                "pair_close_success",
                pair_key   = state.pair_key,
                reason     = reason,
                y_ticket   = state.y_ticket,
                x_ticket   = state.x_ticket,
                spread_at_open = state.spread_at_open,
                spread_now     = spread_now,
                spread_pnl_log = (spread_now - state.spread_at_open) * (1 if state.side == Side.LONG else -1),
                bars_in_position = state.bars_in_position,
            )
            self._notify("CLOSE",
                         f"{state.pair_key} closed ({reason}) "
                         f"z={z_now:+.2f}  bars={state.bars_in_position}")
            return True

        self.logger.event(
            "pair_close_failed",
            pair_key = state.pair_key,
            reason   = reason,
            ok_y     = ok_y,
            ok_x     = ok_x,
            note     = "state NOT removed; will retry next cycle",
        )
        self._notify("CLOSE_PARTIAL",
                     f"{state.pair_key} close failed: y_ok={ok_y} x_ok={ok_x} — retry next cycle.")
        return False

    # ── close helpers ───────────────────────────────────────────────────────

    def _find_position(self, ticket: int, symbol: str):
        """Look up a position by ticket. Returns None if it's gone (already closed)."""
        positions = mt5.positions_get(ticket=ticket) or []
        for p in positions:
            if p.ticket == ticket and p.symbol == symbol:
                return p
        return None

    def _close_one_leg(self, ticket: int, symbol: str, pair_key: str, leg: str) -> bool:
        """
        Close one leg by ticket. Returns True on success or if the position
        is already gone (e.g. broker closed it server-side).
        """
        pos = self._find_position(ticket, symbol)
        if pos is None:
            self.logger.event(
                "pair_leg_already_closed",
                pair_key=pair_key, leg=leg, ticket=ticket, symbol=symbol,
            )
            return True

        # Closing direction = opposite of position direction
        close_price = (
            self._price_for_sell(symbol) if pos.type == mt5.POSITION_TYPE_BUY
            else self._price_for_buy(symbol)
        )
        req = _close_position_request(position=pos, price=close_price)

        t0 = time.monotonic()
        result = mt5.order_send(req)
        self.logger.latency(
            "pair_leg_close_done", started_at=t0,
            pair_key=pair_key, leg=leg,
            ticket=ticket, retcode=getattr(result, "retcode", None),
            comment=getattr(result, "comment", None),
        )
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.event(
                "pair_leg_close_rejected",
                pair_key=pair_key, leg=leg, ticket=ticket,
                retcode=getattr(result, "retcode", None),
                comment=getattr(result, "comment", None),
                last_error=str(mt5.last_error()),
            )
            return False
        return True

    def _emergency_close_by_ticket(
        self, ticket: int, symbol: str, magic: int, pair_key: str, leg: str,
    ) -> None:
        """
        After partial fill: find the open position by ticket and close it.
        We KEEP retrying (with short sleep) for a small number of attempts
        because leaving an unhedged leg open is unacceptable.
        """
        for attempt in range(1, 4):
            ok = self._close_one_leg(ticket, symbol, pair_key, leg=leg)
            if ok:
                self.logger.event(
                    "partial_fill_recovery_success",
                    pair_key=pair_key, leg=leg, ticket=ticket, attempt=attempt,
                )
                return
            time.sleep(1.0)
        # If we get here, operator MUST intervene.
        self.logger.event(
            "partial_fill_recovery_failed",
            pair_key=pair_key, leg=leg, ticket=ticket,
            attempts=3,
            action_required="MANUAL_CLOSE_REQUIRED",
        )
        self._notify("CRITICAL",
                     f"{pair_key} leg {leg} #{ticket} could NOT be auto-closed after 3 tries. "
                     f"MANUAL INTERVENTION REQUIRED.")
