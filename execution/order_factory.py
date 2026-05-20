"""
Pure builders for `mt5.order_send` request dicts.

Why a factory?
    The original code had `send_market_order` and `send_limit_order` each
    doing snap+fill-mode+SL/TP-shift+request-build in one go (~80 lines each).
    That is impossible to unit-test and easy to subtly break (e.g. the
    expiration bug). The factory splits these concerns:

        * snap()            -- round a price to the broker tick grid
        * pick_filling()    -- choose the best supported filling mode
        * build_market()    -- assemble the market-order dict
        * build_limit()     -- assemble the LIMIT order dict (GTC, no expiry)
        * build_cancel()    -- assemble the pending-cancel dict
        * build_modify()    -- assemble the SL/TP modify dict

CRITICAL CHANGE FROM PREVIOUS VERSION
=====================================
We no longer set `type_time = ORDER_TIME_SPECIFIED` with a Unix timestamp,
because Errante (and many other brokers) reads `expiration` as broker-local
seconds-since-epoch -- a UTC value comes out 3 hours in the past on a UTC+3
broker, triggering retcode 10022 "Invalid expiration".

LIMIT orders are now placed with `ORDER_TIME_GTC` and the bot itself cancels
them after `stale_after_seconds` via PendingOrderManager. This is broker-
timezone-independent and avoids the entire class of expiration bugs.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import MetaTrader5 as mt5

from .symbol_config import SymbolConfig


# Default deviation (slippage tolerance) for market orders, in points.
# Overridable via the MT5_DEVIATION_POINTS env var; matches the previous bot.
DEFAULT_DEVIATION_POINTS = int(os.environ.get("MT5_DEVIATION_POINTS", "50"))


@dataclass
class OrderRequest:
    """
    Thin wrapper around the raw dict so the calling code can attach metadata
    (e.g. magic, comment) without mutating the dict in-place.
    """
    request: dict
    side: str            # "buy" or "sell"
    kind: str            # "market" / "buy_limit" / "sell_limit" / "cancel" / "modify"


class OrderFactory:
    """Builds order_send dicts for one bot (one MAGIC, one comment family)."""

    def __init__(
        self,
        magic: int,
        comment_prefix: str = "ob_reaction",
        deviation_points: int = DEFAULT_DEVIATION_POINTS,
    ) -> None:
        self.magic = magic
        self.comment_prefix = comment_prefix
        self.deviation_points = deviation_points

    # ── price helpers ────────────────────────────────────────────────────────-

    @staticmethod
    def snap(price: float, cfg: SymbolConfig, mode: str = "nearest") -> float:
        """
        Round `price` onto the broker tick grid.

        mode:
            "nearest" -- closest tick
            "up"      -- ceil (used for TP on BUY, SL on SELL: keeps risk
                         conservative because rounding moves price AWAY from us)
            "down"    -- floor (used for SL on BUY, TP on SELL)

        The 1e-12 epsilon prevents float-rounding flips when a price is
        already exactly on a tick.
        """
        tick = cfg.tick_size or cfg.point
        x = price / tick

        if mode == "up":
            v = math.ceil(x - 1e-12) * tick
        elif mode == "down":
            v = math.floor(x + 1e-12) * tick
        else:
            v = round(x) * tick

        return round(v, cfg.digits)

    @staticmethod
    def pick_filling(cfg: SymbolConfig) -> int:
        """
        Pick the best filling mode the broker supports for this symbol.

        Priority:
            IOC  -- partial fills allowed, fastest
            FOK  -- all-or-nothing
            RETURN -- fallback for symbols that expose no flags
        """
        ioc = getattr(mt5, "SYMBOL_FILLING_IOC", 2)
        fok = getattr(mt5, "SYMBOL_FILLING_FOK", 1)
        mask = cfg.filling_mode_mask
        if mask & ioc:
            return mt5.ORDER_FILLING_IOC
        if mask & fok:
            return mt5.ORDER_FILLING_FOK
        return mt5.ORDER_FILLING_RETURN

    # ── builders ─────────────────────────────────────────────────────────────-

    def build_market(
        self,
        cfg: SymbolConfig,
        side: str,
        volume: float,
        price: float,
        sl: float,
        tp: float,
        comment_suffix: str = "_market",
    ) -> OrderRequest:
        """
        Build a TRADE_ACTION_DEAL market request. Caller supplies the *intended*
        SL/TP -- the factory snaps them to the grid and the validator should
        have already enforced stops_level distance.
        """
        otype = mt5.ORDER_TYPE_BUY if side == "buy" else mt5.ORDER_TYPE_SELL
        sl_snapped = self.snap(sl, cfg, "down" if side == "buy" else "up")
        tp_snapped = self.snap(tp, cfg, "up"   if side == "buy" else "down")
        req = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       cfg.name,
            "volume":       volume,
            "type":         otype,
            "price":        price,
            "sl":           sl_snapped,
            "tp":           tp_snapped,
            "magic":        self.magic,
            "comment":      self.comment_prefix + comment_suffix,
            "type_filling": self.pick_filling(cfg),
            "deviation":    self.deviation_points,
        }
        return OrderRequest(request=req, side=side, kind="market")

    def build_limit(
        self,
        cfg: SymbolConfig,
        side: str,
        volume: float,
        limit_price: float,
        sl: float,
        tp: float,
        comment_suffix: str = "_limit",
    ) -> OrderRequest:
        """
        Build a TRADE_ACTION_PENDING request with ORDER_TIME_GTC.

        We DELIBERATELY do not set `expiration` (the broker-local-time field
        that previously caused retcode 10022). The PendingOrderManager cancels
        stale orders from the client side instead.
        """
        otype = mt5.ORDER_TYPE_BUY_LIMIT if side == "buy" else mt5.ORDER_TYPE_SELL_LIMIT
        sl_snapped    = self.snap(sl,          cfg, "down" if side == "buy" else "up")
        tp_snapped    = self.snap(tp,          cfg, "up"   if side == "buy" else "down")
        price_snapped = self.snap(limit_price, cfg, "nearest")
        req = {
            "action":       mt5.TRADE_ACTION_PENDING,
            "symbol":       cfg.name,
            "volume":       volume,
            "type":         otype,
            "price":        price_snapped,
            "sl":           sl_snapped,
            "tp":           tp_snapped,
            "magic":        self.magic,
            "comment":      self.comment_prefix + comment_suffix,
            "type_filling": self.pick_filling(cfg),
            "type_time":    mt5.ORDER_TIME_GTC,
            # NOTE: NO `expiration` field. Lifecycle is managed in PendingOrderManager.
        }
        kind = "buy_limit" if side == "buy" else "sell_limit"
        return OrderRequest(request=req, side=side, kind=kind)

    @staticmethod
    def build_cancel(ticket: int) -> OrderRequest:
        """Build a TRADE_ACTION_REMOVE dict to cancel a pending order."""
        req = {"action": mt5.TRADE_ACTION_REMOVE, "order": int(ticket)}
        return OrderRequest(request=req, side="", kind="cancel")

    @staticmethod
    def build_modify(ticket: int, price: float, sl: float, tp: float) -> OrderRequest:
        """Build a TRADE_ACTION_MODIFY dict for an existing pending order."""
        req = {
            "action": mt5.TRADE_ACTION_MODIFY,
            "order":  int(ticket),
            "price":  price,
            "sl":     sl,
            "tp":     tp,
        }
        return OrderRequest(request=req, side="", kind="modify")
