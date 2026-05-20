"""
Fallback cascade: LIMIT -> MARKET, gated by retcode and re-validation.

Stage flow (per signal):
    Stage 1 (LIMIT)
        * geometry valid? (entry on the far side of price by >= stops_level)
        * slippage <= SLIPPAGE_LIMIT_THRESHOLD?
        -> if yes, place LIMIT and STOP
        -> if no, fall through to Stage 2

    Stage 2 (MARKET)
        * re-fetch tick (slippage can change in milliseconds)
        * slippage <  SLIPPAGE_MAX_POINTS?
        * SL/TP outside stops_level vs current price?
        -> if yes, place MARKET with TP recalculated from fill price (RR preserved)
        -> if no, Stage 3

    Stage 3 (REJECT)
        * log skip, notify, return cleanly

Retcode handling
    If Stage 1 LIMIT comes back from the broker with a "soft" failure (price
    moved, invalid_price, invalid_expiration, requote), we automatically
    advance to Stage 2 -- the old code stopped here and dropped the trade.
    Hard failures (autotrading off, invalid stops, market closed) abort.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import MetaTrader5 as mt5

from .broker_validator import BrokerValidator
from .order_factory import OrderFactory
from .pending_manager import PendingOrderManager
from .risk_adapter import RiskAdapter
from .structured_logger import StructuredLogger
from .symbol_config import SymbolConfig


# Retcodes that mean "this exact request did not work, but a different one
# might". These cascade to Stage 2 instead of aborting the trade.
SOFT_RETRY_RETCODES = {
    10004,  # TRADE_RETCODE_REQUOTE
    10015,  # TRADE_RETCODE_INVALID_PRICE
    10021,  # TRADE_RETCODE_PRICE_OFF
    10022,  # TRADE_RETCODE_INVALID_EXPIRATION  <- the bug that started all this
    10024,  # TRADE_RETCODE_TOO_MANY_REQUESTS
    10031,  # TRADE_RETCODE_CONNECTION
}

# Retcodes that abort the trade -- no point falling back to market either.
HARD_ABORT_RETCODES = {
    10018,  # TRADE_RETCODE_MARKET_CLOSED
    10027,  # TRADE_RETCODE_CLIENT_DISABLES_AT
    10026,  # TRADE_RETCODE_TRADE_DISABLED
    10017,  # TRADE_RETCODE_TRADE_NOT_ALLOWED
}


@dataclass
class FallbackResult:
    """What the cascade ended up doing. Useful for telegram / metrics."""
    placed: bool
    stage: str              # "limit", "market", or "rejected"
    ticket: Optional[int]
    reason: str = ""        # populated only when placed=False
    fields: dict = None     # arbitrary extras for the executor to log


class FallbackEngine:
    """
    Wires together validator + pending manager + market sender into one
    "place this signal" call. Called by ExecutionEngine; not used directly.
    """

    def __init__(
        self,
        cfg: SymbolConfig,
        validator: BrokerValidator,
        factory: OrderFactory,
        pendings: PendingOrderManager,
        risk: RiskAdapter,
        logger: StructuredLogger,
        slippage_limit_threshold: float = 4.0,
        slippage_max_points: float = 6.0,
        risk_reward: float = 2.0,
    ) -> None:
        self.cfg = cfg
        self.validator = validator
        self.factory = factory
        self.pendings = pendings
        self.risk = risk
        self.logger = logger
        self.slippage_limit_threshold = slippage_limit_threshold
        self.slippage_max_points = slippage_max_points
        self.risk_reward = risk_reward

    # ── helpers ──────────────────────────────────────────────────────────────-

    def _current_slippage(self, side: str, ob_entry: float) -> tuple[float, float]:
        """
        Return `(market_price, slippage_in_price_units)` for the given side.
        Slippage is positive when the market has moved AWAY from the OB.
        """
        tick = mt5.symbol_info_tick(self.cfg.name)
        if tick is None:
            return float("nan"), float("inf")
        if side == "buy":
            return tick.ask, max(0.0, tick.ask - ob_entry)
        return tick.bid, max(0.0, ob_entry - tick.bid)

    # ── stages ───────────────────────────────────────────────────────────────-

    def execute(
        self,
        side: str,
        ob_entry: float,
        sl: float,
        tp: float,
        ob_key: tuple,
        balance: float,
    ) -> FallbackResult:
        """
        Run the LIMIT -> MARKET -> REJECT cascade. Returns FallbackResult.

        All exceptions from MT5 are converted into rejected results -- the
        executor will surface them via the structured logger.
        """
        t_start = time.monotonic()

        # Initial sizing from the IDEAL OB entry (same as original code).
        # RiskAdapter returns None when mt5.order_calc_profit fails (typically
        # during broker reconnect). We DELIBERATELY refuse to fall back to
        # volume_min -- an accidental min-lot trade during flaky broker state
        # is worse than skipping the signal.
        volume = self.risk.calc_volume(self.cfg, side, ob_entry, sl, balance)
        if volume is None:
            self.logger.error(
                "risk_calc_failed",
                stage="initial_sizing",
                side=side, ob_entry=ob_entry, sl=sl, balance=balance,
                last_error=str(mt5.last_error()),
            )
            return FallbackResult(False, "rejected", None, "risk_calc_failed")
        if volume < self.cfg.volume_min:
            self.logger.event("skip", reason="volume_too_small",
                              volume=volume, min=self.cfg.volume_min)
            return FallbackResult(False, "rejected", None, "volume_too_small")

        market_price, slippage_pts_price = self._current_slippage(side, ob_entry)
        slippage_pts = slippage_pts_price / self.cfg.point if self.cfg.point else 0.0

        # ── Stage 3 pre-check: hard slippage cap ─────────────────────────────-
        if slippage_pts >= self.slippage_max_points:
            self.logger.event("skip",
                              reason="slippage_exceeded",
                              slippage_pts=round(slippage_pts, 2),
                              max_pts=self.slippage_max_points,
                              market_price=market_price, ob_entry=ob_entry)
            return FallbackResult(False, "rejected", None, "slippage_exceeded",
                                  {"slippage_pts": slippage_pts})

        # ── Stage 1: LIMIT ───────────────────────────────────────────────────-
        if slippage_pts <= self.slippage_limit_threshold:
            geom = self.validator.validate_pending_geometry(
                self.cfg, side, ob_entry, sl, tp,
            )
            if geom.ok:
                ticket = self.pendings.place_limit(side, volume, ob_entry, sl, tp, ob_key)
                if ticket is not None:
                    self.logger.latency("execution_latency",
                                        started_at=t_start, stage="limit",
                                        ticket=ticket)
                    return FallbackResult(True, "limit", ticket, fields={
                        "volume": volume, "slippage_pts": slippage_pts,
                    })
                # place_limit logged the failure -- fall through to market
                self.logger.event("fallback_market_execution",
                                  reason="limit_send_failed",
                                  slippage_pts=round(slippage_pts, 2))
            else:
                self.logger.event("fallback_market_execution",
                                  reason=f"limit_geometry_{geom.code}",
                                  detail=geom.detail,
                                  **(geom.fields or {}))

        # ── Stage 2: MARKET (re-validate first) ──────────────────────────────-
        # Re-fetch tick because Stage 1 took non-zero time.
        market_price, slippage_pts_price = self._current_slippage(side, ob_entry)
        slippage_pts = slippage_pts_price / self.cfg.point if self.cfg.point else 0.0
        if slippage_pts >= self.slippage_max_points:
            self.logger.event("skip", reason="slippage_exceeded_after_limit",
                              slippage_pts=round(slippage_pts, 2),
                              max_pts=self.slippage_max_points)
            return FallbackResult(False, "rejected", None, "slippage_exceeded_after_limit",
                                  {"slippage_pts": slippage_pts})

        # Recompute TP from the actual fill price to preserve RR
        risk_from_fill = abs(market_price - sl)
        if side == "buy":
            tp_final = round(market_price + risk_from_fill * self.risk_reward, self.cfg.digits)
        else:
            tp_final = round(market_price - risk_from_fill * self.risk_reward, self.cfg.digits)

        # Resize using the actual market price (slippage-adjusted risk).
        # Same fail-safe contract as the initial sizing: None -> skip, never
        # fall back to min lot.
        volume_final = self.risk.calc_volume(self.cfg, side, market_price, sl, balance)
        if volume_final is None:
            self.logger.error(
                "risk_calc_failed",
                stage="slippage_adjusted_sizing",
                side=side, market_price=market_price, sl=sl, balance=balance,
                last_error=str(mt5.last_error()),
            )
            return FallbackResult(False, "rejected", None, "risk_calc_failed")
        if volume_final < self.cfg.volume_min:
            self.logger.event("skip", reason="volume_too_small_after_slippage",
                              volume=volume_final, min=self.cfg.volume_min)
            return FallbackResult(False, "rejected", None, "volume_too_small_after_slippage")

        market_stops = self.validator.validate_market_stops(self.cfg, side, sl, tp_final)
        if not market_stops.ok:
            self.logger.event("broker_validation_failed",
                              stage="market", code=market_stops.code,
                              detail=market_stops.detail,
                              **(market_stops.fields or {}))
            return FallbackResult(False, "rejected", None, market_stops.code,
                                  market_stops.fields)

        self.logger.event(
            "slippage_adjusted",
            direction="BUY" if side == "buy" else "SELL",
            ob_entry=ob_entry,
            market_price=round(market_price, self.cfg.digits),
            slippage_pts=round(slippage_pts, 2),
            sl_unchanged=sl,
            volume_original=volume,
            volume_adjusted=volume_final,
            tp_original=tp,
            tp_adjusted=tp_final,
        )

        req = self.factory.build_market(self.cfg, side, volume_final,
                                        market_price, sl, tp_final)
        result = mt5.order_send(req.request)

        if result is None:
            self.logger.error("market_order_error",
                              reason="order_send_returned_none",
                              last_error=str(mt5.last_error()))
            return FallbackResult(False, "rejected", None, "order_send_none")

        if result.retcode in HARD_ABORT_RETCODES:
            self.logger.event("market_order_aborted",
                              retcode=result.retcode, comment=result.comment)
            return FallbackResult(False, "rejected", None,
                                  f"hard_retcode_{result.retcode}",
                                  {"retcode": result.retcode})

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.event("market_order_failed",
                              retcode=result.retcode, comment=result.comment,
                              side=side, price=market_price,
                              sl=sl, tp=tp_final)
            return FallbackResult(False, "rejected", None,
                                  f"retcode_{result.retcode}",
                                  {"retcode": result.retcode, "comment": result.comment})

        ticket = result.order
        self.logger.latency("execution_latency",
                            started_at=t_start, stage="market", ticket=ticket)
        self.logger.event(
            "market_order_placed",
            ticket=ticket,
            side=side,
            volume=volume_final,
            price=market_price,
            sl=sl, tp=tp_final,
            slippage_pts=round(slippage_pts, 2),
            ob_entry=ob_entry,
        )
        return FallbackResult(True, "market", ticket, fields={
            "volume": volume_final,
            "slippage_pts": slippage_pts,
            "tp_final": tp_final,
            "market_price": market_price,
        })
