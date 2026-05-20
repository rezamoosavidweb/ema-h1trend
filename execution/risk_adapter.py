"""
Lot sizing from a fixed-percentage risk budget.

Same math as the original `calc_volume()` in run_ob_xauusd.py, lifted into a
class so it can be unit-tested in isolation and so the size of one losing
trade is documented in one place.
"""

from __future__ import annotations

import math
from typing import Optional

import MetaTrader5 as mt5

from .symbol_config import SymbolConfig


class RiskAdapter:
    """
    Lot calculator. One instance per bot; pass `risk_per_trade` at construction
    time so the policy is visible in the bot's startup config.
    """

    def __init__(self, risk_per_trade: float = 0.01) -> None:
        if not (0 < risk_per_trade < 1):
            raise ValueError(f"risk_per_trade must be in (0,1); got {risk_per_trade}")
        self.risk_per_trade = risk_per_trade

    def calc_volume(
        self,
        cfg: SymbolConfig,
        side: str,
        entry: float,
        sl: float,
        balance: float,
    ) -> Optional[float]:
        """
        Return a normalized lot size that risks `balance * risk_per_trade`
        on a move from `entry` to `sl`.

        Uses `mt5.order_calc_profit` so the maths is broker-side (handles
        contract size, currency conversion, leverage correctly).

        Fail-safe behaviour: returns `None` when `order_calc_profit` returns
        None or zero (typically during MT5 reconnect / broker instability).
        The caller MUST treat this as "skip the trade" and log a structured
        error -- we deliberately do NOT fall back to `volume_min` because an
        unintended min-lot trade during a flaky broker state is worse than a
        skipped signal.
        """
        otype = mt5.ORDER_TYPE_BUY if side == "buy" else mt5.ORDER_TYPE_SELL
        pnl_one_lot = mt5.order_calc_profit(otype, cfg.name, 1.0, entry, sl)
        if pnl_one_lot is None or abs(pnl_one_lot) < 1e-9:
            return None

        risk_cash = balance * self.risk_per_trade
        raw_vol = risk_cash / abs(pnl_one_lot)
        return self.normalize(cfg, raw_vol)

    @staticmethod
    def normalize(cfg: SymbolConfig, volume: float) -> float:
        """Snap `volume` to broker constraints (step, min, max)."""
        step = cfg.volume_step
        vmin = cfg.volume_min
        vmax = cfg.volume_max
        v = math.floor(volume / step + 1e-12) * step
        return float(max(vmin, min(vmax, v)))
