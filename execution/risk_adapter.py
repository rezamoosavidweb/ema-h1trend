"""
Lot sizing from a fixed-percentage risk budget.

Same math as the original `calc_volume()` in run_ob_xauusd.py, lifted into a
class so it can be unit-tested in isolation and so the size of one losing
trade is documented in one place.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import MetaTrader5 as mt5

from .symbol_config import SymbolConfig


@dataclass(frozen=True)
class VolumeInfo:
    """
    Outcome of normalising a raw volume against broker volume_min/step/max.

    Carries the *original* request value alongside the broker-grid value so
    callers can detect risk skew (raw < min ⇒ effective risk > intended).
    """
    volume:                  float   # broker-grid-snapped value to actually send
    raw_volume:              float   # what the math said before snapping/clamping
    was_clamped_to_min:      bool    # raw_volume < volume_min
    was_clamped_to_max:      bool    # raw_volume > volume_max
    snapped_to_step:         bool    # raw_volume needed step-grid rounding
    volume_min:              float
    volume_max:              float
    volume_step:             float


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

        For diagnostics use `calc_volume_detailed()` which exposes whether the
        raw value was clamped (i.e. effective risk differs from intent).
        """
        info = self.calc_volume_detailed(cfg, side, entry, sl, balance)
        return None if info is None else info.volume

    def calc_volume_detailed(
        self,
        cfg: SymbolConfig,
        side: str,
        entry: float,
        sl: float,
        balance: float,
    ) -> Optional[VolumeInfo]:
        """Same as `calc_volume` but returns the full VolumeInfo for logging."""
        otype = mt5.ORDER_TYPE_BUY if side == "buy" else mt5.ORDER_TYPE_SELL
        pnl_one_lot = mt5.order_calc_profit(otype, cfg.name, 1.0, entry, sl)
        if pnl_one_lot is None or abs(pnl_one_lot) < 1e-9:
            return None

        risk_cash = balance * self.risk_per_trade
        raw_vol = risk_cash / abs(pnl_one_lot)
        return self.normalize_detailed(cfg, raw_vol)

    @staticmethod
    def normalize(cfg: SymbolConfig, volume: float) -> float:
        """Snap `volume` to broker constraints (step, min, max)."""
        return RiskAdapter.normalize_detailed(cfg, volume).volume

    @staticmethod
    def normalize_detailed(cfg: SymbolConfig, volume: float) -> VolumeInfo:
        """
        Snap `volume` to broker constraints AND report what happened.

        Snapping rules:
            * step grid: floor(volume / step) * step
            * clamp:     max(min, min(max, snapped))

        Reporting flags:
            * was_clamped_to_min: raw was below broker min — effective risk
              is now LARGER than the user asked for. Caller should warn.
            * was_clamped_to_max: raw was above broker max — effective risk
              is now SMALLER than the user asked for. Less urgent but worth
              logging since capital deployment is below intent.
            * snapped_to_step: raw was between two step ticks; rounded down.
              Always true for non-trivial requests, mostly noise.
        """
        step = float(cfg.volume_step)
        vmin = float(cfg.volume_min)
        vmax = float(cfg.volume_max)

        snapped = math.floor(volume / step + 1e-12) * step if step > 0 else volume
        snapped_to_step = abs(snapped - volume) > step * 1e-6

        clamped_to_min = snapped < vmin
        clamped_to_max = snapped > vmax
        final = max(vmin, min(vmax, snapped))

        return VolumeInfo(
            volume              = float(round(final, 8)),
            raw_volume          = float(volume),
            was_clamped_to_min  = bool(clamped_to_min),
            was_clamped_to_max  = bool(clamped_to_max),
            snapped_to_step     = bool(snapped_to_step),
            volume_min          = vmin,
            volume_max          = vmax,
            volume_step         = step,
        )
