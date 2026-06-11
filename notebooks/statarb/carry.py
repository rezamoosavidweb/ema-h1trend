"""FX carry / rollover overlay (Phase-4 item 1).

Carry is the return earned just for *holding* an FX position: long the high-rate currency,
short the low-rate one, you collect the interest-rate differential (the broker's "swap" or
"rollover"). For a low-Sharpe relative-value book this can be the larger, more persistent
signal — so we model it and test the joint reversion+carry book.

**Honest data caveat (stated loudly):** there is no swap/rate feed on disk. We use a small,
*illustrative* table of approximate G8 policy rates by period — it captures the big regime
shifts (ZIRP 2010-21, the 2022+ hiking cycle, JPY persistently lowest) but is NOT a substitute
for a live swap feed. Treat the carry numbers as directional, not precise; production needs the
broker's actual swap points. We therefore also report carry *sensitivity* to a global scale.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import BARS_PER_YEAR

# Approximate annual policy rates (%) by era. Keyed by the first year the level applies; we
# forward-fill between keys. Illustrative — see module caveat.
_RATE_TABLE = {
    2010: {"USD": 0.25, "EUR": 0.75, "GBP": 0.50, "JPY": 0.10, "CHF": 0.05, "AUD": 4.50, "NZD": 3.00, "CAD": 1.00},
    2015: {"USD": 0.50, "EUR": 0.05, "GBP": 0.50, "JPY": -0.10, "CHF": -0.75, "AUD": 2.00, "NZD": 2.75, "CAD": 0.75},
    2017: {"USD": 1.75, "EUR": 0.00, "GBP": 0.75, "JPY": -0.10, "CHF": -0.75, "AUD": 1.50, "NZD": 1.75, "CAD": 1.50},
    2020: {"USD": 0.25, "EUR": 0.00, "GBP": 0.10, "JPY": -0.10, "CHF": -0.75, "AUD": 0.10, "NZD": 0.25, "CAD": 0.25},
    2022: {"USD": 4.50, "EUR": 2.50, "GBP": 4.00, "JPY": -0.10, "CHF": 1.50, "AUD": 3.50, "NZD": 5.00, "CAD": 4.25},
    2024: {"USD": 4.75, "EUR": 3.50, "GBP": 4.75, "JPY": 0.25, "CHF": 1.00, "AUD": 4.35, "NZD": 5.00, "CAD": 4.00},
}


def rate_panel(index: pd.DatetimeIndex, currencies: list[str]) -> pd.DataFrame:
    """Per-currency annual rate (%) aligned to `index` via forward-filled era table."""
    eras = sorted(_RATE_TABLE)
    rows = {}
    for ts in index:
        era = max(e for e in eras if e <= ts.year) if ts.year >= eras[0] else eras[0]
        rows[ts] = _RATE_TABLE[era]
    df = pd.DataFrame.from_dict(rows, orient="index")
    return df.reindex(columns=[c for c in currencies if c in df.columns])


def pair_carry_rate(symbols: list[str], index: pd.DatetimeIndex) -> pd.DataFrame:
    """Annual carry (%) of being LONG each pair = rate(base) - rate(quote). Positive => you are
    paid to hold the pair long."""
    ccys = sorted({s[:3] for s in symbols} | {s[3:] for s in symbols})
    rp = rate_panel(index, ccys)
    out = {}
    for s in symbols:
        b, q = s[:3], s[3:]
        if b in rp.columns and q in rp.columns:
            out[s] = rp[b] - rp[q]
    return pd.DataFrame(out, index=index)


def carry_return_panel(px: pd.DataFrame, *, timeframe: str = "FX_H1") -> pd.DataFrame:
    """Per-bar carry return of holding each pair long (annual differential / bars-per-year)."""
    cr = pair_carry_rate(list(px.columns), px.index) / 100.0
    return cr / BARS_PER_YEAR[timeframe]


def carry_signal(px: pd.DataFrame, *, lookback: int = 1) -> pd.DataFrame:
    """Cross-sectional carry signal: sign/strength of each pair's carry differential, demeaned
    and gross-scaled (long positive-carry, short negative-carry, dollar-neutral)."""
    cr = pair_carry_rate(list(px.columns), px.index)
    w = cr.sub(cr.mean(axis=1), axis=0)
    gross = w.abs().sum(axis=1).replace(0, np.nan)
    return w.div(gross, axis=0).fillna(0.0)


def backtest_carry(px: pd.DataFrame, *, cost=None, timeframe: str = "FX_H1",
                   rebalance: int = 24, carry_scale: float = 1.0) -> dict:
    """Backtest a pure cross-sectional carry book (no reversion). Holds the carry-ranked
    portfolio, rebalancing every `rebalance` bars; PnL = spot move + carry accrual. `carry_scale`
    lets the notebook stress the (uncertain) carry magnitude."""
    from .backtest import CostModel
    from .metrics import perf_stats
    cost = cost or CostModel(fee_bps=1.2, slippage_bps=0.0, exec_lag=1)
    w = carry_signal(px)
    if rebalance > 1:
        w = w.iloc[::1].ffill()
        w = w.rolling(rebalance, min_periods=1).mean()
    w_exec = w.shift(cost.exec_lag).fillna(0.0)
    spot = px.pct_change()
    carry = carry_return_panel(px, timeframe=timeframe) * carry_scale
    gross = (w_exec * (spot + carry)).sum(axis=1)
    turnover = w_exec.diff().abs().sum(axis=1).fillna(0.0)
    net = (gross - turnover * cost.cost_per_turn).fillna(0.0)
    return {"returns": net, "equity": (1 + net).cumprod(), "turnover": turnover,
            "stats": perf_stats(net, timeframe=timeframe),
            "carry_panel": carry, "weights": w_exec}


def combine_reversion_carry(reversion_ret: pd.Series, carry_ret: pd.Series, *,
                            w_rev: float = 0.5, timeframe: str = "FX_H1") -> dict:
    """Blend a reversion return stream with a carry return stream (risk is diversified if they
    are uncorrelated). Returns the combined stats + the correlation between the two sleeves."""
    from .metrics import perf_stats
    df = pd.concat([reversion_ret.rename("rev"), carry_ret.rename("carry")], axis=1).dropna()
    combo = w_rev * df["rev"] + (1 - w_rev) * df["carry"]
    return {"returns": combo, "stats": perf_stats(combo, timeframe=timeframe),
            "corr": float(df["rev"].corr(df["carry"]))}
