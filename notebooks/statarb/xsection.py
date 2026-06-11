"""Cross-sectional mean reversion — a second stat-arb family (Section K).

Pair/basket trading picks specific relationships and is fragile when they decay. **Cross-
sectional** reversion sidesteps selection entirely: at each rebalance, rank the *whole*
universe by recent return and bet that extremes revert — go long the biggest losers, short the
biggest winners, dollar-neutral. Breadth replaces selection; there is nothing to "pick" and so
much less to over-fit.

Mechanics (look-ahead-free):
  * signal at the close of bar t = trailing `lookback`-bar return of each asset.
  * cross-sectionally demean (so the book is market-neutral) and rank into quantiles.
  * weight = long bottom quantile / short top quantile, scaled to unit gross.
  * **execute at t+1** (shift), hold `holding` bars, pay cost on turnover.

We test several lookbacks because the reversion horizon is unknown a priori — but we report the
whole curve rather than cherry-picking the best (which would be data-snooping).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import BARS_PER_YEAR, log_returns
from .backtest import CostModel
from .metrics import perf_stats


def xs_weights(px: pd.DataFrame, *, lookback: int = 24, quantile: float = 0.3,
               signal_logpx: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-bar cross-sectional reversion weights. Long the bottom `quantile` of trailing
    returns, short the top `quantile`, demeaned and scaled to unit gross exposure each bar.

    `signal_logpx`: optional alternative *log-price* frame to rank on (e.g. currency-factor
    residual prices). PnL is still realised on the tradable `px`; only the ranking signal
    changes. Defaults to log(px)."""
    src = signal_logpx if signal_logpx is not None else np.log(px)
    signal = src.diff(lookback)                          # trailing return, known at close t
    # rank within each row (cross-section); lower rank = bigger loser = we go long.
    ranks = signal.rank(axis=1, pct=True)
    long_leg = (ranks <= quantile).astype(float)
    short_leg = (ranks >= 1 - quantile).astype(float)
    w = long_leg - short_leg
    # demean (enforce dollar-neutral) then scale to unit gross per bar.
    w = w.sub(w.mean(axis=1), axis=0)
    gross = w.abs().sum(axis=1).replace(0, np.nan)
    return w.div(gross, axis=0).fillna(0.0)


def backtest_xs(px: pd.DataFrame, *, lookback: int = 24, holding: int = 24, quantile: float = 0.3,
                cost: CostModel | None = None, timeframe: str = "H1",
                signal_logpx: pd.DataFrame | None = None) -> dict:
    """Backtest a cross-sectional reversion book. `holding` smooths the target weights over the
    intended holding period (a simple, low-turnover implementation). t+1 execution and turnover
    costs as elsewhere. `signal_logpx` ranks on an alternative log-price frame (e.g. residuals)
    while PnL is realised on `px`. Returns dict with returns / equity / turnover / stats."""
    cost = cost or CostModel()
    r = px.pct_change()
    w = xs_weights(px, lookback=lookback, quantile=quantile, signal_logpx=signal_logpx)
    if holding > 1:
        w = w.rolling(holding, min_periods=1).mean()   # hold/average over the horizon
    w_exec = w.shift(cost.exec_lag).fillna(0.0) * cost.fill_prob

    gross = (w_exec * r).sum(axis=1)
    turnover = w_exec.diff().abs().sum(axis=1).fillna(0.0)
    cost_series = turnover * cost.cost_per_turn
    net = (gross - cost_series).fillna(0.0)
    stats = perf_stats(net, timeframe=timeframe)
    stats_gross = perf_stats(gross.fillna(0.0), timeframe=timeframe)
    return {"returns": net, "gross_returns": gross.fillna(0.0), "equity": (1 + net).cumprod(),
            "turnover": turnover, "weights": w_exec, "stats": stats,
            "gross_sharpe": stats_gross["sharpe"], "lookback": lookback, "holding": holding}


def lookback_scan(px: pd.DataFrame, lookbacks=(6, 12, 24, 48, 96, 168, 336), *,
                  holding: int | None = None, quantile: float = 0.3,
                  cost: CostModel | None = None, timeframe: str = "H1") -> pd.DataFrame:
    """Scan reversion lookbacks (holding defaults to = lookback). Reports gross & net Sharpe
    for each — the whole curve, so the horizon dependence is visible and not cherry-picked."""
    rows = []
    for lb in lookbacks:
        h = holding or lb
        bt = backtest_xs(px, lookback=lb, holding=h, quantile=quantile, cost=cost, timeframe=timeframe)
        rows.append({"lookback": lb, "holding": h, "gross_sharpe": bt["gross_sharpe"],
                     "net_sharpe": bt["stats"]["sharpe"], "net_cagr": bt["stats"]["cagr"],
                     "max_dd": bt["stats"]["max_dd"]})
    return pd.DataFrame(rows)
