"""Performance & risk metrics — hedge-fund-standard, plus significance tests.

Everything is computed from a per-bar net-return series so the same code grades a single
pair, a whole portfolio, or a benchmark. Annualisation uses the bar count per year for the
timeframe (H1 -> 8760), NOT a hand-waved 252.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .data import BARS_PER_YEAR


def drawdown(equity: pd.Series) -> pd.Series:
    """Drawdown series = equity / running peak - 1 (<= 0)."""
    return equity / equity.cummax() - 1.0


def perf_stats(returns: pd.Series, *, timeframe: str = "H1", rf: float = 0.0) -> dict:
    """Full stat block from a per-bar net-return series."""
    r = returns.dropna()
    if len(r) < 2 or r.std() == 0:
        return {k: np.nan for k in ("cagr", "ann_vol", "sharpe", "sortino", "max_dd",
                                    "calmar", "hit_rate", "n_bars", "total_return")}
    ppy = BARS_PER_YEAR[timeframe]
    equity = (1 + r).cumprod()
    total = equity.iloc[-1] - 1.0
    years = len(r) / ppy
    cagr = equity.iloc[-1] ** (1 / years) - 1.0 if years > 0 else np.nan
    ann_vol = r.std() * np.sqrt(ppy)
    sharpe = (r.mean() - rf / ppy) / r.std() * np.sqrt(ppy)
    downside = r[r < 0].std()
    sortino = (r.mean() - rf / ppy) / downside * np.sqrt(ppy) if downside and downside > 0 else np.nan
    dd = drawdown(equity)
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan
    return {
        "total_return": float(total),
        "cagr": float(cagr),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_dd": float(max_dd),
        "calmar": float(calmar),
        "hit_rate": float((r > 0).mean()),
        "n_bars": int(len(r)),
    }


def sharpe_tstat(returns: pd.Series) -> tuple[float, float]:
    """t-stat and two-sided p-value for 'is mean return > 0'. t ≈ Sharpe·sqrt(n_bars).
    A Sharpe of 1.5 on 3 months of H1 is not the same evidence as on 5 years — this makes
    the sample size explicit, the first thing to check before believing a backtest."""
    r = returns.dropna()
    if len(r) < 2 or r.std() == 0:
        return np.nan, np.nan
    t = r.mean() / r.std() * np.sqrt(len(r))
    p = 2 * (1 - stats.t.cdf(abs(t), df=len(r) - 1))
    return float(t), float(p)


def probabilistic_sharpe(returns: pd.Series, *, timeframe: str = "H1",
                         benchmark_sr: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio (Bailey & López de Prado): P(true annualised SR >
    benchmark) given the *non-normality* (skew/kurt) of returns. Crypto spread returns are
    fat-tailed, which inflates a naive Sharpe; PSR discounts that."""
    r = returns.dropna()
    n = len(r)
    if n < 10 or r.std() == 0:
        return np.nan
    ppy = BARS_PER_YEAR[timeframe]
    sr = r.mean() / r.std()                        # per-bar Sharpe
    sr_bench = benchmark_sr / np.sqrt(ppy)
    sk = stats.skew(r)
    ku = stats.kurtosis(r, fisher=False)
    num = (sr - sr_bench) * np.sqrt(n - 1)
    den = np.sqrt(1 - sk * sr + (ku - 1) / 4 * sr ** 2)
    return float(stats.norm.cdf(num / den)) if den > 0 else np.nan


def summary_table(results: dict, *, timeframe: str = "H1") -> pd.DataFrame:
    """Compare strategies side-by-side. `results` maps label -> per-bar return series."""
    rows = {}
    for label, ret in results.items():
        s = perf_stats(ret, timeframe=timeframe)
        t, p = sharpe_tstat(ret)
        s["t_stat"] = t
        s["p_value"] = p
        s["psr"] = probabilistic_sharpe(ret, timeframe=timeframe)
        rows[label] = s
    cols = ["cagr", "ann_vol", "sharpe", "sortino", "max_dd", "calmar",
            "hit_rate", "t_stat", "p_value", "psr", "n_bars"]
    return pd.DataFrame(rows).T[cols]


def pnl_attribution(pair_returns: pd.DataFrame, *, timeframe: str = "H1") -> pd.DataFrame:
    """Per-pair contribution: total return, Sharpe, and share of gross PnL. Shows whether a
    book's performance is broad-based or one or two pairs carrying everything (a fragility
    flag)."""
    rows = {}
    for col in pair_returns.columns:
        s = perf_stats(pair_returns[col], timeframe=timeframe)
        rows[col] = {"total_return": s["total_return"], "sharpe": s["sharpe"],
                     "max_dd": s["max_dd"]}
    df = pd.DataFrame(rows).T
    tot = df["total_return"].clip(lower=0).sum()
    df["pnl_share"] = df["total_return"].clip(lower=0) / tot if tot > 0 else np.nan
    return df.sort_values("total_return", ascending=False)


def exposure_stats(turnover: pd.Series, *, timeframe: str = "H1") -> dict:
    """Turnover / activity stats — how hard the book trades (cost & capacity proxy)."""
    ppy = BARS_PER_YEAR[timeframe]
    daily = ppy / 365
    return {
        "ann_turnover": float(turnover.sum() / (len(turnover) / ppy)),
        "avg_daily_turnover": float(turnover.mean() * daily),
        "active_frac": float((turnover.abs() > 0).mean()),
    }
