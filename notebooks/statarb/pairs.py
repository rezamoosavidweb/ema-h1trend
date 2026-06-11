"""Pair selection — the baseline (correlation) method and a cointegration filter.

This module answers "which pairs do we trade?" two ways:
  * `correlation_pairs`  — the article's method: rank all pairs by correlation of returns,
    take the top-N. Simple and the thing we are critiquing.
  * `cointegration_*`    — the statistically-correct screen: a pair is tradable only if a
    *linear combination of the price levels is stationary* (mean-reverting). High return
    correlation does NOT imply this (two assets can co-move daily yet drift apart forever).

`select_pairs` is the single entry point used by the backtest; it composes a candidate
generator (correlation OR clustering) with optional cointegration / half-life filters, all
estimated on the formation window only.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint

from .spread import ols_hedge_ratio, static_spread, halflife


@dataclass
class Pair:
    a: str
    b: str
    corr: float = np.nan
    beta: float = np.nan
    alpha: float = np.nan
    coint_p: float = np.nan
    half_life: float = np.nan
    cluster: int = -1

    @property
    def key(self) -> tuple[str, str]:
        return (self.a, self.b)

    def __repr__(self):
        return (f"{self.a}-{self.b}(corr={self.corr:.2f},beta={self.beta:.2f},"
                f"p={self.coint_p:.3f},hl={self.half_life:.0f})")


def correlation_matrix(rets: pd.DataFrame) -> pd.DataFrame:
    return rets.corr()


def correlation_pairs(rets: pd.DataFrame, top_n: int = 20,
                      candidates: list[tuple[str, str]] | None = None) -> list[Pair]:
    """Rank pairs by |return correlation|, descending. If `candidates` is given (e.g. only
    within-cluster pairs), rank within that restricted set — this is exactly how the
    clustering extension plugs in."""
    c = rets.corr()
    cand = candidates if candidates is not None else list(combinations(rets.columns, 2))
    scored = [Pair(a=a, b=b, corr=float(c.loc[a, b])) for a, b in cand]
    scored.sort(key=lambda p: abs(p.corr), reverse=True)
    return scored[:top_n]


def engle_granger_p(px_a: pd.Series, px_b: pd.Series) -> float:
    """Engle-Granger cointegration p-value (statsmodels `coint`, regress A on B). Low p =>
    residual is stationary => the spread mean-reverts. We test on prices (levels), not
    returns — cointegration is a level relationship."""
    try:
        _, p, _ = coint(px_a.values, px_b.values)
        return float(p)
    except Exception:
        return np.nan


def adf_resid_p(px_a: pd.Series, px_b: pd.Series) -> tuple[float, float, float]:
    """Manual Engle-Granger: OLS hedge ratio, then ADF on the residual spread. Returns
    (adf_p, beta, alpha). Gives us beta and the stationarity p-value in one shot."""
    beta, alpha = ols_hedge_ratio(np.log(px_a), np.log(px_b))
    resid = static_spread(px_a, px_b, beta, alpha).dropna()
    try:
        adf_p = float(adfuller(resid.values, maxlag=1, autolag=None)[1])
    except Exception:
        adf_p = np.nan
    return adf_p, beta, alpha


def enrich_pairs(pairs: list[Pair], px: pd.DataFrame) -> list[Pair]:
    """Fill beta/alpha/coint_p/half_life for each pair using formation-window prices."""
    for p in pairs:
        adf_p, beta, alpha = adf_resid_p(px[p.a], px[p.b])
        p.beta, p.alpha, p.coint_p = beta, alpha, adf_p
        sp = static_spread(px[p.a], px[p.b], beta, alpha)
        p.half_life = halflife(sp)
    return pairs


def select_pairs(px_form: pd.DataFrame, *, top_n: int = 12,
                 candidates: list[tuple[str, str]] | None = None,
                 coint_max_p: float | None = None,
                 hl_max: float | None = None,
                 hl_min: float = 2.0) -> list[Pair]:
    """End-to-end pair selection on the formation window.

      1. rank candidates by |correlation| (top 4*top_n as a wide funnel).
      2. enrich with hedge ratio, cointegration p-value, half-life.
      3. optionally keep only cointegrated pairs (`coint_max_p`, e.g. 0.05) and pairs whose
         half-life is in a tradable band [hl_min, hl_max].
      4. return the top_n survivors (still ranked by correlation, the baseline ordering).
    """
    from .data import log_returns
    rets = log_returns(px_form)
    funnel = correlation_pairs(rets, top_n=top_n * 4, candidates=candidates)
    funnel = enrich_pairs(funnel, px_form)

    out = []
    for p in funnel:
        if coint_max_p is not None and not (p.coint_p <= coint_max_p):
            continue
        if hl_max is not None and not (hl_min <= p.half_life <= hl_max):
            continue
        out.append(p)
    return out[:top_n]
