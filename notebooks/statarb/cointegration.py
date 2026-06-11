"""Johansen cointegration — multi-asset baskets beyond pairwise Engle-Granger.

Engle-Granger (in `pairs.py`) tests one pair at a time and needs you to pick the dependent
variable. **Johansen** tests a whole group jointly and returns *all* independent cointegrating
relationships at once, plus the cointegrating vectors (the basket weights that make a
stationary spread). This lets us ask: *can 3-5 assets form a more stable spread than 2?*

The trace test ranks how many cointegrating relationships `r` exist. We take the top
eigenvector (the most-stationary combination) as the basket weight vector, build the basket
spread on log-prices, and trade it with the same z-score engine as the pairs (so the
comparison is apples-to-apples). All estimation is formation-window only.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from .spread import rolling_zscore, halflife


# Johansen trace critical values are indexed [r][90%,95%,99%]; statsmodels returns them.
_CONF = {"90%": 0, "95%": 1, "99%": 2}


@dataclass
class Basket:
    symbols: list[str]
    weights: np.ndarray           # cointegrating vector (first eigenvector), normalised
    rank: int                     # number of cointegrating relations (trace test, 95%)
    trace_stat: float             # trace statistic for r=0
    trace_crit95: float
    eigenvalue: float
    half_life: float

    @property
    def weight_series(self) -> pd.Series:
        return pd.Series(self.weights, index=self.symbols)

    def __repr__(self):
        w = ", ".join(f"{s}:{v:+.2f}" for s, v in zip(self.symbols, self.weights))
        return f"Basket[{w}] rank={self.rank} hl={self.half_life:.0f}"


def johansen(prices: pd.DataFrame, *, det_order: int = 0, k_ar_diff: int = 1) -> dict:
    """Run the Johansen test on log-prices. Returns eigenvalues, eigenvectors, trace stats,
    critical values, and the cointegration rank at 95% (count of trace stats above crit)."""
    logp = np.log(prices.dropna())
    res = coint_johansen(logp.values, det_order, k_ar_diff)
    trace = res.lr1
    crit = res.cvt                                  # columns: 90/95/99%
    rank = int((trace > crit[:, _CONF["95%"]]).sum())
    return {
        "eigenvalues": res.eig,
        "eigenvectors": res.evec,                   # columns are vectors, sorted by eig desc
        "trace_stat": trace,
        "crit": pd.DataFrame(crit, columns=["90%", "95%", "99%"]),
        "rank": rank,
        "symbols": list(prices.columns),
    }


def make_basket(prices: pd.DataFrame, *, det_order: int = 0, k_ar_diff: int = 1) -> Basket | None:
    """Fit Johansen on a group and return the leading cointegrating Basket (or None if the
    group is not cointegrated at 95%). Weights are the top eigenvector, scaled so the largest
    absolute weight is 1 (interpretable: that leg is the unit, others hedge it)."""
    jo = johansen(prices, det_order=det_order, k_ar_diff=k_ar_diff)
    if jo["rank"] < 1:
        return None
    vec = jo["eigenvectors"][:, 0]
    vec = vec / np.abs(vec).max()
    spread = np.log(prices).mul(pd.Series(vec, index=prices.columns), axis=1).sum(axis=1)
    return Basket(symbols=list(prices.columns), weights=vec, rank=jo["rank"],
                  trace_stat=float(jo["trace_stat"][0]),
                  trace_crit95=float(jo["crit"].iloc[0]["95%"]),
                  eigenvalue=float(jo["eigenvalues"][0]), half_life=halflife(spread))


def search_baskets(prices: pd.DataFrame, *, sizes=(3, 4, 5), max_groups: int = 400,
                   candidates: list[list[str]] | None = None,
                   hl_min: float = 4, hl_max: float = 800, top_n: int = 8) -> list[Basket]:
    """Search asset groups of the given sizes for cointegrated baskets with a tradable
    half-life. To avoid a combinatorial blow-up (and data-snooping over thousands of groups),
    restrict to `candidates` (e.g. within-cluster groups) or cap at `max_groups` evaluated.
    Ranked by half-life (fastest reversion first)."""
    if candidates is None:
        candidates = []
        for k in sizes:
            candidates.extend([list(c) for c in combinations(prices.columns, k)])
    if len(candidates) > max_groups:
        # deterministic subsample (seeded) so the search is reproducible, not cherry-picked
        rng = np.random.default_rng(7)
        idx = rng.choice(len(candidates), size=max_groups, replace=False)
        candidates = [candidates[i] for i in sorted(idx)]

    out = []
    for group in candidates:
        sub = prices[group].dropna()
        if sub.shape[0] < 500 or sub.shape[1] < 2:
            continue
        try:
            b = make_basket(sub)
        except Exception:
            continue
        if b is None:
            continue
        if hl_min <= b.half_life <= hl_max:
            out.append(b)
    out.sort(key=lambda b: b.half_life)
    return out[:top_n]


def basket_zscore(prices: pd.DataFrame, basket: Basket, *, window: int = 168) -> pd.Series:
    spread = np.log(prices[basket.symbols]).mul(basket.weight_series, axis=1).sum(axis=1)
    return rolling_zscore(spread, window=window)
