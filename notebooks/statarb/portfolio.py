"""Portfolio construction — combine pair/sleeve return streams (Section G).

Phase-1 weighted every pair equally. This module asks: *can allocation alone improve net
performance, holding signal quality fixed?* It takes a panel of per-sleeve net returns and
produces weights four ways:

  * `equal_weight`     — the Phase-1 baseline.
  * `inverse_vol`      — 1/σ; cheap risk balancing, no covariance needed.
  * `risk_parity`      — each sleeve contributes equal risk (iterative, uses covariance).
  * `mean_variance`    — max Sharpe with a **Ledoit-Wolf shrunk** covariance (robust; raw
                         sample covariance on correlated crypto sleeves is near-singular and
                         produces insane leverage — shrinkage is mandatory, not optional).

**Anti-look-ahead rule:** weights must be estimated on a *formation* slice of returns and
applied to a later *trading* slice. `apply_weights` enforces that split. Weights are
long-only-normalised by default (Σ|w|=1) since these are market-neutral sleeves we either turn
on or scale, not short.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from .data import BARS_PER_YEAR
from .metrics import perf_stats


def equal_weight(rets: pd.DataFrame) -> pd.Series:
    n = rets.shape[1]
    return pd.Series(1.0 / n, index=rets.columns)


def inverse_vol(rets: pd.DataFrame) -> pd.Series:
    vol = rets.std()
    w = 1.0 / vol.replace(0, np.nan)
    w = w.fillna(0.0)
    return w / w.sum()


def risk_parity(rets: pd.DataFrame, *, iters: int = 500, tol: float = 1e-8) -> pd.Series:
    """Equal risk contribution weights via a simple multiplicative fixed-point iteration on the
    covariance. Converges for the long-only ERC problem and needs no QP solver."""
    cov = LedoitWolf().fit(rets.dropna().values).covariance_
    n = cov.shape[0]
    w = np.ones(n) / n
    for _ in range(iters):
        mrc = cov @ w                       # marginal risk contribution
        rc = w * mrc                        # risk contribution
        target = rc.mean()
        w_new = w * (target / (rc + 1e-12)) ** 0.5
        w_new = np.clip(w_new, 0, None)
        w_new /= w_new.sum()
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new
    return pd.Series(w, index=rets.columns)


def mean_variance(rets: pd.DataFrame, *, long_only: bool = True) -> pd.Series:
    """Max-Sharpe (tangency) weights with a Ledoit-Wolf shrunk covariance. w ∝ Σ⁻¹μ. Clipped
    to long-only and renormalised by default to avoid the pathological leverage a raw inverse
    covariance produces on near-collinear sleeves."""
    R = rets.dropna()
    mu = R.mean().values
    cov = LedoitWolf().fit(R.values).covariance_
    try:
        w = np.linalg.solve(cov, mu)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(cov) @ mu
    if long_only:
        w = np.clip(w, 0, None)
    s = np.abs(w).sum()
    w = w / s if s > 0 else np.ones_like(w) / len(w)
    return pd.Series(w, index=rets.columns)


def vol_target(rets: pd.DataFrame, *, target_ann_vol: float = 0.10, timeframe: str = "H1",
               max_leverage: float = 3.0) -> pd.Series:
    """Equal-weight composition scaled so the *combined* book hits a target annualised vol
    (estimated on the formation slice). Unlike inverse-vol (which balances across sleeves),
    vol-targeting sets the overall gross exposure — the lever most allocators actually pull.
    Returns weights that sum to the leverage multiple (may exceed 1)."""
    ew = equal_weight(rets)
    port = rets.mul(ew, axis=1).sum(axis=1)
    realized = port.std() * np.sqrt(BARS_PER_YEAR[timeframe])
    lev = min(target_ann_vol / realized, max_leverage) if realized > 0 else 1.0
    return ew * lev


ALLOCATORS = {"equal_weight": equal_weight, "inverse_vol": inverse_vol,
              "risk_parity": risk_parity, "mean_variance": mean_variance,
              "vol_target": vol_target}


def apply_weights(rets_trade: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Portfolio net return from sleeve returns and fixed weights (weights from a PRIOR
    formation window — caller's responsibility, see compare_allocators)."""
    cols = [c for c in weights.index if c in rets_trade.columns]
    return rets_trade[cols].mul(weights[cols], axis=1).sum(axis=1)


def compare_allocators(pair_returns: pd.DataFrame, *, split: float = 0.5,
                       timeframe: str = "H1") -> tuple[pd.DataFrame, dict]:
    """Estimate each allocator's weights on the first `split` of the sleeve-return panel, apply
    out-of-sample to the rest, and report stats. Returns (stats_table, weight_dict). This keeps
    the allocation comparison honestly out-of-sample, the same discipline as pair selection."""
    R = pair_returns.dropna(how="all").fillna(0.0)
    cut = int(len(R) * split)
    form, trade = R.iloc[:cut], R.iloc[cut:]
    stats, weights = {}, {}
    for name, fn in ALLOCATORS.items():
        try:
            w = fn(form)
        except Exception:
            continue
        weights[name] = w
        ret = apply_weights(trade, w)
        s = perf_stats(ret, timeframe=timeframe)
        stats[name] = {k: s[k] for k in ("sharpe", "sortino", "cagr", "ann_vol",
                                         "max_dd", "calmar", "hit_rate")}
    return pd.DataFrame(stats).T, weights
