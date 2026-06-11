"""Spread construction — hedge ratios (static OLS + dynamic Kalman) and rolling z-scores.

A pairs trade is a bet that a *spread* mean-reverts. The spread is
    s_t = log(P_A,t) - beta_t * log(P_B,t)
where beta is the hedge ratio (how many units of B neutralise one unit of A). Two ways to
get beta, both implemented:
  * `ols_hedge_ratio`     — one static beta over a formation window (the article's implicit
                            choice; simple, but assumes a constant relationship).
  * `kalman_hedge_ratio`  — a time-varying beta from a Kalman filter (institutional upgrade;
                            adapts as the relationship drifts, the #1 weakness of static beta).

The trading signal is the *rolling* z-score of the spread: z_t = (s_t - mean) / std over a
trailing window. Rolling (not full-sample) is mandatory — a full-sample mean uses future
data and is the classic stat-arb look-ahead bug.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ols_hedge_ratio(log_a: pd.Series, log_b: pd.Series) -> tuple[float, float]:
    """Static hedge ratio via OLS of log_a on log_b (with intercept). Returns (beta, alpha).
    Fit this on the *formation* window only, then apply it out-of-sample."""
    x = log_b.values
    y = log_a.values
    A = np.vstack([x, np.ones_like(x)]).T
    beta, alpha = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(beta), float(alpha)


def static_spread(px_a: pd.Series, px_b: pd.Series, beta: float, alpha: float = 0.0) -> pd.Series:
    """Spread with a fixed hedge ratio: log(A) - beta*log(B) - alpha."""
    return np.log(px_a) - beta * np.log(px_b) - alpha


def rolling_hedge_ratio(px_a: pd.Series, px_b: pd.Series, *, window: int = 720) -> pd.Series:
    """Rolling-OLS hedge ratio: beta_t = cov(logA, logB)_t / var(logB)_t over a trailing
    window. The middle ground between a fixed OLS beta (never adapts) and Kalman (adapts every
    bar). Point-in-time by construction."""
    la, lb = np.log(px_a), np.log(px_b)
    cov = la.rolling(window, min_periods=window // 2).cov(lb)
    var = lb.rolling(window, min_periods=window // 2).var()
    return cov / var


def rolling_ols_spread(px_a: pd.Series, px_b: pd.Series, *, window: int = 720) -> pd.Series:
    """Spread under a rolling-OLS hedge ratio: logA - beta_t·logB."""
    beta_t = rolling_hedge_ratio(px_a, px_b, window=window)
    return np.log(px_a) - beta_t * np.log(px_b)


def kalman_hedge_ratio(px_a: pd.Series, px_b: pd.Series, *,
                       delta: float = 1e-4, obs_cov: float = 1e-3) -> pd.DataFrame:
    """Dynamic hedge ratio via a Kalman filter (state = [beta, alpha]).

    Model:  log(A)_t = beta_t*log(B)_t + alpha_t + noise.  The state evolves as a random
    walk; `delta` sets how fast beta is allowed to drift (transition cov = delta/(1-delta)*I),
    `obs_cov` is observation noise. This is the Chan (2013) formulation. Returns a frame with
    columns [beta, alpha, spread, resid_var] — `spread` is the one-step prediction error
    (already point-in-time / look-ahead-free by construction).
    """
    y = np.log(px_a.values)
    x = np.log(px_b.values)
    n = len(y)
    trans_cov = delta / (1 - delta) * np.eye(2)

    # Seed the state with an OLS fit on the first window so the filter starts near the true
    # relationship. Without this the first innovations equal the raw log-price level (~10),
    # which pollutes the rolling z-score warm-up and wrecks the backtest.
    seed = min(max(50, n // 50), n)
    A0 = np.vstack([x[:seed], np.ones(seed)]).T
    b0, a0 = np.linalg.lstsq(A0, y[:seed], rcond=None)[0]
    theta = np.array([b0, a0])
    R = np.eye(2) * 1.0               # moderate initial uncertainty -> adapts quickly

    beta = np.zeros((n, 2))           # [slope, intercept]
    spread = np.full(n, np.nan)
    qvar = np.full(n, np.nan)

    for t in range(n):
        F = np.array([x[t], 1.0])              # observation matrix
        if t > 0:
            R = R + trans_cov                  # predict step
        yhat = F @ theta                       # prediction using PRIOR state -> no look-ahead
        e = y[t] - yhat                         # innovation == the tradable spread
        Q = F @ R @ F.T + obs_cov              # innovation variance
        K = (R @ F) / Q                         # Kalman gain
        theta = theta + K * e                   # update
        R = R - np.outer(K, F) @ R
        beta[t] = theta
        spread[t] = e
        qvar[t] = Q

    return pd.DataFrame({"beta": beta[:, 0], "alpha": beta[:, 1],
                         "spread": spread, "resid_var": qvar}, index=px_a.index)


def rolling_zscore(spread: pd.Series, window: int = 168, min_periods: int | None = None) -> pd.Series:
    """Rolling z-score of a spread (trailing window only). Default 168 H1 bars = 1 week.
    `.shift(0)` is intentional — the spread at t is known at the close of t; the look-ahead
    guard is in execution (t+1), not here."""
    mp = min_periods or window // 2
    mu = spread.rolling(window, min_periods=mp).mean()
    sd = spread.rolling(window, min_periods=mp).std()
    return (spread - mu) / sd


def halflife(spread: pd.Series) -> float:
    """Mean-reversion half-life from an OU/AR(1) fit: ds_t = lambda*(s_{t-1}) + e.
    A short half-life => fast reversion => a tradable spread; a long/negative one => the
    spread wanders (a non-stationary, untradable 'pair'). Used as a quality screen."""
    s = spread.dropna()
    s_lag = s.shift(1).dropna()
    s = s.loc[s_lag.index]
    ds = (s - s_lag)
    A = np.vstack([s_lag.values, np.ones(len(s_lag))]).T
    lam = np.linalg.lstsq(A, ds.values, rcond=None)[0][0]
    if lam >= 0:
        return np.inf            # not mean-reverting
    return float(-np.log(2) / lam)
