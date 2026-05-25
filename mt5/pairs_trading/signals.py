"""
Signal generation for pairs trading.

Three pure functions (no MT5, no IO) — easy to unit-test:

    refit_beta(log_y, log_x)        -> (alpha, beta, adf_p)
    compute_z_series(spread, win)   -> z-score Series  (NaN until win is filled)
    decide_action(z_now, side_now)  -> ActionDecision

Convention: a "spread" is `log(y) - β·log(x) - α`. Going LONG the spread means
buying y and selling x; going SHORT means selling y and buying x.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.stattools import adfuller
    _HAVE_STATSMODELS = True
except ImportError:                                  # graceful — runner will warn
    _HAVE_STATSMODELS = False


# ─────────────────────────────────────────────────────────────────────────────
# Action vocabulary
# ─────────────────────────────────────────────────────────────────────────────


class Side(str, Enum):
    """Pair-level position direction (NOT per-leg)."""
    FLAT  = "flat"
    LONG  = "long"          # long spread  = long y, short x
    SHORT = "short"         # short spread = short y, long x


class Action(str, Enum):
    """What the runner should do this cycle for one pair."""
    HOLD       = "hold"           # do nothing
    OPEN_LONG  = "open_long"      # open long-spread position
    OPEN_SHORT = "open_short"     # open short-spread position
    EXIT       = "exit"           # close existing position (mean-revert reached)
    STOP       = "stop"           # close existing position (stop_z breached)
    TIME_STOP  = "time_stop"      # close existing position (held too long)


@dataclass(frozen=True)
class ActionDecision:
    """Outcome of `decide_action` — includes a human-readable reason for logs."""
    action: Action
    reason: str


# ─────────────────────────────────────────────────────────────────────────────
# β/α refit + cointegration check
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BetaFit:
    alpha:  float
    beta:   float
    adf_p:  float          # p-value of ADF on residuals (cointegration test)
    resid_std: float       # residual std-dev on training window
    n_obs:  int


def refit_beta(log_y: np.ndarray, log_x: np.ndarray) -> BetaFit:
    """
    OLS regression `log_y = α + β·log_x + ε` with ADF cointegration p-value.

    Arrays must be aligned and NaN-free. Length must be >= 50 (else ValueError).
    """
    log_y = np.asarray(log_y, dtype=float).ravel()
    log_x = np.asarray(log_x, dtype=float).ravel()
    if log_y.shape != log_x.shape:
        raise ValueError(f"refit_beta: shapes differ {log_y.shape} vs {log_x.shape}")
    if log_y.size < 50:
        raise ValueError(f"refit_beta: need >= 50 obs, got {log_y.size}")

    X = np.column_stack([np.ones_like(log_x), log_x])
    coef, *_ = np.linalg.lstsq(X, log_y, rcond=None)
    alpha, beta = float(coef[0]), float(coef[1])
    resid = log_y - (alpha + beta * log_x)

    if _HAVE_STATSMODELS:
        try:
            _, adf_p, *_ = adfuller(resid, regression="n", autolag="AIC")
            adf_p = float(adf_p)
        except Exception:
            adf_p = 1.0
    else:
        adf_p = 1.0    # cannot test — caller's ADF gate becomes a no-op

    return BetaFit(
        alpha=alpha, beta=beta, adf_p=adf_p,
        resid_std=float(resid.std()),
        n_obs=int(log_y.size),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Z-score
# ─────────────────────────────────────────────────────────────────────────────


def compute_spread_series(
    prices_y: pd.Series, prices_x: pd.Series, alpha: float, beta: float,
) -> pd.Series:
    """`log(y) - β·log(x) - α`. Indices MUST be already aligned."""
    return np.log(prices_y) - beta * np.log(prices_x) - alpha


def compute_z_series(spread: pd.Series, window: int) -> pd.Series:
    """
    Rolling z-score: `(spread - rolling_mean) / rolling_std`.

    First `window-1` values are NaN — caller must guard against that.
    Uses default ddof=1 (sample std).
    """
    mu = spread.rolling(window, min_periods=window).mean()
    sd = spread.rolling(window, min_periods=window).std()
    return (spread - mu) / sd


# ─────────────────────────────────────────────────────────────────────────────
# Decision
# ─────────────────────────────────────────────────────────────────────────────


def decide_action(
    z_now:           float,
    side_now:        Side,
    bars_in_position: int,
    *,
    entry_z:         float,
    exit_z:          float,
    stop_z:          float,
    time_stop_bars:  int,
) -> ActionDecision:
    """
    Pure state-machine: given current z-score and existing position, decide.

    Symmetric stops:
      * If FLAT and z >  +entry_z   → OPEN_SHORT
      * If FLAT and z <  -entry_z   → OPEN_LONG
      * If FLAT otherwise            → HOLD
      * If in position and |z| <= exit_z                              → EXIT
      * If in position and |z| >= stop_z (same-direction adverse)     → STOP
      * If in position and bars_in_position >= time_stop_bars         → TIME_STOP
      * Otherwise → HOLD

    The stop condition mirrors the backtest (NB26): a SHORT position is stopped
    when z drives even further above +entry_z past stop_z; symmetric for LONG.
    """
    if not np.isfinite(z_now):
        return ActionDecision(Action.HOLD, "z_now is NaN/inf — insufficient history")

    # ── flat: maybe open ───────────────────────────────────────────────────
    if side_now == Side.FLAT:
        if z_now > entry_z:
            return ActionDecision(
                Action.OPEN_SHORT,
                f"z={z_now:+.3f} > +{entry_z} → SHORT spread",
            )
        if z_now < -entry_z:
            return ActionDecision(
                Action.OPEN_LONG,
                f"z={z_now:+.3f} < -{entry_z} → LONG spread",
            )
        return ActionDecision(
            Action.HOLD,
            f"z={z_now:+.3f} inside ±{entry_z} → no entry",
        )

    # ── in position: maybe exit / stop / time_stop ─────────────────────────
    if bars_in_position >= time_stop_bars:
        return ActionDecision(
            Action.TIME_STOP,
            f"bars_in_position={bars_in_position} >= {time_stop_bars} → TIME_STOP",
        )

    if abs(z_now) <= exit_z:
        return ActionDecision(
            Action.EXIT,
            f"|z|={abs(z_now):.3f} <= {exit_z} → mean-revert EXIT",
        )

    if abs(z_now) >= stop_z:
        adverse = (side_now == Side.LONG  and z_now < -entry_z) or \
                  (side_now == Side.SHORT and z_now > +entry_z)
        if adverse:
            return ActionDecision(
                Action.STOP,
                f"|z|={abs(z_now):.3f} >= {stop_z} adverse vs {side_now.value} → STOP",
            )

    return ActionDecision(
        Action.HOLD,
        f"z={z_now:+.3f} side={side_now.value} bars_in={bars_in_position} → HOLD",
    )
