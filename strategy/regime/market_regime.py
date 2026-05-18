"""
Market regime detection — trending vs ranging, volatility classification.

WHY THIS EXISTS
---------------
Order Block strategies perform very differently in different market regimes:

  TRENDING:
    - High-quality OBs with strong follow-through.
    - BOS patterns are clean and reliable.
    - FVGs are respected and filled efficiently.
    - → FULL participation in OB signals.

  RANGING / CHOPPY:
    - OBs are frequently violated and re-tested multiple times.
    - Price oscillates between equal highs/lows without breaking structure.
    - High rate of OB invalidation and false breakouts.
    - → SKIP or heavily filter OB signals.

  HIGH VOLATILITY:
    - Large ATR spikes (news events, manipulated moves).
    - SL distances blow out → poor RR.
    - Spreads widen.
    - → REDUCE position size or skip entirely.

  LOW VOLATILITY:
    - Tight ATR → SL distances extremely small.
    - Single large candle can stop out the position.
    - → Check if signal geometry is still valid.

REGIME DETECTION METHODS
-------------------------
We use two independent indicators and combine them:

1. ADX (Average Directional Index):
   - ADX > 25 = trending market.
   - ADX < 20 = ranging / directionless.
   - ADX 20–25 = transition zone.

2. ATR percentile (volatility regime):
   - Current ATR vs rolling ATR percentile over the lookback window.
   - > 75th percentile = high volatility.
   - < 25th percentile = low volatility.
   - 25th–75th = normal volatility.

WHY ADX OVER SIMPLE TREND INDICATORS
--------------------------------------
ADX measures TREND STRENGTH, not direction.  A strong downtrend has high ADX
just like a strong uptrend.  This is exactly what we want: we don't want the
regime filter to override the OB direction — we only want it to tell us whether
the market is trending AT ALL.

USAGE
-----
    result = detect_regime(df, adx_period=14, lookback=100)
    if result.regime == MarketRegime.RANGING:
        skip_trade("choppy market — OB edge is low")
    elif result.regime == MarketRegime.HIGH_VOLATILITY:
        reduce_size(factor=0.5)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd


# ── Data models ───────────────────────────────────────────────────────────────

class MarketRegime(Enum):
    TRENDING        = "trending"        # ADX > 25, normal volatility
    RANGING         = "ranging"         # ADX < 20
    HIGH_VOLATILITY = "high_volatility" # ATR spike (> 75th pct)
    LOW_VOLATILITY  = "low_volatility"  # ATR compressed (< 25th pct)
    UNKNOWN         = "unknown"         # insufficient data


@dataclass
class RegimeResult:
    """Full regime assessment for a given bar."""
    regime           : MarketRegime
    adx              : float          # ADX value at assessment bar
    atr              : float          # ATR value at assessment bar
    atr_percentile   : float          # ATR rank 0–100 vs lookback window
    is_trending      : bool
    is_ranging       : bool
    is_high_vol      : bool
    description      : str            # human-readable summary


# ── ADX computation ───────────────────────────────────────────────────────────

def _compute_adx(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.Series:
    """
    Compute Average Directional Index (ADX) using Wilder's smoothing.

    ADX does not include direction — it only measures trend strength.
    Range: 0–100.
      > 25 = trending
      < 20 = ranging
      > 50 = very strong trend (often overextended)

    Returns a Series of the same length as df (NaN for first `period` bars).
    """
    high  = df["high"]
    low   = df["low"]
    close = df["close"]

    # True Range
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Directional movement
    up   = high.diff()
    down = -low.diff()

    plus_dm  = pd.Series(np.where((up > down) & (up > 0), up, 0.0),   index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)

    # Wilder's EWM smoothing (equivalent to Wilder's RMA)
    atr_s    = tr.ewm(       alpha=1.0/period, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm( alpha=1.0/period, adjust=False).mean() / atr_s.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1.0/period, adjust=False).mean() / atr_s.replace(0, np.nan)

    dx_denom = (plus_di + minus_di).replace(0, np.nan)
    dx       = 100 * (plus_di - minus_di).abs() / dx_denom

    adx = dx.ewm(alpha=1.0/period, adjust=False).mean()
    return adx.fillna(0.0)


# ── Main regime detection ─────────────────────────────────────────────────────

def detect_regime(
    df: pd.DataFrame,
    adx_period: int       = 14,
    lookback: int         = 100,
    adx_trend_threshold: float  = 25.0,
    adx_range_threshold: float  = 20.0,
    atr_high_pct: float   = 75.0,   # percentile above = high volatility
    atr_low_pct: float    = 25.0,   # percentile below = low volatility
) -> RegimeResult:
    """
    Assess the current market regime based on ADX + ATR percentile.

    Parameters
    ----------
    df                  : Closed OHLCV bars (with 'atr' column preferred).
    adx_period          : Period for ADX computation.
    lookback            : Bars to use for ATR percentile ranking.
    adx_trend_threshold : ADX >= this → trending.
    adx_range_threshold : ADX <= this → ranging.
    atr_high_pct        : ATR percentile above which = high volatility.
    atr_low_pct         : ATR percentile below which = low volatility.

    Returns
    -------
    RegimeResult for the LAST bar in df.
    """
    if len(df) < adx_period * 3:
        return RegimeResult(
            regime         = MarketRegime.UNKNOWN,
            adx            = 0.0,
            atr            = 0.0,
            atr_percentile = 50.0,
            is_trending    = False,
            is_ranging     = False,
            is_high_vol    = False,
            description    = "Insufficient data for regime detection.",
        )

    # ── ADX ──────────────────────────────────────────────────────────────────
    adx_series = _compute_adx(df, period=adx_period)
    current_adx = float(adx_series.iloc[-1])

    # ── ATR ──────────────────────────────────────────────────────────────────
    if "atr" in df.columns:
        atr_series = df["atr"]
    else:
        # Compute ATR on the fly
        prev_close = df["close"].shift(1)
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"]  - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr_series = tr.ewm(span=adx_period, adjust=False).mean()

    current_atr = float(atr_series.iloc[-1])

    # ATR percentile vs rolling lookback window
    recent_atrs = atr_series.iloc[-lookback:].dropna()
    pct = float(pd.Series(recent_atrs).rank(pct=True).iloc[-1]) * 100.0

    # ── Classify ─────────────────────────────────────────────────────────────
    is_trending   = current_adx >= adx_trend_threshold
    is_ranging    = current_adx <= adx_range_threshold
    is_high_vol   = pct >= atr_high_pct
    is_low_vol    = pct <= atr_low_pct

    # Priority: high volatility overrides trend/range classification
    # because high-vol markets are risky regardless of trend strength.
    if is_high_vol:
        regime = MarketRegime.HIGH_VOLATILITY
        desc   = (
            f"HIGH VOLATILITY — ATR={current_atr:.2f} at {pct:.0f}th pct "
            f"(ADX={current_adx:.1f}). Reduce size."
        )
    elif is_low_vol:
        regime = MarketRegime.LOW_VOLATILITY
        desc   = (
            f"LOW VOLATILITY — ATR={current_atr:.2f} at {pct:.0f}th pct "
            f"(ADX={current_adx:.1f}). SL distances may be too small."
        )
    elif is_ranging:
        regime = MarketRegime.RANGING
        desc   = (
            f"RANGING — ADX={current_adx:.1f} (threshold {adx_range_threshold}). "
            f"ATR={current_atr:.2f}. OB edge reduced."
        )
    elif is_trending:
        regime = MarketRegime.TRENDING
        desc   = (
            f"TRENDING — ADX={current_adx:.1f} (threshold {adx_trend_threshold}). "
            f"ATR={current_atr:.2f}. Optimal for OB."
        )
    else:
        # Transition zone: ADX between thresholds
        regime = MarketRegime.RANGING
        desc   = (
            f"TRANSITION — ADX={current_adx:.1f} (between {adx_range_threshold} and "
            f"{adx_trend_threshold}). ATR={current_atr:.2f}. Use caution."
        )

    return RegimeResult(
        regime         = regime,
        adx            = current_adx,
        atr            = current_atr,
        atr_percentile = pct,
        is_trending    = is_trending,
        is_ranging     = is_ranging,
        is_high_vol    = is_high_vol,
        description    = desc,
    )


# ── DataFrame annotation (for notebooks / rolling analysis) ──────────────────

def add_regime_columns(
    df: pd.DataFrame,
    adx_period: int = 14,
) -> pd.DataFrame:
    """
    Add 'adx', 'atr_percentile', 'regime' columns to df for full-history analysis.

    NOTE: This does NOT use rolling regime detection (which would be O(n²)).
    It computes ADX and ATR percentile as series and classifies row-by-row.
    For true walk-forward testing use detect_regime() inside a loop.
    """
    df = df.copy()
    adx_series = _compute_adx(df, period=adx_period)
    df["adx"]   = adx_series

    if "atr" not in df.columns:
        prev_close = df["close"].shift(1)
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"]  - prev_close).abs(),
        ], axis=1).max(axis=1)
        df["atr"] = tr.ewm(span=adx_period, adjust=False).mean()

    # Rolling 100-bar ATR percentile
    df["atr_percentile"] = (
        df["atr"].rolling(100, min_periods=10)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False)
    )

    conditions = [
        df["atr_percentile"] >= 75,
        df["atr_percentile"] <= 25,
        df["adx"] <= 20,
        df["adx"] >= 25,
    ]
    choices = [
        MarketRegime.HIGH_VOLATILITY.value,
        MarketRegime.LOW_VOLATILITY.value,
        MarketRegime.RANGING.value,
        MarketRegime.TRENDING.value,
    ]
    df["regime"] = np.select(conditions, choices, default=MarketRegime.RANGING.value)

    return df
