"""
Liquidity sweep detection — equal highs/lows and stop-hunt reversal signals.

WHY THIS EXISTS
---------------
Before many strong directional moves, smart money / market makers execute a
"liquidity sweep":

  1. They hunt the stop-loss orders clustered above a resistance (equal highs)
     or below a support (equal lows).
  2. Price momentarily breaks through the level (triggering stops = liquidity).
  3. Then immediately reverses.

This reversal is the real move.  The sweep is the trigger.

EXAMPLE
-------
  Price makes 3 highs at ~2650.  Retail traders place stops just above 2650.
  Smart money pushes price to 2651 (sweeping those stops), then dumps hard.
  A bearish OB forms right at 2650–2651 and is the ideal short entry.

HOW WE USE IT
-------------
  - Before taking a bearish OB, check if there was a recent sweep of equal highs.
  - Before taking a bullish OB, check if there was a recent sweep of equal lows.
  - A sweep within the last 20–50 bars adds high-probability confirmation.

EQUAL HIGHS / EQUAL LOWS DEFINITION
-------------------------------------
Two highs are "equal" if they differ by less than `tolerance` price units.
For XAUUSDT: tolerance = 0.50–2.00 USD (about 1–5 ticks).

SWEEP DEFINITION
----------------
A sweep occurs when:
  1. A swing high/low cluster (equal level) exists.
  2. Price wicks above (for highs) or below (for lows) the level.
  3. Price closes BACK below (for highs) or above (for lows) the level.
  → The wick exceeded the level but the close reversed = stop hunt confirmed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
import numpy as np


# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_LOOKBACK       = 100   # bars to scan for equal levels
DEFAULT_TOLERANCE_ATR  = 0.1   # equal-level tolerance as fraction of ATR
DEFAULT_MIN_TOUCHES    = 2     # minimum touches to qualify as a liquidity pool
DEFAULT_SWEEP_LOOKBACK = 50    # bars after level to look for a sweep


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class LiquidityLevel:
    """
    A price level where multiple swing highs/lows clustered (= stop pool).

    Fields
    ------
    price      : The clustered price level.
    kind       : 'high' (equal highs above) or 'low' (equal lows below).
    touches    : Number of times price touched this level.
    first_idx  : Bar index of the first touch.
    last_idx   : Bar index of the most recent touch.
    """
    price     : float
    kind      : str       # 'high' | 'low'
    touches   : int
    first_idx : int
    last_idx  : int
    time_first: pd.Timestamp
    time_last : pd.Timestamp


@dataclass
class LiquiditySweep:
    """
    A confirmed liquidity sweep event.

    Fields
    ------
    sweep_bar_idx  : The bar that executed the sweep (wick through the level).
    sweep_price    : The extreme wick price (high for bull sweep, low for bear sweep).
    close_price    : Close of the sweep bar (should be back below/above level).
    level          : The LiquidityLevel that was swept.
    direction      : 'bearish_sweep' (swept equal highs, expect drop) |
                     'bullish_sweep' (swept equal lows, expect rise).
    reversal_strength : How far price closed back from the extreme (% of sweep distance).
    """
    sweep_bar_idx   : int
    sweep_price     : float
    close_price     : float
    level           : LiquidityLevel
    direction       : str
    reversal_strength: float    # 0.0–1.0
    time            : pd.Timestamp


# ── Equal level detection ─────────────────────────────────────────────────────

def detect_liquidity_levels(
    df: pd.DataFrame,
    lookback: int              = DEFAULT_LOOKBACK,
    tolerance: Optional[float] = None,
    tolerance_atr_fraction: float = DEFAULT_TOLERANCE_ATR,
    min_touches: int           = DEFAULT_MIN_TOUCHES,
    swing_len: int             = 3,
) -> List[LiquidityLevel]:
    """
    Find price levels where equal highs or equal lows clustered.

    Algorithm:
    1. Identify swing highs / swing lows within `lookback` bars.
    2. Group swing points that are within `tolerance` of each other.
    3. Any group with >= `min_touches` members = a liquidity pool.

    Parameters
    ----------
    df                    : OHLCV DataFrame with optional 'atr' column.
    lookback              : Number of bars to look back.
    tolerance             : Absolute price tolerance for "equal" levels.
                            If None, computed from ATR.
    tolerance_atr_fraction: Tolerance = ATR * this fraction (used when tolerance=None).
    min_touches           : Minimum swing points at the level to form a pool.
    swing_len             : Bars on each side for swing detection.

    Returns
    -------
    List of LiquidityLevel sorted by last_idx descending (most recent first).
    """
    n = len(df)
    start = max(0, n - lookback)
    window = df.iloc[start:]

    # Determine tolerance from ATR if not given
    if tolerance is None:
        if "atr" in window.columns:
            atr_val = float(window["atr"].median())
        else:
            # Fallback: use 0.1% of price as tolerance
            atr_val = float(window["close"].median()) * 0.001
        tolerance = atr_val * tolerance_atr_fraction

    highs = window["high"].values
    lows  = window["low"].values
    nw    = len(window)

    # Swing high indices in window
    sh_indices = [
        i for i in range(swing_len, nw - swing_len)
        if highs[i] == max(highs[i - swing_len : i + swing_len + 1])
    ]
    # Swing low indices in window
    sl_indices = [
        i for i in range(swing_len, nw - swing_len)
        if lows[i] == min(lows[i - swing_len : i + swing_len + 1])
    ]

    levels: List[LiquidityLevel] = []

    # ── Cluster swing highs ────────────────────────────────────────────────────
    sh_prices = [float(highs[i]) for i in sh_indices]
    levels.extend(
        _cluster_into_levels(sh_prices, sh_indices, window, "high", tolerance, min_touches, start)
    )

    # ── Cluster swing lows ─────────────────────────────────────────────────────
    sl_prices = [float(lows[i]) for i in sl_indices]
    levels.extend(
        _cluster_into_levels(sl_prices, sl_indices, window, "low", tolerance, min_touches, start)
    )

    return sorted(levels, key=lambda l: l.last_idx, reverse=True)


def _cluster_into_levels(
    prices: List[float],
    indices: List[int],
    window: pd.DataFrame,
    kind: str,
    tolerance: float,
    min_touches: int,
    global_offset: int,
) -> List[LiquidityLevel]:
    """Group nearby prices into clusters (equal-high or equal-low levels)."""
    if not prices:
        return []

    result: List[LiquidityLevel] = []
    used = [False] * len(prices)

    for i in range(len(prices)):
        if used[i]:
            continue
        cluster_prices = [prices[i]]
        cluster_indices = [indices[i]]
        used[i] = True

        for j in range(i + 1, len(prices)):
            if not used[j] and abs(prices[j] - prices[i]) <= tolerance:
                cluster_prices.append(prices[j])
                cluster_indices.append(indices[j])
                used[j] = True

        if len(cluster_prices) >= min_touches:
            level_price = float(np.mean(cluster_prices))
            result.append(LiquidityLevel(
                price      = level_price,
                kind       = kind,
                touches    = len(cluster_prices),
                first_idx  = global_offset + min(cluster_indices),
                last_idx   = global_offset + max(cluster_indices),
                time_first = window.index[min(cluster_indices)],
                time_last  = window.index[max(cluster_indices)],
            ))

    return result


# ── Sweep detection ───────────────────────────────────────────────────────────

def detect_liquidity_sweeps(
    df: pd.DataFrame,
    levels: List[LiquidityLevel],
    sweep_lookback: int = DEFAULT_SWEEP_LOOKBACK,
) -> List[LiquiditySweep]:
    """
    Find bars where price wicked through a liquidity level then closed back.

    For equal HIGHS (kind='high'):
      - high > level.price  AND  close < level.price → bearish sweep
    For equal LOWS (kind='low'):
      - low  < level.price  AND  close > level.price → bullish sweep

    Parameters
    ----------
    df           : Full OHLCV DataFrame.
    levels       : Output of detect_liquidity_levels().
    sweep_lookback: Max bars after the level's last touch to search for sweeps.

    Returns
    -------
    List of LiquiditySweep sorted by sweep_bar_idx descending (most recent first).
    """
    sweeps: List[LiquiditySweep] = []
    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    n      = len(df)

    for level in levels:
        # Only scan bars AFTER the level was established
        start_bar = level.last_idx + 1
        end_bar   = min(level.last_idx + sweep_lookback + 1, n)

        for i in range(start_bar, end_bar):
            if level.kind == "high":
                # Bearish sweep: wick above the equal-high level, close below it
                if highs[i] > level.price and closes[i] < level.price:
                    sweep_dist = highs[i] - level.price
                    reversal   = closes[i] - level.price   # negative (price fell back)
                    strength   = abs(reversal) / highs[i] if highs[i] > 0 else 0.0
                    sweeps.append(LiquiditySweep(
                        sweep_bar_idx    = i,
                        sweep_price      = float(highs[i]),
                        close_price      = float(closes[i]),
                        level            = level,
                        direction        = "bearish_sweep",
                        reversal_strength= min(strength, 1.0),
                        time             = df.index[i],
                    ))
                    break  # One sweep per level is enough

            elif level.kind == "low":
                # Bullish sweep: wick below the equal-low level, close above it
                if lows[i] < level.price and closes[i] > level.price:
                    reversal = closes[i] - lows[i]
                    strength = reversal / (level.price - lows[i]) if (level.price - lows[i]) > 0 else 0.0
                    sweeps.append(LiquiditySweep(
                        sweep_bar_idx    = i,
                        sweep_price      = float(lows[i]),
                        close_price      = float(closes[i]),
                        level            = level,
                        direction        = "bullish_sweep",
                        reversal_strength= min(strength, 1.0),
                        time             = df.index[i],
                    ))
                    break

    return sorted(sweeps, key=lambda s: s.sweep_bar_idx, reverse=True)


# ── Convenience check for OB strategy ─────────────────────────────────────────

def has_recent_sweep(
    df: pd.DataFrame,
    current_bar: int,
    ob_direction: str,         # 'bullish' | 'bearish'
    lookback_bars: int  = 30,
    sweep_tolerance: int = 20,  # bars: how old can the sweep be?
    **level_kwargs,
) -> bool:
    """
    True if there is a recent liquidity sweep aligned with the OB direction.

    For a BULLISH OB (expecting price up):
      We want a prior BULLISH sweep (equal lows swept = buyers absorbed).
    For a BEARISH OB (expecting price down):
      We want a prior BEARISH sweep (equal highs swept = sellers absorbed).

    Parameters
    ----------
    df           : OHLCV DataFrame.
    current_bar  : Index of the current bar (signal bar).
    ob_direction : 'bullish' or 'bearish'.
    lookback_bars: How many bars to scan for levels.
    sweep_tolerance: Maximum age of the sweep (bars before current).
    """
    window = df.iloc[max(0, current_bar - lookback_bars) : current_bar + 1]
    levels  = detect_liquidity_levels(window, lookback=lookback_bars, **level_kwargs)
    sweeps  = detect_liquidity_sweeps(window, levels)

    target_dir = "bullish_sweep" if ob_direction == "bullish" else "bearish_sweep"

    for sweep in sweeps:
        if sweep.direction == target_dir:
            bars_ago = current_bar - sweep.sweep_bar_idx
            if bars_ago <= sweep_tolerance:
                return True

    return False
