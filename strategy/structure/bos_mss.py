"""
Break of Structure (BOS) and Market Structure Shift (MSS) detection.

WHY THIS EXISTS
---------------
The baseline OB strategy fires on every displacement, even in choppy ranging
markets where price has no clear directional bias.  BOS/MSS detection solves
this by requiring that:
  - A real swing high/low is first identified.
  - Price subsequently breaks through that swing level with a closed candle.
  - Only THEN is the market considered structured enough to trade OBs.

CONCEPT DEFINITIONS
-------------------
Swing High : A bar whose high is the highest within ±swing_len bars on each side.
Swing Low  : A bar whose low is the lowest within ±swing_len bars on each side.

BOS (Break of Structure):
  - Bullish BOS : Close > most recent confirmed swing HIGH.
    Means: buyers pushed through resistance → bullish structure intact.
  - Bearish BOS : Close < most recent confirmed swing LOW.
    Means: sellers broke through support → bearish structure intact.

MSS (Market Structure Shift):
  - Was trending up (HH + HL pattern) → price breaks through a swing LOW.
    This is NOT just a BOS but a full trend reversal signal → bearish MSS.
  - Was trending down (LL + LH pattern) → price breaks through a swing HIGH.
    Full trend reversal → bullish MSS.

HOW IT IS USED
--------------
  bias = get_current_bias(m5_df, swing_len=5)
  if bias == MarketBias.BULLISH:
      # only take bullish OB setups
  elif bias == MarketBias.BEARISH:
      # only take bearish OB setups
  else:
      # no trade — market structure is unclear
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import pandas as pd


# ── Configuration ────────────────────────────────────────────────────────────

DEFAULT_SWING_LEN = 5    # bars on each side of a pivot to qualify as swing
DEFAULT_LOOKBACK  = 50   # bars to scan for recent swing points


# ── Data models ───────────────────────────────────────────────────────────────

class MarketBias(Enum):
    """Current directional bias based on market structure."""
    BULLISH  = "bullish"   # HH/HL pattern — only take long OBs
    BEARISH  = "bearish"   # LL/LH pattern — only take short OBs
    NEUTRAL  = "neutral"   # insufficient structure to determine bias
    RANGING  = "ranging"   # conflicting structure — avoid trading


@dataclass
class SwingPoint:
    """A confirmed swing high or swing low."""
    bar_idx : int
    price   : float
    kind    : str          # 'high' | 'low'
    time    : pd.Timestamp


@dataclass
class StructureBreak:
    """A confirmed break of a swing level."""
    break_bar_idx  : int
    break_price    : float     # close price that caused the break
    swing_level    : float     # the swing that was broken
    direction      : str       # 'bullish' | 'bearish'
    is_mss         : bool      # True = reversal (MSS), False = continuation (BOS)
    time           : pd.Timestamp


# ── Swing point detection ─────────────────────────────────────────────────────

def detect_swing_points(
    df: pd.DataFrame,
    swing_len: int = DEFAULT_SWING_LEN,
) -> List[SwingPoint]:
    """
    Find all confirmed swing highs and lows in the DataFrame.

    A swing high at bar i is valid when:
        df['high'].iloc[i] == max(df['high'].iloc[i-swing_len : i+swing_len+1])
    Similarly for swing lows.

    We start from swing_len to ensure the left side is always available.
    The last swing_len bars are excluded because the right side is not yet
    confirmed (would require future bars we don't have).

    Parameters
    ----------
    df       : OHLCV DataFrame with DatetimeIndex (UTC).
    swing_len: Number of bars on each side required to confirm a pivot.

    Returns
    -------
    List of SwingPoint sorted by bar_idx ascending.
    """
    n = len(df)
    points: List[SwingPoint] = []

    # Need at least 2*swing_len+1 bars to detect any pivot
    if n < 2 * swing_len + 1:
        return points

    highs = df["high"].values
    lows  = df["low"].values

    for i in range(swing_len, n - swing_len):
        window_h = highs[i - swing_len : i + swing_len + 1]
        window_l = lows [i - swing_len : i + swing_len + 1]

        # Swing high: bar's high is the max of the surrounding window
        if highs[i] == window_h.max() and highs[i] > highs[i-1] and highs[i] > highs[i+1]:
            points.append(SwingPoint(
                bar_idx = i,
                price   = float(highs[i]),
                kind    = "high",
                time    = df.index[i],
            ))

        # Swing low: bar's low is the min of the surrounding window
        if lows[i] == window_l.min() and lows[i] < lows[i-1] and lows[i] < lows[i+1]:
            points.append(SwingPoint(
                bar_idx = i,
                price   = float(lows[i]),
                kind    = "low",
                time    = df.index[i],
            ))

    return sorted(points, key=lambda p: p.bar_idx)


# ── Structure break detection ─────────────────────────────────────────────────

def detect_structure_breaks(
    df: pd.DataFrame,
    swing_points: List[SwingPoint],
    lookback: int = DEFAULT_LOOKBACK,
) -> List[StructureBreak]:
    """
    Scan all bars for breaks of the most recent confirmed swing high/low.

    Logic:
    1. At each bar i, find the most recent swing HIGH before i.
    2. If df['close'][i] > swing_high → bullish BOS (or MSS if trend was down).
    3. Find the most recent swing LOW before i.
    4. If df['close'][i] < swing_low  → bearish BOS (or MSS if trend was up).

    BOS vs MSS classification:
    - BOS: Break continues in the direction of the prevailing trend.
    - MSS: Break reverses the prevailing trend (LL/LH → break above swing high).

    Parameters
    ----------
    df           : OHLCV DataFrame (same as passed to detect_swing_points).
    swing_points : Output of detect_swing_points().
    lookback     : Maximum bars to look back for the relevant swing point.

    Returns
    -------
    List of StructureBreak sorted by break_bar_idx ascending.
    """
    breaks: List[StructureBreak] = []
    closes = df["close"].values
    n = len(df)

    for i in range(1, n):
        # ── Most recent swing HIGH before bar i ───────────────────────────────
        prior_highs = [
            p for p in swing_points
            if p.kind == "high"
            and p.bar_idx < i
            and (i - p.bar_idx) <= lookback
        ]
        # ── Most recent swing LOW before bar i ────────────────────────────────
        prior_lows = [
            p for p in swing_points
            if p.kind == "low"
            and p.bar_idx < i
            and (i - p.bar_idx) <= lookback
        ]

        if not prior_highs and not prior_lows:
            continue

        close = float(closes[i])

        # ── Bullish break: close above most recent swing HIGH ─────────────────
        if prior_highs:
            last_high = max(prior_highs, key=lambda p: p.bar_idx)

            # Was the prior trend bearish? (most recent swing LOW was AFTER last high)
            # If yes, breaking above swing high = MSS (trend reversal to bullish)
            is_mss = False
            if prior_lows:
                last_low = max(prior_lows, key=lambda p: p.bar_idx)
                # If the most recent structure was lower lows (bearish), this is a reversal
                is_mss = last_low.bar_idx > last_high.bar_idx

            if close > last_high.price:
                breaks.append(StructureBreak(
                    break_bar_idx = i,
                    break_price   = close,
                    swing_level   = last_high.price,
                    direction     = "bullish",
                    is_mss        = is_mss,
                    time          = df.index[i],
                ))

        # ── Bearish break: close below most recent swing LOW ──────────────────
        if prior_lows:
            last_low = max(prior_lows, key=lambda p: p.bar_idx)

            is_mss = False
            if prior_highs:
                last_high = max(prior_highs, key=lambda p: p.bar_idx)
                # If the most recent structure was higher highs (bullish), this is a reversal
                is_mss = last_high.bar_idx > last_low.bar_idx

            if close < last_low.price:
                breaks.append(StructureBreak(
                    break_bar_idx = i,
                    break_price   = close,
                    swing_level   = last_low.price,
                    direction     = "bearish",
                    is_mss        = is_mss,
                    time          = df.index[i],
                ))

    return sorted(breaks, key=lambda b: b.break_bar_idx)


# ── Bias from most recent structure break ─────────────────────────────────────

def get_current_bias(
    df: pd.DataFrame,
    swing_len: int = DEFAULT_SWING_LEN,
    lookback: int  = DEFAULT_LOOKBACK,
    min_breaks: int = 1,
) -> MarketBias:
    """
    Determine the current directional bias for the entire DataFrame.

    Algorithm:
    1. Detect swing highs/lows.
    2. Detect structure breaks from those swings.
    3. Look at the MOST RECENT break to determine current bias.
    4. If no break exists within lookback → NEUTRAL (not enough structure).

    This is the main function used by the live bot and backtest engine.

    Parameters
    ----------
    df        : Closed OHLCV bars (M5 or H1).
    swing_len : Bars on each side to confirm a swing.
    lookback  : Max bars to search backward for bias.
    min_breaks: Minimum number of breaks needed to trust the bias.

    Returns
    -------
    MarketBias enum value.
    """
    swings = detect_swing_points(df, swing_len=swing_len)
    breaks = detect_structure_breaks(df, swings, lookback=lookback)

    if len(breaks) < min_breaks:
        return MarketBias.NEUTRAL

    # Most recent break determines current bias
    last_break = breaks[-1]

    # Check for conflicting breaks in recent history (→ ranging)
    recent = [b for b in breaks if (len(df) - 1 - b.break_bar_idx) <= lookback // 2]
    if len(recent) >= 2:
        directions = {b.direction for b in recent[-4:]}
        if "bullish" in directions and "bearish" in directions:
            # Both directions recently broken → choppy/ranging
            return MarketBias.RANGING

    return MarketBias.BULLISH if last_break.direction == "bullish" else MarketBias.BEARISH


# ── Convenience: add bias column to DataFrame (for backtesting) ───────────────

def add_structure_bias_column(
    df: pd.DataFrame,
    swing_len: int = DEFAULT_SWING_LEN,
    lookback: int  = DEFAULT_LOOKBACK,
) -> pd.DataFrame:
    """
    Add a 'structure_bias' column to df with rolling bias at each bar.

    WHY: Backtesting needs to know the bias at the TIME of each bar,
    not the bias at end-of-dataset.  We compute bias over df.iloc[:i+1]
    for every bar i.  This is O(n²) but n is typically ~500 bars for M5.

    Returns a copy with 'structure_bias' column added (string values).
    """
    df = df.copy()
    biases = []

    # Pre-compute all swings once for speed; breaks are re-scanned per bar
    all_swings = detect_swing_points(df, swing_len=swing_len)

    for i in range(len(df)):
        # Only use bars up to and including i (no lookahead)
        swings_before = [s for s in all_swings if s.bar_idx <= i]
        breaks_before = detect_structure_breaks(df.iloc[: i + 1], swings_before, lookback=lookback)

        if not breaks_before:
            biases.append(MarketBias.NEUTRAL.value)
        else:
            last = breaks_before[-1]
            recent = [b for b in breaks_before if (i - b.break_bar_idx) <= lookback // 2]
            if len(recent) >= 2:
                dirs = {b.direction for b in recent[-4:]}
                if "bullish" in dirs and "bearish" in dirs:
                    biases.append(MarketBias.RANGING.value)
                    continue
            biases.append(last.direction)

    df["structure_bias"] = biases
    return df
