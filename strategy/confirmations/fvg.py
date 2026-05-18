"""
Fair Value Gap (FVG) detection and validation.

WHY THIS EXISTS
---------------
The original ob_core.py mentions "FVG" in comments but does NOT actually
validate whether a Fair Value Gap exists.  A real FVG is an imbalance where
price moved so fast that a gap was left between the wicks of candle[i] and
candle[i+2].  This gap is often revisited when price retests an Order Block.

Requiring a FVG as confirmation dramatically reduces false OB signals because:
  - No FVG = price moved slowly, no real institutional imbalance was left.
  - FVG present = aggressive displacement with measurable imbalance = higher
    probability that smart money is defending that level on retest.

CONCEPT
-------
Bullish FVG (created during an up-move):
    candle[i].high  < candle[i+2].low
    ↑ the high of the pre-displacement bar is BELOW the low of the post bar.
    → An unfilled gap in the upward direction.

Bearish FVG (created during a down-move):
    candle[i].low   > candle[i+2].high
    ↑ the low of the pre-displacement bar is ABOVE the high of the post bar.
    → An unfilled gap in the downward direction.

USAGE IN OB WORKFLOW
---------------------
  # Check if displacement between bar ob_idx and ob_idx+4 left an FVG
  fvg = find_fvg_for_ob(df, ob_bar_idx=42, direction='bullish', search_bars=6)
  if fvg is not None:
      print(f"FVG: {fvg.gap_low:.2f} → {fvg.gap_high:.2f}")
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import pandas as pd


# ── Data models ───────────────────────────────────────────────────────────────

class FVGType(Enum):
    BULLISH = "bullish"   # gap points upward — left during a bullish move
    BEARISH = "bearish"   # gap points downward — left during a bearish move


@dataclass
class FVG:
    """A confirmed Fair Value Gap."""
    bar_idx  : int         # index of candle[i] (the anchor bar of the three-candle pattern)
    fvg_type : FVGType
    gap_low  : float       # lower boundary of the imbalance zone
    gap_high : float       # upper boundary of the imbalance zone
    gap_size : float       # gap_high - gap_low (absolute distance)
    filled   : bool        # True if price has since traded through the entire gap
    time     : pd.Timestamp


# ── Core detection ────────────────────────────────────────────────────────────

def detect_fvgs(
    df: pd.DataFrame,
    min_gap_atr_fraction: float = 0.0,
) -> List[FVG]:
    """
    Scan all bars for three-candle Fair Value Gaps.

    For each trio (i, i+1, i+2):
      Bullish FVG : df['high'].iloc[i]  < df['low'].iloc[i+2]
      Bearish FVG : df['low'].iloc[i]   > df['high'].iloc[i+2]

    The middle candle (i+1) is the one that causes the displacement — it's
    typically a strong-body candle with little to no overlap with candles i and i+2.

    Parameters
    ----------
    df                  : Closed OHLCV bars with 'atr' column (add via add_candle_features).
    min_gap_atr_fraction: Minimum gap size as a fraction of ATR.  Set to 0 to accept all.
                          Example: 0.25 means gap must be at least 0.25 * ATR(14).

    Returns
    -------
    List of FVG objects sorted by bar_idx ascending.
    """
    fvgs: List[FVG] = []
    n = len(df)
    has_atr = "atr" in df.columns

    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    atrs   = df["atr"].values if has_atr else None

    for i in range(n - 2):
        atr_val = float(atrs[i + 1]) if atrs is not None else 0.0
        min_gap = atr_val * min_gap_atr_fraction

        # ── Bullish FVG ───────────────────────────────────────────────────────
        #   Gap between top of candle[i] and bottom of candle[i+2]
        if highs[i] < lows[i + 2]:
            gap_low  = float(highs[i])
            gap_high = float(lows[i + 2])
            gap_size = gap_high - gap_low

            if gap_size >= min_gap:
                # Check if already filled by subsequent price action
                filled = _is_filled_bullish(closes, lows, i + 2, gap_low, n)
                fvgs.append(FVG(
                    bar_idx  = i,
                    fvg_type = FVGType.BULLISH,
                    gap_low  = gap_low,
                    gap_high = gap_high,
                    gap_size = gap_size,
                    filled   = filled,
                    time     = df.index[i],
                ))

        # ── Bearish FVG ───────────────────────────────────────────────────────
        #   Gap between bottom of candle[i] and top of candle[i+2]
        elif lows[i] > highs[i + 2]:
            gap_low  = float(highs[i + 2])
            gap_high = float(lows[i])
            gap_size = gap_high - gap_low

            if gap_size >= min_gap:
                filled = _is_filled_bearish(closes, highs, i + 2, gap_high, n)
                fvgs.append(FVG(
                    bar_idx  = i,
                    fvg_type = FVGType.BEARISH,
                    gap_low  = gap_low,
                    gap_high = gap_high,
                    gap_size = gap_size,
                    filled   = filled,
                    time     = df.index[i],
                ))

    return fvgs


def _is_filled_bullish(closes, lows, start_idx: int, gap_low: float, n: int) -> bool:
    """True if a close below gap_low appeared after the bullish FVG formed."""
    for j in range(start_idx + 1, n):
        if lows[j] < gap_low:
            return True
    return False


def _is_filled_bearish(closes, highs, start_idx: int, gap_high: float, n: int) -> bool:
    """True if a close above gap_high appeared after the bearish FVG formed."""
    for j in range(start_idx + 1, n):
        if highs[j] > gap_high:
            return True
    return False


# ── OB-specific FVG search ────────────────────────────────────────────────────

def find_fvg_for_ob(
    df: pd.DataFrame,
    ob_bar_idx: int,
    direction: str,         # 'bullish' or 'bearish'
    search_bars: int = 8,
    min_gap_atr_fraction: float = 0.0,
) -> Optional[FVG]:
    """
    Find a FVG in the bars immediately AFTER the Order Block.

    When price displaces away from an OB, it often leaves a FVG in the 2–6 bars
    following the OB candle.  This function searches that window.

    Parameters
    ----------
    df               : Full OHLCV DataFrame (with 'atr').
    ob_bar_idx       : Index of the OB candle.
    direction        : 'bullish' (looking for bullish FVG) or 'bearish'.
    search_bars      : How many bars after OB to search.
    min_gap_atr_fraction: Minimum FVG size as fraction of ATR.

    Returns
    -------
    The LARGEST unfilled FVG in the search window, or None.
    """
    start = ob_bar_idx + 1
    end   = min(ob_bar_idx + search_bars + 1, len(df))

    if end - start < 3:
        return None

    window = df.iloc[start:end]
    fvgs   = detect_fvgs(window, min_gap_atr_fraction=min_gap_atr_fraction)

    # Filter to the correct direction and non-filled only
    target_type = FVGType.BULLISH if direction == "bullish" else FVGType.BEARISH
    matching = [f for f in fvgs if f.fvg_type == target_type and not f.filled]

    if not matching:
        return None

    # Return the largest gap (most significant imbalance)
    return max(matching, key=lambda f: f.gap_size)


def has_displacement_fvg(
    df: pd.DataFrame,
    ob_bar_idx: int,
    direction: str,
    search_bars: int = 8,
    min_gap_atr_fraction: float = 0.0,
) -> bool:
    """
    Convenience boolean: True if there is a valid FVG after the OB displacement.

    Used as a filter in list_ob_signals() to require imbalance confirmation.
    """
    return find_fvg_for_ob(
        df, ob_bar_idx, direction,
        search_bars=search_bars,
        min_gap_atr_fraction=min_gap_atr_fraction,
    ) is not None


# ── DataFrame annotation (for notebooks / visualization) ─────────────────────

def add_fvg_columns(df: pd.DataFrame, **detect_kwargs) -> pd.DataFrame:
    """
    Add 'fvg_type', 'fvg_low', 'fvg_high' columns for visualization.

    Non-FVG bars get NaN in these columns.
    """
    df = df.copy()
    df["fvg_type"] = None
    df["fvg_low"]  = float("nan")
    df["fvg_high"] = float("nan")

    fvgs = detect_fvgs(df, **detect_kwargs)
    for fvg in fvgs:
        idx = fvg.bar_idx
        df.at[df.index[idx], "fvg_type"] = fvg.fvg_type.value
        df.at[df.index[idx], "fvg_low"]  = fvg.gap_low
        df.at[df.index[idx], "fvg_high"] = fvg.gap_high

    return df
