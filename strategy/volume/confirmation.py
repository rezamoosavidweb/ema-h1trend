"""
Volume confirmation — relative volume spikes and displacement quality.

WHY THIS EXISTS
---------------
Not all OB displacements are created equal.  A displacement on 3× average
volume is far more significant than one on 0.5× volume.  High volume during
a displacement confirms:
  1. Real institutional participation (not a thin-market false move).
  2. Strong conviction behind the move.
  3. Higher probability that the OB will be defended on retest.

Crypto markets use tick volume (number of trades) since true bid-ask volume
is not available in OHLCV data from Bybit.  Tick volume is a valid proxy for
activity level.

HOW IT IS USED
--------------
  ctx = compute_volume_context(df, ma_period=20)
  if displacement_has_volume(df, disp_start=42, disp_end=47, ctx, mult=1.5):
      # Displacement had above-average volume → higher quality OB
      generate_signal(...)
  else:
      # Low-volume displacement → skip or reduce confidence score

TYPICAL THRESHOLDS FOR XAUUSDT M5
-----------------------------------
  Normal volume multiplier: 1.5–2.0× MA
  Strong displacement:       > 2.0× MA
  Weak / skip:               < 1.0× MA
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class VolumeContext:
    """
    Pre-computed volume statistics for a DataFrame.

    Compute once per cycle and reuse across multiple OB checks.
    """
    ma_period     : int
    volume_ma     : pd.Series     # rolling mean of volume
    volume_std    : pd.Series     # rolling std of volume
    volume_ratio  : pd.Series     # volume / volume_ma (relative volume)
    current_ratio : float         # most recent bar's relative volume
    current_ma    : float         # most recent rolling MA value


# ── Core computation ──────────────────────────────────────────────────────────

def compute_volume_context(
    df: pd.DataFrame,
    ma_period: int = 20,
) -> VolumeContext:
    """
    Compute rolling volume statistics for the DataFrame.

    Parameters
    ----------
    df        : OHLCV DataFrame with 'volume' column.
    ma_period : Rolling window for volume moving average.

    Returns
    -------
    VolumeContext with MA, std, and ratio series.
    """
    vol = df["volume"].astype(float)
    vol_ma  = vol.rolling(ma_period, min_periods=1).mean()
    vol_std = vol.rolling(ma_period, min_periods=1).std().fillna(0)

    # Avoid division by zero on flat-volume symbols
    vol_ratio = vol / vol_ma.replace(0, np.nan)
    vol_ratio = vol_ratio.fillna(1.0)

    return VolumeContext(
        ma_period    = ma_period,
        volume_ma    = vol_ma,
        volume_std   = vol_std,
        volume_ratio = vol_ratio,
        current_ratio= float(vol_ratio.iloc[-1]),
        current_ma   = float(vol_ma.iloc[-1]),
    )


def is_volume_spike(
    bar_idx: int,
    ctx: VolumeContext,
    multiplier: float = 1.5,
) -> bool:
    """
    True if the volume at bar_idx is >= multiplier × rolling MA.

    Parameters
    ----------
    bar_idx    : Integer index into the DataFrame.
    ctx        : Pre-computed VolumeContext.
    multiplier : Threshold (e.g. 1.5 = 150% of average volume).
    """
    if bar_idx < 0 or bar_idx >= len(ctx.volume_ratio):
        return False
    return float(ctx.volume_ratio.iloc[bar_idx]) >= multiplier


def displacement_has_volume(
    df: pd.DataFrame,
    disp_start: int,
    disp_end: int,
    ctx: Optional[VolumeContext] = None,
    multiplier: float = 1.3,
    ma_period: int    = 20,
    require_all: bool = False,
) -> bool:
    """
    True if the displacement segment [disp_start, disp_end) had above-average volume.

    We check the AVERAGE volume of the displacement candles, not just one bar.
    This is more robust than checking a single candle (avoids one-candle flukes).

    Parameters
    ----------
    df          : OHLCV DataFrame.
    disp_start  : First bar index of the displacement.
    disp_end    : Last bar index + 1 (exclusive).
    ctx         : Pre-computed VolumeContext.  If None, computed here.
    multiplier  : Required avg_volume / MA ratio.
    ma_period   : MA period (used if ctx is None).
    require_all : If True, ALL bars must have volume >= multiplier × MA.
                  If False (default), AVERAGE volume of segment must qualify.

    Returns
    -------
    bool
    """
    if ctx is None:
        ctx = compute_volume_context(df, ma_period=ma_period)

    if disp_start >= disp_end or disp_end > len(df):
        return False

    segment_ratios = ctx.volume_ratio.iloc[disp_start:disp_end].values

    if require_all:
        return bool(np.all(segment_ratios >= multiplier))
    else:
        return float(np.mean(segment_ratios)) >= multiplier


# ── DataFrame annotation (for notebooks) ─────────────────────────────────────

def add_volume_columns(
    df: pd.DataFrame,
    ma_period: int = 20,
    spike_multiplier: float = 1.5,
) -> pd.DataFrame:
    """
    Add 'volume_ma', 'volume_ratio', 'volume_spike' columns to df.

    Useful for charting and manual inspection in notebooks.
    """
    df = df.copy()
    ctx = compute_volume_context(df, ma_period=ma_period)

    df["volume_ma"]    = ctx.volume_ma
    df["volume_ratio"] = ctx.volume_ratio
    df["volume_spike"] = ctx.volume_ratio >= spike_multiplier

    return df
