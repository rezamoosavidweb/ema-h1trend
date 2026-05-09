"""Numerically-safe math helpers used across the system."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd


def safe_divide(num: float, denom: float, default: float = float("nan")) -> float:
    """Return ``num / denom`` or *default* when the result is undefined."""
    if not (math.isfinite(num) and math.isfinite(denom)):
        return default
    if denom == 0.0:
        return default
    out = num / denom
    return out if math.isfinite(out) else default


def clip_finite(arr: np.ndarray, replacement: float = 0.0) -> np.ndarray:
    """Replace ±Inf and NaN with *replacement* (returns a copy)."""
    out = np.asarray(arr, dtype=np.float64).copy()
    out[~np.isfinite(out)] = replacement
    return out


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score with ddof=0 mean/std and NaN-safe denominator."""
    rolling = series.rolling(window=window, min_periods=window)
    mean = rolling.mean()
    std = rolling.std(ddof=0)
    z = (series - mean) / std.replace(0.0, np.nan)
    return z


def rolling_percentile(series: pd.Series, window: int, q: float = 0.5) -> pd.Series:
    """Rolling quantile (0 ≤ q ≤ 1) with strict min_periods."""
    return series.rolling(window=window, min_periods=window).quantile(q)
