"""
OHLCV quality checks and data-quality reports.

Used after every download / cache hit before the data is fed to the
strategy / backtester.  Detects:

* missing OHLCV columns
* non-monotonic timestamps
* duplicate timestamps
* missing candles (gaps larger than expected timeframe)
* OHLC integrity violations (High < Low, etc.)
* NaN / Inf values in price columns
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

from utils.logging_setup import get_logger
from utils.timeframes import timeframe_to_minutes

log = get_logger(__name__)


@dataclass
class DataQualityReport:
    """Structured quality report for one OHLCV DataFrame."""
    bars: int
    date_from: pd.Timestamp | None
    date_to:   pd.Timestamp | None
    duplicate_timestamps: int
    non_monotonic_count:  int
    nan_rows:             int
    ohlc_violations:      int
    missing_candles:      int
    issues:               List[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return (
            self.duplicate_timestamps == 0 and
            self.non_monotonic_count == 0 and
            self.nan_rows == 0 and
            self.ohlc_violations == 0
        )

    def to_dict(self) -> dict:
        return {
            "bars":                 self.bars,
            "date_from":            self.date_from,
            "date_to":              self.date_to,
            "duplicate_timestamps": self.duplicate_timestamps,
            "non_monotonic_count":  self.non_monotonic_count,
            "nan_rows":             self.nan_rows,
            "ohlc_violations":      self.ohlc_violations,
            "missing_candles":      self.missing_candles,
            "is_clean":             self.is_clean,
            "issues":               list(self.issues),
        }


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

REQUIRED_COLS = ("time", "Open", "High", "Low", "Close", "Volume")


def validate_ohlcv(
    df: pd.DataFrame,
    timeframe: str | None = None,
) -> DataQualityReport:
    """
    Run every quality check on *df* and return a :class:`DataQualityReport`.

    Parameters
    ----------
    df:
        OHLCV DataFrame.
    timeframe:
        Optional canonical timeframe string (``"5m"``, ``"H1"``, …) used
        for missing-candle detection.  When ``None``, the candle interval
        is inferred from the median time-delta.
    """
    issues: List[str] = []

    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        issues.append(f"missing columns: {missing_cols}")
        return DataQualityReport(
            bars=len(df), date_from=None, date_to=None,
            duplicate_timestamps=0, non_monotonic_count=0,
            nan_rows=0, ohlc_violations=0, missing_candles=0,
            issues=issues,
        )

    times = pd.to_datetime(df["time"], utc=True)

    # Duplicates
    dup = int(times.duplicated().sum())
    if dup:
        issues.append(f"{dup} duplicate timestamps")

    # Monotonicity
    non_mono = int((times.diff().dt.total_seconds() < 0).sum())
    if non_mono:
        issues.append(f"{non_mono} non-monotonic timestamps")

    # NaNs in OHLC
    price_cols = ["Open", "High", "Low", "Close"]
    nan_rows = int(df[price_cols].isna().any(axis=1).sum())
    if nan_rows:
        issues.append(f"{nan_rows} rows with NaN price values")

    # OHLC integrity
    ohlc_bad = (
        (df["High"] < df["Low"]) |
        (df["High"] < df["Open"]) |
        (df["High"] < df["Close"]) |
        (df["Low"]  > df["Open"]) |
        (df["Low"]  > df["Close"]) |
        (df[price_cols] <= 0).any(axis=1)
    )
    n_ohlc_bad = int(ohlc_bad.sum())
    if n_ohlc_bad:
        issues.append(f"{n_ohlc_bad} bars with OHLC integrity violations")

    # Missing candles
    n_missing = detect_missing_candles(df, timeframe=timeframe)
    if n_missing:
        issues.append(f"{n_missing} missing candles detected")

    return DataQualityReport(
        bars=len(df),
        date_from=times.min() if len(df) else None,
        date_to=times.max() if len(df) else None,
        duplicate_timestamps=dup,
        non_monotonic_count=non_mono,
        nan_rows=nan_rows,
        ohlc_violations=n_ohlc_bad,
        missing_candles=n_missing,
        issues=issues,
    )


def detect_missing_candles(
    df: pd.DataFrame,
    timeframe: str | None = None,
    tolerance: float = 1.5,
) -> int:
    """
    Return the count of candles missing under the expected interval.

    A candle is considered missing when the time-delta to the previous
    bar exceeds ``tolerance × expected_interval``.  Weekend/market-close
    gaps for forex are handled by the caller (or by
    :class:`core.DataCleaner`).
    """
    if "time" not in df.columns or len(df) < 3:
        return 0
    times = pd.to_datetime(df["time"], utc=True).sort_values().reset_index(drop=True)
    deltas = times.diff().dt.total_seconds().iloc[1:]

    if timeframe is not None:
        expected = timeframe_to_minutes(timeframe) * 60
    else:
        expected = float(deltas.median())

    if expected <= 0:
        return 0
    threshold = tolerance * expected
    n_gaps = int((deltas > threshold).sum())
    # Approximate count of missing bars
    missing = int(((deltas[deltas > threshold] / expected) - 1).sum())
    return max(missing, n_gaps)
