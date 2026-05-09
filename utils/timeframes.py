"""
Timeframe helpers — string ↔ minutes ↔ pandas offset alias conversions
plus utilities for synchronising a higher-timeframe series onto the
strategy's primary timeframe (without lookahead bias).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


# Canonical minute mapping used everywhere
_TF_MINUTES: Dict[str, int] = {
    "1m":  1,   "M1":  1,
    "3m":  3,   "M3":  3,
    "5m":  5,   "M5":  5,
    "15m": 15,  "M15": 15,
    "30m": 30,  "M30": 30,
    "1h":  60,  "H1":  60,
    "2h":  120, "H2":  120,
    "4h":  240, "H4":  240,
    "6h":  360, "H6":  360,
    "8h":  480, "H8":  480,
    "12h": 720, "H12": 720,
    "1d":  1440,    "D1":  1440,
    "1w":  10080,   "W1":  10080,
    "1M":  43200,   "MN1": 43200,
}

# Pandas resample-rule aliases (used by ``df.resample()``)
_TF_PD_OFFSET: Dict[str, str] = {
    "1m":  "1min",  "M1":  "1min",
    "3m":  "3min",  "M3":  "3min",
    "5m":  "5min",  "M5":  "5min",
    "15m": "15min", "M15": "15min",
    "30m": "30min", "M30": "30min",
    "1h":  "1h",    "H1":  "1h",
    "2h":  "2h",    "H2":  "2h",
    "4h":  "4h",    "H4":  "4h",
    "6h":  "6h",    "H6":  "6h",
    "8h":  "8h",    "H8":  "8h",
    "12h": "12h",   "H12": "12h",
    "1d":  "1D",    "D1":  "1D",
    "1w":  "1W",    "W1":  "1W",
    "1M":  "1ME",   "MN1": "1ME",
}


def timeframe_to_minutes(tf: str) -> int:
    """Return the timeframe length in minutes."""
    if tf not in _TF_MINUTES:
        raise ValueError(f"Unknown timeframe: {tf!r}")
    return _TF_MINUTES[tf]


def timeframe_to_pandas_offset(tf: str) -> str:
    """Return the pandas resample-rule alias for a timeframe string."""
    if tf not in _TF_PD_OFFSET:
        raise ValueError(f"Unknown timeframe: {tf!r}")
    return _TF_PD_OFFSET[tf]


def bars_per_year(tf: str, *, sessions_per_year: int = 252) -> int:
    """
    Annualisation factor used by Sharpe/Sortino calculations.

    For 24/7 markets (crypto) ``sessions_per_year`` should be set to 365.
    For forex/equity sessions the default of 252 trading days is correct.
    """
    minutes = timeframe_to_minutes(tf)
    minutes_per_session = 24 * 60
    return max(1, int(sessions_per_year * minutes_per_session / minutes))


def align_higher_timeframe(
    base_df: pd.DataFrame,
    higher_df: pd.DataFrame,
    *,
    on: str = "time",
    suffix: str = "_htf",
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Backward-merge a higher-timeframe DataFrame onto the lower-timeframe.

    Uses :func:`pandas.merge_asof` with ``direction='backward'`` so each
    base bar receives the latest **closed** higher-timeframe bar — no
    lookahead.

    Parameters
    ----------
    base_df:
        Lower timeframe DataFrame (e.g. 5-minute bars). Must contain ``on``.
    higher_df:
        Higher timeframe DataFrame (e.g. 1-hour bars). Must contain ``on``.
    on:
        Time column name in both frames.
    suffix:
        Suffix appended to higher-frame columns to disambiguate from base.
    columns:
        Subset of ``higher_df`` columns to merge.  ``None`` merges all.

    Returns
    -------
    pd.DataFrame
        ``base_df`` with the additional ``<col><suffix>`` columns.
    """
    if on not in base_df.columns or on not in higher_df.columns:
        raise ValueError(f"Both frames must contain a '{on}' column")

    base = base_df.sort_values(on).reset_index(drop=True)
    high = higher_df.sort_values(on).reset_index(drop=True)

    if columns is not None:
        high = high[[on] + [c for c in columns if c in high.columns]]

    high_renamed = high.rename(
        columns={c: f"{c}{suffix}" for c in high.columns if c != on}
    )

    merged = pd.merge_asof(
        base,
        high_renamed,
        on=on,
        direction="backward",
        allow_exact_matches=True,
    )
    return merged
