"""
MT5 OHLCV fetch + Nicosia timezone normalisation.

Why a separate module?
    The MT5 `time` field is "broker wall-clock encoded as Unix seconds" — not
    real UTC. We relabel as Europe/Nicosia so cointegration tests and rolling
    z-scores align with the CSV cache that notebook 25 produced (which had
    the same relabel applied; see project memory).

Public API:
    fetch_close_bars(symbol, tf, n_bars)             -> pd.Series         (MT5)
    fetch_aligned_panel(symbols, tf, n_bars)         -> pd.DataFrame      (MT5)
    fetch_aligned_panel_from_csv(symbols, tf, n_bars, csv_root)  -> pd.DataFrame
    seconds_until_next_bar_close(tf, grace)          -> float

The CSV path is ONLY for `--dry-run` testing when the broker can't stream all
symbols. Production code must use the MT5 path so it sees actual market state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import MetaTrader5 as mt5
import pandas as pd

from .config import BROKER_TZ_NAME, TF_HOURS


# Map TF name to MT5 timeframe constant; lazily resolved against the live mt5
# module so we don't accidentally rely on a numeric value that ships differently
# across MT5 module versions.
def _mt5_tf(tf: str) -> int:
    if tf == "H1":
        return mt5.TIMEFRAME_H1
    if tf == "H4":
        return mt5.TIMEFRAME_H4
    if tf == "D1":
        return mt5.TIMEFRAME_D1
    raise ValueError(f"unsupported timeframe {tf!r} (expected one of H1/H4/D1)")


BROKER_TZ = ZoneInfo(BROKER_TZ_NAME)


# ─────────────────────────────────────────────────────────────────────────────
# Time helpers
# ─────────────────────────────────────────────────────────────────────────────


def bar_close_to_local(seconds) -> pd.Series | pd.Timestamp:
    """
    Convert MT5 `time` (broker wall-clock as Unix seconds) to tz-aware Series
    in BROKER_TZ. Accepts scalar or array-like.

    Notes
    -----
    The integer is NOT real Unix seconds — it is the broker's clock displayed
    as if it were UTC. We:

        1. parse as naive datetime (treat as wall-clock)
        2. localise to BROKER_TZ (no shift — same wall-clock readings)

    This matches the CSV cache produced by notebook 25/29 so live and backtest
    panels align bar-for-bar.
    """
    parsed = pd.to_datetime(seconds, unit="s")  # naive — no tz
    if hasattr(parsed, "dt"):
        return parsed.dt.tz_localize(
            BROKER_TZ, nonexistent="shift_forward", ambiguous="NaT"
        )
    return parsed.tz_localize(
        BROKER_TZ, nonexistent="shift_forward", ambiguous=False
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fetch
# ─────────────────────────────────────────────────────────────────────────────


def fetch_close_bars(symbol: str, tf: str, n_bars: int) -> pd.Series:
    """
    Fetch the most recent `n_bars` close prices for `symbol` at timeframe `tf`.

    Returns a tz-aware Series indexed by bar-close time (BROKER_TZ).
    Drops the still-forming bar (`iloc[:-1]`) so callers only see fully-closed
    bars — required for any backward-looking statistic.

    Raises RuntimeError on broker fetch failure.
    """
    if n_bars < 2:
        raise ValueError("fetch_close_bars: n_bars must be >= 2 (we drop the open bar)")

    rates = mt5.copy_rates_from_pos(symbol, _mt5_tf(tf), 0, n_bars + 1)
    if rates is None or len(rates) == 0:
        raise RuntimeError(
            f"copy_rates_from_pos({symbol}, {tf}, 0, {n_bars + 1}) returned None: "
            f"{mt5.last_error()}"
        )

    df = pd.DataFrame(rates)
    times = bar_close_to_local(df["time"])
    s = pd.Series(df["close"].values, index=times, name=symbol)
    s = s[s.index.notna()].sort_index()
    return s.iloc[:-1].iloc[-n_bars:]   # drop forming bar, then trim to n_bars


def fetch_aligned_panel(symbols: Iterable[str], tf: str, n_bars: int) -> pd.DataFrame:
    """
    Fetch `n_bars` closes for each symbol and inner-join on bar-close timestamp.

    Returns a (rows × len(symbols)) DataFrame with NO NaN rows. The number of
    rows is <= `n_bars` (gaps reduce overlap).

    Raises RuntimeError if any single symbol fetch fails — partial panels are
    not allowed (would silently bias subsequent stats).
    """
    series: dict[str, pd.Series] = {}
    for sym in symbols:
        series[sym] = fetch_close_bars(sym, tf, n_bars)

    panel = pd.concat(series, axis=1, sort=True).dropna()

    if panel.empty:
        raise RuntimeError(
            f"fetch_aligned_panel: zero overlapping bars across {list(symbols)} "
            f"at {tf}. Check Market Watch — broker may not stream all symbols."
        )
    return panel


# ─────────────────────────────────────────────────────────────────────────────
# CSV fallback (dry-run only — never used in production)
# ─────────────────────────────────────────────────────────────────────────────


def _load_csv_h1(symbol: str, csv_root: Path) -> pd.Series:
    """Read the H1 OHLCV CSV produced by the data-fetcher notebook."""
    path = csv_root / symbol / "H1" / "ohlcv.csv"
    df = pd.read_csv(path, parse_dates=["time"])
    naive = df["time"].dt.tz_localize(None)
    ts = naive.dt.tz_localize(BROKER_TZ, ambiguous="NaT", nonexistent="NaT")
    s = pd.Series(df["close"].values, index=ts, name=symbol)
    return s[s.index.notna()].sort_index()


def fetch_aligned_panel_from_csv(
    symbols: Iterable[str], tf: str, n_bars: int, csv_root: Path,
) -> pd.DataFrame:
    """
    CSV-backed counterpart to `fetch_aligned_panel` for dry-run testing.

    Loads H1 cache, resamples to `tf` with MT5-compatible labelling
    (label='left', closed='left' — same as MT5's bar convention), inner-joins,
    returns the most recent `n_bars` rows.

    Same return contract as `fetch_aligned_panel`: tz-aware index, no NaN rows.
    """
    symbols = list(symbols)
    h1: dict[str, pd.Series] = {sym: _load_csv_h1(sym, csv_root) for sym in symbols}

    if tf != "H1":
        rule = {"H4": "4h", "D1": "1D"}[tf]
        h1 = {
            sym: s.resample(rule, label="left", closed="left").last().dropna()
            for sym, s in h1.items()
        }

    panel = pd.concat(h1, axis=1, sort=True).dropna()
    if panel.empty:
        raise RuntimeError(
            f"CSV fallback: zero overlapping rows across {symbols} at {tf}"
        )
    return panel.iloc[-n_bars:]


# ─────────────────────────────────────────────────────────────────────────────
# Cycle-timing helper
# ─────────────────────────────────────────────────────────────────────────────


def seconds_until_next_bar_close(tf: str, grace_seconds: int = 30) -> float:
    """
    Return real-wall seconds until the next close of TF + a grace period.

    Used by the runner's main loop to sleep efficiently between cycles.

    Example: at 13:47 Nicosia on H4, the next close is 16:00 → ~133 minutes.
    """
    hours_per_bar = TF_HOURS[tf]
    now = pd.Timestamp.now(tz=BROKER_TZ)
    next_bar_hour = (now.hour // hours_per_bar + 1) * hours_per_bar
    if next_bar_hour >= 24:
        next_close = now.normalize() + pd.Timedelta(days=1)
    else:
        next_close = now.normalize() + pd.Timedelta(hours=next_bar_hour)
    delta = (next_close - now).total_seconds() + grace_seconds
    return max(delta, 1.0)
