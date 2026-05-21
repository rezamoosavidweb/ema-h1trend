"""
EMA H1 trend + M5 pending-stop signal math (notebook + MT5 parity).

Used by notebooks/03_strategy03_crypto.ipynb and mt5/run_strategy03_errante.py.
"""

from __future__ import annotations

from datetime import timezone
from zoneinfo import ZoneInfo

import pandas as pd

EMA_FAST = 8
EMA_MID = 13
EMA_SLOW = 21

MIN_WARMUP_BARS_M5 = 200
MIN_WARMUP_BARS_H1 = 200


# ═══════════════════════════════════════════════════════════════════════════
# BROKER TIMEZONE  (paired ingest + egress translation -- both halves must
# agree, or pending expirations land in the past and MT5 returns 10022).
# ═══════════════════════════════════════════════════════════════════════════
#
# MT5's Python `copy_rates_*` returns `time` as an int64 second-count, but
# the integer encodes the BROKER SERVER's wall clock formatted as if it
# were a UTC epoch second. On an EET/EEST broker (Errante) those seconds
# are 2-3h ahead of real UTC.
#
# The SAME mislabel convention is used in the reverse direction by
# `order_send`: the `expiration` field is also an integer that MT5
# interprets as broker wall clock, NOT real UTC. So we need a matched
# pair of translators:
#
#     _mt5_seconds_to_utc(int)     -- inbound  (MT5 -> real UTC)
#     _utc_to_mt5_broker_seconds() -- outbound (real UTC -> MT5 wire)
#
# Why this matters NOW: the previous code in run_strategy03_errante.py
# called `int(expiry_utc.timestamp())` directly. That accidentally
# produced the right wire value ONLY because `rates_to_ohlcv_df` had a
# matching ingest bug -- bar timestamps were broker-mislabeled-UTC, so
# `bar_time + 60min` was also broker-mislabeled-UTC, and `.timestamp()`
# happened to emit broker-seconds. The two bugs cancelled perfectly.
#
# Once the ingest side is corrected here, bar_time becomes real UTC; the
# egress side MUST be corrected in lock-step or every pending order will
# be rejected with "Invalid expiration".
BROKER_TZ = ZoneInfo("Europe/Athens")


def _mt5_seconds_to_utc(seconds):
    """
    INBOUND: MT5 `time` int (broker wall-clock as Unix seconds) -> real UTC
    pandas Timestamp / Series. DST-aware via BROKER_TZ.

    `nonexistent='shift_forward'` and `ambiguous='infer'` keep this robust
    around the two DST transitions per year; broker bars do not normally
    fall in the spring-forward gap, but tagging this explicitly keeps the
    behaviour deterministic if it ever does.
    """
    naive = pd.to_datetime(seconds, unit="s")
    if hasattr(naive, "dt"):  # pandas Series -- 'infer' needs monotonic data, which MT5 bars are.
        return (
            naive.dt.tz_localize(BROKER_TZ, nonexistent="shift_forward",
                                 ambiguous="infer")
                 .dt.tz_convert("UTC")
        )
    # scalar Timestamp: pandas scalar tz_localize does NOT accept 'infer'.
    return (
        naive.tz_localize(BROKER_TZ, nonexistent="shift_forward", ambiguous=False)
             .tz_convert("UTC")
    )


def _utc_to_mt5_broker_seconds(utc_dt) -> int:
    """
    OUTBOUND: real-UTC datetime/Timestamp -> the Unix-seconds integer that
    MT5 expects on the wire (broker wall-clock encoded as if it were UTC).

    This is the exact inverse of `_mt5_seconds_to_utc`:
        UTC -> astimezone(BROKER_TZ)  -- DST-aware broker wall clock
            -> replace tzinfo with UTC -- re-encode wall clock as the MT5
                                          wire convention (no time shift)
            -> .timestamp() -> int
    """
    if isinstance(utc_dt, pd.Timestamp):
        utc_dt = utc_dt.to_pydatetime()
    if utc_dt.tzinfo is None:
        # Treat naive as already-UTC (caller's contract).
        utc_dt = utc_dt.replace(tzinfo=timezone.utc)
    broker_aware = utc_dt.astimezone(BROKER_TZ)
    # Replace the +XX:XX offset label with +00:00 WITHOUT shifting the
    # wall-clock components -- this yields the same encoding MT5 hands
    # us on the inbound side, so the round-trip is exact.
    naive_as_utc = broker_aware.replace(tzinfo=timezone.utc)
    return int(naive_as_utc.timestamp())


def default_crypto_tick(sym: str) -> float:
    """Default price tick step by symbol prefix; override when broker step differs."""
    u = sym.upper()
    if u.startswith("BTC"):
        return 1.0
    if u.startswith("ETH"):
        return 0.01
    if u.startswith(("XRP", "DOGE", "ADA")):
        return 0.0001
    if u.startswith(("BNB", "BCH", "LTC", "SOL")):
        return 0.01
    return 0.01


def add_emas(df: pd.DataFrame) -> pd.DataFrame:
    """Add ema_8 / ema_13 / ema_21 on column close."""
    out = df.copy()
    out[f"ema_{EMA_FAST}"] = out["close"].ewm(span=EMA_FAST, adjust=False).mean()
    out[f"ema_{EMA_MID}"] = out["close"].ewm(span=EMA_MID, adjust=False).mean()
    out[f"ema_{EMA_SLOW}"] = out["close"].ewm(span=EMA_SLOW, adjust=False).mean()
    return out


def h1_trend_series(h1: pd.DataFrame) -> pd.Series:
    """bull / bear / flat per H1 bar (same rules as notebook)."""
    h = h1.copy()
    trend = pd.Series("flat", index=h.index, dtype=object)
    bull = (
        (h[f"ema_{EMA_FAST}"] > h[f"ema_{EMA_MID}"])
        & (h[f"ema_{EMA_MID}"] > h[f"ema_{EMA_SLOW}"])
        & (h["close"] > h[f"ema_{EMA_SLOW}"])
    )
    bear = (
        (h[f"ema_{EMA_FAST}"] < h[f"ema_{EMA_MID}"])
        & (h[f"ema_{EMA_MID}"] < h[f"ema_{EMA_SLOW}"])
        & (h["close"] < h[f"ema_{EMA_SLOW}"])
    )
    trend.loc[bull] = "bull"
    trend.loc[bear] = "bear"
    return trend


def merge_h1_trend_onto_m5(m5: pd.DataFrame, h1: pd.DataFrame) -> pd.DataFrame:
    """
    Attach H1 trend + H1 EMA columns to each M5 row (merge_asof backward).
    suffixes match notebook: M5 keeps plain ema_* , H1 columns get _h1 if overlapping.
    """
    h1_sig = h1.copy()
    h1_sig["trend"] = h1_trend_series(h1_sig)
    cols = ["trend", f"ema_{EMA_FAST}", f"ema_{EMA_MID}", f"ema_{EMA_SLOW}"]
    return pd.merge_asof(
        m5.sort_index(),
        h1_sig[cols].sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",
        suffixes=("", "_h1"),
    )


def rates_to_ohlcv_df(rates) -> pd.DataFrame:
    """MT5 copy_rates_* structured array -> OHLCV DataFrame indexed UTC."""
    df = pd.DataFrame(rates)
    if df.empty:
        return df
    # MT5 returns broker-local wall-clock as a Unix-seconds integer; the old
    # `pd.to_datetime(..., utc=True)` call mislabeled it. See _mt5_seconds_to_utc.
    df["time"] = _mt5_seconds_to_utc(df["time"])
    df = df.rename(columns={"tick_volume": "volume"})
    df = df.set_index("time").sort_index()
    return df[["open", "high", "low", "close", "volume"]]


def compute_pending_setup(
    m5_ctx: pd.DataFrame,
    *,
    bar_index: int,
    lookback_bars: int,
    pending_offset_ticks: float,
    pip_size: float,
    rr: float,
    balance: float,
    risk_per_trade: float,
) -> dict | None:
    """Pending setup at closed bar bar_index; None if flat trend or insufficient rows."""
    if bar_index < lookback_bars:
        return None

    row = m5_ctx.iloc[bar_index]
    window = m5_ctx.iloc[bar_index - lookback_bars : bar_index]
    hh = float(window["high"].max())
    ll = float(window["low"].min())

    trend = row.get("trend", "flat")
    if trend not in ("bull", "bear"):
        return None

    offset = float(pending_offset_ticks) * float(pip_size)

    if trend == "bull":
        entry = hh + offset
        sl = ll - offset
        risk_per_unit = max(entry - sl, 1e-12)
        tp = entry + rr * risk_per_unit
        risk_cash = balance * risk_per_trade
        qty = risk_cash / risk_per_unit
        return {
            "side": "buy",
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "qty": float(qty),
        }

    entry = ll - offset
    sl = hh + offset
    risk_per_unit = max(sl - entry, 1e-12)
    tp = entry - rr * risk_per_unit
    risk_cash = balance * risk_per_trade
    qty = risk_cash / risk_per_unit
    return {
        "side": "sell",
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "qty": float(qty),
    }


def last_closed_bar_index(
    closed_m5: pd.DataFrame,
    lookback_bars: int,
    *,
    min_warmup_bars: int = MIN_WARMUP_BARS_M5,
) -> int | None:
    need = max(lookback_bars + 1, min_warmup_bars)
    if len(closed_m5) < need:
        return None
    return len(closed_m5) - 1


def min_bars_needed_for_signal(
    lookback_bars: int,
    *,
    min_warmup_bars: int = MIN_WARMUP_BARS_M5,
) -> int:
    return max(lookback_bars + 1, min_warmup_bars)
