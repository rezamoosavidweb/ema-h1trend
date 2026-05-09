"""
Signal math aligned with notebooks/03_strategy03_crypto.ipynb (EMA 8/13/21, H1 trend, M5 pending stops).

Constants MIN_WARMUP_BARS_M5 / MIN_WARMUP_BARS_H1 (~200) gate signal generation until enough
closed candles exist so slow EMAs are usable (live MT5 fetches at least this much history).

Core logic: EMA 8/13/21, bull/bear/flat trend on H1, attach trend to M5, define Buy/Sell Stop
from HH/LL with offset and fixed RR (same as the notebook).
"""

from __future__ import annotations

import pandas as pd

# --- Indicator settings (notebook-aligned) ---
# EMA periods on close; used for H1 trend and M5 context.
EMA_FAST = 8
EMA_MID = 13
EMA_SLOW = 21

# Minimum closed bars before EMAs and trend are treated as reliable
# (slow EMA 21 stabilizes after several spans; ~200 bars is a safe margin).
MIN_WARMUP_BARS_M5 = 200
MIN_WARMUP_BARS_H1 = 200


def default_crypto_tick(sym: str) -> float:
    """
    Default "one tick" price step by symbol prefix (same as notebook).
    Used to scale pending offset to price; override with broker tick/pip size when known.
    """
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
    """Adds three EMAs on column close; returns df with columns ema_8 / ema_13 / ema_21."""
    out = df.copy()
    out[f"ema_{EMA_FAST}"] = out["close"].ewm(span=EMA_FAST, adjust=False).mean()
    out[f"ema_{EMA_MID}"] = out["close"].ewm(span=EMA_MID, adjust=False).mean()
    out[f"ema_{EMA_SLOW}"] = out["close"].ewm(span=EMA_SLOW, adjust=False).mean()
    return out


def h1_trend_series(h1: pd.DataFrame) -> pd.Series:
    """
    Per H1 candle, trend label: bull / bear / flat (same rules as notebook).
    bull: EMAs stacked ascending and close above slow EMA; bear the opposite; else flat.
    """
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
    Maps H1 trend and EMAs onto each M5 row: for each M5 time, use the latest fully closed
    H1 bar at or before that time (merge_asof backward). Each M5 row gets trend and concurrent H1 EMAs.
    """
    h1_sig = h1.copy()
    h1_sig["trend"] = h1_trend_series(h1_sig)
    cols = ["trend", f"ema_{EMA_FAST}", f"ema_{EMA_MID}", f"ema_{EMA_SLOW}"]
    merged = pd.merge_asof(
        m5.sort_index(),
        h1_sig[cols].sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",
    )
    return merged


def rates_to_ohlcv_df(rates) -> pd.DataFrame:
    """Converts MT5 copy_rates_* numpy structured array to a DataFrame indexed by UTC time."""
    df = pd.DataFrame(rates)
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
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
    """
    At closed bar index bar_index (same section as notebook):
    swing window = lookback_bars candles strictly before i (bar i is not in the window).
    bull trend: Buy Stop above HH + offset; SL below LL - offset; TP at rr * price risk.
    bear trend: Sell Stop; mirrored geometry.
    qty from dollar risk balance * risk_per_trade vs SL distance from entry.
    Returns None if trend is flat or history is insufficient.
    """
    if bar_index < lookback_bars:
        return None

    row = m5_ctx.iloc[bar_index]
    window = m5_ctx.iloc[bar_index - lookback_bars : bar_index]
    hh = float(window["high"].max())
    ll = float(window["low"].min())

    trend = row.get("trend", "flat")
    if trend not in ("bull", "bear"):
        return None

    # Offset in "ticks"; pip_size is the price multiplier (e.g. 0.01 for ETH).
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
    """
    Index of the last closed M5 bar that can emit a signal; only if dataframe length is enough.
    Required length = max(swing window + 1, min_warmup_bars) so EMAs are warm and lookback is full.
    """
    need = max(lookback_bars + 1, min_warmup_bars)
    if len(closed_m5) < need:
        return None
    return len(closed_m5) - 1


def min_bars_needed_for_signal(lookback_bars: int, *, min_warmup_bars: int = MIN_WARMUP_BARS_M5) -> int:
    """Minimum number of closed M5 bars so last_closed_bar_index returns an index."""
    return max(lookback_bars + 1, min_warmup_bars)
