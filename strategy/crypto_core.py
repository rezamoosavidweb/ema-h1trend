"""
EMA H1 trend + M5 pending-stop signal math.

THIS FILE IS IDENTICAL TO THE MT5 VERSION — DO NOT MODIFY.
All signal logic (EMA, trend, pending setup, risk sizing) must remain unchanged.

Original: strategies/ema_trend/crypto_core.py
"""

from __future__ import annotations

import pandas as pd

EMA_FAST = 8
EMA_MID = 13
EMA_SLOW = 21

MIN_WARMUP_BARS_M5 = 200
MIN_WARMUP_BARS_H1 = 200


def default_crypto_tick(sym: str) -> float:
    """Default price tick step by symbol prefix; override when broker step differs."""
    u = sym.upper().replace("USDT", "").replace("PERP", "").replace(".P", "")
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
    risk_cash: float,
    leverage: int = 1,
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
