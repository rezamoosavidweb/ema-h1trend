"""
Order Block Reaction strategy — signal generation for live trading.

Mirrors notebooks/08_order_block_reaction_crypto.ipynb exactly.

Entry logic:
  1. Displacement: ≥4 consecutive same-direction candles, move ≥ 1.5×ATR(14).
  2. Order Block: last opposite candle immediately before the displacement start.
  3. Signal: price displaces through OB boundary, then retraces into zone with
     rejection wick (wick/range > 0.3).
  4. Bullish OB → entry=ob_high, sl=ob_low, tp=entry+(entry-sl)*rr
     Bearish OB → entry=ob_low,  sl=ob_high, tp=entry-(sl-entry)*rr
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

# ── Strategy constants (match notebook) ──────────────────────────────────────
ATR_PERIOD               = 14
DISPLACEMENT_MIN_CANDLES = 4
DISPLACEMENT_ATR_MULT    = 1.5
OB_EXPIRY_BARS           = 100
REJECTION_WICK_RATIO     = 0.3
DEFAULT_RR               = 2.0
SL_BUFFER                = 0.0
OB_WARMUP_BARS           = 120   # min bars before signals are reliable

# ── Fee-based minimum SL distance ────────────────────────────────────────────
MAKER_FEE_RATE  = 0.0002    # 0.02%  — limit entry orders
TAKER_FEE_RATE  = 0.00055   # 0.055% — stop-loss (market) exit
MIN_SL_FEE_MULT = 1.0       # min_sl_dist = entry × (maker+taker) × this


@dataclass
class _Displacement:
    start_idx: int
    direction: str   # 'UP' | 'DOWN'


@dataclass
class _OB:
    ob_bar_idx: int
    ob_type:    str    # 'bullish' | 'bearish'
    ob_high:    float
    ob_low:     float


# ── Candle features ───────────────────────────────────────────────────────────

def add_candle_features(df: pd.DataFrame, atr_period: int = ATR_PERIOD) -> pd.DataFrame:
    """Add ATR, body bounds, wicks, range, is_bull/is_bear columns (in-place copy)."""
    df = df.copy()
    prev_close  = df["close"].shift(1)
    tr = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            (df["high"] - prev_close).abs(),
            (df["low"]  - prev_close).abs(),
        ),
    )
    df["atr"]        = tr.ewm(span=atr_period, adjust=False).mean()
    df["body_top"]   = df[["open", "close"]].max(axis=1)
    df["body_bot"]   = df[["open", "close"]].min(axis=1)
    df["upper_wick"] = df["high"] - df["body_top"]
    df["lower_wick"] = df["body_bot"] - df["low"]
    df["range"]      = df["high"] - df["low"]
    df["is_bull"]    = df["close"] > df["open"]
    df["is_bear"]    = df["close"] < df["open"]
    return df


# ── Displacement detection ────────────────────────────────────────────────────

def _detect_displacements(
    df: pd.DataFrame,
    min_candles: int = DISPLACEMENT_MIN_CANDLES,
    atr_mult: float  = DISPLACEMENT_ATR_MULT,
) -> List[_Displacement]:
    n, i = len(df), 0
    result: List[_Displacement] = []
    while i < n:
        is_bull = bool(df.iloc[i]["is_bull"])
        is_bear = bool(df.iloc[i]["is_bear"])
        if not is_bull and not is_bear:
            i += 1
            continue
        key = "is_bull" if is_bull else "is_bear"
        j = i + 1
        while j < n and bool(df.iloc[j][key]):
            j += 1
        count = j - i
        if count >= min_candles:
            seg  = df.iloc[i:j]
            move = (
                float(seg["close"].iloc[-1]) - float(seg["open"].iloc[0])
                if is_bull
                else float(seg["open"].iloc[0]) - float(seg["close"].iloc[-1])
            )
            if move >= atr_mult * float(seg["atr"].mean()):
                result.append(_Displacement(i, "UP" if is_bull else "DOWN"))
        i = j
    return result


# ── Order block identification ────────────────────────────────────────────────

def _find_order_blocks(
    df: pd.DataFrame,
    displacements: List[_Displacement],
) -> List[_OB]:
    obs: List[_OB] = []
    for d in displacements:
        look_back = max(0, d.start_idx - 20)
        if d.direction == "UP":
            for k in range(d.start_idx - 1, look_back - 1, -1):
                if df.iloc[k]["is_bear"]:
                    bar = df.iloc[k]
                    obs.append(_OB(k, "bullish", float(bar["high"]), float(bar["low"])))
                    break
        else:
            for k in range(d.start_idx - 1, look_back - 1, -1):
                if df.iloc[k]["is_bull"]:
                    bar = df.iloc[k]
                    obs.append(_OB(k, "bearish", float(bar["high"]), float(bar["low"])))
                    break
    return obs


# ── Rejection candle check ────────────────────────────────────────────────────

def _is_rejection(
    bar: pd.Series,
    ob_type: str,
    ratio: float = REJECTION_WICK_RATIO,
) -> bool:
    r = float(bar["range"])
    if r == 0:
        return False
    if ob_type == "bullish":
        return float(bar["lower_wick"]) / r > ratio
    return float(bar["upper_wick"]) / r > ratio


# ── Public API ────────────────────────────────────────────────────────────────

def list_ob_signals(
    df: pd.DataFrame,
    *,
    risk_cash: float,
    rr: float                    = DEFAULT_RR,
    sl_buffer: float             = SL_BUFFER,
    atr_period: int              = ATR_PERIOD,
    displacement_min_candles: int = DISPLACEMENT_MIN_CANDLES,
    displacement_atr_mult: float  = DISPLACEMENT_ATR_MULT,
    ob_expiry_bars: int           = OB_EXPIRY_BARS,
    rejection_wick_ratio: float   = REJECTION_WICK_RATIO,
    min_sl_fee_mult: float        = MIN_SL_FEE_MULT,
) -> pd.DataFrame:
    """
    Scan all closed M5 bars and return every bar where an OB entry signal fires.

    One signal per Order Block (first valid retest only).
    The live bot takes the LAST row from this DataFrame as the current signal.

    Returned columns:
        signal_bar_index, signal_bar_time, side, entry, sl, tp, qty, ob_type
    """
    df = add_candle_features(df, atr_period=atr_period)
    displacements = _detect_displacements(df, displacement_min_candles, displacement_atr_mult)
    obs           = _find_order_blocks(df, displacements)
    n             = len(df)
    rows: list[dict] = []

    for ob in obs:
        displaced = False
        for i in range(ob.ob_bar_idx + 1, min(ob.ob_bar_idx + ob_expiry_bars, n - 1)):
            bar = df.iloc[i]

            if ob.ob_type == "bullish":
                if not displaced:
                    if float(bar["close"]) > ob.ob_high:
                        displaced = True
                    continue
                # Retest: low touches zone AND close stays above ob_low
                if float(bar["low"]) <= ob.ob_high and float(bar["close"]) >= ob.ob_low:
                    if _is_rejection(bar, ob.ob_type, rejection_wick_ratio):
                        entry    = ob.ob_high
                        sl       = ob.ob_low - sl_buffer
                        dist     = entry - sl
                        min_dist = entry * (MAKER_FEE_RATE + TAKER_FEE_RATE) * min_sl_fee_mult
                        if dist >= min_dist:
                            rows.append({
                                "signal_bar_index": i,
                                "signal_bar_time":  df.index[i],
                                "side":             "buy",
                                "entry":            entry,
                                "sl":               sl,
                                "tp":               entry + dist * rr,
                                "qty":              risk_cash / dist,
                                "ob_type":          ob.ob_type,
                            })
                        break
                if float(bar["close"]) < ob.ob_low:
                    break  # OB invalidated

            else:  # bearish OB
                if not displaced:
                    if float(bar["close"]) < ob.ob_low:
                        displaced = True
                    continue
                # Retest: high touches zone AND close stays below ob_high
                if float(bar["high"]) >= ob.ob_low and float(bar["close"]) <= ob.ob_high:
                    if _is_rejection(bar, ob.ob_type, rejection_wick_ratio):
                        entry    = ob.ob_low
                        sl       = ob.ob_high + sl_buffer
                        dist     = sl - entry
                        min_dist = entry * (MAKER_FEE_RATE + TAKER_FEE_RATE) * min_sl_fee_mult
                        if dist >= min_dist:
                            rows.append({
                                "signal_bar_index": i,
                                "signal_bar_time":  df.index[i],
                                "side":             "sell",
                                "entry":            entry,
                                "sl":               sl,
                                "tp":               entry - dist * rr,
                                "qty":              risk_cash / dist,
                                "ob_type":          ob.ob_type,
                            })
                        break
                if float(bar["close"]) > ob.ob_high:
                    break  # OB invalidated

    return pd.DataFrame(rows)
