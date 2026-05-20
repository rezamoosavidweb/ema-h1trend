#!/usr/bin/env python3
"""
Order Block Reaction Bot — XAUUSD M5 (Errante MT5)

Strategy pipeline (mirrors notebook 08_order_block_reaction.ipynb):
  displacement detection -> order block -> retest -> rejection candle -> market order

Run:
  python mt5/run_ob_xauusd.py              # loop — every M5 close
  python mt5/run_ob_xauusd.py --once       # single evaluation then exit
  python mt5/run_ob_xauusd.py --dry-run    # log signals only, no orders
  python mt5/run_ob_xauusd.py --lot 0.02   # fixed lot size

Env vars (optional — skip if terminal is already open and logged in):
  MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_TERMINAL_PATH

Log file: logs/XAUUSD.json  (JSON Lines — one event per line)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHANGES vs previous version
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[CHANGE 1] Extra wait after M5 close reduced: 2.0 s -> 0.1 s
    Previously the bot waited 2 seconds after each M5 candle closed.
    That 2 s + processing time (~1-2 s) allowed price to move 1-3 pts
    before the order was sent, increasing slippage unnecessarily.
    Now only 0.1 s is used — enough for MT5 to refresh its data feed.

[CHANGE 2] SLIPPAGE_MAX_POINTS raised: 3.0 -> 6.0
    The hard-skip threshold was moved from 3 to 6 pts.
    Signals with slippage between 3 and 6 pts were previously discarded
    entirely. They are now captured with a reduced lot (see CHANGE 3).

[CHANGE 3] Three-tier slippage response (replaces single hard cut-off)
    Before: slippage > 3 pts -> skip signal entirely
    Now:
      <= 4.0 pts  -> LIMIT order at exact OB price (original SL & TP, no RR distortion)
      4.0-6.0 pts -> market order with reduced lot + TP recalculated from fill (RR=2)
      >= 6.0 pts  -> skip (too far from OB, structural thesis weakened)
    The SL always stays at the structural OB edge regardless of slippage.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    print("Install MetaTrader5: pip install MetaTrader5", file=sys.stderr)
    raise

# telegram_bot lives in the repo root; make sure it's importable regardless of
# which directory the script is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from telegram_bot.mt5_notifier import Mt5Notifier  # noqa: E402

# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════════

_REPO_ROOT = Path(__file__).resolve().parent.parent  # mt5/ -> repo root
LOG_PATH = _REPO_ROOT / "logs" / "XAUUSD.json"  # JSON Lines log
SEEN_OBS_PATH = _REPO_ROOT / "logs" / "seen_obs_XAUUSD.json"  # OB dedup state


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY CONFIG  (all parameters in one place — matches notebook exactly)
# ══════════════════════════════════════════════════════════════════════════════

SYMBOL = "XAUUSD"

# Number of M5 bars to fetch — large enough to warm up ATR and find displacements
HISTORY_BARS = 600

# EWM period for ATR (Average True Range)
ATR_PERIOD = 14

# Minimum consecutive same-direction candles to qualify as a displacement
DISPLACEMENT_MIN_CANDLES = 4

# Displacement move must be at least this multiple of the mean ATR
DISPLACEMENT_ATR_MULT = 1.5

# Ignore an OB that is older than this many bars
OB_EXPIRY_BARS = 100

# lower_wick / range threshold for a bullish rejection candle
# upper_wick / range threshold for a bearish rejection candle
REJECTION_WICK_RATIO = 0.3

# Risk-to-reward ratio — TP = entry + risk * RISK_REWARD
RISK_REWARD = 2.0

# Extra buffer beyond the OB edge for the SL (0 = exactly at OB edge)
SL_BUFFER = 0.5

# ── Slippage thresholds ───────────────────────────────────────────────────────
# SLIPPAGE_LIMIT_THRESHOLD: when slippage is at or below this value, place a
#   limit order at the exact OB price so we fill at the structural level.
SLIPPAGE_LIMIT_THRESHOLD  = 4.0   # pts — at or below: limit order at OB price

# SLIPPAGE_MAX_POINTS: skip the trade entirely when slippage reaches this value.
#   Between LIMIT_THRESHOLD and MAX: market order with reduced lot.
SLIPPAGE_MAX_POINTS       = 6.0   # pts — at or above: skip entirely

# How long a limit order stays live before MT5 auto-cancels it (one M5 bar).
LIMIT_ORDER_EXPIRY_MINUTES = 5

# Fraction of balance to risk per trade when --lot is not specified
RISK_PER_TRADE = 0.01  # 1 % of balance

# Unique magic number — isolates this bot's orders from manual trades in MT5
MAGIC = 8088080

# Duration of one M5 candle in seconds
M5_SECONDS = 300


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════


def log(event: str, data: dict | None = None) -> None:
    """
    Append one JSON line to LOG_PATH and print a short summary to the console.

    Log format (JSON Lines — one event per line):
      {"event": "...", "ts": "ISO-8601 UTC", "symbol": "XAUUSD", ...extra fields}

    Each event is an independent line, making the file easy to parse with
    pandas or grep for later analysis.
    """
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "event": event,
        "ts": datetime.now(timezone.utc).isoformat(),
        "symbol": SYMBOL,
    }
    if data:
        entry.update(data)

    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")

    # Print a compact summary — hide ob_bar_idx (too technical for console)
    short = {k: v for k, v in (data or {}).items() if k not in ("ob_bar_idx",)}
    print(f"[{entry['ts']}] {event} | {short}")


# ══════════════════════════════════════════════════════════════════════════════
# SEEN OBS PERSISTENCE  (deduplication across restarts)
# ══════════════════════════════════════════════════════════════════════════════


def load_seen_obs() -> set:
    """
    Load the set of (ob_time, direction) pairs already traded from disk.

    Without this file, a bot restart would re-enter the same OB that was
    already traded in the previous session. The file makes deduplication
    survive restarts.
    """
    if not SEEN_OBS_PATH.exists():
        return set()
    try:
        with SEEN_OBS_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # JSON stores lists; convert back to set of tuples
        return {tuple(item) for item in data}
    except Exception:
        # Corrupted file — start fresh rather than crash
        return set()


def save_seen_obs(seen_obs: set) -> None:
    """
    Persist the set of traded OB keys to disk.
    Called after every successful order placement or dry-run signal.
    """
    SEEN_OBS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SEEN_OBS_PATH.open("w", encoding="utf-8") as f:
        json.dump([list(item) for item in seen_obs], f)


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY LOGIC  (exact replica of notebook 08_order_block_reaction)
# ══════════════════════════════════════════════════════════════════════════════


def add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-candle derived columns and append them to the DataFrame.

    Columns added:
      tr          : True Range — max of (high-low, |high-prev_close|, |low-prev_close|)
      atr         : Exponential moving average of TR over ATR_PERIOD bars
      body        : Absolute distance between open and close
      body_top    : Upper edge of the candle body (max of open, close)
      body_bot    : Lower edge of the candle body (min of open, close)
      upper_wick  : Distance from body_top to high
      lower_wick  : Distance from body_bot to low
      range       : Full candle range (high - low)
      is_bull     : True when close > open
      is_bear     : True when close < open
    """
    df = df.copy()

    # True Range: largest of the three possible price spans for each bar
    df["tr"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            abs(df["high"] - df["close"].shift(1)),
            abs(df["low"] - df["close"].shift(1)),
        ),
    )

    # ATR: exponential moving average of TR (same formula as notebook)
    df["atr"] = df["tr"].ewm(span=ATR_PERIOD, adjust=False).mean()

    # Candle shape metrics
    df["body"] = abs(df["close"] - df["open"])
    df["body_top"] = df[["open", "close"]].max(axis=1)
    df["body_bot"] = df[["open", "close"]].min(axis=1)
    df["upper_wick"] = df["high"] - df["body_top"]
    df["lower_wick"] = df["body_bot"] - df["low"]
    df["range"] = df["high"] - df["low"]
    df["is_bull"] = df["close"] > df["open"]
    df["is_bear"] = df["close"] < df["open"]

    return df


@dataclass
class Displacement:
    """
    Represents a strong, sustained directional move (displacement).

    A run of consecutive same-direction candles qualifies as a displacement when:
      - It contains at least DISPLACEMENT_MIN_CANDLES candles, AND
      - The total price move >= DISPLACEMENT_ATR_MULT * mean ATR of the segment.

    Doji candles (close == open) break a run because they are neither bullish
    nor bearish.
    """

    start_idx: int  # DataFrame index of the first candle in the run
    end_idx: int  # DataFrame index of the last candle in the run
    direction: str  # 'UP' (bullish run) or 'DOWN' (bearish run)
    total_move: float  # total price distance covered (pts)
    candle_count: int  # number of candles in the run


def detect_displacements(df: pd.DataFrame) -> List[Displacement]:
    """
    Find all valid displacements in the DataFrame.

    Iterates through bars grouping consecutive same-direction candles.
    Each group that meets the size and ATR-relative-move thresholds is
    recorded as a Displacement.
    """
    displacements: List[Displacement] = []
    n = len(df)
    i = 0

    while i < n:
        if df.iloc[i]["is_bull"]:
            # Extend the bullish run as far as consecutive bull candles go
            j = i + 1
            while j < n and df.iloc[j]["is_bull"]:
                j += 1

            if j - i >= DISPLACEMENT_MIN_CANDLES:
                seg = df.iloc[i:j]
                move = seg["close"].iloc[-1] - seg["open"].iloc[0]
                if move >= DISPLACEMENT_ATR_MULT * seg["atr"].mean():
                    displacements.append(Displacement(i, j - 1, "UP", move, j - i))
            i = j

        elif df.iloc[i]["is_bear"]:
            # Extend the bearish run
            j = i + 1
            while j < n and df.iloc[j]["is_bear"]:
                j += 1

            if j - i >= DISPLACEMENT_MIN_CANDLES:
                seg = df.iloc[i:j]
                move = seg["open"].iloc[0] - seg["close"].iloc[-1]
                if move >= DISPLACEMENT_ATR_MULT * seg["atr"].mean():
                    displacements.append(Displacement(i, j - 1, "DOWN", move, j - i))
            i = j

        else:
            # Doji — breaks the chain
            i += 1

    return displacements


@dataclass
class OrderBlock:
    """
    Represents an Order Block — the last opposite-direction candle before a displacement.

    Logic:
      - Before a bullish displacement (UP): the last bearish candle = Bullish OB.
        The market bought heavily from that zone, driving price up.
      - Before a bearish displacement (DOWN): the last bullish candle = Bearish OB.
        The market sold heavily from that zone, driving price down.

    When price returns (retests) the OB zone, the same institutional direction
    is expected to resume.

    Entry and SL levels:
      Bullish OB: entry = ob_high (top of zone), SL = ob_low (bottom of zone)
      Bearish OB: entry = ob_low  (bottom of zone), SL = ob_high (top of zone)
    """

    ob_bar_idx: int  # DataFrame index of the OB candle
    ob_type: str  # 'bullish' or 'bearish'
    ob_high: float  # High of the OB candle
    ob_low: float  # Low of the OB candle
    ob_time: object  # Timestamp of the OB candle (used for dedup and logging)
    displaced_by: float  # Size of the displacement that confirmed this OB (pts)


def find_order_blocks(
    df: pd.DataFrame,
    displacements: List[Displacement],
) -> List[OrderBlock]:
    """
    For each displacement, find the last opposite-direction candle before it.

    Why the *last* opposite candle?
    It represents the most recent zone where the market reversed — that zone
    has the highest probability of acting as institutional memory on a retest.

    The look-back window is capped at 20 candles before the displacement start
    to avoid picking up stale, unrelated candles.
    """
    obs: List[OrderBlock] = []

    for disp in displacements:
        start_i = disp.start_idx
        look_back = max(0, start_i - 20)  # search at most 20 bars back

        if disp.direction == "UP":
            # Bullish displacement -> last bearish candle before it = Bullish OB
            for k in range(start_i - 1, look_back - 1, -1):
                if df.iloc[k]["is_bear"]:
                    bar = df.iloc[k]
                    obs.append(
                        OrderBlock(
                            k,
                            "bullish",
                            bar["high"],
                            bar["low"],
                            bar["time"],
                            disp.total_move,
                        )
                    )
                    break  # only the most recent bearish candle is needed

        else:  # direction == "DOWN"
            # Bearish displacement -> last bullish candle before it = Bearish OB
            for k in range(start_i - 1, look_back - 1, -1):
                if df.iloc[k]["is_bull"]:
                    bar = df.iloc[k]
                    obs.append(
                        OrderBlock(
                            k,
                            "bearish",
                            bar["high"],
                            bar["low"],
                            bar["time"],
                            disp.total_move,
                        )
                    )
                    break

    return obs


def is_rejection_candle(bar: pd.Series, ob_type: str) -> bool:
    """
    Return True if the candle shows a valid rejection from the OB zone.

    Bullish OB (BUY signal):
      A long lower wick means price dipped into the OB but buyers pushed it
      back up — strong demand. Condition: lower_wick / range > REJECTION_WICK_RATIO

    Bearish OB (SELL signal):
      A long upper wick means price rose into the OB but sellers pushed it
      back down — strong supply. Condition: upper_wick / range > REJECTION_WICK_RATIO
    """
    if bar["range"] == 0:
        # Zero-range bar (data gap or identical OHLC) — ignore
        return False

    if ob_type == "bullish":
        return bar["lower_wick"] / bar["range"] > REJECTION_WICK_RATIO
    else:
        return bar["upper_wick"] / bar["range"] > REJECTION_WICK_RATIO


def find_signal(
    df: pd.DataFrame,
    order_blocks: List[OrderBlock],
) -> Optional[dict]:
    """
    Evaluate the last closed bar for a valid OB retest + rejection signal.

    This replicates run_ob_backtest() from the notebook but is evaluated only
    on the most recent closed bar (bar n-1).

    BUY signal conditions (Bullish OB):
      1. At least one bar after the OB closed above ob_high  (displacement confirmed)
      2. No bar after the displacement closed below ob_low   (OB not yet invalidated)
      3. Last bar's low  <= ob_high                          (retest reached the zone)
      4. Last bar's close >= ob_low                          (closed inside the zone)
      5. Last bar is a rejection candle                      (large lower wick)

    SELL signal conditions (Bearish OB): mirror of the above.

    Returns a signal dict with full trade parameters, or None if no signal.
    """
    n = len(df)
    last_idx = n - 1
    last_bar = df.iloc[last_idx]

    # Iterate from newest OB to oldest (reversed insertion order)
    for ob in reversed(order_blocks):

        # OB must precede the evaluation bar
        if ob.ob_bar_idx >= last_idx:
            continue

        # OB has expired — too old to be reliable
        if last_idx - ob.ob_bar_idx > OB_EXPIRY_BARS:
            continue

        # ── Bullish OB -> BUY signal ─────────────────────────────────────────
        if ob.ob_type == "bullish":
            displaced = False
            invalidated = False

            for k in range(ob.ob_bar_idx + 1, last_idx):
                bar_k = df.iloc[k]

                if not displaced:
                    # Waiting for a bar to close above ob_high (confirms the breakout)
                    if bar_k["close"] > ob.ob_high:
                        displaced = True
                else:
                    # OB is invalidated if any bar closes below ob_low after displacement
                    if bar_k["close"] < ob.ob_low:
                        invalidated = True
                        break

            if not displaced or invalidated:
                continue

            # Check last bar
            if last_bar["low"] <= ob.ob_high and last_bar["close"] >= ob.ob_low:
                if is_rejection_candle(last_bar, ob.ob_type):
                    # entry = ob_high : upper edge of OB (first support level on retest)
                    # SL    = ob_low  : lower edge of OB (thesis fails if price closes here)
                    # TP    = entry + risk * RR
                    entry = ob.ob_high
                    sl = ob.ob_low - SL_BUFFER
                    risk = entry - sl
                    tp = entry + risk * RISK_REWARD

                    return {
                        "direction": "BUY",
                        "ob_type": ob.ob_type,
                        "ob_high": ob.ob_high,
                        "ob_low": ob.ob_low,
                        "ob_time": str(ob.ob_time),
                        "retest_time": str(last_bar["time"]),
                        "entry": round(entry, 2),
                        "sl": round(sl, 2),
                        "tp": round(tp, 2),
                        "ob_bar_idx": ob.ob_bar_idx,
                        "displaced_by": round(ob.displaced_by, 2),
                    }

        # ── Bearish OB -> SELL signal ────────────────────────────────────────
        else:
            displaced = False
            invalidated = False

            for k in range(ob.ob_bar_idx + 1, last_idx):
                bar_k = df.iloc[k]

                if not displaced:
                    if bar_k["close"] < ob.ob_low:
                        displaced = True
                else:
                    if bar_k["close"] > ob.ob_high:
                        invalidated = True
                        break

            if not displaced or invalidated:
                continue

            if last_bar["high"] >= ob.ob_low and last_bar["close"] <= ob.ob_high:
                if is_rejection_candle(last_bar, ob.ob_type):
                    # entry = ob_low  : lower edge of OB (first resistance level on retest)
                    # SL    = ob_high : upper edge of OB (thesis fails if price closes here)
                    # TP    = entry - risk * RR
                    entry = ob.ob_low
                    sl = ob.ob_high + SL_BUFFER
                    risk = sl - entry
                    tp = entry - risk * RISK_REWARD

                    return {
                        "direction": "SELL",
                        "ob_type": ob.ob_type,
                        "ob_high": ob.ob_high,
                        "ob_low": ob.ob_low,
                        "ob_time": str(ob.ob_time),
                        "retest_time": str(last_bar["time"]),
                        "entry": round(entry, 2),
                        "sl": round(sl, 2),
                        "tp": round(tp, 2),
                        "ob_bar_idx": ob.ob_bar_idx,
                        "displaced_by": round(ob.displaced_by, 2),
                    }

    return None  # no signal found on this bar


# ══════════════════════════════════════════════════════════════════════════════
# MT5 HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def mt5_connect() -> None:
    """
    Initialize the MT5 connection.

    If environment variables MT5_LOGIN, MT5_PASSWORD, MT5_SERVER are set,
    they are passed to mt5.initialize() for programmatic login.
    Otherwise the function assumes the terminal is already open and logged in.
    """
    kwargs: dict = {}
    login = os.environ.get("MT5_LOGIN")
    password = os.environ.get("MT5_PASSWORD")
    server = os.environ.get("MT5_SERVER")

    if login and password and server:
        kwargs["login"] = int(login)
        kwargs["password"] = password
        kwargs["server"] = server

    path = os.environ.get("MT5_TERMINAL_PATH")
    if path:
        kwargs["path"] = path

    if not mt5.initialize(**kwargs):
        raise RuntimeError(f"mt5.initialize failed: {mt5.last_error()}")


def assert_terminal_ready() -> None:
    """
    Verify that the MT5 terminal is connected to the broker and receiving quotes.

    Raises RuntimeError with a clear user-facing message if:
      - terminal_info() returns None (terminal not responding)
      - terminal.connected is False (no live data feed)
    """
    ti = mt5.terminal_info()
    if ti is None:
        raise RuntimeError(
            f"terminal_info() returned None: {mt5.last_error()}\n"
            "Open MT5, log in, enable AutoTrading, then retry.\n"
            "64-bit Python is required."
        )
    if not ti.connected:
        raise RuntimeError(
            "MT5 terminal is not connected to the broker — "
            "wait for quotes to appear in Market Watch."
        )


def resolve_symbol(symbol: str) -> str:
    """
    Resolve the exact symbol name used by the broker.

    Some brokers append suffixes such as .i, .a, or # to standard symbol
    names. For example, Errante uses XAUUSD.i. This function tries the bare
    name first, then common suffixes.
    """
    mt5.symbol_select(symbol, True)
    si = mt5.symbol_info(symbol)

    if si is None:
        for suffix in (".a", ".i", "#", "-raw"):
            candidate = symbol + suffix
            mt5.symbol_select(candidate, True)
            si = mt5.symbol_info(candidate)
            if si is not None:
                break

    if si is None:
        raise RuntimeError(
            f"Symbol {symbol!r} not found in MT5.\n"
            "Open Market Watch -> right-click -> Show All, find the exact name, "
            "then pass it with --symbol."
        )

    mt5.symbol_select(si.name, True)
    return si.name


def fetch_m5(symbol: str, bars: int) -> pd.DataFrame:
    """
    Fetch a fixed number of M5 bars from MT5.

    The last row (the still-forming candle) is always dropped so that
    strategy logic only sees fully closed candles.
    """
    r = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, bars)
    if r is None or len(r) == 0:
        raise RuntimeError(f"No M5 data for {symbol}: {mt5.last_error()}")

    df = pd.DataFrame(r)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={"tick_volume": "volume"})

    keep = ["time", "open", "high", "low", "close", "volume"]
    df = df[[c for c in keep if c in df.columns]].copy()

    # Drop the last bar — it is still forming and has an incomplete close
    return df.iloc[:-1].reset_index(drop=True)


def _tick_size(si) -> float:
    """
    Return the minimum valid price increment (tick size) for the symbol.
    Used to snap SL/TP prices onto the broker's price grid.
    """
    ts = float(getattr(si, "trade_tick_size", 0) or 0)
    if ts > 0:
        return ts
    pt = float(si.point or 0)
    return pt if pt > 0 else float(10 ** (-si.digits))


def pick_filling(si) -> int:
    """
    Select the best order filling mode supported by this broker/symbol.

    MT5 brokers support one or more of:
      IOC (Immediate Or Cancel) — fastest, allows partial fills
      FOK (Fill Or Kill)        — all-or-nothing
      RETURN                    — plain market order
    """
    ioc = getattr(mt5, "SYMBOL_FILLING_IOC", 2)
    fok = getattr(mt5, "SYMBOL_FILLING_FOK", 1)
    fm = int(si.filling_mode)
    if fm & ioc:
        return mt5.ORDER_FILLING_IOC
    if fm & fok:
        return mt5.ORDER_FILLING_FOK
    return mt5.ORDER_FILLING_RETURN


def snap(price: float, si, mode: str = "nearest") -> float:
    """
    Round a price to the nearest valid tick on the broker's price grid.

    mode:
      "nearest" — round to closest tick (used for entry price reference)
      "up"      — round up  (TP for BUY, SL for SELL — conservative direction)
      "down"    — round down (SL for BUY, TP for SELL — conservative direction)

    The 1e-12 epsilon prevents floating-point rounding from pushing a value
    that should be exactly on a tick to the wrong side.
    """
    t = _tick_size(si)
    d = int(si.digits)
    x = price / t

    if mode == "up":
        v = math.ceil(x - 1e-12) * t
    elif mode == "down":
        v = math.floor(x + 1e-12) * t
    else:
        v = round(x) * t

    return round(v, d)


def normalize_vol(vol: float, si) -> float:
    """
    Clamp and round a lot size to the broker's volume constraints.

    Brokers enforce a minimum lot, maximum lot, and lot step (volume_step).
    For example, Errante: min=0.01, step=0.01.
    A calculated volume of 0.0079 rounds down to the minimum 0.01.
    """
    step = float(si.volume_step or 0.01)
    vmin = float(si.volume_min or 0.01)
    vmax = float(si.volume_max or 100.0)
    v = math.floor(vol / step + 1e-12) * step
    return float(max(vmin, min(vmax, v)))


def calc_volume(
    symbol: str,
    side: str,
    entry: float,
    sl: float,
    balance: float,
) -> float:
    """
    Calculate lot size so that a loss from entry to SL equals RISK_PER_TRADE * balance.

    Example:
      balance = 1000 USD, RISK_PER_TRADE = 0.01 -> risk budget = 10 USD
      If 1 lot loses 50 USD from entry to SL -> lot = 10 / 50 = 0.2

    Uses mt5.order_calc_profit for precise broker-side calculation that
    accounts for pip value, leverage, and account currency conversion.

    The entry parameter can be the ideal OB level (ob_high/ob_low) for the
    initial sizing, or the actual market price when adjusting for slippage
    in the three-tier logic [CHANGE 3].
    """
    otype = mt5.ORDER_TYPE_BUY if side == "buy" else mt5.ORDER_TYPE_SELL
    pnl_1l = mt5.order_calc_profit(otype, symbol, 1.0, entry, sl)

    if pnl_1l is None or abs(pnl_1l) < 1e-9:
        # Broker calculation failed — fall back to minimum lot
        return 0.01

    risk_cash = balance * RISK_PER_TRADE
    si = mt5.symbol_info(symbol)
    return normalize_vol(risk_cash / abs(pnl_1l), si)


def send_market_order(
    symbol: str,
    side: str,
    volume: float,
    sl: float,
    tp: float,
) -> Optional[int]:
    """
    Send a market order to MT5 and return the ticket number on success.

    Steps:
      1. Snap SL and TP to the broker price grid.
      2. Enforce the broker's minimum stop distance from the current price.
      3. Send the order via mt5.order_send().
      4. Return None for market-closed (retcode 10018) or any other error.
    """
    si = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)

    if si is None or tick is None:
        log("order_error", {"reason": "no_symbol_or_tick"})
        return None

    price = tick.ask if side == "buy" else tick.bid
    otype = mt5.ORDER_TYPE_BUY if side == "buy" else mt5.ORDER_TYPE_SELL

    # Snap SL/TP to the nearest valid tick
    # BUY:  SL rounds down (protective), TP rounds up (targets more)
    # SELL: SL rounds up  (protective), TP rounds down (targets more)
    sl_snapped = snap(sl, si, "down" if side == "buy" else "up")
    tp_snapped = snap(tp, si, "up" if side == "buy" else "down")

    # Enforce the broker's minimum stop distance (trade_stops_level * point)
    point = float(si.point or 10 ** (-si.digits))
    stops_pts = int(getattr(si, "trade_stops_level", 0) or 0)
    min_dist = stops_pts * point

    if side == "buy":
        if price - sl_snapped < min_dist:
            sl_snapped = snap(price - min_dist - point, si, "down")
        if tp_snapped - price < min_dist:
            tp_snapped = snap(price + min_dist + point, si, "up")
    else:
        if sl_snapped - price < min_dist:
            sl_snapped = snap(price + min_dist + point, si, "up")
        if price - tp_snapped < min_dist:
            tp_snapped = snap(price - min_dist - point, si, "down")

    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": otype,
        "price": price,
        "sl": sl_snapped,
        "tp": tp_snapped,
        "magic": MAGIC,
        "comment": "ob_reaction",
        "type_filling": pick_filling(si),
        # Maximum allowed price deviation in points before the broker rejects the order
        "deviation": int(os.environ.get("MT5_DEVIATION_POINTS", "50")),
    }

    r = mt5.order_send(req)

    if r is None:
        log(
            "order_error",
            {
                "error": str(mt5.last_error()),
                "req_price": price,
                "sl": sl_snapped,
                "tp": tp_snapped,
            },
        )
        return None

    if r.retcode == 10018:
        # Market is closed (weekend or outside trading session)
        log("market_closed", {"comment": r.comment, "req_price": price})
        return None

    if r.retcode != mt5.TRADE_RETCODE_DONE:
        log(
            "order_error",
            {
                "retcode": r.retcode,
                "comment": r.comment,
                "req_price": price,
            },
        )
        return None

    return r.order


def send_limit_order(
    symbol: str,
    side: str,
    volume: float,
    limit_price: float,
    sl: float,
    tp: float,
) -> Optional[int]:
    """
    Place a buy-limit or sell-limit pending order at limit_price.

    The order expires after LIMIT_ORDER_EXPIRY_MINUTES (one M5 bar) so it
    does not linger if the market never retraces to the OB price.
    """
    si = mt5.symbol_info(symbol)
    if si is None:
        log("order_error", {"reason": "no_symbol_info"})
        return None

    otype = mt5.ORDER_TYPE_BUY_LIMIT if side == "buy" else mt5.ORDER_TYPE_SELL_LIMIT

    sl_snapped    = snap(sl,          si, "down" if side == "buy" else "up")
    tp_snapped    = snap(tp,          si, "up"   if side == "buy" else "down")
    price_snapped = snap(limit_price, si)

    expiry = datetime.now(timezone.utc) + timedelta(minutes=LIMIT_ORDER_EXPIRY_MINUTES)

    req = {
        "action":       mt5.TRADE_ACTION_PENDING,
        "symbol":       symbol,
        "volume":       volume,
        "type":         otype,
        "price":        price_snapped,
        "sl":           sl_snapped,
        "tp":           tp_snapped,
        "magic":        MAGIC,
        "comment":      "ob_reaction_limit",
        "type_filling": pick_filling(si),
        "type_time":    mt5.ORDER_TIME_SPECIFIED,
        "expiration":   int(expiry.timestamp()),
    }

    r = mt5.order_send(req)

    if r is None:
        log("order_error", {
            "error":       str(mt5.last_error()),
            "limit_price": price_snapped,
            "sl":          sl_snapped,
            "tp":          tp_snapped,
        })
        return None

    if r.retcode == 10018:
        log("market_closed", {"comment": r.comment, "limit_price": price_snapped})
        return None

    if r.retcode != mt5.TRADE_RETCODE_DONE:
        log("order_error", {
            "retcode":     r.retcode,
            "comment":     r.comment,
            "limit_price": price_snapped,
        })
        return None

    return r.order


def get_our_positions(symbol: str) -> list:
    """
    Return only open positions belonging to this bot (filtered by magic number).
    Prevents the bot from seeing positions opened manually or by other bots.
    """
    return [p for p in (mt5.positions_get(symbol=symbol) or []) if p.magic == MAGIC]


# ══════════════════════════════════════════════════════════════════════════════
# POSITION-CLOSE DETECTOR
# ══════════════════════════════════════════════════════════════════════════════


def _check_closed_positions(
    symbol: str,
    open_tickets: set,
    notifier: "Mt5Notifier",
) -> None:
    """
    Compare previously tracked open tickets against current live positions.
    Any ticket that has disappeared was closed (SL or TP hit).
    Fetches the closing deal profit from MT5 history and sends a balance
    notification for each closed trade.
    """
    if not open_tickets:
        return

    current_tickets = {p.ticket for p in get_our_positions(symbol)}
    closed = open_tickets - current_tickets
    if not closed:
        return

    from datetime import timedelta

    now_utc = datetime.now(timezone.utc)
    look_back = now_utc - timedelta(hours=24)

    deals = mt5.history_deals_get(look_back, now_utc) or []
    profit_by_position: dict[int, float] = {}
    for deal in deals:
        if deal.entry == mt5.DEAL_ENTRY_OUT:
            profit_by_position[deal.position_id] = (
                profit_by_position.get(deal.position_id, 0.0) + deal.profit
            )

    ai = mt5.account_info()
    balance = float(ai.balance) if ai else 0.0
    equity = float(ai.equity) if ai else 0.0

    for ticket in closed:
        profit = profit_by_position.get(ticket, 0.0)
        log(
            "position_closed_detected",
            {
                "ticket": ticket,
                "profit": round(profit, 2),
                "balance": round(balance, 2),
                "equity": round(equity, 2),
            },
        )
        notifier.notify_position_closed(ticket, profit, balance, equity)

    open_tickets -= closed  # remove closed tickets from tracking set


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CYCLE
# ══════════════════════════════════════════════════════════════════════════════


def run_cycle(
    symbol: str,
    fixed_lot: Optional[float],
    dry_run: bool,
    seen_obs: set,
    notifier: "Mt5Notifier",
    open_tickets: set,
) -> None:
    """
    Execute one full evaluation-and-order cycle.

    Called once per M5 bar close. Steps:
      1. Fetch closed M5 bars from MT5.
      2. Run strategy analysis (features -> displacements -> OBs -> signal).
      3. Skip if data is stale (market closed or connection issue).
      4. Skip if this OB was already traded (deduplication).
      5. Skip if a position is already open (one trade at a time).
      6. Calculate initial lot size from the ideal OB entry level.
      7. Measure slippage and apply three-tier response [CHANGE 3].
      8. Send the market order with (possibly adjusted) lot and TP.

    Args:
      symbol    : Exact MT5 symbol name (e.g. "XAUUSD.i").
      fixed_lot : When set, always use this lot; skip auto-sizing.
      dry_run   : Log signals but never send orders.
      seen_obs  : Set of (ob_time, direction) pairs already traded.
    """

    # ── Detect positions closed since the last cycle (SL or TP hit) ──────────
    _check_closed_positions(symbol, open_tickets, notifier)

    # ── Step 1: Fetch data ───────────────────────────────────────────────────
    try:
        df = fetch_m5(symbol, HISTORY_BARS)
    except RuntimeError as e:
        log("error", {"msg": str(e)})
        return

    # ── Step 2: Strategy analysis ────────────────────────────────────────────
    df = add_candle_features(df)
    displacements = detect_displacements(df)
    obs = find_order_blocks(df, displacements)
    signal = find_signal(df, obs)

    # How old is the newest closed bar?
    last_bar_time = df.iloc[-1]["time"]
    age_minutes = (datetime.now(timezone.utc) - last_bar_time).total_seconds() / 60

    # Log cycle state for monitoring and debugging
    log(
        "cycle",
        {
            "bars": len(df),
            "displacements": len(displacements),
            "order_blocks": len(obs),
            "last_bar_time": str(last_bar_time),
            "data_age_min": round(age_minutes, 1),
            "signal": signal is not None,
        },
    )

    # ── Step 3: Stale data guard ─────────────────────────────────────────────
    # If the newest bar is older than 15 minutes the market is closed or the
    # data feed is broken — do not trade on stale information.
    if age_minutes > 15:
        _skip_data = {
            "reason": "market_closed_or_stale",
            "data_age_min": round(age_minutes, 1),
        }
        log("skip", _skip_data)
        notifier.notify_skip(_skip_data)
        return

    if signal is None:
        return  # no signal this bar — wait for the next candle

    # ── Step 4: Deduplication ────────────────────────────────────────────────
    # Use ob_time (stable timestamp) rather than ob_bar_idx (shifts as new
    # bars arrive) to identify whether this OB was already traded.
    ob_key = (signal["ob_time"], signal["direction"])
    if ob_key in seen_obs:
        _skip_data = {
            "reason": "already_traded",
            "ob_time": signal["ob_time"],
            "direction": signal["direction"],
        }
        log("skip", _skip_data)
        notifier.notify_skip(_skip_data)
        return

    # ── Step 5: One position at a time ───────────────────────────────────────
    # The bot holds at most one open position. Log the missed signal so that
    # post-trade analysis can identify how much opportunity was lost.
    positions = get_our_positions(symbol)
    if positions:
        _skip_data = {
            "reason": "position_open",
            "open_positions": len(positions),
            "missed_signal": {
                "direction": signal["direction"],
                "ob_time": signal["ob_time"],
                "entry": signal["entry"],
                "sl": signal["sl"],
                "tp": signal["tp"],
            },
        }
        log("skip", _skip_data)
        notifier.notify_skip(_skip_data)
        return

    # Signal is valid — log it before the order-execution steps
    log("signal", signal)
    notifier.notify_signal(signal)

    # In dry-run mode: record the signal but do not send any order
    if dry_run:
        log("dry_run", {"msg": "no_order_sent"})
        seen_obs.add(ob_key)
        save_seen_obs(seen_obs)
        return

    # ── Step 6: Initial lot sizing ───────────────────────────────────────────
    # Size the position from the ideal OB entry level (ob_high or ob_low).
    # The slippage check in Step 7 may reduce this if the market has moved.
    ai = mt5.account_info()
    balance = float(ai.balance) if ai else 1000.0
    side = "buy" if signal["direction"] == "BUY" else "sell"

    if fixed_lot is not None:
        # User specified a fixed lot — use it as-is, no auto-sizing
        volume_original = fixed_lot
    else:
        # Auto-size: risk exactly RISK_PER_TRADE % from ideal entry to SL
        volume_original = calc_volume(
            symbol, side, signal["entry"], signal["sl"], balance
        )

    # Reject if the calculated lot is below the broker minimum
    si = mt5.symbol_info(symbol)
    if si and volume_original < si.volume_min:
        _skip_data = {
            "reason": "volume_too_small",
            "volume": volume_original,
            "min": si.volume_min,
        }
        log("skip", _skip_data)
        notifier.notify_skip(_skip_data)
        return

    # ── Step 7: Slippage measurement and three-tier response ─────────────────
    #
    # Slippage definition:
    #   BUY:  slippage = tick.ask - ob_high  (positive = market above OB)
    #   SELL: slippage = ob_low - tick.bid   (positive = market below OB)
    #
    # Three tiers:
    #   Tier 1 (slippage <= SLIPPAGE_LIMIT_THRESHOLD = 4.0 pts):
    #       Place a LIMIT order at the exact OB price. The order is valid for
    #       LIMIT_ORDER_EXPIRY_MINUTES so it fills only if price retraces.
    #       Original SL and TP are used — no RR distortion.
    #
    #   Tier 2 (4.0 < slippage < SLIPPAGE_MAX_POINTS = 6.0 pts):
    #       Market order with a reduced lot so that dollar risk from the fill
    #       price to the structural SL stays at RISK_PER_TRADE %.
    #       TP is recalculated from the fill price to keep RR = 2.
    #
    #   Tier 3 (slippage >= SLIPPAGE_MAX_POINTS = 6.0 pts):
    #       Skip entirely — too far from the OB for the thesis to hold.

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        log("error", {"msg": "could not read tick for slippage check"})
        return

    market_price = tick.ask if signal["direction"] == "BUY" else tick.bid

    if signal["direction"] == "BUY":
        slippage = market_price - signal["entry"]
    else:
        slippage = signal["entry"] - market_price

    # ── Tier 3: hard skip ────────────────────────────────────────────────────
    if slippage >= SLIPPAGE_MAX_POINTS:
        _skip_data = {
            "reason":       "slippage_exceeded",
            "direction":    signal["direction"],
            "ob_entry":     signal["entry"],
            "market_price": round(market_price, 2),
            "slippage_pts": round(slippage, 2),
            "max_pts":      SLIPPAGE_MAX_POINTS,
        }
        log("skip", _skip_data)
        notifier.notify_skip(_skip_data)
        return

    # A LIMIT order is only accepted by the broker when entry sits on the far
    # side of the current price by at least trade_stops_level. Otherwise MT5
    # returns retcode 10015 (Invalid price):
    #   SELL LIMIT: entry must be > ask + stops_dist
    #   BUY  LIMIT: entry must be < bid - stops_dist
    # When invalid, skip the limit attempt and fall through to a market order
    # so the signal is still taken immediately.
    si_chk = mt5.symbol_info(symbol)
    _pt = float(si_chk.point) if si_chk and si_chk.point else 0.01
    _stops_dist = (int(getattr(si_chk, "trade_stops_level", 0) or 0) * _pt) if si_chk else 0.0
    if side == "sell":
        limit_valid = (signal["entry"] - tick.ask) > _stops_dist
    else:
        limit_valid = (tick.bid - signal["entry"]) > _stops_dist

    # ── Tier 1: limit order at exact OB price ────────────────────────────────
    if limit_valid and slippage <= SLIPPAGE_LIMIT_THRESHOLD:
        ticket = send_limit_order(
            symbol, side,
            volume_original,
            signal["entry"],   # limit price = exact OB level
            signal["sl"],
            signal["tp"],
        )
        if ticket:
            seen_obs.add(ob_key)
            save_seen_obs(seen_obs)
            # Pending order is not a position yet — tracked via get_our_positions()
            # once it fills; we do not add it to open_tickets to avoid false-close.
            _placed_data = {
                "ticket":       ticket,
                "order_type":   "limit",
                "direction":    signal["direction"],
                "volume":       volume_original,
                "limit_price":  signal["entry"],
                "sl":           signal["sl"],
                "tp":           signal["tp"],
                "slippage_pts": round(slippage, 2),
                "expiry_min":   LIMIT_ORDER_EXPIRY_MINUTES,
                "ob_high":      signal["ob_high"],
                "ob_low":       signal["ob_low"],
                "ob_time":      signal["ob_time"],
            }
            log("limit_order_placed", _placed_data)
            notifier.notify_order_placed(_placed_data)
        else:
            log("order_failed", {"direction": signal["direction"],
                                 "sl": signal["sl"], "tp": signal["tp"]})
        return

    # Limit was skipped because market already passed entry — log and fall
    # through to the Tier 2 market-order path so the trade still gets in.
    if not limit_valid:
        log("limit_invalid_fallback_market", {
            "direction":    signal["direction"],
            "ob_entry":     signal["entry"],
            "ask":          round(tick.ask, 2),
            "bid":          round(tick.bid, 2),
            "stops_dist":   round(_stops_dist, 5),
            "slippage_pts": round(slippage, 2),
        })

    # ── Tier 2: market order, reduced lot (high slippage OR invalid limit) ───
    if fixed_lot is None:
        volume_final = calc_volume(symbol, side, market_price, signal["sl"], balance)
    else:
        volume_final = fixed_lot

    risk_from_fill = abs(market_price - signal["sl"])
    if signal["direction"] == "BUY":
        tp_final = round(market_price + risk_from_fill * RISK_REWARD, 2)
    else:
        tp_final = round(market_price - risk_from_fill * RISK_REWARD, 2)

    _adj_data = {
        "direction":       signal["direction"],
        "ob_entry":        signal["entry"],
        "market_price":    round(market_price, 2),
        "slippage_pts":    round(slippage, 2),
        "sl_unchanged":    signal["sl"],
        "volume_original": volume_original,
        "volume_adjusted": volume_final,
        "tp_original":     signal["tp"],
        "tp_adjusted":     tp_final,
    }
    log("slippage_adjusted", _adj_data)
    notifier.notify_slippage_adjusted(_adj_data)

    # ── Step 8: Send market order (Tier 2 only) ───────────────────────────────
    ticket = send_market_order(
        symbol, side,
        volume_final,
        signal["sl"],
        tp_final,
    )

    if ticket:
        seen_obs.add(ob_key)
        save_seen_obs(seen_obs)
        open_tickets.add(ticket)

        _placed_data = {
            "ticket":       ticket,
            "order_type":   "market",
            "direction":    signal["direction"],
            "volume":       volume_final,
            "sl":           signal["sl"],
            "tp":           tp_final,
            "slippage_pts": round(slippage, 2),
            "ob_entry":     signal["entry"],
            "ob_high":      signal["ob_high"],
            "ob_low":       signal["ob_low"],
            "ob_time":      signal["ob_time"],
        }
        log("order_placed", _placed_data)
        notifier.notify_order_placed(_placed_data)
    else:
        log("order_failed", {"direction": signal["direction"],
                             "sl": signal["sl"], "tp": tp_final})


# ══════════════════════════════════════════════════════════════════════════════
# TIMING
# ══════════════════════════════════════════════════════════════════════════════


def sleep_until_next_m5(extra: float = 0.1) -> None:
    """
    Sleep until just after the next M5 bar closes.

    [CHANGE 1] extra reduced from 2.0 s to 0.1 s.

    Why this matters:
    The previous 2 s buffer, combined with ~1-2 s of processing time, meant
    the order was sent 3-4 s after the candle closed. On XAUUSD that window
    allows 1-3 pts of price movement, adding unnecessary slippage.

    With extra=0.1 s the bot wakes up almost immediately after the bar closes,
    leaving far less time for price to drift before the order is placed.
    The 0.1 s is enough for the MT5 data feed to register the new closed bar.

    Formula:
      next_close = (floor(now / 300) + 1) * 300   (Unix seconds)
      delay      = next_close - now + extra
    """
    now = time.time()
    delay = max(1.0, (int(now // M5_SECONDS) + 1) * M5_SECONDS - now + extra)
    print(f"  -> sleeping {delay:.0f}s until next M5 close ...")
    time.sleep(delay)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """
    Parse command-line arguments, connect to MT5, and run the main loop.

    Command-line flags:
      --symbol  : MT5 symbol name (default: XAUUSD)
      --lot     : Fixed lot size; omit to use automatic 1 % risk sizing
      --risk    : Risk fraction for auto-sizing (default: 0.01)
      --once    : Run one evaluation cycle then exit
      --dry-run : Log signals but send no orders
    """
    p = argparse.ArgumentParser(description="Order Block Reaction Bot — XAUUSD M5")
    p.add_argument("--symbol", default=SYMBOL, help="MT5 symbol name (default: XAUUSD)")
    p.add_argument(
        "--lot",
        type=float,
        default=None,
        help="Fixed lot size; omit for automatic 1%% risk sizing",
    )
    p.add_argument(
        "--risk",
        type=float,
        default=0.01,
        help="Risk fraction for auto-sizing (default: 0.01)",
    )
    p.add_argument("--once", action="store_true", help="Run one evaluation then exit")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Log signals only — do not send any orders",
    )
    args = p.parse_args()

    global RISK_PER_TRADE
    RISK_PER_TRADE = args.risk

    mt5_connect()
    assert_terminal_ready()
    symbol = resolve_symbol(args.symbol)

    # Log all active parameters at startup for auditing and debugging
    log(
        "bot_start",
        {
            "symbol": symbol,
            "lot": args.lot,
            "risk_per_trade": RISK_PER_TRADE,
            "dry_run": args.dry_run,
            "displacement_min_candles": DISPLACEMENT_MIN_CANDLES,
            "displacement_atr_mult": DISPLACEMENT_ATR_MULT,
            "ob_expiry_bars": OB_EXPIRY_BARS,
            "rejection_wick_ratio": REJECTION_WICK_RATIO,
            "risk_reward": RISK_REWARD,
            "sl_buffer": SL_BUFFER,
            "slippage_limit_threshold": SLIPPAGE_LIMIT_THRESHOLD,
            "slippage_max_points": SLIPPAGE_MAX_POINTS,
            "magic": MAGIC,
            "log_path": str(LOG_PATH),
        },
    )

    # Restore OBs already traded in previous sessions
    seen_obs: set = load_seen_obs()

    # Telegram notifier (reads token/chat_id from .env; no-ops if unconfigured)
    notifier = Mt5Notifier()
    open_tickets: set = set()

    try:
        if args.once:
            run_cycle(symbol, args.lot, args.dry_run, seen_obs, notifier, open_tickets)
            return

        print(f"Order Block Bot running on {symbol} M5. Press Ctrl+C to stop.")
        while True:
            run_cycle(symbol, args.lot, args.dry_run, seen_obs, notifier, open_tickets)
            sleep_until_next_m5(extra=0.1)  # [CHANGE 1]

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        log("bot_stop", {})
        mt5.shutdown()


if __name__ == "__main__":
    main()
