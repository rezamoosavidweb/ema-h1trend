#!/usr/bin/env python3
"""
Order Block Reaction Bot — XAUUSD M5 (Errante MT5)

Mirrors notebooks/08_order_block_reaction.ipynb exactly:
  displacement detection → order block → FVG → retest → rejection → market order

Run:
  python mt5/run_ob_xauusd.py              # loop on M5 closes
  python mt5/run_ob_xauusd.py --once       # single evaluation
  python mt5/run_ob_xauusd.py --dry-run    # signal only, no orders
  python mt5/run_ob_xauusd.py --lot 0.02   # fixed lot size

Env (optional — skip if Errante terminal is already open and logged in):
  MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_TERMINAL_PATH

Log: logs/XAUUSD.json  (JSON Lines — one event per line)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    print("Install MetaTrader5: pip install MetaTrader5", file=sys.stderr)
    raise

# ── Paths ─────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = _REPO_ROOT / "logs" / "XAUUSD.json"

# ── Config (matches notebook) ─────────────────────────────────────────────────
SYMBOL                   = "XAUUSD"
HISTORY_BARS             = 600          # M5 bars to fetch (needs warmup for ATR + displacements)
ATR_PERIOD               = 14
DISPLACEMENT_MIN_CANDLES = 4
DISPLACEMENT_ATR_MULT    = 1.5
OB_EXPIRY_BARS           = 100
REJECTION_WICK_RATIO     = 0.3          # lower wick / range threshold
RISK_REWARD              = 2.0
SL_BUFFER                = 0.5          # extra points beyond OB edge for SL
RISK_PER_TRADE           = 0.01         # 1% of balance (used when --lot not given)
MAGIC                    = 8088080      # unique magic number — separates this bot's orders
M5_SECONDS               = 300


# ── Logging ───────────────────────────────────────────────────────────────────

def log(event: str, data: dict | None = None) -> None:
    """Append one JSON line to LOG_PATH and print a summary."""
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
    # Console summary
    short = {k: v for k, v in (data or {}).items() if k not in ("ob_bar_idx",)}
    print(f"[{entry['ts']}] {event} | {short}")


# ── Strategy logic (verbatim from notebook) ───────────────────────────────────

def add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tr"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            abs(df["high"] - df["close"].shift(1)),
            abs(df["low"]  - df["close"].shift(1)),
        ),
    )
    df["atr"]        = df["tr"].ewm(span=ATR_PERIOD, adjust=False).mean()
    df["body"]       = abs(df["close"] - df["open"])
    df["body_top"]   = df[["open", "close"]].max(axis=1)
    df["body_bot"]   = df[["open", "close"]].min(axis=1)
    df["upper_wick"] = df["high"]     - df["body_top"]
    df["lower_wick"] = df["body_bot"] - df["low"]
    df["range"]      = df["high"] - df["low"]
    df["is_bull"]    = df["close"] > df["open"]
    df["is_bear"]    = df["close"] < df["open"]
    return df


@dataclass
class Displacement:
    start_idx:   int
    end_idx:     int
    direction:   str    # 'UP' or 'DOWN'
    total_move:  float
    candle_count: int


def detect_displacements(df: pd.DataFrame) -> List[Displacement]:
    """Find all consecutive bull/bear runs that qualify as displacements."""
    displacements: List[Displacement] = []
    n = len(df)
    i = 0
    while i < n:
        if df.iloc[i]["is_bull"]:
            j = i + 1
            while j < n and df.iloc[j]["is_bull"]:
                j += 1
            if j - i >= DISPLACEMENT_MIN_CANDLES:
                seg  = df.iloc[i:j]
                move = seg["close"].iloc[-1] - seg["open"].iloc[0]
                if move >= DISPLACEMENT_ATR_MULT * seg["atr"].mean():
                    displacements.append(Displacement(i, j - 1, "UP", move, j - i))
            i = j
        elif df.iloc[i]["is_bear"]:
            j = i + 1
            while j < n and df.iloc[j]["is_bear"]:
                j += 1
            if j - i >= DISPLACEMENT_MIN_CANDLES:
                seg  = df.iloc[i:j]
                move = seg["open"].iloc[0] - seg["close"].iloc[-1]
                if move >= DISPLACEMENT_ATR_MULT * seg["atr"].mean():
                    displacements.append(Displacement(i, j - 1, "DOWN", move, j - i))
            i = j
        else:
            i += 1
    return displacements


@dataclass
class OrderBlock:
    ob_bar_idx:   int
    ob_type:      str    # 'bullish' or 'bearish'
    ob_high:      float
    ob_low:       float
    ob_time:      object
    displaced_by: float


def find_order_blocks(df: pd.DataFrame, displacements: List[Displacement]) -> List[OrderBlock]:
    """For each displacement, find the last opposite candle before it — that is the Order Block."""
    obs: List[OrderBlock] = []
    for disp in displacements:
        start_i = disp.start_idx
        look_back = max(0, start_i - 20)
        if disp.direction == "UP":
            for k in range(start_i - 1, look_back - 1, -1):
                if df.iloc[k]["is_bear"]:
                    bar = df.iloc[k]
                    obs.append(OrderBlock(k, "bullish", bar["high"], bar["low"], bar["time"], disp.total_move))
                    break
        else:
            for k in range(start_i - 1, look_back - 1, -1):
                if df.iloc[k]["is_bull"]:
                    bar = df.iloc[k]
                    obs.append(OrderBlock(k, "bearish", bar["high"], bar["low"], bar["time"], disp.total_move))
                    break
    return obs


def is_rejection_candle(bar: pd.Series, ob_type: str) -> bool:
    if bar["range"] == 0:
        return False
    if ob_type == "bullish":
        return bar["lower_wick"] / bar["range"] > REJECTION_WICK_RATIO
    return bar["upper_wick"] / bar["range"] > REJECTION_WICK_RATIO


def find_signal(df: pd.DataFrame, order_blocks: List[OrderBlock]) -> Optional[dict]:
    """
    Check the last closed bar for a valid OB retest + rejection candle.
    Replicates run_ob_backtest() from notebook, evaluated on bar n-1.
    """
    n        = len(df)
    last_idx = n - 1
    last_bar = df.iloc[last_idx]

    for ob in order_blocks:
        # OB must be before last bar, and not expired
        if ob.ob_bar_idx >= last_idx:
            continue
        if last_idx - ob.ob_bar_idx > OB_EXPIRY_BARS:
            continue

        if ob.ob_type == "bullish":
            # Walk bars between OB and last bar to find displacement + invalidation
            displaced   = False
            invalidated = False
            for k in range(ob.ob_bar_idx + 1, last_idx):
                bar_k = df.iloc[k]
                if not displaced:
                    if bar_k["close"] > ob.ob_high:
                        displaced = True
                else:
                    # OB invalid if price closes below OB low after displacement
                    if bar_k["close"] < ob.ob_low:
                        invalidated = True
                        break

            if not displaced or invalidated:
                continue

            # Last bar must touch OB zone and be a bullish rejection
            if last_bar["low"] <= ob.ob_high and last_bar["close"] >= ob.ob_low:
                if is_rejection_candle(last_bar, ob.ob_type):
                    entry = ob.ob_high
                    sl    = ob.ob_low - SL_BUFFER
                    risk  = entry - sl
                    tp    = entry + risk * RISK_REWARD
                    return {
                        "direction":   "BUY",
                        "ob_type":     ob.ob_type,
                        "ob_high":     ob.ob_high,
                        "ob_low":      ob.ob_low,
                        "ob_time":     str(ob.ob_time),
                        "retest_time": str(last_bar["time"]),
                        "entry":       round(entry, 2),
                        "sl":          round(sl,    2),
                        "tp":          round(tp,    2),
                        "ob_bar_idx":  ob.ob_bar_idx,
                        "displaced_by": round(ob.displaced_by, 2),
                    }

        else:  # bearish OB → SELL
            displaced   = False
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
                    entry = ob.ob_low
                    sl    = ob.ob_high + SL_BUFFER
                    risk  = sl - entry
                    tp    = entry - risk * RISK_REWARD
                    return {
                        "direction":   "SELL",
                        "ob_type":     ob.ob_type,
                        "ob_high":     ob.ob_high,
                        "ob_low":      ob.ob_low,
                        "ob_time":     str(ob.ob_time),
                        "retest_time": str(last_bar["time"]),
                        "entry":       round(entry, 2),
                        "sl":          round(sl,    2),
                        "tp":          round(tp,    2),
                        "ob_bar_idx":  ob.ob_bar_idx,
                        "displaced_by": round(ob.displaced_by, 2),
                    }

    return None


# ── MT5 helpers ───────────────────────────────────────────────────────────────

def mt5_connect() -> None:
    kwargs: dict = {}
    login    = os.environ.get("MT5_LOGIN")
    password = os.environ.get("MT5_PASSWORD")
    server   = os.environ.get("MT5_SERVER")
    if login and password and server:
        kwargs["login"]    = int(login)
        kwargs["password"] = password
        kwargs["server"]   = server
    path = os.environ.get("MT5_TERMINAL_PATH")
    if path:
        kwargs["path"] = path
    if not mt5.initialize(**kwargs):
        raise RuntimeError(f"mt5.initialize failed: {mt5.last_error()}")


def assert_terminal_ready() -> None:
    ti = mt5.terminal_info()
    if ti is None:
        raise RuntimeError(
            f"terminal_info() returned None: {mt5.last_error()}\n"
            "Start Errante MT5, log in, enable AutoTrading, then retry.\n"
            "Use 64-bit Python only."
        )
    if not ti.connected:
        raise RuntimeError("MT5 terminal not connected — wait for quotes to appear in Market Watch.")


def resolve_symbol(symbol: str) -> str:
    mt5.symbol_select(symbol, True)
    si = mt5.symbol_info(symbol)
    if si is None:
        # Try common suffixes used by brokers
        for suffix in (".a", ".i", "#", "-raw"):
            candidate = symbol + suffix
            mt5.symbol_select(candidate, True)
            si = mt5.symbol_info(candidate)
            if si is not None:
                break
    if si is None:
        raise RuntimeError(
            f"Symbol {symbol!r} not found in MT5.\n"
            "Open Market Watch → right-click → Show All, find the exact name, "
            "then pass it with --symbol."
        )
    mt5.symbol_select(si.name, True)
    return si.name


def fetch_m5(symbol: str, bars: int) -> pd.DataFrame:
    r = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, bars)
    if r is None or len(r) == 0:
        raise RuntimeError(f"No M5 data for {symbol}: {mt5.last_error()}")
    df = pd.DataFrame(r)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={"tick_volume": "volume"})
    keep = ["time", "open", "high", "low", "close", "volume"]
    df = df[[c for c in keep if c in df.columns]].copy()
    return df.iloc[:-1].reset_index(drop=True)   # drop the still-forming bar


def _tick_size(si) -> float:
    ts = float(getattr(si, "trade_tick_size", 0) or 0)
    if ts > 0:
        return ts
    pt = float(si.point or 0)
    return pt if pt > 0 else float(10 ** (-si.digits))


def pick_filling(si) -> int:
    ioc = getattr(mt5, "SYMBOL_FILLING_IOC", 2)
    fok = getattr(mt5, "SYMBOL_FILLING_FOK", 1)
    fm  = int(si.filling_mode)
    if fm & ioc:
        return mt5.ORDER_FILLING_IOC
    if fm & fok:
        return mt5.ORDER_FILLING_FOK
    return mt5.ORDER_FILLING_RETURN


def snap(price: float, si, mode: str = "nearest") -> float:
    t = _tick_size(si)
    d = int(si.digits)
    x = price / t
    if mode == "up":
        v = math.ceil(x  - 1e-12) * t
    elif mode == "down":
        v = math.floor(x + 1e-12) * t
    else:
        v = round(x) * t
    return round(v, d)


def normalize_vol(vol: float, si) -> float:
    step = float(si.volume_step or 0.01)
    vmin = float(si.volume_min or 0.01)
    vmax = float(si.volume_max or 100.0)
    v = math.floor(vol / step + 1e-12) * step
    return float(max(vmin, min(vmax, v)))


def calc_volume(symbol: str, side: str, entry: float, sl: float, balance: float) -> float:
    """Lot size so that loss at SL = 1% of balance."""
    otype  = mt5.ORDER_TYPE_BUY if side == "buy" else mt5.ORDER_TYPE_SELL
    pnl_1l = mt5.order_calc_profit(otype, symbol, 1.0, entry, sl)
    if pnl_1l is None or abs(pnl_1l) < 1e-9:
        return 0.01
    risk_cash = balance * RISK_PER_TRADE
    si        = mt5.symbol_info(symbol)
    return normalize_vol(risk_cash / abs(pnl_1l), si)


def send_market_order(symbol: str, side: str, volume: float, sl: float, tp: float) -> Optional[int]:
    si   = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if si is None or tick is None:
        log("order_error", {"reason": "no_symbol_or_tick"})
        return None

    price = tick.ask if side == "buy" else tick.bid
    otype = mt5.ORDER_TYPE_BUY if side == "buy" else mt5.ORDER_TYPE_SELL

    # Snap SL/TP to broker tick grid
    sl_snapped = snap(sl, si, "down" if side == "buy" else "up")
    tp_snapped = snap(tp, si, "up"   if side == "buy" else "down")

    # Enforce min stop distance
    point      = float(si.point or 10 ** (-si.digits))
    stops_pts  = int(getattr(si, "trade_stops_level", 0) or 0)
    min_dist   = stops_pts * point
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
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       symbol,
        "volume":       volume,
        "type":         otype,
        "price":        price,
        "sl":           sl_snapped,
        "tp":           tp_snapped,
        "magic":        MAGIC,
        "comment":      "ob_reaction",
        "type_filling": pick_filling(si),
        "deviation":    int(os.environ.get("MT5_DEVIATION_POINTS", "50")),
    }

    r = mt5.order_send(req)
    if r is None:
        log("order_error", {"error": str(mt5.last_error()), "req_price": price, "sl": sl_snapped, "tp": tp_snapped})
        return None
    if r.retcode == 10018:  # Market closed (weekend / outside session)
        log("market_closed", {"comment": r.comment, "req_price": price})
        return None
    if r.retcode != mt5.TRADE_RETCODE_DONE:
        log("order_error", {"retcode": r.retcode, "comment": r.comment, "req_price": price})
        return None

    return r.order


def get_our_positions(symbol: str) -> list:
    return [p for p in (mt5.positions_get(symbol=symbol) or []) if p.magic == MAGIC]


# ── Main cycle ────────────────────────────────────────────────────────────────

def run_cycle(symbol: str, fixed_lot: Optional[float], dry_run: bool, seen_obs: set) -> None:
    # 1. Fetch data
    try:
        df = fetch_m5(symbol, HISTORY_BARS)
    except RuntimeError as e:
        log("error", {"msg": str(e)})
        return

    # 2. Strategy analysis (same as notebook)
    df            = add_candle_features(df)
    displacements = detect_displacements(df)
    obs           = find_order_blocks(df, displacements)
    signal        = find_signal(df, obs)

    last_bar_time = df.iloc[-1]["time"]
    age_minutes   = (datetime.now(timezone.utc) - last_bar_time).total_seconds() / 60

    log("cycle", {
        "bars":          len(df),
        "displacements": len(displacements),
        "order_blocks":  len(obs),
        "last_bar_time": str(last_bar_time),
        "data_age_min":  round(age_minutes, 1),
        "signal":        signal is not None,
    })

    # Market is closed / data is stale — no point trading on old bars
    if age_minutes > 15:
        log("skip", {"reason": "market_closed_or_stale", "data_age_min": round(age_minutes, 1)})
        return

    if signal is None:
        return

    # 3. Deduplicate — same OB + direction already traded this session
    # Use ob_time (stable timestamp) not ob_bar_idx (shifts as new bars arrive)
    ob_key = (signal["ob_time"], signal["direction"])
    if ob_key in seen_obs:
        log("skip", {"reason": "already_traded", "ob_time": signal["ob_time"], "direction": signal["direction"]})
        return

    # 4. Skip if we already have an open position
    positions = get_our_positions(symbol)
    if positions:
        log("skip", {"reason": "position_open", "open_positions": len(positions)})
        return

    log("signal", signal)

    if dry_run:
        log("dry_run", {"msg": "no_order_sent"})
        seen_obs.add(ob_key)
        return

    # 5. Size and send order
    ai      = mt5.account_info()
    balance = float(ai.balance) if ai else 1000.0
    side    = "buy" if signal["direction"] == "BUY" else "sell"

    if fixed_lot is not None:
        volume = fixed_lot
    else:
        volume = calc_volume(symbol, side, signal["entry"], signal["sl"], balance)

    si = mt5.symbol_info(symbol)
    if si and volume < si.volume_min:
        log("skip", {"reason": "volume_too_small", "volume": volume, "min": si.volume_min})
        return

    ticket = send_market_order(symbol, side, volume, signal["sl"], signal["tp"])
    if ticket:
        seen_obs.add(ob_key)
        log("order_placed", {
            "ticket":    ticket,
            "direction": signal["direction"],
            "volume":    volume,
            "sl":        signal["sl"],
            "tp":        signal["tp"],
            "ob_high":   signal["ob_high"],
            "ob_low":    signal["ob_low"],
            "ob_time":   signal["ob_time"],
        })
    else:
        log("order_failed", {"direction": signal["direction"], "sl": signal["sl"], "tp": signal["tp"]})


def sleep_until_next_m5(extra: float = 2.0) -> None:
    """Sleep until just after the next M5 bar closes."""
    now   = time.time()
    delay = max(1.0, (int(now // M5_SECONDS) + 1) * M5_SECONDS - now + extra)
    print(f"  → sleeping {delay:.0f}s until next M5 close …")
    time.sleep(delay)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Order Block Reaction Bot — XAUUSD M5")
    p.add_argument("--symbol",  default=SYMBOL,      help="MT5 symbol name (default XAUUSD)")
    p.add_argument("--lot",     type=float, default=None, help="Fixed lot size; omit for auto 1%% risk")
    p.add_argument("--risk",    type=float, default=0.01, help="Risk fraction when auto-sizing (default 0.01)")
    p.add_argument("--once",    action="store_true",  help="Run one evaluation then exit")
    p.add_argument("--dry-run", action="store_true",  help="Log signals but do not send orders")
    args = p.parse_args()

    global RISK_PER_TRADE
    RISK_PER_TRADE = args.risk

    mt5_connect()
    assert_terminal_ready()
    symbol = resolve_symbol(args.symbol)

    log("bot_start", {
        "symbol":                  symbol,
        "lot":                     args.lot,
        "risk_per_trade":          RISK_PER_TRADE,
        "dry_run":                 args.dry_run,
        "displacement_min_candles": DISPLACEMENT_MIN_CANDLES,
        "displacement_atr_mult":   DISPLACEMENT_ATR_MULT,
        "ob_expiry_bars":          OB_EXPIRY_BARS,
        "rejection_wick_ratio":    REJECTION_WICK_RATIO,
        "risk_reward":             RISK_REWARD,
        "sl_buffer":               SL_BUFFER,
        "magic":                   MAGIC,
        "log_path":                str(LOG_PATH),
    })

    seen_obs: set = set()   # tracks (ob_bar_idx, direction) already acted on this session

    try:
        if args.once:
            run_cycle(symbol, args.lot, args.dry_run, seen_obs)
            return

        print(f"Order Block Bot running on {symbol} M5. Ctrl+C to stop.")
        while True:
            run_cycle(symbol, args.lot, args.dry_run, seen_obs)
            sleep_until_next_m5()

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        log("bot_stop", {})
        mt5.shutdown()


if __name__ == "__main__":
    main()
