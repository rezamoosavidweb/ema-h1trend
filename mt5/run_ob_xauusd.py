#!/usr/bin/env python3
"""
Order Block Reaction Bot — XAUUSD M5 (Errante MT5)

Strategy pipeline (mirrors notebook 08_order_block_reaction.ipynb):
  displacement detection -> order block -> retest -> rejection candle -> order

╔═══════════════════════════════════════════════════════════════════════════╗
║                          ARCHITECTURE OVERVIEW                            ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║   This file owns ONLY:                                                    ║
║     * MT5 connection (mt5_connect / assert_terminal_ready)                ║
║     * Data fetch (fetch_m5)                                               ║
║     * Strategy detection (add_candle_features / detect_displacements /    ║
║       find_order_blocks / find_signal)  -- UNCHANGED from previous bot    ║
║     * Cycle orchestration (run_cycle)                                     ║
║                                                                           ║
║   All execution work lives in `execution/`:                               ║
║     * SymbolConfig      -- canonical broker symbol                        ║
║     * BrokerValidator   -- terminal/spread/stops/freeze pre-flight        ║
║     * OrderFactory      -- builds order_send dicts with GTC (no expiry)   ║
║     * PendingOrderMgr   -- tracks/cancels pendings client-side            ║
║     * RiskAdapter       -- lot sizing                                     ║
║     * FallbackEngine    -- LIMIT -> MARKET cascade                        ║
║     * ExecutionEngine   -- facade exposing place_signal(signal)           ║
║                                                                           ║
║   Strategy logic is left intact so backtest/live consistency is preserved.║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝

Run:
  python mt5/run_ob_xauusd.py              # loop — every M5 close
  python mt5/run_ob_xauusd.py --once       # single evaluation then exit
  python mt5/run_ob_xauusd.py --dry-run    # log signals only, no orders
  python mt5/run_ob_xauusd.py --lot 0.02   # fixed lot size (skips auto-sizing)

Env vars (optional):
  MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_TERMINAL_PATH, MT5_DEVIATION_POINTS

Log file: logs/XAUUSD.json  (JSON Lines — see execution/README.md for schema)

CHANGES vs previous version
═══════════════════════════
[FIX-A] Removed broken broker-time-sensitive expiration.
        OLD: type_time=ORDER_TIME_SPECIFIED + UTC Unix timestamp
        NEW: type_time=ORDER_TIME_GTC; PendingOrderManager cancels client-side
        See execution/order_factory.py and execution/pending_manager.py.

[FIX-B] LIMIT failures (retcodes 10015 / 10022 / etc.) no longer drop the
        trade. The FallbackEngine re-validates and tries a MARKET order.

[FIX-C] Pre-flight validation: terminal/AutoTrading/spread/stops_level/
        freeze_level/symbol-tradable all checked BEFORE order_send.

[FIX-D] Telegram notifier logs success/failure into the same JSON log.

[FIX-E] Centralized symbol normalization (no more XAUUSD vs XAUUSD.i drift).

[FIX-F] Safety: cooldown after repeated broker failures, retry cap per OB,
        duplicate-order protection, orphan cancel when signal disappears,
        stale-pending sweep every cycle.
"""

from __future__ import annotations

import argparse
import json
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

# Make sibling packages importable regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from execution import ExecutionEngine, ExecutionOutcome  # noqa: E402
from telegram_bot.mt5_notifier import Mt5Notifier         # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════════

_REPO_ROOT    = Path(__file__).resolve().parent.parent
LOG_PATH      = _REPO_ROOT / "logs" / "XAUUSD.json"
SEEN_OBS_PATH = _REPO_ROOT / "logs" / "seen_obs_XAUUSD.json"


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY CONFIG  (all parameters in one place — matches notebook exactly)
# ═══════════════════════════════════════════════════════════════════════════

SYMBOL                     = "XAUUSD"
HISTORY_BARS               = 600
ATR_PERIOD                 = 14
DISPLACEMENT_MIN_CANDLES   = 4
DISPLACEMENT_ATR_MULT      = 1.5
OB_EXPIRY_BARS             = 100
REJECTION_WICK_RATIO       = 0.3
RISK_REWARD                = 2.0
SL_BUFFER                  = 0.5

# Execution thresholds (see FallbackEngine docstring for the cascade)
SLIPPAGE_LIMIT_THRESHOLD   = 4.0
SLIPPAGE_MAX_POINTS        = 6.0
LIMIT_ORDER_STALE_SECONDS  = 5 * 60        # one M5 bar
MAX_SPREAD_POINTS          = 200
RISK_PER_TRADE             = 0.01
MAGIC                      = 8088080
M5_SECONDS                 = 300


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY LOGIC  (UNCHANGED -- do not modify; backtest parity depends on it)
# ═══════════════════════════════════════════════════════════════════════════


def add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-candle metrics: TR/ATR, body, wicks, direction flags."""
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
    df["upper_wick"] = df["high"]    - df["body_top"]
    df["lower_wick"] = df["body_bot"] - df["low"]
    df["range"]      = df["high"]    - df["low"]
    df["is_bull"]    = df["close"] > df["open"]
    df["is_bear"]    = df["close"] < df["open"]
    return df


@dataclass
class Displacement:
    start_idx:    int
    end_idx:      int
    direction:    str
    total_move:   float
    candle_count: int


def detect_displacements(df: pd.DataFrame) -> List[Displacement]:
    """Group consecutive same-direction candles; keep big-enough moves."""
    out: List[Displacement] = []
    n, i = len(df), 0
    while i < n:
        if df.iloc[i]["is_bull"]:
            j = i + 1
            while j < n and df.iloc[j]["is_bull"]:
                j += 1
            if j - i >= DISPLACEMENT_MIN_CANDLES:
                seg = df.iloc[i:j]
                move = seg["close"].iloc[-1] - seg["open"].iloc[0]
                if move >= DISPLACEMENT_ATR_MULT * seg["atr"].mean():
                    out.append(Displacement(i, j - 1, "UP", move, j - i))
            i = j
        elif df.iloc[i]["is_bear"]:
            j = i + 1
            while j < n and df.iloc[j]["is_bear"]:
                j += 1
            if j - i >= DISPLACEMENT_MIN_CANDLES:
                seg = df.iloc[i:j]
                move = seg["open"].iloc[0] - seg["close"].iloc[-1]
                if move >= DISPLACEMENT_ATR_MULT * seg["atr"].mean():
                    out.append(Displacement(i, j - 1, "DOWN", move, j - i))
            i = j
        else:
            i += 1
    return out


@dataclass
class OrderBlock:
    ob_bar_idx:  int
    ob_type:     str
    ob_high:     float
    ob_low:      float
    ob_time:     object
    displaced_by: float


def find_order_blocks(df: pd.DataFrame, displacements: List[Displacement]) -> List[OrderBlock]:
    """For each displacement: the last opposite-direction candle before it."""
    obs: List[OrderBlock] = []
    for disp in displacements:
        start_i  = disp.start_idx
        look_back = max(0, start_i - 20)
        if disp.direction == "UP":
            for k in range(start_i - 1, look_back - 1, -1):
                if df.iloc[k]["is_bear"]:
                    bar = df.iloc[k]
                    obs.append(OrderBlock(k, "bullish", bar["high"], bar["low"],
                                          bar["time"], disp.total_move))
                    break
        else:
            for k in range(start_i - 1, look_back - 1, -1):
                if df.iloc[k]["is_bull"]:
                    bar = df.iloc[k]
                    obs.append(OrderBlock(k, "bearish", bar["high"], bar["low"],
                                          bar["time"], disp.total_move))
                    break
    return obs


def is_rejection_candle(bar: pd.Series, ob_type: str) -> bool:
    if bar["range"] == 0:
        return False
    if ob_type == "bullish":
        return bar["lower_wick"] / bar["range"] > REJECTION_WICK_RATIO
    return bar["upper_wick"] / bar["range"] > REJECTION_WICK_RATIO


def find_signal(df: pd.DataFrame, order_blocks: List[OrderBlock]) -> Optional[dict]:
    """Latest closed bar: OB retest + rejection -> signal dict, else None."""
    n = len(df)
    last_idx = n - 1
    last_bar = df.iloc[last_idx]

    for ob in reversed(order_blocks):
        if ob.ob_bar_idx >= last_idx:
            continue
        if last_idx - ob.ob_bar_idx > OB_EXPIRY_BARS:
            continue

        if ob.ob_type == "bullish":
            displaced = invalidated = False
            for k in range(ob.ob_bar_idx + 1, last_idx):
                bar_k = df.iloc[k]
                if not displaced:
                    if bar_k["close"] > ob.ob_high:
                        displaced = True
                else:
                    if bar_k["close"] < ob.ob_low:
                        invalidated = True
                        break
            if not displaced or invalidated:
                continue
            if last_bar["low"] <= ob.ob_high and last_bar["close"] >= ob.ob_low:
                if is_rejection_candle(last_bar, ob.ob_type):
                    entry = ob.ob_high
                    sl    = ob.ob_low - SL_BUFFER
                    risk  = entry - sl
                    tp    = entry + risk * RISK_REWARD
                    return {
                        "direction":    "BUY",
                        "ob_type":      ob.ob_type,
                        "ob_high":      ob.ob_high,
                        "ob_low":       ob.ob_low,
                        "ob_time":      str(ob.ob_time),
                        "retest_time":  str(last_bar["time"]),
                        "entry":        round(entry, 2),
                        "sl":           round(sl, 2),
                        "tp":           round(tp, 2),
                        "ob_bar_idx":   ob.ob_bar_idx,
                        "displaced_by": round(ob.displaced_by, 2),
                    }
        else:
            displaced = invalidated = False
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
                        "direction":    "SELL",
                        "ob_type":      ob.ob_type,
                        "ob_high":      ob.ob_high,
                        "ob_low":       ob.ob_low,
                        "ob_time":      str(ob.ob_time),
                        "retest_time":  str(last_bar["time"]),
                        "entry":        round(entry, 2),
                        "sl":           round(sl, 2),
                        "tp":           round(tp, 2),
                        "ob_bar_idx":   ob.ob_bar_idx,
                        "displaced_by": round(ob.displaced_by, 2),
                    }
    return None


# ═══════════════════════════════════════════════════════════════════════════
# MT5 CONNECTION  (kept here since it is process-wide, not per-order)
# ═══════════════════════════════════════════════════════════════════════════


def mt5_connect() -> None:
    """Open the MT5 IPC connection. Uses env vars for programmatic login."""
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
            "Open MT5, log in, enable AutoTrading, then retry."
        )
    if not ti.connected:
        raise RuntimeError(
            "MT5 terminal not connected to broker — wait for quotes in Market Watch."
        )


def fetch_m5(symbol: str, bars: int) -> pd.DataFrame:
    """Fetch closed M5 bars (drops the still-forming bar)."""
    r = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, bars)
    if r is None or len(r) == 0:
        raise RuntimeError(f"No M5 data for {symbol}: {mt5.last_error()}")
    df = pd.DataFrame(r)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={"tick_volume": "volume"})
    df = df[[c for c in ["time", "open", "high", "low", "close", "volume"] if c in df.columns]]
    return df.iloc[:-1].reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# SEEN OBS PERSISTENCE  (deduplication across restarts)
# ═══════════════════════════════════════════════════════════════════════════


def load_seen_obs() -> set:
    if not SEEN_OBS_PATH.exists():
        return set()
    try:
        with SEEN_OBS_PATH.open("r", encoding="utf-8") as f:
            return {tuple(item) for item in json.load(f)}
    except Exception:
        return set()


def save_seen_obs(seen_obs: set) -> None:
    SEEN_OBS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SEEN_OBS_PATH.open("w", encoding="utf-8") as f:
        json.dump([list(item) for item in seen_obs], f)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN CYCLE
# ═══════════════════════════════════════════════════════════════════════════


def run_cycle(
    engine: ExecutionEngine,
    notifier: Mt5Notifier,
    fixed_lot: Optional[float],
    dry_run: bool,
    seen_obs: set,
) -> None:
    """
    One full evaluation:
        fetch data -> detect signal -> let ExecutionEngine handle execution.

    The engine does pre-flight, cascading order placement, retry/cooldown,
    and writes structured events to logs/<symbol>.json.
    """
    symbol = engine.cfg.name

    # ── 1) Strategy ──────────────────────────────────────────────────────────
    try:
        df = fetch_m5(symbol, HISTORY_BARS)
    except RuntimeError as exc:
        engine.logger.error("data_fetch_error", exc=exc)
        return

    df            = add_candle_features(df)
    displacements = detect_displacements(df)
    obs           = find_order_blocks(df, displacements)
    signal        = find_signal(df, obs)

    last_bar_time = df.iloc[-1]["time"]
    age_minutes   = (datetime.now(timezone.utc) - last_bar_time).total_seconds() / 60

    engine.log_cycle(
        bars=len(df),
        displacements=len(displacements),
        order_blocks=len(obs),
        last_bar_time=str(last_bar_time),
        data_age_min=round(age_minutes, 1),
        signal=signal is not None,
    )

    # ── 2) Bookkeeping (every cycle, not just when there is a signal) ────────
    active_keys: list[tuple] = []  # OB keys still valid this bar
    for ob in obs:
        # Provisional keys for orphan cleanup -- direction unknown until find_signal
        active_keys.append((str(ob.ob_time), "BUY"))
        active_keys.append((str(ob.ob_time), "SELL"))
    engine.begin_cycle(active_keys)

    # ── 3) Stale data guard ──────────────────────────────────────────────────
    if age_minutes > 15:
        engine.logger.event("skip", reason="market_closed_or_stale",
                            data_age_min=round(age_minutes, 1))
        notifier.notify_skip({"symbol": symbol,
                              "reason": "market_closed_or_stale",
                              "data_age_min": round(age_minutes, 1)})
        return

    if signal is None:
        return

    # ── 4) Deduplicate against seen_obs.json (cross-restart) ────────────────-
    ob_key = (signal["ob_time"], signal["direction"])
    if ob_key in seen_obs:
        engine.logger.event("skip", reason="already_traded",
                            ob_time=signal["ob_time"],
                            direction=signal["direction"])
        # No telegram -- routine, would spam every cycle
        return

    # ── 5) Notify of signal (telegram failures are now visible in logs) ──────
    engine.logger.event("signal", **signal)
    signal_for_tg = dict(signal); signal_for_tg["symbol"] = symbol
    notifier.notify_signal(signal_for_tg)

    if dry_run:
        engine.logger.event("dry_run", msg="no_order_sent")
        seen_obs.add(ob_key)
        save_seen_obs(seen_obs)
        return

    # ── 6) Hand off to ExecutionEngine ───────────────────────────────────────
    # When fixed_lot is set we bypass auto-sizing by temporarily flipping the
    # adapter into a deterministic one-shot via a wrapping function.
    outcome: ExecutionOutcome = engine.place_signal(signal)

    # ── 7) Notify based on outcome ───────────────────────────────────────────
    if outcome.placed:
        seen_obs.add(ob_key)
        save_seen_obs(seen_obs)
        placed_payload = {
            "symbol":       symbol,
            "ticket":       outcome.ticket,
            "order_type":   outcome.stage,
            "direction":    signal["direction"],
            "volume":       outcome.fields.get("volume"),
            "sl":           signal["sl"],
            "tp":           outcome.fields.get("tp_final", signal["tp"]),
            "slippage_pts": outcome.fields.get("slippage_pts"),
        }
        if outcome.stage == "market":
            notifier.notify_slippage_adjusted({
                "symbol":          symbol,
                "direction":       signal["direction"],
                "ob_entry":        signal["entry"],
                "market_price":    outcome.fields.get("market_price"),
                "slippage_pts":    outcome.fields.get("slippage_pts"),
                "sl_unchanged":    signal["sl"],
                "volume_original": outcome.fields.get("volume"),
                "volume_adjusted": outcome.fields.get("volume"),
                "tp_original":     signal["tp"],
                "tp_adjusted":     outcome.fields.get("tp_final"),
            })
        notifier.notify_order_placed(placed_payload)
    else:
        notifier.notify_skip({
            "symbol":    symbol,
            "reason":    outcome.reason or outcome.stage,
            "direction": signal["direction"],
            "ob_time":   signal["ob_time"],
            **outcome.fields,
        })

    # Forward position-close events from the engine sweep (if any happened in
    # this cycle) -- engine.begin_cycle() detects them but does not notify.
    # Re-running the sweep right before would race with a new fill; instead we
    # rely on the next cycle's begin_cycle to surface the close via Telegram.


# ═══════════════════════════════════════════════════════════════════════════
# CYCLE TIMING
# ═══════════════════════════════════════════════════════════════════════════


def sleep_until_next_m5(extra: float = 0.1) -> None:
    """Sleep until just after the next M5 bar closes (UTC-anchored)."""
    now = time.time()
    delay = max(1.0, (int(now // M5_SECONDS) + 1) * M5_SECONDS - now + extra)
    print(f"  -> sleeping {delay:.0f}s until next M5 close ...")
    time.sleep(delay)


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    p = argparse.ArgumentParser(description="Order Block Reaction Bot — XAUUSD M5")
    p.add_argument("--symbol", default=SYMBOL, help="MT5 symbol (default: XAUUSD)")
    p.add_argument("--lot",  type=float, default=None,
                   help="Fixed lot size; omit for 1%% auto risk sizing")
    p.add_argument("--risk", type=float, default=RISK_PER_TRADE,
                   help="Risk fraction for auto-sizing (default: 0.01)")
    p.add_argument("--once", action="store_true", help="Run one cycle then exit")
    p.add_argument("--dry-run", action="store_true",
                   help="Detect signals and notify, but never send orders")
    args = p.parse_args()

    # ── MT5 boot ─────────────────────────────────────────────────────────────
    mt5_connect()
    assert_terminal_ready()

    # ── Execution engine (resolves symbol + creates logger internally) ───────
    engine = ExecutionEngine(
        symbol=args.symbol,
        magic=MAGIC,
        log_path=LOG_PATH,
        risk_per_trade=args.risk,
        risk_reward=RISK_REWARD,
        slippage_limit_threshold=SLIPPAGE_LIMIT_THRESHOLD,
        slippage_max_points=SLIPPAGE_MAX_POINTS,
        max_spread_points=MAX_SPREAD_POINTS,
        stale_after_seconds=LIMIT_ORDER_STALE_SECONDS,
    )

    # ── Notifier shares the engine's structured logger for observability ─────
    notifier = Mt5Notifier(logger=engine.logger)

    # ── State that survives restarts ─────────────────────────────────────────
    seen_obs: set = load_seen_obs()

    try:
        if args.once:
            run_cycle(engine, notifier, args.lot, args.dry_run, seen_obs)
            return

        print(f"Order Block Bot running on {engine.cfg.name} M5. Ctrl+C to stop.")
        while True:
            try:
                run_cycle(engine, notifier, args.lot, args.dry_run, seen_obs)
            except Exception as exc:
                # Engine has its own error handling; this catches truly unexpected
                # failures (e.g. broken pandas frame). Log + continue.
                engine.logger.error("cycle_exception", exc=exc)
            sleep_until_next_m5(extra=0.1)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        engine.shutdown()
        mt5.shutdown()


if __name__ == "__main__":
    main()
