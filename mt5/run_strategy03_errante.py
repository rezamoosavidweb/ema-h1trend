#!/usr/bin/env python3
"""
Live / scheduled job: Strategy 03 crypto (notebook parity) on MetaTrader 5 — Errante.

Prerequisites:
  - Errante MT5 terminal installed and logged in (same Windows user as this script).
  - pip install MetaTrader5 pandas numpy
  - Symbol must match Market Watch name (e.g. ETHUSD).

Behavior matches notebooks/03_strategy03_crypto.ipynb pending-stop logic:
  H1 EMA 8/13/21 trend filter, M5 swing +/- offset, TP:SL = 1:1, optional pending expiry.

Environment (optional):
  MT5_LOGIN, MT5_PASSWORD, MT5_SERVER — only if you want the script to log in programmatically.
  MT5_TERMINAL_PATH — directory containing terminal64.exe if auto-detection fails.

Use 64-bit Python only (MetaTrader 5 matches 64-bit). If rates fail: open an M5 chart once,
add the symbol to Market Watch, then run  python run_strategy03_errante.py --find-symbol ETH
to see the broker's exact symbol names.

Summary: connect to MT5, load M5/H1, H1 trend from EMAs, pending signal on M5, send Buy/Sell Stop
with risk sizing — aligned with strategy03_crypto notebook.
"""

from __future__ import annotations

import argparse
import math
import os
import struct
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add this script's folder to sys.path so the sibling module strategy03_crypto_core imports cleanly.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import numpy as np

try:
    import MetaTrader5 as mt5
except ImportError:
    print("Install MetaTrader5: pip install MetaTrader5", file=sys.stderr)
    raise

import pandas as pd

from strategy03_crypto_core import (
    MIN_WARMUP_BARS_H1,
    MIN_WARMUP_BARS_M5,
    add_emas,
    compute_pending_setup,
    default_crypto_tick,
    last_closed_bar_index,
    merge_h1_trend_onto_m5,
    min_bars_needed_for_signal,
    rates_to_ohlcv_df,
)

# --- Strategy parameters (notebook-aligned) ---
LOOKBACK_BARS = 5  # M5 bars in the swing window (HH/LL); signal bar itself is outside the window.
PENDING_OFFSET_TICKS = 3.0  # How many ticks beyond swing boundary for pending entry price.
PENDING_EXPIRY_MIN = 60  # Minutes after signal bar time for pending order expiration.
RR = 1.0  # Reward:risk (TP distance vs SL distance).
RISK_PER_TRADE = 0.01  # Fraction of balance risked if SL hits (1%).
MAGIC_DEFAULT = 310039003  # Magic number to separate this bot's orders from others.
M5_SECONDS = 300  # One M5 candle length in seconds (align loop with bar close).
# Extra history margin so merge_asof / dropping the forming bar never falls below MIN_* warmup.
_FETCH_BUFFER = 50
HISTORY_BARS_M5 = max(800, MIN_WARMUP_BARS_M5 + LOOKBACK_BARS + _FETCH_BUFFER)
HISTORY_BARS_H1 = max(500, MIN_WARMUP_BARS_H1 + _FETCH_BUFFER)

# MQL5 order filling mode; some MetaTrader5 wheels omit SYMBOL_FILLING_* → use getattr fallbacks.
_FILL_IOC = getattr(mt5, "SYMBOL_FILLING_IOC", 2)
_FILL_FOK = getattr(mt5, "SYMBOL_FILLING_FOK", 1)


def pick_filling_mode(si) -> int:
    """Broker bitmask says which of IOC/FOK/Return is allowed; pick first compatible mode."""
    fm = int(si.filling_mode)
    if fm & _FILL_IOC:
        return mt5.ORDER_FILLING_IOC
    if fm & _FILL_FOK:
        return mt5.ORDER_FILLING_FOK
    return mt5.ORDER_FILLING_RETURN


def _parse_env_int(name: str) -> int | None:
    """Parse environment variable as int; empty or invalid → None."""
    v = os.environ.get(name)
    if v is None or not v.strip():
        return None
    try:
        return int(v.strip(), 10)
    except ValueError:
        return None


def round_price(price: float, digits: int) -> float:
    """Round price to symbol digits."""
    return float(np.round(price, digits))


def _tick_size(si) -> float:
    """Tradable price step: trade_tick_size, else point, else 10^-digits."""
    ts = float(getattr(si, "trade_tick_size", 0) or 0)
    if ts > 0:
        return ts
    pt = float(si.point or 0)
    if pt > 0:
        return pt
    return float(10 ** (-si.digits))


def snap_price_to_tick(price: float, si, *, mode: str) -> float:
    """Snap price to broker tick grid (nearest/up/down), then round to digits."""
    t = _tick_size(si)
    d = int(si.digits)
    x = price / t
    if mode == "up":
        snapped = math.ceil(x - 1e-12) * t
    elif mode == "down":
        snapped = math.floor(x + 1e-12) * t
    else:
        snapped = round(x) * t
    return round_price(snapped, d)


def prepare_pending_prices_for_market(
    symbol: str,
    side: str,
    entry: float,
    sl: float,
    tp: float,
    si,
) -> tuple[float, float, float] | None:
    """
    Align model prices with live Ask/Bid and min stop/freeze distance (stops_level, freeze_level),
    snap to ticks so order_send does not fail invalid price (e.g. 10015).
    Buy Stop entry must sit above ask+margin; Sell Stop below bid-margin.
    """
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"prepare_pending_prices: no quote for {symbol}; {mt5.last_error()}")
        return None

    # Margin in price units: minimum allowed distance of SL/TP from market (broker rules).
    point = float(si.point or 10 ** (-si.digits))
    stops_pts = int(getattr(si, "trade_stops_level", 0) or 0)
    freeze_pts = int(getattr(si, "trade_freeze_level", 0) or 0)
    margin_pts = max(stops_pts, freeze_pts)
    margin = margin_pts * point
    tick_sz = _tick_size(si)

    ask = float(tick.ask)
    bid = float(tick.bid)

    raw_entry, raw_sl, raw_tp = entry, sl, tp

    entry = snap_price_to_tick(entry, si, mode="nearest")
    sl = snap_price_to_tick(sl, si, mode="nearest")
    tp = snap_price_to_tick(tp, si, mode="nearest")

    initial = (entry, sl, tp)

    if side == "buy":
        # Buy Stop must be above current price + stop margin; SL below entry, TP above entry.
        floor_e = ask + margin + tick_sz
        if entry < floor_e:
            entry = snap_price_to_tick(floor_e, si, mode="up")

        if entry - sl < margin + tick_sz:
            sl = snap_price_to_tick(entry - margin - tick_sz, si, mode="down")
        if sl >= entry:
            sl = snap_price_to_tick(entry - tick_sz, si, mode="down")

        if tp - entry < margin + tick_sz:
            tp = snap_price_to_tick(entry + margin + tick_sz, si, mode="up")
        if tp <= entry:
            tp = snap_price_to_tick(entry + tick_sz, si, mode="up")

        if not (sl < entry < tp):
            print(
                f"Buy pending geometry invalid after clamp: ask={ask} bid={bid} "
                f"entry={entry} sl={sl} tp={tp} margin_pts={margin_pts}"
            )
            return None
    else:
        # Sell Stop must sit below bid with valid SL/TP spacing per broker rules.
        cap_e = bid - margin - tick_sz
        if entry > cap_e:
            entry = snap_price_to_tick(cap_e, si, mode="down")
        if entry >= bid:
            entry = snap_price_to_tick(bid - tick_sz, si, mode="down")

        if sl - entry < margin + tick_sz:
            sl = snap_price_to_tick(entry + margin + tick_sz, si, mode="up")
        if sl <= entry:
            sl = snap_price_to_tick(entry + tick_sz, si, mode="up")

        if entry - tp < margin + tick_sz:
            tp = snap_price_to_tick(entry - margin - tick_sz, si, mode="down")
        if tp >= entry:
            tp = snap_price_to_tick(entry - tick_sz, si, mode="down")

        if not (tp < entry < sl):
            print(
                f"Sell pending geometry invalid after clamp: ask={ask} bid={bid} "
                f"entry={entry} sl={sl} tp={tp} margin_pts={margin_pts}"
            )
            return None

    if (entry, sl, tp) != initial:
        print(
            f"Prices adjusted for broker rules (tick/stops): "
            f"entry {raw_entry:.5f}->{entry:.5f} sl {raw_sl:.5f}->{sl:.5f} tp {raw_tp:.5f}->{tp:.5f} "
            f"(ask={ask:.5f} bid={bid:.5f} stops_level_pts={stops_pts})"
        )

    return entry, sl, tp


def normalize_volume(vol: float, si) -> float:
    """Round volume to step and clamp to symbol min/max lot size."""
    step = si.volume_step
    vmin = si.volume_min
    vmax = si.volume_max
    if step <= 0:
        return max(vmin, min(vol, vmax))
    steps = np.floor(vol / step + 1e-12)
    v = steps * step
    v = max(vmin, min(vmax, v))
    return float(v)


def loss_per_lot_at_sl(symbol: str, side: str, entry: float, sl: float) -> float:
    """Loss for one full lot if SL is hit (convert dollar risk to lot size)."""
    otype = mt5.ORDER_TYPE_BUY if side == "buy" else mt5.ORDER_TYPE_SELL
    pnl = mt5.order_calc_profit(otype, symbol, 1.0, entry, sl)
    if pnl is None:
        raise RuntimeError(
            f"order_calc_profit failed: {mt5.last_error()} — check symbol & prices"
        )
    return abs(float(pnl))


def mt5_connect(login: int | None, password: str | None, server: str | None) -> None:
    """
    Initialize MetaTrader5 bridge; if login/password/server are all set, log in programmatically,
    otherwise use an already running logged-in terminal. MT5_TERMINAL_PATH points at terminal64.exe folder.
    """
    kwargs = {}
    if login is not None and password and server:
        kwargs["login"] = login
        kwargs["password"] = password
        kwargs["server"] = server
    path = os.environ.get("MT5_TERMINAL_PATH")
    if path:
        kwargs["path"] = path
    if not mt5.initialize(**kwargs):
        raise RuntimeError(f"mt5.initialize failed: {mt5.last_error()}")


def assert_terminal_ready() -> None:
    """Fail with actionable hints if the Python↔MT5 bridge is broken or disconnected."""
    ti = mt5.terminal_info()
    if ti is None:
        err = mt5.last_error()
        py_bits = struct.calcsize("P") * 8
        raise RuntimeError(
            "terminal_info() returned None — Python cannot talk to MT5.\n"
            f"  mt5.last_error()={err}\n"
            f"  This Python is {py_bits}-bit; MetaTrader 5 is 64-bit only — use 64-bit Python.\n"
            "  Start Errante MT5, log in, enable AutoTrading, then retry.\n"
            "  If you run multiple terminals, set MT5_TERMINAL_PATH to terminal64.exe's folder."
        )
    if not ti.connected:
        raise RuntimeError(
            "MT5 terminal reports disconnected — finish login and wait until quotes stream "
            "in Market Watch, then retry."
        )


def symbol_hints(query: str, *, limit: int = 30) -> list[str]:
    """Symbol names containing query substring (resolve broker suffixes like .i or .a)."""
    q = (query or "").strip().upper()
    if not q:
        return []
    found: list[str] = []
    for s in mt5.symbols_get() or []:
        name = s.name
        if q in name.upper():
            found.append(name)
    return sorted(set(found))[:limit]


def ensure_symbol(symbol: str) -> str:
    """
    Enable symbol in Market Watch and return broker canonical name.
    On wrong name, raise with similar symbol hints.
    """
    if not mt5.symbol_select(symbol, True):
        pass  # still try symbol_info; select can fail for invalid names
    si = mt5.symbol_info(symbol)
    if si is None:
        hints = symbol_hints(symbol.replace(".", "").replace("#", ""))
        htxt = ", ".join(hints[:12]) if hints else "(no substring matches — open Market Watch and add the instrument)"
        raise RuntimeError(
            f"Symbol {symbol!r} not found in MT5. Broker-specific names often differ "
            f"(e.g. ETHUSD.a, ETHUSD#).\n"
            f"  Matches containing similar text: {htxt}\n"
            f"  Pass the exact name with --symbol …"
        )
    # Refresh using canonical name from broker
    canonical = si.name
    mt5.symbol_select(canonical, True)
    return canonical


def _copy_rates(symbol: str, timeframe: int, count: int):
    """Try copy_rates_from_pos last count bars; if empty, fall back to copy_rates_range."""
    r = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if r is not None and len(r) > 0:
        return r
    utc_to = datetime.now(timezone.utc)
    # H1 needs a longer lookback to fit ~200 bars; M5 often suffices with 14 days.
    days_back = 120 if timeframe == mt5.TIMEFRAME_H1 else 14
    utc_from = utc_to - timedelta(days=days_back)
    r = mt5.copy_rates_range(symbol, timeframe, utc_from, utc_to)
    if r is not None and len(r) > 0:
        return r
    return None


def fetch_closed_frames(symbol: str) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    """
    Pull M5/H1 history, convert to OHLCV, drop the currently forming candle on each timeframe.
    Returns (canonical symbol name, closed M5 frame, closed H1 frame).
    """
    symbol = ensure_symbol(symbol)

    r_m5 = _copy_rates(symbol, mt5.TIMEFRAME_M5, HISTORY_BARS_M5)
    r_h1 = _copy_rates(symbol, mt5.TIMEFRAME_H1, HISTORY_BARS_H1)

    # After dropping the live candle we still need MIN_WARMUP_* closed bars — require one extra raw row.
    need_raw_m5 = MIN_WARMUP_BARS_M5 + 1
    need_raw_h1 = MIN_WARMUP_BARS_H1 + 1

    if r_m5 is None:
        tick = mt5.symbol_info_tick(symbol)
        err = mt5.last_error()
        py_bits = struct.calcsize("P") * 8
        raise RuntimeError(
            f"No M5 rates for {symbol!r}. mt5.last_error()={err}\n"
            f"  symbol_info_tick={'OK' if tick is not None else 'None'} — "
            "if quotes are missing, open the chart once or add the symbol to Market Watch.\n"
            f"  Python is {py_bits}-bit (MetaTrader5 requires 64-bit Python).\n"
            f"  Try: python run_strategy03_errante.py --find-symbol ETH\n"
        )
    if len(r_m5) < need_raw_m5:
        raise RuntimeError(
            f"Only {len(r_m5)} M5 bars from MT5; need >= {need_raw_m5} so that after dropping "
            f"the forming candle there are still {MIN_WARMUP_BARS_M5} closed bars for EMA/H1 merge warmup "
            f"(strategy03_crypto_core.MIN_WARMUP_BARS_M5). Sync history or open an M5 chart."
        )

    if r_h1 is None:
        raise RuntimeError(
            f"No H1 rates for {symbol!r}: {mt5.last_error()} — wait for history sync or open an H1 chart once."
        )
    if len(r_h1) < need_raw_h1:
        raise RuntimeError(
            f"Only {len(r_h1)} H1 bars from MT5; need >= {need_raw_h1} closed-frame bars "
            f"({MIN_WARMUP_BARS_H1}-bar warmup after dropping forming candle). Open an H1 chart to sync."
        )

    m5 = rates_to_ohlcv_df(r_m5)
    h1 = rates_to_ohlcv_df(r_h1)
    # Last row is the still-forming bar; signals use only fully closed candles.
    m5_closed = m5.iloc[:-1].copy()
    h1_closed = h1.iloc[:-1].copy()

    return symbol, m5_closed, h1_closed


def build_m5_context(symbol: str, pip_size: float) -> tuple[str, pd.DataFrame, int | None]:
    """
    Load raw data, compute EMAs, merge H1 trend onto M5, compute index of last signal-eligible bar.
    pip_size is intentionally unused here (offset defaults come from default_crypto_tick via --symbol).
    """
    sym, m5c, h1c = fetch_closed_frames(symbol)
    m5c = add_emas(m5c)
    h1c = add_emas(h1c)
    m5_ctx = merge_h1_trend_onto_m5(m5c, h1c)
    _ = pip_size
    i = last_closed_bar_index(m5_ctx, LOOKBACK_BARS)
    return sym, m5_ctx, i


def ours_orders_and_positions(symbol: str, magic: int):
    """Positions and pending orders on symbol whose magic matches this bot."""
    positions = mt5.positions_get(symbol=symbol) or []
    positions = [p for p in positions if p.magic == magic]
    orders = mt5.orders_get(symbol=symbol) or []
    orders = [o for o in orders if o.magic == magic]
    return positions, orders


def cancel_orders(orders, magic: int, symbol: str) -> None:
    """If a live quote exists, remove pending orders with matching magic."""
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return
    for o in orders:
        if o.magic != magic:
            continue
        req = {
            "action": mt5.TRADE_ACTION_REMOVE,
            "order": o.ticket,
            "symbol": symbol,
            "magic": magic,
        }
        r = mt5.order_send(req)
        if r is None:
            print(f"remove order {o.ticket} failed: {mt5.last_error()}")
        elif r.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"remove order {o.ticket} retcode={r.retcode} comment={r.comment}")


def send_pending(
    symbol: str,
    magic: int,
    side: str,
    entry: float,
    sl: float,
    tp: float,
    volume: float,
    expiry_utc: datetime,
) -> bool:
    """Submit Buy Stop or Sell Stop with expiry; filling mode matches symbol."""
    si = mt5.symbol_info(symbol)
    if si is None:
        raise RuntimeError(f"symbol_info failed for {symbol}")
    filling = pick_filling_mode(si)

    if side == "buy":
        otype = mt5.ORDER_TYPE_BUY_STOP
    else:
        otype = mt5.ORDER_TYPE_SELL_STOP

    exp_ts = int(expiry_utc.timestamp())

    req = {
        "action": mt5.TRADE_ACTION_PENDING,
        "symbol": symbol,
        "volume": volume,
        "type": otype,
        "price": entry,
        "sl": sl,
        "tp": tp,
        "magic": magic,
        "comment": "s03crypto",
        "type_time": mt5.ORDER_TIME_SPECIFIED,
        "expiration": exp_ts,
        "type_filling": filling,
        "deviation": int(os.environ.get("MT5_DEVIATION_POINTS", "50")),
    }

    order_check = getattr(mt5, "order_check", None)
    if callable(order_check):
        chk = order_check(req)
        if chk is not None and chk.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"order_check retcode={chk.retcode} comment={chk.comment}")

    r = mt5.order_send(req)
    if r is None:
        print(f"order_send failed: {mt5.last_error()}")
        return False
    if r.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"order_send retcode={r.retcode} comment={r.comment}")
        return False
    print(f"Placed pending ticket={r.order} side={side} vol={volume} entry={entry} sl={sl} tp={tp}")
    return True


def run_cycle(
    *,
    symbol: str,
    magic: int,
    pip_size: float,
    risk_per_trade: float,
    dry_run: bool,
    replace_pending: bool,
    verbose: bool = False,
) -> None:
    """
    One full strategy pass: balance, M5+H1 context, compute setup; if flat and allowed, place pending.
    Lot size from percent risk and per-lot loss at SL. dry_run prints only; replace_pending cancels prior pendings.
    """
    ai = mt5.account_info()
    if ai is None:
        raise RuntimeError(f"account_info failed: {mt5.last_error()}")
    balance = float(ai.balance)

    symbol, m5_ctx, i = build_m5_context(symbol, pip_size)
    if verbose:
        print(f"MT5 symbol: {symbol}", flush=True)
    if i is None:
        need = min_bars_needed_for_signal(LOOKBACK_BARS)
        print(
            f"Not enough M5 history for signal ({len(m5_ctx)} closed bars < {need}). "
            f"Warmup requires MIN_WARMUP_BARS_M5={MIN_WARMUP_BARS_M5} plus swing window; skipping."
        )
        return

    # compute_pending_setup returns qty from balance; live sizing recomputes volume via loss_per_lot_at_sl below.
    setup = compute_pending_setup(
        m5_ctx,
        bar_index=i,
        lookback_bars=LOOKBACK_BARS,
        pending_offset_ticks=PENDING_OFFSET_TICKS,
        pip_size=pip_size,
        rr=RR,
        balance=balance,
        risk_per_trade=risk_per_trade,
    )
    if setup is None:
        print(f"No setup (trend flat or data); last closed M5={m5_ctx.index[-1]}")
        return

    positions, orders = ours_orders_and_positions(symbol, magic)
    if positions:
        print(f"Position open ({len(positions)}); not placing pending.")
        return
    if orders:
        if not replace_pending:
            print(f"Pending already exists ({len(orders)}); not replacing (use --replace-pending).")
            return
        cancel_orders(orders, magic, symbol)
        time.sleep(0.5)

    si = mt5.symbol_info(symbol)
    if si is None:
        raise RuntimeError("symbol_info")
    if not si.visible:
        mt5.symbol_select(symbol, True)

    prepared = prepare_pending_prices_for_market(
        symbol, setup["side"], setup["entry"], setup["sl"], setup["tp"], si
    )
    if prepared is None:
        return
    ae, asl, atp = prepared

    try:
        loss_1lot = loss_per_lot_at_sl(symbol, setup["side"], ae, asl)
    except RuntimeError as e:
        print(e)
        return

    # Fixed dollar risk → lot size so loss at SL ≈ risk_cash.
    risk_cash = balance * risk_per_trade
    raw_vol = risk_cash / loss_1lot if loss_1lot > 1e-12 else 0.0
    volume = normalize_volume(raw_vol, si)
    if volume < si.volume_min - 1e-12:
        print(
            f"Volume {raw_vol} -> {volume} below broker minimum {si.volume_min}; "
            "increase balance/risk or reduce SL distance."
        )
        return

    # Expiry anchored to signal bar time + minutes (not "now").
    bar_time = m5_ctx.index[i]
    expiry = bar_time.to_pydatetime()
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=timezone.utc)
    else:
        expiry = expiry.astimezone(timezone.utc)
    expiry = expiry + timedelta(minutes=PENDING_EXPIRY_MIN)

    print(
        f"Signal bar={bar_time} trend={m5_ctx.iloc[i]['trend']} "
        f"side={setup['side']} entry={ae:.5f} sl={asl:.5f} tp={atp:.5f} "
        f"(model entry={setup['entry']:.5f}) vol={volume} dry_run={dry_run}"
    )

    if dry_run:
        return

    send_pending(symbol, magic, setup["side"], ae, asl, atp, volume, expiry)


def sleep_until_next_m5_close(extra_seconds: float = 2.0) -> None:
    """Sleep until after the next M5 bar close (+ small buffer) to stay aligned with closes."""
    now = time.time()
    n = int(now // M5_SECONDS)
    next_close = (n + 1) * M5_SECONDS
    delay = max(1.0, next_close - now + extra_seconds)
    time.sleep(delay)


def main() -> None:
    """Parse args, connect MT5, run once or loop each M5 close, always mt5.shutdown in finally."""
    p = argparse.ArgumentParser(description="Strategy03 crypto — Errante MT5 pending stops")
    p.add_argument("--symbol", default=os.environ.get("MT5_SYMBOL", "ETHUSD"))
    p.add_argument("--pip-size", type=float, default=None, help="Price increment for offset (notebook PIP_SIZE)")
    p.add_argument("--risk", type=float, default=RISK_PER_TRADE, help="Fraction of balance risked if SL hits")
    p.add_argument("--magic", type=int, default=_parse_env_int("MT5_MAGIC") or MAGIC_DEFAULT)
    p.add_argument("--once", action="store_true", help="Single evaluation then exit")
    p.add_argument("--dry-run", action="store_true", help="Print signal only; no orders")
    p.add_argument(
        "--replace-pending",
        action="store_true",
        help="Cancel existing magic-scoped pendings before placing (notebook refreshes when flat)",
    )
    p.add_argument(
        "--find-symbol",
        metavar="TEXT",
        default=None,
        help="List broker symbols whose name contains TEXT (for resolving ETHUSD vs ETHUSD.a etc.), then exit",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Log resolved broker symbol each cycle",
    )
    args = p.parse_args()

    # Without --pip-size, default tick step comes from symbol prefix (BTC/ETH/…).
    pip_size = args.pip_size if args.pip_size is not None else default_crypto_tick(args.symbol)

    login = _parse_env_int("MT5_LOGIN")
    password = os.environ.get("MT5_PASSWORD")
    server = os.environ.get("MT5_SERVER")

    mt5_connect(login, password, server)
    assert_terminal_ready()

    try:
        # Print matching symbols and exit — resolve exact broker names.
        if args.find_symbol:
            hints = symbol_hints(args.find_symbol)
            print(f"Symbols containing {args.find_symbol!r} ({len(hints)} shown, max 30):")
            for n in hints:
                print(f"  {n}")
            return

        if args.once:
            # Single pass — useful for tests or cron.
            run_cycle(
                symbol=args.symbol,
                magic=args.magic,
                pip_size=pip_size,
                risk_per_trade=args.risk,
                dry_run=args.dry_run,
                replace_pending=args.replace_pending,
                verbose=args.verbose,
            )
            return

        print(
            f"Looping on M5 closes: symbol={args.symbol} magic={args.magic} "
            f"pip_size={pip_size} Ctrl+C to stop."
        )
        # After each run_cycle, wait until next M5 close (sleep_until_next_m5_close).
        while True:
            run_cycle(
                symbol=args.symbol,
                magic=args.magic,
                pip_size=pip_size,
                risk_per_trade=args.risk,
                dry_run=args.dry_run,
                replace_pending=args.replace_pending,
                verbose=args.verbose,
            )
            sleep_until_next_m5_close()
    finally:
        # Release connection to MT5 terminal.
        mt5.shutdown()


if __name__ == "__main__":
    main()
