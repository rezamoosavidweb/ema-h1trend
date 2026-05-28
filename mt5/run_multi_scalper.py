#!/usr/bin/env python3
"""
Multi-Symbol Reaction Scalper — Live MT5 (Errante)

Live execution of the strategy from notebooks/24_multi_symbol_scalper.ipynb.

╔═══════════════════════════════════════════════════════════════════════════╗
║                          ARCHITECTURE OVERVIEW                            ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║   Per cycle (driven by M5 close):                                         ║
║       1. Health-check MT5 (watchdog)                                      ║
║       2. For each symbol in the golden basket:                            ║
║            a) Fetch M5 + H1 + D1 history (UTC-corrected)                  ║
║            b) Run Strategy.detect_signal() on last closed bar             ║
║            c) If signal, dedupe via per-symbol seen_signals.json          ║
║            d) Hand off to that symbol's ExecutionEngine (LIMIT→MARKET)    ║
║       3. Aggregate cycle summary line                                     ║
║       4. Sleep until next M5 close                                        ║
║                                                                           ║
║   One ExecutionEngine per symbol => independent:                          ║
║       * logs/<SYMBOL>.json                                                ║
║       * MAGIC (offset per symbol for ticket attribution)                  ║
║       * risk_per_trade (set by CapitalAllocator)                          ║
║       * pending/position state                                            ║
║                                                                           ║
║   Aggregate observability:                                                ║
║       logs/multi_symbol_scalper.json — portfolio-level events             ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝

Run:
    python mt5/run_multi_scalper.py                 # loop, all golden symbols
    python mt5/run_multi_scalper.py --once          # one cycle then exit
    python mt5/run_multi_scalper.py --dry-run       # signals only, no orders
    python mt5/run_multi_scalper.py --symbols GBPUSD XAUUSD   # explicit basket
    python mt5/run_multi_scalper.py --policy score  # score-weighted sizing

Env vars (optional, forwarded to mt5.initialize):
    MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_TERMINAL_PATH
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    print("Install MetaTrader5: pip install MetaTrader5", file=sys.stderr)
    raise

# Make sibling packages importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from execution import ExecutionEngine, ExecutionOutcome, Mt5Watchdog  # noqa: E402
from telegram_bot.mt5_notifier import Mt5Notifier                     # noqa: E402

from mt5.multi_symbol_bot import (                                    # noqa: E402
    Strategy, StrategyConfig, Signal,
    SymbolBasket, SymbolStrategyConfig, load_basket,
    CapitalAllocator, SymbolAllocation,
)
from mt5.multi_symbol_bot.strategy import (                           # noqa: E402
    HISTORY_M5_BARS, HISTORY_H1_BARS, HISTORY_D1_BARS,
    DEFAULT_BROKER_TO_NY_H, RR,
)
# OBSERVABILITY (additive — does not affect trading logic)
from mt5.multi_symbol_bot.observability import (                      # noqa: E402
    make_run_id,
    compute_portfolio_config_hash, compute_symbol_config_hash,
    current_git_commit,
    htf_policy_snapshot, htf_source_info, bar_integrity_snapshot,
    decision_trace, cascade_id as _cascade_id_for,
)


# ═══════════════════════════════════════════════════════════════════════════
# PATHS / RUNTIME CONFIG
# ═══════════════════════════════════════════════════════════════════════════

_REPO_ROOT     = Path(__file__).resolve().parent.parent
LOGS_DIR       = _REPO_ROOT / "logs"
SEEN_DIR       = _REPO_ROOT / "logs" / "seen_signals_multi"
RESULTS_DIR    = _REPO_ROOT / "notebooks" / "results" / "multi_symbol_scalper"
PORTFOLIO_LOG  = LOGS_DIR / "multi_symbol_scalper.json"

# Magic-number base. Each symbol gets MAGIC_BASE + offset (0…N-1) so positions
# from different symbols can be told apart in MT5 history.
MAGIC_BASE     = 24_000_000

# Execution thresholds.
#
# `slippage_max_points` is the HARD cap on price-drift between the strategy's
# desired entry (the close of the signal bar) and the live market price at
# order-send time. Points are the smallest broker tick — meaning is symbol-
# specific:
#
#   FX 5-digit majors/crosses  (e.g. 1.86331)  1 point = 0.00001  → 50 pts = 5 pips
#   FX 3-digit JPY pairs       (e.g. 184.91 )  1 point = 0.001    → 50 pts = 5 pips
#   XAU                        (e.g. 4525.81)  1 point = 0.01     → 30 pts = $0.30
#   USDMXN                     (e.g.   17.29)  1 point = 0.0001   → 200 pts = ~$0.02 quote
#
# The defaults below were derived from R&D on actual live rejections seen on
# Errante demo (2026-05-25):
#   GBPCAD: live drift 19..35 pts  → cap 50  (35 + 40% safety)
#   EURCAD: live drift 15..21 pts  → cap 35  (21 + 65% safety)
#   AUDCAD: live drift 17..31 pts  → cap 50  (31 + 60% safety)
#   GBPUSD/EURJPY/XAUUSD/USDMXN had no live signals yet — sized from typical
#   bar-close volatility on M5 data.
#
# Tune any value individually after live observation. Symbols not in the dict
# fall back to `SLIPPAGE_MAX_POINTS_DEFAULT`.
SLIPPAGE_LIMIT_THRESHOLD = 4.0
SLIPPAGE_MAX_POINTS_BY_SYMBOL: dict[str, float] = {
    'GBPUSD':  40.0,   # FX 5-digit major
    'XAUUSD':  30.0,   # metal, point=0.01 → 30 pts = $0.30 drift
    'GBPCAD':  50.0,   # live max 35 + 40% safety
    'USDMXN': 200.0,   # exotic, low liquidity, wider drift typical
    'EURJPY':  40.0,   # JPY 3-digit
    'EURCAD':  35.0,   # live max 21 + 65% safety
    'AUDCAD':  50.0,   # live max 31 + 60% safety
}
SLIPPAGE_MAX_POINTS_DEFAULT = 40.0   # safe fallback for any new symbol


def slippage_max_for(symbol: str) -> float:
    """Return the symbol-specific slippage cap, or the default if absent."""
    return SLIPPAGE_MAX_POINTS_BY_SYMBOL.get(symbol, SLIPPAGE_MAX_POINTS_DEFAULT)


LIMIT_ORDER_STALE_SECONDS = 5 * 60        # 1 M5 bar
MAX_SPREAD_POINTS         = 200
DEFAULT_PORTFOLIO_RISK    = 0.02          # 2% portfolio at risk if all 1× open

M5_SECONDS               = 300

# ═══════════════════════════════════════════════════════════════════════════
# HTF POLICY  (observability — see notebooks/HTF_POLICY_REPORT.md)
# ═══════════════════════════════════════════════════════════════════════════
#
# Current effective policy is "B" (always-synth both). When the recommended
# C_15 policy ships, flip USE_SYNTH_D1 to False and set H1_FRESHNESS_THRESHOLD_MIN
# to 15 (or whatever threshold the report concludes after stride-1 re-run).
# These constants exist so the live behaviour is hashable + observable, NOT
# so trading logic can be changed via flags. Changing them is a deploy.
USE_SYNTH_H1                  = True
USE_SYNTH_D1                  = True
H1_FRESHNESS_THRESHOLD_MIN    = 0.0
D1_FRESHNESS_THRESHOLD_MIN    = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# TIME HANDLING — KEEP BROKER WALL-CLOCK AS-IS, NO TZ CONVERSION
# ═══════════════════════════════════════════════════════════════════════════
#
# Convention (locked across live + backtest):
#   * MT5 returns `time` as wall-clock-encoded-as-Unix-seconds. We do NOT
#     convert to real UTC — we keep the wall-clock reading verbatim, as a
#     tz-naive timestamp.
#   * Backtest CSVs follow the same convention: time labels are wall-clock
#     even when written with a "+00:00" suffix.
#   * Strategy session windows assume bar `time.hour` is broker wall-clock.
#     With broker_to_ny_h=7 (Errante = EET/EEST), broker 15:00 → NY 08:00.
#   * For age-of-data freshness checks we use `_broker_now()` which returns
#     the current wall-clock time in the same convention.
#
# Why no conversion?
#   Earlier code did `tz_localize(Asia/Nicosia).tz_convert(UTC)` here AND
#   a different conversion in the backtest notebooks → frames drifted out
#   of alignment whenever the CSV cache and the live stream disagreed on
#   bar boundaries. Removing both conversions removes the only source of
#   bar-time disagreement: now BT and live receive identical timestamps.
# ═══════════════════════════════════════════════════════════════════════════

# Offset (hours) from broker wall-clock to real UTC. Errante = EET/EEST so
# this is +2 in winter and +3 in summer. ONLY used by `_broker_now()` so the
# stale-data guard compares wall-clock against wall-clock.
# In May 2026 we are in EEST → +3.
BROKER_WALLCLOCK_OFFSET_HOURS = 3


def _mt5_seconds_to_naive(seconds):
    """MT5 wall-clock-as-unix-seconds → tz-naive Timestamp (no shift)."""
    return pd.to_datetime(seconds, unit="s")


def _broker_now() -> pd.Timestamp:
    """Current real-world UTC time expressed in BROKER wall-clock (tz-naive)."""
    return pd.Timestamp.utcnow().tz_localize(None) + pd.Timedelta(hours=BROKER_WALLCLOCK_OFFSET_HOURS)


# ═══════════════════════════════════════════════════════════════════════════
# MT5 CONNECTION
# ═══════════════════════════════════════════════════════════════════════════

def mt5_connect() -> None:
    """Open the MT5 IPC connection using env vars when present."""
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


# ═══════════════════════════════════════════════════════════════════════════
# DATA FETCH  (per-symbol M5 / H1 / D1)
# ═══════════════════════════════════════════════════════════════════════════

def _fetch_bars(symbol: str, timeframe, n: int) -> pd.DataFrame:
    """Fetch `n` bars; drop the still-forming bar via iloc[:-1].

    Returns a DataFrame whose `time` column is tz-naive BROKER wall-clock.
    """
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No data for {symbol} timeframe={timeframe}: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    df["time"] = _mt5_seconds_to_naive(df["time"])
    df = df.rename(columns={"tick_volume": "volume"})
    cols = [c for c in ["time", "open", "high", "low", "close", "volume"] if c in df.columns]
    return df[cols].iloc[:-1].reset_index(drop=True)


# ─── M5 top-up for stale H1 / D1 ─────────────────────────────────────────────
# Errante demo lags H1/D1 publication by hours-to-days behind M5. When that
# happens, the strategy sees stale HTF context (old EMA50/RSI14) and its
# trend / RSI gates can flip vs what a fully up-to-date feed would say.
# We mirror the fix from notebooks/00_data_feching.ipynb: if M5 has rolled
# past the latest broker H1/D1 bar, synthesise the missing bucket(s) from
# the freshest M5 cache and append them. Native broker history is untouched;
# only the latest forming bucket(s) are synthetic.

_TF_RESAMPLE_RULE = {"H1": "1h", "D1": "1D"}
_TF_BAR_DURATION  = {"H1": pd.Timedelta(hours=1), "D1": pd.Timedelta(days=1)}


def topup_htf_from_m5(htf_df: pd.DataFrame, tf: str, m5_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Return (possibly-extended HTF df, number of synthetic bars appended)."""
    if tf not in _TF_RESAMPLE_RULE or htf_df.empty or m5_df.empty:
        return htf_df, 0

    rule          = _TF_RESAMPLE_RULE[tf]
    bar_duration  = _TF_BAR_DURATION[tf]
    last_htf_time = pd.Timestamp(htf_df["time"].iloc[-1])
    last_m5_time  = pd.Timestamp(m5_df["time"].iloc[-1])

    if last_m5_time < last_htf_time + bar_duration:
        return htf_df, 0     # HTF already covers everything M5 can see

    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in m5_df.columns:
        agg["volume"] = "sum"

    resampled = (
        m5_df.set_index("time")
             .resample(rule, label="left", closed="left")
             .agg(agg)
             .dropna(subset=["open", "high", "low", "close"])
             .reset_index()
    )

    new_bars = resampled[resampled["time"] > last_htf_time].copy()
    if new_bars.empty:
        return htf_df, 0

    # Align column order with the broker frame (fill any missing columns).
    for col in htf_df.columns:
        if col not in new_bars.columns:
            new_bars[col] = 0
    new_bars = new_bars[htf_df.columns]

    return pd.concat([htf_df, new_bars], ignore_index=True), int(len(new_bars))


def fetch_strategy_frames(
    broker_symbol: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int, int, dict]:
    """Fetch all three frames, then top-up H1/D1 from M5 if broker lags.

    Returns ``(m5, h1, d1, h1_topped_up_n, d1_topped_up_n, htf_meta)``.
    `htf_meta` carries the broker-published last bar times CAPTURED
    BEFORE the synth top-up — needed by the observability layer to log
    h1/d1 source provenance and broker freshness per cycle.
    """
    m5 = _fetch_bars(broker_symbol, mt5.TIMEFRAME_M5, HISTORY_M5_BARS)
    h1 = _fetch_bars(broker_symbol, mt5.TIMEFRAME_H1, HISTORY_H1_BARS)
    d1 = _fetch_bars(broker_symbol, mt5.TIMEFRAME_D1, HISTORY_D1_BARS)

    # Capture broker truth BEFORE any synth — observability requires it.
    htf_meta = {
        "h1_last_broker_time": h1["time"].iloc[-1] if not h1.empty else None,
        "d1_last_broker_time": d1["time"].iloc[-1] if not d1.empty else None,
        "h1_last_broker_bars_n": int(len(h1)),
        "d1_last_broker_bars_n": int(len(d1)),
    }

    h1, h1_topped = topup_htf_from_m5(h1, "H1", m5)
    d1, d1_topped = topup_htf_from_m5(d1, "D1", m5)
    return m5, h1, d1, h1_topped, d1_topped, htf_meta


# ═══════════════════════════════════════════════════════════════════════════
# SEEN-SIGNAL PERSISTENCE  (per symbol; dedup across restarts)
# ═══════════════════════════════════════════════════════════════════════════

def _seen_path(symbol: str) -> Path:
    return SEEN_DIR / f"{symbol}.json"


def load_seen_signals(symbol: str) -> set:
    path = _seen_path(symbol)
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {tuple(item) for item in data.get("entries", [])}
    except Exception:
        return set()


def save_seen_signals(symbol: str, seen: set) -> None:
    SEEN_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "utc-v1", "entries": [list(t) for t in seen]}
    _seen_path(symbol).write_text(json.dumps(payload), encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════
# PER-SYMBOL CONTEXT  (everything one symbol needs in the cycle)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SymbolContext:
    """
    One per active symbol. Bundles the strategy, engine, notifier, and
    persistence layer so the cycle loop has a single uniform interface.
    """

    symbol:          str                     # canonical (e.g. "XAUUSD")
    strategy:        Strategy
    engine:          ExecutionEngine
    notifier:        Mt5Notifier
    allocation:      SymbolAllocation
    seen_signals:    set = field(default_factory=set)
    # OBSERVABILITY (additive — not used by any trading code path):
    run_id:                str = ""
    portfolio_config_hash: str = ""
    symbol_config_hash:    str = ""
    htf_policy:            dict = field(default_factory=dict)

    @property
    def broker_symbol(self) -> str:
        """Broker-resolved symbol (e.g. 'XAUUSD.i')."""
        return self.engine.cfg.name


# ═══════════════════════════════════════════════════════════════════════════
# CYCLE  (single evaluation of one symbol)
# ═══════════════════════════════════════════════════════════════════════════

def run_symbol_cycle(ctx: SymbolContext, dry_run: bool) -> dict:
    """
    One full evaluation for one symbol:
        fetch -> detect -> dedupe -> handoff to ExecutionEngine.

    Returns a per-symbol summary dict for the aggregate cycle log.
    Never raises — exceptions are logged and absorbed so one symbol's failure
    doesn't take down the whole portfolio loop.
    """
    sym  = ctx.symbol
    eng  = ctx.engine
    log  = eng.logger

    # ── 0) Bookkeeping that must run every cycle (orphans, stale pendings) ──
    eng.begin_cycle(active_ob_keys=[])    # we set valid keys after detection
    eng.heartbeat_if_due()

    # ── 0b) Forward any newly-closed positions to Telegram ──────────────────
    # `begin_cycle` runs the sweep; we drain the buffer so the operator gets
    # one telegram per closed trade (entry, exit, P&L, balance).
    try:
        for ev in eng.consume_close_events():
            try:
                ctx.notifier.notify_position_closed(
                    ticket      = ev.get("ticket"),
                    profit      = float(ev.get("profit", 0.0)),
                    balance     = float(ev.get("balance", 0.0)),
                    equity      = float(ev.get("equity", 0.0)),
                    symbol      = sym,
                    side        = ev.get("side"),
                    volume      = ev.get("volume"),
                    entry_price = ev.get("entry_price"),
                    exit_price  = ev.get("exit_price"),
                    opened_at   = ev.get("opened_at"),
                    closed_at   = ev.get("closed_at"),
                )
            except Exception as exc:
                log.error("notify_close_failed", exc=exc, ticket=ev.get("ticket"))
    except Exception as exc:
        log.error("consume_close_events_failed", exc=exc)

    if not eng.is_mt5_healthy():
        log.event("cycle_skipped", reason="mt5_unhealthy")
        return {"symbol": sym, "skipped": "mt5_unhealthy"}

    # ── 1) Strategy ─────────────────────────────────────────────────────────-
    try:
        m5, h1, d1, h1_topped, d1_topped, htf_meta = fetch_strategy_frames(ctx.broker_symbol)
    except RuntimeError as exc:
        log.error("data_fetch_error", exc=exc)
        # Telegram alert — data outage is operator-actionable
        try:
            ctx.notifier.notify_error(sym, "data_fetch_error", str(exc))
        except Exception:
            pass
        return {"symbol": sym, "skipped": "data_fetch_error", "error": str(exc)}

    # detect_signal_verbose returns (Signal|None, diagnostics_dict). The
    # diagnostics dump every gate value + OHLC of the last bar — invaluable
    # for backtest/live parity audits in notebook 30/31. Falls back to the
    # legacy `detect_signal` API if the strategy version doesn't expose it
    # (so this runner stays compatible with older strategy.py builds).
    diag: dict = {}
    try:
        if hasattr(ctx.strategy, "detect_signal_verbose"):
            signal, diag = ctx.strategy.detect_signal_verbose(m5, h1, d1)
        else:
            signal = ctx.strategy.detect_signal(m5, h1, d1)
    except Exception as exc:
        log.error("strategy_exception", exc=exc)
        try:
            ctx.notifier.notify_error(sym, "strategy_exception", str(exc))
        except Exception:
            pass
        return {"symbol": sym, "skipped": "strategy_exception", "error": str(exc)}

    # Record any synthetic bars we appended this cycle so post-hoc audits can
    # see when the broker lagged HTF publication.
    if diag is not None and (h1_topped or d1_topped):
        diag["h1_topped_from_m5"] = int(h1_topped)
        diag["d1_topped_from_m5"] = int(d1_topped)

    last_bar_time = m5.iloc[-1]["time"]
    now_broker    = _broker_now()
    age_minutes   = (now_broker - pd.Timestamp(last_bar_time)).total_seconds() / 60

    # ── OBSERVABILITY (additive — see notebooks/observability section) ──────-
    # All derived from data we already have. No new strategy logic.
    try:
        htf_src = htf_source_info(
            h1_topped_from_m5=int(h1_topped),
            d1_topped_from_m5=int(d1_topped),
            h1_last_broker_time=htf_meta.get("h1_last_broker_time"),
            d1_last_broker_time=htf_meta.get("d1_last_broker_time"),
            now_broker_ts=now_broker,
        )
    except Exception:
        htf_src = {}
    try:
        bar_int = bar_integrity_snapshot(m5_len=len(m5), csv_source="live")
    except Exception:
        bar_int = {}
    try:
        dec_trace = decision_trace(diag or {})
    except Exception:
        dec_trace = {}
    # cascade_id: when a position is open for THIS bot's magic, every
    # subsequent signal-bearing cycle shares the same cascade_id with the
    # original signal that opened the position. Lets us reconstruct chains.
    casc_id = None
    try:
        open_pos = next(
            (p for p in (mt5.positions_get(symbol=ctx.broker_symbol) or [])
             if p.magic == ctx.engine.factory.magic),
            None,
        )
        if open_pos is not None:
            open_since_iso = datetime.fromtimestamp(int(open_pos.time),
                                                     tz=timezone.utc).isoformat()
            casc_id = _cascade_id_for(
                symbol=sym, open_ticket=int(open_pos.ticket),
                open_since_iso=open_since_iso,
            )
    except Exception:
        casc_id = None

    log.event(
        "cycle",
        bars_m5=len(m5), bars_h1=len(h1), bars_d1=len(d1),
        last_bar_time=str(last_bar_time),
        data_age_min=round(age_minutes, 1),
        signal=signal is not None,
        risk_per_trade=ctx.allocation.risk_per_trade,
        weight=ctx.allocation.weight,
        diag=diag,
        # ── NEW OBSERVABILITY FIELDS ─────────────────────────────────────-
        run_id=ctx.run_id,
        config_hash=ctx.portfolio_config_hash,
        symbol_config_hash=ctx.symbol_config_hash,
        htf_policy=ctx.htf_policy,
        htf_source=htf_src,
        bar_integrity=bar_int,
        decision_trace=dec_trace,
        cascade_id=casc_id,
    )

    # ── 2) Stale-data guard ─────────────────────────────────────────────────-
    if age_minutes > 15:
        log.event("skip", reason="market_closed_or_stale",
                  data_age_min=round(age_minutes, 1))
        return {"symbol": sym, "skipped": "stale", "age_min": round(age_minutes, 1)}

    if signal is None:
        return {"symbol": sym, "signal": False}

    # ── 3) Dedupe against persistent seen-signals ───────────────────────────-
    sig_key = (signal.bar_time, signal.direction)
    if sig_key in ctx.seen_signals:
        log.event("skip", reason="already_traded",
                  bar_time=signal.bar_time, direction=signal.direction)
        return {"symbol": sym, "skipped": "already_traded"}

    # ── 4) Telegram + structured log of the fresh signal ────────────────────-
    # Always log to disk. Telegram is suppressed when a position with this
    # bot's magic is already open on this symbol — otherwise telegram fills
    # up with repeated signals on consecutive bars while one trade is live
    # (they are all going to be skipped with `position_open` anyway).
    log.event("signal",
              direction=signal.direction, entry=signal.entry,
              sl=signal.sl, tp=signal.tp,
              bar_time=signal.bar_time, **signal.confidence)
    has_open_position = any(
        p.magic == ctx.engine.factory.magic
        for p in (mt5.positions_get(symbol=ctx.broker_symbol) or [])
    )
    if has_open_position:
        log.event("signal_telegram_suppressed",
                  reason="position_open_for_symbol",
                  bar_time=signal.bar_time, direction=signal.direction)
    else:
        tg_payload = {**signal.as_engine_dict(), "symbol": sym}
        try:
            ctx.notifier.notify_signal(tg_payload)
        except Exception as exc:
            log.error("notify_signal_failed", exc=exc)

    if dry_run:
        log.event("dry_run", msg="no_order_sent",
                  direction=signal.direction, entry=signal.entry)
        ctx.seen_signals.add(sig_key)
        save_seen_signals(sym, ctx.seen_signals)
        return {"symbol": sym, "signal": True, "stage": "dry_run"}

    # ── 5) Hand to ExecutionEngine ──────────────────────────────────────────-
    try:
        outcome: ExecutionOutcome = eng.place_signal(signal.as_engine_dict())
    except Exception as exc:
        log.error("execution_exception", exc=exc, bar_time=signal.bar_time)
        try:
            ctx.notifier.notify_error(
                sym, "execution_exception", str(exc),
                extra={"direction": signal.direction, "bar_time": signal.bar_time},
            )
        except Exception:
            pass
        return {"symbol": sym, "signal": True, "stage": "exception", "error": str(exc)}

    # ── 6) Notify outcome ───────────────────────────────────────────────────-
    if outcome.placed:
        ctx.seen_signals.add(sig_key)
        save_seen_signals(sym, ctx.seen_signals)
        # `market_price` is the ACTUAL fill price (only set on market path);
        # for limit fills, the broker fills at the requested price, so fall
        # back to signal.entry which equals the limit price.
        fill_price = outcome.fields.get("market_price", signal.entry)
        try:
            ctx.notifier.notify_order_placed({
                "symbol":       sym,
                "ticket":       outcome.ticket,
                "order_type":   outcome.stage,
                "direction":    signal.direction,
                "volume":       outcome.fields.get("volume"),
                "fill_price":   fill_price,
                "signal_entry": signal.entry,
                "sl":           signal.sl,
                "tp":           outcome.fields.get("tp_final", signal.tp),
                "slippage_pts": outcome.fields.get("slippage_pts"),
            })
        except Exception as exc:
            log.error("notify_placed_failed", exc=exc)
    else:
        try:
            ctx.notifier.notify_skip({
                "symbol":    sym,
                "reason":    outcome.reason or outcome.stage,
                "direction": signal.direction,
                "bar_time":  signal.bar_time,
                **outcome.fields,
            })
        except Exception as exc:
            log.error("notify_skip_failed", exc=exc)

    return {
        "symbol":  sym,
        "signal":  True,
        "stage":   outcome.stage,
        "placed":  outcome.placed,
        "ticket":  outcome.ticket,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PORTFOLIO LOGGER
# ═══════════════════════════════════════════════════════════════════════════

def write_portfolio_event(event: str, **fields) -> None:
    """Single-line JSON to the portfolio aggregate log."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts":    datetime.now(timezone.utc).isoformat(),
        "event": event,
        **fields,
    }
    with PORTFOLIO_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    print(f"[portfolio] {event}: " +
          " ".join(f"{k}={v}" for k, v in fields.items() if k != "exc"))


# ═══════════════════════════════════════════════════════════════════════════
# BUILD PER-SYMBOL CONTEXTS
# ═══════════════════════════════════════════════════════════════════════════

def build_contexts(
    basket: SymbolBasket,
    allocations: list[SymbolAllocation],
    risk_reward: float,
    *,
    run_id: str = "",
    portfolio_config_hash: str = "",
    htf_policy: dict | None = None,
) -> list[SymbolContext]:
    """
    For each basket member: create Strategy, ExecutionEngine, Notifier, seen-set.

    Each ExecutionEngine writes to its own `logs/<SYMBOL>.json` with a unique
    magic so multi-bot ticket attribution stays clean.

    The OBSERVABILITY kwargs (run_id, portfolio_config_hash, htf_policy) are
    threaded into every per-symbol log file via the StructuredLogger's
    default_fields, so each event in logs/<SYMBOL>.json automatically
    carries run identity. They're also stashed on SymbolContext so the
    cycle handler can include them in cycle event diagnostics.
    """
    contexts: list[SymbolContext] = []
    alloc_by_sym = {a.symbol: a for a in allocations}
    htf_policy   = htf_policy or {}

    for idx, member in enumerate(basket):
        sym = member.symbol
        alloc = alloc_by_sym.get(sym)
        if alloc is None:
            # Should never happen — allocator covers every basket member.
            write_portfolio_event("alloc_missing", symbol=sym)
            continue

        log_path = LOGS_DIR / f"{sym}.json"
        magic    = MAGIC_BASE + idx

        # Per-symbol slippage cap — driven by the R&D table near the top of
        # this file. Errante demo R&D (2026-05-25) showed bar-close drift can
        # range from a few pts (XAU) to 35+ pts (FX crosses on news), so a
        # single global value forces a bad trade-off. See SLIPPAGE_MAX_POINTS_BY_SYMBOL.
        slip_max = slippage_max_for(sym)

        # OBSERVABILITY: compute per-symbol config hash and thread run_id +
        # both hashes through to the StructuredLogger so every event in
        # logs/<SYMBOL>.json carries the run identity (cheap; computed once
        # per symbol per process; no per-cycle overhead).
        sym_cfg_hash = compute_symbol_config_hash(
            strategy_cfg        = member.cfg,
            slippage_max_points = slip_max,
            risk_per_trade      = alloc.risk_per_trade,
            risk_reward         = risk_reward,
        )
        logger_defaults = {
            "run_id":             run_id,
            "config_hash":        portfolio_config_hash,
            "symbol_config_hash": sym_cfg_hash,
        }

        engine = ExecutionEngine(
            symbol                    = sym,
            magic                     = magic,
            log_path                  = log_path,
            risk_per_trade            = alloc.risk_per_trade,
            risk_reward               = risk_reward,
            slippage_limit_threshold  = SLIPPAGE_LIMIT_THRESHOLD,
            slippage_max_points       = slip_max,
            max_spread_points         = MAX_SPREAD_POINTS,
            stale_after_seconds       = LIMIT_ORDER_STALE_SECONDS,
            comment_prefix            = "multi_scalp",
            rotate_daily_logs         = True,
            logger_default_fields     = logger_defaults,
        )
        engine.logger.event(
            "slippage_config",
            slippage_max_points=slip_max,
            slippage_limit_threshold=SLIPPAGE_LIMIT_THRESHOLD,
            from_table=sym in SLIPPAGE_MAX_POINTS_BY_SYMBOL,
        )

        # Recover open positions / pendings (idempotent — survives bot restarts).
        engine.initialize_state_from_broker()

        strategy = Strategy(cfg=member.cfg, broker_to_ny_h=DEFAULT_BROKER_TO_NY_H)
        notifier = Mt5Notifier(logger=engine.logger)
        seen     = load_seen_signals(sym)

        # Announce start to telegram so the operator can confirm the bot is
        # actually running (and on the right magic / symbol).
        try:
            notifier.notify_bot_lifecycle(
                symbol=sym, phase="start",
                magic=magic,
                extras={
                    "risk_per_trade":  f"{alloc.risk_per_trade:.4%}",
                    "slippage_max_pts": slip_max,
                    "broker_symbol":   engine.cfg.name,
                },
            )
        except Exception:
            pass

        contexts.append(SymbolContext(
            symbol                 = sym,
            strategy               = strategy,
            engine                 = engine,
            notifier               = notifier,
            allocation             = alloc,
            seen_signals           = seen,
            # observability (additive)
            run_id                 = run_id,
            portfolio_config_hash  = portfolio_config_hash,
            symbol_config_hash     = sym_cfg_hash,
            htf_policy             = htf_policy,
        ))

    return contexts


# ═══════════════════════════════════════════════════════════════════════════
# CYCLE TIMING
# ═══════════════════════════════════════════════════════════════════════════

def sleep_until_next_m5(extra: float = 0.5) -> None:
    """Sleep until just after the next M5 bar closes (UTC-anchored)."""
    now = time.time()
    delay = max(1.0, (int(now // M5_SECONDS) + 1) * M5_SECONDS - now + extra)
    print(f"  → sleeping {delay:.0f}s until next M5 close ...")
    time.sleep(delay)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN CYCLE  (across all symbols)
# ═══════════════════════════════════════════════════════════════════════════

def run_portfolio_cycle(contexts: list[SymbolContext], dry_run: bool) -> None:
    """One pass through every symbol; collects + logs an aggregate summary."""
    started = time.monotonic()
    results: list[dict] = []
    for ctx in contexts:
        try:
            results.append(run_symbol_cycle(ctx, dry_run=dry_run))
        except Exception as exc:
            # Belt-and-suspenders — run_symbol_cycle already catches its own
            # exceptions. This is here so an issue with a single symbol never
            # blocks the rest of the portfolio from running this cycle.
            ctx.engine.logger.error("uncaught_cycle_exception", exc=exc)
            results.append({"symbol": ctx.symbol, "skipped": "uncaught_exception",
                            "error": str(exc)})

    n_signals = sum(1 for r in results if r.get("signal"))
    n_placed  = sum(1 for r in results if r.get("placed"))
    n_skipped = sum(1 for r in results if "skipped" in r)

    write_portfolio_event(
        "cycle_complete",
        symbols=len(contexts),
        signals=n_signals,
        placed=n_placed,
        skipped=n_skipped,
        elapsed_s=round(time.monotonic() - started, 2),
        per_symbol=results,
    )


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", nargs="+", default=None,
                   help="Explicit basket (overrides ranking).")
    p.add_argument("--policy", default="equal", choices=["equal", "score", "custom"],
                   help="Capital allocation policy (default: equal).")
    p.add_argument("--portfolio-risk", type=float, default=DEFAULT_PORTFOLIO_RISK,
                   help="Total fraction of balance at risk if every symbol is in (default: 0.02).")
    p.add_argument("--target-wr", type=float, default=48.0,
                   help="Minimum IS WR%% for auto-basket selection (default: 48).")
    p.add_argument("--once",    action="store_true",
                   help="Run one cycle then exit.")
    p.add_argument("--dry-run", action="store_true",
                   help="Detect + notify, never send orders.")
    p.add_argument("--results-dir", default=str(RESULTS_DIR),
                   help=f"Notebook-24 results directory (default: {RESULTS_DIR}).")
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()

    # ── 1) Load research artifacts ──────────────────────────────────────────-
    basket = load_basket(
        results_dir      = Path(args.results_dir),
        target_wr        = args.target_wr,
        symbols_override = args.symbols,
    )
    print(f"Basket loaded: {len(basket)} symbols — {basket.symbols}")
    print(f"Selection criteria: {basket.criteria}")

    # ── 2) Allocate capital across the basket ───────────────────────────────-
    allocator = CapitalAllocator(total_portfolio_risk=args.portfolio_risk)
    allocations = allocator.allocate(basket, policy=args.policy)
    print(f"\nCapital plan (policy={args.policy}, total_risk={args.portfolio_risk:.3%}):")
    for a in allocations:
        print(f"  {a.symbol:8} weight={a.weight:6.2%}  risk_per_trade={a.risk_per_trade:.4%}")

    # ── 2b) OBSERVABILITY: compute run identity + config fingerprint ─────────
    # Done once per process. All downstream events reference these.
    run_id           = make_run_id()
    htf_policy       = htf_policy_snapshot(
        use_synth_h1               = USE_SYNTH_H1,
        use_synth_d1               = USE_SYNTH_D1,
        h1_freshness_threshold_min = H1_FRESHNESS_THRESHOLD_MIN,
        d1_freshness_threshold_min = D1_FRESHNESS_THRESHOLD_MIN,
    )
    runner_constants = {
        "MAGIC_BASE":                    MAGIC_BASE,
        "SLIPPAGE_LIMIT_THRESHOLD":      SLIPPAGE_LIMIT_THRESHOLD,
        "SLIPPAGE_MAX_POINTS_DEFAULT":   SLIPPAGE_MAX_POINTS_DEFAULT,
        "SLIPPAGE_MAX_POINTS_BY_SYMBOL": dict(SLIPPAGE_MAX_POINTS_BY_SYMBOL),
        "LIMIT_ORDER_STALE_SECONDS":     LIMIT_ORDER_STALE_SECONDS,
        "MAX_SPREAD_POINTS":             MAX_SPREAD_POINTS,
        "DEFAULT_PORTFOLIO_RISK":        DEFAULT_PORTFOLIO_RISK,
        "M5_SECONDS":                    M5_SECONDS,
        "BROKER_WALLCLOCK_OFFSET_HOURS": BROKER_WALLCLOCK_OFFSET_HOURS,
        "RR":                            RR,
        "HISTORY_M5_BARS":               HISTORY_M5_BARS,
        "HISTORY_H1_BARS":               HISTORY_H1_BARS,
        "HISTORY_D1_BARS":               HISTORY_D1_BARS,
        "DEFAULT_BROKER_TO_NY_H":        DEFAULT_BROKER_TO_NY_H,
    }
    per_symbol_payloads = {
        m.symbol: {
            "strategy_config": m.cfg,
            "stats_score":     m.stats.get("score") if isinstance(m.stats, dict) else None,
        }
        for m in basket
    }
    portfolio_config_hash = compute_portfolio_config_hash(
        per_symbol_payloads = per_symbol_payloads,
        runner_constants    = runner_constants,
        htf_policy          = htf_policy,
    )
    git_commit = current_git_commit()

    # ── 3) Boot MT5 ─────────────────────────────────────────────────────────-
    mt5_connect()
    assert_terminal_ready()

    # OBSERVABILITY: one-time bot_run_started event with the full snapshot
    # of what's running. Emitted to portfolio log so a single grep gives
    # the timeline of every process start + its config.
    write_portfolio_event(
        "bot_run_started",
        run_id            = run_id,
        config_hash       = portfolio_config_hash,
        config_version    = {
            "hash":             portfolio_config_hash,
            "loaded_at":        datetime.now(timezone.utc).isoformat(),
            "config_file_path": str(RESULTS_DIR),
            "git_commit":       git_commit,
            "pid":              os.getpid(),
        },
        htf_policy        = htf_policy,
        runner_constants  = runner_constants,
        basket            = basket.symbols,
        per_symbol_config = {
            m.symbol: {
                "config": {
                    "mode":                 m.cfg.mode,
                    "confirms":             list(m.cfg.confirms),
                    "rsi_memory":           m.cfg.rsi_memory,
                    "session":              list(m.cfg.session),
                    "sl_method":            m.cfg.sl_method,
                    "atr_min_mult":         m.cfg.atr_min_mult,
                    "adx_min":              m.cfg.adx_min,
                    "min_reactions":        m.cfg.min_reactions,
                    "require_h1_rsi_align": m.cfg.require_h1_rsi_align,
                    "require_macd_align":   m.cfg.require_macd_align,
                    "vol_spike_mult":       m.cfg.vol_spike_mult,
                },
                "hash": compute_symbol_config_hash(
                    strategy_cfg        = m.cfg,
                    slippage_max_points = slippage_max_for(m.symbol),
                    risk_per_trade      = next((a.risk_per_trade for a in allocations
                                                 if a.symbol == m.symbol), 0.0),
                    risk_reward         = RR,
                ),
            }
            for m in basket
        },
    )

    write_portfolio_event(
        "portfolio_start",
        basket=basket.symbols,
        policy=args.policy,
        portfolio_risk=args.portfolio_risk,
        criteria=basket.criteria,
        dry_run=args.dry_run,
        run_id=run_id,
        config_hash=portfolio_config_hash,
    )

    # ── 4) Build per-symbol contexts ────────────────────────────────────────-
    contexts = build_contexts(
        basket, allocations, risk_reward=RR,
        run_id                = run_id,
        portfolio_config_hash = portfolio_config_hash,
        htf_policy            = htf_policy,
    )

    # Single shared watchdog (MT5 connection is process-wide).
    watchdog = Mt5Watchdog(logger=contexts[0].engine.logger, connect_fn=mt5_connect)
    for ctx in contexts:
        ctx.engine.attach_watchdog(watchdog)

    # ── 5) Run ──────────────────────────────────────────────────────────────-
    try:
        if args.once:
            run_portfolio_cycle(contexts, dry_run=args.dry_run)
            return

        print(f"\nMulti-symbol scalper running on {len(contexts)} symbols. Ctrl+C to stop.")
        while True:
            try:
                run_portfolio_cycle(contexts, dry_run=args.dry_run)
            except Exception as exc:
                write_portfolio_event("portfolio_cycle_exception", error=repr(exc))
            sleep_until_next_m5()

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        write_portfolio_event("portfolio_stop")
        # Telegram: one stop-message per symbol so the operator knows the bot
        # is no longer monitoring. Per-symbol so it survives partial failures.
        for ctx in contexts:
            try:
                ctx.notifier.notify_bot_lifecycle(
                    symbol=ctx.symbol, phase="stop",
                    reason="manual_or_crash",
                )
            except Exception:
                pass
            try:
                if hasattr(ctx.engine, "shutdown"):
                    ctx.engine.shutdown()
            except Exception:
                pass
        mt5.shutdown()


if __name__ == "__main__":
    main()
