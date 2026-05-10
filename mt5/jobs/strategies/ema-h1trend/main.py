# ==============================================================================
# Live trading bot: EMA H1 Trend + M5 Entry  |  Pending stop orders  |  MT5
# ==============================================================================
import warnings
import time
import logging
from datetime import datetime, timezone
from typing import Optional

warnings.filterwarnings("ignore")

import os
import sys
from pathlib import Path

# Repo root must be derived from __file__ so imports work when cwd is mt5/.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import matplotlib.pyplot as plt

from strategies.ema_trend.setup import list_setup_signals
from strategies.ema_trend.crypto_core import (
    EMA_FAST, EMA_MID, EMA_SLOW,
    default_crypto_tick, add_emas, merge_h1_trend_onto_m5,
)
import MetaTrader5 as mt5

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import run_strategy03_errante as r3

plt.style.use("seaborn-v0_8-darkgrid")


# ==============================================================================
# Logging — UTC timestamps, stdout
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s UTC | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.Formatter.converter = time.gmtime   # force UTC in log timestamps
log = logging.getLogger(__name__)


# ==============================================================================
# Configuration
# ==============================================================================

# Broker names often differ (e.g. BTCUSD.i). Set MT5_SYMBOL to the exact Market Watch
# name, or leave a short name: if MT5 returns exactly one substring match it is used.
SYMBOL: str = os.environ.get("MT5_SYMBOL", "BTCUSD").strip()
TF_ENTRY: str = "M5"
TF_TREND: str = "H1"

MT5_TF: dict[str, int] = {
    "M1":  mt5.TIMEFRAME_M1,
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
    "D1":  mt5.TIMEFRAME_D1,
}

LOOKBACK_BARS: int     = 5
ENTRY_TF_MINUTES: int  = 5
MAGIC: int             = 20260510

# Pending stop distance beyond swing high/low  = PENDING_OFFSET_TICKS * PIP_SIZE
PENDING_OFFSET_TICKS: float = 3.0
PENDING_EXPIRY_MIN: int     = 60
RR: float                   = 1.0       # TP:SL = 1:1
RISK_PER_TRADE: float       = 0.01

# PIP_SIZE is tick/step size for crypto (no MT5 connection needed — pure symbol string).
PIP_SIZE: float          = default_crypto_tick(SYMBOL)
PENDING_OFFSET_PIPS: float = float(PENDING_OFFSET_TICKS)   # alias expected by backtest engine

START_BALANCE: float = 10_000.0

# EMA + merge see the full loaded history; these bars limit the signal window only.
BARS_ENTRY: int = 100
BARS_TREND: int = 100

# MT5 reconnect policy
MT5_CONNECT_RETRIES: int    = 3
MT5_CONNECT_RETRY_DELAY: int = 10   # seconds between retries


# ==============================================================================
# Environment helpers
# ==============================================================================

def _parse_env_int(name: str) -> int | None:
    v = os.environ.get(name)
    if v is None or not str(v).strip():
        return None
    try:
        return int(str(v).strip(), 10)
    except ValueError:
        return None


_login:    Optional[int] = _parse_env_int("MT5_LOGIN")
_password: Optional[str] = os.environ.get("MT5_PASSWORD")
_server:   Optional[str] = os.environ.get("MT5_SERVER")


# ==============================================================================
# Data helpers
# ==============================================================================

def _standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names and ensure a DatetimeIndex. Unchanged from original."""
    rename_map = {
        "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume",
        "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume",
        "tick_volume": "volume",
    }
    df = df.rename(columns=rename_map)

    if not isinstance(df.index, pd.DatetimeIndex):
        dt_cols = [c for c in ["time", "datetime", "date", "timestamp"] if c in df.columns]
        if dt_cols:
            dt_col = dt_cols[0]
            df[dt_col] = pd.to_datetime(df[dt_col], utc=False, errors="coerce")
            df = df.set_index(dt_col)
        else:
            first = df.columns[0]
            maybe_dt = pd.to_datetime(df[first], utc=False, errors="coerce")
            if maybe_dt.notna().mean() > 0.8:
                df[first] = maybe_dt
                df = df.set_index(first)

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Could not infer datetime index from data.")

    miss = [c for c in ["open", "high", "low", "close"] if c not in df.columns]
    if miss:
        raise ValueError(f"Missing OHLC columns: {miss}")

    if "volume" not in df.columns:
        df["volume"] = 1.0

    return df[["open", "high", "low", "close", "volume"]].copy().sort_index().dropna()


def _resolve_mt5_symbol(requested: str) -> str:
    """Resolve requested name to the exact broker symbol. Unchanged from original."""
    try:
        return r3.ensure_symbol(requested)
    except RuntimeError:
        stem = requested.replace(".", "").replace("#", "")
        hints = r3.symbol_hints(stem)
        if len(hints) == 1:
            resolved = r3.ensure_symbol(hints[0])
            log.info("MT5: using broker symbol %r (requested %r)", resolved, requested)
            return resolved
        ru = requested.upper()
        for h in sorted(hints):
            hu = h.upper()
            if hu == ru or hu.startswith(ru + ".") or hu.startswith(ru + "#"):
                resolved = r3.ensure_symbol(h)
                log.info("MT5: using broker symbol %r (requested %r)", resolved, requested)
                return resolved
        raise RuntimeError(
            f"Symbol {requested!r} not found in MT5. Candidates: {hints or '(none)'}. "
            "Set env MT5_SYMBOL to the exact symbol from Market Watch."
        ) from None


# ==============================================================================
# MT5 connection
# ==============================================================================

def connect_mt5() -> bool:
    """Connect to MT5 and verify the terminal is ready. Retries up to MT5_CONNECT_RETRIES times."""
    for attempt in range(1, MT5_CONNECT_RETRIES + 1):
        try:
            r3.mt5_connect(_login, _password, _server)
            r3.assert_terminal_ready()
            log.info("MT5 connected (attempt %d/%d).", attempt, MT5_CONNECT_RETRIES)
            return True
        except Exception as exc:
            log.warning("MT5 connect attempt %d/%d failed: %s", attempt, MT5_CONNECT_RETRIES, exc)
            if attempt < MT5_CONNECT_RETRIES:
                time.sleep(MT5_CONNECT_RETRY_DELAY)
    log.error("MT5 connection failed after %d attempts.", MT5_CONNECT_RETRIES)
    return False


# ==============================================================================
# Data fetching
# ==============================================================================

def fetch_data(mt5_symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch BARS_ENTRY M5 bars and BARS_TREND H1 bars from MT5."""

    def _load(tf: str, bars: int) -> pd.DataFrame:
        rates = r3._copy_rates(mt5_symbol, MT5_TF[tf], bars)
        if rates is None or len(rates) == 0:
            last_error = mt5.last_error()
            hint = ""
            if last_error and last_error[0] == -10004:
                hint = (
                    " (-10004 = no IPC: start MetaTrader 5, log in, then retry. "
                    "Use 64-bit Python. Optional: MT5_TERMINAL_PATH, MT5_LOGIN/MT5_PASSWORD/MT5_SERVER.)"
                )
            raise RuntimeError(
                f"No data from MT5 for {mt5_symbol} {tf}. last_error={last_error}{hint}"
            )
        df = pd.DataFrame(rates)
        df["time"]      = pd.to_datetime(df["time"], unit="s", utc=True)
        df["symbol"]    = mt5_symbol
        df["timeframe"] = tf
        return _standardize_ohlcv(df)

    m5 = _load(TF_ENTRY, BARS_ENTRY)
    h1 = _load(TF_TREND, BARS_TREND)

    log.info("Fetched %d M5 bars and %d H1 bars.", len(m5), len(h1))
    return m5, h1


# ==============================================================================
# Context building
# ==============================================================================

def build_context(m5: pd.DataFrame, h1: pd.DataFrame) -> pd.DataFrame:
    """Add EMAs on both timeframes and merge H1 trend direction onto M5."""
    m5 = add_emas(m5)
    h1 = add_emas(h1)

    log.info(
        "H1 EMA tail:\n%s",
        h1[["close", f"ema_{EMA_FAST}", f"ema_{EMA_MID}", f"ema_{EMA_SLOW}"]].tail(3).to_string(),
    )

    m5_ctx = merge_h1_trend_onto_m5(m5, h1)

    log.info("M5 trend distribution:\n%s", m5_ctx["trend"].value_counts(dropna=False).to_string())
    return m5_ctx


# ==============================================================================
# Signal generation
# ==============================================================================

def generate_signals(m5_ctx: pd.DataFrame) -> pd.DataFrame:
    """Run the setup signal engine. Parameters and call signature unchanged from original."""
    signals = list_setup_signals(
        m5_ctx,
        start_balance=START_BALANCE,
        lookback_bars=LOOKBACK_BARS,
        pending_offset_ticks=PENDING_OFFSET_PIPS,
        pip_size=PIP_SIZE,
        rr=RR,
        risk_per_trade=RISK_PER_TRADE,
    )

    log.info("Total entry signals: %d", len(signals))
    if not signals.empty:
        log.info("Signals:\n%s", signals[["signal_bar_time", "side", "entry"]].to_string())
    else:
        log.info("(no setup rows in this window)")

    return signals


# ==============================================================================
# Volume normalisation
# ==============================================================================

def normalize_volume(mt5_symbol: str, raw_volume: float) -> float:
    """Round raw_volume to the nearest valid step and clamp to [volume_min, volume_max]."""
    info = mt5.symbol_info(mt5_symbol)
    if info is None:
        log.warning("symbol_info unavailable for %s — using raw volume %.4f", mt5_symbol, raw_volume)
        return raw_volume

    vol_step = info.volume_step if info.volume_step > 0 else 0.01
    vol_min  = info.volume_min  if info.volume_min  > 0 else 0.01
    vol_max  = info.volume_max  if info.volume_max  > 0 else 100.0

    steps  = max(1, round(raw_volume / vol_step))
    volume = round(steps * vol_step, 8)
    volume = max(vol_min, min(vol_max, volume))
    return volume


# ==============================================================================
# Pending order helpers
# ==============================================================================

def get_existing_pending_order(mt5_symbol: str):
    """Return the first BUY_STOP or SELL_STOP for this strategy's MAGIC, or None."""
    mt5_orders = mt5.orders_get(symbol=mt5_symbol)
    if not mt5_orders:
        return None

    for o in mt5_orders:
        if o.magic != MAGIC:
            continue
        if o.type in (mt5.ORDER_TYPE_BUY_STOP, mt5.ORDER_TYPE_SELL_STOP):
            log.info(
                "Existing pending order | ticket=%d type=%d price_open=%.5f sl=%.5f tp=%.5f magic=%d comment=%s",
                o.ticket, o.type, o.price_open, o.sl, o.tp, o.magic, o.comment,
            )
            return o

    return None


def create_pending_order(
    mt5_symbol: str,
    desired_type: int,
    desired_entry: float,
    desired_sl: float,
    desired_tp: float,
    volume: float,
) -> None:
    """Send a new pending stop order. Request payload is identical to the original."""
    request = {
        "action":       mt5.TRADE_ACTION_PENDING,
        "symbol":       mt5_symbol,
        "volume":       volume,
        "type":         desired_type,
        "price":        desired_entry,
        "sl":           desired_sl,
        "tp":           desired_tp,
        "magic":        MAGIC,
        "comment":      "ema_trend_python",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_RETURN,
    }

    result = mt5.order_send(request)

    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        log.error("create_pending_order FAILED | result=%s", result)
        return

    log.info(
        "Pending order CREATED | ticket=%d result=%s",
        result.order, result,
    )


def modify_pending_order(
    pending_order,
    desired_entry: float,
    desired_sl: float,
    desired_tp: float,
) -> None:
    """Modify price/SL/TP on an existing pending order. Request payload identical to original."""
    request = {
        "action": mt5.TRADE_ACTION_MODIFY,
        "order":  pending_order.ticket,
        "price":  desired_entry,
        "sl":     desired_sl,
        "tp":     desired_tp,
    }

    result = mt5.order_send(request)

    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        log.error("modify_pending_order FAILED | ticket=%d result=%s", pending_order.ticket, result)
        return

    log.info("Pending order MODIFIED | ticket=%d result=%s", pending_order.ticket, result)


def remove_pending_order(pending_order) -> None:
    """Cancel a pending order. Request payload identical to original."""
    request = {
        "action": mt5.TRADE_ACTION_REMOVE,
        "order":  pending_order.ticket,
    }

    result = mt5.order_send(request)

    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        log.error("remove_pending_order FAILED | ticket=%d result=%s", pending_order.ticket, result)
        return

    log.info("Pending order REMOVED | ticket=%d result=%s", pending_order.ticket, result)


# ==============================================================================
# Order synchronisation  (core trading logic — unchanged from original)
# ==============================================================================

def sync_pending_orders(
    mt5_symbol: str,
    entry_signals_df: pd.DataFrame,
    m5_ctx: pd.DataFrame,
) -> None:
    """
    Sync MT5 pending stop orders with the latest signal.

    Logic is exactly identical to the original Section 6:
      - pick last signal
      - compute bars_passed
      - if expired  -> remove any pending order
      - if valid    -> create / update-if-changed / leave-alone
    """
    expiry_bars = max(1, int(PENDING_EXPIRY_MIN / ENTRY_TF_MINUTES))

    if entry_signals_df.empty:
        log.info("No signals — skipping order sync.")
        return

    last_signal  = entry_signals_df.iloc[-1]
    signal_time  = pd.to_datetime(last_signal["signal_bar_time"])
    current_time = m5_ctx.index[-1]

    bars_passed = int(
        (current_time - signal_time).total_seconds()
        / 60
        / ENTRY_TF_MINUTES
    )

    log.info(
        "Latest signal | side=%s  entry=%.5f  signal_time=%s  bars_passed=%d  expiry_bars=%d",
        last_signal["side"], last_signal["entry"], signal_time, bars_passed, expiry_bars,
    )

    pending_order = get_existing_pending_order(mt5_symbol)

    # ------------------------------------------------------------------
    # EXPIRED SIGNAL
    # ------------------------------------------------------------------
    if bars_passed >= expiry_bars:
        log.info("Signal EXPIRED.")
        if pending_order is not None:
            log.info("Removing expired pending order #%d.", pending_order.ticket)
            remove_pending_order(pending_order)
        else:
            log.info("No pending order to remove.")
        return

    # ------------------------------------------------------------------
    # VALID SIGNAL
    # ------------------------------------------------------------------
    log.info("Signal VALID.")

    desired_entry = float(last_signal["entry"])
    desired_sl    = float(last_signal["sl"])
    desired_tp    = float(last_signal["tp"])
    side          = last_signal["side"]

    desired_type = (
        mt5.ORDER_TYPE_BUY_STOP if side == "buy" else mt5.ORDER_TYPE_SELL_STOP
    )

    # NO EXISTING PENDING -> CREATE
    if pending_order is None:
        log.info("No existing pending order — creating new...")
        volume = normalize_volume(mt5_symbol, 0.01)
        create_pending_order(mt5_symbol, desired_type, desired_entry, desired_sl, desired_tp, volume)
        return

    # EXISTING PENDING -> COMPARE and possibly MODIFY
    log.info("Checking existing pending #%d for changes...", pending_order.ticket)

    same_entry = abs(pending_order.price_open - desired_entry) < PIP_SIZE
    same_sl    = abs(pending_order.sl         - desired_sl)    < PIP_SIZE
    same_tp    = abs(pending_order.tp         - desired_tp)    < PIP_SIZE
    same_type  = pending_order.type == desired_type

    if same_entry and same_sl and same_tp and same_type:
        log.info(
            "Pending order already up-to-date — no update needed | "
            "ticket=%d type=%d price_open=%.5f sl=%.5f tp=%.5f magic=%d comment=%s",
            pending_order.ticket, pending_order.type, pending_order.price_open,
            pending_order.sl, pending_order.tp, pending_order.magic, pending_order.comment,
        )
    else:
        log.info("Pending order changed — modifying...")
        modify_pending_order(pending_order, desired_entry, desired_sl, desired_tp)


# ==============================================================================
# Sleep until next M5 candle boundary
# ==============================================================================

def sleep_until_next_candle(tf_minutes: int = 5) -> None:
    """Sleep until the next candle-close boundary (UTC). Logic unchanged from original."""
    now = datetime.now(timezone.utc)

    total_seconds  = now.minute * 60 + now.second
    candle_seconds = tf_minutes * 60
    next_close     = ((total_seconds // candle_seconds) + 1) * candle_seconds
    wait_seconds   = next_close - total_seconds

    if wait_seconds <= 0:
        wait_seconds += candle_seconds

    log.info(
        "[%s] Sleeping %ds until next %dm candle...",
        now.strftime("%Y-%m-%d %H:%M:%S UTC"), wait_seconds, tf_minutes,
    )

    time.sleep(wait_seconds + 1)


# ==============================================================================
# Single strategy cycle
# ==============================================================================

def run_cycle(mt5_symbol: str) -> None:
    """
    One complete strategy cycle:
        connect → fetch → build context → generate signals → sync orders → shutdown.

    MT5 is always shut down in the finally block regardless of errors.
    """
    log.info("=" * 80)
    log.info("RUNNING STRATEGY CYCLE")
    log.info("=" * 80)

    if not connect_mt5():
        raise RuntimeError("MT5 connection failed — skipping cycle.")

    try:
        m5, h1         = fetch_data(mt5_symbol)
        m5_ctx         = build_context(m5, h1)
        entry_signals  = generate_signals(m5_ctx)
        sync_pending_orders(mt5_symbol, entry_signals, m5_ctx)
    finally:
        mt5.shutdown()
        log.info("MT5 shutdown.")


# ==============================================================================
# Entry point
# ==============================================================================

def main() -> None:
    """
    Resolve the broker symbol once, then run the production loop.

    Loop behaviour:
      - Fires immediately on start.
      - After each cycle (success or error) sleeps until the next M5 candle boundary.
      - Guards against processing the same candle twice (restart safety).
      - Reconnects MT5 automatically on drop.
    """
    # One-time symbol resolution (needs a brief MT5 connection)
    if not connect_mt5():
        log.error("Cannot start: initial MT5 connection failed.")
        sys.exit(1)

    try:
        mt5_symbol = _resolve_mt5_symbol(SYMBOL)
    finally:
        mt5.shutdown()

    log.info(
        "Symbol requested=%r  MT5=%r  Trend=%s  Entry=%s  "
        "PIP_SIZE=%.6f  offset_ticks=%.1f  BARS_ENTRY=%d  BARS_TREND=%d",
        SYMBOL, mt5_symbol, TF_TREND, TF_ENTRY,
        PIP_SIZE, PENDING_OFFSET_TICKS, BARS_ENTRY, BARS_TREND,
    )

    last_processed_candle: Optional[pd.Timestamp] = None

    while True:
        # ------------------------------------------------------------------
        # Duplicate-candle guard: peek at the latest M5 bar time cheaply.
        # ------------------------------------------------------------------
        current_candle: Optional[pd.Timestamp] = None

        if connect_mt5():
            try:
                peek = r3._copy_rates(mt5_symbol, MT5_TF[TF_ENTRY], 2)
                if peek is not None and len(peek) > 0:
                    current_candle = pd.Timestamp(peek[-1]["time"], unit="s", tz="UTC")
            finally:
                mt5.shutdown()

        if current_candle is not None and current_candle == last_processed_candle:
            log.info("Candle %s already processed — waiting for next.", current_candle)
            sleep_until_next_candle(ENTRY_TF_MINUTES)
            continue

        # ------------------------------------------------------------------
        # Run the full cycle.
        # ------------------------------------------------------------------
        try:
            run_cycle(mt5_symbol)
            if current_candle is not None:
                last_processed_candle = current_candle
        except Exception as exc:
            log.exception("ERROR in strategy cycle: %s", exc)
            try:
                mt5.shutdown()
            except Exception:
                pass

        sleep_until_next_candle(ENTRY_TF_MINUTES)


if __name__ == "__main__":
    main()
