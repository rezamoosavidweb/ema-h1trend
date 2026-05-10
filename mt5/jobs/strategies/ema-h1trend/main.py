# ==============================================================================
# SECTION 1 - Imports and parameters
import warnings
import time
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

import os
import sys
from pathlib import Path

# Repo root: must use __file__ so imports work when cwd is mt5/ (Path.cwd() would be wrong).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from strategies.ema_trend.setup import list_setup_signals
from strategies.ema_trend.crypto_core import EMA_FAST, EMA_MID, EMA_SLOW, default_crypto_tick
from strategies.ema_trend.crypto_core import add_emas
from strategies.ema_trend.crypto_core import merge_h1_trend_onto_m5
import MetaTrader5 as mt5

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import run_strategy03_errante as r3

plt.style.use("seaborn-v0_8-darkgrid")

# === User parameters (crypto quotes) ===
# Broker names often differ (e.g. BTCUSD.i). Set MT5_SYMBOL to the exact Market Watch name, or
# leave a short name: if MT5 returns exactly one substring match, it is used automatically.
SYMBOL = os.environ.get("MT5_SYMBOL", "BTCUSD").strip()
TF_ENTRY = "M5"
TF_TREND = "H1"

MT5_TF = {
    'M1': mt5.TIMEFRAME_M1,
    'M5': mt5.TIMEFRAME_M5,
    'M15': mt5.TIMEFRAME_M15,
    'M30': mt5.TIMEFRAME_M30,
    'H1': mt5.TIMEFRAME_H1,
    'H4': mt5.TIMEFRAME_H4,
    'D1': mt5.TIMEFRAME_D1,
}

LOOKBACK_BARS = 5
ENTRY_TF_MINUTES = 5
MAGIC = 20260510

# Pending stop distance beyond swing = PENDING_OFFSET_TICKS * PIP_SIZE (price units).
PENDING_OFFSET_TICKS = 3.0
PENDING_EXPIRY_MIN = 60
RR = 1.0  # TP:SL = 1:1
RISK_PER_TRADE = 0.01

PIP_SIZE = default_crypto_tick(SYMBOL)
# Backtest engine uses this name; value is tick/step size for crypto.
PENDING_OFFSET_PIPS = float(PENDING_OFFSET_TICKS)

START_BALANCE = 10000.0



# Limit backtest/chart M5 history to the last BARS rows (trim applied in Section 4 after H1→M5 mapping).
# EMA + merge still see full loaded history first so rows inside the window are warmed up.
BARS_ENTRY = 100  # e.g. 100
BARS_TREND = 100  # e.g. 100


# ==============================================================================

# SECTION 2 - Load data from cache: ./data/<symbol>/<timeframe>
def _standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
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
        raise ValueError("Could not infer datetime index from cache file.")

    need = ["open", "high", "low", "close"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Missing OHLC columns: {miss}")

    if "volume" not in df.columns:
        df["volume"] = 1.0

    out = df[["open", "high", "low", "close", "volume"]].copy()
    out = out.sort_index().dropna()
    return out


def _parse_env_int(name: str) -> int | None:
    v = os.environ.get(name)
    if v is None or not str(v).strip():
        return None
    try:
        return int(str(v).strip(), 10)
    except ValueError:
        return None


def _resolve_mt5_symbol(requested: str) -> str:
    """Resolve ``requested`` to a broker symbol; if unknown, retry when MT5 suggests a single match."""
    try:
        return r3.ensure_symbol(requested)
    except RuntimeError:
        stem = requested.replace(".", "").replace("#", "")
        hints = r3.symbol_hints(stem)
        if len(hints) == 1:
            resolved = r3.ensure_symbol(hints[0])
            print(f"MT5: using broker symbol {resolved!r} (requested {requested!r})", flush=True)
            return resolved
        ru = requested.upper()
        for h in sorted(hints):
            hu = h.upper()
            if hu == ru or hu.startswith(ru + ".") or hu.startswith(ru + "#"):
                resolved = r3.ensure_symbol(h)
                print(f"MT5: using broker symbol {resolved!r} (requested {requested!r})", flush=True)
                return resolved
        raise RuntimeError(
            f"Symbol {requested!r} not found in MT5. Candidate names: {hints or '(none)'}. "
            "Set env MT5_SYMBOL to the exact symbol from Market Watch."
        ) from None


def fetch_mt5(symbol: str, tf: str, bars: int, date_from=None, date_to=None) -> pd.DataFrame:
    timeframe = MT5_TF[tf]
    rates = r3._copy_rates(symbol, timeframe, bars)

    if rates is None or len(rates) == 0:
        last_error = mt5.last_error()
        hint = ""
        if last_error and last_error[0] == -10004:
            hint = (
                " (-10004 = no IPC: start MetaTrader 5, log in, then retry. "
                "Use 64-bit Python. Optional: MT5_TERMINAL_PATH, MT5_LOGIN/MT5_PASSWORD/MT5_SERVER.)"
            )
        raise RuntimeError(f"No data from MT5 for {symbol} {tf}. last_error={last_error}{hint}")

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['symbol'] = symbol
    df['timeframe'] = tf
    df = _standardize_ohlcv(df)
    return df


_login = _parse_env_int("MT5_LOGIN")
_password = os.environ.get("MT5_PASSWORD")
_server = os.environ.get("MT5_SERVER")

r3.mt5_connect(_login, _password, _server)
r3.assert_terminal_ready()
_mt5_symbol = _resolve_mt5_symbol(SYMBOL)

print(
    f"\nSymbol requested={SYMBOL!r}, MT5 symbol={_mt5_symbol!r}, Trend={TF_TREND}, Entry={TF_ENTRY},\n "
    f"PIP_SIZE(tick)={PIP_SIZE}, offset_ticks={PENDING_OFFSET_TICKS}, BARS_ENTRY={BARS_ENTRY}, BARS_TREND={BARS_TREND}\n\n",
    flush=True,
)

def main():  
    m5 = fetch_mt5(_mt5_symbol, TF_ENTRY, BARS_ENTRY)
    h1 = fetch_mt5(_mt5_symbol, TF_TREND, BARS_TREND)


    print("M5 head:\n", m5.head(3))
    print("H1 head:\n", h1.head(3))

    # ==============================================================================

    # SECTION 3 - Build EMA 21/13/8 on both timeframes


    m5 = add_emas(m5)
    h1 = add_emas(h1)

    print("EMA columns added.\n")
    print(h1[["close", f"ema_{EMA_FAST}", f"ema_{EMA_MID}", f"ema_{EMA_SLOW}"]].tail(5))

    # ==============================================================================

    # SECTION 4 - H1 trend direction confirmation
    # Bull trend: EMA8 > EMA13 > EMA21 and close > EMA21
    # Bear trend: EMA8 < EMA13 < EMA21 and close < EMA21

    m5_ctx = merge_h1_trend_onto_m5(m5, h1)

    print(m5_ctx["trend"].value_counts(dropna=False))


    # ==============================================================================

    # SECTION 5 - Strategy engine (pending stop orders + expiry + TP/SL)
    entry_signals_df = list_setup_signals(
        m5_ctx,
        start_balance=START_BALANCE,
        lookback_bars=LOOKBACK_BARS,
        pending_offset_ticks=PENDING_OFFSET_PIPS,
        pip_size=PIP_SIZE,
        rr=RR,
        risk_per_trade=RISK_PER_TRADE,
    )
    print(f"\n\nTotal entry signals: {len(entry_signals_df)}")
    if not entry_signals_df.empty:
        print(entry_signals_df[["signal_bar_time", "side", "entry"]])
    else:
        print("(no setup rows in this window)")


    # ==============================================================================
    # ==============================================================================
    # SECTION 6 - Sync MT5 pending orders with latest signal

    expiry_bars = max(1, int(PENDING_EXPIRY_MIN / ENTRY_TF_MINUTES))

    # reconnect MT5
    r3.mt5_connect(_login, _password, _server)
    r3.assert_terminal_ready()

    try:

        # --------------------------------------------------------------------------
        # Get latest signal
        if entry_signals_df.empty:
            print("No signals.")
            sys.exit(0)

        last_signal = entry_signals_df.iloc[-1]

        signal_time = pd.to_datetime(last_signal["signal_bar_time"])
        current_time = m5_ctx.index[-1]

        bars_passed = int(
            (current_time - signal_time).total_seconds()
            / 60
            / ENTRY_TF_MINUTES
        )

        print(
            f"\nLatest signal:"
            f"\nside={last_signal['side']}"
            f"\nentry={last_signal['entry']}"
            f"\nsignal_time={signal_time}"
            f"\nbars_passed={bars_passed}"
            f"\nexpiry_bars={expiry_bars}\n"
        )

        # --------------------------------------------------------------------------
        # Get current pending orders from MT5
        mt5_orders = mt5.orders_get(symbol=_mt5_symbol)

        pending_order = None

        if mt5_orders:
            for o in mt5_orders:
                # فقط اردرهای همین استراتژی
                if o.magic != MAGIC:
                    continue

                if o.type in (
                    mt5.ORDER_TYPE_BUY_STOP,
                    mt5.ORDER_TYPE_SELL_STOP,
                ):
                    print("-------------------------------------------------------------------------------------------------------------------------------")
                    print(f"Current Pending order found: ticket:{o.ticket} / type:{o.type} / price_open:{o.price_open} / sl:{o.sl} / tp:{o.tp} / magic:{o.magic} / comment:{o.comment}")
                    print("-------------------------------------------------------------------------------------------------------------------------------")
                    pending_order = o
                    break

        # --------------------------------------------------------------------------
        # EXPIRED SIGNAL
        if bars_passed >= expiry_bars:

            print("Signal expired.")

            if pending_order is not None:

                print(f"Removing pending order #{pending_order.ticket}")

                request = {
                    "action": mt5.TRADE_ACTION_REMOVE,
                    "order": pending_order.ticket,
                }

                result = mt5.order_send(request)

                print("REMOVE RESULT:", result)

            else:
                print("No pending order to remove.")

        # --------------------------------------------------------------------------
        # VALID SIGNAL
        else:

            print("Signal valid.")

            desired_entry = float(last_signal["entry"])

            # ----------------------------------------------------------------------
            # Rebuild SL/TP from dataframe row
            desired_sl = float(last_signal["sl"])
            desired_tp = float(last_signal["tp"])

            side = last_signal["side"]

            desired_type = (
                mt5.ORDER_TYPE_BUY_STOP
                if side == "buy"
                else mt5.ORDER_TYPE_SELL_STOP
            )

            # ----------------------------------------------------------------------
            # NO EXISTING PENDING -> CREATE
            if pending_order is None:

                print("Creating new pending order...")

                request = {
                    "action": mt5.TRADE_ACTION_PENDING,
                    "symbol": _mt5_symbol,
                    "volume": 0.01,
                    "type": desired_type,
                    "price": desired_entry,
                    "sl": desired_sl,
                    "tp": desired_tp,
                    "magic": MAGIC,
                    "comment": "ema_trend_python",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_RETURN,
                }

                result = mt5.order_send(request)

                print("----------------------------------------------------------------------------------------------------------------------")
                print(f"New Pending order created: ticket:{result.order} / type:{result.type} / price_open:{result.price_open} / sl:{result.sl} / tp:{result.tp} / magic:{result.magic} / comment:{result.comment}")
                print("----------------------------------------------------------------------------------------------------------------------")
            # ----------------------------------------------------------------------
            # EXISTING PENDING -> COMPARE
            else:

                print(f"Existing pending #{pending_order.ticket}")

                same_entry = abs(pending_order.price_open - desired_entry) < PIP_SIZE
                same_sl = abs(pending_order.sl - desired_sl) < PIP_SIZE
                same_tp = abs(pending_order.tp - desired_tp) < PIP_SIZE
                same_type = pending_order.type == desired_type

                if same_entry and same_sl and same_tp and same_type:


                    print("-------------------------------------------------------------------------------------------------------------------------------")
                    print(f"Pending order already up-to-date and dont need update: ticket:{pending_order.ticket} / type:{pending_order.type} / price_open:{pending_order.price_open} / sl:{pending_order.sl} / tp:{pending_order.tp} / magic:{pending_order.magic} / comment:{pending_order.comment}")
                    print("-------------------------------------------------------------------------------------------------------------------------------")
                else:

                    print("Pending changed -> modifying...")

                    request = {
                        "action": mt5.TRADE_ACTION_MODIFY,
                        "order": pending_order.ticket,
                        "price": desired_entry,
                        "sl": desired_sl,
                        "tp": desired_tp,
                    }

                    result = mt5.order_send(request)
                    print("----------------------------------------------------------------------------------------------------------------------")
                    print(f"Pending order modified: ticket:{result.order} / type:{result.type} / price_open:{result.price_open} / sl:{result.sl} / tp:{result.tp} / magic:{result.magic} / comment:{result.comment}")
                    print("----------------------------------------------------------------------------------------------------------------------")
    finally:
        mt5.shutdown()  
  
    
    
def sleep_until_next_candle(tf_minutes: int = 5):
    now = datetime.now(timezone.utc)

    total_seconds = now.minute * 60 + now.second

    candle_seconds = tf_minutes * 60

    next_close = ((total_seconds // candle_seconds) + 1) * candle_seconds

    wait_seconds = next_close - total_seconds

    if wait_seconds <= 0:
        wait_seconds += candle_seconds

    print(
        f"\n[{now.strftime('%Y-%m-%d %H:%M:%S UTC')}] "
        f"Sleeping {wait_seconds} sec until next {tf_minutes}m candle..."
    )

    time.sleep(wait_seconds + 1)
    
    
# loop
while True:

    try:

        print("\n" + "=" * 120)
        print("RUNNING STRATEGY...")
        print("=" * 120)
        main()
        
    except Exception as e:
        print("ERROR:", e)

    finally:
        try:
            mt5.shutdown()
        except:
            pass

    sleep_until_next_candle(ENTRY_TF_MINUTES)
