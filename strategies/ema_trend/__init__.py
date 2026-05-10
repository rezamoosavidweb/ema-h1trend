"""EMA H1 trend + M5 crypto pending-stop strategy (shared notebook + MT5)."""

from strategies.ema_trend.backtest import run_backtest
from strategies.ema_trend.backtest_new import run_backtest as run_backtest_new
from strategies.ema_trend.crypto_core import (
    EMA_FAST,
    EMA_MID,
    EMA_SLOW,
    MIN_WARMUP_BARS_H1,
    MIN_WARMUP_BARS_M5,
    add_emas,
    compute_pending_setup,
    default_crypto_tick,
    h1_trend_series,
    last_closed_bar_index,
    merge_h1_trend_onto_m5,
    min_bars_needed_for_signal,
    rates_to_ohlcv_df,
)

__all__ = [
    "EMA_FAST",
    "EMA_MID",
    "EMA_SLOW",
    "MIN_WARMUP_BARS_H1",
    "MIN_WARMUP_BARS_M5",
    "add_emas",
    "compute_pending_setup",
    "default_crypto_tick",
    "h1_trend_series",
    "last_closed_bar_index",
    "merge_h1_trend_onto_m5",
    "min_bars_needed_for_signal",
    "rates_to_ohlcv_df",
    "run_backtest",
]
