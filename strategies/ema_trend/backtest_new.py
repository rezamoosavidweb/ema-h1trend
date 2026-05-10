"""Vector-free pending-stop backtest used by notebooks/03_strategy03_crypto.ipynb."""

from __future__ import annotations

import pandas as pd

from strategies.ema_trend.setup import _simulate_walk_forward



def run_backtest(
    data: pd.DataFrame,
    *,
    start_balance: float,
    lookback_bars: int,
    pending_offset_ticks: float,
    pip_size: float,
    rr: float,
    risk_per_trade: float,
    pending_expiry_min: int,
    entry_timeframe_minutes: int = 5,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Walk-forward simulation: pending orders with expiry, TP/SL same candle priority SL first.

    ``data`` must be merged M5 context with ``trend`` column (from merge_h1_trend_onto_m5).
    """
    trades, equity_curve, _ = _simulate_walk_forward(
        data,
        start_balance=start_balance,
        lookback_bars=lookback_bars,
        pending_offset_ticks=pending_offset_ticks,
        pip_size=pip_size,
        rr=rr,
        risk_per_trade=risk_per_trade,
        pending_expiry_min=pending_expiry_min,
        entry_timeframe_minutes=entry_timeframe_minutes,
    )

    trades_df = pd.DataFrame(trades)
    eq = pd.Series({t: v for t, v in equity_curve}).sort_index()
    return trades_df, eq


