"""
Vector-free pending-stop backtest engine.

THIS FILE IS IDENTICAL TO THE MT5 VERSION — DO NOT MODIFY.
Original: strategies/ema_trend/backtest.py
"""

from __future__ import annotations

import pandas as pd

from strategy.crypto_core import compute_pending_setup


def run_backtest(
    data: pd.DataFrame,
    *,
    start_balance: float,
    lookback_bars: int,
    pending_offset_ticks: float,
    pip_size: float,
    rr: float,
    risk_cash: float,
    pending_expiry_min: int,
    entry_timeframe_minutes: int = 5,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Walk-forward simulation: pending orders with expiry, TP/SL same candle priority SL first.

    ``data`` must be merged M5 context with ``trend`` column (from merge_h1_trend_onto_m5).
    """
    data = data.copy().sort_index()

    expiry_bars = max(1, int(pending_expiry_min / entry_timeframe_minutes))

    trades: list[dict] = []
    balance = float(start_balance)
    equity_curve: list[tuple] = []

    pending: dict | None = None
    position: dict | None = None

    for i in range(lookback_bars, len(data)):
        idx = data.index[i]
        row = data.iloc[i]

        floating = 0.0
        if position is not None:
            if position["side"] == "buy":
                floating = (row["close"] - position["entry"]) * position["qty"]
            else:
                floating = (position["entry"] - row["close"]) * position["qty"]
        equity_curve.append((idx, balance + floating))

        if position is not None:
            hit_sl = False
            hit_tp = False

            if position["side"] == "buy":
                hit_sl = row["low"] <= position["sl"]
                hit_tp = row["high"] >= position["tp"]
            else:
                hit_sl = row["high"] >= position["sl"]
                hit_tp = row["low"] <= position["tp"]

            if hit_sl or hit_tp:
                exit_price = position["sl"] if hit_sl else position["tp"]
                pnl = (exit_price - position["entry"]) * position["qty"]
                if position["side"] == "sell":
                    pnl = -pnl

                balance += pnl
                trades.append(
                    {
                        "entry_time": position["entry_time"],
                        "exit_time": idx,
                        "side": position["side"],
                        "entry": position["entry"],
                        "sl": position["sl"],
                        "tp": position["tp"],
                        "exit": exit_price,
                        "qty": position["qty"],
                        "pnl": pnl,
                        "balance_after": balance,
                        "result": "win" if pnl > 0 else "loss",
                    }
                )
                position = None
            continue

        if pending is not None:
            age = i - pending["created_i"]
            if age > expiry_bars:
                pending = None
            else:
                if pending["side"] == "buy":
                    if row["high"] >= pending["entry"]:
                        position = {
                            "side": "buy",
                            "entry": pending["entry"],
                            "sl": pending["sl"],
                            "tp": pending["tp"],
                            "qty": pending["qty"],
                            "entry_time": idx,
                        }
                        pending = None
                else:
                    if row["low"] <= pending["entry"]:
                        position = {
                            "side": "sell",
                            "entry": pending["entry"],
                            "sl": pending["sl"],
                            "tp": pending["tp"],
                            "qty": pending["qty"],
                            "entry_time": idx,
                        }
                        pending = None

        if pending is None and position is None:
            setup = compute_pending_setup(
                data,
                bar_index=i,
                lookback_bars=lookback_bars,
                pending_offset_ticks=pending_offset_ticks,
                pip_size=pip_size,
                rr=rr,
                risk_cash=risk_cash,
            )
            if setup is not None:
                pending = {
                    "side": setup["side"],
                    "entry": float(setup["entry"]),
                    "sl": float(setup["sl"]),
                    "tp": float(setup["tp"]),
                    "qty": float(setup["qty"]),
                    "created_i": i,
                }

    trades_df = pd.DataFrame(trades)
    eq = pd.Series({t: v for t, v in equity_curve}).sort_index()
    return trades_df, eq
