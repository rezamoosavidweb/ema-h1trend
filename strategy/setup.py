"""
Signal generation and walk-forward simulation engine.

THIS FILE IS IDENTICAL TO THE MT5 VERSION — DO NOT MODIFY.
Original: strategies/ema_trend/setup.py
"""

from __future__ import annotations

import pandas as pd

from strategy.crypto_core import compute_pending_setup


def _simulate_walk_forward(
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
) -> tuple[list[dict], list[tuple], list[dict]]:
    """
    Single walk-forward pass: closed trades, equity samples, and each pending→position fill.

    Returns (trades, equity_curve_tuples, trade_entries) where trade_entries rows describe
    actual fills (pending triggered) including the signal bar that created the pending.
    """
    data = data.copy().sort_index()

    expiry_bars = max(1, int(pending_expiry_min / entry_timeframe_minutes))

    trades: list[dict] = []
    trade_entries: list[dict] = []
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
                        created_i = int(pending["created_i"])
                        trade_entries.append(
                            {
                                "signal_bar_time": data.index[created_i],
                                "signal_bar_index": created_i,
                                "entry_time": idx,
                                "entry_bar_index": i,
                                "side": "buy",
                                "entry": float(pending["entry"]),
                                "sl": float(pending["sl"]),
                                "tp": float(pending["tp"]),
                                "qty": float(pending["qty"]),
                            }
                        )
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
                        created_i = int(pending["created_i"])
                        trade_entries.append(
                            {
                                "signal_bar_time": data.index[created_i],
                                "signal_bar_index": created_i,
                                "entry_time": idx,
                                "entry_bar_index": i,
                                "side": "sell",
                                "entry": float(pending["entry"]),
                                "sl": float(pending["sl"]),
                                "tp": float(pending["tp"]),
                                "qty": float(pending["qty"]),
                            }
                        )
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

    return trades, equity_curve, trade_entries


def list_trade_entries(
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
) -> pd.DataFrame:
    """
    Same simulation as ``run_backtest``, but returns only rows where a pending order
    actually filled (realized entry): signal bar, fill bar, side, entry/sl/tp, qty.
    """
    _, _, entries = _simulate_walk_forward(
        data,
        start_balance=start_balance,
        lookback_bars=lookback_bars,
        pending_offset_ticks=pending_offset_ticks,
        pip_size=pip_size,
        rr=rr,
        risk_cash=risk_cash,
        pending_expiry_min=pending_expiry_min,
        entry_timeframe_minutes=entry_timeframe_minutes,
    )
    return pd.DataFrame(entries)


def list_setup_signals(
    data: pd.DataFrame,
    *,
    risk_cash: float,
    lookback_bars: int,
    pending_offset_ticks: float,
    pip_size: float,
    rr: float,
    leverage: int = 1,
) -> pd.DataFrame:
    """
    Every closed M5 bar where the model would place a new pending (bull/bear trend + swing).

    Uses the same ``compute_pending_setup`` as the backtest.
    Does not simulate position overlap, expiry, or fills — only "there is an entry signal."
    """
    data = data.copy().sort_index()
    rows: list[dict] = []
    for i in range(lookback_bars, len(data)):
        setup = compute_pending_setup(
            data,
            bar_index=i,
            lookback_bars=lookback_bars,
            pending_offset_ticks=pending_offset_ticks,
            pip_size=pip_size,
            rr=rr,
            risk_cash=risk_cash,
            leverage=leverage,
        )
        if setup is None:
            continue
        idx = data.index[i]
        trend = data.iloc[i].get("trend", "flat")
        rows.append(
            {
                "signal_bar_index": i,
                "signal_bar_time": idx,
                "trend": trend,
                "side": setup["side"],
                "entry": setup["entry"],
                "sl": setup["sl"],
                "tp": setup["tp"],
                "qty": setup["qty"],
            }
        )
    return pd.DataFrame(rows)


def list_entry_points(
    data: pd.DataFrame,
    *,
    risk_cash: float,
    start_balance: float,
    lookback_bars: int,
    pending_offset_ticks: float,
    pip_size: float,
    rr: float,
    pending_expiry_min: int,
    entry_timeframe_minutes: int = 5,
    mode: str = "filled",
) -> pd.DataFrame:
    """
    Entry points in one of two senses (``mode``):

    - ``filled`` (default): bars where a pending from a prior signal actually filled —
      same rows as ``list_trade_entries`` (walk-forward, respects flat/expiry rules).
    - ``signal``: every bar with a model pending setup — same rows as ``list_setup_signals``.
    """
    if mode == "signal":
        return list_setup_signals(
            data,
            risk_cash=risk_cash,
            lookback_bars=lookback_bars,
            pending_offset_ticks=pending_offset_ticks,
            pip_size=pip_size,
            rr=rr,
        )
    if mode == "filled":
        return list_trade_entries(
            data,
            start_balance=start_balance,
            risk_cash=risk_cash,
            lookback_bars=lookback_bars,
            pending_offset_ticks=pending_offset_ticks,
            pip_size=pip_size,
            rr=rr,
            pending_expiry_min=pending_expiry_min,
            entry_timeframe_minutes=entry_timeframe_minutes,
        )
    raise ValueError("mode must be 'filled' or 'signal'")
