"""
Database schema (SQLAlchemy Core) — portable across SQLite and PostgreSQL.

We use Core (not the ORM) for a small, explicit, reproducible schema. Types are chosen
to work identically on both engines: ``JSON`` (native JSONB on PG, TEXT-backed JSON on
SQLite), ``DateTime(timezone=True)``, ``Float``, ``String``.

Tables
------
market_bars   one fully-closed OHLC bar per (symbol, ts) with spread + volume
signals       every signal the engine emits, fully explainable (z, beta, carry, regime…)
positions     position lifecycle (one row per pair-position, updated on close)
fills         intended vs actual execution with slippage + latency
trades        closed round-trips with realised PnL + all research-logging fields
equity        per-cycle account/equity + exposure snapshot
metrics       rolled-up performance metrics (daily/weekly/monthly windows)
events        append-only audit/event log (bot_start, cycle_*, risk_breach, …)
"""

from __future__ import annotations

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    Index,
)

metadata = MetaData()


market_bars = Table(
    "market_bars", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("symbol", String(32), nullable=False),
    Column("timeframe", String(8), nullable=False),
    Column("ts", DateTime(timezone=True), nullable=False),   # bar-close, broker tz
    Column("open", Float), Column("high", Float),
    Column("low", Float), Column("close", Float, nullable=False),
    Column("spread_bps", Float),                              # quoted spread at close
    Column("volume", Float),
    Column("ingested_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("symbol", "timeframe", "ts", name="uq_bar"),
    Index("ix_bars_sym_ts", "symbol", "ts"),
)


signals = Table(
    "signals", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("cycle_id", String(40), nullable=False),
    Column("signal_ts", DateTime(timezone=True), nullable=False),   # decision bar-close
    Column("pair_key", String(40), nullable=False),                  # "EURUSD~GBPUSD"
    Column("y_symbol", String(32)), Column("x_symbol", String(32)),
    # --- explainability / research-logging fields ---
    Column("zscore", Float),
    Column("hedge_ratio", Float),          # beta
    Column("alpha", Float),
    Column("spread_value", Float),
    Column("half_life_bars", Float),
    Column("adf_p", Float),                # cointegration p in current train window
    Column("carry_value", Float),          # carry contribution / annualised rate diff
    Column("regime_state", String(16)),    # 'calm' | 'crisis' | numeric label
    Column("regime_prob_calm", Float),
    Column("regime_multiplier", Float),
    Column("raw_target", Float),           # reversion target in {-1,0,+1}
    Column("vol_leverage", Float),         # vol-target leverage applied
    Column("target_position", Float),      # final signed gross exposure (fraction of capital)
    Column("expected_spread_move", Float), # expected reversion (z -> 0) in spread units
    Column("confidence", Float),           # 0..1
    Column("action", String(24)),          # hold/open_long/open_short/close/scale
    Column("provenance", JSON),            # frozen strategy params snapshot
    Column("created_at", DateTime(timezone=True), nullable=False),
    Index("ix_sig_cycle", "cycle_id"),
    Index("ix_sig_pair_ts", "pair_key", "signal_ts"),
)


positions = Table(
    "positions", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("pair_key", String(40), nullable=False),
    Column("y_symbol", String(32)), Column("x_symbol", String(32)),
    Column("side", String(8)),             # 'long' | 'short' (of the spread)
    Column("status", String(12), nullable=False),   # 'open' | 'closed'
    Column("opened_cycle", String(40)),
    Column("opened_at", DateTime(timezone=True)),
    Column("closed_at", DateTime(timezone=True)),
    Column("y_volume", Float), Column("x_volume", Float),
    Column("beta_at_open", Float), Column("alpha_at_open", Float),
    Column("z_at_open", Float), Column("z_at_close", Float),
    Column("regime_at_open", String(16)),
    Column("gross_notional", Float),
    Column("realized_pnl", Float),
    Column("meta", JSON),
    Index("ix_pos_status", "status"),
    Index("ix_pos_pair", "pair_key"),
)


fills = Table(
    "fills", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("position_id", Integer),
    Column("pair_key", String(40)),
    Column("leg", String(8)),              # 'y' | 'x'
    Column("symbol", String(32)),
    Column("kind", String(8)),             # 'entry' | 'exit'
    Column("side", String(8)),             # 'buy' | 'sell'
    Column("volume", Float),
    Column("intended_price", Float),
    Column("actual_price", Float),
    Column("slippage_bps", Float),
    Column("spread_bps", Float),
    Column("latency_ms", Float),
    Column("signal_ts", DateTime(timezone=True)),
    Column("fill_ts", DateTime(timezone=True)),       # paper (simulated) fill wall-clock
    # --- real broker execution (populated only when live_orders is on) ---
    Column("exec_mode", String(8)),                   # 'paper' | 'live'
    Column("broker_ticket", Integer),                 # MT5 order/deal ticket
    Column("broker_fill_price", Float),               # actual broker fill price
    Column("broker_fill_ts", DateTime(timezone=True)),# when the real order completed
    Column("broker_latency_ms", Float),               # round-trip order_send latency
    Column("broker_ok", Boolean),                     # True if the real order filled
    Column("broker_comment", String(64)),             # retcode / error if it didn't
    Column("meta", JSON),
    Index("ix_fill_pos", "position_id"),
)


trades = Table(
    "trades", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("position_id", Integer),
    Column("pair_key", String(40), nullable=False),
    Column("y_symbol", String(32)), Column("x_symbol", String(32)),
    Column("side", String(8)),
    # --- research logging (the critical fields from the Phase-5 mandate) ---
    Column("signal_ts", DateTime(timezone=True)),
    Column("entry_ts", DateTime(timezone=True)),
    Column("exit_ts", DateTime(timezone=True)),
    Column("regime", String(16)),
    Column("carry_value", Float),
    Column("zscore_entry", Float),
    Column("zscore_exit", Float),
    Column("hedge_ratio", Float),
    Column("position_size", Float),
    Column("gross_notional", Float),
    Column("entry_slippage_bps", Float),
    Column("exit_slippage_bps", Float),
    Column("realized_pnl", Float),
    Column("reversion_pnl", Float),        # PnL attributable to mean reversion
    Column("carry_pnl", Float),            # PnL attributable to carry
    Column("cost_pnl", Float),             # frictions
    Column("bars_held", Integer),
    Column("exit_reason", String(24)),     # 'z_exit'|'z_stop'|'time_stop'|'regime'|'risk'
    Column("meta", JSON),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Index("ix_trade_pair", "pair_key"),
    Index("ix_trade_regime", "regime"),
)


equity = Table(
    "equity", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("cycle_id", String(40)),
    Column("ts", DateTime(timezone=True), nullable=False),
    Column("equity", Float, nullable=False),
    Column("cash", Float),
    Column("gross_exposure", Float),
    Column("net_exposure", Float),
    Column("leverage", Float),
    Column("open_pairs", Integer),
    Column("daily_pnl", Float),
    Column("regime_state", String(16)),
    Column("regime_multiplier", Float),
    Column("pair_contributions", JSON),    # {pair_key: pnl_contrib}
    UniqueConstraint("ts", name="uq_equity_ts"),
    Index("ix_equity_ts", "ts"),
)


metrics = Table(
    "metrics", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("window", String(12), nullable=False),     # 'daily'|'weekly'|'monthly'|'all'
    Column("period_start", DateTime(timezone=True)),
    Column("period_end", DateTime(timezone=True)),
    Column("pnl", Float),
    Column("return_pct", Float),
    Column("sharpe", Float),
    Column("sortino", Float),
    Column("max_drawdown", Float),
    Column("win_rate", Float),
    Column("turnover", Float),
    Column("n_trades", Integer),
    Column("extra", JSON),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Index("ix_metrics_window", "window", "period_end"),
)


events = Table(
    "events", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ts", DateTime(timezone=True), nullable=False),
    Column("event_type", String(40), nullable=False),
    Column("cycle_id", String(40)),
    Column("severity", String(12)),       # 'info'|'warning'|'error'|'critical'
    Column("message", String(512)),
    Column("payload", JSON),
    Index("ix_events_type", "event_type"),
    Index("ix_events_ts", "ts"),
)


ALL_TABLES = [
    market_bars, signals, positions, fills, trades, equity, metrics, events,
]
