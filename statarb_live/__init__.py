"""
statarb_live — Phase 5 live paper-trading validation system.

A production-quality, modular harness that runs the *frozen* FX statistical-arbitrage
strategy selected before Phase 4 (notebook 38):

    cointegration reversion  +  carry overlay  +  continuous regime sizing

against a Forex demo account, 24/7, on a VPS — for the purpose of RESEARCH VALIDATION,
not profit maximisation. No parameter changes. No re-optimisation. No new alpha.

The strategy logic itself is NOT re-implemented here — it is imported wholesale from the
research engine (`notebooks/statarb/`) via :mod:`statarb_live.engine_bridge`, so that live
signals are bit-for-bit faithful to the backtest. This package only adds the *operational*
layers around that engine: data feed, signal/portfolio wiring, paper execution, storage,
monitoring, reporting, risk controls and deployment.

Sub-packages
------------
data_feed            market-data ingestion, gap handling, tz sync, persistence
signal_engine        wraps the research engine -> explainable target positions
portfolio_engine     volatility targeting, risk limits, sizing, exposure
execution_simulator  paper fills, latency, slippage measurement
broker_adapter       broker-agnostic interface (+ MT5 + simulated backends)
monitoring           live metrics + charts
reporting            daily / weekly / monthly HTML / CSV / PDF
storage              pluggable persistence (SQLite local, PostgreSQL on VPS)
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
