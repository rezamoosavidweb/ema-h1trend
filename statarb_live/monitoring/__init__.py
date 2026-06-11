"""
Monitoring — turn the raw storage tables into the performance metrics and charts the
Phase-5 mandate tracks: daily / weekly / monthly PnL, drawdown, win rate, turnover, regime
statistics, and performance attribution (by pair, regime, carry vs mean-reversion).

Pure read-side: nothing here trades or mutates state — it reads the equity / trades / signals
tables and produces dataclasses + chart files consumed by the reporting layer and any
external dashboard.
"""

from __future__ import annotations

from .metrics import (
    PerformanceMetrics, attribution_by_pair, attribution_by_regime,
    attribution_by_sleeve, compute_metrics, equity_dataframe, trades_dataframe,
)
from .charts import write_charts

__all__ = [
    "PerformanceMetrics", "compute_metrics", "equity_dataframe", "trades_dataframe",
    "attribution_by_pair", "attribution_by_regime", "attribution_by_sleeve", "write_charts",
]
