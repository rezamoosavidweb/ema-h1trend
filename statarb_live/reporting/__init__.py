"""
Reporting — automatic daily / weekly / monthly / final research reports.

Each report bundles: equity curve, drawdown, exposure & regime charts, headline metrics,
and performance attribution (by pair, regime, and reversion-vs-carry sleeve). The 'final'
report adds the Phase-5 success-criteria comparison: backtest vs walk-forward vs live
paper-trading, quantifying performance decay and execution/slippage impact.

Outputs (written under ``<report_dir>/<period>_<stamp>/``):
  * report.html  — self-contained HTML (charts embedded as files alongside)
  * *.csv        — metrics + attribution tables for downstream analysis
  * report.pdf   — if WeasyPrint is installed (optional; HTML always produced)
"""

from __future__ import annotations

from .report import generate_report

__all__ = ["generate_report"]
