"""
Execution layer (paper).

Two pieces:
  * :class:`ExecutionSimulator` — the microstructure model: given an intended price and the
    quoted spread, returns a realistic *actual* fill price (you cross the spread as a taker,
    plus a small slippage), the slippage in bps, and a simulated latency. This is what lets
    the live paper run measure execution drag continuously (Phase-5 'Execution Layer').
  * :class:`PaperBook` — the position ledger: records entry/exit fills, marks open positions
    to market, and on close computes realised PnL decomposed into mean-reversion vs carry vs
    cost contributions (Phase-5 'Performance Attribution').

In 'live' mode the same intents would go to the MT5 broker adapter instead; the PaperBook
attribution maths is shared either way.
"""

from __future__ import annotations

from .simulator import ExecutionSimulator, Fill
from .book import PaperBook, OpenLeg, OpenPosition

__all__ = ["ExecutionSimulator", "Fill", "PaperBook", "OpenLeg", "OpenPosition"]
