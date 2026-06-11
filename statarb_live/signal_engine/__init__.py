"""
Signal engine — the *only* place strategy logic enters the live system, and it does so by
calling the research engine, never by re-implementing it.

Pipeline per cycle (one fully-closed H1 bar):

    price panel ─┬─> reversion sleeve  (frozen cointegration pairs: z-score -> target -> vol size)
                 ├─> carry sleeve      (cross-sectional carry weights)
                 └─> regime multiplier (continuous HMM calm-probability scaling)
                              │
                              ▼
                        CycleSignals  (fully explainable; every field logged to storage)

Faithfulness: the reversion target reuses ``static_spread`` -> ``rolling_zscore`` ->
``zscore_positions`` and the exact vol-targeting from ``backtest.backtest_pair``; the carry
weights reuse ``carry.carry_signal``; the regime multiplier is ``regime.regime_size_multiplier``.
Same frozen parameters as NB38. We take the **last** (latest closed bar) value of each — that is
the live equivalent of the backtest's per-bar decision.
"""

from __future__ import annotations

from .types import CarrySignal, CycleSignals, PairSignal, RegimeState
from .universe import Universe, load_or_select_universe
from .engine import SignalEngine

__all__ = [
    "SignalEngine", "Universe", "load_or_select_universe",
    "CycleSignals", "PairSignal", "CarrySignal", "RegimeState",
]
