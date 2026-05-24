"""
Multi-symbol scalper bot package.

Live execution of the strategy researched in
`notebooks/24_multi_symbol_scalper.ipynb`.

Public surface:
    Strategy        — pure signal detector (data → signal dict | None)
    SymbolBasket    — load per-symbol configs from notebooks/results
    CapitalAllocator— split account balance across the basket

The runner script `mt5/run_multi_scalper.py` wires these together with the
existing `execution.ExecutionEngine` (one engine per symbol).
"""

from .strategy import Strategy, StrategyConfig, Signal
from .config import SymbolBasket, SymbolStrategyConfig, load_basket
from .allocator import CapitalAllocator, SymbolAllocation

__all__ = [
    "Strategy",
    "StrategyConfig",
    "Signal",
    "SymbolBasket",
    "SymbolStrategyConfig",
    "load_basket",
    "CapitalAllocator",
    "SymbolAllocation",
]
