"""
Live pairs-trading runner. Self-contained — does not share any state with
other strategies in this repo.

Public entry points (import from this package):
    PairsConfig, PortfolioSpread, load_portfolio_from_csv
    PairsExecutionEngine
    PairsRunner, assign_magic_numbers, recover_state_from_broker
    PairsStateStore, PairState
    Side, Action

The CLI lives in `mt5/run_pairs_trading.py`.
"""

from .config import (
    BROKER_TZ_NAME, COMMENT_PREFIX, MAGIC_BASE, TF_HOURS,
    PairsConfig, PortfolioSpread,
    load_portfolio_from_csv, filter_portfolio_by_keys,
    default_portfolio_csv, default_log_dir, default_state_file,
)
from .pairs_engine import PairsExecutionEngine
from .runner import PairsRunner, assign_magic_numbers, recover_state_from_broker
from .signals import Action, Side, ActionDecision, BetaFit
from .state import PairsStateStore, PairState

__all__ = [
    "BROKER_TZ_NAME", "COMMENT_PREFIX", "MAGIC_BASE", "TF_HOURS",
    "PairsConfig", "PortfolioSpread",
    "load_portfolio_from_csv", "filter_portfolio_by_keys",
    "default_portfolio_csv", "default_log_dir", "default_state_file",
    "PairsExecutionEngine",
    "PairsRunner", "assign_magic_numbers", "recover_state_from_broker",
    "Action", "Side", "ActionDecision", "BetaFit",
    "PairsStateStore", "PairState",
]
