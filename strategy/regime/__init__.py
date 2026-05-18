"""Market regime detection — trending vs ranging, volatility classification."""
from .market_regime import (
    MarketRegime,
    RegimeResult,
    detect_regime,
)

__all__ = [
    "MarketRegime",
    "RegimeResult",
    "detect_regime",
]
