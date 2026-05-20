"""
Live MT5 execution layer for the Order Block strategy.

Public entry points:
    ExecutionEngine     -- top-level "place this signal" orchestrator
    SymbolConfig        -- centralized symbol normalization
    StructuredLogger    -- JSON-lines event log (logs/<symbol>.json)

The strategy/signal logic stays in mt5/run_ob_xauusd.py untouched; everything
here is the *execution* layer: validation, order construction, pending-order
lifecycle, fallback, retries, and observability.

Module map:
    symbol_config       Canonical symbol resolution + caching
    structured_logger   JSON event logger with consistent fields
    broker_validator    Pre-flight checks (trade mode, stops/freeze, spread, price)
    order_factory       Build mt5 order_send request dicts + price snapping
    pending_manager     Track / cancel / dedupe pending orders (GTC lifecycle)
    risk_adapter        Lot sizing from balance/risk percentage
    fallback_engine     LIMIT -> MARKET cascade keyed by broker retcode
    execution_engine    Public facade tying it all together
"""

from .symbol_config import SymbolConfig, resolve_symbol
from .structured_logger import StructuredLogger
from .broker_validator import BrokerValidator, ValidationResult
from .pending_manager import PendingOrderManager
from .fallback_engine import FallbackEngine, FallbackResult
from .execution_engine import ExecutionEngine, ExecutionOutcome
from .order_factory import OrderFactory
from .risk_adapter import RiskAdapter

__all__ = [
    "SymbolConfig",
    "resolve_symbol",
    "StructuredLogger",
    "BrokerValidator",
    "ValidationResult",
    "PendingOrderManager",
    "FallbackEngine",
    "FallbackResult",
    "ExecutionEngine",
    "ExecutionOutcome",
    "OrderFactory",
    "RiskAdapter",
]
