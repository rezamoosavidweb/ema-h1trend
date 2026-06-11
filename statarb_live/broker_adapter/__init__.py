"""
Broker-agnostic adapter layer.

The rest of the system never imports MetaTrader5 directly — it talks to a
:class:`~statarb_live.broker_adapter.base.BrokerAdapter`. Two backends ship:

  * :class:`~statarb_live.broker_adapter.mt5_adapter.MT5BrokerAdapter` — real demo
    account (Windows VPS): bars, ticks, symbol metadata, account equity, and (for a
    future 'live' demo mode) order placement.
  * :class:`~statarb_live.broker_adapter.sim_adapter.SimBrokerAdapter` — file-backed
    bars from the H1 CSV cache + a simulated account. Lets the whole pipeline run on a
    box with no MT5 (e.g. Linux dev) and underpins backtest-parity replay.

Paper-trading mode uses the adapter only for *market data + symbol metadata + equity*;
order intents are routed to the ExecutionSimulator, not to the broker.
"""

from __future__ import annotations

from .base import AccountInfo, BrokerAdapter, OrderResult, SymbolInfo, Tick
from .factory import create_broker

__all__ = [
    "BrokerAdapter", "AccountInfo", "SymbolInfo", "Tick", "OrderResult",
    "create_broker",
]
