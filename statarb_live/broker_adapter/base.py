"""Broker adapter interface + value objects shared by all backends."""

from __future__ import annotations

import abc
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SymbolInfo:
    """Static contract metadata needed for position sizing and validation."""
    symbol: str
    contract_size: float       # units of base ccy per 1.0 lot
    volume_min: float
    volume_max: float
    volume_step: float
    point: float               # smallest price increment
    digits: int
    tradable: bool = True


@dataclass(frozen=True)
class Tick:
    symbol: str
    ts: pd.Timestamp           # broker-tz aware
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)

    @property
    def spread_bps(self) -> float:
        if self.mid <= 0:
            return float("nan")
        return (self.ask - self.bid) / self.mid * 1e4


@dataclass(frozen=True)
class AccountInfo:
    equity: float
    balance: float
    currency: str = "USD"
    margin_free: float = 0.0
    leverage: int = 100


@dataclass(frozen=True)
class OrderResult:
    ok: bool
    ticket: int = 0
    filled_price: float = 0.0
    filled_volume: float = 0.0
    requested_price: float = 0.0
    latency_ms: float = 0.0
    comment: str = ""
    raw: dict | None = None


class BrokerAdapter(abc.ABC):
    """Minimal surface the live system depends on. Market-data methods are required;
    order methods are only used in 'live' demo mode (paper mode uses the simulator)."""

    name: str = "abstract"

    # ── connection ──────────────────────────────────────────────────────────
    @abc.abstractmethod
    def connect(self) -> bool: ...

    @abc.abstractmethod
    def disconnect(self) -> None: ...

    @abc.abstractmethod
    def healthy(self) -> bool: ...

    # ── market data ─────────────────────────────────────────────────────────
    @abc.abstractmethod
    def get_bars(self, symbol: str, timeframe: str, n_bars: int) -> pd.DataFrame:
        """Most recent `n_bars` *fully-closed* OHLC bars, broker-tz indexed.
        Columns: open, high, low, close, spread_bps (optional), volume (optional)."""

    @abc.abstractmethod
    def get_tick(self, symbol: str) -> Tick: ...

    @abc.abstractmethod
    def symbol_info(self, symbol: str) -> SymbolInfo: ...

    @abc.abstractmethod
    def list_symbols(self) -> list[str]: ...

    # ── account ─────────────────────────────────────────────────────────────
    @abc.abstractmethod
    def account(self) -> AccountInfo: ...

    # ── orders (live demo only; default = unsupported) ──────────────────────
    def market_order(self, symbol: str, side: str, volume: float, *,
                     magic: int = 0, comment: str = "") -> OrderResult:
        raise NotImplementedError(f"{self.name} adapter does not support live orders")

    def close_ticket(self, ticket: int) -> OrderResult:
        raise NotImplementedError(f"{self.name} adapter does not support live orders")

    def list_positions(self, magic: int = 0) -> list:
        """Open broker positions (optionally filtered by magic). Each item must expose
        ``ticket``, ``symbol``, ``volume`` and ``type``. Default: none (paper/sim)."""
        return []

    # ── helpers shared by backends ──────────────────────────────────────────
    def validate_symbols(self, symbols: list[str]) -> tuple[list[str], list[str]]:
        """Split requested symbols into (available, missing) against the broker."""
        available = set(self.list_symbols())
        ok = [s for s in symbols if s in available]
        missing = [s for s in symbols if s not in available]
        return ok, missing
