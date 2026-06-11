"""
File-backed simulated broker.

Reads bars straight from the H1 CSV cache via the *research engine's own loaders*
(`notebooks.statarb.data`) so that data seen here is identical to the backtest — this is
what makes backtest-parity replay meaningful. The account is simulated: equity is whatever
the caller sets (the runner feeds back mark-to-market equity from the execution simulator).

Two clocks:
  * live-ish mode (``as_of=None``): returns the latest bars on disk — useful on a Linux box
    with no MT5 when you just want the pipeline to run.
  * replay mode (``as_of=<timestamp>``): returns only bars at/just-before ``as_of`` — drives
    deterministic historical replay for parity testing. Set via :meth:`set_clock`.

FX contract metadata is synthesised (standard 100k lot, 0.01 min/step) — good enough for
paper sizing; the MT5 adapter supplies true broker values in production.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from ..engine_bridge import eng_data
from .base import AccountInfo, BrokerAdapter, SymbolInfo, Tick


class SimBrokerAdapter(BrokerAdapter):
    name = "sim"

    def __init__(self, data_dir: str | Path, timeframe: str = "H1",
                 *, starting_equity: float = 100_000.0) -> None:
        self.data_dir = str(data_dir)
        self.timeframe = timeframe
        self._equity = starting_equity
        self._balance = starting_equity
        self._as_of: pd.Timestamp | None = None
        self._cache: dict[str, pd.DataFrame] = {}
        self._spread_cache: dict[str, float] = {}

    # ── connection ──────────────────────────────────────────────────────────
    def connect(self) -> bool:
        return Path(self.data_dir).exists()

    def disconnect(self) -> None:
        self._cache.clear()

    def healthy(self) -> bool:
        return Path(self.data_dir).exists()

    # ── replay clock ────────────────────────────────────────────────────────
    def set_clock(self, as_of: pd.Timestamp | None) -> None:
        self._as_of = as_of

    def set_equity(self, equity: float, balance: float | None = None) -> None:
        self._equity = equity
        if balance is not None:
            self._balance = balance

    # ── data ────────────────────────────────────────────────────────────────
    def _load(self, symbol: str) -> pd.DataFrame:
        if symbol not in self._cache:
            self._cache[symbol] = eng_data.load_ohlcv(symbol, self.timeframe, self.data_dir)
        return self._cache[symbol]

    def get_bars(self, symbol: str, timeframe: str, n_bars: int) -> pd.DataFrame:
        df = self._load(symbol)
        if self._as_of is not None:
            df = df[df.index <= self._as_of]
        out = df.iloc[-n_bars:][["open", "high", "low", "close"]].copy()
        # carry a per-bar quoted spread in bps if the source has a 'spread' column
        src = df.loc[out.index]
        if "spread" in src.columns:
            pt = eng_data.fx_point_size(symbol)
            out["spread_bps"] = (src["spread"] * pt / src["close"]) * 1e4
        if "volume" in src.columns:
            out["volume"] = src["volume"]
        return out

    def get_tick(self, symbol: str) -> Tick:
        df = self._load(symbol)
        if self._as_of is not None:
            df = df[df.index <= self._as_of]
        last = df.iloc[-1]
        mid = float(last["close"])
        half = self._median_spread_bps(symbol) / 1e4 * mid / 2.0
        return Tick(symbol=symbol, ts=df.index[-1], bid=mid - half, ask=mid + half)

    def _median_spread_bps(self, symbol: str) -> float:
        if symbol not in self._spread_cache:
            sp = eng_data.quoted_spread_bps(symbol, self.timeframe, self.data_dir)
            self._spread_cache[symbol] = 0.5 if pd.isna(sp) else float(sp)
        return self._spread_cache[symbol]

    def symbol_info(self, symbol: str) -> SymbolInfo:
        point = eng_data.fx_point_size(symbol)
        digits = 3 if symbol.endswith("JPY") else 5
        return SymbolInfo(
            symbol=symbol, contract_size=100_000.0,
            volume_min=0.01, volume_max=100.0, volume_step=0.01,
            point=point, digits=digits, tradable=True,
        )

    def list_symbols(self) -> list[str]:
        return eng_data.discover_symbols(self.data_dir, self.timeframe)

    # ── account ─────────────────────────────────────────────────────────────
    def account(self) -> AccountInfo:
        return AccountInfo(equity=self._equity, balance=self._balance, currency="USD")
