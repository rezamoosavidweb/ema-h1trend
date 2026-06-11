"""
DataFeed — pulls aligned, gap-checked price panels from a broker adapter and persists bars.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..broker_adapter.base import BrokerAdapter
from ..storage.base import Storage


@dataclass
class FeedResult:
    """Outcome of one feed pull."""
    panel: pd.DataFrame                          # aligned closes (index = bar-close, broker tz)
    spreads_bps: dict[str, float]                # latest quoted spread per symbol
    bars_persisted: int = 0
    missing_symbols: list[str] = field(default_factory=list)
    dropped_rows: int = 0                        # rows lost to inner-join (gaps)
    stale_symbols: list[str] = field(default_factory=list)
    last_ts: pd.Timestamp | None = None

    @property
    def ok(self) -> bool:
        return not self.panel.empty and not self.missing_symbols


class DataFeed:
    def __init__(self, broker: BrokerAdapter, storage: Storage, *,
                 timeframe: str = "H1", max_stale_bars: int = 3) -> None:
        self.broker = broker
        self.storage = storage
        self.timeframe = timeframe
        self.max_stale_bars = max_stale_bars

    def validate(self, symbols: list[str]) -> tuple[list[str], list[str]]:
        """Return (available, missing) for the requested symbols."""
        return self.broker.validate_symbols(symbols)

    def pull(self, symbols: list[str], n_bars: int, *, persist: bool = True) -> FeedResult:
        """Fetch ``n_bars`` closed bars for each symbol, persist them, and return an
        inner-joined close panel + diagnostics."""
        available, missing = self.validate(symbols)

        series: dict[str, pd.Series] = {}
        spreads: dict[str, float] = {}
        bar_rows: list[dict] = []
        last_per_symbol: dict[str, pd.Timestamp] = {}

        for sym in available:
            try:
                bars = self.broker.get_bars(sym, self.timeframe, n_bars)
            except Exception:
                missing.append(sym)
                continue
            if bars.empty:
                missing.append(sym)
                continue
            series[sym] = bars["close"]
            last_per_symbol[sym] = bars.index[-1]
            spreads[sym] = float(bars["spread_bps"].iloc[-1]) if "spread_bps" in bars else float("nan")
            if persist:
                for ts, row in bars.iterrows():
                    bar_rows.append({
                        "symbol": sym, "timeframe": self.timeframe, "ts": ts.to_pydatetime(),
                        "open": float(row.get("open", np.nan)),
                        "high": float(row.get("high", np.nan)),
                        "low": float(row.get("low", np.nan)),
                        "close": float(row["close"]),
                        "spread_bps": float(row["spread_bps"]) if "spread_bps" in row else None,
                        "volume": float(row["volume"]) if "volume" in row else None,
                    })

        bars_persisted = self.storage.record_bars(bar_rows) if (persist and bar_rows) else 0

        if not series:
            return FeedResult(panel=pd.DataFrame(), spreads_bps=spreads,
                              bars_persisted=bars_persisted,
                              missing_symbols=sorted(set(missing)))

        raw = pd.concat(series, axis=1, sort=True)
        panel = raw.dropna()
        dropped = len(raw) - len(panel)

        # staleness: any symbol whose last bar lags the panel's max by > max_stale_bars
        last_ts = panel.index[-1] if not panel.empty else None
        stale = []
        if last_ts is not None:
            for sym, ts in last_per_symbol.items():
                lag_bars = len(raw.loc[ts:last_ts]) - 1 if ts <= last_ts else 0
                if lag_bars > self.max_stale_bars:
                    stale.append(sym)

        return FeedResult(
            panel=panel, spreads_bps=spreads, bars_persisted=bars_persisted,
            missing_symbols=sorted(set(missing)), dropped_rows=int(dropped),
            stale_symbols=sorted(stale), last_ts=last_ts,
        )
