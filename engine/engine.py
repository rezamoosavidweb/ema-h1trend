"""
Unified data engine — single facade over every adapter.

Typical usage::

    from scalping_system.configs import default_config
    from scalping_system.data import DataEngine

    cfg = default_config()
    engine = DataEngine.from_config(cfg.data)

    df_5m = engine.fetch("BTC/USDT", "5m", bars=20_000)
    df_1m = engine.fetch("BTC/USDT", "1m", bars=100_000)

    panel = engine.fetch_multi_timeframe("BTC/USDT", ["1m", "5m", "1h"])
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from engine import DataConfig
from engine.cache import CacheKey, OHLCVCache
from engine.sources.base import BaseDataSource

from engine.sources.csv_source import CSVSource
from engine.sources.mt5_source import MT5Source

from engine.validators import DataQualityReport, validate_ohlcv
from utils.logging_setup import get_logger

log = get_logger(__name__)


class DataEngine:
    """Facade over the configured :class:`BaseDataSource`."""

    def __init__(
        self,
        source: BaseDataSource,
        cache: Optional[OHLCVCache] = None,
        *,
        use_cache: bool = True,
        min_bars: int = 500,
        source_name: Optional[str] = None,
        exchange_name: Optional[str] = None,
    ) -> None:
        self.source        = source
        self.cache         = cache
        self.use_cache     = use_cache and cache is not None
        self.min_bars      = min_bars
        self.source_name   = source_name   or getattr(source, "name", "unknown")
        self.exchange_name = exchange_name or getattr(source, "exchange_id",
                                                      getattr(source, "name", "unknown"))

    # ------------------------------------------------------------------ #
    # Factory
    # ------------------------------------------------------------------ #

    @classmethod
    def from_config(cls, cfg: DataConfig) -> "DataEngine":
        """Build a :class:`DataEngine` from a :class:`DataConfig`."""
        if cfg.source == "ccxt":
            api_key    = os.getenv(f"{cfg.exchange.upper()}_API_KEY") or None
            api_secret = os.getenv(f"{cfg.exchange.upper()}_API_SECRET") or None
            sandbox    = os.getenv(f"{cfg.exchange.upper()}_TESTNET", "false").lower() == "true"
            src: BaseDataSource = CCXTSource(
                exchange=cfg.exchange,
                api_key=api_key,
                api_secret=api_secret,
                market_type=cfg.market_type,
                sandbox=sandbox,
            )
            exchange_name = cfg.exchange
        elif cfg.source == "yahoo":
            src = YahooSource()
            exchange_name = "yahoo"
        elif cfg.source == "mt5":
            src = MT5Source()
            exchange_name = "mt5"
        elif cfg.source == "csv":
            src = CSVSource(cfg.csv_dir)
            exchange_name = "csv"
        else:  # pragma: no cover
            raise ValueError(f"Unknown data source: {cfg.source}")

        cache = OHLCVCache(cfg.cache_dir) if cfg.use_cache else None
        return cls(
            source=src, cache=cache,
            use_cache=cfg.use_cache,
            min_bars=cfg.min_bars,
            source_name=cfg.source,
            exchange_name=exchange_name,
        )

    # ------------------------------------------------------------------ #
    # Single-frame fetch
    # ------------------------------------------------------------------ #

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        *,
        bars: int = 5_000,
        date_from: Optional[datetime] = None,
        date_to:   Optional[datetime] = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV with cache fall-back.

        Parameters
        ----------
        force_refresh:
            When ``True``, bypass the cache and always re-download.  The
            new data is still written to the cache afterwards.
        """
        key = CacheKey(
            source=self.source_name,
            exchange=self.exchange_name,
            symbol=symbol,
            timeframe=timeframe,
            date_from=date_from.isoformat() if date_from else None,
            date_to=date_to.isoformat()     if date_to else None,
        )

        if self.use_cache and not force_refresh and self.cache is not None:
            cached = self.cache.get(key)
            if cached is not None and len(cached) >= self.min_bars:
                if not date_from and not date_to and bars and len(cached) > bars:
                    cached = cached.iloc[-bars:].reset_index(drop=True)
                return cached

        df = self.source.fetch(
            symbol=symbol, timeframe=timeframe, bars=bars,
            date_from=date_from, date_to=date_to,
        )

        if self.cache is not None and self.use_cache:
            self.cache.put(key, df)

        return df

    # ------------------------------------------------------------------ #
    # Multi-timeframe sync
    # ------------------------------------------------------------------ #

    def fetch_multi_timeframe(
        self,
        symbol: str,
        timeframes: List[str],
        *,
        bars: int = 5_000,
        date_from: Optional[datetime] = None,
        date_to:   Optional[datetime] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Return ``{timeframe → DataFrame}`` for the requested set."""
        out: Dict[str, pd.DataFrame] = {}
        for tf in timeframes:
            try:
                out[tf] = self.fetch(
                    symbol, tf, bars=bars,
                    date_from=date_from, date_to=date_to,
                )
            except Exception as exc:
                log.error("Failed to fetch %s @ %s — %s", symbol, tf, exc)
        return out

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate(self, df: pd.DataFrame, timeframe: str | None = None) -> DataQualityReport:
        """Run quality checks on *df* and return a structured report."""
        return validate_ohlcv(df, timeframe=timeframe)
