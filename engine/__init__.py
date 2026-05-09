"""
Data engine — historical OHLCV downloads, multi-timeframe sync, caching.

Adapters
--------
* :class:`CCXTSource`  — Binance, Bybit, OKX, Kraken, etc. via the ``ccxt``
  library.  Supports REST batch downloads with pagination.
* :class:`YahooSource` — Yahoo Finance via ``yfinance`` (forex, equities,
  metals).  Best for daily / hourly research data.
* :class:`MT5Source`   — Wraps :class:`core.MT5Connector` so the same
  ``DataEngine`` interface works for MT5-only workflows.
* :class:`CSVSource`   — Loads OHLCV CSVs cached from previous runs.

The unified entry-point :class:`DataEngine` selects an adapter based on
:class:`scalping_system.configs.DataConfig.source` and provides:

* ``fetch(symbol, timeframe, …)``        — single-symbol download
* ``fetch_multi_timeframe(symbol, [tfs])`` — list of synced frames
* ``cache_get / cache_put``               — Parquet on-disk cache
* ``validate(df)``                        — quality checks
"""

from .engine import DataEngine
from .cache import OHLCVCache
from .validators import (
    validate_ohlcv,
    detect_missing_candles,
    DataQualityReport,
)
from .sources.base import BaseDataSource
from .sources.mt5_source import MT5Source
from .sources.csv_source import CSVSource

__all__ = [
    "DataEngine",
    "OHLCVCache",
    "validate_ohlcv",
    "detect_missing_candles",
    "DataQualityReport",
    "BaseDataSource",
    "MT5Source",
    "CSVSource",
]
