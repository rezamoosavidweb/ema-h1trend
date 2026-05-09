"""MT5 adapter — wraps :class:`core.MT5Connector` + :class:`core.DataFetcher`."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from scalping_system.data.sources.base import BaseDataSource
from scalping_system.utils.logging_setup import get_logger

log = get_logger(__name__)


class MT5Source(BaseDataSource):
    """
    MetaTrader 5 OHLCV adapter.

    Re-uses the project's ``core.MT5Connector`` so every notebook /
    process shares one connection.  When *connector* is ``None`` a fresh
    one is opened via the ``with`` protocol on each ``fetch()`` call.

    The MT5 broker often appends a suffix (``.i``, ``-cash``, …) to the
    symbol — pass ``add_suffix=True`` (default) and configure the suffix
    in :class:`config.settings.MT5Config`.
    """

    name = "mt5"

    def __init__(self, *, add_suffix: bool = True) -> None:
        try:
            from core.mt5_connector import MT5Connector
            from core.data_fetcher  import DataFetcher
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "MT5 modules not importable from `core`. Ensure the project "
                "is on sys.path."
            ) from e

        self._MT5Connector = MT5Connector
        self._DataFetcher  = DataFetcher
        self._add_suffix   = add_suffix

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        *,
        bars: int = 5_000,
        date_from: Optional[datetime] = None,
        date_to:   Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Download bars from MT5 and return a canonical DataFrame."""
        with self._MT5Connector() as conn:
            fetcher = self._DataFetcher(conn)
            if date_from is not None and date_to is not None:
                df = fetcher.fetch_range(
                    symbol=symbol, timeframe=timeframe,
                    date_from=date_from, date_to=date_to,
                    add_suffix=self._add_suffix,
                )
            else:
                df = fetcher.fetch(
                    symbol=symbol, timeframe=timeframe,
                    bars=bars, add_suffix=self._add_suffix,
                )

        df["time"] = pd.to_datetime(df["time"], utc=True)
        if "symbol" not in df.columns:
            df["symbol"] = symbol
        if "timeframe" not in df.columns:
            df["timeframe"] = timeframe

        return df[["time", "Open", "High", "Low", "Close", "Volume", "symbol", "timeframe"]]
