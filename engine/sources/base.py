"""Base data-source interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import pandas as pd


class BaseDataSource(ABC):
    """
    Abstract OHLCV download adapter.

    Subclasses must return a DataFrame with the canonical schema:
    ``time`` (UTC-aware), ``Open``, ``High``, ``Low``, ``Close``, ``Volume``.
    """

    name: str = "base"

    @abstractmethod
    def fetch(
        self,
        symbol: str,
        timeframe: str,
        *,
        bars: int = 5_000,
        date_from: Optional[datetime] = None,
        date_to:   Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Download OHLCV data and return a canonical DataFrame.

        The implementation must:

        * apply paging when the requested range exceeds API single-call limits
        * convert exchange-native timestamps to UTC datetimes
        * attach ``symbol`` and ``timeframe`` meta-columns
        """
