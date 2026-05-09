"""CSV / Parquet local-file adapter."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from scalping_system.data.sources.base import BaseDataSource
from scalping_system.utils.logging_setup import get_logger

log = get_logger(__name__)


class CSVSource(BaseDataSource):
    """
    Read OHLCV from a local CSV or Parquet file.

    File-naming convention
    ----------------------
    ``<csv_dir>/<symbol>_<timeframe>.csv``  or
    ``<csv_dir>/<symbol>_<timeframe>.parquet``

    The file must contain the canonical columns ``time``, ``Open``,
    ``High``, ``Low``, ``Close``, ``Volume``.  Symbols containing ``/``
    or ``:`` are sanitised to underscores when looking up the path.
    """

    name = "csv"

    def __init__(self, csv_dir: str | Path) -> None:
        self.csv_dir = Path(csv_dir)

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        *,
        bars: int = 5_000,
        date_from: Optional[datetime] = None,
        date_to:   Optional[datetime] = None,
    ) -> pd.DataFrame:
        sym_safe = symbol.replace("/", "_").replace(":", "_")
        candidates = [
            self.csv_dir / f"{sym_safe}_{timeframe}.parquet",
            self.csv_dir / f"{sym_safe}_{timeframe}.csv",
        ]
        for path in candidates:
            if path.exists():
                if path.suffix == ".parquet":
                    df = pd.read_parquet(path)
                else:
                    df = pd.read_csv(path, parse_dates=["time"])
                break
        else:
            raise FileNotFoundError(
                f"No cached file for {symbol} @ {timeframe} in {self.csv_dir} "
                f"(tried {[c.name for c in candidates]})"
            )

        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.sort_values("time").reset_index(drop=True)

        if date_from is not None:
            df = df[df["time"] >= pd.Timestamp(date_from, tz="UTC")]
        if date_to is not None:
            df = df[df["time"] <= pd.Timestamp(date_to, tz="UTC")]
        if bars and len(df) > bars and date_from is None and date_to is None:
            df = df.iloc[-bars:].reset_index(drop=True)

        if "symbol" not in df.columns:
            df["symbol"] = symbol
        if "timeframe" not in df.columns:
            df["timeframe"] = timeframe

        log.info("CSV loaded %d bars | %s %s", len(df), symbol, timeframe)
        return df[["time", "Open", "High", "Low", "Close", "Volume", "symbol", "timeframe"]]
