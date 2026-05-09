"""
OHLCV on-disk cache.

Used by :class:`scalping_system.data.engine.DataEngine` to avoid
re-downloading bars across runs.  Cache keys encode the source name,
exchange, symbol, timeframe, and optional date range.

Storage backends
----------------
Parquet is preferred (fast, columnar, pandas-native) when ``pyarrow`` or
``fastparquet`` is available.  When neither is installed the cache
transparently falls back to pickle — fully lossless for tz-aware
datetimes and any pandas dtype, slightly larger on disk but never
requires an optional dependency.

The detection happens once at import time; reads probe the file
extension so caches written by older versions (or by a teammate using
the other engine) are still usable on the current machine.

File layout
-----------
::

    data_cache/
    └── ccxt_binance/
        └── BTC_USDT/
            └── 5m/
                └── 2023-01-01_2024-01-01.parquet   # or .pkl
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from utils.logging_setup import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Backend detection — once at import time
# ---------------------------------------------------------------------------

def _detect_parquet_engine() -> Optional[str]:
    """Return the available parquet engine name, or ``None`` if missing."""
    for engine in ("pyarrow", "fastparquet"):
        try:
            __import__(engine)
            return engine
        except Exception:
            continue
    return None


_PARQUET_ENGINE: Optional[str] = _detect_parquet_engine()
_DEFAULT_EXT: str = ".parquet" if _PARQUET_ENGINE else ".pkl"

if _PARQUET_ENGINE is None:
    log.warning(
        "OHLCVCache: neither pyarrow nor fastparquet found — falling back "
        "to pickle. Install pyarrow for faster, smaller cache files: "
        "`pip install pyarrow`."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAFE_RE = re.compile(r"[^A-Za-z0-9_\-\.]+")


def _safe(name: str) -> str:
    """Replace path-unsafe characters with underscores."""
    return _SAFE_RE.sub("_", name)


@dataclass
class CacheKey:
    """Composite key uniquely identifying an OHLCV slice."""
    source:    str
    exchange:  str
    symbol:    str
    timeframe: str
    date_from: Optional[str] = None
    date_to:   Optional[str] = None

    def to_path(self, root: Path, *, ext: str = _DEFAULT_EXT) -> Path:
        df_part = self.date_from or "earliest"
        dt_part = self.date_to or "latest"
        fname = f"{_safe(df_part)}_{_safe(dt_part)}{ext}"
        return (
            root
            / f"{_safe(self.source)}_{_safe(self.exchange)}"
            / _safe(self.symbol)
            / _safe(self.timeframe)
            / fname
        )


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class OHLCVCache:
    """
    On-disk OHLCV cache with pluggable backend (parquet or pickle).

    Reads/writes are atomic — partial writes are written to ``*.tmp`` and
    renamed on success.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Read
    # ------------------------------------------------------------------ #

    def get(self, key: CacheKey) -> Optional[pd.DataFrame]:
        # Probe both extensions so a cache written with one backend can be
        # read by either backend (resilient across environments).
        for ext in (_DEFAULT_EXT, ".parquet", ".pkl"):
            path = key.to_path(self.root, ext=ext)
            if not path.exists():
                continue
            try:
                if path.suffix == ".parquet":
                    df = pd.read_parquet(path)
                else:
                    df = pd.read_pickle(path)
            except Exception as exc:  # pragma: no cover
                log.warning("Cache read failed %s — %s", path, exc)
                return None
            log.debug("Cache hit | %s rows=%d", path, len(df))
            return df
        return None

    # ------------------------------------------------------------------ #
    # Write
    # ------------------------------------------------------------------ #

    def put(self, key: CacheKey, df: pd.DataFrame) -> Path:
        """
        Persist *df*, choosing the parquet backend when available and
        gracefully falling back to pickle.

        Returns the final on-disk path.
        """
        path = key.to_path(self.root, ext=_DEFAULT_EXT)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")

        wrote_parquet = False
        if _PARQUET_ENGINE is not None:
            try:
                df.to_parquet(tmp, index=False, engine=_PARQUET_ENGINE)
                wrote_parquet = True
            except Exception as exc:  # pragma: no cover
                log.warning(
                    "Parquet write failed (%s) — falling back to pickle for %s",
                    exc, path,
                )
                tmp.unlink(missing_ok=True)

        if not wrote_parquet:
            # Fallback: pickle.  Force a .pkl extension so the file
            # contents always match the suffix.
            path = path.with_suffix(".pkl")
            tmp  = path.with_suffix(path.suffix + ".tmp")
            df.to_pickle(tmp)

        tmp.replace(path)
        log.debug("Cache write | %s rows=%d", path, len(df))
        return path

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #

    def list(self) -> list[Path]:
        """Return all cached files (parquet *and* pickle)."""
        out = list(self.root.rglob("*.parquet")) + list(self.root.rglob("*.pkl"))
        return sorted(out)

    def clear(self) -> int:
        """Delete every cached file. Returns the count of removed files."""
        n = 0
        for pattern in ("*.parquet", "*.pkl"):
            for p in self.root.rglob(pattern):
                try:
                    p.unlink()
                    n += 1
                except OSError:  # pragma: no cover
                    pass
        log.info("Cleared %d cached files from %s", n, self.root)
        return n
