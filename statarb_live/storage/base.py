"""Backend-agnostic storage interface used by every engine layer."""

from __future__ import annotations

import abc
from datetime import datetime
from typing import Any, Mapping, Sequence


class Storage(abc.ABC):
    """Append-mostly repository. Implementations must be safe to call from the
    single-threaded runner loop. All ``record_*`` methods persist immediately
    (the runner relies on durability across restarts)."""

    # ── lifecycle ───────────────────────────────────────────────────────────
    @abc.abstractmethod
    def init_schema(self) -> None: ...

    @abc.abstractmethod
    def close(self) -> None: ...

    # ── writes ──────────────────────────────────────────────────────────────
    @abc.abstractmethod
    def record_bars(self, rows: Sequence[Mapping[str, Any]]) -> int:
        """Upsert market bars; returns number of new rows inserted."""

    @abc.abstractmethod
    def record_signal(self, row: Mapping[str, Any]) -> int: ...

    @abc.abstractmethod
    def open_position(self, row: Mapping[str, Any]) -> int:
        """Insert an 'open' position; returns its id."""

    @abc.abstractmethod
    def close_position(self, position_id: int, updates: Mapping[str, Any]) -> None: ...

    @abc.abstractmethod
    def record_fill(self, row: Mapping[str, Any]) -> int: ...

    @abc.abstractmethod
    def record_trade(self, row: Mapping[str, Any]) -> int: ...

    @abc.abstractmethod
    def record_equity(self, row: Mapping[str, Any]) -> int: ...

    @abc.abstractmethod
    def record_metric(self, row: Mapping[str, Any]) -> int: ...

    @abc.abstractmethod
    def record_event(self, row: Mapping[str, Any]) -> int: ...

    # ── reads (for monitoring / reporting / recovery) ───────────────────────
    @abc.abstractmethod
    def open_positions(self) -> list[dict]: ...

    @abc.abstractmethod
    def trades_between(self, start: datetime, end: datetime) -> list[dict]: ...

    @abc.abstractmethod
    def equity_curve(self, start: datetime | None = None,
                     end: datetime | None = None) -> list[dict]: ...

    @abc.abstractmethod
    def fetch_df(self, table: str, where: str | None = None):
        """Return a table (optionally filtered by a raw SQL predicate) as a DataFrame."""

    @abc.abstractmethod
    def last_equity(self) -> dict | None: ...
