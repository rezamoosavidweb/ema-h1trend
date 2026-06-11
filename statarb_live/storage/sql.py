"""
SQLAlchemy-Core storage backend — serves SQLite (dev) and PostgreSQL (VPS) identically.

The only thing that changes between environments is the URL passed to ``create_engine``:

    sqlite:///.../statarb_live.db
    postgresql+psycopg://user:pass@host:5432/statarb

Everything else (schema, inserts, queries) is identical, which is exactly the
"pluggable: SQLite now, Postgres adapter" requirement satisfied with one class.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from sqlalchemy import create_engine, insert, select, text, update
from sqlalchemy.engine import Engine

from . import schema as S
from .base import Storage


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SqlStorage(Storage):
    def __init__(self, url: str, *, echo: bool = False) -> None:
        self.url = url
        # SQLite needs check_same_thread off only if multithreaded; the runner is
        # single-threaded, so defaults are fine. future=True for 2.0 style.
        connect_args = {}
        if url.startswith("sqlite"):
            connect_args = {"timeout": 30}
        self.engine: Engine = create_engine(
            url, echo=echo, future=True, connect_args=connect_args,
            pool_pre_ping=not url.startswith("sqlite"),
        )

    # ── lifecycle ───────────────────────────────────────────────────────────
    def init_schema(self) -> None:
        S.metadata.create_all(self.engine)
        # SQLite durability/concurrency niceties.
        if self.url.startswith("sqlite"):
            with self.engine.begin() as cx:
                cx.execute(text("PRAGMA journal_mode=WAL"))
                cx.execute(text("PRAGMA synchronous=NORMAL"))

    def close(self) -> None:
        self.engine.dispose()

    # ── generic insert helper ───────────────────────────────────────────────
    def _insert(self, table, row: Mapping[str, Any]) -> int:
        data = dict(row)
        with self.engine.begin() as cx:
            res = cx.execute(insert(table).values(**data))
            pk = res.inserted_primary_key
            return int(pk[0]) if pk and pk[0] is not None else 0

    # ── writes ──────────────────────────────────────────────────────────────
    def record_bars(self, rows: Sequence[Mapping[str, Any]]) -> int:
        if not rows:
            return 0
        now = _utcnow()
        inserted = 0
        with self.engine.begin() as cx:
            for r in rows:
                # Skip duplicates (symbol, timeframe, ts) — idempotent ingestion.
                exists = cx.execute(
                    select(S.market_bars.c.id).where(
                        S.market_bars.c.symbol == r["symbol"],
                        S.market_bars.c.timeframe == r["timeframe"],
                        S.market_bars.c.ts == r["ts"],
                    )
                ).first()
                if exists:
                    continue
                payload = dict(r)
                payload.setdefault("ingested_at", now)
                cx.execute(insert(S.market_bars).values(**payload))
                inserted += 1
        return inserted

    def record_signal(self, row: Mapping[str, Any]) -> int:
        row = {**row}
        row.setdefault("created_at", _utcnow())
        return self._insert(S.signals, row)

    def open_position(self, row: Mapping[str, Any]) -> int:
        row = {**row, "status": "open"}
        return self._insert(S.positions, row)

    def close_position(self, position_id: int, updates: Mapping[str, Any]) -> None:
        upd = {**updates, "status": "closed"}
        upd.setdefault("closed_at", _utcnow())
        with self.engine.begin() as cx:
            cx.execute(
                update(S.positions).where(S.positions.c.id == position_id).values(**upd)
            )

    def record_fill(self, row: Mapping[str, Any]) -> int:
        return self._insert(S.fills, row)

    def record_trade(self, row: Mapping[str, Any]) -> int:
        row = {**row}
        row.setdefault("created_at", _utcnow())
        return self._insert(S.trades, row)

    def record_equity(self, row: Mapping[str, Any]) -> int:
        return self._insert(S.equity, row)

    def record_metric(self, row: Mapping[str, Any]) -> int:
        row = {**row}
        row.setdefault("created_at", _utcnow())
        return self._insert(S.metrics, row)

    def record_event(self, row: Mapping[str, Any]) -> int:
        row = {**row}
        row.setdefault("ts", _utcnow())
        return self._insert(S.events, row)

    # ── reads ───────────────────────────────────────────────────────────────
    def _rows(self, stmt) -> list[dict]:
        with self.engine.connect() as cx:
            return [dict(m) for m in cx.execute(stmt).mappings().all()]

    def open_positions(self) -> list[dict]:
        return self._rows(select(S.positions).where(S.positions.c.status == "open"))

    def trades_between(self, start: datetime, end: datetime) -> list[dict]:
        stmt = (
            select(S.trades)
            .where(S.trades.c.exit_ts >= start, S.trades.c.exit_ts < end)
            .order_by(S.trades.c.exit_ts)
        )
        return self._rows(stmt)

    def equity_curve(self, start: datetime | None = None,
                     end: datetime | None = None) -> list[dict]:
        stmt = select(S.equity).order_by(S.equity.c.ts)
        if start is not None:
            stmt = stmt.where(S.equity.c.ts >= start)
        if end is not None:
            stmt = stmt.where(S.equity.c.ts < end)
        return self._rows(stmt)

    def last_equity(self) -> dict | None:
        stmt = select(S.equity).order_by(S.equity.c.ts.desc()).limit(1)
        rows = self._rows(stmt)
        return rows[0] if rows else None

    def fetch_df(self, table: str, where: str | None = None):
        import pandas as pd  # local import keeps storage importable without pandas tools
        q = f'SELECT * FROM {table}'
        if where:
            q += f' WHERE {where}'
        with self.engine.connect() as cx:
            return pd.read_sql(text(q), cx)
