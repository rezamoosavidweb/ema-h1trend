"""
Pluggable persistence layer.

Design: one SQLAlchemy-Core backend (:class:`~statarb_live.storage.sql.SqlStorage`)
serves *both* dev (SQLite) and production (PostgreSQL) — the only difference is the
connection URL (``sqlite:///...`` vs ``postgresql+psycopg://...``). The abstract
:class:`~statarb_live.storage.base.Storage` interface keeps call-sites backend-agnostic
so a non-SQL backend could be slotted in later without touching the engines.

Every persisted event is reproducible: market bars, signals, positions, fills, trades,
equity snapshots, performance metrics, and an append-only event/audit log.
"""

from __future__ import annotations

from .base import Storage
from .factory import create_storage

__all__ = ["Storage", "create_storage"]
