"""Storage factory — picks a backend from the resolved DB URL."""

from __future__ import annotations

from ..config import SystemConfig
from .base import Storage
from .sql import SqlStorage


def create_storage(config: SystemConfig, *, init: bool = True) -> Storage:
    """Build the storage backend for the given config.

    The URL alone determines the engine (sqlite vs postgresql), so one
    :class:`SqlStorage` class covers both the dev box and the VPS.
    """
    url = config.resolved_db_url()
    store = SqlStorage(url)
    if init:
        store.init_schema()
    return store
