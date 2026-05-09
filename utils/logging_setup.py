"""
Centralised logging configuration for the scalping system.

Wraps the project-level :func:`core.logger.get_logger` and adds an
``configure_logging()`` helper used by CLI entry-points to set verbosity
and optionally enable JSON output for log-aggregation backends.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from core.logger import get_logger as _get_core_logger


_CONFIGURED: bool = False


def get_logger(name: str) -> logging.Logger:
    """Return a project-configured logger by name."""
    return _get_core_logger(name)


def configure_logging(
    level: str = "INFO",
    *,
    log_file: Optional[Path] = None,
    json_output: bool = False,
) -> None:
    """
    One-shot logging setup invoked by CLI / live entry-points.

    The project's ``core.logger.get_logger`` already wires per-module
    rotating-file handlers; this function reconfigures the *root* logger
    so third-party libraries (ccxt, websockets, asyncio) emit at the same
    level and through the same formatter.

    Parameters
    ----------
    level:
        ``"DEBUG"`` | ``"INFO"`` | ``"WARNING"`` | ``"ERROR"`` | ``"CRITICAL"``.
    log_file:
        Optional path to mirror logs to a dedicated file (in addition to
        the rotating-file handlers attached by ``core.logger``).
    json_output:
        When ``True`` the formatter emits structured JSON lines suitable
        for ingestion by Loki/ELK/Datadog.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    numeric = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(numeric)

    if json_output:
        fmt = (
            '{"ts":"%(asctime)s","lvl":"%(levelname)s",'
            '"logger":"%(name)s","msg":"%(message)s"}'
        )
    else:
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(numeric)
        sh.setFormatter(formatter)
        root.addHandler(sh)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(numeric)
        fh.setFormatter(formatter)
        root.addHandler(fh)

    _CONFIGURED = True
