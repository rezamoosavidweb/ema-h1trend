"""
Risk controls — hard limits that can only ever make the book *smaller*.

Per the Phase-5 mandate, when a limit is breached the system:
  * stops opening new positions,
  * keeps monitoring (and may close on normal exit logic),
  * emits an alert event.

Limits (fractions of equity unless noted):
  max daily loss, max weekly loss, max gross exposure, max per-position size.

These never alter the frozen strategy parameters — they are an outer safety envelope.
"""

from __future__ import annotations

from .controls import RiskDecision, RiskManager

__all__ = ["RiskManager", "RiskDecision"]
