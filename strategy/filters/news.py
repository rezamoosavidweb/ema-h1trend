"""
News event filter — pause trading around high-impact economic releases.

WHY THIS EXISTS
---------------
High-impact news events (NFP, CPI, FOMC, Powell speeches) cause:
  1. Extreme spread widening immediately before and after the release.
  2. Sharp directional moves that instantly invalidate pending OB orders.
  3. Fake liquidity sweeps that trigger stop losses before true direction.
  4. Stop hunting by market makers exploiting the volatility.

These moves are essentially random from the perspective of a technical OB
strategy.  The correct response is to:
  - Pause new signals X minutes before the event.
  - Pause new signals Y minutes after the event.
  - Allow existing positions to run (do NOT close them — exits are random too).

DESIGN PRINCIPLE — ABSTRACT INTERFACE
--------------------------------------
We use an abstract base class so the news provider can be swapped:
  - NullNewsFilter : No filtering (default, always returns allowed).
  - StaticNewsFilter: Load a pre-defined CSV/list of events (for backtesting).
  - ForexFactoryFilter: Live fetch from Forex Factory or Econoday API (future).

The live bot uses StaticNewsFilter or NullNewsFilter depending on configuration.
The backtest uses StaticNewsFilter with a downloaded calendar CSV.

USAGE
-----
    # Null (no filter):
    filt = NullNewsFilter()

    # Static list (backtesting or pre-loaded schedule):
    events = [
        NewsEvent("NFP",  pd.Timestamp("2024-02-02 13:30", tz="UTC"), impact="high"),
        NewsEvent("CPI",  pd.Timestamp("2024-02-13 13:30", tz="UTC"), impact="high"),
    ]
    filt = StaticNewsFilter(events, minutes_before=30, minutes_after=60)

    result = filt.check(pd.Timestamp.now(tz="UTC"))
    if result.blocked:
        log.info("News block: %s", result.reason)
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import timedelta
from typing import List, Optional

import pandas as pd


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class NewsEvent:
    """
    A single economic news release.

    Fields
    ------
    name          : Human-readable name ("NFP", "CPI", "FOMC Rate Decision").
    release_time  : UTC timestamp of the scheduled release.
    impact        : "high" | "medium" | "low"  — only "high" is blocked by default.
    currency      : Affected currency ("USD", "EUR", etc.).
    """
    name         : str
    release_time : pd.Timestamp
    impact       : str = "high"      # "high" | "medium" | "low"
    currency     : str = "USD"


@dataclass
class NewsCheckResult:
    """Output of NewsFilter.check()."""
    blocked       : bool
    reason        : str           # empty string if not blocked
    nearest_event : Optional[NewsEvent] = None
    minutes_to_event: float = 0.0  # negative = event is in the past


# ── Abstract base class ───────────────────────────────────────────────────────

class NewsFilter(abc.ABC):
    """
    Abstract interface for news event filters.

    All implementations must provide check().
    """

    @abc.abstractmethod
    def check(self, current_time: pd.Timestamp) -> NewsCheckResult:
        """
        Check whether the given UTC timestamp is within a news blackout window.

        Parameters
        ----------
        current_time : Current bar close time (UTC).

        Returns
        -------
        NewsCheckResult with blocked flag and diagnostic fields.
        """
        ...

    @abc.abstractmethod
    def upcoming_events(
        self,
        current_time: pd.Timestamp,
        within_hours: float = 24.0,
    ) -> List[NewsEvent]:
        """Return list of events within the next N hours."""
        ...


# ── Null implementation (default — no blocking) ───────────────────────────────

class NullNewsFilter(NewsFilter):
    """
    Passes all timestamps without blocking.

    Use this when:
      - News filtering is not configured.
      - Running in backtesting mode without a calendar.
      - User explicitly disables news filtering.
    """

    def check(self, current_time: pd.Timestamp) -> NewsCheckResult:
        return NewsCheckResult(blocked=False, reason="")

    def upcoming_events(
        self, current_time: pd.Timestamp, within_hours: float = 24.0
    ) -> List[NewsEvent]:
        return []


# ── Static list implementation (backtesting + live with pre-loaded calendar) ──

class StaticNewsFilter(NewsFilter):
    """
    Blocks trading within [minutes_before, minutes_after] of any high-impact event.

    Parameters
    ----------
    events          : List of NewsEvent objects.
    minutes_before  : How many minutes before release to block (default 30).
    minutes_after   : How many minutes after release to block (default 60).
    blocked_impacts : Set of impact levels that trigger a block (default: {"high"}).
    blocked_currencies: Set of currencies to filter on (default: {"USD"}).
                        Gold reacts most strongly to USD data.
    """

    def __init__(
        self,
        events: List[NewsEvent],
        minutes_before: int = 30,
        minutes_after: int  = 60,
        blocked_impacts: Optional[set] = None,
        blocked_currencies: Optional[set] = None,
    ) -> None:
        self._events = sorted(events, key=lambda e: e.release_time)
        self._before = timedelta(minutes=minutes_before)
        self._after  = timedelta(minutes=minutes_after)
        self._impacts    = blocked_impacts    or {"high"}
        self._currencies = blocked_currencies or {"USD"}

    def check(self, current_time: pd.Timestamp) -> NewsCheckResult:
        # Ensure timezone-aware comparison
        if current_time.tzinfo is None:
            current_time = current_time.tz_localize("UTC")

        for event in self._events:
            release = event.release_time
            if release.tzinfo is None:
                release = release.tz_localize("UTC")

            # Skip events that don't match the filter criteria
            if event.impact not in self._impacts:
                continue
            if event.currency not in self._currencies:
                continue

            window_start = release - self._before
            window_end   = release + self._after

            if window_start <= current_time <= window_end:
                minutes_to = (release - current_time).total_seconds() / 60.0
                return NewsCheckResult(
                    blocked=True,
                    reason=(
                        f"News blackout: {event.name} ({event.impact.upper()}) "
                        f"at {release.strftime('%H:%M UTC')} "
                        f"({abs(minutes_to):.0f}min {'until' if minutes_to > 0 else 'ago'})"
                    ),
                    nearest_event=event,
                    minutes_to_event=minutes_to,
                )

        return NewsCheckResult(blocked=False, reason="")

    def upcoming_events(
        self,
        current_time: pd.Timestamp,
        within_hours: float = 24.0,
    ) -> List[NewsEvent]:
        if current_time.tzinfo is None:
            current_time = current_time.tz_localize("UTC")

        cutoff = current_time + timedelta(hours=within_hours)
        return [
            e for e in self._events
            if current_time <= e.release_time <= cutoff
            and e.impact in self._impacts
            and e.currency in self._currencies
        ]

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        minutes_before: int = 30,
        minutes_after: int  = 60,
        **kwargs,
    ) -> "StaticNewsFilter":
        """
        Load events from a CSV file with columns:
            name, release_time (ISO 8601 UTC), impact, currency

        Example row:
            NFP, 2024-02-02 13:30:00+00:00, high, USD
        """
        import csv
        events: List[NewsEvent] = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    events.append(NewsEvent(
                        name=row["name"].strip(),
                        release_time=pd.Timestamp(row["release_time"]).tz_localize("UTC")
                            if pd.Timestamp(row["release_time"]).tzinfo is None
                            else pd.Timestamp(row["release_time"]),
                        impact=row.get("impact", "high").strip().lower(),
                        currency=row.get("currency", "USD").strip().upper(),
                    ))
                except Exception:
                    continue  # Skip malformed rows
        return cls(events, minutes_before=minutes_before, minutes_after=minutes_after, **kwargs)
