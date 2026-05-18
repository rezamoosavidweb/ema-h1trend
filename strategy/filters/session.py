"""
Trading session filters — restrict signals to high-liquidity market hours.

WHY THIS EXISTS
---------------
Gold (XAUUSDT) and most crypto pairs show dramatically different behaviour
across market sessions:

  Asian session   : Low volatility, choppy, OBs fill quickly but with minimal
                    follow-through.  High false-signal rate.
  London session  : Strong directional moves as European institutions open.
                    Best for OB breakouts with displacement.
  New York session: High volume, major economic data releases, continuation
                    of London moves or sharp reversals.
  London/NY overlap: Highest liquidity in the day → cleanest OB reactions.

By restricting OB entries to London + NY hours we avoid:
  - Asian-session fake-outs with no follow-through.
  - Pre-session low-volume chop.
  - Weekend / holiday thin-market conditions.

All times are UTC.  Crypto markets are 24/7, but the institutional flow
that creates clean OB reactions follows traditional FX/equities hours.

USAGE
-----
    from strategy.filters.session import SessionFilter, SessionConfig

    cfg = SessionConfig(
        allow_london=True,
        allow_new_york=True,
        allow_overlap=True,
        allow_asia=False,
    )
    filt = SessionFilter(cfg)
    if filt.is_allowed(timestamp):
        # generate signal
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from enum import Enum
from typing import List, Optional, Set

import pandas as pd


# ── Session time windows (UTC) ────────────────────────────────────────────────
# Based on traditional FX session hours adjusted for crypto liquidity patterns.

class TradingSession(Enum):
    ASIA          = "asia"         # 00:00 – 09:00 UTC  (Tokyo/Sydney)
    LONDON        = "london"       # 07:00 – 16:00 UTC  (London open to NY open)
    NEW_YORK      = "new_york"     # 12:00 – 21:00 UTC  (NY open to NY close)
    LONDON_NY     = "london_ny"    # 12:00 – 16:00 UTC  (overlap)
    OFF_HOURS     = "off_hours"    # everything else


# Session boundaries in UTC (hour, minute)
_SESSION_RANGES = {
    TradingSession.ASIA      : (time(0,  0), time(9,  0)),
    TradingSession.LONDON    : (time(7,  0), time(16, 0)),
    TradingSession.NEW_YORK  : (time(12, 0), time(21, 0)),
    TradingSession.LONDON_NY : (time(12, 0), time(16, 0)),
}


def _time_in_range(t: time, start: time, end: time) -> bool:
    """True if time t falls within [start, end) (start inclusive, end exclusive)."""
    if start <= end:
        return start <= t < end
    # Overnight range (wraps midnight) — not used here but included for completeness
    return t >= start or t < end


def get_session(ts: pd.Timestamp) -> TradingSession:
    """
    Return the primary session label for a UTC timestamp.

    Overlap takes priority over individual sessions.
    """
    # Ensure UTC
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    elif str(ts.tzinfo) != "UTC":
        ts = ts.tz_convert("UTC")

    t = ts.time()

    # Overlap is the highest-liquidity sub-session — check first
    lo_start, lo_end = _SESSION_RANGES[TradingSession.LONDON_NY]
    if _time_in_range(t, lo_start, lo_end):
        return TradingSession.LONDON_NY

    ld_start, ld_end = _SESSION_RANGES[TradingSession.LONDON]
    if _time_in_range(t, ld_start, ld_end):
        return TradingSession.LONDON

    ny_start, ny_end = _SESSION_RANGES[TradingSession.NEW_YORK]
    if _time_in_range(t, ny_start, ny_end):
        return TradingSession.NEW_YORK

    as_start, as_end = _SESSION_RANGES[TradingSession.ASIA]
    if _time_in_range(t, as_start, as_end):
        return TradingSession.ASIA

    return TradingSession.OFF_HOURS


# ── Filter configuration ──────────────────────────────────────────────────────

@dataclass
class SessionConfig:
    """
    Controls which sessions are allowed for OB signal generation.

    Defaults are tuned for XAUUSDT M5 Order Block trading:
      - London and NY open sessions are prime time.
      - Asian session is disabled (high OB failure rate).
      - Overlap is always allowed (highest liquidity).
    """
    allow_london   : bool = True    # 07:00–12:00 UTC
    allow_new_york : bool = True    # 16:00–21:00 UTC (excl. overlap)
    allow_overlap  : bool = True    # 12:00–16:00 UTC (London + NY overlap)
    allow_asia     : bool = False   # 00:00–09:00 UTC — disabled by default
    allow_off_hours: bool = False   # everything else — disabled by default

    def allowed_sessions(self) -> Set[TradingSession]:
        """Return set of TradingSession values that are currently allowed."""
        result: Set[TradingSession] = set()
        if self.allow_london:
            result.add(TradingSession.LONDON)
        if self.allow_new_york:
            result.add(TradingSession.NEW_YORK)
        if self.allow_overlap:
            result.add(TradingSession.LONDON_NY)
        if self.allow_asia:
            result.add(TradingSession.ASIA)
        if self.allow_off_hours:
            result.add(TradingSession.OFF_HOURS)
        return result


class SessionFilter:
    """
    Stateless filter: tells callers whether a given UTC timestamp is tradeable.

    Usage:
        cfg  = SessionConfig(allow_london=True, allow_new_york=True)
        filt = SessionFilter(cfg)
        if filt.is_allowed(bar_timestamp):
            ...
    """

    def __init__(self, config: Optional[SessionConfig] = None) -> None:
        self._config = config or SessionConfig()

    def is_allowed(self, ts: pd.Timestamp) -> bool:
        """True if the bar timestamp falls within an allowed session."""
        session = get_session(ts)
        return session in self._config.allowed_sessions()

    def session_of(self, ts: pd.Timestamp) -> TradingSession:
        """Return the session name for a timestamp (for logging / analytics)."""
        return get_session(ts)


# ── Convenience free function ─────────────────────────────────────────────────

def is_in_session(
    ts: pd.Timestamp,
    allow_london: bool   = True,
    allow_new_york: bool = True,
    allow_overlap: bool  = True,
    allow_asia: bool     = False,
) -> bool:
    """
    Quick boolean check without constructing SessionFilter / SessionConfig.

    Useful inside backtest loops where object construction overhead matters.
    """
    cfg  = SessionConfig(
        allow_london=allow_london,
        allow_new_york=allow_new_york,
        allow_overlap=allow_overlap,
        allow_asia=allow_asia,
    )
    filt = SessionFilter(cfg)
    return filt.is_allowed(ts)


# ── DataFrame annotation (for notebooks) ─────────────────────────────────────

def add_session_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'session' column with session name string for each bar."""
    df = df.copy()
    df["session"] = df.index.map(lambda ts: get_session(ts).value)
    return df
