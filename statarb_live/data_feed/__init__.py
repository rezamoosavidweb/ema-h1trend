"""
Data feed — the boundary between the broker and the strategy.

Responsibilities (Phase-5 'Data Feed' mandate):
  * automatic market-data updates — pull the latest fully-closed bars each cycle;
  * missing-data handling — detect gaps, inner-join to an aligned panel, flag staleness;
  * time synchronisation — all timestamps land in broker tz (Europe/Nicosia), matching the
    research cache (see project memory 'Data timezone is Nicosia not UTC');
  * symbol validation — verify the universe's symbols exist on the broker before trading;
  * persistence — every incoming bar (ts/OHLC/spread/volume) is written to storage.
"""

from __future__ import annotations

from .feed import DataFeed, FeedResult

__all__ = ["DataFeed", "FeedResult"]
