"""
Spread filter — reject entries when bid-ask spread is too wide.

WHY THIS EXISTS
---------------
During:
  - News events (NFP, CPI, FOMC)
  - Session gaps (Friday close, Sunday open)
  - Low-liquidity periods (Asian session, holidays)
  - Exchange maintenance / connectivity issues

... spreads can spike to 10–100× their normal level.  Entering an OB trade
into a wide spread means you are instantly losing money on the entry fill.
A 1-pip spread on XAUUSDT is negligible; a 30-pip spread eats your entire
risk budget before the trade even starts.

THRESHOLD GUIDANCE (XAUUSDT)
------------------------------
  Normal spread  : 0.3–1.0 pip (0.03–0.10 USD)
  Caution spread : 1.0–3.0 pip
  Wide spread    : > 3.0 pip → skip trade
  Extreme spread : > 10 pip → possible connectivity issue

The filter provides both absolute (USD) and relative (fraction of SL distance)
thresholds so it adapts to different symbols.

USAGE
-----
    filt   = SpreadFilter(max_spread_abs=3.0, max_spread_fraction=0.15)
    result = filt.check(bid=2650.10, ask=2650.45, sl_distance=20.0)
    if result.allowed:
        place_order(...)
    else:
        log.info("Trade skipped: %s", result.reason)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SpreadResult:
    """Output of SpreadFilter.check()."""
    allowed       : bool
    spread_abs    : float   # absolute bid-ask spread in price units
    spread_pct    : float   # spread as % of mid-price
    spread_frac   : float   # spread as fraction of sl_distance (0 if sl_distance=0)
    reason        : str     # empty string if allowed


class SpreadFilter:
    """
    Validates that the current bid-ask spread is within acceptable limits.

    Two independent thresholds (both must pass for allowed=True):
    1. max_spread_abs    : Hard cap on absolute spread (e.g. 3.0 USD for XAUUSDT).
    2. max_spread_fraction: Spread must be <= this fraction of the SL distance.
                            Prevents entries where spread > meaningful portion of risk.

    Parameters
    ----------
    max_spread_abs      : Maximum allowed absolute spread.  None = no limit.
    max_spread_fraction : Maximum allowed spread / sl_distance.  None = no limit.
    """

    def __init__(
        self,
        max_spread_abs: Optional[float] = None,
        max_spread_fraction: Optional[float] = None,
    ) -> None:
        self._max_abs  = max_spread_abs
        self._max_frac = max_spread_fraction

    def check(
        self,
        bid: float,
        ask: float,
        sl_distance: float = 0.0,
    ) -> SpreadResult:
        """
        Check whether the current spread is acceptable for an OB entry.

        Parameters
        ----------
        bid         : Current best bid price.
        ask         : Current best ask price.
        sl_distance : Absolute distance between entry and SL (price units).
                      Used only for the fraction check.

        Returns
        -------
        SpreadResult with allowed flag and diagnostic fields.
        """
        spread_abs = ask - bid
        mid        = (bid + ask) / 2.0
        spread_pct = (spread_abs / mid * 100.0) if mid > 0 else 0.0
        spread_frac = (spread_abs / sl_distance) if sl_distance > 0 else 0.0

        # ── Absolute threshold ────────────────────────────────────────────────
        if self._max_abs is not None and spread_abs > self._max_abs:
            return SpreadResult(
                allowed=False,
                spread_abs=spread_abs,
                spread_pct=spread_pct,
                spread_frac=spread_frac,
                reason=(
                    f"Spread {spread_abs:.4f} exceeds max_spread_abs={self._max_abs:.4f}"
                ),
            )

        # ── Fraction of SL threshold ──────────────────────────────────────────
        if (
            self._max_frac is not None
            and sl_distance > 0
            and spread_frac > self._max_frac
        ):
            return SpreadResult(
                allowed=False,
                spread_abs=spread_abs,
                spread_pct=spread_pct,
                spread_frac=spread_frac,
                reason=(
                    f"Spread/SL {spread_frac:.3f} exceeds max_fraction={self._max_frac:.3f} "
                    f"(spread={spread_abs:.4f} sl_dist={sl_distance:.4f})"
                ),
            )

        return SpreadResult(
            allowed=True,
            spread_abs=spread_abs,
            spread_pct=spread_pct,
            spread_frac=spread_frac,
            reason="",
        )

    @classmethod
    def from_thresholds(
        cls,
        max_abs: Optional[float] = 3.0,
        max_fraction: Optional[float] = 0.20,
    ) -> "SpreadFilter":
        """
        Factory: sensible defaults for XAUUSDT M5 OB strategy.

        max_abs=3.0   : Skip if spread > $3 (normal is < $1).
        max_fraction=0.20: Skip if spread > 20% of the SL distance.
        """
        return cls(max_spread_abs=max_abs, max_spread_fraction=max_fraction)
