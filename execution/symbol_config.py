"""
Centralized symbol normalization.

Brokers add suffixes to standard symbols (Errante uses XAUUSD.i, others use .a,
.raw, #, etc). Mixing the raw and resolved names in logs, requests, and
notifications causes confusing duplicates and the occasional "symbol not
selected" failure. SymbolConfig resolves the broker-side name ONCE and is
threaded through every downstream component.

Usage:
    cfg = resolve_symbol("XAUUSD")          # -> SymbolConfig(name="XAUUSD.i", ...)
    cfg.name        # use this everywhere (orders, logs, telegram)
    cfg.point       # tick size in price units
    cfg.digits      # broker decimal digits
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import MetaTrader5 as mt5


# Common broker suffix variants we try when the bare symbol is not found.
_SUFFIX_CANDIDATES: tuple[str, ...] = (".i", ".a", ".raw", "-raw", "#", ".m", ".pro")


@dataclass(frozen=True)
class SymbolConfig:
    """
    Snapshot of broker-side symbol metadata captured at resolution time.

    Static fields only -- live tick / spread are fetched fresh by the validator
    so this object can be cached for the lifetime of the bot process.
    """

    requested: str          # what the user/config asked for (e.g. "XAUUSD")
    name: str               # actual broker symbol name (e.g. "XAUUSD.i")
    digits: int             # broker decimal digits
    point: float            # smallest price increment in price units
    tick_size: float        # trade_tick_size (may differ from point on some brokers)
    volume_min: float
    volume_max: float
    volume_step: float
    stops_level_points: int # min distance entry<->sl/tp in *points* (multiply by point for price)
    freeze_level_points: int
    filling_mode_mask: int  # bitmask of supported SYMBOL_FILLING_* flags
    trade_mode: int         # SYMBOL_TRADE_MODE_FULL / LONG_ONLY / etc.

    @property
    def stops_distance(self) -> float:
        """Minimum SL/TP distance from current price, in price units."""
        return self.stops_level_points * self.point

    @property
    def freeze_distance(self) -> float:
        """Within this distance from the order price, MT5 forbids modify/cancel."""
        return self.freeze_level_points * self.point

    @property
    def tradable(self) -> bool:
        """Is the symbol currently accepting orders?"""
        return self.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL


def _snapshot(si) -> dict:
    """Pull only the fields we actually consume from a symbol_info struct."""
    return {
        "digits":             int(si.digits),
        "point":              float(si.point or (10 ** (-si.digits))),
        "tick_size":          float(getattr(si, "trade_tick_size", 0) or si.point or 0.0),
        "volume_min":         float(si.volume_min or 0.01),
        "volume_max":         float(si.volume_max or 100.0),
        "volume_step":        float(si.volume_step or 0.01),
        "stops_level_points": int(getattr(si, "trade_stops_level", 0) or 0),
        "freeze_level_points": int(getattr(si, "trade_freeze_level", 0) or 0),
        "filling_mode_mask":  int(getattr(si, "filling_mode", 0) or 0),
        "trade_mode":         int(getattr(si, "trade_mode", mt5.SYMBOL_TRADE_MODE_FULL)),
    }


def resolve_symbol(requested: str) -> SymbolConfig:
    """
    Resolve `requested` to the exact broker symbol and snapshot its metadata.

    Raises RuntimeError if the symbol cannot be found in MT5 Market Watch -- caller
    should treat this as a hard fail (no point trying to trade an unknown symbol).
    """
    candidates: list[str] = [requested]
    candidates.extend(requested + suffix for suffix in _SUFFIX_CANDIDATES)

    # Errante (and several other brokers) expose BOTH a non-tradable bare
    # quote ("GBPUSD", trade_mode=0) and a tradable suffixed contract
    # ("GBPUSD.i", trade_mode=4). Picking the first hit silently grabbed the
    # disabled one and every order failed preflight with trade_mode=0. Prefer
    # a SYMBOL_TRADE_MODE_FULL candidate; fall back to the first-found name
    # only if nothing tradable is available.
    resolved: Optional[str] = None
    fallback: Optional[str] = None
    for name in candidates:
        # symbol_select(True) forces visibility in Market Watch (required for orders)
        if not mt5.symbol_select(name, True):
            continue
        si = mt5.symbol_info(name)
        if si is None:
            continue
        if int(getattr(si, "trade_mode", 0)) == mt5.SYMBOL_TRADE_MODE_FULL:
            resolved = name
            break
        if fallback is None:
            fallback = name

    if resolved is None:
        resolved = fallback

    if resolved is None:
        raise RuntimeError(
            f"Symbol {requested!r} not found in MT5. Tried: {candidates}. "
            "Open Market Watch -> Show All in the terminal and pick the exact name."
        )

    si = mt5.symbol_info(resolved)
    fields = _snapshot(si)

    return SymbolConfig(requested=requested, name=resolved, **fields)
