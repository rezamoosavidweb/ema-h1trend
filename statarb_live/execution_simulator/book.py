"""
PaperBook — the paper account ledger.

Tracks open positions (one per pair_key — a reversion pair has two legs, a carry holding one),
marks them to market at reference mid prices, and on close decomposes realised PnL into the
three buckets the Phase-5 attribution mandate asks for:

    realized_pnl   = price_pnl_actual + carry_accrual
    cost_pnl       = price_pnl_actual - price_pnl_mid          (execution drag, <= 0)
    carry_pnl      = carry_accrual  (+ price move for carry-sleeve holdings)
    reversion_pnl  = price_pnl_mid for reversion-sleeve holdings

Units: PnL is in account currency (USD), using ``contract_size`` per lot. Carry accrual uses
the stored annual rate differential and the bars held (``rate/100 * bars_held / bars_per_year``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class OpenLeg:
    symbol: str
    direction: int            # +1 long, -1 short
    lots: float
    contract_size: float
    entry_actual: float       # filled price (incl. spread/slippage)
    entry_mid: float          # reference mid at entry
    carry_rate_annual: float  # rate(base)-rate(quote), % — for carry accrual
    entry_slippage_bps: float = 0.0

    @property
    def notional(self) -> float:
        return self.direction * self.lots * self.contract_size * self.entry_mid


@dataclass
class OpenPosition:
    pair_key: str
    kind: str                 # 'reversion' | 'carry'
    legs: list[OpenLeg]
    opened_ts: pd.Timestamp
    signal_ts: pd.Timestamp
    z_at_open: float = 0.0
    beta_at_open: float = 0.0
    alpha_at_open: float = 0.0
    regime_at_open: str = ""
    carry_value: float = 0.0
    side: str = ""            # 'long' | 'short' (of spread / position)
    storage_id: int | None = None
    bars_held: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def gross_notional(self) -> float:
        return sum(abs(l.notional) for l in self.legs)

    def price_pnl_mid(self, prices: dict[str, float]) -> float:
        pnl = 0.0
        for l in self.legs:
            p = prices.get(l.symbol, l.entry_mid)
            pnl += l.direction * (p - l.entry_mid) * l.lots * l.contract_size
        return pnl

    def carry_accrual(self, bars_held: int, bars_per_year: float) -> float:
        frac = bars_held / max(bars_per_year, 1e-9)
        return sum(l.notional * (l.carry_rate_annual / 100.0) * frac for l in self.legs)


@dataclass
class ClosedTrade:
    pair_key: str
    kind: str
    realized_pnl: float
    reversion_pnl: float
    carry_pnl: float
    cost_pnl: float
    bars_held: int
    entry_slippage_bps: float
    exit_slippage_bps: float
    gross_notional: float


class PaperBook:
    def __init__(self, starting_equity: float, bars_per_year: float) -> None:
        self.starting_equity = starting_equity
        self.realized_cum = 0.0
        self.bpy = bars_per_year
        self.positions: dict[str, OpenPosition] = {}

    # ── lifecycle ────────────────────────────────────────────────────────────
    def open(self, pos: OpenPosition) -> None:
        self.positions[pos.pair_key] = pos

    def has(self, pair_key: str) -> bool:
        return pair_key in self.positions

    def get(self, pair_key: str) -> OpenPosition | None:
        return self.positions.get(pair_key)

    def close(self, pair_key: str, exit_fills: dict[str, "object"],
              prices: dict[str, float]) -> ClosedTrade | None:
        """Close a position. ``exit_fills`` maps symbol -> Fill (actual exit prices)."""
        pos = self.positions.pop(pair_key, None)
        if pos is None:
            return None

        price_pnl_actual = 0.0
        price_pnl_mid = 0.0
        exit_slip = 0.0
        for l in pos.legs:
            f = exit_fills.get(l.symbol)
            exit_actual = float(getattr(f, "actual_price", prices.get(l.symbol, l.entry_mid)))
            exit_mid = float(prices.get(l.symbol, l.entry_mid))
            exit_slip = max(exit_slip, float(getattr(f, "slippage_bps", 0.0)))
            price_pnl_actual += l.direction * (exit_actual - l.entry_actual) * l.lots * l.contract_size
            price_pnl_mid += l.direction * (exit_mid - l.entry_mid) * l.lots * l.contract_size

        carry = pos.carry_accrual(pos.bars_held, self.bpy)
        cost_pnl = price_pnl_actual - price_pnl_mid
        realized = price_pnl_actual + carry
        self.realized_cum += realized

        reversion_pnl = price_pnl_mid if pos.kind == "reversion" else 0.0
        carry_pnl = carry + (price_pnl_mid if pos.kind == "carry" else 0.0)
        entry_slip = max((l.entry_slippage_bps for l in pos.legs), default=0.0)
        return ClosedTrade(
            pair_key=pair_key, kind=pos.kind, realized_pnl=realized,
            reversion_pnl=reversion_pnl, carry_pnl=carry_pnl, cost_pnl=cost_pnl,
            bars_held=pos.bars_held, entry_slippage_bps=entry_slip,
            exit_slippage_bps=exit_slip, gross_notional=pos.gross_notional,
        )

    # ── valuation ─────────────────────────────────────────────────────────────
    def unrealized(self, prices: dict[str, float]) -> float:
        return sum(p.price_pnl_mid(prices) + p.carry_accrual(p.bars_held, self.bpy)
                   for p in self.positions.values())

    def equity(self, prices: dict[str, float]) -> float:
        return self.starting_equity + self.realized_cum + self.unrealized(prices)

    def gross_notional(self) -> float:
        return sum(p.gross_notional for p in self.positions.values())

    def pair_contributions(self, prices: dict[str, float]) -> dict[str, float]:
        return {k: p.price_pnl_mid(prices) + p.carry_accrual(p.bars_held, self.bpy)
                for k, p in self.positions.items()}

    def increment_bars(self) -> None:
        for p in self.positions.values():
            p.bars_held += 1
