"""PortfolioEngine — sizing + exposure constraints + book risk metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..broker_adapter.base import BrokerAdapter, SymbolInfo
from ..config import SystemConfig
from ..signal_engine.types import CycleSignals


@dataclass
class Leg:
    symbol: str
    side: str                 # 'buy' | 'sell'
    lots: float
    notional: float           # signed USD-ish notional (sign follows side)
    price: float


@dataclass
class TargetHolding:
    pair_key: str
    kind: str                 # 'reversion' | 'carry'
    target_position: float    # signed fraction of equity (gross of the spread / leg)
    legs: list[Leg] = field(default_factory=list)
    beta: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def gross_notional(self) -> float:
        return sum(abs(l.notional) for l in self.legs)

    @property
    def is_flat(self) -> bool:
        return all(l.lots <= 0 for l in self.legs) or abs(self.target_position) < 1e-9


@dataclass
class ExposureSnapshot:
    gross_exposure: float     # multiples of equity
    net_exposure: float
    leverage: float
    n_holdings: int
    per_symbol_net: dict[str, float]
    scaled_by: float = 1.0    # <1 if exposure cap forced a downscale


def _round_step(x: float, step: float) -> float:
    if step <= 0:
        return x
    return math.floor(x / step + 1e-9) * step


class PortfolioEngine:
    def __init__(self, broker: BrokerAdapter, config: SystemConfig) -> None:
        self.broker = broker
        self.cfg = config
        self._sym_cache: dict[str, SymbolInfo] = {}

    def _info(self, symbol: str) -> SymbolInfo:
        if symbol not in self._sym_cache:
            self._sym_cache[symbol] = self.broker.symbol_info(symbol)
        return self._sym_cache[symbol]

    def _lots_for_notional(self, symbol: str, notional: float, price: float) -> float:
        """Convert a target notional (base-ccy units) to clamped, step-rounded lots."""
        info = self._info(symbol)
        if price <= 0 or info.contract_size <= 0:
            return 0.0
        raw = abs(notional) / (info.contract_size * price)
        lots = _round_step(raw, info.volume_step)
        if lots < info.volume_min:
            # too small to trade at this size; treat as flat (under-risk) rather than
            # forcing volume_min, which would distort the frozen sizing.
            return 0.0
        return min(lots, info.volume_max)

    def build_targets(self, signals: CycleSignals, equity: float,
                      prices: dict[str, float]) -> tuple[list[TargetHolding], ExposureSnapshot]:
        """Size every sleeve into lot-based target holdings and enforce exposure caps."""
        holdings: list[TargetHolding] = []

        # ── reversion pairs (two legs each) ─────────────────────────────────
        for ps in signals.pair_signals:
            if abs(ps.target_position) < 1e-9:
                holdings.append(TargetHolding(ps.pair_key, "reversion", 0.0, beta=ps.beta))
                continue
            pa, pb = prices.get(ps.y_symbol), prices.get(ps.x_symbol)
            if not pa or not pb:
                continue
            T = ps.target_position
            denom = 1.0 + abs(ps.beta)
            notion_a = T / denom * equity                      # signed
            notion_b = -T * ps.beta / denom * equity           # signed
            # per-position cap
            cap = self.cfg.max_position_pct * equity
            scale = min(1.0, cap / max(abs(notion_a) + abs(notion_b), 1e-9))
            notion_a *= scale; notion_b *= scale
            leg_a = self._leg(ps.y_symbol, notion_a, pa)
            leg_b = self._leg(ps.x_symbol, notion_b, pb)
            holdings.append(TargetHolding(
                ps.pair_key, "reversion", T * scale, legs=[leg_a, leg_b], beta=ps.beta,
                meta={"zscore": ps.zscore, "raw_target": ps.raw_target},
            ))

        # ── carry legs (single leg each) ────────────────────────────────────
        for cs in signals.carry_signals:
            if abs(cs.target_position) < 1e-9:
                continue
            price = prices.get(cs.symbol)
            if not price:
                continue
            notion = cs.target_position * equity
            cap = self.cfg.max_position_pct * equity
            if abs(notion) > cap:
                notion = math.copysign(cap, notion)
            leg = self._leg(cs.symbol, notion, price)
            holdings.append(TargetHolding(
                cs.pair_key, "carry", cs.target_position, legs=[leg],
                meta={"carry_rate": cs.carry_rate_annual, "weight": cs.weight},
            ))

        snap = self._exposure(holdings, equity)

        # ── gross-exposure cap (scale the whole book down if needed) ────────
        if snap.gross_exposure > self.cfg.max_gross_exposure and snap.gross_exposure > 0:
            factor = self.cfg.max_gross_exposure / snap.gross_exposure
            for h in holdings:
                for l in h.legs:
                    l.notional *= factor
                    l.lots = self._lots_for_notional(l.symbol, l.notional, l.price)
                h.target_position *= factor
            snap = self._exposure(holdings, equity)
            snap.scaled_by = factor

        # drop holdings that rounded to zero lots on every leg
        holdings = [h for h in holdings if any(l.lots > 0 for l in h.legs) or h.target_position == 0.0]
        return holdings, snap

    def _leg(self, symbol: str, notional: float, price: float) -> Leg:
        lots = self._lots_for_notional(symbol, notional, price)
        side = "buy" if notional >= 0 else "sell"
        return Leg(symbol=symbol, side=side, lots=lots, notional=notional, price=price)

    def _exposure(self, holdings: list[TargetHolding], equity: float) -> ExposureSnapshot:
        gross = 0.0
        per_symbol: dict[str, float] = {}
        for h in holdings:
            for l in h.legs:
                if l.lots <= 0:
                    continue
                gross += abs(l.notional)
                per_symbol[l.symbol] = per_symbol.get(l.symbol, 0.0) + l.notional
        net = sum(per_symbol.values())
        eq = max(equity, 1e-9)
        return ExposureSnapshot(
            gross_exposure=gross / eq, net_exposure=net / eq, leverage=gross / eq,
            n_holdings=len([h for h in holdings if any(l.lots > 0 for l in h.legs)]),
            per_symbol_net={k: v / eq for k, v in per_symbol.items()},
        )
