"""ExecutionSimulator — paper microstructure: spread crossing, slippage, latency."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Fill:
    symbol: str
    side: str               # 'buy' | 'sell'
    lots: float
    intended_price: float
    actual_price: float
    slippage_bps: float     # signed, positive = adverse (worse than intended)
    spread_bps: float
    latency_ms: float


class ExecutionSimulator:
    """Taker fill model.

    A buy lifts the ask, a sell hits the bid, so you pay half the quoted spread on entry and
    again on exit — the dominant, *real* FX cost (matches the research cost model, which uses
    the quoted spread as ``fee_bps``). On top we add a small, configurable random slippage to
    represent queue/latency effects, and a simulated latency in milliseconds.
    """

    def __init__(self, *, extra_slippage_bps: float = 0.2, slippage_jitter_bps: float = 0.1,
                 base_latency_ms: float = 45.0, latency_jitter_ms: float = 25.0,
                 seed: int | None = 7) -> None:
        self.extra_slippage_bps = extra_slippage_bps
        self.slippage_jitter_bps = slippage_jitter_bps
        self.base_latency_ms = base_latency_ms
        self.latency_jitter_ms = latency_jitter_ms
        self._rng = random.Random(seed)

    def fill(self, symbol: str, side: str, lots: float, ref_price: float,
             spread_bps: float) -> Fill:
        spread_bps = spread_bps if spread_bps == spread_bps and spread_bps > 0 else 0.4  # NaN guard
        half_spread_frac = (spread_bps / 2.0) / 1e4
        slip_bps = self.extra_slippage_bps + abs(self._rng.gauss(0.0, self.slippage_jitter_bps))
        slip_frac = slip_bps / 1e4
        direction = 1.0 if side == "buy" else -1.0
        # cross the spread + adverse slippage, both against you
        actual = ref_price * (1.0 + direction * (half_spread_frac + slip_frac))
        total_slip_bps = (half_spread_frac + slip_frac) * 1e4
        latency = max(1.0, self.base_latency_ms + self._rng.gauss(0.0, self.latency_jitter_ms))
        return Fill(symbol=symbol, side=side, lots=lots, intended_price=ref_price,
                    actual_price=actual, slippage_bps=total_slip_bps, spread_bps=spread_bps,
                    latency_ms=latency)
