"""Value objects produced by the signal engine — all fully explainable / loggable."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class RegimeState:
    """Continuous regime sizing state for the cycle (NB38 §5)."""
    method: str
    proxy_symbol: str
    prob_calm: float           # filtered P(calm) at the latest bar
    multiplier: float          # exposure multiplier in [min_mult, max_mult]
    label: str                 # 'calm' | 'crisis' (argmax-ish summary)

    def as_row(self) -> dict:
        return {"regime_state": self.label, "regime_prob_calm": self.prob_calm,
                "regime_multiplier": self.multiplier}


@dataclass(frozen=True)
class PairSignal:
    """One cointegration-pair reversion signal at the latest closed bar."""
    pair_key: str
    y_symbol: str              # leg A (long when long-spread)
    x_symbol: str              # leg B (hedge)
    zscore: float
    beta: float                # hedge ratio (formation, static)
    alpha: float
    spread_value: float
    spread_mean: float
    half_life_bars: float
    adf_p: float
    raw_target: float          # reversion target in {-1, 0, +1}
    vol_leverage: float        # vol-target leverage applied this bar
    # final signed gross exposure of THIS sleeve as a fraction of total equity
    target_position: float
    expected_spread_move: float
    confidence: float          # 0..1
    action: str                # hold | open_long | open_short | close | scale

    def as_signal_row(self, cycle_id: str, signal_ts: pd.Timestamp,
                      regime: RegimeState, carry_value: float, provenance: dict) -> dict:
        return {
            "cycle_id": cycle_id, "signal_ts": signal_ts, "pair_key": self.pair_key,
            "y_symbol": self.y_symbol, "x_symbol": self.x_symbol,
            "zscore": self.zscore, "hedge_ratio": self.beta, "alpha": self.alpha,
            "spread_value": self.spread_value, "half_life_bars": self.half_life_bars,
            "adf_p": self.adf_p, "carry_value": carry_value,
            **regime.as_row(),
            "raw_target": self.raw_target, "vol_leverage": self.vol_leverage,
            "target_position": self.target_position,
            "expected_spread_move": self.expected_spread_move,
            "confidence": self.confidence, "action": self.action,
            "provenance": provenance,
        }


@dataclass(frozen=True)
class CarrySignal:
    """One cross-sectional carry-sleeve weight at the latest closed bar."""
    symbol: str
    carry_rate_annual: float   # rate(base) - rate(quote), %
    weight: float              # gross-normalised cross-sectional weight (signed)
    target_position: float     # final fraction-of-equity (after w_carry scaling)

    @property
    def pair_key(self) -> str:
        return f"CARRY:{self.symbol}"


@dataclass
class CycleSignals:
    """Everything decided in one cycle."""
    cycle_id: str
    signal_ts: pd.Timestamp
    regime: RegimeState
    pair_signals: list[PairSignal] = field(default_factory=list)
    carry_signals: list[CarrySignal] = field(default_factory=list)
    panel_symbols: list[str] = field(default_factory=list)
    carry_rates: dict = field(default_factory=dict)   # symbol -> annual rate diff (%), all panel symbols
    notes: dict = field(default_factory=dict)

    def net_targets_by_symbol(self) -> dict[str, float]:
        """Consolidated net target position per FX symbol across both sleeves, in
        fraction-of-equity signed notional. Reversion pair p contributes its spread
        position split across its two legs by the hedge ratio."""
        net: dict[str, float] = {}
        for ps in self.pair_signals:
            denom = 1.0 + abs(ps.beta)
            net[ps.y_symbol] = net.get(ps.y_symbol, 0.0) + ps.target_position / denom
            net[ps.x_symbol] = net.get(ps.x_symbol, 0.0) - ps.target_position * ps.beta / denom
        for cs in self.carry_signals:
            net[cs.symbol] = net.get(cs.symbol, 0.0) + cs.target_position
        return net
