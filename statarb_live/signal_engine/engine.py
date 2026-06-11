"""
SignalEngine — turns a live price panel into a fully-explainable CycleSignals.

Every number is produced by the research engine with the frozen NB38 parameters; this module
only orchestrates the calls and extracts the *latest closed bar*. No thresholds are invented
here. The composition of the three sleeves follows NB38:

  * reversion  (cell 1):  per pair  z-score -> {-1,0,+1} target -> vol-target leverage.
  * carry      (cell 3):  cross-sectional carry weights, blended at ``w_rev`` (0.5/0.5).
  * regime     (cell 17): a continuous HMM calm-probability multiplier scales the reversion
                          sleeve (this is exactly the variant NB38 cell 17 evaluated;
                          ``regime_scales_carry`` is exposed but defaults False to match it).

Capital convention: ``target_position`` is signed gross exposure as a fraction of total
equity. Reversion pair p: ``w_rev * (1/n_pairs) * regime_mult * (target * leverage)``.
Carry symbol s: ``(1 - w_rev) * weight_s``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import STRATEGY
from ..engine_bridge import (
    bars_per_year, eng_backtest, eng_carry, eng_pairs, eng_regime, eng_spread,
    eng_signals,
)
from .types import CarrySignal, CycleSignals, PairSignal, RegimeState
from .universe import Universe, pair_key


class SignalEngine:
    def __init__(self, universe: Universe, *, regime_scales_carry: bool = False) -> None:
        self.uni = universe
        self.n_pairs = max(len(universe.pairs), 1)
        self.regime_scales_carry = regime_scales_carry
        self._provenance = STRATEGY.as_provenance()

    # ── regime sleeve ────────────────────────────────────────────────────────
    def _regime(self, panel: pd.DataFrame) -> RegimeState:
        proxy = STRATEGY.regime_proxy_symbol
        if proxy not in panel.columns:
            proxy = panel.columns[0]
        if not STRATEGY.regime_enabled:
            return RegimeState(method="off", proxy_symbol=proxy, prob_calm=1.0,
                               multiplier=1.0, label="off")
        px = panel[proxy].dropna()
        mult_series = eng_regime.regime_size_multiplier(
            px, method=STRATEGY.regime_method,
            min_mult=STRATEGY.regime_min_mult, max_mult=STRATEGY.regime_max_mult,
            timeframe=STRATEGY.bars_per_year_key,
        )
        mult = float(mult_series.reindex(panel.index).ffill().iloc[-1])
        span = STRATEGY.regime_max_mult - STRATEGY.regime_min_mult
        prob_calm = (mult - STRATEGY.regime_min_mult) / span if span > 0 else 1.0
        prob_calm = float(np.clip(prob_calm, 0.0, 1.0))
        label = "calm" if prob_calm >= 0.5 else "crisis"
        return RegimeState(method=STRATEGY.regime_method, proxy_symbol=proxy,
                           prob_calm=prob_calm, multiplier=mult, label=label)

    # ── reversion sleeve ─────────────────────────────────────────────────────
    def _pair_signal(self, panel: pd.DataFrame, p, regime: RegimeState) -> PairSignal | None:
        if p.a not in panel.columns or p.b not in panel.columns:
            return None
        sub = panel[[p.a, p.b]].dropna()
        if len(sub) < STRATEGY.z_window // 2:
            return None
        px_a, px_b = sub[p.a], sub[p.b]

        spread = eng_spread.static_spread(px_a, px_b, p.beta, p.alpha)
        z_series = eng_spread.rolling_zscore(spread, window=STRATEGY.z_window)
        target_series = eng_signals.zscore_positions(
            z_series, z_entry=STRATEGY.z_entry, z_exit=STRATEGY.z_exit, z_stop=STRATEGY.z_stop
        )
        raw_target = float(target_series.iloc[-1])

        # Vol-target leverage — identical maths to backtest.backtest_pair.
        ra, rb = px_a.pct_change(), px_b.pct_change()
        denom = 1.0 + abs(p.beta)
        r_pair = (ra - p.beta * rb) / denom
        tgt_bar = STRATEGY.target_ann_vol / np.sqrt(bars_per_year())
        vol_hat = r_pair.rolling(STRATEGY.vol_window,
                                 min_periods=STRATEGY.vol_window // 2).std()
        lev = float((tgt_bar / vol_hat).clip(upper=STRATEGY.max_leverage).fillna(0.0).iloc[-1])

        signed_exposure = raw_target * lev
        regime_mult = regime.multiplier if STRATEGY.regime_enabled else 1.0
        sleeve_weight = STRATEGY.carry_w_rev if STRATEGY.carry_enabled else 1.0
        target_position = sleeve_weight * (1.0 / self.n_pairs) * regime_mult * signed_exposure

        z_now = float(z_series.iloc[-1]) if np.isfinite(z_series.iloc[-1]) else 0.0
        spread_mean = float(spread.rolling(STRATEGY.z_window,
                                           min_periods=STRATEGY.z_window // 2).mean().iloc[-1])
        spread_val = float(spread.iloc[-1])

        # current-window cointegration p-value (diagnostic only — NB38 does NOT re-gate)
        try:
            adf_p = float(eng_pairs.engle_granger_p(px_a, px_b))
        except Exception:
            adf_p = float("nan")

        confidence = float(np.clip(abs(z_now) / STRATEGY.z_stop, 0.0, 1.0) * regime.prob_calm)
        action = ("target_long" if raw_target > 0 else
                  "target_short" if raw_target < 0 else "flat")

        return PairSignal(
            pair_key=pair_key(p.a, p.b), y_symbol=p.a, x_symbol=p.b,
            zscore=z_now, beta=float(p.beta), alpha=float(p.alpha),
            spread_value=spread_val, spread_mean=spread_mean,
            half_life_bars=float(p.half_life), adf_p=adf_p,
            raw_target=raw_target, vol_leverage=lev, target_position=target_position,
            expected_spread_move=spread_mean - spread_val, confidence=confidence,
            action=action,
        )

    # ── carry sleeve ─────────────────────────────────────────────────────────
    def _carry_signals(self, panel: pd.DataFrame, regime: RegimeState) -> list[CarrySignal]:
        if not STRATEGY.carry_enabled:
            return []
        cols = [s for s in self.uni.carry_symbols if s in panel.columns]
        if len(cols) < 2:
            return []
        sub = panel[cols].dropna()
        w = eng_carry.carry_signal(sub)
        # smooth to the configured rebalance cadence, matching backtest_carry
        if STRATEGY.carry_rebalance > 1:
            w = w.rolling(STRATEGY.carry_rebalance, min_periods=1).mean()
        w_last = w.iloc[-1]
        rates = eng_carry.pair_carry_rate(cols, sub.index).iloc[-1]
        w_carry = (1.0 - STRATEGY.carry_w_rev)
        regime_mult = regime.multiplier if (self.regime_scales_carry and STRATEGY.regime_enabled) else 1.0
        out: list[CarrySignal] = []
        for s in cols:
            weight = float(w_last.get(s, 0.0))
            if abs(weight) < 1e-9:
                continue
            out.append(CarrySignal(
                symbol=s, carry_rate_annual=float(rates.get(s, float("nan"))),
                weight=weight, target_position=w_carry * regime_mult * weight,
            ))
        return out

    # ── public API ───────────────────────────────────────────────────────────
    def evaluate(self, panel: pd.DataFrame, cycle_id: str) -> CycleSignals:
        """Compute all sleeves for the latest closed bar in ``panel``."""
        if panel.empty:
            raise ValueError("SignalEngine.evaluate received an empty panel")
        signal_ts = panel.index[-1]
        regime = self._regime(panel)

        pair_signals = []
        for p in self.uni.pairs:
            sig = self._pair_signal(panel, p, regime)
            if sig is not None:
                pair_signals.append(sig)

        carry_signals = self._carry_signals(panel, regime)

        # full symbol -> annual carry-rate map (for per-leg carry PnL attribution)
        carry_rates: dict[str, float] = {}
        fx_cols = [s for s in panel.columns if len(s) == 6]
        if fx_cols:
            try:
                rates = eng_carry.pair_carry_rate(fx_cols, panel.index).iloc[-1]
                carry_rates = {s: float(rates.get(s, float("nan"))) for s in fx_cols}
            except Exception:
                carry_rates = {}

        return CycleSignals(
            cycle_id=cycle_id, signal_ts=signal_ts, regime=regime,
            pair_signals=pair_signals, carry_signals=carry_signals,
            panel_symbols=list(panel.columns), carry_rates=carry_rates,
            notes={"n_pairs_evaluated": len(pair_signals),
                   "n_carry_legs": len(carry_signals)},
        )

    def provenance(self) -> dict:
        return dict(self._provenance)

    def carry_value_for(self, signals: CycleSignals, pair: PairSignal) -> float:
        """Carry rate of the reversion pair's long leg (for per-pair carry logging)."""
        for cs in signals.carry_signals:
            if cs.symbol == pair.y_symbol:
                return cs.carry_rate_annual
        return float("nan")
