"""
Enhanced Order Block Signal Engine — integrates all strategy improvements.

This module is the UPGRADED replacement for the baseline ob_core.py.
It builds on the same displacement → OB → retest foundation but adds:

  1. Higher Timeframe (H1) structural bias
     Only trade bullish OBs in H1 bullish structure and vice versa.
     Prevents counter-trend OB trades in strong trending markets.

  2. BOS/MSS confirmation
     Require that the displacement broke real market structure.
     Prevents trading OBs formed during choppy ranging price action.

  3. Fair Value Gap (FVG) validation
     Require an imbalance gap in the displacement zone.
     No FVG = no real institutional imbalance = skip.

  4. Session filter
     Restrict signals to London + New York sessions.
     Avoids low-liquidity Asian session false signals.

  5. Market regime filter
     Skip all signals in ranging (ADX < 20) market conditions.
     Reduces OB invalidation rate in choppy markets.

  6. Volume confirmation
     Displacement segment must show above-average volume.
     Confirms real institutional participation vs thin-market noise.

  7. Liquidity sweep confirmation (optional)
     Check if a recent stop-hunt sweep preceded the OB formation.
     Highest-probability OBs follow a liquidity grab.

ARCHITECTURE
------------
Each filter is INDEPENDENTLY configurable via OBSignalConfig.
You can enable/disable any subset of filters for research comparison.
The base ob_core.py logic is preserved unchanged as the foundation.

RESULT FORMAT
-------------
Identical to ob_core.list_ob_signals() output — same columns, same schema.
Additional columns are added with filter diagnostic information.

USAGE
-----
    from strategy.ob_signals import list_ob_signals_enhanced, OBSignalConfig

    cfg = OBSignalConfig(
        require_htf_bias=True,
        require_bos=True,
        require_fvg=True,
        session_filter_enabled=True,
        regime_filter_enabled=True,
        volume_filter_enabled=True,
        liquidity_sweep_filter=False,  # optional extra confirmation
    )
    signals = list_ob_signals_enhanced(m5_df, h1_df, risk_cash=20.0, config=cfg)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

# ── Baseline OB engine (unchanged, always imported) ──────────────────────────
from strategy.ob_core import (
    add_candle_features,
    _detect_displacements,
    _find_order_blocks,
    _is_rejection,
    _Displacement,
    _OB,
    ATR_PERIOD,
    DISPLACEMENT_MIN_CANDLES,
    DISPLACEMENT_ATR_MULT,
    OB_EXPIRY_BARS,
    REJECTION_WICK_RATIO,
    DEFAULT_RR,
    SL_BUFFER,
    MAKER_FEE_RATE,
    TAKER_FEE_RATE,
    MIN_SL_FEE_MULT,
)

# ── New strategy modules ──────────────────────────────────────────────────────
from strategy.structure.bos_mss import (
    detect_swing_points,
    detect_structure_breaks,
    get_current_bias,
    MarketBias,
)
from strategy.confirmations.fvg import has_displacement_fvg, find_fvg_for_ob
from strategy.filters.session import SessionFilter, SessionConfig
from strategy.liquidity.sweeps import (
    detect_liquidity_levels,
    detect_liquidity_sweeps,
    has_recent_sweep,
)
from strategy.volume.confirmation import compute_volume_context, displacement_has_volume
from strategy.regime.market_regime import detect_regime, MarketRegime

# ── EMA H1 trend (the existing baseline H1 bias) ─────────────────────────────
from strategy.crypto_core import add_emas, h1_trend_series


# ── Signal configuration dataclass ───────────────────────────────────────────

@dataclass
class OBSignalConfig:
    """
    Controls which filters and confirmations are active.

    All flags default to OFF so this module can be used as a drop-in
    replacement with progressive opt-in of each improvement.

    The notebooks use these flags to do A/B comparisons: run with one
    flag True, measure the change in expectancy, then decide to keep it.
    """

    # ── Core OB parameters (same defaults as ob_core.py) ────────────────────
    atr_period               : int   = ATR_PERIOD
    displacement_min_candles : int   = DISPLACEMENT_MIN_CANDLES
    displacement_atr_mult    : float = DISPLACEMENT_ATR_MULT
    ob_expiry_bars           : int   = OB_EXPIRY_BARS
    rejection_wick_ratio     : float = REJECTION_WICK_RATIO
    rr                       : float = DEFAULT_RR
    sl_buffer                : float = SL_BUFFER
    min_sl_fee_mult          : float = MIN_SL_FEE_MULT

    # ── HTF bias: H1 structural direction ───────────────────────────────────
    require_htf_bias   : bool = False   # if True, H1 must agree with OB direction
    htf_bias_source    : str  = "ema"   # "ema" | "bos_mss" (which H1 method to use)

    # ── BOS/MSS on M5 ─────────────────────────────────────────────────────
    require_bos        : bool = False   # if True, displacement must break M5 structure
    bos_swing_len      : int  = 5
    bos_lookback       : int  = 50

    # ── FVG validation ─────────────────────────────────────────────────────
    require_fvg        : bool  = False  # if True, displacement must leave an FVG
    fvg_search_bars    : int   = 8      # bars after OB to search for FVG
    fvg_min_atr_frac   : float = 0.0    # minimum FVG size as fraction of ATR

    # ── Session filter ──────────────────────────────────────────────────────
    session_filter_enabled : bool = False
    allow_london           : bool = True
    allow_new_york         : bool = True
    allow_overlap          : bool = True
    allow_asia             : bool = False

    # ── Market regime filter ────────────────────────────────────────────────
    regime_filter_enabled  : bool = False
    skip_ranging           : bool = True   # skip signals when ADX < threshold
    skip_high_volatility   : bool = False  # skip signals in ATR spikes
    adx_period             : int  = 14
    adx_lookback           : int  = 100

    # ── Volume confirmation ─────────────────────────────────────────────────
    volume_filter_enabled  : bool = False
    volume_ma_period       : int  = 20
    volume_multiplier      : float = 1.3   # displacement must avg >= this × volume MA

    # ── Liquidity sweep confirmation ─────────────────────────────────────────
    liquidity_sweep_filter : bool = False   # require prior stop hunt sweep
    sweep_lookback_bars    : int  = 50      # bars to scan for sweep
    sweep_max_age_bars     : int  = 30      # how old can the sweep be?


# ── Public API ────────────────────────────────────────────────────────────────

def list_ob_signals_enhanced(
    m5: pd.DataFrame,
    h1: Optional[pd.DataFrame] = None,
    *,
    risk_cash: float,
    config: Optional[OBSignalConfig] = None,
) -> pd.DataFrame:
    """
    Scan M5 bars and return enhanced OB signals with all configured filters.

    Mirrors ob_core.list_ob_signals() signature but accepts an OBSignalConfig.
    If h1 is None, HTF bias filter is automatically disabled.

    Parameters
    ----------
    m5        : Closed M5 OHLCV bars (DatetimeIndex UTC).
    h1        : Closed H1 OHLCV bars for HTF bias (optional).
    risk_cash : Fixed USD risk per trade.
    config    : OBSignalConfig controlling which filters are active.

    Returns
    -------
    DataFrame with signal columns plus filter diagnostics.
    Same schema as ob_core.list_ob_signals() with extra columns:
        htf_bias, bos_break, has_fvg, session, regime, vol_confirmed, has_sweep
        filter_reason (first filter that rejected the signal, empty = passed all)
    """
    cfg = config or OBSignalConfig()

    # ── Step 0: Add candle features (ATR, body, wicks) ───────────────────────
    m5 = add_candle_features(m5, atr_period=cfg.atr_period)

    # ── Step 1: H1 bias (compute once for entire dataset) ────────────────────
    h1_trend: Optional[pd.Series] = None
    if cfg.require_htf_bias and h1 is not None:
        if cfg.htf_bias_source == "ema":
            h1_trend = _compute_h1_ema_trend(h1)
        else:
            # BOS/MSS-based H1 bias: get bias at each H1 bar
            h1_trend = _compute_h1_bos_trend(h1)

    # ── Step 2: Market regime (compute once for entire dataset) ───────────────
    regime_result = None
    if cfg.regime_filter_enabled:
        regime_result = detect_regime(
            m5,
            adx_period=cfg.adx_period,
            lookback=cfg.adx_lookback,
        )

    # ── Step 3: Volume context (compute once for entire dataset) ──────────────
    vol_ctx = None
    if cfg.volume_filter_enabled:
        vol_ctx = compute_volume_context(m5, ma_period=cfg.volume_ma_period)

    # ── Step 4: Session filter instance ──────────────────────────────────────
    session_filt = None
    if cfg.session_filter_enabled:
        session_filt = SessionFilter(SessionConfig(
            allow_london=cfg.allow_london,
            allow_new_york=cfg.allow_new_york,
            allow_overlap=cfg.allow_overlap,
            allow_asia=cfg.allow_asia,
        ))

    # ── Step 5: Detect displacements and OBs ─────────────────────────────────
    displacements = _detect_displacements(
        m5, cfg.displacement_min_candles, cfg.displacement_atr_mult
    )
    obs = _find_order_blocks(m5, displacements)

    # Build a map from OB bar_idx → displacement object for volume check
    disp_map = {
        d.start_idx: d for d in displacements
    }
    # Map each OB to the displacement that spawned it
    ob_to_disp: dict[int, _Displacement] = {}
    for ob in obs:
        for d in displacements:
            # The OB is the candle just before the displacement start
            if d.start_idx == ob.ob_bar_idx + 1:
                ob_to_disp[ob.ob_bar_idx] = d
                break

    n = len(m5)
    rows: list[dict] = []

    # ── Step 6: Process each OB ───────────────────────────────────────────────
    for ob in obs:
        # ── Regime filter (evaluated once per OB, not per signal bar) ────────
        if cfg.regime_filter_enabled and regime_result is not None:
            if cfg.skip_ranging and regime_result.is_ranging:
                continue   # entire dataset is ranging — skip all OBs
            if cfg.skip_high_volatility and regime_result.is_high_vol:
                continue

        # ── BOS check: did the displacement that formed this OB break structure?
        bos_confirmed = True
        if cfg.require_bos:
            # Check if price broke a swing level in the window before the OB
            window_for_bos = m5.iloc[max(0, ob.ob_bar_idx - cfg.bos_lookback) : ob.ob_bar_idx + 3]
            swings = detect_swing_points(window_for_bos, swing_len=cfg.bos_swing_len)
            bos_breaks = detect_structure_breaks(window_for_bos, swings)
            # A relevant BOS must be in the direction matching the OB type
            target_dir = "bullish" if ob.ob_type == "bullish" else "bearish"
            bos_confirmed = any(b.direction == target_dir for b in bos_breaks)

        # ── FVG check: did the displacement leave an imbalance? ───────────────
        fvg_confirmed = True
        if cfg.require_fvg:
            fvg_confirmed = has_displacement_fvg(
                m5,
                ob_bar_idx=ob.ob_bar_idx,
                direction=ob.ob_type,
                search_bars=cfg.fvg_search_bars,
                min_gap_atr_fraction=cfg.fvg_min_atr_frac,
            )

        # ── Volume check: was the displacement on high volume? ────────────────
        vol_confirmed = True
        if cfg.volume_filter_enabled and vol_ctx is not None:
            disp = ob_to_disp.get(ob.ob_bar_idx)
            if disp is not None:
                disp_end = min(disp.start_idx + cfg.displacement_min_candles, n - 1)
                vol_confirmed = displacement_has_volume(
                    m5, disp.start_idx, disp_end, vol_ctx,
                    multiplier=cfg.volume_multiplier,
                )

        # ── Liquidity sweep check ─────────────────────────────────────────────
        sweep_confirmed = True
        if cfg.liquidity_sweep_filter:
            sweep_confirmed = has_recent_sweep(
                m5,
                current_bar=ob.ob_bar_idx,
                ob_direction=ob.ob_type,
                lookback_bars=cfg.sweep_lookback_bars,
                sweep_tolerance=cfg.sweep_max_age_bars,
            )

        # ── Now scan for valid retest bars (same as ob_core.py) ──────────────
        displaced = False
        for i in range(ob.ob_bar_idx + 1, min(ob.ob_bar_idx + cfg.ob_expiry_bars, n - 1)):
            bar = m5.iloc[i]

            if ob.ob_type == "bullish":
                if not displaced:
                    if float(bar["close"]) > ob.ob_high:
                        displaced = True
                    continue
                if float(bar["low"]) <= ob.ob_high and float(bar["close"]) >= ob.ob_low:
                    if _is_rejection(bar, ob.ob_type, cfg.rejection_wick_ratio):
                        entry    = ob.ob_high
                        sl       = ob.ob_low - cfg.sl_buffer
                        dist     = entry - sl
                        min_dist = entry * (MAKER_FEE_RATE + TAKER_FEE_RATE) * cfg.min_sl_fee_mult
                        if dist >= min_dist:
                            _emit_signal(
                                rows, m5, i, ob, "buy", entry, sl, dist,
                                risk_cash, cfg,
                                h1_trend=h1_trend,
                                bos_confirmed=bos_confirmed,
                                fvg_confirmed=fvg_confirmed,
                                session_filt=session_filt,
                                regime_result=regime_result,
                                vol_confirmed=vol_confirmed,
                                sweep_confirmed=sweep_confirmed,
                            )
                        break
                if float(bar["close"]) < ob.ob_low:
                    break

            else:  # bearish OB
                if not displaced:
                    if float(bar["close"]) < ob.ob_low:
                        displaced = True
                    continue
                if float(bar["high"]) >= ob.ob_low and float(bar["close"]) <= ob.ob_high:
                    if _is_rejection(bar, ob.ob_type, cfg.rejection_wick_ratio):
                        entry    = ob.ob_low
                        sl       = ob.ob_high + cfg.sl_buffer
                        dist     = sl - entry
                        min_dist = entry * (MAKER_FEE_RATE + TAKER_FEE_RATE) * cfg.min_sl_fee_mult
                        if dist >= min_dist:
                            _emit_signal(
                                rows, m5, i, ob, "sell", entry, sl, dist,
                                risk_cash, cfg,
                                h1_trend=h1_trend,
                                bos_confirmed=bos_confirmed,
                                fvg_confirmed=fvg_confirmed,
                                session_filt=session_filt,
                                regime_result=regime_result,
                                vol_confirmed=vol_confirmed,
                                sweep_confirmed=sweep_confirmed,
                            )
                        break
                if float(bar["close"]) > ob.ob_high:
                    break

    return pd.DataFrame(rows)


# ── Internal signal emission helper ──────────────────────────────────────────

def _emit_signal(
    rows: list,
    m5: pd.DataFrame,
    signal_bar: int,
    ob: _OB,
    side: str,
    entry: float,
    sl: float,
    dist: float,
    risk_cash: float,
    cfg: OBSignalConfig,
    h1_trend,
    bos_confirmed: bool,
    fvg_confirmed: bool,
    session_filt,
    regime_result,
    vol_confirmed: bool,
    sweep_confirmed: bool,
) -> None:
    """
    Evaluate all filters and append a signal row if all required filters pass.

    This is extracted here to keep list_ob_signals_enhanced() readable.
    The 'filter_reason' column documents WHY a signal was rejected.
    Empty filter_reason = signal passed all active filters.
    """
    ts = m5.index[signal_bar]

    # ── Evaluate each filter in priority order ────────────────────────────────
    filter_reason = ""

    # 1. Session filter
    session_label = "unknown"
    if session_filt is not None:
        session_label = session_filt.session_of(ts).value
        if not session_filt.is_allowed(ts):
            filter_reason = f"session:{session_label}"

    # 2. HTF bias filter
    htf_bias_label = "none"
    if not filter_reason and h1_trend is not None:
        # Find the H1 bias at this M5 bar time (forward-fill H1 → M5)
        prior_h1 = h1_trend[h1_trend.index <= ts]
        if not prior_h1.empty:
            htf_bias_label = str(prior_h1.iloc[-1])
            if side == "buy" and htf_bias_label not in ("bull", "bullish"):
                filter_reason = f"htf_bias:{htf_bias_label}"
            elif side == "sell" and htf_bias_label not in ("bear", "bearish"):
                filter_reason = f"htf_bias:{htf_bias_label}"
        else:
            htf_bias_label = "no_h1_data"
            filter_reason  = "htf_bias:no_data"

    # 3. BOS confirmation
    if not filter_reason and cfg.require_bos and not bos_confirmed:
        filter_reason = "no_bos"

    # 4. FVG confirmation
    if not filter_reason and cfg.require_fvg and not fvg_confirmed:
        filter_reason = "no_fvg"

    # 5. Volume confirmation
    if not filter_reason and cfg.volume_filter_enabled and not vol_confirmed:
        filter_reason = "low_volume"

    # 6. Liquidity sweep
    if not filter_reason and cfg.liquidity_sweep_filter and not sweep_confirmed:
        filter_reason = "no_sweep"

    # 7. Regime filter
    regime_label = "none"
    if regime_result is not None:
        regime_label = regime_result.regime.value
        if not filter_reason and cfg.regime_filter_enabled:
            if cfg.skip_ranging and regime_result.is_ranging:
                filter_reason = f"regime:{regime_label}"
            elif cfg.skip_high_volatility and regime_result.is_high_vol:
                filter_reason = f"regime:{regime_label}"

    # ── Build signal row (all signals emitted, rejected ones flagged) ──────────
    tp = entry + dist * cfg.rr if side == "buy" else entry - dist * cfg.rr

    rows.append({
        "signal_bar_index"  : signal_bar,
        "signal_bar_time"   : ts,
        "side"              : side,
        "entry"             : entry,
        "sl"                : sl,
        "tp"                : tp,
        "qty"               : risk_cash / dist,
        "ob_type"           : ob.ob_type,
        # ── Filter diagnostics ─────────────────────────────────────────────
        "htf_bias"          : htf_bias_label,
        "bos_confirmed"     : bos_confirmed,
        "fvg_confirmed"     : fvg_confirmed,
        "vol_confirmed"     : vol_confirmed,
        "sweep_confirmed"   : sweep_confirmed,
        "session"           : session_label,
        "regime"            : regime_label,
        "filter_reason"     : filter_reason,   # empty = passed all filters
        "passed_all_filters": filter_reason == "",
    })


# ── H1 trend helpers ──────────────────────────────────────────────────────────

def _compute_h1_ema_trend(h1: pd.DataFrame) -> pd.Series:
    """
    Compute H1 EMA trend series using the existing crypto_core logic.

    Returns a Series indexed by H1 timestamps with values: 'bull' | 'bear' | 'flat'.
    """
    h1_with_emas = add_emas(h1)
    return h1_trend_series(h1_with_emas)


def _compute_h1_bos_trend(h1: pd.DataFrame) -> pd.Series:
    """
    Compute H1 BOS/MSS trend series.

    Returns a Series indexed by H1 timestamps with values: 'bullish' | 'bearish' | 'neutral'.
    """
    swings = detect_swing_points(h1, swing_len=5)
    breaks = detect_structure_breaks(h1, swings, lookback=50)

    trend: dict = {}
    for i, ts in enumerate(h1.index):
        prior = [b for b in breaks if b.break_bar_idx < i]
        if not prior:
            trend[ts] = "neutral"
        else:
            trend[ts] = prior[-1].direction  # "bullish" or "bearish"

    return pd.Series(trend)


# ── Convenience: signals that passed ALL active filters ───────────────────────

def get_passed_signals(signals: pd.DataFrame) -> pd.DataFrame:
    """Filter the enhanced signals DataFrame to only rows that passed all filters."""
    if signals.empty or "passed_all_filters" not in signals.columns:
        return signals
    return signals[signals["passed_all_filters"]].reset_index(drop=True)


# ── Backward-compatible wrapper ───────────────────────────────────────────────

def list_ob_signals_baseline(
    df: pd.DataFrame,
    *,
    risk_cash: float,
    rr: float = DEFAULT_RR,
    **kwargs,
) -> pd.DataFrame:
    """
    Backward-compatible wrapper that runs with ALL filters disabled.

    Produces identical output to ob_core.list_ob_signals() but in the
    enhanced DataFrame format with filter diagnostic columns.
    Used in notebooks to establish the baseline before adding filters.
    """
    cfg = OBSignalConfig(rr=rr, **kwargs)
    return list_ob_signals_enhanced(df, h1=None, risk_cash=risk_cash, config=cfg)
