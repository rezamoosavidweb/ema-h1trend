"""Regime detection — volatility regimes and a Markov-switching (HMM) model (Section F).

Two ways to label the market state, both used to ask *where does the mean-reversion edge live
and where does it die*:

  * `vol_regime` (re-exported from `factor`) — simple expanding-quantile vol buckets.
  * `markov_regimes` — a 2/3-state Markov-switching model on returns with switching variance
    (statsmodels). This is the econometric cousin of a Gaussian HMM: it infers hidden states
    and the transition matrix. Crucially we expose **filtered** (causal, point-in-time) state
    probabilities for anything that could become a trading rule, and **smoothed** (full-sample)
    only for descriptive labelling — using smoothed states to gate trades would be look-ahead.

The output is consumed by `conditional_performance`, which slices a strategy's return series by
regime and reports Sharpe / drawdown / activity per state — the basis for a "switch the book
off in regime X" decision.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .data import BARS_PER_YEAR
from .factor import vol_regime  # noqa: F401  (re-export)
from .metrics import perf_stats


def markov_regimes(px: pd.Series, *, k_regimes: int = 2, timeframe: str = "H1",
                   downsample: int = 4) -> dict:
    """Fit a Markov-switching variance model on (optionally downsampled) log returns.

    Returns hard `labels` (argmax of FILTERED probs — causal), the filtered/smoothed
    probability frames, and the per-state annualised vol. States are re-ordered by volatility
    so 0=calmest .. k-1=wildest, making the labels comparable across runs. We downsample to
    e.g. 4h to keep the EM fit stable and fast on long H1 samples (documented; the regime is a
    slow-moving object so 4h labelling is plenty)."""
    r = np.log(px).diff().dropna()
    if downsample > 1:
        r_fit = r.iloc[::downsample]
    else:
        r_fit = r
    mod = sm.tsa.MarkovRegression(r_fit.values, k_regimes=k_regimes, trend="c",
                                  switching_variance=True)
    res = mod.fit(disp=False)

    filt_raw = pd.DataFrame(res.filtered_marginal_probabilities, index=r_fit.index)
    smoo_raw = pd.DataFrame(res.smoothed_marginal_probabilities, index=r_fit.index)
    lab_raw = filt_raw.idxmax(axis=1)

    # Order states by *empirical* return volatility of the bars assigned to each raw state
    # (robust — avoids fragile statsmodels param-name indexing). 0 = calmest .. k-1 = wildest.
    emp_vol = {s: r_fit[lab_raw == s].std() for s in range(k_regimes)}
    order = sorted(range(k_regimes), key=lambda s: emp_vol[s])
    remap = {old: new for new, old in enumerate(order)}

    filt = filt_raw[order]; filt.columns = range(k_regimes)
    smoo = smoo_raw[order]; smoo.columns = range(k_regimes)

    labels_ds = lab_raw.map(remap)
    # reindex labels back to the full (H1) series by forward-fill (state persists between
    # downsampled observations) — still causal.
    labels = labels_ds.reindex(px.index).ffill()
    ann = np.sqrt(BARS_PER_YEAR[timeframe])
    state_vol = pd.Series([emp_vol[order[i]] * ann * np.sqrt(downsample) for i in range(k_regimes)],
                          index=range(k_regimes), name="ann_vol")
    return {"labels": labels, "filtered": filt, "smoothed": smoo,
            "state_vol": state_vol, "transition": res.regime_transition, "result": res}


def trend_regime(px: pd.Series, *, fast: int = 168, slow: int = 720) -> pd.Series:
    """Trend regime from a fast-vs-slow moving-average cross on a market proxy: 'up' when the
    fast MA is above the slow MA, 'down' otherwise. Both MAs are trailing, so the label is
    causal. Mean-reversion strategies typically suffer in strong trends — this lets us test
    that directly (label 1=up-trend, 0=down-trend)."""
    f = px.rolling(fast, min_periods=fast // 2).mean()
    s = px.rolling(slow, min_periods=slow // 2).mean()
    out = (f > s).astype(float)
    out[f.isna() | s.isna()] = np.nan
    return out


def regime_size_multiplier(px: pd.Series, *, method: str = "hmm", min_mult: float = 0.0,
                           max_mult: float = 1.5, vol_window: int = 168, target_vol: float = 0.10,
                           timeframe: str = "H1", k_regimes: int = 2, downsample: int = 6) -> pd.Series:
    """A *continuous*, causal exposure multiplier — scale up when the regime is favourable, down
    in crisis — instead of a hard on/off gate. A gate throws away information at the threshold and
    whipsaws; a smooth multiplier sizes proportionally to conviction.

      method="hmm"  -> multiplier rises with the FILTERED probability of the calm state.
      method="vol"  -> inverse-vol target (clip(target_vol / trailing_vol)), the classic lever.
    Both use only trailing/filtered data, so applying mult.shift(1) to returns is look-ahead-free.
    """
    if method == "vol":
        rv = np.log(px).diff().rolling(vol_window, min_periods=vol_window // 2).std() \
            * np.sqrt(BARS_PER_YEAR[timeframe])
        mult = (target_vol / rv).clip(min_mult, max_mult)
        return mult.reindex(px.index).ffill().fillna(min_mult)
    mk = markov_regimes(px, k_regimes=k_regimes, downsample=downsample, timeframe=timeframe)
    p_calm = mk["filtered"][0].reindex(px.index).ffill()     # state 0 = calmest (ordered)
    mult = min_mult + (max_mult - min_mult) * p_calm
    return mult.fillna(min_mult)


def conditional_performance(returns: pd.Series, labels: pd.Series, *,
                            timeframe: str = "H1", min_bars: int = 100) -> pd.DataFrame:
    """Sharpe / drawdown / activity of a return series conditioned on regime labels. Answers:
    which regimes produce profit, which destroy alpha, and how much time is spent in each."""
    lab = labels.reindex(returns.index).ffill()
    rows = {}
    for st in sorted(pd.unique(lab.dropna())):
        mask = (lab == st)
        rr = returns[mask]
        if mask.sum() < min_bars:
            continue
        s = perf_stats(rr, timeframe=timeframe)
        rows[f"state_{int(st)}"] = {"n_bars": int(mask.sum()), "time_frac": float(mask.mean()),
                                    "sharpe": s["sharpe"], "cagr": s["cagr"],
                                    "max_dd": s["max_dd"], "hit_rate": s["hit_rate"]}
    return pd.DataFrame(rows).T
