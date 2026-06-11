"""Purged, embargoed walk-forward validation — the institutional rigor upgrade.

A single OOS split can get lucky (or unlucky) on one regime. Walk-forward re-runs the *whole*
selection+trade pipeline on a sequence of expanding-train / forward-test folds, so we see the
*distribution* of performance, not one number. Two anti-leakage controls (López de Prado):

  * **Purge / embargo gap** — a buffer of `embargo` bars is removed between the end of each
    train window and the start of its test window. Pair selection, hedge ratios and the
    z-score warm-up all use trailing data; without a gap, the bars at the boundary share
    information across the split (serial correlation + overlapping estimation windows). The
    gap severs that.
  * **Out-of-sample-only metrics** — every reported statistic is computed on test bars the
    selection never saw.

`fold_consistency` then asks the question that actually matters for capital allocation: not
"what was the best fold" but "is the edge *stable across folds* or does it depend on one
window?". A strategy with mean Sharpe 0.5 ± 1.5 across folds is not deployable; 0.3 ± 0.2 is.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from itertools import combinations

from scipy import stats

from .data import BARS_PER_YEAR
from .metrics import perf_stats, exposure_stats


@dataclass
class Fold:
    i: int
    train: pd.DatetimeIndex
    test: pd.DatetimeIndex


def walk_forward_folds(index: pd.DatetimeIndex, *, n_folds: int = 6,
                       train_min_frac: float = 0.35, embargo: int = 168,
                       expanding: bool = True) -> list[Fold]:
    """Build expanding (or rolling) walk-forward folds with an embargo gap.

    The first `train_min_frac` of the sample seeds the initial train window; the remainder is
    cut into `n_folds` equal forward test blocks. Train end and test start are separated by
    `embargo` bars (default 1 week of H1).
    """
    n = len(index)
    start_test = int(n * train_min_frac)
    test_len = (n - start_test) // n_folds
    folds = []
    for i in range(n_folds):
        t0 = start_test + i * test_len
        t1 = t0 + test_len if i < n_folds - 1 else n
        train_end = t0 - embargo
        if train_end <= 0 or t1 <= t0:
            continue
        train_start = 0 if expanding else max(0, train_end - (start_test - embargo))
        folds.append(Fold(i=i, train=index[train_start:train_end], test=index[t0:t1]))
    return folds


def run_walk_forward(px: pd.DataFrame, select_fn, backtest_fn, *,
                     n_folds: int = 6, train_min_frac: float = 0.35, embargo: int = 168,
                     timeframe: str = "H1", expanding: bool = True) -> pd.DataFrame:
    """Run a full walk-forward.

    `select_fn(px_train) -> selection`   : chooses pairs/baskets on train data only.
    `backtest_fn(px_test, selection) -> (returns: Series, turnover: Series)` : trades on test.

    Returns one row of OOS stats per fold. Folds with an empty selection are recorded as flat
    (Sharpe NaN, n_pairs 0) rather than skipped — a method that finds nothing in a window is a
    real (bad) outcome, not a missing data point.
    """
    folds = walk_forward_folds(px.index, n_folds=n_folds, train_min_frac=train_min_frac,
                               embargo=embargo, expanding=expanding)
    rows = []
    for f in folds:
        px_tr, px_te = px.loc[f.train], px.loc[f.test]
        sel = select_fn(px_tr)
        n_sel = len(sel) if hasattr(sel, "__len__") else np.nan
        if not n_sel:
            rows.append({"fold": f.i, "test_start": f.test[0], "test_end": f.test[-1],
                         "n_pairs": 0, "sharpe": np.nan, "sortino": np.nan, "calmar": np.nan,
                         "max_dd": np.nan, "cagr": np.nan, "hit_rate": np.nan,
                         "ann_turnover": np.nan, "exposure": np.nan})
            continue
        ret, turn = backtest_fn(px_te, sel)
        s = perf_stats(ret, timeframe=timeframe)
        ex = (exposure_stats(turn, timeframe=timeframe) if turn is not None
              else {"ann_turnover": np.nan, "active_frac": np.nan})
        rows.append({"fold": f.i, "test_start": f.test[0], "test_end": f.test[-1],
                     "n_pairs": n_sel, "sharpe": s["sharpe"], "sortino": s["sortino"],
                     "calmar": s["calmar"], "max_dd": s["max_dd"], "cagr": s["cagr"],
                     "hit_rate": s["hit_rate"], "ann_turnover": ex["ann_turnover"],
                     "exposure": ex.get("active_frac", np.nan)})
    return pd.DataFrame(rows)


def deflated_sharpe(returns: pd.Series, *, n_trials: int, trial_sharpes=None,
                    timeframe: str = "H1") -> dict:
    """Deflated Sharpe Ratio (Bailey & López de Prado 2014). The observed Sharpe is inflated by
    (a) the number of strategy variants tried and (b) non-normal returns. DSR is the probability
    the *true* Sharpe exceeds the benchmark you'd expect from `n_trials` random tries — i.e. it
    discounts selection / multiple-testing bias, the thing that makes most backtests lie.

    `trial_sharpes` (annualised Sharpes of all variants tried) sets the benchmark via its
    variance; if absent we use a default spread. DSR < 0.95 => not convincingly better than luck.
    """
    r = returns.dropna()
    n = len(r)
    if n < 10 or r.std() == 0:
        return {"dsr": np.nan, "benchmark_sr_ann": np.nan, "observed_sr_ann": np.nan}
    ppy = BARS_PER_YEAR[timeframe]
    sr = r.mean() / r.std()                                  # per-bar Sharpe
    sk = stats.skew(r); ku = stats.kurtosis(r, fisher=False)
    gamma = 0.5772156649                                     # Euler-Mascheroni
    if trial_sharpes is not None and len(trial_sharpes) > 1:
        var_sr = np.var(np.asarray(trial_sharpes) / np.sqrt(ppy))
    else:
        var_sr = (0.5 / np.sqrt(ppy)) ** 2                   # default: ~0.5 annual SR dispersion
    N = max(int(n_trials), 1)
    if N > 1 and var_sr > 0:
        sr0 = np.sqrt(var_sr) * ((1 - gamma) * stats.norm.ppf(1 - 1.0 / N)
                                 + gamma * stats.norm.ppf(1 - 1.0 / (N * np.e)))
    else:
        sr0 = 0.0
    den = np.sqrt(1 - sk * sr + (ku - 1) / 4 * sr ** 2)
    dsr = float(stats.norm.cdf((sr - sr0) * np.sqrt(n - 1) / den)) if den > 0 else np.nan
    return {"dsr": dsr, "benchmark_sr_ann": float(sr0 * np.sqrt(ppy)),
            "observed_sr_ann": float(sr * np.sqrt(ppy))}


def cpcv(px: pd.DataFrame, select_fn, backtest_fn, *, n_groups: int = 6, k_test: int = 2,
         embargo: int = 168, timeframe: str = "H1") -> tuple[pd.DataFrame, dict]:
    """Combinatorial Purged Cross-Validation (López de Prado). Split the timeline into
    `n_groups` blocks; for every combination of `k_test` test blocks, train on the rest (purged
    of bars within `embargo` of any test block) and evaluate on each held-out block. This yields
    C(n_groups, k_test) train sets and many more *test paths* than sequential walk-forward, so
    the Sharpe distribution is far harder to fluke. Returns (per-evaluation table, summary)."""
    idx = px.index
    n = len(idx)
    edges = np.linspace(0, n, n_groups + 1).astype(int)
    blocks = [idx[edges[i]:edges[i + 1]] for i in range(n_groups)]
    rows = []
    for combo in combinations(range(n_groups), k_test):
        test_pos = set()
        for g in combo:
            test_pos |= set(range(edges[g], edges[g + 1]))
        # purge: drop train bars within `embargo` of any test block boundary
        purged = set()
        for g in combo:
            purged |= set(range(max(0, edges[g] - embargo), min(n, edges[g + 1] + embargo)))
        train_pos = [i for i in range(n) if i not in purged]
        if len(train_pos) < 2000:
            continue
        sel = select_fn(px.iloc[train_pos])
        if not (len(sel) if hasattr(sel, "__len__") else 0):
            for g in combo:
                rows.append({"combo": combo, "test_block": g, "sharpe": np.nan, "n_pairs": 0})
            continue
        for g in combo:
            ret, _ = backtest_fn(px.loc[blocks[g]], sel)
            rows.append({"combo": combo, "test_block": g,
                         "sharpe": perf_stats(ret, timeframe=timeframe)["sharpe"],
                         "n_pairs": len(sel)})
    tab = pd.DataFrame(rows)
    sh = tab["sharpe"].dropna()
    summary = {"n_paths": len(sh), "mean_sharpe": float(sh.mean()) if len(sh) else np.nan,
               "std_sharpe": float(sh.std()) if len(sh) else np.nan,
               "frac_positive": float((sh > 0).mean()) if len(sh) else np.nan,
               "p05": float(sh.quantile(0.05)) if len(sh) else np.nan,
               "p95": float(sh.quantile(0.95)) if len(sh) else np.nan}
    return tab, summary


def fold_consistency(wf: pd.DataFrame) -> dict:
    """Summarise a walk-forward result the way an allocator reads it."""
    sh = wf["sharpe"].dropna()
    if len(sh) == 0:
        return {"mean_sharpe": np.nan, "std_sharpe": np.nan, "frac_positive": np.nan,
                "worst_fold": np.nan, "best_fold": np.nan, "info_ratio_of_folds": np.nan,
                "n_folds": 0}
    return {
        "mean_sharpe": float(sh.mean()),
        "std_sharpe": float(sh.std()),
        "frac_positive": float((sh > 0).mean()),
        "worst_fold": float(sh.min()),
        "best_fold": float(sh.max()),
        # stability score: mean/std of fold Sharpes — high only if the edge is consistent
        "info_ratio_of_folds": float(sh.mean() / sh.std()) if sh.std() > 0 else np.nan,
        "n_folds": int(len(sh)),
    }
