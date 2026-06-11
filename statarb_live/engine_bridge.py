"""
The single, sanctioned gateway to the research engine (``notebooks/statarb/``).

Phase-5 discipline: live signals must be *bit-for-bit faithful* to the backtest, so we
do NOT re-implement any strategy maths. We import the exact functions the notebooks used
and call them with the exact frozen parameters. This module just makes that import robust:

  * adds the repo root to ``sys.path`` so ``import notebooks.statarb`` resolves;
  * re-exports the sub-modules the live layers need under stable names;
  * exposes :func:`build_signal_params`, :func:`build_size_params`, :func:`build_cost_model`
    constructed straight from :data:`statarb_live.config.STRATEGY`.

If the engine is missing (it was absent from git earlier in the project's history), import
fails loudly here with a clear remediation message — never silently degrade into a
re-implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

from .config import STRATEGY, repo_root

_ROOT = repo_root()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from notebooks.statarb import data as eng_data          # noqa: E402
    from notebooks.statarb import pairs as eng_pairs        # noqa: E402
    from notebooks.statarb import spread as eng_spread      # noqa: E402
    from notebooks.statarb import signals as eng_signals    # noqa: E402
    from notebooks.statarb import backtest as eng_backtest  # noqa: E402
    from notebooks.statarb import carry as eng_carry        # noqa: E402
    from notebooks.statarb import regime as eng_regime      # noqa: E402
    from notebooks.statarb import metrics as eng_metrics    # noqa: E402
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "statarb_live could not import the research engine 'notebooks.statarb'. "
        "The engine modules (pairs/backtest/carry/regime/...) must be present under "
        f"{_ROOT / 'notebooks' / 'statarb'} and 'scikit-learn' must be installed. "
        "Note: a stale 'notebooks/statarb/data/' package shadows 'data.py' — remove it. "
        f"Original error: {exc}"
    ) from exc


# ── Frozen-parameter constructors ───────────────────────────────────────────


def build_signal_params() -> "eng_backtest.SignalParams":
    return eng_backtest.SignalParams(
        z_entry=STRATEGY.z_entry,
        z_exit=STRATEGY.z_exit,
        z_stop=STRATEGY.z_stop,
        z_window=STRATEGY.z_window,
    )


def build_size_params() -> "eng_backtest.SizeParams":
    return eng_backtest.SizeParams(
        target_ann_vol=STRATEGY.target_ann_vol,
        vol_window=STRATEGY.vol_window,
        max_leverage=STRATEGY.max_leverage,
        timeframe=STRATEGY.bars_per_year_key,
    )


def build_cost_model(fee_bps: float, *, slippage_bps: float = 0.0,
                     exec_lag: int = 1) -> "eng_backtest.CostModel":
    """NB38 used CostModel(fee_bps=median quoted spread, slippage_bps=0, exec_lag=1)."""
    return eng_backtest.CostModel(
        fee_bps=fee_bps, slippage_bps=slippage_bps, exec_lag=exec_lag
    )


def bars_per_year() -> float:
    return float(eng_backtest.BARS_PER_YEAR[STRATEGY.bars_per_year_key])


__all__ = [
    "eng_data", "eng_pairs", "eng_spread", "eng_signals", "eng_backtest",
    "eng_carry", "eng_regime", "eng_metrics",
    "build_signal_params", "build_size_params", "build_cost_model", "bars_per_year",
]
