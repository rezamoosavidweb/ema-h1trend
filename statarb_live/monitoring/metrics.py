"""Performance metrics + attribution computed from the storage tables."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from ..engine_bridge import bars_per_year, eng_metrics
from ..storage.base import Storage


@dataclass
class PerformanceMetrics:
    window: str
    period_start: datetime | None
    period_end: datetime | None
    pnl: float = 0.0
    return_pct: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    turnover: float = 0.0
    n_trades: int = 0
    avg_bars_held: float = 0.0
    extra: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        return {
            "window": self.window, "period_start": self.period_start,
            "period_end": self.period_end, "pnl": self.pnl, "return_pct": self.return_pct,
            "sharpe": self.sharpe, "sortino": self.sortino, "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate, "turnover": self.turnover, "n_trades": self.n_trades,
            "extra": self.extra,
        }


def _utc_ts(dt) -> pd.Timestamp | None:
    """Coerce a datetime (naive or aware) to a UTC-aware Timestamp."""
    if dt is None:
        return None
    ts = pd.Timestamp(dt)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def equity_dataframe(storage: Storage) -> pd.DataFrame:
    df = storage.fetch_df("equity")
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.sort_values("ts").set_index("ts")


def trades_dataframe(storage: Storage) -> pd.DataFrame:
    df = storage.fetch_df("trades")
    if df.empty:
        return df
    for c in ("entry_ts", "exit_ts", "signal_ts"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True)
    return df.sort_values("exit_ts")


def _drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = (equity / peak - 1.0)
    return float(dd.min())


def compute_metrics(storage: Storage, *, window: str = "all",
                    start: datetime | None = None, end: datetime | None = None,
                    starting_equity: float = 100_000.0) -> PerformanceMetrics:
    """Compute headline metrics over an optional [start, end) window."""
    eq = equity_dataframe(storage)
    tr = trades_dataframe(storage)
    start_ts = _utc_ts(start)
    end_ts = _utc_ts(end)
    if start_ts is not None and not eq.empty:
        eq = eq[eq.index >= start_ts]
    if end_ts is not None and not eq.empty:
        eq = eq[eq.index < end_ts]
    if start_ts is not None and not tr.empty:
        tr = tr[tr["exit_ts"] >= start_ts]
    if end_ts is not None and not tr.empty:
        tr = tr[tr["exit_ts"] < end_ts]

    m = PerformanceMetrics(window=window, period_start=start, period_end=end)
    if not eq.empty:
        equity_series = eq["equity"].astype(float)
        rets = equity_series.pct_change().dropna()
        base = float(equity_series.iloc[0]) or starting_equity
        m.pnl = float(equity_series.iloc[-1] - equity_series.iloc[0])
        m.return_pct = float(equity_series.iloc[-1] / base - 1.0) if base else 0.0
        m.max_drawdown = _drawdown(equity_series)
        if len(rets) > 2:
            try:
                stats = eng_metrics.perf_stats(rets, timeframe="FX_H1")
                m.sharpe = float(stats.get("sharpe", 0.0))
                m.sortino = float(stats.get("sortino", 0.0))
            except Exception:
                std = rets.std()
                m.sharpe = float(rets.mean() / std * np.sqrt(bars_per_year())) if std else 0.0
        if "leverage" in eq.columns:
            m.extra["avg_leverage"] = float(eq["leverage"].mean())
        if "regime_multiplier" in eq.columns:
            m.extra["avg_regime_mult"] = float(eq["regime_multiplier"].mean())
    if not tr.empty:
        pnl = tr["realized_pnl"].astype(float)
        m.n_trades = int(len(tr))
        m.win_rate = float((pnl > 0).mean())
        m.turnover = float(tr["gross_notional"].astype(float).sum())
        if "bars_held" in tr.columns:
            m.avg_bars_held = float(tr["bars_held"].astype(float).mean())
        m.extra.update({
            "reversion_pnl": float(tr.get("reversion_pnl", pd.Series(dtype=float)).sum()),
            "carry_pnl": float(tr.get("carry_pnl", pd.Series(dtype=float)).sum()),
            "cost_pnl": float(tr.get("cost_pnl", pd.Series(dtype=float)).sum()),
            "gross_trade_pnl": float(pnl.sum()),
        })
    return m


# ── attribution ──────────────────────────────────────────────────────────────
def _group_pnl(tr: pd.DataFrame, key: str) -> pd.DataFrame:
    if tr.empty or key not in tr.columns:
        return pd.DataFrame()
    g = tr.groupby(key).agg(
        realized_pnl=("realized_pnl", "sum"),
        reversion_pnl=("reversion_pnl", "sum"),
        carry_pnl=("carry_pnl", "sum"),
        cost_pnl=("cost_pnl", "sum"),
        n_trades=("realized_pnl", "size"),
        win_rate=("realized_pnl", lambda s: float((s > 0).mean())),
    ).sort_values("realized_pnl", ascending=False)
    return g


def attribution_by_pair(storage: Storage) -> pd.DataFrame:
    return _group_pnl(trades_dataframe(storage), "pair_key")


def attribution_by_regime(storage: Storage) -> pd.DataFrame:
    return _group_pnl(trades_dataframe(storage), "regime")


def attribution_by_sleeve(storage: Storage) -> pd.DataFrame:
    """PnL split into the two sleeves (reversion pairs vs carry holdings) + cost."""
    tr = trades_dataframe(storage)
    if tr.empty:
        return pd.DataFrame()
    tr = tr.copy()
    tr["sleeve"] = tr["pair_key"].apply(lambda k: "carry" if str(k).startswith("CARRY:") else "reversion")
    return _group_pnl(tr, "sleeve")
