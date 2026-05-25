#!/usr/bin/env python
# coding: utf-8

# # Pairs Trading — Notebook 4: Portfolio Construction
# 
# نوت‌بوک ۲۷ نشون داد که در true walk-forward OOS، **۸۵% spreads در H4 سودده هستن** (median Sharpe 0.65). حالا می‌خواییم با ترکیب چند spread یه portfolio بسازیم که Sharpe بالاتر، DD کمتر و **concentration risk کنترل‌شده** داشته باشه.
# 
# ## فریم‌ورک
# 1. **Universe filter:** spreads با Sharpe_wf ≥ `MIN_SHARPE` (پیش‌فرض 0.5)
# 2. **Currency concentration cap:** هیچ ارز در بیش از `MAX_PER_CURRENCY` spread ظاهر نشه — جلوگیری از خطر NZD-concentration که در NB27 دیدیم
# 3. **Weighting schemes (مقایسه):**
#    - `equal` — وزن مساوی
#    - `risk_parity` — وزن ∝ 1/std(net_pnl)
#    - `sharpe_weighted` — وزن ∝ Sharpe (positive only)
# 4. **Portfolio backtest:** aggregation بر اساس trade (exit time-stamped)؛ equity curve = sum of weighted PnLs
# 5. **متریک‌ها:** annualized Sharpe، MaxDD، Calmar، monthly returns
# 
# **ورودی:** خروجی‌های NB27 (walkforward_trades_*.csv + walkforward_results_*.csv)
# **خروجی:** `portfolio_summary.csv` + `portfolio_equity_{H1,H4}.csv` + لیست spreads انتخابی

# In[ ]:


from __future__ import annotations
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

from stat_arb.identity.currency_graph import parse_symbol
from stat_arb.config import DEFAULT_KNOWN_CURRENCIES

pd.set_option("display.float_format", lambda x: f"{x:.4f}")
print("ready")


# ## Config

# In[ ]:


TIMEFRAMES         = ["H1", "H4"]
MIN_SHARPE         = 0.5         # universe filter: walk-forward Sharpe
MAX_PER_CURRENCY   = 3           # NZD-concentration cap (set to 2 for stricter)
MIN_TRADES         = 5           # require this many WF trades to enter portfolio
OOS_START          = "2024-01-01"   # first true-OOS bar (= train_months end of NB27)

STAT_DIR = PROJECT_ROOT / "notebooks" / "data" / "stat_arb"

print(f"universe filter: Sharpe_wf >= {MIN_SHARPE},  n_trades_wf >= {MIN_TRADES}")
print(f"concentration cap: max {MAX_PER_CURRENCY} spreads per currency")


# ## ۱) Load walk-forward results

# In[ ]:


metrics = {tf: pd.read_csv(STAT_DIR / f"walkforward_results_{tf}.csv") for tf in TIMEFRAMES}
trades  = {tf: pd.read_csv(STAT_DIR / f"walkforward_trades_{tf}.csv",
                            parse_dates=["entry_time", "exit_time"]) for tf in TIMEFRAMES}

for tf in TIMEFRAMES:
    print(f"[{tf}] metrics: {metrics[tf].shape}  |  trades: {trades[tf].shape}")
    print(f"  date range: {trades[tf]['exit_time'].min()} → {trades[tf]['exit_time'].max()}")


# ## ۲) Universe filter + concentration cap (greedy by Sharpe)

# In[ ]:


def currencies_of(y: str, x: str) -> set[str]:
    py = parse_symbol(y, DEFAULT_KNOWN_CURRENCIES)
    px = parse_symbol(x, DEFAULT_KNOWN_CURRENCIES)
    return {py.base, py.quote, px.base, px.quote}


def select_portfolio(
    metrics_df: pd.DataFrame,
    min_sharpe: float = MIN_SHARPE,
    min_trades: int = MIN_TRADES,
    max_per_currency: int = MAX_PER_CURRENCY,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Greedy: sort by Sharpe desc, accept spread if no currency exceeds cap."""
    univ = metrics_df[
        (metrics_df["sharpe"] >= min_sharpe) & (metrics_df["n_trades"] >= min_trades)
    ].sort_values("sharpe", ascending=False).reset_index(drop=True)
    counts: dict[str, int] = defaultdict(int)
    accepted_idx, rejected_idx = [], []
    for idx, row in univ.iterrows():
        ccys = currencies_of(row["y"], row["x"])
        if any(counts[c] >= max_per_currency for c in ccys):
            rejected_idx.append(idx)
        else:
            for c in ccys:
                counts[c] += 1
            accepted_idx.append(idx)
    return univ.loc[accepted_idx].reset_index(drop=True), \
           univ.loc[rejected_idx].reset_index(drop=True), \
           dict(counts)


selections: dict[str, dict] = {}
for tf in TIMEFRAMES:
    accepted, rejected, counts = select_portfolio(metrics[tf])
    selections[tf] = dict(accepted=accepted, rejected=rejected, ccy_counts=counts)
    print(f"\n=== [{tf}] portfolio selection ===")
    print(f"  universe (Sharpe>={MIN_SHARPE}, trades>={MIN_TRADES}): {len(accepted) + len(rejected)}")
    print(f"  accepted (after cap):                          {len(accepted)}")
    print(f"  rejected by cap:                               {len(rejected)}")
    print(f"  currency counts: {counts}")
    print(f"  --- accepted spreads ---")
    print(accepted[["y", "x", "sharpe", "total_pnl", "n_trades", "max_dd", "hit_rate"]].to_string(index=False))


# ## ۳) Compute weights (three schemes)

# In[ ]:


def weights_equal(accepted: pd.DataFrame) -> dict[str, float]:
    n = len(accepted)
    return {f"{r.y}~{r.x}": 1.0/n for _, r in accepted.iterrows()} if n else {}


def weights_risk_parity(accepted: pd.DataFrame, trades_df: pd.DataFrame) -> dict[str, float]:
    inv = {}
    for _, r in accepted.iterrows():
        key = f"{r.y}~{r.x}"
        sub = trades_df[(trades_df["y"] == r.y) & (trades_df["x"] == r.x)]
        sd = sub["net_pnl"].std()
        inv[key] = (1.0/sd) if sd and np.isfinite(sd) else 1.0
    s = sum(inv.values())
    return {k: v/s for k, v in inv.items()} if s else {}


def weights_sharpe(accepted: pd.DataFrame) -> dict[str, float]:
    w = {f"{r.y}~{r.x}": max(float(r.sharpe), 0.0) for _, r in accepted.iterrows()}
    s = sum(w.values())
    return {k: v/s for k, v in w.items()} if s else {}


schemes = ["equal", "risk_parity", "sharpe_weighted"]
weights_by_tf_scheme: dict[tuple[str, str], dict[str, float]] = {}

for tf in TIMEFRAMES:
    acc = selections[tf]["accepted"]
    weights_by_tf_scheme[(tf, "equal")]           = weights_equal(acc)
    weights_by_tf_scheme[(tf, "risk_parity")]     = weights_risk_parity(acc, trades[tf])
    weights_by_tf_scheme[(tf, "sharpe_weighted")] = weights_sharpe(acc)

    print(f"\n[{tf}] weights summary:")
    for sc in schemes:
        w = weights_by_tf_scheme[(tf, sc)]
        if w:
            wmax, wmin = max(w.values()), min(w.values())
            print(f"  {sc:<18s}  n={len(w)}  max={wmax:.3f}  min={wmin:.3f}  sum={sum(w.values()):.3f}")


# ## ۴) Portfolio backtest (per-trade aggregation)
# 
# **روش:** هر trade در portfolio با وزن متناظر spread‌ش ضرب می‌شه. equity curve = cumulative weighted PnL، time-indexed بر اساس exit_time.
# 
# **محدودیت‌ها:**
# - intra-trade DD دیده نمی‌شه (فقط در exits مارک می‌شیم)
# - اگه چند spread همزمان open هستن، هیچ position-netting نداریم (هر leg مستقل)
# - این حد پایین DD هست؛ DD واقعی mark-to-market کمی بدتر

# In[ ]:


def portfolio_equity(trades_df: pd.DataFrame, weights: dict[str, float]) -> tuple[pd.Series, pd.DataFrame]:
    if not weights:
        return pd.Series(dtype=float), pd.DataFrame()
    t = trades_df.copy()
    t["pair"] = t["y"] + "~" + t["x"]
    t = t[t["pair"].isin(weights.keys())].copy()
    t["weight"]       = t["pair"].map(weights)
    t["weighted_pnl"] = t["net_pnl"] * t["weight"]
    t = t.sort_values("exit_time").reset_index(drop=True)
    equity = t.set_index("exit_time")["weighted_pnl"].cumsum()
    return equity, t


def portfolio_metrics(equity: pd.Series, trades_pf: pd.DataFrame) -> dict:
    if equity.empty:
        return dict(n_trades=0, total_pnl=0.0, sharpe=0.0, max_dd=0.0,
                    calmar=0.0, hit_rate=0.0, avg_dur_days=0.0)
    pnl = trades_pf["weighted_pnl"]
    dd  = float((equity - equity.cummax()).min())
    # Annualize via trades-per-year (avg duration based)
    avg_dur = float(trades_pf["duration_days"].mean()) if len(trades_pf) else 1.0
    trades_per_year = 365.0 / max(avg_dur, 1.0)
    sharpe = (pnl.mean() / pnl.std()) * np.sqrt(trades_per_year) if pnl.std() > 0 else 0.0
    total = float(equity.iloc[-1])
    # Calmar = annualized return / |maxDD|
    years = (equity.index[-1] - equity.index[0]).total_seconds() / (365.25 * 24 * 3600)
    ann_return = total / max(years, 1e-6)
    calmar = ann_return / abs(dd) if dd < 0 else np.nan
    return dict(
        n_trades=int(len(trades_pf)),
        total_pnl=total,
        sharpe=float(sharpe),
        max_dd=dd,
        calmar=float(calmar),
        hit_rate=float((pnl > 0).mean()),
        avg_dur_days=avg_dur,
    )


summary_rows: list[dict] = []
portfolio_equities: dict[tuple[str, str], pd.Series] = {}
for tf in TIMEFRAMES:
    for sc in schemes:
        w = weights_by_tf_scheme[(tf, sc)]
        eq, t_pf = portfolio_equity(trades[tf], w)
        m = portfolio_metrics(eq, t_pf)
        portfolio_equities[(tf, sc)] = eq
        summary_rows.append(dict(timeframe=tf, scheme=sc, n_spreads=len(w), **m))

summary = pd.DataFrame(summary_rows)
for c in ("total_pnl", "max_dd"):
    summary[c + "_bps"] = summary[c] * 1e4
print("=== portfolio metrics (TRUE OOS) ===")
print(summary[["timeframe", "scheme", "n_spreads", "n_trades", "total_pnl_bps",
                "sharpe", "max_dd_bps", "calmar", "hit_rate", "avg_dur_days"]].to_string(index=False))


# ## ۵) Compare to single-best spread baseline

# In[ ]:


for tf in TIMEFRAMES:
    best = metrics[tf].sort_values("sharpe", ascending=False).iloc[0]
    print(f"[{tf}] best single spread:  {best['y']}~{best['x']}  Sharpe={best['sharpe']:.2f}  "
          f"PnL={best['total_pnl']*1e4:.0f}bps  maxDD={best['max_dd']*1e4:.0f}bps  "
          f"n_trades={int(best['n_trades'])}")

    for sc in schemes:
        m = summary[(summary["timeframe"] == tf) & (summary["scheme"] == sc)].iloc[0]
        print(f"     portfolio ({sc:<16s})    Sharpe={m['sharpe']:.2f}  "
              f"PnL={m['total_pnl_bps']:.0f}bps  maxDD={m['max_dd_bps']:.0f}bps  n_trades={int(m['n_trades'])}")
    print()


# ## ۶) Portfolio equity curves (per TF, all schemes overlaid)

# In[ ]:


for tf in TIMEFRAMES:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                        subplot_titles=("cumulative net PnL (bps)", "underwater (drawdown, bps)"),
                        vertical_spacing=0.08)
    for sc in schemes:
        eq = portfolio_equities[(tf, sc)]
        if eq.empty:
            continue
        fig.add_trace(go.Scatter(x=eq.index, y=eq.values * 1e4,
                                  mode="lines", name=sc), row=1, col=1)
        dd = (eq - eq.cummax()) * 1e4
        fig.add_trace(go.Scatter(x=dd.index, y=dd.values, mode="lines",
                                  name=f"{sc} DD", showlegend=False,
                                  line=dict(dash="dot")), row=2, col=1)
    fig.update_layout(title=f"[{tf}] portfolio equity & drawdown (true OOS)",
                      height=600, hovermode="x unified")
    fig.show()


# ## ۷) Monthly PnL heatmap (best scheme per TF)

# In[ ]:


for tf in TIMEFRAMES:
    # Pick highest-Sharpe scheme
    sub = summary[summary["timeframe"] == tf].sort_values("sharpe", ascending=False)
    best_scheme = sub.iloc[0]["scheme"]
    eq = portfolio_equities[(tf, best_scheme)]
    if eq.empty:
        continue
    monthly = eq.diff().resample("MS").sum() * 1e4
    monthly_df = monthly.to_frame("pnl")
    monthly_df["year"]  = monthly_df.index.year
    monthly_df["month"] = monthly_df.index.month
    pivot = monthly_df.pivot(index="year", columns="month", values="pnl")

    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=[f"{m:02d}" for m in pivot.columns], y=pivot.index.astype(str),
        colorscale="RdYlGn", zmid=0,
        text=np.round(pivot.values, 0).astype(int), texttemplate="%{text}", hovertemplate="%{y}-%{x}: %{z:.0f} bps",
    ))
    fig.update_layout(title=f"[{tf}] monthly PnL — best scheme = {best_scheme} (bps)",
                      xaxis_title="month", yaxis_title="year", height=350)
    fig.show()


# ## ۸) Save

# In[ ]:


summary_out = STAT_DIR / "portfolio_summary.csv"
summary.to_csv(summary_out, index=False)
print(f"summary -> {summary_out}")

for tf in TIMEFRAMES:
    acc = selections[tf]["accepted"][["y", "x", "sharpe", "total_pnl", "n_trades"]]
    acc_out = STAT_DIR / f"portfolio_selected_{tf}.csv"
    acc.to_csv(acc_out, index=False)
    print(f"  [{tf}] selected spreads -> {acc_out}")

    for sc in schemes:
        eq = portfolio_equities[(tf, sc)]
        if not eq.empty:
            eq_out = STAT_DIR / f"portfolio_equity_{tf}_{sc}.csv"
            eq.to_csv(eq_out, header=["cum_pnl"])
            print(f"  [{tf}] {sc} equity -> {eq_out}")

print("\nnext options (notebook 29):")
print("  A. Parameter sensitivity sweep on top portfolio spreads (entry_z, exit_z)")
print("  B. Continuous mark-to-market backtest (true DD with intra-trade markings)")
print("  C. Live MT5 execution skeleton — signal generator + position manager")

