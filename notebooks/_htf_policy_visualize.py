"""Generate trend-flip timelines and policy-comparison visualizations.

Outputs PNG files into notebooks/data/htf_policy/figs/
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT     = Path(__file__).resolve().parent.parent
SIM_DIR  = ROOT / "notebooks" / "data" / "htf_policy"
FIG_DIR  = SIM_DIR / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def fig_metrics_grid():
    """4-up bar chart: trades / sum_R / WR / max_dd per (symbol, policy)."""
    m = pd.read_csv(SIM_DIR / "metrics.csv")
    syms = sorted(m["symbol"].unique())
    pols = ["A", "B", "C_15", "C_60", "C_120", "D_2", "D_3"]
    pols = [p for p in pols if p in set(m["policy"].unique())]
    if not pols:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    metrics_to_plot = [("trades", "Trades"),
                       ("sum_R", "Sum R"),
                       ("WR",    "Win rate %"),
                       ("max_dd_R", "Max drawdown (R)")]

    for ax, (col, title) in zip(axes.flat, metrics_to_plot):
        bottoms = np.arange(len(syms))
        bar_width = 0.85 / max(1, len(pols))
        for j, p in enumerate(pols):
            vals = [float(m[(m["symbol"] == s) & (m["policy"] == p)][col].iloc[0])
                    if not m[(m["symbol"] == s) & (m["policy"] == p)].empty else 0.0
                    for s in syms]
            ax.bar(bottoms + j * bar_width - 0.425, vals, bar_width, label=p)
        ax.set_xticks(bottoms)
        ax.set_xticklabels(syms, rotation=20)
        ax.set_title(title)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.suptitle("HTF Policy comparison — per-symbol metrics")
    plt.tight_layout()
    out = FIG_DIR / "metrics_grid.png"
    plt.savefig(out, dpi=110)
    plt.close()
    print(f"saved {out}", flush=True)


def fig_signal_funnel():
    """Show signal_count vs trade_count per policy, aggregated."""
    m = pd.read_csv(SIM_DIR / "metrics.csv")
    agg = m.groupby("policy", as_index=False).agg(
        signals=("signal_count", "sum"),
        trades=("trades", "sum"),
        sum_R=("sum_R", "sum"),
        net_=("net_$", "sum"),
    )
    pols = ["A", "B", "C_15", "C_60", "C_120", "D_2", "D_3"]
    agg = agg.set_index("policy").loc[[p for p in pols if p in agg.index]].reset_index()

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(agg))
    ax.bar(x - 0.2, agg["signals"], 0.4, label="Signals fired")
    ax.bar(x + 0.2, agg["trades"],  0.4, label="Trades placed (after 1-trade-at-a-time)")
    for i, (s, t) in enumerate(zip(agg["signals"], agg["trades"])):
        ax.text(i - 0.2, s + 1, str(int(s)), ha="center", fontsize=9)
        ax.text(i + 0.2, t + 1, str(int(t)), ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(agg["policy"])
    ax.set_title("Portfolio signal funnel by HTF policy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = FIG_DIR / "signal_funnel.png"
    plt.tight_layout(); plt.savefig(out, dpi=110); plt.close()
    print(f"saved {out}", flush=True)


def fig_trend_flip_timeline(symbol: str):
    """Per-symbol: timeline showing when each policy says trend_dir != 0."""
    p = SIM_DIR / "diags.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    d = d[d["symbol"] == symbol].copy()
    if d.empty:
        return
    d["bar_time"] = pd.to_datetime(d["bar_time"])
    d = d.sort_values(["policy", "bar_time"])

    pols = ["A", "B", "C_15", "C_60", "C_120", "D_2", "D_3"]
    pols = [p for p in pols if p in set(d["policy"].unique())]
    if not pols:
        return

    fig, axes = plt.subplots(len(pols), 1, figsize=(15, 1.6 * len(pols)),
                              sharex=True)
    if len(pols) == 1:
        axes = [axes]

    for ax, pol in zip(axes, pols):
        sub = d[d["policy"] == pol]
        if "trend_dir" not in sub.columns:
            continue
        ax.scatter(sub["bar_time"], sub["trend_dir"], s=4, c="C0", alpha=0.6)
        ax.set_ylabel(pol)
        ax.set_yticks([-1, 0, 1])
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("bar_time")
    plt.suptitle(f"{symbol}: trend_dir per HTF policy over window")
    plt.tight_layout()
    out = FIG_DIR / f"trend_timeline_{symbol}.png"
    plt.savefig(out, dpi=110)
    plt.close()
    print(f"saved {out}", flush=True)


def main():
    fig_metrics_grid()
    fig_signal_funnel()
    # Pick symbols with most signal activity
    m = pd.read_csv(SIM_DIR / "metrics.csv")
    top = (m.groupby("symbol")["signal_count"].sum()
             .sort_values(ascending=False).head(3).index.tolist())
    for s in top:
        fig_trend_flip_timeline(s)
    print("done", flush=True)


if __name__ == "__main__":
    main()
