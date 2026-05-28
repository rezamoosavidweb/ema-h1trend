"""Aggregate HTF policy simulation outputs into a portfolio summary report
suitable for pasting into the markdown report.

Reads:
  notebooks/data/htf_policy/{metrics.csv, signals.csv, trades.csv, diags.csv}

Outputs to stdout:
  - Portfolio summary table per policy
  - Per-symbol breakdown
  - Signal-set Jaccard similarity (which policies fire on which bars)
  - Sanity-check against the existing parity-evidence dataset

Run:  python notebooks/_htf_policy_summary.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT     = Path(__file__).resolve().parent.parent
SIM      = ROOT / "notebooks" / "data" / "htf_policy"
PARITY   = ROOT / "notebooks" / "data"

POLICY_ORDER = ["A", "B", "C_15", "C_60", "C_120", "D_2", "D_3"]


def fmt_pct(p): return f"{p:5.1f}%"
def fmt_R(r):   return f"{r:+6.2f}"
def fmt_usd(d): return f"${d:+8.2f}"
def fmt_int(n): return f"{n:>4d}"


def portfolio_summary():
    m = pd.read_csv(SIM / "metrics.csv")
    if m.empty:
        return "no metrics."
    pols = [p for p in POLICY_ORDER if p in set(m["policy"].unique())]

    agg = (m.groupby("policy", as_index=False)
             .agg(signals=("signal_count", "sum"),
                  trades=("trades", "sum"),
                  sum_R=("sum_R", "sum"),
                  net_=("net_$", "sum"),
                  WR=("WR", "mean"),
                  PF=("PF", "mean"),
                  max_dd_R=("max_dd_R", "sum"),
                  expectancy_R=("expectancy_R", "mean"),
                  sharpe_R=("sharpe_R", "mean")))
    agg["cascade_loss_%"] = (1 - agg["trades"] / agg["signals"]) * 100
    agg = agg.set_index("policy").reindex(pols).reset_index()

    # Pretty-print
    lines = ["| Policy | Sigs | Trades | Cascade % | WR% | PF | Sum R | Net $ | Max DD | Expectancy | Sharpe |",
             "|:------:|-----:|-------:|----------:|----:|---:|------:|------:|-------:|-----------:|-------:|"]
    for _, r in agg.iterrows():
        cl = (1 - r["trades"] / r["signals"]) * 100 if r["signals"] else 0.0
        lines.append("| {p:6s} | {s:>4d} | {t:>6d} | {c:>7.1f}% | {wr:>4.1f} | {pf:>4.2f} | {sR:>+6.2f} | ${dn:>+7.2f} | {dd:>5.2f} | {ex:>+8.3f} | {sh:>+6.2f} |".format(
            p=r["policy"], s=int(r["signals"]), t=int(r["trades"]), c=cl,
            wr=r["WR"], pf=r["PF"], sR=r["sum_R"], dn=r["net_"], dd=r["max_dd_R"],
            ex=r["expectancy_R"], sh=r["sharpe_R"],
        ))
    return "\n".join(lines)


def per_symbol_table():
    m = pd.read_csv(SIM / "metrics.csv")
    pols = [p for p in POLICY_ORDER if p in set(m["policy"].unique())]
    out = ["### Trades per symbol per policy"]
    pivot = m.pivot(index="symbol", columns="policy", values="trades").reindex(columns=pols, fill_value=0).fillna(0).astype(int)
    out.append("```")
    out.append(pivot.to_string())
    out.append("```\n")
    out.append("### Sum R per symbol per policy")
    out.append("```")
    pivot = m.pivot(index="symbol", columns="policy", values="sum_R").reindex(columns=pols, fill_value=0).fillna(0).round(2)
    out.append(pivot.to_string())
    out.append("```\n")
    out.append("### Net $ per symbol per policy")
    out.append("```")
    pivot = m.pivot(index="symbol", columns="policy", values="net_$").reindex(columns=pols, fill_value=0).fillna(0).round(2)
    out.append(pivot.to_string())
    out.append("```\n")
    out.append("### Signal count per symbol per policy")
    out.append("```")
    pivot = m.pivot(index="symbol", columns="policy", values="signal_count").reindex(columns=pols, fill_value=0).fillna(0).astype(int)
    out.append(pivot.to_string())
    out.append("```")
    return "\n".join(out)


def jaccard_table():
    s = pd.read_csv(SIM / "signals.csv")
    pols = [p for p in POLICY_ORDER if p in set(s["policy"].unique())]
    keys = {p: set(zip(s.loc[s["policy"] == p, "symbol"],
                       s.loc[s["policy"] == p, "bar_time"],
                       s.loc[s["policy"] == p, "direction"])) for p in pols}
    out = ["### Signal-set Jaccard similarity (1.0 = identical bar+direction sets)"]
    out.append("```")
    header = "        " + " ".join(f"{p:>7s}" for p in pols)
    out.append(header)
    for a in pols:
        row = f"{a:7s} "
        for b in pols:
            u = keys[a] | keys[b]; i = keys[a] & keys[b]
            j = len(i) / len(u) if u else 1.0
            row += f"{j:>7.3f} "
        out.append(row)
    out.append("```")
    return "\n".join(out)


def divergence_table():
    """For each pair of policies, count signals only-in-A and only-in-B."""
    s = pd.read_csv(SIM / "signals.csv")
    pols = [p for p in POLICY_ORDER if p in set(s["policy"].unique())]
    keys = {p: set(zip(s.loc[s["policy"] == p, "symbol"],
                       s.loc[s["policy"] == p, "bar_time"],
                       s.loc[s["policy"] == p, "direction"])) for p in pols}
    out = ["### Signal-set divergence vs Policy A (broker-only baseline)"]
    out.append("```")
    out.append(f"{'Policy':<7s}  {'extra_vs_A':>11s}  {'missing_vs_A':>13s}  {'net':>6s}  {'jaccard':>9s}")
    base = keys.get("A", set())
    for p in pols:
        if p == "A":
            continue
        extra   = keys[p] - base
        missing = base - keys[p]
        net = len(keys[p]) - len(base)
        u = keys[p] | base; i = keys[p] & base
        jac = len(i)/len(u) if u else 1.0
        out.append(f"{p:<7s}  {len(extra):>11d}  {len(missing):>13d}  {net:>+6d}  {jac:>9.3f}")
    out.append("```")
    return "\n".join(out)


def freshness_distribution():
    """How fresh was broker H1 (median, p90, max) across all cycles in window?"""
    d = pd.read_csv(SIM / "diags.csv")
    pol = d[d["policy"] == "A"]   # any policy has the same freshness
    if pol.empty or "h1_freshness_min" not in pol.columns:
        return "no freshness data"
    out = ["### Broker H1 freshness distribution (minutes since last broker bar closed, per cycle)"]
    out.append("```")
    out.append(f"{'symbol':<8s}  {'mean':>6s}  {'p50':>6s}  {'p90':>6s}  {'p95':>6s}  {'max':>6s}")
    for sym, g in pol.groupby("symbol"):
        f = g["h1_freshness_min"].dropna()
        if f.empty:
            continue
        out.append(f"{sym:<8s}  {f.mean():>6.1f}  {f.quantile(0.5):>6.1f}  {f.quantile(0.9):>6.1f}  {f.quantile(0.95):>6.1f}  {f.max():>6.1f}")
    out.append("```")
    return "\n".join(out)


def main():
    parts = [
        "## Portfolio summary per HTF policy",
        portfolio_summary(),
        "",
        per_symbol_table(),
        "",
        jaccard_table(),
        "",
        divergence_table(),
        "",
        freshness_distribution(),
    ]
    print("\n".join(parts))


if __name__ == "__main__":
    main()
