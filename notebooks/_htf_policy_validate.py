"""Sanity-check simulation outputs against ground truth.

1) Policy A (broker-only) signals should approximate the BT diagnostics in
   parity_per_bar_diff.csv (where signal_dir from NB31's BT side != 0).
2) Policy B (synth current) signals should approximate the LIVE
   `event=signal` events in the per-symbol JSON logs.

Discrepancies are NOT necessarily bugs (different windows, different
warmup, slight numerical differences), but >50% disagreement would
indicate a structural mismatch.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT       = Path(__file__).resolve().parent.parent
SIM_DIR    = ROOT / "notebooks" / "data" / "htf_policy"
LOG_DIR    = ROOT / "logs"
PARITY_DIR = ROOT / "notebooks" / "data"


def load_sim_signals() -> pd.DataFrame:
    p = SIM_DIR / "signals.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    s = pd.read_csv(p)
    s["bar_time"] = pd.to_datetime(s["bar_time"])
    return s


def load_live_signals_from_logs() -> pd.DataFrame:
    rows = []
    for f in sorted(LOG_DIR.glob("*-2026-*.json")):
        m = re.match(r"([A-Z]{6,8})-\d{4}-\d{2}-\d{2}\.json", f.name)
        if not m:
            continue
        sym = m.group(1)
        for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("event") == "signal":
                rows.append({
                    "symbol":    sym,
                    "bar_time":  e.get("bar_time"),
                    "direction": e.get("direction"),
                    "entry":     e.get("entry"),
                    "sl":        e.get("sl"),
                    "tp":        e.get("tp"),
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["bar_time"] = pd.to_datetime(df["bar_time"])
    return df


def load_nb31_bt_signals() -> pd.DataFrame:
    """Reconstruct NB31's BT-side signals from parity_per_bar_diff.csv +
    full per-bar diagnostics. We only have the *divergent* bars, so we
    can only identify bars where BT and LV disagreed — but where signal_dir
    is in the diff we know exactly what BT said.
    """
    p = PARITY_DIR / "parity_per_bar_diff.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    rows = []
    for _, r in df.iterrows():
        try:
            d = json.loads(r["diff"])
        except Exception:
            continue
        if "signal_dir" in d:
            bt_dir, lv_dir = d["signal_dir"]
            rows.append({
                "symbol":   r["symbol"],
                "bar_time": pd.to_datetime(r["bar_time"]),
                "bt_signal_dir": int(bt_dir),
                "lv_signal_dir": int(lv_dir),
            })
    return pd.DataFrame(rows)


def main():
    sim = load_sim_signals()
    live_logs = load_live_signals_from_logs()
    nb31_diffs = load_nb31_bt_signals()

    print("=== Policy B (synth) vs ACTUAL live `event=signal` logs ===")
    sim_b = sim[sim["policy"] == "B"].copy()
    print(f"Sim Policy B signals:     {len(sim_b)}")
    print(f"Live log signals:         {len(live_logs)}")
    if not live_logs.empty and not sim_b.empty:
        # match per (symbol, bar_time)
        sim_b_keys  = set(zip(sim_b["symbol"], sim_b["bar_time"]))
        live_keys   = set(zip(live_logs["symbol"], live_logs["bar_time"]))
        common      = sim_b_keys & live_keys
        sim_only    = sim_b_keys - live_keys
        live_only   = live_keys - sim_b_keys
        print(f"Match (common):           {len(common)}")
        print(f"Sim-only (in sim, not in live logs): {len(sim_only)}")
        print(f"Live-only (in live logs, not in sim): {len(live_only)}")
        if sim_only:
            print("\nExamples of sim_only (Policy B signal sim produced but live log didn't have):")
            for k in list(sim_only)[:5]:
                print(f"  {k}")
        if live_only:
            print("\nExamples of live_only (live log has, sim doesn't):")
            for k in list(live_only)[:5]:
                print(f"  {k}")

    print("\n=== Policy A (broker-only) vs NB31 BT signal_dir disagreement cases ===")
    sim_a = sim[sim["policy"] == "A"].copy()
    if not nb31_diffs.empty and not sim_a.empty:
        # For each NB31 case where BT signal_dir = +1 or -1, does our Policy A also fire?
        nb31_bt_fires = nb31_diffs[nb31_diffs["bt_signal_dir"] != 0].copy()
        nb31_keys = set(zip(nb31_bt_fires["symbol"], nb31_bt_fires["bar_time"]))
        sim_a_keys = set(zip(sim_a["symbol"], sim_a["bar_time"]))
        intersect = nb31_keys & sim_a_keys
        nb31_only  = nb31_keys - sim_a_keys
        print(f"NB31 BT-fires (excl. LV): {len(nb31_bt_fires)}")
        print(f"Sim Policy A signals:     {len(sim_a)}")
        print(f"Match:                    {len(intersect)}")
        print(f"NB31-only (BT fires, sim A doesn't): {len(nb31_only)}")
        if nb31_only:
            for k in list(nb31_only)[:5]:
                print(f"  {k}")

    # Per-symbol Policy A vs Policy B counts
    print("\n=== Per-symbol signal counts per policy ===")
    pivot = sim.groupby(["symbol", "policy"]).size().unstack(fill_value=0)
    print(pivot.to_string())


if __name__ == "__main__":
    main()
