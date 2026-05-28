"""Extract the 23 bars where BT signal_dir != LV signal_dir, with full diag."""
from __future__ import annotations
import json
from pathlib import Path

import pandas as pd

ROOT  = Path(__file__).resolve().parent.parent
p = ROOT / "notebooks" / "data" / "parity_per_bar_diff.csv"
df = pd.read_csv(p)

cases = []
for _, r in df.iterrows():
    try:
        d = json.loads(r["diff"])
    except Exception:
        continue
    if "signal_dir" not in d:
        continue
    cases.append({
        "symbol":   r["symbol"],
        "bar_time": r["bar_time"],
        "log_ts":   r["log_ts"],
        "signal_dir_diff": d["signal_dir"],
        "other_fields":    {k: v for k, v in d.items() if k != "signal_dir"},
    })

print(f"Total signal_dir-disagreement bars: {len(cases)}\n")
for c in cases:
    bt, lv = c["signal_dir_diff"]
    direction = "REPLAY ALONE FIRED" if lv == 0 else ("LIVE ALONE FIRED" if bt == 0 else "OPPOSITE DIRECTIONS")
    print(f"  {c['symbol']:7s}  {c['bar_time']}  BT={bt:+d}  LV={lv:+d}   [{direction}]")
    if c["other_fields"]:
        # Show top 4 most informative fields
        items = list(c["other_fields"].items())[:4]
        for k, v in items:
            print(f"      {k}: BT={v[0]}  LV={v[1]}")
    print()

# Per-symbol breakdown
from collections import Counter
sym_counter = Counter(c["symbol"] for c in cases)
print("\nPer-symbol signal_dir-disagreement count:")
for s, n in sym_counter.most_common():
    print(f"  {s}: {n}")
