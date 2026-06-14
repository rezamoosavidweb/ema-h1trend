"""
Reconcile the paper (simulated) fills against the real broker orders.

When `live_orders` is on, every fill row in the `fills` table holds BOTH the simulated
execution (``actual_price``, ``slippage_bps``, ``latency_ms``, ``fill_ts``) and the real
MT5 execution (``broker_fill_price``, ``broker_latency_ms``, ``broker_fill_ts``,
``broker_ok``). This module turns that into a like-for-like comparison so you can answer,
after a few days: *how far apart are the log and the real orders — in price and in time?*

Per filled leg it computes:
  * ``price_gap_pips``  = real fill price − paper (simulated) fill price
  * ``slip_real_pips``  = real fill price − intended (mid) price        (real slippage)
  * ``slip_paper_pips`` = paper fill price − intended (mid) price       (simulated slippage)
  * ``time_gap_s``      = real fill time − paper fill time
  * ``signal_to_real_s``= real fill time − signal (bar-close) time
  * latency: real ``broker_latency_ms`` vs simulated ``latency_ms``
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import SystemConfig
from .storage import create_storage


def _pip(symbol: str) -> float:
    return 1e-3 if str(symbol).endswith("JPY") else 1e-5


def build_reconciliation(storage) -> pd.DataFrame:
    """Return a per-fill comparison DataFrame (only rows that have a real broker fill)."""
    df = storage.fetch_df("fills")
    if df.empty or "exec_mode" not in df.columns:
        return pd.DataFrame()
    live = df[(df["exec_mode"] == "live") & (df["broker_ok"] == True)].copy()  # noqa: E712
    if live.empty:
        return pd.DataFrame()

    for c in ("fill_ts", "broker_fill_ts", "signal_ts"):
        if c in live.columns:
            live[c] = pd.to_datetime(live[c], utc=True, errors="coerce")

    pip = live["symbol"].map(_pip)
    live["price_gap_pips"] = (live["broker_fill_price"] - live["actual_price"]) / pip
    live["slip_real_pips"] = (live["broker_fill_price"] - live["intended_price"]) / pip
    live["slip_paper_pips"] = (live["actual_price"] - live["intended_price"]) / pip
    live["time_gap_s"] = (live["broker_fill_ts"] - live["fill_ts"]).dt.total_seconds()
    live["signal_to_real_s"] = (live["broker_fill_ts"] - live["signal_ts"]).dt.total_seconds()
    live["latency_paper_ms"] = live["latency_ms"]
    cols = ["id", "pair_key", "symbol", "kind", "side", "volume",
            "intended_price", "actual_price", "broker_fill_price",
            "price_gap_pips", "slip_paper_pips", "slip_real_pips",
            "fill_ts", "broker_fill_ts", "time_gap_s", "signal_to_real_s",
            "latency_paper_ms", "broker_latency_ms", "broker_ticket"]
    return live[[c for c in cols if c in live.columns]].reset_index(drop=True)


def summarize(recon: pd.DataFrame, storage) -> dict:
    df = storage.fetch_df("fills")
    n_total = int((df.get("exec_mode") == "live").sum()) if not df.empty else 0
    if recon.empty:
        return {"live_fills": n_total, "filled_ok": 0,
                "note": "no successfully-filled live orders yet"}
    return {
        "live_fills_attempted": n_total,
        "filled_ok": int(len(recon)),
        "fill_rate": round(len(recon) / max(n_total, 1), 3),
        "price_gap_pips_median": round(float(recon["price_gap_pips"].median()), 2),
        "price_gap_pips_mean": round(float(recon["price_gap_pips"].mean()), 2),
        "price_gap_pips_p95": round(float(recon["price_gap_pips"].abs().quantile(0.95)), 2),
        "slip_paper_pips_median": round(float(recon["slip_paper_pips"].median()), 2),
        "slip_real_pips_median": round(float(recon["slip_real_pips"].median()), 2),
        "time_gap_s_median": round(float(recon["time_gap_s"].median()), 2),
        "signal_to_real_s_median": round(float(recon["signal_to_real_s"].median()), 1),
        "latency_real_ms_median": round(float(recon["broker_latency_ms"].median()), 1),
        "latency_paper_ms_median": round(float(recon["latency_paper_ms"].median()), 1),
    }


def run_reconcile(config: SystemConfig, *, export: bool = True) -> dict:
    """Build + summarize the reconciliation; optionally export the per-fill CSV."""
    storage = create_storage(config, init=False)
    try:
        recon = build_reconciliation(storage)
        summary = summarize(recon, storage)
        out_path = None
        if export and not recon.empty:
            out_path = config.report_path() / "reconciliation.csv"
            recon.to_csv(out_path, index=False)
        return {"summary": summary, "rows": len(recon),
                "csv": str(out_path) if out_path else None, "table": recon}
    finally:
        storage.close()
