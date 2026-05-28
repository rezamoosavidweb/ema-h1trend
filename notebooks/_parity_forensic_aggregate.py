"""Forensic aggregation across parity artifacts + live logs.

Run from project root:  python notebooks/_parity_forensic_aggregate.py

Produces a single JSON report on stdout summarising:
  1. Per-bar gate divergence frequency  (parity_per_bar_diff.csv)
  2. Timestamp-format hypothesis check  (parity_multi_scalper_detail.csv)
  3. Trade-list counts vs replay trades (live logs vs replay_trades_*.csv)
  4. Live skip-reason histogram          (logs/<SYM>*.json event=skip)
  5. Signal-bar entry-price drift sample (close-of-i vs open-of-i+1)
"""
from __future__ import annotations
import json, re
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime

import pandas as pd

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "notebooks" / "data"
LOG_DIR  = ROOT / "logs"

BASKET = ["GBPUSD", "XAUUSD", "GBPCAD", "USDMXN", "EURJPY", "EURCAD", "AUDCAD"]


# ============================================================================
# 1) Per-bar gate divergence frequency
# ============================================================================
def analyze_per_bar_diff() -> dict:
    p = DATA_DIR / "parity_per_bar_diff.csv"
    if not p.exists():
        return {"error": "parity_per_bar_diff.csv missing"}

    df = pd.read_csv(p)
    # Parse the JSON `diff` column into {field: (bt_val, lv_val)}
    field_freq         = Counter()
    field_pairs        = defaultdict(Counter)
    by_symbol_field    = defaultdict(Counter)
    by_symbol_total    = Counter()

    for _, r in df.iterrows():
        sym = r["symbol"]
        by_symbol_total[sym] += 1
        try:
            diff = json.loads(r["diff"])
        except Exception:
            continue
        for k, pair in diff.items():
            field_freq[k] += 1
            by_symbol_field[sym][k] += 1
            # Bucket pair like (-1,1) so we see directional flips
            if k in ("trend_dir", "h1_trend", "d1_trend", "f_bb", "f_ema", "f_rsi",
                     "f_candle", "f_rsiR", "f_macd", "f_stoch", "f_vol",
                     "signal_dir", "in_session", "atr_ok", "adx_ok"):
                field_pairs[k][f"{pair[0]}->{pair[1]}"] += 1

    # signal_dir specifically — how often does it disagree?
    sig_dir_total = field_freq.get("signal_dir", 0)
    return {
        "total_divergent_bars": int(len(df)),
        "field_frequency": dict(field_freq.most_common()),
        "categorical_pair_buckets": {k: dict(v.most_common()) for k, v in field_pairs.items()},
        "by_symbol_total_divergent_bars": dict(by_symbol_total),
        "by_symbol_top_fields": {
            s: dict(by_symbol_field[s].most_common(5)) for s in by_symbol_field
        },
        "signal_dir_disagreement_count": int(sig_dir_total),
    }


# ============================================================================
# 2) Timestamp-format hypothesis
# ============================================================================
def analyze_timestamp_format() -> dict:
    p = DATA_DIR / "parity_multi_scalper_detail.csv"
    if not p.exists():
        return {"error": "parity_multi_scalper_detail.csv missing"}
    df = pd.read_csv(p)
    has_suffix = df["bar_time"].astype(str).str.contains(r"\+00:00")
    in_bt_only = (df["in_backtest"] == True) & (df["in_live"] == False)
    in_lv_only = (df["in_backtest"] == False) & (df["in_live"] == True)

    bt_with_suffix = int((in_bt_only & has_suffix).sum())
    bt_no_suffix   = int((in_bt_only & ~has_suffix).sum())
    lv_with_suffix = int((in_lv_only & has_suffix).sum())
    lv_no_suffix   = int((in_lv_only & ~has_suffix).sum())

    # Strip suffix → if both sides have same time, they could have matched.
    df["bar_time_naive"] = df["bar_time"].astype(str).str.replace(r"\+00:00$", "", regex=True)
    bt_times = set(df.loc[in_bt_only, "bar_time_naive"])
    lv_times = set(df.loc[in_lv_only, "bar_time_naive"])
    naive_overlap = bt_times & lv_times

    # Per-symbol overlap
    per_sym_overlap = {}
    for sym in BASKET:
        bt_s = set(df.loc[(df["symbol"] == sym) & in_bt_only, "bar_time_naive"])
        lv_s = set(df.loc[(df["symbol"] == sym) & in_lv_only, "bar_time_naive"])
        per_sym_overlap[sym] = {
            "bt_only": len(bt_s),
            "lv_only": len(lv_s),
            "would_match_if_naive": len(bt_s & lv_s),
        }

    return {
        "backtest_rows_with_utc_suffix": bt_with_suffix,
        "backtest_rows_without_suffix":  bt_no_suffix,
        "live_rows_with_utc_suffix":     lv_with_suffix,
        "live_rows_without_suffix":      lv_no_suffix,
        "would_match_if_suffix_stripped_total": len(naive_overlap),
        "would_match_per_symbol": per_sym_overlap,
        "note": (
            "If backtest rows carry +00:00 but live rows don't, the bar_time JOIN "
            "key never matches and the summary CSV shows 0 matches even when "
            "both sides actually produced a signal at the same wall-clock minute."
        ),
    }


# ============================================================================
# 3) Live log skip-reason histogram
# ============================================================================
def analyze_live_skips() -> dict:
    counts = Counter()
    per_sym_per_reason = defaultdict(Counter)
    suppressed_telegram = Counter()
    signals_per_sym = Counter()
    placed_per_sym  = Counter()
    cycle_count_per_sym = Counter()

    log_files = sorted(LOG_DIR.glob("*-2026-*.json"))
    for f in log_files:
        sym_match = re.match(r"([A-Z]{6,8})-\d{4}-\d{2}-\d{2}\.json", f.name)
        if not sym_match:
            continue
        sym = sym_match.group(1)
        for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            ev = e.get("event")
            if ev == "skip":
                reason = e.get("reason", "?")
                counts[reason] += 1
                per_sym_per_reason[sym][reason] += 1
            elif ev == "signal":
                signals_per_sym[sym] += 1
            elif ev == "signal_telegram_suppressed":
                suppressed_telegram[sym] += 1
            elif ev == "order_placed" or ev == "market_order_placed" or ev == "limit_order_placed":
                placed_per_sym[sym] += 1
            elif ev == "cycle":
                cycle_count_per_sym[sym] += 1

    return {
        "skip_reason_global_histogram": dict(counts.most_common()),
        "skip_reasons_per_symbol":      {s: dict(v.most_common()) for s, v in per_sym_per_reason.items()},
        "signal_count_per_symbol":      dict(signals_per_sym),
        "order_placed_per_symbol":      dict(placed_per_sym),
        "tg_suppressed_per_symbol":     dict(suppressed_telegram),
        "cycle_count_per_symbol":       dict(cycle_count_per_sym),
    }


# ============================================================================
# 4) Replay vs live trade lists (use existing replay CSVs)
# ============================================================================
def analyze_replay_vs_live_trades() -> dict:
    out = {}
    # Use the freshest replay trades file
    replay_files = sorted(DATA_DIR.glob("replay_trades_2026*.csv"))
    if not replay_files:
        return {"error": "no replay_trades_2026*.csv found"}
    replay = pd.read_csv(replay_files[-1])
    replay["entry_time"] = pd.to_datetime(replay["entry_time"])
    out["replay_file_used"] = replay_files[-1].name
    out["replay_trades_per_symbol"] = (
        replay.groupby("symbol").size().to_dict()
    )

    # Live "trades" extracted from logs (order_placed events as a proxy)
    live_placed_per_sym = defaultdict(list)
    log_files = sorted(LOG_DIR.glob("*-2026-*.json"))
    placed_events = ("order_placed", "market_order_placed", "limit_order_placed")
    for f in log_files:
        sym_match = re.match(r"([A-Z]{6,8})-\d{4}-\d{2}-\d{2}\.json", f.name)
        if not sym_match:
            continue
        sym = sym_match.group(1)
        for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("event") in placed_events:
                live_placed_per_sym[sym].append({
                    "ts": e.get("ts"),
                    "bar_time": e.get("bar_time") or e.get("ob_time"),
                    "direction": e.get("direction") or e.get("side"),
                    "entry": e.get("entry") or e.get("market_price") or e.get("requested_entry"),
                })

    out["live_orders_placed_per_symbol"] = {s: len(v) for s, v in live_placed_per_sym.items()}

    # Compare counts side-by-side
    out["compare_counts"] = {
        sym: {
            "replay_trades": int(replay[replay["symbol"] == sym].shape[0]),
            "live_orders_placed":  len(live_placed_per_sym.get(sym, [])),
        }
        for sym in BASKET
    }
    return out


# ============================================================================
# 5) Entry-price drift sample (open(i+1) vs close(i)) for the replay window
# ============================================================================
def sample_entry_drift() -> dict:
    """For each replay trade, compute |open(i+1) - close(i)| using the CSV M5
    if it exists; this is the inherent entry-price gap between live's
    close-of-bar convention and replay's open-of-next-bar convention.
    """
    replay_files = sorted(DATA_DIR.glob("replay_trades_2026*.csv"))
    if not replay_files:
        return {"error": "no replay trades CSV"}
    replay = pd.read_csv(replay_files[-1])
    replay["entry_time"] = pd.to_datetime(replay["entry_time"])

    drifts_per_sym = defaultdict(list)
    for _, t in replay.iterrows():
        sym = t["symbol"]
        m5p = DATA_DIR / sym / "M5" / "ohlcv.csv"
        if not m5p.exists():
            continue
        m5 = pd.read_csv(m5p, parse_dates=["time"])
        if m5["time"].dt.tz is not None:
            m5["time"] = m5["time"].dt.tz_localize(None)
        # `entry_time` in NB33 is df['time'].iat[ei] which is the **i+1 bar's
        # time** (the bar we open at). Live "close" was on the previous M5 bar
        # (entry_time - 5min).
        prev_bar_time = t["entry_time"] - pd.Timedelta(minutes=5)
        row_prev = m5[m5["time"] == prev_bar_time]
        row_cur  = m5[m5["time"] == t["entry_time"]]
        if row_prev.empty or row_cur.empty:
            continue
        close_prev = float(row_prev["close"].iat[0])
        open_cur   = float(row_cur["open"].iat[0])
        drift = open_cur - close_prev
        drifts_per_sym[sym].append({
            "entry_time":  str(t["entry_time"]),
            "side":        t["side"],
            "close_prev":  close_prev,
            "open_cur":    open_cur,
            "drift":       round(drift, 6),
        })

    summary = {}
    for sym, lst in drifts_per_sym.items():
        if not lst:
            continue
        ds = [abs(d["drift"]) for d in lst]
        summary[sym] = {
            "samples":      len(lst),
            "mean_abs_drift": round(sum(ds) / len(ds), 6),
            "max_abs_drift":  round(max(ds), 6),
            "examples":     lst[:3],
        }
    return summary


# ============================================================================
def main():
    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "section_1_per_bar_gate_diff":      analyze_per_bar_diff(),
        "section_2_timestamp_format_check": analyze_timestamp_format(),
        "section_3_live_skip_histogram":    analyze_live_skips(),
        "section_4_replay_vs_live_trades":  analyze_replay_vs_live_trades(),
        "section_5_entry_drift_sample":     sample_entry_drift(),
    }
    print(json.dumps(report, indent=2, default=str))

if __name__ == "__main__":
    main()
