#!/usr/bin/env python
# coding: utf-8

# # Multi-Scalper — Backtest vs Live Parity Check
# 
# این نوت‌بوک **دقیقاً همان منطق `mt5/run_multi_scalper.py` رو replay می‌کنه** روی دیتای CSV ذخیره‌شده، تا ببینیم آیا سیگنال‌های live با backtest یکی هستن.
# 
# ## معماری
# 
# برای هر symbol در golden basket:
# 1. Config رو از `notebooks/results/multi_symbol_scalper/<SYM>/config.json` لود می‌کنیم.
# 2. M5 + H1 + D1 رو از CSV می‌خونیم (با timezone fix مطابق memory).
# 3. برای هر M5 bar در بازه‌ی replay، یه snapshot از history می‌سازیم (تا اون bar) و `Strategy.detect_signal()` رو call می‌کنیم — دقیقاً همون چیزی که runner زنده می‌کنه.
# 4. سیگنال‌های backtest رو جمع می‌کنیم.
# 5. سیگنال‌های live رو از `logs/<SYM>-YYYY-MM-DD.json` parse می‌کنیم (event=`signal` یا `cycle` با `signal=True`).
# 6. side-by-side مقایسه می‌کنیم.
# 
# ## انتظار
# 
# اگر منطق یکسانه و دیتا یکسانه:
# - تعداد سیگنال‌ها در یه بازه‌ی زمانی **مساوی** باشه
# - timestampها (`bar_time`) **دقیقاً منطبق** باشن
# - direction/entry/SL/TP **یکسان** باشن
# 
# هر divergence → یا تفاوت در دیتای منبع (CSV vs broker stream) یا یه باگ subtle در زمان‌گذاری.
# 
# **خروجی:** `notebooks/data/parity_multi_scalper.csv`

# In[10]:


from __future__ import annotations
import json
import sys
import warnings
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

from mt5.multi_symbol_bot.strategy import (
    Strategy, StrategyConfig, Signal,
    HISTORY_M5_BARS, HISTORY_H1_BARS, HISTORY_D1_BARS,
    DEFAULT_BROKER_TO_NY_H,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.set_option("display.float_format", lambda x: f"{x:.5f}")
print("ready")


# ## ۱) Config
# 
# Basket و بازه‌ی زمانی replay (هم‌بازه با live).

# In[11]:


# Golden basket — exactly the 7 symbols the live runner picked from symbol_ranking.csv
BASKET = ["GBPUSD", "XAUUSD", "GBPCAD", "USDMXN", "EURJPY", "EURCAD", "AUDCAD"]

# Same broker→NY offset as the live runner (Errante = EET/EEST → NY = UTC-4/-5)
BROKER_TO_NY_H = DEFAULT_BROKER_TO_NY_H   # = 7

# Replay window — same window we want to compare against the live logs.
# Set REPLAY_FROM=None to replay the full history (slow on long CSVs).
REPLAY_FROM = "2026-05-25T00:00:00+00:00"
REPLAY_TO   = None                      # None = latest M5 bar in CSV

# Paths
DATA_DIR    = PROJECT_ROOT / "notebooks" / "data"
RESULTS_DIR = PROJECT_ROOT / "notebooks" / "results" / "multi_symbol_scalper"
LOGS_DIR    = PROJECT_ROOT / "logs"
OUT_DIR     = PROJECT_ROOT / "notebooks" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Same timezone convention as the live runner / NB29:
# CSV time labels are wall-clock-as-UTC, real tz is Europe/Nicosia.
# We relabel to Nicosia then convert to UTC — matching the live runner's
# `_mt5_seconds_to_utc()` output.
BROKER_TZ = "Europe/Nicosia"

print(f"basket: {BASKET}")
print(f"replay window: {REPLAY_FROM} → {REPLAY_TO or 'latest'}")


# ## ۲) Load per-symbol config
# 
# `notebooks/results/multi_symbol_scalper/<SYM>/config.json` رو می‌خونیم.

# In[12]:


def load_strategy_config(symbol: str) -> StrategyConfig:
    path = RESULTS_DIR / symbol / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"missing config.json for {symbol}: {path}")
    return StrategyConfig.from_dict(json.loads(path.read_text(encoding="utf-8")))

configs = {sym: load_strategy_config(sym) for sym in BASKET}
for sym, cfg in configs.items():
    print(f"  {sym:<8s}  mode={cfg.mode:<16s} session={cfg.session}  confirms={cfg.confirms}")


# ## ۳) Loader (با timezone fix منطبق با live runner)
# 
# Live runner: MT5 → `_mt5_seconds_to_utc()` → tz=UTC.
# این تابع زمان wall-clock بروکر (که با offset Nicosia ست) رو به UTC واقعی تبدیل می‌کنه.
# 
# CSV file: زمان با label `+00:00` ذخیره شده ولی واقعاً Nicosia‌ـه (memory).
# پس همون تبدیل: strip label → localize as Nicosia → convert to UTC.
# نتیجه: تایم‌اِستمپ‌ها در backtest و live **یکسان** خواهند بود.

# In[ ]:


def load_ohlcv(symbol: str, tf: str) -> pd.DataFrame:
    """Load CSV, normalise tz to real UTC (match live runner output)."""
    p = DATA_DIR / symbol / tf / "ohlcv.csv"
    df = pd.read_csv(p, parse_dates=["time"])
    # CSV is wall-clock-as-UTC; relabel as Nicosia (no shift), then convert to UTC.
    naive = df["time"].dt.tz_localize(None) if df["time"].dt.tz is not None else df["time"]
    nicosia = naive.dt.tz_localize(BROKER_TZ, ambiguous="infer", nonexistent="shift_forward")
    df["time"] = nicosia.dt.tz_convert("UTC")

    # Live runner renames tick_volume → volume
    if "tick_volume" in df.columns:
        df = df.rename(columns={"tick_volume": "volume"})
    elif "volume" not in df.columns:
        df["volume"] = 0

    cols = ["time", "open", "high", "low", "close", "volume"]
    return df[cols].sort_values("time").reset_index(drop=True)

# Pre-load everything once
data = {}
for sym in BASKET:
    data[sym] = {
        "M5": load_ohlcv(sym, "M5"),
        "H1": load_ohlcv(sym, "H1"),
        "D1": load_ohlcv(sym, "D1"),
    }
    m5 = data[sym]["M5"]
    print(f"  {sym:<8s}  M5={len(m5)}  H1={len(data[sym]['H1'])}  D1={len(data[sym]['D1'])}  "
          f"last={m5.iloc[-1]['time']}")


# ## ۴) Replay engine — exactly like the live runner
# 
# برای هر M5 bar در بازه‌ی replay:
# - M5 frame = همه‌ی barهای ≤ این timestamp (مانند runner که `iloc[:-1]` می‌زنه)
# - H1 frame = همه‌ی H1 barهایی که بسته شدن (`<= this_m5.time`)
# - D1 frame = همه‌ی D1 barهایی که بسته شدن
# - `Strategy.detect_signal(m5_frame, h1_frame, d1_frame)` رو call می‌کنیم

# In[ ]:


def replay_signals(
    symbol: str, cfg: StrategyConfig,
    replay_from: pd.Timestamp, replay_to: pd.Timestamp | None = None,
) -> list[dict]:
    """Walk M5 bars; at each, replay the runner's detect_signal call."""
    m5_all = data[symbol]["M5"]
    h1_all = data[symbol]["H1"]
    d1_all = data[symbol]["D1"]

    if m5_all.empty:
        return []
    if replay_to is None:
        replay_to = m5_all.iloc[-1]["time"]

    # Indices of M5 bars whose CLOSE time is in the replay window.
    mask = (m5_all["time"] >= replay_from) & (m5_all["time"] <= replay_to)
    replay_idx = m5_all.index[mask].tolist()
    if not replay_idx:
        return []

    strategy = Strategy(cfg=cfg, broker_to_ny_h=BROKER_TO_NY_H)
    signals: list[dict] = []

    # Pre-extract numpy views for the HTF frames so slicing in the loop is cheap.
    h1_times = h1_all["time"].values
    d1_times = d1_all["time"].values

    for i in replay_idx:
        # Live runner fetches the latest HISTORY_*_BARS, then drops the still-
        # forming bar with iloc[:-1]. Here we already have CLOSED bars in CSV,
        # so we take a tail of length HISTORY_M5_BARS ending at index `i`.
        m5_tail = m5_all.iloc[max(0, i - HISTORY_M5_BARS + 1) : i + 1]
        t_now = m5_all.iloc[i]["time"]

        # H1: include only bars whose close is <= this M5 bar's close.
        h1_cutoff = np.searchsorted(h1_times, t_now.to_datetime64(), side="right")
        h1_tail = h1_all.iloc[max(0, h1_cutoff - HISTORY_H1_BARS) : h1_cutoff]

        d1_cutoff = np.searchsorted(d1_times, t_now.to_datetime64(), side="right")
        d1_tail = d1_all.iloc[max(0, d1_cutoff - HISTORY_D1_BARS) : d1_cutoff]

        sig: Signal | None = strategy.detect_signal(m5_tail, h1_tail, d1_tail)
        if sig is not None:
            signals.append({
                "symbol":    symbol,
                "bar_time":  sig.bar_time,
                "direction": sig.direction,
                "entry":     sig.entry,
                "sl":        sig.sl,
                "tp":        sig.tp,
                **{f"conf_{k}": v for k, v in sig.confidence.items()},
            })
    return signals

# Run for each symbol
replay_from_ts = pd.Timestamp(REPLAY_FROM)
replay_to_ts   = pd.Timestamp(REPLAY_TO) if REPLAY_TO else None

backtest_signals: dict[str, list[dict]] = {}
for sym in BASKET:
    sigs = replay_signals(sym, configs[sym], replay_from_ts, replay_to_ts)
    backtest_signals[sym] = sigs
    print(f"  {sym:<8s}  signals in window: {len(sigs)}")


# ## ۵) Parse live signals from log files
# 
# Live runner برای هر symbol یه فایل `logs/<SYM>-YYYY-MM-DD.json` می‌نویسه. سیگنال‌ها در event=`signal` ظاهر می‌شن.

# In[ ]:


def parse_live_signals(symbol: str) -> list[dict]:
    """Read all daily log files for `symbol` and extract signal events."""
    out: list[dict] = []
    for log_file in sorted(LOGS_DIR.glob(f"{symbol}-*.json")):
        for line in log_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("event") != "signal":
                continue
            out.append({
                "symbol":    symbol,
                "log_ts":    e["ts"],
                "bar_time":  e.get("bar_time"),
                "direction": e.get("direction"),
                "entry":     e.get("entry"),
                "sl":        e.get("sl"),
                "tp":        e.get("tp"),
            })
    return out

live_signals: dict[str, list[dict]] = {sym: parse_live_signals(sym) for sym in BASKET}
for sym, sigs in live_signals.items():
    print(f"  {sym:<8s}  live signals so far: {len(sigs)}")


# ## ۶) Side-by-side comparison
# 
# برای هر `bar_time` در یکی از دو طرف، ببینیم در طرف دیگه هم هست یا نه.

# In[ ]:


rows: list[dict] = []
for sym in BASKET:
    bt = pd.DataFrame(backtest_signals[sym])
    lv = pd.DataFrame(live_signals[sym])

    bt_keys = set() if bt.empty else set(bt["bar_time"].astype(str))
    lv_keys = set() if lv.empty else set(lv["bar_time"].astype(str))
    only_bt = bt_keys - lv_keys
    only_lv = lv_keys - bt_keys
    both    = bt_keys & lv_keys

    rows.append({
        "symbol":         sym,
        "backtest_total": len(bt_keys),
        "live_total":     len(lv_keys),
        "matched":        len(both),
        "only_backtest":  len(only_bt),
        "only_live":      len(only_lv),
    })

summary = pd.DataFrame(rows)
print("=== summary ===")
print(summary.to_string(index=False))
print()
totals = summary[["backtest_total","live_total","matched","only_backtest","only_live"]].sum()
print("=== totals ===")
print(totals.to_string())


# ## ۷) Detailed diff — every signal in either side, side-by-side

# In[ ]:


details: list[dict] = []
for sym in BASKET:
    bt = pd.DataFrame(backtest_signals[sym])
    lv = pd.DataFrame(live_signals[sym])
    bt_by = {} if bt.empty else {str(r['bar_time']): r for _, r in bt.iterrows()}
    lv_by = {} if lv.empty else {str(r['bar_time']): r for _, r in lv.iterrows()}
    all_keys = sorted(set(bt_by) | set(lv_by))
    for k in all_keys:
        b = bt_by.get(k); l = lv_by.get(k)
        details.append({
            "symbol":          sym,
            "bar_time":        k,
            "in_backtest":     b is not None,
            "in_live":         l is not None,
            "bt_direction":    b["direction"] if b is not None else None,
            "lv_direction":    l["direction"] if l is not None else None,
            "bt_entry":        round(float(b["entry"]), 5) if b is not None else None,
            "lv_entry":        round(float(l["entry"]), 5) if l is not None else None,
            "bt_sl":           round(float(b["sl"]), 5) if b is not None else None,
            "lv_sl":           round(float(l["sl"]), 5) if l is not None else None,
            "bt_tp":           round(float(b["tp"]), 5) if b is not None else None,
            "lv_tp":           round(float(l["tp"]), 5) if l is not None else None,
            "match_direction": (b is not None and l is not None and b["direction"] == l["direction"]),
        })
diff = pd.DataFrame(details)
if diff.empty:
    print("no signals on either side in the replay window")
else:
    print(f"total rows: {len(diff)}")
    print(diff.to_string(index=False, max_rows=60))


# ## ۸) Save

# In[ ]:


out_summary = OUT_DIR / "parity_multi_scalper_summary.csv"
out_detail  = OUT_DIR / "parity_multi_scalper_detail.csv"
summary.to_csv(out_summary, index=False)
diff.to_csv(out_detail, index=False)
print(f"saved → {out_summary}")
print(f"saved → {out_detail}")
print()
print("interpretation guide:")
print("  matched > 0 + only_backtest=0 + only_live=0 → ✅ perfect parity")
print("  only_backtest > 0  → live MISSED signals that backtest sees (data freshness?)")
print("  only_live > 0      → live FIRED signals backtest doesn't (CSV stale vs broker stream)")
print("  match_direction=False in diff → SAME bar but OPPOSITE side (BUG; investigate)")

