#!/usr/bin/env python
# coding: utf-8

# # Multi-Scalper — Parity Debug (per-bar gate diff)
# 
# نسخه‌ی precise از NB30 با ۲ تغییر:
# 
# 1. **هیچ tz conversion** — هم backtest هم live از همان `time` ناخالص (broker wall-clock) استفاده می‌کنن.
#    سرور Errante زمان‌ها رو با broker wall-clock می‌فرسته (مثال: `15:00` = `8:00 NY`).
#    `Strategy._build_frame` با `broker_to_ny_h=7` این رو به NY hour تبدیل می‌کنه.
# 
# 2. **per-cycle diff** — برای هر cycle live، diagnostics قسمت backtest رو محاسبه می‌کنیم و
#    اگه `signal_dir` متفاوت بود، **مقدار هر filter رو side-by-side چاپ می‌کنیم**.
# 
# **ورودی:** `logs/<SYM>-YYYY-MM-DD.json` با event=`cycle` که حالا `diag={...}` دارن  
# (پس از patch روی `strategy.py` + `run_multi_scalper.py` — کاربر باید سرور رو restart کنه).
# 
# **خروجی:** `notebooks/data/parity_per_bar_diff.csv` با فقط ردیف‌هایی که BT vs live اختلاف دارن.

# In[1]:


from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

from mt5.multi_symbol_bot.strategy import (
    Strategy, StrategyConfig,
    HISTORY_M5_BARS, HISTORY_H1_BARS, HISTORY_D1_BARS,
    DEFAULT_BROKER_TO_NY_H,
)
warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.set_option("display.float_format", lambda x: f"{x:.6f}")
print("ready")


# ## ۱) Config

# In[2]:


BASKET = ["GBPUSD", "XAUUSD", "GBPCAD", "USDMXN", "EURJPY", "EURCAD", "AUDCAD"]
BROKER_TO_NY_H = DEFAULT_BROKER_TO_NY_H   # 7
DATA_DIR    = PROJECT_ROOT / "notebooks" / "data"
RESULTS_DIR = PROJECT_ROOT / "notebooks" / "results" / "multi_symbol_scalper"
LOGS_DIR    = PROJECT_ROOT / "logs"
OUT_DIR     = PROJECT_ROOT / "notebooks" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Backtest window — limit how many cycles to compare. None = all live cycles.
REPLAY_FROM = "2026-05-25T00:00:00"   # NO tz suffix — wall-clock
REPLAY_TO   = None
print(f"basket: {BASKET}")


# ## ۲) Loaders (naive wall-clock — NO tz conversion)
# 
# **این مهم‌ترین قسمت‌ـه**: همان قراردادی که حالا runner استفاده می‌کنه.  
# CSV با `+00:00` label نوشته شده — ولی `dt.tz_localize(None)` می‌زنیم تا فقط عددها بمونن (wall-clock).

# In[3]:


def load_cfg(sym: str) -> StrategyConfig:
    p = RESULTS_DIR / sym / "config.json"
    return StrategyConfig.from_dict(json.loads(p.read_text(encoding="utf-8")))

def load_ohlcv_naive(sym: str, tf: str) -> pd.DataFrame:
    p = DATA_DIR / sym / tf / "ohlcv.csv"
    df = pd.read_csv(p, parse_dates=["time"])
    # Strip any timezone label — we want wall-clock readings ONLY.
    if df["time"].dt.tz is not None:
        df["time"] = df["time"].dt.tz_localize(None)
    if "tick_volume" in df.columns:
        df = df.rename(columns={"tick_volume": "volume"})
    elif "volume" not in df.columns:
        df["volume"] = 0
    cols = ["time", "open", "high", "low", "close", "volume"]
    return df[cols].sort_values("time").reset_index(drop=True)

configs = {sym: load_cfg(sym) for sym in BASKET}
data = {
    sym: {tf: load_ohlcv_naive(sym, tf) for tf in ("M5", "H1", "D1")}
    for sym in BASKET
}
for sym in BASKET:
    m5 = data[sym]["M5"]
    print(f"  {sym:<8s}  M5={len(m5)}  last={m5.iloc[-1]['time']}  (naive — wall-clock)")


# ## ۳) Replay BT diagnostics for every M5 bar in the window
# 
# برای هر bar، تمام diagnostics رو ذخیره می‌کنیم (نه فقط وقتی سیگنال هست).

# In[4]:


def replay_diagnostics(sym: str, cfg: StrategyConfig,
                       replay_from: pd.Timestamp, replay_to: pd.Timestamp | None) -> pd.DataFrame:
    m5 = data[sym]["M5"]
    h1 = data[sym]["H1"]
    d1 = data[sym]["D1"]
    if replay_to is None:
        replay_to = m5.iloc[-1]["time"]
    mask = (m5["time"] >= replay_from) & (m5["time"] <= replay_to)
    idx = m5.index[mask].tolist()

    strategy = Strategy(cfg=cfg, broker_to_ny_h=BROKER_TO_NY_H)
    h1_times = h1["time"].values
    d1_times = d1["time"].values

    rows = []
    for i in idx:
        m5_tail = m5.iloc[max(0, i - HISTORY_M5_BARS + 1) : i + 1]
        t_now = m5.iloc[i]["time"]
        h1_cut = np.searchsorted(h1_times, t_now.to_datetime64(), side="right")
        d1_cut = np.searchsorted(d1_times, t_now.to_datetime64(), side="right")
        h1_tail = h1.iloc[max(0, h1_cut - HISTORY_H1_BARS) : h1_cut]
        d1_tail = d1.iloc[max(0, d1_cut - HISTORY_D1_BARS) : d1_cut]
        _, diag = strategy.detect_signal_verbose(m5_tail, h1_tail, d1_tail)
        diag["symbol"] = sym
        rows.append(diag)
    return pd.DataFrame(rows)

replay_from_ts = pd.Timestamp(REPLAY_FROM)   # naive
replay_to_ts   = pd.Timestamp(REPLAY_TO) if REPLAY_TO else None

bt_diags: dict[str, pd.DataFrame] = {}
for sym in BASKET:
    df = replay_diagnostics(sym, configs[sym], replay_from_ts, replay_to_ts)
    bt_diags[sym] = df
    n_sig = (df.get("signal_dir", pd.Series(dtype=int)) != 0).sum() if "signal_dir" in df.columns else 0
    print(f"  {sym:<8s}  diagnosed bars={len(df)}  with signal_dir!=0: {n_sig}")


# ## ۴) Parse live `cycle` events (must contain `diag` after the patch)

# In[5]:


def parse_live_cycle_diags(sym: str) -> pd.DataFrame:
    rows = []
    for log_file in sorted(LOGS_DIR.glob(f"{sym}-*.json")):
        for line in log_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("event") != "cycle":
                continue
            diag = e.get("diag") or {}
            if not diag:
                # Pre-patch cycles — no diag. Keep just enough to detect coverage.
                rows.append({"symbol": sym, "log_ts": e["ts"],
                             "bar_time": e.get("last_bar_time"),
                             "signal":   e.get("signal"),
                             "diag_present": False})
                continue
            rows.append({
                "symbol":       sym,
                "log_ts":       e["ts"],
                "diag_present": True,
                **diag,
            })
    return pd.DataFrame(rows)

live_diags = {sym: parse_live_cycle_diags(sym) for sym in BASKET}
for sym, df in live_diags.items():
    if df.empty:
        print(f"  {sym:<8s}  no live cycles found")
        continue
    n_with_diag = int(df.get("diag_present", pd.Series(dtype=bool)).sum())
    print(f"  {sym:<8s}  live cycles={len(df)}  with diag={n_with_diag}")


# ## ۵) Per-bar diff — only print rows where BT diag ≠ live diag
# 
# Match by `bar_time`. اگر هر فیلد عددی > 1e-6 اختلاف داشت یا هر filter integer متفاوت بود → divergence.

# In[6]:


INT_FIELDS = ["trend_dir", "h1_trend", "d1_trend", "in_session", "atr_ok", "adx_ok",
              "f_bb", "f_ema", "f_rsi", "f_candle", "f_rsiR", "f_macd", "f_stoch", "f_vol",
              "signal_dir"]
FLOAT_FIELDS = ["open", "high", "low", "close", "volume",
                "h1_rsi", "ema20", "rsi", "atr", "adx", "bb_up", "bb_lo"]
TOL = 1e-4   # forgiving threshold for floats (broker price quantisation)

def diff_diag(a: dict, b: dict) -> dict:
    """Return {field: (a, b)} for fields that disagree."""
    out = {}
    for k in INT_FIELDS:
        if k in a and k in b and int(a[k]) != int(b[k]):
            out[k] = (int(a[k]), int(b[k]))
    for k in FLOAT_FIELDS:
        if k in a and k in b:
            try:
                if abs(float(a[k]) - float(b[k])) > TOL:
                    out[k] = (round(float(a[k]), 6), round(float(b[k]), 6))
            except (ValueError, TypeError):
                pass
    return out

all_diffs = []
for sym in BASKET:
    bt = bt_diags[sym]
    lv = live_diags[sym]
    if lv.empty or "bar_time" not in lv.columns:
        continue
    lv_with_diag = lv[lv["diag_present"] == True] if "diag_present" in lv.columns else lv
    if lv_with_diag.empty:
        print(f"  {sym:<8s}  no live diag yet — restart server to pick up the patch")
        continue
    bt_by = {str(r["bar_time"]): r.to_dict() for _, r in bt.iterrows() if "bar_time" in bt.columns}
    for _, lv_row in lv_with_diag.iterrows():
        bt_match = bt_by.get(str(lv_row["bar_time"]))
        if bt_match is None:
            continue
        d = diff_diag(bt_match, lv_row.to_dict())
        if d:
            all_diffs.append({
                "symbol":   sym,
                "bar_time": lv_row["bar_time"],
                "log_ts":   lv_row["log_ts"],
                "n_diffs":  len(d),
                "diff":     json.dumps(d, default=str),
            })

diff_df = pd.DataFrame(all_diffs)
if diff_df.empty:
    print("=== ✅ NO DIVERGENCES detected across the analysed cycles ===")
    print("(if you expected divergences, check that the live runner was restarted")
    print(" so cycle events carry the `diag` field)")
else:
    print(f"=== ⚠️  {len(diff_df)} divergent bars ===")
    print(diff_df.to_string(index=False, max_rows=50))


# ## ۶) Save

# In[7]:


out = OUT_DIR / "parity_per_bar_diff.csv"
diff_df.to_csv(out, index=False)
print(f"saved → {out}")
print()
print("Interpretation:")
print("  * Empty → BT and live agree on every gate. Any signal mismatch is")
print("            due to factors OUTSIDE detect_signal (dedupe, stale gate, etc).")
print("  * Float-only diffs (open/high/low/close) → CSV cache OHLC differs from")
print("            broker stream. Re-fetch CSVs and re-run NB30 to confirm.")
print("  * volume diff → broker uses real_volume but our CSV has tick_volume")
print("            (or vice-versa). Only matters if `f_vol` is in cfg.confirms.")
print("  * Integer-only diffs (e.g. f_bb=-1 vs 0) with no float diff → bug or")
print("            indicator init difference (warmup needed).")

