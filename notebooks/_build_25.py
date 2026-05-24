#!/usr/bin/env python
# coding: utf-8

# # Pairs Trading — Notebook 1: Discovery (H1 + H4 موازی)
# 
# هدف: شناسایی **کاندیداهای cointegrated pairs** روی ۲۸ جفت FX major+cross، **در دو تایم‌فریم موازی** (H1 و H4 از resample) طی ۳ سال اخیر.
# 
# - **In-sample:** `2023-01-01` تا `2025-12-31`
# - **Out-of-sample:** از `2026-01-01` به بعد (held out برای نوت‌بوک ۲۶)
# 
# **چرا دو TF؟**  در اولین اجرا با H1 دیدیم که همه‌ی ۳۷۸ جفت از نظر آماری cointegrated هستن اما min half-life ≈ 319 ساعت (~۱۳ روز کاری). یعنی mean-reversion هست ولی خیلی آرومه. در H4 همون half-life می‌شه ۸۰ بار، که برای swing-trading مناسبه. این نوت‌بوک هر دو رویکرد رو می‌سنجه.
# 
# **مراحل (برای هر TF):**
# 1. لود دیتا (H1 از CSV، H4 با resample)
# 2. Currency-graph rank check
# 3. Engle-Granger cointegration scan روی C(28,2)=378 جفت
# 4. فیلتر: p<0.05 + half-life **در ساعت** بین MIN و MAX (یکسان برای دو TF — مقایسه‌پذیر)
# 5. Stage-4 lite: حذف spread‌هایی که با یک پیر سوم corr>0.95
# 6. Top-5 spread chart
# 7. Save shortlist
# 
# **خروجی:** `notebooks/data/stat_arb/cointegrated_shortlist_{H1,H4}.csv`

# In[10]:


from __future__ import annotations
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import adfuller

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

from stat_arb.identity.currency_graph import CurrencyGraph, parse_symbol
from stat_arb.config import DEFAULT_KNOWN_CURRENCIES

warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")
print("ready")


# ## Configuration

# In[11]:


UNIVERSE = [
    # Majors
    "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD",
    "USDCAD", "USDCHF", "USDJPY",
    # JPY crosses
    "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY", "CHFJPY",
    # EUR crosses
    "EURGBP", "EURAUD", "EURNZD", "EURCAD", "EURCHF",
    # GBP crosses
    "GBPAUD", "GBPNZD", "GBPCAD", "GBPCHF",
    # Commodity / CHF crosses
    "AUDCAD", "AUDCHF", "AUDNZD", "NZDCAD", "NZDCHF", "CADCHF",
]
TIMEFRAMES           = ["H1", "H4"]
IN_SAMPLE_START      = "2023-01-01"
IN_SAMPLE_END        = "2025-12-31"
OUT_OF_SAMPLE_START  = "2026-01-01"
MIN_HALF_LIFE_HOURS  = 5            # below this = noise / over-trading
MAX_HALF_LIFE_HOURS  = 500          # cap on real-time half-life (~3 weeks); same wall-clock for both TFs
PVALUE_MAX           = 0.05
SPREAD_CORR_MAX      = 0.95         # Stage-4 lite threshold
TOP_N_PLOT           = 5

DATA_DIR   = PROJECT_ROOT / "notebooks" / "data"
OUTPUT_DIR = PROJECT_ROOT / "notebooks" / "data" / "stat_arb"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Per project memory: CSVs label time as +00:00 but real broker time is Europe/Nicosia.
REAL_TZ = "Europe/Nicosia"

TF_HOURS = {"H1": 1, "H4": 4, "D1": 24}   # bar→hours mapping
n_pairs  = len(UNIVERSE) * (len(UNIVERSE) - 1) // 2
print(f"universe: {len(UNIVERSE)} symbols  |  C({len(UNIVERSE)},2) = {n_pairs} candidate pairs per TF")
print(f"timeframes: {TIMEFRAMES}")
print(f"half-life cap: {MIN_HALF_LIFE_HOURS}h ≤ HL ≤ {MAX_HALF_LIFE_HOURS}h  (~ {MAX_HALF_LIFE_HOURS/24:.0f} days)")


# ## ۱) Loader
# 
# - `load_pair_h1` — لود مستقیم CSV با fix timezone (UTC label → Europe/Nicosia)
# - `load_all_h1_aligned` — inner-join همه‌ی ۲۸ جفت روی H1 → DataFrame متراکم بدون NaN
# - `to_tf(prices_h1, tf)` — resample wide-DataFrame اسلاید‌شده به H4 یا D1
# 
# **نکته‌ی مهم:** اول H1 رو inner-join می‌کنیم، **بعد** resample. اگه برعکس انجام بدیم، bucket‌های weekend در پیرهای مختلف ناهماهنگ‌اند → هر ردیف حداقل یه NaN داره → dropna همه رو می‌بره.

# In[ ]:


def load_pair_h1(symbol: str) -> pd.Series:
    """Load H1 close prices for one symbol; return tz-aware Series indexed by Nicosia time."""
    path = DATA_DIR / symbol / "H1" / "ohlcv.csv"
    df = pd.read_csv(path, parse_dates=["time"])
    naive = df["time"].dt.tz_localize(None)
    ts = naive.dt.tz_localize(REAL_TZ, ambiguous="NaT", nonexistent="NaT")
    s = pd.Series(df["close"].values, index=ts, name=symbol)
    return s[s.index.notna()].sort_index()


def load_all_h1_aligned() -> pd.DataFrame:
    """Inner-join all UNIVERSE pairs on H1 — only rows where every pair has data."""
    h1 = {sym: load_pair_h1(sym) for sym in UNIVERSE}
    return pd.concat(h1, axis=1, sort=True).dropna()


def to_tf(prices_h1: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Resample the *aligned* H1 wide-DataFrame to H4 / D1 (last close per bucket).

    Resampling AFTER inner-join avoids the trap where different pairs have
    different weekend/gap buckets — which would make a per-pair resample leave
    a wide DataFrame whose every row has at least one NaN.
    """
    if tf == "H1":
        return prices_h1
    rule = {"H4": "4h", "D1": "1D"}[tf]
    return prices_h1.resample(rule, label="right", closed="right").last().dropna()


# Load H1 once; H4 is a cheap resample on top.
all_h1 = load_all_h1_aligned()
print(f"H1 aligned (full history): {all_h1.shape}")
print(f"H1 range: {all_h1.index.min()} to {all_h1.index.max()}")

_demo_h4 = to_tf(all_h1, "H4").loc[IN_SAMPLE_START:IN_SAMPLE_END]
print(f"H4 in-sample sanity: {_demo_h4.shape}  (range {_demo_h4.index.min()} to {_demo_h4.index.max()})")
_demo_h4.head()


# ## ۲) Currency-graph: rank check
# 
# این universe ۲۸-تایی به‌طور ریاضی rank-deficient‌ـه (۸ ارز → rank max=7، پس ۲۱ پیر redundant‌ـه). فقط flag می‌کنیم — فیلتر اصلی برای synthetic-trap در §5.

# In[13]:


parsed = [parse_symbol(s, DEFAULT_KNOWN_CURRENCIES, suffix_strip=()) for s in UNIVERSE]
graph = CurrencyGraph(parsed)
print(f"currencies: {graph.currencies}")
print(f"graph rank: {graph.rank()} / {len(graph)}  |  rank-deficient: {graph.is_rank_deficient()}")
print(f"redundant symbols: {len(graph.redundant_pairs())}")


# ## ۳) Cointegration & filter pipeline (functions)
# 
# همه‌ی منطق در توابع — یک بار تعریف، چند بار اجرا برای H1 و H4.

# In[14]:


def cointegration_test(y: np.ndarray, x: np.ndarray) -> dict:
    """Engle-Granger on a single (y, x) log-price pair."""
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta = float(coef[0]), float(coef[1])
    resid = y - (alpha + beta * x)
    _, adf_p, *_ = adfuller(resid, regression="n", autolag="AIC")
    d_resid   = np.diff(resid)
    lag_resid = resid[:-1]
    lam_c, *_ = np.linalg.lstsq(lag_resid.reshape(-1, 1), d_resid, rcond=None)
    lam = -float(lam_c[0])
    hl_bars = float(np.log(2) / lam) if lam > 0 else np.nan
    return dict(alpha=alpha, beta=beta, adf_p=float(adf_p),
                half_life_bars=hl_bars, resid_std=float(resid.std()))


def scan_universe(prices: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Run EG over all C(N,2) pairs (both directions); keep better p-value."""
    log_p = np.log(prices)
    bar_hours = TF_HOURS[tf]
    records, pairs = [], list(combinations(prices.columns, 2))
    for s1, s2 in pairs:
        y, x = log_p[s1].values, log_p[s2].values
        f12, f21 = cointegration_test(y, x), cointegration_test(x, y)
        if f12["adf_p"] <= f21["adf_p"]:
            rec = dict(y=s1, x=s2, **f12)
        else:
            rec = dict(y=s2, x=s1, **f21)
        rec["half_life_hours"] = rec["half_life_bars"] * bar_hours
        records.append(rec)
    return pd.DataFrame(records).sort_values("adf_p").reset_index(drop=True)


def apply_filters(scan: pd.DataFrame, prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """p-value + half-life filter, then Stage-4 lite (max third-pair correlation)."""
    shortlist = scan.query(
        "adf_p < @PVALUE_MAX and @MIN_HALF_LIFE_HOURS <= half_life_hours <= @MAX_HALF_LIFE_HOURS"
    ).reset_index(drop=True).copy()

    if shortlist.empty:
        return shortlist, shortlist

    log_p_ret = np.log(prices).diff().dropna()
    best_thirds, best_cors = [], []
    for _, row in shortlist.iterrows():
        sr = log_p_ret[row["y"]] - row["beta"] * log_p_ret[row["x"]]
        others = [c for c in log_p_ret.columns if c not in (row["y"], row["x"])]
        cors = log_p_ret[others].corrwith(sr).abs()
        best_thirds.append(cors.idxmax())
        best_cors.append(float(cors.max()))

    shortlist["max_third"]      = best_thirds
    shortlist["max_third_corr"] = best_cors
    shortlist["synthetic_flag"] = shortlist["max_third_corr"] > SPREAD_CORR_MAX
    clean = shortlist[~shortlist["synthetic_flag"]].reset_index(drop=True)
    synth = shortlist[ shortlist["synthetic_flag"]].reset_index(drop=True)
    return clean, synth


def plot_top(clean: pd.DataFrame, prices: pd.DataFrame, tf: str, n: int = TOP_N_PLOT) -> None:
    """Plot top-N spreads with ±2σ bands."""
    if clean.empty:
        print(f"[{tf}] no CLEAN candidates to plot.")
        return
    log_p = np.log(prices)
    n = min(n, len(clean))
    titles = [
        f"{r.y} ~ {r.x}  (β={r.beta:.3f},  p={r.adf_p:.3g},  HL={r.half_life_hours:.0f}h)"
        for r in clean.head(n).itertuples()
    ]
    fig = make_subplots(rows=n, cols=1, shared_xaxes=False,
                        subplot_titles=titles, vertical_spacing=0.04)
    for i, (_, r) in enumerate(clean.head(n).iterrows(), 1):
        sp = log_p[r["y"]] - r["beta"] * log_p[r["x"]] - r["alpha"]
        mu, sd = sp.mean(), sp.std()
        fig.add_trace(go.Scatter(x=sp.index, y=sp.values, mode="lines", line=dict(width=1)), row=i, col=1)
        for k, color in [(0, "gray"), (2, "orange"), (-2, "orange")]:
            fig.add_hline(y=mu + k * sd, line=dict(color=color, width=1, dash="dot"), row=i, col=1)
    fig.update_layout(height=220 * n, showlegend=False,
                      title=f"[{tf}] Top {n} Cointegrated Spreads (in-sample)")
    fig.show()


print("functions defined.")


# ## ۴) اجرای موازی H1 + H4

# In[ ]:


results: dict[str, dict] = {}

for tf in TIMEFRAMES:
    print(f"\n{'='*60}\n[{tf}] running pipeline\n{'='*60}")
    prices = to_tf(all_h1, tf).loc[IN_SAMPLE_START:IN_SAMPLE_END]
    print(f"  in-sample bars: {len(prices)}  (range {prices.index.min()} to {prices.index.max()})")

    scan = scan_universe(prices, tf)
    print(f"  scan done: {len(scan)} pairs")
    print(f"  p-value range:           [{scan['adf_p'].min():.2e}, {scan['adf_p'].max():.2e}]")
    print(f"  half-life (hours) range: [{scan['half_life_hours'].min():.0f}, {scan['half_life_hours'].max():.0f}]")

    clean, synth = apply_filters(scan, prices)
    pre_stage4 = len(clean) + len(synth)
    print(f"  filter p<{PVALUE_MAX} + {MIN_HALF_LIFE_HOURS}h <= HL <= {MAX_HALF_LIFE_HOURS}h:  {pre_stage4} pass")
    print(f"  Stage-4 lite (drop max_third_corr > {SPREAD_CORR_MAX}):")
    print(f"    CLEAN:     {len(clean)}")
    print(f"    synthetic: {len(synth)}")

    results[tf] = dict(prices=prices, scan=scan, clean=clean, synth=synth)


# ## ۵) خلاصه و مقایسه

# In[ ]:


summary_rows = []
for tf, r in results.items():
    summary_rows.append(dict(
        timeframe=tf,
        bars=len(r["prices"]),
        scanned=len(r["scan"]),
        passed_filter=len(r["clean"]) + len(r["synth"]),
        clean=len(r["clean"]),
        synthetic_dropped=len(r["synth"]),
        median_hl_hours=float(r["scan"]["half_life_hours"].median()),
        min_hl_hours=float(r["scan"]["half_life_hours"].min()),
    ))
summary = pd.DataFrame(summary_rows)
print("=== summary across timeframes ===")
print(summary.to_string(index=False))


# ## ۶) Top-15 CLEAN در هر TF

# In[ ]:


for tf, r in results.items():
    print(f"\n=== [{tf}] CLEAN top 15 ===")
    cols = ["y", "x", "beta", "adf_p", "half_life_hours", "max_third", "max_third_corr"]
    if r["clean"].empty:
        print("  (empty)")
    else:
        print(r["clean"][cols].head(15).to_string(index=False))


# ## ۷) چارت Top spreads

# In[ ]:


for tf, r in results.items():
    plot_top(r["clean"], r["prices"], tf)


# ## ۸) Save shortlists
# 
# هر TF یه فایل جداگانه: `cointegrated_shortlist_{TF}.csv`. نوت‌بوک ۲۶ (Backtest) همین فایل‌ها رو می‌خونه.

# In[ ]:


for tf, r in results.items():
    out = OUTPUT_DIR / f"cointegrated_shortlist_{tf}.csv"
    r["clean"].to_csv(out, index=False)
    print(f"  [{tf}] saved {len(r['clean'])} candidates -> {out}")

print("\nnext (notebook 26): load shortlist(s) -> z-score signals -> backtest with realistic costs (spread + commission + swap).")

