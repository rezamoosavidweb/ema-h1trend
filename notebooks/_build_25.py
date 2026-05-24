#!/usr/bin/env python
# coding: utf-8

# # Pairs Trading — Notebook 1: Discovery
# 
# هدف این نوت‌بوک شناسایی **کاندیداهای cointegrated pairs** برای statistical-arbitrage از روی ۲۸ جفت FX major+cross روی H1 طی ۳ سال اخیر.
# 
# - **In-sample:** `2023-01-01` تا `2025-12-31`
# - **Out-of-sample (held out for backtest in notebook 26):** `2026-01-01` به بعد
# 
# **مراحل:**
# 1. Universe + Loader (با timezone fix طبق memory: داده‌ها برچسب UTC دارن ولی واقعاً Europe/Nicosia هستن)
# 2. Currency-graph redundancy check (با ماژول `stat_arb.identity.currency_graph`)
# 3. Engle-Granger cointegration scan روی همه‌ی C(28,2)=378 ترکیب
# 4. فیلتر اولیه: p-value < 0.05 + half-life معقول (5 تا 100 ساعت)
# 5. **Stage-4 lite**: حذف spread‌هایی که با یه پیر سوم بیش از 0.95 correlate هستن (تله‌ی synthetic identity)
# 6. Top candidates → چارت اسپرد + باند ±۲σ
# 7. ذخیره‌ی shortlist برای نوت‌بوک ۲ (backtest)
# 
# **خروجی:** `notebooks/data/stat_arb/cointegrated_shortlist.csv`

# In[ ]:


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

# In[ ]:


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
TIMEFRAME            = "H1"
IN_SAMPLE_START      = "2023-01-01"
IN_SAMPLE_END        = "2025-12-31"
OUT_OF_SAMPLE_START  = "2026-01-01"

DATA_DIR   = PROJECT_ROOT / "notebooks" / "data"
OUTPUT_DIR = PROJECT_ROOT / "notebooks" / "data" / "stat_arb"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Per project memory: CSVs label time as +00:00 but real broker time is Europe/Nicosia.
REAL_TZ = "Europe/Nicosia"

n_pairs = len(UNIVERSE) * (len(UNIVERSE) - 1) // 2
print(f"universe: {len(UNIVERSE)} symbols  |  C({len(UNIVERSE)},2) = {n_pairs} candidate pairs")


# ## ۱) Loader
# 
# ستون `time` به‌صورت UTC label می‌خونیم، بعد به Europe/Nicosia تبدیل می‌کنیم (طبق memory). همه‌ی جفت‌ها روی timestamp مشترک inner-join می‌شن. ساعات بسته‌ی بازار خودبه‌خود drop می‌شن چون داده‌ای ندارن.

# In[ ]:


def load_pair(symbol: str, timeframe: str = TIMEFRAME) -> pd.Series:
    """Load close prices for one symbol; return tz-aware Series indexed by Nicosia time."""
    path = DATA_DIR / symbol / timeframe / "ohlcv.csv"
    df = pd.read_csv(path, parse_dates=["time"])
    # parse_dates yields tz-aware UTC. Drop the (incorrect) UTC label, then
    # relabel as Nicosia — wall-clock is preserved (no shift).
    naive = df["time"].dt.tz_localize(None)
    ts = naive.dt.tz_localize(REAL_TZ, ambiguous="NaT", nonexistent="NaT")
    s = pd.Series(df["close"].values, index=ts, name=symbol)
    return s[s.index.notna()]


series = {sym: load_pair(sym) for sym in UNIVERSE}
prices_raw = pd.concat(series, axis=1)
print("raw shape:", prices_raw.shape, "|  max NaN per col:", int(prices_raw.isna().sum().max()))

prices_in = prices_raw.loc[IN_SAMPLE_START:IN_SAMPLE_END].dropna()
print("in-sample aligned shape:", prices_in.shape)
print("range:", prices_in.index.min(), "→", prices_in.index.max())
prices_in.head()


# ## ۲) Currency-graph: شناسایی redundant pairs
# 
# اگه یه جفت در فضای vector ارز قابل بیان به‌صورت ترکیب خطی integer از جفت‌های قبلی باشه، تریدش هیچ تنوعی نمی‌ده. این universe کاملاً rank-deficient‌ـه (28 پیر، 8 ارز → rank max = 7). فقط **flag** می‌کنیم و در §5 با Stage-4 lite فیلتر می‌کنیم.

# In[ ]:


parsed = [parse_symbol(s, DEFAULT_KNOWN_CURRENCIES, suffix_strip=()) for s in UNIVERSE]
graph = CurrencyGraph(parsed)
redundant = graph.redundant_pairs()
print(f"currencies in universe: {graph.currencies}")
print(f"symbols: {len(graph)}  |  graph rank: {graph.rank()}  |  rank-deficient: {graph.is_rank_deficient()}")
print(f"redundant (collinear with earlier symbols): {len(redundant)}")
print(f"  → {redundant}")


# ## ۳) Engle-Granger Cointegration Scan
# 
# برای هر جفت `(S1, S2)`:
# 1. **OLS hedge ratio:** `log(S1) = α + β · log(S2) + ε`
# 2. **ADF تست روی residual ε:** اگر stationary → cointegrated
# 3. **Half-life mean-reversion:** از `Δε_t = -λ · ε_{t-1}` → `half_life = ln(2)/λ` (بر حسب bar / ساعت)
# 
# تست دوطرفه (S1~S2 و S2~S1) — بهترین p-value نگه می‌داریم.

# In[ ]:


def cointegration_test(y: pd.Series, x: pd.Series) -> dict:
    """Engle-Granger: regress y on x, ADF on residuals, OU half-life."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta = float(coef[0]), float(coef[1])
    resid = y - (alpha + beta * x)
    adf_stat, adf_p, *_ = adfuller(resid, regression="n", autolag="AIC")
    d_resid = np.diff(resid)
    lag_resid = resid[:-1]
    lam_coef, *_ = np.linalg.lstsq(lag_resid.reshape(-1, 1), d_resid, rcond=None)
    lam = -float(lam_coef[0])
    half_life = float(np.log(2) / lam) if lam > 0 else np.nan
    return dict(
        alpha=alpha, beta=beta,
        adf_p=float(adf_p), adf_stat=float(adf_stat),
        half_life=half_life,
        resid_std=float(resid.std()),
    )


log_p = np.log(prices_in)
pairs = list(combinations(UNIVERSE, 2))

records = []
for i, (s1, s2) in enumerate(pairs, 1):
    f12 = cointegration_test(log_p[s1], log_p[s2])
    f21 = cointegration_test(log_p[s2], log_p[s1])
    if f12["adf_p"] <= f21["adf_p"]:
        rec = dict(y=s1, x=s2, **f12)
    else:
        rec = dict(y=s2, x=s1, **f21)
    records.append(rec)
    if i % 50 == 0:
        print(f"  scanned {i}/{len(pairs)}")

scan = pd.DataFrame(records).sort_values("adf_p").reset_index(drop=True)
print(f"\ndone. {len(scan)} pairs scanned.")
scan.head(15)


# ## ۴) فیلتر اولیه: p<0.05 + half-life معقول
# 
# - **p < 0.05** → spread stationary در سطح اطمینان 95٪
# - **half-life بین 5 تا 100 bar (H1)** = بین ~5 ساعت و ~4 روز کاری. کوتاه‌تر یعنی نویز / over-trading، بلندتر یعنی پوزیشن‌گیری بی‌نهایت.

# In[ ]:


shortlist = scan.query("adf_p < 0.05 and 5 <= half_life <= 100").reset_index(drop=True)
print(f"after p<0.05 + 5≤half-life≤100:  {len(shortlist)} / {len(scan)} survive")
shortlist


# ## ۵) Stage-4 lite: حذف spread‌هایی که مشابه یه پیر سوم هستن
# 
# اگه return‌های ساعتی یه spread با log-return یه پیر سوم correlation > 0.95 داشته باشن، اون spread درواقع داره همون پیر سوم رو بازسازی می‌کنه — هیچ edge مستقلی نداره و فقط هزینه‌ی دو پوزیشن رو می‌دیم.
# 
# این یه نسخه‌ی ساده‌ی Stage 4 detector کامله. detector کامل (residual ratio روی دو-regressor) رو بعداً در `stat_arb/identity/detector.py` می‌نویسیم.

# In[ ]:


SPREAD_CORR_MAX = 0.95
log_p_ret = log_p.diff().dropna()


def spread_returns(row: pd.Series) -> pd.Series:
    return log_p_ret[row["y"]] - row["beta"] * log_p_ret[row["x"]]


def max_third_pair_corr(row: pd.Series) -> tuple[str, float]:
    sr = spread_returns(row)
    others = [c for c in log_p_ret.columns if c not in (row["y"], row["x"])]
    cors = log_p_ret[others].corrwith(sr).abs()
    best = cors.idxmax()
    return best, float(cors.loc[best])


best_thirds, best_cors = [], []
for _, row in shortlist.iterrows():
    sym, c = max_third_pair_corr(row)
    best_thirds.append(sym)
    best_cors.append(c)

shortlist["max_third"]      = best_thirds
shortlist["max_third_corr"] = best_cors
shortlist["synthetic_flag"] = shortlist["max_third_corr"] > SPREAD_CORR_MAX

clean = shortlist[~shortlist["synthetic_flag"]].reset_index(drop=True)
synth = shortlist[ shortlist["synthetic_flag"]].reset_index(drop=True)

print(f"after Stage-4 lite (drop max_third_corr > {SPREAD_CORR_MAX}):")
print(f"  CLEAN candidates    : {len(clean)}")
print(f"  flagged as synthetic: {len(synth)}")
if len(synth):
    print("\n— synthetic (dropped) —")
    print(synth[["y", "x", "beta", "adf_p", "half_life", "max_third", "max_third_corr"]].head(10).to_string(index=False))
print("\n— CLEAN top 15 —")
clean.head(15)


# ## ۶) چارت Top Candidates
# 
# برای ۵ کاندید اول CLEAN: spread، میانگین in-sample، باندهای ±2σ. اگر spread خوب mean-revert کنه باید مرتب از باند بالا/پایین به وسط برگرده.

# In[ ]:


TOP_N = min(5, len(clean))


def build_spread(row: pd.Series, prices: pd.DataFrame = log_p) -> pd.Series:
    return prices[row["y"]] - row["beta"] * prices[row["x"]] - row["alpha"]


if TOP_N == 0:
    print("no CLEAN candidates to plot.")
else:
    titles = [
        f"{r.y} ~ {r.x}  (β={r.beta:.3f},  p={r.adf_p:.3g},  HL={r.half_life:.1f}h)"
        for r in clean.head(TOP_N).itertuples()
    ]
    fig = make_subplots(rows=TOP_N, cols=1, shared_xaxes=False,
                        subplot_titles=titles, vertical_spacing=0.04)
    for i, row in enumerate(clean.head(TOP_N).iterrows(), 1):
        _, r = row
        sp = build_spread(r)
        mu, sd = sp.mean(), sp.std()
        fig.add_trace(go.Scatter(x=sp.index, y=sp.values, mode="lines",
                                 line=dict(width=1)), row=i, col=1)
        for k, color in [(0, "gray"), (2, "orange"), (-2, "orange")]:
            fig.add_hline(y=mu + k * sd, line=dict(color=color, width=1, dash="dot"),
                          row=i, col=1)
    fig.update_layout(height=220 * TOP_N, showlegend=False,
                      title=f"Top {TOP_N} Cointegrated Spreads (in-sample)")
    fig.show()


# ## ۷) Save shortlist
# 
# ذخیره برای نوت‌بوک ۲۶ (Backtest).

# In[ ]:


out = OUTPUT_DIR / "cointegrated_shortlist.csv"
clean.to_csv(out, index=False)
print(f"saved {len(clean)} candidates → {out}")
print("\nnext (notebook 26): load shortlist → build z-score signals → backtest with realistic costs (spread + commission + swap).")

