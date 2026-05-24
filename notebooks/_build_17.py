"""Builds notebooks/17_h4_erl_ifvg_model.ipynb from a list of cells."""
import json
from pathlib import Path

CELLS: list[tuple[str, str]] = []  # (cell_type, source)

def md(s): CELLS.append(("markdown", s))
def code(s): CELLS.append(("code", s))


md(r"""# Notebook #17 — H4 ERL Sweep → M15 IFVG Retest (ICT Delivery Cycle)
## مدل سواپ نقدینگی بیرونی H4 + ورودی IFVG روی M15 — XAUUSD

---

### ایده‌ی استراتژی (به زبان ICT/SMC)

قیمت همیشه بین دو سطح نقدینگی در حال انتقال است:

```
External Liquidity  ──sweep──▶  Internal Liquidity  ──rebalance──▶  External (مخالف)
```

این چرخه (delivery cycle) روی هر تایم‌فریمی تکرار می‌شود. در این نوت‌بوک از **H4 برای بایاس** و از **M15 برای اجرا** استفاده می‌کنیم — همان جفت تایم‌فریمی که سخنران ویدئو معرفی می‌کند.

### مراحل مدل

1. **بایاس H4 — تعریف ERL**
   - برای هر کندل H4 نزدیک‌ترین Swing High و Swing Low (در یک پنجره‌ی N کندلی) به‌عنوان ERL درنظر گرفته می‌شود.
   - میانه‌ی این محدوده = `equilibrium` (یکی از کاندیداهای IRL).

2. **سواپ ERL روی H4**
   - **Bearish sweep (→ SELL):** فتیله از Swing High بالاتر می‌رود ولی کندل **زیر** Swing High بسته می‌شود.
   - **Bullish sweep (→ BUY):** فتیله از Swing Low پایین‌تر می‌رود ولی کندل **بالای** Swing Low بسته می‌شود.

3. **Displacement روی M15**
   - بعد از سواپ، انتظار داریم حرکت تهاجمی (displacement) در جهت مخالف سواپ ظاهر شود.
   - این displacement روی M15 یک یا چند **Fair Value Gap (FVG)** باقی می‌گذارد.

4. **IFVG — Inverted Fair Value Gap**
   - یک **bullish FVG** وقتی **inverted** می‌شود که قیمت با بدنه از پایین آن عبور و بسته شود → دیگر FVG حمایتی نیست، بلکه **مقاومت** است.
   - یک **bearish FVG** وقتی inverted می‌شود که قیمت بالای آن بسته شود → **حمایت** می‌شود.
   - این IFVG ناحیه‌ی ورود ماست.

5. **ورود (Retest of IFVG)**
   - برای SELL: لیمیت در میانه‌ی IFVG (bullish FVG که inverted شده).
   - برای BUY: لیمیت در میانه‌ی IFVG (bearish FVG که inverted شده).
   - SL پشت فتیله‌ی سواپ H4 + بافر.
   - TP حداقل **2R** (طبق ویدئو) — اگر فاصله تا equilibrium کمتر از 2R بود، SL را تنگ‌تر می‌کنیم تا دقیقاً 2R شود؛ اگر این تنگ‌تر کردن SL را داخل خود IFVG ببرد، setup رد می‌شود.

---

| پارامتر | مقدار |
|---|---|
| نماد | XAUUSD |
| TF بایاس | H4 (resample از H1) |
| TF اجرا | M15 (resample از M5) |
| H4 swing window | 12 کندل (≈ 2 روز) |
| سواپ حداقل | ≥ 1.5 USD فراتر از ERL |
| پنجره‌ی displacement | ۸ کندل M15 (= 2 ساعت) |
| پنجره‌ی ورود (لیمیت معتبر) | ۲۴ کندل M15 (= 6 ساعت) |
| نسبت R:R هدف | 2.0 (قفل‌شده) |
| Max trade duration | ۹۶ کندل M15 (= 24h) |
""")

md(r"""## Step 1 — Imports & Configuration""")

code(r"""import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

pio.renderers.default = 'notebook'
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.4f}'.format)

# ── Strategy identity ────────────────────────────────────────────────────────
STRATEGY_NAME    = 'h4_erl_ifvg_model'
SYMBOL           = 'XAUUSD'
DATA_DIR         = Path('./data')
RESULTS_DIR      = Path('./results') / STRATEGY_NAME / SYMBOL
LOOKBACK_DAYS    = 120

# Broker clock: broker = NY + 7h (per data_timezone memory). We keep naive timestamps.
BROKER_TO_NY_OFFSET_H = 7

# ── HTF: H4 ERL definition ───────────────────────────────────────────────────
H4_SWING_WINDOW  = 12          # rolling lookback for ERL high/low (12 H4 bars ≈ 2 days)
ERL_MIN_WIDTH    = 15.0        # min H4 range width (USD); too-tight ranges are skipped
SWEEP_MIN        = 1.5         # min USD by which wick exceeds ERL
SWEEP_MAX_BODY   = 0.45        # H4 close-body / bar-range ratio (we want a rejection wick)

# ── LTF: M15 IFVG entry ──────────────────────────────────────────────────────
FVG_LOOKBACK_BARS    = 24      # how far back (M15 bars) to scan for the FVG that gets inverted
DISPLACEMENT_BARS    = 8       # bars after sweep in which IFVG inversion must occur
ENTRY_WINDOW_BARS    = 24      # after IFVG, how many M15 bars the limit stays valid
FVG_MIN_WIDTH        = 0.5     # min FVG height (USD) to qualify
FVG_BODY_MULT        = 1.0     # middle-candle body must exceed body_mult × avg(body, last 10)
DISPLACEMENT_BODY_MULT = 1.3   # inversion candle body must be aggressive (× avg)

# ── Risk / target ───────────────────────────────────────────────────────────
SL_BUFFER        = 0.5         # USD beyond H4 sweep wick
RR_TARGET        = 2.0         # video: minimum 2R, lock SL geometry to exactly 2R
MAX_TRADE_BARS   = 96          # 24h on M15

print('H4 ERL Sweep → M15 IFVG Retest — Configuration')
print(f'  Symbol           : {SYMBOL}')
print(f'  Bias TF          : H4   (swing window = {H4_SWING_WINDOW} bars)')
print(f'  Exec TF          : M15  (resampled from M5)')
print(f'  Sweep min        : {SWEEP_MIN} USD beyond ERL')
print(f'  ERL min width    : {ERL_MIN_WIDTH} USD')
print(f'  FVG lookback     : {FVG_LOOKBACK_BARS} M15 bars before sweep')
print(f'  Displacement win : {DISPLACEMENT_BARS} M15 bars (IFVG must form inside this)')
print(f'  Entry window     : {ENTRY_WINDOW_BARS} M15 bars (limit valid)')
print(f'  Risk target      : exactly {RR_TARGET}R (SL tightened to match)')
print(f'  Max trade bars   : {MAX_TRADE_BARS} (24h M15)')
print(f'  Results dir      : {RESULTS_DIR}')
""")

md(r"""## Step 2 — Load H1 + M5 (broker wall-clock)

Timestamps are loaded as **naive broker time** — the `+00:00` suffix in the CSVs is misleading. Per the `data_timezone` rule, broker == NY + 7h. No tz conversion needed inside the notebook.""")

code(r"""def load_ohlcv(symbol: str, tf: str, lookback_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    path = DATA_DIR / symbol / tf / 'ohlcv.csv'
    if not path.exists():
        raise FileNotFoundError(f'Missing: {path}')
    df = pd.read_csv(path)
    df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)   # naive broker clock
    df = df.dropna(subset=['time']).sort_values('time').reset_index(drop=True)
    keep = ['time', 'open', 'high', 'low', 'close', 'tick_volume']
    df = df[[c for c in keep if c in df.columns]].copy()
    df.rename(columns={'tick_volume': 'volume'}, inplace=True)
    cutoff = df['time'].max() - pd.Timedelta(days=lookback_days)
    return df[df['time'] >= cutoff].reset_index(drop=True)


df_h1 = load_ohlcv(SYMBOL, 'H1')
df_m5 = load_ohlcv(SYMBOL, 'M5')

print(f'H1 bars : {len(df_h1):>8,}  [{df_h1["time"].min()} → {df_h1["time"].max()}]')
print(f'M5 bars : {len(df_m5):>8,}  [{df_m5["time"].min()} → {df_m5["time"].max()}]')
df_h1.tail(3)
""")

md(r"""## Step 3 — Resample to H4 and M15

H4 bars use the broker-clock 4-hour grid (origin = epoch, no NY anchor; the H4 ERL we care about is structural, not session-aligned). M15 likewise rolls up from M5.""")

code(r"""def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'}
    out = (df.set_index('time')
             .resample(rule, label='left', closed='left')
             .agg(agg)
             .dropna(subset=['open','high','low','close'])
             .reset_index())
    return out


df_h4  = resample(df_h1, '4h')
df_m15 = resample(df_m5, '15min')

print(f'H4  bars : {len(df_h4):>6,}  [{df_h4["time"].min()} → {df_h4["time"].max()}]')
print(f'M15 bars : {len(df_m15):>6,}  [{df_m15["time"].min()} → {df_m15["time"].max()}]')
df_h4.tail(3)
""")

md(r"""## Step 4 — H4 ERL (External Range Liquidity)

For each H4 bar we take the rolling **swing high / swing low** over the previous `H4_SWING_WINDOW` bars *(strictly prior to the current bar — no look-ahead)*. These two levels are the ERL. The midpoint is the H4 equilibrium and is our default IRL target.""")

code(r"""def compute_h4_erl(df_h4: pd.DataFrame, window: int = H4_SWING_WINDOW) -> pd.DataFrame:
    df = df_h4.copy()
    # Shifted rolling → uses bars strictly before current bar
    df['erl_high'] = df['high'].shift(1).rolling(window, min_periods=window).max()
    df['erl_low']  = df['low'].shift(1).rolling(window, min_periods=window).min()
    df['eq']       = (df['erl_high'] + df['erl_low']) / 2.0
    df['width']    = df['erl_high'] - df['erl_low']
    df['valid_range'] = df['width'] >= ERL_MIN_WIDTH
    return df


df_h4 = compute_h4_erl(df_h4)
valid = int(df_h4['valid_range'].sum())
print(f'H4 bars with valid ERL : {valid} / {len(df_h4)}')
print(f'Mean range width       : {df_h4["width"].mean():.2f} USD')
print(f'Median range width     : {df_h4["width"].median():.2f} USD')
df_h4[['time','erl_high','erl_low','eq','width','valid_range']].tail(5)
""")

md(r"""## Step 5 — H4 ERL Sweep Detection (Trigger)

**Bearish ERL Sweep** (→ SELL bias):
- `high > erl_high + SWEEP_MIN`  (wick crosses the swing high)
- `close < erl_high`             (close back inside → rejection)
- body / range < `SWEEP_MAX_BODY` (small body + long wick)

**Bullish ERL Sweep** (→ BUY bias) is the mirror.""")

code(r"""def detect_h4_sweeps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['sweep_type']   = None
    df['swept_level']  = np.nan
    df['sweep_extent'] = np.nan

    h, l, o, c = df['high'].values, df['low'].values, df['open'].values, df['close'].values
    erl_h = df['erl_high'].values
    erl_l = df['erl_low'].values
    valid = df['valid_range'].fillna(False).values

    for i in range(len(df)):
        if not valid[i] or np.isnan(erl_h[i]) or np.isnan(erl_l[i]):
            continue
        bar_range = h[i] - l[i]
        if bar_range <= 0:
            continue
        body_ratio = abs(c[i] - o[i]) / bar_range

        if (h[i] > erl_h[i] + SWEEP_MIN and c[i] < erl_h[i]
                and body_ratio < SWEEP_MAX_BODY):
            df.iat[i, df.columns.get_loc('sweep_type')]   = 'bearish_sweep'
            df.iat[i, df.columns.get_loc('swept_level')]  = erl_h[i]
            df.iat[i, df.columns.get_loc('sweep_extent')] = h[i] - erl_h[i]
            continue
        if (l[i] < erl_l[i] - SWEEP_MIN and c[i] > erl_l[i]
                and body_ratio < SWEEP_MAX_BODY):
            df.iat[i, df.columns.get_loc('sweep_type')]   = 'bullish_sweep'
            df.iat[i, df.columns.get_loc('swept_level')]  = erl_l[i]
            df.iat[i, df.columns.get_loc('sweep_extent')] = erl_l[i] - l[i]
    return df


df_h4 = detect_h4_sweeps(df_h4)
n_bear = int((df_h4['sweep_type'] == 'bearish_sweep').sum())
n_bull = int((df_h4['sweep_type'] == 'bullish_sweep').sum())
print(f'H4 Bearish Sweeps (→ SELL) : {n_bear}')
print(f'H4 Bullish Sweeps (→ BUY)  : {n_bull}')
print(f'Total H4 setup triggers     : {n_bear + n_bull}')
if (n_bear + n_bull) > 0:
    avg_ext = df_h4['sweep_extent'].dropna().mean()
    print(f'Avg sweep extent (beyond ERL): {avg_ext:.2f} USD')
df_h4[df_h4['sweep_type'].notna()][['time','sweep_type','swept_level','sweep_extent','high','low','close']].tail(5)
""")

md(r"""## Step 6 — M15 FVG Detection

A 3-candle gap, with the middle candle aggressive enough that this is a real displacement (not a noise gap).

- **Bullish FVG** at index `i`: `low[i] > high[i-2]`, FVG zone = `[high[i-2], low[i]]`.
- **Bearish FVG** at index `i`: `high[i] < low[i-2]`, FVG zone = `[high[i], low[i-2]]`.

Each FVG also tracks whether it has been **inverted** later (price closed through to the opposite side) — that inversion is what creates the IFVG we trade.""")

code(r"""def detect_m15_fvgs(df: pd.DataFrame,
                    lookback_avg: int = 10,
                    body_mult: float = FVG_BODY_MULT,
                    min_width: float = FVG_MIN_WIDTH) -> pd.DataFrame:
    df = df.copy()
    n = len(df)
    fvg_type = [None] * n
    fvg_low  = np.full(n, np.nan)
    fvg_high = np.full(n, np.nan)

    high = df['high'].values
    low  = df['low'].values
    op   = df['open'].values
    cl   = df['close'].values

    for i in range(2, n):
        s = max(0, i - 1 - lookback_avg)
        if i - 1 - s <= 0:
            continue
        avg_body = np.mean(np.abs(cl[s:i-1] - op[s:i-1]))
        if avg_body <= 0:
            continue
        mid_body = abs(cl[i-1] - op[i-1])
        if mid_body < body_mult * avg_body:
            continue

        if low[i] > high[i-2]:
            width = low[i] - high[i-2]
            if width >= min_width:
                fvg_type[i] = 'bullish'
                fvg_low[i]  = high[i-2]
                fvg_high[i] = low[i]
        elif high[i] < low[i-2]:
            width = low[i-2] - high[i]
            if width >= min_width:
                fvg_type[i] = 'bearish'
                fvg_low[i]  = high[i]
                fvg_high[i] = low[i-2]

    df['fvg_type'] = fvg_type
    df['fvg_low']  = fvg_low
    df['fvg_high'] = fvg_high
    df['fvg_mid']  = (df['fvg_low'] + df['fvg_high']) / 2.0
    return df


df_m15 = detect_m15_fvgs(df_m15)
n_bull_fvg = int((df_m15['fvg_type'] == 'bullish').sum())
n_bear_fvg = int((df_m15['fvg_type'] == 'bearish').sum())
print(f'M15 Bullish FVGs : {n_bull_fvg}')
print(f'M15 Bearish FVGs : {n_bear_fvg}')
print(f'Total            : {n_bull_fvg + n_bear_fvg}')
""")

md(r"""## Step 7 — Find IFVG and Build Trade Plans

For every H4 sweep we walk forward on M15:

- For a **bearish H4 sweep** (→ SELL): we need a **bullish FVG that becomes inverted** — i.e. an M15 candle closes **below** the FVG bottom **with an aggressive body**. The FVG zone then acts as bearish resistance — that is our IFVG (sell-on-retest zone).
- For a **bullish H4 sweep** (→ BUY): mirror — a **bearish FVG that becomes inverted** (close **above** the FVG top, aggressive body) → bullish support → buy-on-retest.

The qualifying FVG must be **recent** (within the last `FVG_LOOKBACK_BARS` M15 bars *before* the H4 sweep — it's old structure that's now being broken).

**Order placement**
- Entry: limit at FVG midpoint.
- SL: pessimistically tightened to lock exactly `RR_TARGET` R **relative to TP** = H4 equilibrium. If that tightened SL would sit inside the FVG zone, we skip (the structure is too compressed for a clean 2R).""")

code(r"""def find_qualifying_ifvg(df_m15: pd.DataFrame,
                          sweep_time: pd.Timestamp,
                          direction: str) -> Optional[Dict]:
    # SELL → recent BULLISH FVG closed below with aggressive body (now bearish IFVG).
    # BUY  → recent BEARISH FVG closed above with aggressive body (now bullish IFVG).
    # Returns None if no qualifying inversion occurs in DISPLACEMENT_BARS after sweep.
    # Index of first M15 bar at or after the sweep time
    forward = df_m15[df_m15['time'] >= sweep_time]
    if forward.empty:
        return None
    sweep_i = forward.index[0]

    # FVGs in the recent past (must be old structure to invert)
    lo_i = max(0, sweep_i - FVG_LOOKBACK_BARS)
    history = df_m15.iloc[lo_i:sweep_i]

    if direction == 'SELL':
        candidates = history[history['fvg_type'] == 'bullish']
    else:
        candidates = history[history['fvg_type'] == 'bearish']
    if candidates.empty:
        return None

    high = df_m15['high'].values
    low  = df_m15['low'].values
    op   = df_m15['open'].values
    cl   = df_m15['close'].values

    end_i = min(len(df_m15), sweep_i + DISPLACEMENT_BARS + 1)

    for k in range(sweep_i, end_i):
        body = abs(cl[k] - op[k])
        s = max(0, k - 10)
        avg_body = float(np.mean(np.abs(cl[s:k] - op[s:k]))) if k - s > 0 else 0.0
        if avg_body <= 0 or body < DISPLACEMENT_BODY_MULT * avg_body:
            continue

        # Iterate from the most recent FVG backward — newest inversion wins
        for fv_idx, fv in candidates[::-1].iterrows():
            f_low  = float(fv['fvg_low'])
            f_high = float(fv['fvg_high'])
            if direction == 'SELL':
                # bullish FVG; inversion = close below f_low
                if cl[k] < f_low and op[k] >= f_low - 5 * (f_high - f_low):
                    # additionally, the bar can't have inverted before (older fv preferred)
                    return {
                        'ifvg_index'   : k,
                        'ifvg_time'    : df_m15.iloc[k]['time'],
                        'fvg_origin'   : fv['time'],
                        'fvg_low'      : f_low,
                        'fvg_high'     : f_high,
                        'fvg_mid'      : (f_low + f_high) / 2.0,
                        'fvg_type'     : 'bullish',
                        'inversion_close': float(cl[k]),
                    }
            else:
                # bearish FVG; inversion = close above f_high
                if cl[k] > f_high and op[k] <= f_high + 5 * (f_high - f_low):
                    return {
                        'ifvg_index'   : k,
                        'ifvg_time'    : df_m15.iloc[k]['time'],
                        'fvg_origin'   : fv['time'],
                        'fvg_low'      : f_low,
                        'fvg_high'     : f_high,
                        'fvg_mid'      : (f_low + f_high) / 2.0,
                        'fvg_type'     : 'bearish',
                        'inversion_close': float(cl[k]),
                    }
    return None


def build_trade_plans(df_h4: pd.DataFrame, df_m15: pd.DataFrame) -> pd.DataFrame:
    plans = []
    sweeps = df_h4[df_h4['sweep_type'].notna()]
    for _, sw in sweeps.iterrows():
        direction = 'SELL' if sw['sweep_type'] == 'bearish_sweep' else 'BUY'
        ifvg = find_qualifying_ifvg(df_m15, sw['time'], direction)
        if ifvg is None:
            plans.append({
                'sweep_time'   : sw['time'],
                'direction'    : direction,
                'sweep_type'   : sw['sweep_type'],
                'erl_high'     : float(sw['erl_high']),
                'erl_low'      : float(sw['erl_low']),
                'eq'           : float(sw['eq']),
                'sweep_high'   : float(sw['high']),
                'sweep_low'    : float(sw['low']),
                'status_plan'  : 'NO_IFVG',
            })
            continue

        # Entry / stop / target geometry
        entry = ifvg['fvg_mid']
        if direction == 'SELL':
            tp = float(sw['eq'])                       # H4 equilibrium (IRL)
            raw_sl = float(sw['high']) + SL_BUFFER     # behind H4 sweep wick
            reward = entry - tp
            risk   = raw_sl - entry
            if reward <= 0 or risk <= 0:
                plans.append({**ifvg, 'sweep_time': sw['time'], 'direction': direction,
                              'sweep_type': sw['sweep_type'],
                              'erl_high': float(sw['erl_high']), 'erl_low': float(sw['erl_low']),
                              'eq': float(sw['eq']), 'sweep_high': float(sw['high']),
                              'sweep_low': float(sw['low']),
                              'status_plan': 'INVALID_GEOMETRY'})
                continue
            # Tighten SL to lock exactly RR_TARGET R
            tight_sl = entry + reward / RR_TARGET
            sl = min(raw_sl, tight_sl)
            # Reject if tightened SL sits INSIDE the IFVG zone
            if sl <= ifvg['fvg_high']:
                plans.append({**ifvg, 'sweep_time': sw['time'], 'direction': direction,
                              'sweep_type': sw['sweep_type'],
                              'erl_high': float(sw['erl_high']), 'erl_low': float(sw['erl_low']),
                              'eq': float(sw['eq']), 'sweep_high': float(sw['high']),
                              'sweep_low': float(sw['low']),
                              'status_plan': 'SL_IN_IFVG'})
                continue
        else:  # BUY
            tp = float(sw['eq'])
            raw_sl = float(sw['low']) - SL_BUFFER
            reward = tp - entry
            risk   = entry - raw_sl
            if reward <= 0 or risk <= 0:
                plans.append({**ifvg, 'sweep_time': sw['time'], 'direction': direction,
                              'sweep_type': sw['sweep_type'],
                              'erl_high': float(sw['erl_high']), 'erl_low': float(sw['erl_low']),
                              'eq': float(sw['eq']), 'sweep_high': float(sw['high']),
                              'sweep_low': float(sw['low']),
                              'status_plan': 'INVALID_GEOMETRY'})
                continue
            tight_sl = entry - reward / RR_TARGET
            sl = max(raw_sl, tight_sl)
            if sl >= ifvg['fvg_low']:
                plans.append({**ifvg, 'sweep_time': sw['time'], 'direction': direction,
                              'sweep_type': sw['sweep_type'],
                              'erl_high': float(sw['erl_high']), 'erl_low': float(sw['erl_low']),
                              'eq': float(sw['eq']), 'sweep_high': float(sw['high']),
                              'sweep_low': float(sw['low']),
                              'status_plan': 'SL_IN_IFVG'})
                continue

        # Entry deadline = ENTRY_WINDOW_BARS M15 after IFVG forms
        ifvg_i = int(ifvg['ifvg_index'])
        deadline_i = min(len(df_m15) - 1, ifvg_i + ENTRY_WINDOW_BARS)
        entry_deadline = df_m15.iloc[deadline_i]['time']

        plans.append({
            'sweep_time'    : sw['time'],
            'direction'     : direction,
            'sweep_type'    : sw['sweep_type'],
            'erl_high'      : float(sw['erl_high']),
            'erl_low'       : float(sw['erl_low']),
            'eq'            : float(sw['eq']),
            'sweep_high'    : float(sw['high']),
            'sweep_low'     : float(sw['low']),
            'ifvg_time'     : ifvg['ifvg_time'],
            'ifvg_index'    : ifvg_i,
            'fvg_origin'    : ifvg['fvg_origin'],
            'fvg_low'       : ifvg['fvg_low'],
            'fvg_high'      : ifvg['fvg_high'],
            'fvg_type'      : ifvg['fvg_type'],
            'entry_price'   : round(entry, 4),
            'sl'            : round(sl, 4),
            'tp'            : round(tp, 4),
            'risk'          : round(abs(entry - sl), 4),
            'reward'        : round(abs(tp - entry), 4),
            'planned_rr'    : round(abs(tp - entry) / abs(entry - sl), 3),
            'entry_deadline': entry_deadline,
            'status_plan'   : 'READY',
        })
    return pd.DataFrame(plans)


plans_df = build_trade_plans(df_h4, df_m15)
print(f'Total H4 sweeps      : {len(plans_df)}')
print(f"  → READY            : {(plans_df['status_plan']=='READY').sum()}")
print(f"  → NO_IFVG          : {(plans_df['status_plan']=='NO_IFVG').sum()}")
print(f"  → SL_IN_IFVG       : {(plans_df['status_plan']=='SL_IN_IFVG').sum()}")
print(f"  → INVALID_GEOMETRY : {(plans_df['status_plan']=='INVALID_GEOMETRY').sum()}")
if (plans_df['status_plan']=='READY').any():
    rdy = plans_df[plans_df['status_plan']=='READY']
    print(f"  Avg planned R:R    : {rdy['planned_rr'].mean():.2f}  (target {RR_TARGET})")
plans_df[plans_df['status_plan']=='READY'].head(5)
""")

md(r"""## Step 8 — Trade Simulator (M15, pessimistic intrabar)

For each `READY` plan:
1. Wait for **limit fill**: the M15 bar's range touches `entry_price` within `ENTRY_WINDOW_BARS`.
2. If filled, simulate until SL or TP within `MAX_TRADE_BARS`. If both hit on the same bar → SL (worst case).
""")

code(r"""def simulate_trade(df_m15: pd.DataFrame, plan: pd.Series) -> dict:
    direction = plan['direction']
    entry     = float(plan['entry_price'])
    sl        = float(plan['sl'])
    tp        = float(plan['tp'])
    risk      = abs(entry - sl)

    ifvg_i = int(plan['ifvg_index'])
    fill_start = ifvg_i + 1
    fill_end   = min(len(df_m15), fill_start + ENTRY_WINDOW_BARS)
    fill_window = df_m15.iloc[fill_start:fill_end]
    if fill_window.empty:
        return {'status': 'NO_FILL', 'result': None, 'pnl_r': 0.0}

    # Direction-specific limit fill check
    if direction == 'SELL':
        fill_hits = fill_window[fill_window['high'] >= entry]
    else:
        fill_hits = fill_window[fill_window['low']  <= entry]
    if fill_hits.empty:
        return {'status': 'NO_FILL', 'result': None, 'pnl_r': 0.0}

    fill_bar = fill_hits.iloc[0]
    entry_time = fill_bar['time']
    entry_idx  = fill_hits.index[0]

    # IMMEDIATE SL/TP check on the FILL bar (pessimistic — if both touched, SL wins)
    if direction == 'SELL':
        if fill_bar['high'] >= sl:
            return {'status':'FILLED','result':'SL','entry_time':entry_time,
                    'exit_time':fill_bar['time'],'exit_price':sl,'pnl_r':-1.0,'bars_held':0}
        if fill_bar['low']  <= tp:
            return {'status':'FILLED','result':'TP','entry_time':entry_time,
                    'exit_time':fill_bar['time'],'exit_price':tp,'pnl_r':RR_TARGET,'bars_held':0}
    else:
        if fill_bar['low']  <= sl:
            return {'status':'FILLED','result':'SL','entry_time':entry_time,
                    'exit_time':fill_bar['time'],'exit_price':sl,'pnl_r':-1.0,'bars_held':0}
        if fill_bar['high'] >= tp:
            return {'status':'FILLED','result':'TP','entry_time':entry_time,
                    'exit_time':fill_bar['time'],'exit_price':tp,'pnl_r':RR_TARGET,'bars_held':0}

    # Subsequent bars
    mgmt_start = entry_idx + 1
    mgmt_end   = min(len(df_m15), mgmt_start + MAX_TRADE_BARS)
    mgmt = df_m15.iloc[mgmt_start:mgmt_end].reset_index(drop=True)

    for i, b in mgmt.iterrows():
        if direction == 'SELL':
            hit_sl = b['high'] >= sl
            hit_tp = b['low']  <= tp
        else:
            hit_sl = b['low']  <= sl
            hit_tp = b['high'] >= tp
        if hit_sl and hit_tp:
            return {'status':'FILLED','result':'SL','entry_time':entry_time,
                    'exit_time':b['time'],'exit_price':sl,'pnl_r':-1.0,'bars_held':i+1}
        if hit_sl:
            return {'status':'FILLED','result':'SL','entry_time':entry_time,
                    'exit_time':b['time'],'exit_price':sl,'pnl_r':-1.0,'bars_held':i+1}
        if hit_tp:
            return {'status':'FILLED','result':'TP','entry_time':entry_time,
                    'exit_time':b['time'],'exit_price':tp,'pnl_r':RR_TARGET,'bars_held':i+1}

    if not mgmt.empty:
        last = mgmt.iloc[-1]
        pnl = ((entry - last['close']) / risk if direction == 'SELL'
               else (last['close'] - entry) / risk)
        return {'status':'FILLED','result':'OPEN','entry_time':entry_time,
                'exit_time':last['time'],'exit_price':float(last['close']),
                'pnl_r':round(float(pnl),3),'bars_held':len(mgmt)}
    return {'status':'FILLED','result':'OPEN','entry_time':entry_time,
            'exit_time':entry_time,'exit_price':entry,'pnl_r':0.0,'bars_held':0}


print('Simulator defined.')
""")

md(r"""## Step 9 — Backtest""")

code(r"""def run_backtest(plans_df: pd.DataFrame, df_m15: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ready = plans_df[plans_df['status_plan'] == 'READY'].copy()
    for _, plan in ready.iterrows():
        sim = simulate_trade(df_m15, plan)
        rows.append({
            'sweep_time'   : plan['sweep_time'],
            'direction'    : plan['direction'],
            'sweep_type'   : plan['sweep_type'],
            'erl_high'     : round(plan['erl_high'], 2),
            'erl_low'      : round(plan['erl_low'],  2),
            'eq'           : round(plan['eq'],       2),
            'ifvg_time'    : plan['ifvg_time'],
            'fvg_origin'   : plan['fvg_origin'],
            'fvg_low'      : round(plan['fvg_low'],  2),
            'fvg_high'     : round(plan['fvg_high'], 2),
            'fvg_type'     : plan['fvg_type'],
            'entry_price'  : plan['entry_price'],
            'sl'           : plan['sl'],
            'tp'           : plan['tp'],
            'risk'         : plan['risk'],
            'planned_rr'   : plan['planned_rr'],
            **sim,
        })
    return pd.DataFrame(rows)


trades_df = run_backtest(plans_df, df_m15)

if trades_df.empty:
    print('No trades generated.')
else:
    n_plans  = len(trades_df)
    n_fill   = int((trades_df['status'] == 'FILLED').sum())
    n_nofill = n_plans - n_fill
    print(f'READY setups   : {n_plans}')
    print(f'Filled trades  : {n_fill}  ({n_fill / max(1,n_plans) * 100:.1f}%)')
    print(f'No-fill        : {n_nofill}')
    if n_fill:
        vc = trades_df[trades_df['status']=='FILLED']['result'].value_counts()
        for k, v in vc.items():
            print(f'  {str(k):5s} : {v}')
    trades_df.head(6)
""")

md(r"""## Step 10 — Performance Metrics""")

code(r"""def calc_metrics(trades_df: pd.DataFrame) -> dict:
    if trades_df.empty:
        return {}
    closed = trades_df[trades_df['result'].isin(['TP', 'SL'])].copy()
    if closed.empty:
        return {'closed': closed}

    n   = len(closed)
    w   = int((closed['result'] == 'TP').sum())
    wr  = w / n
    closed['cum_r'] = closed['pnl_r'].cumsum()
    dd  = closed['cum_r'] - closed['cum_r'].cummax()
    pos = closed.loc[closed['pnl_r'] > 0, 'pnl_r'].sum()
    neg = abs(closed.loc[closed['pnl_r'] < 0, 'pnl_r'].sum())
    pf  = (pos / neg) if neg > 0 else float('inf')
    arr = (closed['result'] == 'SL').astype(int).values
    max_cl = streak = 0
    for v in arr:
        streak = streak + 1 if v else 0
        max_cl = max(max_cl, streak)
    return {
        'total_trades'    : n,
        'wins'            : w,
        'losses'          : n - w,
        'win_rate'        : wr,
        'total_r'         : round(float(closed['pnl_r'].sum()), 3),
        'avg_r'           : round(float(closed['pnl_r'].mean()), 3),
        'profit_factor'   : round(float(pf), 3),
        'max_dd_r'        : round(float(dd.min()), 3),
        'max_consec_loss' : int(max_cl),
        'expectancy'      : round(wr * RR_TARGET - (1 - wr), 3),
        'avg_bars'        : round(float(closed['bars_held'].mean()), 1),
        'cum_r'           : closed['cum_r'].reset_index(drop=True),
        'drawdown'        : dd.reset_index(drop=True),
        'closed'          : closed.reset_index(drop=True),
    }


metrics = calc_metrics(trades_df)

if metrics and metrics.get('total_trades', 0):
    be_wr = 1.0 / (1.0 + RR_TARGET)
    sep = '=' * 60
    print(sep)
    print('  PERFORMANCE — H4 ERL → M15 IFVG Retest')
    print(sep)
    print(f'  Trades         : {metrics["total_trades"]}')
    print(f'  Wins / Losses  : {metrics["wins"]} / {metrics["losses"]}')
    print(f'  Win Rate       : {metrics["win_rate"]*100:.1f}%')
    print(f'  Break-even WR  : {be_wr*100:.1f}%   (R:R = {RR_TARGET})')
    print(f'  Total R        : {metrics["total_r"]:+.2f} R')
    print(f'  Avg R / trade  : {metrics["avg_r"]:+.3f} R')
    print(f'  Profit Factor  : {metrics["profit_factor"]:.2f}')
    print(f'  Expectancy     : {metrics["expectancy"]:+.3f} R')
    print(f'  Max Drawdown   : {metrics["max_dd_r"]:.2f} R')
    print(f'  Max CL streak  : {metrics["max_consec_loss"]}')
    print(f'  Avg duration   : {metrics["avg_bars"]:.0f} M15 bars')
    edge = '✅ POSITIVE' if (metrics["win_rate"] > be_wr and metrics["profit_factor"] > 1) else '❌ NEGATIVE'
    print(f'  Edge           : {edge}')
    print(sep)
else:
    print('No closed trades.')
""")

md(r"""## Step 11 — Equity Curve""")

code(r"""def plot_equity(metrics: dict, name: str = 'H4 ERL → M15 IFVG'):
    if not metrics or 'cum_r' not in metrics or metrics.get('cum_r', pd.Series(dtype=float)).empty:
        print('No equity data.')
        return
    cum_r  = metrics['cum_r']
    dd     = metrics['drawdown']
    closed = metrics['closed']

    fig = make_subplots(rows=3, cols=1, row_heights=[0.5, 0.25, 0.25],
                        subplot_titles=['Cumulative Equity (R)', 'Drawdown', 'Per-Trade PnL (R)'],
                        vertical_spacing=0.08)
    fig.add_trace(go.Scatter(x=cum_r.index, y=cum_r.values, mode='lines',
                             line=dict(color='#00E5FF', width=2.5),
                             fill='tozeroy', fillcolor='rgba(0,229,255,0.08)',
                             name='Equity'), row=1, col=1)
    fig.add_trace(go.Scatter(x=cum_r.index, y=cum_r.cummax().values, mode='lines',
                             line=dict(color='gold', width=1, dash='dot'),
                             name='Peak'), row=1, col=1)
    fig.add_hline(y=0, line_color='gray', line_dash='dash', row=1, col=1)
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, mode='lines',
                             fill='tozeroy', fillcolor='rgba(255,23,68,0.15)',
                             line=dict(color='#FF1744', width=1.5),
                             name='Drawdown'), row=2, col=1)
    colors = ['#00E676' if r == 'TP' else '#FF1744' for r in closed['result']]
    fig.add_trace(go.Bar(x=closed.index, y=closed['pnl_r'],
                         marker_color=colors, name='PnL'), row=3, col=1)
    fig.add_hline(y=0, line_color='gray', line_dash='dash', row=3, col=1)

    fig.update_layout(
        title=dict(text=(f'{name} — {SYMBOL}  ·  '
                         f'WR={metrics["win_rate"]*100:.0f}%  '
                         f'Total={metrics["total_r"]:+.1f}R  '
                         f'PF={metrics["profit_factor"]:.2f}  '
                         f'MaxDD={metrics["max_dd_r"]:.1f}R'), x=0.5),
        height=720, template='plotly_dark', showlegend=True,
    )
    fig.show()


if metrics and metrics.get('total_trades', 0):
    plot_equity(metrics)
""")

md(r"""## Step 12 — Setup Overview Chart (M15)

Shows H4 ERL band projected onto M15 candles, H4 sweep events, all IFVG zones found, and per-trade outcomes.""")

code(r"""def project_h4_to_m15(df_m15: pd.DataFrame, df_h4: pd.DataFrame) -> pd.DataFrame:
    h4 = df_h4.copy()
    h4['close_time'] = h4['time'] + pd.Timedelta(hours=4)
    h4 = h4[['close_time','erl_high','erl_low','eq']].sort_values('close_time')
    m15 = df_m15.sort_values('time').copy()
    merged = pd.merge_asof(m15, h4, left_on='time', right_on='close_time',
                            direction='backward').drop(columns=['close_time'])
    return merged


df_m15_view = project_h4_to_m15(df_m15, df_h4)


def plot_overview(df: pd.DataFrame, trades_df: pd.DataFrame, plans_df: pd.DataFrame,
                  tail_bars: int = 1500):
    data = df.tail(tail_bars).copy()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=data['time'], open=data['open'], high=data['high'],
        low=data['low'], close=data['close'], name='M15',
        increasing_line_color='#26a69a', decreasing_line_color='#ef5350'))

    fig.add_trace(go.Scatter(x=data['time'], y=data['erl_high'], mode='lines',
                             line=dict(color='#FF1744', width=1.2), name='H4 ERL High'))
    fig.add_trace(go.Scatter(x=data['time'], y=data['erl_low'], mode='lines',
                             line=dict(color='#00E676', width=1.2), name='H4 ERL Low'))
    fig.add_trace(go.Scatter(x=data['time'], y=data['eq'], mode='lines',
                             line=dict(color='#FFC107', width=1, dash='dot'),
                             name='H4 Equilibrium (IRL)'))

    view_min = data['time'].min()
    if not plans_df.empty:
        ready = plans_df[plans_df['status_plan']=='READY']
        ready = ready[ready['ifvg_time'] >= view_min]
        for _, p in ready.iterrows():
            fig.add_shape(type='rect',
                          x0=p['ifvg_time'], x1=p['entry_deadline'],
                          y0=p['fvg_low'],   y1=p['fvg_high'],
                          line=dict(color='rgba(255,193,7,0.6)', width=1),
                          fillcolor='rgba(255,193,7,0.10)')

    if not trades_df.empty:
        for d, r, sym, col in [
            ('BUY',  'TP', 'triangle-up',        '#00E676'),
            ('BUY',  'SL', 'triangle-up-open',   '#FF9800'),
            ('SELL', 'TP', 'triangle-down',      '#00B0FF'),
            ('SELL', 'SL', 'triangle-down-open', '#FF1744'),
        ]:
            sub = trades_df[(trades_df['direction']==d) & (trades_df['result']==r)]
            sub = sub[sub['entry_time'] >= view_min]
            if sub.empty: continue
            fig.add_trace(go.Scatter(
                x=sub['entry_time'], y=sub['entry_price'],
                mode='markers', name=f'{d} {r}',
                marker=dict(symbol=sym, size=11, color=col,
                            line=dict(color='white', width=1))))

    fig.update_layout(
        title=f'H4 ERL → M15 IFVG — {SYMBOL}',
        xaxis_rangeslider_visible=False,
        template='plotly_dark', height=620,
        legend=dict(orientation='h', yanchor='bottom', y=1.01))
    fig.show()


plot_overview(df_m15_view, trades_df, plans_df)
""")

md(r"""## Step 13 — Per-Trade Review (last 5 setups)""")

code(r"""def plot_trade(trade: pd.Series, df_m15: pd.DataFrame, pad: int = 40) -> go.Figure:
    ifvg_t = pd.Timestamp(trade['ifvg_time'])
    entry_t = pd.Timestamp(trade['entry_time']) if pd.notna(trade.get('entry_time')) else ifvg_t
    exit_t  = pd.Timestamp(trade['exit_time'])  if pd.notna(trade.get('exit_time'))  else entry_t

    ifvg_i  = df_m15.index[df_m15['time'] <= ifvg_t].max() if (df_m15['time'] <= ifvg_t).any() else 0
    exit_i  = df_m15.index[df_m15['time'] <= exit_t].max() if (df_m15['time'] <= exit_t).any() else ifvg_i + 30
    s = max(0, ifvg_i - pad)
    e = min(len(df_m15) - 1, exit_i + pad // 2)
    view = df_m15.iloc[s:e+1]

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=view['time'], open=view['open'], high=view['high'],
        low=view['low'], close=view['close'], name='M15',
        increasing_line_color='#26a69a', decreasing_line_color='#ef5350'))

    # IFVG zone
    fig.add_shape(type='rect',
                  x0=trade['fvg_origin'], x1=exit_t,
                  y0=trade['fvg_low'],    y1=trade['fvg_high'],
                  line=dict(color='rgba(255,193,7,0.85)', width=1),
                  fillcolor='rgba(255,193,7,0.15)')

    fig.add_hline(y=trade['erl_high'], line_color='#FF1744', line_dash='solid',
                  annotation_text=f'H4 ERL High {trade["erl_high"]:.2f}',
                  annotation_position='top right')
    fig.add_hline(y=trade['erl_low'], line_color='#00E676', line_dash='solid',
                  annotation_text=f'H4 ERL Low {trade["erl_low"]:.2f}',
                  annotation_position='bottom right')
    fig.add_hline(y=trade['eq'], line_color='#FFC107', line_dash='dot',
                  annotation_text=f'H4 Equilibrium {trade["eq"]:.2f}',
                  annotation_position='top left')

    fig.add_hline(y=trade['entry_price'], line_color='white', line_dash='dash',
                  annotation_text=f'Limit {trade["entry_price"]:.2f}',
                  annotation_position='top left')
    fig.add_hline(y=trade['sl'], line_color='rgba(255,23,68,0.85)', line_dash='dot',
                  annotation_text=f'SL {trade["sl"]:.2f}',
                  annotation_position='bottom left')
    fig.add_hline(y=trade['tp'], line_color='rgba(0,230,118,0.85)', line_dash='dot',
                  annotation_text=f'TP {trade["tp"]:.2f}',
                  annotation_position='top left')

    if trade.get('status') == 'FILLED':
        ec   = '#26a69a' if trade['direction']=='BUY' else '#ef5350'
        esym = 'triangle-up' if trade['direction']=='BUY' else 'triangle-down'
        fig.add_trace(go.Scatter(x=[entry_t], y=[trade['entry_price']],
                                 mode='markers+text', name='Entry',
                                 marker=dict(symbol=esym, size=16, color=ec,
                                             line=dict(color='white', width=2)),
                                 text=['ENTRY'], textposition='top center',
                                 textfont=dict(size=10, color='white')))
        result = trade.get('result')
        if result in ('TP','SL'):
            xc   = '#00E676' if result == 'TP' else '#FF1744'
            xsym = 'star' if result == 'TP' else 'x'
            fig.add_trace(go.Scatter(x=[exit_t], y=[trade['exit_price']],
                                     mode='markers+text', name=f'Exit ({result})',
                                     marker=dict(symbol=xsym, size=16, color=xc,
                                                 line=dict(color='white', width=2)),
                                     text=[result], textposition='top center',
                                     textfont=dict(size=10, color='white')))

    emoji = {'TP':'✅', 'SL':'❌', 'OPEN':'⏳', None:'⛔'}.get(trade.get('result'), '?')
    pnl   = trade.get('pnl_r', 0.0) or 0.0
    fig.update_layout(
        title=(f'{emoji} {trade["direction"]} | IFVG from {trade["fvg_type"]} FVG | '
               f'PnL: {pnl:+.2f}R | Planned RR: {trade["planned_rr"]:.2f}'),
        xaxis_rangeslider_visible=False,
        template='plotly_dark', height=560)
    return fig


if not trades_df.empty:
    filled = trades_df[trades_df['status'] == 'FILLED']
    N_SHOW = min(5, len(filled))
    for i in range(N_SHOW):
        t = filled.iloc[-(N_SHOW - i)]
        plot_trade(t, df_m15).show()
        pnl = t.get('pnl_r', 0.0) or 0.0
        print(f'  {t["direction"]:4s} | {t["sweep_time"]} → IFVG {t["ifvg_time"]} → '
              f'{(t.get("result") or "-"):4s} {pnl:+.2f}R')
""")

md(r"""## Step 14 — Save Results""")

code(r"""RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if trades_df.empty:
    print('No trades to save.')
else:
    trades_path = RESULTS_DIR / 'trades.csv'
    trades_df.to_csv(trades_path, index=False)
    print(f'Saved → {trades_path}')

    plans_path = RESULTS_DIR / 'plans.csv'
    plans_df.to_csv(plans_path, index=False)
    print(f'Saved → {plans_path}')

    if metrics and metrics.get('total_trades', 0):
        summary = pd.DataFrame([{
            'symbol'           : SYMBOL,
            'strategy'         : STRATEGY_NAME,
            'lookback_days'    : LOOKBACK_DAYS,
            'h4_swing_window'  : H4_SWING_WINDOW,
            'erl_min_width'    : ERL_MIN_WIDTH,
            'sweep_min'        : SWEEP_MIN,
            'fvg_lookback'     : FVG_LOOKBACK_BARS,
            'displacement_bars': DISPLACEMENT_BARS,
            'entry_window'     : ENTRY_WINDOW_BARS,
            'rr_target'        : RR_TARGET,
            'h4_sweeps'        : int(len(plans_df)),
            'ready_setups'     : int((plans_df['status_plan']=='READY').sum()),
            'no_ifvg'          : int((plans_df['status_plan']=='NO_IFVG').sum()),
            'sl_in_ifvg'       : int((plans_df['status_plan']=='SL_IN_IFVG').sum()),
            'filled_trades'    : int((trades_df['status']=='FILLED').sum()),
            'total_trades'     : metrics['total_trades'],
            'wins'             : metrics['wins'],
            'losses'           : metrics['losses'],
            'win_rate_pct'     : round(metrics['win_rate']*100, 2),
            'total_r'          : metrics['total_r'],
            'avg_r'            : metrics['avg_r'],
            'profit_factor'    : metrics['profit_factor'],
            'max_dd_r'         : metrics['max_dd_r'],
            'expectancy'       : metrics['expectancy'],
            'max_consec_loss'  : metrics['max_consec_loss'],
        }])
        summary_path = RESULTS_DIR / 'summary.csv'
        summary.to_csv(summary_path, index=False)
        print(f'Saved → {summary_path}')

        eq_path = RESULTS_DIR / 'equity.csv'
        metrics['closed'][['entry_time','direction','result','pnl_r','cum_r']].to_csv(eq_path, index=False)
        print(f'Saved → {eq_path}')

        display(summary.T.rename(columns={0:'value'}))
""")

md(r"""## Step 15 — Final Analysis""")

code(r"""if metrics and metrics.get('total_trades', 0):
    be_wr  = 1.0 / (1.0 + RR_TARGET)
    closed = metrics['closed']
    sep = '=' * 60
    print(sep)
    print('  H4 ERL → M15 IFVG — FINAL ANALYSIS')
    print(sep)

    print('\n[EDGE]')
    print(f'  R:R locked       : {RR_TARGET}:1')
    print(f'  Break-even WR    : {be_wr*100:.1f}%')
    print(f'  Actual WR        : {metrics["win_rate"]*100:.1f}%')
    print(f'  Expectancy       : {metrics["expectancy"]:+.3f} R/trade')
    edge = '✅ POSITIVE' if (metrics["win_rate"] > be_wr and metrics["profit_factor"] > 1) else '❌ NEGATIVE'
    print(f'  Edge             : {edge}')

    print('\n[BY DIRECTION]')
    for d in ['BUY', 'SELL']:
        sub = closed[closed['direction'] == d]
        if sub.empty: continue
        wr_d = (sub['result']=='TP').mean()
        print(f'  {d:5s} : n={len(sub):3d}  WR={wr_d*100:5.1f}%  AvgR={sub["pnl_r"].mean():+.3f}')

    print('\n[STRATEGY INSIGHTS]')
    print('  + H4 bias filters out lower-TF noise — only structural sweeps trigger')
    print('  + IFVG is a real-structure entry (mitigation), not a fixed retracement')
    print('  + 2R lock makes WR math exact: anything above 33.3% is profitable')
    print('  + Limit at IFVG midpoint gives better R:R than market execution at sweep close')

    print('\n[KNOWN WEAKNESSES]')
    print('  - Trend days: the H4 ERL just keeps extending → repeated sweep failures')
    print('  - Range too wide: equilibrium is far → SL must be very tight → SL_IN_IFVG rejections rise')
    print('  - News spikes: H4 sweep wick is huge → raw SL would be 2R+ before tightening')
    print('  - Subtle bias: IFVG midpoint may be unreachable if displacement is large')

    print('\n[NEXT-STEP OPTIMIZATIONS]')
    print('  > Tune H4_SWING_WINDOW (8, 12, 20) and re-evaluate edge')
    print('  > Replace H4 equilibrium TP with the nearest unmitigated H4 FVG (true IRL)')
    print('  > Require alignment with H4 trend direction (only counter-trend or only trend)')
    print('  > Add session filter (London / NY) — skip Asian-session sweeps which often fail')
    print('  > Try market-execution variant on aggressive displacements (entry at close, wider SL)')
    print(sep)
else:
    print('No closed trades to analyze. Try expanding LOOKBACK_DAYS or relaxing thresholds.')
""")


# ── Build the notebook ──────────────────────────────────────────────────────
def make_cell(cell_type: str, source: str, idx: int) -> dict:
    lines = source.splitlines(keepends=True)
    if not lines:
        lines = [""]
    base = {
        "cell_type": cell_type,
        "id": f"c-{idx:03d}",
        "metadata": {},
        "source": lines,
    }
    if cell_type == "code":
        base["execution_count"] = None
        base["outputs"] = []
    return base


nb = {
    "cells": [make_cell(t, s, i) for i, (t, s) in enumerate(CELLS)],
    "metadata": {
        "kernelspec": {
            "display_name": ".venv (3.13.5)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.13.5",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).parent / "17_h4_erl_ifvg_model.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Wrote {out}  ({len(CELLS)} cells)")
