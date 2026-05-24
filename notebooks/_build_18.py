"""Generate notebook 18: Strong-H/L + Fibonacci Scalper with stackable filters."""
import json
from pathlib import Path

cells = []

def md(text):
    cells.append({
        "cell_type": "markdown",
        "id": f"cell-{len(cells)}",
        "metadata": {},
        "source": text,
    })

def code(src):
    cells.append({
        "cell_type": "code",
        "id": f"cell-{len(cells)}",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src,
    })

# ─────────────────────────────────────────────────────────────────────────────
md("""# Notebook #18 — Strong High/Low + Fibonacci Scalper (M1 & M5)
### OTE-style retracement scalping with stackable indicator filters

---

### Concept

Every impulse leg between a confirmed swing low and a confirmed swing high creates a **Fibonacci retracement zone**. Price often pulls back into the **0.618 – 0.786 "OTE" zone** before continuing in the direction of the original impulse.

We exploit this with **M1 / M5 scalping**:

1. **Strong swing detection** — fractal pivots (lookback `L` bars on each side). A swing is "strong" while its opposite extreme remains unbroken. The most recent unbroken swing high (SH) and swing low (SL) define our leg.
2. **Fibonacci zones** — drawn from the leg start (0%) to the leg end (100%). We use the 0.618 – 0.786 retracement zone.
3. **Entry** — when price touches the OTE zone against the leg, we wait for a closing bar that resumes the leg direction.
4. **Risk** — SL beyond the swing extreme + ATR buffer. TP at fixed RR (default 2:1) — every variant shares the same expectancy math.

### The five strategies

| # | Strategy | Filter set |
|---|---|---|
| 1 | **01_base_fib_ote** | Pure swing + fib retracement |
| 2 | **02_rsi**          | + RSI(14) confirmation |
| 3 | **03_dma**          | + Displaced MA filter |
| 4 | **04_htf_trend**    | + H1 EMA20/50 trend alignment |
| 5 | **05_best_combo**   | All filters stacked + **spread-aware net profit** |

Each variant runs on **both M1 and M5**, gets its own weekly PnL report, and is saved to:

```
./results/{strategy}/{timeframe}/{symbol}/{trades,weekly,summary,equity}.csv
```

> Data note: timestamps are kept as **naive broker wall-clock**. The CSV's `+00:00` suffix is misleading — the underlying clock is broker time. Our logic is timezone-agnostic so this is harmless.""")

md("## Step 1 — Imports & Configuration")

code("""import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

pio.renderers.default = 'notebook'
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.5f}'.format)

# ── Identity ────────────────────────────────────────────────────────────────
STRATEGY_FAMILY  = 'fib_scalper_strong_hl'
SYMBOL           = 'EURUSD'
TIMEFRAMES       = ['M1', 'M5']
DATA_DIR         = Path('./data')
RESULTS_ROOT     = Path('./results')

# ── Date window (narrow it to keep backtests snappy) ────────────────────────
DATE_FROM        = '2024-06-01'
DATE_TO          = None       # None → up to last bar

# ── Swing / Fibonacci ───────────────────────────────────────────────────────
SWING_LOOKBACK   = 20         # bars on each side of a pivot → higher = stronger swings
FIB_LO           = 0.618      # entry zone lower bound (closer to leg end)
FIB_HI           = 0.786      # entry zone upper bound (deeper into the leg)
TRIGGER_BARS     = 5          # OTE touch must occur within last N bars

# ── Risk / reward ───────────────────────────────────────────────────────────
ATR_PERIOD       = 14
SL_ATR_BUFFER    = 0.15       # SL = swing_extreme ± SL_ATR_BUFFER · ATR
RR_TARGET        = 2.0        # TP at RR_TARGET · risk_distance
MAX_HOLD_BARS_M1 = 240        # M1: cap at 4h
MAX_HOLD_BARS_M5 = 96         # M5: cap at 8h
COOLDOWN_BARS    = 20         # bars in trading-TF after exit before next entry

# ── RSI filter ─────────────────────────────────────────────────────────────
RSI_PERIOD       = 14
RSI_LONG_MAX     = 45         # LONG only if RSI < this (some oversold-ness)
RSI_SHORT_MIN    = 55         # SHORT only if RSI > this

# ── DMA (Displaced MA) filter ──────────────────────────────────────────────
DMA_PERIOD       = 50
DMA_SHIFT        = 10         # use EMA from N bars ago (shifted right)

# ── HTF Trend filter ───────────────────────────────────────────────────────
HTF_TF           = 'H1'
HTF_EMA_FAST     = 20
HTF_EMA_SLOW     = 50

# ── Spread model (for net-profit backtest) ─────────────────────────────────
# 'spread' column is in broker points → multiply by POINT_SIZE for price units.
POINT_SIZES = {
    'XAUUSD': 0.01,   'XAGUSD': 0.001,
    'BTCUSD': 0.01,   'ETHUSD': 0.01,    'LTCUSD': 0.01,
    'XRPUSD': 0.0001, 'ADAUSD': 0.0001,  'DOGEUSD': 0.00001,
    'LNKUSD': 0.001,  'UNIUSD': 0.001,   'TRXUSD': 0.00001,
}
def get_point_size(symbol: str) -> float:
    return POINT_SIZES.get(symbol, 1e-5)   # default: 5-digit FX

print(f'{STRATEGY_FAMILY} — {SYMBOL}')
print(f'  Date window     : {DATE_FROM} → {DATE_TO or "latest"}')
print(f'  Timeframes      : {TIMEFRAMES}')
print(f'  Swing lookback  : {SWING_LOOKBACK} bars each side')
print(f'  Fib OTE zone    : [{FIB_LO:.3f}, {FIB_HI:.3f}]')
print(f'  RR              : {RR_TARGET}:1   (SL buffer = {SL_ATR_BUFFER}·ATR)')
print(f'  Cooldown        : {COOLDOWN_BARS} bars')
print(f'  Spread point    : {get_point_size(SYMBOL):.5f}')""")

# ─────────────────────────────────────────────────────────────────────────────
md("## Step 2 — Load Data (M1, M5, H1)")

code("""def load_ohlcv(symbol: str, tf: str,
               date_from: Optional[str] = DATE_FROM,
               date_to:   Optional[str] = DATE_TO) -> pd.DataFrame:
    path = DATA_DIR / symbol / tf / 'ohlcv.csv'
    if not path.exists():
        raise FileNotFoundError(f'Missing: {path}')
    df = pd.read_csv(path)
    df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)   # naive broker clock
    df = df.dropna(subset=['time']).sort_values('time').reset_index(drop=True)
    keep = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread']
    df   = df[[c for c in keep if c in df.columns]].copy()
    df.rename(columns={'tick_volume': 'volume'}, inplace=True)
    if 'spread' not in df.columns:
        df['spread'] = 0
    if date_from:
        df = df[df['time'] >= pd.Timestamp(date_from)]
    if date_to:
        df = df[df['time'] <= pd.Timestamp(date_to)]
    return df.reset_index(drop=True)


df_m1 = load_ohlcv(SYMBOL, 'M1')
df_m5 = load_ohlcv(SYMBOL, 'M5')
df_h1 = load_ohlcv(SYMBOL, 'H1')

for name, d in [('M1', df_m1), ('M5', df_m5), ('H1', df_h1)]:
    print(f'{name:>3}: {len(d):>8,} bars   [{d["time"].min()} → {d["time"].max()}]')
df_m5.tail(3)""")

# ─────────────────────────────────────────────────────────────────────────────
md("## Step 3 — Core Indicators (EMA · RSI · ATR · Swing pivots)")

code("""def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    prev_close = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low']  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()


def detect_swings(df: pd.DataFrame, lookback: int = SWING_LOOKBACK
                  ) -> Tuple[pd.Series, pd.Series]:
    \"\"\"Fractal swing detection.

    A swing high at bar i means
        high[i] == max(high[i-lookback : i+lookback+1])
    A swing low is the symmetric minimum.

    The swing is only **confirmed** `lookback` bars later, so consumers must
    use `shift(lookback)` (handled in `build_legs`) to avoid look-ahead.
    \"\"\"
    w = 2 * lookback + 1
    h_max = df['high'].rolling(w, center=True).max()
    l_min = df['low'].rolling(w,  center=True).min()
    is_sh = (df['high'] == h_max) & h_max.notna()
    is_sl = (df['low']  == l_min) & l_min.notna()
    return is_sh.fillna(False), is_sl.fillna(False)


print('Indicators ready.')""")

# ─────────────────────────────────────────────────────────────────────────────
md("""## Step 4 — Strong High/Low Tracker + Fibonacci Legs

For every bar `t`, we attach the most recent **confirmed** swing high and swing low (confirmed at `pivot_index + lookback`). Their relative order defines the active leg direction:

- Most-recent confirmed swing = **high** → previous impulse was bullish (LL → HH) → expect pullback DOWN → **LONG OTE** entry.
- Most-recent confirmed swing = **low**  → previous impulse was bearish (HH → LL) → expect pullback UP → **SHORT OTE** entry.

The leg is "strong" only while neither extreme has been broken since confirmation. Once broken, the setup is invalidated until the next pivot prints.""")

code("""def build_legs(df: pd.DataFrame, lookback: int = SWING_LOOKBACK) -> pd.DataFrame:
    \"\"\"
    Per-bar most-recent confirmed swing high / swing low + fib OTE bounds.

    Confirmation rule: a pivot at bar i is visible only from bar `i + lookback`
    onwards (so the right-side bars have actually printed).
    \"\"\"
    d = df.copy()
    is_sh, is_sl = detect_swings(d, lookback)

    sh_val  = d['high'].where(is_sh).shift(lookback)
    sh_time = d['time'].where(is_sh).shift(lookback)
    sl_val  = d['low'].where(is_sl).shift(lookback)
    sl_time = d['time'].where(is_sl).shift(lookback)

    d['recent_sh']      = sh_val.ffill()
    d['recent_sh_time'] = sh_time.ffill()
    d['recent_sl']      = sl_val.ffill()
    d['recent_sl_time'] = sl_time.ffill()

    sh_t = pd.to_datetime(d['recent_sh_time'])
    sl_t = pd.to_datetime(d['recent_sl_time'])
    d['leg_dir'] = np.where(sh_t > sl_t, 'BULL',
                    np.where(sl_t > sh_t, 'BEAR', None))
    d['leg_range'] = d['recent_sh'] - d['recent_sl']

    is_bull = d['leg_dir'] == 'BULL'
    is_bear = d['leg_dir'] == 'BEAR'

    # Fib OTE zone:
    #   BULL leg (pullback DOWN from SH toward SL, LONG entry):
    #       fib_lo = SH − FIB_HI · range   (deepest acceptable pullback)
    #       fib_hi = SH − FIB_LO · range
    #   BEAR leg (pullback UP from SL toward SH, SHORT entry):
    #       fib_lo = SL + FIB_LO · range
    #       fib_hi = SL + FIB_HI · range
    d['fib_lo'] = np.where(is_bull, d['recent_sh'] - FIB_HI * d['leg_range'],
                  np.where(is_bear, d['recent_sl'] + FIB_LO * d['leg_range'], np.nan))
    d['fib_hi'] = np.where(is_bull, d['recent_sh'] - FIB_LO * d['leg_range'],
                  np.where(is_bear, d['recent_sl'] + FIB_HI * d['leg_range'], np.nan))

    # "Strong" = the swing extreme defining the leg hasn't been broken.
    # BULL leg invalidates when low < recent_sl (took out the swing low).
    # BEAR leg invalidates when high > recent_sh (took out the swing high).
    d['leg_strong'] = np.where(is_bull, d['low']  > d['recent_sl'],
                       np.where(is_bear, d['high'] < d['recent_sh'], False))
    return d


m5_legs = build_legs(df_m5)
print(f'M5 rows with a valid leg : {m5_legs["leg_dir"].notna().sum():,}')
print(f'    strong-leg rows      : {(m5_legs["leg_strong"] & m5_legs["leg_dir"].notna()).sum():,}')
print(m5_legs['leg_dir'].value_counts().to_string())
m5_legs[['time','recent_sl','recent_sh','leg_dir','fib_lo','fib_hi','leg_strong']].tail(5)""")

# ─────────────────────────────────────────────────────────────────────────────
md("""## Step 5 — Enrichment Pipeline (single source of truth)

Bundle swings + fib zones + indicators (EMA / RSI / ATR / DMA) + HTF trend onto every trading-TF bar **once**. Strategy variants then just toggle which filter flags are active — no recomputation per strategy.""")

code("""def htf_trend(df_h1: pd.DataFrame, fast: int = HTF_EMA_FAST,
              slow: int = HTF_EMA_SLOW) -> pd.DataFrame:
    h = df_h1.copy()
    h['ema_fast'] = ema(h['close'], fast)
    h['ema_slow'] = ema(h['close'], slow)
    long_t  = (h['close'] > h['ema_slow']) & (h['ema_fast'] > h['ema_slow'])
    short_t = (h['close'] < h['ema_slow']) & (h['ema_fast'] < h['ema_slow'])
    h['htf_trend'] = np.where(long_t, 'LONG',
                       np.where(short_t, 'SHORT', 'FLAT'))
    return h[['time', 'htf_trend']]


def enrich(df_tf: pd.DataFrame, df_h1: pd.DataFrame, tf_label: str) -> pd.DataFrame:
    d = build_legs(df_tf, SWING_LOOKBACK)

    d['rsi']     = rsi(d['close'], RSI_PERIOD)
    d['ema_dma'] = ema(d['close'], DMA_PERIOD).shift(DMA_SHIFT)
    d['atr']     = atr(d, ATR_PERIOD)

    htf = htf_trend(df_h1)
    d = pd.merge_asof(d.sort_values('time'),
                      htf.sort_values('time'),
                      on='time', direction='backward')
    d['htf_trend'] = d['htf_trend'].fillna('FLAT')

    # Did this bar touch the OTE zone (any overlap of [low,high] with [fib_lo,fib_hi])?
    in_zone = (d['leg_dir'].notna() & d['leg_strong'] &
               (d['high'] >= d['fib_lo']) & (d['low'] <= d['fib_hi']))
    d['touched_ote'] = in_zone

    # Was the zone touched within the last TRIGGER_BARS bars?
    d['recent_touch'] = (d['touched_ote']
                         .rolling(TRIGGER_BARS, min_periods=1).max()
                         .fillna(0).astype(bool))

    d.attrs['tf'] = tf_label
    return d.reset_index(drop=True)


tf_m1 = enrich(df_m1, df_h1, 'M1')
tf_m5 = enrich(df_m5, df_h1, 'M5')

for d in (tf_m1, tf_m5):
    print(f'{d.attrs["tf"]:>3}: rows={len(d):>8,}  '
          f'leg_strong={(d["leg_strong"] & d["leg_dir"].notna()).sum():>7,}  '
          f'ote_touches={d["touched_ote"].sum():>6,}')""")

# ─────────────────────────────────────────────────────────────────────────────
md("""## Step 6 — Generic Signal Engine (filters as booleans)

A single signal builder, parameterised by which filters are active. This keeps every variant honest: they only differ by the filter set passed in.

**Always-on base trigger:**
1. Active leg exists and is still strong.
2. OTE zone was touched within the last `TRIGGER_BARS` bars.
3. Current bar closes *back in the leg's direction* (reversal trigger off the OTE).
4. ATR is available (warm-up complete).

**Optional filters:**

| flag | LONG requires | SHORT requires |
|---|---|---|
| `use_rsi` | RSI &lt; `RSI_LONG_MAX` | RSI &gt; `RSI_SHORT_MIN` |
| `use_dma` | close &gt; displaced EMA | close &lt; displaced EMA |
| `use_htf` | `htf_trend == 'LONG'` | `htf_trend == 'SHORT'` |""")

code("""def make_signals(df: pd.DataFrame,
                 use_rsi: bool = False,
                 use_dma: bool = False,
                 use_htf: bool = False) -> pd.DataFrame:
    d = df

    has_leg = d['leg_dir'].notna() & d['leg_strong'] & d['atr'].notna()
    touched = d['recent_touch'] & has_leg

    is_bull = (d['leg_dir'] == 'BULL') & touched
    is_bear = (d['leg_dir'] == 'BEAR') & touched

    # Reversal trigger off the OTE (close moves back in leg direction)
    prev_high = d['high'].shift(1)
    prev_low  = d['low'].shift(1)
    trigger_long  = is_bull & (d['close'] > prev_high)
    trigger_short = is_bear & (d['close'] < prev_low)

    if use_rsi:
        trigger_long  &= (d['rsi'] < RSI_LONG_MAX)
        trigger_short &= (d['rsi'] > RSI_SHORT_MIN)
    if use_dma:
        trigger_long  &= (d['close'] > d['ema_dma'])
        trigger_short &= (d['close'] < d['ema_dma'])
    if use_htf:
        trigger_long  &= (d['htf_trend'] == 'LONG')
        trigger_short &= (d['htf_trend'] == 'SHORT')

    sig = pd.Series(np.where(trigger_long, 'LONG',
                     np.where(trigger_short, 'SHORT', None)),
                    index=d.index, dtype=object)

    cols = ['time','open','high','low','close','volume','spread',
            'recent_sh','recent_sl','leg_dir','leg_range','fib_lo','fib_hi',
            'rsi','ema_dma','atr','htf_trend']
    out = d.loc[sig.notna(), cols].copy()
    out['signal'] = sig[sig.notna()].values
    return out.reset_index(drop=True)


print('Signal engine ready.')""")

# ─────────────────────────────────────────────────────────────────────────────
md("""## Step 7 — Trade Simulator (next-bar fill, swing-anchored SL, fixed RR)

- **Entry** at the *open* of the next bar (no look-ahead).
- **SL** = swing extreme (recent_sl for LONG / recent_sh for SHORT) ± `SL_ATR_BUFFER · ATR` buffer.
- **TP** = entry ± `RR_TARGET · risk` so every variant shares the same RR math.
- **Cooldown** = `COOLDOWN_BARS` trading-TF bars after each exit.
- If SL and TP both trigger on the same bar → SL hits first (pessimistic).
- If `max_hold_bars` elapses → mark-to-market on the last bar's close (`TIME` outcome).

The simulator is vectorised over the management window (numpy `argmax` on the hit masks) so it runs in seconds even on thousands of signals.""")

code("""def simulate_signal_arr(arrs: Dict, sig_idx: int, sig: pd.Series,
                        max_hold: int) -> Optional[Dict]:
    n = arrs['n']
    if sig_idx + 1 >= n:
        return None

    o, h, l, c, t, sp = arrs['o'], arrs['h'], arrs['l'], arrs['c'], arrs['t'], arrs['sp']

    entry   = float(o[sig_idx + 1])
    atr_val = float(sig['atr'])
    if not np.isfinite(atr_val) or atr_val <= 0:
        return None

    if sig['signal'] == 'LONG':
        sl = float(sig['recent_sl']) - SL_ATR_BUFFER * atr_val
        if sl >= entry:
            return None
        risk = entry - sl
        tp   = entry + RR_TARGET * risk
    else:
        sl = float(sig['recent_sh']) + SL_ATR_BUFFER * atr_val
        if sl <= entry:
            return None
        risk = sl - entry
        tp   = entry - RR_TARGET * risk

    fill_idx = sig_idx + 1
    end_idx  = min(fill_idx + max_hold, n - 1)
    hi = h[fill_idx:end_idx + 1]
    lo = l[fill_idx:end_idx + 1]

    if sig['signal'] == 'LONG':
        hit_sl = lo <= sl
        hit_tp = hi >= tp
    else:
        hit_sl = hi >= sl
        hit_tp = lo <= tp

    sl_first = int(np.argmax(hit_sl)) if hit_sl.any() else len(hit_sl)
    tp_first = int(np.argmax(hit_tp)) if hit_tp.any() else len(hit_tp)

    base = {'entry_idx': fill_idx, 'entry_time': pd.Timestamp(t[fill_idx]),
            'entry_price': entry, 'sl': sl, 'tp': tp, 'risk': risk, 'atr': atr_val,
            'spread_in': float(sp[fill_idx])}

    if sl_first < len(hit_sl) and sl_first <= tp_first:
        j = sl_first
        return {**base, 'exit_idx': fill_idx + j,
                'exit_time': pd.Timestamp(t[fill_idx + j]),
                'exit_price': sl, 'result': 'SL', 'pnl_r': -1.0,
                'bars_held': j + 1, 'spread_out': float(sp[fill_idx + j])}
    if tp_first < len(hit_tp):
        j = tp_first
        return {**base, 'exit_idx': fill_idx + j,
                'exit_time': pd.Timestamp(t[fill_idx + j]),
                'exit_price': tp, 'result': 'TP', 'pnl_r': float(RR_TARGET),
                'bars_held': j + 1, 'spread_out': float(sp[fill_idx + j])}

    last_idx = end_idx
    last_close = float(c[last_idx])
    if sig['signal'] == 'LONG':
        pnl_r = (last_close - entry) / risk
    else:
        pnl_r = (entry - last_close) / risk
    return {**base, 'exit_idx': last_idx,
            'exit_time': pd.Timestamp(t[last_idx]),
            'exit_price': last_close, 'result': 'TIME',
            'pnl_r': round(float(pnl_r), 3),
            'bars_held': last_idx - sig_idx,
            'spread_out': float(sp[last_idx])}


def run_backtest(tf_df: pd.DataFrame, signals: pd.DataFrame,
                 max_hold: int, cooldown: int = COOLDOWN_BARS) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame()
    time_to_idx = pd.Series(np.arange(len(tf_df)), index=tf_df['time'])
    arrs = {
        'o' : tf_df['open'].values,
        'h' : tf_df['high'].values,
        'l' : tf_df['low'].values,
        'c' : tf_df['close'].values,
        't' : tf_df['time'].values,
        'sp': tf_df['spread'].fillna(0).values,
        'n' : len(tf_df),
    }
    trades: List[Dict] = []
    cooldown_until = -1
    for _, sig in signals.iterrows():
        idx = int(time_to_idx.get(sig['time'], -1))
        if idx < 0 or idx <= cooldown_until:
            continue
        out = simulate_signal_arr(arrs, idx, sig, max_hold)
        if out is None:
            continue
        trades.append({
            'signal_time': sig['time'],
            'direction'  : sig['signal'],
            'leg_dir'    : sig['leg_dir'],
            'fib_lo'     : round(float(sig['fib_lo']), 5),
            'fib_hi'     : round(float(sig['fib_hi']), 5),
            'rsi'        : round(float(sig['rsi']), 2) if pd.notna(sig['rsi']) else np.nan,
            'htf_trend'  : sig['htf_trend'],
            **out,
        })
        cooldown_until = out['exit_idx'] + cooldown
    return pd.DataFrame(trades)


print('Simulator ready.')""")

# ─────────────────────────────────────────────────────────────────────────────
md("## Step 8 — Metrics · Weekly PnL · Persistence")

code("""def calc_metrics(trades_df: pd.DataFrame, tf_label: str) -> Dict:
    if trades_df.empty:
        return {'tf': tf_label, 'total_trades': 0}
    c = trades_df.copy().reset_index(drop=True)
    n  = len(c)
    w  = int((c['result'] == 'TP').sum())
    l  = int((c['result'] == 'SL').sum())
    t  = int((c['result'] == 'TIME').sum())
    wr = w / n
    c['cum_r'] = c['pnl_r'].cumsum()
    dd  = c['cum_r'] - c['cum_r'].cummax()
    pos = c.loc[c['pnl_r'] > 0, 'pnl_r'].sum()
    neg = abs(c.loc[c['pnl_r'] < 0, 'pnl_r'].sum())
    pf  = (pos / neg) if neg > 0 else float('inf')

    arr = (c['result'] == 'SL').astype(int).values
    max_cl = streak = 0
    for v in arr:
        streak = streak + 1 if v else 0
        max_cl = max(max_cl, streak)

    return {
        'tf'              : tf_label,
        'total_trades'    : n,
        'wins'            : w,
        'losses'          : l,
        'timeouts'        : t,
        'win_rate'        : wr,
        'total_r'         : round(float(c['pnl_r'].sum()), 3),
        'avg_r'           : round(float(c['pnl_r'].mean()), 3),
        'profit_factor'   : round(float(pf), 3),
        'max_dd_r'        : round(float(dd.min()), 3),
        'max_consec_loss' : int(max_cl),
        'expectancy'      : round(wr * RR_TARGET - (1 - wr), 3),
        'avg_bars'        : round(float(c['bars_held'].mean()), 1),
        'cum_r'           : c['cum_r'].reset_index(drop=True),
        'drawdown'        : dd.reset_index(drop=True),
        'closed'          : c,
    }


def weekly_report(trades_df: pd.DataFrame, pnl_col: str = 'pnl_r') -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame()
    c = trades_df.copy()
    c['entry_time'] = pd.to_datetime(c['entry_time'])
    # ISO week starting Monday
    c['week'] = c['entry_time'].dt.to_period('W-SUN').dt.start_time.dt.date
    g = c.groupby('week')
    out = pd.DataFrame({
        'trades'       : g.size(),
        'wins'         : g.apply(lambda x: int((x['result'] == 'TP').sum())),
        'losses'       : g.apply(lambda x: int((x['result'] == 'SL').sum())),
        'timeouts'     : g.apply(lambda x: int((x['result'] == 'TIME').sum())),
        'total_r'      : g[pnl_col].sum().round(3),
        'win_rate_pct' : (g.apply(lambda x: (x[pnl_col] > 0).mean()) * 100).round(1),
    }).reset_index()
    return out


def save_strategy_results(strategy_name: str, tf_label: str,
                          trades_df: pd.DataFrame, metrics: Dict,
                          weekly: pd.DataFrame) -> Path:
    out_dir = RESULTS_ROOT / strategy_name / tf_label / SYMBOL
    out_dir.mkdir(parents=True, exist_ok=True)
    if not trades_df.empty:
        trades_df.to_csv(out_dir / 'trades.csv', index=False)
    if not weekly.empty:
        weekly.to_csv(out_dir / 'weekly.csv', index=False)
    if metrics.get('total_trades', 0):
        summary = pd.DataFrame([{
            'symbol'        : SYMBOL,
            'strategy'      : strategy_name,
            'tf'            : tf_label,
            'date_from'     : DATE_FROM,
            'date_to'       : DATE_TO,
            'total_trades'  : metrics['total_trades'],
            'wins'          : metrics['wins'],
            'losses'        : metrics['losses'],
            'timeouts'      : metrics['timeouts'],
            'win_rate_pct'  : round(metrics['win_rate'] * 100, 2),
            'total_r'       : metrics['total_r'],
            'avg_r'         : metrics['avg_r'],
            'profit_factor' : metrics['profit_factor'],
            'max_dd_r'      : metrics['max_dd_r'],
            'expectancy'    : metrics['expectancy'],
            'max_consec_loss': metrics['max_consec_loss'],
            'rr_target'     : RR_TARGET,
            'swing_lookback': SWING_LOOKBACK,
            'fib_lo'        : FIB_LO,
            'fib_hi'        : FIB_HI,
        }])
        summary.to_csv(out_dir / 'summary.csv', index=False)
        metrics['closed'][['entry_time','direction','result','pnl_r','cum_r']].to_csv(
            out_dir / 'equity.csv', index=False)
    return out_dir


print('Metrics + persistence ready.')""")

# ─────────────────────────────────────────────────────────────────────────────
md("""## Step 9 — Universal Strategy Runner

A small wrapper so every variant runs the same way: **enrich → signals → backtest → metrics → weekly → save**.""")

code("""MAX_HOLD = {'M1': MAX_HOLD_BARS_M1, 'M5': MAX_HOLD_BARS_M5}
TF_DATA  = {'M1': tf_m1, 'M5': tf_m5}

def run_strategy(name: str, *, use_rsi=False, use_dma=False, use_htf=False
                 ) -> Dict[str, Dict]:
    results = {}
    title = f' {name}  (rsi={use_rsi} · dma={use_dma} · htf={use_htf}) '
    print('\\n' + '═' * 78)
    print(title.center(78, '═'))
    print('═' * 78)
    for tf_label, tf_df in TF_DATA.items():
        signals  = make_signals(tf_df, use_rsi=use_rsi, use_dma=use_dma, use_htf=use_htf)
        trades   = run_backtest(tf_df, signals, MAX_HOLD[tf_label])
        metrics  = calc_metrics(trades, tf_label)
        weekly   = weekly_report(trades)
        out_dir  = save_strategy_results(name, tf_label, trades, metrics, weekly)

        n  = metrics.get('total_trades', 0)
        wr = metrics.get('win_rate', 0) * 100 if n else 0
        tr = metrics.get('total_r', 0)
        pf = metrics.get('profit_factor', 0)
        dd = metrics.get('max_dd_r', 0)
        print(f'  [{tf_label}] sig={len(signals):>5}  trades={n:>4}  '
              f'WR={wr:5.1f}%  PF={pf:5.2f}  Total={tr:+7.2f}R  '
              f'MaxDD={dd:+6.2f}R   → {out_dir}')

        results[tf_label] = {'metrics': metrics, 'trades': trades,
                             'weekly': weekly, 'out_dir': out_dir}
    return results


print('Strategy runner ready.')""")

# ─── STRATEGY 1 ──────────────────────────────────────────────────────────────
md("## Strategy 1 — **Base** : Strong High/Low + Fibonacci OTE")

code("""r_base = run_strategy('01_base_fib_ote')

print('\\n— Weekly PnL (M5) — last 12 weeks —')
w = r_base['M5']['weekly']
if not w.empty:
    display(w.tail(12))
else:
    print('  (no trades)')""")

# ─── STRATEGY 2 ──────────────────────────────────────────────────────────────
md("## Strategy 2 — Base + **RSI** confirmation")

code("""r_rsi = run_strategy('02_rsi', use_rsi=True)

print('\\n— Weekly PnL (M5) — last 12 weeks —')
w = r_rsi['M5']['weekly']
if not w.empty:
    display(w.tail(12))
else:
    print('  (no trades)')""")

# ─── STRATEGY 3 ──────────────────────────────────────────────────────────────
md("""## Strategy 3 — Base + **DMA** (Displaced Moving Average)

A *displaced moving average* uses an EMA value from `DMA_SHIFT` bars ago — comparing current price against a lagged trend baseline. Filters trades that are clearly counter to the local trend memory.""")

code("""r_dma = run_strategy('03_dma', use_dma=True)

print('\\n— Weekly PnL (M5) — last 12 weeks —')
w = r_dma['M5']['weekly']
if not w.empty:
    display(w.tail(12))
else:
    print('  (no trades)')""")

# ─── STRATEGY 4 ──────────────────────────────────────────────────────────────
md("""## Strategy 4 — Base + **H1 Higher-Timeframe Trend** filter

Only takes the OTE setup if the H1 EMA20/50 trend agrees with the trade direction. The classical "trade-with-trend" filter.""")

code("""r_htf = run_strategy('04_htf_trend', use_htf=True)

print('\\n— Weekly PnL (M5) — last 12 weeks —')
w = r_htf['M5']['weekly']
if not w.empty:
    display(w.tail(12))
else:
    print('  (no trades)')""")

# ─── COMPARISON ──────────────────────────────────────────────────────────────
md("## Cross-Strategy Comparison — which filter set wins?")

code("""def cmp_row(name, r, tf):
    m = r[tf]['metrics']
    if m.get('total_trades', 0) == 0:
        return {'strategy': name, 'tf': tf, 'trades': 0}
    return {
        'strategy'    : name,
        'tf'          : tf,
        'trades'      : m['total_trades'],
        'WR_pct'      : round(m['win_rate'] * 100, 1),
        'total_R'     : m['total_r'],
        'avg_R'       : m['avg_r'],
        'PF'          : m['profit_factor'],
        'maxDD_R'     : m['max_dd_r'],
        'expectancy'  : m['expectancy'],
    }


comparison = pd.DataFrame([
    cmp_row('01_base_fib_ote', r_base, 'M1'), cmp_row('01_base_fib_ote', r_base, 'M5'),
    cmp_row('02_rsi',          r_rsi,  'M1'), cmp_row('02_rsi',          r_rsi,  'M5'),
    cmp_row('03_dma',          r_dma,  'M1'), cmp_row('03_dma',          r_dma,  'M5'),
    cmp_row('04_htf_trend',    r_htf,  'M1'), cmp_row('04_htf_trend',    r_htf,  'M5'),
])
print('All strategies ranked by total_R:')
display(comparison.sort_values('total_R', ascending=False, na_position='last'))

# Identify the filter combo that beat the base
base_total = {tf: r_base[tf]['metrics'].get('total_r', 0) for tf in TIMEFRAMES}

print(f'\\nBaseline (no filters): M1={base_total["M1"]:+.2f}R · M5={base_total["M5"]:+.2f}R')
improvements = []
for name, r in [('rsi', r_rsi), ('dma', r_dma), ('htf', r_htf)]:
    for tf in TIMEFRAMES:
        tr = r[tf]['metrics'].get('total_r', 0)
        if tr > base_total[tf]:
            improvements.append((name, tf, tr - base_total[tf]))
print('\\nFilters that *improved* over base:')
for n, tf, delta in sorted(improvements, key=lambda x: -x[2]):
    print(f'  + {n:>4s} on {tf} → +{delta:.2f}R over base')""")

# ─── STRATEGY 5 / SPREAD ─────────────────────────────────────────────────────
md("""## Strategy 5 — **Best Combo (auto-selected)**  +  Spread-aware Net Profit

We sweep **all 2³ = 8 filter combinations** (rsi × dma × htf), require a minimum trade count for statistical relevance, and pick the combo that **maximises total R-net of spread** per timeframe.

Spread is modeled in R-units so it stacks against the trade's R-multiple:

```
spread_cost_price = (spread_in + spread_out) × point_size
spread_cost_R     = spread_cost_price / trade_risk
pnl_r_net         = pnl_r − spread_cost_R
```

Each TF gets its own winner (M1 and M5 may legitimately prefer different filter sets).""")

code("""from itertools import product

MIN_TRADES_FOR_PICK = 30      # require at least this many trades for the auto-pick


def add_spread_columns(trades_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if trades_df.empty:
        return trades_df
    ps = get_point_size(symbol)
    c = trades_df.copy()
    c['spread_cost_price'] = (c['spread_in'].fillna(0) + c['spread_out'].fillna(0)) * ps
    c['spread_cost_r']     = (c['spread_cost_price'] / c['risk']).round(4)
    c['pnl_r_net']         = (c['pnl_r'] - c['spread_cost_r']).round(4)
    return c


def net_summary(c: pd.DataFrame, tf_label: str, combo: Tuple[bool, bool, bool]) -> Dict:
    if c.empty:
        return {'tf': tf_label, 'use_rsi': combo[0], 'use_dma': combo[1], 'use_htf': combo[2],
                'total_trades': 0}
    n   = len(c)
    w   = int((c['pnl_r_net'] > 0).sum())
    wr  = w / n
    cum = c['pnl_r_net'].cumsum()
    dd  = cum - cum.cummax()
    pos = c.loc[c['pnl_r_net'] > 0, 'pnl_r_net'].sum()
    neg = abs(c.loc[c['pnl_r_net'] < 0, 'pnl_r_net'].sum())
    pf  = (pos / neg) if neg > 0 else float('inf')
    return {
        'tf'              : tf_label,
        'use_rsi'         : combo[0],
        'use_dma'         : combo[1],
        'use_htf'         : combo[2],
        'total_trades'    : n,
        'wins'            : w,
        'losses'          : n - w,
        'win_rate_pct'    : round(wr * 100, 2),
        'gross_total_r'   : round(float(c['pnl_r'].sum()), 3),
        'spread_total_r'  : round(float(c['spread_cost_r'].sum()), 3),
        'net_total_r'     : round(float(c['pnl_r_net'].sum()), 3),
        'avg_r_net'       : round(float(c['pnl_r_net'].mean()), 3),
        'profit_factor_net': round(float(pf), 3),
        'max_dd_r_net'    : round(float(dd.min()), 3),
        'avg_spread_pts'  : round(float((c['spread_in'].fillna(0) + c['spread_out'].fillna(0)).mean()), 1),
        'avg_spread_r'    : round(float(c['spread_cost_r'].mean()), 4),
    }


# ── Sweep every filter combination on both timeframes ────────────────────────
sweep_rows: List[Dict] = []
combo_trades: Dict[Tuple[str, Tuple[bool, bool, bool]], pd.DataFrame] = {}

print('Sweeping 8 filter combinations × 2 timeframes…')
for tf_label, tf_df in TF_DATA.items():
    for combo in product([False, True], repeat=3):
        use_rsi, use_dma, use_htf = combo
        signals = make_signals(tf_df, use_rsi=use_rsi, use_dma=use_dma, use_htf=use_htf)
        trades  = run_backtest(tf_df, signals, MAX_HOLD[tf_label])
        if trades.empty:
            sweep_rows.append({'tf': tf_label, 'use_rsi': use_rsi, 'use_dma': use_dma,
                               'use_htf': use_htf, 'total_trades': 0})
            continue
        trades_net = add_spread_columns(trades, SYMBOL)
        sweep_rows.append(net_summary(trades_net, tf_label, combo))
        combo_trades[(tf_label, combo)] = trades_net

sweep = pd.DataFrame(sweep_rows).sort_values(['tf', 'net_total_r'],
                                             ascending=[True, False], na_position='last')
print('\\nAll combinations (sorted within each TF by NET total R):')
display(sweep)

# ── Pick the winner per TF — best NET total R subject to MIN_TRADES_FOR_PICK ──
print(f'\\nPicking winner per TF (min {MIN_TRADES_FOR_PICK} trades for statistical relevance)…')
winners: Dict[str, Dict] = {}
for tf_label in TIMEFRAMES:
    sub = sweep[(sweep['tf'] == tf_label) &
                (sweep['total_trades'] >= MIN_TRADES_FOR_PICK)].copy()
    if sub.empty:
        sub = sweep[sweep['tf'] == tf_label].copy()
        print(f'  [{tf_label}] no combo cleared {MIN_TRADES_FOR_PICK} trades — falling back to most-traded.')
        sub = sub.sort_values('total_trades', ascending=False)
    else:
        sub = sub.sort_values('net_total_r', ascending=False)
    win = sub.iloc[0].to_dict()
    winners[tf_label] = win
    combo = (bool(win['use_rsi']), bool(win['use_dma']), bool(win['use_htf']))
    print(f'  [{tf_label}] best combo = rsi={combo[0]} · dma={combo[1]} · htf={combo[2]}  '
          f'→ trades={int(win["total_trades"])}  '
          f'gross={win["gross_total_r"]:+.2f}R  '
          f'net={win["net_total_r"]:+.2f}R  '
          f'WR_net={win["win_rate_pct"]:.1f}%  '
          f'PF_net={win["profit_factor_net"]:.2f}')

# ── Persist the auto-selected combo as strategy '05_best_combo' ──────────────
print('\\nSaving auto-selected combo as 05_best_combo …')
r_best: Dict[str, Dict] = {}
for tf_label in TIMEFRAMES:
    combo = (bool(winners[tf_label]['use_rsi']),
             bool(winners[tf_label]['use_dma']),
             bool(winners[tf_label]['use_htf']))
    trades_net = combo_trades.get((tf_label, combo))
    if trades_net is None or trades_net.empty:
        r_best[tf_label] = {'metrics': {'total_trades': 0, 'tf': tf_label}, 'trades': pd.DataFrame()}
        continue

    # Rebuild gross metrics from the (gross) trades subset
    gross_trades = trades_net[['signal_time','direction','leg_dir','entry_time','exit_time',
                               'entry_price','exit_price','sl','tp','risk','atr',
                               'result','pnl_r','bars_held','spread_in','spread_out']].copy()
    metrics = calc_metrics(gross_trades, tf_label)
    weekly  = weekly_report(gross_trades)

    out_dir = save_strategy_results('05_best_combo', tf_label, gross_trades, metrics, weekly)

    # And the net-of-spread artifacts
    trades_net.to_csv(out_dir / 'trades_with_spread.csv', index=False)
    pd.DataFrame([net_summary(trades_net, tf_label, combo)]).to_csv(out_dir / 'summary_net.csv', index=False)

    # Weekly NET
    trades_net['entry_time'] = pd.to_datetime(trades_net['entry_time'])
    trades_net['week'] = trades_net['entry_time'].dt.to_period('W-SUN').dt.start_time.dt.date
    weekly_net = (trades_net.groupby('week')
                  .agg(trades=('pnl_r_net', 'size'),
                       gross_R=('pnl_r', 'sum'),
                       spread_R=('spread_cost_r', 'sum'),
                       net_R=('pnl_r_net', 'sum'),
                       WR_pct=('pnl_r_net', lambda x: round((x > 0).mean() * 100, 1)))
                  .round(3).reset_index())
    weekly_net.to_csv(out_dir / 'weekly_net.csv', index=False)

    print(f'  [{tf_label}] combo=rsi={combo[0]}·dma={combo[1]}·htf={combo[2]}  saved → {out_dir}')

    r_best[tf_label] = {
        'metrics'   : metrics,
        'trades'    : gross_trades,
        'trades_net': trades_net,
        'weekly'    : weekly,
        'weekly_net': weekly_net,
        'combo'     : combo,
        'out_dir'   : out_dir,
    }

# ── Display weekly NET PnL for M5 winner ─────────────────────────────────────
print('\\n— Weekly NET PnL (M5 winner) — last 12 weeks —')
if 'weekly_net' in r_best.get('M5', {}):
    display(r_best['M5']['weekly_net'].tail(12))""")

# ─── EQUITY PLOT ─────────────────────────────────────────────────────────────
md("## Equity Curves — Gross vs Net (Best Combo) on M5")

code("""def plot_equity_compare(r_all: Dict[str, Dict], tf: str = 'M5'):
    fig = make_subplots(rows=2, cols=1, row_heights=[0.6, 0.4],
                        subplot_titles=[f'Cumulative Equity (R) — {SYMBOL}  ·  {tf}',
                                        'Drawdown (R)'],
                        vertical_spacing=0.10)
    palette = {
        '01_base_fib_ote': '#00E5FF',
        '02_rsi'         : '#FFB300',
        '03_dma'         : '#AB47BC',
        '04_htf_trend'   : '#26A69A',
        '05_best_combo'  : '#FF1744',
    }
    for name, r in r_all.items():
        m = r[tf]['metrics']
        if m.get('total_trades', 0) == 0:
            continue
        color = palette.get(name, '#888')
        fig.add_trace(go.Scatter(x=m['cum_r'].index, y=m['cum_r'].values,
                                 mode='lines', line=dict(color=color, width=2),
                                 name=f'{name}  ({m["total_trades"]} · {m["total_r"]:+.1f}R)'),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=m['drawdown'].index, y=m['drawdown'].values,
                                 mode='lines', line=dict(color=color, width=1.2),
                                 fill='tozeroy', showlegend=False,
                                 name=f'{name} DD'),
                      row=2, col=1)

    # Net equity for best combo, plotted dashed
    t = r_all['05_best_combo'][tf]['trades']
    if not t.empty:
        tn = add_spread_columns(t, SYMBOL)
        cum_net = tn['pnl_r_net'].cumsum()
        fig.add_trace(go.Scatter(x=cum_net.index, y=cum_net.values,
                                 mode='lines', line=dict(color='#FF1744', width=2, dash='dot'),
                                 name=f'05_best_combo NET ({cum_net.iloc[-1]:+.1f}R)'),
                      row=1, col=1)

    fig.add_hline(y=0, line_color='gray', line_dash='dash', row=1, col=1)
    fig.update_layout(title=f'Fib OTE Scalper — {SYMBOL}  ·  {tf}  ·  All strategies',
                      height=720, template='plotly_dark', showlegend=True)
    fig.show()


R_ALL = {'01_base_fib_ote': r_base, '02_rsi': r_rsi, '03_dma': r_dma,
         '04_htf_trend': r_htf, '05_best_combo': r_best}

plot_equity_compare(R_ALL, 'M5')
plot_equity_compare(R_ALL, 'M1')""")

# ─── FINAL SUMMARY ───────────────────────────────────────────────────────────
md("## Final Summary & Verdict")

code("""sep = '═' * 78
print(sep)
print(f'  FIB OTE SCALPER — {SYMBOL}   ·   {DATE_FROM} → {DATE_TO or "latest"}'.ljust(76) + '  ')
print(f'  RR={RR_TARGET}:1  ·  SWING_LB={SWING_LOOKBACK}  ·  OTE=[{FIB_LO},{FIB_HI}]  ·  cooldown={COOLDOWN_BARS}'.ljust(76))
print(sep)

rows = []
for name, r in R_ALL.items():
    for tf in TIMEFRAMES:
        m = r[tf]['metrics']
        if m.get('total_trades', 0) == 0:
            continue
        rows.append({
            'strategy'   : name, 'tf': tf, 'trades': m['total_trades'],
            'WR_pct'     : round(m['win_rate']*100, 1),
            'total_R'    : m['total_r'], 'PF': m['profit_factor'],
            'maxDD'      : m['max_dd_r'], 'expectancy': m['expectancy'],
        })

final_df = pd.DataFrame(rows).sort_values('total_R', ascending=False).reset_index(drop=True)
print('\\n[GROSS — RR-units, no spread]')
display(final_df)

print('\\n[NET — best combo only, after spread cost]')
net_lines = []
for tf in TIMEFRAMES:
    t = r_best[tf]['trades']
    if t.empty:
        continue
    tn = add_spread_columns(t, SYMBOL)
    net = float(tn['pnl_r_net'].sum())
    gross = float(tn['pnl_r'].sum())
    wr_net = (tn['pnl_r_net'] > 0).mean() * 100
    pos = tn.loc[tn['pnl_r_net'] > 0, 'pnl_r_net'].sum()
    neg = abs(tn.loc[tn['pnl_r_net'] < 0, 'pnl_r_net'].sum())
    pf_net = (pos / neg) if neg > 0 else float('inf')
    net_lines.append({
        'tf': tf, 'trades': len(tn),
        'gross_R': round(gross, 2),
        'spread_R': round(gross - net, 2),
        'net_R': round(net, 2),
        'WR_net_pct': round(wr_net, 1),
        'PF_net': round(pf_net, 2),
    })
if net_lines:
    display(pd.DataFrame(net_lines).set_index('tf'))

print('\\n[VERDICT]')
if final_df.empty:
    print('  No viable trades. Widen DATE_FROM or lower SWING_LOOKBACK.')
else:
    top = final_df.iloc[0]
    print(f'  ★ Best gross setup : {top["strategy"]} / {top["tf"]}  →  '
          f'{top["total_R"]:+.2f}R   (WR={top["WR_pct"]:.1f}%, PF={top["PF"]:.2f})')
    if net_lines:
        best_net = max(net_lines, key=lambda x: x['net_R'])
        print(f'  ★ Best  NET  setup : 05_best_combo / {best_net["tf"]}  →  '
              f'{best_net["net_R"]:+.2f}R   (WR_net={best_net["WR_net_pct"]:.1f}%, '
              f'PF_net={best_net["PF_net"]:.2f})')
        print(f'    spread cost reduced gross by {best_net["spread_R"]:.2f}R '
              f'on {best_net["trades"]} trades.')
print(sep)
print(f'  Results saved under: {RESULTS_ROOT.resolve()}')
print(f'  Layout: ./results/{{strategy}}/{{timeframe}}/{{symbol}}/(trades|weekly|summary|equity).csv')
print(sep)""")

# ─── WRITE NOTEBOOK ──────────────────────────────────────────────────────────
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = Path(__file__).parent / '18_fib_scalper_strong_hl.ipynb'
out_path.write_text(json.dumps(nb, indent=1), encoding='utf-8')
print(f'Wrote {out_path}  ({len(cells)} cells)')
