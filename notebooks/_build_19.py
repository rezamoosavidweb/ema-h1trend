"""Generate notebook 19: RSI Divergence / Convergence Scalper (pivot-to-pivot)."""
import json
from pathlib import Path

CELLS: list[tuple[str, str]] = []  # (cell_type, source)


def md(s: str) -> None:
    CELLS.append(("markdown", s))


def code(s: str) -> None:
    CELLS.append(("code", s))


# ─────────────────────────────────────────────────────────────────────────────
md(r"""# Notebook #19 — RSI Divergence / Convergence Scalper

### Automated pivot-to-pivot divergence engine between price and RSI(14)

---

### Concept

Price and momentum *should* move together. When they stop confirming each other, it often signals reversals (regular divergence) or trend continuation (hidden divergence). This notebook is a fully mechanical detector:

1. **Detect significant swing highs and lows** on the price chart using a confirmed-pivot rule (left/right strength = `PIVOT_STRENGTH` bars).
2. **Read the RSI value at the exact same bar** as each price pivot.
3. **Connect consecutive pivot pairs** on both price and RSI with straight lines.
4. **Compare slopes** of the price-line vs the RSI-line and classify:

| Pattern                  | Price (pivot pair) | RSI (pivot pair) | Bias              |
|--------------------------|--------------------|------------------|-------------------|
| **Regular Bearish Div**  | Higher High        | Lower  High      | SHORT (reversal)  |
| **Regular Bullish Div**  | Lower  Low         | Higher Low       | LONG  (reversal)  |
| **Hidden  Bearish Div**  | Lower  High        | Higher High      | SHORT (continuation) |
| **Hidden  Bullish Div**  | Higher Low         | Lower  Low       | LONG  (continuation) |
| **Convergence**          | Same direction     | Same direction   | Trend OK (logged, no trade) |

Entries are triggered on the **bar *after* the second pivot's confirmation bar** (no look-ahead — the pivot is only "known" once `PIVOT_STRENGTH` bars have passed). SL sits beyond the structural pivot; TP is a fixed `RR_TARGET` multiple.

### Why M5

The strategy is described as a scalper system. M5 keeps the pivot count manageable for visual debugging while remaining a lower-timeframe horizon. Switch `TIMEFRAME = 'M1'` for the faster variant.
""")

md(r"""## Step 1 — Imports & Configuration""")

code(r"""import warnings
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

# ── Strategy identity ───────────────────────────────────────────
STRATEGY_NAME   = 'rsi_divergence_scalper'
SYMBOL          = 'XAUUSD'
TIMEFRAME       = 'M5'                 # 'M1' or 'M5'
DATA_DIR        = Path('./data')
RESULTS_DIR     = Path('./results') / STRATEGY_NAME / SYMBOL / TIMEFRAME
LOOKBACK_DAYS   = 30

# ── Broker→NY clock (broker = NY + 7h, per project memory) ────────────────────
BROKER_TO_NY_OFFSET_H = 7

# ── RSI ───────────────────────────────────────────────────────────
RSI_PERIOD          = 14
RSI_OVERBOUGHT      = 70.0
RSI_OVERSOLD        = 30.0

# ── Pivot detection (noise filter) ──────────────────────────────────────
PIVOT_STRENGTH      = 5        # bars each side must be lower (for high) / higher (for low)
PIVOT_MIN_GAP_BARS  = 5        # min spacing between two consecutive same-side pivots
PIVOT_LOOKBACK_BARS = 60       # max bars between paired pivots (avoid stale lines)

# ── Divergence filters ─────────────────────────────────────────────────
REQUIRE_OB_OS       = True     # regular bearish needs RSI[1st pivot] >= OB; bullish needs RSI[1st] <= OS
MIN_RSI_DELTA       = 1.0      # minimum RSI difference (points) between the two pivots to count
MIN_PRICE_DELTA     = 0.0      # >0 to require a min price separation (in $); 0 disables

# ── Trade management ──────────────────────────────────────────────────
RR_TARGET           = 2.0      # take-profit at +RR_TARGET * R
SL_BUFFER_FRAC      = 0.10     # SL extends this fraction of (entry-pivot distance) beyond the pivot
MAX_TRADE_BARS      = 96       # safety cap on holding time in bars (M5 × 96 = 8h)
ALLOW_HIDDEN        = True     # trade hidden divergences as continuation
ALLOW_REGULAR       = True     # trade regular divergences as reversal

print('RSI Divergence Scalper — Configuration')
print(f'  Symbol             : {SYMBOL}')
print(f'  Timeframe          : {TIMEFRAME}')
print(f'  Lookback           : {LOOKBACK_DAYS} days')
print(f'  RSI                : period={RSI_PERIOD}  OB={RSI_OVERBOUGHT}  OS={RSI_OVERSOLD}')
print(f'  Pivot strength     : ±{PIVOT_STRENGTH} bars  ({2*PIVOT_STRENGTH+1}-bar window)')
print(f'  Pivot pairing      : gap >= {PIVOT_MIN_GAP_BARS}, distance <= {PIVOT_LOOKBACK_BARS} bars')
print(f'  Require OB/OS      : {REQUIRE_OB_OS}  (regular divergences only)')
print(f'  RR target          : {RR_TARGET}:1')
print(f'  SL buffer          : {SL_BUFFER_FRAC*100:.0f}% beyond pivot')
print(f'  Max trade duration : {MAX_TRADE_BARS} {TIMEFRAME} bars')
print(f'  Trade hidden div   : {ALLOW_HIDDEN}')
print(f'  Trade regular div  : {ALLOW_REGULAR}')
print(f'  Results dir        : {RESULTS_DIR}')
""")

md(r"""## Step 2 — Load Data

Timestamps are kept as **naive broker wall-clock** (per `data_timezone` memory: the `+00:00` suffix is misleading — broker clock 15:00 == 08:00 NY).""")

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


df = load_ohlcv(SYMBOL, TIMEFRAME)
print(f'{TIMEFRAME} bars : {len(df):>8,}  [{df["time"].min()} → {df["time"].max()}]')
df.tail(3)
""")

md(r"""## Step 3 — Compute RSI(14)

Standard Wilder RSI on the close. We attach it directly to the dataframe so pivot indices line up trivially.""")

code(r"""def compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


df['rsi'] = compute_rsi(df['close'])

print(f'RSI stats   : min={df["rsi"].min():.2f}  max={df["rsi"].max():.2f}  mean={df["rsi"].mean():.2f}')
print(f'OB touches  : {int((df["rsi"] >= RSI_OVERBOUGHT).sum())}')
print(f'OS touches  : {int((df["rsi"] <= RSI_OVERSOLD).sum())}')
df[['time','close','rsi']].tail(5)
""")

md(r"""## Step 4 — Detect Significant Pivots

A pivot is confirmed only when **all `PIVOT_STRENGTH` bars on each side are strictly lower (for a high) or higher (for a low)**. The confirmation time is `pivot_bar_index + PIVOT_STRENGTH` — i.e. the earliest moment we could possibly act on the pivot. Backtest entries respect this delay (no look-ahead).

We also enforce a minimum gap between consecutive same-side pivots to avoid micro-pivots that produce visually-cluttered divergence lines.""")

code(r"""def detect_pivots(df: pd.DataFrame, strength: int = PIVOT_STRENGTH,
                  min_gap: int = PIVOT_MIN_GAP_BARS) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # returns (highs_df, lows_df) with columns: idx, time, price, rsi, confirm_idx, confirm_time
    high = df['high'].values
    low  = df['low'].values
    n    = len(df)

    is_high = np.zeros(n, dtype=bool)
    is_low  = np.zeros(n, dtype=bool)
    for i in range(strength, n - strength):
        win_h = high[i-strength:i+strength+1]
        win_l = low [i-strength:i+strength+1]
        if high[i] == win_h.max() and (win_h[:strength] < high[i]).all() and (win_h[strength+1:] < high[i]).all():
            is_high[i] = True
        if low[i]  == win_l.min() and (win_l[:strength] > low[i] ).all() and (win_l[strength+1:] > low[i] ).all():
            is_low[i]  = True

    def _collect(mask: np.ndarray, prices: np.ndarray) -> pd.DataFrame:
        rows = []
        last_idx = -10**9
        for i in np.where(mask)[0]:
            if i - last_idx < min_gap:
                continue
            rows.append({
                'idx':          int(i),
                'time':         df['time'].iat[i],
                'price':        float(prices[i]),
                'rsi':          float(df['rsi'].iat[i]) if pd.notna(df['rsi'].iat[i]) else np.nan,
                'confirm_idx':  int(i + strength),
                'confirm_time': df['time'].iat[min(i + strength, n-1)],
            })
            last_idx = i
        return pd.DataFrame(rows)

    return _collect(is_high, high), _collect(is_low, low)


highs_df, lows_df = detect_pivots(df)
highs_df = highs_df.dropna(subset=['rsi']).reset_index(drop=True)
lows_df  = lows_df .dropna(subset=['rsi']).reset_index(drop=True)

print(f'Significant highs : {len(highs_df):>5}')
print(f'Significant lows  : {len(lows_df):>5}')
print('--- last 3 highs ---'); print(highs_df.tail(3))
print('--- last 3 lows  ---'); print(lows_df.tail(3))
""")

md(r"""## Step 5 — Plot Detected Pivots on Price + RSI

Sanity check that the pivot detector behaves before we build divergences on top of it.""")

code(r"""def plot_pivots(df: pd.DataFrame, highs_df: pd.DataFrame, lows_df: pd.DataFrame,
                bars: int = 600, title_suffix: str = ''):
    # Plot the last `bars` of price + RSI with detected pivots overlaid.
    view = df.tail(bars).reset_index(drop=True)
    if view.empty:
        return
    t0, t1 = view['time'].iloc[0], view['time'].iloc[-1]
    hi = highs_df[(highs_df['time'] >= t0) & (highs_df['time'] <= t1)]
    lo = lows_df [(lows_df ['time'] >= t0) & (lows_df ['time'] <= t1)]

    fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], shared_xaxes=True,
                        vertical_spacing=0.05,
                        subplot_titles=[f'{SYMBOL} {TIMEFRAME} — Pivots {title_suffix}', 'RSI'])

    fig.add_trace(go.Candlestick(
        x=view['time'], open=view['open'], high=view['high'],
        low=view['low'], close=view['close'], name='price',
        increasing_line_color='#26a69a', decreasing_line_color='#ef5350'), row=1, col=1)

    fig.add_trace(go.Scatter(x=hi['time'], y=hi['price'], mode='markers',
                             marker=dict(symbol='triangle-down', size=10, color='#FF1744',
                                         line=dict(color='white', width=1)),
                             name='swing high'), row=1, col=1)
    fig.add_trace(go.Scatter(x=lo['time'], y=lo['price'], mode='markers',
                             marker=dict(symbol='triangle-up', size=10, color='#00E676',
                                         line=dict(color='white', width=1)),
                             name='swing low'), row=1, col=1)

    fig.add_trace(go.Scatter(x=view['time'], y=view['rsi'], mode='lines',
                             line=dict(color='#00E5FF', width=1.5), name='RSI(14)'),
                  row=2, col=1)
    fig.add_hline(y=RSI_OVERBOUGHT, line_color='#FF1744', line_dash='dot', row=2, col=1)
    fig.add_hline(y=RSI_OVERSOLD,   line_color='#00E676', line_dash='dot', row=2, col=1)
    fig.add_hline(y=50,             line_color='gray',    line_dash='dash', row=2, col=1)
    fig.add_trace(go.Scatter(x=hi['time'], y=hi['rsi'], mode='markers',
                             marker=dict(symbol='triangle-down', size=8, color='#FF1744'),
                             name='RSI@high', showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=lo['time'], y=lo['rsi'], mode='markers',
                             marker=dict(symbol='triangle-up', size=8, color='#00E676'),
                             name='RSI@low', showlegend=False), row=2, col=1)

    fig.update_layout(template='plotly_dark', height=720, xaxis_rangeslider_visible=False,
                      xaxis2_rangeslider_visible=False)
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    fig.show()


plot_pivots(df, highs_df, lows_df, bars=600)
""")

md(r"""## Step 6 — Build Pivot-to-Pivot Pairs and Classify Each

For each side (highs / lows), pair every pivot with its immediate predecessor (subject to the gap & lookback rules). Compute price slope and RSI slope, then classify:

- **Regular Bearish** : highs → price up, RSI down (+ optional OB filter on first pivot)
- **Regular Bullish** : lows  → price down, RSI up (+ optional OS filter)
- **Hidden  Bearish** : highs → price down, RSI up
- **Hidden  Bullish** : lows  → price up, RSI down
- **Convergence**      : price and RSI move the same direction (no edge — logged but not traded)""")

code(r"""def classify_pair(side: str, p1: pd.Series, p2: pd.Series) -> str:
    # side='high' or 'low'. Returns: REG_BEAR / REG_BULL / HID_BEAR / HID_BULL / CONV_* / NONE.
    price_up = p2['price'] > p1['price']
    rsi_up   = p2['rsi']   > p1['rsi']

    if abs(p2['rsi'] - p1['rsi']) < MIN_RSI_DELTA:
        return 'NONE'
    if MIN_PRICE_DELTA > 0 and abs(p2['price'] - p1['price']) < MIN_PRICE_DELTA:
        return 'NONE'

    if side == 'high':
        if price_up and not rsi_up:
            if REQUIRE_OB_OS and p1['rsi'] < RSI_OVERBOUGHT:
                return 'NONE'
            return 'REG_BEAR'
        if not price_up and rsi_up:
            return 'HID_BEAR'
        return 'CONV_UP' if price_up else 'CONV_DOWN'
    else:  # low
        if not price_up and rsi_up:
            if REQUIRE_OB_OS and p1['rsi'] > RSI_OVERSOLD:
                return 'NONE'
            return 'REG_BULL'
        if price_up and not rsi_up:
            return 'HID_BULL'
        return 'CONV_UP' if price_up else 'CONV_DOWN'


def build_pairs(pivots: pd.DataFrame, side: str) -> pd.DataFrame:
    rows = []
    for i in range(1, len(pivots)):
        p1 = pivots.iloc[i-1]
        p2 = pivots.iloc[i]
        gap = p2['idx'] - p1['idx']
        if gap < PIVOT_MIN_GAP_BARS or gap > PIVOT_LOOKBACK_BARS:
            continue
        kind = classify_pair(side, p1, p2)
        if kind == 'NONE':
            continue
        rows.append({
            'side':         side,
            'kind':         kind,
            'p1_idx':       int(p1['idx']),
            'p1_time':      p1['time'],
            'p1_price':     float(p1['price']),
            'p1_rsi':       float(p1['rsi']),
            'p2_idx':       int(p2['idx']),
            'p2_time':      p2['time'],
            'p2_price':     float(p2['price']),
            'p2_rsi':       float(p2['rsi']),
            'confirm_idx':  int(p2['confirm_idx']),
            'confirm_time': p2['confirm_time'],
            'bars_between': int(gap),
            'price_slope':  float(p2['price'] - p1['price']) / max(gap, 1),
            'rsi_slope':    float(p2['rsi']   - p1['rsi'])   / max(gap, 1),
        })
    return pd.DataFrame(rows)


pairs_high = build_pairs(highs_df, 'high')
pairs_low  = build_pairs(lows_df,  'low')
pairs_all  = pd.concat([pairs_high, pairs_low], ignore_index=True).sort_values('confirm_idx').reset_index(drop=True)

print('Classified pivot pairs:')
for k, v in pairs_all['kind'].value_counts().items():
    print(f'  {k:10s} : {v}')
pairs_all.tail(8)
""")

md(r"""## Step 7 — Visualize Divergence Lines (last window)

Price-side line + RSI-side line for every classified pair in the last window, colour-coded:

- 🔴 Regular Bearish · 🟢 Regular Bullish
- 🟠 Hidden Bearish · 🔵 Hidden Bullish
- ⚪ Convergence (no trade)""")

code(r"""KIND_COLOR = {
    'REG_BEAR':  '#FF1744',
    'REG_BULL':  '#00E676',
    'HID_BEAR':  '#FFA726',
    'HID_BULL':  '#2196F3',
    'CONV_UP':   '#9E9E9E',
    'CONV_DOWN': '#9E9E9E',
}
KIND_LABEL = {
    'REG_BEAR':  'Reg Bear Div',
    'REG_BULL':  'Reg Bull Div',
    'HID_BEAR':  'Hid Bear Div',
    'HID_BULL':  'Hid Bull Div',
    'CONV_UP':   'Convergence',
    'CONV_DOWN': 'Convergence',
}


def plot_divergences(df: pd.DataFrame, pairs: pd.DataFrame,
                     bars: int = 800, only_kinds: Optional[List[str]] = None):
    view = df.tail(bars).reset_index(drop=True)
    if view.empty:
        return
    t0, t1 = view['time'].iloc[0], view['time'].iloc[-1]
    sub = pairs[(pairs['p2_time'] >= t0) & (pairs['p2_time'] <= t1)]
    if only_kinds:
        sub = sub[sub['kind'].isin(only_kinds)]

    fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], shared_xaxes=True,
                        vertical_spacing=0.05,
                        subplot_titles=[f'{SYMBOL} {TIMEFRAME} — Divergence / Convergence', 'RSI'])

    fig.add_trace(go.Candlestick(
        x=view['time'], open=view['open'], high=view['high'],
        low=view['low'], close=view['close'], name='price',
        increasing_line_color='#26a69a', decreasing_line_color='#ef5350'), row=1, col=1)
    fig.add_trace(go.Scatter(x=view['time'], y=view['rsi'], mode='lines',
                             line=dict(color='#00E5FF', width=1.4), name='RSI(14)'), row=2, col=1)
    fig.add_hline(y=RSI_OVERBOUGHT, line_color='#FF1744', line_dash='dot', row=2, col=1)
    fig.add_hline(y=RSI_OVERSOLD,   line_color='#00E676', line_dash='dot', row=2, col=1)
    fig.add_hline(y=50,             line_color='gray',    line_dash='dash', row=2, col=1)

    seen_legend = set()
    for _, r in sub.iterrows():
        c     = KIND_COLOR[r['kind']]
        label = KIND_LABEL[r['kind']]
        show  = label not in seen_legend
        seen_legend.add(label)

        fig.add_trace(go.Scatter(
            x=[r['p1_time'], r['p2_time']],
            y=[r['p1_price'], r['p2_price']],
            mode='lines+markers', line=dict(color=c, width=2.4),
            marker=dict(size=8, color=c, line=dict(color='white', width=1)),
            name=label, legendgroup=label, showlegend=show,
            hovertemplate=f'{label}<br>%{{x}}<br>price=%{{y:.5f}}<extra></extra>',
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[r['p1_time'], r['p2_time']],
            y=[r['p1_rsi'],  r['p2_rsi']],
            mode='lines+markers', line=dict(color=c, width=2.4, dash='dot'),
            marker=dict(size=7, color=c, line=dict(color='white', width=1)),
            name=label, legendgroup=label, showlegend=False,
            hovertemplate=f'{label} RSI<br>%{{x}}<br>rsi=%{{y:.2f}}<extra></extra>',
        ), row=2, col=1)

    fig.update_layout(template='plotly_dark', height=780, xaxis_rangeslider_visible=False,
                      xaxis2_rangeslider_visible=False)
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    fig.show()


plot_divergences(df, pairs_all, bars=800)
""")

md(r"""## Step 8 — Build Trade Plans From Tradeable Pairs

Each tradeable pair (regular or hidden divergence) becomes one trade plan:

- **Entry** : open of the bar immediately *after* the second pivot's confirm bar (no look-ahead).
- **SL**    : just beyond the structural pivot (`p2_price`), with `SL_BUFFER_FRAC` extra.
- **TP**    : entry ± `RR_TARGET * R`, where `R = |entry − SL|`.""")

code(r"""TRADEABLE_KINDS = []
if ALLOW_REGULAR: TRADEABLE_KINDS += ['REG_BEAR', 'REG_BULL']
if ALLOW_HIDDEN:  TRADEABLE_KINDS += ['HID_BEAR', 'HID_BULL']

DIR_MAP = {'REG_BEAR': 'SHORT', 'HID_BEAR': 'SHORT',
           'REG_BULL': 'LONG',  'HID_BULL': 'LONG'}


def build_trade_plans(df: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(df)
    for _, r in pairs.iterrows():
        if r['kind'] not in TRADEABLE_KINDS:
            continue
        entry_idx = int(r['confirm_idx']) + 1
        if entry_idx >= n:
            continue
        direction = DIR_MAP[r['kind']]
        entry_price = float(df['open'].iat[entry_idx])
        pivot_price = float(r['p2_price'])

        if direction == 'SHORT':
            raw_R = max(pivot_price - entry_price, 1e-9)
            sl = pivot_price + raw_R * SL_BUFFER_FRAC
            R  = sl - entry_price
            if R <= 0:
                continue
            tp = entry_price - RR_TARGET * R
        else:  # LONG
            raw_R = max(entry_price - pivot_price, 1e-9)
            sl = pivot_price - raw_R * SL_BUFFER_FRAC
            R  = entry_price - sl
            if R <= 0:
                continue
            tp = entry_price + RR_TARGET * R

        rows.append({
            **r.to_dict(),
            'direction':   direction,
            'entry_idx':   entry_idx,
            'entry_time':  df['time'].iat[entry_idx],
            'entry_price': entry_price,
            'sl':          sl,
            'tp':          tp,
            'risk':        abs(R),
            'rr':          RR_TARGET,
        })
    return pd.DataFrame(rows)


plans_df = build_trade_plans(df, pairs_all)
print(f'Trade plans built : {len(plans_df)}')
if not plans_df.empty:
    print(plans_df['kind'].value_counts().to_string())
plans_df[['confirm_time','kind','direction','entry_price','sl','tp','risk','rr']].tail(6)
""")

md(r"""## Step 9 — Simulate Trades

Walk forward bar-by-bar from `entry_idx` until SL or TP is touched (or `MAX_TRADE_BARS` elapses). If both touch in the same bar, assume SL hits first (pessimistic).""")

code(r"""def simulate_trade(df: pd.DataFrame, plan: pd.Series) -> Dict:
    entry_idx = int(plan['entry_idx'])
    last_idx  = min(entry_idx + MAX_TRADE_BARS, len(df) - 1)
    direction = plan['direction']
    sl        = float(plan['sl'])
    tp        = float(plan['tp'])
    entry_price = float(plan['entry_price'])

    for j in range(entry_idx, last_idx + 1):
        bar = df.iloc[j]
        if direction == 'LONG':
            hit_sl = bar['low']  <= sl
            hit_tp = bar['high'] >= tp
        else:
            hit_sl = bar['high'] >= sl
            hit_tp = bar['low']  <= tp

        if hit_sl and hit_tp:
            return {'status':'CLOSED','result':'SL','exit_idx':j,'exit_time':bar['time'],
                    'exit_price':sl,'pnl_r':-1.0,'bars_held':j-entry_idx+1}
        if hit_sl:
            return {'status':'CLOSED','result':'SL','exit_idx':j,'exit_time':bar['time'],
                    'exit_price':sl,'pnl_r':-1.0,'bars_held':j-entry_idx+1}
        if hit_tp:
            return {'status':'CLOSED','result':'TP','exit_idx':j,'exit_time':bar['time'],
                    'exit_price':tp,'pnl_r':RR_TARGET,'bars_held':j-entry_idx+1}

    last = df.iloc[last_idx]
    if direction == 'LONG':
        pnl = (last['close'] - entry_price) / plan['risk']
    else:
        pnl = (entry_price - last['close']) / plan['risk']
    return {'status':'TIMEOUT','result':'OPEN','exit_idx':last_idx,'exit_time':last['time'],
            'exit_price':float(last['close']),'pnl_r':round(float(pnl),3),
            'bars_held':last_idx - entry_idx + 1}


def run_backtest(df: pd.DataFrame, plans_df: pd.DataFrame) -> pd.DataFrame:
    if plans_df.empty:
        return pd.DataFrame()
    rows = []
    for _, plan in plans_df.iterrows():
        sim = simulate_trade(df, plan)
        rows.append({**plan.to_dict(), **sim})
    return pd.DataFrame(rows)


trades_df = run_backtest(df, plans_df)
n_total  = len(trades_df)
n_closed = int((trades_df['result'].isin(['TP','SL'])).sum()) if n_total else 0
print(f'Plans            : {len(plans_df)}')
print(f'Closed trades    : {n_closed}')
if n_total:
    print(trades_df['result'].value_counts().to_string())
trades_df[['confirm_time','kind','direction','entry_price','sl','tp','exit_time','result','pnl_r','bars_held']].tail(8)
""")

md(r"""## Step 10 — Performance Metrics""")

code(r"""def calc_metrics(trades_df: pd.DataFrame) -> dict:
    if trades_df.empty:
        return {}
    closed = trades_df[trades_df['result'].isin(['TP','SL'])].copy().reset_index(drop=True)
    if closed.empty:
        return {'closed': closed}
    n  = len(closed)
    w  = int((closed['result']=='TP').sum())
    wr = w / n
    closed['cum_r'] = closed['pnl_r'].cumsum()
    dd = closed['cum_r'] - closed['cum_r'].cummax()
    pos = closed.loc[closed['pnl_r']>0, 'pnl_r'].sum()
    neg = abs(closed.loc[closed['pnl_r']<0, 'pnl_r'].sum())
    pf  = (pos / neg) if neg > 0 else float('inf')
    arr = (closed['result']=='SL').astype(int).values
    max_cl = streak = 0
    for v in arr:
        streak = streak+1 if v else 0
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
        'cum_r'           : closed['cum_r'],
        'drawdown'        : dd,
        'closed'          : closed,
    }


metrics = calc_metrics(trades_df)

if metrics and metrics.get('total_trades', 0):
    sep = '=' * 60
    print(sep)
    print('  PERFORMANCE — RSI Divergence Scalper')
    print(sep)
    print(f'  Symbol         : {SYMBOL} {TIMEFRAME}')
    print(f'  Trades         : {metrics["total_trades"]}')
    print(f'  Wins / Losses  : {metrics["wins"]} / {metrics["losses"]}')
    print(f'  Win Rate       : {metrics["win_rate"]*100:.1f}%')
    print(f'  Total R        : {metrics["total_r"]:+.2f} R')
    print(f'  Avg R / trade  : {metrics["avg_r"]:+.3f} R')
    print(f'  Profit Factor  : {metrics["profit_factor"]:.2f}')
    print(f'  Expectancy     : {metrics["expectancy"]:+.3f} R')
    print(f'  Max Drawdown   : {metrics["max_dd_r"]:.2f} R')
    print(f'  Max CL streak  : {metrics["max_consec_loss"]}')
    print(f'  Avg duration   : {metrics["avg_bars"]:.0f} {TIMEFRAME} bars')
    print(sep)
    be_wr = 1.0 / (1.0 + RR_TARGET)
    print(f'  Break-even WR  : {be_wr*100:.1f}%   →   Edge: '
          f'{"POSITIVE" if metrics["win_rate"]>be_wr else "NEGATIVE"}')

    print()
    print('By divergence kind:')
    for k in ['REG_BEAR','REG_BULL','HID_BEAR','HID_BULL']:
        sub = metrics['closed'][metrics['closed']['kind']==k]
        if sub.empty:
            continue
        wr_k = (sub['result']=='TP').mean()
        print(f'  {k:9s} n={len(sub):3d}  WR={wr_k*100:5.1f}%  AvgR={sub["pnl_r"].mean():+.3f}')
else:
    print('No closed trades — try smaller PIVOT_STRENGTH or longer LOOKBACK_DAYS.')
""")

md(r"""## Step 11 — Equity Curve, Drawdown & Per-Trade PnL""")

code(r"""def plot_equity(metrics: dict, name: str = 'RSI Divergence Scalper'):
    if not metrics or 'cum_r' not in metrics or len(metrics['closed']) == 0:
        print('No equity data.')
        return
    cum_r  = metrics['cum_r'].reset_index(drop=True)
    dd     = metrics['drawdown'].reset_index(drop=True)
    closed = metrics['closed'].reset_index(drop=True)

    fig = make_subplots(
        rows=3, cols=1, row_heights=[0.5, 0.25, 0.25],
        subplot_titles=['Cumulative Equity (R)', 'Drawdown', 'Per-Trade PnL (R)'],
        vertical_spacing=0.08,
    )
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

    colors = ['#00E676' if r=='TP' else '#FF1744' for r in closed['result']]
    fig.add_trace(go.Bar(x=closed.index, y=closed['pnl_r'],
                         marker_color=colors, name='PnL'), row=3, col=1)
    fig.add_hline(y=0, line_color='gray', line_dash='dash', row=3, col=1)

    fig.update_layout(
        title=dict(text=(f'{name} — {SYMBOL} {TIMEFRAME}  ·  WR={metrics["win_rate"]*100:.0f}%  '
                         f'Total={metrics["total_r"]:+.1f}R  '
                         f'PF={metrics["profit_factor"]:.2f}  '
                         f'MaxDD={metrics["max_dd_r"]:.1f}R'), x=0.5),
        height=760, template='plotly_dark', showlegend=True,
    )
    fig.show()


if metrics and metrics.get('total_trades', 0):
    plot_equity(metrics)
""")

md(r"""## Step 12 — Per-Trade Review (last 5 trades on price + RSI)""")

code(r"""def plot_trade(trade: pd.Series, df: pd.DataFrame, pad_bars: int = 25) -> go.Figure:
    p1_idx  = int(trade['p1_idx'])
    exit_ix = int(trade['exit_idx']) if pd.notna(trade.get('exit_idx')) else int(trade['entry_idx'])
    a = max(0, p1_idx - pad_bars)
    b = min(len(df), exit_ix + pad_bars + 1)
    view = df.iloc[a:b].reset_index(drop=True)
    if view.empty:
        return go.Figure()

    c = KIND_COLOR[trade['kind']]
    fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], shared_xaxes=True,
                        vertical_spacing=0.05,
                        subplot_titles=[f'{KIND_LABEL[trade["kind"]]} — {trade["direction"]}', 'RSI'])

    fig.add_trace(go.Candlestick(
        x=view['time'], open=view['open'], high=view['high'],
        low=view['low'], close=view['close'], name='price',
        increasing_line_color='#26a69a', decreasing_line_color='#ef5350'), row=1, col=1)
    fig.add_trace(go.Scatter(x=view['time'], y=view['rsi'], mode='lines',
                             line=dict(color='#00E5FF', width=1.4), name='RSI'), row=2, col=1)
    fig.add_hline(y=RSI_OVERBOUGHT, line_color='#FF1744', line_dash='dot', row=2, col=1)
    fig.add_hline(y=RSI_OVERSOLD,   line_color='#00E676', line_dash='dot', row=2, col=1)
    fig.add_hline(y=50,             line_color='gray',    line_dash='dash', row=2, col=1)

    # divergence lines
    fig.add_trace(go.Scatter(x=[trade['p1_time'], trade['p2_time']],
                             y=[trade['p1_price'], trade['p2_price']],
                             mode='lines+markers', line=dict(color=c, width=2.6),
                             marker=dict(size=10, color=c, line=dict(color='white', width=1.2)),
                             name='Price line'), row=1, col=1)
    fig.add_trace(go.Scatter(x=[trade['p1_time'], trade['p2_time']],
                             y=[trade['p1_rsi'],  trade['p2_rsi']],
                             mode='lines+markers', line=dict(color=c, width=2.6, dash='dot'),
                             marker=dict(size=9, color=c, line=dict(color='white', width=1)),
                             name='RSI line'), row=2, col=1)

    entry_t = trade['entry_time']
    fig.add_hline(y=trade['entry_price'], line_color='white',  line_dash='dash',
                  annotation_text=f'Entry {trade["entry_price"]:.5f}',
                  annotation_position='top left', row=1, col=1)
    fig.add_hline(y=trade['sl'], line_color='#FF1744', line_dash='dot',
                  annotation_text=f'SL {trade["sl"]:.5f}',
                  annotation_position='bottom left', row=1, col=1)
    fig.add_hline(y=trade['tp'], line_color='#00E676', line_dash='dot',
                  annotation_text=f'TP {trade["tp"]:.5f}',
                  annotation_position='top left', row=1, col=1)

    esym = 'triangle-up' if trade['direction']=='LONG' else 'triangle-down'
    ec   = '#00E676'     if trade['direction']=='LONG' else '#FF1744'
    fig.add_trace(go.Scatter(x=[entry_t], y=[trade['entry_price']],
                             mode='markers+text', marker=dict(symbol=esym, size=15, color=ec,
                                                              line=dict(color='white', width=2)),
                             text=['ENTRY'], textposition='top center',
                             textfont=dict(size=10, color='white'), name='Entry'), row=1, col=1)
    if pd.notna(trade.get('exit_time')) and trade.get('result') in ('TP','SL','OPEN'):
        xc = '#00E676' if trade['result']=='TP' else ('#FF1744' if trade['result']=='SL' else 'gray')
        xs = 'star' if trade['result']=='TP' else ('x' if trade['result']=='SL' else 'circle-open')
        fig.add_trace(go.Scatter(x=[trade['exit_time']], y=[trade['exit_price']],
                                 mode='markers+text',
                                 marker=dict(symbol=xs, size=16, color=xc,
                                             line=dict(color='white', width=2)),
                                 text=[trade['result']], textposition='top center',
                                 textfont=dict(size=10, color='white'),
                                 name=f'Exit {trade["result"]}'), row=1, col=1)

    pnl = trade.get('pnl_r', 0.0) or 0.0
    fig.update_layout(
        title=(f'{KIND_LABEL[trade["kind"]]} | {trade["direction"]:5s} | '
               f'{trade["confirm_time"]} | {trade.get("result","-"):4s} | PnL: {pnl:+.2f}R'),
        template='plotly_dark', height=620,
        xaxis_rangeslider_visible=False, xaxis2_rangeslider_visible=False,
    )
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    return fig


if not trades_df.empty:
    N_SHOW = min(5, len(trades_df))
    for i in range(N_SHOW):
        t = trades_df.iloc[-(N_SHOW - i)]
        plot_trade(t, df).show()
        pnl = t.get('pnl_r', 0.0) or 0.0
        print(f'  {t["kind"]:9s} {t["direction"]:5s} | {t["confirm_time"]} | '
              f'{t.get("result","-"):4s} {pnl:+.2f}R')
else:
    print('No trades to plot.')
""")

md(r"""## Step 13 — Save Results to `./results/rsi_divergence_scalper/{SYMBOL}/{TF}/`""")

code(r"""RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if trades_df.empty:
    print('No trades to save.')
else:
    trades_path = RESULTS_DIR / 'trades.csv'
    pairs_path  = RESULTS_DIR / 'pivot_pairs.csv'
    trades_df.to_csv(trades_path, index=False)
    pairs_all.to_csv(pairs_path,  index=False)

    if metrics and metrics.get('total_trades', 0):
        summary = pd.DataFrame([{
            'symbol'           : SYMBOL,
            'timeframe'        : TIMEFRAME,
            'strategy'         : STRATEGY_NAME,
            'lookback_days'    : LOOKBACK_DAYS,
            'rsi_period'       : RSI_PERIOD,
            'rsi_overbought'   : RSI_OVERBOUGHT,
            'rsi_oversold'     : RSI_OVERSOLD,
            'pivot_strength'   : PIVOT_STRENGTH,
            'pivot_min_gap'    : PIVOT_MIN_GAP_BARS,
            'pivot_lookback'   : PIVOT_LOOKBACK_BARS,
            'min_rsi_delta'    : MIN_RSI_DELTA,
            'require_ob_os'    : REQUIRE_OB_OS,
            'allow_hidden'     : ALLOW_HIDDEN,
            'allow_regular'    : ALLOW_REGULAR,
            'rr_target'        : RR_TARGET,
            'sl_buffer_frac'   : SL_BUFFER_FRAC,
            'max_trade_bars'   : MAX_TRADE_BARS,
            'pivot_highs'      : int(len(highs_df)),
            'pivot_lows'       : int(len(lows_df)),
            'pairs_total'      : int(len(pairs_all)),
            'pairs_tradeable'  : int(len(plans_df)),
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

        eq_path = RESULTS_DIR / 'equity.csv'
        metrics['closed'][['entry_time','kind','direction','result','pnl_r','cum_r']].to_csv(eq_path, index=False)

        print(f'Saved → {trades_path}')
        print(f'Saved → {pairs_path}')
        print(f'Saved → {summary_path}')
        print(f'Saved → {eq_path}')
        display(summary.T.rename(columns={0:'value'}))
    else:
        print(f'Saved → {trades_path}   (no closed trades — summary skipped)')
        print(f'Saved → {pairs_path}')
""")

md(r"""## Step 14 — Final Analysis & Knobs to Tune""")

code(r"""if metrics and metrics.get('total_trades', 0):
    be_wr  = 1.0 / (1.0 + RR_TARGET)
    closed = metrics['closed']
    sep = '=' * 60
    print(sep)
    print('  RSI DIVERGENCE SCALPER — FINAL ANALYSIS')
    print(sep)
    print('\n[EDGE]')
    print(f'  RR fixed       : {RR_TARGET:.2f}:1')
    print(f'  Break-even WR  : {be_wr*100:.1f}%')
    print(f'  Actual WR      : {metrics["win_rate"]*100:.1f}%')
    print(f'  Expectancy     : {metrics["expectancy"]:+.3f} R/trade')
    edge = 'POSITIVE' if (metrics["win_rate"] > be_wr and metrics["profit_factor"] > 1) else 'NEGATIVE'
    print(f'  Edge           : {edge}')

    print('\n[BY DIRECTION]')
    for d in ['LONG','SHORT']:
        sub = closed[closed['direction']==d]
        if sub.empty: continue
        wr_d = (sub['result']=='TP').mean()
        print(f'  {d:5s} : n={len(sub):3d}  WR={wr_d*100:4.1f}%  AvgR={sub["pnl_r"].mean():+.3f}')

    print('\n[BY KIND]')
    for k in ['REG_BEAR','REG_BULL','HID_BEAR','HID_BULL']:
        sub = closed[closed['kind']==k]
        if sub.empty: continue
        wr_k = (sub['result']=='TP').mean()
        print(f'  {k:9s} n={len(sub):3d}  WR={wr_k*100:5.1f}%  AvgR={sub["pnl_r"].mean():+.3f}  '
              f'TotalR={sub["pnl_r"].sum():+.2f}')

    print('\n[STRATEGY INSIGHTS]')
    print('  + Confirmed pivots (±N bars) remove noise; no look-ahead in entries')
    print('  + Trade plan uses next-bar open after confirm — mechanically reproducible')
    print('  + Hidden vs regular divergences scored separately so you can drop the loser')

    print('\n[KNOBS TO TUNE]')
    print('  > PIVOT_STRENGTH (3–7)        : higher = fewer / cleaner pivots')
    print('  > MIN_RSI_DELTA (0.5–3.0)     : minimum RSI gap to call divergence')
    print('  > REQUIRE_OB_OS               : drop weak regular divergences mid-range')
    print('  > RR_TARGET (1.0 → 3.0)       : changes BE win-rate; sweep for best PF')
    print('  > SL_BUFFER_FRAC              : tighter SL = more stop-outs but bigger R')
    print('  > Confluence: only take signals near S/R from notebook 01')
    print(sep)
else:
    print('No closed trades to analyze. Try:')
    print('  - Lower PIVOT_STRENGTH (e.g. 3)')
    print('  - Increase LOOKBACK_DAYS')
    print('  - Disable REQUIRE_OB_OS')
    print('  - Lower MIN_RSI_DELTA')
""")

md(r"""## Step 15 — SHARP-Divergence Filter (subset backtest)

A "sharp" divergence is one where momentum is being torn apart **quickly** — not drifting slowly across many bars. Three hard gates + a percentile cut:

1. `|Δ RSI|` between the two pivots ≥ `SHARP_MIN_RSI_DELTA` (absolute RSI points)
2. `|Δ RSI| / bars_between` ≥ `SHARP_MIN_RSI_PER_BAR` (RSI points per bar — the **steepness**)
3. `bars_between` ≤ `SHARP_MAX_BARS` (no slow long-range pairs)
4. Of the survivors, keep only the **top `SHARP_TOP_PCT` %** by combined `sharpness` score
   (steep RSI per bar × steep normalized price per bar).

The same `calc_metrics` / `plot_equity` are re-used so the sharp subset is directly comparable to the full backtest above.""")

code(r"""# ── Sharp-divergence thresholds (tune freely) ────────────────────────────────
SHARP_MIN_RSI_DELTA   = 5.0    # absolute RSI points between the two pivots
SHARP_MIN_RSI_PER_BAR = 0.20   # RSI points per bar between pivots (steepness)
SHARP_MAX_BARS        = 30     # divergences spread over more bars are 'slow'
SHARP_TOP_PCT         = 50.0   # additionally keep only top X% by sharpness score


def sharpness_score(row: pd.Series) -> float:
    bars = max(int(row['bars_between']), 1)
    rsi_per_bar       = abs(row['p2_rsi']   - row['p1_rsi'])   / bars
    price_per_bar_pct = abs(row['p2_price'] - row['p1_price']) / max(float(row['p1_price']), 1e-9) / bars
    return float(rsi_per_bar * price_per_bar_pct * 10000)   # scaled for readability


if trades_df.empty:
    print('No trades to filter.')
else:
    t = trades_df.copy()
    t['rsi_delta']   = (t['p2_rsi'] - t['p1_rsi']).abs()
    t['rsi_per_bar'] = t['rsi_delta'] / t['bars_between'].clip(lower=1)
    t['sharpness']   = t.apply(sharpness_score, axis=1)

    hard_mask = (
        (t['rsi_delta']    >= SHARP_MIN_RSI_DELTA)   &
        (t['rsi_per_bar']  >= SHARP_MIN_RSI_PER_BAR) &
        (t['bars_between'] <= SHARP_MAX_BARS)
    )
    if hard_mask.sum() > 0:
        cutoff = t.loc[hard_mask, 'sharpness'].quantile(1.0 - SHARP_TOP_PCT/100.0)
        final_mask = hard_mask & (t['sharpness'] >= cutoff)
    else:
        final_mask = hard_mask

    sharp_trades = t[final_mask].reset_index(drop=True)
    n_all, n_sharp = len(t), len(sharp_trades)
    print(f'Total trades        : {n_all}')
    print(f'After hard gates    : {int(hard_mask.sum())}  ({hard_mask.mean()*100:.1f}%)')
    print(f'After top-{SHARP_TOP_PCT:.0f}% sharpness : {n_sharp}  ({n_sharp/max(n_all,1)*100:.1f}% of all)')

    if n_sharp == 0:
        print('No sharp trades survived — try lowering SHARP_MIN_RSI_DELTA or SHARP_MIN_RSI_PER_BAR.')
    else:
        sharp_metrics = calc_metrics(sharp_trades)
        if sharp_metrics and sharp_metrics.get('total_trades', 0):
            sep = '=' * 60
            print()
            print(sep)
            print('  SHARP-ONLY PERFORMANCE')
            print(sep)
            print(f'  Trades         : {sharp_metrics["total_trades"]}')
            print(f'  Win Rate       : {sharp_metrics["win_rate"]*100:.1f}%')
            print(f'  Total R        : {sharp_metrics["total_r"]:+.2f} R')
            print(f'  Avg R / trade  : {sharp_metrics["avg_r"]:+.3f} R')
            print(f'  Profit Factor  : {sharp_metrics["profit_factor"]:.2f}')
            print(f'  Expectancy     : {sharp_metrics["expectancy"]:+.3f} R')
            print(f'  Max Drawdown   : {sharp_metrics["max_dd_r"]:.2f} R')
            print(f'  Max CL streak  : {sharp_metrics["max_consec_loss"]}')
            be_wr = 1.0 / (1.0 + RR_TARGET)
            print(f'  Break-even WR  : {be_wr*100:.1f}%   →   Edge: '
                  f'{"POSITIVE" if sharp_metrics["win_rate"]>be_wr else "NEGATIVE"}')
            print(sep)

            print('\nBy divergence kind (sharp only):')
            for k in ['REG_BEAR','REG_BULL','HID_BEAR','HID_BULL']:
                sub = sharp_metrics['closed'][sharp_metrics['closed']['kind']==k]
                if sub.empty: continue
                wr_k = (sub['result']=='TP').mean()
                print(f'  {k:9s} n={len(sub):3d}  WR={wr_k*100:5.1f}%  AvgR={sub["pnl_r"].mean():+.3f}  '
                      f'TotalR={sub["pnl_r"].sum():+.2f}')

            plot_equity(sharp_metrics, name='RSI Divergence — SHARP ONLY')

            # save the sharp-only outputs alongside the full ones
            sharp_path = RESULTS_DIR / 'trades_sharp.csv'
            sharp_trades.to_csv(sharp_path, index=False)
            print(f'\nSaved → {sharp_path}')
""")

# ─────────────────────────────────────────────────────────────────────────────
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

out = Path(__file__).parent / "19_rsi_divergence_scalper.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Wrote {out}  ({len(CELLS)} cells)")
