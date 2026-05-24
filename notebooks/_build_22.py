"""Generate notebook 22: 08:00 NY H1 anchor candle plotted on M5 with Fib expansion + H1 trend."""
import json
from pathlib import Path

CELLS: list[tuple[str, str]] = []


def md(s: str) -> None:
    CELLS.append(("markdown", s))


def code(s: str) -> None:
    CELLS.append(("code", s))


md(r"""# Notebook #22 — 08:00 NY H1 Anchor Candle + Fib Expansion (H1 over M5)

Two stacked Plotly subplots for a chosen date:

1. **H1 chart (top)** — H1 candles with EMA 8 / 13 / 21 and a colored trend ribbon (bull / bear / flat). The 08:00 NY anchor candle is highlighted.
2. **M5 chart (bottom)** — zoomed-in M5 window around the anchor, the H1 anchor candle drawn as a translucent box, the H1 EMAs overlaid (step-projected onto M5), and Fibonacci **expansion** levels:
   - **Bullish anchor** (close ≥ open) → leg = **low → high**, levels project upward.
   - **Bearish anchor** (close < open) → leg = **high → low**, levels project downward.

Trend definition matches `strategies/ema_trend/crypto_core.py`:
`bull` ⇔ `ema8 > ema13 > ema21 AND close > ema21`, `bear` ⇔ all inequalities flipped, else `flat`.
""")

md(r"""## Step 1 — Imports & Config""")

code(r"""import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Config ───────────────────────────────────────────────────────────────────
SYMBOL              = 'EURUSD'
ANCHOR_DATE         = '2025-09-15'      # any date with M5 + H1 data
H1_BARS_BEFORE      = 48                # H1 bars before anchor (top chart context)
H1_BARS_AFTER       = 24                # H1 bars after  anchor
M5_BARS_BEFORE      = 24                # M5 bars before  the H1 candle starts
M5_BARS_AFTER       = 60                # M5 bars after   the H1 candle ends
BROKER_TO_NY_OFFSET_H = 7               # broker 15:00 == NY 08:00 (see memory)
ANCHOR_HOUR_BROKER  = 15                # = 08:00 NY

# EMA trend (parity with strategies/ema_trend/crypto_core.py)
EMA_FAST, EMA_MID, EMA_SLOW = 8, 13, 21

# Fib expansion ratios (extensions beyond the leg)
FIB_LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0,
              1.272, 1.618, 2.0, 2.618]

DATA_ROOT = Path('data') / SYMBOL
H1_CSV    = DATA_ROOT / 'H1' / 'ohlcv.csv'
M5_CSV    = DATA_ROOT / 'M5' / 'ohlcv.csv'
print(H1_CSV.exists(), M5_CSV.exists())
""")

md(r"""## Step 2 — Load H1 & M5 (broker clock, naive)

The CSV `+00:00` suffix is **misleading** — these are broker wall-clock times where `15:00 broker == 08:00 NY`. We strip the tz info and work with naive timestamps; only when we need NY-day labels do we subtract the offset.""")

code(r"""def load_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)   # naive broker time
    return df[['time', 'open', 'high', 'low', 'close']].sort_values('time').reset_index(drop=True)

h1 = load_ohlcv(H1_CSV)
m5 = load_ohlcv(M5_CSV)

h1['broker_hour'] = h1['time'].dt.hour
h1['ny_time']    = h1['time'] - pd.Timedelta(hours=BROKER_TO_NY_OFFSET_H)
h1['ny_date']    = h1['ny_time'].dt.date

print(f'H1 rows: {len(h1):,}   range: {h1.time.min()}  →  {h1.time.max()}')
print(f'M5 rows: {len(m5):,}   range: {m5.time.min()}  →  {m5.time.max()}')
""")

md(r"""## Step 3 — Compute H1 EMAs + trend (bull / bear / flat)""")

code(r"""h1[f'ema_{EMA_FAST}'] = h1['close'].ewm(span=EMA_FAST, adjust=False).mean()
h1[f'ema_{EMA_MID}']  = h1['close'].ewm(span=EMA_MID,  adjust=False).mean()
h1[f'ema_{EMA_SLOW}'] = h1['close'].ewm(span=EMA_SLOW, adjust=False).mean()

bull = (
    (h1[f'ema_{EMA_FAST}'] > h1[f'ema_{EMA_MID}'])
    & (h1[f'ema_{EMA_MID}']  > h1[f'ema_{EMA_SLOW}'])
    & (h1['close']           > h1[f'ema_{EMA_SLOW}'])
)
bear = (
    (h1[f'ema_{EMA_FAST}'] < h1[f'ema_{EMA_MID}'])
    & (h1[f'ema_{EMA_MID}']  < h1[f'ema_{EMA_SLOW}'])
    & (h1['close']           < h1[f'ema_{EMA_SLOW}'])
)
h1['trend'] = np.where(bull, 'bull', np.where(bear, 'bear', 'flat'))

h1[['time', 'close', f'ema_{EMA_FAST}', f'ema_{EMA_MID}', f'ema_{EMA_SLOW}', 'trend']].tail(5)
""")

md(r"""## Step 4 — Find the 08:00-NY anchor candle for the chosen date""")

code(r"""target_ny_date = pd.to_datetime(ANCHOR_DATE).date()

mask = (h1.ny_date == target_ny_date) & (h1.broker_hour == ANCHOR_HOUR_BROKER)
if not mask.any():
    raise ValueError(f'No 08:00-NY H1 candle for {ANCHOR_DATE} (likely weekend/holiday)')

anchor_idx   = h1.index[mask][0]
anchor       = h1.loc[anchor_idx]
anchor_start = anchor['time']                          # broker 15:00
anchor_end   = anchor_start + pd.Timedelta(hours=1)    # broker 16:00 (= NY 09:00)

is_bullish = anchor['close'] >= anchor['open']
direction  = 'BULLISH' if is_bullish else 'BEARISH'

print(f'Anchor H1 candle (broker {anchor_start} → {anchor_end})  ·  H1 trend = {anchor.trend.upper()}')
print(f'  open  = {anchor.open:.5f}')
print(f'  high  = {anchor.high:.5f}')
print(f'  low   = {anchor.low:.5f}')
print(f'  close = {anchor.close:.5f}')
print(f'  candle direction = {direction}')
""")

md(r"""## Step 5 — Slice H1 (context) and M5 (zoom) windows around the anchor""")

code(r"""# H1 window (wider context for the top chart)
h1_ws = anchor_start - pd.Timedelta(hours=H1_BARS_BEFORE)
h1_we = anchor_end   + pd.Timedelta(hours=H1_BARS_AFTER)
h1_win = h1[(h1.time >= h1_ws) & (h1.time < h1_we)].reset_index(drop=True)

# M5 window (zoom for the bottom chart)
m5_ws = anchor_start - pd.Timedelta(minutes=5 * M5_BARS_BEFORE)
m5_we = anchor_end   + pd.Timedelta(minutes=5 * M5_BARS_AFTER)
m5_win = m5[(m5.time >= m5_ws) & (m5.time < m5_we)].reset_index(drop=True)

# Project H1 EMAs onto M5 (step / forward-fill) so they overlay correctly
ema_cols = [f'ema_{EMA_FAST}', f'ema_{EMA_MID}', f'ema_{EMA_SLOW}']
m5_win = pd.merge_asof(
    m5_win.sort_values('time'),
    h1[['time', *ema_cols, 'trend']].sort_values('time'),
    on='time', direction='backward',
)

print(f'H1 window: {h1_ws} → {h1_we}   ({len(h1_win)} bars)')
print(f'M5 window: {m5_ws} → {m5_we}   ({len(m5_win)} bars)')
""")

md(r"""## Step 6 — Build Fib expansion levels for the anchor leg""")

code(r"""rng = anchor.high - anchor.low

if is_bullish:
    leg_from, leg_to = anchor.low, anchor.high
    fib_prices = {r: anchor.low + r * rng for r in FIB_LEVELS}
else:
    leg_from, leg_to = anchor.high, anchor.low
    fib_prices = {r: anchor.high - r * rng for r in FIB_LEVELS}

print(f'Leg: {leg_from:.5f}  →  {leg_to:.5f}   ({direction})')
for r, p in fib_prices.items():
    tag = '   (anchor)' if r in (0.0, 1.0) else ''
    print(f'  {r:>5.3f}  →  {p:.5f}{tag}')
""")

md(r"""## Step 7 — Plot: H1 (top, with trend ribbon & EMAs) + M5 (bottom, with anchor box & Fib)""")

code(r"""# ── Helpers ──────────────────────────────────────────────────────────────────
TREND_COLOR = {'bull': 'rgba(38,166,154,0.35)',     # green
               'bear': 'rgba(239,83,80,0.35)',      # red
               'flat': 'rgba(120,144,156,0.18)'}    # grey
EMA_COLOR   = {EMA_FAST: '#ffb74d',                 # orange
               EMA_MID : '#ba68c8',                 # purple
               EMA_SLOW: '#4fc3f7'}                 # blue

def trend_bands(df: pd.DataFrame, bar_minutes: int) -> list[dict]:
    # rectangle shapes that color the background by H1 trend over a window
    bar_w  = pd.Timedelta(minutes=bar_minutes)
    shapes = []
    for _, r in df.iterrows():
        shapes.append(dict(
            type='rect', xref='x', yref='paper',
            x0=r['time'], x1=r['time'] + bar_w,
            y0=0, y1=1,
            line=dict(width=0),
            fillcolor=TREND_COLOR.get(r.get('trend', 'flat'), TREND_COLOR['flat']),
            layer='below',
        ))
    return shapes

# ── Figure with 2 subplots ───────────────────────────────────────────────────
fig = make_subplots(
    rows=2, cols=1, shared_xaxes=False,
    row_heights=[0.45, 0.55], vertical_spacing=0.06,
    subplot_titles=(
        f'H1 — {SYMBOL}  ·  anchor {anchor_start:%Y-%m-%d %H:%M}  ·  trend = {anchor.trend.upper()}',
        f'M5 — zoom around anchor  ·  candle = {direction}  ·  leg {leg_from:.5f} → {leg_to:.5f}',
    ),
)

# ─── Top: H1 candles + trend ribbon + EMAs ───────────────────────────────────
h1_shapes = trend_bands(h1_win, bar_minutes=60)

fig.add_trace(go.Candlestick(
    x=h1_win['time'],
    open=h1_win['open'], high=h1_win['high'],
    low =h1_win['low'],  close=h1_win['close'],
    name='H1', increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
    showlegend=False,
), row=1, col=1)

for span in (EMA_FAST, EMA_MID, EMA_SLOW):
    fig.add_trace(go.Scatter(
        x=h1_win['time'], y=h1_win[f'ema_{span}'],
        mode='lines', name=f'EMA{span}',
        line=dict(color=EMA_COLOR[span], width=1.4),
        legendgroup=f'ema{span}',
    ), row=1, col=1)

# anchor box on H1
fig.add_shape(type='rect',
              x0=anchor_start, x1=anchor_end,
              y0=anchor.low,   y1=anchor.high,
              line=dict(color='rgba(255,193,7,0.95)', width=2),
              fillcolor='rgba(255,193,7,0.08)',
              row=1, col=1)
fig.add_annotation(
    x=anchor_start, y=anchor.high,
    text=f'08:00 NY anchor ({direction})',
    showarrow=False, yshift=12,
    font=dict(color='#ffb300', size=11),
    xanchor='left', row=1, col=1,
)

# ─── Bottom: M5 candles + anchor box + projected H1 EMAs + Fib ───────────────
m5_shapes = trend_bands(m5_win, bar_minutes=5)

fig.add_trace(go.Candlestick(
    x=m5_win['time'],
    open=m5_win['open'], high=m5_win['high'],
    low =m5_win['low'],  close=m5_win['close'],
    name='M5', increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
    showlegend=False,
), row=2, col=1)

# H1 EMAs projected onto M5 (step lines)
for span in (EMA_FAST, EMA_MID, EMA_SLOW):
    fig.add_trace(go.Scatter(
        x=m5_win['time'], y=m5_win[f'ema_{span}'],
        mode='lines', name=f'H1 EMA{span}',
        line=dict(color=EMA_COLOR[span], width=1.1, dash='dot', shape='hv'),
        legendgroup=f'ema{span}', showlegend=False,
    ), row=2, col=1)

# anchor box on M5
fig.add_shape(type='rect',
              x0=anchor_start, x1=anchor_end,
              y0=anchor.low,   y1=anchor.high,
              line=dict(color='rgba(255,193,7,0.95)', width=2),
              fillcolor='rgba(255,193,7,0.10)',
              row=2, col=1)

# Fib expansion levels (annotated at right edge)
for r, price in fib_prices.items():
    if r in (0.0, 1.0):
        col, wd, dash = '#42a5f5', 2,   'solid'
    elif r > 1.0:
        col, wd, dash = '#ab47bc', 1.5, 'dash'
    else:
        col, wd, dash = '#90a4ae', 1,   'dot'
    fig.add_shape(type='line',
                  x0=m5_ws, x1=m5_we, y0=price, y1=price,
                  line=dict(color=col, width=wd, dash=dash),
                  row=2, col=1)
    fig.add_annotation(x=m5_we, y=price,
                       text=f'  {r:.3f}   {price:.5f}',
                       showarrow=False, xanchor='left',
                       font=dict(color=col, size=10),
                       row=2, col=1)

# Apply background trend bands to both rows (have to do after add_shape since
# row/col aware shapes don't accept paper-yref directly — we use y-min/y-max).
def add_bg_bands(df, bar_minutes, row, ymin, ymax):
    bar_w = pd.Timedelta(minutes=bar_minutes)
    for _, r in df.iterrows():
        fig.add_shape(
            type='rect',
            x0=r['time'], x1=r['time'] + bar_w,
            y0=ymin, y1=ymax,
            line=dict(width=0),
            fillcolor=TREND_COLOR.get(r.get('trend', 'flat'), TREND_COLOR['flat']),
            layer='below',
            row=row, col=1,
        )

# Padded y-range so bands fill the panel cleanly
def _pad(lo, hi, pct=0.05):
    span = hi - lo
    return lo - pct * span, hi + pct * span

h1_lo, h1_hi = _pad(h1_win[['low']].min().iloc[0], h1_win[['high']].max().iloc[0])
m5_lo, m5_hi = _pad(min(m5_win[['low']].min().iloc[0],  min(fib_prices.values())),
                    max(m5_win[['high']].max().iloc[0], max(fib_prices.values())))

add_bg_bands(h1_win, 60, row=1, ymin=h1_lo, ymax=h1_hi)
add_bg_bands(m5_win,  5, row=2, ymin=m5_lo, ymax=m5_hi)

# Layout
fig.update_layout(
    title=f'{SYMBOL}  —  {ANCHOR_DATE}  ·  H1 trend = {anchor.trend.upper()}  ·  anchor candle = {direction}',
    template='plotly_dark',
    height=900,
    margin=dict(l=40, r=140, t=80, b=40),
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
)
fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
fig.update_xaxes(rangeslider_visible=False, row=2, col=1)
fig.update_yaxes(range=[h1_lo, h1_hi], row=1, col=1, title_text='Price (H1)')
fig.update_yaxes(range=[m5_lo, m5_hi], row=2, col=1, title_text='Price (M5)')

fig.show()
""")

md(r"""## Step 8 — Reusable: plot any date

Wraps Steps 4–7 into a single function. Background ribbon = H1 trend, dotted lines on M5 = H1 EMAs forward-filled across the M5 window.""")

code(r"""def plot_anchor(anchor_date: str):
    target = pd.to_datetime(anchor_date).date()
    mk = (h1.ny_date == target) & (h1.broker_hour == ANCHOR_HOUR_BROKER)
    if not mk.any():
        print(f'skip {anchor_date}: no anchor (weekend/holiday?)')
        return
    a       = h1.loc[h1.index[mk][0]]
    a_start = a['time']
    a_end   = a_start + pd.Timedelta(hours=1)
    bull_   = a['close'] >= a['open']
    dir_    = 'BULLISH' if bull_ else 'BEARISH'
    rng_    = a.high - a.low
    if bull_:
        fibs = {r: a.low + r * rng_ for r in FIB_LEVELS}
        lf, lt = a.low, a.high
    else:
        fibs = {r: a.high - r * rng_ for r in FIB_LEVELS}
        lf, lt = a.high, a.low

    hws = a_start - pd.Timedelta(hours=H1_BARS_BEFORE)
    hwe = a_end   + pd.Timedelta(hours=H1_BARS_AFTER)
    mws = a_start - pd.Timedelta(minutes=5 * M5_BARS_BEFORE)
    mwe = a_end   + pd.Timedelta(minutes=5 * M5_BARS_AFTER)

    hw = h1[(h1.time >= hws) & (h1.time < hwe)].reset_index(drop=True)
    mw = m5[(m5.time >= mws) & (m5.time < mwe)].reset_index(drop=True)
    mw = pd.merge_asof(
        mw.sort_values('time'),
        h1[['time', *ema_cols, 'trend']].sort_values('time'),
        on='time', direction='backward',
    )

    f = make_subplots(
        rows=2, cols=1, shared_xaxes=False,
        row_heights=[0.45, 0.55], vertical_spacing=0.06,
        subplot_titles=(
            f'H1 — {SYMBOL}  ·  {anchor_date}  ·  trend = {a.trend.upper()}',
            f'M5 — anchor = {dir_}  ·  leg {lf:.5f} → {lt:.5f}',
        ),
    )
    # H1
    f.add_trace(go.Candlestick(x=hw['time'], open=hw['open'], high=hw['high'],
                               low=hw['low'], close=hw['close'], showlegend=False,
                               increasing_line_color='#26a69a',
                               decreasing_line_color='#ef5350'), row=1, col=1)
    for sp in (EMA_FAST, EMA_MID, EMA_SLOW):
        f.add_trace(go.Scatter(x=hw['time'], y=hw[f'ema_{sp}'], name=f'EMA{sp}',
                               line=dict(color=EMA_COLOR[sp], width=1.4)), row=1, col=1)
    f.add_shape(type='rect', x0=a_start, x1=a_end, y0=a.low, y1=a.high,
                line=dict(color='rgba(255,193,7,0.95)', width=2),
                fillcolor='rgba(255,193,7,0.08)', row=1, col=1)

    # M5
    f.add_trace(go.Candlestick(x=mw['time'], open=mw['open'], high=mw['high'],
                               low=mw['low'], close=mw['close'], showlegend=False,
                               increasing_line_color='#26a69a',
                               decreasing_line_color='#ef5350'), row=2, col=1)
    for sp in (EMA_FAST, EMA_MID, EMA_SLOW):
        f.add_trace(go.Scatter(x=mw['time'], y=mw[f'ema_{sp}'], showlegend=False,
                               line=dict(color=EMA_COLOR[sp], width=1.1,
                                         dash='dot', shape='hv')), row=2, col=1)
    f.add_shape(type='rect', x0=a_start, x1=a_end, y0=a.low, y1=a.high,
                line=dict(color='rgba(255,193,7,0.95)', width=2),
                fillcolor='rgba(255,193,7,0.10)', row=2, col=1)
    for r_, p_ in fibs.items():
        if r_ in (0.0, 1.0): col_, dash_, wd_ = '#42a5f5', 'solid', 2
        elif r_ > 1.0:       col_, dash_, wd_ = '#ab47bc', 'dash',  1.5
        else:                col_, dash_, wd_ = '#90a4ae', 'dot',   1
        f.add_shape(type='line', x0=mws, x1=mwe, y0=p_, y1=p_,
                    line=dict(color=col_, width=wd_, dash=dash_), row=2, col=1)
        f.add_annotation(x=mwe, y=p_, text=f'  {r_:.3f}   {p_:.5f}',
                         showarrow=False, xanchor='left',
                         font=dict(color=col_, size=10), row=2, col=1)

    # background trend ribbon
    h1_lo_, h1_hi_ = _pad(hw[['low']].min().iloc[0], hw[['high']].max().iloc[0])
    m5_lo_, m5_hi_ = _pad(min(mw[['low']].min().iloc[0],  min(fibs.values())),
                          max(mw[['high']].max().iloc[0], max(fibs.values())))
    for _, rr in hw.iterrows():
        f.add_shape(type='rect', x0=rr['time'],
                    x1=rr['time'] + pd.Timedelta(minutes=60),
                    y0=h1_lo_, y1=h1_hi_, line=dict(width=0),
                    fillcolor=TREND_COLOR.get(rr.get('trend', 'flat'), TREND_COLOR['flat']),
                    layer='below', row=1, col=1)
    for _, rr in mw.iterrows():
        f.add_shape(type='rect', x0=rr['time'],
                    x1=rr['time'] + pd.Timedelta(minutes=5),
                    y0=m5_lo_, y1=m5_hi_, line=dict(width=0),
                    fillcolor=TREND_COLOR.get(rr.get('trend', 'flat'), TREND_COLOR['flat']),
                    layer='below', row=2, col=1)

    f.update_layout(template='plotly_dark', height=900,
                    margin=dict(l=40, r=140, t=80, b=40),
                    title=f'{SYMBOL}  —  {anchor_date}  ·  H1 trend = {a.trend.upper()}  ·  candle = {dir_}',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02,
                                xanchor='right', x=1))
    f.update_xaxes(rangeslider_visible=False, row=1, col=1)
    f.update_xaxes(rangeslider_visible=False, row=2, col=1)
    f.update_yaxes(range=[h1_lo_, h1_hi_], row=1, col=1, title_text='Price (H1)')
    f.update_yaxes(range=[m5_lo_, m5_hi_], row=2, col=1, title_text='Price (M5)')
    f.show()

# Example sweep
for d in ['2025-09-15', '2025-09-16', '2025-09-17']:
    plot_anchor(d)
""")


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

out = Path(__file__).parent / "22_8am_ny_anchor_fib_expansion.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Wrote {out}  ({len(CELLS)} cells)")
