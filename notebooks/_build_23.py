"""Generate notebook 23: multi-strategy trend-following scalper on XAUUSD M5.

Pipeline:
  D1 + H1 trend consensus  →  M5 reaction filters (BB / EMA20 / RSI / Pin-Engulf / OB)
  Score-based aggregator  →  trades with RR=2  →  two SL methods (structural vs ATR)
  Per-filter ablation table  →  roadmap for iterative improvement.
"""
import json
from pathlib import Path

CELLS: list[tuple[str, str]] = []


def md(s: str) -> None:
    CELLS.append(("markdown", s))


def code(s: str) -> None:
    CELLS.append(("code", s))


# ─────────────────────────────────────────────────────────────────────────────
md(r"""# Notebook #23 — Multi-Strategy Trend-Following Scalper (XAUUSD M5)

**Goal**: stack several independent *price-reaction* detectors behind a multi-timeframe
trend filter, and require a **score** (multiple confirmations) before every entry.
With **RR=2** the break-even win-rate is 33.3%; the target here is **win-rate ≥ 45%**
with a *high trade count* on a short timeframe.

### Components

| Layer | Filter | Role |
|---|---|---|
| Trend | D1 EMA50 slope | Macro direction (only long if D1 up, only short if D1 down) |
| Trend | H1 EMA50 vs price | Tactical alignment with H1 |
| Reaction | Bollinger lower/upper touch | Mean-reversion edge inside trend |
| Reaction | EMA20 pullback + bounce | Dynamic support/resistance |
| Reaction | RSI exit from OS/OB | Momentum re-engagement |
| Reaction | Pin-bar / Engulfing | Rejection candle confirmation |
| Reaction | Order-Block retest | Institutional zone retest |
| Context | London + NY session | Avoid Asia chop |

### Aggregation

A signal fires when **all trend filters agree** AND **at least `MIN_REACTIONS` reaction
filters fire on the same side**. The score acts as a quality knob — raise it to lift
win-rate at the cost of trade count, lower it to do the opposite.

### Two SL methods, side-by-side

Per the discussion we report **both**:
1. **Structural** — SL beyond the last swing low/high (size depends on structure)
2. **ATR-based** — `SL = entry ± k·ATR14` (uniform sizing)

Both run on the same signal stream so the comparison is apples-to-apples.

> **Important**: the OHLCV `+00:00` suffix is misleading — the broker clock is Europe/Nicosia
> (UTC+2/+3). For session filtering we shift broker → NY by **−7 h** (per project memory).
""")

# ─────────────────────────────────────────────────────────────────────────────
md(r"""## Step 1 — Imports & Configuration""")

code(r"""import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

pio.renderers.default = 'notebook'
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.4f}'.format)

# ── Identity ────────────────────────────────────────────────────────────────
STRATEGY     = 'multi_strategy_scalper'
SYMBOL       = 'XAUUSD'
DATA_DIR     = Path('./data')
RESULTS_DIR  = Path('./results') / STRATEGY / SYMBOL
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Windows: in-sample (sweep + tuning) vs forward (Phase 8 — strictly unseen) ─
# Forward window is the most recent 30 days. The sweep / Phase 7 only see data up
# to DATE_TO; Phase 8 then runs the frozen best config on FORWARD_TO data
# the optimisation never touched.
DATE_FROM    = '2025-11-01'
DATE_TO      = '2026-04-21'   # in-sample cutoff
FORWARD_FROM = '2026-04-21'   # honest forward starts here
FORWARD_TO   = '2026-05-22'   # end of available data

# ── Broker → NY offset (per data_timezone memory) ───────────────────────────
BROKER_TO_NY_H = 7
NY_SESSION_START_H = 8    # NY 08:00 — peak London/NY overlap starts
NY_SESSION_END_H   = 13   # NY 13:00 — London close. Tighter window = denser quality.

# ── Indicators ──────────────────────────────────────────────────────────────
EMA_FAST       = 20
EMA_TREND_H1   = 50
EMA_TREND_D1   = 50
BB_PERIOD      = 20
BB_STD         = 2.0
RSI_PERIOD     = 14
RSI_OS         = 35.0     # widened back from 30 so RSI fires more often → more overlap with confirms
RSI_OB         = 65.0
ATR_PERIOD     = 14

# ── Reaction-filter parameters ──────────────────────────────────────────────
PULLBACK_TOLERANCE_ATR = 0.4   # tighter pullback band (was 0.5)
PIN_BAR_WICK_RATIO     = 0.60  # stronger rejection wick (was 0.55)
ENGULF_LOOKBACK        = 1
OB_DISPLACEMENT_BARS   = 4
OB_DISPLACEMENT_ATR    = 1.5
OB_RETEST_TOLERANCE    = 0.0
OB_EXPIRY_BARS         = 96

# ── Aggregator ──────────────────────────────────────────────────────────────
# Active reaction filters — derived from ablation in v1 run:
#   f_rsi WR 48.2%, f_candle 45.8%, f_ema 42.9% (kept)
#   f_bb  WR 37.7%, f_ob     38.6%               (dropped — were dragging WR down)
ACTIVE_FILTERS = ['f_ema', 'f_rsi', 'f_candle']
MIN_REACTIONS  = 2   # 2 of 3 active filters must agree

# ── Risk ────────────────────────────────────────────────────────────────────
RR             = 2.0
ATR_SL_MULT    = 1.0          # for the ATR-SL variant
STRUCT_LOOKBACK_BARS = 12     # swing window for the structural-SL variant
SL_BUFFER_ATR  = 0.10         # tiny buffer beyond structure (in ATR units)
MAX_HOLD_BARS  = 96
ONE_TRADE_AT_A_TIME = True

print(f'{STRATEGY} on {SYMBOL}  window {DATE_FROM} → {DATE_TO}')
print(f'  MIN_REACTIONS = {MIN_REACTIONS}   RR = {RR}   max-hold = {MAX_HOLD_BARS} bars')
""")

# ─────────────────────────────────────────────────────────────────────────────
md(r"""## Step 2 — Load M5 / H1 / D1""")

code(r"""def load_ohlcv(symbol: str, tf: str, date_from: str, date_to: str) -> pd.DataFrame:
    path = DATA_DIR / symbol / tf / 'ohlcv.csv'
    df = pd.read_csv(path)
    # tz-naive "broker wall-clock" — the +00:00 suffix is misleading (per memory)
    df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
    df = df.sort_values('time').reset_index(drop=True)
    keep = ['time', 'open', 'high', 'low', 'close', 'tick_volume']
    df = df[[c for c in keep if c in df.columns]].copy()
    df.rename(columns={'tick_volume': 'volume'}, inplace=True)
    df = df[(df['time'] >= pd.Timestamp(date_from)) & (df['time'] <= pd.Timestamp(date_to))]
    return df.reset_index(drop=True)


df_m5 = load_ohlcv(SYMBOL, 'M5', DATE_FROM, DATE_TO)
df_h1 = load_ohlcv(SYMBOL, 'H1', DATE_FROM, DATE_TO)
df_d1 = load_ohlcv(SYMBOL, 'D1', DATE_FROM, DATE_TO)

print(f'M5: {len(df_m5):>7,}   H1: {len(df_h1):>5,}   D1: {len(df_d1):>4,}')
print(f'Range: {df_m5["time"].min()} → {df_m5["time"].max()}')
df_m5.tail(2)
""")

# ─────────────────────────────────────────────────────────────────────────────
md(r"""## Step 3 — Indicators

EMA20, BB(20,2), RSI(14), ATR(14), and the candle-body / wick features used by the
pin-bar and engulfing detectors. The trend on H1 and D1 is just `close vs EMA50` plus
the EMA50 slope direction.
""")

code(r"""def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0)
    loss = (-d).clip(lower=0)
    ag = gain.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    al = loss.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low']  - df['close'].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()


def add_m5_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ema20']  = ema(df['close'], EMA_FAST)
    df['rsi']    = rsi(df['close'], RSI_PERIOD)
    df['atr']    = atr(df, ATR_PERIOD)
    mid = df['close'].rolling(BB_PERIOD).mean()
    std = df['close'].rolling(BB_PERIOD).std()
    df['bb_mid'] = mid
    df['bb_up']  = mid + BB_STD * std
    df['bb_lo']  = mid - BB_STD * std
    # Candle anatomy
    df['body']       = (df['close'] - df['open']).abs()
    df['range']      = (df['high']  - df['low']).clip(lower=1e-9)
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['is_bull']    = df['close'] > df['open']
    df['is_bear']    = df['close'] < df['open']
    return df


def add_htf_trend(df_htf: pd.DataFrame, ema_period: int) -> pd.DataFrame:
    df_htf = df_htf.copy()
    e = ema(df_htf['close'], ema_period)
    slope = e.diff()
    df_htf['ema_trend'] = e
    df_htf['trend_dir'] = np.where(
        (df_htf['close'] > e) & (slope > 0), 1,
        np.where((df_htf['close'] < e) & (slope < 0), -1, 0)
    )
    return df_htf[['time', 'ema_trend', 'trend_dir']]


df_m5 = add_m5_features(df_m5)
df_h1_t = add_htf_trend(df_h1, EMA_TREND_H1)
df_d1_t = add_htf_trend(df_d1, EMA_TREND_D1)


# Project H1 + D1 trend onto M5 with merge_asof (left-join, no look-ahead)
df_m5 = pd.merge_asof(
    df_m5.sort_values('time'),
    df_h1_t.rename(columns={'ema_trend': 'h1_ema', 'trend_dir': 'h1_trend'}),
    on='time', direction='backward',
)
df_m5 = pd.merge_asof(
    df_m5,
    df_d1_t.rename(columns={'ema_trend': 'd1_ema', 'trend_dir': 'd1_trend'}),
    on='time', direction='backward',
)

print(f'Coverage: {df_m5["h1_trend"].notna().sum()/len(df_m5):.1%} H1, '
      f'{df_m5["d1_trend"].notna().sum()/len(df_m5):.1%} D1')
df_m5[['time', 'close', 'ema20', 'rsi', 'atr', 'bb_lo', 'bb_up', 'h1_trend', 'd1_trend']].tail(3)
""")

# ─────────────────────────────────────────────────────────────────────────────
md(r"""## Step 4 — Trend & Session Gates

`trend_dir`: +1 (only longs), −1 (only shorts), 0 (no trade) — both D1 and H1 must
agree. `in_session`: True for London + NY (07:00–16:00 NY ≈ broker 14:00–23:00).
""")

code(r"""def compute_gates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    same  = df['h1_trend'] == df['d1_trend']
    nonzero = df['h1_trend'] != 0
    df['trend_dir'] = np.where(same & nonzero, df['h1_trend'], 0).astype(int)

    # Broker hour → NY hour
    broker_hour = df['time'].dt.hour
    ny_hour = (broker_hour - BROKER_TO_NY_H) % 24
    df['in_session'] = (ny_hour >= NY_SESSION_START_H) & (ny_hour < NY_SESSION_END_H)
    return df


df_m5 = compute_gates(df_m5)
print('Trend distribution :', df_m5['trend_dir'].value_counts().to_dict())
print('In-session bars    :', int(df_m5['in_session'].sum()), '/', len(df_m5))
""")

# ─────────────────────────────────────────────────────────────────────────────
md(r"""## Step 5 — Reaction Filters

Each filter returns a vector of `+1` (long bias), `−1` (short bias) or `0`.
All filters are computed up-front on the M5 frame; the aggregator below just sums
the ones that agree with the trend.

> Every filter is bar-closing — we read OHLC of the *just-closed* bar at entry time.
""")

code(r"""# ── F1: Bollinger touch ─────────────────────────────────────────────────────
def f_bb_touch(df: pd.DataFrame) -> pd.Series:
    long  = df['low']  <= df['bb_lo']
    short = df['high'] >= df['bb_up']
    return pd.Series(np.where(long, 1, np.where(short, -1, 0)), index=df.index)


# ── F2: EMA20 pullback (close pulled to EMA, candle rejects) ────────────────
def f_ema_pullback(df: pd.DataFrame) -> pd.Series:
    tol = PULLBACK_TOLERANCE_ATR * df['atr']
    touched_long  = (df['low']  <= df['ema20'] + tol) & (df['close'] > df['ema20'])
    touched_short = (df['high'] >= df['ema20'] - tol) & (df['close'] < df['ema20'])
    return pd.Series(np.where(touched_long, 1, np.where(touched_short, -1, 0)), index=df.index)


# ── F3: RSI exits OS / OB on the closing bar ────────────────────────────────
def f_rsi_exit(df: pd.DataFrame) -> pd.Series:
    prev = df['rsi'].shift(1)
    long  = (prev <= RSI_OS) & (df['rsi'] > RSI_OS)
    short = (prev >= RSI_OB) & (df['rsi'] < RSI_OB)
    return pd.Series(np.where(long, 1, np.where(short, -1, 0)), index=df.index)


# ── F4: Pin-bar OR Engulfing ────────────────────────────────────────────────
def f_pin_engulf(df: pd.DataFrame) -> pd.Series:
    rng = df['range']
    bull_pin = (df['lower_wick'] / rng >= PIN_BAR_WICK_RATIO) & (df['close'] > df['open'])
    bear_pin = (df['upper_wick'] / rng >= PIN_BAR_WICK_RATIO) & (df['close'] < df['open'])
    prev_o, prev_c = df['open'].shift(1), df['close'].shift(1)
    bull_eng = (prev_c < prev_o) & (df['close'] > df['open']) & \
               (df['close'] >= prev_o) & (df['open'] <= prev_c)
    bear_eng = (prev_c > prev_o) & (df['close'] < df['open']) & \
               (df['close'] <= prev_o) & (df['open'] >= prev_c)
    long  = bull_pin | bull_eng
    short = bear_pin | bear_eng
    return pd.Series(np.where(long, 1, np.where(short, -1, 0)), index=df.index)


# ── F5: Order-Block retest (vectorized-ish) ─────────────────────────────────
@dataclass
class OB:
    idx: int         # index of the OB candle itself
    expires_at: int  # bar index after which OB is dead
    direction: int   # +1 bullish OB (last bearish before up-displacement), -1 the reverse
    top: float
    bottom: float


def detect_order_blocks(df: pd.DataFrame) -> List[OB]:
    obs: List[OB] = []
    n = len(df)
    o, c = df['open'].values, df['close'].values
    is_bull = (c > o)
    is_bear = (c < o)
    a = df['atr'].values

    i = 0
    while i < n - OB_DISPLACEMENT_BARS:
        # Try bullish displacement starting at i
        j = i
        while j < n and is_bull[j]:
            j += 1
        if j - i >= OB_DISPLACEMENT_BARS:
            move = c[j-1] - o[i]
            if move >= OB_DISPLACEMENT_ATR * a[i]:
                # find last bearish candle before i
                k = i - 1
                while k >= 0 and not is_bear[k]:
                    k -= 1
                if k >= 0:
                    obs.append(OB(idx=k, expires_at=k + OB_EXPIRY_BARS,
                                  direction=+1,
                                  top=float(df['high'].iat[k]),
                                  bottom=float(df['low'].iat[k])))
            i = j
            continue

        j = i
        while j < n and is_bear[j]:
            j += 1
        if j - i >= OB_DISPLACEMENT_BARS:
            move = o[i] - c[j-1]
            if move >= OB_DISPLACEMENT_ATR * a[i]:
                k = i - 1
                while k >= 0 and not is_bull[k]:
                    k -= 1
                if k >= 0:
                    obs.append(OB(idx=k, expires_at=k + OB_EXPIRY_BARS,
                                  direction=-1,
                                  top=float(df['high'].iat[k]),
                                  bottom=float(df['low'].iat[k])))
            i = j
            continue

        i += 1
    return obs


def f_ob_retest(df: pd.DataFrame, obs: List[OB]) -> pd.Series:
    # +1 if current bar wicks back into an unmitigated bullish OB; -1 the reverse.
    sig = np.zeros(len(df), dtype=int)
    high = df['high'].values
    low  = df['low'].values
    for ob in obs:
        start = ob.idx + OB_DISPLACEMENT_BARS  # cannot retest before displacement ends
        end   = min(ob.expires_at, len(df) - 1)
        if start >= end:
            continue
        if ob.direction == +1:
            # bullish OB → wick down into [bottom, top]
            touched = (low[start:end+1] <= ob.top + OB_RETEST_TOLERANCE) & \
                      (high[start:end+1] >= ob.bottom - OB_RETEST_TOLERANCE)
            idx_touched = np.where(touched)[0] + start
            if len(idx_touched):
                sig[idx_touched[0]] = +1  # only first retest
        else:
            touched = (high[start:end+1] >= ob.bottom - OB_RETEST_TOLERANCE) & \
                      (low[start:end+1]  <= ob.top + OB_RETEST_TOLERANCE)
            idx_touched = np.where(touched)[0] + start
            if len(idx_touched):
                sig[idx_touched[0]] = -1
    return pd.Series(sig, index=df.index)


obs = detect_order_blocks(df_m5)
print(f'Order blocks detected: {len(obs)}')

df_m5['f_bb']       = f_bb_touch(df_m5)
df_m5['f_ema']      = f_ema_pullback(df_m5)
df_m5['f_rsi']      = f_rsi_exit(df_m5)
df_m5['f_candle']   = f_pin_engulf(df_m5)
df_m5['f_ob']       = f_ob_retest(df_m5, obs)


# ── RSI-recent gate (5-bar memory of the RSI exit signal) ───────────────────
# Acts as a "fresh momentum snapback zone" — pairs well with a confirming filter.
def f_rsi_recent(df: pd.DataFrame, memory: int = 5) -> pd.Series:
    long_fresh  = (df['f_rsi'] ==  1).rolling(memory).max().fillna(0).astype(bool)
    short_fresh = (df['f_rsi'] == -1).rolling(memory).max().fillna(0).astype(bool)
    return pd.Series(np.where(long_fresh, 1, np.where(short_fresh, -1, 0)), index=df.index)


df_m5['f_rsiR'] = f_rsi_recent(df_m5, memory=10)   # widened to 10-bar memory

filter_cols = ['f_bb', 'f_ema', 'f_rsi', 'f_candle', 'f_ob']   # all computed
fired = {c: int((df_m5[c] != 0).sum()) for c in filter_cols}
print('Fires per filter      :', fired)
print('Active filters (vote) :', ACTIVE_FILTERS)
""")

# ─────────────────────────────────────────────────────────────────────────────
md(r"""## Step 6 — Score-Based Aggregator

`score_long` = number of reaction filters that voted +1 (when trend allows long).
A signal fires when `trend_dir`, session, and `score ≥ MIN_REACTIONS` all line up.
""")

code(r"""def aggregate_signals(df: pd.DataFrame, filter_cols: List[str], min_reactions: int) -> pd.DataFrame:
    df = df.copy()
    f = df[filter_cols].values
    long_votes  = (f ==  1).sum(axis=1)
    short_votes = (f == -1).sum(axis=1)
    df['long_score']  = long_votes
    df['short_score'] = short_votes

    can_long  = (df['trend_dir'] ==  1) & df['in_session'] & (long_votes  >= min_reactions)
    can_short = (df['trend_dir'] == -1) & df['in_session'] & (short_votes >= min_reactions)

    df['signal'] = np.where(can_long, 1, np.where(can_short, -1, 0))
    return df


df_m5 = aggregate_signals(df_m5, ACTIVE_FILTERS, MIN_REACTIONS)
n_long  = int((df_m5['signal'] ==  1).sum())
n_short = int((df_m5['signal'] == -1).sum())
print(f'Raw signals: long={n_long}  short={n_short}  total={n_long+n_short}')
""")

# ─────────────────────────────────────────────────────────────────────────────
md(r"""## Step 7 — Backtest: structural-SL vs ATR-SL (side by side)

Same signal stream, two SL methods. Entry is at the **next bar open** after a signal
(no look-ahead). On each forward bar we check SL first then TP (bar-conservative). If
neither hits by `MAX_HOLD_BARS`, we close at that bar's close.

> When `ONE_TRADE_AT_A_TIME` is true, new signals while in a trade are ignored.
""")

code(r"""@dataclass
class Trade:
    side: int
    entry_idx: int
    entry_time: pd.Timestamp
    entry: float
    sl: float
    tp: float
    exit_idx: int = -1
    exit_time: pd.Timestamp = None
    exit: float = 0.0
    reason: str = ''
    r_multiple: float = 0.0


def structural_sl(df: pd.DataFrame, idx: int, side: int) -> float:
    lo_lookback = max(0, idx - STRUCT_LOOKBACK_BARS)
    a = df['atr'].iat[idx]
    if side == 1:
        return float(df['low'].iloc[lo_lookback:idx].min()) - SL_BUFFER_ATR * a
    else:
        return float(df['high'].iloc[lo_lookback:idx].max()) + SL_BUFFER_ATR * a


def atr_sl(df: pd.DataFrame, idx: int, side: int, entry: float) -> float:
    a = df['atr'].iat[idx]
    return entry - ATR_SL_MULT * a if side == 1 else entry + ATR_SL_MULT * a


def backtest(df: pd.DataFrame, sl_method: str) -> Tuple[List[Trade], pd.Series]:
    trades: List[Trade] = []
    n = len(df)
    in_trade = False
    cur: Trade = None
    equity = 1.0
    eq_curve = np.full(n, np.nan)

    for i in range(n - 1):
        # Manage open trade
        if in_trade:
            hi, lo = df['high'].iat[i], df['low'].iat[i]
            hit_sl = (cur.side == 1 and lo <= cur.sl) or (cur.side == -1 and hi >= cur.sl)
            hit_tp = (cur.side == 1 and hi >= cur.tp) or (cur.side == -1 and lo <= cur.tp)
            exit_now = False; reason = ''; px = 0.0
            if hit_sl and hit_tp:
                exit_now, reason, px = True, 'sl', cur.sl  # conservative: assume SL hit first
            elif hit_sl:
                exit_now, reason, px = True, 'sl', cur.sl
            elif hit_tp:
                exit_now, reason, px = True, 'tp', cur.tp
            elif i - cur.entry_idx >= MAX_HOLD_BARS:
                exit_now, reason, px = True, 'time', float(df['close'].iat[i])
            if exit_now:
                cur.exit_idx = i; cur.exit_time = df['time'].iat[i]
                cur.exit = px; cur.reason = reason
                r_unit = abs(cur.entry - cur.sl)
                cur.r_multiple = ((cur.exit - cur.entry) * cur.side) / r_unit if r_unit > 0 else 0
                equity *= 1 + 0.01 * cur.r_multiple  # 1% risk per trade
                trades.append(cur)
                in_trade = False; cur = None

        # New entry on signal at this bar → executed at next bar open
        sig = int(df['signal'].iat[i])
        if (not in_trade or not ONE_TRADE_AT_A_TIME) and sig != 0:
            entry_idx = i + 1
            entry_px  = float(df['open'].iat[entry_idx])
            if sl_method == 'structural':
                sl = structural_sl(df, entry_idx, sig)
            else:
                sl = atr_sl(df, entry_idx, sig, entry_px)
            r = abs(entry_px - sl)
            if r <= 0 or r > 5 * df['atr'].iat[entry_idx]:   # sanity
                eq_curve[i] = equity
                continue
            tp = entry_px + RR * r * sig
            cur = Trade(side=sig, entry_idx=entry_idx,
                        entry_time=df['time'].iat[entry_idx],
                        entry=entry_px, sl=sl, tp=tp)
            in_trade = True

        eq_curve[i] = equity
    eq_curve[-1] = equity
    return trades, pd.Series(eq_curve, index=df.index)


def stats(trades: List[Trade]) -> Dict:
    if not trades:
        return {'trades': 0, 'wins': 0, 'win_rate': 0.0,
                'expectancy_R': 0.0, 'avg_R': 0.0, 'profit_factor': 0.0}
    wins = [t for t in trades if t.r_multiple > 0]
    losses = [t for t in trades if t.r_multiple <= 0]
    gross_win  = sum(t.r_multiple for t in wins)
    gross_loss = -sum(t.r_multiple for t in losses)
    return {
        'trades'       : len(trades),
        'wins'         : len(wins),
        'win_rate'     : len(wins) / len(trades) * 100,
        'avg_R'        : np.mean([t.r_multiple for t in trades]),
        'expectancy_R' : (len(wins)/len(trades)) * RR - (1 - len(wins)/len(trades)),
        'profit_factor': gross_win / gross_loss if gross_loss > 0 else float('inf'),
    }


trades_struct, eq_struct = backtest(df_m5, 'structural')
trades_atr,    eq_atr    = backtest(df_m5, 'atr')

comparison = pd.DataFrame({
    'structural': stats(trades_struct),
    'atr':        stats(trades_atr),
}).T
comparison
""")

# ─────────────────────────────────────────────────────────────────────────────
md(r"""## Step 8 — Per-Filter Ablation

For each reaction filter, recompute the signal stream with `MIN_REACTIONS=1` and **only
that filter enabled**, then report the resulting win-rate / trade count. This tells us
which filters are pulling their weight on this window.
""")

code(r"""def ablate_one(df: pd.DataFrame, col: str) -> Dict:
    long_votes  = (df[col] ==  1).astype(int).values
    short_votes = (df[col] == -1).astype(int).values
    can_long  = (df['trend_dir'] ==  1) & df['in_session'] & (long_votes  >= 1)
    can_short = (df['trend_dir'] == -1) & df['in_session'] & (short_votes >= 1)
    sig = np.where(can_long, 1, np.where(can_short, -1, 0))
    tmp = df.copy()
    tmp['signal'] = sig
    t, _ = backtest(tmp, 'structural')
    s = stats(t)
    s['filter'] = col
    return s


abl = pd.DataFrame([ablate_one(df_m5, c) for c in filter_cols]).set_index('filter')
abl
""")

md(r"""## Step 9 — Filter-Drop Ablation (sensitivity check)

Same idea, the other direction: keep all filters **except one**, with the original
`MIN_REACTIONS`. If win-rate jumps when a filter is removed, that filter is hurting us.
""")

code(r"""def ablate_drop(df: pd.DataFrame, drop_col: str) -> Dict:
    keep = [c for c in filter_cols if c != drop_col]
    f = df[keep].values
    long_votes  = (f ==  1).sum(axis=1)
    short_votes = (f == -1).sum(axis=1)
    can_long  = (df['trend_dir'] ==  1) & df['in_session'] & (long_votes  >= MIN_REACTIONS)
    can_short = (df['trend_dir'] == -1) & df['in_session'] & (short_votes >= MIN_REACTIONS)
    sig = np.where(can_long, 1, np.where(can_short, -1, 0))
    tmp = df.copy()
    tmp['signal'] = sig
    t, _ = backtest(tmp, 'structural')
    s = stats(t)
    s['dropped'] = drop_col
    return s


drop_abl = pd.DataFrame([ablate_drop(df_m5, c) for c in filter_cols]).set_index('dropped')
drop_abl
""")

# ─────────────────────────────────────────────────────────────────────────────
md(r"""## Step 9b — Config Sweep (auto-pick best WR config)

Search over `(active_filter_subset × MIN_REACTIONS × session_window)` and rank by
win-rate (with a floor on trade count so we don't pick a config that fires twice).
This is the empirical replacement for hand-tuning Phase 1+2 in the roadmap.
""")

code(r"""SUBSETS = [
    ['f_rsi', 'f_candle'],
    ['f_rsi', 'f_ema'],
    ['f_candle', 'f_ema'],
    ['f_rsi', 'f_candle', 'f_ema'],
]
SESSION_WINDOWS = [
    (7, 16),   # full London + NY
    (8, 13),   # London/NY overlap
    (8, 12),   # NY morning only
    (13, 16),  # NY afternoon only
]
MIN_REACT_GRID = [1, 2]
MIN_TRADES_FLOOR = 30   # ≈ 5 trades / month — keeps RSI-gated combos in view

# RSI-required combos: f_rsiR (10-bar memory of the RSI exit) must agree, AND
# the listed confirming filter(s) must fire on the same bar in the same direction.
RSI_GATED = [
    ['f_candle'],                         # gate AND candle confirms
    ['f_ema'],                            # gate AND ema confirms
    ['f_candle', 'f_ema'],                # gate AND (candle OR ema)
]
RSI_GATED_AND = [
    ['f_candle', 'f_ema'],                # gate AND candle AND ema (strict)
]


def sweep_or(df, subset, mr, sess):
    sh, eh = sess
    broker_h = df['time'].dt.hour
    ny_h = (broker_h - BROKER_TO_NY_H) % 24
    in_sess = (ny_h >= sh) & (ny_h < eh)
    f = df[subset].values
    long_votes  = (f ==  1).sum(axis=1)
    short_votes = (f == -1).sum(axis=1)
    can_long  = (df['trend_dir'] ==  1) & in_sess & (long_votes  >= mr)
    can_short = (df['trend_dir'] == -1) & in_sess & (short_votes >= mr)
    sig = np.where(can_long, 1, np.where(can_short, -1, 0))
    tmp = df.copy(); tmp['signal'] = sig
    t, _ = backtest(tmp, 'structural')
    s = stats(t)
    s['subset']    = '+'.join(subset)
    s['min_react'] = mr
    s['session']   = f'NY {sh:02d}-{eh:02d}'
    s['mode']      = 'OR'
    return s


def sweep_rsi_gated(df, confirm_subset, sess, require_all=False):
    # RSI-recent gate (must fire). If require_all: ALL confirms must fire; else ANY.
    sh, eh = sess
    broker_h = df['time'].dt.hour
    ny_h = (broker_h - BROKER_TO_NY_H) % 24
    in_sess = (ny_h >= sh) & (ny_h < eh)

    rsi_long  = (df['f_rsiR'] ==  1)
    rsi_short = (df['f_rsiR'] == -1)
    conf = df[confirm_subset].values
    if require_all:
        conf_long  = (conf ==  1).all(axis=1)
        conf_short = (conf == -1).all(axis=1)
        glue = '&'
        mode = 'RSI-gated-AND'
    else:
        conf_long  = (conf ==  1).any(axis=1)
        conf_short = (conf == -1).any(axis=1)
        glue = '|'
        mode = 'RSI-gated'

    can_long  = (df['trend_dir'] ==  1) & in_sess & rsi_long  & conf_long
    can_short = (df['trend_dir'] == -1) & in_sess & rsi_short & conf_short
    sig = np.where(can_long, 1, np.where(can_short, -1, 0))
    tmp = df.copy(); tmp['signal'] = sig
    t, _ = backtest(tmp, 'structural')
    s = stats(t)
    s['subset']    = 'f_rsiR & (' + glue.join(confirm_subset) + ')'
    s['min_react'] = 'gate'
    s['session']   = f'NY {sh:02d}-{eh:02d}'
    s['mode']      = mode
    return s


rows = []
for sub in SUBSETS:
    for mr in MIN_REACT_GRID:
        if mr > len(sub):
            continue
        for sess in SESSION_WINDOWS:
            rows.append(sweep_or(df_m5, sub, mr, sess))
for confirm in RSI_GATED:
    for sess in SESSION_WINDOWS:
        rows.append(sweep_rsi_gated(df_m5, confirm, sess, require_all=False))
for confirm in RSI_GATED_AND:
    for sess in SESSION_WINDOWS:
        rows.append(sweep_rsi_gated(df_m5, confirm, sess, require_all=True))

TARGET_WR = 48.0   # user requirement: win-rate ≥ 48 %

sweep_all = pd.DataFrame(rows)
sweep = sweep_all[sweep_all['trades'] >= MIN_TRADES_FLOOR].copy()

# Tier 1 — configs that meet the WR target, ranked by *trade count* (more trades win)
tier1 = sweep[sweep['win_rate'] >= TARGET_WR].sort_values(
    by=['trades', 'win_rate'], ascending=False
).reset_index(drop=True)

# Tier 2 — configs below target, ranked by WR (so user sees what's close)
tier2 = sweep[sweep['win_rate'] < TARGET_WR].sort_values(
    by=['win_rate', 'expectancy_R'], ascending=False
).reset_index(drop=True)

print(f'All configurations tested        : {len(sweep_all)}')
print(f'Passing trade-count floor ({MIN_TRADES_FLOOR}) : {len(sweep)}')
print(f'Meeting WR target ({TARGET_WR}%)              : {len(tier1)}')
print()
print(f'--- Tier 1: WR >= {TARGET_WR}%, ranked by trades ---')
print(tier1[['subset','session','trades','win_rate','expectancy_R','profit_factor','mode']].to_string(index=False))
print()
print('--- Tier 2: below target (top 10) ---')
print(tier2[['subset','session','trades','win_rate','expectancy_R','profit_factor','mode']].head(10).to_string(index=False))

sweep = pd.concat([tier1, tier2], ignore_index=True)
sweep.head(15)
""")

md(r"""### Apply best config and re-run

The top-ranked row of `sweep` is used to recompute the signal stream and the final
two-method backtest. This is the *empirical* answer to "which combination works".
""")

code(r"""if len(sweep) == 0:
    print('No configs passed the trade-count floor — relax MIN_TRADES_FLOOR or expand the grid.')
else:
    best = sweep.iloc[0]
    print('=== Best config ===')
    print(best.to_string())
    sh, eh = [int(x) for x in best['session'].replace('NY ', '').split('-')]

    # Recompute session
    broker_h = df_m5['time'].dt.hour
    ny_h = (broker_h - BROKER_TO_NY_H) % 24
    df_m5['in_session'] = (ny_h >= sh) & (ny_h < eh)

    if best['mode'] in ('RSI-gated', 'RSI-gated-AND'):
        # subset string looks like:  "f_rsiR & (f_candle|f_ema)" or "(f_candle&f_ema)"
        inner = best['subset'].split('(')[1].rstrip(')')
        glue  = '&' if best['mode'] == 'RSI-gated-AND' else '|'
        confirm = inner.split(glue)
        rsi_long  = (df_m5['f_rsiR'] ==  1)
        rsi_short = (df_m5['f_rsiR'] == -1)
        conf = df_m5[confirm].values
        if best['mode'] == 'RSI-gated-AND':
            conf_long  = (conf ==  1).all(axis=1)
            conf_short = (conf == -1).all(axis=1)
        else:
            conf_long  = (conf ==  1).any(axis=1)
            conf_short = (conf == -1).any(axis=1)
        can_long  = (df_m5['trend_dir'] ==  1) & df_m5['in_session'] & rsi_long  & conf_long
        can_short = (df_m5['trend_dir'] == -1) & df_m5['in_session'] & rsi_short & conf_short
        df_m5['signal'] = np.where(can_long, 1, np.where(can_short, -1, 0))
    else:
        best_subset = best['subset'].split('+')
        df_m5 = aggregate_signals(df_m5, best_subset, int(best['min_react']))

    trades_struct, eq_struct = backtest(df_m5, 'structural')
    trades_atr,    eq_atr    = backtest(df_m5, 'atr')

    final_compare = pd.DataFrame({
        'structural': stats(trades_struct),
        'atr':        stats(trades_atr),
    }).T
    print()
    print('=== Final backtest with best config ===')
    print(final_compare.to_string())
""")

# ─────────────────────────────────────────────────────────────────────────────
md(r"""## Step 9c — Phase 7 Robustness Check (out-of-sample windows)

The in-sample window (2025-11 → 2026-05) is what the sweep tuned on, so its high WR
could be partly curve-fit. To test if the edge is structural, we **freeze the best
config** and run it unchanged on four **non-overlapping 90-day windows** that come
*before* the in-sample data — the model has never seen them.

**Pass criteria** (per the roadmap stop condition):
- WR ≥ 48 % in **each** window
- Standard deviation of WR across windows ≤ 5 percentage points
- Average trade count ≥ ~10/month (≥ 30 trades / 90-day window)
""")

code(r"""# ── Freeze the best config from Step 9b ─────────────────────────────────────
BEST_CONFIG = {
    'mode'       : 'RSI-gated',
    'confirms'   : ['f_candle', 'f_ema'],   # OR of these alongside the RSI gate
    'rsi_memory' : 10,
    'session'    : (7, 16),                 # NY 07-16
    'sl_method'  : 'structural',
}


# ── Self-contained pipeline runner: load → features → signal → backtest ─────
def run_window(date_from: str, date_to: str, cfg: dict, symbol: str = SYMBOL,
               return_trades: bool = False):
    m5 = load_ohlcv(symbol, 'M5', date_from, date_to)
    h1 = load_ohlcv(symbol, 'H1', date_from, date_to)
    d1 = load_ohlcv(symbol, 'D1', date_from, date_to)
    if len(m5) < 1000 or len(h1) < 50 or len(d1) < 10:
        empty = {'window': f'{date_from} → {date_to}', 'trades': 0, 'win_rate': 0.0,
                 'expectancy_R': 0.0, 'profit_factor': 0.0, 'note': 'insufficient data'}
        return (empty, []) if return_trades else empty

    m5 = add_m5_features(m5)
    h1t = add_htf_trend(h1, EMA_TREND_H1)
    d1t = add_htf_trend(d1, EMA_TREND_D1)
    m5 = pd.merge_asof(m5.sort_values('time'),
                       h1t.rename(columns={'ema_trend':'h1_ema','trend_dir':'h1_trend'}),
                       on='time', direction='backward')
    m5 = pd.merge_asof(m5,
                       d1t.rename(columns={'ema_trend':'d1_ema','trend_dir':'d1_trend'}),
                       on='time', direction='backward')

    # trend gate
    same = m5['h1_trend'] == m5['d1_trend']
    nonzero = m5['h1_trend'] != 0
    m5['trend_dir'] = np.where(same & nonzero, m5['h1_trend'], 0).astype(int)

    # session
    sh, eh = cfg['session']
    ny_h = (m5['time'].dt.hour - BROKER_TO_NY_H) % 24
    m5['in_session'] = (ny_h >= sh) & (ny_h < eh)

    # reaction filters
    m5['f_bb']     = f_bb_touch(m5)
    m5['f_ema']    = f_ema_pullback(m5)
    m5['f_rsi']    = f_rsi_exit(m5)
    m5['f_candle'] = f_pin_engulf(m5)
    obs = detect_order_blocks(m5)
    m5['f_ob']     = f_ob_retest(m5, obs)
    m5['f_rsiR']   = f_rsi_recent(m5, memory=cfg['rsi_memory'])

    # signal: RSI-gated OR mode (gate AND any-of-confirms)
    rsi_long  = (m5['f_rsiR'] ==  1)
    rsi_short = (m5['f_rsiR'] == -1)
    conf = m5[cfg['confirms']].values
    conf_long  = (conf ==  1).any(axis=1)
    conf_short = (conf == -1).any(axis=1)
    can_long  = (m5['trend_dir'] ==  1) & m5['in_session'] & rsi_long  & conf_long
    can_short = (m5['trend_dir'] == -1) & m5['in_session'] & rsi_short & conf_short
    m5['signal'] = np.where(can_long, 1, np.where(can_short, -1, 0))

    trades_, _ = backtest(m5, cfg['sl_method'])
    s = stats(trades_)
    s['window'] = f'{date_from} → {date_to}'
    s['bars']   = len(m5)
    return (s, trades_) if return_trades else s


# ── Four OOS 90-day windows ending just before the in-sample period ─────────
OOS_WINDOWS = [
    ('2024-11-01', '2025-02-01'),
    ('2025-02-01', '2025-05-01'),
    ('2025-05-01', '2025-08-01'),
    ('2025-08-01', '2025-11-01'),
]
IS_WINDOW = (DATE_FROM, DATE_TO)   # the in-sample reference window

rows = []
for df_, dt_ in OOS_WINDOWS:
    rows.append({'set': 'OOS', **run_window(df_, dt_, BEST_CONFIG)})
rows.append({'set': 'IN', **run_window(*IS_WINDOW, BEST_CONFIG)})

robust = pd.DataFrame(rows)
robust = robust[['set', 'window', 'bars', 'trades', 'win_rate', 'expectancy_R', 'profit_factor']]
print('=== Phase 7: per-window results ===')
print(robust.to_string(index=False))

oos = robust[robust['set'] == 'OOS']
mean_wr = oos['win_rate'].mean()
std_wr  = oos['win_rate'].std()
min_wr  = oos['win_rate'].min()
avg_tr  = oos['trades'].mean()

print()
print(f'OOS  mean WR : {mean_wr:.2f} %')
print(f'OOS  std  WR : {std_wr:.2f} pp')
print(f'OOS  min  WR : {min_wr:.2f} %')
print(f'OOS  avg trades / 90-day window : {avg_tr:.1f}')
print()

pass_wr_floor = (oos['win_rate'] >= 48.0).all()
pass_std      = std_wr <= 5.0
pass_volume   = avg_tr >= 30
verdict = pass_wr_floor and pass_std and pass_volume

print('=== VERDICT ===')
print(f'  All windows WR ≥ 48 %   : {pass_wr_floor}')
print(f'  WR std ≤ 5 pp           : {pass_std}')
print(f'  ≥ 30 trades / window    : {pass_volume}')
print(f'  → ROBUSTNESS {"PASS" if verdict else "FAIL"}')

robust.to_csv(RESULTS_DIR / 'phase7_robustness.csv', index=False)
""")

# ─────────────────────────────────────────────────────────────────────────────
md(r"""## Step 9d — Phase 8 Forward Test (the honest one)

The strictest test: the **last 30 days** of data were excluded from the sweep
(in-sample ends at `DATE_TO`). Phase 8 runs the **exact same frozen config** on
this period without any retuning. If the edge holds here, it's real — not a story
the historical data told us in hindsight.

**Pass criteria**:
- WR ≥ 48 %
- Profit factor > 1 (i.e. profitable)
- ≥ 10 trades (otherwise statistically meaningless on 30 days)

This is what the strategy will actually do on day-1 of live deployment.
""")

code(r"""print(f'Forward window: {FORWARD_FROM} → {FORWARD_TO}')
fwd = run_window(FORWARD_FROM, FORWARD_TO, BEST_CONFIG)

# Print result + verdict
forward_df = pd.DataFrame([fwd])
forward_df = forward_df[['window', 'bars', 'trades', 'win_rate', 'expectancy_R', 'profit_factor']]
print()
print('=== Phase 8: forward result ===')
print(forward_df.to_string(index=False))
print()

pass_wr = fwd['win_rate'] >= 48.0
pass_pf = fwd['profit_factor'] > 1.0
pass_n  = fwd['trades'] >= 10
verdict = pass_wr and pass_pf and pass_n
print('=== FORWARD VERDICT ===')
print(f'  WR ≥ 48 %               : {pass_wr}  ({fwd["win_rate"]:.2f} %)')
print(f'  Profit factor > 1       : {pass_pf}  ({fwd["profit_factor"]:.2f})')
print(f'  Trades ≥ 10             : {pass_n}  ({fwd["trades"]})')
print(f'  → FORWARD {"PASS — ready for paper trading" if verdict else "FAIL — do not deploy"}')

forward_df.to_csv(RESULTS_DIR / 'phase8_forward.csv', index=False)
""")

# ─────────────────────────────────────────────────────────────────────────────
md(r"""## Step 9e — Real-Money Net P&L (Errante spread + 0.02 lot)

A high WR doesn't automatically mean real-money profit. This step models the
**actual cost of trading XAUUSD on Errante**:

- **Bid/Ask spread**: 0.25 USD per oz (round-trip cost on the price).
- **Lot size**: 0.02 standard lot (= 2 oz, since 1 lot = 100 oz on XAUUSD).
- **Fee per trade** = `spread × contract_size × lot` = `0.25 × 100 × 0.02 = 0.50 USD` round-trip.
- **Starting capital**: 1 000 USD (for the return-% calculation).

For each trade we report `gross_$ - fee_$ = net_$`. A trade is **net-positive** only
if the price move covers the spread; backtest WR can differ slightly from cost-aware
WR for trades that "win" by less than the spread.
""")

code(r"""# ── Cost model ───────────────────────────────────────────────────────────────
SPREAD_USD     = 0.25
CONTRACT_SIZE  = 100         # oz per standard lot (XAUUSD)
LOT_SIZE       = 0.02
START_CAPITAL  = 1000.0      # USD assumed for return-% calc
FEE_PER_TRADE  = SPREAD_USD * CONTRACT_SIZE * LOT_SIZE     # = 0.50 USD


def trades_to_pnl_df(trades_list, lot=LOT_SIZE, spread=SPREAD_USD):
    rows = []
    cumulative = 0.0
    for t in trades_list:
        gross_price = (t.exit - t.entry) * t.side
        gross_usd   = gross_price * CONTRACT_SIZE * lot
        fee_usd     = spread * CONTRACT_SIZE * lot
        net_usd     = gross_usd - fee_usd
        cumulative += net_usd
        rows.append({
            'entry_time': t.entry_time,
            'side'      : 'L' if t.side == 1 else 'S',
            'entry'     : round(t.entry, 2),
            'exit'      : round(t.exit, 2),
            'price_mv'  : round(gross_price, 2),
            'gross_$'   : round(gross_usd, 2),
            'fee_$'     : round(fee_usd, 2),
            'net_$'     : round(net_usd, 2),
            'cum_$'     : round(cumulative, 2),
            'reason'    : t.reason,
        })
    return pd.DataFrame(rows)


# Run all windows + collect trades
windows_to_eval = [
    ('OOS-1', '2024-11-01', '2025-02-01'),
    ('OOS-2', '2025-02-01', '2025-05-01'),
    ('OOS-3', '2025-05-01', '2025-08-01'),
    ('OOS-4', '2025-08-01', '2025-11-01'),
    ('IN'   , DATE_FROM,    DATE_TO),
    ('FWD'  , FORWARD_FROM, FORWARD_TO),
]

summary_rows = []
all_pnl: dict = {}

for label, df_, dt_ in windows_to_eval:
    s, trades_w = run_window(df_, dt_, BEST_CONFIG, return_trades=True)
    pnl_df = trades_to_pnl_df(trades_w)
    all_pnl[label] = pnl_df

    n          = len(pnl_df)
    wins_back  = int(s['wins'])                     # WR before fees
    wins_net   = int((pnl_df['net_$'] > 0).sum()) if n else 0
    gross_sum  = float(pnl_df['gross_$'].sum())     if n else 0.0
    fee_sum    = float(pnl_df['fee_$'].sum())       if n else 0.0
    net_sum    = float(pnl_df['net_$'].sum())       if n else 0.0
    summary_rows.append({
        'window'      : label,
        'period'      : f'{df_} → {dt_}',
        'trades'      : n,
        'wr_backtest' : round(s['win_rate'], 2),
        'wr_net'      : round(wins_net / n * 100, 2) if n else 0,
        'gross_$'     : round(gross_sum, 2),
        'fee_$'       : round(fee_sum,   2),
        'net_$'       : round(net_sum,   2),
        'return_%'    : round(net_sum / START_CAPITAL * 100, 2),
    })

summary = pd.DataFrame(summary_rows)

print(f'Cost model: spread = {SPREAD_USD} USD/oz,  lot = {LOT_SIZE},  contract = {CONTRACT_SIZE} oz/lot')
print(f'           → fee per round-trip trade = ${FEE_PER_TRADE:.2f}')
print(f'Starting capital = ${START_CAPITAL:.0f}')
print()
print('=== Per-window net P&L ===')
print(summary.to_string(index=False))

oos_net = summary[summary['window'].str.startswith('OOS')]['net_$'].sum()
oos_ret = oos_net / START_CAPITAL * 100
all_oos_trades = sum(summary[summary['window'].str.startswith('OOS')]['trades'])
print()
print(f'OOS total      (4 × 90-day, frozen config) : ${oos_net:>+8.2f}  → {oos_ret:+.2f}% on $1k')
print(f'In-sample total                            : ${summary.loc[summary["window"]=="IN","net_$"].iat[0]:>+8.2f}')
print(f'Forward (30d, completely unseen)           : ${summary.loc[summary["window"]=="FWD","net_$"].iat[0]:>+8.2f}')

summary.to_csv(RESULTS_DIR / 'phase9_net_pnl.csv', index=False)
""")

md(r"""### Per-trade detail (Forward window)

The forward window is the most honest test. Below is every individual trade with
`gross_$`, `fee_$`, `net_$`, and a running `cum_$` (running P&L on a 1 000 USD
account at 0.02 lot).
""")

code(r"""fwd_pnl = all_pnl.get('FWD')
if fwd_pnl is None or fwd_pnl.empty:
    print('No forward trades.')
else:
    print(fwd_pnl.to_string(index=False))
    fwd_pnl.to_csv(RESULTS_DIR / 'phase9_forward_trades.csv', index=False)
    print()
    print(f'Forward trades   : {len(fwd_pnl)}')
    print(f'Forward gross $  : ${fwd_pnl["gross_$"].sum():+.2f}')
    print(f'Forward fees $   : ${fwd_pnl["fee_$"].sum():.2f}')
    print(f'Forward net $    : ${fwd_pnl["net_$"].sum():+.2f}')
    print(f'Forward return % : {fwd_pnl["net_$"].sum()/START_CAPITAL*100:+.2f}% on $1k')
""")

md(r"""### Optimization scan — same WR, more return

If the strategy is already net-profitable, the natural next question is:
**how do we lift return % without breaking win-rate?**

Three changes that historically lift profit while keeping (or even improving) WR:

1. **Break-even stop at +1 R** — once a trade reaches +1 R, move SL to entry.
   Wins are unaffected (they still hit TP at +2 R), but trades that *would* have
   reversed back to SL now exit at flat → losses convert into break-evens.
   *Effect*: WR may go up; expectancy goes up; gross profit stays similar but
   gross loss shrinks → return % rises.
2. **Larger lot (linear scale)** — 0.05 lot, 0.10 lot. Pure scaling. Risk grows linearly.
3. **Partial close at +1 R, runner to +2 R** — locks 50 % at 1 R, lets the rest run to TP.
   Increases hit-rate of "at-least-break-even" trades but caps upside on full runners.

The cell below runs **option 1 (break-even stop)** on every window and reports the delta.
""")

code(r"""# ── Break-even stop variant: once trade reaches +1R, move SL to entry ───────
def backtest_with_be(df, sl_method='structural'):
    trades_ = []
    n = len(df)
    in_trade = False
    cur = None
    for i in range(n - 1):
        if in_trade:
            hi, lo = df['high'].iat[i], df['low'].iat[i]
            # Promote SL to entry if +1R reached on this bar
            r_unit = abs(cur.entry - cur.sl)
            if r_unit > 0 and not getattr(cur, 'be_armed', False):
                price_extreme = hi if cur.side == 1 else lo
                profit_now = (price_extreme - cur.entry) * cur.side
                if profit_now >= r_unit:                       # +1R or better
                    cur.sl = cur.entry                         # break-even
                    cur.be_armed = True
            hit_sl = (cur.side == 1 and lo <= cur.sl) or (cur.side == -1 and hi >= cur.sl)
            hit_tp = (cur.side == 1 and hi >= cur.tp) or (cur.side == -1 and lo <= cur.tp)
            exit_now = False; reason = ''; px = 0.0
            if hit_sl and hit_tp:
                exit_now, reason, px = True, 'sl', cur.sl
            elif hit_sl:
                exit_now, reason, px = True, 'sl', cur.sl
            elif hit_tp:
                exit_now, reason, px = True, 'tp', cur.tp
            elif i - cur.entry_idx >= MAX_HOLD_BARS:
                exit_now, reason, px = True, 'time', float(df['close'].iat[i])
            if exit_now:
                cur.exit_idx = i; cur.exit_time = df['time'].iat[i]
                cur.exit = px; cur.reason = reason
                r_unit2 = abs(cur.entry - (cur.sl if not getattr(cur, 'be_armed', False) else cur.entry - 0.0001 * cur.side))
                # For R reporting, use the original SL distance via re-derivation:
                if cur.reason == 'tp':   cur.r_multiple = RR
                elif cur.reason == 'sl' and getattr(cur, 'be_armed', False): cur.r_multiple = 0.0
                elif cur.reason == 'sl': cur.r_multiple = -1.0
                else:                    cur.r_multiple = ((cur.exit - cur.entry) * cur.side) / max(r_unit2, 1e-9)
                trades_.append(cur); in_trade = False; cur = None

        sig = int(df['signal'].iat[i])
        if (not in_trade or not ONE_TRADE_AT_A_TIME) and sig != 0:
            entry_idx = i + 1
            entry_px  = float(df['open'].iat[entry_idx])
            sl = structural_sl(df, entry_idx, sig) if sl_method == 'structural' else atr_sl(df, entry_idx, sig, entry_px)
            r = abs(entry_px - sl)
            if r <= 0 or r > 5 * df['atr'].iat[entry_idx]:
                continue
            tp = entry_px + RR * r * sig
            cur = Trade(side=sig, entry_idx=entry_idx,
                        entry_time=df['time'].iat[entry_idx],
                        entry=entry_px, sl=sl, tp=tp)
            in_trade = True
    return trades_


def run_window_be(date_from, date_to, cfg):
    # Same as run_window, but uses backtest_with_be.
    m5 = load_ohlcv(SYMBOL, 'M5', date_from, date_to)
    h1 = load_ohlcv(SYMBOL, 'H1', date_from, date_to)
    d1 = load_ohlcv(SYMBOL, 'D1', date_from, date_to)
    m5 = add_m5_features(m5)
    h1t = add_htf_trend(h1, EMA_TREND_H1)
    d1t = add_htf_trend(d1, EMA_TREND_D1)
    m5 = pd.merge_asof(m5.sort_values('time'),
                       h1t.rename(columns={'ema_trend':'h1_ema','trend_dir':'h1_trend'}),
                       on='time', direction='backward')
    m5 = pd.merge_asof(m5,
                       d1t.rename(columns={'ema_trend':'d1_ema','trend_dir':'d1_trend'}),
                       on='time', direction='backward')
    same = m5['h1_trend'] == m5['d1_trend']; nonzero = m5['h1_trend'] != 0
    m5['trend_dir'] = np.where(same & nonzero, m5['h1_trend'], 0).astype(int)
    sh, eh = cfg['session']
    ny_h = (m5['time'].dt.hour - BROKER_TO_NY_H) % 24
    m5['in_session'] = (ny_h >= sh) & (ny_h < eh)
    m5['f_bb']=f_bb_touch(m5); m5['f_ema']=f_ema_pullback(m5); m5['f_rsi']=f_rsi_exit(m5)
    m5['f_candle']=f_pin_engulf(m5)
    obs = detect_order_blocks(m5); m5['f_ob']=f_ob_retest(m5, obs)
    m5['f_rsiR'] = f_rsi_recent(m5, memory=cfg['rsi_memory'])
    rsi_long  = (m5['f_rsiR']== 1); rsi_short = (m5['f_rsiR']==-1)
    conf = m5[cfg['confirms']].values
    conf_long  = (conf== 1).any(axis=1); conf_short = (conf==-1).any(axis=1)
    can_long  = (m5['trend_dir']== 1) & m5['in_session'] & rsi_long  & conf_long
    can_short = (m5['trend_dir']==-1) & m5['in_session'] & rsi_short & conf_short
    m5['signal'] = np.where(can_long, 1, np.where(can_short, -1, 0))
    trades_ = backtest_with_be(m5, cfg['sl_method'])
    return trades_


rows = []
for label, df_, dt_ in windows_to_eval:
    tr_be = run_window_be(df_, dt_, BEST_CONFIG)
    pnl_be = trades_to_pnl_df(tr_be)
    n = len(pnl_be); wins = int((pnl_be['net_$']>0).sum())
    rows.append({
        'window'    : label,
        'trades'    : n,
        'wr_net_%'  : round(wins/n*100, 2) if n else 0,
        'net_$'     : round(pnl_be['net_$'].sum(), 2) if n else 0,
        'return_%'  : round(pnl_be['net_$'].sum()/START_CAPITAL*100, 2) if n else 0,
    })
be_summary = pd.DataFrame(rows)

# Compare to baseline (no BE)
compare = summary[['window','trades','wr_net','net_$','return_%']].copy()
compare.columns = ['window','tr_base','wr_base','net_base_$','ret_base_%']
compare = compare.merge(be_summary, on='window', how='left')
compare['delta_net_$']  = compare['net_$'] - compare['net_base_$']
compare['delta_ret_%']  = compare['return_%'] - compare['ret_base_%']
compare['delta_wr_pp']  = compare['wr_net_%'] - compare['wr_base']

print('=== Baseline  vs  Break-even-stop variant ===')
print(compare.to_string(index=False))

be_summary.to_csv(RESULTS_DIR / 'phase9_break_even_variant.csv', index=False)
""")

code(r"""fig = go.Figure()
fig.add_trace(go.Scatter(x=df_m5['time'], y=eq_struct, name='structural SL', line=dict(width=2)))
fig.add_trace(go.Scatter(x=df_m5['time'], y=eq_atr,    name='ATR SL',        line=dict(width=2)))
fig.update_layout(title='Equity (1% risk per trade)', height=350,
                  template='plotly_white', margin=dict(l=40, r=10, t=40, b=30))
fig.show()


def plot_window_with_trades(df: pd.DataFrame, trades: List[Trade], around_idx: int, span: int = 240):
    lo = max(0, around_idx - span // 2)
    hi = min(len(df), lo + span)
    seg = df.iloc[lo:hi]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                        vertical_spacing=0.04, subplot_titles=(SYMBOL, 'RSI'))
    fig.add_trace(go.Candlestick(x=seg['time'], open=seg['open'], high=seg['high'],
                                 low=seg['low'], close=seg['close'], name='M5'), row=1, col=1)
    fig.add_trace(go.Scatter(x=seg['time'], y=seg['ema20'], name='EMA20',
                             line=dict(color='orange', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=seg['time'], y=seg['bb_up'], name='BB up',
                             line=dict(color='gray', width=1, dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=seg['time'], y=seg['bb_lo'], name='BB lo',
                             line=dict(color='gray', width=1, dash='dot')), row=1, col=1)
    for t in trades:
        if t.entry_idx >= lo and t.entry_idx <= hi:
            color = 'green' if t.r_multiple > 0 else 'red'
            fig.add_shape(type='line', x0=t.entry_time, x1=t.exit_time or seg['time'].iat[-1],
                          y0=t.entry, y1=t.entry, line=dict(color=color, width=2), row=1, col=1)
            fig.add_shape(type='line', x0=t.entry_time, x1=t.exit_time or seg['time'].iat[-1],
                          y0=t.sl, y1=t.sl, line=dict(color='red', dash='dot', width=1), row=1, col=1)
            fig.add_shape(type='line', x0=t.entry_time, x1=t.exit_time or seg['time'].iat[-1],
                          y0=t.tp, y1=t.tp, line=dict(color='green', dash='dot', width=1), row=1, col=1)
    fig.add_trace(go.Scatter(x=seg['time'], y=seg['rsi'], name='RSI14',
                             line=dict(color='purple', width=1)), row=2, col=1)
    fig.add_hline(y=RSI_OS, line=dict(color='blue', dash='dot', width=1), row=2, col=1)
    fig.add_hline(y=RSI_OB, line=dict(color='blue', dash='dot', width=1), row=2, col=1)
    fig.update_layout(height=700, xaxis_rangeslider_visible=False,
                      template='plotly_white', margin=dict(l=40, r=10, t=40, b=30))
    return fig


if trades_struct:
    sample = trades_struct[len(trades_struct) // 2]
    plot_window_with_trades(df_m5, trades_struct, sample.entry_idx).show()
else:
    print('No structural-SL trades to plot.')
""")

# ─────────────────────────────────────────────────────────────────────────────
md(r"""## Step 11 — Save Trades & Stats""")

code(r"""def trades_to_df(trades: List[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame([{
        'entry_time': t.entry_time, 'exit_time': t.exit_time,
        'side': t.side, 'entry': t.entry, 'sl': t.sl, 'tp': t.tp,
        'exit': t.exit, 'reason': t.reason, 'R': t.r_multiple,
    } for t in trades])


trades_to_df(trades_struct).to_csv(RESULTS_DIR / 'trades_structural.csv', index=False)
trades_to_df(trades_atr).to_csv   (RESULTS_DIR / 'trades_atr.csv',        index=False)
comparison.to_csv(RESULTS_DIR / 'stats_comparison.csv')
abl.to_csv(RESULTS_DIR / 'ablation_keep_one.csv')
drop_abl.to_csv(RESULTS_DIR / 'ablation_drop_one.csv')
print('Saved →', RESULTS_DIR)
""")

# ─────────────────────────────────────────────────────────────────────────────
md(r"""## Step 12 — Roadmap

Use this notebook as the **harness**, not the answer. The win-rate / trade-count
combo depends entirely on which knobs we turn. The plan below is ordered so each
phase isolates one variable and tells us *why* the number moved.

### Phase 0 — Smoke (this run)
- Verify the pipeline executes end-to-end on the chosen window.
- Read the **`comparison` table**: structural vs ATR SL. Pick the one with higher
  *expectancy* (not just win-rate — a tight SL inflates win-rate but shrinks edge).

### Phase 1 — Establish a baseline
- Set `MIN_REACTIONS = 1` → maximises trade count, reveals raw per-filter quality.
- Read **`abl` (keep-one-filter)**: any filter with WR ≥ 45% on its own is gold.
- Read **`drop_abl`**: any filter whose removal *improves* expectancy is noise — drop it.

### Phase 2 — Tune the score
- Loop `MIN_REACTIONS ∈ {1, 2, 3}`. Plot `(win_rate, trades, expectancy)` for each.
- Choose the smallest score that lifts WR above 45% while keeping ≥ 1 trade / day.

### Phase 3 — Reaction parameters (one at a time)
- `RSI_OS / RSI_OB`: 30/70 → 35/65 → 40/60. Stricter = fewer, cleaner.
- `BB_STD`: 1.5 / 2.0 / 2.5. Wider BBs = rarer touches but stronger reversion.
- `PIN_BAR_WICK_RATIO`: 0.5 → 0.6 → 0.7.
- `OB_DISPLACEMENT_ATR`: 1.5 → 2.0 → 2.5.
- After each, log WR / trades / expectancy in a small CSV.

### Phase 4 — Trend gates
- Try `h1_trend` only (drop D1) — XAUUSD D1 trend can be too lagging on M5.
- Try `EMA_TREND_H1 ∈ {21, 50, 100}`. Faster H1 = more whipsaw but more trades.

### Phase 5 — Session windows
- Restrict to **NY 08–12** (London-NY overlap, peak liquidity).
- Compare against 07–16. Smaller, denser session usually wins on WR.

### Phase 6 — SL/TP geometry
- Pick the SL method that survived Phase 0 and stress-test it:
  - `STRUCT_LOOKBACK_BARS ∈ {6, 12, 20}` for structural
  - `ATR_SL_MULT ∈ {0.75, 1.0, 1.5}` for ATR
- Try moving to break-even at +1R (paste the logic into `backtest()` once we agree).

### Phase 7 — Robustness
- Re-run with `MIN_REACTIONS` & best parameters across multiple 90-day windows
  (e.g. four non-overlapping quarters). WR variance across windows is the real signal.
- Then port to a second symbol (EURUSD M5) without changing any knobs — if it survives,
  the edge is structural; if it collapses, it was curve-fit.

### Phase 8 — Live forward
- Once Phase 7 numbers hold, run forward on the last 30 unseen days
  (`DATE_FROM = today - 30d`) without re-tuning. That's the honest WR.

---

**Stop condition**: WR ≥ 45% **and** ≥ 1 trade/day **and** standard deviation of WR
across the four Phase-7 windows ≤ 5pp. Anything looser is curve-fitting noise.
""")

# ─────────────────────────────────────────────────────────────────────────────
# Assemble notebook
nb = {
    "cells": [
        ({
            "cell_type": "markdown",
            "metadata": {},
            "source": [src]
        } if ctype == "markdown" else {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [src]
        })
        for ctype, src in CELLS
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT = Path(__file__).parent / "23_multi_strategy_scalper.ipynb"
OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {OUT}  ({len(CELLS)} cells)")
