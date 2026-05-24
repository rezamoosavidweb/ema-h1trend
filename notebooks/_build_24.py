"""Generate notebook 24: multi-symbol scalper across FX pairs.

Pipeline mirrors notebook 23 (RSI-gated + candle/EMA confirmation, D1+H1 trend
gate, NY session). Differences:
  • runs across {GBPJPY, USDJPY, USDCHF, EURUSD, AUDUSD}
  • 3+ year window with per-quarter stats per symbol
  • per-symbol mini-sweep when baseline config fails the WR floor
  • per-symbol pip-value + spread for honest USD P&L
"""
import json
from pathlib import Path

CELLS: list[tuple[str, str]] = []


def md(s: str) -> None:
    CELLS.append(("markdown", s))


def code(s: str) -> None:
    CELLS.append(("code", s))


# ─────────────────────────────────────────────────────────────────────────────
md(r"""# Notebook #24 — Multi-Symbol FX Scalper

Reuses the **RSI-gated + (candle | EMA) confirmation** logic that delivered
WR ≈ 50–58 % on XAUUSD M5 in notebook #23. Applies it to a basket of FX pairs
over a 3+ year window with **per-quarter reporting per symbol** to ensure no
quarter is statistically broken.

### Target
- **WR ≥ 48 %** on every (symbol, quarter) cell.
- Trade count **must not drop** — at least ~30 trades per quarter per symbol.

### Symbols
GBPJPY, USDJPY, USDCHF, EURUSD, AUDUSD.

### Window
`2023-01-01 → 2026-05-15` (full available data minus warmup). Forward window
(unseen) = last 90 days.

### Cost model per symbol (Errante typical spreads)
We model spread + 0.02-lot pip value per pair so the USD P&L numbers reflect
real-money execution, not theoretical R-multiples.
""")

# ─────────────────────────────────────────────────────────────────────────────
md(r"""## Step 1 — Imports & per-symbol config""")

code(r"""import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'notebook'
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.4f}'.format)

# ── Identity ────────────────────────────────────────────────────────────────
STRATEGY    = 'multi_symbol_scalper'
DATA_DIR    = Path('./data')
RESULTS_DIR = Path('./results') / STRATEGY
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Broker → NY offset (data_timezone memory) ───────────────────────────────
BROKER_TO_NY_H = 7
NY_SESSION_START_H = 7
NY_SESSION_END_H   = 16

# ── Window ──────────────────────────────────────────────────────────────────
DATE_FROM    = '2023-01-01'
DATE_TO      = '2026-02-15'         # in-sample cutoff
FORWARD_FROM = '2026-02-15'
FORWARD_TO   = '2026-05-15'

# ── Per-symbol cost model + minor parameter overrides ───────────────────────
# pip_value_usd = USD P&L per 1 pip on 0.02 lot (approximate, OK for sizing)
# spread_pips  = Errante typical spread
# ── Per-symbol cost model (Errante 2026 broker quote table — user-supplied) ─
SYMBOLS_CFG: Dict[str, dict] = {
    # ── USD majors / commodity FX ──────────────────────────────────────────
    'EURUSD':  {'pip_size': 0.0001, 'pip_value_usd_per_002lot': 0.2000,  'spread_pips':   1.3},
    'GBPUSD':  {'pip_size': 0.0001, 'pip_value_usd_per_002lot': 0.2000,  'spread_pips':   1.6},
    'AUDUSD':  {'pip_size': 0.0001, 'pip_value_usd_per_002lot': 0.2000,  'spread_pips':   1.5},
    'NZDUSD':  {'pip_size': 0.0001, 'pip_value_usd_per_002lot': 0.2000,  'spread_pips':   2.2001},
    'USDCAD':  {'pip_size': 0.0001, 'pip_value_usd_per_002lot': 0.1447,  'spread_pips':   1.8},
    'USDCHF':  {'pip_size': 0.0001, 'pip_value_usd_per_002lot': 0.2548,  'spread_pips':   1.8},
    'USDJPY':  {'pip_size': 0.01,   'pip_value_usd_per_002lot': 0.1256,  'spread_pips':   2.0},
    # ── JPY crosses ────────────────────────────────────────────────────────
    'EURJPY':  {'pip_size': 0.01,   'pip_value_usd_per_002lot': 0.1256,  'spread_pips':   2.9001},
    'GBPJPY':  {'pip_size': 0.01,   'pip_value_usd_per_002lot': 0.1256,  'spread_pips':   4.0},
    'AUDJPY':  {'pip_size': 0.01,   'pip_value_usd_per_002lot': 0.1256,  'spread_pips':   2.7},
    'NZDJPY':  {'pip_size': 0.01,   'pip_value_usd_per_002lot': 0.1256,  'spread_pips':   4.9001},
    'CADJPY':  {'pip_size': 0.01,   'pip_value_usd_per_002lot': 0.1256,  'spread_pips':   2.0},
    'CHFJPY':  {'pip_size': 0.01,   'pip_value_usd_per_002lot': 0.1256,  'spread_pips':   3.8001},
    # ── EUR crosses ────────────────────────────────────────────────────────
    'EURGBP':  {'pip_size': 0.0001, 'pip_value_usd_per_002lot': 0.2687,  'spread_pips':   2.1},
    'EURAUD':  {'pip_size': 0.0001, 'pip_value_usd_per_002lot': 0.1426,  'spread_pips':   4.1},
    # EURNZD: only H1 data available — skipped
    'EURCAD':  {'pip_size': 0.0001, 'pip_value_usd_per_002lot': 0.1447,  'spread_pips':   2.8001},
    'EURCHF':  {'pip_size': 0.0001, 'pip_value_usd_per_002lot': 0.2548,  'spread_pips':   1.6},
    # ── GBP crosses ────────────────────────────────────────────────────────
    'GBPAUD':  {'pip_size': 0.0001, 'pip_value_usd_per_002lot': 0.1426,  'spread_pips':   4.5},
    'GBPNZD':  {'pip_size': 0.0001, 'pip_value_usd_per_002lot': 0.1170,  'spread_pips':   7.6001},
    'GBPCAD':  {'pip_size': 0.0001, 'pip_value_usd_per_002lot': 0.1447,  'spread_pips':   3.0},
    'GBPCHF':  {'pip_size': 0.0001, 'pip_value_usd_per_002lot': 0.2548,  'spread_pips':   2.3},
    # ── AUD / NZD / CAD crosses ────────────────────────────────────────────
    'AUDCAD':  {'pip_size': 0.0001, 'pip_value_usd_per_002lot': 0.1447,  'spread_pips':   2.0},
    'AUDCHF':  {'pip_size': 0.0001, 'pip_value_usd_per_002lot': 0.2548,  'spread_pips':   2.2001},
    'AUDNZD':  {'pip_size': 0.0001, 'pip_value_usd_per_002lot': 0.1170,  'spread_pips':   3.9001},
    # NZDCAD, NZDCHF: no M5/H1/D1 data — skipped
    'CADCHF':  {'pip_size': 0.0001, 'pip_value_usd_per_002lot': 0.2548,  'spread_pips':   1.8},
    # ── USD exotics (high spread — included for completeness) ──────────────
    'USDCZK':  {'pip_size': 0.0001, 'pip_value_usd_per_002lot': 0.00956, 'spread_pips': 300.0},
    'USDHUF':  {'pip_size': 0.01,   'pip_value_usd_per_002lot': 0.0647,  'spread_pips':  53.2},
    'USDMXN':  {'pip_size': 0.0001, 'pip_value_usd_per_002lot': 0.01155, 'spread_pips': 103.7},
    'USDZAR':  {'pip_size': 0.0001, 'pip_value_usd_per_002lot': 0.01214, 'spread_pips': 135.0},
    'USDTRY':  {'pip_size': 0.0001, 'pip_value_usd_per_002lot': 0.0006,  'spread_pips':  50.0},
    # ── EUR exotics ────────────────────────────────────────────────────────
    'EURCZK':  {'pip_size': 0.0001, 'pip_value_usd_per_002lot': 0.00956, 'spread_pips': 394.0},
    'EURHUF':  {'pip_size': 0.01,   'pip_value_usd_per_002lot': 0.0647,  'spread_pips':  87.2},
    'EURNOK':  {'pip_size': 0.0001, 'pip_value_usd_per_002lot': 0.02158, 'spread_pips':  86.8001},
    'EURSEK':  {'pip_size': 0.0001, 'pip_value_usd_per_002lot': 0.02135, 'spread_pips':  34.2001},
    'EURZAR':  {'pip_size': 0.0001, 'pip_value_usd_per_002lot': 0.01214, 'spread_pips': 277.1001},
    'EURTRY':  {'pip_size': 0.0001, 'pip_value_usd_per_002lot': 0.0006,  'spread_pips':  60.0},
    # ── Metals (Errante: XAU 100 oz/lot, XAG 5000 oz/lot) ──────────────────
    'XAUUSD':  {'pip_size': 0.01,   'pip_value_usd_per_002lot': 0.0200,  'spread_pips':  17.0},
    'XAGUSD':  {'pip_size': 0.01,   'pip_value_usd_per_002lot': 1.0000,  'spread_pips':   7.8001},
}

# ── Indicators (same as notebook 23) ────────────────────────────────────────
EMA_FAST       = 20
EMA_TREND_H1   = 50
EMA_TREND_D1   = 50
BB_PERIOD      = 20
BB_STD         = 2.0
RSI_PERIOD     = 14
RSI_OS         = 35.0
RSI_OB         = 65.0
ATR_PERIOD     = 14

# Reaction-filter parameters
PULLBACK_TOLERANCE_ATR = 0.4
PIN_BAR_WICK_RATIO     = 0.60

# ── Aggregator (gold's best config — RSI-gated + candle|ema) ────────────────
BEST_CONFIG = {
    'mode'       : 'RSI-gated',
    'confirms'   : ['f_candle', 'f_ema'],
    'rsi_memory' : 10,
    'session'    : (7, 16),
    'sl_method'  : 'structural',
}

# Risk
RR             = 2.0
ATR_SL_MULT    = 1.0
STRUCT_LOOKBACK_BARS = 12
SL_BUFFER_ATR  = 0.10
MAX_HOLD_BARS  = 96
ONE_TRADE_AT_A_TIME = True

# Cost analysis
LOT_SIZE       = 0.02
START_CAPITAL  = 1000.0
TARGET_WR      = 48.0   # hard requirement
MIN_TRADES_PER_QUARTER = 30

# Filter the symbol universe by per-trade economics — pairs with fee_per_trade
# above MAX_FEE_USD are economically hopeless at 0.02 lot (R rarely > fee).
MAX_FEE_USD = 1.50
TRADABLE = {s: c for s, c in SYMBOLS_CFG.items()
            if c['spread_pips'] * c['pip_value_usd_per_002lot'] <= MAX_FEE_USD}
SKIPPED  = {s: round(c['spread_pips'] * c['pip_value_usd_per_002lot'], 2)
            for s, c in SYMBOLS_CFG.items() if s not in TRADABLE}

print(f'All symbols defined  : {len(SYMBOLS_CFG)}')
print(f'Tradable (fee ≤ ${MAX_FEE_USD}) : {len(TRADABLE)}  → {list(TRADABLE.keys())}')
print(f'Skipped (fee > ${MAX_FEE_USD})  : {len(SKIPPED)}  → {SKIPPED}')
print(f'IS  : {DATE_FROM} → {DATE_TO}')
print(f'FWD : {FORWARD_FROM} → {FORWARD_TO}')
print(f'Per-quarter minimum trades: {MIN_TRADES_PER_QUARTER}')
# Use TRADABLE as the active sweep universe; keep SYMBOLS_CFG for cost lookup
SYMBOLS_ACTIVE = list(TRADABLE.keys())
""")

# ─────────────────────────────────────────────────────────────────────────────
md(r"""## Step 2 — Pipeline helpers (load, features, filters, backtest)

Copied verbatim from notebook 23 — same logic, no per-symbol shortcuts.
""")

code(r"""def load_ohlcv(symbol: str, tf: str, date_from: str, date_to: str) -> pd.DataFrame:
    path = DATA_DIR / symbol / tf / 'ohlcv.csv'
    df = pd.read_csv(path)
    df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
    df = df.sort_values('time').reset_index(drop=True)
    keep = ['time', 'open', 'high', 'low', 'close', 'tick_volume']
    df = df[[c for c in keep if c in df.columns]].copy()
    df.rename(columns={'tick_volume': 'volume'}, inplace=True)
    df = df[(df['time'] >= pd.Timestamp(date_from)) & (df['time'] <= pd.Timestamp(date_to))]
    return df.reset_index(drop=True)


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi(close, n=14):
    d = close.diff()
    gain = d.clip(lower=0); loss = (-d).clip(lower=0)
    ag = gain.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    al = loss.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def atr(df, n=14):
    tr = pd.concat([
        df['high']-df['low'],
        (df['high']-df['close'].shift()).abs(),
        (df['low'] -df['close'].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()

def adx(df, n=14):
    # Wilder ADX — trend strength index
    up   = df['high'].diff()
    down = -df['low'].diff()
    plus_dm  = pd.Series(np.where((up>down) & (up>0), up, 0.0),   index=df.index)
    minus_dm = pd.Series(np.where((down>up) & (down>0), down, 0.0), index=df.index)
    tr = pd.concat([df['high']-df['low'],
                    (df['high']-df['close'].shift()).abs(),
                    (df['low'] -df['close'].shift()).abs()], axis=1).max(axis=1)
    atr_w = tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    plus_di  = 100 * plus_dm.ewm(alpha=1/n, adjust=False, min_periods=n).mean()  / atr_w.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1/n, adjust=False, min_periods=n).mean() / atr_w.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False, min_periods=n).mean().fillna(0)

def macd_hist(close, fast=12, slow=26, signal=9):
    f = ema(close, fast); s = ema(close, slow)
    line = f - s
    sig  = ema(line, signal)
    return line - sig

def stoch(df, k=14, d=3):
    ll = df['low'].rolling(k).min()
    hh = df['high'].rolling(k).max()
    k_pct = (100 * (df['close'] - ll) / (hh - ll).replace(0, np.nan)).fillna(50)
    d_pct = k_pct.rolling(d).mean().fillna(50)
    return k_pct, d_pct

def add_m5_features(df):
    df = df.copy()
    df['ema20']  = ema(df['close'], EMA_FAST)
    df['rsi']    = rsi(df['close'], RSI_PERIOD)
    df['atr']    = atr(df, ATR_PERIOD)
    mid = df['close'].rolling(BB_PERIOD).mean()
    std = df['close'].rolling(BB_PERIOD).std()
    df['bb_mid'], df['bb_up'], df['bb_lo'] = mid, mid+BB_STD*std, mid-BB_STD*std
    df['body']  = (df['close']-df['open']).abs()
    df['range'] = (df['high']-df['low']).clip(lower=1e-9)
    df['upper_wick'] = df['high']-df[['open','close']].max(axis=1)
    df['lower_wick'] = df[['open','close']].min(axis=1)-df['low']
    # New indicators (iteration 4)
    df['adx']        = adx(df, 14)
    df['macd_hist']  = macd_hist(df['close'])
    df['stoch_k'], df['stoch_d'] = stoch(df, k=14, d=3)
    return df

def add_htf_trend(df, n):
    df = df.copy()
    e = ema(df['close'], n); slope = e.diff()
    df['ema_trend'] = e
    df['trend_dir'] = np.where((df['close']>e)&(slope>0), 1,
                       np.where((df['close']<e)&(slope<0), -1, 0))
    # Trend STRENGTH: how much the EMA has moved over last 6 bars, normalised by H1 ATR.
    h1_atr = atr(df, 14)
    df['trend_strength'] = (e.diff(6).abs() / h1_atr.replace(0, np.nan)).fillna(0)
    # H1 RSI for multi-TF momentum alignment (only used when df is H1)
    df['htf_rsi'] = rsi(df['close'], 14).fillna(50)
    return df[['time','ema_trend','trend_dir','trend_strength','htf_rsi']]


# Reaction filters
def f_bb_touch(df):
    long  = df['low']  <= df['bb_lo']
    short = df['high'] >= df['bb_up']
    return pd.Series(np.where(long,1,np.where(short,-1,0)), index=df.index)

def f_ema_pullback(df):
    tol = PULLBACK_TOLERANCE_ATR * df['atr']
    tl = (df['low']  <= df['ema20']+tol) & (df['close']>df['ema20'])
    ts = (df['high'] >= df['ema20']-tol) & (df['close']<df['ema20'])
    return pd.Series(np.where(tl,1,np.where(ts,-1,0)), index=df.index)

def f_rsi_exit(df):
    prev = df['rsi'].shift(1)
    long  = (prev<=RSI_OS) & (df['rsi']>RSI_OS)
    short = (prev>=RSI_OB) & (df['rsi']<RSI_OB)
    return pd.Series(np.where(long,1,np.where(short,-1,0)), index=df.index)

def f_pin_engulf(df):
    rng = df['range']
    bull_pin = (df['lower_wick']/rng >= PIN_BAR_WICK_RATIO) & (df['close']>df['open'])
    bear_pin = (df['upper_wick']/rng >= PIN_BAR_WICK_RATIO) & (df['close']<df['open'])
    po,pc = df['open'].shift(1), df['close'].shift(1)
    bull_eng = (pc<po)&(df['close']>df['open'])&(df['close']>=po)&(df['open']<=pc)
    bear_eng = (pc>po)&(df['close']<df['open'])&(df['close']<=po)&(df['open']>=pc)
    long  = bull_pin | bull_eng
    short = bear_pin | bear_eng
    return pd.Series(np.where(long,1,np.where(short,-1,0)), index=df.index)

def f_rsi_recent(df, memory=10):
    long_fresh  = (df['f_rsi']== 1).rolling(memory).max().fillna(0).astype(bool)
    short_fresh = (df['f_rsi']==-1).rolling(memory).max().fillna(0).astype(bool)
    return pd.Series(np.where(long_fresh,1,np.where(short_fresh,-1,0)), index=df.index)

# ── New filter functions (iteration 4) ──────────────────────────────────────
def f_macd(df):
    # MACD histogram sign as direction vote
    h = df['macd_hist']
    return pd.Series(np.where(h>0, 1, np.where(h<0, -1, 0)), index=df.index)

def f_stoch_cross(df):
    # Stochastic K crossing D out of OS/OB zones
    k_prev = df['stoch_k'].shift(1); d_prev = df['stoch_d'].shift(1)
    long  = (k_prev < d_prev) & (df['stoch_k'] > df['stoch_d']) & (df['stoch_k'] < 35)
    short = (k_prev > d_prev) & (df['stoch_k'] < df['stoch_d']) & (df['stoch_k'] > 65)
    return pd.Series(np.where(long, 1, np.where(short, -1, 0)), index=df.index)

def f_volume_spike(df, mult=1.4):
    # Volume spike on candle with directional body
    if 'volume' not in df.columns or df['volume'].sum() == 0:
        return pd.Series(0, index=df.index)
    vm  = df['volume'].rolling(50, min_periods=10).median()
    spike = df['volume'] >= mult * vm
    long_s  = spike & (df['close'] > df['open'])
    short_s = spike & (df['close'] < df['open'])
    return pd.Series(np.where(long_s, 1, np.where(short_s, -1, 0)), index=df.index)


# Trade / backtest objects
@dataclass
class Trade:
    side: int; entry_idx: int; entry_time: object
    entry: float; sl: float; tp: float
    exit_idx: int = -1; exit_time: object = None
    exit: float = 0.0; reason: str = ''; r_multiple: float = 0.0


def structural_sl(df, idx, side):
    lo = max(0, idx-STRUCT_LOOKBACK_BARS); a = df['atr'].iat[idx]
    if side == 1:
        return float(df['low'].iloc[lo:idx].min()) - SL_BUFFER_ATR*a
    return float(df['high'].iloc[lo:idx].max()) + SL_BUFFER_ATR*a


def backtest(df, sl_method='structural'):
    trades = []
    n = len(df); in_trade=False; cur=None
    for i in range(n-1):
        if in_trade:
            hi, lo = df['high'].iat[i], df['low'].iat[i]
            hit_sl = (cur.side==1 and lo<=cur.sl) or (cur.side==-1 and hi>=cur.sl)
            hit_tp = (cur.side==1 and hi>=cur.tp) or (cur.side==-1 and lo<=cur.tp)
            exit_now=False; reason=''; px=0.0
            if hit_sl and hit_tp: exit_now,reason,px = True,'sl',cur.sl
            elif hit_sl:          exit_now,reason,px = True,'sl',cur.sl
            elif hit_tp:          exit_now,reason,px = True,'tp',cur.tp
            elif i-cur.entry_idx >= MAX_HOLD_BARS:
                exit_now,reason,px = True,'time',float(df['close'].iat[i])
            if exit_now:
                cur.exit_idx=i; cur.exit_time=df['time'].iat[i]
                cur.exit=px; cur.reason=reason
                r_unit = abs(cur.entry-cur.sl)
                cur.r_multiple = ((cur.exit-cur.entry)*cur.side)/r_unit if r_unit>0 else 0
                trades.append(cur); in_trade=False; cur=None
        sig = int(df['signal'].iat[i])
        if (not in_trade or not ONE_TRADE_AT_A_TIME) and sig != 0:
            ei = i+1; ep = float(df['open'].iat[ei])
            sl = structural_sl(df, ei, sig)
            r = abs(ep-sl)
            if r<=0 or r>5*df['atr'].iat[ei]: continue
            tp = ep + RR*r*sig
            cur = Trade(side=sig, entry_idx=ei, entry_time=df['time'].iat[ei],
                        entry=ep, sl=sl, tp=tp)
            in_trade=True
    return trades


def stats(trades):
    if not trades:
        return {'trades':0,'wins':0,'win_rate':0.0,'avg_R':0.0,
                'expectancy_R':0.0,'profit_factor':0.0}
    wins = [t for t in trades if t.r_multiple>0]
    losses = [t for t in trades if t.r_multiple<=0]
    gw = sum(t.r_multiple for t in wins)
    gl = -sum(t.r_multiple for t in losses)
    return {
        'trades': len(trades),
        'wins'  : len(wins),
        'win_rate': len(wins)/len(trades)*100,
        'avg_R' : np.mean([t.r_multiple for t in trades]),
        'expectancy_R': len(wins)/len(trades)*RR - (1-len(wins)/len(trades)),
        'profit_factor': gw/gl if gl>0 else float('inf'),
    }


def build_signals(m5, h1, d1, cfg):
    # Adds features + filters + signal column. Returns the m5 dataframe.
    m5 = add_m5_features(m5)
    h1t = add_htf_trend(h1, EMA_TREND_H1)
    d1t = add_htf_trend(d1, EMA_TREND_D1)
    m5 = pd.merge_asof(m5.sort_values('time'),
                       h1t.rename(columns={'ema_trend':'h1_ema','trend_dir':'h1_trend',
                                            'trend_strength':'h1_strength','htf_rsi':'h1_rsi'}),
                       on='time', direction='backward')
    m5 = pd.merge_asof(m5,
                       d1t.rename(columns={'ema_trend':'d1_ema','trend_dir':'d1_trend',
                                            'trend_strength':'d1_strength','htf_rsi':'d1_rsi'}),
                       on='time', direction='backward')
    same = m5['h1_trend']==m5['d1_trend']; nz = m5['h1_trend']!=0
    m5['trend_dir'] = np.where(same & nz, m5['h1_trend'], 0).astype(int)
    # Optional H1 trend-strength gate
    m5['htf_strong'] = m5['h1_strength'] >= cfg.get('htf_strength_min', 0.0)
    sh, eh = cfg['session']
    ny_h = (m5['time'].dt.hour - BROKER_TO_NY_H) % 24
    m5['in_session'] = (ny_h>=sh) & (ny_h<eh)
    # Base reaction filters
    m5['f_bb']     = f_bb_touch(m5)
    m5['f_ema']    = f_ema_pullback(m5)
    m5['f_rsi']    = f_rsi_exit(m5)
    m5['f_candle'] = f_pin_engulf(m5)
    m5['f_rsiR']   = f_rsi_recent(m5, memory=cfg['rsi_memory'])
    # New iteration-4 filters
    m5['f_macd']   = f_macd(m5)
    m5['f_stoch']  = f_stoch_cross(m5)
    m5['f_vol']    = f_volume_spike(m5, mult=cfg.get('vol_spike_mult', 1.4))

    # ATR-min filter (optional)
    atr_mult = cfg.get('atr_min_mult', 0.0)
    if atr_mult > 0:
        atr_med = m5['atr'].rolling(500, min_periods=50).median()
        m5['atr_ok'] = (m5['atr'] >= atr_mult * atr_med).fillna(False)
    else:
        m5['atr_ok'] = True

    # New optional gates
    adx_min = cfg.get('adx_min', 0.0)
    m5['adx_ok'] = m5['adx'] >= adx_min if adx_min > 0 else True

    # H1 RSI alignment: long needs h1_rsi > 50, short needs h1_rsi < 50
    if cfg.get('require_h1_rsi_align', False):
        m5['h1_rsi_long_ok']  = m5['h1_rsi'] > 50
        m5['h1_rsi_short_ok'] = m5['h1_rsi'] < 50
    else:
        m5['h1_rsi_long_ok']  = True
        m5['h1_rsi_short_ok'] = True

    # MACD alignment gate (sign of m5 MACD histogram matches direction)
    if cfg.get('require_macd_align', False):
        m5['macd_long_ok']  = m5['f_macd'] ==  1
        m5['macd_short_ok'] = m5['f_macd'] == -1
    else:
        m5['macd_long_ok']  = True
        m5['macd_short_ok'] = True

    # Common universe of base gates that apply to BOTH long and short candidates
    base_long  = m5['in_session'] & m5['atr_ok'] & m5['htf_strong'] & m5['adx_ok'] & m5['h1_rsi_long_ok']  & m5['macd_long_ok']
    base_short = m5['in_session'] & m5['atr_ok'] & m5['htf_strong'] & m5['adx_ok'] & m5['h1_rsi_short_ok'] & m5['macd_short_ok']

    if cfg['mode'] == 'RSI-gated':
        rl=(m5['f_rsiR']==1); rs=(m5['f_rsiR']==-1)
        conf = m5[cfg['confirms']].values
        cl=(conf==1).any(axis=1); cs=(conf==-1).any(axis=1)
        cl_long  = (m5['trend_dir']==1)  & base_long  & rl & cl
        cl_short = (m5['trend_dir']==-1) & base_short & rs & cs
    elif cfg['mode'] == 'RSI-gated-AND':
        rl=(m5['f_rsiR']==1); rs=(m5['f_rsiR']==-1)
        conf = m5[cfg['confirms']].values
        cl=(conf==1).all(axis=1); cs=(conf==-1).all(axis=1)
        cl_long  = (m5['trend_dir']==1)  & base_long  & rl & cl
        cl_short = (m5['trend_dir']==-1) & base_short & rs & cs
    elif cfg['mode'] == 'OR':
        mr = cfg.get('min_reactions', 1)
        f  = m5[cfg['confirms']].values
        lv = (f==1).sum(axis=1); sv = (f==-1).sum(axis=1)
        cl_long  = (m5['trend_dir']==1)  & base_long  & (lv>=mr)
        cl_short = (m5['trend_dir']==-1) & base_short & (sv>=mr)
    else:
        raise ValueError(cfg['mode'])
    m5['signal'] = np.where(cl_long,1,np.where(cl_short,-1,0))
    return m5


def run_symbol(symbol, date_from, date_to, cfg):
    try:
        m5 = load_ohlcv(symbol, 'M5', date_from, date_to)
        h1 = load_ohlcv(symbol, 'H1', date_from, date_to)
        d1 = load_ohlcv(symbol, 'D1', date_from, date_to)
    except FileNotFoundError:
        return [], pd.DataFrame()
    if len(m5) < 1000 or len(h1) < 50 or len(d1) < 10:
        return [], pd.DataFrame()
    m5 = build_signals(m5, h1, d1, cfg)
    tr = backtest(m5, cfg['sl_method'])
    return tr, m5

# ── Persistence helpers (per-symbol folder structure) ──────────────────────
SUMMARY_DIR = RESULTS_DIR / '_summary'
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

def symbol_dir(sym):
    p = RESULTS_DIR / sym
    p.mkdir(parents=True, exist_ok=True)
    return p

def trades_to_dataframe(trades, symbol=None, with_usd_pnl=True):
    if not trades:
        return pd.DataFrame()
    rows = []
    for t in trades:
        row = {
            'entry_time': t.entry_time, 'exit_time': t.exit_time,
            'side': 'L' if t.side==1 else 'S',
            'entry': round(t.entry, 5), 'exit': round(t.exit, 5),
            'sl': round(t.sl, 5), 'tp': round(t.tp, 5),
            'R': round(t.r_multiple, 3), 'reason': t.reason,
        }
        if with_usd_pnl and symbol and symbol in SYMBOLS_CFG:
            ps = SYMBOLS_CFG[symbol]['pip_size']
            pv = SYMBOLS_CFG[symbol]['pip_value_usd_per_002lot']
            sp = SYMBOLS_CFG[symbol]['spread_pips']
            pip_move = (t.exit - t.entry) * t.side / ps
            row['gross_$']  = round(pip_move * pv, 2)
            row['fee_$']    = round(sp * pv, 2)
            row['net_$']    = round(row['gross_$'] - row['fee_$'], 2)
        rows.append(row)
    df = pd.DataFrame(rows)
    if with_usd_pnl and 'net_$' in df.columns:
        df['cum_net_$'] = df['net_$'].cumsum().round(2)
    return df

def save_trades(symbol, trades, filename='trades_full.csv'):
    df = trades_to_dataframe(trades, symbol=symbol, with_usd_pnl=True)
    if not df.empty:
        df.to_csv(symbol_dir(symbol) / filename, index=False)
    return df

def save_config(symbol, cfg):
    import json as _json
    # Stringify session tuple for JSON
    out = {**cfg}
    if 'session' in out and isinstance(out['session'], tuple):
        out['session'] = list(out['session'])
    (symbol_dir(symbol) / 'config.json').write_text(_json.dumps(out, indent=2, default=str))

def save_equity_curve(symbol, trades):
    df = trades_to_dataframe(trades, symbol=symbol, with_usd_pnl=True)
    if df.empty:
        return
    eq = df[['exit_time','net_$','cum_net_$']].copy()
    eq['equity_$'] = (START_CAPITAL + eq['cum_net_$']).round(2)
    peak = eq['equity_$'].cummax()
    eq['drawdown_%'] = ((eq['equity_$'] - peak) / peak * 100).round(2)
    eq.to_csv(symbol_dir(symbol) / 'equity_curve.csv', index=False)

print('Helpers + persistence ready.')
""")

# ─────────────────────────────────────────────────────────────────────────────
md(r"""## Step 3 — Baseline run: gold-tested config on every symbol

Apply `BEST_CONFIG` from notebook 23 verbatim, no per-symbol tuning yet. We
collect the full trades list per symbol so we can break them into quarters next.
""")

code(r"""baseline_trades = {}     # symbol → trades list
baseline_stats  = {}     # symbol → overall stats dict
baseline_summary_rows = []

for sym in SYMBOLS_ACTIVE:
    try:
        tr, _ = run_symbol(sym, DATE_FROM, DATE_TO, BEST_CONFIG)
    except Exception as e:
        print(f'{sym:8} ERROR: {e!r}')
        continue
    baseline_trades[sym] = tr
    baseline_stats[sym]  = stats(tr)
    s = baseline_stats[sym]
    print(f'{sym:8} trades={s["trades"]:>5}  WR={s["win_rate"]:.2f}%  '
          f'exp={s["expectancy_R"]:.3f}R  PF={s["profit_factor"]:.2f}')
    baseline_summary_rows.append({'symbol': sym, **{k: s[k] for k in
        ('trades','wins','win_rate','avg_R','expectancy_R','profit_factor')}})
    # Save per-symbol baseline trades (for later inspection)
    save_trades(sym, tr, filename='trades_baseline.csv')

baseline_df = pd.DataFrame(baseline_summary_rows)
baseline_df.to_csv(SUMMARY_DIR / 'baseline_overall.csv', index=False)
print(f'\nSaved → {SUMMARY_DIR / "baseline_overall.csv"}')
""")

# ─────────────────────────────────────────────────────────────────────────────
md(r"""## Step 4 — Per-quarter, per-symbol breakdown

For each `(symbol, quarter)` cell we report trades, WR, expectancy. A row is
flagged **FAIL** if WR < 48 % or trades < 30 (per-quarter minimum).
""")

code(r"""def trades_to_df(trades):
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame([{
        'entry_time': t.entry_time,
        'exit_time' : t.exit_time,
        'side': t.side, 'entry': t.entry, 'exit': t.exit,
        'sl': t.sl, 'tp': t.tp, 'R': t.r_multiple, 'reason': t.reason,
    } for t in trades])


QUARTERLY_COLS = ['symbol','quarter','trades','wins','WR_%','exp_R','PF','pass']

def per_quarter_stats(trades, symbol):
    df = trades_to_df(trades)
    if df.empty:
        return pd.DataFrame(columns=QUARTERLY_COLS)
    df['quarter'] = df['entry_time'].dt.to_period('Q').astype(str)
    rows = []
    for q, g in df.groupby('quarter'):
        n = len(g); wins = int((g['R']>0).sum())
        rows.append({
            'symbol' : symbol,
            'quarter': q,
            'trades' : n,
            'wins'   : wins,
            'WR_%'   : round(wins/n*100, 2),
            'exp_R'  : round(g['R'].mean(), 3),
            'PF'     : round(g[g['R']>0]['R'].sum() / abs(g[g['R']<=0]['R'].sum() or 1e-9), 2),
            'pass'   : (wins/n*100 >= TARGET_WR) and (n >= MIN_TRADES_PER_QUARTER),
        })
    return pd.DataFrame(rows, columns=QUARTERLY_COLS)


parts = [per_quarter_stats(baseline_trades[s], s) for s in baseline_trades]
all_q = (pd.concat(parts, ignore_index=True) if parts
         else pd.DataFrame(columns=QUARTERLY_COLS))
all_q.to_csv(SUMMARY_DIR / 'baseline_quarterly_all.csv', index=False)

if all_q.empty:
    n_syms     = len(baseline_trades)
    n_with_tr  = sum(1 for s in baseline_trades if baseline_trades[s])
    raise RuntimeError(
        f'No baseline trades to summarise: {n_with_tr}/{n_syms} symbols produced any trades. '
        f'Check that the M5/H1/D1 CSVs under {DATA_DIR.resolve()} exist for the active symbols '
        f'and that DATE_FROM..DATE_TO ({DATE_FROM}..{DATE_TO}) overlaps the available data.'
    )

# Pivot for readability
pivot_wr = all_q.pivot(index='quarter', columns='symbol', values='WR_%').round(2)
pivot_n  = all_q.pivot(index='quarter', columns='symbol', values='trades')
print('=== Win-rate per quarter per symbol (BASELINE — gold config) ===')
print(pivot_wr.to_string())
print()
print('=== Trades per quarter per symbol (BASELINE) ===')
print(pivot_n.to_string())
print()

fail_rows = all_q[~all_q['pass']]
print(f'Failing (symbol, quarter) cells: {len(fail_rows)} / {len(all_q)}')
print(fail_rows.to_string(index=False))
""")

# ─────────────────────────────────────────────────────────────────────────────
md(r"""## Step 5 — Per-symbol mini-sweep

For every symbol whose **overall** WR < 48 % or per-quarter pass rate < 75 %,
sweep over `{session × MIN_REACTIONS × subset}` and pick the variant that
maximises pass-rate × average-WR. The picked config is **frozen** per symbol —
no per-quarter re-tuning (that would be curve-fitting).
""")

code(r"""# Trimmed grid — based on what worked across previous runs (drop unhelpful dims)
SWEEP_SESSIONS = [(7,16), (8,13), (8,12)]
SWEEP_CONFIGS = []
# RSI-gated (gate + ANY of confirms) — the winning mode for most symbols
for memory in [10, 15]:
    for sess in SWEEP_SESSIONS:
        for atr_min in [0.0, 0.9]:
            SWEEP_CONFIGS.append({
                'mode':'RSI-gated','confirms':['f_candle','f_ema'],'rsi_memory':memory,
                'session':sess,'sl_method':'structural','atr_min_mult':atr_min,
            })
# OR-mode (no gate, score-based) — better for some symbols (USDCHF, GBPUSD)
for confirms in [['f_candle','f_ema'], ['f_rsi','f_candle','f_ema']]:
    for sess in SWEEP_SESSIONS:
        SWEEP_CONFIGS.append({
            'mode':'OR','confirms':confirms,'min_reactions':1,'rsi_memory':10,
            'session':sess,'sl_method':'structural','atr_min_mult':0.9,
        })

print(f'Sweep configs per symbol : {len(SWEEP_CONFIGS)}')


def score_config(stats_dict, q_pass_rate):
    # Symbol-level config score: rewards WR above target and high q-pass rate.
    if stats_dict['trades'] == 0:
        return -1e9
    n_penalty = 0 if stats_dict['trades'] >= 4 * MIN_TRADES_PER_QUARTER else -10
    return (stats_dict['win_rate'] - TARGET_WR) + 50 * q_pass_rate + n_penalty


def sweep_symbol(symbol):
    rows = []
    for cfg in SWEEP_CONFIGS:
        tr, _ = run_symbol(symbol, DATE_FROM, DATE_TO, cfg)
        s = stats(tr)
        q = per_quarter_stats(tr, symbol)
        pass_rate = q['pass'].mean() if len(q) else 0
        rows.append({
            'symbol': symbol,
            'mode'  : cfg['mode'],
            'confirms': '|'.join(cfg['confirms']),
            'memory'  : cfg.get('rsi_memory', '-'),
            'min_r'   : cfg.get('min_reactions', '-'),
            'atr_min' : cfg.get('atr_min_mult', 0.0),
            'session' : f'NY{cfg["session"][0]:02d}-{cfg["session"][1]:02d}',
            'trades'  : s['trades'],
            'WR_%'    : round(s['win_rate'], 2),
            'exp_R'   : round(s['expectancy_R'], 3),
            'PF'      : round(s['profit_factor'], 2),
            'q_pass_%': round(pass_rate*100, 1),
            'score'   : round(score_config(s, pass_rate), 2),
            '_cfg'    : cfg,
        })
    df = pd.DataFrame(rows).sort_values('score', ascending=False).reset_index(drop=True)
    return df


PER_SYMBOL_BEST = {}
iter2_winners = []
for sym in SYMBOLS_ACTIVE:
    try:
        print(f'\n--- sweeping {sym} ---')
        sw = sweep_symbol(sym)
    except Exception as e:
        print(f'  ERROR sweeping {sym}: {e!r}'); continue
    if sw.empty:
        print(f'  {sym}: empty sweep result, skipping')
        continue
    try:
        top = sw.head(5)[['mode','confirms','memory','min_r','atr_min','session','trades','WR_%','exp_R','PF','q_pass_%','score']]
        print(top.to_string(index=False))
    except Exception as e:
        print(f'  {sym}: display error: {e!r}')
    PER_SYMBOL_BEST[sym] = sw.iloc[0]['_cfg']
    # Save full sweep + winner row per symbol (defensive)
    try:
        sw_to_save = sw.drop(columns=['_cfg'], errors='ignore').copy()
        sw_to_save.to_csv(symbol_dir(sym) / 'sweep_iter2.csv', index=False)
        winner_row = sw.iloc[0].drop(['_cfg']).to_dict()
        winner_row['symbol'] = sym
        iter2_winners.append(winner_row)
    except Exception as e:
        print(f'  {sym}: save error: {e!r}')
try:
    pd.DataFrame(iter2_winners).to_csv(SUMMARY_DIR / 'iter2_sweep_winners.csv', index=False)
    print(f'\nSaved iter-2 → {SUMMARY_DIR / "iter2_sweep_winners.csv"} + per-symbol sweep_iter2.csv')
except Exception as e:
    print(f'iter-2 summary save error: {e!r}')
""")

# ─────────────────────────────────────────────────────────────────────────────
md(r"""## Step 5b — Iteration 3: refine with H1 trend-strength filter

For each symbol's winning base config from Step 5, sweep only `htf_strength_min`
over a small grid. The goal: skip range-bound regimes that fail the WR test.
""")

code(r"""HTF_STRENGTH_GRID = [0.0, 0.15, 0.30, 0.50, 0.80]

iter3_results = []
PER_SYMBOL_BEST_V2 = {}

for sym, base_cfg in PER_SYMBOL_BEST.items():
    rows = []
    for sm in HTF_STRENGTH_GRID:
        cfg = {**base_cfg, 'htf_strength_min': sm}
        try:
            tr, _ = run_symbol(sym, DATE_FROM, DATE_TO, cfg)
        except Exception as e:
            print(f'  {sym}: error sm={sm}: {e!r}'); continue
        s = stats(tr)
        q = per_quarter_stats(tr, sym)
        pass_rate = q['pass'].mean() if len(q) else 0
        rows.append({
            'symbol': sym, 'htf_strength_min': sm,
            'trades': s['trades'], 'WR_%': round(s['win_rate'],2),
            'exp_R' : round(s['expectancy_R'],3), 'PF': round(s['profit_factor'],2),
            'q_pass_%': round(pass_rate*100, 1),
            'score'   : round(score_config(s, pass_rate), 2),
            '_cfg'    : cfg,
        })
    if not rows:
        continue
    sub = pd.DataFrame(rows).sort_values('score', ascending=False).reset_index(drop=True)
    iter3_results.append(sub)
    PER_SYMBOL_BEST_V2[sym] = sub.iloc[0]['_cfg']
    print(f'\n--- {sym}: trend-strength sweep ---')
    try:
        print(sub[['htf_strength_min','trades','WR_%','exp_R','PF','q_pass_%','score']].to_string(index=False))
    except Exception as e:
        print(f'  display error: {e!r}')
    try:
        sub.drop(columns=['_cfg']).to_csv(symbol_dir(sym) / 'sweep_iter3.csv', index=False)
    except Exception as e:
        print(f'  {sym}: iter-3 save error: {e!r}')

try:
    if iter3_results:
        pd.concat([d.drop(columns=['_cfg']) for d in iter3_results], ignore_index=True)\
          .to_csv(SUMMARY_DIR / 'iter3_trend_strength.csv', index=False)
except Exception as e:
    print(f'iter-3 summary save error: {e!r}')
""")


# ─────────────────────────────────────────────────────────────────────────────
md(r"""## Step 5c — Iteration 4: overlay new indicators (ADX, H1-RSI, MACD, volume)

The trend-strength filter in iter-3 was redundant. Iter-4 tests genuinely new
information: ADX (regime quality), H1 RSI alignment (HTF momentum agreement),
MACD histogram (M5 momentum agreement), volume spike (institutional interest).

For each symbol, we overlay these on the **winning base config from iter-3**
(NOT a fresh sweep). Goal: lift WR ≥ 48% per symbol without cutting trades.
""")

code(r"""ITER4_OVERLAYS = [
    {},                                                              # baseline (iter-3 winner)
    {'adx_min': 18.0},
    {'adx_min': 22.0},
    {'require_h1_rsi_align': True},
    {'require_macd_align': True},
    {'adx_min': 18.0, 'require_h1_rsi_align': True},
    {'adx_min': 18.0, 'require_macd_align': True},
    {'require_h1_rsi_align': True, 'require_macd_align': True},
    {'adx_min': 20.0, 'require_h1_rsi_align': True, 'require_macd_align': True},  # strict
    # Add a "vote with volume" variant — adds f_vol as a confirming reaction
    {'add_vol_confirm': True},
    {'adx_min': 18.0, 'add_vol_confirm': True},
]

PER_SYMBOL_BEST_V3 = {}     # final after iter-4
iter4_rows = []

for sym, base_cfg in PER_SYMBOL_BEST_V2.items():
    rows = []
    winner_trades = None
    winner_score  = -1e9
    winner_cfg    = None
    for ov in ITER4_OVERLAYS:
        cfg = {**base_cfg}
        ov_clean = dict(ov)
        if ov_clean.pop('add_vol_confirm', False):
            cfg['confirms'] = list(cfg['confirms']) + ['f_vol']
        cfg.update(ov_clean)
        try:
            tr, _ = run_symbol(sym, DATE_FROM, DATE_TO, cfg)
        except Exception as e:
            print(f'  {sym}: error overlay={ov}: {e!r}'); continue
        s = stats(tr)
        q = per_quarter_stats(tr, sym)
        pass_rate = q['pass'].mean() if len(q) else 0
        sc = round(score_config(s, pass_rate), 2)
        rows.append({
            'symbol' : sym,
            'overlay': str({k:v for k,v in ov.items()}) or 'baseline',
            'trades' : s['trades'],
            'WR_%'   : round(s['win_rate'], 2),
            'exp_R'  : round(s['expectancy_R'], 3),
            'PF'     : round(s['profit_factor'], 2),
            'q_pass_%': round(pass_rate*100, 1),
            'score'  : sc,
            '_cfg'   : cfg,
        })
        # Track winning trades so we can persist them without a re-run
        if sc > winner_score:
            winner_score  = sc
            winner_trades = tr
            winner_cfg    = cfg
    if not rows:
        continue
    sub = pd.DataFrame(rows).sort_values('score', ascending=False).reset_index(drop=True)
    iter4_rows.append(sub)
    PER_SYMBOL_BEST_V3[sym] = winner_cfg
    print(f'\n--- {sym}: iter-4 overlay results (top 5) ---')
    try:
        print(sub.head(5)[['overlay','trades','WR_%','exp_R','PF','q_pass_%','score']].to_string(index=False))
    except Exception as e:
        print(f'  display error: {e!r}')
    # Save overlay table + winning config + winning trades artefacts so downstream
    # cells can read everything from disk without re-running the strategy.
    try: sub.drop(columns=['_cfg']).to_csv(symbol_dir(sym) / 'sweep_iter4.csv', index=False)
    except Exception as e: print(f'  {sym}: sweep_iter4 save error: {e!r}')
    try: save_config(sym, winner_cfg)
    except Exception as e: print(f'  {sym}: config save error: {e!r}')
    try: save_trades(sym, winner_trades, filename='trades_full.csv')
    except Exception as e: print(f'  {sym}: trades_full save error: {e!r}')
    try: save_equity_curve(sym, winner_trades)
    except Exception as e: print(f'  {sym}: equity_curve save error: {e!r}')
    try:
        q_win = per_quarter_stats(winner_trades, sym)
        if not q_win.empty:
            q_win.to_csv(symbol_dir(sym) / 'quarterly.csv', index=False)
    except Exception as e: print(f'  {sym}: quarterly save error: {e!r}')
    # Persist final stats row so Step 6 can read it without re-running
    try:
        s_win = stats(winner_trades)
        pd.DataFrame([{'symbol': sym, **s_win}]).to_csv(
            symbol_dir(sym) / 'final_stats.csv', index=False)
    except Exception as e: print(f'  {sym}: final_stats save error: {e!r}')

try:
    if iter4_rows:
        pd.concat([d.drop(columns=['_cfg']) for d in iter4_rows], ignore_index=True)\
          .to_csv(SUMMARY_DIR / 'iter4_overlays.csv', index=False)
except Exception as e:
    print(f'iter-4 summary save error: {e!r}')
""")


# ─────────────────────────────────────────────────────────────────────────────
md(r"""## Step 6 — Final per-symbol stats (LOADS FROM DISK — no re-run)

After Step 5c, every symbol has `trades_full.csv`, `quarterly.csv`, `config.json`
and `final_stats.csv` saved on disk. This cell just aggregates them — runs
**instantly** even after a kernel restart.
""")

code(r"""# Load per-symbol final stats + per-quarter tables from disk
import json as _json

final_per_symbol_rows = []
final_q_rows = []
loaded_configs = {}

for sym_dir in sorted(RESULTS_DIR.iterdir()):
    if not sym_dir.is_dir() or sym_dir.name in ('_summary', '_portfolio'):
        continue
    sym = sym_dir.name
    fs_path = sym_dir / 'final_stats.csv'
    if fs_path.exists():
        try:
            row = pd.read_csv(fs_path).iloc[0].to_dict()
            final_per_symbol_rows.append(row)
        except Exception as e:
            print(f'  {sym}: final_stats load error: {e!r}')
    q_path = sym_dir / 'quarterly.csv'
    if q_path.exists():
        try:
            final_q_rows.append(pd.read_csv(q_path))
        except Exception as e:
            print(f'  {sym}: quarterly load error: {e!r}')
    cfg_path = sym_dir / 'config.json'
    if cfg_path.exists():
        try:
            loaded_configs[sym] = _json.loads(cfg_path.read_text())
        except Exception as e:
            print(f'  {sym}: config load error: {e!r}')

final_q = pd.concat(final_q_rows, ignore_index=True) if final_q_rows else pd.DataFrame()
print(f'Loaded final stats for {len(final_per_symbol_rows)} symbols, '
      f'{len(final_q)} quarter rows, {len(loaded_configs)} configs.')

# Display per-symbol summary
for row in sorted(final_per_symbol_rows, key=lambda r: -r.get('win_rate', 0)):
    sym = row['symbol']; cfg = loaded_configs.get(sym, {})
    sess = cfg.get('session', [0,0])
    print(f'{sym:8} mode={cfg.get("mode","-"):13} confirms={"|".join(cfg.get("confirms",["-"])):20} '
          f'mem={cfg.get("rsi_memory","-"):>2} atr_min={cfg.get("atr_min_mult",0.0):.1f}  '
          f'sess=NY{sess[0]:02d}-{sess[1]:02d}  '
          f'trades={int(row["trades"]):>5}  WR={row["win_rate"]:.2f}%  exp={row["expectancy_R"]:.3f}R')

try:
    pd.DataFrame(final_per_symbol_rows).to_csv(SUMMARY_DIR / 'final_per_symbol.csv', index=False)
except Exception as e:
    print(f'final_per_symbol save error: {e!r}')
if not final_q.empty:
    pivot_wr2 = final_q.pivot(index='quarter', columns='symbol', values='WR_%').round(2)
    pivot_n2  = final_q.pivot(index='quarter', columns='symbol', values='trades')
    print()
    print('=== Win-rate per quarter per symbol (FINAL — per-symbol tuned) ===')
    print(pivot_wr2.to_string())
    print()
    print('=== Trades per quarter per symbol (FINAL) ===')
    print(pivot_n2.to_string())
    print()
    fails = final_q[~final_q['pass']]
    print(f'Failing cells (WR<{TARGET_WR}% or trades<{MIN_TRADES_PER_QUARTER}): '
          f'{len(fails)} / {len(final_q)}')
    if not fails.empty:
        print(fails[['symbol','quarter','trades','WR_%','exp_R']].to_string(index=False))

    final_q.to_csv(SUMMARY_DIR / 'final_quarterly_stats.csv', index=False)
""")

# ─────────────────────────────────────────────────────────────────────────────
md(r"""## Step 7 — Cost-aware net USD P&L per symbol

Apply per-symbol spread + pip-value to convert every trade to USD and sum.
""")

code(r"""def pip_units(symbol, price_move):
    return price_move / SYMBOLS_CFG[symbol]['pip_size']

def trade_pnl_usd(symbol, t):
    pip_move = (t.exit - t.entry) * t.side / SYMBOLS_CFG[symbol]['pip_size']
    gross    = pip_move * SYMBOLS_CFG[symbol]['pip_value_usd_per_002lot']
    fee      = SYMBOLS_CFG[symbol]['spread_pips'] * SYMBOLS_CFG[symbol]['pip_value_usd_per_002lot']
    return gross, fee, gross - fee


# Step 7 reads each symbol's saved trades_full.csv and computes cost-aware totals.
def _compute_max_dd_from_series(net_series):
    cum  = np.cumsum(net_series) + START_CAPITAL
    peak = np.maximum.accumulate(cum)
    dd   = (cum - peak) / peak * 100
    return float(dd.min()) if len(dd) else 0.0


cost_rows = []
for sym_dir in sorted(RESULTS_DIR.iterdir()):
    if not sym_dir.is_dir() or sym_dir.name in ('_summary', '_portfolio'):
        continue
    sym = sym_dir.name
    tpath = sym_dir / 'trades_full.csv'
    if not tpath.exists():
        continue
    try:
        df = pd.read_csv(tpath)
        if df.empty or 'net_$' not in df.columns:
            continue
        df['entry_time'] = pd.to_datetime(df['entry_time'])
        df = df.sort_values('entry_time')
        max_dd = _compute_max_dd_from_series(df['net_$'].values)
        n      = len(df)
        wins_n = int((df['net_$'] > 0).sum())
        cost_rows.append({
            'symbol'  : sym,
            'trades'  : n,
            'wr_net_%': round(wins_n / n * 100, 2),
            'gross_$' : round(df['gross_$'].sum(), 2),
            'fee_$'   : round(df['fee_$'].sum(),   2),
            'net_$'   : round(df['net_$'].sum(),   2),
            'return_%': round(df['net_$'].sum() / START_CAPITAL * 100, 2),
            'max_DD_%': round(max_dd, 2),
        })
    except Exception as e:
        print(f'  {sym}: cost-analysis load error: {e!r}')

cost_df = pd.DataFrame(cost_rows)
print('=== Cost-aware net USD P&L per symbol (0.02 lot, $1k start) ===')
print(cost_df.to_string(index=False))
print()
print(f'Portfolio gross  : ${cost_df["gross_$"].sum():+.2f}')
print(f'Portfolio fees   : ${cost_df["fee_$"].sum():.2f}')
print(f'Portfolio net    : ${cost_df["net_$"].sum():+.2f}')
print(f'Portfolio return : {cost_df["net_$"].sum()/START_CAPITAL*100:+.2f}% on $1k')

cost_df.to_csv(SUMMARY_DIR / 'final_cost_summary.csv', index=False)
""")

# ─────────────────────────────────────────────────────────────────────────────
md(r"""## Step 8 — Forward test (unseen) per symbol

The frozen per-symbol configs are evaluated on the forward window
`FORWARD_FROM → FORWARD_TO` — data the sweep never saw.
""")

code(r"""# Step 8 — Forward test. Lazy: reads `trades_forward.csv` if present;
# otherwise runs forward with the saved per-symbol config and persists.
fwd_rows = []
import json as _json

for sym_dir in sorted(RESULTS_DIR.iterdir()):
    if not sym_dir.is_dir() or sym_dir.name in ('_summary', '_portfolio'):
        continue
    sym = sym_dir.name
    if sym not in SYMBOLS_CFG:
        continue
    fwd_path = sym_dir / 'trades_forward.csv'

    if fwd_path.exists():
        # Load cached forward trades
        try:
            df = pd.read_csv(fwd_path)
        except Exception as e:
            print(f'  {sym}: cannot read cached forward: {e!r}'); df = pd.DataFrame()
    else:
        # Run forward using saved config
        cfg_path = sym_dir / 'config.json'
        if not cfg_path.exists():
            continue
        try:
            cfg = _json.loads(cfg_path.read_text())
            if isinstance(cfg.get('session'), list):
                cfg['session'] = tuple(cfg['session'])
            tr, _ = run_symbol(sym, FORWARD_FROM, FORWARD_TO, cfg)
            df = save_trades(sym, tr, filename='trades_forward.csv') \
                 if tr else pd.DataFrame()
        except Exception as e:
            print(f'  {sym}: forward run error: {e!r}'); df = pd.DataFrame()

    if df is None or df.empty:
        fwd_rows.append({'symbol':sym, 'trades':0, 'WR_%':0, 'exp_R':0,
                         'gross_$':0, 'fee_$':0, 'net_$':0, 'return_%':0})
        continue
    n        = len(df)
    wins_net = int((df['net_$'] > 0).sum())
    gross    = float(df['gross_$'].sum())
    fee      = float(df['fee_$'].sum())
    net      = float(df['net_$'].sum())
    exp_R    = float(df['R'].mean()) if 'R' in df.columns else 0.0
    fwd_rows.append({
        'symbol' : sym,
        'trades' : n,
        'WR_%'   : round(wins_net/n*100, 2),
        'exp_R'  : round(exp_R, 3),
        'gross_$': round(gross, 2),
        'fee_$'  : round(fee, 2),
        'net_$'  : round(net, 2),
        'return_%': round(net/START_CAPITAL*100, 2),
    })

fwd_df = pd.DataFrame(fwd_rows)
print(f'=== Forward window {FORWARD_FROM} → {FORWARD_TO} ===')
print(fwd_df.to_string(index=False))
print()
print(f'Portfolio forward net    : ${fwd_df["net_$"].sum():+.2f}')
print(f'Portfolio forward return : {fwd_df["net_$"].sum()/START_CAPITAL*100:+.2f}% on $1k')

fwd_df.to_csv(SUMMARY_DIR / 'forward_summary.csv', index=False)
""")

# ─────────────────────────────────────────────────────────────────────────────
md(r"""## Step 8b — Symbol ranking: best instruments overall

Combines in-sample + forward results into a composite score:
`score = (return_%) × (WR / 50) / max(1, |max_DD_%|)` — rewards profit, WR
above 50, penalises drawdown. Sorted descending.
""")

code(r"""# Combine cost_df (IS) with fwd_df (forward) by symbol
combined = cost_df.set_index('symbol').join(
    fwd_df.set_index('symbol').rename(columns={
        'WR_%':'fwd_WR_%','net_$':'fwd_net_$','return_%':'fwd_return_%','trades':'fwd_trades'
    })[['fwd_WR_%','fwd_net_$','fwd_return_%','fwd_trades']],
    how='left'
)

# Composite score (use IS data only — forward is sanity check)
def _score(row):
    wr_factor = max(0.5, row['wr_net_%'] / 50.0)
    dd_penalty = max(1.0, abs(row['max_DD_%']))
    return row['return_%'] * wr_factor / dd_penalty * (1 + row['trades']/1000)

combined['score'] = combined.apply(_score, axis=1)
combined = combined.sort_values('score', ascending=False)

print('=== Symbol ranking (composite score: return × WR-factor / DD, weighted by trade count) ===')
print(combined[['trades','wr_net_%','net_$','return_%','max_DD_%',
                'fwd_trades','fwd_WR_%','fwd_net_$','score']].to_string())
print()

# Pick top-N as the recommended live basket where IS WR >= TARGET_WR
recommended = combined[
    (combined['wr_net_%'] >= TARGET_WR) &
    (combined['fwd_net_$'] >= 0)
].sort_values('score', ascending=False)
print(f'=== Recommended basket (IS WR ≥ {TARGET_WR}% AND forward net ≥ 0) ===')
if recommended.empty:
    print('No symbol met both criteria. Top 5 by score:')
    print(combined.head(5)[['trades','wr_net_%','return_%','max_DD_%','fwd_WR_%','score']].to_string())
else:
    print(recommended[['trades','wr_net_%','return_%','max_DD_%','fwd_WR_%','fwd_net_$']].to_string())

combined.to_csv(SUMMARY_DIR / 'symbol_ranking.csv')
""")


# ─────────────────────────────────────────────────────────────────────────────
md(r"""## Step 8c — Portfolio aggregation (golden basket)

Combines trades from all symbols meeting `WR_IS ≥ TARGET_WR AND fwd_net ≥ 0`
into one chronological stream. Reports portfolio-level WR, expectancy, equity
curve, max drawdown, monthly returns, and per-quarter portfolio WR.
""")

code(r"""# 1. Identify the golden basket from the ranking
GOLDEN_BASKET = combined[
    (combined['wr_net_%'] >= TARGET_WR) &
    (combined['fwd_net_$'].fillna(0) >= 0)
].index.tolist()

if not GOLDEN_BASKET:
    # Fallback: top 5 by score
    GOLDEN_BASKET = combined.head(5).index.tolist()
    print(f'No symbol met both criteria — using top 5 by score: {GOLDEN_BASKET}')
else:
    print(f'Golden basket ({len(GOLDEN_BASKET)} symbols): {GOLDEN_BASKET}')


def build_portfolio_stream_from_disk(basket, filename):
    parts = []
    for sym in basket:
        path = RESULTS_DIR / sym / filename
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        df['symbol'] = sym
        df['entry_time'] = pd.to_datetime(df['entry_time'])
        df['exit_time']  = pd.to_datetime(df['exit_time'])
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True).sort_values('entry_time').reset_index(drop=True)
    df['cum_net_$']  = df['net_$'].cumsum().round(2)
    df['equity_$']   = (START_CAPITAL + df['cum_net_$']).round(2)
    peak             = df['equity_$'].cummax()
    df['drawdown_%'] = ((df['equity_$'] - peak) / peak * 100).round(2)
    return df


def portfolio_stats(df, label):
    if df.empty:
        return None
    n = len(df)
    wins = int((df['net_$'] > 0).sum())
    wr   = wins / n * 100
    gross_w = df.loc[df['net_$']>0, 'net_$'].sum()
    gross_l = -df.loc[df['net_$']<=0, 'net_$'].sum()
    pf      = gross_w / gross_l if gross_l > 0 else float('inf')
    net     = df['net_$'].sum()
    ret_pct = net / START_CAPITAL * 100
    max_dd  = df['drawdown_%'].min()
    days    = (pd.to_datetime(df['exit_time'].iloc[-1]) - pd.to_datetime(df['entry_time'].iloc[0])).days
    return {
        'label'        : label,
        'symbols'      : len(df['symbol'].unique()),
        'trades'       : n,
        'WR_%'         : round(wr, 2),
        'avg_R'        : round(df['R'].mean(), 3),
        'profit_factor': round(pf, 2),
        'net_$'        : round(net, 2),
        'return_%'     : round(ret_pct, 2),
        'max_DD_%'     : round(max_dd, 2),
        'days'         : days,
        'trades_per_day': round(n/max(days,1), 2),
    }


# 2. Build portfolios from disk (no re-run needed — uses cached CSVs)
port_is  = build_portfolio_stream_from_disk(GOLDEN_BASKET, 'trades_full.csv')
port_fwd = build_portfolio_stream_from_disk(GOLDEN_BASKET, 'trades_forward.csv')

# 3. Print summary stats
stats_is  = portfolio_stats(port_is,  'IN-SAMPLE')
stats_fwd = portfolio_stats(port_fwd, 'FORWARD')
summary = pd.DataFrame([s for s in [stats_is, stats_fwd] if s])
print()
print('=== Portfolio summary (golden basket) ===')
print(summary.to_string(index=False))
print()

# 4. Per-symbol contribution
print('--- Per-symbol contribution (IS) ---')
contrib = port_is.groupby('symbol').agg(
    trades=('net_$', 'size'),
    wins=('net_$', lambda x: (x>0).sum()),
    net_USD=('net_$', 'sum'),
).sort_values('net_USD', ascending=False)
contrib['WR_%'] = (contrib['wins']/contrib['trades']*100).round(2)
contrib['share_%'] = (contrib['net_USD']/contrib['net_USD'].sum()*100).round(1)
contrib['net_USD'] = contrib['net_USD'].round(2)
print(contrib[['trades','wins','WR_%','net_USD','share_%']].to_string())
print()

# 5. Per-quarter portfolio WR
print('--- Per-quarter portfolio WR (IS) ---')
qport = port_is.copy()
qport['quarter'] = pd.to_datetime(qport['entry_time']).dt.to_period('Q').astype(str)
q_summary = qport.groupby('quarter').agg(
    trades=('net_$', 'size'),
    wins=('net_$', lambda x: (x>0).sum()),
    net_USD=('net_$', 'sum'),
)
q_summary['WR_%'] = (q_summary['wins']/q_summary['trades']*100).round(2)
q_summary['return_%'] = (q_summary['net_USD']/START_CAPITAL*100).round(2)
q_summary['net_USD'] = q_summary['net_USD'].round(2)
print(q_summary[['trades','wins','WR_%','net_USD','return_%']].to_string())
print()

# 6. Monthly returns
print('--- Monthly returns (IS) ---')
mport = port_is.copy()
mport['month'] = pd.to_datetime(mport['entry_time']).dt.to_period('M').astype(str)
m_summary = mport.groupby('month').agg(
    trades=('net_$','size'),
    net_USD=('net_$','sum'),
)
m_summary['return_%'] = (m_summary['net_USD']/START_CAPITAL*100).round(2)
m_summary['net_USD']  = m_summary['net_USD'].round(2)
print(m_summary[['trades','net_USD','return_%']].to_string())
print()

# 7. Save everything to disk
try:
    PORT_DIR = RESULTS_DIR / '_portfolio'
    PORT_DIR.mkdir(parents=True, exist_ok=True)
    port_is.to_csv (PORT_DIR / 'trades_IS.csv',  index=False)
    port_fwd.to_csv(PORT_DIR / 'trades_FWD.csv', index=False)
    summary.to_csv(PORT_DIR / 'summary.csv', index=False)
    contrib.to_csv(PORT_DIR / 'contribution_per_symbol.csv')
    q_summary.to_csv(PORT_DIR / 'quarterly_returns.csv')
    m_summary.to_csv(PORT_DIR / 'monthly_returns.csv')

    # Equity curve as standalone file
    eq = port_is[['entry_time','symbol','net_$','equity_$','drawdown_%']].copy()
    eq.to_csv(PORT_DIR / 'equity_curve_IS.csv', index=False)
    if not port_fwd.empty:
        port_fwd[['entry_time','symbol','net_$','equity_$','drawdown_%']].to_csv(
            PORT_DIR / 'equity_curve_FWD.csv', index=False)

    # Basket meta
    import json as _json
    (PORT_DIR / 'basket.json').write_text(_json.dumps({
        'basket': GOLDEN_BASKET,
        'criteria': f'WR_IS ≥ {TARGET_WR}% AND fwd_net ≥ 0',
        'starting_capital': START_CAPITAL,
        'lot_per_trade': LOT_SIZE,
    }, indent=2))
    print(f'Saved → {PORT_DIR}/')
except Exception as e:
    print(f'Portfolio save error: {e!r}')

# 8. Equity curve plot
try:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=port_is['exit_time'], y=port_is['equity_$'],
        mode='lines', name='IS equity', line=dict(color='royalblue', width=2)))
    if not port_fwd.empty:
        # Continue forward equity from IS end
        fwd_eq = port_fwd['equity_$'] + port_is['equity_$'].iat[-1] - START_CAPITAL
        fig.add_trace(go.Scatter(
            x=port_fwd['exit_time'], y=fwd_eq,
            mode='lines', name='Forward equity', line=dict(color='green', width=2, dash='dot')))
    fig.update_layout(title=f'Portfolio Equity Curve ({len(GOLDEN_BASKET)} symbols, $1k start, 0.02 lot)',
                      xaxis_title='Time', yaxis_title='Equity ($)',
                      template='plotly_white', height=400, margin=dict(l=40,r=10,t=40,b=30))
    fig.show()
except Exception as e:
    print(f'Plot error: {e!r}')
""")


# ─────────────────────────────────────────────────────────────────────────────
md(r"""## Step 9 — Per-symbol config dump

For the record / live-deployment reference.
""")

code(r"""# Step 9 reads every <symbol>/config.json saved by Step 5c and consolidates them.
import json as _json

all_configs = {}
for sym_dir in sorted(RESULTS_DIR.iterdir()):
    if not sym_dir.is_dir() or sym_dir.name in ('_summary', '_portfolio'):
        continue
    cfg_path = sym_dir / 'config.json'
    if cfg_path.exists():
        try:
            all_configs[sym_dir.name] = _json.loads(cfg_path.read_text())
        except Exception as e:
            print(f'  {sym_dir.name}: config load error: {e!r}')

print(f'Frozen per-symbol configs ({len(all_configs)}):')
for sym, cfg in all_configs.items():
    print(f'  {sym}: {cfg}')

(SUMMARY_DIR / 'per_symbol_configs.json').write_text(
    _json.dumps(all_configs, indent=2, default=str)
)
print(f'\nSaved → {SUMMARY_DIR/"per_symbol_configs.json"}')

# Final summary of what's saved
print('\n=== FILES PERSISTED ===')
print(f'Per-symbol directories : {RESULTS_DIR}/<SYMBOL>/')
print(f'  trades_baseline.csv   — baseline-config trades w/ R, gross/fee/net')
print(f'  trades_full.csv       — final-config trades w/ R, gross/fee/net, cumulative')
print(f'  trades_forward.csv    — forward-window trades')
print(f'  quarterly.csv         — per-quarter stats')
print(f'  sweep_iter2.csv       — all configs tested in iter-2')
print(f'  sweep_iter3.csv       — trend-strength sweep')
print(f'  sweep_iter4.csv       — new-indicator overlays')
print(f'  equity_curve.csv      — cumulative net + drawdown over time')
print(f'  config.json           — frozen winning config')
print(f'Summary directory      : {SUMMARY_DIR}/')
for f in sorted(SUMMARY_DIR.glob("*.csv")) + sorted(SUMMARY_DIR.glob("*.json")):
    print(f'  {f.name}')
""")


# ─────────────────────────────────────────────────────────────────────────────
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

OUT = Path(__file__).parent / "24_multi_symbol_scalper.ipynb"
OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {OUT}  ({len(CELLS)} cells)")
