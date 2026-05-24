"""Standalone runner for Steps 6 through 9 of notebook 24 — reads CSVs only.

Lets you re-run the post-sweep analysis without redoing the 50-minute sweep.
Use after the notebook's sweep has populated `results/multi_symbol_scalper/`.

Usage: python _post_sweep_24.py
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ── Constants (mirror notebook config) ─────────────────────────────────────
DATA_DIR        = Path('./data')
STRATEGY        = 'multi_symbol_scalper'
RESULTS_DIR     = Path('./results') / STRATEGY
SUMMARY_DIR     = RESULTS_DIR / '_summary'
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

DATE_FROM       = '2023-01-01'
DATE_TO         = '2026-02-15'
FORWARD_FROM    = '2026-02-15'
FORWARD_TO      = '2026-05-15'
START_CAPITAL   = 1000.0
LOT_SIZE        = 0.02
TARGET_WR       = 48.0
MIN_TRADES_PER_QUARTER = 30


# =============================================================================
# Step 6 — Load per-symbol final stats from disk
# =============================================================================
print('=== Step 6: Loading final per-symbol stats from disk ===\n')

final_per_symbol_rows = []
final_q_rows = []
loaded_configs = {}

for sym_dir in sorted(RESULTS_DIR.iterdir()):
    if not sym_dir.is_dir() or sym_dir.name in ('_summary', '_portfolio'):
        continue
    sym = sym_dir.name
    fs_path  = sym_dir / 'final_stats.csv'
    q_path   = sym_dir / 'quarterly.csv'
    cfg_path = sym_dir / 'config.json'
    if fs_path.exists():
        try:
            final_per_symbol_rows.append(pd.read_csv(fs_path).iloc[0].to_dict())
        except Exception as e:
            print(f'  {sym}: final_stats load error: {e!r}')
    if q_path.exists():
        try:
            final_q_rows.append(pd.read_csv(q_path))
        except Exception as e:
            print(f'  {sym}: quarterly load error: {e!r}')
    if cfg_path.exists():
        try:
            loaded_configs[sym] = json.loads(cfg_path.read_text())
        except Exception as e:
            print(f'  {sym}: config load error: {e!r}')

final_q = pd.concat(final_q_rows, ignore_index=True) if final_q_rows else pd.DataFrame()
print(f'Loaded final stats : {len(final_per_symbol_rows)} symbols')
print(f'Loaded quarterly   : {len(final_q)} rows')
print(f'Loaded configs     : {len(loaded_configs)}\n')

if final_per_symbol_rows:
    pd.DataFrame(final_per_symbol_rows).to_csv(SUMMARY_DIR / 'final_per_symbol.csv', index=False)
if not final_q.empty:
    final_q.to_csv(SUMMARY_DIR / 'final_quarterly_stats.csv', index=False)


# =============================================================================
# Step 7 — Cost-aware net USD P&L per symbol (reads trades_full.csv)
# =============================================================================
print('=== Step 7: Cost-aware P&L from disk ===\n')

def _max_dd_from_series(net_series):
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
            'max_DD_%': round(_max_dd_from_series(df['net_$'].values), 2),
        })
    except Exception as e:
        print(f'  {sym}: cost-analysis load error: {e!r}')

cost_df = pd.DataFrame(cost_rows)
print(cost_df.to_string(index=False))
print(f'\nPortfolio gross  : ${cost_df["gross_$"].sum():+.2f}')
print(f'Portfolio fees   : ${cost_df["fee_$"].sum():.2f}')
print(f'Portfolio net    : ${cost_df["net_$"].sum():+.2f}\n')
cost_df.to_csv(SUMMARY_DIR / 'final_cost_summary.csv', index=False)


# =============================================================================
# Step 8 — Forward results (reads trades_forward.csv — no re-run)
# =============================================================================
print('=== Step 8: Forward results from disk ===\n')

fwd_rows = []
for sym_dir in sorted(RESULTS_DIR.iterdir()):
    if not sym_dir.is_dir() or sym_dir.name in ('_summary', '_portfolio'):
        continue
    sym = sym_dir.name
    fwd_path = sym_dir / 'trades_forward.csv'
    if not fwd_path.exists():
        continue
    try:
        df = pd.read_csv(fwd_path)
    except Exception:
        df = pd.DataFrame()
    if df.empty or 'net_$' not in df.columns:
        fwd_rows.append({'symbol': sym, 'trades': 0, 'WR_%': 0, 'exp_R': 0,
                         'gross_$': 0, 'fee_$': 0, 'net_$': 0, 'return_%': 0})
        continue
    n = len(df); wins_n = int((df['net_$'] > 0).sum())
    fwd_rows.append({
        'symbol' : sym,
        'trades' : n,
        'WR_%'   : round(wins_n/n*100, 2),
        'exp_R'  : round(df['R'].mean(), 3) if 'R' in df else 0.0,
        'gross_$': round(df['gross_$'].sum(), 2),
        'fee_$'  : round(df['fee_$'].sum(),   2),
        'net_$'  : round(df['net_$'].sum(),   2),
        'return_%': round(df['net_$'].sum()/START_CAPITAL*100, 2),
    })

fwd_df = pd.DataFrame(fwd_rows)
print(fwd_df.to_string(index=False))
fwd_df.to_csv(SUMMARY_DIR / 'forward_summary.csv', index=False)
print()


# =============================================================================
# Step 8b — Symbol ranking
# =============================================================================
print('=== Step 8b: Symbol ranking ===\n')

combined = cost_df.set_index('symbol').join(
    fwd_df.set_index('symbol').rename(columns={
        'WR_%':'fwd_WR_%','net_$':'fwd_net_$','return_%':'fwd_return_%','trades':'fwd_trades'
    })[['fwd_WR_%','fwd_net_$','fwd_return_%','fwd_trades']],
    how='left'
)

def _score(row):
    wr_factor = max(0.5, row['wr_net_%'] / 50.0)
    dd_penalty = max(1.0, abs(row['max_DD_%']))
    return row['return_%'] * wr_factor / dd_penalty * (1 + row['trades']/1000)

combined['score'] = combined.apply(_score, axis=1)
combined = combined.sort_values('score', ascending=False)
print(combined[['trades','wr_net_%','net_$','return_%','max_DD_%',
                'fwd_trades','fwd_WR_%','fwd_net_$','score']].to_string())
combined.to_csv(SUMMARY_DIR / 'symbol_ranking.csv')
print()


# =============================================================================
# Step 8c — Portfolio aggregation
# =============================================================================
print('=== Step 8c: Portfolio (golden basket) ===\n')

GOLDEN_BASKET = combined[
    (combined['wr_net_%'] >= TARGET_WR) &
    (combined['fwd_net_$'].fillna(0) >= 0)
].index.tolist()
if not GOLDEN_BASKET:
    GOLDEN_BASKET = combined.head(5).index.tolist()
print(f'Basket: {GOLDEN_BASKET}\n')


def _stream_from_disk(basket, filename):
    parts = []
    for sym in basket:
        p = RESULTS_DIR / sym / filename
        if not p.exists(): continue
        df = pd.read_csv(p)
        if df.empty: continue
        df['symbol'] = sym
        df['entry_time'] = pd.to_datetime(df['entry_time'])
        df['exit_time']  = pd.to_datetime(df['exit_time'])
        parts.append(df)
    if not parts: return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True).sort_values('entry_time').reset_index(drop=True)
    df['cum_net_$']  = df['net_$'].cumsum().round(2)
    df['equity_$']   = (START_CAPITAL + df['cum_net_$']).round(2)
    peak             = df['equity_$'].cummax()
    df['drawdown_%'] = ((df['equity_$'] - peak) / peak * 100).round(2)
    return df


def _port_stats(df, label):
    if df.empty: return None
    n=len(df); wins=int((df['net_$']>0).sum())
    gw=df.loc[df['net_$']>0,'net_$'].sum(); gl=-df.loc[df['net_$']<=0,'net_$'].sum()
    return {
        'label':label, 'symbols':df['symbol'].nunique(), 'trades':n,
        'WR_%':round(wins/n*100,2), 'avg_R':round(df['R'].mean(),3),
        'profit_factor':round(gw/gl,2) if gl>0 else float('inf'),
        'net_$':round(df['net_$'].sum(),2),
        'return_%':round(df['net_$'].sum()/START_CAPITAL*100,2),
        'max_DD_%':round(df['drawdown_%'].min(),2),
        'days':(df['exit_time'].iloc[-1]-df['entry_time'].iloc[0]).days,
    }

port_is  = _stream_from_disk(GOLDEN_BASKET, 'trades_full.csv')
port_fwd = _stream_from_disk(GOLDEN_BASKET, 'trades_forward.csv')

summary = pd.DataFrame([s for s in [_port_stats(port_is,'IN-SAMPLE'),
                                     _port_stats(port_fwd,'FORWARD')] if s])
print(summary.to_string(index=False))
print()

PORT_DIR = RESULTS_DIR / '_portfolio'
PORT_DIR.mkdir(parents=True, exist_ok=True)
port_is.to_csv(PORT_DIR/'trades_IS.csv', index=False)
port_fwd.to_csv(PORT_DIR/'trades_FWD.csv', index=False)
summary.to_csv(PORT_DIR/'summary.csv', index=False)
(PORT_DIR/'basket.json').write_text(json.dumps({
    'basket': GOLDEN_BASKET,
    'criteria': f'WR_IS >= {TARGET_WR} AND fwd_net >= 0',
}, indent=2))


# =============================================================================
# Step 9 — Consolidated config dump
# =============================================================================
print('=== Step 9: Per-symbol configs ===\n')
all_configs = {}
for sym_dir in sorted(RESULTS_DIR.iterdir()):
    if not sym_dir.is_dir() or sym_dir.name in ('_summary', '_portfolio'):
        continue
    cfg_path = sym_dir / 'config.json'
    if cfg_path.exists():
        try:
            all_configs[sym_dir.name] = json.loads(cfg_path.read_text())
        except Exception as e:
            print(f'  {sym_dir.name}: {e!r}')

(SUMMARY_DIR / 'per_symbol_configs.json').write_text(
    json.dumps(all_configs, indent=2, default=str)
)
print(f'Saved {len(all_configs)} configs → {SUMMARY_DIR/"per_symbol_configs.json"}')
print()
print('========================================')
print('All post-sweep cells ran from disk only.')
print('========================================')
