"""Standalone portfolio aggregator for notebook 24.

Reads per-symbol trades from results/multi_symbol_scalper/<SYMBOL>/trades_*.csv
and computes portfolio-level metrics WITHOUT re-running any sweep.

Usage: python _portfolio_24.py
"""
from pathlib import Path
import pandas as pd
import json

# ── Config ──────────────────────────────────────────────────────────────────
RESULTS_DIR    = Path('./results/multi_symbol_scalper')
SUMMARY_DIR    = RESULTS_DIR / '_summary'
PORT_DIR       = RESULTS_DIR / '_portfolio'
PORT_DIR.mkdir(parents=True, exist_ok=True)

START_CAPITAL  = 1000.0
LOT_SIZE       = 0.02
TARGET_WR      = 48.0


# ── 1. Pick golden basket from the saved ranking ────────────────────────────
ranking = pd.read_csv(SUMMARY_DIR / 'symbol_ranking.csv').rename(columns={'Unnamed: 0':'symbol'})
ranking.set_index('symbol', inplace=True)
GOLDEN_BASKET = ranking[
    (ranking['wr_net_%'] >= TARGET_WR) &
    (ranking['fwd_net_$'].fillna(0) >= 0)
].sort_values('score', ascending=False).index.tolist()
print(f'Golden basket ({len(GOLDEN_BASKET)} symbols): {GOLDEN_BASKET}')


# ── 2. Load each symbol's trade history (IS + Forward) ──────────────────────
def load_symbol_trades(symbol, filename):
    path = RESULTS_DIR / symbol / filename
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df['symbol'] = symbol
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['exit_time']  = pd.to_datetime(df['exit_time'])
    return df


def build_stream(basket, filename):
    parts = []
    for sym in basket:
        df = load_symbol_trades(sym, filename)
        if not df.empty:
            parts.append(df)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True).sort_values('entry_time').reset_index(drop=True)
    df['cum_net_$']  = df['net_$'].cumsum().round(2)
    df['equity_$']   = (START_CAPITAL + df['cum_net_$']).round(2)
    peak             = df['equity_$'].cummax()
    df['drawdown_%'] = ((df['equity_$'] - peak) / peak * 100).round(2)
    return df


port_is  = build_stream(GOLDEN_BASKET, 'trades_full.csv')
port_fwd = build_stream(GOLDEN_BASKET, 'trades_forward.csv')


# ── 3. Portfolio-level stats ────────────────────────────────────────────────
def portfolio_stats(df, label):
    if df.empty:
        return None
    n     = len(df)
    wins  = int((df['net_$'] > 0).sum())
    wr    = wins / n * 100
    gw    = df.loc[df['net_$']>0, 'net_$'].sum()
    gl    = -df.loc[df['net_$']<=0, 'net_$'].sum()
    pf    = gw / gl if gl > 0 else float('inf')
    net   = df['net_$'].sum()
    ret_pct = net / START_CAPITAL * 100
    max_dd  = df['drawdown_%'].min()
    days  = (df['exit_time'].iloc[-1] - df['entry_time'].iloc[0]).days
    return {
        'label'         : label,
        'symbols'       : df['symbol'].nunique(),
        'trades'        : n,
        'WR_%'          : round(wr, 2),
        'avg_R'         : round(df['R'].mean(), 3),
        'profit_factor' : round(pf, 2),
        'net_$'         : round(net, 2),
        'return_%'      : round(ret_pct, 2),
        'max_DD_%'      : round(max_dd, 2),
        'days'          : days,
        'trades_per_day': round(n/max(days,1), 2),
    }


stats_is  = portfolio_stats(port_is,  'IN-SAMPLE')
stats_fwd = portfolio_stats(port_fwd, 'FORWARD')
summary = pd.DataFrame([s for s in [stats_is, stats_fwd] if s])
print()
print('=== Portfolio summary (golden basket) ===')
print(summary.to_string(index=False))
print()


# ── 4. Per-symbol contribution (IS) ─────────────────────────────────────────
contrib = port_is.groupby('symbol').agg(
    trades  = ('net_$', 'size'),
    wins    = ('net_$', lambda x: (x > 0).sum()),
    net_USD = ('net_$', 'sum'),
).sort_values('net_USD', ascending=False)
contrib['WR_%']    = (contrib['wins'] / contrib['trades'] * 100).round(2)
contrib['share_%'] = (contrib['net_USD'] / contrib['net_USD'].sum() * 100).round(1)
contrib['net_USD'] = contrib['net_USD'].round(2)
print('--- Per-symbol contribution (IS) ---')
print(contrib[['trades','wins','WR_%','net_USD','share_%']].to_string())
print()


# ── 5. Per-quarter portfolio WR (IS) ────────────────────────────────────────
qport = port_is.copy()
qport['quarter'] = qport['entry_time'].dt.to_period('Q').astype(str)
q_summary = qport.groupby('quarter').agg(
    trades  = ('net_$', 'size'),
    wins    = ('net_$', lambda x: (x > 0).sum()),
    net_USD = ('net_$', 'sum'),
)
q_summary['WR_%']    = (q_summary['wins'] / q_summary['trades'] * 100).round(2)
q_summary['return_%'] = (q_summary['net_USD'] / START_CAPITAL * 100).round(2)
q_summary['net_USD']  = q_summary['net_USD'].round(2)
print('--- Per-quarter portfolio WR (IS) ---')
print(q_summary[['trades','wins','WR_%','net_USD','return_%']].to_string())
print()


# ── 6. Monthly returns ──────────────────────────────────────────────────────
mport = port_is.copy()
mport['month'] = mport['entry_time'].dt.to_period('M').astype(str)
m_summary = mport.groupby('month').agg(
    trades  = ('net_$', 'size'),
    net_USD = ('net_$', 'sum'),
)
m_summary['return_%'] = (m_summary['net_USD'] / START_CAPITAL * 100).round(2)
m_summary['net_USD']  = m_summary['net_USD'].round(2)
print('--- Monthly returns (IS) ---')
print(m_summary[['trades','net_USD','return_%']].to_string())
print()


# ── 7. Save everything ──────────────────────────────────────────────────────
port_is.to_csv (PORT_DIR / 'trades_IS.csv',  index=False)
port_fwd.to_csv(PORT_DIR / 'trades_FWD.csv', index=False)
summary.to_csv (PORT_DIR / 'summary.csv', index=False)
contrib.to_csv (PORT_DIR / 'contribution_per_symbol.csv')
q_summary.to_csv(PORT_DIR / 'quarterly_returns.csv')
m_summary.to_csv(PORT_DIR / 'monthly_returns.csv')

eq_is = port_is[['exit_time','symbol','net_$','equity_$','drawdown_%']]
eq_is.to_csv(PORT_DIR / 'equity_curve_IS.csv', index=False)
if not port_fwd.empty:
    port_fwd[['exit_time','symbol','net_$','equity_$','drawdown_%']].to_csv(
        PORT_DIR / 'equity_curve_FWD.csv', index=False)

(PORT_DIR / 'basket.json').write_text(json.dumps({
    'basket': GOLDEN_BASKET,
    'criteria': f'WR_IS >= {TARGET_WR} AND fwd_net >= 0',
    'starting_capital': START_CAPITAL,
    'lot_per_trade': LOT_SIZE,
}, indent=2))

print(f'Saved -> {PORT_DIR}/')
