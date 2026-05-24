# Multi-Symbol Reaction Scalper — Live MT5 Bot

Live execution layer for the strategy researched in
[`notebooks/24_multi_symbol_scalper.ipynb`](../../notebooks/24_multi_symbol_scalper.ipynb).

The notebook produces, for each tested symbol:
- A frozen `config.json` (mode, confirms, RSI memory, ATR-min, ADX-min,
  H1-RSI alignment, MACD alignment, session)
- A ranking row in `symbol_ranking.csv`

This bot **reads those artifacts** from `notebooks/results/multi_symbol_scalper/`
and trades the symbols that pass the WR/forward-net filter, with capital split
across the basket.

---

## Run

```bash
# Default: trade every "golden" symbol (WR_IS ≥ 48 % AND fwd_net ≥ 0)
python mt5/run_multi_scalper.py

# One cycle then exit (good for cron / first deploy)
python mt5/run_multi_scalper.py --once

# Detect + notify, never send orders
python mt5/run_multi_scalper.py --dry-run

# Force an explicit basket (skips ranking-based selection)
python mt5/run_multi_scalper.py --symbols GBPUSD XAUUSD EURJPY

# Score-weighted capital (high-score symbols get more risk_per_trade)
python mt5/run_multi_scalper.py --policy score
```

Environment variables (optional, forwarded to `mt5.initialize`):
`MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER`, `MT5_TERMINAL_PATH`.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│  mt5/run_multi_scalper.py            ← main process                │
│  ──────────────────────────                                        │
│  1. load_basket() ──────► notebooks/results/.../symbol_ranking.csv │
│                          ├──► <SYM>/config.json   (per-symbol cfg) │
│  2. CapitalAllocator ──► per-symbol risk_per_trade                 │
│  3. build_contexts() ──► [SymbolContext × N]                       │
│       │                                                            │
│       ├─ Strategy(cfg)                                             │
│       ├─ ExecutionEngine(symbol, magic, risk_per_trade, ...)       │
│       ├─ Mt5Notifier (telegram)                                    │
│       └─ seen_signals (dedup across restarts)                      │
│                                                                    │
│  loop:                                                             │
│    sleep_until_next_m5()                                           │
│    for ctx in contexts:                                            │
│        run_symbol_cycle(ctx)  ── fetch M5/H1/D1 → detect → execute │
│    write_portfolio_event("cycle_complete", ...)                    │
└────────────────────────────────────────────────────────────────────┘
```

## Package layout

```
mt5/multi_symbol_bot/
├── __init__.py        — public surface
├── strategy.py        — Strategy, StrategyConfig, Signal + all indicators
├── config.py          — SymbolBasket loader (reads notebook artifacts)
├── allocator.py       — CapitalAllocator (equal / score / custom policies)
└── README.md          — you are here

mt5/run_multi_scalper.py   — entrypoint that wires everything together
```

## Strategy — what runs every M5 close

For each symbol the bot fetches the last 600 M5 + 250 H1 + 120 D1 bars (UTC-
corrected from broker wall-clock) and computes:

| Layer | What it does |
|---|---|
| **Trend gate** | H1 and D1 EMA50 must agree on direction (else skip). |
| **Session gate** | NY-local hours from the config (e.g. NY 08-13). |
| **ATR-min**   | Optional: ATR ≥ N× its 500-bar median (skip dead-vol regimes). |
| **ADX-min**   | Optional: ADX ≥ threshold (skip ranging markets). |
| **H1 RSI align** | Optional: H1 RSI > 50 for long, < 50 for short. |
| **MACD align** | Optional: M5 MACD histogram sign matches direction. |
| **Reaction filters** | Per-mode combination of `f_bb`, `f_ema`, `f_rsi`, `f_candle`, `f_rsiR`. |
| **Signal modes** | `RSI-gated` (RSI + any confirm) / `RSI-gated-AND` (RSI + all confirms) / `OR` (≥ min_reactions confirms). |

When a signal fires on the last closed bar, the bot builds:
- **entry** = close of that bar
- **SL** = swing low/high of last 12 bars ± `0.10 × ATR`
- **TP** = entry + `2 × R` (RR = 2)
- Skipped if `R > 5 × ATR` (degenerate setup)

This pipeline is byte-for-byte the same logic the notebook validates against
3 years of history — `_post_sweep_24.py` produces those CSVs from the same code.

## Capital allocation

`CapitalAllocator(total_portfolio_risk=0.02)` means **2 % of the account is at
risk when every symbol holds a 1× position simultaneously**. The allocator
divides that 2 % across the basket:

- **equal** — `0.02 / N` per symbol (default).
- **score** — proportional to `symbol_ranking.csv` `score`; capped at 40 %
  to keep one strong symbol from dominating.
- **custom** — caller supplies a `{symbol: weight}` mapping.

Each symbol's allocation is passed to its `ExecutionEngine` as `risk_per_trade`,
so MT5's `order_calc_profit` math handles contract-size and currency conversion
correctly.

## Logging

Every event is JSON-Lines, append-only, daily-rotated:

| Path | Owner | Contents |
|---|---|---|
| `logs/<SYMBOL>.json` | `ExecutionEngine` per symbol | bot_start, cycle, signal, order_placed, broker_validation_failed, position_closed, heartbeat, … |
| `logs/multi_symbol_scalper.json` | runner aggregate | portfolio_start, cycle_complete (with per-symbol summary), portfolio_stop |
| `logs/seen_signals_multi/<SYMBOL>.json` | dedup store | UTC-timestamped (bar_time, direction) tuples — never re-trade the same bar |

Grep on `event=signal symbol=XAUUSD` from any day to reconstruct strategy
behaviour; `event=cycle_complete` gives the portfolio heartbeat.

## Safety / production checklist

The bot inherits all the guards from `execution.ExecutionEngine`:
- Pre-flight validation (terminal connected, AutoTrading on, spread ≤ max, stops/freeze levels).
- Duplicate-order protection (one pending OR position per `(bar_time, direction)`).
- Retry cap per OB key + cooldown after repeated broker failures.
- Stale pending sweep every cycle.
- LIMIT → MARKET fallback for slippage > threshold.
- MT5 watchdog with auto-reconnect.

It adds:
- Per-symbol seen-signal persistence (survives restarts).
- Per-symbol log file (one symbol's broker error never poisons another's log).
- One try/except per symbol in the cycle loop — a single symbol's exception
  never blocks the rest of the portfolio.

## Tweaking after the live data tells you something

The whole pipeline is config-driven. If you decide to drop a symbol:

1. **Forcibly drop it now**:
   `python mt5/run_multi_scalper.py --symbols GBPUSD XAUUSD EURJPY EURCAD ...`

2. **Or re-tune the basket**: edit `SYMBOLS_CFG` and re-run the sweep in
   `notebooks/24_multi_symbol_scalper.ipynb`, then restart the bot — the
   new basket comes from disk automatically.

No code changes needed for either path.
