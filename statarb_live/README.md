# statarb_live — Phase 5: Live Paper-Trading Validation System

A production-quality, modular harness that runs the **frozen** FX statistical-arbitrage
strategy selected before Phase 4 (notebook 38) against a Forex demo account, continuously,
for **research validation** — not profit maximisation.

> **Frozen strategy:** cointegration reversion **+** carry overlay **+** continuous regime
> sizing. No parameter changes. No re-optimisation. No new alpha. The strategy maths is
> imported wholesale from the research engine (`notebooks/statarb/`) so live signals are
> bit-for-bit faithful to the backtest — `statarb_live` only adds the operational layers.

## Why this exists

The research (NB35–38) found a **real but thin** FX edge: reversion OOS Sharpe ≈ 0.29,
≈ 0.49 blended 50/50 with a carry sleeve (corr +0.17), but the Deflated Sharpe (0.65) sits
below the 0.95 bar and CPCV gives a low-positive, wide distribution (mean ≈ +0.1, p05 < 0).
The verdict was *"investable only after a small, ring-fenced pilot."* This system **is** that
pilot harness: it runs the exact book live on a demo account and records everything needed to
decide, after 2–3 months, whether the edge survives real execution.

## Architecture

```
statarb_live/
  config.py            FrozenStrategy (NB38 params) + SystemConfig (env-driven)
  engine_bridge.py     the ONLY gateway to notebooks/statarb (faithful import)
  data_feed/           aligned bars, gap/tz handling, symbol validation, persistence
  signal_engine/       universe selection (frozen) + per-cycle explainable signals
  portfolio_engine/    lot sizing + exposure caps + book risk metrics
  execution_simulator/ paper fills (spread+slippage+latency) + position ledger w/ attribution
  broker_adapter/      broker-agnostic interface (+ MT5 + sim backends)
  risk/                hard limits (daily/weekly loss, exposure, position size)
  monitoring/          metrics + charts (read-side)
  reporting/           daily/weekly/monthly/final HTML + CSV + (optional) PDF
  storage/             pluggable persistence — SQLite (dev) / PostgreSQL (VPS), one URL
  orchestrator.py      the cycle loop (+ deterministic historical replay)
  cli.py               `python -m statarb_live ...`
  deploy/              Dockerfile, docker-compose (Postgres), .env.example
```

Every cycle persists: market bars, **every signal** (z, β, carry, regime, target, confidence),
positions, fills (intended vs actual + slippage + latency), closed trades (with reversion /
carry / cost PnL attribution), equity + exposure snapshots, and an append-only event log.

## Quick start (local, SQLite + sim broker)

```bash
pip install -r requirements.txt           # adds scikit-learn, SQLAlchemy, psycopg, jinja2

# show resolved config + the frozen universe
python -m statarb_live info

# one paper cycle on the latest bars
SAL_BROKER=sim python -m statarb_live once

# deterministic historical replay (exercises the live code path on cached data)
SAL_BROKER=sim python -m statarb_live replay --start 2025-01-01 --end 2025-06-01 --step 4

# generate reports from whatever is in the DB
python -m statarb_live report --period monthly
python -m statarb_live report --period final     # adds backtest-vs-live comparison
```

The frozen pair book is selected once (NB38 formation window) and cached to
`statarb_live/_data/universe.json`. Re-selection only happens with `reselect`.

## Deployment

### Linux VPS (paper / sim + Postgres) — Docker

```bash
cp statarb_live/deploy/.env.example statarb_live/deploy/.env   # edit secrets
docker compose -f statarb_live/deploy/docker-compose.yml --env-file statarb_live/deploy/.env up -d --build
docker compose -f statarb_live/deploy/docker-compose.yml run --rm reporter   # on-demand report
```

### Live MT5 demo on Windows

`MetaTrader5` is **Windows-only** and cannot run in a Linux container. For live demo
execution, run natively on a Windows VPS with MT5 open + AutoTrading on:

```powershell
$env:SAL_BROKER="mt5"; $env:SAL_MODE="paper"   # paper still simulates fills; mt5 only feeds data
python -m statarb_live run
```

Set `MT5_LOGIN/PASSWORD/SERVER/TERMINAL_PATH` in `.env`. Point `SAL_DB_URL` at the VPS
Postgres if you want a single shared store; otherwise it falls back to local SQLite.

## Storage backends

One `SqlStorage` class serves both — the URL decides the engine:
- dev: `sqlite:///statarb_live/_data/statarb_live.db` (default)
- VPS: `postgresql+psycopg://user:pass@host:5432/statarb` (set `SAL_DB_URL`)

## Risk controls

Hard limits that can only shrink the book (never alter strategy params). On breach: stop
opening new positions, keep monitoring, emit a `risk_breach` event. Limits in `.env`:
`SAL_MAX_DAILY_LOSS_PCT`, `SAL_MAX_WEEKLY_LOSS_PCT`, `SAL_MAX_GROSS_EXPOSURE`,
`SAL_MAX_POSITION_PCT`.

## Success criteria (the final report)

After 2–3 months, `report --period final` compares **backtest vs walk-forward vs live**
(reference Sharpes from NB37/38 are embedded as validation targets) and quantifies
performance decay, slippage/execution drag (the `cost_pnl` bucket), and regime dependency —
to answer the only question that matters: *is the strategy investable?*

## Caveats (read these)

- **Carry magnitudes are illustrative.** The engine has no live swap feed — carry uses an
  approximate G8 policy-rate table (NB38 caveat). Carry PnL here is directional, not precise.
- **MT5 timestamps are Europe/Nicosia**, not UTC (repo-wide gotcha) — handled in the adapters.
- **The edge is thin.** Expect periods of negative PnL; the point is measurement, not return.
