#!/usr/bin/env python3
"""
Pairs-Trading Live MT5 Runner
=============================

Run the cointegration-based statistical-arbitrage strategy from
`notebooks/27_pairs_trading_walkforward.ipynb` against a live MT5 terminal.

╔═══════════════════════════════════════════════════════════════════════════╗
║                                ARCHITECTURE                                ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║   Per cycle (driven by H4 bar close):                                      ║
║     1. Watchdog: ensure MT5 healthy                                        ║
║     2. Fetch aligned H4 panel for all needed symbols                       ║
║     3. For each pair in portfolio:                                         ║
║          - Refit β/α on last 12 months                                     ║
║          - ADF gate (skip if cointegration broke)                          ║
║          - Compute spread + rolling z-score                                ║
║          - Decide action (open/close/hold)                                 ║
║          - Enact via PairsExecutionEngine (two-leg, partial-fill safe)     ║
║     4. Heartbeat (every 10 min)                                            ║
║     5. Sleep until next bar close                                          ║
║                                                                            ║
║   Logging:   logs/pairs_trading/pairs-YYYY-MM-DD.json (JSON-lines)        ║
║   State:     logs/pairs_trading/state.json (open positions)                ║
║   Magic:     MAGIC_BASE = 28_000_000  → one offset per pair               ║
║   Comment:   "pairs_v1:<pair>:<leg>"  in every order                       ║
║                                                                            ║
╚═══════════════════════════════════════════════════════════════════════════╝

Run examples:
    # ALWAYS start with dry-run for 1–2 weeks on demo
    python mt5/run_pairs_trading.py --dry-run --once
    python mt5/run_pairs_trading.py --dry-run

    # After confidence:
    python mt5/run_pairs_trading.py --once
    python mt5/run_pairs_trading.py

    # Subset of portfolio (override CSV):
    python mt5/run_pairs_trading.py --dry-run --pairs EURCHF~GBPJPY NZDCAD~EURUSD

    # Different portfolio CSV (e.g. produced by a re-run of NB28):
    python mt5/run_pairs_trading.py --portfolio-csv path/to/portfolio_selected_H4.csv

Env vars (optional):
    MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_TERMINAL_PATH
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

try:
    import MetaTrader5 as mt5
except ImportError:
    print("Install MetaTrader5: pip install MetaTrader5", file=sys.stderr)
    raise

# Make sibling packages importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from execution.mt5_watchdog import Mt5Watchdog, WatchdogConfig                  # noqa: E402
from execution.structured_logger import StructuredLogger                        # noqa: E402

from mt5.pairs_trading import (                                                 # noqa: E402
    PairsConfig, PortfolioSpread,
    PairsExecutionEngine, PairsRunner, PairsStateStore,
    assign_magic_numbers, recover_state_from_broker,
    load_portfolio_from_csv, filter_portfolio_by_keys,
    default_portfolio_csv, default_log_dir, default_state_file,
)


# ─────────────────────────────────────────────────────────────────────────────
# Optional Telegram notifier
# ─────────────────────────────────────────────────────────────────────────────


def _maybe_build_notifier(logger: StructuredLogger):
    """Build a Mt5Notifier if env vars are set; otherwise return None."""
    try:
        from telegram_bot.mt5_notifier import Mt5Notifier   # type: ignore
    except ImportError:
        return None
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat  = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        logger.event("telegram_disabled", reason="env_not_set")
        return None
    try:
        return Mt5Notifier(bot_token=token, chat_id=chat)
    except Exception as exc:
        logger.error("telegram_init_failed", exc=exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MT5 connection
# ─────────────────────────────────────────────────────────────────────────────


def mt5_connect() -> None:
    """Open the MT5 IPC connection, using env vars when present."""
    kwargs: dict = {}
    login    = os.environ.get("MT5_LOGIN")
    password = os.environ.get("MT5_PASSWORD")
    server   = os.environ.get("MT5_SERVER")
    path     = os.environ.get("MT5_TERMINAL_PATH")
    if login and password and server:
        kwargs["login"]    = int(login)
        kwargs["password"] = password
        kwargs["server"]   = server
    if path:
        kwargs["path"] = path
    if not mt5.initialize(**kwargs):
        raise RuntimeError(f"mt5.initialize failed: {mt5.last_error()}")


def assert_terminal_ready() -> None:
    ti = mt5.terminal_info()
    if ti is None:
        raise RuntimeError(
            f"terminal_info() returned None: {mt5.last_error()}\n"
            "Open MT5, log in, enable AutoTrading, then retry."
        )
    if not ti.connected:
        raise RuntimeError(
            "MT5 terminal not connected to broker — wait for quotes in Market Watch."
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_pairs_trading",
        description="Pairs-trading live MT5 runner. See README for full operator manual.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--tf", choices=("H1", "H4"), default="H4",
                   help="Timeframe (default: H4 — recommended)")
    p.add_argument("--once", action="store_true",
                   help="Run a single cycle and exit (else loop forever)")
    p.add_argument("--dry-run", action="store_true",
                   help="Evaluate + log signals but DO NOT send orders")
    p.add_argument("--pairs", nargs="+", default=None,
                   help="Subset of portfolio pair_keys to run (e.g. EURCHF~GBPJPY)")
    p.add_argument("--portfolio-csv", type=Path, default=None,
                   help="Override portfolio CSV (defaults to NB28 output for the TF)")
    p.add_argument("--risk-per-leg-pct", type=float, default=0.10,
                   help="Equity %% at risk per LEG (default 0.10 — extremely conservative)")
    p.add_argument("--max-open-pairs", type=int, default=4,
                   help="Cap on simultaneously open pairs (default 4 = NB28 portfolio size)")
    p.add_argument("--log-dir", type=Path, default=None,
                   help="Override log directory (default logs/pairs_trading/)")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> PairsConfig:
    return PairsConfig(
        timeframe        = args.tf,
        dry_run          = args.dry_run,
        once             = args.once,
        risk_per_leg_pct = args.risk_per_leg_pct,
        max_open_pairs   = args.max_open_pairs,
        portfolio_csv    = args.portfolio_csv or default_portfolio_csv(args.tf),
        log_dir          = args.log_dir       or default_log_dir(),
    )


def load_portfolio(cfg: PairsConfig, requested_keys: Optional[list[str]]) -> list[PortfolioSpread]:
    spreads = load_portfolio_from_csv(cfg.portfolio_csv)
    if requested_keys:
        spreads = filter_portfolio_by_keys(spreads, requested_keys)
    if not spreads:
        raise RuntimeError(
            f"portfolio is empty (csv={cfg.portfolio_csv}, pairs={requested_keys})"
        )
    return spreads


def main() -> int:
    args = parse_args()
    cfg = build_config(args)

    log_dir = cfg.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = StructuredLogger(
        symbol="pairs",                          # used as filename prefix
        log_path=log_dir / "pairs.json",
        rotate_daily=True,
    )

    # bot_start envelope — useful for incident triage
    logger.event(
        "bot_start",
        tf               = cfg.timeframe,
        dry_run          = cfg.dry_run,
        once             = cfg.once,
        portfolio_csv    = str(cfg.portfolio_csv),
        risk_per_leg_pct = cfg.risk_per_leg_pct,
        max_open_pairs   = cfg.max_open_pairs,
        entry_z          = cfg.entry_z,
        exit_z           = cfg.exit_z,
        stop_z           = cfg.stop_z,
        adf_gate_p       = cfg.adf_gate_p,
        train_months     = cfg.train_months,
        z_window_bars    = cfg.z_window_bars,
        requested_pairs  = args.pairs,
    )

    # 1) Load portfolio FIRST so we fail fast on a bad CSV before touching MT5
    try:
        portfolio = load_portfolio(cfg, args.pairs)
    except Exception as exc:
        logger.error("portfolio_load_failed", exc=exc)
        return 2

    magic_for_pair = assign_magic_numbers(portfolio)
    logger.event(
        "portfolio_loaded",
        n_spreads = len(portfolio),
        pair_keys = [s.key for s in portfolio],
        magics    = magic_for_pair,
    )

    # 2) Connect MT5 + watchdog
    try:
        mt5_connect()
        assert_terminal_ready()
    except Exception as exc:
        logger.error("mt5_startup_failed", exc=exc)
        return 3

    ai = mt5.account_info()
    if ai:
        logger.event(
            "account_info",
            login=ai.login, server=ai.server, currency=ai.currency,
            balance=float(ai.balance), equity=float(ai.equity),
            margin=float(ai.margin), free_margin=float(ai.margin_free),
        )

    watchdog = Mt5Watchdog(
        logger=logger,
        connect_fn=mt5_connect,
        config=WatchdogConfig(),
    )

    # 3) State store + recovery
    state_store = PairsStateStore(default_state_file())
    state_store.load()
    if not cfg.dry_run:
        try:
            recover_state_from_broker(state_store, magic_for_pair, logger)
        except Exception as exc:
            logger.error("state_recovery_failed", exc=exc)

    # 4) Build engine + runner
    notifier = _maybe_build_notifier(logger)
    engine = PairsExecutionEngine(
        cfg            = cfg,
        logger         = logger,
        state_store    = state_store,
        magic_for_pair = magic_for_pair,
        dry_run        = cfg.dry_run,
        notifier       = notifier,
    )
    runner = PairsRunner(
        cfg         = cfg,
        portfolio   = portfolio,
        logger      = logger,
        state_store = state_store,
        engine      = engine,
        watchdog    = watchdog,
    )

    # 5) Run
    exit_code = 0
    try:
        if cfg.once:
            runner.run_once()
        else:
            runner.run_forever()
    except KeyboardInterrupt:
        logger.event("bot_stop", reason="KeyboardInterrupt")
    except Exception as exc:
        logger.error("bot_stop", exc=exc, reason="unhandled_exception")
        exit_code = 1
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass
        logger.event("bot_stop_clean", code=exit_code)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
