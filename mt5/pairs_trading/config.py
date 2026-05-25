"""
Configuration for the pairs-trading live runner.

Everything tunable lives here. Two dataclasses:

    PortfolioSpread   -- one entry per pair to trade (loaded from CSV)
    PairsConfig       -- global thresholds + costs + risk + paths

The CSV format matches the output of notebook 28
(`notebooks/data/stat_arb/portfolio_selected_{TF}.csv`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Magic number space — keep WELL clear of other strategies
# ─────────────────────────────────────────────────────────────────────────────
# OB strategy uses ~10_000_000
# Multi-symbol scalper uses 24_000_000
# Pairs trading uses 28_000_000  → MAGIC_BASE + index (0..N-1)
MAGIC_BASE: int = 28_000_000

# Order comment prefix. Visible in MT5 history → easy strategy attribution.
COMMENT_PREFIX: str = "pairs_v1"

# Broker reports timestamps as wall-clock-encoded-as-UTC; real tz is Cyprus.
# Same convention as run_multi_scalper.py — keep them consistent across bots.
BROKER_TZ_NAME: str = "Europe/Nicosia"

# Timeframes the runner understands (just H1 / H4 for now)
TF_HOURS: dict[str, int] = {"H1": 1, "H4": 4}


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio definition (loaded from notebook 28 output)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PortfolioSpread:
    """
    One cointegrated pair to trade.

    Fields after construction are NEVER mutated. β/α are refit each cycle
    inside `signals.refit_beta()` — these values are only initial seeds /
    sanity-check defaults.
    """

    y: str                        # base leg symbol (long-side when LONG spread)
    x: str                        # hedge leg symbol
    initial_beta: float           # β from NB28 (refit each cycle anyway)
    initial_alpha: float = 0.0    # α from NB28 (recomputed)
    half_life_bars: float = 80.0  # OU half-life in BARS (TF-scaled)
    historical_sharpe: float = 0.0   # walk-forward Sharpe (informational)
    historical_pnl: float = 0.0      # walk-forward total PnL (informational)

    @property
    def key(self) -> str:
        """Stable identifier used in logs, state file, and order comments."""
        return f"{self.y}~{self.x}"

    @property
    def symbols(self) -> tuple[str, str]:
        return (self.y, self.x)


def load_portfolio_from_csv(
    csv_path: Path,
    half_life_default_bars: float = 80.0,
) -> list[PortfolioSpread]:
    """
    Read `portfolio_selected_{TF}.csv` from notebook 28 and build the spread list.

    Required columns: y, x, sharpe, total_pnl
    Optional:         alpha, beta, half_life_bars  (else seeded with defaults)
    """
    df = pd.read_csv(csv_path)

    required = {"y", "x"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"portfolio CSV {csv_path} missing columns: {missing}")

    spreads: list[PortfolioSpread] = []
    for _, row in df.iterrows():
        spreads.append(PortfolioSpread(
            y                 = str(row["y"]),
            x                 = str(row["x"]),
            initial_beta      = float(row.get("beta", 0.0)),
            initial_alpha     = float(row.get("alpha", 0.0)),
            half_life_bars    = float(row.get("half_life_bars", half_life_default_bars)),
            historical_sharpe = float(row.get("sharpe",     0.0)),
            historical_pnl    = float(row.get("total_pnl",  0.0)),
        ))
    return spreads


def filter_portfolio_by_keys(
    spreads: Sequence[PortfolioSpread],
    keys: Iterable[str],
) -> list[PortfolioSpread]:
    """Keep only spreads whose `.key` is in `keys` (for --pairs CLI override)."""
    wanted = set(keys)
    out = [s for s in spreads if s.key in wanted]
    missing = wanted - {s.key for s in out}
    if missing:
        raise ValueError(
            f"--pairs requested {sorted(missing)} but they are not in the portfolio. "
            f"Available: {[s.key for s in spreads]}"
        )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Runner config
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PairsConfig:
    """
    Global config for the runner. All tunables in one place.

    Defaults match notebook 27/28 — bring them up when the strategy proves
    itself in demo, never down without re-running a walk-forward.
    """

    # ─── Universe & timing ──────────────────────────────────────────────────
    timeframe: str = "H4"                # H1 or H4
    cycle_extra_grace_seconds: int = 30  # wait after bar close before evaluating
    train_months: int = 12               # β refit window (mirror NB27)
    z_window_bars: int = 200             # rolling z-score window
    bars_warmup_buffer: int = 50         # extra bars fetched beyond train+z

    # ─── Strategy thresholds (DO NOT TUNE without walk-forward) ─────────────
    entry_z: float = 2.0
    exit_z:  float = 0.5
    stop_z:  float = 4.0
    time_stop_bars_mult: int = 4         # time-stop = mult × half_life_bars

    # ─── ADF gate (skip pair if cointegration breaks in current train window)
    adf_gate_p: float = 0.10

    # ─── Risk ───────────────────────────────────────────────────────────────
    risk_per_leg_pct: float = 0.10       # % of equity at risk per LEG (so ~0.2% per pair)
    assumed_stop_pct: float = 0.01       # for lot sizing — 1% adverse y-leg move
    max_open_pairs:   int   = 4          # hard cap (matches NB28 portfolio size)

    # ─── Heartbeat ──────────────────────────────────────────────────────────
    heartbeat_interval_seconds: int = 10 * 60

    # ─── Path overrides (None → defaults under repo root) ───────────────────
    portfolio_csv:   Path | None = None    # falls back to NB28 output
    log_dir:         Path | None = None    # falls back to logs/pairs_trading/
    state_file:      Path | None = None    # falls back to <log_dir>/state.json

    # ─── Operational ────────────────────────────────────────────────────────
    dry_run: bool = False                  # if True, no orders sent
    once:    bool = False                  # run one cycle then exit


# ─────────────────────────────────────────────────────────────────────────────
# Path resolution
# ─────────────────────────────────────────────────────────────────────────────


def repo_root() -> Path:
    """Resolve project root from this file's location."""
    return Path(__file__).resolve().parents[2]


def default_portfolio_csv(tf: str) -> Path:
    return repo_root() / "notebooks" / "data" / "stat_arb" / f"portfolio_selected_{tf}.csv"


def default_log_dir() -> Path:
    return repo_root() / "logs" / "pairs_trading"


def default_state_file() -> Path:
    return default_log_dir() / "state.json"


def default_seen_signals_file() -> Path:
    return default_log_dir() / "seen_signals.json"
