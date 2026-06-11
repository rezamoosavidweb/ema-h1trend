"""
Single source of truth for configuration.

Two distinct concerns, deliberately separated:

1.  :class:`FrozenStrategy` — the strategy parameters selected before Phase 4
    (notebook 38). These are **frozen**: changing them violates the Phase-5 mandate
    (no re-optimisation, no curve fitting). They are a frozen dataclass, so any
    accidental mutation raises at runtime. Every value here is annotated with its
    provenance in NB38.

2.  :class:`SystemConfig` — operational/runtime knobs (paths, DB URL, broker creds,
    risk limits, cycle timing). These are environment-driven (``.env`` / env vars)
    and may legitimately differ between the dev box (Windows + SQLite) and the VPS
    (Linux + PostgreSQL).

Nothing in (1) may be overridden by env vars — that is the whole point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# ─────────────────────────────────────────────────────────────────────────────
# (1) FROZEN STRATEGY  — provenance: notebooks/38_statarb_forex_phase4.ipynb
# ─────────────────────────────────────────────────────────────────────────────
# DO NOT EDIT these values to "improve" performance. The Phase-5 objective is to
# validate THIS exact configuration live. If research selects a new parameter set,
# bump `version` and record the new NB provenance — never silently tweak.


@dataclass(frozen=True)
class FrozenStrategy:
    """The exact 'cointegration reversion + carry + regime sizing' config from NB38."""

    version: str = "nb38-phase4-frozen-2026"

    # ── Universe (NB38 cell 1) ──────────────────────────────────────────────
    # G8 majors+crosses, both legs in G8, symbol's data history starts <= 2011.
    timeframe: str = "H1"                 # research engine timeframe key for FX = "FX_H1"
    bars_per_year_key: str = "FX_H1"      # key into engine BARS_PER_YEAR
    g8: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {"USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF"}
        )
    )
    history_start_year_max: int = 2011
    formation_fraction: float = 0.40      # first 40% of panel = formation window

    # ── Pair selection (NB38 cell 1: select_pairs(...)) ─────────────────────
    pairs_top_n: int = 6
    coint_max_p: float = 0.10
    hl_max: float = 4000.0
    hl_min: float = 4.0

    # ── Signal (NB38 cell 1: SignalParams(...)) ─────────────────────────────
    z_entry: float = 2.0
    z_exit: float = 0.0
    z_stop: float = 3.5
    z_window: int = 480

    # ── Sizing (NB38 cell 1: SizeParams(...)) ───────────────────────────────
    target_ann_vol: float = 0.10
    vol_window: int = 480
    max_leverage: float = 3.0

    # ── Hedge ratio estimation ──────────────────────────────────────────────
    hedge: str = "static"                 # NB38 reversion book used static formation beta

    # ── Carry overlay (NB38 cell 3: combine_reversion_carry(w_rev=0.5)) ─────
    carry_enabled: bool = True
    carry_w_rev: float = 0.50             # 50/50 reversion / carry blend
    carry_rebalance: int = 24             # engine default (bars)
    carry_scale: float = 1.0              # illustrative rate table magnitude

    # ── Continuous regime sizing (NB38 cell 17) ─────────────────────────────
    regime_enabled: bool = True
    regime_proxy_symbol: str = "EURUSD"   # falls back to first panel column if absent
    regime_method: str = "hmm"
    regime_min_mult: float = 0.0
    regime_max_mult: float = 1.5

    def as_provenance(self) -> dict:
        """Flat dict for logging into every signal row (audit trail)."""
        return {
            "strategy_version": self.version,
            "timeframe": self.timeframe,
            "pairs_top_n": self.pairs_top_n,
            "coint_max_p": self.coint_max_p,
            "hl_max": self.hl_max,
            "hl_min": self.hl_min,
            "z_entry": self.z_entry,
            "z_exit": self.z_exit,
            "z_stop": self.z_stop,
            "z_window": self.z_window,
            "target_ann_vol": self.target_ann_vol,
            "vol_window": self.vol_window,
            "max_leverage": self.max_leverage,
            "hedge": self.hedge,
            "carry_enabled": self.carry_enabled,
            "carry_w_rev": self.carry_w_rev,
            "regime_enabled": self.regime_enabled,
            "regime_method": self.regime_method,
            "regime_max_mult": self.regime_max_mult,
        }


# A module-level singleton — import this everywhere a strategy param is needed.
STRATEGY = FrozenStrategy()


# ─────────────────────────────────────────────────────────────────────────────
# (2) SYSTEM CONFIG  — operational, environment-driven
# ─────────────────────────────────────────────────────────────────────────────

try:  # pydantic-settings is in requirements; degrade gracefully if absent.
    from pydantic import Field
    from pydantic_settings import BaseSettings, SettingsConfigDict

    _HAVE_PYDANTIC = True
except Exception:  # pragma: no cover
    _HAVE_PYDANTIC = False


def repo_root() -> Path:
    """Project root resolved from this file (statarb_live/ is at repo root)."""
    return Path(__file__).resolve().parents[1]


if _HAVE_PYDANTIC:

    class SystemConfig(BaseSettings):
        """Operational config. Reads from env / .env with prefix ``SAL_``.

        Example: ``SAL_DB_URL=postgresql+psycopg://...`` overrides ``db_url``.
        Broker creds use the existing ``MT5_*`` names (no prefix) for continuity
        with the other bots in this repo.
        """

        model_config = SettingsConfigDict(
            env_prefix="SAL_",
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )

        # ── Mode ────────────────────────────────────────────────────────────
        # 'paper'  -> route fills through ExecutionSimulator (no broker orders)
        # 'live'   -> (future) send demo-account orders via MT5 broker adapter
        mode: Literal["paper", "live"] = "paper"
        broker: Literal["mt5", "sim"] = "mt5"
        dry_run: bool = False
        once: bool = False

        # ── Data / timing ───────────────────────────────────────────────────
        data_dir: str = "notebooks/data"  # H1 CSV cache root (engine data_dir layout)
        timeframe: str = "H1"
        broker_tz: str = "Europe/Nicosia"
        cycle_grace_seconds: int = 30
        warmup_extra_bars: int = 100

        # ── Storage ─────────────────────────────────────────────────────────
        # SQLite on the dev box; set SAL_DB_URL to a postgresql+psycopg URL on the VPS.
        db_url: str = ""                  # empty -> default sqlite under storage_dir
        storage_dir: str = ""             # empty -> <repo>/statarb_live/_data

        # ── Capital / risk ──────────────────────────────────────────────────
        starting_equity: float = 100_000.0
        # Hard risk limits (see risk/controls.py). Fractions of equity.
        max_daily_loss_pct: float = 0.02
        max_weekly_loss_pct: float = 0.05
        max_gross_exposure: float = 4.0   # multiples of equity (matches max 6 pairs * sizing)
        max_position_pct: float = 1.0     # per-pair gross as fraction of equity

        # ── Broker (MT5) — reuse existing names ─────────────────────────────
        mt5_login: int = Field(default=0, alias="MT5_LOGIN")
        mt5_password: str = Field(default="", alias="MT5_PASSWORD")
        mt5_server: str = Field(default="", alias="MT5_SERVER")
        mt5_terminal_path: str = Field(default="", alias="MT5_TERMINAL_PATH")

        # ── Reporting / monitoring ──────────────────────────────────────────
        report_dir: str = ""              # empty -> <repo>/statarb_live/_reports
        log_dir: str = ""                 # empty -> <repo>/logs/statarb_live

        # ── Magic-number space (keep clear of other bots: OB 10M, scalper 24M, pairs 28M)
        magic_base: int = 29_000_000
        comment_prefix: str = "sal_v1"

        # ── Derived paths ───────────────────────────────────────────────────
        def storage_path(self) -> Path:
            p = Path(self.storage_dir) if self.storage_dir else repo_root() / "statarb_live" / "_data"
            p.mkdir(parents=True, exist_ok=True)
            return p

        def resolved_db_url(self) -> str:
            if self.db_url:
                return self.db_url
            return f"sqlite:///{(self.storage_path() / 'statarb_live.db').as_posix()}"

        def report_path(self) -> Path:
            p = Path(self.report_dir) if self.report_dir else repo_root() / "statarb_live" / "_reports"
            p.mkdir(parents=True, exist_ok=True)
            return p

        def log_path(self) -> Path:
            p = Path(self.log_dir) if self.log_dir else repo_root() / "logs" / "statarb_live"
            p.mkdir(parents=True, exist_ok=True)
            return p

        def data_path(self) -> Path:
            d = Path(self.data_dir)
            return d if d.is_absolute() else repo_root() / self.data_dir

else:  # pragma: no cover — minimal fallback if pydantic-settings missing

    @dataclass
    class SystemConfig:  # type: ignore[no-redef]
        mode: str = "paper"
        broker: str = "mt5"
        dry_run: bool = False
        once: bool = False
        data_dir: str = "notebooks/data"
        timeframe: str = "H1"
        broker_tz: str = "Europe/Nicosia"
        cycle_grace_seconds: int = 30
        warmup_extra_bars: int = 100
        db_url: str = ""
        storage_dir: str = ""
        starting_equity: float = 100_000.0
        max_daily_loss_pct: float = 0.02
        max_weekly_loss_pct: float = 0.05
        max_gross_exposure: float = 4.0
        max_position_pct: float = 1.0
        mt5_login: int = 0
        mt5_password: str = ""
        mt5_server: str = ""
        mt5_terminal_path: str = ""
        report_dir: str = ""
        log_dir: str = ""
        magic_base: int = 29_000_000
        comment_prefix: str = "sal_v1"

        def storage_path(self) -> Path:
            p = Path(self.storage_dir) if self.storage_dir else repo_root() / "statarb_live" / "_data"
            p.mkdir(parents=True, exist_ok=True)
            return p

        def resolved_db_url(self) -> str:
            if self.db_url:
                return self.db_url
            return f"sqlite:///{(self.storage_path() / 'statarb_live.db').as_posix()}"

        def report_path(self) -> Path:
            p = Path(self.report_dir) if self.report_dir else repo_root() / "statarb_live" / "_reports"
            p.mkdir(parents=True, exist_ok=True)
            return p

        def log_path(self) -> Path:
            p = Path(self.log_dir) if self.log_dir else repo_root() / "logs" / "statarb_live"
            p.mkdir(parents=True, exist_ok=True)
            return p

        def data_path(self) -> Path:
            d = Path(self.data_dir)
            return d if d.is_absolute() else repo_root() / self.data_dir


def load_config() -> SystemConfig:
    """Construct SystemConfig from environment / .env."""
    return SystemConfig()
