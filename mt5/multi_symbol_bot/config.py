"""
Symbol-basket configuration loader.

Reads the artifacts that `notebooks/24_multi_symbol_scalper.ipynb` writes to
`notebooks/results/multi_symbol_scalper/` and turns them into typed objects
the runner can use directly.

The "golden basket" is computed by `symbol_ranking.csv`:
    WR_IS ≥ TARGET_WR AND fwd_net ≥ 0  → in the basket.

Why not hardcode the symbol list?
    Backtest output is the source of truth. If we re-run the sweep with new
    data, the basket may shift (e.g. EURJPY drops, NZDUSD enters). Reading
    from disk keeps the live bot honest -- it trades exactly the symbols the
    research said were tradable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

from .strategy import StrategyConfig


# Default location of the notebook's results dir (relative to repo root).
DEFAULT_RESULTS_DIR = Path("notebooks/results/multi_symbol_scalper")

# Default basket selection criteria (kept identical to notebook Step 8b).
DEFAULT_TARGET_WR        = 48.0
DEFAULT_MIN_FORWARD_NET  = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# TYPES
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class SymbolStrategyConfig:
    """
    Bundle of strategy config + research stats for one symbol.

    The runner uses `cfg` for live signal detection and `stats` for the
    capital allocator (weighting by score).
    """

    symbol:    str
    cfg:       StrategyConfig
    stats:     dict = field(default_factory=dict)   # row from symbol_ranking.csv


@dataclass
class SymbolBasket:
    """The active live basket plus per-symbol stats and selection criteria."""

    members:   list[SymbolStrategyConfig]
    criteria:  str
    source_dir: Path

    @property
    def symbols(self) -> list[str]:
        return [m.symbol for m in self.members]

    def get(self, symbol: str) -> SymbolStrategyConfig:
        for m in self.members:
            if m.symbol == symbol:
                return m
        raise KeyError(symbol)

    def __len__(self) -> int:
        return len(self.members)

    def __iter__(self):
        return iter(self.members)


# ═══════════════════════════════════════════════════════════════════════════
# LOADER
# ═══════════════════════════════════════════════════════════════════════════


def load_basket(
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
    target_wr:   float = DEFAULT_TARGET_WR,
    min_forward_net: float = DEFAULT_MIN_FORWARD_NET,
    symbols_override: Iterable[str] | None = None,
) -> SymbolBasket:
    """
    Build a SymbolBasket from disk.

    Selection logic (in order of precedence):
        1. If `symbols_override` is given, use exactly those symbols.
        2. Otherwise read `_summary/symbol_ranking.csv` and apply
           `WR_IS >= target_wr AND fwd_net >= min_forward_net`.
        3. If neither yields anything, raises FileNotFoundError -- the user
           must run the notebook's sweep first.

    Each chosen symbol MUST have a `<sym>/config.json` file (saved by Step 5c).
    """
    results_dir = Path(results_dir).resolve()
    if not results_dir.exists():
        raise FileNotFoundError(
            f"Results directory not found: {results_dir}\n"
            f"Run the sweep in notebooks/24_multi_symbol_scalper.ipynb first."
        )

    # ── Decide which symbols to include ──────────────────────────────────────-
    if symbols_override:
        wanted = [s.upper() for s in symbols_override]
        criteria = f"symbols_override={wanted}"
        ranking = _read_ranking(results_dir, optional=True)
    else:
        ranking = _read_ranking(results_dir, optional=False)
        criteria = f"WR_IS >= {target_wr:.1f}% AND fwd_net_$ >= {min_forward_net:.2f}"
        keep = ranking[
            (ranking["wr_net_%"] >= target_wr)
            & (ranking["fwd_net_$"].fillna(0) >= min_forward_net)
        ]
        wanted = keep.sort_values("score", ascending=False).index.tolist()
        if not wanted:
            raise ValueError(
                f"No symbol met criteria ({criteria}). Either lower target_wr "
                f"or pass symbols_override to force a basket."
            )

    # ── Load per-symbol config + stats row ───────────────────────────────────-
    members: list[SymbolStrategyConfig] = []
    missing: list[str] = []
    for sym in wanted:
        cfg_path = results_dir / sym / "config.json"
        if not cfg_path.exists():
            missing.append(sym)
            continue
        try:
            cfg = StrategyConfig.from_dict(json.loads(cfg_path.read_text()))
        except Exception as exc:
            raise RuntimeError(f"{sym}: cannot parse {cfg_path.name}: {exc!r}") from exc

        stats_row = {}
        if ranking is not None and sym in ranking.index:
            stats_row = ranking.loc[sym].to_dict()

        members.append(SymbolStrategyConfig(symbol=sym, cfg=cfg, stats=stats_row))

    if missing:
        raise FileNotFoundError(
            f"Missing per-symbol config.json for: {missing}. "
            f"Run the notebook so these are persisted to {results_dir}/<symbol>/config.json"
        )

    return SymbolBasket(members=members, criteria=criteria, source_dir=results_dir)


# ═══════════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════════


def _read_ranking(results_dir: Path, optional: bool) -> pd.DataFrame | None:
    """Read `_summary/symbol_ranking.csv`. Returns None if absent + optional."""
    path = results_dir / "_summary" / "symbol_ranking.csv"
    if not path.exists():
        if optional:
            return None
        raise FileNotFoundError(
            f"symbol_ranking.csv not found at {path}.\n"
            f"Run notebooks/24_multi_symbol_scalper.ipynb so the ranking is built, "
            f"or use load_basket(..., symbols_override=[...]) to bypass."
        )
    df = pd.read_csv(path)
    # The CSV's index column is the symbol — match the notebook's `.to_csv(...)`
    # behaviour which keeps the index named "symbol".
    if "symbol" in df.columns:
        df = df.set_index("symbol")
    elif df.columns[0] in ("", "Unnamed: 0"):
        df = df.rename(columns={df.columns[0]: "symbol"}).set_index("symbol")
    return df
