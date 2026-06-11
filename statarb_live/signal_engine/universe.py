"""
Universe / pair selection — run ONCE on the formation window, then frozen.

Mirrors NB38 cell 1 exactly:
  * candidate symbols = G8 majors+crosses whose history starts <= 2011 (both legs in G8);
  * build an inner-joined close panel;
  * formation window = first ``formation_fraction`` (40%) of the panel;
  * ``select_pairs(form, top_n=6, coint_max_p=0.10, hl_max=4000, hl_min=4)``.

The selected pairs (with their formation betas/alphas) are persisted to JSON so the live
runner uses the SAME book across restarts — re-selecting every boot would be a silent
parameter drift. Re-selection only happens on an explicit ``--reselect`` request.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..config import STRATEGY
from ..engine_bridge import eng_data, eng_pairs

_PAIR_FIELDS = ["a", "b", "corr", "beta", "alpha", "coint_p", "half_life", "cluster"]


def pair_key(a: str, b: str) -> str:
    return f"{a}~{b}"


@dataclass
class Universe:
    """The frozen tradable book + the symbols needed to compute every sleeve."""
    pairs: list                       # list[eng_pairs.Pair]
    carry_symbols: list[str]          # FX universe used by the carry sleeve
    selected_at: str
    formation_start: str
    formation_end: str

    @property
    def pair_keys(self) -> list[str]:
        return [pair_key(p.a, p.b) for p in self.pairs]

    @property
    def reversion_symbols(self) -> list[str]:
        out: list[str] = []
        for p in self.pairs:
            for s in (p.a, p.b):
                if s not in out:
                    out.append(s)
        return out

    def all_symbols(self) -> list[str]:
        out = list(self.reversion_symbols)
        for s in self.carry_symbols:
            if s not in out:
                out.append(s)
        if STRATEGY.regime_proxy_symbol not in out:
            out.append(STRATEGY.regime_proxy_symbol)
        return out

    # ── persistence ─────────────────────────────────────────────────────────
    def to_json(self) -> dict:
        return {
            "strategy_version": STRATEGY.version,
            "selected_at": self.selected_at,
            "formation_start": self.formation_start,
            "formation_end": self.formation_end,
            "carry_symbols": self.carry_symbols,
            "pairs": [{f: getattr(p, f) for f in _PAIR_FIELDS} for p in self.pairs],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2, default=str), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Universe":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        pairs = [eng_pairs.Pair(**{f: row.get(f) for f in _PAIR_FIELDS}) for row in d["pairs"]]
        return cls(pairs=pairs, carry_symbols=d.get("carry_symbols", []),
                   selected_at=d.get("selected_at", ""),
                   formation_start=d.get("formation_start", ""),
                   formation_end=d.get("formation_end", ""))


# ──────────────────────────────────────────────────────────────────────────────


def _g8_long_history_symbols(data_dir: str, timeframe: str) -> list[str]:
    """G8-only FX symbols (both legs in G8) whose data starts on/before the cutoff year."""
    fx = eng_data.classify_fx(eng_data.discover_symbols(data_dir, timeframe))["fx"]
    out = []
    for s in fx:
        if len(s) != 6:
            continue
        if s[:3] in STRATEGY.g8 and s[3:] in STRATEGY.g8:
            try:
                start = eng_data.load_ohlcv(s, timeframe, data_dir).index.min()
            except FileNotFoundError:
                continue
            if start is not None and start.year <= STRATEGY.history_start_year_max:
                out.append(s)
    return out


def select_universe(data_dir: str, timeframe: str = "H1") -> Universe:
    """Run the NB38 formation-window pair selection from scratch."""
    longs = _g8_long_history_symbols(data_dir, timeframe)
    if len(longs) < 4:
        raise RuntimeError(
            f"only {len(longs)} G8 long-history symbols found under {data_dir} — "
            "cannot reproduce the NB38 universe"
        )
    px, _dv, _reports = eng_data.build_panel(
        longs, timeframe, how="inner", data_dir=data_dir, min_obs=5000
    )
    cut = int(len(px) * STRATEGY.formation_fraction)
    form = px.iloc[:cut]
    pairs = eng_pairs.select_pairs(
        form, top_n=STRATEGY.pairs_top_n, coint_max_p=STRATEGY.coint_max_p,
        hl_max=STRATEGY.hl_max, hl_min=STRATEGY.hl_min,
    )
    if not pairs:
        raise RuntimeError("select_pairs returned no cointegrated pairs on the formation window")
    return Universe(
        pairs=pairs,
        carry_symbols=list(px.columns),       # carry sleeve trades the full G8 panel
        selected_at=datetime.now(timezone.utc).isoformat(),
        formation_start=str(form.index.min()),
        formation_end=str(form.index.max()),
    )


def load_or_select_universe(data_dir: str, storage_dir: Path, *,
                            timeframe: str = "H1", reselect: bool = False) -> Universe:
    """Load the persisted frozen universe, or select+persist it on first run."""
    path = Path(storage_dir) / "universe.json"
    if path.exists() and not reselect:
        return Universe.load(path)
    uni = select_universe(data_dir, timeframe)
    uni.save(path)
    return uni
