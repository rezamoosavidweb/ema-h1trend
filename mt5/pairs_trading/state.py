"""
Persistent state store for open pair positions.

Why?
    After a restart we MUST know which pairs are currently open and which
    MT5 tickets back them — otherwise the runner would re-open positions on
    top of existing ones, or fail to close them on EXIT.

Storage:
    A single JSON file (default `logs/pairs_trading/state.json`) overwritten
    atomically after every change. One snapshot is enough — historical
    transitions are reconstructable from the structured event log.

Schema:
    {
      "schema":   "pairs_state_v1",
      "updated":  "<ISO-8601 UTC>",
      "pairs": {
        "EURCHF~GBPJPY": {
          "side":           "long",
          "y_symbol":       "EURCHF",
          "x_symbol":       "GBPJPY",
          "y_ticket":       123456,
          "x_ticket":       123457,
          "y_volume":       0.10,
          "x_volume":       0.02,
          "y_entry_price":  1.0710,
          "x_entry_price":  187.45,
          "beta_at_open":   -0.197,
          "alpha_at_open":   2.31,
          "spread_at_open":  0.0014,
          "z_at_open":      -2.27,
          "opened_at":      "2026-05-25T12:00:00+00:00",
          "opened_bar":     "2026-05-25T12:00:00+03:00"
        }
      }
    }

Public API:
    PairsStateStore(file_path)
        .load()            -> dict[pair_key, PairState]
        .save_all(states)  -> None
        .add(state)        -> None
        .remove(pair_key)  -> Optional[PairState]
        .get(pair_key)     -> Optional[PairState]
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .signals import Side


SCHEMA_VERSION = "pairs_state_v1"


@dataclass
class PairState:
    """One row in the state table — everything needed to close the pair later."""

    pair_key:        str
    side:            Side
    y_symbol:        str
    x_symbol:        str
    y_ticket:        int
    x_ticket:        int
    y_volume:        float
    x_volume:        float
    y_entry_price:   float
    x_entry_price:   float
    beta_at_open:    float
    alpha_at_open:   float
    spread_at_open:  float
    z_at_open:       float
    opened_at:       str           # ISO-8601 UTC
    opened_bar:      str           # ISO-8601 bar-close (BROKER_TZ)
    bars_in_position: int = 0      # incremented by runner on each cycle

    def to_dict(self) -> dict:
        d = asdict(self)
        d["side"] = self.side.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PairState":
        d = dict(d)
        d["side"] = Side(d["side"])
        return cls(**d)


class PairsStateStore:
    """
    File-backed dict of {pair_key -> PairState}. Atomic writes (write to
    temp then `os.replace`) so a crash mid-write cannot corrupt the file.
    """

    def __init__(self, file_path: Path) -> None:
        self.path = Path(file_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, PairState] = {}
        self._loaded = False

    # ── persistence ─────────────────────────────────────────────────────────

    def load(self) -> dict[str, PairState]:
        """Load from disk (or return empty dict if file missing/corrupt)."""
        self._cache = {}
        if self.path.exists():
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
                if payload.get("schema") == SCHEMA_VERSION:
                    for key, raw in payload.get("pairs", {}).items():
                        self._cache[key] = PairState.from_dict(raw)
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                # Don't fail-open into trading — but the runner will note this
                # via `state_load_corrupted` and stop. We just return empty.
                self._cache = {}
        self._loaded = True
        return dict(self._cache)

    def _flush(self) -> None:
        """Write atomically: temp file then os.replace."""
        payload = {
            "schema":  SCHEMA_VERSION,
            "updated": datetime.now(timezone.utc).isoformat(),
            "pairs":   {k: v.to_dict() for k, v in self._cache.items()},
        }
        # Write to sibling temp file so a crash mid-write doesn't corrupt
        # the real file.
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix=self.path.name + ".",
            suffix=".tmp",
            dir=self.path.parent,
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, default=str)
            os.replace(tmp_path, self.path)
        except Exception:
            # Best-effort cleanup of the tmp file
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # ── lookups ─────────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def get(self, pair_key: str) -> Optional[PairState]:
        self._ensure_loaded()
        return self._cache.get(pair_key)

    def all(self) -> dict[str, PairState]:
        self._ensure_loaded()
        return dict(self._cache)

    # ── mutations (flushed immediately) ─────────────────────────────────────

    def add(self, state: PairState) -> None:
        self._ensure_loaded()
        self._cache[state.pair_key] = state
        self._flush()

    def remove(self, pair_key: str) -> Optional[PairState]:
        self._ensure_loaded()
        removed = self._cache.pop(pair_key, None)
        if removed is not None:
            self._flush()
        return removed

    def increment_bars(self, pair_key: str) -> None:
        """Called once per cycle for every currently-open pair."""
        self._ensure_loaded()
        st = self._cache.get(pair_key)
        if st is not None:
            st.bars_in_position += 1
            self._flush()
