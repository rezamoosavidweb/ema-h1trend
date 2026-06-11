"""
FX currency-graph algebra.

In log-price space every FX pair `XY` (base `X`, quote `Y`) maps to the
vector ``e_X - e_Y`` in R^|C| where C is the set of currencies. The space
of log-pair processes therefore has dimension at most ``|C| - 1`` (one
currency acts as the implicit numeraire). Any pair whose currency vector
lies in the integer span of two others is a *deterministic triangular
identity* and cannot supply a tradable arbitrage edge that is not already
priced by every market-maker in milliseconds.

This module supplies the symbol-parsing, vector-construction, and
rank-deficiency primitives the detector needs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

import numpy as np


_SEPARATOR_RE = re.compile(r"[\s/\-_.:]")


@dataclass(frozen=True)
class ParsedSymbol:
    """Outcome of parsing a broker symbol into (base, quote)."""

    raw: str
    canonical: str   # e.g. "EURUSD"
    base: str        # e.g. "EUR"
    quote: str       # e.g. "USD"


# ---------------------------------------------------------------- parsing


def parse_symbol(
    symbol: str,
    known_currencies: FrozenSet[str],
    suffix_strip: Sequence[str] = (),
    alias: Optional[Dict[str, str]] = None,
) -> ParsedSymbol:
    """
    Parse a broker symbol into (base, quote).

    Strategy
    --------
    1. Apply manual alias if provided.
    2. Uppercase and strip common separators (``/``, ``-``, ``_``, ``.``,
       whitespace).
    3. Strip configured broker suffixes (``.m``, ``.ecn``, ``i``, ``c``,
       ...).
    4. Try fixed 3/3 split first (covers ~98% of FX symbols).
    5. Fall back to greedy 3/4 and 4/3 split for crypto-style tickers
       like ``BTCUSDT`` or ``USDTBTC``.
    6. Raise ``ValueError`` if no valid (known-base, known-quote)
       interpretation exists. We refuse to guess.

    Parameters
    ----------
    symbol:
        Raw broker ticker (case-insensitive, may contain punctuation).
    known_currencies:
        Set of recognised currency tokens. Anything not in this set is
        treated as parse failure rather than a novel currency.
    suffix_strip:
        Broker-specific suffixes (lowercased) to remove before parsing.
    alias:
        Optional explicit override map.

    Raises
    ------
    ValueError: when the symbol cannot be parsed.
    """
    if alias and symbol.upper() in alias:
        canonical = alias[symbol.upper()]
    else:
        canonical = symbol.upper()
        canonical = _SEPARATOR_RE.sub("", canonical)
        for suf in suffix_strip:
            suf_u = suf.upper()
            if canonical.endswith(suf_u) and len(canonical) - len(suf_u) >= 6:
                canonical = canonical[: -len(suf_u)]
                break

    n = len(canonical)
    candidates: List[Tuple[int, int]] = []
    if n == 6:
        candidates.append((3, 3))
    elif n == 7:
        candidates.extend([(3, 4), (4, 3)])
    elif n == 8:
        candidates.extend([(4, 4), (3, 5), (5, 3)])
    else:
        # We could try (3, n-3) etc., but the false-positive risk is too
        # high; demand an explicit alias instead.
        raise ValueError(
            f"Cannot parse symbol {symbol!r}: unsupported length {n} "
            f"(canonical={canonical!r}). Provide an explicit alias."
        )

    for base_len, quote_len in candidates:
        base = canonical[:base_len]
        quote = canonical[base_len:base_len + quote_len]
        if base in known_currencies and quote in known_currencies:
            return ParsedSymbol(
                raw=symbol,
                canonical=base + quote,
                base=base,
                quote=quote,
            )

    raise ValueError(
        f"Cannot parse symbol {symbol!r}: no split into known currencies "
        f"({canonical!r}). Add to known_currencies or symbol_alias."
    )


def safe_parse_symbol(
    symbol: str,
    known_currencies: FrozenSet[str],
    suffix_strip: Sequence[str] = (),
    alias: Optional[Dict[str, str]] = None,
) -> Optional[ParsedSymbol]:
    """Non-raising wrapper. Returns ``None`` on failure."""
    try:
        return parse_symbol(symbol, known_currencies, suffix_strip, alias)
    except ValueError:
        return None


# ----------------------------------------------- currency-vector algebra


class CurrencyGraph:
    """
    Currency-vector representation of an FX universe.

    Each symbol maps to a row of the universe matrix ``V`` with +1 in the
    base column and -1 in the quote column. Operations on the graph are
    pure linear algebra over this representation.
    """

    def __init__(self, parsed: Sequence[ParsedSymbol]) -> None:
        if not parsed:
            raise ValueError("CurrencyGraph requires at least one symbol.")

        self.symbols: Tuple[str, ...] = tuple(p.canonical for p in parsed)
        self._symbol_to_idx: Dict[str, int] = {
            s: i for i, s in enumerate(self.symbols)
        }
        currencies = sorted({p.base for p in parsed} | {p.quote for p in parsed})
        self.currencies: Tuple[str, ...] = tuple(currencies)
        self._currency_to_idx: Dict[str, int] = {
            c: i for i, c in enumerate(self.currencies)
        }

        V = np.zeros((len(self.symbols), len(self.currencies)), dtype=np.int8)
        for i, p in enumerate(parsed):
            V[i, self._currency_to_idx[p.base]] = +1
            V[i, self._currency_to_idx[p.quote]] = -1
        self.V: np.ndarray = V

    # -------------------------------------------------------- accessors
    def index_of(self, symbol: str) -> int:
        return self._symbol_to_idx[symbol]

    def vector(self, symbol: str) -> np.ndarray:
        """Currency vector of a symbol as a float row."""
        return self.V[self._symbol_to_idx[symbol]].astype(float)

    def __len__(self) -> int:
        return len(self.symbols)

    def __contains__(self, symbol: str) -> bool:
        return symbol in self._symbol_to_idx

    # ---------------------------------------------------------- algebra
    def rank(self) -> int:
        """Numerical rank of the currency-vector matrix."""
        return int(np.linalg.matrix_rank(self.V.astype(float)))

    def is_rank_deficient(self) -> bool:
        """
        True when the universe contains more pairs than independent
        directions, i.e. at least one pair is a deterministic
        combination of the others.

        For FX universes this is *typical* — with N currencies you
        cannot have more than N-1 independent pair processes.
        """
        return self.rank() < len(self.symbols)

    def spread_vector(
        self, p1: str, p2: str, beta: float
    ) -> np.ndarray:
        """Currency vector of the log-spread ``log(P1) - beta*log(P2)``."""
        return self.vector(p1) - beta * self.vector(p2)

    def matches_existing_pair(
        self,
        v: np.ndarray,
        *,
        atol: float,
        exclude: Iterable[str] = (),
        test_inverse: bool = True,
    ) -> Optional[Tuple[str, int]]:
        """
        Find a symbol whose currency vector equals ``v`` (or ``-v``).

        Returns ``(symbol, sign)`` where ``sign`` is +1 for direct match
        and -1 for inverse-pair match, or ``None`` if no symbol matches
        within ``atol``.
        """
        exclude_set = set(exclude)
        Vf = self.V.astype(float)
        diff = Vf - v[np.newaxis, :]
        norms = np.linalg.norm(diff, axis=1)
        order = np.argsort(norms)
        for idx in order:
            sym = self.symbols[idx]
            if sym in exclude_set:
                continue
            if norms[idx] <= atol:
                return sym, +1
            break  # rest are further away

        if test_inverse:
            diff = Vf + v[np.newaxis, :]
            norms = np.linalg.norm(diff, axis=1)
            order = np.argsort(norms)
            for idx in order:
                sym = self.symbols[idx]
                if sym in exclude_set:
                    continue
                if norms[idx] <= atol:
                    return sym, -1
                break

        return None

    def redundant_pairs(self) -> List[str]:
        """
        Return symbols that can be expressed as integer linear
        combinations of earlier symbols (in the iteration order of the
        constructor). Useful for pre-screening: trading both legs of a
        triangular identity gives you no diversification.
        """
        redundant: List[str] = []
        seen_rows: List[np.ndarray] = []
        for i, sym in enumerate(self.symbols):
            row = self.V[i].astype(float)
            if not seen_rows:
                seen_rows.append(row)
                continue
            M = np.vstack(seen_rows + [row])
            r_before = np.linalg.matrix_rank(np.vstack(seen_rows))
            r_after = np.linalg.matrix_rank(M)
            if r_after == r_before:
                redundant.append(sym)
            else:
                seen_rows.append(row)
        return redundant
