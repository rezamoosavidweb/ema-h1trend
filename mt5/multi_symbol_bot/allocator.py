"""
Capital allocation across the symbol basket.

Three weighting policies are supported:

    equal       — each symbol gets `total_risk / N`. Default; matches the
                  notebook's "treat the basket as one portfolio" assumption.
    score       — weight by `symbol_ranking.csv` score; high-score symbols
                  get more of the budget. Caps a single symbol at MAX_WEIGHT
                  so one star performer cannot dominate.
    custom      — caller supplies a {symbol: weight} dict; weights are
                  normalised to sum to 1.

Output per symbol:
    risk_per_trade  — the fraction of account balance to risk on one trade,
                      passed straight to `execution.RiskAdapter`.
    weight          — informational; share of the portfolio risk budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .config import SymbolBasket


# A single symbol cannot consume more than this fraction of the total risk
# budget under any policy (sanity guard against a runaway score weighting).
MAX_WEIGHT = 0.40


@dataclass(frozen=True)
class SymbolAllocation:
    """The portion of the per-trade risk budget assigned to one symbol."""

    symbol:          str
    weight:          float    # share of total portfolio risk in [0, 1]
    risk_per_trade:  float    # fraction of account balance per trade for this symbol


class CapitalAllocator:
    """
    Computes per-symbol risk_per_trade given a portfolio-level risk budget.

    Example
    -------
        basket = load_basket(...)              # 7 symbols
        alloc  = CapitalAllocator(total_portfolio_risk=0.02)
        plan   = alloc.allocate(basket, policy="score")

        for a in plan:
            engine = ExecutionEngine(symbol=a.symbol,
                                      risk_per_trade=a.risk_per_trade, ...)
    """

    def __init__(self, total_portfolio_risk: float = 0.02) -> None:
        """
        `total_portfolio_risk` is the cumulative fraction of the account at risk
        if EVERY symbol opens its 1× position at the same time. With 0.02 and
        7 symbols, each symbol risks ~0.286% per trade.
        """
        if not (0 < total_portfolio_risk < 0.5):
            raise ValueError(
                f"total_portfolio_risk must be in (0, 0.5); got {total_portfolio_risk}"
            )
        self.total_portfolio_risk = total_portfolio_risk

    # ── public API ───────────────────────────────────────────────────────────-

    def allocate(
        self,
        basket: SymbolBasket,
        policy: str = "equal",
        custom_weights: Mapping[str, float] | None = None,
    ) -> list[SymbolAllocation]:
        if len(basket) == 0:
            return []

        weights = self._compute_weights(basket, policy, custom_weights)
        out: list[SymbolAllocation] = []
        for sym, w in weights.items():
            out.append(SymbolAllocation(
                symbol         = sym,
                weight         = round(w, 6),
                risk_per_trade = round(w * self.total_portfolio_risk, 6),
            ))
        return out

    # ── policies ─────────────────────────────────────────────────────────────-

    def _compute_weights(
        self,
        basket: SymbolBasket,
        policy: str,
        custom_weights: Mapping[str, float] | None,
    ) -> dict[str, float]:
        symbols = basket.symbols

        if policy == "equal":
            w = 1.0 / len(symbols)
            return {s: w for s in symbols}

        if policy == "score":
            # Score from symbol_ranking.csv. Symbols missing a score get the
            # median so they aren't accidentally zeroed out.
            raw = {m.symbol: float(m.stats.get("score", 0.0) or 0.0) for m in basket}
            # All scores ≤ 0 → fall back to equal weighting (degenerate case).
            if max(raw.values(), default=0.0) <= 0:
                return self._compute_weights(basket, "equal", None)
            # Clip negatives to 0, then normalise.
            clipped = {s: max(0.0, v) for s, v in raw.items()}
            total = sum(clipped.values()) or 1.0
            weights = {s: v / total for s, v in clipped.items()}
            return self._cap_and_renormalise(weights)

        if policy == "custom":
            if not custom_weights:
                raise ValueError("policy='custom' requires custom_weights")
            unknown = set(custom_weights) - set(symbols)
            if unknown:
                raise ValueError(f"custom_weights references unknown symbols: {unknown}")
            # Symbols not listed get 0; the rest are normalised.
            raw = {s: max(0.0, float(custom_weights.get(s, 0.0))) for s in symbols}
            total = sum(raw.values())
            if total <= 0:
                raise ValueError("custom_weights sum to 0 or are all negative")
            return self._cap_and_renormalise({s: v / total for s, v in raw.items()})

        raise ValueError(f"unknown policy: {policy!r}")

    @staticmethod
    def _cap_and_renormalise(weights: dict[str, float]) -> dict[str, float]:
        """
        Apply MAX_WEIGHT cap, then renormalise so weights sum to 1.

        Iterative: capping creates "spare" weight to redistribute, which can
        itself push another symbol above the cap. Two passes are usually
        enough but we keep iterating until stable.
        """
        for _ in range(10):
            capped = {s: min(MAX_WEIGHT, w) for s, w in weights.items()}
            spare = 1.0 - sum(capped.values())
            if spare <= 1e-9:
                weights = capped
                break
            # Distribute spare among symbols not yet at the cap, proportional
            # to their existing weight.
            uncapped = {s: w for s, w in capped.items() if w < MAX_WEIGHT - 1e-9}
            if not uncapped:
                weights = capped
                break
            denom = sum(uncapped.values()) or 1.0
            new_weights = {}
            for s, w in capped.items():
                if s in uncapped:
                    new_weights[s] = w + spare * (uncapped[s] / denom)
                else:
                    new_weights[s] = w
            if max(abs(new_weights[s] - weights[s]) for s in weights) < 1e-9:
                weights = new_weights
                break
            weights = new_weights
        # Final normalisation guard against floating-point drift.
        total = sum(weights.values()) or 1.0
        return {s: w / total for s, w in weights.items()}
