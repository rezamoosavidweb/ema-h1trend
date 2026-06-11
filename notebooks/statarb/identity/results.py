"""
Result dataclasses for synthetic-identity detection.

The detector returns a fully-typed structured result that downstream
screening stages (cost gate, walk-forward, etc.) can serialise to JSON
or Parquet without ad-hoc post-processing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


class RejectionReason(str, Enum):
    """Reason a candidate pair was flagged as synthetic, or CLEAN."""

    CLEAN = "clean"
    TRIANGULAR_IDENTITY = "triangular_identity"
    ONE_PAIR_EQUIVALENT = "one_pair_equivalent"
    TWO_PAIR_EQUIVALENT = "two_pair_equivalent"
    HIGH_SPREAD_CORRELATION = "high_spread_correlation"
    CURRENCY_GRAPH_RANK_DEFICIENT = "currency_graph_rank_deficient"

    # Failure modes that prevent a verdict
    INSUFFICIENT_DATA = "insufficient_data"
    PARSE_FAILURE = "parse_failure"
    NUMERICAL_INSTABILITY = "numerical_instability"


# A "detected equivalent" can be either a single ticker, a (ticker, ticker)
# pair (for 2-regressor matches), or None (clean / undetectable).
DetectedEquivalent = Optional[Union[str, Tuple[str, str]]]


@dataclass
class IdentityResult:
    """
    Verdict for a single candidate pair.

    Attributes
    ----------
    pair:
        Two-tuple of canonical symbols, e.g. ``("EURUSD", "GBPUSD")``.
    hedge_ratio:
        The beta used to construct the spread (``None`` when the
        detector ran in symbol-only / triangular-only mode).
    is_synthetic:
        True if the pair's spread is mechanically replicable (and
        therefore should be excluded from stat-arb candidates), or
        if the detector could not produce a clean verdict.
    confidence:
        Detector's confidence in the verdict, in [0, 1]. For positive
        verdicts, higher = stronger evidence the spread is synthetic.
        For CLEAN, the value represents the *negative* evidence margin
        (i.e., how far the best alternative was from the threshold).
    reason:
        Which test fired (or CLEAN).
    detected_equivalent:
        The third instrument (or pair of instruments) that explains
        the spread, when applicable.
    diagnostics:
        Free-form floats / scalars from each test stage. Always populated
        so downstream code can audit borderline cases.
    """

    pair: Tuple[str, str]
    hedge_ratio: Optional[float]
    is_synthetic: bool
    confidence: float
    reason: RejectionReason
    detected_equivalent: DetectedEquivalent = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------- helpers
    @property
    def passes(self) -> bool:
        """Whether this pair should advance to the next screening stage."""
        return self.reason == RejectionReason.CLEAN and not self.is_synthetic

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["reason"] = self.reason.value
        # Tuples don't round-trip JSON cleanly.
        d["pair"] = list(self.pair)
        if isinstance(self.detected_equivalent, tuple):
            d["detected_equivalent"] = list(self.detected_equivalent)
        return d

    def summary(self) -> str:
        """One-line human summary for logging."""
        p1, p2 = self.pair
        head = f"{p1:>10s} / {p2:<10s}"
        beta = (
            f"beta={self.hedge_ratio:+.4f}"
            if self.hedge_ratio is not None
            else "beta=  n/a "
        )
        verdict = "SYNTH" if self.is_synthetic else "CLEAN"
        equiv = (
            f" -> {self.detected_equivalent}"
            if self.detected_equivalent is not None
            else ""
        )
        return (
            f"{head} | {beta} | {verdict:5s} "
            f"({self.reason.value}, conf={self.confidence:.2f}){equiv}"
        )


@dataclass
class UniverseReport:
    """
    Output of running the detector over an entire candidate universe.

    Holds per-pair results plus universe-level diagnostics (graph rank,
    redundant symbols, etc.) that downstream stages may consume.
    """

    results: List[IdentityResult]
    graph_rank: int
    graph_dim: int                      # number of currencies
    n_symbols: int
    redundant_symbols: List[str]
    config_snapshot: Dict[str, Any] = field(default_factory=dict)

    # --------------------------------------------------------- helpers
    @property
    def n_synthetic(self) -> int:
        return sum(1 for r in self.results if r.is_synthetic)

    @property
    def n_clean(self) -> int:
        return sum(1 for r in self.results if r.passes)

    def clean(self) -> List[IdentityResult]:
        return [r for r in self.results if r.passes]

    def synthetic(self) -> List[IdentityResult]:
        return [r for r in self.results if r.is_synthetic]

    def to_records(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self.results]
