"""
Configuration schema for the synthetic-identity detector.

All thresholds live here, with sensible defaults calibrated for retail FX
H1/M15 data. Override per-call or per-environment via YAML/env vars using
pydantic-settings.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Optional

from pydantic import BaseModel, Field, field_validator


# Currencies recognised by the symbol parser. Extend as needed.
DEFAULT_KNOWN_CURRENCIES: FrozenSet[str] = frozenset(
    {
        # G10 fiat
        "USD", "EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD",
        # Scandi / emerging
        "NOK", "SEK", "DKK", "PLN", "TRY", "ZAR", "MXN", "SGD",
        "HKD", "CNH", "CNY", "CZK", "HUF", "ILS", "THB", "RUB",
        # Metals (treat as currency in the graph)
        "XAU", "XAG", "XPT", "XPD",
        # Crypto (treat as currency in the graph; broker-dependent)
        "BTC", "ETH", "XRP", "LTC", "BCH", "ADA", "DOT", "SOL",
        "USDT", "USDC",
    }
)


class DetectorConfig(BaseModel):
    """
    Tunable thresholds for synthetic-identity detection.

    Defaults are calibrated for retail FX H1 data on majors+crosses.
    Tighter thresholds reduce false negatives but raise false positives;
    looser thresholds do the opposite. See `docs/calibration.md` (TBD).
    """

    # ----------------------------------------------------------------- I/O
    known_currencies: FrozenSet[str] = Field(
        default=DEFAULT_KNOWN_CURRENCIES,
        description="Tokens treated as currencies during symbol parsing.",
    )
    symbol_suffix_strip: tuple[str, ...] = Field(
        default=(".m", ".ecn", ".pro", "_m", "_ecn", "-m", "-ecn", "i", "c"),
        description=(
            "Lowercase broker-specific suffixes stripped before parsing. "
            "Order matters; longer suffixes first."
        ),
    )
    symbol_alias: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Manual symbol-to-canonical map. Use for broker-specific "
            "names that the heuristic parser cannot resolve "
            '(e.g., {"GOLD": "XAUUSD"}).'
        ),
    )

    # ------------------------------------------------------ Stage 1: graph
    # Tolerance for the algebraic equality v_spread == v_other.
    # Vectors are integer-valued ±1, so a tight tolerance is appropriate.
    graph_atol: float = Field(default=1e-9, ge=0.0)

    # Beta proximity to 1 for the same-base / same-quote triangular check.
    # If |beta - 1| > beta_tol, we do NOT claim a deterministic triangular
    # identity, even if the currency vectors would algebraically cancel.
    # This is intentional: a fitted beta far from unity means market quotes
    # disagree with the algebraic identity (e.g., NDF dislocation).
    beta_unity_tol: float = Field(default=0.20, ge=0.0, le=1.0)

    # ------------------------------------ Stage 2: one-pair residual test
    # Residual-std / spread-std ratio below which we flag the spread as
    # essentially a tradable third pair.
    # 0.15 means the third pair explains >97.75% of spread-return variance.
    one_pair_resid_ratio_max: float = Field(default=0.15, gt=0.0, le=1.0)

    # Required absolute slope in the one-pair regression to consider the
    # match meaningful (filters numerical noise on near-zero spreads).
    one_pair_min_abs_slope: float = Field(default=0.10, ge=0.0)

    # --------------------------------- Stage 3: two-pair residual test
    # Residual ratio threshold for the two-regressor identity case
    # (4-currency triangular collapses, e.g., EURJPY = EURUSD + USDJPY).
    two_pair_resid_ratio_max: float = Field(default=0.20, gt=0.0, le=1.0)

    # Skip the two-pair test if it would explode combinatorially.
    # We cap the search to top-K candidates by univariate corr.
    two_pair_top_k: int = Field(default=8, ge=2, le=50)

    # ------------------------- Stage 4: high spread-return correlation
    # Maximum |Pearson| of spread-returns vs. any single other pair's
    # log-returns. Above this, the spread is degenerate with that pair.
    spread_corr_max: float = Field(default=0.95, gt=0.0, le=1.0)

    # ------------------------------------------------------- Robustness
    min_obs: int = Field(default=500, ge=50,
                         description="Minimum aligned observations required.")
    rolling_window: Optional[int] = Field(
        default=None,
        description=(
            "If set, run residual tests over rolling windows of this size "
            "and aggregate (median ratio). Catches transient identities."
        ),
    )

    # If the regression's condition number exceeds this, treat as
    # numerically unstable rather than as a clean result.
    max_condition_number: float = Field(default=1e10, gt=0.0)

    # Whether to also test the spread against the *inverse* of each pair
    # (since log(B/A) = -log(A/B), broker symbol direction shouldn't matter).
    test_inverse_pairs: bool = Field(default=True)

    # ----------------------------------------------------------- Logging
    log_level: str = Field(default="INFO")
    progress_every: int = Field(
        default=50, ge=1,
        description="Emit a progress log line every N pairs scanned.",
    )

    # -------------------------------------------------------- Validators
    @field_validator("known_currencies", mode="before")
    @classmethod
    def _coerce_currencies(cls, v):
        if isinstance(v, (list, tuple, set)):
            return frozenset(s.upper() for s in v)
        return v

    @field_validator("symbol_alias", mode="before")
    @classmethod
    def _upper_alias_keys(cls, v):
        if isinstance(v, dict):
            return {k.upper(): val.upper() for k, val in v.items()}
        return v

    model_config = {
        "arbitrary_types_allowed": True,
        "frozen": False,
    }
