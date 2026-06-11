"""Currency-factor residual reversion (Phase-4 item 3).

A pair return is mechanically the difference of two *currency* strengths:
    r(BASEQUOTE)_t ≈ strength(BASE)_t − strength(QUOTE)_t + idiosyncratic_t
Phase-3 showed FX is ~85% idiosyncratic, so the clean play is to (a) estimate the currency
strength factors, (b) strip them out, and (c) trade the *residual* cross-sectionally. This is
cleaner than pairwise selection: there is nothing to pick, and we trade exactly the
idiosyncratic mean-reversion the PCA said dominates FX.

We recover the currency factors by least squares each bar: build the pair→currency incidence
matrix A (+1 for base, −1 for quote), solve A·s ≈ r for the strength vector s (fixing one
currency to zero / using the pseudo-inverse for identifiability). The fitted residual
r − A·s is the tradable idiosyncratic return.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import log_returns


def incidence_matrix(symbols: list[str]) -> tuple[np.ndarray, list[str]]:
    """A: (n_pairs x n_ccy), +1 base / -1 quote. Returns (A, currency_order)."""
    ccys = sorted({s[:3] for s in symbols} | {s[3:] for s in symbols})
    idx = {c: i for i, c in enumerate(ccys)}
    A = np.zeros((len(symbols), len(ccys)))
    for r, s in enumerate(symbols):
        A[r, idx[s[:3]]] = 1.0
        A[r, idx[s[3:]]] = -1.0
    return A, ccys


def currency_factor_returns(px: pd.DataFrame) -> pd.DataFrame:
    """Per-bar currency strength returns (one column per currency) recovered by least squares
    from the pair returns. Identification: the system is rank-deficient by 1 (only differences
    are observed), so we use the minimum-norm pseudo-inverse, which fixes the average strength
    to zero — i.e. strengths are *relative*, exactly what FX is."""
    symbols = list(px.columns)
    A, ccys = incidence_matrix(symbols)
    Ainv = np.linalg.pinv(A)                       # (n_ccy x n_pairs), minimum-norm
    R = log_returns(px).values                     # (T x n_pairs)
    S = R @ Ainv.T                                 # (T x n_ccy) strength returns
    return pd.DataFrame(S, index=px.index, columns=ccys)


def residual_returns(px: pd.DataFrame) -> pd.DataFrame:
    """Idiosyncratic (factor-neutral) pair returns: r − A·s. These are what the currency
    factors cannot explain — the pure relative-value signal."""
    symbols = list(px.columns)
    A, ccys = incidence_matrix(symbols)
    S = currency_factor_returns(px)
    fitted = S.values @ A.T                         # (T x n_pairs) explained part
    R = log_returns(px)
    return pd.DataFrame(R.values - fitted, index=px.index, columns=symbols)


def residual_log_prices(px: pd.DataFrame) -> pd.DataFrame:
    """Cumulative residual returns as synthetic log-prices — feed to the cross-sectional engine
    so we trade reversion of the *idiosyncratic* component, not the raw pair."""
    return residual_returns(px).cumsum()


def factor_strength_table(px: pd.DataFrame) -> pd.DataFrame:
    """Diagnostic: how much of each pair's variance the currency factors explain (R²). High R²
    => the pair is mostly its two currencies; the residual is what's left to trade."""
    symbols = list(px.columns)
    A, _ = incidence_matrix(symbols)
    S = currency_factor_returns(px)
    fitted = pd.DataFrame(S.values @ A.T, index=px.index, columns=symbols)
    R = log_returns(px)
    out = {}
    for s in symbols:
        tot = R[s].var()
        resid = (R[s] - fitted[s]).var()
        out[s] = 1 - resid / tot if tot > 0 else np.nan
    return pd.Series(out, name="factor_R2").sort_values(ascending=False)
