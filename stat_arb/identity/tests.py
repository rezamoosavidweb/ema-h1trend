"""
Statistical primitives used by the synthetic-identity detector.

All functions:
* operate on aligned NumPy arrays (caller handles index alignment);
* are numerically guarded (condition-number check, NaN-safe);
* are free of look-ahead — they treat the input window as a closed set.

Numerical convention: regressions are computed via SVD-based ``lstsq``
rather than the normal equations, because pair log-returns are often
extremely correlated and the normal equations are ill-conditioned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class OLSFit:
    """Outcome of a guarded least-squares fit."""

    intercept: float
    slopes: np.ndarray        # shape (k,)
    residuals: np.ndarray     # shape (n,)
    residual_std: float
    target_std: float
    r_squared: float
    condition_number: float
    ok: bool                  # False if numerically unstable

    @property
    def residual_ratio(self) -> float:
        if self.target_std <= 0.0 or not np.isfinite(self.target_std):
            return 1.0
        return float(self.residual_std / self.target_std)


def _design_matrix(X: np.ndarray, add_intercept: bool = True) -> np.ndarray:
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if add_intercept:
        return np.column_stack([np.ones(X.shape[0], dtype=float), X])
    return X


def guarded_ols(
    y: np.ndarray,
    X: np.ndarray,
    *,
    add_intercept: bool = True,
    max_condition: float = 1e10,
) -> OLSFit:
    """
    SVD-based OLS with explicit numerical guards.

    Parameters
    ----------
    y, X:
        Aligned, NaN-free arrays. ``X`` may be 1-D (single regressor) or
        2-D (multiple regressors as columns).
    add_intercept:
        Whether to prepend a constant column.
    max_condition:
        If the design matrix's condition number exceeds this, mark the
        fit as ``ok=False``.
    """
    y = np.asarray(y, dtype=float).ravel()
    Xd = _design_matrix(np.asarray(X, dtype=float), add_intercept)

    if Xd.shape[0] != y.shape[0]:
        raise ValueError(
            f"y has {y.shape[0]} rows but X has {Xd.shape[0]} rows"
        )
    if Xd.shape[0] <= Xd.shape[1]:
        return _failed_fit(y)

    # SVD-based pseudoinverse path. Guards against rank deficiency too.
    try:
        cond = float(np.linalg.cond(Xd))
    except np.linalg.LinAlgError:
        return _failed_fit(y)

    ok = np.isfinite(cond) and cond <= max_condition

    try:
        coef, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    except np.linalg.LinAlgError:
        return _failed_fit(y)

    yhat = Xd @ coef
    resid = y - yhat

    if add_intercept:
        intercept = float(coef[0])
        slopes = coef[1:].astype(float)
    else:
        intercept = 0.0
        slopes = coef.astype(float)

    target_std = float(np.std(y, ddof=1)) if y.size > 1 else 0.0
    resid_std = float(np.std(resid, ddof=1)) if resid.size > 1 else 0.0
    if target_std > 0:
        r2 = 1.0 - (resid.var(ddof=1) / max(y.var(ddof=1), 1e-300))
    else:
        r2 = 0.0

    return OLSFit(
        intercept=intercept,
        slopes=slopes,
        residuals=resid,
        residual_std=resid_std,
        target_std=target_std,
        r_squared=float(r2),
        condition_number=cond,
        ok=ok,
    )


def _failed_fit(y: np.ndarray) -> OLSFit:
    return OLSFit(
        intercept=float("nan"),
        slopes=np.array([np.nan]),
        residuals=np.full_like(y, np.nan, dtype=float),
        residual_std=float("nan"),
        target_std=float(np.std(y, ddof=1)) if y.size > 1 else 0.0,
        r_squared=float("nan"),
        condition_number=float("inf"),
        ok=False,
    )


# ---------------------------------------------------------- correlation


def pearson_corr_vector(
    y: np.ndarray, X: np.ndarray
) -> np.ndarray:
    """
    Compute Pearson correlation of ``y`` against every column of ``X``.

    Vectorised over columns. Returns shape ``(X.shape[1],)``. Columns
    with zero variance get correlation = 0 (rather than NaN).
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    y_centred = y - y.mean()
    X_centred = X - X.mean(axis=0, keepdims=True)

    y_std = float(np.sqrt((y_centred * y_centred).sum()))
    X_std = np.sqrt((X_centred * X_centred).sum(axis=0))

    # Avoid 0/0 -> NaN. Pairs with zero variance simply have undefined
    # correlation; treat as zero so downstream argmax doesn't pick them.
    denom = y_std * X_std
    safe = denom > 0
    out = np.zeros(X.shape[1], dtype=float)
    if not safe.any():
        return out
    num = X_centred.T @ y_centred                       # shape (k,)
    out[safe] = num[safe] / denom[safe]
    return out


def variance_ratio(
    y_resid_std: float, y_total_std: float
) -> float:
    """Residual / total std. Returns 1.0 when ``y_total_std`` is 0."""
    if y_total_std <= 0.0 or not np.isfinite(y_total_std):
        return 1.0
    return float(y_resid_std / y_total_std)


# ---------------------------------------------------------- spread maths


def log_spread(
    log_p1: np.ndarray, log_p2: np.ndarray, beta: float
) -> np.ndarray:
    """``log(P1) - beta * log(P2)`` with NaN propagation."""
    return np.asarray(log_p1, dtype=float) - beta * np.asarray(log_p2, dtype=float)


def aligned_log_returns(
    log_prices: np.ndarray,
) -> np.ndarray:
    """First differences along axis 0. Rows with any NaN are kept; the
    caller is responsible for masking before consuming them."""
    diff = np.diff(log_prices, axis=0)
    return diff


# ----------------------------------------------------- top-K preselect


def top_k_correlated(
    y: np.ndarray,
    X: np.ndarray,
    k: int,
    *,
    exclude_cols: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Indices of the K columns in ``X`` most correlated (by |rho|) with ``y``.

    Used to bound the combinatorial cost of the two-regressor search.

    Parameters
    ----------
    exclude_cols:
        Boolean array of shape ``(X.shape[1],)``. Columns marked True
        are excluded from the ranking (typically the two source pairs).
    """
    rho = np.abs(pearson_corr_vector(y, X))
    if exclude_cols is not None:
        rho = rho.copy()
        rho[exclude_cols] = -np.inf
    k = min(k, int(np.isfinite(rho).sum()))
    if k <= 0:
        return np.array([], dtype=int)
    # argpartition is O(n) and adequate; we sort the top-k for determinism.
    part = np.argpartition(-rho, k - 1)[:k]
    return part[np.argsort(-rho[part])]
