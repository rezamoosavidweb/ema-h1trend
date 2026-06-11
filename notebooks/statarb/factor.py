"""Institutional upgrades — PCA factor neutralisation and volatility-regime detection.

Two extras beyond the core pipeline, both aimed at the same enemy: *common-factor risk*.
In crypto almost everything is long-BTC-beta, so a "market-neutral" pairs book can quietly
become a leveraged beta bet when correlations spike.

  * `pca_factors` / `neutralize` — decompose the return panel; PC1 is essentially the
    crypto market. Removing the top factors leaves *idiosyncratic* residual returns; pairs
    built on residuals are genuinely market-neutral, not just dollar-neutral.
  * `vol_regime` — label each bar low/mid/high vol from a market proxy. Lets us measure
    whether the strategy's edge is regime-dependent (it usually is: mean-reversion pays in
    calm regimes and gets run over in trending/high-vol ones).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .data import BARS_PER_YEAR


def pca_factors(rets: pd.DataFrame, n_components: int = 5) -> dict:
    """Fit PCA on standardised returns. Returns components, explained-variance ratio, and
    the factor time series (scores). PC1's explained-variance share is the 'how much is one
    market factor' number."""
    R = rets.dropna(how="any")
    Z = (R - R.mean()) / R.std()
    n_components = min(n_components, Z.shape[1])
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(Z.values)
    return {
        "explained_var": pd.Series(pca.explained_variance_ratio_,
                                   index=[f"PC{i+1}" for i in range(n_components)]),
        "loadings": pd.DataFrame(pca.components_.T, index=R.columns,
                                 columns=[f"PC{i+1}" for i in range(n_components)]),
        "scores": pd.DataFrame(scores, index=R.index,
                               columns=[f"PC{i+1}" for i in range(n_components)]),
        "pca": pca,
    }


def pca_structure(rets: pd.DataFrame, n_components: int | None = None) -> dict:
    """Full PCA structure for the universe (Section A). Returns eigenvalues, explained- and
    cumulative-variance, loadings, and the number of factors needed to reach 80%/90% of
    variance. Built on the *correlation* matrix (standardised returns) so it answers "how many
    independent directions of co-movement are there", the question that decides whether
    market-neutral pairs can ever be diversified."""
    R = rets.dropna(how="any")
    Z = (R - R.mean()) / R.std()
    k = n_components or Z.shape[1]
    pca = PCA(n_components=k)
    scores = pca.fit_transform(Z.values)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    names = [f"PC{i+1}" for i in range(k)]
    return {
        "eigenvalues": pd.Series(pca.explained_variance_, index=names),
        "explained_var": pd.Series(evr, index=names),
        "cumulative_var": pd.Series(cum, index=names),
        "loadings": pd.DataFrame(pca.components_.T, index=R.columns, columns=names),
        "scores": pd.DataFrame(scores, index=R.index, columns=names),
        "n_factors_80": int(np.searchsorted(cum, 0.80) + 1),
        "n_factors_90": int(np.searchsorted(cum, 0.90) + 1),
        "idiosyncratic_frac": float(1 - evr[0]),     # variance NOT in the market factor
        "pca": pca,
    }


def neutralize(rets: pd.DataFrame, n_factors: int = 1) -> pd.DataFrame:
    """Return factor-neutral residual returns: regress each asset's returns on the top
    `n_factors` principal components (estimated in-sample) and keep the residual. Spreads
    built on these are neutral to the market factor, not just to each other."""
    R = rets.dropna(how="any")
    Z = (R - R.mean()) / R.std()
    pca = PCA(n_components=min(n_factors, Z.shape[1]))
    F = pca.fit_transform(Z.values)                # factor scores (in-sample)
    F1 = np.hstack([F, np.ones((len(F), 1))])      # add intercept
    resid = {}
    for c in R.columns:
        beta = np.linalg.lstsq(F1, R[c].values, rcond=None)[0]
        resid[c] = R[c].values - F1 @ beta
    return pd.DataFrame(resid, index=R.index)


def residual_log_prices(rets: pd.DataFrame, n_factors: int = 1) -> pd.DataFrame:
    """Cumulative factor-neutral residual returns as a synthetic 'log-price' panel. Feeding
    these into the normal spread/z-score machinery yields *residual spreads* — pair spreads
    with the market factor stripped out. The research test (Section A): are residual spreads
    more stationary (shorter half-life, lower ADF p) than raw price spreads?"""
    resid = neutralize(rets, n_factors=n_factors)
    return resid.cumsum()


def vol_regime(px: pd.Series, *, window: int = 168, timeframe: str = "H1",
               q=(0.33, 0.66)) -> pd.Series:
    """Label each bar 'low'/'mid'/'high' vol from a market proxy's trailing realised vol.
    Thresholds are *expanding-window* quantiles (point-in-time — a bar is classified using
    only vol seen up to then, so the labels themselves are not look-ahead)."""
    r = np.log(px).diff()
    rv = r.rolling(window, min_periods=window // 2).std() * np.sqrt(BARS_PER_YEAR[timeframe])
    lo = rv.expanding(min_periods=window).quantile(q[0])
    hi = rv.expanding(min_periods=window).quantile(q[1])
    out = pd.Series("mid", index=px.index, dtype=object)
    out[rv <= lo] = "low"
    out[rv >= hi] = "high"
    out[rv.isna()] = np.nan
    return out
