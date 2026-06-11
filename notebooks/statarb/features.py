"""Feature engineering — a per-asset feature matrix for clustering and diagnostics.

Two products:
  * `panel_features` — point-in-time, *cross-sectional* feature vector per asset measured
    over a window (used by the clustering layer to group assets). One row per asset.
  * helper time-series features (rolling vol, beta, rolling correlation) used both as
    clustering inputs and as backtest diagnostics.

Standardisation matters: KMeans is distance-based, so an un-scaled feature (e.g. dollar
volume ~1e6) would dominate a scaled one (e.g. skew ~0.3). We default to RobustScaler
(median / IQR) because crypto features are fat-tailed and a single 2021-bull outlier
would wreck a plain z-score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

from .data import BARS_PER_YEAR, log_returns


# --------------------------------------------------------------------------- #
# Time-series features (per asset, indexed by time).
# --------------------------------------------------------------------------- #
def rolling_returns(px: pd.DataFrame, horizons=(1, 4, 24)) -> dict[int, pd.DataFrame]:
    """Multi-horizon log returns. On H1 bars: 1=1h, 4=4h, 24=1d momentum."""
    logp = np.log(px)
    return {h: logp.diff(h) for h in horizons}


def ewma_vol(rets: pd.DataFrame, halflife: int = 72) -> pd.DataFrame:
    """EWMA volatility (default halflife 72 H1 bars = 3 days). Reacts faster to regime
    changes than a flat rolling window — the natural vol estimate for sizing."""
    return rets.ewm(halflife=halflife, min_periods=halflife // 2).std()


def rolling_vol(rets: pd.DataFrame, window: int = 168) -> pd.DataFrame:
    """Flat rolling-std vol (default 168 H1 bars = 1 week)."""
    return rets.rolling(window, min_periods=window // 2).std()


def rolling_beta(rets: pd.DataFrame, market: str = "BTCUSDT", window: int = 720) -> pd.DataFrame:
    """Rolling beta of each asset vs a market proxy (default BTC, 720h≈30d window).
    beta = cov(asset, mkt) / var(mkt). Captures how much of a name is just market
    direction — a key axis for clustering (high-beta majors vs idiosyncratic names)."""
    m = rets[market]
    var_m = m.rolling(window, min_periods=window // 2).var()
    out = {}
    for c in rets.columns:
        cov = rets[c].rolling(window, min_periods=window // 2).cov(m)
        out[c] = cov / var_m
    return pd.DataFrame(out)


def rolling_corr_vector(rets: pd.DataFrame, window: int = 720) -> pd.DataFrame:
    """For each asset, its *average* rolling correlation to the rest of the universe at
    each time — a compact 'how connected am I' feature (full pairwise matrices are too
    high-dimensional to cluster on directly)."""
    cols = rets.columns
    out = pd.DataFrame(index=rets.index, columns=cols, dtype=float)
    # Pairwise rolling corr is O(n^2); fine for a ~16-name universe.
    corr = {}
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            corr[(a, b)] = rets[a].rolling(window, min_periods=window // 2).corr(rets[b])
    for a in cols:
        partners = [corr[(a, b)] if (a, b) in corr else corr[(b, a)]
                    for b in cols if b != a]
        out[a] = pd.concat(partners, axis=1).mean(axis=1)
    return out


def amihud_illiquidity(rets: pd.DataFrame, dollar_volume: pd.DataFrame,
                       window: int = 168) -> pd.DataFrame:
    """Amihud (2002) illiquidity: mean(|return| / dollar_volume). High = price moves a lot
    per $ traded = illiquid. A cleaner liquidity axis than raw volume for clustering."""
    impact = rets.abs() / dollar_volume.replace(0, np.nan)
    return impact.rolling(window, min_periods=window // 2).mean()


# --------------------------------------------------------------------------- #
# Cross-sectional feature matrix (one row per asset) — clustering input.
# --------------------------------------------------------------------------- #
def panel_features(px: pd.DataFrame, dollar_volume: pd.DataFrame, *,
                   timeframe: str = "H1", market: str = "BTCUSDT",
                   corr_window: int = 720) -> pd.DataFrame:
    """Collapse each asset's behaviour over the supplied window into one feature vector.

    Features chosen to span the axes a quant cares about for pairing:
      * vol_ann            — annualised volatility (risk scale)
      * skew, kurt         — return-distribution shape (tail behaviour)
      * beta_btc, beta_eth — market exposure (factor structure)
      * mean_corr          — average correlation to the universe (connectedness)
      * trend              — |sum of returns| / vol, persistence vs choppiness
      * autocorr1          — lag-1 return autocorrelation (mean-reverting vs trending)
      * illiq              — log Amihud illiquidity (tradability)
      * adv                — log median dollar volume (size)
    All are *point-in-time over the formation window* — no future data leaks in.
    """
    rets = log_returns(px).dropna(how="all")
    ann = np.sqrt(BARS_PER_YEAR[timeframe])
    feats = {}
    for c in px.columns:
        r = rets[c].dropna()
        feats[c] = {
            "vol_ann": r.std() * ann,
            "skew": r.skew(),
            "kurt": r.kurt(),
            "trend": abs(r.sum()) / (r.std() * np.sqrt(len(r)) + 1e-12),
            "autocorr1": r.autocorr(lag=1),
        }
    f = pd.DataFrame(feats).T

    # Market betas (full-window OLS slope). For non-crypto universes (e.g. FX) the crypto
    # proxies are absent, so fall back to data-driven proxies: the two assets with the highest
    # average correlation to the rest stand in for "the market". Column names are kept
    # (beta_btc/beta_eth) for backward compatibility — read them as "beta vs primary/secondary
    # market proxy".
    cmat0 = rets.corr()
    avg_corr = cmat0.abs().mean().sort_values(ascending=False)
    mkt1 = market if market in rets.columns else avg_corr.index[0]
    mkt2 = "ETHUSDT" if "ETHUSDT" in rets.columns else avg_corr.index[1]
    for mkt, name in [(mkt1, "beta_btc"), (mkt2, "beta_eth")]:
        if mkt in rets.columns:
            m = rets[mkt]
            var_m = m.var()
            f[name] = [rets[c].cov(m) / var_m if var_m else np.nan for c in f.index]

    # Average pairwise correlation over the window (static, full-window).
    cmat = rets.corr()
    f["mean_corr"] = [cmat.loc[c].drop(c).mean() for c in f.index]

    # Liquidity axes.
    f["adv"] = np.log(dollar_volume.median().reindex(f.index) + 1.0)
    impact = (rets.abs() / dollar_volume.replace(0, np.nan)).mean()
    f["illiq"] = np.log(impact.reindex(f.index) + 1e-12)

    return f


def standardize(features: pd.DataFrame, *, robust: bool = True) -> tuple[pd.DataFrame, object]:
    """Scale features for distance-based clustering. Returns (scaled_df, fitted_scaler)."""
    scaler = RobustScaler() if robust else StandardScaler()
    clean = features.replace([np.inf, -np.inf], np.nan).fillna(features.median(numeric_only=True))
    X = scaler.fit_transform(clean.values)
    return pd.DataFrame(X, index=features.index, columns=features.columns), scaler
