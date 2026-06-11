"""Clustering layer — regime-aware pair selection (the Part-4 extension).

Thesis being tested: naive correlation pairs two assets that *happened* to co-move, even
if they are economically unrelated (a DeFi token and a meme coin can show corr 0.8 in a
risk-on month). If instead we first **cluster assets by their structural fingerprint**
(volatility, tails, market beta, connectedness, liquidity) and only pair *within* a
cluster, the pairs should be more economically coherent and their spreads more stable.

Three clusterers, same interface:
  * `kmeans_clusters`        — baseline, spherical clusters, fast.
  * `gmm_clusters`           — soft/elliptical clusters (Gaussian Mixture).
  * `hierarchical_clusters`  — agglomerative (Ward) — bonus, gives a dendrogram story.

`within_cluster_candidates` turns labels into the candidate pair list consumed by
`pairs.select_pairs`, so the *entire* downstream pipeline (spread, signals, backtest) is
shared between baseline and enhanced — the only thing that changes is the candidate set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from itertools import combinations

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


def _labels_series(labels, index) -> pd.Series:
    return pd.Series(labels, index=index, name="cluster")


def kmeans_clusters(X: pd.DataFrame, k: int, *, seed: int = 7) -> pd.Series:
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    return _labels_series(km.fit_predict(X.values), X.index)


def gmm_clusters(X: pd.DataFrame, k: int, *, seed: int = 7) -> pd.Series:
    gm = GaussianMixture(n_components=k, covariance_type="full", n_init=5, random_state=seed)
    return _labels_series(gm.fit_predict(X.values), X.index)


def hierarchical_clusters(X: pd.DataFrame, k: int, *, linkage: str = "ward") -> pd.Series:
    ag = AgglomerativeClustering(n_clusters=k, linkage=linkage)
    return _labels_series(ag.fit_predict(X.values), X.index)


def choose_k(X: pd.DataFrame, k_range=range(2, 8), *, method: str = "kmeans",
             seed: int = 7) -> pd.DataFrame:
    """Silhouette score across k. Higher = better-separated clusters. We pick k by the
    silhouette peak rather than guessing — and report the whole curve so the choice is
    auditable (a flat curve is itself a finding: 'the universe has no strong cluster
    structure', which is a legitimate critique of the whole approach)."""
    rows = []
    n = len(X)
    for k in k_range:
        if k >= n:
            continue
        if method == "kmeans":
            lab = kmeans_clusters(X, k, seed=seed)
        elif method == "gmm":
            lab = gmm_clusters(X, k, seed=seed)
        else:
            lab = hierarchical_clusters(X, k)
        try:
            sil = silhouette_score(X.values, lab.values)
        except Exception:
            sil = np.nan
        rows.append({"k": k, "silhouette": sil, "n_clusters_used": lab.nunique()})
    return pd.DataFrame(rows).set_index("k")


def within_cluster_candidates(labels: pd.Series) -> list[tuple[str, str]]:
    """All within-cluster asset pairs — the candidate set for the enhanced strategy.
    Cross-cluster pairs are *excluded by construction* (the whole point)."""
    cands = []
    for c in sorted(labels.unique()):
        members = labels.index[labels == c].tolist()
        cands.extend(combinations(members, 2))
    return cands


def cross_cluster_candidates(labels: pd.Series) -> list[tuple[str, str]]:
    """The complement — used only to *demonstrate* that cross-cluster pairs are worse."""
    within = set(within_cluster_candidates(labels))
    allp = set(combinations(labels.index, 2))
    return sorted(allp - within)


def cluster_summary(labels: pd.Series, features: pd.DataFrame) -> pd.DataFrame:
    """Human-readable cluster profile: members + mean of each feature per cluster. This is
    the 'do these clusters mean anything?' sanity check — a good clustering should yield
    interpretable groups (e.g. 'high-beta majors', 'idiosyncratic low-liquidity')."""
    df = features.copy()
    df["cluster"] = labels
    prof = df.groupby("cluster").mean(numeric_only=True)
    prof["members"] = df.groupby("cluster").apply(
        lambda g: ", ".join(g.index), include_groups=False)
    prof["n"] = df.groupby("cluster").size()
    return prof
