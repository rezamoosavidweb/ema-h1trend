"""Chart generation — equity curve, drawdown, exposure, regime. Saved as PNG for reports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .metrics import equity_dataframe
from ..storage.base import Storage


def write_charts(storage: Storage, out_dir: Path, *, prefix: str = "") -> dict[str, Path]:
    """Render the standard chart set to ``out_dir``; returns {name: path}. Charts that can't
    be drawn (no data / matplotlib missing) are skipped silently so reporting still runs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return paths

    eq = equity_dataframe(storage)
    if eq.empty:
        return paths
    equity = eq["equity"].astype(float)

    # 1) equity curve
    p = out_dir / f"{prefix}equity.png"
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(equity.index, equity.values, color="tab:blue", lw=1.2)
    ax.set_title("Equity curve"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig)
    paths["equity"] = p

    # 2) drawdown
    p = out_dir / f"{prefix}drawdown.png"
    dd = equity / equity.cummax() - 1.0
    fig, ax = plt.subplots(figsize=(10, 2.6))
    ax.fill_between(dd.index, dd.values, 0, color="tab:red", alpha=0.5)
    ax.set_title("Drawdown"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig)
    paths["drawdown"] = p

    # 3) exposure + regime multiplier
    if "gross_exposure" in eq.columns:
        p = out_dir / f"{prefix}exposure.png"
        fig, ax = plt.subplots(figsize=(10, 2.6))
        ax.plot(eq.index, eq["gross_exposure"], label="gross", color="tab:purple", lw=1)
        if "net_exposure" in eq.columns:
            ax.plot(eq.index, eq["net_exposure"], label="net", color="tab:green", lw=1)
        if "regime_multiplier" in eq.columns:
            ax.plot(eq.index, eq["regime_multiplier"], label="regime mult", color="tab:orange",
                    lw=0.8, alpha=0.7)
        ax.legend(fontsize=8); ax.set_title("Exposure & regime"); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig)
        paths["exposure"] = p

    return paths
