"""
Portfolio engine — translate per-sleeve target positions (fraction of equity) into concrete,
broker-feasible lot sizes, subject to exposure constraints, and report book-level risk.

Volatility targeting itself lives in the signal engine (it is part of the frozen strategy —
``SizeParams(target_ann_vol=0.10, …)``); the portfolio engine adds the *operational* risk
envelope around it: gross/net exposure caps, per-position caps, leverage, and the exposure
metrics the Phase-5 mandate asks to log (gross, net, leverage, pair contributions).
"""

from __future__ import annotations

from .engine import Leg, PortfolioEngine, TargetHolding, ExposureSnapshot

__all__ = ["PortfolioEngine", "TargetHolding", "Leg", "ExposureSnapshot"]
