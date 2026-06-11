"""Execution models — taker vs a simplified maker model (Section E).

The Phase-1 conclusion was that the cointegration edge is *sub-frictional* at taker fees. The
obvious escape hatch is **maker execution**: post limit orders, earn (or save) the spread
instead of crossing it. But maker fills are not free — you trade fee for *fill uncertainty*:

  * **fill probability** — your resting order may never get hit before the signal moves.
  * **partial fills** — you may get only part of your size.
  * **adverse selection** — the orders that *do* fill tend to fill right before the price
    moves against you (you provided liquidity to an informed taker).

We model this as an **expected-value** overlay on the backtester (no order-book simulation —
honestly flagged as a simplification): `fill_prob` scales realised exposure (partial/missed
fills), `fee_bps` becomes the maker fee/rebate, and `adverse_bps` charges the filled flow for
adverse selection. Three scenarios bracket the uncertainty rather than pretending to one true
number. The research question is deliberately framed as falsifiable: *does maker execution
move net Sharpe above zero, and under which scenario does it stop working?*
"""
from __future__ import annotations

from .backtest import CostModel

# Bybit-like reference: taker 5.5 bps, maker ≈ 1.5 bps (some tiers rebate). We do NOT assume a
# rebate in the base case — that would flatter the result.

TAKER = CostModel(fee_bps=5.5, slippage_bps=2.0, exec_lag=1, fill_prob=1.0, adverse_bps=0.0)

# Maker scenarios. fee_bps = maker fee (can be negative = rebate); fill_prob < 1 = miss some
# trades; adverse_bps = the penalty on filled flow. slippage_bps=0 (you set the price).
MAKER_SCENARIOS = {
    "maker_conservative": CostModel(fee_bps=1.5, slippage_bps=0.0, exec_lag=1,
                                    fill_prob=0.40, adverse_bps=3.0),
    "maker_moderate":     CostModel(fee_bps=1.0, slippage_bps=0.0, exec_lag=1,
                                    fill_prob=0.60, adverse_bps=1.5),
    "maker_optimistic":   CostModel(fee_bps=-0.5, slippage_bps=0.0, exec_lag=1,  # small rebate
                                    fill_prob=0.80, adverse_bps=0.5),
}


def scenario(name: str) -> CostModel:
    if name == "taker":
        return TAKER
    if name in MAKER_SCENARIOS:
        return MAKER_SCENARIOS[name]
    raise KeyError(f"unknown execution scenario {name!r}")


def all_scenarios() -> dict:
    """Taker + the three maker scenarios, for the side-by-side comparison."""
    return {"taker": TAKER, **MAKER_SCENARIOS}


def describe() -> str:
    lines = ["Execution scenarios (expected-value maker overlay; NOT an order-book sim):"]
    for name, c in all_scenarios().items():
        lines.append(f"  {name:20s} fee={c.fee_bps:+.1f}bps  fill_prob={c.fill_prob:.0%}  "
                     f"adverse={c.adverse_bps:.1f}bps  slip={c.slippage_bps:.1f}bps")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Data-driven maker calibration from M1 bars (Phase-4 item 2).
#
# Replaces the *assumed* maker fill_prob/adverse with numbers measured from the data: post a
# passive limit `delta_bps` away from each H1 bar's open on the entry side, and use the M1 path
# WITHIN that hour to ask (a) did price touch the limit (a fill?) and (b) where did it go right
# after (adverse selection?). This is still a simplification (no queue position, assumes one
# resting order) but it is grounded in the realised intrabar path, not a guess.
# --------------------------------------------------------------------------- #
def calibrate_maker_from_m1(symbol: str, *, data_dir: str = "data_forex",
                            delta_bps_grid=(0.5, 1.0, 2.0, 4.0), max_bars: int = 200_000,
                            side: str = "buy") -> "object":
    """Calibrate (fill_prob, adverse_bps) vs passive offset `delta_bps` from M1 data.

    For a BUY we post `delta_bps` BELOW the H1 open; a fill requires the intrabar low to reach
    it. Adverse selection = how far price continued *against* a filled order over the rest of the
    hour (close vs fill), in bps. Returns a DataFrame indexed by delta_bps with columns
    [fill_prob, adverse_bps, net_edge_bps] where net_edge = half-spread saved − adverse cost.
    """
    import numpy as np
    import pandas as pd
    from .data import load_ohlcv, quoted_spread_bps

    m1 = load_ohlcv(symbol, "M1", data_dir)
    if max_bars and len(m1) > max_bars:
        m1 = m1.iloc[-max_bars:]                       # most-recent sample (bounded for speed)
    h1 = m1.index.floor("h")
    g = m1.groupby(h1)
    bar = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(),
                        "low": g["low"].min(), "close": g["close"].last()}).dropna()
    half_spread = quoted_spread_bps(symbol, "M1", data_dir)
    half_spread = (half_spread if np.isfinite(half_spread) else 1.0)

    rows = []
    o = bar["open"].values
    lo, hi, cl = bar["low"].values, bar["high"].values, bar["close"].values
    for d in delta_bps_grid:
        if side == "buy":
            limit = o * (1 - d / 1e4)
            filled = lo <= limit
            # adverse = price below fill at close (you bought, it kept falling) in bps
            adverse = np.where(filled, np.clip((limit - cl) / limit * 1e4, 0, None), np.nan)
        else:
            limit = o * (1 + d / 1e4)
            filled = hi >= limit
            adverse = np.where(filled, np.clip((cl - limit) / limit * 1e4, 0, None), np.nan)
        fp = float(np.mean(filled))
        adv = float(np.nanmean(adverse)) if filled.any() else np.nan
        rows.append({"delta_bps": d, "fill_prob": fp, "adverse_bps": adv,
                     "net_edge_bps": half_spread - (adv if np.isfinite(adv) else 0.0)})
    return pd.DataFrame(rows).set_index("delta_bps")


def maker_cost_from_calibration(calib, *, maker_fee_bps: float = 1.0) -> CostModel:
    """Pick the operating point that maximises net edge and turn it into a CostModel: fee =
    maker fee minus the half-spread captured, fill_prob and adverse from the calibration."""
    best = calib["net_edge_bps"].idxmax()
    row = calib.loc[best]
    eff_fee = maker_fee_bps - max(row["net_edge_bps"], 0.0)   # capturing spread reduces net fee
    return CostModel(fee_bps=float(eff_fee), slippage_bps=0.0, exec_lag=1,
                     fill_prob=float(row["fill_prob"]), adverse_bps=float(row["adverse_bps"]))
