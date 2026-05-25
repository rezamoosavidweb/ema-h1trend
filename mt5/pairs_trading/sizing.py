"""
β-scaled position sizing with broker volume bounds.

The function `calculate_lot_sizes` returns volumes that:
  * size the y-leg from `risk_per_leg_pct × equity` and an assumed adverse move
  * size the x-leg so |β × notional_y| matches notional_x (delta-neutral on β)
  * respect each symbol's `volume_min`, `volume_step`, `volume_max`
  * fall back to broker minima with an `UNDER_RISK` warning when the raw
    computation is below the broker minimum (typical on tiny demo equity)

Returned `SizingResult` carries every input → easy to audit in logs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LegSpec:
    """Per-leg sizing inputs read from MT5 symbol_info."""
    symbol:         str
    price:          float        # last close (or current bid/ask)
    contract_size:  float        # 100_000 for standard FX, 100 for metals, etc.
    volume_min:     float
    volume_step:    float
    volume_max:     float


@dataclass(frozen=True)
class SizingResult:
    """Full breakdown — printed verbatim into logs."""
    lots_y:        float
    lots_x:        float
    notional_y:    float
    notional_x:    float
    risk_usd_per_leg: float
    raw_lots_y:    float          # before clamping (for logs)
    raw_lots_x:    float
    warning:       str            # "" if all clean; "UNDER_RISK_*" otherwise


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def round_to_step(value: float, step: float) -> float:
    """Round `value` to the nearest multiple of `step`."""
    if step <= 0:
        return value
    return round(round(value / step) * step, 8)


def clamp_volume(raw: float, leg: LegSpec) -> tuple[float, bool]:
    """
    Apply broker volume_min / volume_step / volume_max.

    Returns (final_lots, was_clamped_to_min).
    """
    if raw < leg.volume_min:
        return leg.volume_min, True
    rounded = round_to_step(raw, leg.volume_step)
    if rounded < leg.volume_min:
        return leg.volume_min, True
    if rounded > leg.volume_max:
        return leg.volume_max, False
    return rounded, False


# ─────────────────────────────────────────────────────────────────────────────
# Main entry
# ─────────────────────────────────────────────────────────────────────────────


def calculate_lot_sizes(
    y:                 LegSpec,
    x:                 LegSpec,
    beta:              float,
    equity_usd:        float,
    risk_per_leg_pct:  float,
    assumed_stop_pct:  float = 0.01,
) -> SizingResult:
    """
    Compute paired lot sizes.

    Sizing logic
    ------------
        risk_usd        = equity × risk_per_leg_pct / 100
        notional_y/lot  = contract_size_y × price_y
        raw_lots_y      = risk_usd / (notional_y/lot × assumed_stop_pct)
                          ← so an `assumed_stop_pct` adverse move on y costs risk_usd

        notional_y      = lots_y × contract_size_y × price_y
        target_x        = |β| × notional_y
        raw_lots_x      = target_x / (contract_size_x × price_x)

    Both then clamped to broker volume bounds.

    Parameters
    ----------
    y, x:
        Per-leg specs from MT5 `symbol_info`.
    beta:
        Hedge ratio from `signals.refit_beta` (current cycle's value).
    equity_usd:
        Account equity (`mt5.account_info().equity`).
    risk_per_leg_pct:
        Percent of equity allocated to ONE leg. ~0.1 means very conservative.
    assumed_stop_pct:
        Hypothetical adverse move used to back out lot size. Defaults to 1%
        (i.e. 100 pips on a 1.0000-quoted pair) — matches NB29.

    Returns
    -------
    SizingResult with everything you need to audit the order in logs.
    """
    if equity_usd <= 0:
        raise ValueError(f"equity_usd must be positive, got {equity_usd}")
    if y.price <= 0 or x.price <= 0:
        raise ValueError(f"prices must be positive (y={y.price}, x={x.price})")

    risk_usd = equity_usd * (risk_per_leg_pct / 100.0)

    notional_per_lot_y = y.contract_size * y.price
    raw_lots_y = risk_usd / (notional_per_lot_y * assumed_stop_pct)
    lots_y, y_underrisk = clamp_volume(raw_lots_y, y)

    notional_y = lots_y * notional_per_lot_y
    target_notional_x = abs(beta) * notional_y
    raw_lots_x = target_notional_x / (x.contract_size * x.price)
    lots_x, x_underrisk = clamp_volume(raw_lots_x, x)

    notional_x = lots_x * x.contract_size * x.price

    warning_parts: list[str] = []
    if y_underrisk:
        warning_parts.append(f"UNDER_RISK_Y(raw={raw_lots_y:.4f}<min={y.volume_min})")
    if x_underrisk:
        warning_parts.append(f"UNDER_RISK_X(raw={raw_lots_x:.4f}<min={x.volume_min})")

    return SizingResult(
        lots_y           = lots_y,
        lots_x           = lots_x,
        notional_y       = notional_y,
        notional_x       = notional_x,
        risk_usd_per_leg = risk_usd,
        raw_lots_y       = raw_lots_y,
        raw_lots_x       = raw_lots_x,
        warning          = " ".join(warning_parts),
    )
