"""
Pre-flight order validation against broker constraints.

Order rejections we have seen in production:
    10015  "Invalid price"         -- limit price on the wrong side of market /
                                      inside stops_level / outside freeze_level
    10022  "Invalid expiration"    -- handled in pending_manager (we use GTC)
    10027  "AutoTrading disabled"  -- terminal-side toggle, surfaced here
    10018  "Market is closed"      -- weekend / outside session
    10016  "Invalid stops"         -- SL/TP inside stops_level of price

Every one of these is preventable with a static check against `symbol_info`
and the current tick. This module performs those checks and returns a single
ValidationResult that the executor can act on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import MetaTrader5 as mt5

from .symbol_config import SymbolConfig


# Spread threshold (in points) above which we refuse to trade. Conservative
# default for XAUUSD where typical spread is 15-30 points; pumped to 200 to
# cover news/illiquid hours. Tighten per-symbol via `max_spread_points`.
DEFAULT_MAX_SPREAD_POINTS = 200


@dataclass
class ValidationResult:
    """
    Outcome of one pre-flight check. `ok=True` means "go ahead and place the order".

    The `code` field is intentionally a stable string (not an int) so it shows
    up in logs as a human-readable category, e.g. "stops_too_close".
    """
    ok: bool
    code: str = "ok"
    detail: str = ""
    fields: dict = field(default_factory=dict)


class BrokerValidator:
    """
    Stateless validator. Construct once per bot, call per signal.

    Methods return ValidationResult instead of raising so the caller can choose
    to skip, log, fall back to market, etc.
    """

    def __init__(
        self,
        max_spread_points: int = DEFAULT_MAX_SPREAD_POINTS,
        max_data_age_minutes: float = 15.0,
    ) -> None:
        self.max_spread_points = max_spread_points
        self.max_data_age_minutes = max_data_age_minutes

    # ── account / terminal ───────────────────────────────────────────────────-

    def validate_terminal(self) -> ValidationResult:
        """Check terminal is connected and AutoTrading is enabled."""
        ti = mt5.terminal_info()
        if ti is None:
            return ValidationResult(False, "terminal_unreachable",
                                    f"terminal_info() returned None: {mt5.last_error()}")
        if not ti.connected:
            return ValidationResult(False, "broker_disconnected",
                                    "Terminal not connected to broker")
        if not ti.trade_allowed:
            # Matches MT5 retcode 10027 ("AutoTrading disabled by client")
            return ValidationResult(False, "autotrading_disabled",
                                    "AutoTrading is OFF in the terminal")

        ai = mt5.account_info()
        if ai is None:
            return ValidationResult(False, "account_unreachable",
                                    "account_info() returned None")
        if not ai.trade_allowed:
            return ValidationResult(False, "account_trade_disabled",
                                    "Account-side trading is disabled")

        return ValidationResult(True)

    # ── symbol / market ──────────────────────────────────────────────────────-

    def validate_symbol(self, cfg: SymbolConfig) -> ValidationResult:
        """Symbol must be tradable in FULL mode (not CLOSEONLY / DISABLED)."""
        if not cfg.tradable:
            return ValidationResult(False, "symbol_not_tradable",
                                    f"trade_mode={cfg.trade_mode}",
                                    {"trade_mode": cfg.trade_mode})
        return ValidationResult(True)

    def validate_spread(self, cfg: SymbolConfig) -> ValidationResult:
        """Reject when spread is wider than `max_spread_points`."""
        tick = mt5.symbol_info_tick(cfg.name)
        if tick is None:
            return ValidationResult(False, "no_tick", "symbol_info_tick returned None")

        spread_pts = round((tick.ask - tick.bid) / cfg.point) if cfg.point else 0
        if spread_pts > self.max_spread_points:
            return ValidationResult(
                False, "spread_too_high",
                f"spread {spread_pts} pts > {self.max_spread_points}",
                {"spread_pts": spread_pts, "max_pts": self.max_spread_points,
                 "ask": tick.ask, "bid": tick.bid},
            )
        return ValidationResult(True, fields={"spread_pts": spread_pts,
                                              "ask": tick.ask, "bid": tick.bid})

    # ── pending order geometry ───────────────────────────────────────────────-

    def validate_pending_geometry(
        self,
        cfg: SymbolConfig,
        side: str,          # "buy" or "sell"
        entry: float,
        sl: float,
        tp: float,
    ) -> ValidationResult:
        """
        Run the price-grid checks that broker would do server-side -- before
        we send the request -- so we can pick LIMIT vs MARKET locally.

        Rules (mirror MT5 internals):
            BUY_LIMIT:  entry < bid - stops_distance, and SL < entry < TP
            SELL_LIMIT: entry > ask + stops_distance, and TP < entry < SL
            All prices must respect symbol digits / tick_size.
        """
        tick = mt5.symbol_info_tick(cfg.name)
        if tick is None:
            return ValidationResult(False, "no_tick")

        stops_dist = cfg.stops_distance

        # SL/TP side-of-entry sanity
        if side == "buy":
            if not (sl < entry < tp):
                return ValidationResult(False, "geometry_inverted",
                    f"BUY needs sl<entry<tp; got sl={sl} entry={entry} tp={tp}")
            limit_room = tick.bid - entry
            if limit_room <= stops_dist:
                return ValidationResult(
                    False, "limit_inside_stops_level",
                    f"bid {tick.bid} - entry {entry} = {limit_room:.5f} <= stops {stops_dist:.5f}",
                    {"bid": tick.bid, "entry": entry, "stops_dist": stops_dist},
                )
        else:  # sell
            if not (tp < entry < sl):
                return ValidationResult(False, "geometry_inverted",
                    f"SELL needs tp<entry<sl; got tp={tp} entry={entry} sl={sl}")
            limit_room = entry - tick.ask
            if limit_room <= stops_dist:
                return ValidationResult(
                    False, "limit_inside_stops_level",
                    f"entry {entry} - ask {tick.ask} = {limit_room:.5f} <= stops {stops_dist:.5f}",
                    {"ask": tick.ask, "entry": entry, "stops_dist": stops_dist},
                )

        return ValidationResult(True, fields={"limit_room": limit_room,
                                              "stops_dist": stops_dist})

    # ── volume ───────────────────────────────────────────────────────────────-

    def validate_volume(self, cfg: SymbolConfig, volume: float) -> ValidationResult:
        """
        Final volume sanity check before submitting an order. Mirrors what the
        broker would do server-side (retcode 10014 "Invalid volume"):

            * volume >= volume_min       (otherwise broker rejects)
            * volume <= volume_max
            * volume is an exact multiple of volume_step within float tolerance

        ``RiskAdapter.normalize`` already snaps to step + clamps to [min, max],
        so this is a safety-net for any code path that builds a custom volume
        (manual override, future strategies, etc). Cheap to run — call it once
        per order from FallbackEngine right before order_send.
        """
        if volume is None or volume <= 0:
            return ValidationResult(
                False, "volume_not_positive",
                f"volume must be > 0; got {volume!r}",
                {"volume": volume},
            )

        vmin = float(cfg.volume_min)
        vmax = float(cfg.volume_max)
        step = float(cfg.volume_step)

        if volume + 1e-9 < vmin:
            return ValidationResult(
                False, "volume_below_min",
                f"volume {volume} < broker min {vmin}",
                {"volume": volume, "volume_min": vmin},
            )
        if volume > vmax + 1e-9:
            return ValidationResult(
                False, "volume_above_max",
                f"volume {volume} > broker max {vmax}",
                {"volume": volume, "volume_max": vmax},
            )

        # Float-safe step check: volume should land on the broker's step grid
        # within one part in 10^6 of a step.
        if step > 0:
            steps_off = round(volume / step)
            grid_volume = steps_off * step
            if abs(volume - grid_volume) > step * 1e-6:
                return ValidationResult(
                    False, "volume_off_step_grid",
                    f"volume {volume} not a multiple of step {step} (closest grid: {grid_volume})",
                    {"volume": volume, "volume_step": step, "closest_grid": grid_volume},
                )

        return ValidationResult(True, fields={
            "volume": volume, "volume_min": vmin, "volume_max": vmax,
            "volume_step": step,
        })

    # ── market order geometry ────────────────────────────────────────────────-

    def validate_market_stops(
        self,
        cfg: SymbolConfig,
        side: str,
        sl: float,
        tp: float,
    ) -> ValidationResult:
        """
        For a market order, SL/TP must be at least `stops_distance` away from
        the *fill* side of the market (ask for BUY, bid for SELL).
        """
        tick = mt5.symbol_info_tick(cfg.name)
        if tick is None:
            return ValidationResult(False, "no_tick")

        stops_dist = cfg.stops_distance
        price = tick.ask if side == "buy" else tick.bid

        if side == "buy":
            sl_dist = price - sl
            tp_dist = tp - price
        else:
            sl_dist = sl - price
            tp_dist = price - tp

        if sl_dist < stops_dist:
            return ValidationResult(
                False, "sl_inside_stops_level",
                f"sl_dist {sl_dist:.5f} < stops {stops_dist:.5f}",
                {"price": price, "sl": sl, "stops_dist": stops_dist},
            )
        if tp_dist < stops_dist:
            return ValidationResult(
                False, "tp_inside_stops_level",
                f"tp_dist {tp_dist:.5f} < stops {stops_dist:.5f}",
                {"price": price, "tp": tp, "stops_dist": stops_dist},
            )

        return ValidationResult(True, fields={"price": price, "stops_dist": stops_dist})

    # ── orchestrator ─────────────────────────────────────────────────────────-

    def preflight_all(self, cfg: SymbolConfig) -> Optional[ValidationResult]:
        """
        Run the cheap checks (no order-specific data needed). Returns the FIRST
        failure or None if everything is OK -- so the caller can do:

            v = validator.preflight_all(cfg)
            if v is not None:
                logger.event("broker_validation_failed", **v.fields, code=v.code)
                return
        """
        for check in (self.validate_terminal,):
            r = check()
            if not r.ok:
                return r
        for check in (self.validate_symbol, self.validate_spread):
            r = check(cfg)
            if not r.ok:
                return r
        return None
