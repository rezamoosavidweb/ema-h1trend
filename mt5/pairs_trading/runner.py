"""
Main cycle loop for pairs trading.

Lifecycle per cycle (driven by H4 bar close):

    1. Watchdog: ensure MT5 healthy. If not → cycle_skipped.
    2. Fetch aligned panel: H4 bars for all needed symbols.
    3. For each pair in portfolio:
         a. Refit (α, β) on last `train_months` of bars.
         b. ADF gate: if p > adf_gate_p → skip pair this cycle.
         c. Build spread series, rolling z-score.
         d. Look up current pair state.
         e. Decide action (Action enum).
         f. Enact:
              OPEN_LONG / OPEN_SHORT → calculate sizing, PairsExecutionEngine.open_pair()
              EXIT / STOP / TIME_STOP → PairsExecutionEngine.close_pair()
              HOLD → just log spread_evaluated.
    4. Heartbeat (cycle-driven, throttled).
    5. Cycle summary log.
    6. (loop mode) sleep until next bar close + grace.

Everything is logged. Nothing happens silently.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from execution.mt5_watchdog import Mt5Watchdog
from execution.structured_logger import StructuredLogger
from execution.symbol_config import SymbolConfig

from .config import MAGIC_BASE, PairsConfig, PortfolioSpread, TF_HOURS, repo_root
from .data_fetcher import (
    fetch_aligned_panel, fetch_aligned_panel_from_csv,
    seconds_until_next_bar_close,
)
from .pairs_engine import PairsExecutionEngine
from .signals import (
    Action, ActionDecision, BetaFit, Side,
    compute_spread_series, compute_z_series, decide_action, refit_beta,
)
from .sizing import LegSpec, SizingResult, calculate_lot_sizes
from .state import PairsStateStore, PairState


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────


class PairsRunner:
    """
    Owns one cycle's worth of work. Constructed once per process; the runner's
    `run_forever()` is the main loop, `run_once()` does a single cycle.
    """

    def __init__(
        self,
        cfg:          PairsConfig,
        portfolio:    list[PortfolioSpread],
        logger:       StructuredLogger,
        state_store:  PairsStateStore,
        engine:       PairsExecutionEngine,
        watchdog:     Optional[Mt5Watchdog] = None,
    ) -> None:
        self.cfg = cfg
        self.portfolio = portfolio
        self.logger = logger
        self.state_store = state_store
        self.engine = engine
        self.watchdog = watchdog

        self._last_heartbeat: float = 0.0
        self._started_at: float = time.monotonic()

    # ── public entrypoints ─────────────────────────────────────────────────

    def run_once(self) -> dict:
        """Execute exactly one cycle. Returns a summary dict."""
        return self._cycle()

    def run_forever(self) -> None:
        """Loop forever. Sleeps until the next bar close + grace between cycles."""
        try:
            while True:
                self._cycle()
                wait_s = seconds_until_next_bar_close(
                    self.cfg.timeframe, self.cfg.cycle_extra_grace_seconds,
                )
                self.logger.event("loop_sleep", seconds=round(wait_s, 1))
                time.sleep(wait_s)
        except KeyboardInterrupt:
            self.logger.event("bot_stop", reason="KeyboardInterrupt")
            raise

    # ── one cycle ───────────────────────────────────────────────────────────

    def _cycle(self) -> dict:
        cycle_t0 = time.monotonic()
        cycle_id = datetime.now(timezone.utc).isoformat(timespec="seconds")

        self.logger.event(
            "cycle_start",
            cycle_id=cycle_id,
            tf=self.cfg.timeframe,
            n_pairs=len(self.portfolio),
            dry_run=self.cfg.dry_run,
        )

        # 1) Watchdog
        if self.watchdog is not None and not self.watchdog.ensure_healthy():
            self.logger.event("cycle_skipped", reason="mt5_unhealthy")
            return {"cycle_id": cycle_id, "skipped": "mt5_unhealthy"}

        # 2) Fetch aligned panel
        symbols = sorted({s for sp in self.portfolio for s in sp.symbols})
        n_bars = self._bars_needed()
        try:
            panel = fetch_aligned_panel(symbols, self.cfg.timeframe, n_bars)
            panel_source = "mt5"
        except Exception as exc:
            # Dry-run safety net: if the broker can't stream all symbols (e.g.
            # demo restrictions), fall back to the CSV cache so the operator
            # can still validate cycle / signal / sizing logic end-to-end.
            # Production runs (dry_run=False) refuse to fall back — a missing
            # symbol there is a hard fail that needs operator attention.
            if self.cfg.dry_run:
                self.logger.event(
                    "panel_fetch_mt5_failed_falling_back_csv",
                    error=str(exc), symbols=symbols, n_bars=n_bars,
                )
                try:
                    panel = fetch_aligned_panel_from_csv(
                        symbols, self.cfg.timeframe, n_bars,
                        csv_root=repo_root() / "notebooks" / "data",
                    )
                    panel_source = "csv"
                except Exception as exc2:
                    self.logger.error(
                        "panel_fetch_csv_fallback_failed", exc=exc2,
                        symbols=symbols, n_bars=n_bars,
                    )
                    self.logger.event("cycle_skipped", reason="panel_fetch_failed")
                    return {"cycle_id": cycle_id, "skipped": "panel_fetch_failed"}
            else:
                self.logger.error("panel_fetch_failed", exc=exc,
                                  symbols=symbols, n_bars=n_bars)
                self.logger.event("cycle_skipped", reason="panel_fetch_failed")
                return {"cycle_id": cycle_id, "skipped": "panel_fetch_failed"}

        last_bar_iso = panel.index[-1].isoformat()
        self.logger.event(
            "panel_ready",
            cycle_id=cycle_id, rows=len(panel),
            range_start=panel.index[0].isoformat(),
            range_end=last_bar_iso,
            source=panel_source,
        )

        # 3) For each pair → decide and enact
        per_pair_summaries: list[dict] = []
        for spread in self.portfolio:
            summary = self._handle_pair(spread, panel, last_bar_iso)
            per_pair_summaries.append(summary)

        # 4) Heartbeat
        self._heartbeat_if_due()

        # 5) Cycle summary
        elapsed_ms = round((time.monotonic() - cycle_t0) * 1000, 1)
        self.logger.event(
            "cycle_end",
            cycle_id=cycle_id,
            elapsed_ms=elapsed_ms,
            n_pairs_evaluated=len(per_pair_summaries),
            n_actions=sum(1 for s in per_pair_summaries if s.get("action") not in (None, "hold")),
            open_pairs=len(self.state_store.all()),
        )

        return {
            "cycle_id":  cycle_id,
            "elapsed_ms": elapsed_ms,
            "per_pair":  per_pair_summaries,
        }

    # ── per-pair handler ────────────────────────────────────────────────────

    def _handle_pair(
        self, spread: PortfolioSpread, panel: pd.DataFrame, last_bar_iso: str,
    ) -> dict:
        """Evaluate ONE spread for this cycle. Returns a summary dict."""

        # Bars per year derived from TF; train window in bars
        bars_per_year = {"H1": 24 * 252, "H4": 6 * 252}[self.cfg.timeframe]
        train_bars = self.cfg.train_months * bars_per_year // 12

        if len(panel) < train_bars + self.cfg.z_window_bars:
            self.logger.event(
                "spread_evaluated", pair_key=spread.key,
                skipped="insufficient_bars",
                have=len(panel),
                need=train_bars + self.cfg.z_window_bars,
            )
            return {"pair_key": spread.key, "action": None, "skipped": "insufficient_bars"}

        # 1) Refit β/α on training window (exclude the very recent z_window bars
        #    so today's spread isn't fitted in — same convention as NB29).
        train_slice = panel.iloc[-train_bars - self.cfg.z_window_bars : -self.cfg.z_window_bars]
        try:
            fit: BetaFit = refit_beta(
                np.log(train_slice[spread.y].values),
                np.log(train_slice[spread.x].values),
            )
        except Exception as exc:
            self.logger.error("refit_beta_failed", exc=exc, pair_key=spread.key)
            return {"pair_key": spread.key, "action": None, "skipped": "refit_failed"}

        # 2) ADF gate
        if fit.adf_p > self.cfg.adf_gate_p:
            self.logger.event(
                "spread_evaluated", pair_key=spread.key,
                action="hold", reason="adf_gate",
                adf_p=fit.adf_p, gate=self.cfg.adf_gate_p,
                beta=fit.beta, alpha=fit.alpha,
            )
            return {"pair_key": spread.key, "action": "hold", "reason": "adf_gate"}

        # 3) Spread + z-score on full panel
        spread_series = compute_spread_series(
            panel[spread.y], panel[spread.x], fit.alpha, fit.beta,
        )
        z_series = compute_z_series(spread_series, self.cfg.z_window_bars)
        z_now      = float(z_series.iloc[-1])
        spread_now = float(spread_series.iloc[-1])

        # 4) Look up current state
        state = self.state_store.get(spread.key)
        side_now = state.side if state else Side.FLAT
        bars_in_position = state.bars_in_position if state else 0

        # 5) Decide
        time_stop_bars = int(self.cfg.time_stop_bars_mult * spread.half_life_bars)
        decision: ActionDecision = decide_action(
            z_now=z_now,
            side_now=side_now,
            bars_in_position=bars_in_position,
            entry_z=self.cfg.entry_z,
            exit_z=self.cfg.exit_z,
            stop_z=self.cfg.stop_z,
            time_stop_bars=time_stop_bars,
        )

        self.logger.event(
            "spread_evaluated",
            pair_key=spread.key,
            bar_close=last_bar_iso,
            beta=fit.beta, alpha=fit.alpha, adf_p=fit.adf_p,
            spread_now=spread_now, z_now=z_now,
            side_now=side_now.value, bars_in_position=bars_in_position,
            action=decision.action.value, reason=decision.reason,
            time_stop_bars=time_stop_bars,
        )

        # 6) Enact
        action = decision.action
        if action == Action.HOLD:
            # Tick the bar counter for open positions
            if state is not None:
                self.state_store.increment_bars(spread.key)
            return {"pair_key": spread.key, "action": "hold", "z_now": z_now}

        if action in (Action.OPEN_LONG, Action.OPEN_SHORT):
            # Refuse if we've hit max_open_pairs
            n_open = len(self.state_store.all())
            if n_open >= self.cfg.max_open_pairs:
                self.logger.event(
                    "open_skipped_max_pairs",
                    pair_key=spread.key, n_open=n_open, cap=self.cfg.max_open_pairs,
                )
                return {"pair_key": spread.key, "action": "skipped_cap"}

            return self._enact_open(spread, action, fit, spread_now, z_now, last_bar_iso)

        # EXIT / STOP / TIME_STOP
        if action in (Action.EXIT, Action.STOP, Action.TIME_STOP):
            if state is None:
                # Should not happen — decide_action only emits EXIT/STOP from non-FLAT.
                self.logger.event(
                    "close_skipped_no_state",
                    pair_key=spread.key, action=action.value,
                )
                return {"pair_key": spread.key, "action": "noop"}
            ok = self.engine.close_pair(
                state=state, reason=action.value,
                spread_now=spread_now, z_now=z_now,
            )
            return {"pair_key": spread.key, "action": action.value, "closed": ok}

        return {"pair_key": spread.key, "action": "unhandled", "decision": decision.action.value}

    # ── open subroutine ─────────────────────────────────────────────────────

    def _enact_open(
        self,
        spread:       PortfolioSpread,
        action:       Action,
        fit:          BetaFit,
        spread_now:   float,
        z_now:        float,
        last_bar_iso: str,
    ) -> dict:
        # Symbol metadata
        y_cfg = self.engine.resolve_symbol(spread.y)
        x_cfg = self.engine.resolve_symbol(spread.x)

        # Current prices (use bid for SELL leg, ask for BUY leg — sizing uses ask
        # for both as a conservative approximation; engine re-fetches per leg).
        y_tick = mt5.symbol_info_tick(y_cfg.name)
        x_tick = mt5.symbol_info_tick(x_cfg.name)
        if y_tick is None or x_tick is None:
            self.logger.event(
                "open_skipped_no_tick",
                pair_key=spread.key,
                y_tick_ok=y_tick is not None, x_tick_ok=x_tick is not None,
            )
            return {"pair_key": spread.key, "action": action.value, "skipped": "no_tick"}

        y_leg = LegSpec(
            symbol=y_cfg.name, price=float(y_tick.ask),
            contract_size=_contract_size_for(y_cfg.name),
            volume_min=y_cfg.volume_min, volume_step=y_cfg.volume_step,
            volume_max=y_cfg.volume_max,
        )
        x_leg = LegSpec(
            symbol=x_cfg.name, price=float(x_tick.ask),
            contract_size=_contract_size_for(x_cfg.name),
            volume_min=x_cfg.volume_min, volume_step=x_cfg.volume_step,
            volume_max=x_cfg.volume_max,
        )

        # Equity
        ai = mt5.account_info()
        equity = float(ai.equity) if ai else 0.0
        if equity <= 0:
            self.logger.event("open_skipped_no_equity", pair_key=spread.key)
            return {"pair_key": spread.key, "action": action.value, "skipped": "no_equity"}

        try:
            sizing: SizingResult = calculate_lot_sizes(
                y=y_leg, x=x_leg, beta=fit.beta,
                equity_usd=equity,
                risk_per_leg_pct=self.cfg.risk_per_leg_pct,
                assumed_stop_pct=self.cfg.assumed_stop_pct,
            )
        except Exception as exc:
            self.logger.error("sizing_failed", exc=exc, pair_key=spread.key)
            return {"pair_key": spread.key, "action": action.value, "skipped": "sizing_failed"}

        side = Side.LONG if action == Action.OPEN_LONG else Side.SHORT
        new_state = self.engine.open_pair(
            pair_key      = spread.key,
            y_symbol      = spread.y,
            x_symbol      = spread.x,
            side          = side,
            sizing        = sizing,
            beta          = fit.beta,
            alpha         = fit.alpha,
            spread_now    = spread_now,
            z_now         = z_now,
            bar_close_iso = last_bar_iso,
        )
        return {
            "pair_key": spread.key,
            "action":   action.value,
            "opened":   new_state is not None,
            "lots_y":   sizing.lots_y,
            "lots_x":   sizing.lots_x,
        }

    # ── heartbeat ───────────────────────────────────────────────────────────

    def _heartbeat_if_due(self) -> None:
        now = time.monotonic()
        if now - self._last_heartbeat < self.cfg.heartbeat_interval_seconds:
            return
        self._last_heartbeat = now

        ai = mt5.account_info()
        ti = mt5.terminal_info()
        self.logger.event(
            "heartbeat",
            uptime_s=round(now - self._started_at, 1),
            mt5_connected=bool(ti and ti.connected),
            mt5_trade_allowed=bool(ti and ti.trade_allowed),
            open_pairs=len(self.state_store.all()),
            equity=float(ai.equity) if ai else None,
            balance=float(ai.balance) if ai else None,
        )

    # ── helpers ─────────────────────────────────────────────────────────────

    def _bars_needed(self) -> int:
        bars_per_year = {"H1": 24 * 252, "H4": 6 * 252}[self.cfg.timeframe]
        train_bars = self.cfg.train_months * bars_per_year // 12
        return train_bars + self.cfg.z_window_bars + self.cfg.bars_warmup_buffer


# ─────────────────────────────────────────────────────────────────────────────
# Small helper kept here (vs sizing.py) to avoid importing mt5 from sizing.
# ─────────────────────────────────────────────────────────────────────────────


def _contract_size_for(broker_symbol: str) -> float:
    si = mt5.symbol_info(broker_symbol)
    if si is not None:
        return float(si.trade_contract_size or 100_000.0)
    return 100_000.0


# ─────────────────────────────────────────────────────────────────────────────
# Magic-number assignment (one per pair, stable order)
# ─────────────────────────────────────────────────────────────────────────────


def assign_magic_numbers(portfolio: list[PortfolioSpread]) -> dict[str, int]:
    """
    Deterministic magic-number assignment: MAGIC_BASE + sorted_index.

    Sorted by `pair_key` so the same portfolio always yields the same magics,
    regardless of CSV row order — important across restarts so state recovery
    can match positions by magic.
    """
    keys = sorted({s.key for s in portfolio})
    return {key: MAGIC_BASE + idx for idx, key in enumerate(keys)}


# ─────────────────────────────────────────────────────────────────────────────
# Startup state recovery
# ─────────────────────────────────────────────────────────────────────────────


def recover_state_from_broker(
    state_store: PairsStateStore,
    magic_for_pair: dict[str, int],
    logger: StructuredLogger,
) -> None:
    """
    Compare on-disk state with live broker positions filtered by magic.

    Outcomes:
      * state row exists AND broker position exists → keep (no change)
      * state row exists AND broker position MISSING → drop with warning
        (probably closed during downtime — TP/SL/SL/manual)
      * broker position exists AND no state row → log warning, leave alone
        (could be manual user trade with same magic — needs operator review)
    """
    disk_state = state_store.all()
    broker_positions = mt5.positions_get() or []
    our_positions = [p for p in broker_positions if p.magic in magic_for_pair.values()]

    # Build {magic: [positions]}
    by_magic: dict[int, list] = {}
    for p in our_positions:
        by_magic.setdefault(p.magic, []).append(p)

    magic_to_key = {v: k for k, v in magic_for_pair.items()}

    # 1) Dropping stale rows
    for key, st in list(disk_state.items()):
        magic = magic_for_pair.get(key)
        if magic is None:
            # Portfolio changed since last run — drop the orphan state.
            logger.event("state_drop_unknown_pair", pair_key=key)
            state_store.remove(key)
            continue
        broker_ones = by_magic.get(magic, [])
        broker_tickets = {p.ticket for p in broker_ones}
        if st.y_ticket not in broker_tickets and st.x_ticket not in broker_tickets:
            logger.event(
                "state_drop_no_broker_position", pair_key=key,
                y_ticket=st.y_ticket, x_ticket=st.x_ticket,
                broker_tickets=sorted(broker_tickets),
                note="probably closed during downtime (TP/SL/manual)",
            )
            state_store.remove(key)

    # 2) Warning for orphan broker positions
    for magic, plist in by_magic.items():
        key = magic_to_key.get(magic, f"<unknown_magic_{magic}>")
        st = state_store.get(key)
        for p in plist:
            tickets_on_disk = {st.y_ticket, st.x_ticket} if st else set()
            if p.ticket not in tickets_on_disk:
                logger.event(
                    "state_orphan_broker_position",
                    pair_key=key, ticket=p.ticket, symbol=p.symbol,
                    volume=p.volume, magic=magic,
                    action_required="OPERATOR_REVIEW",
                )

    logger.event(
        "state_recovered",
        n_on_disk=len(disk_state),
        n_in_broker=len(our_positions),
        n_in_state_now=len(state_store.all()),
    )
