"""
Orchestrator — the cycle loop that wires every layer together.

One cycle (per fully-closed H1 bar):

  1. resolve equity (paper book MTM, or broker in live mode)
  2. data feed: pull + persist aligned bars for the whole universe
  3. signal engine: evaluate all three sleeves -> CycleSignals (every signal logged)
  4. portfolio engine: size targets into lots + exposure snapshot
  5. risk manager: evaluate hard limits -> gate new entries
  6. reconcile targets vs open positions -> open / close via the execution simulator
  7. persist fills, trades, position lifecycle, equity snapshot, exposure
  8. heartbeat / cycle events

Designed to run unattended: open positions are recovered from storage on startup, so a
restart never double-opens or loses the book.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from .broker_adapter import create_broker
from .broker_adapter.base import BrokerAdapter
from .config import STRATEGY, SystemConfig, load_config
from .data_feed import DataFeed
from .engine_bridge import bars_per_year
from .eventlog import EventLogger
from .execution_simulator import ExecutionSimulator, PaperBook
from .execution_simulator.book import OpenLeg, OpenPosition
from .portfolio_engine import PortfolioEngine, TargetHolding
from .risk import RiskManager
from .signal_engine import SignalEngine, load_or_select_universe
from .signal_engine.types import CycleSignals
from .storage import create_storage


@dataclass
class CycleReport:
    cycle_id: str
    signal_ts: pd.Timestamp
    equity: float
    opened: list[str]
    closed: list[str]
    held: int
    gross_exposure: float
    net_exposure: float
    regime_label: str
    regime_mult: float
    risk_ok: bool


class Orchestrator:
    def __init__(self, config: SystemConfig | None = None, *,
                 broker: BrokerAdapter | None = None, reselect: bool = False) -> None:
        self.cfg = config or load_config()
        self.storage = create_storage(self.cfg)
        self.broker = broker or create_broker(self.cfg)
        self.events = EventLogger(self.storage, self.cfg.log_path())
        self.feed = DataFeed(self.broker, self.storage, timeframe=self.cfg.timeframe)

        self.uni = load_or_select_universe(
            str(self.cfg.data_path()), self.cfg.storage_path(),
            timeframe=self.cfg.timeframe, reselect=reselect,
        )
        self.signal_engine = SignalEngine(self.uni)
        self.portfolio = PortfolioEngine(self.broker, self.cfg)
        self.exec_sim = ExecutionSimulator()
        self.risk = RiskManager(self.cfg, self.storage)
        self.book = PaperBook(self.cfg.starting_equity, bars_per_year())
        self.warmup_bars = max(
            STRATEGY.z_window + STRATEGY.vol_window + self.cfg.warmup_extra_bars, 2000
        )
        self._connected = False
        self._last_processed_bar: pd.Timestamp | None = None  # no-new-bar guard
        # Shadow-live: in addition to the paper book, fire real MT5 demo orders and record
        # their execution next to the simulated fill (for later paper-vs-real reconciliation).
        self.send_live = bool(self.cfg.send_live_orders)

    # ── lifecycle ──────────────────────────────────────────────────────────
    def start(self) -> None:
        self._connected = self.broker.connect()
        self.events.emit("bot_start", message=(
            f"mode={self.cfg.mode} broker={self.broker.name} live_orders={self.send_live} "
            f"pairs={self.uni.pair_keys} strategy={STRATEGY.version}"
        ), payload={"db": self.cfg.resolved_db_url(), "connected": self._connected,
                    "live_orders": self.send_live})
        if self.send_live:
            self.events.emit("live_orders_enabled", severity="warning", message=(
                "LIVE ORDERS ON — real market orders will be sent to the demo account "
                f"(magic={self.cfg.magic_base}, comment={self.cfg.comment_prefix})"))
        if not self._connected:
            self.events.emit("broker_unavailable", severity="warning",
                             message=f"{self.broker.name} did not connect")
        # Warm-up: pre-select every universe symbol so the broker (MT5) starts loading
        # their history immediately. Until that history is fresh, the staleness guard in
        # run_cycle skips trading — this is what prevents the first-cycle stale-data bug.
        if self._connected:
            try:
                ok, missing = self.broker.validate_symbols(self.uni.all_symbols())
                self.events.emit("symbols_warmed_up",
                                 message=f"{len(ok)} symbols selected, {len(missing)} missing",
                                 payload={"missing": missing[:10]})
            except Exception as exc:
                self.events.emit("symbol_warmup_error", severity="warning", message=str(exc))
        # When sending real orders, size the book on the REAL account equity so orders fit
        # the actual account (avoids NO_MONEY rejections / over-leverage) and keeps the
        # paper sim sized identically to the live orders for a fair reconciliation.
        if self.send_live and self._connected:
            try:
                acct = self.broker.account()
                self.book.starting_equity = float(acct.equity)
                self.events.emit("equity_synced", message=(
                    f"sizing on REAL account equity ${acct.equity:,.2f} "
                    f"(balance ${acct.balance:,.2f}) — paper sim matched to it"))
                if acct.equity < 5_000:
                    self.events.emit("low_account_balance", severity="warning", message=(
                        f"account equity ${acct.equity:,.2f} is small — many positions will "
                        f"round below min lot and be skipped; use a larger demo for full sizing"))
            except Exception as exc:
                self.events.emit("account_sync_error", severity="warning", message=str(exc))
        self._recover()

    def stop(self) -> None:
        self.events.emit("bot_stop", message="shutting down")
        try:
            self.broker.disconnect()
        finally:
            self.storage.close()

    def _recover(self) -> None:
        """Rebuild the in-memory paper book from storage's open positions."""
        rows = self.storage.open_positions()
        for r in rows:
            meta = r.get("meta") or {}
            legs_meta = meta.get("legs", [])
            if not legs_meta:
                continue
            legs = [OpenLeg(
                symbol=l["symbol"], direction=int(l["direction"]), lots=float(l["lots"]),
                contract_size=float(l["contract_size"]), entry_actual=float(l["entry_actual"]),
                entry_mid=float(l["entry_mid"]), carry_rate_annual=float(l.get("carry_rate", 0.0)),
                entry_slippage_bps=float(l.get("entry_slip", 0.0)),
                broker_ticket=int(l.get("broker_ticket") or 0),
            ) for l in legs_meta]
            pos = OpenPosition(
                pair_key=r["pair_key"], kind=meta.get("kind", "reversion"), legs=legs,
                opened_ts=r.get("opened_at") or datetime.now(timezone.utc),
                signal_ts=meta.get("signal_ts") or datetime.now(timezone.utc),
                z_at_open=float(r.get("z_at_open") or 0.0),
                beta_at_open=float(r.get("beta_at_open") or 0.0),
                regime_at_open=r.get("regime_at_open") or "", side=r.get("side") or "",
                storage_id=r.get("id"), bars_held=int(meta.get("bars_held", 0)),
            )
            self.book.open(pos)
        if rows:
            self.events.emit("state_recovered", message=f"recovered {len(self.book.positions)} positions")

    def _bars_behind_now(self, last_ts: pd.Timestamp) -> float:
        """How many `timeframe` bars the panel's latest bar lags wall-clock now."""
        tf_hours = {"H1": 1, "H4": 4, "D1": 24}.get(self.cfg.timeframe, 1)
        tz = getattr(last_ts, "tz", None)
        now = pd.Timestamp.now(tz=tz) if tz is not None else pd.Timestamp.now(tz=timezone.utc)
        return (now - last_ts).total_seconds() / 3600.0 / tf_hours

    # ── one cycle ────────────────────────────────────────────────────────────
    def run_cycle(self, *, as_of: pd.Timestamp | None = None) -> CycleReport | None:
        cycle_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        self.events.emit("cycle_start", cycle_id=cycle_id, message=f"as_of={as_of}")

        # replay clock for the sim broker
        if as_of is not None and hasattr(self.broker, "set_clock"):
            self.broker.set_clock(as_of)

        feed = self.feed.pull(self.uni.all_symbols(), self.warmup_bars)
        if not feed.ok:
            self.events.emit("cycle_skipped", cycle_id=cycle_id, severity="warning",
                             message=f"feed not ok; missing={feed.missing_symbols[:5]}",
                             payload={"dropped_rows": feed.dropped_rows})
            return None
        panel = feed.panel
        last_ts = feed.last_ts if feed.last_ts is not None else panel.index[-1]

        # ── staleness guard (live MT5 only) ──────────────────────────────────
        # Never trade on a panel whose latest bar is far behind wall-clock. This is the
        # core fix for the bug where the first cycle after connect aligned the inner-join
        # to a weeks-old common bar (freshly selected symbols returned stale history).
        if as_of is None and self.broker.name == "mt5":
            bars_behind = self._bars_behind_now(last_ts)
            if bars_behind > self.cfg.max_staleness_bars:
                self.events.emit("cycle_skipped", cycle_id=cycle_id, severity="warning",
                                 message=(f"stale data: last bar {last_ts} is "
                                          f"{bars_behind:.1f} bars behind now — not trading"),
                                 payload={"last_ts": str(last_ts),
                                          "bars_behind": round(bars_behind, 1)})
                return None

        # ── no-new-bar guard ─────────────────────────────────────────────────
        # If the latest closed bar hasn't advanced since last cycle (weekends, market
        # closed, or sub-bar polling), skip before doing any work — avoids duplicate
        # signals and the equity.ts duplicate-insert crash.
        if self._last_processed_bar is not None and last_ts == self._last_processed_bar:
            self.events.emit("cycle_skipped", cycle_id=cycle_id,
                             message=f"no new bar since {last_ts}",
                             payload={"last_ts": str(last_ts)})
            return None
        self._last_processed_bar = last_ts

        prices = {s: float(panel[s].iloc[-1]) for s in panel.columns}

        signals = self.signal_engine.evaluate(panel, cycle_id)
        self._log_signals(signals)

        # keep the sim account equity in step with the paper book
        equity = self.book.equity(prices)
        if hasattr(self.broker, "set_equity"):
            self.broker.set_equity(equity)

        risk = self.risk.evaluate(equity=equity, gross_exposure=self.book.gross_notional() / max(equity, 1e-9))
        if not risk.ok:
            self.events.emit("risk_breach", cycle_id=cycle_id, severity="critical",
                             message="; ".join(risk.breaches), payload=risk.metrics)

        targets, exposure = self.portfolio.build_targets(signals, equity, prices)
        opened, closed = self._reconcile(cycle_id, signals, targets, feed.spreads_bps,
                                         prices, allow_new=risk.allow_new_entries)

        self.book.increment_bars()
        equity = self.book.equity(prices)
        self._record_equity(cycle_id, signals, equity, prices)

        report = CycleReport(
            cycle_id=cycle_id, signal_ts=signals.signal_ts, equity=equity,
            opened=opened, closed=closed, held=len(self.book.positions),
            gross_exposure=exposure.gross_exposure, net_exposure=exposure.net_exposure,
            regime_label=signals.regime.label, regime_mult=signals.regime.multiplier,
            risk_ok=risk.allow_new_entries,
        )
        self.events.emit("cycle_end", cycle_id=cycle_id, message=(
            f"eq={equity:,.0f} open={report.held} +{len(opened)} -{len(closed)} "
            f"gross={exposure.gross_exposure:.2f}x regime={signals.regime.label}"
        ), payload={"opened": opened, "closed": closed})
        return report

    # ── signal logging ────────────────────────────────────────────────────────
    def _log_signals(self, signals: CycleSignals) -> None:
        prov = self.signal_engine.provenance()
        for ps in signals.pair_signals:
            carry_val = self.signal_engine.carry_value_for(signals, ps)
            self.storage.record_signal(
                ps.as_signal_row(signals.cycle_id, signals.signal_ts.to_pydatetime(),
                                 signals.regime, carry_val, prov)
            )

    # ── reconciliation ─────────────────────────────────────────────────────────
    def _reconcile(self, cycle_id: str, signals: CycleSignals,
                   targets: list[TargetHolding], spreads: dict[str, float],
                   prices: dict[str, float], *, allow_new: bool) -> tuple[list[str], list[str]]:
        opened: list[str] = []
        closed: list[str] = []
        target_by_key = {t.pair_key: t for t in targets}

        # 1) close positions whose target is now flat or flipped sign
        for key in list(self.book.positions.keys()):
            t = target_by_key.get(key)
            cur = self.book.get(key)
            desired_dir = _holding_direction(t) if t else 0
            cur_dir = 1 if cur.side == "long" else -1
            if t is None or desired_dir == 0 or desired_dir != cur_dir:
                self._close_holding(cycle_id, key, prices, spreads, reason="z_exit")
                closed.append(key)

        # 2) open new positions where target is non-flat and not already held
        if allow_new:
            for t in targets:
                if self.book.has(t.pair_key):
                    continue
                if abs(t.target_position) < 1e-9 or not any(l.lots > 0 for l in t.legs):
                    continue
                self._open_holding(cycle_id, signals, t, prices, spreads)
                opened.append(t.pair_key)
        return opened, closed

    # ── real (shadow) order helpers ──────────────────────────────────────────
    def _send_real_order(self, cycle_id: str, symbol: str, side: str, lots: float,
                         *, kind: str) -> dict:
        """Fire a real MT5 market order (when live_orders is on) and return the broker
        fields to store next to the simulated fill. No-op (exec_mode='paper') otherwise."""
        if not self.send_live:
            return {"exec_mode": "paper"}
        try:
            res = self.broker.market_order(
                symbol, side, lots, magic=self.cfg.magic_base,
                comment=f"{self.cfg.comment_prefix}:{kind}")
        except Exception as exc:
            self.events.emit("live_order_error", severity="error", cycle_id=cycle_id,
                             message=f"{symbol} {side} {lots}: {exc}")
            return {"exec_mode": "live", "broker_ok": False, "broker_comment": str(exc)[:64]}
        if not res.ok:
            self.events.emit("live_order_rejected", severity="error", cycle_id=cycle_id,
                             message=f"{symbol} {side} {lots}: {res.comment}")
        return {
            "exec_mode": "live",
            "broker_ticket": int(res.ticket) or None,
            "broker_fill_price": float(res.filled_price) or None,
            "broker_fill_ts": datetime.now(timezone.utc),
            "broker_latency_ms": float(res.latency_ms),
            "broker_ok": bool(res.ok),
            "broker_comment": (res.comment or "")[:64],
        }

    def _close_real_ticket(self, cycle_id: str, ticket: int) -> dict:
        if not self.send_live or not ticket:
            return {"exec_mode": "paper"}
        try:
            res = self.broker.close_ticket(int(ticket))
        except Exception as exc:
            self.events.emit("live_close_error", severity="error", cycle_id=cycle_id,
                             message=f"ticket {ticket}: {exc}")
            return {"exec_mode": "live", "broker_ok": False, "broker_comment": str(exc)[:64]}
        if not res.ok:
            self.events.emit("live_close_rejected", severity="error", cycle_id=cycle_id,
                             message=f"ticket {ticket}: {res.comment}")
        return {
            "exec_mode": "live", "broker_ticket": int(ticket),
            "broker_fill_price": float(res.filled_price) or None,
            "broker_fill_ts": datetime.now(timezone.utc),
            "broker_latency_ms": float(res.latency_ms),
            "broker_ok": bool(res.ok), "broker_comment": (res.comment or "")[:64],
        }

    def _open_holding(self, cycle_id: str, signals: CycleSignals, t: TargetHolding,
                      prices: dict[str, float], spreads: dict[str, float]) -> None:
        legs: list[OpenLeg] = []
        fill_rows: list[dict] = []
        side = "long" if t.target_position > 0 else "short"
        for leg in t.legs:
            if leg.lots <= 0:
                continue
            ref = prices.get(leg.symbol, leg.price)
            fill = self.exec_sim.fill(leg.symbol, leg.side, leg.lots, ref,
                                      spreads.get(leg.symbol, float("nan")))
            info = self.portfolio._info(leg.symbol)
            direction = 1 if leg.side == "buy" else -1
            broker = self._send_real_order(cycle_id, leg.symbol, leg.side, leg.lots,
                                           kind="entry")
            legs.append(OpenLeg(
                symbol=leg.symbol, direction=direction, lots=leg.lots,
                contract_size=info.contract_size, entry_actual=fill.actual_price,
                entry_mid=ref, carry_rate_annual=signals.carry_rates.get(leg.symbol, 0.0),
                entry_slippage_bps=fill.slippage_bps,
                broker_ticket=int(broker.get("broker_ticket") or 0),
            ))
            fill_rows.append({
                "pair_key": t.pair_key, "leg": "y" if leg is t.legs[0] else "x",
                "symbol": leg.symbol, "kind": "entry", "side": leg.side, "volume": leg.lots,
                "intended_price": ref, "actual_price": fill.actual_price,
                "slippage_bps": fill.slippage_bps, "spread_bps": fill.spread_bps,
                "latency_ms": fill.latency_ms,
                "signal_ts": signals.signal_ts.to_pydatetime(),
                "fill_ts": datetime.now(timezone.utc),
                **broker,
            })
        if not legs:
            return

        ps = next((p for p in signals.pair_signals if p.pair_key == t.pair_key), None)
        pos = OpenPosition(
            pair_key=t.pair_key, kind=t.kind, legs=legs,
            opened_ts=datetime.now(timezone.utc), signal_ts=signals.signal_ts,
            z_at_open=ps.zscore if ps else 0.0, beta_at_open=t.beta,
            alpha_at_open=ps.alpha if ps else 0.0,
            regime_at_open=signals.regime.label, carry_value=t.meta.get("carry_rate", 0.0),
            side=side,
        )
        meta = {
            "kind": t.kind, "bars_held": 0, "signal_ts": str(signals.signal_ts),
            "legs": [{"symbol": l.symbol, "direction": l.direction, "lots": l.lots,
                      "contract_size": l.contract_size, "entry_actual": l.entry_actual,
                      "entry_mid": l.entry_mid, "carry_rate": l.carry_rate_annual,
                      "entry_slip": l.entry_slippage_bps,
                      "broker_ticket": l.broker_ticket} for l in legs],
        }
        # Track in the in-memory book FIRST. If the real order(s) already fired, the
        # position is real — a storage failure must NOT prevent tracking, otherwise the
        # next cycle would think the pair is flat and re-open it (duplicate real orders).
        self.book.open(pos)
        try:
            pos.storage_id = self.storage.open_position({
                "pair_key": t.pair_key, "y_symbol": legs[0].symbol,
                "x_symbol": legs[1].symbol if len(legs) > 1 else None, "side": side,
                "opened_cycle": cycle_id, "opened_at": pos.opened_ts,
                "y_volume": legs[0].lots, "x_volume": legs[1].lots if len(legs) > 1 else None,
                "beta_at_open": t.beta, "alpha_at_open": pos.alpha_at_open,
                "z_at_open": pos.z_at_open, "regime_at_open": signals.regime.label,
                "gross_notional": pos.gross_notional, "meta": meta,
            })
            for fr in fill_rows:
                fr["position_id"] = pos.storage_id
                self.storage.record_fill(fr)
        except Exception as exc:
            # Position is tracked in-memory; just couldn't persist. Loud, but non-fatal.
            self.events.emit("storage_write_failed", severity="error", cycle_id=cycle_id,
                             message=f"{t.pair_key} open not persisted (check DB schema): {exc}")
        self.events.emit("pair_open", cycle_id=cycle_id,
                         message=f"{t.pair_key} {side} gross={pos.gross_notional:,.0f}",
                         payload={"z": pos.z_at_open, "kind": t.kind})

    def _close_holding(self, cycle_id: str, key: str, prices: dict[str, float],
                       spreads: dict[str, float], *, reason: str) -> None:
        pos = self.book.get(key)
        if pos is None:
            return
        exit_fills = {}
        for l in pos.legs:
            exit_side = "sell" if l.direction > 0 else "buy"
            ref = prices.get(l.symbol, l.entry_mid)
            f = self.exec_sim.fill(l.symbol, exit_side, l.lots, ref,
                                   spreads.get(l.symbol, float("nan")))
            exit_fills[l.symbol] = f
            broker = self._close_real_ticket(cycle_id, l.broker_ticket)
            self.storage.record_fill({
                "position_id": pos.storage_id, "pair_key": key, "symbol": l.symbol,
                "kind": "exit", "side": exit_side, "volume": l.lots,
                "intended_price": ref, "actual_price": f.actual_price,
                "slippage_bps": f.slippage_bps, "spread_bps": f.spread_bps,
                "latency_ms": f.latency_ms, "fill_ts": datetime.now(timezone.utc),
                **broker,
            })
        trade = self.book.close(key, exit_fills, prices)
        if trade is None:
            return
        ps = None  # current-bar z for exit logging
        if pos.storage_id is not None:
            self.storage.close_position(pos.storage_id, {
                "closed_at": datetime.now(timezone.utc), "realized_pnl": trade.realized_pnl,
            })
        self.storage.record_trade({
            "position_id": pos.storage_id, "pair_key": key,
            "y_symbol": pos.legs[0].symbol,
            "x_symbol": pos.legs[1].symbol if len(pos.legs) > 1 else None,
            "side": pos.side, "signal_ts": pos.signal_ts.to_pydatetime() if hasattr(pos.signal_ts, "to_pydatetime") else pos.signal_ts,
            "entry_ts": pos.opened_ts, "exit_ts": datetime.now(timezone.utc),
            "regime": pos.regime_at_open, "carry_value": pos.carry_value,
            "zscore_entry": pos.z_at_open, "hedge_ratio": pos.beta_at_open,
            "position_size": pos.legs[0].lots, "gross_notional": trade.gross_notional,
            "entry_slippage_bps": trade.entry_slippage_bps,
            "exit_slippage_bps": trade.exit_slippage_bps,
            "realized_pnl": trade.realized_pnl, "reversion_pnl": trade.reversion_pnl,
            "carry_pnl": trade.carry_pnl, "cost_pnl": trade.cost_pnl,
            "bars_held": trade.bars_held, "exit_reason": reason,
        })
        self.events.emit("pair_close", cycle_id=cycle_id,
                         message=f"{key} pnl={trade.realized_pnl:,.2f} ({reason})",
                         payload={"reversion": trade.reversion_pnl, "carry": trade.carry_pnl,
                                  "cost": trade.cost_pnl, "bars": trade.bars_held})

    def _record_equity(self, cycle_id: str, signals: CycleSignals, equity: float,
                       prices: dict[str, float]) -> None:
        gross = self.book.gross_notional()
        contribs = self.book.pair_contributions(prices)
        net = sum(p.legs[0].direction * abs(p.legs[0].notional) for p in self.book.positions.values())
        # daily pnl from realised trades today
        self.storage.record_equity({
            "cycle_id": cycle_id, "ts": signals.signal_ts.to_pydatetime(),
            "equity": equity, "cash": self.cfg.starting_equity + self.book.realized_cum,
            "gross_exposure": gross / max(equity, 1e-9),
            "net_exposure": net / max(equity, 1e-9),
            "leverage": gross / max(equity, 1e-9), "open_pairs": len(self.book.positions),
            "regime_state": signals.regime.label, "regime_multiplier": signals.regime.multiplier,
            "pair_contributions": contribs,
        })

    # ── loops ────────────────────────────────────────────────────────────────
    def run_once(self) -> CycleReport | None:
        self.start()
        try:
            return self.run_cycle()
        finally:
            self.stop()

    def replay(self, *, start: str | None = None, end: str | None = None,
               step: int = 1, max_cycles: int | None = None) -> list[CycleReport]:
        """Drive the full pipeline deterministically over historical bars (sim broker only).

        This is the backtest-parity harness: it walks the bar index of the regime proxy
        symbol and runs a cycle at each (stepped) close, so the live code path is exercised
        against the same data the research engine backtested. Returns per-cycle reports.
        """
        if not hasattr(self.broker, "set_clock"):
            raise RuntimeError("replay requires the sim broker (set SAL_BROKER=sim)")
        self.start()
        reports: list[CycleReport] = []
        try:
            proxy = STRATEGY.regime_proxy_symbol
            bars = self.broker.get_bars(proxy, self.cfg.timeframe, 100_000).index
            if start:
                bars = bars[bars >= pd.Timestamp(start, tz=bars.tz)]
            if end:
                bars = bars[bars <= pd.Timestamp(end, tz=bars.tz)]
            bars = bars[::step]
            if max_cycles:
                bars = bars[:max_cycles]
            for ts in bars:
                rep = self.run_cycle(as_of=ts)
                if rep is not None:
                    reports.append(rep)
        finally:
            self.stop()
        return reports

    def run_forever(self) -> None:
        from .data_feed import DataFeed  # noqa
        self.start()
        try:
            while True:
                try:
                    self.run_cycle()
                except Exception as exc:  # keep the loop alive
                    self.events.emit("cycle_error", severity="error", message=str(exc))
                wait = _seconds_until_next_bar(self.cfg.timeframe, self.cfg.broker_tz,
                                               self.cfg.cycle_grace_seconds)
                self.events.emit("heartbeat", message=f"sleeping {wait:.0f}s until next bar")
                if self.cfg.once:
                    break
                time.sleep(wait)
        finally:
            self.stop()


# ── helpers ─────────────────────────────────────────────────────────────────
def _holding_direction(t: TargetHolding | None) -> int:
    if t is None or abs(t.target_position) < 1e-9:
        return 0
    return 1 if t.target_position > 0 else -1


def _seconds_until_next_bar(tf: str, tz_name: str, grace: int) -> float:
    from zoneinfo import ZoneInfo
    hours = {"H1": 1, "H4": 4, "D1": 24}.get(tf, 1)
    now = pd.Timestamp.now(tz=ZoneInfo(tz_name))
    nb = (now.hour // hours + 1) * hours
    if nb >= 24:
        nxt = now.normalize() + pd.Timedelta(days=1)
    else:
        nxt = now.normalize() + pd.Timedelta(hours=nb)
    return max((nxt - now).total_seconds() + grace, 1.0)
