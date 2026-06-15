"""
`statarb_live health` — a one-shot health + strategy-correctness snapshot.

Reads the event logs and the storage DB and writes a compact, human-readable report to
``logs/statarb_live/health.txt`` (which IS committed to git, so you can ``git pull`` it and
read it without transferring the DB). Also printed to stdout.

It answers two questions at a glance:
  1. Is the bot ALIVE and error-free?  (last cycle recency, error counts, heartbeats)
  2. Is it running the FROZEN strategy CORRECTLY?  (the 6 NB38 pairs, frozen params,
     live-order health, regime sizing reacting, recent trades/attribution)
"""

from __future__ import annotations

import glob
import json
from datetime import datetime, timezone
from collections import Counter
from pathlib import Path

from .config import STRATEGY, SystemConfig
from .signal_engine.universe import Universe


def _load_events(log_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for f in sorted(glob.glob(str(log_dir / "events-*.json"))):
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def _age_str(ts_iso: str) -> tuple[float, str]:
    try:
        t = datetime.fromisoformat(ts_iso)
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        mins = (datetime.now(timezone.utc) - t).total_seconds() / 60.0
        return mins, f"{mins:.0f} min ago" if mins < 120 else f"{mins/60:.1f} h ago"
    except Exception:
        return 1e9, "unknown"


def build_health(config: SystemConfig) -> str:
    L: list[str] = []
    now = datetime.now(timezone.utc)
    L.append("=" * 64)
    L.append(f" statarb_live HEALTH  —  generated {now:%Y-%m-%d %H:%M:%S} UTC")
    L.append("=" * 64)

    warnings: list[str] = []
    rows = _load_events(config.log_path())

    # ── 1) liveness ─────────────────────────────────────────────────────────
    L.append("\n[1] LIVENESS")
    if not rows:
        L.append("  no events found — bot may never have started.")
        warnings.append("no events")
    else:
        last = rows[-1]
        age_min, age = _age_str(last["ts"])
        L.append(f"  events           : {len(rows)}  ({rows[0]['ts'][:19]} -> {rows[-1]['ts'][:19]})")
        L.append(f"  last event       : {last['event_type']}  ({age})")
        ce = [r for r in rows if r["event_type"] == "cycle_end"]
        L.append(f"  last cycle_end   : {ce[-1]['message'] if ce else '(none yet)'}")
        cs = [r for r in rows if r["event_type"] == "cycle_skipped"]
        if cs:
            L.append(f"  last skip        : {cs[-1]['message'][:70]}")
        st = [r for r in rows if r["event_type"] == "bot_start"]
        if st:
            L.append(f"  last bot_start   : {st[-1]['message'][:90]}")
        L.append(f"  event counts     : {dict(Counter(r['event_type'] for r in rows))}")
        # errors
        errs = [r for r in rows if r.get("severity") == "error"
                or r["event_type"] in ("cycle_error", "storage_write_failed",
                                        "live_order_rejected", "orphan_close_error")]
        L.append(f"  errors (all)     : {len(errs)}")
        for r in errs[-4:]:
            L.append(f"     ! {r['ts'][:19]} {r['event_type']}: {r['message'][:64]}")
        if errs:
            warnings.append(f"{len(errs)} error events")
        if age_min > 130:   # > ~2 H1 bars with no event (weekends excepted by the operator)
            warnings.append(f"last event {age} (bot may be stopped/stuck)")

    # ── 2) strategy correctness ─────────────────────────────────────────────
    L.append("\n[2] STRATEGY (must be the FROZEN NB38 config)")
    L.append(f"  version          : {STRATEGY.version}")
    L.append(f"  signal           : z_entry={STRATEGY.z_entry} z_exit={STRATEGY.z_exit} "
             f"z_stop={STRATEGY.z_stop} z_window={STRATEGY.z_window}")
    L.append(f"  sizing           : target_ann_vol={STRATEGY.target_ann_vol} "
             f"vol_window={STRATEGY.vol_window} max_lev={STRATEGY.max_leverage}")
    L.append(f"  carry / regime   : w_rev={STRATEGY.carry_w_rev}  regime={STRATEGY.regime_method}"
             f"/{STRATEGY.regime_max_mult}")
    EXPECTED = {"CHFJPY~EURJPY", "CADCHF~EURCAD", "AUDJPY~EURJPY",
                "AUDUSD~USDCAD", "AUDJPY~CHFJPY", "EURCAD~EURGBP"}
    try:
        uni = Universe.load(config.storage_path() / "universe.json")
        keys = set(uni.pair_keys)
        match = keys == EXPECTED
        L.append(f"  universe pairs   : {sorted(keys)}")
        L.append(f"  matches NB38     : {'YES ✓' if match else 'NO ✗ — universe drifted!'}")
        if not match:
            warnings.append("universe pairs != frozen NB38 set")
    except Exception as exc:
        L.append(f"  universe         : <could not load: {exc}>")
        warnings.append("universe.json missing")

    # ── 3) book / account / live-order health (from DB) ─────────────────────
    L.append("\n[3] BOOK / ACCOUNT (from DB)")
    try:
        from .storage import create_storage
        st = create_storage(config, init=False)
        try:
            openpos = st.open_positions()
            L.append(f"  open positions   : {len(openpos)}")
            for p in openpos[:8]:
                L.append(f"     - {p.get('pair_key'):16s} {p.get('side') or '':5s} "
                         f"z@open={p.get('z_at_open')}  regime={p.get('regime_at_open')}")
            eq = st.last_equity()
            if eq:
                L.append(f"  last equity      : {eq.get('equity'):,.2f}  gross={eq.get('gross_exposure'):.2f}x "
                         f"net={eq.get('net_exposure'):.2f}x  regime={eq.get('regime_state')} "
                         f"(mult={eq.get('regime_multiplier'):.2f})")
            # live-order health from fills
            fills = st.fetch_df("fills")
            if not fills.empty and "exec_mode" in fills.columns:
                live = fills[fills["exec_mode"] == "live"]
                if len(live):
                    okrate = float((live["broker_ok"] == True).mean())  # noqa: E712
                    L.append(f"  live orders      : {len(live)} sent, fill_ok={okrate:.0%}")
                    if okrate < 0.9:
                        warnings.append(f"live order fill rate {okrate:.0%}")
            # recent closed trades + attribution
            trades = st.fetch_df("trades")
            if not trades.empty:
                import pandas as pd
                trades["exit_ts"] = pd.to_datetime(trades["exit_ts"], errors="coerce")
                t = trades.sort_values("exit_ts").tail(5)
                L.append(f"  trades (total {len(trades)}) — last {len(t)}:")
                for _, r in t.iterrows():
                    L.append(f"     {str(r.get('pair_key'))[:16]:16s} pnl={float(r.get('realized_pnl') or 0):+8.2f} "
                             f"(rev={float(r.get('reversion_pnl') or 0):+.2f} carry={float(r.get('carry_pnl') or 0):+.2f} "
                             f"cost={float(r.get('cost_pnl') or 0):+.2f}) {r.get('exit_reason')}")
                tot = float(trades["realized_pnl"].fillna(0).sum())
                L.append(f"  realized PnL     : {tot:+,.2f}")

            # ── 4) paper (logged/replay-style) vs REAL broker fills ─────────
            L.append("\n[4] PAPER (simulated) vs REAL broker fills")
            try:
                from .reconcile import build_reconciliation, summarize
                recon = build_reconciliation(st)
                rs = summarize(recon, st)
                if rs.get("filled_ok"):
                    L.append(f"  real fills        : {rs['filled_ok']} ok / {rs.get('live_fills_attempted')} sent "
                             f"(fill rate {rs.get('fill_rate')})")
                    L.append(f"  price gap (pips)  : median {rs.get('price_gap_pips_median')}  "
                             f"p95 {rs.get('price_gap_pips_p95')}   (real fill - simulated fill)")
                    L.append(f"  slippage (pips)   : simulated {rs.get('slip_paper_pips_median')}  vs  "
                             f"real {rs.get('slip_real_pips_median')}")
                    L.append(f"  time gap (s)      : log->real {rs.get('time_gap_s_median')}  "
                             f"signal->real {rs.get('signal_to_real_s_median')}")
                    L.append(f"  latency (ms)      : simulated {rs.get('latency_paper_ms_median')}  vs  "
                             f"real {rs.get('latency_real_ms_median')}")
                else:
                    L.append(f"  {rs.get('note', 'no real fills yet')}")
            except Exception as exc:
                L.append(f"  <reconcile unavailable: {exc}>")
        finally:
            st.close()
    except Exception as exc:
        L.append(f"  <DB unavailable: {exc}>")

    # ── verdict ─────────────────────────────────────────────────────────────
    L.append("\n" + "=" * 64)
    if warnings:
        L.append(" VERDICT: ⚠️  ATTENTION — " + "; ".join(warnings))
    else:
        L.append(" VERDICT: ✅  HEALTHY — running the frozen strategy, no errors")
    L.append("=" * 64)
    return "\n".join(L)


def run_health(config: SystemConfig, *, write: bool = True) -> tuple[str, Path | None]:
    text = build_health(config)
    path = None
    if write:
        path = config.log_path() / "health.txt"
        path.write_text(text + "\n", encoding="utf-8")
    return text, path
