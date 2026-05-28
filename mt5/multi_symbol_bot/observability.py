"""
Observability helpers — pure functions that the runner and execution engine
call to enrich log events.

Design rules (locked):
  * Pure functions only. No I/O at module scope.
  * Never raises in the trading-hot path. Every helper has a safe fallback.
  * O(1) per cycle. Hash is computed ONCE at startup (run_config_hash).
  * Adds fields only; never mutates inputs.

Output of this module is consumed by:
  * mt5/run_multi_scalper.py    (cycle / bot_run_started / portfolio events)
  * execution/execution_engine.py (position_open skip enrichment)

After 30 days of production logs, these fields let us answer:
  - Did D1 synth produce negative-expectancy trades?  (filter on htf_source.d1_source == "synth")
  - How many signals were blocked by ONE_TRADE vs truly absent?  (skip events with cascade_id)
  - Did config drift across restarts?  (compare run_id × config_hash)
  - Can every live trade be replayed exactly?  (bar_integrity + htf_source + run_config snapshot)
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════
# RUN ID  — one UUID per process invocation
# ═══════════════════════════════════════════════════════════════════════════

def make_run_id() -> str:
    """UUID-v4 string. Stable for the whole process; regenerated on restart."""
    return str(uuid.uuid4())


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG HASHING
# ═══════════════════════════════════════════════════════════════════════════

def _canonicalize(obj: Any) -> Any:
    """Recursively convert any value to a JSON-stable shape (sorted keys,
    primitive leaves). Tolerates dataclasses, Paths, tuples, sets, frozensets."""
    if is_dataclass(obj):
        obj = asdict(obj)
    if isinstance(obj, dict):
        return {str(k): _canonicalize(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_canonicalize(x) for x in obj]
    if isinstance(obj, (set, frozenset)):
        return [_canonicalize(x) for x in sorted(obj, key=lambda v: str(v))]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (int, float, bool, str)) or obj is None:
        return obj
    # Fallback — never raise from the hashing pipeline.
    return str(obj)


def sha256_of(obj: Any) -> str:
    canonical = json.dumps(_canonicalize(obj), sort_keys=True, separators=(",", ":"),
                           default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_symbol_config_hash(strategy_cfg: Any,
                                slippage_max_points: float,
                                risk_per_trade: float,
                                risk_reward: float) -> str:
    """Hash everything that affects ONE symbol's live behaviour."""
    payload = {
        "strategy":            _canonicalize(strategy_cfg),
        "slippage_max_points": float(slippage_max_points),
        "risk_per_trade":      float(risk_per_trade),
        "risk_reward":         float(risk_reward),
    }
    return sha256_of(payload)


def compute_portfolio_config_hash(per_symbol_payloads: dict,
                                    runner_constants: dict,
                                    htf_policy: dict) -> str:
    """Hash everything that affects portfolio-level behaviour."""
    payload = {
        "per_symbol":        _canonicalize(per_symbol_payloads),
        "runner_constants":  _canonicalize(runner_constants),
        "htf_policy":        _canonicalize(htf_policy),
    }
    return sha256_of(payload)


# ═══════════════════════════════════════════════════════════════════════════
# GIT COMMIT (best-effort)
# ═══════════════════════════════════════════════════════════════════════════

def current_git_commit() -> Optional[str]:
    """Return the current git HEAD commit hash, or None if unavailable.
    Never raises; never blocks for more than 1 second."""
    try:
        cwd = Path(__file__).resolve().parent.parent.parent
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(cwd),
            stderr=subprocess.DEVNULL, timeout=1.0,
        )
        return out.decode("ascii", errors="ignore").strip() or None
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
# HTF POLICY SNAPSHOT
# ═══════════════════════════════════════════════════════════════════════════

def htf_policy_snapshot(
    use_synth_h1: bool = True,
    use_synth_d1: bool = True,
    h1_freshness_threshold_min: float = 0.0,
    d1_freshness_threshold_min: float = 0.0,
) -> dict:
    """Current effective HTF policy. Default reflects existing live
    behaviour (always-synth both). Override via constants in run_multi_scalper.py
    when phasing in the C_15 policy from HTF_POLICY_REPORT.md."""
    return {
        "use_synth_h1": bool(use_synth_h1),
        "use_synth_d1": bool(use_synth_d1),
        "h1_freshness_threshold_min": float(h1_freshness_threshold_min),
        "d1_freshness_threshold_min": float(d1_freshness_threshold_min),
        "h1_synth_enabled": bool(use_synth_h1),
        "d1_synth_enabled": bool(use_synth_d1),
    }


# ═══════════════════════════════════════════════════════════════════════════
# HTF SOURCE TRACEABILITY
# ═══════════════════════════════════════════════════════════════════════════

H1_DURATION = pd.Timedelta(hours=1)
D1_DURATION = pd.Timedelta(days=1)


def _freshness_min(last_bar_time: Any, bar_duration: pd.Timedelta,
                    now_broker_ts: pd.Timestamp) -> float:
    """Minutes since the most recent broker bar fully closed. Never negative."""
    if last_bar_time is None or pd.isna(last_bar_time):
        return float("inf")
    last_close = pd.Timestamp(last_bar_time) + bar_duration
    delta = (now_broker_ts - last_close).total_seconds() / 60.0
    return max(0.0, float(delta))


def htf_source_info(
    h1_topped_from_m5: int,
    d1_topped_from_m5: int,
    h1_last_broker_time: Any,
    d1_last_broker_time: Any,
    now_broker_ts: pd.Timestamp,
) -> dict:
    """
    Return per-cycle HTF source traceability.

    `h1_last_broker_time` / `d1_last_broker_time` must be the time of the
    *broker-published* last bar, BEFORE any synth was appended (i.e. the
    last row of the H1/D1 dataframe at the moment `_fetch_bars` returned,
    NOT after topup_htf_from_m5).
    """
    h1_age = _freshness_min(h1_last_broker_time, H1_DURATION, now_broker_ts)
    d1_age = _freshness_min(d1_last_broker_time, D1_DURATION, now_broker_ts)
    return {
        "h1_source":              "synth" if h1_topped_from_m5 else "broker",
        "h1_synth_used":          bool(h1_topped_from_m5),
        "h1_synth_count":         int(h1_topped_from_m5),
        "h1_broker_age_min":      round(h1_age, 1),
        "h1_last_broker_time":    str(h1_last_broker_time) if h1_last_broker_time is not None else None,
        "d1_source":              "synth" if d1_topped_from_m5 else "broker",
        "d1_synth_used":          bool(d1_topped_from_m5),
        "d1_synth_count":         int(d1_topped_from_m5),
        "d1_broker_age_min":      round(d1_age, 1),
        "d1_last_broker_time":    str(d1_last_broker_time) if d1_last_broker_time is not None else None,
    }


# ═══════════════════════════════════════════════════════════════════════════
# BAR INTEGRITY
# ═══════════════════════════════════════════════════════════════════════════

def bar_integrity_snapshot(m5_len: int, csv_source: str = "live") -> dict:
    """
    Bar-integrity flags. `lookahead_protection=True` reflects the
    `iloc[:-1]` in `_fetch_bars` that drops the still-forming bar.
    `is_final_closed_bar=True` is the consequence: the bar we run the
    strategy on closed before this cycle started.
    """
    return {
        "is_final_closed_bar":  True,
        "lookahead_protection": True,
        "bar_index":            int(m5_len - 1) if m5_len else None,
        "csv_source":           str(csv_source),
    }


# ═══════════════════════════════════════════════════════════════════════════
# DECISION TRACE (derived from existing strategy diag — no logic change)
# ═══════════════════════════════════════════════════════════════════════════

# Reasons we record when signal_dir is 0 OR a gate blocked something.
# Order matters: returned `blocked_reasons` keeps insertion order.
_GATE_ORDER = ["HTF", "SESSION", "ATR", "ADX", "H1_RSI", "MACD", "NO_REACTION"]


def decision_trace(diag: dict) -> dict:
    """
    Convert the existing strategy diag dict into a structured decision
    trace + blocked_reasons list.

    Pure derivation — no strategy logic change. Uses the same fields that
    `Strategy.detect_signal_verbose` already emits (trend_dir, in_session,
    atr_ok, adx_ok, f_*, signal_dir, h1_rsi).

    Outputs:
      blocked_reasons   list[str]   ordered: which gates would block
      blocked_reason    str         primary reason (first in list, or "NONE")
      trend_gate_passed bool
      session_gate_passed bool
      atr_gate_passed   bool
      adx_gate_passed   bool
      reactions_agree_long  bool   any reaction filter fired +1 on the bar
      reactions_agree_short bool
      final_signal_dir  int  (mirror of diag.signal_dir, for convenience)
    """
    out: dict = {}
    sig    = int(diag.get("signal_dir", 0)) if diag.get("signal_dir") is not None else 0
    tdir   = int(diag.get("trend_dir", 0))  if diag.get("trend_dir") is not None else 0
    in_ses = bool(diag.get("in_session", False))
    atr_ok = bool(diag.get("atr_ok", True))
    adx_ok = bool(diag.get("adx_ok", True))

    out["trend_gate_passed"]   = tdir != 0
    out["session_gate_passed"] = in_ses
    out["atr_gate_passed"]     = atr_ok
    out["adx_gate_passed"]     = adx_ok

    # Reaction agreement at the M5 layer (mode-agnostic indicator).
    f_keys = ("f_bb", "f_ema", "f_rsi", "f_candle", "f_rsiR", "f_macd", "f_stoch", "f_vol")
    long_votes  = sum(1 for k in f_keys if int(diag.get(k, 0) or 0) ==  1)
    short_votes = sum(1 for k in f_keys if int(diag.get(k, 0) or 0) == -1)
    out["reactions_agree_long"]  = long_votes  >= 1
    out["reactions_agree_short"] = short_votes >= 1
    out["reaction_long_votes"]   = int(long_votes)
    out["reaction_short_votes"]  = int(short_votes)
    out["final_signal_dir"]      = int(sig)

    # Blocked reasons (only meaningful when signal_dir == 0).
    reasons: list[str] = []
    if sig == 0:
        if not out["trend_gate_passed"]:
            reasons.append("HTF")
        elif not out["session_gate_passed"]:
            reasons.append("SESSION")
        elif not out["atr_gate_passed"]:
            reasons.append("ATR")
        elif not out["adx_gate_passed"]:
            reasons.append("ADX")
        else:
            # All gates passed but strategy chose 0 — must be reaction-side
            if tdir == 1 and not out["reactions_agree_long"]:
                reasons.append("NO_REACTION")
            elif tdir == -1 and not out["reactions_agree_short"]:
                reasons.append("NO_REACTION")
            else:
                reasons.append("MODE_CONFIRM")  # mode-specific aggregation rejected

    out["blocked_reasons"] = reasons
    out["blocked_reason"]  = reasons[0] if reasons else "NONE"
    return out


# ═══════════════════════════════════════════════════════════════════════════
# CASCADE ID
# ═══════════════════════════════════════════════════════════════════════════

def cascade_id(symbol: str, open_ticket: Optional[int],
               open_since_iso: Optional[str], now_ts: Optional[Any] = None) -> Optional[str]:
    """
    Group together all signals suppressed by the same open position.

    Format: ``"<symbol>-<open_iso_date>-<ticket>"``.

    Returns None when there is no open position (no cascade in progress).
    The id is stable for the lifetime of the open position; once the
    position closes, the next cascade gets a new ticket → new id.
    """
    if not open_ticket:
        return None
    # ISO date of when the cascade started (when the position opened).
    if open_since_iso:
        date_part = str(open_since_iso)[:10]
    else:
        if now_ts is None:
            now_ts = datetime.now(timezone.utc)
        date_part = pd.Timestamp(now_ts).strftime("%Y-%m-%d")
    return f"{symbol}-{date_part}-{open_ticket}"


# ═══════════════════════════════════════════════════════════════════════════
# POSITION STATE (used by execution_engine when blocking on ONE_TRADE)
# ═══════════════════════════════════════════════════════════════════════════

def position_state_snapshot(
    has_open_position: bool,
    open_ticket: Optional[int] = None,
    open_symbol: Optional[str] = None,
    open_since_iso: Optional[str] = None,
    open_pnl: Optional[float] = None,
    blocks_new_signal: bool = True,
    now_ts: Optional[Any] = None,
) -> dict:
    """Snapshot of the open position that's blocking new signals."""
    duration_min = None
    if open_since_iso:
        try:
            t0 = pd.Timestamp(open_since_iso)
            t1 = pd.Timestamp(now_ts) if now_ts is not None else pd.Timestamp.utcnow()
            duration_min = round((t1 - t0).total_seconds() / 60.0, 2)
        except Exception:
            duration_min = None
    return {
        "has_open_position":  bool(has_open_position),
        "open_ticket":        int(open_ticket) if open_ticket else None,
        "open_symbol":        open_symbol,
        "open_since":         open_since_iso,
        "open_duration_min":  duration_min,
        "open_pnl":           float(open_pnl) if open_pnl is not None else None,
        "blocks_new_signal":  bool(blocks_new_signal),
    }
