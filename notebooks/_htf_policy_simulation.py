"""HTF synchronization policy simulation.

Runs the live `Strategy.detect_signal_verbose` against the basket over a
recent window, varying ONLY the H1/D1 frames it sees per policy. Then
backtests each policy's signal stream with NB33-identical execution
semantics so all comparisons are apples-to-apples.

Output: CSVs + a markdown summary the report builder consumes.

Run:  python notebooks/_htf_policy_simulation.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Live strategy — single source of truth
from mt5.multi_symbol_bot.strategy import (  # noqa: E402
    Strategy, StrategyConfig,
    HISTORY_M5_BARS, HISTORY_H1_BARS, HISTORY_D1_BARS,
    DEFAULT_BROKER_TO_NY_H, RR, STRUCT_LOOKBACK_BARS, SL_BUFFER_ATR,
    MAX_R_OVER_ATR,
)

# ---------- config ----------------------------------------------------------
DATA_DIR    = ROOT / "notebooks" / "data"
RESULTS_DIR = ROOT / "notebooks" / "results" / "multi_symbol_scalper"
OUT_DIR     = ROOT / "notebooks" / "data" / "htf_policy"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASKET = ["GBPUSD", "XAUUSD", "GBPCAD", "USDMXN", "EURJPY", "EURCAD", "AUDCAD"]

# Window matches the parity-evidence window (so we can sanity-check against
# the actual live cycle logs + the existing parity_per_bar_diff.csv).
WINDOW_FROM = pd.Timestamp("2026-05-25 00:00:00")
WINDOW_TO   = pd.Timestamp("2026-05-28 00:00:00")
WARMUP_DAYS = 60   # generous; live runner uses ~600 M5 bars
# Cycle stride: live runs every M5 bar, but for the policy comparison the
# meaningful resolution is the HTF bar boundary. Sampling every 3rd M5
# (every 15min) cuts work 3x with no loss of policy-divergence resolution
# (broker vs synth disagreement persists across consecutive M5s within
# the same hour). For final production deployment we still use M5-per-cycle.
M5_STRIDE = 3

# Cost model from NB33 — needed for $ metrics
SYMBOLS_CFG = {
    'GBPUSD':  {'pip_size': 0.0001, 'pip_value_usd_per_002lot': 0.2000,  'spread_pips':   1.6},
    'EURJPY':  {'pip_size': 0.01,   'pip_value_usd_per_002lot': 0.1256,  'spread_pips':   2.9001},
    'EURCAD':  {'pip_size': 0.0001, 'pip_value_usd_per_002lot': 0.1447,  'spread_pips':   2.8001},
    'GBPCAD':  {'pip_size': 0.0001, 'pip_value_usd_per_002lot': 0.1447,  'spread_pips':   3.0},
    'AUDCAD':  {'pip_size': 0.0001, 'pip_value_usd_per_002lot': 0.1447,  'spread_pips':   2.0},
    'USDMXN':  {'pip_size': 0.0001, 'pip_value_usd_per_002lot': 0.01155, 'spread_pips': 103.7},
    'XAUUSD':  {'pip_size': 0.01,   'pip_value_usd_per_002lot': 0.0200,  'spread_pips':  17.0},
}


# ---------- data loaders ----------------------------------------------------
def load_ohlcv_naive(sym: str, tf: str) -> pd.DataFrame:
    p = DATA_DIR / sym / tf / "ohlcv.csv"
    df = pd.read_csv(p, parse_dates=["time"])
    if df["time"].dt.tz is not None:
        df["time"] = df["time"].dt.tz_localize(None)
    if "tick_volume" in df.columns:
        df = df.rename(columns={"tick_volume": "volume"})
    elif "volume" not in df.columns:
        df["volume"] = 0
    cols = ["time", "open", "high", "low", "close", "volume"]
    return df[cols].sort_values("time").reset_index(drop=True)


def load_cfg(sym: str) -> tuple[StrategyConfig, dict]:
    p = RESULTS_DIR / sym / "config.json"
    raw = json.loads(p.read_text(encoding="utf-8"))
    return StrategyConfig.from_dict(raw), raw


# ---------- HTF view builders (the policies) --------------------------------
TF_RESAMPLE_RULE = {"H1": "1h", "D1": "1D"}
TF_BAR_DURATION  = {"H1": pd.Timedelta(hours=1), "D1": pd.Timedelta(days=1)}


def topup_htf_from_m5(htf_df: pd.DataFrame, tf: str, m5_df: pd.DataFrame) -> pd.DataFrame:
    """Replica of mt5/run_multi_scalper.py:topup_htf_from_m5 — kept verbatim
    so simulation Policy B exactly matches live behaviour."""
    if tf not in TF_RESAMPLE_RULE or htf_df.empty or m5_df.empty:
        return htf_df

    rule         = TF_RESAMPLE_RULE[tf]
    bar_duration = TF_BAR_DURATION[tf]
    last_htf_time = pd.Timestamp(htf_df["time"].iloc[-1])
    last_m5_time  = pd.Timestamp(m5_df["time"].iloc[-1])

    if last_m5_time < last_htf_time + bar_duration:
        return htf_df

    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in m5_df.columns:
        agg["volume"] = "sum"

    resampled = (
        m5_df.set_index("time")
             .resample(rule, label="left", closed="left")
             .agg(agg)
             .dropna(subset=["open", "high", "low", "close"])
             .reset_index()
    )

    new_bars = resampled[resampled["time"] > last_htf_time].copy()
    if new_bars.empty:
        return htf_df

    for col in htf_df.columns:
        if col not in new_bars.columns:
            new_bars[col] = 0
    new_bars = new_bars[htf_df.columns]
    return pd.concat([htf_df, new_bars], ignore_index=True)


def broker_h1_view(broker_h1_full: pd.DataFrame, t: pd.Timestamp, n: int) -> pd.DataFrame:
    """All broker H1 bars that closed by time t. tail(n) for memory."""
    cut = broker_h1_full[broker_h1_full["time"] + TF_BAR_DURATION["H1"] <= t]
    return cut.tail(n).reset_index(drop=True)


def broker_d1_view(broker_d1_full: pd.DataFrame, t: pd.Timestamp, n: int) -> pd.DataFrame:
    cut = broker_d1_full[broker_d1_full["time"] + TF_BAR_DURATION["D1"] <= t]
    return cut.tail(n).reset_index(drop=True)


def build_htf_views(policy: str, broker_h1_full: pd.DataFrame, broker_d1_full: pd.DataFrame,
                    m5_tail: pd.DataFrame, t: pd.Timestamp,
                    policy_params: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (h1_view, d1_view, debug_info) per policy at time t.

    Policies:
      A           — broker-only, no synth. Stale HTF allowed.
      B           — synth-only (current live). Topup from M5 to current.
      C_<minutes> — hybrid: A if broker lag <= minutes else B.
      D_<n>       — broker-only but require last n broker H1 bars
                    to agree on trend direction (smoothing).
      D1_<n>      — same but applied to D1 not H1.
    """
    params = policy_params or {}
    h1 = broker_h1_view(broker_h1_full, t, HISTORY_H1_BARS)
    d1 = broker_d1_view(broker_d1_full, t, HISTORY_D1_BARS)

    # "Freshness" = minutes since the most recent broker bar fully closed.
    # That's the operationally meaningful lag: at t=:35 with the last
    # H1 bar [15:00,16:00) closed at 16:00, broker is 35 min fresh — not
    # 95 min behind (which is the duration-from-start measure that
    # collapses with the 1h bar duration).
    last_h1_close = (h1["time"].iloc[-1] + TF_BAR_DURATION["H1"]) if not h1.empty else None
    last_d1_close = (d1["time"].iloc[-1] + TF_BAR_DURATION["D1"]) if not d1.empty else None
    h1_freshness_min = max(0.0, (t - last_h1_close).total_seconds() / 60.0) if last_h1_close is not None else float("inf")
    d1_freshness_min = max(0.0, (t - last_d1_close).total_seconds() / 60.0) if last_d1_close is not None else float("inf")
    debug = {"h1_freshness_min": h1_freshness_min,
             "d1_freshness_min": d1_freshness_min,
             "policy": policy}

    if policy == "A":
        return h1, d1, debug

    if policy == "B":
        h1 = topup_htf_from_m5(h1, "H1", m5_tail)
        d1 = topup_htf_from_m5(d1, "D1", m5_tail)
        debug["h1_topped"] = int(max(0, len(h1) - HISTORY_H1_BARS) + 0)  # rough
        return h1, d1, debug

    if policy.startswith("C_"):
        # Format: "C_<h1_threshold_min>" — D1 NEVER synth (D1 evolves slowly;
        # always-synth gives noisy intraday D1, which is the dominant source
        # of `d1_trend` flips we want to avoid).
        threshold_min = float(policy.split("_")[1])
        if h1_freshness_min > threshold_min:
            h1 = topup_htf_from_m5(h1, "H1", m5_tail)
        # D1 broker-only — no synth regardless.
        return h1, d1, debug

    if policy.startswith("D_"):
        # Broker-only on H1 + delayed-confirmation smoothing handled
        # outside this fn (we annotate; caller applies smoothing on h1_trend)
        debug["delay_n"] = int(policy.split("_")[1])
        return h1, d1, debug

    raise ValueError(f"unknown policy {policy!r}")


# ---------- run strategy per policy -----------------------------------------
def policy_signal_at(strategy: Strategy, m5_tail: pd.DataFrame,
                     h1_view: pd.DataFrame, d1_view: pd.DataFrame,
                     policy: str, debug: dict) -> tuple[Optional[dict], dict]:
    """Wrapper around detect_signal_verbose. For Policy D, post-process the
    diag to require last-N H1 bars to agree before declaring trend.
    """
    sig, diag = strategy.detect_signal_verbose(m5_tail, h1_view, d1_view)
    diag.update(debug)

    if policy.startswith("D_") and sig is not None:
        # Smooth: require last N broker H1 closes to agree on trend_dir.
        # We recompute H1 trend on the broker h1_view, check the last N.
        n = int(policy.split("_")[1])
        if len(h1_view) >= n:
            from mt5.multi_symbol_bot.strategy import ema as _ema
            h1c = h1_view["close"]
            e   = _ema(h1c, 50)
            slope = e.diff()
            trend = np.where(
                (h1c > e) & (slope > 0), 1,
                np.where((h1c < e) & (slope < 0), -1, 0),
            )
            last_n = trend[-n:]
            if not np.all(last_n == last_n[0]) or last_n[0] == 0:
                # not confirmed → suppress
                sig = None
                diag["d_suppressed"] = True

    sig_dict = None
    if sig is not None:
        sig_dict = {
            "direction": sig.direction,
            "entry":     sig.entry,
            "sl":        sig.sl,
            "tp":        sig.tp,
            "bar_time":  sig.bar_time,
            **(sig.confidence or {}),
        }
    return sig_dict, diag


# ---------- walk the window for a single symbol/policy ----------------------
def walk_policy(sym: str, policy: str, m5_full: pd.DataFrame,
                h1_full: pd.DataFrame, d1_full: pd.DataFrame,
                cfg: StrategyConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (signals_df, diags_df) for this (sym, policy) over WINDOW_*."""
    strategy = Strategy(cfg=cfg, broker_to_ny_h=DEFAULT_BROKER_TO_NY_H)

    # Cycle iteration: stride to reduce work; policy decision varies only at
    # HTF boundaries, so 15-min stride is sufficient resolution.
    m5_in_window = m5_full[(m5_full["time"] >= WINDOW_FROM) & (m5_full["time"] < WINDOW_TO)]
    m5_in_window = m5_in_window.iloc[::M5_STRIDE]
    sigs, diags = [], []
    m5_full_t = m5_full["time"].values  # for searchsorted

    for t in m5_in_window["time"]:
        # M5 tail: last HISTORY_M5_BARS bars closing at/before t (excluding still-forming)
        # Since we walk on closed M5 bars only, t IS the close of the latest bar.
        m5_cut_end = np.searchsorted(m5_full_t, t.to_datetime64(), side="right")
        m5_tail = m5_full.iloc[max(0, m5_cut_end - HISTORY_M5_BARS): m5_cut_end]

        h1_view, d1_view, dbg = build_htf_views(
            policy, h1_full, d1_full, m5_tail, t,
        )
        # Guards for early window
        if len(m5_tail) < 200 or len(h1_view) < 60 or len(d1_view) < 30:
            continue

        sig, diag = policy_signal_at(strategy, m5_tail, h1_view, d1_view, policy, dbg)
        diag["bar_time"] = str(t)
        diag["symbol"]   = sym
        diag["policy"]   = policy
        diags.append(diag)

        if sig is not None:
            sigs.append({
                "symbol":    sym,
                "policy":    policy,
                "bar_time":  sig["bar_time"],
                "direction": sig["direction"],
                "entry":     sig["entry"],
                "sl":        sig["sl"],
                "tp":        sig["tp"],
            })

    return pd.DataFrame(sigs), pd.DataFrame(diags)


# ---------- backtest engine (NB33-compatible) --------------------------------
@dataclass
class Trade:
    side: int  # +1 / -1
    entry_idx: int
    entry_time: pd.Timestamp
    entry: float
    sl: float
    tp: float
    exit_idx: int = -1
    exit_time: Optional[pd.Timestamp] = None
    exit: float = 0.0
    reason: str = ""
    r_multiple: float = 0.0


def backtest_policy_signals(sym: str, signals_df: pd.DataFrame,
                            m5_full: pd.DataFrame, cfg_raw: dict,
                            one_trade_at_a_time: bool = True) -> pd.DataFrame:
    """NB33-identical engine: enter at open(i+1), SL/TP via OHLC walk,
    time-stop at max_hold_bars, ONE_TRADE_AT_A_TIME by default.

    Signals are matched to M5 bars by bar_time. We need M5 ATR for the
    SL distance & degenerate-setup guard, so we compute it on m5_full
    once and merge.
    """
    if signals_df.empty:
        return pd.DataFrame()

    from mt5.multi_symbol_bot.strategy import atr as _atr
    df = m5_full.copy()
    df["atr"] = _atr(df, 14)
    df = df.reset_index(drop=True)

    max_hold_bars = int(cfg_raw.get("max_hold_bars", 288))

    # Build a bar_time -> idx map
    bar_time_to_idx = {str(t): i for i, t in enumerate(df["time"])}

    # Convert signals to (entry_idx, side, entry_price, sl, tp) at i+1
    raw = []
    for _, s in signals_df.iterrows():
        i = bar_time_to_idx.get(str(s["bar_time"]))
        if i is None or i + 1 >= len(df):
            continue
        ei = i + 1
        side = 1 if s["direction"] == "BUY" else -1
        ep = float(df["open"].iat[ei])
        # Same structural SL as NB33 — recompute on m5_full to match
        lo = max(0, ei - STRUCT_LOOKBACK_BARS)
        a  = float(df["atr"].iat[ei])
        if side == 1:
            sl = float(df["low"].iloc[lo:ei].min()) - SL_BUFFER_ATR * a
        else:
            sl = float(df["high"].iloc[lo:ei].max()) + SL_BUFFER_ATR * a
        r = abs(ep - sl)
        if r <= 0 or r > MAX_R_OVER_ATR * a:
            continue
        tp = ep + RR * r * side
        raw.append({"entry_idx": ei, "side": side,
                    "entry_time": df["time"].iat[ei],
                    "entry": ep, "sl": sl, "tp": tp})

    raw_sorted = sorted(raw, key=lambda x: x["entry_idx"])

    # Walk: ONE_TRADE_AT_A_TIME means we skip a new entry while in_trade.
    trades = []
    in_trade = False
    cur: Optional[Trade] = None
    raw_iter = iter(raw_sorted)
    next_raw = next(raw_iter, None)

    for i in range(len(df) - 1):
        # First, exit logic if in trade
        if in_trade and cur is not None:
            hi, lo = df["high"].iat[i], df["low"].iat[i]
            hit_sl = (cur.side == 1 and lo <= cur.sl) or (cur.side == -1 and hi >= cur.sl)
            hit_tp = (cur.side == 1 and hi >= cur.tp) or (cur.side == -1 and lo <= cur.tp)
            exit_now, reason, px = False, "", 0.0
            if hit_sl:
                exit_now, reason, px = True, "sl", cur.sl
            elif hit_tp:
                exit_now, reason, px = True, "tp", cur.tp
            elif i - cur.entry_idx >= max_hold_bars:
                exit_now, reason, px = True, "time", float(df["close"].iat[i])
            if exit_now:
                cur.exit_idx, cur.exit_time, cur.exit, cur.reason = (
                    i, df["time"].iat[i], px, reason
                )
                r_unit = abs(cur.entry - cur.sl)
                cur.r_multiple = ((cur.exit - cur.entry) * cur.side) / r_unit if r_unit > 0 else 0
                trades.append(cur)
                in_trade, cur = False, None

        # Then, entry logic: take next raw signal whose entry_idx == i+1
        # (signals come from M5 bar i triggering at i+1's open)
        while next_raw and next_raw["entry_idx"] <= i:
            next_raw = next(raw_iter, None)
        if next_raw and next_raw["entry_idx"] == i + 1:
            if not one_trade_at_a_time or not in_trade:
                cur = Trade(
                    side=next_raw["side"], entry_idx=next_raw["entry_idx"],
                    entry_time=next_raw["entry_time"],
                    entry=next_raw["entry"], sl=next_raw["sl"], tp=next_raw["tp"],
                )
                in_trade = True
            next_raw = next(raw_iter, None)

    # Convert
    rows = []
    for t in trades:
        ps = SYMBOLS_CFG[sym]["pip_size"]
        pv = SYMBOLS_CFG[sym]["pip_value_usd_per_002lot"]
        sp = SYMBOLS_CFG[sym]["spread_pips"]
        gross = ((t.exit - t.entry) * t.side / ps) * pv
        fee   = sp * pv
        net   = gross - fee
        rows.append({
            "symbol":     sym,
            "entry_time": t.entry_time,
            "exit_time":  t.exit_time,
            "side":       "BUY" if t.side == 1 else "SELL",
            "entry":      t.entry, "sl": t.sl, "tp": t.tp, "exit": t.exit,
            "reason":     t.reason, "R": t.r_multiple,
            "hold_bars":  (t.exit_idx - t.entry_idx) if t.exit_idx >= 0 else None,
            "gross_$":    round(gross, 2),
            "fee_$":      round(fee, 2),
            "net_$":      round(net, 2),
        })
    return pd.DataFrame(rows)


# ---------- metrics ---------------------------------------------------------
def metrics_from_trades(tr: pd.DataFrame) -> dict:
    if tr.empty:
        return {"trades": 0, "WR": 0.0, "PF": 0.0, "expectancy_R": 0.0,
                "sum_R": 0.0, "net_$": 0.0, "max_dd_R": 0.0,
                "recovery_factor": 0.0, "avg_RR": 0.0, "sharpe_R": 0.0}

    R = tr["R"].astype(float)
    wins = R[R > 0]
    losses = R[R < 0]
    pf = float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else float("inf")

    cum = R.cumsum()
    peak = cum.cummax()
    dd = (cum - peak)
    max_dd_R = float(-dd.min()) if not dd.empty else 0.0

    # Sharpe-like: mean(R) / std(R) * sqrt(n_trades), per-trade Sharpe-ish
    if R.std() > 0:
        sharpe = float(R.mean() / R.std() * np.sqrt(max(1, len(R))))
    else:
        sharpe = 0.0

    return {
        "trades":          int(len(tr)),
        "WR":              round(float((R > 0).mean() * 100), 2),
        "PF":              round(pf, 3),
        "expectancy_R":    round(float(R.mean()), 4),
        "sum_R":           round(float(R.sum()), 3),
        "net_$":           round(float(tr["net_$"].sum()), 2),
        "max_dd_R":        round(max_dd_R, 3),
        "recovery_factor": round(float(R.sum() / max_dd_R) if max_dd_R > 0 else 0.0, 3),
        "avg_RR":          round(float(wins.mean() / -losses.mean()) if len(wins) and len(losses) else 0.0, 3),
        "sharpe_R":        round(sharpe, 3),
    }


# ---------- main driver -----------------------------------------------------
POLICIES = [
    "A",           # broker-only, no synth
    "B",           # current production: always synth-extend
    "C_15",        # hybrid: synth only if broker H1 stale > 15 min
    "C_60",        # hybrid: synth only if broker H1 stale > 60 min
    "C_120",       # hybrid: synth only if broker H1 stale > 120 min
    "D_2",         # broker-only + require last 2 broker H1 bars to agree
    "D_3",         # broker-only + require last 3 broker H1 bars to agree
]


def run() -> None:
    import sys as _sys
    _sys.stdout.reconfigure(line_buffering=True)
    print(f"Window: {WINDOW_FROM} -> {WINDOW_TO}", flush=True)
    print(f"Policies: {POLICIES}", flush=True)
    print(f"Symbols:  {BASKET}", flush=True)
    print(f"Stride: {M5_STRIDE} (~{5*M5_STRIDE}min sampling)\n", flush=True)

    all_signals  = []
    all_trades   = []
    all_metrics  = []
    all_diags    = []
    started = time.monotonic()

    import argparse as _argparse
    ap = _argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="Run on only GBPUSD for fast iteration")
    ap.add_argument("--symbols", nargs="+", default=None)
    args, _ = ap.parse_known_args()
    syms = ["GBPUSD"] if args.quick else (args.symbols or BASKET)

    for sym in syms:
        cfg, cfg_raw = load_cfg(sym)
        # Load M5 with warmup before window AND 2 days past window so trades
        # opened near the end can still close via SL/TP/time-stop.
        # H1/D1 are tiny — load full file.
        t_load_from = WINDOW_FROM - pd.Timedelta(days=WARMUP_DAYS)
        t_load_to   = WINDOW_TO + pd.Timedelta(days=2)
        m5 = load_ohlcv_naive(sym, "M5")
        h1 = load_ohlcv_naive(sym, "H1")
        d1 = load_ohlcv_naive(sym, "D1")
        m5 = m5[(m5["time"] >= t_load_from) & (m5["time"] < t_load_to)].reset_index(drop=True)
        h1 = h1[h1["time"] < t_load_to].reset_index(drop=True)
        d1 = d1[d1["time"] < t_load_to].reset_index(drop=True)

        for policy in POLICIES:
            t0 = time.monotonic()
            sigs, diags = walk_policy(sym, policy, m5, h1, d1, cfg)
            trades      = backtest_policy_signals(sym, sigs, m5, cfg_raw, one_trade_at_a_time=True)
            m           = metrics_from_trades(trades)
            m.update({"symbol": sym, "policy": policy,
                      "signal_count": len(sigs),
                      "elapsed_s": round(time.monotonic() - t0, 1)})
            all_metrics.append(m)
            if not sigs.empty:
                all_signals.append(sigs)
            if not trades.empty:
                trades["policy"] = policy
                all_trades.append(trades)
            all_diags.append(diags)
            print(f"  {sym:7s}  {policy:6s}  sigs={len(sigs):4d}  trades={m['trades']:3d}  "
                  f"WR={m['WR']:5.1f}%  R={m['sum_R']:+6.2f}  PF={m['PF']:5.2f}  "
                  f"net=${m['net_$']:+7.2f}  elapsed={m['elapsed_s']:.1f}s", flush=True)

    print(f"\nTotal elapsed: {time.monotonic() - started:.1f}s")

    # Persist
    pd.DataFrame(all_metrics).to_csv(OUT_DIR / "metrics.csv", index=False)
    if all_signals:
        pd.concat(all_signals, ignore_index=True).to_csv(OUT_DIR / "signals.csv", index=False)
    if all_trades:
        pd.concat(all_trades, ignore_index=True).to_csv(OUT_DIR / "trades.csv", index=False)
    if all_diags:
        pd.concat(all_diags, ignore_index=True).to_csv(OUT_DIR / "diags.csv", index=False)
    print(f"\nWrote outputs to: {OUT_DIR}")


if __name__ == "__main__":
    run()
