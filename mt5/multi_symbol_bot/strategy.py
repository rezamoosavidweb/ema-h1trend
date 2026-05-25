"""
Pure signal-detection logic for the multi-symbol scalper.

This file is a faithful, side-effect-free port of the `build_signals` pipeline
from `notebooks/24_multi_symbol_scalper.ipynb`. Backtest parity matters: any
deviation here makes the live behaviour diverge from the validated edge.

The only public entry point is `Strategy.detect_signal(m5, h1, d1)`, which
returns a `Signal` (BUY/SELL with entry/SL/TP) on the *last closed bar* or
None.

Architecture
────────────
    indicators           — EMA / RSI / ATR / BB / ADX / MACD-hist / Stochastic
    reaction filters     — f_bb, f_ema, f_rsi, f_candle, f_rsi_recent,
                           f_macd, f_stoch, f_volume_spike
    gate filters         — H1+D1 trend agreement, NY session window,
                           ATR-min, ADX-min, H1-RSI alignment, MACD alignment
    aggregator           — config-driven combination of votes/gates
    SL computation       — structural (lookback swing) + ATR buffer
    sanity guard         — drops signals with R > 5×ATR (degenerate setups)

Time conventions
────────────────
    * All input frames carry UTC timestamps (broker-tz conversion happens at
      the data-fetch boundary in the runner).
    * Session windows are expressed in **NY local hours** (broker hour minus
      `broker_to_ny_h`). The runner supplies this via StrategyConfig.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Optional

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS  (locked to notebook 24 — do NOT change without re-running sweep)
# ═══════════════════════════════════════════════════════════════════════════

EMA_FAST              = 20
EMA_TREND_H1          = 50
EMA_TREND_D1          = 50
BB_PERIOD             = 20
BB_STD                = 2.0
RSI_PERIOD            = 14
RSI_OS                = 35.0
RSI_OB                = 65.0
ATR_PERIOD            = 14
PULLBACK_TOLERANCE_ATR = 0.4
PIN_BAR_WICK_RATIO    = 0.60

# Risk management
RR                    = 2.0
STRUCT_LOOKBACK_BARS  = 12
SL_BUFFER_ATR         = 0.10
MAX_R_OVER_ATR        = 5.0  # skip degenerate setups where R > 5×ATR

# How much history per timeframe is needed for stable indicator output.
# Tuned generous — fetching extra is cheap, recomputing missing indicator
# warm-up bars is expensive (gives subtle off-by-one signal drift).
HISTORY_M5_BARS       = 600
HISTORY_H1_BARS       = 250
HISTORY_D1_BARS       = 120

# Default broker → NY hour offset (Errante = EET/EEST = UTC+2/+3).
DEFAULT_BROKER_TO_NY_H = 7


# ═══════════════════════════════════════════════════════════════════════════
# INDICATORS  (vectorised pandas; pure functions for testability)
# ═══════════════════════════════════════════════════════════════════════════


def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    diff = close.diff()
    up = diff.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-diff.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder ADX — trend-strength index in [0, 100]."""
    up   =  df["high"].diff()
    down = -df["low"].diff()
    plus_dm  = pd.Series(np.where((up > down) & (up > 0), up, 0.0),   index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr_w = tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    plus_di  = 100 * plus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()  / atr_w.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atr_w.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean().fillna(0)


def macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    line = ema(close, fast) - ema(close, slow)
    return line - ema(line, signal)


def stoch(df: pd.DataFrame, k: int = 14, d: int = 3) -> tuple[pd.Series, pd.Series]:
    ll = df["low"].rolling(k).min()
    hh = df["high"].rolling(k).max()
    k_pct = (100 * (df["close"] - ll) / (hh - ll).replace(0, np.nan)).fillna(50)
    return k_pct, k_pct.rolling(d).mean().fillna(50)


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE PIPELINE
# ═══════════════════════════════════════════════════════════════════════════


def _add_m5_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema20"]  = ema(df["close"], EMA_FAST)
    df["rsi"]    = rsi(df["close"], RSI_PERIOD)
    df["atr"]    = atr(df, ATR_PERIOD)
    mid = df["close"].rolling(BB_PERIOD).mean()
    std = df["close"].rolling(BB_PERIOD).std()
    df["bb_mid"] = mid
    df["bb_up"]  = mid + BB_STD * std
    df["bb_lo"]  = mid - BB_STD * std
    df["range"]      = (df["high"] - df["low"]).clip(lower=1e-9)
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["adx"]        = adx(df, 14)
    df["macd_hist"]  = macd_hist(df["close"])
    df["stoch_k"], df["stoch_d"] = stoch(df, k=14, d=3)
    return df


def _add_htf_trend(df: pd.DataFrame, ema_period: int) -> pd.DataFrame:
    """Same as notebook: trend_dir = sign(close vs EMA) AND sign(slope)."""
    df = df.copy()
    e = ema(df["close"], ema_period)
    slope = e.diff()
    df["ema_trend"]      = e
    df["trend_dir"]      = np.where(
        (df["close"] > e) & (slope > 0), 1,
        np.where((df["close"] < e) & (slope < 0), -1, 0),
    )
    df["htf_rsi"] = rsi(df["close"], 14).fillna(50)
    return df[["time", "ema_trend", "trend_dir", "htf_rsi"]]


# ═══════════════════════════════════════════════════════════════════════════
# REACTION FILTERS  (each returns -1 / 0 / +1 per bar)
# ═══════════════════════════════════════════════════════════════════════════


def _f_bb_touch(df: pd.DataFrame) -> pd.Series:
    long  = df["low"]  <= df["bb_lo"]
    short = df["high"] >= df["bb_up"]
    return pd.Series(np.where(long, 1, np.where(short, -1, 0)), index=df.index)


def _f_ema_pullback(df: pd.DataFrame) -> pd.Series:
    tol = PULLBACK_TOLERANCE_ATR * df["atr"]
    long  = (df["low"]  <= df["ema20"] + tol) & (df["close"] > df["ema20"])
    short = (df["high"] >= df["ema20"] - tol) & (df["close"] < df["ema20"])
    return pd.Series(np.where(long, 1, np.where(short, -1, 0)), index=df.index)


def _f_rsi_exit(df: pd.DataFrame) -> pd.Series:
    prev = df["rsi"].shift(1)
    long  = (prev <= RSI_OS) & (df["rsi"] > RSI_OS)
    short = (prev >= RSI_OB) & (df["rsi"] < RSI_OB)
    return pd.Series(np.where(long, 1, np.where(short, -1, 0)), index=df.index)


def _f_pin_engulf(df: pd.DataFrame) -> pd.Series:
    rng = df["range"]
    bull_pin = (df["lower_wick"] / rng >= PIN_BAR_WICK_RATIO) & (df["close"] > df["open"])
    bear_pin = (df["upper_wick"] / rng >= PIN_BAR_WICK_RATIO) & (df["close"] < df["open"])
    prev_o, prev_c = df["open"].shift(1), df["close"].shift(1)
    bull_eng = (prev_c < prev_o) & (df["close"] > df["open"]) & (df["close"] >= prev_o) & (df["open"] <= prev_c)
    bear_eng = (prev_c > prev_o) & (df["close"] < df["open"]) & (df["close"] <= prev_o) & (df["open"] >= prev_c)
    return pd.Series(
        np.where(bull_pin | bull_eng, 1,
                 np.where(bear_pin | bear_eng, -1, 0)),
        index=df.index,
    )


def _f_rsi_recent(f_rsi: pd.Series, memory: int = 10) -> pd.Series:
    """OR-merge of recent f_rsi exits — the 'fresh momentum snapback' zone."""
    long_fresh  = (f_rsi == 1).rolling(memory).max().fillna(0).astype(bool)
    short_fresh = (f_rsi == -1).rolling(memory).max().fillna(0).astype(bool)
    return pd.Series(np.where(long_fresh, 1, np.where(short_fresh, -1, 0)), index=f_rsi.index)


def _f_macd(df: pd.DataFrame) -> pd.Series:
    h = df["macd_hist"]
    return pd.Series(np.where(h > 0, 1, np.where(h < 0, -1, 0)), index=df.index)


def _f_stoch_cross(df: pd.DataFrame) -> pd.Series:
    k_prev = df["stoch_k"].shift(1)
    d_prev = df["stoch_d"].shift(1)
    long  = (k_prev < d_prev) & (df["stoch_k"] > df["stoch_d"]) & (df["stoch_k"] < 35)
    short = (k_prev > d_prev) & (df["stoch_k"] < df["stoch_d"]) & (df["stoch_k"] > 65)
    return pd.Series(np.where(long, 1, np.where(short, -1, 0)), index=df.index)


def _f_volume_spike(df: pd.DataFrame, mult: float = 1.4) -> pd.Series:
    if "volume" not in df.columns or float(df["volume"].sum()) == 0.0:
        return pd.Series(0, index=df.index)
    median = df["volume"].rolling(50, min_periods=10).median()
    spike  = df["volume"] >= mult * median
    long_s  = spike & (df["close"] > df["open"])
    short_s = spike & (df["close"] < df["open"])
    return pd.Series(np.where(long_s, 1, np.where(short_s, -1, 0)), index=df.index)


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION + SIGNAL TYPES
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class StrategyConfig:
    """
    Per-symbol strategy configuration -- one of these comes from
    `notebooks/results/multi_symbol_scalper/<SYMBOL>/config.json`.

    The fields below match the keys the notebook persists. Unknown extras
    (e.g. `htf_strength_min` which proved redundant in iter-3) are accepted
    via **kwargs in `from_dict` so future config-format changes don't break
    loading.
    """

    mode:                  str                 # "RSI-gated" | "RSI-gated-AND" | "OR"
    confirms:              tuple[str, ...]     # e.g. ("f_candle", "f_ema")
    rsi_memory:            int                 # 5 / 10 / 15
    session:               tuple[int, int]     # NY-local (start_h, end_h)
    sl_method:             str = "structural"
    atr_min_mult:          float = 0.0         # 0.0 disables
    min_reactions:         int = 1             # only used by "OR" mode
    adx_min:               float = 0.0
    require_h1_rsi_align:  bool = False
    require_macd_align:    bool = False
    vol_spike_mult:        float = 1.4

    @classmethod
    def from_dict(cls, data: dict) -> "StrategyConfig":
        """Tolerant loader: accepts JSON-from-disk including legacy keys."""
        return cls(
            mode                  = data["mode"],
            confirms              = tuple(data["confirms"]),
            rsi_memory            = int(data["rsi_memory"]),
            session               = tuple(data["session"]),
            sl_method             = data.get("sl_method", "structural"),
            atr_min_mult          = float(data.get("atr_min_mult", 0.0)),
            min_reactions         = int(data.get("min_reactions", 1)),
            adx_min               = float(data.get("adx_min", 0.0)),
            require_h1_rsi_align  = bool(data.get("require_h1_rsi_align", False)),
            require_macd_align    = bool(data.get("require_macd_align", False)),
            vol_spike_mult        = float(data.get("vol_spike_mult", 1.4)),
        )


@dataclass
class Signal:
    """
    A live trading signal compatible with `execution.ExecutionEngine.place_signal()`.

    The runner converts this to the dict shape the engine expects.
    """

    direction:    str           # "BUY" | "SELL"
    entry:        float
    sl:           float
    tp:           float
    bar_time:     str           # UTC ISO; dedup key
    confidence:   dict = field(default_factory=dict)  # for diagnostics only

    def as_engine_dict(self) -> dict:
        """Shape expected by ExecutionEngine.place_signal()."""
        return {
            "direction":   self.direction,
            "entry":       self.entry,
            "sl":          self.sl,
            "tp":          self.tp,
            # The execution engine uses ob_time as the dedup key. Reuse our
            # bar_time so duplicate-signal protection / seen-signals tracking
            # still works without engine changes.
            "ob_time":     self.bar_time,
            **self.confidence,
        }


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY
# ═══════════════════════════════════════════════════════════════════════════


class Strategy:
    """
    Stateless detector: feed M5/H1/D1 frames, get a Signal or None for the
    last closed bar. One instance per symbol; configuration is frozen at
    construction time.
    """

    def __init__(self, cfg: StrategyConfig, broker_to_ny_h: int = DEFAULT_BROKER_TO_NY_H) -> None:
        self.cfg = cfg
        self.broker_to_ny_h = broker_to_ny_h

    # ── public API ───────────────────────────────────────────────────────────-

    def detect_signal(
        self,
        m5: pd.DataFrame,
        h1: pd.DataFrame,
        d1: pd.DataFrame,
    ) -> Optional[Signal]:
        """
        Returns a Signal on the latest closed M5 bar if all gates fire, else None.

        The input frames MUST have a `time` column (tz-naive broker wall-clock
        — same convention as the live runner) and OHLC[+volume]. They MUST
        exclude the still-forming bar (do `iloc[:-1]` at the data boundary).

        For debugging parity issues, prefer `detect_signal_verbose()` which
        returns the same Signal plus a dict of every gate value that led to
        the decision (the runner logs that dict on each cycle).
        """
        sig, _ = self.detect_signal_verbose(m5, h1, d1)
        return sig

    def detect_signal_verbose(
        self,
        m5: pd.DataFrame,
        h1: pd.DataFrame,
        d1: pd.DataFrame,
    ) -> tuple[Optional[Signal], dict]:
        """
        Same as `detect_signal` but ALSO returns a diagnostics dict so callers
        can log exactly why a signal fired (or didn't). Designed to be cheap
        to serialise as JSON — all values are int / float / bool / str.

        Diagnostics keys:
            n_m5 / n_h1 / n_d1  — input frame sizes
            skip                — set when we bail out early
            bar_time, open, high, low, close, volume  — last-bar OHLC
            trend_dir, h1_trend, d1_trend, h1_rsi
            in_session, atr_ok, adx_ok
            ema20, rsi, atr, adx, bb_up, bb_lo
            f_bb, f_ema, f_rsi, f_candle, f_rsiR, f_macd, f_stoch, f_vol
            signal_dir          — -1 / 0 / +1 raw output of `_signal_at`
        """
        diag: dict = {
            "n_m5": int(len(m5)),
            "n_h1": int(len(h1)),
            "n_d1": int(len(d1)),
        }
        if len(m5) < HISTORY_M5_BARS * 0.5 or len(h1) < 100 or len(d1) < 60:
            diag["skip"] = "insufficient_data"
            return None, diag

        frame = self._build_frame(m5, h1, d1)
        i = len(frame) - 1
        row = frame.iloc[i]

        diag.update({
            "bar_time":   str(row["time"]),
            "open":       float(row["open"]),
            "high":       float(row["high"]),
            "low":        float(row["low"]),
            "close":      float(row["close"]),
            "volume":     float(row.get("volume", 0.0)),
            "trend_dir":  int(row["trend_dir"]),
            "h1_trend":   int(row["h1_trend"]),
            "d1_trend":   int(row["d1_trend"]),
            "h1_rsi":     round(float(row["h1_rsi"]), 3),
            "in_session": bool(row["in_session"]),
            "atr_ok":     bool(row["atr_ok"]),
            "adx_ok":     bool(row["adx_ok"]),
            "ema20":      round(float(row["ema20"]), 6),
            "rsi":        round(float(row["rsi"]), 3),
            "atr":        round(float(row["atr"]), 6),
            "adx":        round(float(row["adx"]), 3),
            "bb_up":      round(float(row["bb_up"]), 6),
            "bb_lo":      round(float(row["bb_lo"]), 6),
            "f_bb":       int(row["f_bb"]),
            "f_ema":      int(row["f_ema"]),
            "f_rsi":      int(row["f_rsi"]),
            "f_candle":   int(row["f_candle"]),
            "f_rsiR":     int(row["f_rsiR"]),
            "f_macd":     int(row["f_macd"]),
            "f_stoch":    int(row["f_stoch"]),
            "f_vol":      int(row["f_vol"]),
        })

        sig_direction = self._signal_at(frame, i)
        diag["signal_dir"] = int(sig_direction)
        if sig_direction == 0:
            return None, diag

        return self._build_signal(frame, i, sig_direction), diag

    # ── internals: feature pipeline ──────────────────────────────────────────-

    def _build_frame(self, m5: pd.DataFrame, h1: pd.DataFrame, d1: pd.DataFrame) -> pd.DataFrame:
        """Compute M5 features + merge HTF trend + apply all filters."""
        m5f  = _add_m5_features(m5)
        h1t  = _add_htf_trend(h1, EMA_TREND_H1)
        d1t  = _add_htf_trend(d1, EMA_TREND_D1)

        # merge_asof requires both sides sorted by `time`.
        m5f = pd.merge_asof(
            m5f.sort_values("time"),
            h1t.rename(columns={"ema_trend": "h1_ema",
                                 "trend_dir": "h1_trend",
                                 "htf_rsi":   "h1_rsi"}),
            on="time", direction="backward",
        )
        m5f = pd.merge_asof(
            m5f,
            d1t.rename(columns={"ema_trend": "d1_ema",
                                 "trend_dir": "d1_trend",
                                 "htf_rsi":   "d1_rsi"}),
            on="time", direction="backward",
        )

        # Trend gate: trade only when H1 and D1 agree and are non-zero.
        same = m5f["h1_trend"] == m5f["d1_trend"]
        nz   = m5f["h1_trend"] != 0
        m5f["trend_dir"] = np.where(same & nz, m5f["h1_trend"], 0).astype(int)

        # Session gate (NY local hours).
        sh, eh = self.cfg.session
        ny_h = (m5f["time"].dt.hour - self.broker_to_ny_h) % 24
        m5f["in_session"] = (ny_h >= sh) & (ny_h < eh)

        # Reaction filters.
        m5f["f_bb"]     = _f_bb_touch(m5f)
        m5f["f_ema"]    = _f_ema_pullback(m5f)
        m5f["f_rsi"]    = _f_rsi_exit(m5f)
        m5f["f_candle"] = _f_pin_engulf(m5f)
        m5f["f_rsiR"]   = _f_rsi_recent(m5f["f_rsi"], memory=self.cfg.rsi_memory)
        m5f["f_macd"]   = _f_macd(m5f)
        m5f["f_stoch"]  = _f_stoch_cross(m5f)
        m5f["f_vol"]    = _f_volume_spike(m5f, mult=self.cfg.vol_spike_mult)

        # ATR-min regime filter.
        if self.cfg.atr_min_mult > 0:
            atr_med = m5f["atr"].rolling(500, min_periods=50).median()
            m5f["atr_ok"] = (m5f["atr"] >= self.cfg.atr_min_mult * atr_med).fillna(False)
        else:
            m5f["atr_ok"] = True

        # ADX-min trend-quality filter.
        m5f["adx_ok"] = m5f["adx"] >= self.cfg.adx_min if self.cfg.adx_min > 0 else True

        # H1 RSI alignment gate.
        if self.cfg.require_h1_rsi_align:
            m5f["h1_rsi_long_ok"]  = m5f["h1_rsi"] > 50
            m5f["h1_rsi_short_ok"] = m5f["h1_rsi"] < 50
        else:
            m5f["h1_rsi_long_ok"]  = True
            m5f["h1_rsi_short_ok"] = True

        # MACD alignment gate.
        if self.cfg.require_macd_align:
            m5f["macd_long_ok"]  = m5f["f_macd"] ==  1
            m5f["macd_short_ok"] = m5f["f_macd"] == -1
        else:
            m5f["macd_long_ok"]  = True
            m5f["macd_short_ok"] = True

        return m5f.reset_index(drop=True)

    # ── internals: signal aggregation ────────────────────────────────────────-

    def _signal_at(self, frame: pd.DataFrame, i: int) -> int:
        """
        Returns +1 (long), -1 (short), or 0 at index `i`.

        Implements the three notebook modes verbatim. The reaction-vote logic
        differs between modes:
            * RSI-gated      = RSI must fire AND any confirm fires
            * RSI-gated-AND  = RSI must fire AND all confirms fire
            * OR             = >= min_reactions confirms fire (no RSI gate)
        """
        row = frame.iloc[i]
        if row["trend_dir"] == 0:
            return 0
        if not bool(row["in_session"]):
            return 0
        if not bool(row["atr_ok"]) or not bool(row["adx_ok"]):
            return 0

        confirms = self.cfg.confirms

        if self.cfg.mode == "RSI-gated":
            rsi_long  = int(row["f_rsiR"]) == 1
            rsi_short = int(row["f_rsiR"]) == -1
            conf_long  = any(int(row[c]) ==  1 for c in confirms)
            conf_short = any(int(row[c]) == -1 for c in confirms)
        elif self.cfg.mode == "RSI-gated-AND":
            rsi_long  = int(row["f_rsiR"]) == 1
            rsi_short = int(row["f_rsiR"]) == -1
            conf_long  = all(int(row[c]) ==  1 for c in confirms)
            conf_short = all(int(row[c]) == -1 for c in confirms)
        elif self.cfg.mode == "OR":
            rsi_long = rsi_short = True
            mr = self.cfg.min_reactions
            long_votes  = sum(1 for c in confirms if int(row[c]) ==  1)
            short_votes = sum(1 for c in confirms if int(row[c]) == -1)
            conf_long  = long_votes  >= mr
            conf_short = short_votes >= mr
        else:
            raise ValueError(f"unknown mode: {self.cfg.mode!r}")

        can_long  = (int(row["trend_dir"]) ==  1) and rsi_long  and conf_long  and bool(row["h1_rsi_long_ok"])  and bool(row["macd_long_ok"])
        can_short = (int(row["trend_dir"]) == -1) and rsi_short and conf_short and bool(row["h1_rsi_short_ok"]) and bool(row["macd_short_ok"])

        if can_long:  return  1
        if can_short: return -1
        return 0

    # ── internals: SL / TP / entry construction ──────────────────────────────-

    def _build_signal(self, frame: pd.DataFrame, i: int, direction: int) -> Optional[Signal]:
        """
        Build the live Signal for bar `i`. Entry = close of signal bar (the
        bot will market-execute right after bar close, which lines up with the
        backtest's 'open of bar i+1' assumption within seconds).
        """
        bar    = frame.iloc[i]
        entry  = float(bar["close"])
        atr_i  = float(bar["atr"])

        # Structural SL using the 12 bars ENDING at the signal bar (inclusive).
        # The backtest computes SL at bar i+1 using rows [i+1-12 : i+1], which
        # equals our [i-11 : i+1] -> the same 12 bars we see now.
        lo = max(0, i + 1 - STRUCT_LOOKBACK_BARS)
        if direction == 1:
            swing  = float(frame["low"].iloc[lo:i + 1].min())
            sl     = swing - SL_BUFFER_ATR * atr_i
        else:
            swing  = float(frame["high"].iloc[lo:i + 1].max())
            sl     = swing + SL_BUFFER_ATR * atr_i

        risk = abs(entry - sl)
        if risk <= 0 or risk > MAX_R_OVER_ATR * atr_i:
            # Degenerate setup (SL too far / too close) — skip.
            return None

        tp = entry + (RR * risk * (1 if direction == 1 else -1))

        return Signal(
            direction = "BUY" if direction == 1 else "SELL",
            entry     = round(entry, 5),
            sl        = round(sl, 5),
            tp        = round(tp, 5),
            bar_time  = str(bar["time"]),
            confidence={
                "atr":          round(atr_i, 5),
                "risk":         round(risk, 5),
                "rr":           RR,
                "trend_dir":    int(bar["trend_dir"]),
                "adx":          round(float(bar["adx"]), 1),
                "h1_rsi":       round(float(bar["h1_rsi"]), 1),
                "rsi":          round(float(bar["rsi"]), 1),
            },
        )
