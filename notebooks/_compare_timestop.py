"""NB33-engine, run over a long window with two different MAX_HOLD_BARS settings,
and report per-symbol comparison so we can decide 8h vs 24h time-stop per pair."""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "notebooks" / "data"
RESULTS_DIR  = PROJECT_ROOT / "notebooks" / "results" / "multi_symbol_scalper"
OUT_DIR      = PROJECT_ROOT / "notebooks" / "data"

BACKTEST_FROM = "2022-01-01"
BACKTEST_TO   = "2026-05-28"
BASKET        = ["GBPUSD", "XAUUSD", "GBPCAD", "USDMXN", "EURJPY", "EURCAD", "AUDCAD"]
WARMUP_DAYS   = 60

# ── Constants — identical to NB24/NB33 ────────────────────────────────────────
BROKER_TO_NY_H = 7
EMA_FAST = 20
EMA_TREND_H1 = 50
EMA_TREND_D1 = 50
BB_PERIOD = 20
BB_STD = 2.0
RSI_PERIOD = 14
RSI_OS = 35.0
RSI_OB = 65.0
ATR_PERIOD = 14
PULLBACK_TOLERANCE_ATR = 0.4
PIN_BAR_WICK_RATIO = 0.60
RR = 2.0
STRUCT_LOOKBACK_BARS = 12
SL_BUFFER_ATR = 0.10
ONE_TRADE_AT_A_TIME = True

SYMBOLS_CFG: Dict[str, dict] = {
    "GBPUSD":  {"pip_size": 0.0001, "pip_value_usd_per_002lot": 0.2000,  "spread_pips":   1.6},
    "EURJPY":  {"pip_size": 0.01,   "pip_value_usd_per_002lot": 0.1256,  "spread_pips":   2.9001},
    "EURCAD":  {"pip_size": 0.0001, "pip_value_usd_per_002lot": 0.1447,  "spread_pips":   2.8001},
    "GBPCAD":  {"pip_size": 0.0001, "pip_value_usd_per_002lot": 0.1447,  "spread_pips":   3.0},
    "AUDCAD":  {"pip_size": 0.0001, "pip_value_usd_per_002lot": 0.1447,  "spread_pips":   2.0},
    "USDMXN":  {"pip_size": 0.0001, "pip_value_usd_per_002lot": 0.01155, "spread_pips": 103.7},
    "XAUUSD":  {"pip_size": 0.01,   "pip_value_usd_per_002lot": 0.0200,  "spread_pips":  17.0},
}

# ── Indicators (copy from NB33) ───────────────────────────────────────────────
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi(close, n=14):
    d = close.diff()
    gain = d.clip(lower=0); loss = (-d).clip(lower=0)
    ag = gain.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    al = loss.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def atr(df, n=14):
    tr = pd.concat([df["high"]-df["low"],
                    (df["high"]-df["close"].shift()).abs(),
                    (df["low"] -df["close"].shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()

def adx(df, n=14):
    up = df["high"].diff(); down = -df["low"].diff()
    plus_dm  = pd.Series(np.where((up>down)&(up>0),  up,   0.0), index=df.index)
    minus_dm = pd.Series(np.where((down>up)&(down>0),down, 0.0), index=df.index)
    tr = pd.concat([df["high"]-df["low"],
                    (df["high"]-df["close"].shift()).abs(),
                    (df["low"] -df["close"].shift()).abs()], axis=1).max(axis=1)
    atr_w = tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    plus_di  = 100 * plus_dm.ewm(alpha=1/n, adjust=False, min_periods=n).mean()  / atr_w.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1/n, adjust=False, min_periods=n).mean() / atr_w.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False, min_periods=n).mean().fillna(0)

def macd_hist(c, fast=12, slow=26, signal=9):
    line = ema(c, fast) - ema(c, slow); sig = ema(line, signal); return line - sig

def stoch(df, k=14, d=3):
    ll = df["low"].rolling(k).min(); hh = df["high"].rolling(k).max()
    k_pct = (100 * (df["close"]-ll) / (hh-ll).replace(0, np.nan)).fillna(50)
    return k_pct, k_pct.rolling(d).mean().fillna(50)

def add_m5_features(df):
    df = df.copy()
    df["ema20"] = ema(df["close"], EMA_FAST)
    df["rsi"]   = rsi(df["close"], RSI_PERIOD)
    df["atr"]   = atr(df, ATR_PERIOD)
    mid = df["close"].rolling(BB_PERIOD).mean(); std = df["close"].rolling(BB_PERIOD).std()
    df["bb_mid"], df["bb_up"], df["bb_lo"] = mid, mid+BB_STD*std, mid-BB_STD*std
    df["body"]  = (df["close"]-df["open"]).abs()
    df["range"] = (df["high"]-df["low"]).clip(lower=1e-9)
    df["upper_wick"] = df["high"]-df[["open","close"]].max(axis=1)
    df["lower_wick"] = df[["open","close"]].min(axis=1)-df["low"]
    df["adx"] = adx(df, 14)
    df["macd_hist"] = macd_hist(df["close"])
    df["stoch_k"], df["stoch_d"] = stoch(df, k=14, d=3)
    return df

def add_htf_trend(df, n):
    df = df.copy()
    e = ema(df["close"], n); slope = e.diff()
    df["ema_trend"] = e
    df["trend_dir"] = np.where((df["close"]>e)&(slope>0), 1,
                       np.where((df["close"]<e)&(slope<0), -1, 0))
    h1_atr = atr(df, 14)
    df["trend_strength"] = (e.diff(6).abs() / h1_atr.replace(0, np.nan)).fillna(0)
    df["htf_rsi"] = rsi(df["close"], 14).fillna(50)
    return df[["time","ema_trend","trend_dir","trend_strength","htf_rsi"]]

def f_bb_touch(df):
    long  = df["low"]  <= df["bb_lo"]; short = df["high"] >= df["bb_up"]
    return pd.Series(np.where(long,1,np.where(short,-1,0)), index=df.index)

def f_ema_pullback(df):
    tol = PULLBACK_TOLERANCE_ATR * df["atr"]
    tl = (df["low"]  <= df["ema20"]+tol) & (df["close"]>df["ema20"])
    ts = (df["high"] >= df["ema20"]-tol) & (df["close"]<df["ema20"])
    return pd.Series(np.where(tl,1,np.where(ts,-1,0)), index=df.index)

def f_rsi_exit(df):
    prev = df["rsi"].shift(1)
    long  = (prev<=RSI_OS) & (df["rsi"]>RSI_OS)
    short = (prev>=RSI_OB) & (df["rsi"]<RSI_OB)
    return pd.Series(np.where(long,1,np.where(short,-1,0)), index=df.index)

def f_pin_engulf(df):
    rng = df["range"]
    bull_pin = (df["lower_wick"]/rng >= PIN_BAR_WICK_RATIO) & (df["close"]>df["open"])
    bear_pin = (df["upper_wick"]/rng >= PIN_BAR_WICK_RATIO) & (df["close"]<df["open"])
    po,pc = df["open"].shift(1), df["close"].shift(1)
    bull_eng = (pc<po)&(df["close"]>df["open"])&(df["close"]>=po)&(df["open"]<=pc)
    bear_eng = (pc>po)&(df["close"]<df["open"])&(df["close"]<=po)&(df["open"]>=pc)
    long  = bull_pin | bull_eng; short = bear_pin | bear_eng
    return pd.Series(np.where(long,1,np.where(short,-1,0)), index=df.index)

def f_rsi_recent(df, memory=10):
    long_fresh  = (df["f_rsi"]== 1).rolling(memory).max().fillna(0).astype(bool)
    short_fresh = (df["f_rsi"]==-1).rolling(memory).max().fillna(0).astype(bool)
    return pd.Series(np.where(long_fresh,1,np.where(short_fresh,-1,0)), index=df.index)

def f_macd(df):
    h = df["macd_hist"]; return pd.Series(np.where(h>0,1,np.where(h<0,-1,0)), index=df.index)

def f_stoch_cross(df):
    kp = df["stoch_k"].shift(1); dp = df["stoch_d"].shift(1)
    long  = (kp<dp)&(df["stoch_k"]>df["stoch_d"])&(df["stoch_k"]<35)
    short = (kp>dp)&(df["stoch_k"]<df["stoch_d"])&(df["stoch_k"]>65)
    return pd.Series(np.where(long,1,np.where(short,-1,0)), index=df.index)

def f_volume_spike(df, mult=1.4):
    if "volume" not in df.columns or df["volume"].sum()==0:
        return pd.Series(0, index=df.index)
    vm = df["volume"].rolling(50, min_periods=10).median()
    sp = df["volume"] >= mult*vm
    return pd.Series(np.where(sp & (df["close"]>df["open"]), 1,
                     np.where(sp & (df["close"]<df["open"]),-1, 0)), index=df.index)

# ── Engine ────────────────────────────────────────────────────────────────────
@dataclass
class Trade:
    side: int; entry_idx: int; entry_time: object
    entry: float; sl: float; tp: float
    exit_idx: int = -1; exit_time: object = None
    exit: float = 0.0; reason: str = ""; r_multiple: float = 0.0

def structural_sl(df, idx, side):
    lo = max(0, idx-STRUCT_LOOKBACK_BARS); a = df["atr"].iat[idx]
    if side == 1:
        return float(df["low"].iloc[lo:idx].min()) - SL_BUFFER_ATR*a
    return float(df["high"].iloc[lo:idx].max()) + SL_BUFFER_ATR*a

def backtest(df, max_hold_bars: int):
    trades: List[Trade] = []
    n = len(df); in_trade=False; cur: Trade | None = None
    sig_arr  = df["signal"].values
    open_arr = df["open"].values; high_arr=df["high"].values
    low_arr  = df["low"].values;  close_arr=df["close"].values
    time_arr = df["time"].values; atr_arr = df["atr"].values
    for i in range(n-1):
        if in_trade:
            hi, lo = high_arr[i], low_arr[i]
            hit_sl = (cur.side==1 and lo<=cur.sl) or (cur.side==-1 and hi>=cur.sl)
            hit_tp = (cur.side==1 and hi>=cur.tp) or (cur.side==-1 and lo<=cur.tp)
            exit_now=False; reason=""; px=0.0
            if hit_sl and hit_tp: exit_now,reason,px = True,"sl",cur.sl
            elif hit_sl:          exit_now,reason,px = True,"sl",cur.sl
            elif hit_tp:          exit_now,reason,px = True,"tp",cur.tp
            elif i-cur.entry_idx >= max_hold_bars:
                exit_now,reason,px = True,"time",float(close_arr[i])
            if exit_now:
                cur.exit_idx=i; cur.exit_time=time_arr[i]
                cur.exit=px; cur.reason=reason
                r_unit = abs(cur.entry-cur.sl)
                cur.r_multiple = ((cur.exit-cur.entry)*cur.side)/r_unit if r_unit>0 else 0
                trades.append(cur); in_trade=False; cur=None
        sig = int(sig_arr[i])
        if (not in_trade or not ONE_TRADE_AT_A_TIME) and sig != 0:
            ei = i+1; ep = float(open_arr[ei])
            sl = structural_sl(df, ei, sig)
            r = abs(ep-sl)
            if r<=0 or r>5*atr_arr[ei]: continue
            tp = ep + RR*r*sig
            cur = Trade(side=sig, entry_idx=ei, entry_time=time_arr[ei],
                        entry=ep, sl=sl, tp=tp)
            in_trade=True
    return trades

def build_signals(m5, h1, d1, cfg):
    m5 = add_m5_features(m5)
    h1t = add_htf_trend(h1, EMA_TREND_H1)
    d1t = add_htf_trend(d1, EMA_TREND_D1)
    m5 = pd.merge_asof(m5.sort_values("time"),
                       h1t.rename(columns={"ema_trend":"h1_ema","trend_dir":"h1_trend",
                                            "trend_strength":"h1_strength","htf_rsi":"h1_rsi"}),
                       on="time", direction="backward")
    m5 = pd.merge_asof(m5,
                       d1t.rename(columns={"ema_trend":"d1_ema","trend_dir":"d1_trend",
                                            "trend_strength":"d1_strength","htf_rsi":"d1_rsi"}),
                       on="time", direction="backward")
    same = m5["h1_trend"]==m5["d1_trend"]; nz = m5["h1_trend"]!=0
    m5["trend_dir"] = np.where(same & nz, m5["h1_trend"], 0).astype(int)
    m5["htf_strong"] = m5["h1_strength"] >= cfg.get("htf_strength_min", 0.0)
    sh, eh = cfg["session"]
    ny_h = (m5["time"].dt.hour - BROKER_TO_NY_H) % 24
    m5["in_session"] = (ny_h>=sh) & (ny_h<eh)
    m5["f_bb"]     = f_bb_touch(m5)
    m5["f_ema"]    = f_ema_pullback(m5)
    m5["f_rsi"]    = f_rsi_exit(m5)
    m5["f_candle"] = f_pin_engulf(m5)
    m5["f_rsiR"]   = f_rsi_recent(m5, memory=cfg.get("rsi_memory", 10))
    m5["f_macd"]   = f_macd(m5)
    m5["f_stoch"]  = f_stoch_cross(m5)
    m5["f_vol"]    = f_volume_spike(m5, mult=cfg.get("vol_spike_mult", 1.4))
    atr_mult = cfg.get("atr_min_mult", 0.0)
    if atr_mult > 0:
        atr_med = m5["atr"].rolling(500, min_periods=50).median()
        m5["atr_ok"] = (m5["atr"] >= atr_mult * atr_med).fillna(False)
    else:
        m5["atr_ok"] = True
    adx_min = cfg.get("adx_min", 0.0)
    m5["adx_ok"] = m5["adx"] >= adx_min if adx_min > 0 else True
    if cfg.get("require_h1_rsi_align", False):
        m5["h1_rsi_long_ok"]  = m5["h1_rsi"] > 50
        m5["h1_rsi_short_ok"] = m5["h1_rsi"] < 50
    else:
        m5["h1_rsi_long_ok"] = True; m5["h1_rsi_short_ok"] = True
    if cfg.get("require_macd_align", False):
        m5["macd_long_ok"]  = m5["f_macd"]== 1
        m5["macd_short_ok"] = m5["f_macd"]==-1
    else:
        m5["macd_long_ok"] = True; m5["macd_short_ok"] = True
    base_long  = m5["in_session"] & m5["atr_ok"] & m5["htf_strong"] & m5["adx_ok"] & m5["h1_rsi_long_ok"]  & m5["macd_long_ok"]
    base_short = m5["in_session"] & m5["atr_ok"] & m5["htf_strong"] & m5["adx_ok"] & m5["h1_rsi_short_ok"] & m5["macd_short_ok"]
    if cfg["mode"] == "RSI-gated":
        rl=(m5["f_rsiR"]==1); rs=(m5["f_rsiR"]==-1)
        conf = m5[cfg["confirms"]].values
        cl=(conf==1).any(axis=1); cs=(conf==-1).any(axis=1)
        cl_long  = (m5["trend_dir"]==1)  & base_long  & rl & cl
        cl_short = (m5["trend_dir"]==-1) & base_short & rs & cs
    elif cfg["mode"] == "RSI-gated-AND":
        rl=(m5["f_rsiR"]==1); rs=(m5["f_rsiR"]==-1)
        conf = m5[cfg["confirms"]].values
        cl=(conf==1).all(axis=1); cs=(conf==-1).all(axis=1)
        cl_long  = (m5["trend_dir"]==1)  & base_long  & rl & cl
        cl_short = (m5["trend_dir"]==-1) & base_short & rs & cs
    elif cfg["mode"] == "OR":
        mr = cfg.get("min_reactions", 1)
        f  = m5[cfg["confirms"]].values
        lv = (f==1).sum(axis=1); sv = (f==-1).sum(axis=1)
        cl_long  = (m5["trend_dir"]==1)  & base_long  & (lv>=mr)
        cl_short = (m5["trend_dir"]==-1) & base_short & (sv>=mr)
    else:
        raise ValueError(cfg["mode"])
    m5["signal"] = np.where(cl_long,1,np.where(cl_short,-1,0))
    return m5

def load_ohlcv(symbol, tf, t_from, t_to):
    path = DATA_DIR / symbol / tf / "ohlcv.csv"
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    df = df.sort_values("time").reset_index(drop=True)
    keep = ["time","open","high","low","close","tick_volume"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df.rename(columns={"tick_volume":"volume"}, inplace=True)
    if "volume" not in df.columns: df["volume"] = 0
    df = df[(df["time"]>=t_from) & (df["time"]<t_to)].reset_index(drop=True)
    return df

def load_cfg(sym):
    p = RESULTS_DIR / sym / "config.json"
    cfg = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(cfg.get("session"), list):
        cfg["session"] = tuple(cfg["session"])
    return cfg

def trade_pnl_dollars(t: Trade, sym: str) -> float:
    if not t.exit: return 0.0
    s = SYMBOLS_CFG[sym]
    pip_move = (t.exit - t.entry) * t.side / s["pip_size"]
    return pip_move * s["pip_value_usd_per_002lot"] - s["spread_pips"] * s["pip_value_usd_per_002lot"]

def stats_for_trades(trades: List[Trade], sym: str) -> dict:
    if not trades:
        return dict(trades=0, wins=0, losses=0, wr=0.0,
                    sum_R=0.0, avg_R=0.0, expectancy_R=0.0,
                    profit_factor=0.0, max_dd_R=0.0,
                    sum_net_d=0.0, avg_net_d=0.0, max_dd_d=0.0,
                    avg_hold_bars=0.0, p95_hold_bars=0,
                    tp=0, sl=0, time_exits=0, time_exit_pct=0.0,
                    time_exit_avg_R=0.0, swap_nights=0)
    Rs = np.array([t.r_multiple for t in trades])
    dollars = np.array([trade_pnl_dollars(t, sym) for t in trades])
    holds = np.array([t.exit_idx - t.entry_idx for t in trades if t.exit_idx>=0])
    wins = int((Rs>0).sum()); losses = int((Rs<=0).sum())
    gross_win  = float(Rs[Rs>0].sum())
    gross_loss = float(-Rs[Rs<0].sum())
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf") if gross_win>0 else 0.0
    eq_R = np.cumsum(Rs); dd_R = float(np.max(np.maximum.accumulate(eq_R) - eq_R)) if len(eq_R) else 0.0
    eq_d = np.cumsum(dollars); dd_d = float(np.max(np.maximum.accumulate(eq_d) - eq_d)) if len(eq_d) else 0.0
    n_tp   = sum(1 for t in trades if t.reason=="tp")
    n_sl   = sum(1 for t in trades if t.reason=="sl")
    n_time = sum(1 for t in trades if t.reason=="time")
    time_R = float(np.mean([t.r_multiple for t in trades if t.reason=="time"])) if n_time else 0.0
    # rough swap exposure: how many trades held over a 22:00–23:00 broker boundary
    swap_nights = 0
    for t in trades:
        if t.exit_time is None: continue
        et = pd.Timestamp(t.entry_time); xt = pd.Timestamp(t.exit_time)
        # count number of 22:00 broker-time crossings in [et, xt]
        # broker tz is irrelevant for *counting* crossings of any fixed wall-clock hour
        cur = et.floor("D") + pd.Timedelta(hours=22)
        if cur < et: cur += pd.Timedelta(days=1)
        while cur <= xt:
            swap_nights += 1; cur += pd.Timedelta(days=1)
    return dict(
        trades=len(trades), wins=wins, losses=losses,
        wr=round(wins/len(trades)*100,1),
        sum_R=round(float(Rs.sum()),2), avg_R=round(float(Rs.mean()),3),
        expectancy_R=round(float(Rs.mean()),3),
        profit_factor=round(pf,2) if pf!=float("inf") else 99.0,
        max_dd_R=round(dd_R,2),
        sum_net_d=round(float(dollars.sum()),2),
        avg_net_d=round(float(dollars.mean()),2),
        max_dd_d=round(dd_d,2),
        avg_hold_bars=round(float(holds.mean()),1) if len(holds) else 0,
        p95_hold_bars=int(np.percentile(holds,95)) if len(holds) else 0,
        tp=n_tp, sl=n_sl, time_exits=n_time,
        time_exit_pct=round(n_time/len(trades)*100,1),
        time_exit_avg_R=round(time_R,2),
        swap_nights=swap_nights,
    )

# ── Run ───────────────────────────────────────────────────────────────────────
def main():
    t_from_user = pd.Timestamp(BACKTEST_FROM)
    t_to_user   = pd.Timestamp(BACKTEST_TO)
    t_from_load = t_from_user - pd.Timedelta(days=WARMUP_DAYS)
    t_to_load   = t_to_user
    print(f"Window: {BACKTEST_FROM} .. {BACKTEST_TO}  (warmup {WARMUP_DAYS} d)")
    print()
    print(f"{'symbol':<8s} {'cfg':<5s} {'trades':>6s} {'wr':>5s} {'sumR':>7s} {'avgR':>6s} "
          f"{'PF':>5s} {'ddR':>6s} {'net$':>9s} {'dd$':>8s} {'avgH':>5s} {'p95H':>5s} "
          f"{'tp':>3s} {'sl':>3s} {'tim':>4s} {'tim%':>5s} {'timR':>5s} {'swp':>4s}")
    summary_rows = []
    for sym in BASKET:
        try:
            cfg = load_cfg(sym)
            m5 = load_ohlcv(sym, "M5", t_from_load, t_to_load)
            h1 = load_ohlcv(sym, "H1", t_from_load, t_to_load)
            d1 = load_ohlcv(sym, "D1", t_from_load, t_to_load)
        except Exception as e:
            print(f"  {sym}  SKIP — {e}")
            continue
        if len(m5) < 200:
            print(f"  {sym}  SKIP — too little data"); continue
        m5 = build_signals(m5, h1, d1, cfg)
        # Restrict trades to the user-requested window
        for hold, label in [(96,"8h"), (288,"24h")]:
            trades_all = backtest(m5, max_hold_bars=hold)
            trades = [t for t in trades_all if (t.entry_time >= t_from_user) and (t.entry_time < t_to_user)]
            st = stats_for_trades(trades, sym)
            row = {"symbol": sym, "cfg": label, **st}
            summary_rows.append(row)
            print(f"  {sym:<6s} {label:<5s} "
                  f"{st['trades']:>6d} {st['wr']:>4.1f}% {st['sum_R']:>+7.2f} {st['avg_R']:>+6.3f} "
                  f"{st['profit_factor']:>5.2f} {st['max_dd_R']:>6.2f} "
                  f"{st['sum_net_d']:>+9.2f} {st['max_dd_d']:>8.2f} "
                  f"{st['avg_hold_bars']:>5.1f} {st['p95_hold_bars']:>5d} "
                  f"{st['tp']:>3d} {st['sl']:>3d} {st['time_exits']:>4d} {st['time_exit_pct']:>4.1f}% "
                  f"{st['time_exit_avg_R']:>+5.2f} {st['swap_nights']:>4d}")
    df = pd.DataFrame(summary_rows)
    out = OUT_DIR / "compare_timestop_2022_20260528.csv"
    df.to_csv(out, index=False)
    print(); print(f"saved -> {out}")

if __name__ == "__main__":
    main()
