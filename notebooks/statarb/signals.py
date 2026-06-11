"""Signal generation — a z-score state machine with entry, exit, and stop-loss.

The spread position is path-dependent (you stay in until exit/stop), so this is a genuine
state machine, not a vectorised threshold. States: +1 (long spread, z is low), -1 (short
spread, z is high), 0 (flat).

Rules (all on the z-score known at the close of bar t):
  * enter long  when z <= -z_entry   (spread cheap -> expect it to rise)
  * enter short when z >= +z_entry   (spread rich  -> expect it to fall)
  * exit        when |z| <= z_exit    (reverted to the mean -> take profit)
  * stop-loss   when |z| >= z_stop    (spread blew through -> the relationship may be broken)

The stop-loss is the single most important addition the article omits: without it, a
non-stationary / regime-broken spread can lose unboundedly while the z-score keeps telling
you to "add". The stop converts an unbounded tail into a bounded one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def zscore_positions(z: pd.Series, *, z_entry: float = 2.0, z_exit: float = 0.5,
                     z_stop: float = 4.0) -> pd.Series:
    """Target spread position in {-1, 0, +1} from the z-score. Returns a series aligned to
    `z`; NaN z (warm-up) -> flat. This is the *desired* position at the close of each bar;
    the backtester applies a t+1 execution lag so nothing is acted on with future info."""
    zv = z.values
    n = len(zv)
    pos = np.zeros(n)
    state = 0
    for t in range(n):
        zt = zv[t]
        if not np.isfinite(zt):
            pos[t] = 0.0
            state = 0
            continue
        if state == 0:
            if zt <= -z_entry:
                state = 1
            elif zt >= z_entry:
                state = -1
        elif state == 1:                       # long spread
            if zt >= z_stop or abs(zt) <= z_exit:
                state = 0                       # stop or take-profit
        elif state == -1:                      # short spread
            if zt <= -z_stop or abs(zt) <= z_exit:
                state = 0
        pos[t] = state
    return pd.Series(pos, index=z.index, name="pos")


def trade_log(pos: pd.Series) -> pd.DataFrame:
    """Compress a position series into discrete trades (entry ts, exit ts, side, bars)."""
    rows = []
    cur = 0
    entry_t = None
    for t, p in pos.items():
        if p != cur:
            if cur != 0:                        # closing a trade
                rows.append({"entry": entry_t, "exit": t, "side": cur})
            if p != 0:                          # opening a trade
                entry_t = t
            cur = p
    if cur != 0:
        rows.append({"entry": entry_t, "exit": pos.index[-1], "side": cur})
    df = pd.DataFrame(rows)
    if len(df):
        df["bars"] = [(pos.index.get_loc(r.exit) - pos.index.get_loc(r.entry)) for r in df.itertuples()]
    return df
