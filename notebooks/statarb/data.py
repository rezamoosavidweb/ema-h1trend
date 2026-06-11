"""Data layer — institutional-quality OHLCV loading, cleaning, alignment, liquidity.

The on-disk format is one CSV per `data/<SYMBOL>/<TF>/ohlcv.csv` with columns
`time,open,high,low,close,volume`. Timestamps are *mixed* timezone (some +03:30/+04:30
Tehran-with-DST, some +00:00) — we normalise everything to tz-aware UTC.

Why each cleaning step exists is documented inline; the philosophy is "fail loud on
structure, fail soft on values": structural problems (no file, no overlap) raise; value
problems (a single bad print, a short gap) are repaired and *counted* so the notebook can
report exactly how much surgery was done.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Universe definition.
#
# The framework can ingest any symbol with on-disk data, but the *default* universe is a
# curated set of liquid USDT-perpetuals with >=4y of history, chosen so the baseline-vs-
# clustering comparison runs on a clean common window. Exclusions and their reasons are
# explicit (this is also where survivorship bias enters — see SURVIVORSHIP_NOTE).
# --------------------------------------------------------------------------- #
DEFAULT_UNIVERSE = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT",
    "DOTUSDT", "LTCUSDT", "BCHUSDT", "LINKUSDT", "SOLUSDT", "AVAXUSDT",
    "XLMUSDT", "NEARUSDT", "ZECUSDT", "SHIB1000USDT",
]

# Newer listings: liquid but short history. The framework supports them; they are excluded
# from the *default* common-window comparison to avoid shrinking the intersection.
SHORT_HISTORY = ["1000PEPEUSDT", "SUIUSDT", "TONUSDT", "ONDOUSDT", "TAOUSDT", "MNTUSDT", "HYPEUSDT"]

# Hard exclusions with reasons (documented, not silent).
EXCLUSIONS = {
    "USDCUSDT": "stablecoin — ~zero vol, no mean-reversion signal, contaminates clustering",
    "USDEUSDT": "stablecoin — same as USDC",
    "XAUUSDT":  "gold (not crypto) + only ~1.6k bars on disk",
    "XAGUSDT":  "silver (not crypto) + trades only as closed-pnl history",
    "DOGEUSD":  "duplicate of DOGEUSDT (USD-quoted, shorter) — keep the USDT perp",
    "LTCUSD":   "duplicate of LTCUSDT (USD-quoted, shorter) — keep the USDT perp",
}

MARKET_PROXIES = ("BTCUSDT", "ETHUSDT")  # used for beta / market-neutralisation

SURVIVORSHIP_NOTE = """\
SURVIVORSHIP BIAS — explicit statement.
The on-disk universe is the set of symbols *currently* listed (and worth fetching) on
Bybit. Coins that were delisted, died, or never grew liquid enough to fetch are absent.
Any backtest on this universe therefore over-states achievable performance: pairs built
from survivors mean-revert partly *because* both legs survived. We cannot fully repair
this without a point-in-time listing database, so we (a) state it, (b) keep the universe
large and liquidity-filtered rather than cherry-picked, and (c) treat absolute returns as
optimistic and lean on *relative* (baseline vs clustering) conclusions, which share the
same bias and so cancel much of it."""

# Crypto trades 24/7 (8760 H1 bars/yr). Forex trades ~24h on ~260 weekdays/yr, so its bar
# count is ~71% of crypto's — using the crypto figure would over-state FX Sharpe by ~1.18x.
# FX_* keys let the same metrics/backtest code annualise correctly by passing timeframe="FX_H1".
BARS_PER_YEAR = {"M5": 105_120, "M30": 17_520, "H1": 8_760, "H4": 2_190, "D1": 365,
                 "FX_M5": 74_880, "FX_M30": 12_480, "FX_H1": 6_240, "FX_H4": 1_560, "FX_D1": 260}

# Fiat ISO-4217 codes seen on disk — used to classify true FX pairs vs crypto/metals.
FIAT_CCY = {"USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF", "CZK", "HUF",
            "PLN", "SEK", "NOK", "DKK", "HKD", "MXN", "TRY", "ZAR", "SGD", "CNH"}


def discover_symbols(data_dir: str | Path, timeframe: str = "H1") -> list[str]:
    """Dynamically discover every symbol directory that has an OHLCV file for `timeframe`.
    No hardcoded lists — the universe is whatever is on disk."""
    root = Path(data_dir)
    if not root.exists():
        return []
    out = [p.name for p in sorted(root.iterdir())
           if p.is_dir() and (p / timeframe / "ohlcv.csv").exists()]
    return out


def classify_fx(symbols: list[str]) -> dict:
    """Split discovered symbols into fx / crypto / metal / other by their ISO currency codes.
    A symbol is FX iff it is BASE+QUOTE with both legs fiat (e.g. EURUSD, GBPJPY)."""
    metals = {"XAU", "XAG", "XPT", "XPD"}
    out = {"fx": [], "metal": [], "crypto": [], "other": []}
    for s in symbols:
        if len(s) == 6 and s[:3] in FIAT_CCY and s[3:] in FIAT_CCY:
            out["fx"].append(s)
        elif len(s) == 6 and (s[:3] in metals or s[3:] in metals):
            out["metal"].append(s)
        elif len(s) == 6 and s[3:] in FIAT_CCY:        # e.g. BTCUSD, XRPUSD
            out["crypto"].append(s)
        else:
            out["other"].append(s)
    return out


# --------------------------------------------------------------------------- #
@dataclass
class CleaningReport:
    """Audit trail of every repair, so cleaning is transparent, not magic."""
    symbol: str
    rows_in: int = 0
    dup_index: int = 0
    nonpositive_px: int = 0
    return_outliers: int = 0
    gaps_filled: int = 0
    rows_out: int = 0
    first: pd.Timestamp | None = None
    last: pd.Timestamp | None = None

    def as_row(self) -> dict:
        return {k: getattr(self, k) for k in
                ("symbol", "rows_in", "dup_index", "nonpositive_px",
                 "return_outliers", "gaps_filled", "rows_out", "first", "last")}


def load_ohlcv(symbol: str, timeframe: str = "H1", data_dir: str | Path = "data") -> pd.DataFrame:
    """Load one symbol's OHLCV as a tz-aware (UTC) DataFrame indexed by time, sorted."""
    f = Path(data_dir) / symbol / timeframe / "ohlcv.csv"
    if not f.exists():
        raise FileNotFoundError(f"no OHLCV for {symbol} {timeframe}: {f}")
    df = pd.read_csv(f)
    # `utc=True` resolves the mixed-offset timestamps into a single UTC instant each.
    idx = pd.to_datetime(df["time"], utc=True)
    df = df.drop(columns=["time"]).set_index(idx).sort_index()
    df.index.name = "time"
    # Schema normalisation: FX/MetaTrader files use `tick_volume` (+ a `spread` column in
    # price points) and have no crypto-style `volume`. Map tick_volume -> volume so the rest
    # of the pipeline is market-agnostic; keep `spread` for the FX cost study.
    if "volume" not in df.columns:
        if "tick_volume" in df.columns:
            df["volume"] = df["tick_volume"]
        elif "real_volume" in df.columns:
            df["volume"] = df["real_volume"].replace(0, np.nan)
        else:
            df["volume"] = np.nan
    return df


def clean_ohlcv(df: pd.DataFrame, symbol: str, *, ret_mad_k: float = 12.0,
                winsorize: bool = True) -> tuple[pd.DataFrame, CleaningReport]:
    """Repair value-level problems and return (clean_df, report).

    Steps, in order, each counted:
      1. drop duplicate timestamps (keep last — latest fetch wins).
      2. drop non-positive prices (a log-return on <=0 is undefined).
      3. flag/winsorize extreme close-to-close return outliers using a robust MAD rule.
         We use a *generous* k=12 MADs: real crypto moves are fat-tailed, so we only clip
         data-error spikes (e.g. a single bad print), not genuine volatility.
    """
    rep = CleaningReport(symbol=symbol, rows_in=len(df))

    dup = df.index.duplicated(keep="last")
    rep.dup_index = int(dup.sum())
    df = df[~dup]

    bad = (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    rep.nonpositive_px = int(bad.sum())
    df = df[~bad]

    r = np.log(df["close"]).diff()
    med = r.median()
    mad = (r - med).abs().median()
    if mad and np.isfinite(mad):
        thr = ret_mad_k * 1.4826 * mad           # 1.4826 -> MAD≈sigma for normal
        out = (r - med).abs() > thr
        rep.return_outliers = int(out.fillna(False).sum())
        if winsorize and rep.return_outliers:
            # Rebuild close from winsorized returns so the level stays self-consistent.
            r_clip = r.clip(med - thr, med + thr)
            base = np.log(df["close"].iloc[0])
            df = df.copy()
            df["close"] = np.exp(base + r_clip.fillna(0).cumsum())

    rep.rows_out = len(df)
    rep.first, rep.last = df.index.min(), df.index.max()
    return df, rep


def build_panel(symbols: list[str], timeframe: str = "H1", *, field: str = "close",
                data_dir: str | Path = "data", how: str = "inner",
                max_gap_bars: int = 6, min_obs: int = 2000,
                clean: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, list[CleaningReport]]:
    """Build an aligned panel (one column per symbol) for `field`, plus a volume panel.

    Alignment: reindex every series onto a common time grid.
      * how="inner"  -> intersection of all timestamps (clean common window; default).
      * how="outer"  -> union, then forward-fill short gaps up to `max_gap_bars`.
    Short gaps are forward-filled (a missing H1 bar = market didn't print; last price
    persists). Long gaps are *not* filled — a 2-day hole is real missing data, and
    fabricating it would inject fake mean-reversion. Symbols with < `min_obs` usable
    rows on the common grid are dropped (and reported).
    """
    closes, vols, reports = {}, {}, []
    for s in symbols:
        try:
            raw = load_ohlcv(s, timeframe, data_dir)
        except FileNotFoundError:
            continue
        if clean:
            raw, rep = clean_ohlcv(raw, s)
            reports.append(rep)
        closes[s] = raw[field]
        vols[s] = raw["close"] * raw["volume"]      # quote-currency (USDT) dollar volume

    if not closes:
        raise ValueError("no symbols loaded")

    px = pd.DataFrame(closes).sort_index()
    dv = pd.DataFrame(vols).reindex(px.index)

    if how == "inner":
        px = px.dropna(how="any")
    elif how == "outer":
        px = px.ffill(limit=max_gap_bars)
    else:
        raise ValueError("how must be 'inner' or 'outer'")
    dv = dv.reindex(px.index).ffill(limit=max_gap_bars)

    # Drop thin symbols.
    keep = [c for c in px.columns if px[c].notna().sum() >= min_obs]
    px, dv = px[keep], dv[keep]
    if how == "inner":
        px = px.dropna(how="any")
        dv = dv.reindex(px.index)
    return px, dv, reports


def liquidity_table(dollar_volume: pd.DataFrame, *, window: int | None = None) -> pd.DataFrame:
    """Per-symbol liquidity stats from dollar volume. Liquidity filtering is *critical*:
    correlation/cointegration on an illiquid name is a mirage — you cannot trade the
    spread without moving it, and thin-volume coves create spurious co-movement (both
    legs jump on the same exchange-wide liquidity event, not on a tradable relationship).
    """
    dv = dollar_volume if window is None else dollar_volume.tail(window)
    out = pd.DataFrame({
        "median_dollar_vol": dv.median(),
        "mean_dollar_vol": dv.mean(),
        "p05_dollar_vol": dv.quantile(0.05),
    })
    out["adv_usd_m"] = out["median_dollar_vol"] / 1e6      # per-bar ADV in $m
    out["liq_rank"] = out["median_dollar_vol"].rank(ascending=False).astype(int)
    return out.sort_values("median_dollar_vol", ascending=False)


def liquidity_filter(px: pd.DataFrame, dollar_volume: pd.DataFrame, *,
                     min_median_dollar_vol: float = 50_000.0,
                     window: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only symbols whose median per-bar dollar volume clears a floor."""
    lt = liquidity_table(dollar_volume, window=window)
    keep = lt.index[lt["median_dollar_vol"] >= min_median_dollar_vol].tolist()
    keep = [c for c in px.columns if c in keep]
    return px[keep], dollar_volume[keep]


def fx_point_size(symbol: str) -> float:
    """MetaTrader point size: JPY-quoted pairs quote to 3 decimals (point=1e-3), the rest to
    5 (point=1e-5). Used to turn the `spread` column (in points) into an absolute price."""
    return 1e-3 if symbol.endswith("JPY") else 1e-5


def quoted_spread_bps(symbol: str, timeframe: str = "H1", data_dir: str | Path = "data") -> float:
    """Median *quoted* half-spread in basis points from the broker's `spread` column — the
    single biggest reason FX may succeed where crypto failed: real spreads are tiny. Returns
    NaN if the file has no spread column. (Full round-trip cost ≈ this value per leg.)"""
    try:
        df = load_ohlcv(symbol, timeframe, data_dir)
    except FileNotFoundError:
        return np.nan
    if "spread" not in df.columns:
        return np.nan
    sp = df["spread"]
    sp = sp[sp > 0]                       # 0 = unpopulated quote, not a real zero spread
    if len(sp) < 100:
        return np.nan
    abs_spread = sp * fx_point_size(symbol)
    rel_bps = (abs_spread / df["close"].loc[sp.index]).median() * 1e4
    return float(rel_bps)


def log_returns(px: pd.DataFrame) -> pd.DataFrame:
    """Close-to-close log returns. Log space: additive over time, symmetric, and the
    natural unit for spread = logA - beta*logB."""
    return np.log(px).diff()
