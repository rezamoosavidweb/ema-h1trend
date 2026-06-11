"""Event-driven, look-ahead-free pairs backtester with realistic frictions.

What "realistic" means here, concretely:
  * t+1 execution      — a position decided from the z-score at the close of bar t is held
                         over bar t+1 (positions are shifted forward one bar). No bar's own
                         signal trades inside that bar.
  * transaction costs  — taker fee per leg, applied to *turnover* (both legs trade).
  * slippage           — extra bps per leg on every execution.
  * latency            — modelled as the t+1 execution lag (we act on the next bar, not the
                         signal bar). An extra `exec_lag` knob allows >1 bar latency.
  * volatility sizing  — each pair is scaled to a target per-bar vol using a *trailing*
                         estimate of its own spread-return vol (point-in-time), with a
                         leverage cap. Inverse-vol sizing stops one wild pair dominating PnL.

A pair's per-bar P&L (long-spread convention = long A, short beta·B), per $1 gross:
    r_pair_t = (r_A,t - beta·r_B,t) / (1 + |beta|)
where r are simple returns. Costs are charged when the *signed, vol-scaled* position
changes. The portfolio equally allocates capital across the selected pairs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import BARS_PER_YEAR
from .spread import static_spread, kalman_hedge_ratio, rolling_zscore
from .signals import zscore_positions, trade_log
from .pairs import Pair


@dataclass
class CostModel:
    """All frictions in basis points (per leg, per execution) unless noted.

    Taker (default): fee_bps>0, slippage_bps>0, fill_prob=1 (you always fill, you pay).
    Maker (see execution.py): fee_bps can be negative (rebate), fill_prob<1 (you may not
    fill), adverse_bps>0 (the orders that DO fill are mildly adversely selected). fill_prob
    scales the *realised* exposure — an expected-value model of partial / missed fills.
    """
    fee_bps: float = 5.5          # Bybit taker ≈ 0.055%
    slippage_bps: float = 2.0     # conservative extra for a liquid USDT perp
    exec_lag: int = 1             # bars between signal and fill (>=1 => no look-ahead)
    fill_prob: float = 1.0        # expected fill ratio (maker<1); 1.0 = taker
    adverse_bps: float = 0.0      # adverse-selection cost on filled maker orders

    @property
    def cost_per_turn(self) -> float:
        return (self.fee_bps + self.slippage_bps + self.adverse_bps) / 1e4


@dataclass
class SignalParams:
    z_entry: float = 2.0
    z_exit: float = 0.5
    z_stop: float = 4.0
    z_window: int = 168           # rolling z-score window (1 week of H1)


@dataclass
class SizeParams:
    target_ann_vol: float = 0.10  # per-pair annualised vol target
    vol_window: int = 168
    max_leverage: float = 3.0
    timeframe: str = "H1"


@dataclass
class PairResult:
    pair: Pair
    returns: pd.Series                    # net per-bar return on the pair's capital
    gross_returns: pd.Series
    position: pd.Series                   # executed (lagged) signed exposure
    trades: pd.DataFrame
    cost: pd.Series

    @property
    def n_trades(self) -> int:
        return len(self.trades)


@dataclass
class PortfolioResult:
    returns: pd.Series                    # portfolio net per-bar return
    equity: pd.Series                     # cumulative (compounded) equity, starts at 1.0
    pair_returns: pd.DataFrame            # per-pair net returns (PnL attribution)
    pair_results: dict
    turnover: pd.Series
    config: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
def _simple_returns(px: pd.Series) -> pd.Series:
    return px.pct_change()


def backtest_pair(px_a: pd.Series, px_b: pd.Series, pair: Pair, *,
                  sig: SignalParams, size: SizeParams, cost: CostModel,
                  hedge: str = "static") -> PairResult:
    """Backtest one pair over the supplied (trading-window) price series.

    `hedge`:
      * "static" — use the formation-window beta stored on `pair` (constant).
      * "kalman" — re-estimate a time-varying beta with a Kalman filter (the dynamic-hedge
                   institutional upgrade). The Kalman spread is its own innovation series.
    """
    ra, rb = _simple_returns(px_a), _simple_returns(px_b)

    if hedge == "kalman":
        kf = kalman_hedge_ratio(px_a, px_b)
        beta_t = kf["beta"]
        spread = kf["spread"]
    elif hedge == "rolling":
        from .spread import rolling_hedge_ratio
        beta_t = rolling_hedge_ratio(px_a, px_b).fillna(pair.beta)
        spread = np.log(px_a) - beta_t * np.log(px_b)
    else:
        beta_t = pd.Series(pair.beta, index=px_a.index)
        spread = static_spread(px_a, px_b, pair.beta, pair.alpha)

    z = rolling_zscore(spread, window=sig.z_window)
    target = zscore_positions(z, z_entry=sig.z_entry, z_exit=sig.z_exit, z_stop=sig.z_stop)

    # Gross per-$1 pair return (long-spread convention), using the *prevailing* beta.
    denom = 1.0 + beta_t.abs()
    r_pair = (ra - beta_t * rb) / denom

    # Volatility sizing — trailing (point-in-time) vol of the pair return.
    ann = np.sqrt(BARS_PER_YEAR[size.timeframe])
    tgt_bar = size.target_ann_vol / ann
    vol_hat = r_pair.rolling(size.vol_window, min_periods=size.vol_window // 2).std()
    lev = (tgt_bar / vol_hat).clip(upper=size.max_leverage).fillna(0.0)

    # Executed position: decided at close t, filled `exec_lag` bars later (no look-ahead).
    # fill_prob<1 (maker) scales realised exposure: an expected-value partial/missed-fill model.
    signed_exposure = (target * lev).shift(cost.exec_lag).fillna(0.0) * cost.fill_prob

    gross = signed_exposure * r_pair
    # Turnover = change in signed exposure; both legs trade => multiply gross by (1+|beta|).
    dpos = signed_exposure.diff().abs().fillna(signed_exposure.abs())
    cost_series = dpos * denom * cost.cost_per_turn
    net = (gross - cost_series).fillna(0.0)

    trades = trade_log(np.sign(signed_exposure).replace(0, 0.0))
    return PairResult(pair=pair, returns=net, gross_returns=gross.fillna(0.0),
                      position=signed_exposure, trades=trades, cost=cost_series)


def backtest_portfolio(px_trade: pd.DataFrame, pairs: list[Pair], *,
                       sig: SignalParams | None = None, size: SizeParams | None = None,
                       cost: CostModel | None = None, hedge: str = "static",
                       label: str = "") -> PortfolioResult:
    """Backtest a book of pairs with equal capital per pair. Returns a PortfolioResult with
    the equity curve, per-pair attribution, and turnover."""
    sig = sig or SignalParams()
    size = size or SizeParams()
    cost = cost or CostModel()
    if not pairs:
        raise ValueError("no pairs to backtest")

    per_pair, cols = {}, {}
    turn = pd.Series(0.0, index=px_trade.index)
    for p in pairs:
        if p.a not in px_trade or p.b not in px_trade:
            continue
        res = backtest_pair(px_trade[p.a], px_trade[p.b], p, sig=sig, size=size,
                            cost=cost, hedge=hedge)
        per_pair[p.key] = res
        cols[f"{p.a}-{p.b}"] = res.returns
        turn = turn.add(res.position.diff().abs().fillna(0.0), fill_value=0.0)

    pair_rets = pd.DataFrame(cols)
    w = 1.0 / pair_rets.shape[1]                       # equal capital per pair
    port_ret = pair_rets.mul(w).sum(axis=1)
    equity = (1.0 + port_ret).cumprod()

    return PortfolioResult(returns=port_ret, equity=equity, pair_returns=pair_rets * w,
                           pair_results=per_pair, turnover=turn * w,
                           config={"label": label, "n_pairs": pair_rets.shape[1],
                                   "hedge": hedge, "sig": sig.__dict__,
                                   "size": size.__dict__, "cost": cost.__dict__})


# --------------------------------------------------------------------------- #
# Generic basket-spread backtest (a pair is the 2-asset case with weights [1, -beta]).
# Used by the Johansen multi-asset baskets in cointegration.py.
# --------------------------------------------------------------------------- #
@dataclass
class SpreadResult:
    returns: pd.Series
    gross_returns: pd.Series
    position: pd.Series
    turnover: pd.Series
    equity: pd.Series
    weights: pd.Series


def backtest_spread(prices: pd.DataFrame, weights: pd.Series, *,
                    sig: SignalParams | None = None, size: SizeParams | None = None,
                    cost: CostModel | None = None, label: str = "") -> SpreadResult:
    """Backtest a fixed-weight basket spread  s = Σ w_i·log(P_i).

    Same engine as the pair: rolling z-score of the spread, t+1 execution, vol-targeted
    sizing, costs on turnover scaled by the basket's gross leg count (Σ|w|). `weights` is
    the cointegrating vector (e.g. a Johansen eigenvector), estimated on the FORMATION
    window only and held fixed out-of-sample."""
    sig = sig or SignalParams()
    size = size or SizeParams()
    cost = cost or CostModel()
    cols = [c for c in weights.index if c in prices.columns]
    w = weights.loc[cols]
    P = prices[cols]

    logP = np.log(P)
    spread = logP.mul(w, axis=1).sum(axis=1)
    z = rolling_zscore(spread, window=sig.z_window)
    target = zscore_positions(z, z_entry=sig.z_entry, z_exit=sig.z_exit, z_stop=sig.z_stop)

    gross_legs = float(w.abs().sum())
    r = P.pct_change()
    r_basket = r.mul(w, axis=1).sum(axis=1) / gross_legs      # per-$1 gross basket return

    ann = np.sqrt(BARS_PER_YEAR[size.timeframe])
    tgt_bar = size.target_ann_vol / ann
    vol_hat = r_basket.rolling(size.vol_window, min_periods=size.vol_window // 2).std()
    lev = (tgt_bar / vol_hat).clip(upper=size.max_leverage).fillna(0.0)

    signed = (target * lev).shift(cost.exec_lag).fillna(0.0) * cost.fill_prob
    gross = signed * r_basket
    dpos = signed.diff().abs().fillna(signed.abs())
    cost_series = dpos * gross_legs * cost.cost_per_turn
    net = (gross - cost_series).fillna(0.0)
    return SpreadResult(returns=net, gross_returns=gross.fillna(0.0), position=signed,
                        turnover=dpos, equity=(1 + net).cumprod(), weights=w)
