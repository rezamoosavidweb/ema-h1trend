"""
MetaTrader 5 broker adapter (Windows / VPS, real demo account).

Timezone discipline matches the other bots in this repo: MT5's ``time`` field is the
broker's wall-clock encoded as Unix seconds (NOT real UTC), so we relabel to
``Europe/Nicosia`` — see project memory 'Data timezone is Nicosia not UTC' and
``mt5/pairs_trading/data_fetcher.py`` for the canonical implementation this mirrors.

The ``MetaTrader5`` module is imported lazily so this file is importable on Linux
(it just won't ``connect()``).
"""

from __future__ import annotations

import time
from zoneinfo import ZoneInfo

import pandas as pd

from .base import AccountInfo, BrokerAdapter, OrderResult, SymbolInfo, Tick

_TF_MAP_NAMES = {"H1": "TIMEFRAME_H1", "H4": "TIMEFRAME_H4", "D1": "TIMEFRAME_D1"}


class MT5BrokerAdapter(BrokerAdapter):
    name = "mt5"

    def __init__(self, *, login: int = 0, password: str = "", server: str = "",
                 terminal_path: str = "", broker_tz: str = "Europe/Nicosia") -> None:
        self.login = login
        self.password = password
        self.server = server
        self.terminal_path = terminal_path
        self.tz = ZoneInfo(broker_tz)
        self._mt5 = None
        self._connected = False

    # ── connection ──────────────────────────────────────────────────────────
    def _import_mt5(self):
        if self._mt5 is None:
            import MetaTrader5 as mt5  # type: ignore
            self._mt5 = mt5
        return self._mt5

    def connect(self) -> bool:
        mt5 = self._import_mt5()
        kwargs = {}
        if self.terminal_path:
            kwargs["path"] = self.terminal_path
        if self.login:
            kwargs.update(login=int(self.login), password=self.password, server=self.server)
        ok = mt5.initialize(**kwargs) if kwargs else mt5.initialize()
        self._connected = bool(ok)
        return self._connected

    def disconnect(self) -> None:
        if self._mt5 is not None and self._connected:
            self._mt5.shutdown()
        self._connected = False

    def healthy(self) -> bool:
        if not self._connected or self._mt5 is None:
            return False
        return self._mt5.terminal_info() is not None and self._mt5.account_info() is not None

    # ── time ────────────────────────────────────────────────────────────────
    def _to_local(self, seconds) -> pd.DatetimeIndex:
        parsed = pd.to_datetime(seconds, unit="s")
        return parsed.tz_localize(self.tz, nonexistent="shift_forward", ambiguous="NaT")

    def _tf(self, timeframe: str) -> int:
        mt5 = self._import_mt5()
        return getattr(mt5, _TF_MAP_NAMES[timeframe])

    # ── market data ─────────────────────────────────────────────────────────
    def get_bars(self, symbol: str, timeframe: str, n_bars: int) -> pd.DataFrame:
        mt5 = self._import_mt5()
        rates = mt5.copy_rates_from_pos(symbol, self._tf(timeframe), 0, n_bars + 1)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"MT5 copy_rates {symbol} {timeframe} failed: {mt5.last_error()}")
        df = pd.DataFrame(rates)
        idx = self._to_local(df["time"])
        df = df.set_index(idx).sort_index()
        df = df[df.index.notna()]
        out = pd.DataFrame(index=df.index)
        out["open"], out["high"] = df["open"], df["high"]
        out["low"], out["close"] = df["low"], df["close"]
        if "spread" in df.columns:
            # MT5 'spread' is in points; convert to bps of price.
            pt = 1e-3 if symbol.endswith("JPY") else 1e-5
            out["spread_bps"] = (df["spread"] * pt / df["close"]) * 1e4
        if "tick_volume" in df.columns:
            out["volume"] = df["tick_volume"]
        return out.iloc[:-1].iloc[-n_bars:]   # drop forming bar

    def get_tick(self, symbol: str) -> Tick:
        mt5 = self._import_mt5()
        t = mt5.symbol_info_tick(symbol)
        if t is None:
            raise RuntimeError(f"MT5 tick {symbol} failed: {mt5.last_error()}")
        ts = pd.to_datetime(t.time, unit="s").tz_localize(self.tz, nonexistent="shift_forward",
                                                          ambiguous="NaT")
        return Tick(symbol=symbol, ts=ts, bid=float(t.bid), ask=float(t.ask))

    def symbol_info(self, symbol: str) -> SymbolInfo:
        mt5 = self._import_mt5()
        info = mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"MT5 symbol_info {symbol} failed: {mt5.last_error()}")
        if not info.visible:
            mt5.symbol_select(symbol, True)
            info = mt5.symbol_info(symbol)
        return SymbolInfo(
            symbol=symbol,
            contract_size=float(info.trade_contract_size),
            volume_min=float(info.volume_min),
            volume_max=float(info.volume_max),
            volume_step=float(info.volume_step),
            point=float(info.point),
            digits=int(info.digits),
            tradable=bool(info.visible),
        )

    def list_symbols(self) -> list[str]:
        mt5 = self._import_mt5()
        syms = mt5.symbols_get()
        return [s.name for s in syms] if syms else []

    # ── account ─────────────────────────────────────────────────────────────
    def account(self) -> AccountInfo:
        mt5 = self._import_mt5()
        a = mt5.account_info()
        if a is None:
            raise RuntimeError(f"MT5 account_info failed: {mt5.last_error()}")
        return AccountInfo(equity=float(a.equity), balance=float(a.balance),
                           currency=a.currency, margin_free=float(a.margin_free),
                           leverage=int(a.leverage))

    # ── orders (live demo mode) ─────────────────────────────────────────────
    def market_order(self, symbol: str, side: str, volume: float, *,
                     magic: int = 0, comment: str = "") -> OrderResult:
        mt5 = self._import_mt5()
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return OrderResult(ok=False, comment=f"no tick for {symbol}")
        is_buy = side.lower() == "buy"
        price = tick.ask if is_buy else tick.bid
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL,
            "price": price,
            "deviation": 20,
            "magic": int(magic),
            "comment": comment[:31],
            "type_filling": mt5.ORDER_FILLING_IOC,
            "type_time": mt5.ORDER_TIME_GTC,
        }
        t0 = time.perf_counter()
        res = mt5.order_send(req)
        latency_ms = (time.perf_counter() - t0) * 1e3
        if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
            return OrderResult(ok=False, requested_price=price, latency_ms=latency_ms,
                               comment=f"retcode={getattr(res, 'retcode', None)}",
                               raw=res._asdict() if res else None)
        return OrderResult(ok=True, ticket=int(res.order), filled_price=float(res.price),
                           filled_volume=float(res.volume), requested_price=price,
                           latency_ms=latency_ms, comment="ok", raw=res._asdict())

    def close_ticket(self, ticket: int) -> OrderResult:
        mt5 = self._import_mt5()
        pos = mt5.positions_get(ticket=ticket)
        if not pos:
            return OrderResult(ok=False, comment=f"no open position {ticket}")
        p = pos[0]
        symbol = p.symbol
        tick = mt5.symbol_info_tick(symbol)
        is_buy_close = p.type == mt5.POSITION_TYPE_SELL  # close short with a buy
        price = tick.ask if is_buy_close else tick.bid
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(p.volume),
            "type": mt5.ORDER_TYPE_BUY if is_buy_close else mt5.ORDER_TYPE_SELL,
            "position": int(ticket),
            "price": price,
            "deviation": 20,
            "magic": int(p.magic),
            "comment": "sal_close",
            "type_filling": mt5.ORDER_FILLING_IOC,
            "type_time": mt5.ORDER_TIME_GTC,
        }
        t0 = time.perf_counter()
        res = mt5.order_send(req)
        latency_ms = (time.perf_counter() - t0) * 1e3
        ok = res is not None and res.retcode == mt5.TRADE_RETCODE_DONE
        return OrderResult(ok=ok, ticket=ticket, filled_price=float(getattr(res, "price", 0.0)),
                           requested_price=price, latency_ms=latency_ms,
                           comment="ok" if ok else f"retcode={getattr(res, 'retcode', None)}")
