"""
Telethon command handlers for EMA H1 Trend Exchange.

Commands (send to your Saved Messages or the configured listen_chat):
  /start                          — help menu
  /status                         — bot runtime status (all symbols)
  /positions [SYMBOL]             — live open positions
  /wallet                         — account balance
  /pending [SYMBOL]               — pending stop orders on Bybit
  /config [SYMBOL]                — current bot configuration
  /trend SYMBOL                   — live H1 trend & EMA values
  /signals [SYMBOL] [N=10]        — recent strategy signals from CSV log
  /performance [SYMBOL] [days=30] — closed P&L statistics
  /journal SYMBOL [N=20]          — event journal entries (JSONL)
  /cancel_pending SYMBOL          — cancel all pending orders for symbol

Registration:
    from telegram_bot.commands import register_handlers
    register_handlers(telethon_client, services, bybit_client, base_cfg, listen_chat)
"""
from __future__ import annotations

import csv
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd
from telethon import TelegramClient, events

from config.settings import Settings
from exchange.bybit_client import BybitClient
from strategy.crypto_core import add_emas, h1_trend_series

log = logging.getLogger(__name__)

TEHRAN_TZ = ZoneInfo("Asia/Tehran")
MAX_MSG_LEN = 4000


# ── Formatting helpers ─────────────────────────────────────────────────────────

def _fmt_price(p: float) -> str:
    if p <= 0:
        return "0"
    if p >= 1000:
        return f"{p:,.2f}"
    if p >= 1:
        return f"{p:,.4f}"
    return f"{p:,.6f}"


def _fmt_pnl(pnl: float) -> str:
    sign = "+" if pnl >= 0 else ""
    return f"{sign}{pnl:.2f}"


def _ts_tehran(ts_ms: int) -> str:
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).astimezone(TEHRAN_TZ)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _raw_klines_to_df(raw: list) -> pd.DataFrame:
    rows = list(reversed(raw))
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume", "turnover"])
    df["time"] = pd.to_datetime(df["time"].astype("int64"), unit="ms", utc=True)
    df = df.set_index("time").sort_index()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df[["open", "high", "low", "close", "volume"]]


async def _send_chunked(event, text: str) -> None:
    """Reply to an event, splitting at MAX_MSG_LEN if needed."""
    if len(text) <= MAX_MSG_LEN:
        await event.reply(text)
        return
    lines = text.split("\n")
    chunk = ""
    for line in lines:
        if len(chunk) + len(line) + 1 > MAX_MSG_LEN:
            if chunk.strip():
                await event.reply(chunk)
            chunk = line + "\n"
        else:
            chunk += line + "\n"
    if chunk.strip():
        await event.reply(chunk)


# ── Data-fetch helpers (return formatted strings) ──────────────────────────────

async def _status_text(services: Dict) -> str:
    if not services:
        return "No active trading services."
    lines = ["Bot Status\n"]
    for symbol, svc in services.items():
        state = svc.state
        cfg = svc.settings
        summary = state.summary()
        mode = "DEMO" if cfg.bybit_demo else ("TESTNET" if cfg.bybit_testnet else "LIVE")
        pending_info = "None"
        if state.pending:
            p = state.pending
            age_min = p.age_minutes()
            pending_info = (
                f"{p.side.value.upper()} "
                f"entry={_fmt_price(p.entry)} "
                f"sl={_fmt_price(p.sl)} "
                f"tp={_fmt_price(p.tp)} "
                f"qty={p.qty} "
                f"(age={age_min:.0f}m)"
            )
        last_candle = str(summary.get("last_candle", "None"))
        if last_candle != "None" and "+" in last_candle:
            last_candle = last_candle.split("+")[0].strip()
        lines.append(f"{symbol}  [{mode}]")
        lines.append(f"  Cycles:    {summary['cycles']}  (errors: {summary['errors']})")
        lines.append(f"  Last bar:  {last_candle}")
        lines.append(f"  Pending:   {pending_info}")
        lines.append(f"  Recovery:  {'Done' if summary['recovery_done'] else 'Pending'}")
        lines.append(f"  Dry run:   {'Yes' if cfg.dry_run else 'No'}")
        lines.append("")
    return "\n".join(lines)


async def _positions_text(client: BybitClient, symbol: Optional[str]) -> str:
    positions = await client.get_positions(symbol) if symbol else await client.get_all_positions()
    if not positions:
        return f"No open positions{' for ' + symbol if symbol else ''}."
    lines = ["Open Positions\n"]
    for p in positions:
        pnl_pct = ""
        if p.entry_price > 0 and p.leverage > 0:
            margin = p.entry_price * p.size / p.leverage
            if margin > 0:
                pct = (p.unrealized_pnl / margin) * 100
                pnl_pct = f"  ({_fmt_pnl(pct)}%)"
        lines.append(f"{p.symbol}  {p.side.value.upper()}")
        lines.append(f"  Size:    {p.size}")
        lines.append(f"  Entry:   {_fmt_price(p.entry_price)}")
        lines.append(f"  uPnL:    {_fmt_pnl(p.unrealized_pnl)} USDT{pnl_pct}")
        lines.append(f"  Lev:     {int(p.leverage)}x")
        lines.append("")
    return "\n".join(lines)


async def _wallet_text(client: BybitClient) -> str:
    bal = await client.get_balance()
    return (
        "Wallet Balance\n\n"
        f"  Equity:      {bal.total_equity:,.2f} USDT\n"
        f"  Available:   {bal.available_balance:,.2f} USDT\n"
        f"  Used Margin: {bal.used_margin:,.2f} USDT\n"
    )


async def _pending_text(client: BybitClient, services: Dict, req_symbol: Optional[str]) -> str:
    symbols_to_check = [req_symbol] if req_symbol else list(services.keys())
    if not symbols_to_check:
        return "No active symbols configured."
    lines = ["Pending Orders\n"]
    found_any = False
    for symbol in symbols_to_check:
        try:
            all_orders = await client.get_open_stop_orders(symbol)
            ema_orders = [o for o in all_orders if o.get("orderLinkId", "").startswith("ema-")]
        except Exception as exc:
            lines.append(f"{symbol}: Error — {exc}\n")
            continue
        if not ema_orders:
            if req_symbol:
                lines.append(f"{symbol}: No pending EMA orders.")
            continue
        for o in ema_orders:
            found_any = True
            trigger = float(o.get("triggerPrice", 0) or 0)
            sl_val = float(o.get("stopLoss", 0) or 0)
            tp_val = float(o.get("takeProfit", 0) or 0)
            qty = o.get("qty", "?")
            status = o.get("orderStatus", "?")
            link_id = o.get("orderLinkId", "?")
            side = o.get("side", "?")
            created_ms = int(o.get("createdTime", 0) or 0)
            created_str = _ts_tehran(created_ms) if created_ms else "?"
            lines.append(f"{symbol}  {side.upper()}")
            lines.append(f"  Link:    {link_id}")
            lines.append(f"  Trigger: {_fmt_price(trigger)}")
            lines.append(f"  SL:      {_fmt_price(sl_val)}")
            lines.append(f"  TP:      {_fmt_price(tp_val)}")
            lines.append(f"  Qty:     {qty}")
            lines.append(f"  Status:  {status}")
            lines.append(f"  Created: {created_str}")
            lines.append("")
    if not found_any and not req_symbol:
        return "No pending EMA orders found."
    return "\n".join(lines)


async def _config_text(services: Dict, base: Settings, req_symbol: Optional[str]) -> str:
    if req_symbol:
        if req_symbol not in services:
            return f"Symbol {req_symbol} not in active services."
        cfgs: List = [(req_symbol, services[req_symbol].settings)]
    elif services:
        cfgs = [(sym, svc.settings) for sym, svc in services.items()]
    else:
        cfgs = [("(base)", base)]
    lines = []
    for sym, cfg in cfgs:
        mode = "DEMO" if cfg.bybit_demo else ("TESTNET" if cfg.bybit_testnet else "LIVE")
        lines.append(f"Config — {sym}")
        lines.append(f"  Mode:         {mode}")
        lines.append(f"  Dry Run:      {'Yes' if cfg.dry_run else 'No'}")
        lines.append(f"  Risk/Trade:   {cfg.risk_fixed_usdt:.2f} USDT")
        lines.append(f"  R:R:          {cfg.rr}")
        lines.append(f"  Leverage:     {cfg.leverage}x")
        lines.append(f"  Lookback:     {cfg.lookback_bars} bars")
        lines.append(f"  Expiry:       {cfg.pending_expiry_min} min  ({cfg.expiry_bars} bars M5)")
        lines.append(f"  Offset:       {cfg.pending_offset_ticks} ticks")
        lines.append(f"  Pip Size:     {cfg.effective_pip_size():.6f}")
        lines.append(f"  Magic:        {cfg.effective_magic()}")
        lines.append(f"  Pos Mode:     {cfg.position_mode}")
        lines.append(f"  Replace:      {'Yes' if cfg.replace_pending else 'No'}")
        lines.append("")
    return "\n".join(lines)


async def _trend_text(client: BybitClient, services: Dict, symbol: str) -> str:
    raw_h1 = await client.get_kline(symbol, "60", 250)
    h1 = _raw_klines_to_df(raw_h1)
    if len(h1) < 30:
        return f"Insufficient H1 data for {symbol} ({len(h1)} bars)."
    h1 = add_emas(h1)
    trend_series = h1_trend_series(h1)
    current_trend = trend_series.iloc[-1] if not trend_series.empty else "unknown"
    last_bar = h1.index[-1]
    last_close = h1["close"].iloc[-1]
    ema8 = h1["ema_8"].iloc[-1]
    ema13 = h1["ema_13"].iloc[-1]
    ema21 = h1["ema_21"].iloc[-1]
    recent_trend_str = " → ".join(t.upper() for t in trend_series.tail(3).tolist())
    if current_trend == "bull":
        alignment = f"Close({_fmt_price(last_close)}) > EMA8({_fmt_price(ema8)}) > EMA13({_fmt_price(ema13)}) > EMA21({_fmt_price(ema21)})"
    elif current_trend == "bear":
        alignment = f"Close({_fmt_price(last_close)}) < EMA8({_fmt_price(ema8)}) < EMA13({_fmt_price(ema13)}) < EMA21({_fmt_price(ema21)})"
    else:
        alignment = f"Close: {_fmt_price(last_close)}  EMA8: {_fmt_price(ema8)}  EMA13: {_fmt_price(ema13)}  EMA21: {_fmt_price(ema21)}"
    trend_label = {"bull": "BULL (uptrend)", "bear": "BEAR (downtrend)", "flat": "FLAT (no signal)"}.get(
        current_trend, current_trend.upper()
    )
    pending_info = "None"
    if symbol in services:
        state = services[symbol].state
        if state.pending:
            p = state.pending
            age_min = p.age_minutes()
            cfg = services[symbol].settings
            expires_in = max(0, cfg.pending_expiry_min - age_min)
            pending_info = (
                f"{p.side.value.upper()}  "
                f"entry={_fmt_price(p.entry)}  "
                f"sl={_fmt_price(p.sl)}  "
                f"tp={_fmt_price(p.tp)}\n"
                f"   qty={p.qty}  age={age_min:.0f}m  expires_in={expires_in:.0f}m"
            )
    return (
        f"Trend Analysis — {symbol}\n\n"
        f"H1 Trend:  {trend_label}\n"
        f"Recent:    {recent_trend_str}\n"
        f"Last bar:  {last_bar.strftime('%Y-%m-%d %H:%M UTC')}\n\n"
        f"H1 EMA Alignment:\n"
        f"  {alignment}\n\n"
        f"Pending Order:\n"
        f"  {pending_info}\n"
    )


async def _signals_text(base: Settings, services: Dict, req_symbol: Optional[str], n: int) -> str:
    csv_path: Optional[str] = base.signal_log_csv
    if not csv_path:
        for svc in services.values():
            if svc.settings.signal_log_csv:
                csv_path = svc.settings.signal_log_csv
                break
    if not csv_path or not Path(csv_path).exists():
        return (
            "Signal log CSV not found.\n"
            "Enable: SIGNAL_LOG_CSV=logs/signals.csv in .env"
        )
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if req_symbol and row.get("symbol") != req_symbol:
                continue
            rows.append(row)
    rows = rows[-n:]
    if not rows:
        sym_info = f" for {req_symbol}" if req_symbol else ""
        return f"No signals found{sym_info}."
    sym_label = f" — {req_symbol}" if req_symbol else ""
    lines = [f"Recent Signals{sym_label}  (last {len(rows)})\n"]
    for i, row in enumerate(reversed(rows), 1):
        logged_at = row.get("logged_at_utc", "?")[:16].replace("T", " ")
        sym = row.get("symbol", "?")
        side_raw = row.get("side", "?")
        side = side_raw.split(".")[-1].upper()
        trend = row.get("trend", "?")
        entry = float(row.get("adjusted_entry") or row.get("model_entry") or 0)
        sl_v = float(row.get("adjusted_sl") or row.get("model_sl") or 0)
        tp_v = float(row.get("adjusted_tp") or row.get("model_tp") or 0)
        qty = row.get("normalized_qty") or row.get("setup_qty", "?")
        action = row.get("action", "?")
        link = row.get("order_link_id", "")
        rr = row.get("rr", "?")
        lines.append(f"{i}. [{logged_at} UTC]  {sym}  {side}  ({trend})")
        lines.append(f"   Entry: {_fmt_price(entry)}  SL: {_fmt_price(sl_v)}  TP: {_fmt_price(tp_v)}")
        lines.append(f"   Qty: {qty}  R:R {rr}  Action: {action}")
        if link:
            lines.append(f"   Link: {link}")
        lines.append("")
    return "\n".join(lines)


async def _performance_text(client: BybitClient, req_symbol: Optional[str], days: int) -> str:
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 24 * 60 * 60 * 1000
    records = await client.get_closed_pnl(
        symbol=req_symbol,
        start_ms=start_ms,
        end_ms=end_ms,
        limit=100,
    )
    ema_records = [r for r in records if r.get("orderLinkId", "").startswith("ema-")]
    data = ema_records if ema_records else records
    bot_label = "  (EMA bot trades only)" if ema_records else ""
    if not data:
        sym_info = f" for {req_symbol}" if req_symbol else ""
        return f"No closed trades found{sym_info} in the last {days} days."
    pnls = [float(r.get("closedPnl", 0) or 0) for r in data]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    total = len(pnls)
    win_rate = (len(wins) / total * 100) if total > 0 else 0.0
    total_pnl = sum(pnls)
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    pf_str = f"{pf:.2f}" if pf != float("inf") else "∞"
    best = max(pnls) if pnls else 0.0
    worst = min(pnls) if pnls else 0.0
    breakdown = ""
    if not req_symbol:
        by_sym: Dict[str, List[float]] = {}
        for r in data:
            sym = r.get("symbol", "?")
            by_sym.setdefault(sym, []).append(float(r.get("closedPnl", 0) or 0))
        sym_lines = []
        for sym, sym_pnls in sorted(by_sym.items()):
            sym_wins = len([p for p in sym_pnls if p > 0])
            sym_lines.append(
                f"  {sym:<12} {len(sym_pnls)} trades  {sym_wins}W/{len(sym_pnls)-sym_wins}L  PnL: {_fmt_pnl(sum(sym_pnls))} USDT"
            )
        if sym_lines:
            breakdown = "\nBy Symbol:\n" + "\n".join(sym_lines) + "\n"
    sym_label = f" — {req_symbol}" if req_symbol else ""
    sep = "─" * 30
    return (
        f"Performance{sym_label}  (last {days} days){bot_label}\n\n"
        f"Trades:         {total}\n"
        f"Wins:           {len(wins)}\n"
        f"Losses:         {len(losses)}\n"
        f"Win Rate:       {win_rate:.1f}%\n"
        f"{sep}\n"
        f"Total PnL:      {_fmt_pnl(total_pnl)} USDT\n"
        f"Avg Win:        {_fmt_pnl(avg_win)} USDT\n"
        f"Avg Loss:       {_fmt_pnl(avg_loss)} USDT\n"
        f"Profit Factor:  {pf_str}\n"
        f"{sep}\n"
        f"Best Trade:     {_fmt_pnl(best)} USDT\n"
        f"Worst Trade:    {_fmt_pnl(worst)} USDT\n"
        f"{breakdown}"
    )


def _journal_text(base: Settings, services: Dict, symbol: str, n: int) -> str:
    log_dir = base.event_log_dir
    if not log_dir and symbol in services:
        log_dir = services[symbol].settings.event_log_dir
    if not log_dir:
        return "Event journal not enabled. Set EVENT_LOG_DIR in .env."
    journal_path = Path(log_dir) / f"events_{symbol}.jsonl"
    if not journal_path.exists():
        return f"No event journal found for {symbol}.\nExpected: {journal_path}"
    with open(journal_path, encoding="utf-8") as f:
        all_lines = f.readlines()
    last_lines = all_lines[-n:] if len(all_lines) > n else all_lines
    lines = [f"Journal — {symbol}  (last {len(last_lines)} events)\n"]
    for raw_line in last_lines:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            entry = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        ts = entry.get("ts", "?")[:19].replace("T", " ")
        event = entry.get("event", "?")
        extras: list = []
        if event == "cycle_start":
            extras.append(f"cycle={entry.get('cycle', '?')}")
        elif event == "data_fetched":
            h1_trend = entry.get("h1_trend", {})
            last_trend = list(h1_trend.values())[-1] if h1_trend else "?"
            extras.append(f"m5={entry.get('m5_bars','?')} h1={entry.get('h1_bars','?')} trend={last_trend}")
        elif event == "signal":
            total_sig = int(entry.get("total_signals", 0) or 0)
            if total_sig > 0:
                side_raw = str(entry.get("side", "?"))
                side_disp = side_raw.split(".")[-1].upper()
                entry_p = float(entry.get("entry", 0) or 0)
                sl_p = float(entry.get("sl", 0) or 0)
                tp_p = float(entry.get("tp", 0) or 0)
                extras.append(f"{side_disp} entry={_fmt_price(entry_p)} sl={_fmt_price(sl_p)} tp={_fmt_price(tp_p)}")
            else:
                extras.append("no signal")
        elif event == "balance":
            equity = entry.get("equity")
            avail = entry.get("available")
            extras.append(
                f"equity={float(equity):.2f}" if equity is not None else "equity=?"
            )
            extras.append(
                f"avail={float(avail):.2f}" if avail is not None else "avail=?"
            )
        elif event in ("order_created", "order_filled"):
            link = entry.get("link_id") or entry.get("order_link_id", "?")
            side_disp = str(entry.get("side", "")).split(".")[-1].upper()
            extras.append(f"link={link}" + (f" side={side_disp}" if side_disp else ""))
        elif event in ("order_cancelled", "order_expired"):
            link = entry.get("link_id") or entry.get("order_link_id", "?")
            reason = str(entry.get("reason", ""))[:40]
            extras.append(f"link={link}" + (f" ({reason})" if reason else ""))
        elif event == "cycle_error":
            err = str(entry.get("error", ""))[:60]
            extras.append(f"cycle={entry.get('cycle','?')} {err}")
        elif event == "cycle_complete":
            extras.append(f"cycle={entry.get('cycle', '?')}")
        elif event == "position_check":
            extras.append(f"has_position={entry.get('has_position', '?')}")
        elif event == "bot_start":
            extras.append(f"mode={entry.get('mode','?')} dry_run={entry.get('dry_run','?')}")
        elif event == "account_snapshot":
            equity = entry.get("equity")
            eq_str = f"{float(equity):.2f}" if equity is not None else "?"
            n_pos = len(entry.get("open_positions") or [])
            n_ord = len(entry.get("pending_orders") or [])
            extras.append(f"equity={eq_str} positions={n_pos} pending={n_ord}")
        extra_str = "  |  ".join(extras)
        lines.append(f"[{ts}]  {event}" + (f"  —  {extra_str}" if extra_str else ""))
    return "\n".join(lines)


# ── Telethon command handler registration ──────────────────────────────────────

def register_handlers(
    tg_client: TelegramClient,
    services: Dict,
    bybit_client: BybitClient,
    base_cfg: Settings,
    listen_chat: Optional[int] = None,
) -> None:
    """
    Register Telethon event handlers for slash commands.

    listen_chat: Telegram chat ID to listen in (e.g. your Saved Messages = None means
                 all incoming messages to your account). Set to a specific chat ID to
                 restrict command handling.
    """
    pattern_kw = {"chats": listen_chat} if listen_chat else {}

    @tg_client.on(events.NewMessage(pattern=r"^/start", **pattern_kw))
    async def start_handler(event):
        symbols = list(services.keys())
        text = (
            "EMA H1 Trend Exchange — Telethon Bot\n\n"
            f"Active symbols: {', '.join(symbols) if symbols else 'none'}\n\n"
            "Commands:\n"
            "/status — bot runtime status (all symbols)\n"
            "/positions [SYMBOL] — open positions\n"
            "/wallet — account balance\n"
            "/pending [SYMBOL] — pending stop orders\n"
            "/config [SYMBOL] — bot configuration\n"
            "/trend SYMBOL — live H1 trend & EMA values\n"
            "/signals [SYMBOL] [N] — recent signals from log (default N=10)\n"
            "/performance [SYMBOL] [days] — P&L stats (default 30 days)\n"
            "/journal SYMBOL [N] — event log entries (default N=20)\n"
            "/cancel_pending SYMBOL — cancel pending orders\n"
        )
        await _send_chunked(event, text)

    @tg_client.on(events.NewMessage(pattern=r"^/status", **pattern_kw))
    async def status_handler(event):
        text = await _status_text(services)
        await _send_chunked(event, text)

    @tg_client.on(events.NewMessage(pattern=r"^/positions", **pattern_kw))
    async def positions_handler(event):
        args = event.raw_text.split()[1:]
        symbol = args[0].upper() if args else None
        try:
            text = await _positions_text(bybit_client, symbol)
        except Exception as exc:
            text = f"Error fetching positions: {exc}"
        await _send_chunked(event, text)

    @tg_client.on(events.NewMessage(pattern=r"^/wallet", **pattern_kw))
    async def wallet_handler(event):
        try:
            text = await _wallet_text(bybit_client)
        except Exception as exc:
            text = f"Error fetching wallet: {exc}"
        await _send_chunked(event, text)

    @tg_client.on(events.NewMessage(pattern=r"^/pending", **pattern_kw))
    async def pending_handler(event):
        args = event.raw_text.split()[1:]
        req_symbol = args[0].upper() if args else None
        try:
            text = await _pending_text(bybit_client, services, req_symbol)
        except Exception as exc:
            text = f"Error fetching pending orders: {exc}"
        await _send_chunked(event, text)

    @tg_client.on(events.NewMessage(pattern=r"^/config", **pattern_kw))
    async def config_handler(event):
        args = event.raw_text.split()[1:]
        req_symbol = args[0].upper() if args else None
        text = await _config_text(services, base_cfg, req_symbol)
        await _send_chunked(event, text)

    @tg_client.on(events.NewMessage(pattern=r"^/trend", **pattern_kw))
    async def trend_handler(event):
        args = event.raw_text.split()[1:]
        if not args:
            symbol = list(services.keys())[0] if services else None
            if not symbol:
                await event.reply("Usage: /trend SYMBOL")
                return
        else:
            symbol = args[0].upper()
        await event.reply(f"Fetching trend data for {symbol}...")
        try:
            text = await _trend_text(bybit_client, services, symbol)
        except Exception as exc:
            text = f"Error fetching trend data: {exc}"
        await _send_chunked(event, text)

    @tg_client.on(events.NewMessage(pattern=r"^/signals", **pattern_kw))
    async def signals_handler(event):
        args = event.raw_text.split()[1:]
        req_symbol: Optional[str] = None
        n = 10
        for arg in args:
            if arg.isdigit():
                n = max(1, min(50, int(arg)))
            else:
                req_symbol = arg.upper()
        try:
            text = await _signals_text(base_cfg, services, req_symbol, n)
        except Exception as exc:
            text = f"Error reading signals: {exc}"
        await _send_chunked(event, text)

    @tg_client.on(events.NewMessage(pattern=r"^/performance", **pattern_kw))
    async def performance_handler(event):
        args = event.raw_text.split()[1:]
        req_symbol: Optional[str] = None
        days = 30
        for arg in args:
            if arg.isdigit():
                days = max(1, min(365, int(arg)))
            else:
                req_symbol = arg.upper()
        await event.reply(f"Fetching P&L{' for ' + req_symbol if req_symbol else ''} (last {days} days)...")
        try:
            text = await _performance_text(bybit_client, req_symbol, days)
        except Exception as exc:
            text = f"Error fetching P&L: {exc}"
        await _send_chunked(event, text)

    @tg_client.on(events.NewMessage(pattern=r"^/journal", **pattern_kw))
    async def journal_handler(event):
        args = event.raw_text.split()[1:]
        if not args:
            await event.reply("Usage: /journal SYMBOL [N]\nExample: /journal BTCUSDT 20")
            return
        symbol = args[0].upper()
        try:
            n = max(1, min(100, int(args[1]))) if len(args) > 1 else 20
        except (ValueError, IndexError):
            n = 20
        try:
            text = _journal_text(base_cfg, services, symbol, n)
        except Exception as exc:
            text = f"Error reading journal: {exc}"
        await _send_chunked(event, text)

    @tg_client.on(events.NewMessage(pattern=r"^/cancel_pending", **pattern_kw))
    async def cancel_pending_handler(event):
        args = event.raw_text.split()[1:]
        if not args:
            await event.reply("Usage: /cancel_pending SYMBOL\nExample: /cancel_pending BTCUSDT")
            return
        symbol = args[0].upper()
        try:
            await bybit_client.cancel_all_stop_orders(symbol)
            if symbol in services:
                services[symbol].state.clear_pending("cancelled via Telegram command")
            text = f"All pending EMA orders cancelled for {symbol}."
        except Exception as exc:
            text = f"Error cancelling orders for {symbol}: {exc}"
        await _send_chunked(event, text)

    log.info("Registered 11 Telethon command handlers (listen_chat=%s)", listen_chat or "all")
