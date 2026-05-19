"""
Telegram message formatters for Bybit WebSocket events.

Converts raw Bybit order/position WebSocket dicts into readable, emoji-rich
Telegram messages — one formatter per topic type.
"""

from __future__ import annotations

from datetime import datetime, timezone


# ── Lookup tables ─────────────────────────────────────────────────────────────

_STATUS_EMOJI: dict[str, str] = {
    "untriggered":     "⏸",
    "triggered":       "🎯",
    "new":             "📋",
    "partiallyfilled": "🔄",
    "filled":          "✅",
    "cancelled":       "❌",
    "deactivated":     "🚫",
    "rejected":        "⛔",
}

_STATUS_LABEL: dict[str, str] = {
    "untriggered":     "Conditional order created (not triggered)",
    "triggered":       "Order triggered — entering market",
    "new":             "Order placed",
    "partiallyfilled": "Partially filled",
    "filled":          "Order filled",
    "cancelled":       "Order cancelled",
    "deactivated":     "Order deactivated",
    "rejected":        "Order rejected",
}

_SIDE_EMOJI: dict[str, str] = {
    "buy":  "🟢",
    "sell": "🔴",
}

_STOP_LABEL: dict[str, str] = {
    "StopLoss":         "🛑 Stop Loss",
    "TakeProfit":       "🎯 Take Profit",
    "TrailingStop":     "📈 Trailing Stop",
    "PartialStopLoss":  "🛑 Partial Stop Loss",
    "PartialTakeProfit":"🎯 Partial Take Profit",
}

_CANCEL_LABEL: dict[str, str] = {
    "CancelByUser":           "Cancelled by user",
    "CancelByReduceOnly":     "Cancelled — reduce only",
    "CancelByPrepareLiq":     "Cancelled — pre-liquidation",
    "CancelAllBeforeLiq":     "Cancelled — liquidation",
    "CancelByAdmin":          "Cancelled by admin",
    "CancelByTpSlTsClear":    "Cancelled — TP/SL cleared",
    "CancelByPzSideCh":       "Cancelled — position side change",
    "UNKNOWN":                "",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ms_to_str(ts_ms: str | int | None) -> str:
    if not ts_ms:
        return "—"
    try:
        return datetime.fromtimestamp(
            int(ts_ms) / 1000, tz=timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return str(ts_ms)


def _fmt(value: str | float | None, decimals: int = 4) -> str:
    if value is None or value == "" or value == "0" or value == 0:
        return ""
    try:
        f = float(value)
        if f == 0:
            return ""
        if decimals == 0:
            return f"{f:,.0f}"
        return f"{f:,.{decimals}f}"
    except Exception:
        return str(value)


def _line(label: str, value: str, unit: str = "") -> str | None:
    """Return 'label: value unit' or None if value is empty."""
    v = value.strip() if value else ""
    if not v:
        return None
    return f"{label}: {v}{' ' + unit if unit else ''}"


# ── Order formatter ───────────────────────────────────────────────────────────

def format_order_event(message: dict) -> str:
    """
    Format a Bybit private `order` WebSocket message as a Telegram notification.

    Handles batches — one Bybit message can carry multiple order updates.
    """
    data: list[dict] = message.get("data", [])
    if not data:
        return ""

    topic     = message.get("topic", "order")
    ts_ms     = message.get("creationTime") or message.get("ts")
    msg_time  = _ms_to_str(ts_ms)
    msg_id    = message.get("id", "—")
    count     = len(data)

    header = (
        f"📡 WebSocket Message\n"
        f"\n"
        f"Topic: {topic}\n"
        f"ID: {msg_id}\n"
        f"Time: {msg_time}\n"
        f"Orders Count: {count}"
    )

    blocks = [header]
    for idx, order in enumerate(data, 1):
        blocks.append(_format_single_order(order, idx))

    return "\n\n\n".join(blocks)


def _format_single_order(order: dict, index: int) -> str:
    status_raw  = (order.get("orderStatus") or "").lower()
    emoji       = _STATUS_EMOJI.get(status_raw, "📋")
    status_text = _STATUS_LABEL.get(status_raw, status_raw.capitalize())

    side_raw    = (order.get("side") or "").lower()
    side_emoji  = _SIDE_EMOJI.get(side_raw, "")

    stop_type   = order.get("stopOrderType") or ""
    stop_label  = _STOP_LABEL.get(stop_type, "")

    cancel_type = order.get("cancelType") or "UNKNOWN"
    cancel_text = _CANCEL_LABEL.get(cancel_type, cancel_type)

    # Header emoji: stop-loss triggered → 🛑, take-profit → 🎯, else status emoji
    if stop_type == "StopLoss":
        hdr_emoji = "🛑"
    elif stop_type == "TakeProfit":
        hdr_emoji = "🎯"
    else:
        hdr_emoji = emoji

    qty       = order.get("qty") or order.get("leavesQty") or "—"
    price     = _fmt(order.get("price"), 4)
    trigger   = _fmt(order.get("triggerPrice"), 4)
    avg       = _fmt(order.get("avgPrice") or order.get("lastExecPrice"), 4)
    exec_qty  = _fmt(order.get("cumExecQty"), 4)
    exec_val  = _fmt(order.get("cumExecValue"), 2)
    fee       = _fmt(order.get("cumExecFee"), 5)
    sl        = _fmt(order.get("stopLoss"), 4)
    tp        = _fmt(order.get("takeProfit"), 4)
    created   = _ms_to_str(order.get("createdTime"))
    order_type = order.get("orderType", "—")

    rows = [f"{hdr_emoji} Order #{index}:\n"]

    for item in [
        _line("Symbol",      order.get("symbol", "—")),
        _line("Side",        f"{side_emoji} {order.get('side', '—')}"),
        _line("Order ID",    order.get("orderId", "—")),
        _line("Status",      f"{emoji} {status_text}"),
        _line("Order Type",  stop_label if stop_label else order_type),
        _line("Quantity",    qty),
        _line("Price",       price),
        _line("Trigger Price", trigger),
        _line("Avg Price",   avg),
        _line("Executed Qty",  exec_qty),
        _line("Executed Value", exec_val, "USDT"),
        _line("Fee",         fee, "USDT"),
        _line("Stop Loss",   sl),
        _line("Take Profit", tp),
        _line("Created",     created),
        _line("Cancel Reason", cancel_text if cancel_text else None),
    ]:
        if item is not None:
            rows.append(item)

    return "\n".join(rows)


# ── Position formatter ────────────────────────────────────────────────────────

def format_position_event(message: dict) -> str:
    """
    Format a Bybit private `position` WebSocket message as a Telegram notification.
    """
    data: list[dict] = message.get("data", [])
    if not data:
        return ""

    ts_ms    = message.get("creationTime") or message.get("ts")
    msg_time = _ms_to_str(ts_ms)

    header = f"📊 Position Update\n\nTime: {msg_time}"
    blocks = [header]

    for pos in data:
        blocks.append(_format_single_position(pos))

    return "\n\n\n".join(blocks)


def _format_single_position(pos: dict) -> str:
    symbol    = pos.get("symbol", "—")
    side_raw  = (pos.get("side") or "").lower()
    side_emoji = _SIDE_EMOJI.get(side_raw, "")
    size      = pos.get("size", "0")
    entry     = _fmt(pos.get("avgPrice") or pos.get("entryPrice"), 4)
    upnl_raw  = pos.get("unrealisedPnl", "0")
    lev       = pos.get("leverage", "1")
    sl        = _fmt(pos.get("stopLoss"), 4)
    tp        = _fmt(pos.get("takeProfit"), 4)

    try:
        upnl_f   = float(upnl_raw or 0)
        pnl_icon = "📈" if upnl_f >= 0 else "📉"
        upnl_str = f"{pnl_icon} {upnl_f:+.2f} USDT"
    except Exception:
        upnl_str = f"{upnl_raw} USDT"

    try:
        is_closed = float(size) == 0
    except Exception:
        is_closed = size in ("0", "")

    if is_closed:
        rpnl_raw = pos.get("cumRealisedPnl") or pos.get("realisedPnl") or ""
        try:
            rpnl_f   = float(rpnl_raw or 0)
            rpnl_str = f"{'📈' if rpnl_f >= 0 else '📉'} {rpnl_f:+.2f} USDT"
        except Exception:
            rpnl_str = ""

        rows = [f"❌ Position Closed\n", f"Symbol: {symbol}"]
        if rpnl_str:
            rows.append(f"Realised PnL: {rpnl_str}")
        return "\n".join(rows)

    rows = [f"{side_emoji} Position Update\n"]
    for item in [
        _line("Symbol",         symbol),
        _line("Side",           f"{side_emoji} {pos.get('side', '—')}"),
        _line("Size",           size),
        _line("Entry Price",    entry),
        _line("Leverage",       f"{lev}x"),
        _line("Unrealised PnL", upnl_str),
        _line("Stop Loss",      sl),
        _line("Take Profit",    tp),
    ]:
        if item is not None:
            rows.append(item)

    return "\n".join(rows)
