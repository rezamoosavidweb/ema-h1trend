"""
Synchronous Telegram notifier for the MT5 Order Block bot.

Reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from the repo-root .env file
(or from environment variables if the .env is absent).

Uses only stdlib — no extra packages needed beyond what the MT5 script requires.

Usage:
    notifier = Mt5Notifier()
    notifier.send("Hello from the bot!")
    notifier.notify_signal(signal_dict)
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_env(path: Path) -> dict[str, str]:
    """Parse KEY=VALUE lines from a .env file; ignore comments and blanks."""
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key   = key.strip()
        value = value.strip().strip("'\"")
        env[key] = value
    return env


class Mt5Notifier:
    """Fire-and-forget Telegram sender for the synchronous MT5 bot."""

    def __init__(self) -> None:
        env     = _load_env(_REPO_ROOT / ".env")
        token   = env.get("TELEGRAM_BOT_TOKEN")   or os.environ.get("TELEGRAM_BOT_TOKEN",  "")
        chat_id = env.get("TELEGRAM_CHAT_ID")      or os.environ.get("TELEGRAM_CHAT_ID",    "")
        self._url     = f"https://api.telegram.org/bot{token}/sendMessage"
        self._chat_id = chat_id
        self._enabled = bool(token and chat_id)

    # ── Low-level send ─────────────────────────────────────────────────────────

    def send(self, text: str) -> None:
        """Send an HTML-formatted message. Silently swallows all errors."""
        if not self._enabled:
            return
        try:
            payload = json.dumps({
                "chat_id":                  self._chat_id,
                "text":                     text,
                "parse_mode":               "HTML",
                "disable_web_page_preview": True,
            }).encode()
            req = urllib.request.Request(
                self._url,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=10)
        except Exception:
            pass  # never crash the trading bot over a notification failure

    # ── Event formatters ───────────────────────────────────────────────────────

    def notify_signal(self, signal: dict) -> None:
        direction = signal.get("direction", "?")
        icon = "🟢" if direction == "BUY" else "🔴"
        text = (
            f"{icon} <b>SIGNAL — XAUUSD</b>\n"
            f"Direction:    <b>{direction}</b>\n"
            f"Entry:        {signal.get('entry')}\n"
            f"SL:           {signal.get('sl')}\n"
            f"TP:           {signal.get('tp')}\n"
            f"OB Time:      {str(signal.get('ob_time', ''))[:16]}\n"
            f"Displaced by: {signal.get('displaced_by')} pts"
        )
        self.send(text)

    def notify_slippage_adjusted(self, data: dict) -> None:
        direction = data.get("direction", "?")
        text = (
            f"⚠️ <b>SLIPPAGE ADJUSTED — XAUUSD</b>\n"
            f"Direction: {direction}\n"
            f"OB Entry:  {data.get('ob_entry')}\n"
            f"Market:    {data.get('market_price')}\n"
            f"Slippage:  {data.get('slippage_pts')} pts\n"
            f"Volume:    {data.get('volume_original')} → {data.get('volume_adjusted')}\n"
            f"TP:        {data.get('tp_original')} → {data.get('tp_adjusted')}\n"
            f"SL:        {data.get('sl_unchanged')} (unchanged)"
        )
        self.send(text)

    def notify_order_placed(self, data: dict) -> None:
        direction = data.get("direction", "?")
        icon = "🟢" if direction == "BUY" else "🔴"
        text = (
            f"✅ <b>ORDER PLACED — XAUUSD</b>\n"
            f"Ticket:    {data.get('ticket')}\n"
            f"Direction: {icon} <b>{direction}</b>\n"
            f"Volume:    {data.get('volume')}\n"
            f"SL:        {data.get('sl')}\n"
            f"TP:        {data.get('tp')}\n"
            f"Slippage:  {data.get('slippage_pts')} pts"
        )
        self.send(text)

    def notify_skip(self, data: dict) -> None:
        """Send a skip notification; silently drops routine/noisy reasons."""
        reason = data.get("reason", "?")
        # These happen every cycle during off-hours or after an OB is already traded
        if reason in ("market_closed_or_stale", "already_traded"):
            return
        icon = "⏭"
        lines = [f"{icon} <b>SKIP — XAUUSD</b>", f"Reason: <code>{reason}</code>"]
        if "slippage_pts" in data:
            lines.append(f"Slippage: {data['slippage_pts']} pts (max {data.get('max_pts')})")
        if "direction" in data:
            lines.append(f"Direction: {data['direction']}")
        if "ob_time" in data:
            lines.append(f"OB Time: {str(data['ob_time'])[:16]}")
        if reason == "position_open" and "missed_signal" in data:
            ms = data["missed_signal"]
            lines.append(
                f"Missed: {ms.get('direction')} entry={ms.get('entry')} "
                f"sl={ms.get('sl')} tp={ms.get('tp')}"
            )
        self.send("\n".join(lines))

    def notify_position_closed(
        self,
        ticket: int,
        profit: float,
        balance: float,
        equity: float,
    ) -> None:
        pnl_icon = "📈" if profit >= 0 else "📉"
        outcome  = "TP HIT ✅" if profit >= 0 else "SL HIT 🛑"
        text = (
            f"{pnl_icon} <b>POSITION CLOSED — XAUUSD</b>\n"
            f"Ticket: {ticket}\n"
            f"Result: <b>{outcome}</b>\n"
            f"Profit: {profit:+.2f} USD\n\n"
            f"💰 <b>Account Balance</b>\n"
            f"Balance: {balance:,.2f} USD\n"
            f"Equity:  {equity:,.2f} USD"
        )
        self.send(text)
