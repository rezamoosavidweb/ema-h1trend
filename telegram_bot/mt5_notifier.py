"""
Synchronous Telegram notifier with proper error observability.

Why a rewrite?
    The previous version did `except Exception: pass`, which made network or
    token errors invisible. After the 2026-05-20 BUY 4543.21 incident the user
    had no way to tell whether telegram WAS notified for the trade open.

This version:
    * accepts an optional `logger` (any object with .event(name, **kwargs))
    * logs every send attempt as `telegram_sent` or `telegram_error`
    * captures HTTP status + body excerpt for non-2xx responses
    * captures URLError/timeout details for network failures
    * still never raises (would crash the trading bot)
    * supports a `retries` count for transient HTTP 5xx / 429 responses

Configuration unchanged (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID from .env or env).
"""

from __future__ import annotations

import json
import os
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional, Protocol

_REPO_ROOT = Path(__file__).resolve().parent.parent


# ── SSL context (fixes "self-signed certificate" failures on hardened servers) ─
#
# On Windows servers behind a corporate proxy / older OpenSSL builds, the
# default OS CA store can be missing or contain a TLS-intercepting cert. The
# symptom is:
#     URLError: [SSL: CERTIFICATE_VERIFY_FAILED] self-signed certificate
# repeated for every Telegram POST.
#
# Fix: install `certifi` (Mozilla's vendored CA bundle) and use it explicitly
# as the trust store for urllib. Falls back to OS default if certifi isn't
# importable (so dev boxes without it still work).
def _build_ssl_context() -> ssl.SSLContext:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


_SSL_CONTEXT = _build_ssl_context()


# ── tiny duck-typed logger interface ─────────────────────────────────────────-

class _LoggerLike(Protocol):
    def event(self, event: str, **fields: Any) -> None: ...


# ── env loader (unchanged) ────────────────────────────────────────────────────

def _load_env(path: Path) -> dict[str, str]:
    """Parse KEY=VALUE lines from a .env file; ignore comments and blanks."""
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip().strip("'\"")
    return env


# ── notifier ─────────────────────────────────────────────────────────────────-

class Mt5Notifier:
    """
    Fire-and-(observability-)forget Telegram sender.

    Backwards-compatible: existing callers that do `Mt5Notifier()` and then
    `notifier.notify_signal(signal)` still work; the only new optional kwarg
    is `logger`.

    Example:
        from execution import StructuredLogger
        log = StructuredLogger("XAUUSD.i", "logs/XAUUSD.json")
        notifier = Mt5Notifier(logger=log)
        notifier.notify_signal({...})  # success: telegram_sent event
                                       # failure: telegram_error event
    """

    def __init__(
        self,
        logger: Optional[_LoggerLike] = None,
        timeout: float = 10.0,
        max_retries: int = 2,
        retry_delay_seconds: float = 1.5,
    ) -> None:
        env       = _load_env(_REPO_ROOT / ".env")
        token     = env.get("TELEGRAM_BOT_TOKEN") or os.environ.get("TELEGRAM_BOT_TOKEN",  "")
        chat_id   = env.get("TELEGRAM_CHAT_ID")    or os.environ.get("TELEGRAM_CHAT_ID",    "")
        self._url        = f"https://api.telegram.org/bot{token}/sendMessage"
        self._chat_id    = chat_id
        self._enabled    = bool(token and chat_id)
        self._logger     = logger
        self._timeout    = timeout
        self._max_retries = max(0, int(max_retries))
        self._retry_delay = max(0.0, float(retry_delay_seconds))

        if self._logger and not self._enabled:
            self._logger.event(
                "telegram_disabled",
                reason="missing_token_or_chat_id",
                has_token=bool(token), has_chat_id=bool(chat_id),
            )

    # ── low-level ────────────────────────────────────────────────────────────-

    def send(self, text: str, category: str = "generic") -> bool:
        """
        POST `text` (HTML) to Telegram. Returns True on success, False otherwise.

        Errors are logged through `self._logger` but never raised. Callers
        should NOT rely on the return value to decide whether to trade -- it
        is purely for diagnostics.
        """
        if not self._enabled:
            return False

        payload = json.dumps({
            "chat_id":                  self._chat_id,
            "text":                     text,
            "parse_mode":               "HTML",
            "disable_web_page_preview": True,
        }).encode()

        last_failure: dict | None = None

        for attempt in range(1, self._max_retries + 2):  # 1 + retries
            t0 = time.monotonic()
            try:
                req = urllib.request.Request(
                    self._url, data=payload,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=self._timeout, context=_SSL_CONTEXT) as resp:
                    status = resp.status
                    body = resp.read(512).decode("utf-8", errors="replace")

                if 200 <= status < 300:
                    if self._logger:
                        self._logger.event(
                            "telegram_sent",
                            category=category,
                            attempt=attempt,
                            status=status,
                            latency_ms=round((time.monotonic() - t0) * 1000, 1),
                        )
                    return True

                last_failure = {
                    "category": category, "attempt": attempt,
                    "status": status, "body": body[:200],
                }
                # 429 / 5xx are transient -- retry; other 4xx are not
                if status != 429 and not (500 <= status < 600):
                    break

            except urllib.error.HTTPError as exc:
                body = ""
                try:
                    body = exc.read(512).decode("utf-8", errors="replace")
                except Exception:
                    pass
                last_failure = {
                    "category": category, "attempt": attempt,
                    "status": exc.code, "body": body[:200],
                    "error_type": "HTTPError",
                }
                if exc.code != 429 and not (500 <= exc.code < 600):
                    break

            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_failure = {
                    "category": category, "attempt": attempt,
                    "error_type": type(exc).__name__,
                    "error_msg":  str(exc),
                }

            if attempt <= self._max_retries:
                time.sleep(self._retry_delay)

        if self._logger and last_failure is not None:
            self._logger.event("telegram_error", **last_failure)
        return False

    # ── event formatters (text unchanged except for slight tightening) ───────-

    def notify_signal(self, signal: dict) -> bool:
        direction = signal.get("direction", "?")
        icon = "🟢" if direction == "BUY" else "🔴"
        text = (
            f"{icon} <b>SIGNAL — {signal.get('symbol', 'XAUUSD')}</b>\n"
            f"Direction:    <b>{direction}</b>\n"
            f"Entry:        {signal.get('entry')}\n"
            f"SL:           {signal.get('sl')}\n"
            f"TP:           {signal.get('tp')}\n"
            f"OB Time:      {str(signal.get('ob_time', ''))[:16]}\n"
            f"Displaced by: {signal.get('displaced_by')} pts"
        )
        return self.send(text, category="signal")

    def notify_slippage_adjusted(self, data: dict) -> bool:
        text = (
            f"⚠️ <b>SLIPPAGE ADJUSTED — {data.get('symbol', 'XAUUSD')}</b>\n"
            f"Direction: {data.get('direction')}\n"
            f"OB Entry:  {data.get('ob_entry')}\n"
            f"Market:    {data.get('market_price')}\n"
            f"Slippage:  {data.get('slippage_pts')} pts\n"
            f"Volume:    {data.get('volume_original')} → {data.get('volume_adjusted')}\n"
            f"TP:        {data.get('tp_original')} → {data.get('tp_adjusted')}\n"
            f"SL:        {data.get('sl_unchanged')} (unchanged)"
        )
        return self.send(text, category="slippage_adjusted")

    def notify_order_placed(self, data: dict) -> bool:
        """
        ORDER PLACED message — now shows the ACTUAL fill price the order was
        executed at (`fill_price`), alongside the originally-requested entry
        from the signal (`signal_entry`). If only one is supplied, that one
        is shown without the parenthetical comparison.
        """
        direction = data.get("direction", "?")
        icon = "🟢" if direction == "BUY" else "🔴"
        order_type = data.get("order_type", "market").upper()
        slippage_pts = data.get("slippage_pts")
        slippage_str = (f"{float(slippage_pts):.1f}"
                        if isinstance(slippage_pts, (int, float)) else "—")

        fill_price    = data.get("fill_price")
        signal_entry  = data.get("signal_entry")
        if fill_price is not None and signal_entry is not None and \
           abs(float(fill_price) - float(signal_entry)) > 1e-9:
            fill_line = f"Fill Price: {fill_price}  (signal asked: {signal_entry})"
        elif fill_price is not None:
            fill_line = f"Fill Price: {fill_price}"
        elif signal_entry is not None:
            fill_line = f"Entry:      {signal_entry}"
        else:
            fill_line = "Entry:      —"

        text = (
            f"✅ <b>ORDER PLACED ({order_type}) — {data.get('symbol', 'XAUUSD')}</b>\n"
            f"Ticket:    {data.get('ticket')}\n"
            f"Direction: {icon} <b>{direction}</b>\n"
            f"Volume:    {data.get('volume')}\n"
            f"{fill_line}\n"
            f"SL:        {data.get('sl')}\n"
            f"TP:        {data.get('tp')}\n"
            f"Slippage:  {slippage_str} pts"
        )
        return self.send(text, category="order_placed")

    def notify_skip(self, data: dict) -> bool:
        reason = data.get("reason", "?")
        # Noisy reasons are dropped (would spam every cycle out-of-session)
        if reason in ("market_closed_or_stale", "already_traded", "duplicate_pending"):
            return False
        lines = [
            f"⏭ <b>SKIP — {data.get('symbol', 'XAUUSD')}</b>",
            f"Reason: <code>{reason}</code>",
        ]
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
        return self.send("\n".join(lines), category="skip")

    def notify_position_closed(
        self,
        ticket: int,
        profit: float,
        balance: float,
        equity: float,
        symbol: str = "XAUUSD",
        side: Optional[str] = None,
        volume: Optional[float] = None,
        entry_price: Optional[float] = None,
        exit_price: Optional[float] = None,
        opened_at: Optional[str] = None,
        closed_at: Optional[str] = None,
    ) -> bool:
        """
        Per-position close report. Sent automatically after the cycle sweep
        detects a ticket has disappeared from positions_get. Carries entry,
        exit, side, volume and dollar P&L so the operator can audit each
        trade without opening MT5.
        """
        pnl_icon = "📈" if profit >= 0 else "📉"
        outcome  = "WIN ✅" if profit >= 0 else "LOSS 🛑"
        side_icon = ""
        if side:
            side_icon = "🟢 BUY " if side.lower() == "buy" else "🔴 SELL "

        # Pip move (informational — only when we have both prices). For
        # 5-digit FX 1 pip = 10 points (0.0001); for XAU/JPY it's symbol-
        # specific. We just show raw price delta to keep this universal.
        delta_line = ""
        if entry_price is not None and exit_price is not None:
            delta = exit_price - entry_price
            sign  = "+" if delta >= 0 else ""
            delta_line = f"Δ price:  {sign}{delta:.5f}\n"

        lines = [f"{pnl_icon} <b>POSITION CLOSED — {symbol}</b>",
                 f"Ticket: {ticket}",
                 f"Result: <b>{outcome}</b>"]
        if side or volume is not None:
            lines.append(f"Side:   {side_icon}{volume if volume is not None else ''}".rstrip())
        if entry_price is not None:
            lines.append(f"Entry:  {entry_price}")
        if exit_price is not None:
            lines.append(f"Exit:   {exit_price}")
        if delta_line:
            lines.append(delta_line.rstrip())
        if opened_at:
            lines.append(f"Opened: {opened_at[:19]}")
        if closed_at:
            lines.append(f"Closed: {closed_at[:19]}")
        lines.append(f"<b>Profit: {profit:+.2f} USD</b>")
        lines.append("")
        lines.append("💰 <b>Account</b>")
        lines.append(f"Balance: {balance:,.2f} USD")
        lines.append(f"Equity:  {equity:,.2f} USD")

        return self.send("\n".join(lines), category="position_closed")

    # ── new: error / warning / lifecycle events ──────────────────────────────-

    def notify_error(
        self,
        symbol: str,
        event_name: str,
        error_msg: str,
        extra: Optional[dict] = None,
    ) -> bool:
        """
        Send a serious error/exception to telegram so the operator can react.
        Use for: data_fetch_error, strategy_exception, execution_exception,
        broker_validation_failed, partial_fill_emergency, etc.
        """
        lines = [
            f"🚨 <b>ERROR — {symbol}</b>",
            f"Event:  <code>{event_name}</code>",
            f"Detail: {error_msg[:300]}",
        ]
        if extra:
            for k, v in list(extra.items())[:6]:   # cap to 6 keys
                lines.append(f"{k}: {v}")
        return self.send("\n".join(lines), category="error")

    def notify_volume_clamped(
        self,
        symbol: str,
        raw_volume: float,
        snapped_volume: float,
        volume_min: float,
        effective_risk_multiplier: Optional[float],
    ) -> bool:
        """
        Send a warning when raw computed volume was below broker min and got
        clamped UP. Means effective risk > target risk_per_trade.
        """
        mult = f"{effective_risk_multiplier:.2f}x" if effective_risk_multiplier else "—"
        text = (
            f"⚠️ <b>VOLUME CLAMPED TO MIN — {symbol}</b>\n"
            f"Raw lots:       {raw_volume:.6f}\n"
            f"Broker min:     {volume_min}\n"
            f"Sent lots:      {snapped_volume}\n"
            f"Risk multiplier: <b>{mult}</b>\n"
            f"(actual risk is larger than risk_per_trade target)"
        )
        return self.send(text, category="volume_clamped")

    def notify_bot_lifecycle(
        self,
        symbol: str,
        phase: str,            # "start" or "stop"
        magic: Optional[int] = None,
        reason: Optional[str] = None,
        extras: Optional[dict] = None,
    ) -> bool:
        """Compact start / stop notification — one per process lifetime."""
        icon = "🟢" if phase == "start" else "🔴"
        lines = [f"{icon} <b>BOT {phase.upper()} — {symbol}</b>"]
        if magic is not None:
            lines.append(f"Magic: {magic}")
        if reason:
            lines.append(f"Reason: {reason}")
        if extras:
            for k, v in list(extras.items())[:5]:
                lines.append(f"{k}: {v}")
        return self.send("\n".join(lines), category=f"bot_{phase}")

    def notify_pair_open(
        self,
        pair_key: str,
        side: str,
        y_symbol: str, y_volume: float, y_price: float,
        x_symbol: str, x_volume: float, x_price: float,
        z_now: float,
    ) -> bool:
        """Two-leg open notification for pairs trading."""
        icon = "🟢" if side == "long" else "🔴"
        text = (
            f"{icon} <b>PAIR OPEN — {pair_key}</b>\n"
            f"Side:  <b>{side.upper()} spread</b>  (z={z_now:+.3f})\n"
            f"Y:     {y_symbol} {y_volume} @ {y_price}\n"
            f"X:     {x_symbol} {x_volume} @ {x_price}"
        )
        return self.send(text, category="pair_open")

    def notify_pair_close(
        self,
        pair_key: str,
        reason: str,
        bars_in_position: int,
        z_now: float,
        spread_pnl_log: Optional[float] = None,
    ) -> bool:
        """Two-leg close notification for pairs trading."""
        pnl_str = f"{spread_pnl_log:+.4f}" if spread_pnl_log is not None else "n/a"
        text = (
            f"⏹ <b>PAIR CLOSE — {pair_key}</b>\n"
            f"Reason: <code>{reason}</code>\n"
            f"Bars in position: {bars_in_position}\n"
            f"z at close: {z_now:+.3f}\n"
            f"Spread PnL (log-space): {pnl_str}"
        )
        return self.send(text, category="pair_close")
