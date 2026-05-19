"""
Telethon-based Telegram notifier for EMA H1 Trend Exchange.
Uses the user's own Telegram account (via Telethon) to send messages to TARGET_CHANNEL.
Login is done once at startup — subsequent runs reuse the saved session.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Dict, List, Optional

from config.settings import Settings
from exchange.bybit_client import BybitClient
from telegram_bot.telethon_notifier import TelethonNotifier

if TYPE_CHECKING:
    from services.trading_service import TradingService

log = logging.getLogger(__name__)


class TelegramBot:
    """
    Wraps TelethonNotifier and keeps the same interface as the previous
    python-telegram-bot implementation (run / stop / send / client property)
    so the rest of the app requires no changes.
    """

    def __init__(
        self,
        api_id: int,
        api_hash: str,
        phone: Optional[str],
        services: List["TradingService"],
        base_settings: Settings,
        session_file: Optional[str] = None,
    ) -> None:
        notifier_kwargs: dict = {}
        if session_file:
            notifier_kwargs["session_file"] = session_file

        self._notifier = TelethonNotifier(api_id, api_hash, **notifier_kwargs)
        self._phone = phone
        self._services: Dict[str, "TradingService"] = {s.symbol: s for s in services}
        self._base_cfg = base_settings
        # Dedicated BybitClient for any query-based helpers
        self._client = BybitClient(
            api_key=base_settings.bybit_api_key,
            api_secret=base_settings.bybit_api_secret,
            testnet=base_settings.bybit_testnet,
            demo=base_settings.bybit_demo,
        )
        self._stop_event = asyncio.Event()

    @property
    def client(self) -> BybitClient:
        return self._client

    async def login(self) -> None:
        """
        Authenticate with Telegram. Call this once before starting other tasks.
        If a session file already exists this returns immediately without prompts.
        """
        await self._notifier.login(phone=self._phone)

    async def run(self) -> None:
        """Hold the Telethon connection open until stop() is called."""
        log.info(
            "Telethon notifier active — messages will be sent to channel %d",
            self._notifier._target,
        )
        await self._stop_event.wait()

    async def stop(self) -> None:
        self._stop_event.set()
        await self._notifier.stop()

    async def send(self, text: str) -> None:
        """Send a proactive notification to TARGET_CHANNEL."""
        await self._notifier.send(text)
