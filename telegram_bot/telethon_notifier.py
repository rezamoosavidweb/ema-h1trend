"""
Telethon-based notifier — sends messages to TARGET_CHANNEL using the user's own account.
Handles login at startup: phone number → verification code → optional 2FA password.
Session is persisted to disk so subsequent starts skip the login prompts.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from telethon import TelegramClient

TARGET_CHANNEL = -1002383929199  # BullishBearish
_DEFAULT_SESSION = str(Path(__file__).resolve().parent.parent / "telegram_session")

log = logging.getLogger(__name__)


class TelethonNotifier:
    """Sends messages to a Telegram channel via the user's own Telegram account."""

    def __init__(
        self,
        api_id: int,
        api_hash: str,
        session_file: str = _DEFAULT_SESSION,
    ) -> None:
        self._api_id = api_id
        self._api_hash = api_hash
        self._session_file = session_file
        self._client: Optional[TelegramClient] = None
        self._target = TARGET_CHANNEL

    async def login(self, phone: Optional[str] = None) -> None:
        """
        Connect and authenticate.
        - If a saved session exists: silent re-use, no prompts.
        - Otherwise: prompts for phone number, verification code, and 2FA password if needed.
        phone: pre-set phone number string (e.g. '+989123456789'). If None, will prompt.
        """
        self._client = TelegramClient(self._session_file, self._api_id, self._api_hash)

        if phone:
            await self._client.start(phone=lambda: phone)
        else:
            # TelegramClient.start() handles prompts for phone/code/password automatically
            await self._client.start()

        me = await self._client.get_me()
        log.info(
            "Telethon: logged in as %s (@%s) — target channel %s",
            me.first_name,
            me.username or "no username",
            self._target,
        )

    async def send(self, text: str, parse_mode: str = "md") -> None:
        """
        Send a text message to TARGET_CHANNEL.
        parse_mode: 'md' (Markdown), 'html', or None (plain text).
        """
        if self._client is None or not self._client.is_connected():
            log.warning("Telethon: client not connected — message dropped: %.80s", text)
            return
        try:
            await self._client.send_message(
                self._target,
                text,
                parse_mode=parse_mode,
                link_preview=False,
            )
        except Exception as exc:
            log.warning("Telethon: send_message failed: %s", exc)

    async def stop(self) -> None:
        if self._client and self._client.is_connected():
            await self._client.disconnect()
            log.info("Telethon: client disconnected")
