"""Telegram notification helper."""

import requests
from loguru import logger

from config import NotificationConfig


class Notifier:
    def __init__(self, cfg: NotificationConfig):
        self.cfg = cfg

    def send(self, message: str):
        """Send a Telegram message."""
        if not self.cfg.enabled:
            return

        try:
            requests.post(
                f"https://api.telegram.org/bot{self.cfg.telegram_bot_token}/sendMessage",
                json={"chat_id": self.cfg.telegram_chat_id, "text": message, "parse_mode": "Markdown"},
                timeout=10,
            )
        except Exception as e:
            logger.warning(f"Telegram notification failed: {e}")
