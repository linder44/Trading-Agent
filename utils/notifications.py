"""Telegram notification helper."""

import time

import requests
from loguru import logger

from config import NotificationConfig


class Notifier:
    def __init__(self, cfg: NotificationConfig):
        self.cfg = cfg

    def send(self, message: str, parse_mode: str = "HTML"):
        """Send a Telegram message with retry logic.

        Uses HTML parse mode by default to avoid issues with
        Markdown special characters (_, *, [, etc.) in order data.
        """
        if not self.cfg.enabled:
            return

        payload = {
            "chat_id": self.cfg.telegram_chat_id,
            "text": message,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode

        url = f"https://api.telegram.org/bot{self.cfg.telegram_bot_token}/sendMessage"

        for attempt in range(3):
            try:
                resp = requests.post(url, json=payload, timeout=10)

                if resp.status_code == 200:
                    return True

                # If HTML parsing failed, retry without parse_mode
                if resp.status_code == 400 and parse_mode:
                    logger.warning(
                        f"Telegram отклонил сообщение (parse_mode={parse_mode}), "
                        f"отправляю без форматирования"
                    )
                    payload.pop("parse_mode", None)
                    resp = requests.post(url, json=payload, timeout=10)
                    if resp.status_code == 200:
                        return True
                    logger.warning(f"Telegram ошибка (повтор без формата): {resp.status_code} {resp.text}")
                else:
                    logger.warning(f"Telegram ошибка: {resp.status_code} {resp.text}")

            except requests.exceptions.Timeout:
                logger.warning(f"Telegram таймаут (попытка {attempt + 1}/3)")
            except Exception as e:
                logger.warning(f"Ошибка отправки в Telegram (попытка {attempt + 1}/3): {e}")

            if attempt < 2:
                time.sleep(2 ** attempt)

        logger.error("Не удалось отправить сообщение в Telegram после 3 попыток")
        return False
