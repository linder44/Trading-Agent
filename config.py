"""Trading Agent Configuration."""

import os
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


class BitgetConfig(BaseModel):
    api_key: str = os.getenv("BITGET_API_KEY", "")
    secret_key: str = os.getenv("BITGET_SECRET_KEY", "")
    passphrase: str = os.getenv("BITGET_PASSPHRASE", "")
    sandbox: bool = os.getenv("TRADING_MODE", "testnet") == "testnet"


class ClaudeConfig(BaseModel):
    api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096


class TradingConfig(BaseModel):
    symbols: list[str] = [
        "BTC/USDT",
        "ETH/USDT",
        "SOL/USDT",
        "XRP/USDT",
        "BNB/USDT",
    ]
    timeframes: list[str] = ["1h", "4h", "1d"]
    max_position_pct: float = 0.1  # Max 10% of portfolio per trade
    max_open_positions: int = 5
    default_leverage: int = 1  # No leverage by default (safe)
    stop_loss_pct: float = 0.03  # 3% stop loss
    take_profit_pct: float = 0.06  # 6% take profit (2:1 RR)
    trailing_stop_pct: float = 0.02  # 2% trailing stop
    analysis_interval_minutes: int = 30


class NotificationConfig(BaseModel):
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")
    enabled: bool = bool(os.getenv("TELEGRAM_BOT_TOKEN", ""))


class NewsConfig(BaseModel):
    api_key: str = os.getenv("NEWS_API_KEY", "")
    keywords: list[str] = ["bitcoin", "ethereum", "crypto", "SEC", "Fed", "inflation"]
    check_interval_minutes: int = 60


class Config(BaseModel):
    bitget: BitgetConfig = BitgetConfig()
    claude: ClaudeConfig = ClaudeConfig()
    trading: TradingConfig = TradingConfig()
    notifications: NotificationConfig = NotificationConfig()
    news: NewsConfig = NewsConfig()


config = Config()
