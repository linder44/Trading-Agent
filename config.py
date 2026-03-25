"""Trading Agent Configuration."""

import os
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

# paper = только анализ и логирование, без реальных ордеров
# demo = демо-счёт Bitget (виртуальные деньги, реальные ордера на демо)
# live = реальная торговля (настоящие деньги!)
_raw_mode = os.getenv("TRADING_MODE", "demo").lower()
# "testnet" is an alias for "demo"
TRADING_MODE: str = "demo" if _raw_mode in ("testnet", "demo") else _raw_mode


class BitgetConfig(BaseModel):
    api_key: str = os.getenv("BITGET_API_KEY", "")
    secret_key: str = os.getenv("BITGET_SECRET_KEY", "")
    passphrase: str = os.getenv("BITGET_PASSPHRASE", "")
    demo: bool = os.getenv("TRADING_MODE", "demo").lower() in ("demo", "paper", "testnet")


class ClaudeConfig(BaseModel):
    api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 16384


class TradingConfig(BaseModel):
    symbols: list[str] = [
        # Высоколиквидные пары для краткосрочной торговли (скальпинг)
        "BTC/USDT",
        "ETH/USDT",
        "SOL/USDT",
        "XRP/USDT",
        "BNB/USDT",
        "DOGE/USDT",
        "PEPE/USDT",
    ]
    timeframes: list[str] = ["1m", "5m", "15m"]  # Короткие таймфреймы для скальпинга
    max_position_pct: float = 0.08  # Max 8% портфеля на сделку
    max_open_positions: int = 5  # Меньше позиций, больше фокуса
    default_leverage: int = 1  # Без плеча (безопасно)
    stop_loss_pct: float = 0.005  # 0.5% стоп-лосс (для скальпинга)
    take_profit_pct: float = 0.01  # 1.0% тейк-профит (2:1 RR)
    trailing_stop_pct: float = 0.003  # 0.3% трейлинг-стоп
    analysis_interval_minutes: int = 3  # Анализ каждые 3 минуты
    max_position_age_minutes: int = 120  # Макс. время жизни позиции: 2 часа


class PaperConfig(BaseModel):
    initial_balance: float = float(os.getenv("PAPER_BALANCE", "10000"))


class NotificationConfig(BaseModel):
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")
    enabled: bool = bool(os.getenv("TELEGRAM_BOT_TOKEN", ""))


class NewsConfig(BaseModel):
    api_key: str = os.getenv("NEWS_API_KEY", "")
    newsdata_api_key: str = os.getenv("NEWSDATA_API_KEY", "")
    keywords: list[str] = [
        # Крипто
        "bitcoin", "ethereum", "crypto", "altcoin",
        # Регуляторы и финансы
        "SEC", "Fed", "interest rate", "inflation", "CPI",
        # Геополитика, влияющая на рынки
        "sanctions", "trade war", "tariffs", "war", "conflict",
        "China", "Russia", "NATO", "OPEC",
        # Макроэкономика
        "recession", "GDP", "unemployment", "banking crisis",
        "dollar", "treasury", "stock market crash",
    ]
    check_interval_minutes: int = 60


class Config(BaseModel):
    bitget: BitgetConfig = BitgetConfig()
    claude: ClaudeConfig = ClaudeConfig()
    trading: TradingConfig = TradingConfig()
    paper: PaperConfig = PaperConfig()
    notifications: NotificationConfig = NotificationConfig()
    news: NewsConfig = NewsConfig()


config = Config()
