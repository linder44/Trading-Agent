"""Bitget exchange client via ccxt."""

import ccxt
import pandas as pd
from loguru import logger
from config import BitgetConfig


class _BitgetDemo(ccxt.bitget):
    """Bitget subclass that adds the PAPTRADING header for demo/simulated trading.

    ccxt's built-in set_sandbox_mode() is broken for Bitget — it mixes demo-symbol
    prefixes with the paptrading header, causing 40099 errors and missing markets.
    The correct approach: use production endpoints + PAPTRADING=1 header on every
    authenticated request.  See https://github.com/ccxt/ccxt/issues/25523
    """

    def sign(self, path, api="public", method="GET", params=None, headers=None, body=None):
        result = super().sign(path, api, method, params or {}, headers, body)
        if result.get("headers") is None:
            result["headers"] = {}
        result["headers"]["PAPTRADING"] = "1"
        return result


class ExchangeClient:
    def __init__(self, cfg: BitgetConfig):
        params = {
            "enableRateLimit": True,
            "options": {
                "defaultType": "swap",
            },
        }
        # Only add auth if keys are provided (paper mode may not have them)
        if cfg.api_key:
            params["apiKey"] = cfg.api_key
            params["secret"] = cfg.secret_key
            params["password"] = cfg.passphrase

        # Use custom subclass for demo trading (adds PAPTRADING header),
        # regular ccxt.bitget for live/public-only mode.
        if cfg.demo and cfg.api_key:
            self.exchange = _BitgetDemo(params)
        else:
            self.exchange = ccxt.bitget(params)

        self._has_auth = bool(cfg.api_key)
        self._is_demo = cfg.demo
        mode = "демо" if cfg.demo else ("авторизован" if self._has_auth else "только публичный")
        logger.info(f"Биржа инициализирована (режим={mode})")

        # Pre-load markets so we can validate symbols later
        self.exchange.load_markets()

        # Set one-way position mode (Bitget default may be hedge mode)
        if self._has_auth:
            try:
                self.exchange.set_position_mode(hedged=False)
                logger.info("Режим позиций: one-way (односторонний)")
            except Exception as e:
                logger.debug(f"set_position_mode: {e} (может быть уже установлен)")

    def validate_symbols(self, symbols: list[str]) -> list[str]:
        """Возвращает только символы, доступные на фьючерсах (swap) биржи.

        Bitget swap markets используют формат 'BTC/USDT:USDT'.
        В конфиге используется короткий формат 'BTC/USDT' — конвертируем в swap.
        Приоритет всегда отдаётся фьючерсному (swap) рынку.
        """
        valid = []
        for s in symbols:
            swap_symbol = f"{s}:USDT" if ":USDT" not in s else s
            if swap_symbol in self.exchange.markets:
                if swap_symbol != s:
                    logger.info(f"Маппинг {s} -> {swap_symbol} (фьючерсы)")
                valid.append(swap_symbol)
            elif s in self.exchange.markets:
                market = self.exchange.markets[s]
                if market.get("swap") or market.get("future"):
                    valid.append(s)
                    logger.info(f"Символ {s} — фьючерсный рынок")
                else:
                    logger.warning(f"Символ {s} доступен только на споте, пропускаем (бот только фьючерсный)")
            else:
                logger.warning(f"Символ {s} недоступен на бирже (демо={self._is_demo}), пропускаем")
        return valid

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 200) -> pd.DataFrame:
        """Fetch OHLCV candles as DataFrame. Works without auth."""
        try:
            raw = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        except Exception:
            # Некоторые версии ccxt маппят "1d" → "1Dutc", что Bitget не принимает.
            # Пробуем с явным params override.
            if timeframe in ("1d", "1D"):
                raw = self.exchange.fetch_ohlcv(
                    symbol, timeframe, limit=limit,
                    params={"granularity": "1d", "productType": "USDT-FUTURES"},
                )
            else:
                raise
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df

    def fetch_ticker(self, symbol: str) -> dict:
        """Get current ticker data. Works without auth."""
        return self.exchange.fetch_ticker(symbol)

    def fetch_balance(self) -> dict:
        """Get account balance. Requires auth."""
        balance = self.exchange.fetch_balance()
        return {
            "total": balance.get("total", {}),
            "free": balance.get("free", {}),
            "used": balance.get("used", {}),
        }

    def fetch_usdt_balance(self) -> float:
        """Get available USDT balance."""
        balance = self.fetch_balance()
        return float(balance["free"].get("USDT", 0))

    def fetch_open_orders(self, symbol: str | None = None) -> list:
        """Get all open orders."""
        return self.exchange.fetch_open_orders(symbol)

    def fetch_positions(self, symbols: list[str] | None = None) -> list[dict]:
        """Get open positions."""
        positions = self.exchange.fetch_positions(symbols)
        return [p for p in positions if float(p.get("contracts", 0)) > 0]

    def fetch_order_book(self, symbol: str, limit: int = 20) -> dict:
        """Get order book. Works without auth."""
        return self.exchange.fetch_order_book(symbol, limit)

    def create_market_order(self, symbol: str, side: str, amount: float, params: dict | None = None) -> dict:
        """Place a market order."""
        params = params or {}
        order = self.exchange.create_order(symbol, "market", side, amount, params=params)
        logger.info(f"Рыночный ордер {side} {amount} {symbol} -> {order['id']}")
        return order

    def create_limit_order(self, symbol: str, side: str, amount: float, price: float, params: dict | None = None) -> dict:
        """Place a limit order."""
        params = params or {}
        order = self.exchange.create_order(symbol, "limit", side, amount, price, params=params)
        logger.info(f"Лимитный ордер {side} {amount} {symbol} @ {price} -> {order['id']}")
        return order

    def create_stop_loss(self, symbol: str, side: str, amount: float, stop_price: float, params: dict | None = None) -> dict:
        """Place a stop-loss order."""
        params = params or {}
        params["stopPrice"] = stop_price
        params["triggerPrice"] = stop_price
        order = self.exchange.create_order(symbol, "market", side, amount, params=params)
        logger.info(f"Стоп-лосс {side} {amount} {symbol} триггер @ {stop_price} -> {order['id']}")
        return order

    def create_take_profit(self, symbol: str, side: str, amount: float, tp_price: float, params: dict | None = None) -> dict:
        """Place a take-profit order."""
        params = params or {}
        params["stopPrice"] = tp_price
        params["triggerPrice"] = tp_price
        order = self.exchange.create_order(symbol, "market", side, amount, params=params)
        logger.info(f"Тейк-профит {side} {amount} {symbol} триггер @ {tp_price} -> {order['id']}")
        return order

    def create_trigger_order(self, symbol: str, side: str, amount: float, trigger_price: float, params: dict | None = None) -> dict:
        """Триггерный ордер — рыночный ордер, активирующийся при достижении trigger_price."""
        params = params or {}
        params["triggerPrice"] = trigger_price
        params["triggerType"] = "market_price"
        order = self.exchange.create_order(symbol, "market", side, amount, params=params)
        logger.info(f"Триггерный ордер {side} {amount} {symbol} триггер @ {trigger_price} -> {order['id']}")
        return order

    def cancel_order(self, order_id: str, symbol: str) -> dict:
        """Cancel an order."""
        result = self.exchange.cancel_order(order_id, symbol)
        logger.info(f"Отменён ордер {order_id} на {symbol}")
        return result

    def cancel_all_orders(self, symbol: str) -> list:
        """Cancel all orders for a symbol."""
        orders = self.fetch_open_orders(symbol)
        results = []
        for o in orders:
            results.append(self.cancel_order(o["id"], symbol))
        return results

    def set_leverage(self, symbol: str, leverage: int) -> dict:
        """Set leverage for a symbol."""
        result = self.exchange.set_leverage(leverage, symbol)
        logger.info(f"Установлено плечо {leverage}x для {symbol}")
        return result

    def get_market_info(self, symbol: str) -> dict:
        """Get market info (min order size, tick size, etc.)."""
        self.exchange.load_markets()
        return self.exchange.market(symbol)

    def round_amount(self, symbol: str, amount: float) -> float:
        """Round amount to market precision using ccxt's built-in method."""
        return float(self.exchange.amount_to_precision(symbol, amount))

    def round_price(self, symbol: str, price: float) -> float:
        """Round price to market precision using ccxt's built-in method."""
        return float(self.exchange.price_to_precision(symbol, price))
