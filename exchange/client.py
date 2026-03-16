"""Bitget exchange client via ccxt."""

import ccxt
import pandas as pd
from loguru import logger
from config import BitgetConfig


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

        self.exchange = ccxt.bitget(params)

        # Activate demo/sandbox mode if configured
        if cfg.demo and cfg.api_key:
            self.exchange.set_sandbox_mode(True)

        self._has_auth = bool(cfg.api_key)
        self._is_demo = cfg.demo
        mode = "demo" if cfg.demo else ("auth" if self._has_auth else "public only")
        logger.info(f"Exchange initialized (mode={mode})")

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 200) -> pd.DataFrame:
        """Fetch OHLCV candles as DataFrame. Works without auth."""
        raw = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
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
        logger.info(f"Market {side} {amount} {symbol} -> {order['id']}")
        return order

    def create_limit_order(self, symbol: str, side: str, amount: float, price: float, params: dict | None = None) -> dict:
        """Place a limit order."""
        params = params or {}
        order = self.exchange.create_order(symbol, "limit", side, amount, price, params=params)
        logger.info(f"Limit {side} {amount} {symbol} @ {price} -> {order['id']}")
        return order

    def create_stop_loss(self, symbol: str, side: str, amount: float, stop_price: float, params: dict | None = None) -> dict:
        """Place a stop-loss order."""
        params = params or {}
        params["stopPrice"] = stop_price
        params["triggerPrice"] = stop_price
        order = self.exchange.create_order(symbol, "market", side, amount, params=params)
        logger.info(f"Stop-loss {side} {amount} {symbol} trigger @ {stop_price} -> {order['id']}")
        return order

    def create_take_profit(self, symbol: str, side: str, amount: float, tp_price: float, params: dict | None = None) -> dict:
        """Place a take-profit order."""
        params = params or {}
        params["stopPrice"] = tp_price
        params["triggerPrice"] = tp_price
        order = self.exchange.create_order(symbol, "market", side, amount, params=params)
        logger.info(f"Take-profit {side} {amount} {symbol} trigger @ {tp_price} -> {order['id']}")
        return order

    def cancel_order(self, order_id: str, symbol: str) -> dict:
        """Cancel an order."""
        result = self.exchange.cancel_order(order_id, symbol)
        logger.info(f"Cancelled order {order_id} on {symbol}")
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
        logger.info(f"Set leverage {leverage}x for {symbol}")
        return result

    def get_market_info(self, symbol: str) -> dict:
        """Get market info (min order size, tick size, etc.)."""
        self.exchange.load_markets()
        return self.exchange.market(symbol)
